#!/usr/bin/env python3
"""Gate peak emitter RSS using deterministic final-ACIR fixtures.

Fixture generation and linking are deliberately outside the measured command.
Only ``pycircuit-emit`` and the processes in its isolated process group count
toward the fixed platform limit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO / "tests/perf/emitter/cases.json"
DEFAULT_LIMITS = REPO / "tests/perf/emitter/rss_limits.json"
GENERATOR = REPO / "benchmarks/source-units/generate.py"
TARGETS = ("cpp", "verilog")


class GateError(RuntimeError):
    """A measurement precondition or measured run failed."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise GateError(f"unsupported or missing schema_version in {path}")
    return value


def platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{system}-{machine}"


def selected_cases(config: dict[str, Any], tier: str) -> list[tuple[str, dict]]:
    cases = config.get("cases")
    if not isinstance(cases, dict):
        raise GateError("cases config must contain an object named 'cases'")
    selected = []
    for name, case in cases.items():
        if not isinstance(case, dict):
            raise GateError(f"case {name!r} must be an object")
        if tier in case.get("tiers", []):
            selected.append((name, case))
    if not selected:
        raise GateError(f"no emitter RSS cases select tier {tier!r}")
    return selected


def evaluate_runs(runs: list[dict[str, Any]], limit_mib: int) -> dict[str, Any]:
    if not isinstance(limit_mib, int) or isinstance(limit_mib, bool) or limit_mib <= 0:
        raise GateError("RSS limit must be a positive integer MiB value")
    peaks = []
    failures = []
    for index, run in enumerate(runs, 1):
        rss = run.get("rss", {})
        if run.get("timed_out"):
            failures.append(f"run {index} timed out")
        if run.get("exit_status") != 0:
            failures.append(f"run {index} exited {run.get('exit_status')}")
        if not rss.get("available"):
            failures.append(
                f"run {index} RSS unavailable: {rss.get('unavailable_reason')}"
            )
        peak = rss.get("peak_sampled_sum_kib")
        if isinstance(peak, int):
            peaks.append(peak)
    if len(peaks) != len(runs):
        failures.append("one or more runs have no sampled peak RSS")

    limit_kib = limit_mib * 1024
    maximum = max(peaks) if peaks else None
    if maximum is not None and maximum > limit_kib:
        failures.append(f"peak {maximum / 1024:.1f} MiB exceeds {limit_mib} MiB limit")
    return {
        "status": "pass" if not failures else "fail",
        "limit_mib": limit_mib,
        "median_peak_rss_kib": statistics.median(peaks) if peaks else None,
        "max_peak_rss_kib": maximum,
        "failures": failures,
    }


def _load_generator():
    spec = importlib.util.spec_from_file_location("emitter_rss_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise GateError(f"cannot load fixture generator at {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_logged(
    argv: list[str], *, cwd: Path, env: dict[str, str], log: Path, timeout: int
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as output:
        try:
            result = subprocess.run(
                argv,
                cwd=cwd,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise GateError(f"command timed out; see {log}: {argv}") from exc
    if result.returncode:
        raise GateError(f"command exited {result.returncode}; see {log}: {argv}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise GateError(f"output directory must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _case_limit(limits: dict[str, Any], key: str, case_name: str, target: str) -> int:
    try:
        value = limits[key]["release"][case_name][f"{target}_mib"]
    except (KeyError, TypeError) as exc:
        raise GateError(f"no release RSS limit for {key}/{case_name}/{target}") from exc
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise GateError(f"invalid RSS limit for {key}/{case_name}/{target}")
    return value


def _build_fixture(
    name: str,
    case: dict[str, Any],
    *,
    output: Path,
    prefix: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[Path, dict[str, Any]]:
    case_root = output / name
    fixture_root = case_root / "fixture"
    build_root = case_root / "build"
    fixture = _load_generator().generate_case(fixture_root, case["axis"], case["size"])
    cmake = shutil.which("cmake")
    if cmake is None:
        raise GateError("cmake executable unavailable")
    configure = [
        cmake,
        "-S",
        str(fixture_root),
        "-B",
        str(build_root),
        "-G",
        "Ninja",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-DPYCIRCUIT_PREFIX={prefix}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    _run_logged(
        configure,
        cwd=REPO,
        env=env,
        log=case_root / "configure.log",
        timeout=timeout,
    )
    _run_logged(
        [
            cmake,
            "--build",
            str(build_root),
            "--target",
            "source-unit-all",
            "--parallel",
            "4",
        ],
        cwd=REPO,
        env=env,
        log=case_root / "build.log",
        timeout=timeout,
    )
    final_ac = build_root / "design_top.ac"
    if not final_ac.is_file():
        raise GateError(f"fixture did not produce {final_ac}")
    size = final_ac.stat().st_size
    lower = case["final_ac_min_bytes"]
    upper = case["final_ac_max_bytes"]
    if not lower <= size <= upper:
        raise GateError(
            f"{name} final ACIR is {size} bytes, outside [{lower}, {upper}]"
        )
    return final_ac, {
        "fixture": fixture,
        "final_ac_size_bytes": size,
        "final_ac_sha256": _sha256(final_ac),
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    from process_usage import run_sampled

    cases = load_json(args.cases)
    limits = load_json(args.limits)
    key = args.platform_key or platform_key()
    if key not in limits:
        raise GateError(f"unsupported measurement platform {key!r}")
    prefix = args.prefix.resolve()
    emitter = prefix / "bin/pycircuit-emit"
    driver = prefix / "bin/pycircuit"
    config = prefix / "share/pycircuit/cmake/pycircuitConfig.cmake"
    for required in (emitter, driver, config):
        if not required.exists():
            raise GateError(f"installed toolchain file is missing: {required}")

    output = args.output_dir.resolve()
    _prepare_output(output)
    env = os.environ.copy()
    env["PATH"] = (
        f"{Path(sys.executable).parent}{os.pathsep}{prefix / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "tier": args.tier,
        "platform": key,
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "toolchain": {
            "prefix": str(prefix),
            "emitter_sha256": _sha256(emitter),
        },
        "cases": {},
    }
    for name, case in selected_cases(cases, args.tier):
        final_ac, case_report = _build_fixture(
            name,
            case,
            output=output,
            prefix=prefix,
            env=env,
            timeout=args.timeout,
        )
        case_report["targets"] = {}
        repeats = args.repeats if args.repeats is not None else case["repeats"]
        if repeats <= 0:
            raise GateError("repeats must be positive")
        for target in TARGETS:
            runs = []
            for repeat in range(1, repeats + 1):
                runs.append(
                    run_sampled(
                        [
                            str(emitter),
                            str(final_ac),
                            "--target",
                            target,
                            "--format",
                            "json",
                        ],
                        cwd=REPO,
                        env=env,
                        log_dir=output / name / "logs",
                        label=f"{target}-{repeat}",
                        sample_interval_seconds=args.rss_interval,
                        timeout_seconds=args.timeout,
                    )
                )
            evaluation = evaluate_runs(runs, _case_limit(limits, key, name, target))
            case_report["targets"][target] = {"runs": runs, **evaluation}
            if evaluation["status"] != "pass":
                report["status"] = "fail"
        report["cases"][name] = case_report
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("gate", "nightly"), required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--rss-interval", type=float, default=0.02)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--platform-key", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report_path = args.output_dir.resolve() / "report.json"
    try:
        report = run_gate(args)
    except (GateError, KeyError, TypeError, ValueError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {"schema_version": 1, "status": "error", "error": str(exc)}
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"emitter RSS gate error: {exc}", file=sys.stderr)
        return 1
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for case_name, case in report["cases"].items():
        for target, result in case["targets"].items():
            peak = result["max_peak_rss_kib"]
            peak_mib = "unavailable" if peak is None else f"{peak / 1024:.1f} MiB"
            print(
                f"{case_name}/{target}: {result['status']} "
                f"(max {peak_mib}, limit {result['limit_mib']} MiB)"
            )
            for failure in result["failures"]:
                print(f"  {failure}")
    print(f"report: {report_path}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
