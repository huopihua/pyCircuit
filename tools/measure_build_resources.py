#!/usr/bin/env python3
"""Measure current source builds, backend emission, and Structural C++ runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from process_usage import run_sampled

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "benchmarks/source-units/generate.py"
SMALL_SIZES = {"shared": (1,), "distinct": (1,)}
SELECTED_SIZES = {"shared": (1, 16), "distinct": (1, 8)}


def _load_generator():
    spec = importlib.util.spec_from_file_location("resource_source_graph", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source graph generator: {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True, check=False
    )
    if result.returncode:
        return f"unavailable (exit {result.returncode}): {result.stderr.strip()}"
    return (result.stdout or result.stderr).splitlines()[0]


def _phase(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    logs: Path,
    label: str,
    interval: float,
    timeout: float,
) -> dict[str, Any]:
    result = run_sampled(
        argv,
        cwd=cwd,
        env=env,
        log_dir=logs,
        label=label,
        sample_interval_seconds=interval,
        timeout_seconds=timeout,
    )
    if result["exit_status"] != 0 or result["timed_out"]:
        stdout = Path(result["stdout_log"]).read_text(
            encoding="utf-8", errors="replace"
        )[-4000:]
        stderr = Path(result["stderr_log"]).read_text(
            encoding="utf-8", errors="replace"
        )[-8000:]
        raise RuntimeError(
            f"stage failed: {argv!r}, exit={result['exit_status']}, "
            f"timeout={result['timed_out']}\nstdout tail:\n{stdout}\nstderr tail:\n{stderr}"
        )
    return result


def _executed_commands(stdout_path: str) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for line in Path(stdout_path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        try:
            arguments = shlex.split(line.strip())
        except ValueError:
            continue
        kind: str | None = None
        source: str | None = None
        for index, argument in enumerate(arguments[:-1]):
            if Path(argument).name == "pycircuit":
                action = arguments[index + 1]
                if action in {"compile", "link"}:
                    kind = f"pycircuit_{action}"
                elif action == "emit" and "--target" in arguments:
                    kind = f"pycircuit_emit_{arguments[arguments.index('--target') + 1]}"
                if "-c" in arguments:
                    source = arguments[arguments.index("-c") + 1]
                break
        is_cpp = any(
            Path(argument).name in {"c++", "g++", "clang++", "clang-cl"}
            for argument in arguments
        )
        if is_cpp and "-c" in arguments:
            kind = "cpp_compile"
            source = arguments[arguments.index("-c") + 1].strip('"')
        elif arguments and Path(arguments[0]).name in {"ar", "llvm-ar"}:
            kind = "cpp_archive"
        if kind:
            commands.append({"kind": kind, "command": line.strip(), "source": source})
    return commands


def _record(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "argv": result["argv"],
        "cwd": result["cwd"],
        "exit_status": result["exit_status"],
        "timed_out": result["timed_out"],
        "timeout_cleanup": result["timeout_cleanup"],
        "wall_seconds": result["wall_seconds"],
        "rss": result["rss"],
        "process_group": result["process_group"],
        "stdout_log": result["stdout_log"],
        "stderr_log": result["stderr_log"],
        "executed_commands": _executed_commands(result["stdout_log"]),
    }


def _configure(cmake: str, source: Path, build: Path, prefix: Path) -> list[str]:
    return [
        cmake,
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DPYCIRCUIT_PREFIX={prefix}",
        f"-DCMAKE_PREFIX_PATH={prefix}",
    ]


def _build(cmake: str, build: Path, target: str, jobs: int) -> list[str]:
    return [
        cmake,
        "--build",
        str(build),
        "--target",
        target,
        "--verbose",
        "--parallel",
        str(jobs),
    ]


def _source_inventory(build: Path, stems: list[str]) -> dict[str, Any]:
    rows = []
    for stem in stems:
        unit = build / "units" / stem
        paths = {
            "body": unit / f"{stem}.ac",
            "interface": unit / f"{stem}.interface.ac",
            "receipt": unit / "unit.json",
        }
        if not all(path.is_file() for path in paths.values()):
            raise RuntimeError(f"incomplete source unit: {unit}")
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        if receipt.get("source", {}).get("path") != f"{stem}.py":
            raise RuntimeError(f"source receipt mismatch: {receipt}")
        rows.append({"source": f"src/{stem}.py", **{key: str(value) for key, value in paths.items()}})
    final = build / "design_top.ac"
    if not final.is_file():
        raise RuntimeError(f"linked final artifact missing: {final}")
    return {"source_units": rows, "linked_final": str(final)}


def _backend_inventory(build: Path, stems: list[str]) -> dict[str, Any]:
    expected_owners = {("measurement", f"{stem}.py") for stem in stems}
    result: dict[str, Any] = {}
    for target in ("structural-cpp", "verilog"):
        root = build / target
        receipt_path = root / "generated.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"missing {target} receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        files = receipt.get("files")
        groups = receipt.get("source_groups")
        if not isinstance(files, list) or not isinstance(groups, list):
            raise RuntimeError(f"malformed {target} receipt")
        roles: dict[str, str] = {}
        audited = []
        root_resolved = root.resolve()
        for row in files:
            relative, role = row.get("path"), row.get("role")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or relative in roles
                or not isinstance(role, str)
            ):
                raise RuntimeError(f"unsafe {target} receipt row: {row!r}")
            path = root / relative
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root_resolved) or path.is_symlink() or not path.is_file():
                raise RuntimeError(f"unsafe {target} generated file: {relative}")
            roles[relative] = role
            audited.append({"path": relative, "role": role, "resolved_path": str(resolved)})
        required = (
            {"CMakeLists.txt": "cmake", "design.hpp": "header", "manifest.json": "runtime-glue"}
            if target == "structural-cpp"
            else {
                "CMakeLists.txt": "cmake",
                "design_top.sv": "rtl",
                "ports.json": "runtime-glue",
                "verilator_policy.json": "runtime-glue",
            }
        )
        if any(roles.get(path) != role for path, role in required.items()):
            raise RuntimeError(f"{target} required file roles are incomplete: {roles}")
        owners = set()
        normalized = []
        extension = ".cpp" if target == "structural-cpp" else ".sv"
        for group in groups:
            source = group.get("source", {})
            owner = (source.get("package"), source.get("path"))
            if owner in owners or not all(isinstance(item, str) for item in owner):
                raise RuntimeError(f"malformed or duplicate {target} source owner: {owner}")
            owners.add(owner)
            owned = group.get("files")
            if not isinstance(owned, list) or not owned or not set(owned) <= set(roles):
                raise RuntimeError(f"invalid {target} source group: {group}")
            implementations = [path for path in owned if path.endswith(extension)]
            if len(implementations) != 1:
                raise RuntimeError(f"{target} source group needs one implementation: {group}")
            if roles[implementations[0]] != ("source" if target == "structural-cpp" else "rtl"):
                raise RuntimeError(f"wrong implementation role in {target}: {group}")
            normalized.append({**group, "implementation_files": implementations})
        if owners != expected_owners:
            raise RuntimeError(f"{target} owners differ: {owners} != {expected_owners}")
        result[target] = {
            "groups": normalized,
            "receipt_files": audited,
            "implementation_file_count": len(normalized),
            "definition_count": len(stems),
        }
    return result


CONSUMER = r'''#include "design.hpp"
#include <gfsim/SimExecutor.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static auto bit(unsigned value) {
  return gfsim::wire<gfsim::Bits<1>>::known(gfsim::Bits<1>{value});
}

static bool run(pyc_dut &dut, gfsim::SimExecutor &executor,
                std::uint64_t epochs, std::uint64_t &elapsed) {
  pyc_dut::Inputs inputs;
  inputs.pyc_7079635f636c6b = bit(0);
  inputs.pyc_7079635f727374 = bit(1);
  dut.drive(inputs);
  if (executor.Reset() != PYCIRCUIT_MODEL_STATUS_V1_OK) return false;
  inputs.pyc_7079635f727374 = bit(0);
  const auto started = std::chrono::steady_clock::now();
  for (std::uint64_t epoch = 0; epoch < epochs; ++epoch) {
    inputs.pyc_7079635f636c6b = bit(static_cast<unsigned>(epoch & 1));
    dut.drive(inputs);
    PycircuitModelStepResultV1 result{};
    result.struct_size = sizeof(result);
    if (executor.Step(&result) != PYCIRCUIT_MODEL_STATUS_V1_OK ||
        result.state != PYCIRCUIT_MODEL_STEP_V1_RUNNING) return false;
  }
  elapsed = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started).count());
  return dut.sample().result.isFullyKnown();
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  char *end = nullptr;
  const auto epochs = std::strtoull(argv[1], &end, 10);
  if (!end || *end || !epochs) return 3;
  const auto setup_started = std::chrono::steady_clock::now();
  pyc_dut dut;
  gfsim::SimExecutor executor(dut.system(), dut.observations(), {});
  constexpr char config[] = "{}";
  if (!executor.created() ||
      executor.ConfigureJson(reinterpret_cast<const std::uint8_t *>(config), 2) !=
          PYCIRCUIT_MODEL_STATUS_V1_OK) return 4;
  const auto setup_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - setup_started).count());
  std::uint64_t first = 0, replay = 0;
  if (!run(dut, executor, epochs, first) ||
      !run(dut, executor, epochs, replay)) return 4;
  std::cout << "{\"epochs\":" << epochs
            << ",\"setup_ns\":" << setup_ns
            << ",\"step_loop_ns\":" << first
            << ",\"reset_replay_ns\":" << replay
            << ",\"terminal_state\":\"COMPLETED\""
            << ",\"epoch_time\":" << epochs
            << ",\"result_valid\":true"
            << ",\"reset_replay_valid\":true}\n";
  return 0;
}
'''


def _parse_consumer(path: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Structural C++ consumer emitted {len(rows)} records")
    return rows[0]


def _measure_case(
    *,
    axis: str,
    size: int,
    output: Path,
    prefix: Path,
    cmake: str,
    cxx: str,
    env: dict[str, str],
    interval: float,
    timeout: float,
    jobs: int,
    epochs: list[int],
    repeats: int,
    report: dict[str, Any],
) -> None:
    root = output / f"{axis}-{size}"
    fixture = _load_generator().generate_case(root / "fixture", axis, size)
    build = root / "build"
    logs = root / "logs"
    case = {
        "case": {
            "axis": axis,
            "size": size,
            "source_units": fixture["source_units"],
            "definitions": fixture["definitions"],
            "instances": fixture["instances"],
        },
        "phases": [],
    }
    report["cases"].append(case)

    def run(name: str, argv: list[str], cwd: Path) -> dict[str, Any]:
        result = _phase(
            argv, cwd=cwd, env=env, logs=logs, label=name, interval=interval, timeout=timeout
        )
        record = _record(name, result)
        case["phases"].append(record)
        _write_json(output / "report.json", report)
        return record

    run("configure_source_graph", _configure(cmake, root / "fixture", build, prefix), root)
    run("source_compile_link", _build(cmake, build, "source-unit-all", jobs), root)
    case["source_inventory"] = _source_inventory(build, fixture["stems"])
    run("emit_structural_cpp", _build(cmake, build, "emit_cpp", jobs), root)
    run("emit_verilog", _build(cmake, build, "emit_verilog", jobs), root)
    case["backend_inventory"] = _backend_inventory(build, fixture["stems"])

    generated = build / "structural-cpp"
    model_build = root / "model-build"
    run("configure_structural_cpp", _configure(cmake, generated, model_build, prefix), root)
    run("build_structural_cpp_translation_units", _build(cmake, model_build, "all", jobs), root)
    library = model_build / "libpycircuit_modules.a"
    if not library.is_file():
        raise RuntimeError(f"generated module library missing: {library}")

    consumer_source = root / "structural-cpp-consumer.cpp"
    consumer_source.write_text(CONSUMER, encoding="utf-8")
    consumer = root / "structural-cpp-consumer"
    compile_command = [
        cxx,
        "-std=c++20",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-Wno-unused-variable",
        "-I",
        str(generated),
        "-I",
        str(prefix / "include"),
        str(consumer_source),
        str(library),
        str(prefix / "lib/libpyc6_runtime.a"),
        "-pthread",
        "-o",
        str(consumer),
    ]
    run("compile_structural_cpp_consumer", compile_command, root)
    smoke = run("three_epoch_structural_cpp", [str(consumer), "3"], root)
    case["runtime_smoke"] = _parse_consumer(smoke["stdout_log"])

    for epoch_count in epochs:
        runs = []
        for repeat in range(1, repeats + 1):
            measured = _phase(
                [str(consumer), str(epoch_count)],
                cwd=root,
                env=env,
                logs=logs,
                label=f"structural-cpp-{epoch_count}-{repeat}",
                interval=interval,
                timeout=timeout,
            )
            row = _parse_consumer(measured["stdout_log"])
            if row.get("epochs") != epoch_count or not row.get("result_valid") or not row.get("reset_replay_valid"):
                raise RuntimeError(f"Structural C++ run oracle mismatch: {row}")
            first_seconds = row["step_loop_ns"] / 1_000_000_000
            replay_seconds = row["reset_replay_ns"] / 1_000_000_000
            runs.append(
                {
                    **row,
                    "repeat": repeat,
                    "process_wall_seconds_including_startup": measured["wall_seconds"],
                    "process_rss": measured["rss"],
                    "structural_cpp_epochs_per_second": epoch_count / first_seconds if first_seconds else None,
                    "structural_cpp_replay_epochs_per_second": epoch_count / replay_seconds if replay_seconds else None,
                    "amortized_epochs_per_second_including_process_startup": (
                        epoch_count * 2 / measured["wall_seconds"] if measured["wall_seconds"] else None
                    ),
                    "timing_note": "finite Structural C++ Work/Validate/Xfer loop; not a steady-state guarantee",
                    "argv": measured["argv"],
                    "exit_status": measured["exit_status"],
                    "timed_out": measured["timed_out"],
                    "wall_seconds": measured["wall_seconds"],
                    "stdout_log": measured["stdout_log"],
                    "stderr_log": measured["stderr_log"],
                }
            )
        report["long_runs"].append(
            {"case": case["case"], "measurements": [{"epochs": epoch_count, "runs": runs}]}
        )
        _write_json(output / "report.json", report)


def _epoch_values(values: list[str]) -> list[int]:
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed) or parsed != sorted(set(parsed)):
        raise ValueError("epoch counts must be unique, increasing positive integers")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", choices=("small", "selected", "all"), default="selected")
    parser.add_argument("--epochs", nargs="+", default=["1000", "10000", "100000"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--rss-interval", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--build-jobs", type=int, default=2)
    args = parser.parse_args()
    try:
        epochs = _epoch_values(args.epochs)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.repeats <= 0 or args.build_jobs <= 0:
        parser.error("--repeats and --build-jobs must be positive")
    if not math.isfinite(args.rss_interval) or args.rss_interval <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--rss-interval and --timeout must be finite and positive")

    prefix = args.prefix.resolve()
    output = args.output_dir.resolve()
    metadata_path = prefix / "share/pycircuit/toolchain-metadata.json"
    required = (prefix / "bin/pycircuit", prefix / "share/pycircuit/cmake/pycircuitConfig.cmake", metadata_path)
    if not all(path.is_file() for path in required):
        parser.error(f"installed toolchain is incomplete: {prefix}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be empty or absent: {output}")
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if not cmake or not ninja or not cxx:
        parser.error("cmake, ninja, and a C++ compiler are required")

    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CMAKE_PREFIX_PATH"] = str(prefix)
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("git_sha") != head:
        parser.error("installed toolchain does not match checkout HEAD")
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    sizes = SMALL_SIZES if args.sizes == "small" else SELECTED_SIZES
    if args.sizes == "all":
        sizes = SELECTED_SIZES
    report: dict[str, Any] = {
        "status": "running",
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "repository": str(REPO),
            "head": head,
            "working_tree_status": status,
            "prefix": str(prefix),
            "toolchain_metadata": metadata,
            "installed_toolchain_metadata_sha256": _sha256(metadata_path),
            "platform": platform.platform(),
            "python": sys.version,
            "cmake": _version([cmake, "--version"], cwd=REPO, env=env),
            "ninja": _version([ninja, "--version"], cwd=REPO, env=env),
            "cxx_compiler": _version([cxx, "--version"], cwd=REPO, env=env),
            "build_jobs": args.build_jobs,
            "epochs": epochs,
            "repeats": args.repeats,
            "rss_limits": [
                "short-lived children may escape samples",
                "shared pages can be counted for more than one process",
                "detached/new-session processes are excluded",
            ],
            "throughput_boundary": "finite Structural C++ execution, not steady-state throughput",
        },
        "cases": [],
        "long_runs": [],
    }
    _write_json(output / "report.json", report)
    try:
        for axis, case_sizes in sizes.items():
            for size in case_sizes:
                _measure_case(
                    axis=axis,
                    size=size,
                    output=output,
                    prefix=prefix,
                    cmake=cmake,
                    cxx=cxx,
                    env=env,
                    interval=args.rss_interval,
                    timeout=args.timeout,
                    jobs=args.build_jobs,
                    epochs=epochs,
                    repeats=args.repeats,
                    report=report,
                )
        report["status"] = "complete"
        _write_json(output / "report.json", report)
        digest = _sha256(output / "report.json")
        (output / "report.developer.sha256").write_text(
            f"{digest}  report.json\n", encoding="utf-8"
        )
        print(f"resource report: {output / 'report.json'}\nSHA-256: {digest}")
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output / "report.json", report)
        print(f"resource measurement failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
