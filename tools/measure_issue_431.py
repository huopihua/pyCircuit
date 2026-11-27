"""Validate or run the locked 40-node WSE reproducer for issue #431."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "benchmarks/issue-431/wse40.lock.json"


def load_lock(path: Path = LOCK) -> dict:
    value = json.loads(path.read_text())
    required = {
        "schema",
        "repository",
        "revision",
        "directory",
        "build_script",
        "top_source",
        "top_source_materialization",
        "top_source_sha256",
        "compiler_revision",
        "python",
        "profile",
    }
    if set(value) != required or value["schema"] != (
        "pycircuit-issue-431-wse40-lock-v1"
    ):
        raise ValueError("issue #431 WSE lock has an unknown schema")
    profile = value["profile"]
    if value["python"] != {
        "implementation": "cpython",
        "version": "3.14.6",
    }:
        raise ValueError("issue #431 WSE Python runtime changed")
    if profile != {
        "nodes": 40,
        "early_slots": 2,
        "seen_slots": 56,
        "beat_bytes": 64,
        "model": "timed",
        "optimization": "O1",
        "build_parallelism": 1,
    }:
        raise ValueError("issue #431 WSE profile changed")
    for key in ("revision", "top_source_sha256", "compiler_revision"):
        if len(value[key]) != 40 and key != "top_source_sha256":
            raise ValueError(f"issue #431 lock has an invalid {key}")
        if key == "top_source_sha256" and len(value[key]) != 64:
            raise ValueError("issue #431 lock has an invalid top source digest")
    return value


def runtime_identity() -> dict:
    return {
        "implementation": sys.implementation.name,
        "version": ".".join(str(part) for part in sys.version_info[:3]),
        "executable": str(Path(sys.executable).resolve()),
    }


def verify_runtime(lock: dict) -> dict:
    actual = runtime_identity()
    expected = lock["python"]
    if actual["implementation"] != expected["implementation"]:
        raise ValueError(
            "issue #431 WSE requires "
            f"{expected['implementation']}, got {actual['implementation']}"
        )
    if actual["version"] != expected["version"]:
        raise ValueError(
            "issue #431 WSE requires CPython "
            f"{expected['version']}, got {actual['version']}"
        )
    return actual


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def source_candidate_identity(root: Path = ROOT) -> dict:
    root = root.resolve()
    try:
        commit = _git(root, "rev-parse", "HEAD").decode().strip()
        tracked_diff = _git(root, "diff", "--binary", "HEAD", "--")
        untracked = [
            Path(raw.decode())
            for raw in _git(
                root, "ls-files", "--others", "--exclude-standard", "-z"
            ).split(b"\0")
            if raw
        ]
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return {
            "kind": "unavailable",
            "root": str(root),
            "commit": None,
            "dirty": None,
            "content_identity": None,
        }
    untracked_hash = hashlib.sha256()
    for relative in sorted(untracked, key=lambda path: path.as_posix()):
        path = root / relative
        untracked_hash.update(relative.as_posix().encode())
        untracked_hash.update(b"\0")
        if path.is_symlink():
            untracked_hash.update(b"symlink\0")
            untracked_hash.update(os.readlink(path).encode())
        elif path.is_file():
            untracked_hash.update(b"file\0")
            untracked_hash.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            untracked_hash.update(b"other\0")
    facts = {
        "commit": commit,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_count": len(untracked),
        "untracked_sha256": untracked_hash.hexdigest(),
    }
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    return {
        "kind": "git-worktree",
        "root": str(root),
        **facts,
        "dirty": bool(tracked_diff or untracked),
        "content_identity": hashlib.sha256(encoded).hexdigest(),
    }


def installed_toolchain_identity(toolchain: Path) -> tuple[dict, dict]:
    metadata_path = toolchain / "share/pycircuit/toolchain-metadata.json"
    raw_metadata = metadata_path.read_bytes()
    metadata = json.loads(raw_metadata)
    entrypoints = {}
    for name in (
        "pycircuit",
        "pycircuit-source-unit",
        "pycircuit-link",
        "pycircuit-emit",
    ):
        path = toolchain / "bin" / name
        if path.is_file():
            entrypoints[f"bin/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return metadata, {
        "root": str(toolchain),
        "metadata": metadata,
        "metadata_sha256": hashlib.sha256(raw_metadata).hexdigest(),
        "entrypoint_sha256": entrypoints,
    }


def candidate_binding(source: dict, metadata: dict) -> dict:
    commit_matches = source.get("commit") is not None and source.get(
        "commit"
    ) == metadata.get("git_sha")
    clean = source.get("dirty") is False
    return {
        "commit_matches": commit_matches,
        "source_clean": clean,
        "status": (
            "exact-clean-commit"
            if commit_matches and clean
            else (
                "commit-matches-dirty-source"
                if commit_matches
                else "installed-toolchain-differs-from-source"
            )
        ),
    }


def fetch(lock: dict, cache: Path) -> Path:
    destination = cache / lock["revision"]
    if destination.is_dir():
        return destination
    if destination.exists():
        raise ValueError("issue #431 cache destination is not a directory")
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{lock['revision']}.tar.gz"
    url = f"{lock['repository']}/archive/{lock['revision']}.tar.gz"
    subprocess.run(
        ["curl", "--fail", "--location", "--output", str(archive), url],
        check=True,
    )
    with tarfile.open(archive, "r:gz") as source:
        roots = {member.name.split("/", 1)[0] for member in source.getmembers()}
        if len(roots) != 1:
            raise ValueError("issue #431 archive has an unexpected root")
        source.extractall(cache, filter="data")
    extracted = cache / roots.pop()
    extracted.rename(destination)
    return destination


def verify_materialized_source(lock: dict, checkout: Path, output: Path) -> None:
    design = checkout / lock["directory"]
    script = design / lock["top_source_materialization"]
    source = design / lock["top_source"]
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; from fixed_timed_ledger import materialize; "
            "materialize(Path(__import__('sys').argv[1]), "
            "Path(__import__('sys').argv[2]))"
        ),
        str(source),
        str(output),
    ]
    subprocess.run(command, cwd=script.parent, check=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    if digest != lock["top_source_sha256"]:
        raise ValueError("issue #431 materialized top source digest changed")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_rtl_closure(
    checkout: Path,
    build: Path,
    output: Path,
    toolchain: Path,
    timeout: int,
) -> dict:
    """Compile and execute the locked WSE RTL with the large-design backend."""
    script = checkout / "tools/verify_machine_rtl.py"
    if not script.is_file():
        raise ValueError("issue #431 checkout lacks the locked RTL verifier")
    verifier = output / "rtl-verifier"
    verifier.mkdir(parents=True)
    for source in (script, script.with_name("machine_cases.py")):
        shutil.copy2(source, verifier / source.name)
    path = verifier / "verify_machine_rtl.py"
    text = path.read_text()
    old = '''        compiler, simulator = shutil.which("iverilog"), shutil.which("vvp")
        if not compiler or not simulator:
            raise ValueError("Icarus Verilog and vvp are required")
        binary = output / "machine.vvp"
        run("compile", [compiler, "-g2012", "-s", "tb", "-o", binary, *rtl, *primitives, bench])
'''
    new = '''        compiler = shutil.which("verilator")
        if not compiler:
            raise ValueError("Verilator is required for the large WSE RTL closure")
        directory = output / "verilator"
        binary = directory / "Vtb"
        run("compile", [compiler, "--binary", "--timing", "-Wno-fatal",
                        "--top-module", "tb", "--Mdir", directory,
                        "-j", "2", "-DPYCIRCUIT_FAST_SIM",
                        *rtl, *primitives, bench])
        simulator = binary
'''
    if text.count(old) != 1:
        raise ValueError("locked RTL verifier backend boundary changed")
    path.write_text(
        text.replace(old, new).replace(
            '"backend": "iverilog"', '"backend": "verilator"'
        )
    )
    destination = output / "rtl-closure"
    command = [
        sys.executable,
        str(path),
        "--build",
        str(build),
        "--toolchain",
        str(toolchain),
        "--output",
        str(destination),
        "--timeout",
        str(timeout),
    ]
    started = time.monotonic()
    result = subprocess.run(command, check=False)
    elapsed = time.monotonic() - started
    verification = destination / "verification.json"
    payload = json.loads(verification.read_text()) if verification.is_file() else {}
    compiled = [
        row for row in payload.get("commands", []) if row.get("label") == "compile"
    ]
    executed = payload.get("runs", [])
    status = (
        "PASS"
        if result.returncode == 0
        and payload.get("status") == "PASS"
        and len(compiled) == 1
        and compiled[0].get("exit_code") == 0
        and executed
        and all(row.get("status") == "PASS" for row in executed)
        else "FAIL"
    )
    return {
        "backend": "verilator",
        "status": status,
        "returncode": result.returncode,
        "compile_seconds": compiled[0].get("seconds") if compiled else None,
        "simulation_seconds": sum(
            row.get("seconds", 0)
            for row in payload.get("commands", [])
            if row.get("label") != "compile"
        ),
        "elapsed_seconds": elapsed,
        "simulation_ran": bool(executed),
        "passed_runs": len(executed),
        "receipt": (
            str(verification.relative_to(output)) if verification.is_file() else None
        ),
        "receipt_sha256": digest(verification) if verification.is_file() else None,
    }


def run(
    lock: dict,
    checkout: Path,
    output: Path,
    *,
    rtl_closure: bool = False,
    timeout: int = 1800,
) -> int:
    python = verify_runtime(lock)
    output.mkdir(parents=True, exist_ok=False)
    verify_materialized_source(lock, checkout, output / "timed_machine.py")
    locked_design = checkout / lock["directory"]
    design = output / "source"
    shutil.copytree(locked_design, design)
    toolchain = Path(os.environ["PYC_TOOLCHAIN_ROOT"]).resolve()
    metadata, installed_toolchain = installed_toolchain_identity(toolchain)
    source_candidate = source_candidate_identity()
    pin_path = design / "models/machine/timed-pycircuit.lock.json"
    pin = json.loads(pin_path.read_text())
    locked_pin = dict(pin)
    pin.update(revision=metadata["git_sha"], llvm_mlir=metadata["llvm_version"])
    pin_path.write_text(json.dumps(pin, indent=2) + "\n")
    arguments = [
        str(design / lock["build_script"]),
        "--model",
        lock["profile"]["model"],
        "--optimization",
        lock["profile"]["optimization"],
        "--emit-verilog",
        "--output",
        str(output / "build"),
    ]
    receipt = {
        "schema": lock["schema"],
        "command": [sys.executable, *arguments],
        "status": "RUNNING",
        "python": python,
        "source_candidate": source_candidate,
        "installed_toolchain": installed_toolchain,
        "candidate_binding": candidate_binding(source_candidate, metadata),
        "native_metrics": [],
        "rtl_closure": {
            "backend": "verilator",
            "status": "NOT RUN",
            "simulation_ran": False,
        },
        "compiler_overlay": {
            "path": str(pin_path.relative_to(output)),
            "locked": locked_pin,
            "measured": pin,
        },
    }
    path = output / "measurement.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    metrics = output / "metrics"
    metrics.mkdir()
    script = design / lock["build_script"]
    sys.path.insert(0, str(script.parent))
    specification = importlib.util.spec_from_file_location(
        "issue_431_locked_builder", script
    )
    if specification is None or specification.loader is None:
        raise ValueError("cannot load issue #431 locked build script")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    original_run = module.subprocess.run
    counter = 0

    def measured_run(command, *args, **kwargs):
        nonlocal counter
        command = [str(part) for part in command]
        executable = Path(command[0]).name
        if executable in {
            "python",
            "python3",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        }:
            command[0] = sys.executable
        elif executable == "pycircuit":
            command = [
                sys.executable,
                str(toolchain / "bin/pycircuit"),
                *command[1:],
            ]
        if (
            len(command) > 2
            and Path(command[0]).resolve() == Path(sys.executable).resolve()
            and Path(command[1]).name == "pycircuit"
            and command[2]
            in {
                "compile",
                "link",
                "emit",
            }
        ):
            target = ""
            if command[2] == "emit" and "--target" in command:
                target = "-" + command[command.index("--target") + 1]
            report = metrics / f"{counter:02d}-{command[2]}{target}.json"
            counter += 1
            command += ["--metrics-out", str(report)]
            receipt["native_metrics"].append(str(report.relative_to(output)))
            path.write_text(json.dumps(receipt, indent=2) + "\n")
        environment = os.environ.copy()
        environment.update(kwargs.get("env", {}))
        environment["PATH"] = os.pathsep.join(
            [
                str(Path(sys.executable).resolve().parent),
                str(toolchain / "bin"),
                environment.get("PATH", ""),
            ]
        )
        kwargs["env"] = environment
        return original_run(command, *args, **kwargs)

    module.subprocess.run = measured_run
    prior_argv = sys.argv
    prior_cwd = Path.cwd()
    returncode = 1
    try:
        sys.argv = arguments
        __import__("os").chdir(design)
        returncode = int(module.main())
    except (RuntimeError, ValueError) as error:
        receipt["error"] = str(error)
    finally:
        __import__("os").chdir(prior_cwd)
        sys.argv = prior_argv
        module.subprocess.run = original_run
        sys.path.remove(str(script.parent))
    stage_reports = []
    for report in sorted(metrics.glob("*.json")):
        payload = json.loads(report.read_text())
        stage_reports.append(
            {
                "path": str(report.relative_to(output)),
                "status": payload.get("status"),
                "timing": payload.get("timing", {}),
                "phases": [phase.get("name") for phase in payload.get("phases", [])],
            }
        )
    receipt["stage_reports"] = stage_reports
    if returncode == 0 and rtl_closure:
        receipt["rtl_closure"] = run_rtl_closure(
            design, output / "build", output, toolchain, timeout
        )
        if receipt["rtl_closure"]["status"] != "PASS":
            returncode = 1
    receipt.update(status="PASS" if returncode == 0 else "FAIL", returncode=returncode)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--rtl-closure",
        action="store_true",
        help="compile and execute the large WSE RTL with Verilator",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    lock = load_lock(args.lock)
    if not args.fetch and not args.run:
        print(json.dumps(lock, indent=2))
        return 0
    if args.cache is None:
        parser.error("--fetch/--run requires --cache")
    checkout = fetch(lock, args.cache.resolve())
    if not args.run:
        return 0
    if args.output is None:
        parser.error("--run requires --output")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return run(
        lock,
        checkout,
        args.output.resolve(),
        rtl_closure=args.rtl_closure,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
