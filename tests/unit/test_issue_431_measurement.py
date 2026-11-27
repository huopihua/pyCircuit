"""The issue #431 WSE reproducer stays immutable and opt-in."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_wse40_lock_is_closed_and_current():
    spec = importlib.util.spec_from_file_location(
        "measure_issue_431", ROOT / "tools/measure_issue_431.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock = module.load_lock()
    assert lock["revision"] == "c0bc7088a68e9e0693c2f26ee4f6776f19c67bea"
    assert lock["profile"]["nodes"] == 40
    assert lock["profile"]["build_parallelism"] == 1
    assert lock["python"] == {
        "implementation": "cpython",
        "version": "3.14.6",
    }
    assert lock["top_source_sha256"] == (
        "30a641aae00bf4616b97278bfe2a8db6c88a36bca5d9f47fab7d1082c022ed19"
    )


def test_source_candidate_identity_distinguishes_dirty_content(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "measure_issue_431_identity", ROOT / "tools/measure_issue_431.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(
        ["git", "-C", tmp_path, "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", tmp_path, "config", "user.name", "Test"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n")
    subprocess.run(["git", "-C", tmp_path, "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", tmp_path, "commit", "-qm", "base"], check=True)
    clean = module.source_candidate_identity(tmp_path)
    assert clean["dirty"] is False
    tracked.write_text("changed\n")
    (tmp_path / "new.txt").write_text("new\n")
    dirty = module.source_candidate_identity(tmp_path)
    assert dirty["dirty"] is True
    assert dirty["untracked_count"] == 1
    assert dirty["content_identity"] != clean["content_identity"]


def test_runtime_lock_requires_exact_patch_version(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "measure_issue_431_runtime", ROOT / "tools/measure_issue_431.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock = module.load_lock()
    monkeypatch.setattr(
        module,
        "runtime_identity",
        lambda: {
            "implementation": "cpython",
            "version": "3.14.5",
            "executable": "/python",
        },
    )
    with pytest.raises(ValueError, match="requires CPython 3.14.6"):
        module.verify_runtime(lock)


def test_rtl_closure_requires_compile_and_simulation(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "measure_issue_431_rtl", ROOT / "tools/measure_issue_431.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkout = tmp_path / "checkout"
    tools = checkout / "tools"
    tools.mkdir(parents=True)
    verifier = tools / "verify_machine_rtl.py"
    verifier.write_text(
        '''        compiler, simulator = shutil.which("iverilog"), shutil.which("vvp")
        if not compiler or not simulator:
            raise ValueError("Icarus Verilog and vvp are required")
        binary = output / "machine.vvp"
        run("compile", [compiler, "-g2012", "-s", "tb", "-o", binary, *rtl, *primitives, bench])
receipt = {"backend": "iverilog"}
'''
    )
    (tools / "machine_cases.py").write_text("CASES = ()\n")

    def fake_run(command, check=False):
        del check
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir()
        (destination / "verification.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "commands": [
                        {"label": "compile", "exit_code": 0, "seconds": 1.25},
                        {"label": "case", "exit_code": 0, "seconds": 0.75},
                    ],
                    "runs": [{"case": "case", "status": "PASS"}],
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module.run_rtl_closure(
        checkout, tmp_path / "build", tmp_path / "out", tmp_path / "toolchain", 10
    )
    assert result["status"] == "PASS"
    assert result["backend"] == "verilator"
    assert result["compile_seconds"] == 1.25
    assert result["simulation_seconds"] == 0.75
    assert result["simulation_ran"] is True
    assert result["passed_runs"] == 1
