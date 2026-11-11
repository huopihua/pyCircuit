"""Fresh source build, backend inventory, and finite Structural C++ evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.system
ROOT = Path(__file__).resolve().parents[2]
MEASURER = ROOT / "tools/measure_build_resources.py"
INSTALL = Path(
    os.environ.get("PYCIRCUIT_TEST_PREFIX", ROOT / ".pycircuit_out/toolchain/install")
).resolve()


def _finite_nonnegative(value: object) -> None:
    assert type(value) in {int, float}
    assert math.isfinite(value)
    assert value >= 0


def _assert_rss(rss: dict[str, Any]) -> None:
    assert rss["unit"] == "KiB"
    assert type(rss["sample_count"]) is int and rss["sample_count"] >= 0
    if rss["available"]:
        assert rss["sample_count"] > 0
        _finite_nonnegative(rss["peak_sampled_sum_kib"])
        _finite_nonnegative(rss["peak_sampled_member_count"])
    else:
        assert rss["unavailable_reason"]


def _run_report(output: Path) -> dict[str, Any]:
    metadata = INSTALL / "share/pycircuit/toolchain-metadata.json"
    if not (INSTALL / "bin/pycircuit").is_file() or not metadata.is_file():
        pytest.fail(f"build and install this checkout before resource test: {INSTALL}")
    command = [
        sys.executable,
        str(MEASURER),
        "--prefix",
        str(INSTALL),
        "--output-dir",
        str(output),
        "--sizes",
        "small",
        "--epochs",
        "100",
        "200",
        "--repeats",
        "1",
        "--rss-interval",
        "0.05",
        "--timeout",
        "900",
        "--build-jobs",
        "2",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=1800,
    )
    assert result.returncode == 0, (
        f"resource measurement failed ({result.returncode}): {command!r}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return json.loads((output / "report.json").read_text(encoding="utf-8"))


def test_fresh_models_emit_both_backends_and_run_structural_cpp(
    tmp_path: Path,
) -> None:
    output = tmp_path / "resource-run"
    report = _run_report(output)
    metadata = report["metadata"]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert report["status"] == "complete"
    assert metadata["head"] == head
    assert metadata["prefix"] == str(INSTALL)
    assert metadata["toolchain_metadata"]["git_sha"] == head
    installed_metadata = INSTALL / "share/pycircuit/toolchain-metadata.json"
    assert metadata["installed_toolchain_metadata_sha256"] == hashlib.sha256(
        installed_metadata.read_bytes()
    ).hexdigest()
    checksum = (output / "report.developer.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    assert checksum == hashlib.sha256((output / "report.json").read_bytes()).hexdigest()
    assert "shared pages" in " ".join(metadata["rss_limits"])
    assert "detached/new-session" in " ".join(metadata["rss_limits"])
    assert "steady-state" in metadata["throughput_boundary"]
    assert len(report["cases"]) == 2

    for case in report["cases"]:
        assert case["case"]["source_units"] == 2
        assert case["case"]["definitions"] == 2
        assert len(case["source_inventory"]["source_units"]) == 2
        assert Path(case["source_inventory"]["linked_final"]).is_file()
        leaf = "leaf.py" if case["case"]["axis"] == "shared" else "leaf_0.py"
        expected_owners = {("measurement", leaf), ("measurement", "design_top.py")}
        for target, extension in (("structural-cpp", ".cpp"), ("verilog", ".sv")):
            inventory = case["backend_inventory"][target]
            assert {
                (row["source"]["package"], row["source"]["path"])
                for row in inventory["groups"]
            } == expected_owners
            assert inventory["implementation_file_count"] == 2
            backend_root = (
                output
                / f"{case['case']['axis']}-{case['case']['size']}"
                / "build"
                / target
            )
            receipt = json.loads(
                (backend_root / "generated.json").read_text(encoding="utf-8")
            )
            declared = {row["path"] for row in receipt["files"]}
            assert len(declared) == len(receipt["files"])
            assert all((backend_root / path).is_file() for path in declared)
            assert all(set(group["files"]) <= declared for group in receipt["source_groups"])
            assert all(
                len([path for path in group["files"] if path.endswith(extension)]) == 1
                for group in receipt["source_groups"]
            )
            assert {row["path"] for row in inventory["receipt_files"]} == declared

        phases = {phase["name"]: phase for phase in case["phases"]}
        assert {
            "source_compile_link",
            "emit_structural_cpp",
            "emit_verilog",
            "build_structural_cpp_translation_units",
            "compile_structural_cpp_consumer",
            "three_epoch_structural_cpp",
        } <= phases.keys()
        source_commands = phases["source_compile_link"]["executed_commands"]
        assert sum(row["kind"] == "pycircuit_compile" for row in source_commands) == 2
        assert any(row["kind"] == "pycircuit_link" for row in source_commands)
        assert any(
            row["kind"] == "pycircuit_emit_structural-cpp"
            for row in phases["emit_structural_cpp"]["executed_commands"]
        )
        assert any(
            row["kind"] == "pycircuit_emit_verilog"
            for row in phases["emit_verilog"]["executed_commands"]
        )
        model_commands = phases["build_structural_cpp_translation_units"][
            "executed_commands"
        ]
        assert sum(
            row["kind"] == "cpp_compile"
            and row["source"]
            and Path(row["source"]).name.startswith("source_unit_")
            for row in model_commands
        ) == 2
        smoke = case["runtime_smoke"]
        assert smoke["epochs"] == 3
        assert smoke["terminal_state"] == "COMPLETED"
        assert smoke["result_valid"] and smoke["reset_replay_valid"]
        for phase in case["phases"]:
            assert phase["exit_status"] == 0 and phase["timed_out"] is False
            _finite_nonnegative(phase["wall_seconds"])
            assert Path(phase["stdout_log"]).is_file()
            assert Path(phase["stderr_log"]).is_file()
            _assert_rss(phase["rss"])

    runs = [
        run
        for long_run in report["long_runs"]
        for measurement in long_run["measurements"]
        for run in measurement["runs"]
    ]
    assert len(runs) == 4
    assert {row["epochs"] for row in runs} == {100, 200}
    for row in runs:
        assert row["terminal_state"] == "COMPLETED"
        assert row["epoch_time"] == row["epochs"]
        assert row["result_valid"] and row["reset_replay_valid"]
        for key in (
            "setup_ns",
            "step_loop_ns",
            "reset_replay_ns",
            "process_wall_seconds_including_startup",
        ):
            _finite_nonnegative(row[key])
        for key in (
            "structural_cpp_epochs_per_second",
            "structural_cpp_replay_epochs_per_second",
            "amortized_epochs_per_second_including_process_startup",
        ):
            if row[key] is not None:
                _finite_nonnegative(row[key])
        assert row["exit_status"] == 0 and row["timed_out"] is False
        _assert_rss(row["process_rss"])
        assert Path(row["argv"][0]).resolve().is_relative_to(output.resolve())
        assert Path(row["stdout_log"]).is_file()
        assert Path(row["stderr_log"]).is_file()
