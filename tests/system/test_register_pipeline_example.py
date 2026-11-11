"""Public multi-unit example build and both current hardware backends."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.system
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/module_loop"
SOURCES = ("child", "module_loop")


def _prefix() -> Path:
    prefix = Path(
        os.environ.get("PYCIRCUIT_TEST_PREFIX", ROOT / ".pycircuit_out/toolchain/install")
    ).resolve()
    required = (
        prefix / "bin/pycircuit",
        prefix / "share/pycircuit/toolchain-metadata.json",
        prefix / "share/pycircuit/cmake/pycircuitConfig.cmake",
        *(prefix / "bin" / name for name in (
            "pycircuit-source-unit",
            "pycircuit-link",
            "pycircuit-emit",
            "pycircuit-opt",
        )),
    )
    assert all(path.is_file() for path in required), f"incomplete prefix: {prefix}"
    return prefix


def _run(
    command: list[str], *, cwd: Path, timeout: int = 900
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {command!r}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _commands(output: str) -> list[list[str]]:
    parsed: list[list[str]] = []
    for line in output.splitlines():
        try:
            parsed.append(shlex.split(line))
        except ValueError:
            continue
    return parsed


def _receipt(bundle: Path) -> dict[str, object]:
    receipt = json.loads((bundle / "generated.json").read_text(encoding="utf-8"))
    files = receipt["files"]
    assert isinstance(files, list)
    assert all((bundle / row["path"]).is_file() for row in files)
    return receipt


def test_multi_unit_public_build_and_both_backends(tmp_path: Path) -> None:
    prefix = _prefix()
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    ctest = shutil.which("ctest")
    assert cmake and ninja and ctest, "system test requires CMake, Ninja, and CTest"
    assert shutil.which("verilator"), "system test requires Verilator"

    source = tmp_path / "module_loop"
    shutil.copytree(EXAMPLE, source)
    build = tmp_path / "module-loop-build"
    _run(
        [
            cmake,
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={prefix}",
        ],
        cwd=tmp_path,
    )

    for stem in SOURCES:
        unit = build / "units" / stem
        query = _run(
            [ninja, "-C", str(build), "-t", "query", str(unit / f"{stem}.ac")],
            cwd=tmp_path,
        ).stdout
        assert str(source / f"{stem}.py") in query
        if stem == "module_loop":
            assert "units/child/child.interface.ac" in query
            assert "units/child/unit.json" in query
            assert "units/child/child.ac" not in query

    built = _run([ninja, "-C", str(build), "-v", "all"], cwd=tmp_path)
    compile_sources: Counter[str] = Counter()
    emit_targets: Counter[str] = Counter()
    for arguments in _commands(built.stdout):
        if "compile" in arguments:
            for index, argument in enumerate(arguments[:-1]):
                if argument == "-c" and arguments[index + 1].endswith(".py"):
                    compile_sources[Path(arguments[index + 1]).name] += 1
                    break
        if "emit" in arguments:
            emit_targets.update(
                target
                for target in ("structural-cpp", "verilog")
                if target in arguments
            )
    assert compile_sources == Counter({f"{stem}.py": 1 for stem in SOURCES})
    assert emit_targets == Counter({"structural-cpp": 1, "verilog": 1})

    final = build / "module_loop.ac"
    assert final.is_file()
    final_text = final.read_text(encoding="utf-8")
    assert final_text.count('callee = @pycircuit.__builtins__.dffe') == 1
    assert final_text.count('callee = @example_loop.child.Child') == 2

    receipts = {
        target: _receipt(build / target)
        for target in ("structural-cpp", "verilog")
    }
    expected_sources = {f"{stem}.py" for stem in SOURCES}
    for receipt in receipts.values():
        assert {
            group["source"]["path"] for group in receipt["source_groups"]
        } == expected_sources

    cpp_files = {row["path"] for row in receipts["structural-cpp"]["files"]}
    assert {"design.hpp", "source_unit_0.cpp", "source_unit_1.cpp"} <= cpp_files
    rtl_files = {
        row["path"]: row["role"] for row in receipts["verilog"]["files"]
    }
    assert rtl_files["design_top.sv"] == "rtl"
    assert rtl_files["ports.json"] == "runtime-glue"
    assert rtl_files["verilator_policy.json"] == "runtime-glue"
    assert sum(path.endswith(".sv") and role == "rtl" for path, role in rtl_files.items()) >= 3

    runner = build / "pycircuit_system"
    assert runner.is_file()
    _run([str(runner), "--config", str(source / "config.json")], cwd=tmp_path)
    _run(
        [ctest, "--test-dir", str(build), "--output-on-failure", "--no-tests=error"],
        cwd=tmp_path,
        timeout=1800,
    )
