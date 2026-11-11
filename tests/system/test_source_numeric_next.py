"""Finite numeric next-state range admission and stable serialization."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from pycircuit._source_capture import _capture_source_file
from pycircuit._source_transport import _emit_source_transport

pytestmark = pytest.mark.system


@dataclass(frozen=True)
class CompiledUnit:
    process: subprocess.CompletedProcess[str]
    body: Path
    interface: Path


def _harness() -> Path:
    candidates: list[str | None] = [os.environ.get("PYCIRCUIT_SOURCE_COMPILER")]
    toolchain = os.environ.get("PYC_TOOLCHAIN_ROOT")
    if toolchain:
        candidates.append(str(Path(toolchain) / "bin/pycircuit-source-unit"))
    candidates.append(shutil.which("pycircuit-source-unit"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise AssertionError("set PYCIRCUIT_SOURCE_COMPILER for numeric next tests")


def _compile(
    tmp_path: Path,
    stem: str,
    source_text: str,
    *,
    headers: tuple[Path, ...] = (),
) -> CompiledUnit:
    source_root = tmp_path / f"{stem}-source"
    source_root.mkdir()
    source = source_root / f"{stem}.py"
    source.write_text(source_text, encoding="utf-8")
    output = tmp_path / f"{stem}-out"
    output.mkdir()
    transport = output / f"{stem}.transport.mlir"
    body = output / f"{stem}.body.mlir"
    interface = output / f"{stem}.interface.mlir"
    transport.write_text(
        _emit_source_transport(_capture_source_file(source, source_root=source_root)),
        encoding="utf-8",
    )
    command = [
        str(_harness()),
        "--capture",
        str(transport),
        "--package",
        "numeric_next",
        "--path",
        source.name,
        "--body-out",
        str(body),
        "--interface-out",
        str(interface),
    ]
    for header in headers:
        command.extend(("--header", str(header)))
    return CompiledUnit(
        subprocess.run(command, text=True, capture_output=True, check=False),
        body,
        interface,
    )


def _counter(
    expression: str = "(state + 1) & 255",
    *,
    system_root: bool = False,
    statements: str | None = None,
) -> str:
    decorator = "system" if system_root else "module"
    body = statements if statements is not None else f"state = {expression}"
    indented_body = "".join(f"        {line}\n" for line in body.splitlines())
    return (
        "from typing import Annotated\n"
        f"from pycircuit import {decorator}, rule\n\n"
        "Word = Annotated[int, range(256)]\n\n"
        f"@{decorator}\n"
        "def Counter():\n"
        "    state: Word = 0\n\n"
        "    @rule\n"
        "    def advance() -> None:\n"
        "        nonlocal state\n"
        f"{indented_body}"
        "\n"
        "    advance()\n"
    )


def test_numeric_next_lowering_is_serialization_stable(tmp_path: Path) -> None:
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _compile(first_root, "counter", _counter())
    second = _compile(second_root, "counter", _counter())

    assert first.process.returncode == 0, first.process.stderr
    assert second.process.returncode == 0, second.process.stderr
    assert first.body.read_bytes() == second.body.read_bytes()


@pytest.mark.parametrize("expression", ["True", "256"])
def test_numeric_next_rejects_invalid_type_or_static_boundary(
    tmp_path: Path, expression: str
) -> None:
    unit = _compile(tmp_path, "invalid_next", _counter(expression))

    assert unit.process.returncode != 0
    assert not unit.body.exists()
    assert not unit.interface.exists()


def test_numeric_next_accepts_mask_within_declared_range(tmp_path: Path) -> None:
    # A seven-bit result fits the eight-bit register without implicit wrapping.
    unit = _compile(tmp_path, "bounded_mask", _counter("(state + 1) & 127"))

    assert unit.process.returncode == 0, unit.process.stderr
    assert unit.body.is_file()


def test_imported_non_power_of_two_alias_checks_declared_range(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "provider"
    consumer_root = tmp_path / "consumer"
    provider_root.mkdir()
    consumer_root.mkdir()
    provider = _compile(
        provider_root,
        "provider",
        "from typing import Annotated\nWord = Annotated[int, range(200)]\n",
    )
    assert provider.process.returncode == 0, provider.process.stderr
    consumer = _compile(
        consumer_root,
        "consumer",
        "from pycircuit import system, rule\n"
        "from .provider import Word\n\n"
        "@system\n"
        "def Counter():\n"
        "    state: Word = 198\n\n"
        "    @rule\n"
        "    def advance() -> None:\n"
        "        nonlocal state\n"
        "        state = state + 1\n\n"
        "    advance()\n",
        headers=(provider.interface,),
    )
    assert consumer.process.returncode == 0, consumer.process.stderr
    body = consumer.body.read_text(encoding="utf-8")
    assert re.search(
        r'#ac\.math_int<200>.*\n\s+%\d+ = "ac\.bits\.compare".*'
        r'predicate = "ult"',
        body,
    )
    assert 'mode = "trunc"' in body
    # One declaration in the rule contract and one published expectation.
    assert body.count('kind = "range"') == 2


def _tool(env: str, name: str) -> str:
    tool = os.environ.get(env) or shutil.which(name)
    assert tool and Path(tool).is_file(), f"set {env} or put {name} on PATH"
    return tool


def _toolchain_root() -> Path:
    configured = os.environ.get("PYC_TOOLCHAIN_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.append(_harness().parent.parent)
    for candidate in candidates:
        prefix = candidate.resolve()
        if (prefix / "share/pycircuit/cmake/pycircuitConfig.cmake").is_file():
            return prefix
    raise AssertionError("set PYC_TOOLCHAIN_ROOT to the installed test toolchain")


@pytest.mark.parametrize(
    ("expression", "statements", "expected_checks"),
    [
        ("state + 1", None, 1),
        ("(state + 1) & 511", None, 1),
        (
            "path-gated",
            "if state == 254:\n"
            "    state = state + 1\n"
            "else:\n"
            "    state = state + 2",
            2,
        ),
    ],
)
def test_numeric_next_dynamic_boundary_fails_without_commit(
    tmp_path: Path,
    expression: str,
    statements: str | None,
    expected_checks: int,
) -> None:
    # Mathematical addition gives 256 at Q=255; range checking rejects the entire
    # precommit before any narrowed value can be driven or transferred.
    unit = _compile(
        tmp_path,
        "dynamic_boundary",
        _counter(
            expression, system_root=True, statements=statements
        ).replace(
            "state: Word = 0", "state: Word = 254"
        ),
    )
    assert unit.process.returncode == 0, unit.process.stderr
    linker = _tool("PYCIRCUIT_LINKER", "pycircuit-link")
    design = tmp_path / "boundary.ac"
    linked = subprocess.run(
        [
            linker,
            "--body",
            str(unit.body),
            "--header",
            str(unit.interface),
            "--top",
            "numeric_next.dynamic_boundary.Counter",
            "--target",
            "final",
            "--output",
            str(design),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert linked.returncode == 0, linked.stderr
    final_ir = design.read_text(encoding="utf-8")
    # One occurrence in RequiredCheck and one in ac.expect per boundary.
    assert final_ir.count('kind = "range"') == expected_checks * 2
    assert 'predicate = "ult"' in final_ir
    assert 'mode = "trunc"' in final_ir
    generated = tmp_path / "boundary-cpp"
    environment = dict(os.environ)
    environment["PYCIRCUIT_LINKER"] = linker
    environment["PYCIRCUIT_EMITTER"] = _tool(
        "PYCIRCUIT_EMITTER", "pycircuit-emit"
    )
    environment["PYCIRCUIT_OPTIMIZER"] = _tool(
        "PYCIRCUIT_OPTIMIZER", "pycircuit-opt"
    )
    emitted = subprocess.run(
        [
            sys.executable,
            "-m",
            "pycircuit.cli",
            "emit",
            str(design),
            "--target",
            "structural-cpp",
            "-o",
            str(generated),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    assert (generated / "generated.json").is_file()

    # Test-only hierarchical probe. Derive the member from the emitted state
    # declaration; no fixed generated identifier or graph layout is asserted.
    model = generated / "design.hpp"
    text = model.read_text(encoding="utf-8")
    states = re.findall(r"gfsim::dffe<.*>\s+(\w+);", text)
    assert len(states) == 1, "generated model must have one owned register"
    model.write_text(text.replace("private:", "public:"), encoding="utf-8")
    state = states[0]
    driver = generated / "boundary_driver.cpp"
    driver.write_text(
        '#include "design.hpp"\n'
        "#include <iostream>\n"
        "int main() {\n"
        "  const auto bit = [](bool value) {\n"
        "    return gfsim::wire<gfsim::Bits<1>>::known(\n"
        "        gfsim::Bits<1>{value ? 1u : 0u});\n"
        "  };\n"
        "  pyc_dut dut;\n"
        "  dut.drive({bit(false), bit(false)});\n"
        "  dut.system().Build();\n"
        "  for (int run = 0; run < 2; ++run) {\n"
        "    dut.drive({bit(false), bit(false)});\n"
        "    dut.system().Reset();\n"
        f"    if (dut.root().{state}.currentState().q.value().value() != 254) return 1;\n"
        "    dut.drive({bit(true), bit(false)});\n"
        "    if (dut.system().Step() != gfsim::SimStepResult::Running) return 2;\n"
        f"    if (dut.root().{state}.currentState().q.value().value() != 255) return 3;\n"
        "    if (dut.system().cycle() != 1) return 4;\n"
        "    if (dut.system().Step() != gfsim::SimStepResult::Failed) return 5;\n"
        "    if (dut.system().cycle() != 1) return 6;\n"
        "    if (dut.system().failureInfo().phase != gfsim::SimFailurePhase::Check) return 7;\n"
        f"    if (dut.root().{state}.currentState().q.value().value() != 255) return 8;\n"
        "    if (dut.system().Step() != gfsim::SimStepResult::Failed) return 9;\n"
        f"    if (dut.root().{state}.currentState().q.value().value() != 255) return 10;\n"
        "  }\n"
        '  std::cout << "BOUNDARY_ZERO_COMMIT_OK\\n";\n'
        "}\n",
        encoding="utf-8",
    )
    with (generated / "CMakeLists.txt").open("a", encoding="utf-8") as cmake:
        cmake.write(
            "\nadd_executable(boundary_driver boundary_driver.cpp)\n"
            "target_compile_features(boundary_driver PRIVATE cxx_std_20)\n"
            "target_link_libraries(boundary_driver PRIVATE pycircuit_modules)\n"
        )
    build = tmp_path / "boundary-build"
    configured = subprocess.run(
        [
            "cmake",
            "-S",
            str(generated),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-Dpycircuit_DIR={_toolchain_root() / 'share/pycircuit/cmake'}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert configured.returncode == 0, configured.stdout + configured.stderr
    built = subprocess.run(
        ["cmake", "--build", str(build), "--target", "boundary_driver"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run(
        [str(build / "boundary_driver")], text=True, capture_output=True, check=False
    )
    assert ran.returncode == 0, f"{ran.stdout}\n{ran.stderr}\nexit={ran.returncode}"
    assert "BOUNDARY_ZERO_COMMIT_OK" in ran.stdout
