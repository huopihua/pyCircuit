"""Approved control-flow Python source programs and importer inventories."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from pycircuit._source_capture import _capture_source_file
from pycircuit._source_transport import _emit_source_transport

pytestmark = pytest.mark.system


TYPES = """\
from pycircuit import struct, u8

@struct
class IncrementResult:
    outgoing: u8

@struct
class PipelineResult:
    outgoing: u8
"""

INCREMENT = """\
from pycircuit import module, rule, u1, u8
from .types import IncrementResult

@module
def Increment(enabled: u1, incoming: u8) -> IncrementResult:
    outgoing: u8 = 0

    @rule
    def update() -> IncrementResult:
        if enabled and incoming != 0:
            outgoing = (incoming + 1) & 255
        return IncrementResult(outgoing=outgoing)

    return update()
"""

PIPELINE = """\
from pycircuit import module, u1, u8
from .increment import Increment
from .types import PipelineResult

@module
def Pipeline(enabled: u1, incoming: u8) -> PipelineResult:
    first = Increment(enabled, incoming)
    second = Increment(enabled, first.outgoing)
    return PipelineResult(outgoing=second.outgoing)
"""

TEST_INCREMENT = """\
from pycircuit import system, rule, log, report, u1, u3, u8
from .increment import Increment

@rule
def transfer_output(outgoing, child_outgoing) -> None:
    outgoing = child_outgoing

@system
def TestIncrement():
    enabled: u1 = True
    incoming: u8 = 1
    outgoing: u8 = 0
    phase: u3 = 0
    dut = Increment(enabled, incoming)
    transfer_output(outgoing, dut.outgoing)

    @rule
    def fixture() -> None:
        nonlocal enabled, incoming, phase
        if phase == 0:
            assert outgoing == 0, "reset output"
            incoming = 4
        elif phase == 1:
            assert outgoing == 2, "first input"
            incoming = 0
        elif phase == 2:
            assert outgoing == 5, "second input"
        elif phase == 3:
            assert outgoing == 5, "disabled input holds"
            enabled = False
        elif phase == 4:
            assert outgoing == 5, "final output"
        if phase < 4:
            phase = phase + 1

    fixture()
"""

TEST_PIPELINE = """\
from pycircuit import system, rule, log, report, u1, u3, u8
from .pipeline import Pipeline

@rule
def transfer_output(outgoing, child_outgoing) -> None:
    outgoing = child_outgoing

@system
def TestPipeline():
    enabled: u1 = True
    incoming: u8 = 1
    outgoing: u8 = 0
    phase: u3 = 0
    dut = Pipeline(enabled, incoming)
    transfer_output(outgoing, dut.outgoing)

    @rule
    def fixture() -> None:
        nonlocal enabled, incoming, phase
        if phase == 0:
            assert outgoing == 0, "reset output"
            incoming = 4
        elif phase == 1:
            assert outgoing == 0, "first stage only"
            incoming = 0
        elif phase == 2:
            assert outgoing == 3, "first end-to-end result"
        elif phase == 3:
            assert outgoing == 6, "second end-to-end result"
            enabled = False
        elif phase == 4:
            assert outgoing == 6, "final output"
        if phase < 4:
            phase = phase + 1

    fixture()
"""

SHORT_CIRCUIT = """\
from typing import Annotated
from pycircuit import rule, system
Word = Annotated[int, range(256)]
@system
def ShortCircuit():
    enabled: bool = True
    incoming: Word = 1
    outgoing: Word = 0
    @rule
    def update() -> None:
        nonlocal outgoing
        if enabled and incoming != 0:
            outgoing = incoming
    update()
"""

LIVE_PATH_ASSERT = """\
from typing import Annotated
from pycircuit import rule, system
Word = Annotated[int, range(256)]
@system
def LivePathAssert():
    ready: bool = True
    source: Word = 7
    sink: Word = 0
    @rule
    def guarded() -> None:
        nonlocal sink
        cached = source
        if ready:
            assert cached == 7, "guarded value"
            sink = cached
    guarded()
"""

EXCLUSIVE_WRITES = """\
from typing import Annotated
from pycircuit import rule, system
Word = Annotated[int, range(256)]
@system
def ExclusiveWrites():
    choose_left: bool = True
    left: Word = 3
    right: Word = 5
    sink: Word = 0
    @rule
    def choose() -> None:
        nonlocal sink
        if choose_left:
            sink = left
        else:
            sink = right
    choose()
"""

COMPOSITION_AUTHORITY = """\
from typing import Annotated
from pycircuit import rule, system
Word = Annotated[int, range(256)]
@system
def CompositionAuthority():
    guard_a: bool = True
    left: Word = 1
    right: Word = 2
    sink: Word = 0
    @rule
    def choose() -> None:
        nonlocal sink
        if guard_a and left < 4:
            assert left != 0, "left is live"
            sink = (left + 1) & 255
        else:
            assert right != 0, "right is live"
            sink = (right + 1) & 255
    choose()
"""

RUNTIME_RUNTIME_COMPARE = """\
from typing import Annotated
from pycircuit import rule, system
Word = Annotated[int, range(256)]
@system
def RuntimeRuntimeCompare():
    left: Word = 1
    right: Word = 2
    sink: Word = 0
    @rule
    def compare() -> None:
        nonlocal sink
        if left < right:
            sink = left
    compare()
"""


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
    raise AssertionError("set PYCIRCUIT_SOURCE_COMPILER for control-flow source tests")


def _compile(
    source_root: Path,
    output_root: Path,
    relative: str,
    source_text: str,
    *,
    headers: tuple[Path, ...] = (),
) -> CompiledUnit:
    source = source_root / relative
    source.write_text(source_text, encoding="utf-8")
    stem = relative.replace("/", "_").replace(".", "_")
    transport = output_root / f"{stem}.transport.mlir"
    body = output_root / f"{stem}.body.mlir"
    interface = output_root / f"{stem}.interface.mlir"
    transport.write_text(
        _emit_source_transport(_capture_source_file(source, source_root=source_root)),
        encoding="utf-8",
    )
    command = [
        str(_harness()),
        "--capture",
        str(transport),
        "--package",
        "control_flow",
        "--path",
        relative,
    ]
    for header in headers:
        command.extend(("--header", str(header)))
    command.extend(("--body-out", str(body), "--interface-out", str(interface)))
    return CompiledUnit(
        subprocess.run(command, text=True, capture_output=True, check=False),
        body,
        interface,
    )


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    return source, output


def _operation_lines(text: str, operation: str) -> list[str]:
    quoted = f'"{operation}"'
    custom = re.compile(rf"(?:^|\s){re.escape(operation)}(?:\s|\()")
    return [
        line.strip()
        for line in text.splitlines()
        if quoted in line or custom.search(line)
    ]


def _operation_operands(line: str, operation: str) -> list[str]:
    match = re.search(rf'"{re.escape(operation)}"\(([^)]*)\)', line)
    assert match is not None, line
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def _dictionary_attr(line: str, name: str) -> str:
    marker = f"{name} = "
    start = line.index(marker) + len(marker)
    assert line[start] == "{", line
    depth = 0
    for index in range(start, len(line)):
        if line[index] == "{":
            depth += 1
        elif line[index] == "}":
            depth -= 1
            if depth == 0:
                return line[start : index + 1]
    raise AssertionError(f"unterminated {name}: {line}")


def _compile_types(source: Path, output: Path) -> CompiledUnit:
    unit = _compile(source, output, "types.py", TYPES)
    assert unit.process.returncode == 0, unit.process.stderr
    return unit


def _compile_increment(source: Path, output: Path, types: CompiledUnit) -> CompiledUnit:
    unit = _compile(
        source, output, "increment.py", INCREMENT, headers=(types.interface,)
    )
    assert unit.process.returncode == 0, unit.process.stderr
    return unit


def _assert_increment_inventory(text: str) -> None:
    assert text.count('"ac.module"') == 1
    assert text.count('callee = @pycircuit.__builtins__.dffe') == 1
    assert text.count('"ac.rule"') == 1
    assert 'predicate = "ne"' in text
    assert 'opcode = "add"' in text
    assert 'opcode = "and"' in text
    assert text.count('"ac.bits.select"') == 1
    assert '!ac.struct<"control_flow.types.IncrementResult">' in text


def _assert_fixture_inventory(text: str, *, pipeline: bool) -> None:
    assert 'ac.root_kind = "system"' in text
    assert text.count('callee = @pycircuit.__builtins__.dffe') == 4
    assert text.count('"ac.instance"') == 5
    assert text.count('"ac.rule"') == 2
    checks = _operation_lines(text, "ac.expect")
    assert sum('kind = "assert"' in check for check in checks) == 5
    assert text.count("ac.required_checks =") == 1
    assert "#ac.math_int<3>" in text
    assert 'instance_name = "__pyc_call_0"' in text
    callee = (
        "@control_flow.pipeline.Pipeline"
        if pipeline
        else "@control_flow.increment.Increment"
    )
    assert f"callee = {callee}" in text


def _compile_inline(tmp_path: Path, stem: str, source_text: str) -> CompiledUnit:
    source, output = _workspace(tmp_path)
    return _compile(source, output, f"{stem}.py", source_text)


def test_short_circuit_guard_preserves_left_to_right_path(tmp_path: Path) -> None:
    unit = _compile_inline(tmp_path, "short_circuit", SHORT_CIRCUIT)

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    # Current source bodies pass each live register value into the rule once;
    # they no longer materialize a separate ac.source.read for every use.  The
    # dataflow itself is the stable short-circuit contract: compare incoming,
    # combine that predicate with enabled in left-to-right operand order, and
    # use the combined guard to select incoming as outgoing's next value.
    rule = _operation_lines(body, "ac.rule")
    assert len(rule) == 1
    assert len(_operation_operands(rule[0], "ac.rule")) == 5

    compares = _operation_lines(body, "ac.bits.compare")
    assert len(compares) == 1
    assert 'predicate = "ne"' in compares[0]
    compare_result = compares[0].split(" = ", 1)[0]
    compare_operands = _operation_operands(compares[0], "ac.bits.compare")

    guards = [
        line
        for line in _operation_lines(body, "ac.bits.binary")
        if f'({compare_result})' not in line
        and f", {compare_result})" in line
        and 'opcode = "and"' in line
    ]
    assert len(guards) == 1
    guard_result = guards[0].split(" = ", 1)[0]
    guard_operands = _operation_operands(guards[0], "ac.bits.binary")
    assert guard_operands[1] == compare_result
    assert guard_operands[0] != compare_operands[0]

    next_values = [
        line
        for line in _operation_lines(body, "ac.bits.select")
        if _operation_operands(line, "ac.bits.select")[:2]
        == [guard_result, compare_operands[0]]
    ]
    assert len(next_values) == 1
    assert body.index(compares[0]) < body.index(guards[0]) < body.index(next_values[0])
    assert body.count('"ac.yield"') >= 2


def test_assertion_uses_the_same_live_branch_path(tmp_path: Path) -> None:
    unit = _compile_inline(tmp_path, "live_path_assert", LIVE_PATH_ASSERT)

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    checks = _operation_lines(body, "ac.expect")
    assert sum('kind = "assert"' in check for check in checks) == 1
    assert sum('kind = "range"' in check for check in checks) == 1
    assert body.count('"ac.bits.compare"') == 1
    assert 'predicate = "eq"' in body
    assert body.count('"ac.bits.select"') >= 1
    assert body.count("ac.required_checks =") == 1
    assert "#ac.math_int<7>" in body


def test_mutually_exclusive_branch_writes_remain_two_contributions(
    tmp_path: Path,
) -> None:
    unit = _compile_inline(tmp_path, "exclusive_writes", EXCLUSIVE_WRITES)

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    value_selects = [
        line
        for line in _operation_lines(body, "ac.bits.select")
        if line.count("#ac.math_int<8>") == 3
    ]
    assert len(value_selects) == 1
    condition, left, right = _operation_operands(
        value_selects[0], "ac.bits.select"
    )
    assert len({condition, left, right}) == 3
    checks = _operation_lines(body, "ac.expect")
    assert sum('kind = "range"' in check for check in checks) == 2
    assert body.count('"ac.yield"') >= 2


def test_composition_preserves_proof_local_inputs_and_negative_polarity(
    tmp_path: Path,
) -> None:
    unit = _compile_inline(tmp_path, "composition_authority", COMPOSITION_AUTHORITY)

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    checks = _operation_lines(body, "ac.expect")
    assert sum('kind = "assert"' in check for check in checks) == 2
    assert sum('kind = "range"' in check for check in checks) == 2
    assert body.count("ac.required_checks =") == 1
    assert 'predicate = "ult"' in body
    assert body.count('predicate = "ne"') == 2
    assert body.count('opcode = "add"') == 2
    assert body.count('"ac.bits.select"') == 3
    branch_value_selects = [
        line
        for line in _operation_lines(body, "ac.bits.select")
        if line.count("#ac.math_int<8>") == 3
    ]
    assert len(branch_value_selects) == 1
    condition, left, right = _operation_operands(
        branch_value_selects[0], "ac.bits.select"
    )
    assert len({condition, left, right}) == 3


def test_runtime_runtime_compare_is_preserved_in_hardware_dataflow(
    tmp_path: Path,
) -> None:
    unit = _compile_inline(tmp_path, "runtime_runtime_compare", RUNTIME_RUNTIME_COMPARE)

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    compares = _operation_lines(body, "ac.bits.compare")
    assert len(compares) == 1
    assert 'predicate = "ult"' in compares[0]
    left, right = _operation_operands(compares[0], "ac.bits.compare")
    assert left != right
    value_selects = [
        line
        for line in _operation_lines(body, "ac.bits.select")
        if line.count("#ac.math_int<8>") == 3
    ]
    assert len(value_selects) == 1
    assert _operation_operands(value_selects[0], "ac.bits.select")[0] == (
        compares[0].split(" = ", 1)[0]
    )
    checks = _operation_lines(body, "ac.expect")
    assert sum('kind = "range"' in check for check in checks) == 1


@pytest.mark.parametrize(
    ("stem", "condition", "update", "predicate", "opcode"),
    [
        ("le", "value <= 4", "(value + 1) & 255", "ule", "add"),
        ("gt", "value > 0", "(value + 1) & 255", "ugt", "add"),
        ("ge", "value >= 1", "(value + 1) & 255", "uge", "add"),
        ("sub", "value < 4", "(value - 1) & 255", "ult", "sub"),
    ],
)
def test_composition_relations_and_arithmetic_lower_to_hardware(
    tmp_path: Path,
    stem: str,
    condition: str,
    update: str,
    predicate: str,
    opcode: str,
) -> None:
    source_text = f"""\
from typing import Annotated
from pycircuit import rule, system
Word = Annotated[int, range(256)]
@system
def Mapping():
    value: Word = 3
    sink: Word = 0
    @rule
    def update() -> None:
        nonlocal sink
        if {condition}:
            sink = {update}
    update()
"""
    unit = _compile_inline(tmp_path, f"composition_{stem}", source_text)

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert f'predicate = "{predicate}"' in body
    assert f'opcode = "{opcode}"' in body
    assert 'opcode = "and"' in body
    checks = _operation_lines(body, "ac.expect")
    assert sum('kind = "range"' in check for check in checks) == 1
    assert body.count("ac.required_checks =") == 1


def test_approved_sources_publish_complete_semantic_inventory(
    tmp_path: Path,
) -> None:
    source, output = _workspace(tmp_path)
    types = _compile_types(source, output)
    increment = _compile_increment(source, output, types)
    system = _compile(
        source,
        output,
        "test_increment.py",
        TEST_INCREMENT,
        headers=(types.interface, increment.interface),
    )

    assert system.process.returncode == 0, system.process.stderr
    _assert_increment_inventory(increment.body.read_text(encoding="utf-8"))
    _assert_fixture_inventory(system.body.read_text(encoding="utf-8"), pipeline=False)


def test_approved_sources_publish_hierarchy_without_relay_registers(
    tmp_path: Path,
) -> None:
    source, output = _workspace(tmp_path)
    types = _compile_types(source, output)
    increment = _compile_increment(source, output, types)
    pipeline = _compile(
        source,
        output,
        "pipeline.py",
        PIPELINE,
        headers=(types.interface, increment.interface),
    )
    assert pipeline.process.returncode == 0, pipeline.process.stderr
    system = _compile(
        source,
        output,
        "test_pipeline.py",
        TEST_PIPELINE,
        headers=(types.interface, increment.interface, pipeline.interface),
    )

    assert system.process.returncode == 0, system.process.stderr
    pipeline_text = pipeline.body.read_text(encoding="utf-8")
    assert pipeline_text.count('callee = @pycircuit.__builtins__.dffe') == 0
    assert pipeline_text.count('"ac.instance"') == 2
    assert pipeline_text.count("callee = @control_flow.increment.Increment") == 2
    assert 'instance_name = "__pyc_call_0"' in pipeline_text
    assert 'instance_name = "__pyc_call_1"' in pipeline_text
    assert '!ac.struct<"control_flow.types.PipelineResult">' in pipeline_text
    _assert_fixture_inventory(system.body.read_text(encoding="utf-8"), pipeline=True)
