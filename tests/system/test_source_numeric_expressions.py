"""Python-to-ACIR contracts for the bounded value numeric bridge."""

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
    raise AssertionError("set PYCIRCUIT_SOURCE_COMPILER for numeric source tests")


def _compile(
    tmp_path: Path,
    stem: str,
    source_text: str,
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
        "numeric",
        "--path",
        source.name,
        "--body-out",
        str(body),
        "--interface-out",
        str(interface),
    ]
    return CompiledUnit(
        subprocess.run(command, text=True, capture_output=True, check=False),
        body,
        interface,
    )


def _module(
    expression: str,
    *,
    parameters: str = "current: Word",
    statements: tuple[str, ...] = (),
    word_upper: int = 16,
) -> str:
    rule_lines = [f"temporary = {expression}", *statements, "return"]
    return (
        "from typing import Annotated\n"
        "from pycircuit import module, rule\n\n"
        f"Word = Annotated[int, range({word_upper})]\n\n"
        "@module\n"
        f"def NumericInput({parameters}):\n"
        "    @rule\n"
        "    def calculate():\n"
        + "".join(f"        {line}\n" for line in rule_lines)
        + "\n"
        "    calculate()\n"
    )


def _operation_lines(text: str, operation: str) -> list[str]:
    quoted = f'"{operation}"'
    custom = re.compile(rf"(?:^|\s){re.escape(operation)}(?:\s|\()")
    return [
        line.strip()
        for line in text.splitlines()
        if quoted in line or custom.search(line)
    ]


def _assert_source_recipe(body: str, operator: str, *, compare: bool) -> None:
    # Numeric source now lowers directly from the ac.rule block argument into
    # finite-width ACIR.  The retired source.read/ac.math proof recipe must not
    # reappear as a second numeric path.
    assert body.count('"ac.rule"') == 1
    assert '"ac.source.read"' not in body
    assert "ac.math." not in body and "!ac.math_int" not in body
    assert body.count('"ac.bits.constant"') == 1
    resize_count = body.count('"ac.bits.resize"')
    assert resize_count == 0 if compare else 1 <= resize_count <= 2
    operation = "ac.bits.compare" if compare else "ac.bits.binary"
    assert body.count(f'"{operation}"') == 1
    predicate = {
        "lt": "ult",
        "le": "ule",
        "gt": "ugt",
        "ge": "uge",
    }.get(operator, operator)
    assert f'{"predicate" if compare else "opcode"} = "{predicate}"' in body
    operation_line = _operation_lines(body, operation)[0]
    assert re.search(
        r'"ac\.bits\.(?:binary|compare)"\(%(?:arg)?\d+, %(?:arg)?\d+\)',
        operation_line,
    )


@pytest.mark.parametrize(
    ("expression", "operator", "compare"),
    [
        ("current + 7", "add", False),
        ("current - 7", "sub", False),
        ("current * 7", "mul", False),
        ("current == 7", "eq", True),
        ("current != 7", "ne", True),
        ("current < 7", "lt", True),
        ("current <= 7", "le", True),
        ("current > 7", "gt", True),
        ("current >= 7", "ge", True),
    ],
)
def test_python_value_expression_emits_closed_source_recipe(
    tmp_path: Path, expression: str, operator: str, compare: bool
) -> None:
    unit = _compile(tmp_path, operator, _module(expression))

    assert unit.process.returncode == 0, unit.process.stderr
    _assert_source_recipe(
        unit.body.read_text(encoding="utf-8"), operator, compare=compare
    )


def test_integer_constant_keeps_arbitrary_precision_and_is_not_a_bool(
    tmp_path: Path,
) -> None:
    literal = 9_007_199_254_740_993
    unit = _compile(tmp_path, "wide_constant", _module(f"current + {literal}"))

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert f"#ac.math_int<{literal}>" in body
    assert "#ac.math_int<true>" not in body
    assert "#ac.math_int<false>" not in body
    _assert_source_recipe(body, "add", compare=False)

def test_comparison_result_is_a_one_bit_hardware_value(tmp_path: Path) -> None:
    unit = _compile(tmp_path, "bool_result", _module("current == 1"))

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert '"ac.bits.compare"' in body
    comparison = _operation_lines(body, "ac.bits.compare")
    assert len(comparison) == 1 and "-> !ac.bits<" in comparison[0]
    assert "#ac.math_int<1>" in comparison[0]
    assert "#ac.math_int<1>" in body


@pytest.mark.parametrize(
    ("expression", "lowered_operation", "opcode"),
    [
        ("current + 7", "ac.bits.binary", "add"),
        ("current - 7", "ac.bits.binary", "sub"),
        ("current * 7", "ac.bits.binary", "mul"),
        ("current == 7", "ac.bits.compare", "eq"),
        ("current != 7", "ac.bits.compare", "ne"),
        ("current < 7", "ac.bits.compare", "ult"),
        ("current <= 7", "ac.bits.compare", "ule"),
        ("current > 7", "ac.bits.compare", "ugt"),
        ("current >= 7", "ac.bits.compare", "uge"),
    ],
)
def test_private_harness_lowers_real_python_numeric_source(
    tmp_path: Path,
    expression: str,
    lowered_operation: str,
    opcode: str,
) -> None:
    unit = _compile(
        tmp_path,
        lowered_operation.replace(".", "_"),
        _module(expression),
    )

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert "ac.math." not in body and "!ac.math_int" not in body
    assert f'"{lowered_operation}"' in body
    operations = _operation_lines(body, lowered_operation)
    assert len(operations) == 1
    attribute = "predicate" if lowered_operation.endswith("compare") else "opcode"
    assert f'{attribute} = "{opcode}"' in operations[0]
    assert "ac.numeric.proof" not in body
    assert "ac.required_numeric" not in body and "ac.proof_scope" not in body


def test_direct_lowering_is_stable_and_binds_the_rule_argument(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source-stage"
    lowered_dir = tmp_path / "lowered-stage"
    source_dir.mkdir()
    lowered_dir.mkdir()
    text = _module("current - 7")
    source = _compile(source_dir, "identity", text)
    lowered = _compile(lowered_dir, "identity", text)

    assert source.process.returncode == 0, source.process.stderr
    assert lowered.process.returncode == 0, lowered.process.stderr
    source_body = source.body.read_text(encoding="utf-8")
    lowered_body = lowered.body.read_text(encoding="utf-8")
    assert source_body == lowered_body
    assert '"ac.source.read"' not in lowered_body
    rule = _operation_lines(lowered_body, "ac.rule")
    assert len(rule) == 1 and "(%arg0)" in rule[0]
    first_resize = _operation_lines(lowered_body, "ac.bits.resize")[0]
    assert "(%arg1)" in first_resize


@pytest.mark.parametrize(
    ("stem", "source", "diagnostic"),
    [
        (
            "numeric_check",
            _module("current + 1", statements=('assert temporary, "numeric"',)),
            "conditional requires a Boolean source condition",
        ),
        (
            "numeric_observation",
            _module("current + 1", statements=('print("numeric", temporary)',)),
            "unsupported behavioral rule statement 'Expr'",
        ),
        (
            "bool_literal_is_not_an_integer",
            _module("current + True"),
            "Boolean literal cannot be implicitly converted",
        ),
        (
            "bool_current_is_not_an_integer",
            _module("current + 1", parameters="current: bool"),
            "requires known Integer kind and finite interval",
        ),
        (
            "chained_compare",
            _module("current < 7 < 9"),
            "hardware comparisons do not support chaining",
        ),
    ],
)
def test_unsupported_numeric_shapes_are_rejected_without_publishing_body(
    tmp_path: Path, stem: str, source: str, diagnostic: str
) -> None:
    unit = _compile(tmp_path, stem, source)

    assert unit.process.returncode != 0
    assert not unit.body.exists()
    assert diagnostic.lower() in unit.process.stderr.lower()


def test_wide_unused_intermediate_remains_exact_in_source_body(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "i65_result",
        _module("current + 18446744073709551615", word_upper=2),
    )

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert "#ac.math_int<65>" in body
    assert "#ac.math_int<18446744073709551615>" in body
    assert 'opcode = "add"' in body
