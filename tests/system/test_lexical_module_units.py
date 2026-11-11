"""Independent module/register lexical source contract tests."""

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
    raise AssertionError("set PYCIRCUIT_SOURCE_COMPILER for lexical module tests")


def _compile(
    tmp_path: Path,
    stem: str,
    text: str,
    *,
    headers: tuple[Path, ...] = (),
) -> CompiledUnit:
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    source = source_root / f"{stem}.py"
    source.write_text(text, encoding="utf-8")
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
        "verify",
        "--path",
        source.name,
    ]
    for header in headers:
        command.extend(("--header", str(header)))
    command.extend(("--body-out", str(body), "--interface-out", str(interface)))
    return CompiledUnit(
        subprocess.run(command, text=True, capture_output=True, check=False),
        body,
        interface,
    )


WORD_ALIAS = "from typing import Annotated\nWord = Annotated[int, range(16)]\n"


def _operation_lines(text: str, name: str) -> list[str]:
    quoted = f'"{name}"'
    custom = re.compile(rf"(?:^|\s){re.escape(name)}(?:\s|\()")
    return [
        line.strip()
        for line in text.splitlines()
        if quoted in line or custom.search(line)
    ]


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
    raise AssertionError(f"unterminated {name} dictionary: {line}")


def _operation_operands(line: str, name: str) -> list[str]:
    match = re.search(rf'(?:"{re.escape(name)}"|{re.escape(name)})\(([^)]*)\)', line)
    assert match is not None, line
    return [operand.strip() for operand in match.group(1).split(",")]


def _operation_results(line: str, count: int) -> list[str]:
    prefix = line.split("=", 1)[0].strip()
    explicit = [result.strip() for result in prefix.split(",")]
    if len(explicit) == count and all(
        re.fullmatch(r"%[A-Za-z0-9_]+", result) for result in explicit
    ):
        return explicit
    multiple = re.fullmatch(r"%([A-Za-z0-9_]+):(\d+)", prefix)
    if multiple is not None:
        assert int(multiple.group(2)) == count, line
        return [f"%{multiple.group(1)}#{index}" for index in range(count)]
    assert count == 1 and re.fullmatch(r"%[A-Za-z0-9_]+", prefix), line
    return [prefix]


def _rule_text(body: str) -> str:
    return '"ac.rule"' + body.split('"ac.rule"', 1)[1].split(
        '"ac.yield"() : () -> ()', 1
    )[0]


def _rule_arguments(rule: str) -> list[tuple[str, str]]:
    marker = "^bb0("
    start = rule.find(marker)
    assert start >= 0, rule
    opening = start + len(marker) - 1
    depth = 0
    quoted = False
    escaped = False
    closing = -1
    for index in range(opening, len(rule)):
        character = rule[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                closing = index
                break
    assert closing >= 0 and rule[closing + 1 :].lstrip().startswith(":"), rule
    arguments = rule[opening + 1 : closing]
    return re.findall(r"(%[A-Za-z0-9_]+):\s*([^,\s]+)", arguments)


def _yield_operands(rule: str) -> list[str]:
    lines = _operation_lines(rule, "ac.yield")
    assert len(lines) == 1, lines
    return _operation_operands(lines[0], "ac.yield")


def _assert_condition_path(rule: str, path: str, ready_read: str) -> None:
    ready_value = _operation_results(ready_read, 1)[0]
    definitions = {}
    for line in rule.splitlines():
        line = line.strip()
        if " = " in line:
            definitions[line.split(" = ", 1)[0]] = line

    def evaluate(value: str, ready: bool) -> bool:
        if value == ready_value:
            return ready
        line = definitions[value]
        if "arith.constant" in line:
            return " true" in line or "value = true" in line
        if "arith.andi" in line:
            # Handle either custom or generic operation printing.
            if '"arith.andi"' in line:
                operands = _operation_operands(line, "arith.andi")
            else:
                operands = line.split("arith.andi ", 1)[1].split(" :", 1)[0].split(", ")
            return all(evaluate(operand, ready) for operand in operands)
        raise AssertionError(f"unsupported boolean path operation: {line}")

    assert not evaluate(path, False), "inactive conditional must hold"
    assert evaluate(path, True), "active conditional must propose"


def test_v02_function_module_nested_rule_and_bare_return_are_accepted(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "counter",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def Counter():\n"
        + "    q: Word = 0\n\n"
        + "    @rule\n"
        + "    def hold():\n"
        + "        nonlocal q\n"
        + "        q = q\n"
        + "        return\n\n"
        + "    hold()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert '"ac.module"()' in body
    assert body.count('callee = @pycircuit.__builtins__.dffe') == 1
    assert 'instance_name = "q"' in body
    assert body.count('"ac.rule"') == 1
    assert '"ac.source.read"' not in body
    assert '"ac.source.use"' not in body


def test_v02_class_and_self_module_authoring_is_a_hard_error(tmp_path: Path) -> None:
    unit = _compile(
        tmp_path,
        "retired_class",
        WORD_ALIAS
        + "from pycircuit import module\n\n"
        + "@module\n"
        + "class Retired:\n"
        + "    def __init__(self):\n"
        + "        self.q: Word = 0\n",
    )

    assert unit.process.returncode != 0
    assert "source top level supports imports, @module, and @system definitions" in (
        unit.process.stderr
    )


def test_v03_nonlocal_write_reads_one_canonical_q_slot(tmp_path: Path) -> None:
    unit = _compile(
        tmp_path,
        "alias_read",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def AliasRead(source: Word, sink: Word):\n"
        + "    alias = source\n\n"
        + "    @rule\n"
        + "    def forward():\n"
        + "        nonlocal sink\n"
        + "        sink = alias\n"
        + "        return\n\n"
        + "    forward()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    rule = _rule_text(unit.body.read_text(encoding="utf-8"))
    rule_line = _operation_lines(rule, "ac.rule")[0]
    assert len(_operation_operands(rule_line, "ac.rule")) == 3
    assert 'name = "forward"' in rule_line
    assert '"ac.source.read"' not in rule
    assert '"ac.source.use"' not in rule


def test_v03_assignment_without_nonlocal_uses_current_lexical_capture(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "missing_nonlocal",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def MissingNonlocal():\n"
        + "    q: Word = 0\n\n"
        + "    @rule\n"
        + "    def update():\n"
        + "        q = q\n"
        + "        return\n\n"
        + "    update()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert body.count('callee = @pycircuit.__builtins__.dffe') == 1
    assert body.count('"ac.rule"') == 1
    assert '"ac.source.read"' not in body


def test_v03_local_ssa_can_feed_a_nonlocal_proposal(tmp_path: Path) -> None:
    unit = _compile(
        tmp_path,
        "local_ssa",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def LocalSSA(source: Word, sink: Word):\n"
        + "    @rule\n"
        + "    def forward():\n"
        + "        nonlocal sink\n"
        + "        candidate = source\n"
        + "        sink = candidate\n"
        + "        return\n\n"
        + "    forward()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    rule = _rule_text(unit.body.read_text(encoding="utf-8"))
    rule_line = _operation_lines(rule, "ac.rule")[0]
    operands = _operation_operands(rule_line, "ac.rule")
    assert len(operands) == 2
    assert len(set(operands)) == 2
    assert '"ac.source.read"' not in rule
    assert '"ac.source.use"' not in rule


def test_v03_owned_reg_alias_reuses_one_state_for_read_and_write(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "owned_alias",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def OwnedAlias():\n"
        + "    q: Word = 0\n"
        + "    alias = q\n\n"
        + "    @rule\n"
        + "    def hold():\n"
        + "        nonlocal alias\n"
        + "        alias = alias\n"
        + "        return\n\n"
        + "    hold()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert body.count('callee = @pycircuit.__builtins__.dffe') == 1
    assert 'instance_name = "q"' in body
    assert "ac.queue" not in body
    rule_line = _operation_lines(body, "ac.rule")[0]
    assert len(_operation_operands(rule_line, "ac.rule")) == 4


def test_v14_conditional_write_uses_condition_as_enable_and_bare_return_holds(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "conditional_write",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def ConditionalWrite(source: Word, sink: Word, ready: bool):\n"
        + "    @rule\n"
        + "    def forward():\n"
        + "        nonlocal sink\n"
        + "        if ready:\n"
        + "            sink = source\n"
        + "        return\n\n"
        + "    forward()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    rule = _rule_text(unit.body.read_text(encoding="utf-8"))
    rule_line = _operation_lines(rule, "ac.rule")[0]
    assert len(_operation_operands(rule_line, "ac.rule")) == 3
    assert '"ac.source.read"' not in rule
    assert '"ac.source.use"' not in rule


def test_w04_each_direct_q_read_has_a_distinct_source_occurrence(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "repeated_reads",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def RepeatedReads(source: Word, sink: Word):\n"
        + "    @rule\n"
        + "    def forward():\n"
        + "        nonlocal sink\n"
        + "        first = source\n"
        + "        second = source\n"
        + "        sink = second\n"
        + "        return\n\n"
        + "    forward()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    rule = _rule_text(unit.body.read_text(encoding="utf-8"))
    rule_line = _operation_lines(rule, "ac.rule")[0]
    operands = _operation_operands(rule_line, "ac.rule")
    assert len(operands) == 2
    assert len(set(operands)) == 2
    assert '"ac.source.read"' not in rule


def test_w04_local_cache_reuse_does_not_duplicate_source_read(tmp_path: Path) -> None:
    unit = _compile(
        tmp_path,
        "cached_read",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def CachedRead(source: Word, sink: Word):\n"
        + "    @rule\n"
        + "    def forward():\n"
        + "        nonlocal sink\n"
        + "        cached = source\n"
        + "        sink = cached\n"
        + "        return\n\n"
        + "    forward()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    rule = _rule_text(unit.body.read_text(encoding="utf-8"))
    rule_line = _operation_lines(rule, "ac.rule")[0]
    assert len(_operation_operands(rule_line, "ac.rule")) == 2
    assert '"ac.source.read"' not in rule


def test_w04_each_nonlocal_assignment_uses_canonical_target_and_yield(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "two_next_uses",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def TwoNextUses(source: Word, left: Word, right: Word):\n"
        + "    @rule\n"
        + "    def forward():\n"
        + "        nonlocal left, right\n"
        + "        left = source\n"
        + "        right = source\n"
        + "        return\n\n"
        + "    forward()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    rule = _rule_text(unit.body.read_text(encoding="utf-8"))
    rule_line = _operation_lines(rule, "ac.rule")[0]
    operands = _operation_operands(rule_line, "ac.rule")
    assert len(operands) == 3
    assert len(set(operands)) == 3
    assert '"ac.source.use"' not in rule


def test_w04_alias_read_keeps_source_origin_on_the_canonical_q_slot(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "alias_source_fact",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def AliasSourceFact(source: Word, sink: Word):\n"
        + "    alias = source\n\n"
        + "    @rule\n"
        + "    def forward():\n"
        + "        nonlocal sink\n"
        + "        sink = alias\n"
        + "        return\n\n"
        + "    forward()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    rule = _rule_text(body)
    rule_line = _operation_lines(rule, "ac.rule")[0]
    assert len(_operation_operands(rule_line, "ac.rule")) == 3
    assert 'input_names = ["source", "sink"]' in body
    assert 'parameter = "alias"' not in body


def test_w04_conditional_next_use_preserves_typed_rule_dataflow(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "conditional_active_facts",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def ConditionalActiveFacts(source: Word, sink: Word, ready: bool):\n"
        + "    @rule\n"
        + "    def active():\n"
        + "        nonlocal sink\n"
        + "        if ready:\n"
        + "            sink = source\n"
        + "        return\n\n"
        + "    active()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert body.count('"ac.rule"') == 1
    assert 'name = "active"' in body
    rule = _rule_text(body)
    rule_line = _operation_lines(rule, "ac.rule")[0]
    assert len(_operation_operands(rule_line, "ac.rule")) == 3
    assert '"ac.source.read"' not in rule
    assert '"ac.source.use"' not in rule


def test_w04_duplicate_rule_registration_is_rejected(
    tmp_path: Path,
) -> None:
    unit = _compile(
        tmp_path,
        "two_registrations",
        "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def TwoRegistrations(source: bool, sink: bool):\n"
        + "    @rule\n"
        + "    def forward():\n"
        + "        nonlocal sink\n"
        + "        sink = source\n"
        + "        return\n\n"
        + "    forward()\n"
        + "    forward()\n",
    )

    assert unit.process.returncode != 0
    assert "Rule declaration must be registered exactly once" in unit.process.stderr


def test_w05_child_result_uses_parent_instance_occurrence(tmp_path: Path) -> None:
    child = _compile(
        tmp_path,
        "child_effects",
        "from pycircuit import module, rule, struct, u1\n\n"
        + "@struct\n"
        + "class ChildResult:\n"
        + "    sink: u1\n\n"
        + "@module\n"
        + "def ChildEffects(source: u1) -> ChildResult:\n"
        + "    @rule\n"
        + "    def forward() -> ChildResult:\n"
        + "        return ChildResult(sink=source)\n\n"
        + "    return forward()\n",
    )
    assert child.process.returncode == 0, child.process.stderr

    parent = _compile(
        tmp_path,
        "parent_effects",
        "from pycircuit import module, rule, u1\n"
        + "from .child_effects import ChildEffects\n\n"
        + "@rule\n"
        + "def transfer(sink: u1, child_sink: u1) -> None:\n"
        + "    sink = child_sink\n\n"
        + "@module\n"
        + "def ParentEffects(source: u1, sink: u1):\n"
        + "    child = ChildEffects(source)\n"
        + "    transfer(sink, child.sink)\n",
        headers=(child.interface,),
    )
    assert parent.process.returncode == 0, parent.process.stderr

    body = parent.body.read_text(encoding="utf-8")
    instances = _operation_lines(body, "ac.instance")
    assert len(instances) == 1
    instance_occurrence = _dictionary_attr(instances[0], "occurrence")
    interface = parent.interface.read_text(encoding="utf-8")
    declarations = [
        declaration
        for declaration in _operation_lines(interface, "ac.module.import")
        if "verify.parent_effects.ParentEffects" in declaration
    ]
    assert len(declarations) == 1
    contract_line = declarations[0]
    assert instance_occurrence
    assert "@verify.child_effects.ChildEffects" in body
    assert "verify.parent_effects.ParentEffects" in contract_line


@pytest.mark.parametrize("imported", ["Queue", "FIFO", "Queue as LocalQueue"])
def test_v05_framework_storage_identities_are_rejected(
    tmp_path: Path, imported: str
) -> None:
    unit = _compile(
        tmp_path,
        "framework_storage",
        f"from pycircuit import module, {imported}\n\n"
        + "@module\n"
        + "def BadStorage():\n"
        + "    return\n",
    )

    assert unit.process.returncode != 0
    name = imported.split()[0]
    assert f"imported declaration 'pycircuit.{name}' is absent from explicit headers" in (
        unit.process.stderr
    )


def test_v05_ordinary_queue_fifo_ready_push_names_are_accepted(tmp_path: Path) -> None:
    unit = _compile(
        tmp_path,
        "ordinary_names",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def OrdinaryNames(Queue: Word, FIFO: Word, ready: Word, push: Word):\n"
        + "    @rule\n"
        + "    def copy():\n"
        + "        nonlocal ready, push\n"
        + "        ready = Queue\n"
        + "        push = FIFO\n"
        + "        return\n\n"
        + "    copy()\n",
    )

    assert unit.process.returncode == 0, unit.process.stderr
    body = unit.body.read_text(encoding="utf-8")
    assert body.count('"ac.rule"') == 1
    assert len(_operation_operands(_operation_lines(body, "ac.rule")[0], "ac.rule")) == 4
    assert "ac.queue" not in body


def test_v02_lexical_parent_rejects_retired_self_child_actual(tmp_path: Path) -> None:
    child = _compile(
        tmp_path,
        "child",
        WORD_ALIAS
        + "from pycircuit import module, rule\n\n"
        + "@module\n"
        + "def Child(source: Word, sink: Word):\n"
        + "    @rule\n"
        + "    def forward():\n"
        + "        nonlocal sink\n"
        + "        sink = source\n"
        + "        return\n\n"
        + "    forward()\n",
    )
    assert child.process.returncode == 0, child.process.stderr

    parent = _compile(
        tmp_path,
        "parent",
        WORD_ALIAS
        + "from pycircuit import module\n"
        + "from .child import Child\n\n"
        + "@module\n"
        + "def Parent():\n"
        + "    source: Word = 0\n"
        + "    sink: Word = 0\n"
        + "    child = Child(self.source, sink)\n",
        headers=(child.interface,),
    )

    assert parent.process.returncode != 0
    assert "unknown hardware value 'self'" in parent.process.stderr


def test_v02_imported_system_header_cannot_be_instantiated_as_child(
    tmp_path: Path,
) -> None:
    root = _compile(
        tmp_path,
        "root_system",
        WORD_ALIAS
        + "from pycircuit import system, rule\n\n"
        + "@system\n"
        + "def RootSystem():\n"
        + "    q: Word = 0\n\n"
        + "    @rule\n"
        + "    def hold():\n"
        + "        nonlocal q\n"
        + "        q = q\n"
        + "        return\n\n"
        + "    hold()\n",
    )
    assert root.process.returncode == 0, root.process.stderr
    assert 'ac.root_kind = "system"' in root.interface.read_text(encoding="utf-8")

    parent = _compile(
        tmp_path,
        "system_parent",
        "from pycircuit import module\n"
        + "from .root_system import RootSystem\n\n"
        + "@module\n"
        + "def SystemParent():\n"
        + "    child = RootSystem()\n",
        headers=(root.interface,),
    )

    assert parent.process.returncode != 0
    assert "@system is root-only and cannot be instantiated as a child" in (
        parent.process.stderr
    )
