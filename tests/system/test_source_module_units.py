"""Current source module, nominal result, and header-only composition tests."""

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
ROOT = Path(__file__).resolve().parents[2]
PACKET_FIXTURE = ROOT / "tests/integration/fixtures/source_language/packet.py"

PROBE = """\
from pycircuit import module, rule
from .packet import ProbeResult, Request

@module
def AccumulatorProbe(request: Request) -> ProbeResult:
    @rule
    def forward() -> ProbeResult:
        return ProbeResult(value=request.value)

    return forward()
"""

ROOT_SOURCE = """\
from pycircuit import module
from .packet import ProbeResult, Request
from .accumulator_probe import AccumulatorProbe

@module
def ProbeRoot() -> ProbeResult:
    request = Request(value=3, valid=1)
    child = AccumulatorProbe(request)
    return ProbeResult(value=child.value)
"""


@dataclass(frozen=True)
class Unit:
    completed: subprocess.CompletedProcess[str]
    transport: Path
    body: Path
    interface: Path


def _harness() -> Path:
    candidates = [os.environ.get("PYCIRCUIT_SOURCE_COMPILER")]
    root = os.environ.get("PYC_TOOLCHAIN_ROOT")
    if root:
        candidates.append(str(Path(root) / "bin/pycircuit-source-unit"))
    candidates.append(shutil.which("pycircuit-source-unit"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise AssertionError("set PYCIRCUIT_SOURCE_COMPILER for source module tests")


def _compile(
    source: Path,
    *,
    root: Path,
    output: Path,
    headers: tuple[Path, ...] = (),
) -> Unit:
    output.mkdir(parents=True, exist_ok=True)
    transport = output / f"{source.stem}.transport.mlir"
    body = output / f"{source.stem}.body.mlir"
    interface = output / f"{source.stem}.interface.mlir"
    transport.write_text(
        _emit_source_transport(_capture_source_file(source, source_root=root)),
        encoding="utf-8",
    )
    command = [
        str(_harness()),
        "--capture",
        str(transport),
        "--package",
        "demo",
        "--path",
        source.relative_to(root).as_posix(),
    ]
    for header in headers:
        command.extend(("--header", str(header)))
    command.extend(("--body-out", str(body), "--interface-out", str(interface)))
    return Unit(
        subprocess.run(command, text=True, capture_output=True, check=False),
        transport,
        body,
        interface,
    )


def _workspace(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(PACKET_FIXTURE, source / "packet.py")
    (source / "accumulator_probe.py").write_text(PROBE, encoding="utf-8")
    (source / "probe_root.py").write_text(ROOT_SOURCE, encoding="utf-8")
    return source


def _compile_probe(tmp_path: Path) -> tuple[Path, Unit, Unit]:
    source = _workspace(tmp_path)
    packet = _compile(source / "packet.py", root=source, output=tmp_path / "packet")
    assert packet.completed.returncode == 0, packet.completed.stderr
    probe = _compile(
        source / "accumulator_probe.py",
        root=source,
        output=tmp_path / "probe",
        headers=(packet.interface,),
    )
    assert probe.completed.returncode == 0, probe.completed.stderr
    return source, packet, probe


def _operation_lines(text: str, operation: str) -> list[str]:
    quoted = f'"{operation}"'
    custom = re.compile(rf"(?:^|\s){re.escape(operation)}(?:\s|\()")
    return [
        line.strip()
        for line in text.splitlines()
        if quoted in line or custom.search(line)
    ]


def _module_declaration(text: str, symbol: str) -> str:
    matches = [
        line
        for line in _operation_lines(text, "ac.module.import")
        if f'sym_name = "{symbol}"' in line
    ]
    assert len(matches) == 1, matches
    return matches[0]


def test_packet_and_probe_emit_body_interface_and_exact_transport(tmp_path: Path) -> None:
    source, packet, probe = _compile_probe(tmp_path)
    assert packet.interface.is_file() and probe.body.is_file() and probe.interface.is_file()
    assert probe.transport.read_text(encoding="utf-8") == _emit_source_transport(
        _capture_source_file(source / "accumulator_probe.py", source_root=source)
    )


def test_class_self_module_authoring_is_an_explicit_negative(tmp_path: Path) -> None:
    source = _workspace(tmp_path)
    retired = source / "retired.py"
    retired.write_text(
        "from pycircuit import module\n\n@module\nclass Retired:\n"
        "    def __init__(self):\n        self.value: int = 0\n",
        encoding="utf-8",
    )
    result = _compile(retired, root=source, output=tmp_path / "retired")
    assert result.completed.returncode != 0
    assert "source top level supports imports, @module, and @system definitions" in result.completed.stderr


def test_probe_interface_publishes_typed_nominal_result_contract(tmp_path: Path) -> None:
    _, _, probe = _compile_probe(tmp_path)
    interface = probe.interface.read_text(encoding="utf-8")
    declaration = _module_declaration(
        interface, "demo.accumulator_probe.AccumulatorProbe"
    )
    assert 'input_names = ["request"]' in declaration
    assert 'output_names = ["result"]' in declaration
    assert 'parameters = []' in declaration
    assert '!ac.struct<"demo.packet.Request">' in declaration
    assert '!ac.struct<"demo.packet.ProbeResult">' in declaration


def test_probe_body_contains_import_snapshots_and_typed_rule_dataflow(
    tmp_path: Path,
) -> None:
    _, _, probe = _compile_probe(tmp_path)
    body = probe.body.read_text(encoding="utf-8")
    assert 'ac.struct "demo.packet.Request"' in body
    assert 'ac.struct "demo.packet.ProbeResult"' in body
    assert 'ac.declaration_role = "import_snapshot"' in body
    assert body.count('"ac.rule"') == 1
    assert "ac.struct.get" in body
    assert "ac.struct.create" in body
    assert '"ac.source.read"' not in body
    assert '"ac.source.use"' not in body


def test_header_only_parent_compiles_to_current_child_instance(tmp_path: Path) -> None:
    source, packet, probe = _compile_probe(tmp_path)
    (source / "packet.py").unlink()
    (source / "accumulator_probe.py").unlink()
    packet.body.unlink()
    probe.body.unlink()
    root = _compile(
        source / "probe_root.py",
        root=source,
        output=tmp_path / "root",
        headers=(packet.interface, probe.interface),
    )
    assert root.completed.returncode == 0, root.completed.stderr
    body = root.body.read_text(encoding="utf-8")
    assert 'callee = @demo.accumulator_probe.AccumulatorProbe' in body
    assert "ac.struct.create" in body
    assert "ac.struct.get" in body
    assert '#ac.math_int<3>' in body


@pytest.mark.parametrize(
    "inactive_body",
    (
        "        return\n",
        "        return ProbeResult(value=request.value)\n",
        "        missing = request.value\n        return ProbeResult(value=missing)\n",
    ),
)
def test_unregistered_rules_are_rejected(
    tmp_path: Path, inactive_body: str
) -> None:
    source = _workspace(tmp_path)
    packet = _compile(source / "packet.py", root=source, output=tmp_path / "packet")
    assert packet.completed.returncode == 0, packet.completed.stderr
    text = PROBE.replace(
        "    return forward()\n",
        "    @rule\n    def inactive() -> ProbeResult:\n"
        + inactive_body
        + "\n    return forward()\n",
    )
    path = source / "unregistered.py"
    path.write_text(text, encoding="utf-8")
    result = _compile(
        path, root=source, output=tmp_path / "unregistered", headers=(packet.interface,)
    )
    assert result.completed.returncode != 0
    assert "Rule declaration must be registered exactly once" in result.completed.stderr


def test_duplicate_rule_registration_is_rejected(tmp_path: Path) -> None:
    source = _workspace(tmp_path)
    packet = _compile(source / "packet.py", root=source, output=tmp_path / "packet")
    text = PROBE.replace("    return forward()\n", "    forward()\n    return forward()\n")
    path = source / "duplicate.py"
    path.write_text(text, encoding="utf-8")
    result = _compile(
        path, root=source, output=tmp_path / "duplicate", headers=(packet.interface,)
    )
    assert result.completed.returncode != 0
    assert "Rule declaration must be registered exactly once" in result.completed.stderr


@pytest.mark.parametrize(
    "expression",
    (
        "Request(unknown=3)",
        "Request(value=3, missing=1)",
        "Request(value=3)",
        "Request(3, 1, 0)",
        "Request(value=3, valid=1, value=4)",
    ),
)
def test_invalid_nominal_constructor_bindings_are_rejected(
    tmp_path: Path, expression: str
) -> None:
    source, packet, probe = _compile_probe(tmp_path)
    text = ROOT_SOURCE.replace("Request(value=3, valid=1)", expression)
    path = source / "invalid_constructor.py"
    path.write_text(text, encoding="utf-8")
    result = _compile(
        path,
        root=source,
        output=tmp_path / "invalid-constructor",
        headers=(packet.interface, probe.interface),
    )
    assert result.completed.returncode != 0
    assert result.completed.stderr.strip()


def test_wrong_nominal_child_argument_is_rejected(tmp_path: Path) -> None:
    source, packet, probe = _compile_probe(tmp_path)
    other = source / "other.py"
    other.write_text(
        "from pycircuit import struct, u1, u8\n\n@struct\nclass Other:\n"
        "    value: u8\n    valid: u1\n",
        encoding="utf-8",
    )
    other_unit = _compile(other, root=source, output=tmp_path / "other")
    assert other_unit.completed.returncode == 0, other_unit.completed.stderr
    text = ROOT_SOURCE.replace(
        "from .packet import ProbeResult, Request",
        "from .packet import ProbeResult, Request\nfrom .other import Other",
    ).replace("Request(value=3, valid=1)", "Other(value=3, valid=1)")
    path = source / "wrong_nominal.py"
    path.write_text(text, encoding="utf-8")
    result = _compile(
        path,
        root=source,
        output=tmp_path / "wrong-nominal",
        headers=(packet.interface, probe.interface, other_unit.interface),
    )
    assert result.completed.returncode != 0
    assert result.completed.stderr.strip()


def test_child_call_accepts_header_declared_positional_connection(tmp_path: Path) -> None:
    source, packet, probe = _compile_probe(tmp_path)
    path = source / "positional_parent.py"
    path.write_text(ROOT_SOURCE, encoding="utf-8")
    result = _compile(
        path,
        root=source,
        output=tmp_path / "positional-parent",
        headers=(packet.interface, probe.interface),
    )
    assert result.completed.returncode == 0, result.completed.stderr
    assert 'callee = @demo.accumulator_probe.AccumulatorProbe' in result.body.read_text(
        encoding="utf-8"
    )
