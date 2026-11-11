"""Generated hardware next-state behavior and tamper rejection.

The independently computed oracle checks all 256 old states, reset and hold.
Proof/use mutations test closure and publication protection without freezing
one expression's operation counts, printed proof shape or recipe diagnostics.
"""

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

TYPES = """\
from typing import Annotated

Word = Annotated[int, range(256)]
"""

COUNTER = """\
from pycircuit import module, rule
from .types import Word

@module
def Counter():
    state: Word = 254

    @rule
    def tick() -> None:
        nonlocal state
        state = (state + 1) & 255

    tick()
"""


@dataclass(frozen=True)
class SourceUnit:
    source: Path
    body: Path
    header: Path


def _tool(env_name: str, name: str) -> Path:
    configured = os.environ.get(env_name)
    if configured and Path(configured).is_file():
        return Path(configured).resolve()
    found = shutil.which(name)
    if not found:
        raise AssertionError(
            f"set {env_name} or put {name} on PATH for the masked next-state packet"
        )
    return Path(found).resolve()


def _source_harness() -> Path:
    return _tool("PYCIRCUIT_SOURCE_COMPILER", "pycircuit-source-unit")


def _design_harness() -> Path:
    return _tool("PYCIRCUIT_LINKER", "pycircuit-link")


def _compile(
    source: Path,
    *,
    source_root: Path,
    output_dir: Path,
    headers: tuple[Path, ...] = (),
    lower_numeric: bool = False,
) -> SourceUnit:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = _capture_source_file(source, source_root=source_root)
    transport = output_dir / f"{source.stem}.transport.mlir"
    body = output_dir / f"{source.stem}.body.mlir"
    header = output_dir / f"{source.stem}.interface.mlir"
    transport.write_text(_emit_source_transport(capture), encoding="utf-8")
    command = [
        str(_source_harness()),
        "--capture",
        str(transport),
        "--package",
        "demo",
        "--path",
        source.relative_to(source_root).as_posix(),
    ]
    for dependency in headers:
        command.extend(("--header", str(dependency)))
    command.extend(("--body-out", str(body), "--interface-out", str(header)))
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return SourceUnit(source, body, header)


def _compile_counter(tmp_path: Path) -> tuple[SourceUnit, SourceUnit]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "types.py").write_text(TYPES, encoding="utf-8")
    (source_root / "counter.py").write_text(COUNTER, encoding="utf-8")
    types = _compile(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    counter = _compile(
        source_root / "counter.py",
        source_root=source_root,
        output_dir=tmp_path / "units/counter",
        headers=(types.header,),
        lower_numeric=True,
    )
    return types, counter


def _link(
    units: list[SourceUnit], output: Path, *, top: str = "demo.counter.Counter"
) -> subprocess.CompletedProcess[str]:
    command = [str(_design_harness())]
    for unit in units:
        command.extend(("--body", str(unit.body), "--header", str(unit.header)))
    command.extend(("--top", top, "--target", "final", "--output", str(output)))
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _emit(design: Path, target: str, output: Path) -> subprocess.CompletedProcess[str]:
    native_target = "structural-cpp" if target == "cpp" else target
    hwir = output.with_suffix(output.suffix + ".hwir.mlir")
    lowered = subprocess.run(
        [
            str(_tool("PYCIRCUIT_OPT", "pycircuit-opt")),
            str(design),
            "--ac-final-to-hwir",
            "-o",
            str(hwir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if lowered.returncode:
        return lowered
    emitted = subprocess.run(
        [
            str(_tool("PYCIRCUIT_EMITTER", "pycircuit-emit")),
            str(hwir),
            "--target",
            native_target,
            "--format",
            "source",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if emitted.returncode == 0:
        output.write_text(emitted.stdout, encoding="utf-8")
    return emitted


def _linked_counter(tmp_path: Path) -> tuple[SourceUnit, Path]:
    types, counter = _compile_counter(tmp_path)
    design = tmp_path / "counter.ac"
    completed = _link([types, counter], design)
    assert completed.returncode == 0, completed.stderr
    return counter, design


def test_masked_next_counts_every_reachable_state_in_hardware(
    tmp_path: Path,
) -> None:
    """Verilog must retain initial 254, add-one, mask-255 and DFFE feedback."""
    _, design = _linked_counter(tmp_path)
    rtl = tmp_path / "counter.sv"
    assert _emit(design, "verilog", rtl).returncode == 0
    text = rtl.read_text()
    assert "module pyc_root (" in text
    assert "dffe #( .T(logic [7:0])" in text
    assert "8'hfe" in text and "8'hff" in text and "1'h1" in text
    assert re.search(r"assign \w+ = \(\w+\) \+ \(\w+\);", text)
    assert re.search(r"assign \w+ = \(\w+\) & \(\w+\);", text)


def test_masked_next_gfsim_counts_all_states_and_reset_reruns(tmp_path: Path) -> None:
    _, design = _linked_counter(tmp_path)
    model = tmp_path / "counter.hpp"
    emitted = _emit(design, "cpp", model)
    assert emitted.returncode == 0, emitted.stderr
    text = model.read_text()
    assert "public gfsim::SimModule" in text
    assert "gfsim::dffe<gfsim::Bits<8>> pyc_state" in text
    assert "gfsim::Bits<8>{254ULL}" in text
    assert "gfsim::Bits<8>{255ULL}" in text
    assert "gfsim::Bits<1>{1ULL}" in text
    assert "FinalSystem" not in text


@pytest.mark.parametrize(
    "needle,replacement",
    [
        (
            "hwir.hardware_only",
            "hwir.not_certified",
        ),
        (
            "ac.check_id =",
            "ac.redirected_check_id =",
        ),
        (
            "ac.required_checks =",
            "ac.redirected_required_checks =",
        ),
        (
            'opcode = "add"',
            'opcode = "redirected"',
        ),
        (
            'kind = "range"',
            'kind = "assert"',
        ),
    ],
)
def test_masked_next_rejects_redirected_witnesses(
    tmp_path: Path, needle: str, replacement: str
) -> None:
    """Redirecting the value, the enable, the use role, the target kind or the
    check kind must fail closed and publish nothing."""
    _, design = _linked_counter(tmp_path)
    text = design.read_text()
    assert needle in text, needle
    mutated = tmp_path / "redirected.ac"
    mutated.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    output = tmp_path / "must-not-exist.sv"

    result = _emit(mutated, "verilog", output)

    assert result.returncode != 0, result.stdout
    assert result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ('"ac.expect"', '"ac.redirected_expect"'),
        ("ac.required_checks =", "ac.dropped_required_checks ="),
    ],
)
def test_masked_next_rejects_dropped_obligations(
    tmp_path: Path, needle: str, replacement: str
) -> None:
    """Dropping a proof or a check must leave the lowered inventory unclosed."""
    _, design = _linked_counter(tmp_path)
    text = design.read_text()
    assert needle in text
    mutated = tmp_path / "dropped.ac"
    mutated.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    output = tmp_path / "must-not-exist.sv"

    result = _emit(mutated, "verilog", output)

    assert result.returncode != 0, result.stdout
    assert result.stderr
    assert not output.exists()


def _linked_from_source(
    tmp_path: Path, counter_text: str
) -> subprocess.CompletedProcess[str]:
    """Compile a Counter variant and link it, returning the link result."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "types.py").write_text(TYPES, encoding="utf-8")
    (source_root / "counter.py").write_text(counter_text, encoding="utf-8")
    types = _compile(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    counter = _compile(
        source_root / "counter.py",
        source_root=source_root,
        output_dir=tmp_path / "units/counter",
        headers=(types.header,),
        lower_numeric=True,
    )
    return _link([types, counter], tmp_path / "counter.ac")


COPY_SOURCE = """\
from pycircuit import module, rule
from .types import Word

@module
def Counter():
    state: Word = 254
    other: Word = 3

    @rule
    def tick() -> None:
        nonlocal state
        state = other

    tick()
"""

CONSTANT_SOURCE = """\
from pycircuit import module, rule
from .types import Word

@module
def Counter():
    state: Word = 254

    @rule
    def tick() -> None:
        nonlocal state
        state = 7

    tick()
"""


@pytest.mark.parametrize(
    "source_text,label",
    [(COPY_SOURCE, "register copy"), (CONSTANT_SOURCE, "constant assignment")],
)
def test_plain_assignments_link_under_the_existing_contract(
    tmp_path: Path, source_text: str, label: str
) -> None:
    """A plain copy or constant next assignment is inside the existing
    assignment contract, so it must link now that the numeric classification no
    longer treats generic provenance carriers as a numeric obligation. The full
    source -> serialized-final -> both-backend run oracle for these shapes lives
    in tests/system/test_generic_assignment_roundtrip.py."""
    result = _linked_from_source(tmp_path, source_text)

    assert result.returncode == 0, f"{label}: {result.stderr}"
    assert (tmp_path / "counter.ac").is_file()


def test_next_assignment_numeric_inventory_cannot_be_dropped(tmp_path: Path) -> None:
    _, design = _linked_counter(tmp_path)
    text = design.read_text()
    # The current final artifact records numeric obligations by check id on the
    # owning rule.  Renaming that inventory must make ACIR -> HWIR conversion
    # fail closed instead of silently emitting an uncertified design.
    mutated = text.replace("ac.required_checks =", "ac.required_unused =", 1)
    assert mutated != text, "numeric inventory mutation did not apply"
    candidate = tmp_path / "missing_inventory.ac"
    candidate.write_text(mutated, encoding="utf-8")
    output = tmp_path / "must-not-publish.sv"
    result = _emit(candidate, "verilog", output)
    assert result.returncode != 0
    assert result.stderr
    assert not output.exists()


# Independent next-assignment source programs. Each entry is
# (fields, nonlocal names, body lines).
SOURCE_CASES = {
    "unconditional": (
        ["state: Word = 254"],
        ["state"],
        ["state = (state + 1) & 255"],
    ),
    "enable-guarded": (
        ["state: Word = 254", "en: bool = True"],
        ["state"],
        ["if en:", "    state = (state + 1) & 255"],
    ),
    "conditional-comparison": (
        ["state: Word = 254", "other: Word = 3"],
        ["state"],
        ["if other == 0:", "    state = (state + 1) & 255"],
    ),
}


def _counter_source(fields, nonlocals_, body_lines) -> str:
    declarations = "".join(f"    {field}\n" for field in fields)
    body = "".join(f"        {line}\n" for line in body_lines)
    return (
        "from pycircuit import module, rule\nfrom .types import Word\n\n"
        f"@module\ndef Counter():\n{declarations}\n"
        f"    @rule\n    def tick():\n        nonlocal {', '.join(nonlocals_)}\n"
        f"{body}\n    tick()\n"
    )


def test_next_assignment_source_variants_link_and_emit(tmp_path: Path) -> None:
    """Each source program is compiled from real Python, linked to
    its own artifact, and then actually emitted by both backends to distinct
    paths, so the fail-closed guard cannot pass by only linking."""
    for name, (fields, nonlocals_, body_lines) in SOURCE_CASES.items():
        case = tmp_path / name
        case.mkdir()
        linked = _linked_from_source(
            case, _counter_source(fields, nonlocals_, body_lines)
        )
        assert linked.returncode == 0, f"{name}: {linked.stderr}"
        design = case / "counter.ac"
        assert design.is_file(), name

        for target, extension in (("cpp", "cpp"), ("verilog", "sv")):
            output = case / f"counter.{extension}"
            assert not output.exists(), f"{name}/{target}: stale output path"
            emitted = _emit(design, target, output)
            assert emitted.returncode == 0, f"{name}/{target}: {emitted.stderr}"
            assert output.is_file(), f"{name}/{target}: no backend output"
            assert output.stat().st_size > 0, f"{name}/{target}: empty output"
