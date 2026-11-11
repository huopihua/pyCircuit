"""Independent declaration-only record profile checks for final nominal record declarations.

These cases intentionally stop at declarations. They exercise source/header
authority, the bounded two-field final projection, and final-only verification;
they do not claim record state or backend execution support.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import test_cpp_source_parts as cpp
import test_driver_compile_link as driver
import test_final_value_declarations as value

pytestmark = pytest.mark.system


PROVIDER = """\
from pycircuit import struct, u1, u8, u64

@struct
class Pair:
    lo: u8
    hi: u1

@struct
class Wide:
    low: u64
    high: u64

@struct
class _Private:
    first: u1
    second: u8
"""

THREE_FIELDS = """\
from pycircuit import struct, u4

@struct
class TooWideForThisSlice:
    first: u4
    second: u4
    third: u4
"""

ROOT_SOURCE = """\
from pycircuit import module, struct, u1, u8
from .facade import ExportedPair

@struct
class LocalPair:
    first: u1
    second: u8

@module
def Top():
    value: u8 = 11
"""


def _build(
    directory: Path,
    *,
    provider: str = PROVIDER,
    ordinary_helper: bool = False,
    root_source: str = ROOT_SOURCE,
    constructor_mutation: tuple[str, str, str] | None = None,
):
    source_root = directory / "source"
    source_root.mkdir(parents=True)
    source_texts = {
        "provider.py": provider,
        "facade.py": "from .provider import Pair as ExportedPair\n",
        "empty.py": '"""An intentionally empty source unit."""\n',
    }
    if ordinary_helper:
        source_texts["helper.py"] = (
            "from .provider import Pair\ndef unused() -> Pair:\n"
            "    return Pair(lo=3, hi=0)\n"
        )
    source_texts["root.py"] = root_source
    units = {}
    provider_body = None
    for index, (relative, contents) in enumerate(source_texts.items()):
        if relative != "provider.py" and provider_body is None:
            # Prove consumers compile with the owning source and body absent;
            # the provider interface header is their only declaration source.
            provider_body = units["provider.py"].body.read_bytes()
            (source_root / "provider.py").unlink()
            units["provider.py"].body.unlink()
        source = source_root / relative
        source.write_text(contents, encoding="utf-8")
        units[relative] = cpp._compile_source(
            source,
            source_root=source_root,
            output_dir=directory / "units" / str(index),
            headers=tuple(unit.header for unit in units.values()),
        )
    mutation_count = 0
    if constructor_mutation is not None:
        symbol, before, after = constructor_mutation
        for unit in units.values():
            for artifact in (unit.body, unit.header):
                if not artifact.is_file():
                    continue
                source = artifact.read_text(encoding="utf-8")
                changed, count = _mutate_constructor_body(source, symbol, before, after)
                if count:
                    artifact.write_text(changed, encoding="utf-8")
                    mutation_count += count
        minimum = 3 if symbol.endswith("Pair.__init__") else 1
        assert mutation_count >= minimum, (symbol, mutation_count, before)
    (source_root / "provider.py").write_text(provider, encoding="utf-8")
    units["provider.py"].body.write_bytes(provider_body)
    final = directory / "root.final.ac"
    linked = cpp._link(list(units.values()), final, top="demo.root.Top")
    return source_root, units, final, linked


def _mutate_constructor_body(
    text: str, symbol: str, before: str, after: str
) -> tuple[str, int]:
    """Edit only a named constructor body while keeping all snapshots aligned."""
    marker = f"func.func @{symbol}"
    count = 0
    offset = 0
    while (start := text.find(marker, offset)) >= 0:
        header_end = text.find("\n", start)
        if header_end < 0 or not text[start:header_end].rstrip().endswith("{"):
            offset = start + len(marker)
            continue
        body_end = text.find("\n  }", header_end)
        if body_end < 0:
            raise AssertionError(f"unclosed constructor function {symbol}")
        body = text[header_end:body_end]
        occurrences = body.count(before)
        if occurrences:
            body = body.replace(before, after, 1)
            text = text[:header_end] + body + text[body_end:]
            count += 1
            offset = header_end + len(body)
        else:
            offset = body_end + 4
    return text, count


def _verify(final: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cpp._design_harness(), "--design", str(final), "--verify-only"],
        text=True,
        capture_output=True,
        check=False,
    )


def _error_text(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        line.split("error:", 1)[1]
        for line in result.stderr.splitlines()
        if "error:" in line
    )


def _record_lines(text: str) -> dict[str, str]:
    records = {}
    for line in text.splitlines():
        if '"ac.struct"' not in line and not re.match(r"\s*ac\.struct\b", line):
            continue
        symbol = re.search(r'ac\.struct\s+"([^"]+)"', line)
        if symbol:
            records[symbol.group(1)] = line.strip()
            continue
        symbol = re.search(r'\bsym_name\s*=\s*"([^"]+)"', line)
        if symbol:
            records[symbol.group(1)] = line.strip()
        else:
            custom = re.search(r"ac\.struct\s+@([^\s{]+)", line)
            assert custom, line
            records[custom.group(1)] = line.strip()
    return records


def _expect_invalid(final: Path, text: str, tmp_path: Path):
    mutated = tmp_path / "mutated.final.ac"
    tmp_path.mkdir(parents=True, exist_ok=True)
    mutated.write_text(text, encoding="utf-8")
    checked = _verify(mutated)
    assert checked.returncode == 1, checked.stderr
    for target in ("cpp", "verilog"):
        output = tmp_path / f"mutated.{target}"
        assert not output.exists()
        native_target = "structural-cpp" if target == "cpp" else target
        emitted = subprocess.run(
            [
                cpp._design_harness(),
                "--design",
                str(mutated),
                "--target",
                native_target,
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert emitted.returncode == 1, emitted.stderr
        assert not output.exists()


def test_two_field_declarations_are_owned_ordered_and_final_only(tmp_path: Path):
    source_root, units, final, linked = _build(tmp_path)
    assert linked.returncode == 0, linked.stderr
    assert final.is_file()

    text = final.read_text(encoding="utf-8")
    records = _record_lines(text)
    expected = {
        "demo.provider.Pair": ("lo", "hi"),
        "demo.provider.Wide": ("low", "high"),
        "demo.provider._Private": ("first", "second"),
        "demo.root.LocalPair": ("first", "second"),
    }
    assert set(records) == set(expected)
    assert list(records) == [
        "demo.provider._Private",
        "demo.provider.Pair",
        "demo.provider.Wide",
        "demo.root.LocalPair",
    ]
    for symbol, field_names in expected.items():
        line = records[symbol]
        assert 'ac.declaration_role = "definition"' in line
        positions = [line.index(f'name = "{name}"') for name in field_names]
        assert positions == sorted(positions), (symbol, line)
        assert "func.func" not in line

    for symbol in expected:
        final_fields = re.search(
            r" fields (.*) \{ac\.declaration_role =", records[symbol]
        )
        assert final_fields, records[symbol]
        owner_path = symbol.removeprefix("demo.").rsplit(".", 1)[0] + ".py"
        source_unit = units[owner_path]
        header_line = _record_lines(source_unit.header.read_text())[symbol]
        header_fields = re.search(
            r" fields (.*) \{ac\.declaration_role =", header_line
        )
        assert header_fields, header_line
        assert final_fields.group(1) == header_fields.group(1), symbol

    provider_header = units["provider.py"].header.read_text(encoding="utf-8")
    assert "demo.provider.Pair" in provider_header
    assert "demo.provider.Wide" in provider_header
    facade_header = units["facade.py"].header.read_text(encoding="utf-8")
    assert "demo.provider.Pair" in facade_header
    assert "demo.facade.ExportedPair" not in records
    assert "demo.facade.Pair" not in records
    assert "demo.facade.ExportedPair" not in text
    assert ("demo", "empty.py") in value._units(text)
    assert "demo.root.Top" in text

    # Nominal struct construction is carried by ac.struct itself; final ACIR
    # retains one canonical declaration and no source helper body.
    assert text.count('ac.struct "demo.provider.Pair"') == 1
    assert text.count('ac.struct "demo.provider.Wide"') == 1
    assert "func.func @demo.provider.Pair.__init__" not in text
    assert "func.func @demo.provider.Wide.__init__" not in text

    # The saved package verifies in a fresh process after all Python sources
    # and source transports are unavailable.
    shutil.rmtree(source_root)
    shutil.rmtree(tmp_path / "units")
    checked = _verify(final)
    assert checked.returncode == 0, checked.stderr


def test_source_header_mismatch_rejects_but_three_field_structs_link(
    tmp_path: Path,
):
    source_root, units, _final, _linked = _build(tmp_path / "valid")

    # Alter only the supplied provider header after body capture. The link
    # boundary must detect that a consumer cannot redefine provider authority.
    header = units["provider.py"].header
    original = header.read_text(encoding="utf-8")
    pair = _record_lines(original)["demo.provider.Pair"]
    changed = original.replace(pair, pair.replace('name = "lo"', 'name = "bad"', 1), 1)
    assert changed != original
    header.write_text(changed, encoding="utf-8")
    mismatch = cpp._link(
        list(units.values()),
        tmp_path / "mismatch.final.ac",
        top="demo.root.Top",
    )
    assert mismatch.returncode == 1, mismatch.stderr

    # Independent HWIR supports nominal structs beyond the old two-field
    # final-only slice, including unused declarations.
    source_root2 = tmp_path / "wide-source"
    source_root2.mkdir()
    wide_source = source_root2 / "provider.py"
    wide_source.write_text(THREE_FIELDS, encoding="utf-8")
    unit = cpp._compile_source(
        wide_source, source_root=source_root2, output_dir=tmp_path / "wide-unit"
    )
    assert unit.header.is_file()
    root = source_root2 / "root.py"
    root.write_text(
        "from typing import Annotated\nfrom pycircuit import module\n"
        "Word = Annotated[int, range(16)]\n"
        "@module\ndef Top():\n    state: Word = 0\n"
    )
    root_unit = cpp._compile_source(
        root,
        source_root=source_root2,
        output_dir=tmp_path / "wide-root",
        headers=(unit.header,),
    )
    linked = cpp._link(
        [unit, root_unit],
        tmp_path / "wide.final.ac",
        top="demo.root.Top",
    )
    assert linked.returncode == 0, linked.stderr
    value._accept_hwir_backends(tmp_path / "wide.final.ac", tmp_path / "wide-backends")


def test_final_record_authority_and_shape_mutations_fail_closed(tmp_path: Path):
    _source_root, _units, final, linked = _build(tmp_path)
    assert linked.returncode == 0, linked.stderr
    text = final.read_text(encoding="utf-8")
    record = _record_lines(text)["demo.provider.Pair"]

    mutations = {
        "missing-field": text.replace(
            record,
            re.sub(r'name = "hi"', "", record, count=1),
            1,
        ),
        "duplicate-field": text.replace(
            record, record.replace('name = "hi"', 'name = "lo"', 1), 1
        ),
    }
    for name, mutated in mutations.items():
        assert mutated != text, name
        _expect_invalid(final, mutated, tmp_path / name)
    package = text.index("module attributes {")
    attributes_open = text.index("{", package)
    attributes_end = value._balanced(text, attributes_open)
    body_open = text.index("{", attributes_end + 1)
    body_end = value._balanced(text, body_open)
    _expect_invalid(
        final, text[:body_end] + text[body_end + 1 :], tmp_path / "unclosed"
    )


@pytest.mark.parametrize(
    "case", ["unknown-property", "field-origin", "field-location-path", "owner"]
)
def test_unused_struct_provenance_does_not_constrain_hwir_projection(
    tmp_path: Path, case: str
):
    _source_root, _units, final, linked = _build(tmp_path)
    assert linked.returncode == 0, linked.stderr
    text = final.read_text(encoding="utf-8")
    record = _record_lines(text)["demo.provider.Pair"]
    if case == "unknown-property":
        changed = record.replace(
            "ac.source_owner = {", "ac.unknown = 1, ac.source_owner = {", 1
        )
    elif case == "field-origin":
        changed = record.replace(
            "definition = @demo.provider.Pair", "definition = @demo.provider.Wide", 1
        )
    elif case == "field-location-path":
        changed = record.replace('path = "provider.py"', 'path = "facade.py"', 1)
    else:
        changed = record.replace(
            'ac.source_owner = {package = "demo", path = "provider.py"}',
            'ac.source_owner = {package = "demo", path = "facade.py"}',
            1,
        )
    assert changed != record
    variant = tmp_path / f"{case}.ac"
    variant.write_text(text.replace(record, changed, 1), encoding="utf-8")
    value._accept_hwir_backends(variant, tmp_path / f"{case}-backends")


def test_record_declaration_final_emit_is_explicitly_not_backend_acceptance(
    tmp_path: Path,
):
    _source_root, _units, final, linked = _build(tmp_path)
    assert linked.returncode == 0, linked.stderr
    # declaration-only record profile establishes declaration verification. Until later record storage/backend integration,
    # no final-only record artifact may accidentally claim executable support.
    for target in ("cpp", "verilog"):
        output = tmp_path / f"not-yet-supported.{target}"
        native_target = "structural-cpp" if target == "cpp" else target
        emitted = subprocess.run(
            [
                cpp._design_harness(),
                "--design",
                str(final),
                "--target",
                native_target,
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert emitted.returncode == 1, emitted.stderr
        assert not output.exists()


def test_unused_ordinary_helper_is_erased_before_hwir_projection(tmp_path: Path):
    _source_root, units, final, linked = _build(tmp_path, ordinary_helper=True)
    assert units["helper.py"].body.is_file()
    assert linked.returncode == 0, linked.stderr
    assert "demo.helper.unused" not in final.read_text(encoding="utf-8")
    value._accept_hwir_backends(final, tmp_path / "helper-backends")


def test_nested_struct_declarations_link_and_reach_both_hwir_backends(
    tmp_path: Path,
):
    nested = (
        PROVIDER
        + """\
@struct
class Inner:
    value: u8

@struct
class Outer:
    inner: Inner
    flag: u1
"""
    )
    _source_root, units, final, linked = _build(tmp_path, provider=nested)
    assert units["provider.py"].header.is_file()
    assert linked.returncode == 0, linked.stderr
    value._accept_hwir_backends(final, tmp_path / "nested-backends")


def test_imported_struct_persistent_initializer_is_rejected_by_frontend(
    tmp_path: Path,
):
    stateful = """\
from pycircuit import module
from .provider import Pair

@module
def Top():
    state: Pair = Pair(lo=3, hi=0)
"""
    with pytest.raises(AssertionError, match="persistent initializer must be a static complete value"):
        _build(tmp_path, root_source=stateful)


def test_public_compile_link_publishes_final_declarations_and_emit_fails_closed(
    tmp_path: Path,
):
    cli_environment = dict(os.environ)
    cli_environment["PYCIRCUIT_SOURCE_COMPILER"] = cpp._tool(
        "PYCIRCUIT_SOURCE_COMPILER", "pycircuit-source-unit"
    )
    cli_environment["PYCIRCUIT_LINKER"] = cpp._tool(
        "PYCIRCUIT_LINKER", "pycircuit-link"
    )
    roots = (
        cpp.ROOT / "python",
        cpp.ROOT / "python/semantic-core/src",
    )
    inherited = cli_environment.get("PYTHONPATH")
    cli_environment["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in roots), *([inherited] if inherited else [])]
    )
    workspace = driver._Workspace(
        root=tmp_path / "cli-source",
        units=tmp_path / "cli-units",
        out=tmp_path / "cli-out",
    )
    workspace.root.mkdir()
    workspace.units.mkdir()
    workspace.out.mkdir()
    sources = {
        "provider.py": PROVIDER,
        "facade.py": "from .provider import Pair as ExportedPair\n",
        "empty.py": '"""Empty declaration source."""\n',
        "root.py": ROOT_SOURCE,
    }
    for relative, content in sources.items():
        (workspace.root / relative).write_text(content, encoding="utf-8")

    assert driver._ok(driver._compile(cli_environment, workspace, "provider.py"))
    provider_unit = workspace.units / "provider"
    receipt = json.loads((provider_unit / "unit.json").read_text())
    assert receipt["files"] == {
        "body": "provider.ac",
        "interface": "provider.interface.ac",
        "depfile": "provider.d",
    }
    provider_body = (provider_unit / receipt["files"]["body"]).read_bytes()
    (workspace.root / "provider.py").unlink()
    (provider_unit / receipt["files"]["body"]).unlink()

    assert driver._ok(
        driver._compile(
            cli_environment,
            workspace,
            "facade.py",
            interface=(provider_unit,),
        )
    )
    facade_unit = workspace.units / "facade"
    assert driver._ok(driver._compile(cli_environment, workspace, "empty.py"))
    empty_unit = workspace.units / "empty"
    assert driver._ok(
        driver._compile(
            cli_environment,
            workspace,
            "root.py",
            interface=(provider_unit, facade_unit),
        )
    )
    root_unit = workspace.units / "root"
    (workspace.root / "provider.py").write_text(PROVIDER, encoding="utf-8")
    (provider_unit / receipt["files"]["body"]).write_bytes(provider_body)

    final = workspace.out / "design_top.ac"
    linked = driver._link(
        cli_environment,
        (provider_unit, facade_unit, empty_unit, root_unit),
        "demo.root.Top",
        final,
    )
    assert linked.returncode == 0, linked.stderr
    final_text = final.read_text(encoding="utf-8")
    assert "demo.provider.Pair" in final_text
    assert "demo.provider._Private" in final_text
    assert "demo.facade.ExportedPair" not in final_text

    for target in ("structural-cpp", "verilog"):
        destination = workspace.out / f"sentinel.{target}"
        destination.write_bytes(b"preserve this existing output\n")
        emitted = driver._cli(
            cli_environment,
            "emit",
            str(final),
            "--target",
            target,
            "-o",
            str(destination),
        )
        driver._diagnostic(emitted, "emit")
        assert destination.read_bytes() == b"preserve this existing output\n"
