"""Independent value-declaration source inventory, final-only, and C++ header oracles."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import test_cpp_source_parts as cpp

pytestmark = pytest.mark.system

PROVIDER = """\
from typing import Annotated
Word = Annotated[int, range(256)]
Unused = Annotated[int, range(16)]
_Hidden = Annotated[int, range(8)]
Flag = bool
UnsignedOne = Annotated[int, range(2)]
SignedOne = Annotated[int, range(2)]
NarrowSigned = Annotated[int, range(256)]
Std = Annotated[int, range(16)]
Gfsim = Annotated[int, range(16)]
FalseValue = False
TrueValue = True
Zero = 0
One = 1
MinusOne = 1
Minimum = 0
Maximum = 18446744073709551615
"""

ROOT_SOURCE = """\
from typing import Annotated
from pycircuit import module
from .facade import PublicWord
Own = Annotated[int, range(32)]
Std = Annotated[int, range(16)]
Gfsim = Annotated[int, range(16)]
Limit = 9
@module
def Root():
    state: PublicWord = 7
"""

PROVIDER_NAMES = {
    "Word",
    "Unused",
    "_Hidden",
    "Flag",
    "UnsignedOne",
    "SignedOne",
    "NarrowSigned",
    "Std",
    "Gfsim",
    "FalseValue",
    "TrueValue",
    "Zero",
    "One",
    "MinusOne",
    "Minimum",
    "Maximum",
}
ROOT_NAMES = {"Own", "Std", "Gfsim", "Limit"}


def _balanced(text: str, start: int) -> int:
    """Match braces while ignoring quoted MLIR strings."""
    assert text[start] == "{"
    depth = 0
    quoted = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if not depth:
                return index
    raise AssertionError("unclosed MLIR region or attribute")


def _units(text: str) -> dict[tuple[str, str], str]:
    package = re.search(r"(?m)^module attributes \{", text)
    assert package, "final package has no top-level module"
    attributes_start = text.index("{", package.start())
    attributes_end = _balanced(text, attributes_start)
    attributes = text[attributes_start : attributes_end + 1]
    inventory = re.search(r"ac\.source_units = \[(.*?)\]", attributes, re.S)
    assert inventory, "final package has no source-unit inventory"
    owners = [
        (match.group(1), match.group(2))
        for match in re.finditer(
            r'\{package = "([^"]*)", path = "([^"]+)"\}',
            inventory.group(1),
        )
    ]
    assert owners and len(owners) == len(set(owners)), "invalid source-unit inventory"
    units = {owner: "" for owner in owners}

    body_start = text.index("{", attributes_end + 1) + 1
    body_end = _balanced(text, body_start - 1)
    starts_and_ends: list[tuple[int, int]] = []
    start = body_start
    while start < body_end:
        while start < body_end and text[start].isspace():
            start += 1
        if start >= body_end:
            break
        depth = 0
        quoted = escaped = False
        end = start
        while end < body_end:
            char = text[end]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "({[":
                depth += 1
            elif char in ")} ]".replace(" ", ""):
                depth -= 1
            elif char == "\n" and depth == 0:
                end += 1
                break
            end += 1
        starts_and_ends.append((start, end))
        start = end
    for start, end in starts_and_ends:
        operation = text[start:end]
        owner = re.search(
            r'(?:ac\.)?source_owner = \{package = "([^"]*)", path = "([^"]+)"\}',
            operation,
        )
        if not owner:
            continue
        key = owner.group(1), owner.group(2)
        if key in units:
            units[key] += operation
    return units


def _top_level_operation(text: str, needle: str) -> str:
    matches = [operation for operation in _all_top_level_operations(text) if needle in operation]
    assert len(matches) == 1, (needle, len(matches))
    return matches[0]


def _all_top_level_operations(text: str) -> list[str]:
    package = re.search(r"(?m)^module attributes \{", text)
    assert package
    attributes_start = text.index("{", package.start())
    attributes_end = _balanced(text, attributes_start)
    body_start = text.index("{", attributes_end + 1) + 1
    body_end = _balanced(text, body_start - 1)
    operations: list[str] = []
    start = body_start
    while start < body_end:
        while start < body_end and text[start].isspace():
            start += 1
        if start >= body_end:
            break
        depth = 0
        quoted = escaped = False
        end = start
        while end < body_end:
            char = text[end]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "({[":
                depth += 1
            elif char in ")} ]".replace(" ", ""):
                depth -= 1
            elif char == "\n" and depth == 0:
                end += 1
                break
            end += 1
        operations.append(text[start:end])
        start = end
    return operations


def _insert_top_level_operation(text: str, operation: str) -> str:
    package = re.search(r"(?m)^module attributes \{", text)
    assert package
    attributes_start = text.index("{", package.start())
    attributes_end = _balanced(text, attributes_start)
    body_start = text.index("{", attributes_end + 1) + 1
    body_end = _balanced(text, body_start - 1)
    if not operation.endswith("\n"):
        operation += "\n"
    return text[:body_end] + operation + text[body_end:]


def _declarations(text: str) -> dict[str, str]:
    declarations = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(
            ('"ac.type_alias"', "ac.type_alias ", '"ac.constant"')
        ):
            continue
        symbol = re.search(r'\bsym_name = "([^"]+)"', line)
        if symbol:
            name = symbol.group(1)
        else:
            custom = re.match(
                r'ac\.type_alias\s+(?:"([^"]+)"|@"([^"]+)"|@([^\s]+))',
                stripped,
            )
            assert custom, line
            name = next(value for value in custom.groups() if value is not None)
        assert name not in declarations, f"duplicate declaration: {name}"
        declarations[name] = line
    return declarations


def _build(
    directory: Path,
    *,
    provider: str = PROVIDER,
    extra: dict[str, str] | None = None,
    reverse: bool = False,
) -> tuple[Path, dict[str, cpp.SourceUnit]]:
    source_root = directory / "source"
    source_root.mkdir(parents=True)
    sources = {
        "provider.py": provider,
        "facade.py": "from .provider import Word as PublicWord\n",
        "empty.py": '"""Empty declaration source."""\n',
        **(extra or {}),
        "root.py": ROOT_SOURCE,
    }
    units = {}
    for index, (relative, contents) in enumerate(sources.items()):
        source = source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(contents, encoding="utf-8")
        headers = tuple(unit.header for unit in units.values())
        units[relative] = cpp._compile_source(
            source,
            source_root=source_root,
            output_dir=directory / "units" / str(index),
            headers=headers,
        )
    final = directory / "root.final.ac"
    ordered = list(units.values())
    linked = cpp._link(
        list(reversed(ordered)) if reverse else ordered,
        final,
        top="demo.root.Root",
    )
    assert linked.returncode == 0, linked.stderr
    assert final.is_file()
    return final, units


def _verify(final: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cpp._design_harness(), "--design", str(final), "--verify-only"],
        text=True,
        capture_output=True,
        check=False,
    )


def _emit(final: Path, target: str, output: Path) -> subprocess.CompletedProcess[str]:
    return cpp._emit_source(final, target, output)


def _parts(final: Path) -> dict:
    emitted = cpp._emit_parts(final)
    assert emitted.returncode == 0, emitted.stderr
    return json.loads(emitted.stdout)


def test_complete_owning_inventory_survives_final_and_unit_reordering(tmp_path: Path):
    extra = {
        "std.py": "from typing import Annotated\nValue = Annotated[int, range(4)]\n",
        "pkg/__init__.py": "from typing import Annotated\nTag = Annotated[int, range(8)]\n",
    }
    final, sources = _build(tmp_path, extra=extra)
    text = final.read_text(encoding="utf-8")
    units = _units(text)
    assert set(units) == {("demo", path) for path in sources}
    expected = {
        "provider.py": {f"demo.provider.{name}" for name in PROVIDER_NAMES},
        "root.py": {f"demo.root.{name}" for name in ROOT_NAMES},
        "facade.py": set(),
        "empty.py": set(),
        "std.py": {"demo.std.Value"},
        "pkg/__init__.py": {"demo.pkg.Tag"},
    }
    for path, names in expected.items():
        actual = _declarations(units["demo", path])
        assert set(actual) == names, path
        assert list(actual) == sorted(actual), f"{path}: canonical declaration order"
        owning = {
            name
            for name, line in _declarations(sources[path].header.read_text()).items()
            if 'ac.declaration_role = "definition"' in line
        }
        assert owning == names, f"{path}: independent source inventory"
        assert units["demo", path].count('"ac.module"') == (path == "root.py")
    for residue in (
        "ac.interfaces",
        "ac.exports",
        "ac.import_bindings",
    ):
        assert residue not in text
    # Final ACIR keeps only the closed snapshots needed by the hardware body;
    # it no longer wraps every source in a nested unit module.
    assert text.count('"ac.module.import"') == 1
    assert text.count('callee = @pycircuit.__builtins__.dffe') == 1
    reversed_final, _ = _build(tmp_path / "reversed", extra=extra, reverse=True)
    assert _parts(final) == _parts(reversed_final)


def test_final_only_structural_header_is_standalone_and_groups_are_complete(
    tmp_path: Path,
):
    final, _ = _build(
        tmp_path,
        extra={
            "std.py": "from typing import Annotated\nValue = Annotated[int, range(4)]\n",
            "pkg/__init__.py": "from typing import Annotated\nTag = Annotated[int, range(8)]\n",
        },
    )
    shutil.rmtree(tmp_path / "source")
    shutil.rmtree(tmp_path / "units")
    checked = _verify(final)
    assert checked.returncode == 0, checked.stderr
    for target in ("cpp", "verilog"):
        emitted = _emit(final, target, tmp_path / f"standalone.{target}")
        assert emitted.returncode == 0, emitted.stderr
    payload = _parts(final)
    output, groups = cpp._write_parts(tmp_path, payload)
    by_path = {group["source"]["path"]: group for group in groups}
    assert set(by_path) == {
        "provider.py",
        "facade.py",
        "empty.py",
        "std.py",
        "pkg/__init__.py",
        "root.py",
    }
    assert len({group["path"] for group in groups}) == len(groups)
    assert all(group["path"].endswith(".cpp") for group in groups)
    assert all(group["text"].startswith('#include "design.hpp"') for group in groups)
    for path in ("facade.py", "empty.py", "std.py", "pkg/__init__.py"):
        assert by_path[path]["text"].strip() == '#include "design.hpp"'
    objects = cpp._compile_groups(output, groups, tmp_path)
    assert len(objects) == len(groups)
    first = tmp_path / "first.cpp"
    second = tmp_path / "second.cpp"
    first.write_text(
        '#include "design.hpp"\n'
        "#include <type_traits>\n"
        "static_assert(!std::is_copy_constructible_v<pyc_dut>);\n"
        "int second_consumer();\n"
        "int main() { return second_consumer(); }\n",
        encoding="utf-8",
    )
    second.write_text(
        '#include "design.hpp"\n'
        "int second_consumer() { return sizeof(pyc_dut) > 0 ? 0 : 1; }\n",
        encoding="utf-8",
    )
    for index, source in enumerate((first, second)):
        obj = tmp_path / f"consumer-{index}.o"
        result = subprocess.run(
            [
                cpp._cxx(),
                "-std=c++20",
                *cpp._include_args(output),
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        objects.append(obj)
    executable = tmp_path / "headers-odr"
    linked = subprocess.run(
        [
            cpp._cxx(),
            *(str(obj) for obj in objects),
            "-o",
            str(executable),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert linked.returncode == 0, linked.stderr
    ran = subprocess.run([str(executable)], text=True, capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr


def _edit_declaration(text: str, symbol: str, change) -> str:
    original = _declarations(text)[symbol]
    replacement = change(original)
    assert replacement != original, f"mutation did not apply: {symbol}"
    return text.replace(original, replacement, 1)


def _set_dictionary(line: str, field: str, value: str) -> str:
    match = re.search(rf"\b{re.escape(field)}(?:\s*=\s*|\s+)\{{", line)
    assert match, (field, line)
    start = line.index("{", match.start())
    return line[:start] + value + line[_balanced(line, start) + 1 :]


def _edit_unit(text: str, path: str, change) -> str:
    original = _units(text)["demo", path]
    assert original, f"source unit has no owned operations: {path}"
    replacement = change(original)
    assert replacement != original, path
    return text.replace(original, replacement, 1)


def _insert_unit(text: str, path: str, body: str = "", package: str = "demo") -> str:
    key = package, path
    units = _units(text)
    assert key not in units
    package_match = re.search(r"(?m)^module attributes \{", text)
    assert package_match
    attributes_start = text.index("{", package_match.start())
    attributes_end = _balanced(text, attributes_start)
    attributes = text[attributes_start : attributes_end + 1]
    inventory = re.search(r"ac\.source_units = \[(.*?)\]", attributes, re.S)
    assert inventory
    owners = sorted([*units, key])
    rendered = ", ".join(
        f'{{package = "{owner_package}", path = "{owner_path}"}}'
        for owner_package, owner_path in owners
    )
    inventory_start = attributes_start + inventory.start(1)
    inventory_end = attributes_start + inventory.end(1)
    text = text[:inventory_start] + rendered + text[inventory_end:]
    if not body:
        return text
    operation = re.sub(r"(?m)^    " , "  ", body)
    if not operation.endswith("\n"):
        operation += "\n"
    package_match = re.search(r"(?m)^module attributes \{", text)
    attributes_start = text.index("{", package_match.start())
    attributes_end = _balanced(text, attributes_start)
    body_start = text.index("{", attributes_end + 1) + 1
    body_end = _balanced(text, body_start - 1)
    return text[:body_end] + operation + text[body_end:]


def _error_text(result: subprocess.CompletedProcess[str]) -> str:
    # Do not let a pytest temp directory's name satisfy diagnostic assertions.
    lines = [
        line.split("error:", 1)[1]
        for line in result.stderr.splitlines()
        if "error:" in line
    ]
    return "\n".join(lines)


def _reject_final(path: Path, pattern: str) -> None:
    checked = _verify(path)
    assert checked.returncode == 1, checked.stderr
    assert re.search(pattern, _error_text(checked), re.IGNORECASE), checked.stderr
    for target in ("cpp", "verilog"):
        output = path.parent / f"rejected.{target}"
        assert not output.exists()
        emitted = _emit(path, target, output)
        assert emitted.returncode == 1, emitted.stderr
        assert re.search(pattern, _error_text(emitted), re.IGNORECASE), emitted.stderr
        assert not output.exists()
    parts = cpp._emit_parts(path)
    assert parts.returncode == 1 and parts.stdout == "", parts.stderr


def _reject_cpp_only(final: Path, pattern: str) -> None:
    checked = _verify(final)
    assert checked.returncode == 0, checked.stderr
    rtl = _emit(final, "verilog", final.parent / "accepted.sv")
    assert rtl.returncode == 0, rtl.stderr
    sentinel = final.parent / "existing.cpp"
    sentinel.write_bytes(b"previous user artifact\n")
    cpp_result = _emit(final, "cpp", sentinel)
    assert cpp_result.returncode == 1, cpp_result.stderr
    assert re.search(pattern, _error_text(cpp_result), re.IGNORECASE), cpp_result.stderr
    assert sentinel.read_bytes() == b"previous user artifact\n"
    parts = cpp._emit_parts(final)
    assert parts.returncode == 1 and parts.stdout == "", parts.stderr
    assert re.search(pattern, _error_text(parts), re.IGNORECASE), parts.stderr


def _accept_hwir_backends(final: Path, tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checked = _verify(final)
    assert checked.returncode == 0, checked.stderr
    for target in ("cpp", "verilog"):
        emitted = _emit(final, target, tmp_path / f"accepted.{target}")
        assert emitted.returncode == 0, emitted.stderr
    output, groups = cpp._write_parts(tmp_path, _parts(final))
    cpp._compile_groups(output, groups, tmp_path)


def test_signed_and_integer_one_bit_final_declarations_survive_hwir_emit(
    tmp_path: Path,
):
    # Focused compiler-input transformations: this test does not claim new
    # signed-range or negative-literal Python frontend support.
    final, _ = _build(tmp_path)
    text = final.read_text()
    for symbol, target in (
        (
            "SignedOne",
            '{kind = "integer", storage = i1, lower = #ac.math_int<-1>, '
            'upper = #ac.math_int<1>, interpretation = "signed"}',
        ),
        (
            "NarrowSigned",
            '{kind = "integer", storage = i4, lower = #ac.math_int<-8>, '
            'upper = #ac.math_int<8>, interpretation = "signed"}',
        ),
    ):
        text = _edit_declaration(
            text,
            "demo.provider." + symbol,
            lambda line, target=target: _set_dictionary(line, "target", target),
        )
    for symbol, value in (("MinusOne", -1), ("Minimum", -(1 << 63))):
        text = _edit_declaration(
            text,
            "demo.provider." + symbol,
            lambda line, value=value: _set_dictionary(
                line, "value", f'{{kind = "integer", value = #ac.math_int<{value}>}}'
            ),
        )
    variant = tmp_path / "signed.ac"
    variant.write_text(text)
    verified = _verify(variant)
    assert verified.returncode == 0, verified.stderr
    output, groups = cpp._write_parts(tmp_path, _parts(variant))
    cpp._compile_groups(output, groups, tmp_path)
    source = tmp_path / "signed.cpp"
    source.write_text(
        r"""
#include "design.hpp"
#include <type_traits>
static_assert(!::std::is_copy_constructible_v<pyc_dut>);
static_assert(!::std::is_copy_assignable_v<pyc_dut>);
"""
    )
    result = subprocess.run(
        [
            cpp._cxx(),
            "-std=c++20",
            *cpp._include_args(output),
            "-fsyntax-only",
            str(source),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", [-(1 << 63) - 1, 1 << 64, 1 << 180])
def test_unused_wide_constants_do_not_leak_past_hwir_projection(
    tmp_path: Path, value: int
):
    final, units = _build(
        tmp_path, provider=PROVIDER + f"\nOversized = {max(0, value)}\n"
    )
    if value < 0:
        # Feed a legal negative MathInt source declaration to the native linker.
        # The unsigned cases above originate directly in real Python literals.
        for unit in units.values():
            for artifact in (unit.header, unit.body):
                original = artifact.read_text()
                if "demo.provider.Oversized" not in original:
                    continue
                artifact.write_text(
                    _edit_declaration(
                        original,
                        "demo.provider.Oversized",
                        lambda line: _set_dictionary(
                            line,
                            "value",
                            f'{{kind = "integer", value = #ac.math_int<{value}>}}',
                        ),
                    )
                )
        final = tmp_path / "negative.final.ac"
        linked = cpp._link(list(units.values()), final, top="demo.root.Root")
        assert linked.returncode == 0, linked.stderr
    assert f"#ac.math_int<{value}>" in final.read_text()
    verified = _verify(final)
    assert verified.returncode == 0, verified.stderr
    rtl = _emit(final, "verilog", tmp_path / "accepted.sv")
    assert rtl.returncode == 0, rtl.stderr
    direct = _emit(final, "cpp", tmp_path / "accepted.cpp")
    assert direct.returncode == 0, direct.stderr
    output, groups = cpp._write_parts(tmp_path, _parts(final))
    cpp._compile_groups(output, groups, tmp_path)


def _malformed(text: str, case: str) -> tuple[str, str]:
    symbol = "demo.provider.Word"
    if case == "symbol":
        return (
            _edit_declaration(
                text,
                symbol,
                lambda line: line.replace("demo.provider.Word", "demo.foreign.Word", 1),
            ),
            r"owner|symbol|origin|definition",
        )
    if case == "owner":
        return (
            _edit_declaration(
                text,
                symbol,
                lambda line: line.replace(
                    'path = "provider.py"', 'path = "foreign.py"', 1
                ),
            ),
            r"owner|source",
        )
    if case == "origin":
        return (
            _edit_declaration(
                text,
                symbol,
                lambda line: re.sub(
                    r'definition = @(?:"demo\.provider\.Word"|demo\.provider\.Word)',
                    'definition = @"demo.provider.Other"',
                    line,
                    count=1,
                ),
            ),
            r"origin|definition",
        )
    if case in {"snapshot", "unknown-role", "unknown-field"}:
        replacement = {
            "snapshot": 'ac.declaration_role = "import_snapshot"',
            "unknown-role": 'ac.declaration_role = "bogus"',
            "unknown-field": 'ac.declaration_role = "definition", ac.unapproved = true',
        }[case]
        return (
            _edit_declaration(
                text,
                symbol,
                lambda line: line.replace(
                    'ac.declaration_role = "definition"', replacement, 1
                ),
            ),
            r"role|snapshot|field|attribute|declaration",
        )
    if case == "width":
        return (
            _edit_declaration(
                text,
                symbol,
                lambda line: line.replace("storage = i8", "storage = i7", 1),
            ),
            r"width|storage|range",
        )
    if case == "bool-value":
        return (
            _edit_declaration(
                text,
                "demo.provider.TrueValue",
                lambda line: line.replace("value = true", "value = #ac.math_int<1>", 1),
            ),
            r"value|bool|type",
        )
    if case in {"exports", "imports", "interfaces"}:
        attr = {
            "exports": "ac.exports",
            "imports": "ac.import_bindings",
            "interfaces": "ac.interfaces",
        }[case]
        return (
            text.replace(
                'ac.stage = "final"', f'{attr} = [], ac.stage = "final"', 1
            ),
            r"attribute|envelope|metadata|package",
        )
    if case == "hardware":
        return (
            _insert_top_level_operation(
                text,
                "  %illegal = arith.constant false\n",
            ),
            r"declaration|operation|hardware|package",
        )
    if case == "helper":
        return (
            _insert_top_level_operation(
                text,
                "  func.func @forbidden() { return }\n",
            ),
            r"declaration|helper|operation|package",
        )
    if case == "nested":
        line = _declarations(text)["demo.root.Own"]
        without = text.replace(line + "\n", "", 1)
        module = _top_level_operation(without, 'sym_name = "demo.root.Root"')
        changed = re.sub(
            r"(?m)^([ \t]*\^bb0[^\n]*\n)",
            lambda match: match.group(1) + "      " + line.strip() + "\n",
            module,
            count=1,
        )
        assert changed != module
        return (
            without.replace(module, changed, 1),
            r"parent|placement|declaration|operation",
        )
    if case == "duplicate":
        line = _declarations(text)[symbol]
        return (
            text.replace(line, line + "\n" + line, 1),
            r"duplicate|redefin|symbol|order",
        )
    if case == "order":
        lines = list(_declarations(_units(text)["demo", "provider.py"]).values())
        first, second = lines[:2]
        return (
            text.replace(first, "__SWAP__", 1)
            .replace(second, first, 1)
            .replace("__SWAP__", second, 1),
            r"order|sort",
        )
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "owner",
        "width",
        "bool-value",
        "hardware",
        "helper",
        "duplicate",
    ],
)
def test_common_declaration_rejections_cover_verify_and_both_backends(
    tmp_path: Path, case: str
):
    final, _ = _build(tmp_path)
    original = final.read_text()
    text, pattern = _malformed(original, case)
    assert text != original, case
    invalid = tmp_path / "invalid.ac"
    invalid.write_text(text)
    _reject_final(invalid, pattern)


@pytest.mark.parametrize(
    "case",
    [
        "symbol",
        "origin",
        "snapshot",
        "unknown-role",
        "unknown-field",
        "exports",
        "imports",
        "interfaces",
        "order",
    ],
)
def test_non_hardware_declaration_metadata_does_not_constrain_hwir_projection(
    tmp_path: Path, case: str
):
    final, _ = _build(tmp_path)
    original = final.read_text()
    text, _ = _malformed(original, case)
    assert text != original, case
    variant = tmp_path / "variant.ac"
    variant.write_text(text)
    verified = _verify(variant)
    assert verified.returncode == 0, verified.stderr
    for target in ("cpp", "verilog"):
        emitted = _emit(variant, target, tmp_path / f"accepted.{target}")
        assert emitted.returncode == 0, emitted.stderr
    output, groups = cpp._write_parts(tmp_path, _parts(variant))
    cpp._compile_groups(output, groups, tmp_path)


def test_nested_source_declaration_is_rejected_at_hwir_projection(tmp_path: Path):
    final, _ = _build(tmp_path)
    text, _ = _malformed(final.read_text(), "nested")
    variant = tmp_path / "nested.ac"
    variant.write_text(text)
    verified = _verify(variant)
    assert verified.returncode == 0, verified.stderr
    for result in (
        cpp._emit_parts(variant),
        _emit(variant, "cpp", tmp_path / "rejected.cpp"),
        _emit(variant, "verilog", tmp_path / "rejected.sv"),
    ):
        assert result.returncode == 1
        assert re.search(
            r"operation is not permitted in an HWIR module body",
            _error_text(result),
            re.I,
        ), result.stderr


@pytest.mark.parametrize(
    "paths",
    [
        ("foo.py", "foo/__init__.py"),
        ("case.py", "CASE.py"),
        ("σ.py", "Σ.py"),
        ("ß.py", "ss.py"),
        ("foo/__init__.py", "FOO/__INIT__.py"),
    ],
)
def test_empty_unit_owner_aliases_keep_distinct_structural_source_groups(
    tmp_path: Path, paths
):
    final, _ = _build(tmp_path)
    text = final.read_text()
    for path in paths:
        text = _insert_unit(text, path)
    variant = tmp_path / "variant.ac"
    variant.write_text(text)
    verified = _verify(variant)
    assert verified.returncode == 0, verified.stderr
    payload = _parts(variant)
    groups = payload["structural_cpp_groups"]
    by_source = {
        (group["source"]["package"], group["source"]["path"]): group
        for group in groups
    }
    for path in paths:
        assert ("demo", path) in by_source
    assert len({group["path"] for group in groups}) == len(groups)
    output, groups = cpp._write_parts(tmp_path, payload)
    cpp._compile_groups(output, groups, tmp_path)


@pytest.mark.parametrize("kind", ["list-alias", "list-constant"])
def test_unused_unsupported_value_definitions_are_not_silently_erased(
    tmp_path: Path, kind: str
):
    _, units = _build(tmp_path)
    header = units["provider.py"].header
    if kind == "list-alias":

        def transform(line):
            return _set_dictionary(
                line,
                "target",
                '{kind = "list", element = {kind = "bool", storage = i1}, length = 2 : i64}',
            )

        symbol = "demo.provider.Unused"
    else:

        def transform(line):
            line = _set_dictionary(
                line,
                "type",
                '{kind = "list", element = {kind = "integer"}, length = 2 : i64}',
            )
            return _set_dictionary(
                line,
                "value",
                '{kind = "list", values = [{kind = "integer", value = #ac.math_int<0>}, '
                '{kind = "integer", value = #ac.math_int<1>}]}',
            )

        symbol = "demo.provider.One"
    header.write_text(_edit_declaration(header.read_text(), symbol, transform))
    output = tmp_path / "unsupported.ac"
    result = cpp._link(list(units.values()), output, top="demo.root.Root")
    assert result.returncode == 1, result.stderr
    assert re.search(
        r"value|unsupported|declaration|projection", _error_text(result), re.I
    ), result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "source",
    [
        "\nunused = 3\n",
        "\né = Annotated[int, range(4)]\nue9 = 3\n",
    ],
)
def test_source_declaration_names_are_not_public_cpp_names(
    tmp_path: Path, source: str
):
    final, _ = _build(tmp_path, provider=PROVIDER + source)
    _accept_hwir_backends(final, tmp_path)


@pytest.mark.parametrize("name", ["Std", "Gfsim"])
def test_unused_global_runtime_names_do_not_collide_with_structural_cpp(
    tmp_path: Path, name: str
):
    final, _ = _build(tmp_path)
    body = (
        f'    "ac.type_alias"() <{{sym_name = "{name}", '
        'target = {kind = "bool", storage = i1}}> {'
        'ac.source_owner = {package = "", path = "__init__.py"}, '
        f'ac.origin = {{site = {{definition = @"{name}", ast_path = []}}, expansion = []}}, '
        'ac.declaration_role = "definition"} : () -> () loc("__init__.py":1:1)\n'
    )
    variant = tmp_path / "global-name.ac"
    variant.write_text(_insert_unit(final.read_text(), "__init__.py", body, package=""))
    _accept_hwir_backends(variant, tmp_path)


@pytest.mark.parametrize("case", ["module", "namespace", "path"])
def test_declaration_group_names_are_isolated_from_structural_cpp_names(
    tmp_path: Path, case: str
):
    extra = {
        "namespace": {"provider/unused.py": "Value = 7\n"},
        "path": {"fooBar.py": "Value = 7\n", "foo_bar.py": "Value = 8\n"},
    }.get(case)
    final, _ = _build(tmp_path, extra=extra)
    if case == "module":
        # ROOT and Root are distinct canonical source symbols but both legalize
        # to the C++ family name root. ROOT keeps the existing declaration order.
        text = _edit_declaration(
            final.read_text(),
            "demo.root.Own",
            lambda line: line.replace("demo.root.Own", "demo.root.ROOT"),
        )
        final = tmp_path / "same-cpp-name.ac"
        final.write_text(text)
    _accept_hwir_backends(final, tmp_path)


@pytest.mark.parametrize("name", ["Std", "Gfsim"])
def test_global_module_names_are_mangled_away_from_cpp_runtime_names(
    tmp_path: Path,
    name: str,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "__init__.py"
    source.write_text(
        f"from pycircuit import module\n\n@module\ndef {name}():\n    state: bool = False\n",
        encoding="utf-8",
    )
    unit = cpp._compile_source(
        source,
        source_root=source_root,
        output_dir=tmp_path / "unit",
        package="",
    )
    final = tmp_path / "module.ac"
    linked = cpp._link([unit], final, top=name)
    assert linked.returncode == 0, linked.stderr

    _accept_hwir_backends(final, tmp_path)


def test_casefolded_raw_init_owners_keep_distinct_structural_groups(tmp_path: Path):
    source_root = tmp_path / "source"
    units = []
    for index, (path, contents) in enumerate(
        (
            ("foo/__init__.py", '"""lower package"""\n'),
            ("FOO/__INIT__.py", '"""upper package"""\n'),
            (
                "root.py",
                "from pycircuit import module\n@module\ndef Root():\n    state: bool = False\n",
            ),
        )
    ):
        source = source_root / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(contents)
        units.append(
            cpp._compile_source(
                source,
                source_root=source_root,
                output_dir=tmp_path / "units" / str(index),
                package="demo",
            )
        )
    output = tmp_path / "linked.ac"
    result = cpp._link(units, output, top="demo.root.Root")
    assert result.returncode == 0, result.stderr
    payload = _parts(output)
    groups = payload["structural_cpp_groups"]
    sources = {
        (group["source"]["package"], group["source"]["path"])
        for group in groups
    }
    assert ("demo", "foo/__init__.py") in sources
    assert ("demo", "FOO/__INIT__.py") in sources
    assert len({group["path"] for group in groups}) == len(groups)
    _accept_hwir_backends(output, tmp_path)
