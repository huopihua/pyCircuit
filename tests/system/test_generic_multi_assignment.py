"""Generic rules keep source-order current/next semantics across assignments.

This regression compiles each Python file independently, links the saved final
program, then asks a fresh process to emit both backends. The child rule writes
its owned ``count`` next from ``incoming`` and then writes the parent-owned
``outgoing`` next from ``count``. The source contract requires that second read to observe the
cycle-start current value, so the observable trace is reset, 0, incoming.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from pycircuit._source_capture import _capture_source_file
from pycircuit._source_transport import _emit_source_transport

pytestmark = pytest.mark.system

ROOT = Path(__file__).resolve().parents[2]
GFSIM_INCLUDE = ROOT / "include"

TYPES = """\
from typing import Annotated

Word = Annotated[int, range(256)]
Phase = Annotated[int, range(8)]
"""

CHILD = """\
@rule
def tick_left(count, incoming, outgoing) -> None:
    count = incoming
    outgoing = count

@rule
def tick_right(count, incoming, outgoing) -> None:
    count = incoming
    outgoing = count
"""

SYSTEM = """\
from pycircuit import system, rule, log
from .types import Phase, Word

{provider}

@system
def TestCounters():
    left_input: Word = 3
    right_input: Word = 10
    left_output: Word = 100
    right_output: Word = 200
    left_count: Word = 0
    right_count: Word = 0
    phase: Phase = 0
    tick_left(left_count, left_input, left_output)
    tick_right(right_count, right_input, right_output)

    @rule
    def fixture() -> None:
        nonlocal phase
        log("info", "left", left_output)
        log("info", "right", right_output)
        phase = (phase + 1) & 7

    fixture()
"""

LOCAL_CANDIDATE_CHILD = CHILD.replace(
    "    count = incoming\n    outgoing = count",
    "    candidate = incoming\n    count = candidate\n    outgoing = candidate",
)
LOCAL_CANDIDATE_SYSTEM = SYSTEM
ENABLED_CHILD = CHILD.replace(
    "(count, incoming, outgoing) -> None:",
    "(count, incoming, outgoing, enabled) -> None:",
).replace(
    "    outgoing = count",
    "    if enabled:\n        outgoing = count",
)
ENABLED_SYSTEM = (
    SYSTEM.replace(
        "    phase: Phase = 0\n    tick_left(left_count, left_input, left_output)\n    tick_right(right_count, right_input, right_output)",
        "    phase: Phase = 0\n    left_enable: bool = True\n    right_enable: bool = False\n"
        "    tick_left(left_count, left_input, left_output, left_enable)\n"
        "    tick_right(right_count, right_input, right_output, right_enable)",
    )
)


@dataclass(frozen=True)
class SourceUnit:
    source: Path
    body: Path
    header: Path


def _tool(env_name: str, name: str) -> str:
    configured = os.environ.get(env_name)
    if configured:
        return configured
    found = shutil.which(name)
    if not found:
        raise AssertionError(f"set {env_name} or put {name} on PATH")
    return found


def _source_parts_harness() -> str:
    return _tool("PYCIRCUIT_EMITTER", "pycircuit-emit")


def _optimizer() -> str:
    return _tool("PYCIRCUIT_OPT", "pycircuit-opt")


def _emit_hwir(final: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="pycircuit-test-hwir-") as temporary:
        hwir = Path(temporary) / "design.hwir.mlir"
        lowered = subprocess.run(
            [_optimizer(), str(final), "--ac-final-to-hwir", "-o", str(hwir)],
            text=True,
            capture_output=True,
            check=False,
        )
        if lowered.returncode != 0:
            return lowered
        return subprocess.run(
            [_source_parts_harness(), str(hwir), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )


def _compile(
    source: Path, source_root: Path, output: Path, headers: tuple[Path, ...] = ()
) -> SourceUnit:
    output.mkdir(parents=True, exist_ok=True)
    capture = _capture_source_file(source, source_root=source_root)
    transport = output / f"{source.stem}.transport.mlir"
    body = output / f"{source.stem}.body.mlir"
    header = output / f"{source.stem}.interface.mlir"
    transport.write_text(_emit_source_transport(capture), encoding="utf-8")
    command = [
        _tool("PYCIRCUIT_SOURCE_COMPILER", "pycircuit-source-unit"),
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
    assert body.is_file() and header.is_file()
    return SourceUnit(source, body, header)


def _link(units: list[SourceUnit], final: Path) -> subprocess.CompletedProcess[str]:
    command = [_tool("PYCIRCUIT_LINKER", "pycircuit-link")]
    for unit in units:
        command.extend(("--body", str(unit.body), "--header", str(unit.header)))
    command.extend(
        (
            "--top",
            "demo.test_counters.TestCounters",
            "--target",
            "final",
            "--output",
            str(final),
        )
    )
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _emit(final: Path, target: str, output: Path) -> subprocess.CompletedProcess[str]:
    native_target = "structural-cpp" if target == "cpp" else target
    # This subprocess reparses the persisted final program; no in-memory linked
    # module is passed from the test process.
    result = _emit_hwir(
        final,
        "--target",
        native_target,
        "--format",
        "json" if target == "cpp" else "source",
    )
    if result.returncode == 0:
        if target == "cpp":
            payload = json.loads(result.stdout)
            header = payload["structural_cpp_header"]
            output.write_text(header, encoding="utf-8")
            (output.parent / "design.hpp").write_text(header, encoding="utf-8")
            for group in payload["structural_cpp_groups"]:
                path = output.parent / group["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(group["text"], encoding="utf-8")
        else:
            output.write_text(result.stdout, encoding="utf-8")
    return result


def _build_final(
    tmp_path: Path, child_source: str = CHILD, system_source: str = SYSTEM
) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "types.py").write_text(TYPES, encoding="utf-8")
    (source_root / "counter.py").write_text(
        '"""Reserved provider source unit."""\n', encoding="utf-8"
    )
    (source_root / "test_counters.py").write_text(
        system_source.format(provider=child_source), encoding="utf-8"
    )
    types = _compile(source_root / "types.py", source_root, tmp_path / "units/types")
    counter = _compile(
        source_root / "counter.py",
        source_root,
        tmp_path / "units/counter",
        (types.header,),
    )
    system = _compile(
        source_root / "test_counters.py",
        source_root,
        tmp_path / "units/test_counters",
        (types.header, counter.header),
    )
    final = tmp_path / "test_counters.final.ac"
    linked = _link([types, counter, system], final)
    assert linked.returncode == 0, linked.stderr
    assert final.is_file()
    return final


def _run_cpp(tmp_path: Path, model: Path) -> list[int]:
    driver = tmp_path / "driver.cpp"
    driver.write_text(
        """\
#include \"gfsim/SimSystem.h\"
#include \"model.hpp\"
#include <iostream>
int main() {
  const auto bit = [](bool value) {
    return gfsim::wire<gfsim::Bits<1>>::known(gfsim::Bits<1>{value ? 1u : 0u});
  };
  pyc_dut dut;
  auto &model = dut.system();
  dut.drive({bit(false), bit(false)});
  model.Build();
  for (int run = 0; run != 2; ++run) {
    dut.drive({bit(false), bit(false)});
    model.Reset();
    for (int i = 0; i != 3; ++i) {
      dut.drive({bit(false), bit(false)});
      if (model.Step() != gfsim::SimStepResult::Running) return 5;
      dut.drive({bit(true), bit(false)});
      auto step = model.Step();
      for (const auto &event : dut.observations().Events())
        std::cout << event.value.bits << \"\\n\";
      if (step == gfsim::SimStepResult::Failed ||
          step == gfsim::SimStepResult::InvalidState) return 5;
    }
  }
}
""",
        encoding="utf-8",
    )
    shutil.copy2(model, tmp_path / "model.hpp")
    binary = tmp_path / "model"
    compiled = subprocess.run(
        [
            _tool("CXX", "clang++"),
            "-std=c++20",
            "-I",
            str(tmp_path),
            "-I",
            str(GFSIM_INCLUDE),
            str(driver),
            *(str(path) for path in sorted(tmp_path.glob("source_unit_*.cpp"))),
            "-o",
            str(binary),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    return [int(line) for line in ran.stdout.splitlines()]


def _run_verilog(tmp_path: Path, model: Path) -> list[int]:
    binary = tmp_path / "rtl.vvp"
    compiled = subprocess.run(
        [
            _tool("IVERILOG", "iverilog"),
            "-g2012",
            "-s",
            "pycircuit_sim",
            "-o",
            str(binary),
            str(model),
            str(ROOT / "include/verilog/dff.v"),
            str(ROOT / "include/verilog/dffe.v"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    result: list[int] = []
    for _ in range(2):
        ran = subprocess.run(
            [_tool("VVP", "vvp"), str(binary), "+cycles=3"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert ran.returncode == 0, ran.stderr
        values = [
            int(value["value"])
            for line in ran.stdout.splitlines()
            if line.startswith('{"kind":"log"')
            for value in json.loads(line)["values"]
        ]
        # Two observations per edge and two edges per logical cycle.
        result.extend(
            value
            for edge in range(1, 6, 2)
            for value in values[edge * 2 : edge * 2 + 2]
        )
    return result


def _run_grouped_cpp(tmp_path: Path, final: Path) -> str:
    emitted = _emit_hwir(final, "--target", "structural-cpp", "--format", "json")
    assert emitted.returncode == 0, emitted.stderr
    payload = json.loads(emitted.stdout)
    assert "structural_cpp_header" in payload
    assert "structural_cpp_groups" in payload
    output = tmp_path / "grouped"
    output.mkdir()
    (output / "design.hpp").write_text(
        payload["structural_cpp_header"], encoding="utf-8"
    )
    groups = payload["structural_cpp_groups"]
    owners = [(group["source"]["package"], group["source"]["path"]) for group in groups]
    assert sorted(owners) == [
        ("demo", "counter.py"),
        ("demo", "test_counters.py"),
        ("demo", "types.py"),
    ]
    for group in groups:
        relative = Path(group["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(group["text"], encoding="utf-8")
    includes = ["-I", str(output), "-I", str(GFSIM_INCLUDE)]
    cxx = _tool("CXX", "clang++")
    objects = []
    for index, group in enumerate(groups):
        source = output / group["path"]
        obj = tmp_path / f"group-{index}.o"
        compiled = subprocess.run(
            [cxx, "-std=c++20", *includes, "-c", str(source), "-o", str(obj)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert compiled.returncode == 0, compiled.stderr
        objects.append(obj)
    driver = tmp_path / "grouped.cpp"
    driver.write_text(
        """\
#include \"design.hpp\"
#include <iostream>
static auto bit(bool value) {
  return gfsim::wire<gfsim::Bits<1>>::known(gfsim::Bits<1>{value ? 1u : 0u});
}
static int run(pyc_dut &dut, const char *tag) {
  auto &model = dut.system();
  dut.drive({bit(false), bit(false)});
  model.Reset();
  for (int i = 0; i != 3; ++i) {
    dut.drive({bit(false), bit(false)});
    if (model.Step() != gfsim::SimStepResult::Running) return 5;
    dut.drive({bit(true), bit(false)});
    auto step = model.Step();
    for (const auto &event : dut.observations().Events())
      std::cout << tag << \" EV \" << event.value.bits << \"\\n\";
    if (step == gfsim::SimStepResult::Failed ||
        step == gfsim::SimStepResult::InvalidState) return 5;
  }
  return 0;
}
int main() {
  pyc_dut dut; dut.drive({bit(false), bit(false)}); dut.system().Build();
  if (int rc = run(dut, \"A\")) return rc;
  if (int rc = run(dut, \"B\")) return rc;
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "grouped-model"
    compiled = subprocess.run(
        [
            cxx,
            "-std=c++20",
            *includes,
            str(driver),
            *(str(obj) for obj in objects),
            "-o",
            str(binary),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
    assert ran.returncode == 0, ran.stderr
    return ran.stdout


def _cpp_trace(stdout: str, tag: str) -> list[int]:
    return [
        int(match.group(1))
        for match in re.finditer(rf"^{tag} EV (\d+)$", stdout, re.MULTILINE)
    ]


def _balanced_close(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    quoted = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError(f"unclosed MLIR attribute beginning at {start}")


def _array_elements(array_text: str) -> list[str]:
    elements = []
    start = 1
    depth = 0
    quoted = False
    escaped = False
    for index, char in enumerate(array_text[1:-1], 1):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1
        elif char == "," and depth == 0:
            elements.append(array_text[start:index].strip())
            start = index + 1
    tail = array_text[start:-1].strip()
    if tail:
        elements.append(tail)
    return elements


def _multi_use_array(text: str) -> tuple[int, int, list[str]]:
    for match in re.finditer(r"ac.required_uses\s*=\s*\[", text):
        start = text.find("[", match.start())
        end = _balanced_close(text, start, "[", "]")
        array = text[start : end + 1]
        elements = _array_elements(array)
        if len(elements) >= 2 and all('role = "next"' in item for item in elements[:2]):
            return start, end + 1, elements
    raise AssertionError("final has no rule with two generic next-use proofs")


def _nested_field(record: str, field: str) -> str:
    match = re.search(rf"\b{field}\s*=\s*\{{", record)
    assert match, f"missing {field} in {record}"
    start = record.find("{", match.start())
    end = _balanced_close(record, start, "{", "}")
    return record[start : end + 1]


def _corrupt_second_target(text: str, *, duplicate_id: bool) -> str:
    start, end, uses = _multi_use_array(text)
    first, second = uses[0], uses[1]
    if duplicate_id:
        replacement = _nested_field(first, "id")
        old = _nested_field(second, "id")
    else:
        replacement = _nested_field(first, "target")
        old = _nested_field(second, "target")
    assert old != replacement
    mutated_second = second.replace(old, replacement, 1)
    assert mutated_second != second
    updated = text[start:end].replace(second, mutated_second, 1)
    return text[:start] + updated + text[end:]


@pytest.mark.parametrize(
    ("child_source", "system_source", "expected_one_run", "serialized_regs"),
    [
        pytest.param(
            CHILD,
            SYSTEM,
            [100, 200, 3, 10, 3, 10],
            7,
            id="persistent-current-next",
        ),
        pytest.param(
            LOCAL_CANDIDATE_CHILD,
            LOCAL_CANDIDATE_SYSTEM,
            [100, 200, 3, 10, 3, 10],
            7,
            id="local-candidate-reuse",
        ),
        pytest.param(
            ENABLED_CHILD,
            ENABLED_SYSTEM,
            [100, 200, 3, 200, 3, 200],
            9,
            id="independent-yield-enables",
        ),
    ],
)
def test_multi_assignment_saved_final_cpp_and_rtl(
    tmp_path: Path,
    child_source: str,
    system_source: str,
    expected_one_run: list[int],
    serialized_regs: int,
) -> None:
    final = _build_final(tmp_path, child_source, system_source)
    # Five parent registers plus one owned count per child. Aliased output
    # connections must not create relay state; repeated child definitions must
    # retain separate count state and output targets.
    assert (
        final.read_text(encoding="utf-8").count(
            "callee = @pycircuit.__builtins__.dffe"
        )
        == serialized_regs
    )

    cpp = tmp_path / "model.generated.hpp"
    rtl = tmp_path / "model.sv"
    cpp_emitted = _emit(final, "cpp", cpp)
    assert cpp_emitted.returncode == 0, cpp_emitted.stderr
    rtl_emitted = _emit(final, "verilog", rtl)
    assert rtl_emitted.returncode == 0, rtl_emitted.stderr
    assert _run_cpp(tmp_path, cpp) == expected_one_run * 2
    grouped_out = _run_grouped_cpp(tmp_path, final)
    assert _cpp_trace(grouped_out, "A") == expected_one_run
    assert _cpp_trace(grouped_out, "B") == expected_one_run

    # Unconditional observations keep the structural child composition route.
    per_run = expected_one_run
    rtl_values = _run_verilog(tmp_path, rtl)
    assert rtl_values == per_run * 2, rtl_values


@pytest.mark.parametrize(
    ("label", "mutate", "expected_diagnostic"),
    [
        (
            "duplicate-entry",
            lambda text: "".join(
                line * 2 if line.lstrip().startswith("hwir.system ") else line
                for line in text.splitlines(keepends=True)
            ),
            "requires exactly one hwir.system",
        ),
        (
            "invalid-root-kind",
            lambda text: text.replace(
                'hwir.root_kind = "system"', 'hwir.root_kind = "module"', 1
            ),
            "structural C++ root_kind must be system",
        ),
    ],
)
def test_multi_assignment_hwir_contract_fails_closed(
    tmp_path: Path, label: str, mutate, expected_diagnostic: str
) -> None:
    """Both backends reject corrupt independent-HWIR package contracts."""
    final = _build_final(tmp_path)
    lowered = subprocess.run(
        [_optimizer(), str(final), "--ac-final-to-hwir"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert lowered.returncode == 0, lowered.stderr
    text = lowered.stdout
    mutated = mutate(text)
    assert mutated != text
    candidate = tmp_path / f"{label}.hwir.mlir"
    candidate.write_text(mutated, encoding="utf-8")

    for target in ("structural-cpp", "verilog"):
        result = subprocess.run(
            [
                _source_parts_harness(),
                str(candidate),
                "--target",
                target,
                "--format",
                "source",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 1, f"{label}/{target}: {result.returncode}"
        assert expected_diagnostic in result.stderr, result.stderr
        assert not result.stdout, "rejected HWIR must not publish partial output"
