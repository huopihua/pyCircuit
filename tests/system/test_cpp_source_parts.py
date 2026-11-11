"""Source-owned C++ groups survive final-program serialization and link.

The source units in this test are captured and linked through the current
private source bridges. The C++ source-parts bridge receives only the saved
final artifact in a fresh process, so passing this test requires the final
program to retain the source ownership needed by backend emission.
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
from pycircuit import struct, u8 as Word

@struct
class CounterResult:
    count: Word
"""

# The child owns one hidden register per instance and writes to a register
# borrowed from its parent. The two instances keep distinct copies of that
# state without duplicating either parent output.
COUNTER_CHILD = """\
from pycircuit import module, rule
from pycircuit import u8 as Word
from .types import CounterResult

@rule
def capture(count, incoming, outgoing) -> CounterResult:
    result = CounterResult(count=count)
    count = incoming
    return result

@module
def Counter(incoming: Word, outgoing: Word) -> CounterResult:
    count: Word = 0
    return capture(count, incoming, outgoing)
"""

COUNTER_SYSTEM = """\
from pycircuit import system, rule, log
from pycircuit import u3 as Phase, u8 as Word
from .counter import Counter

@rule
def transfer(left_output, right_output, left_count, right_count) -> None:
    left_output = left_count
    right_output = right_count

@system
def TestCounters():
    left_input: Word = 3
    right_input: Word = 10
    left_output: Word = 100
    right_output: Word = 200
    phase: Phase = 0
    left = Counter(left_input, left_output)
    right = Counter(right_input, right_output)
    transfer(left_output, right_output, left.count, right.count)

    @rule
    def fixture() -> None:
        nonlocal phase
        log("info", "left", left_output)
        log("info", "right", right_output)
        phase = (phase + 1) & 7

    fixture()
"""


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


def _source_unit_harness() -> str:
    return _tool("PYCIRCUIT_SOURCE_COMPILER", "pycircuit-source-unit")


def _design_harness() -> str:
    return _tool("PYCIRCUIT_LINKER", "pycircuit-link")


def _source_parts_harness() -> str:
    return _tool("PYCIRCUIT_EMITTER", "pycircuit-emit")


def _optimizer() -> str:
    return _tool("PYCIRCUIT_OPT", "pycircuit-opt")


def _emit_hwir(
    final: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
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


def _emit_source(
    final: Path, target: str, output: Path
) -> subprocess.CompletedProcess[str]:
    native_target = "structural-cpp" if target == "cpp" else target
    result = _emit_hwir(
        final, "--target", native_target, "--format", "source"
    )
    if result.returncode == 0:
        output.write_text(result.stdout, encoding="utf-8")
    return result


def _compile_source(
    source: Path,
    *,
    source_root: Path,
    output_dir: Path,
    headers: tuple[Path, ...] = (),
    package: str = "demo",
) -> SourceUnit:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = _capture_source_file(source, source_root=source_root)
    transport = output_dir / f"{source.stem}.transport.mlir"
    body = output_dir / f"{source.stem}.body.mlir"
    header = output_dir / f"{source.stem}.interface.mlir"
    transport.write_text(_emit_source_transport(capture), encoding="utf-8")
    command = [
        _source_unit_harness(),
        "--capture",
        str(transport),
        "--package",
        package,
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


def _link(
    units: list[SourceUnit],
    output: Path,
    *,
    top: str = "demo.test_counters.TestCounters",
) -> subprocess.CompletedProcess[str]:
    command = [_design_harness()]
    for unit in units:
        command.extend(("--body", str(unit.body), "--header", str(unit.header)))
    command.extend(("--top", top, "--target", "final", "--output", str(output)))
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _build_final(tmp_path: Path, *, reverse_units: bool = False) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True)
    (source_root / "types.py").write_text(TYPES, encoding="utf-8")
    (source_root / "counter.py").write_text(COUNTER_CHILD, encoding="utf-8")
    (source_root / "test_counters.py").write_text(COUNTER_SYSTEM, encoding="utf-8")

    types = _compile_source(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    counter = _compile_source(
        source_root / "counter.py",
        source_root=source_root,
        output_dir=tmp_path / "units/counter",
        headers=(types.header,),
    )
    system = _compile_source(
        source_root / "test_counters.py",
        source_root=source_root,
        output_dir=tmp_path / "units/test_counters",
        headers=(types.header, counter.header),
    )
    final = tmp_path / "test_counters.final.ac"
    units = [types, counter, system]
    linked = _link(list(reversed(units)) if reverse_units else units, final)
    assert linked.returncode == 0, linked.stderr
    assert final.is_file()
    return final


def _build_static_profile_final(tmp_path: Path) -> Path:
    source_root = tmp_path / "static-source"
    source_root.mkdir(parents=True)
    (source_root / "types.py").write_text(TYPES, encoding="utf-8")
    source = source_root / "static_root.py"
    source.write_text(
        "from pycircuit import module, u8 as Word\n\n"
        "@module\n"
        "def StaticRoot():\n"
        "    value: Word = 0\n",
        encoding="utf-8",
    )
    types = _compile_source(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "static-units/types",
    )
    unit = _compile_source(
        source,
        source_root=source_root,
        output_dir=tmp_path / "static-units/static_root",
        headers=(types.header,),
    )
    final = tmp_path / "static_root.final.ac"
    linked = _link([types, unit], final, top="demo.static_root.StaticRoot")
    assert linked.returncode == 0, linked.stderr
    return final


def _emit_parts(final: Path) -> subprocess.CompletedProcess[str]:
    return _emit_hwir(
        final, "--target", "structural-cpp", "--format", "json"
    )


def _emit_legacy(
    final: Path, target: str, output: Path
) -> subprocess.CompletedProcess[str]:
    return _emit_source(final, target, output)


def _write_parts(tmp_path: Path, payload: dict) -> tuple[Path, list[dict]]:
    output = tmp_path / "cpp-parts"
    output.mkdir()
    (output / "design.hpp").write_text(
        payload["structural_cpp_header"], encoding="utf-8"
    )
    groups = payload["structural_cpp_groups"]
    for group in groups:
        relative = Path(group["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        source_path = output / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(group["text"], encoding="utf-8")
    return output, groups


def _cxx() -> str:
    return _tool("CXX", "clang++")


def _include_args(output: Path) -> list[str]:
    return ["-I", str(output), "-I", str(GFSIM_INCLUDE)]


def _compile_groups(output: Path, groups: list[dict], tmp_path: Path) -> list[Path]:
    cxx = _cxx()
    include_args = _include_args(output)
    headers = [output / "design.hpp"]
    for header in headers:
        checked = subprocess.run(
            [
                cxx,
                "-std=c++20",
                *include_args,
                "-x",
                "c++",
                "-fsyntax-only",
                str(header),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert checked.returncode == 0, f"{header}:\n{checked.stderr}"

    objects: list[Path] = []
    for index, group in enumerate(groups):
        source = output / group["path"]
        object_file = tmp_path / f"source-group-{index}.o"
        compiled = subprocess.run(
            [
                cxx,
                "-std=c++20",
                *include_args,
                "-c",
                str(source),
                "-o",
                str(object_file),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert compiled.returncode == 0, f"{source}:\n{compiled.stderr}"
        objects.append(object_file)
    return objects


DRIVER = """\
#include "design.hpp"
#include <iostream>

int second_consumer();

static auto bit(bool value) {
  return gfsim::wire<gfsim::Bits<1>>::known(gfsim::Bits<1>{value ? 1u : 0u});
}

static int run_once(pyc_dut &dut, const char *tag) {
  auto &model = dut.system();
  dut.drive({bit(false), bit(false)});
  model.Reset();
  for (int i = 0; i < 3; ++i) {
    dut.drive({bit(false), bit(false)});
    if (model.Step() != gfsim::SimStepResult::Running) return 5;
    dut.drive({bit(true), bit(false)});
    const auto step = model.Step();
    for (const auto &event : dut.observations().Events())
      std::cout << tag << " EV " << event.value.bits << "\\n";
    if (step == gfsim::SimStepResult::Failed ||
        step == gfsim::SimStepResult::InvalidState) return 5;
  }
  return 0;
}

int main() {
  pyc_dut dut;
  dut.drive({bit(false), bit(false)});
  dut.system().Build();
  if (int first = run_once(dut, "A")) return first;
  if (int second = run_once(dut, "B")) return second;
  if (second_consumer() != 0) return 9;
  std::cout << "DONE\\n";
  return 0;
}
"""


def _run_cpp(tmp_path: Path, output: Path, objects: list[Path]) -> str:
    driver = tmp_path / "source_parts_driver.cpp"
    driver.write_text(DRIVER, encoding="utf-8")
    consumer = tmp_path / "source_parts_second_consumer.cpp"
    consumer.write_text(
        """\
#include "design.hpp"

int second_consumer() {
  pyc_dut dut;
  dut.system().Build();
  dut.system().Reset();
  return dut.system().cycle() == 0 ? 0 : 1;
}
""",
        encoding="utf-8",
    )
    consumer_object = tmp_path / "source-parts-second-consumer.o"
    consumer_compiled = subprocess.run(
        [
            _cxx(),
            "-std=c++20",
            *_include_args(output),
            "-c",
            str(consumer),
            "-o",
            str(consumer_object),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert consumer_compiled.returncode == 0, consumer_compiled.stderr
    binary = tmp_path / "source_parts_cpp"
    compiled = subprocess.run(
        [
            _cxx(),
            "-std=c++20",
            *_include_args(output),
            str(driver),
            *(str(obj) for obj in objects),
            str(consumer_object),
            "-o",
            str(binary),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
    assert (
        ran.returncode == 0
    ), f"cpp run rc={ran.returncode}\n{ran.stdout}\n{ran.stderr}"
    return ran.stdout


def _run_legacy_cpp(tmp_path: Path, model: Path) -> str:
    driver = tmp_path / "legacy_source_parts_driver.cpp"
    driver.write_text(
        """\
#include "gfsim/SimSystem.h"
#include "legacy_model.hpp"
#include <iostream>

static int run_once(FinalSystem &model, const char *tag) {
  model.Reset();
  for (int i = 0; i < 5; ++i) {
    const auto step = model.Step();
    for (const auto &event : model.Observations().Events())
      std::cout << tag << " EV " << event.value.bits << "\\n";
    if (step == gfsim::SimStepResult::Failed ||
        step == gfsim::SimStepResult::InvalidState) return 5;
    if (step == gfsim::SimStepResult::Quiescent) break;
  }
  return 0;
}

int main() {
  FinalSystem model;
  model.Build();
  if (int first = run_once(model, "A")) return first;
  if (int second = run_once(model, "B")) return second;
  return 0;
}
""",
        encoding="utf-8",
    )
    shutil.copy2(model, tmp_path / "legacy_model.hpp")
    binary = tmp_path / "legacy_source_parts_cpp"
    compiled = subprocess.run(
        [
            _cxx(),
            "-std=c++20",
            "-I",
            str(GFSIM_INCLUDE),
            str(driver),
            "-o",
            str(binary),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
    assert (
        ran.returncode == 0
    ), f"legacy cpp rc={ran.returncode}\n{ran.stdout}\n{ran.stderr}"
    return ran.stdout


def _run_legacy_verilog(tmp_path: Path, model: Path) -> str:
    testbench = tmp_path / "source_parts_tb.sv"
    testbench.write_text(
        """\
module tb;
  logic clk = 1'b0;
  logic reset = 1'b1;
  FinalModelSim dut(.clk(clk), .reset(reset));
  task automatic tick;
    begin #1 clk = 1'b1; #1 clk = 1'b0; #1; end
  endtask
  initial begin
    tick();
    reset = 1'b0;
    repeat (5) tick();
    reset = 1'b1;
    tick();
    reset = 1'b0;
    repeat (5) tick();
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    binary = tmp_path / "source_parts.vvp"
    compiled = subprocess.run(
        [
            _tool("IVERILOG", "iverilog"),
            "-g2012",
            "-s",
            "tb",
            "-o",
            str(binary),
            str(model),
            str(testbench),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run(
        [_tool("VVP", "vvp"), str(binary)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    return ran.stdout


def _rtl_trace(stdout: str) -> list[int]:
    values = []
    for line in stdout.splitlines():
        fields = line.split()
        if len(fields) == 8 and fields[0] == "AC_OBS":
            values.append(int(fields[5]))
    return values


def _trace(stdout: str, tag: str) -> list[int]:
    return [
        int(match.group(1))
        for match in re.finditer(rf"^{tag} EV (\d+)$", stdout, re.MULTILINE)
    ]


def test_source_parts_preserve_per_source_cpp_tus_and_behavior(tmp_path: Path) -> None:
    final = _build_final(tmp_path)
    reversed_final = _build_final(tmp_path / "reverse", reverse_units=True)

    # The final file is the complete backend input. A new process receives no
    # source directory or source-unit paths from which it could recover owners.
    for source in (tmp_path / "source").glob("*.py"):
        source.unlink()
    for source in (tmp_path / "reverse/source").glob("*.py"):
        source.unlink()
    shutil.rmtree(tmp_path / "units")
    shutil.rmtree(tmp_path / "reverse/units")

    expected = [100, 200, 0, 0, 3, 10]

    emitted = _emit_parts(final)
    assert emitted.returncode == 0, emitted.stderr
    payload = json.loads(emitted.stdout)
    assert payload["structural_cpp_header"]
    assert payload["structural_cpp_groups"]
    repeated_emission = _emit_parts(final)
    reversed_emission = _emit_parts(reversed_final)
    assert repeated_emission.returncode == 0, repeated_emission.stderr
    assert reversed_emission.returncode == 0, reversed_emission.stderr
    assert repeated_emission.stdout == emitted.stdout, "emission must be deterministic"
    assert (
        reversed_emission.stdout == emitted.stdout
    ), "reversing final unit-link order must preserve source-owned output"

    output, groups = _write_parts(tmp_path, payload)
    owners = [(group["source"]["package"], group["source"]["path"]) for group in groups]
    assert sorted(owners) == [
        ("demo", "counter.py"),
        ("demo", "test_counters.py"),
        ("demo", "types.py"),
    ]
    executable_groups = groups
    assert len({group["path"] for group in groups}) == len(groups)
    assert all(group["path"].endswith(".cpp") for group in groups)

    child_groups = [
        group for group in groups if group["source"]["path"] == "counter.py"
    ]
    assert len(child_groups) == 1, "repeated child instances must share one source TU"
    child_group = child_groups[0]
    child_text = child_group["text"]
    assert '#include "design.hpp"' in child_text
    system_group = next(
        group for group in groups if group["source"]["path"] == "test_counters.py"
    )
    assert "class pyc_structural_system final" in payload["structural_cpp_header"]
    assert "class pyc_dut final" in payload["structural_cpp_header"]
    # Final serialization carries one child-definition register; its two
    # runtime instances each own a distinct copy of that state. Five system
    # registers plus the child definition yield six serialized register ops.
    assert final.read_text(encoding="utf-8").count(
        "callee = @pycircuit.__builtins__.dffe"
    ) == 6

    objects = _compile_groups(output, groups, tmp_path)
    stdout = _run_cpp(tmp_path, output, objects)
    assert _trace(stdout, "A") == expected, stdout
    assert (
        _trace(stdout, "B") == expected
    ), "reset must restore both hidden child states"
    assert "DONE" in stdout

    # If the child implementation is accidentally emitted inline in a header,
    # linking without its owner object would still succeed. Require the missing
    # owner object to produce a method-symbol link error.
    driver = tmp_path / "source_parts_driver.cpp"
    child_index = next(
        i
        for i, group in enumerate(executable_groups)
        if group["source"]["path"] == "counter.py"
    )
    omitted = subprocess.run(
        [
            _cxx(),
            "-std=c++20",
            *_include_args(output),
            str(driver),
            *(str(obj) for i, obj in enumerate(objects) if i != child_index),
            str(tmp_path / "source-parts-second-consumer.o"),
            "-o",
            str(tmp_path / "missing-child-owner"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert omitted.returncode != 0, "child method definitions must live in the child TU"
    assert re.search(
        r"undefined reference|undefined symbol|unresolved external",
        omitted.stderr,
        re.IGNORECASE,
    ), omitted.stderr


@pytest.mark.parametrize(
    "mutation", ["bad-stage", "unsafe-source-owner", "nonempty-static-arguments"]
)
def test_source_parts_refuses_invalid_final_without_partial_stdout(
    tmp_path: Path,
    mutation: str,
) -> None:
    final = (
        _build_static_profile_final(tmp_path)
        if mutation == "nonempty-static-arguments"
        else _build_final(tmp_path)
    )
    text = final.read_text(encoding="utf-8")
    if mutation == "bad-stage":
        needle = 'ac.stage = "final"'
        assert needle in text
        text = text.replace(needle, 'ac.stage = "source"', 1)
    elif mutation == "unsafe-source-owner":
        needle = 'path = "counter.py"'
        assert needle in text
        text = text.replace(needle, 'path = "../counter.py"')
    else:
        needle = (
            "callee = @demo.static_root.StaticRoot, parameters = [], "
            "type_arguments = []"
        )
        # SpecKey arguments use the StaticValue dictionary form. This is a
        # structurally valid nonempty integer argument, not a malformed
        # StaticArgument wrapper or an undeclared family parameter.
        static_integer = '{kind = "integer", value = #ac.math_int<1>}'
        assert needle in text
        text = text.replace(
            needle,
            "callee = @demo.static_root.StaticRoot, "
            f"parameters = [{static_integer}], type_arguments = []",
        )
    invalid = tmp_path / f"{mutation}.ac"
    invalid.write_text(text, encoding="utf-8")

    completed = _emit_parts(invalid)
    assert completed.returncode == 1
    assert completed.stdout == "", "refusal must not publish partial JSON"
    if mutation == "nonempty-static-arguments":
        # The value has valid shape, but its nonempty root SpecKey has no
        # matching serialized definition. Current final rehydration refuses it
        # before source-owned C++ layout is entered.
        assert "root integer arguments require StaticExpr" in completed.stderr


def test_source_parts_disambiguates_legalized_source_path_collision(tmp_path: Path) -> None:
    source_root = tmp_path / "collision-source"
    source_root.mkdir()
    (source_root / "types.py").write_text(TYPES, encoding="utf-8")
    for filename, definition in (
        ("fooBar.py", "FirstUnit"),
        ("foo_bar.py", "SecondUnit"),
    ):
        (source_root / filename).write_text(
            f"""\
from pycircuit import module, rule, u8 as Word

@module
def {definition}(incoming: Word, outgoing: Word):
    @rule
    def forward() -> None:
        nonlocal outgoing
        outgoing = incoming
    forward()
""",
            encoding="utf-8",
        )
    (source_root / "top.py").write_text(
        """\
from pycircuit import system, u8 as Word
from .fooBar import FirstUnit
from .foo_bar import SecondUnit

@system
def CollisionTop():
    first_in: Word = 3
    second_in: Word = 10
    first_out: Word = 0
    second_out: Word = 0
    first = FirstUnit(first_in, first_out)
    second = SecondUnit(second_in, second_out)
""",
        encoding="utf-8",
    )
    types = _compile_source(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "collision-units/types",
    )
    first = _compile_source(
        source_root / "fooBar.py",
        source_root=source_root,
        output_dir=tmp_path / "collision-units/fooBar",
        headers=(types.header,),
    )
    second = _compile_source(
        source_root / "foo_bar.py",
        source_root=source_root,
        output_dir=tmp_path / "collision-units/foo_bar",
        headers=(types.header,),
    )
    top = _compile_source(
        source_root / "top.py",
        source_root=source_root,
        output_dir=tmp_path / "collision-units/top",
        headers=(types.header, first.header, second.header),
    )
    final = tmp_path / "collision.final.ac"
    linked = subprocess.run(
        [
            _design_harness(),
            "--body",
            str(types.body),
            "--header",
            str(types.header),
            "--body",
            str(first.body),
            "--header",
            str(first.header),
            "--body",
            str(second.body),
            "--header",
            str(second.header),
            "--body",
            str(top.body),
            "--header",
            str(top.header),
            "--top",
            "demo.top.CollisionTop",
            "--target",
            "final",
            "--output",
            str(final),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert linked.returncode == 0, linked.stderr

    completed = _emit_parts(final)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    groups = payload["structural_cpp_groups"]
    assert {group["source"]["path"] for group in groups} >= {
        "fooBar.py",
        "foo_bar.py",
    }
    assert len({group["path"] for group in groups}) == len(groups)
    output, materialized = _write_parts(tmp_path, payload)
    _compile_groups(output, materialized, tmp_path)
