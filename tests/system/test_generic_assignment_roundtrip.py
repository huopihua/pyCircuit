"""Generic assignment roundtrip: plain register copy and constant assignment.

These cases exercise the existing assignment contract end to end:

    per-source Python capture -> source body/header -> link -> final
    materialization -> saved .ac -> reparse in a new process -> C++/Verilog
    emit -> real compile and run

The oracle is the observed register value per phase (``[254, 7, 7]`` for the
constant assignment and ``[254, 3, 3]`` for the register copy), read from the
model's own observations on both backends, plus reset/rerun equality. It is never
derived from the generated code.
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

# A provider Rule that writes a system-owned register through the ordinary
# assignment contract. The owner remains in the system source unit.
COPY_CHILD = """\
@rule
def holder_tick(other, state) -> None:
    state = other
"""

CONSTANT_CHILD = """\
@rule
def holder_tick(state) -> None:
    state = 7
"""

# The testbench owns the registers, instantiates the DUT and checks the value it
# reads at each phase. It supplies the oracle, not the design.
BENCH = """\
from pycircuit import system, rule, log
from .types import Phase, Word

{provider}

@system
def TestHolder():
    other: Word = 3
    state: Word = 254
    phase: Phase = 0
    holder_tick({arguments})

    @rule
    def fixture() -> None:
        nonlocal phase
        log("info", "observed", state)
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


def _source_harness() -> str:
    return _tool("PYCIRCUIT_SOURCE_COMPILER", "pycircuit-source-unit")


def _design_harness() -> str:
    return _tool("PYCIRCUIT_LINKER", "pycircuit-link")


def _emitter() -> str:
    return _tool("PYCIRCUIT_EMITTER", "pycircuit-emit")


def _optimizer() -> str:
    return _tool("PYCIRCUIT_OPT", "pycircuit-opt")


def _emit_hwir(design: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="pycircuit-test-hwir-") as temporary:
        hwir = Path(temporary) / "design.hwir.mlir"
        lowered = subprocess.run(
            [_optimizer(), str(design), "--ac-final-to-hwir", "-o", str(hwir)],
            text=True,
            capture_output=True,
            check=False,
        )
        if lowered.returncode != 0:
            return lowered
        return subprocess.run(
            [_emitter(), str(hwir), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )


def _compile(
    source: Path, *, source_root: Path, output_dir: Path, headers: tuple[Path, ...] = ()
) -> SourceUnit:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = _capture_source_file(source, source_root=source_root)
    transport = output_dir / f"{source.stem}.transport.mlir"
    body = output_dir / f"{source.stem}.body.mlir"
    header = output_dir / f"{source.stem}.interface.mlir"
    transport.write_text(_emit_source_transport(capture), encoding="utf-8")
    command = [
        _source_harness(),
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


def _link(
    units: list[SourceUnit], output: Path, top: str
) -> subprocess.CompletedProcess[str]:
    command = [_design_harness()]
    for unit in units:
        command.extend(("--body", str(unit.body), "--header", str(unit.header)))
    command.extend(
        (
            "--top",
            top,
            "--target",
            "final",
            "--output",
            str(output),
        )
    )
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _emit(design: Path, target: str, output: Path) -> subprocess.CompletedProcess[str]:
    native_target = "structural-cpp" if target == "cpp" else target
    result = _emit_hwir(
        design,
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


def _build_and_link(tmp_path: Path, child_text: str, bench_text: str) -> Path:
    """Compile the child and the testbench per source and link the closure."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "types.py").write_text(TYPES, encoding="utf-8")
    (source_root / "holder.py").write_text(
        '"""Reserved provider source unit."""\n', encoding="utf-8"
    )
    (source_root / "test_holder.py").write_text(bench_text, encoding="utf-8")
    types = _compile(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    holder = _compile(
        source_root / "holder.py",
        source_root=source_root,
        output_dir=tmp_path / "units/holder",
        headers=(types.header,),
    )
    bench = _compile(
        source_root / "test_holder.py",
        source_root=source_root,
        output_dir=tmp_path / "units/test_holder",
        headers=(types.header, holder.header),
    )
    design = tmp_path / "test_holder.ac"
    linked = _link([types, holder, bench], design, "demo.test_holder.TestHolder")
    assert linked.returncode == 0, linked.stderr
    assert design.is_file()
    return design


CPP_DRIVER = """\
#include "gfsim/SimSystem.h"
#include "generic_model.hpp"
#include <iostream>

static int run_once(const char *tag) {
  const auto bit = [](bool value) {
    return gfsim::wire<gfsim::Bits<1>>::known(gfsim::Bits<1>{value ? 1u : 0u});
  };
  pyc_dut dut;
  auto &model = dut.system();
  dut.drive({bit(false), bit(false)});
  model.Build();
  model.Reset();
  for (int i = 0; i < 3; ++i) {
    dut.drive({bit(false), bit(false)});
    if (model.Step() != gfsim::SimStepResult::Running) return 5;
    dut.drive({bit(true), bit(false)});
    const gfsim::SimStepResult step = model.Step();
    for (const auto &event : dut.observations().Events())
      std::cout << tag << " EV " << event.epoch << " " << event.value.bits << "\\n";
    if (step == gfsim::SimStepResult::Failed ||
        step == gfsim::SimStepResult::InvalidState)
      return 5;
  }
  return 0;
}

int main() {
  if (int first = run_once("A")) return first;
  if (int second = run_once("B")) return second;
  std::cout << "DONE\\n";
  return 0;
}
"""

VERILOG_TESTBENCH = """\
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
    repeat (3) tick();
    reset = 1'b1;
    tick();
    reset = 1'b0;
    repeat (3) tick();
    $finish;
  end
endmodule
"""


def _run_cpp(tmp_path: Path, model: Path) -> str:
    cxx = _tool("CXX", "clang++")
    driver = tmp_path / "generic_driver.cpp"
    driver.write_text(CPP_DRIVER, encoding="utf-8")
    shutil.copy2(model, tmp_path / "generic_model.hpp")
    binary = tmp_path / "generic_cpp"
    compiled = subprocess.run(
        [
            cxx,
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
    assert (
        ran.returncode == 0
    ), f"cpp run rc={ran.returncode}\n{ran.stdout}\n{ran.stderr}"
    return ran.stdout


def _run_verilog(tmp_path: Path, model: Path) -> str:
    iverilog = _tool("IVERILOG", "iverilog")
    vvp = _tool("VVP", "vvp")
    binary = tmp_path / "generic.vvp"
    compiled = subprocess.run(
        [
            iverilog,
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
    outputs = []
    for _ in range(2):
        ran = subprocess.run(
            [vvp, str(binary), "+cycles=3"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert ran.returncode == 0, ran.stderr
        outputs.append(ran.stdout)
    return "".join(outputs)


def _cpp_trace(stdout: str, tag: str) -> list[int]:
    return [
        int(m.group(1))
        for m in re.finditer(rf"^{tag} EV \d+ (\d+)$", stdout, re.MULTILINE)
    ]


def _rtl_trace(stdout: str) -> list[int]:
    values: list[int] = []
    for line in stdout.splitlines():
        if not line.startswith('{"kind":"log"'):
            continue
        record = json.loads(line)
        values.append(int(record["values"][0]["value"]))
    # The managed runner evaluates both low and high clock levels. Keep the
    # high-edge observation from each pair to match one C++ cycle.
    return values[1::2]


# case -> (child source, DUT instantiation arguments, expected per-phase trace)
GENERIC_CASES = {
    "constant": (CONSTANT_CHILD, "state", [254, 7, 7]),
    "register-copy": (COPY_CHILD, "other, state", [254, 3, 3]),
}


@pytest.mark.parametrize("case", sorted(GENERIC_CASES))
def test_generic_assignment_roundtrip_both_backends(tmp_path: Path, case: str) -> None:
    child, arguments, expected = GENERIC_CASES[case]
    bench = BENCH.format(
        provider=child,
        arguments=arguments,
        reset_value=expected[0],
        committed_value=expected[1],
    )
    design = _build_and_link(tmp_path, child, bench)

    # Parent/child alias: the DUT writes the register the testbench owns, so the
    # linked design declares exactly the three registers the testbench holds
    # (other, state, phase) and the child contributes none. A relay or double
    # register would raise this count.
    assert (
        design.read_text().count("callee = @pycircuit.__builtins__.dffe") == 3
    ), "alias must not add state"

    cpp = tmp_path / "generic.cpp"
    verilog = tmp_path / "generic.sv"
    assert _emit(design, "cpp", cpp).returncode == 0
    assert _emit(design, "verilog", verilog).returncode == 0

    cpp_out = _run_cpp(tmp_path, cpp)
    cpp_first, cpp_second = _cpp_trace(cpp_out, "A"), _cpp_trace(cpp_out, "B")
    assert cpp_first == expected, cpp_out
    assert cpp_second == expected, "reset/rerun must reproduce the same trace"
    assert "DONE" in cpp_out

    # The unconditional observation keeps this fixture on the structural route;
    # branch-local observations intentionally select typed transactions instead.
    per_run = expected
    rtl_values = _rtl_trace(_run_verilog(tmp_path, verilog))
    assert rtl_values[: len(per_run)] == per_run, rtl_values
    assert rtl_values[len(per_run) : 2 * len(per_run)] == per_run, rtl_values


GENERIC_NEGATIVES = [
    (
        "unsupported-contract",
        lambda text: text.replace(
            "hwir.contract = 2 : i64", "hwir.contract = 99 : i64", 1
        ),
        "unsupported hwir.contract version 99",
    ),
    (
        "missing-entry",
        lambda text: "".join(
            line
            for line in text.splitlines(keepends=True)
            if not line.lstrip().startswith("hwir.system ")
        ),
        "requires exactly one hwir.system",
    ),
    (
        "unresolved-entry",
        lambda text: text.replace(
            "callee = @demo.test_holder.TestHolder",
            "callee = @demo.test_holder.Missing",
            1,
        ),
        "does not resolve to an HWIR module",
    ),
    (
        "malformed-entry",
        lambda text: text.replace("hwir.system {callee", "hwir.system {bad_callee", 1),
        "entry requires callee, parameters and type_arguments",
    ),
]


@pytest.mark.parametrize(
    "label,apply,diagnostic",
    GENERIC_NEGATIVES,
    ids=[case[0] for case in GENERIC_NEGATIVES],
)
def test_generic_use_carriers_are_verified(
    tmp_path: Path, label: str, apply, diagnostic: str
) -> None:
    """Independent HWIR rejects corrupt package/entry contracts before emit."""
    child, arguments, expected = GENERIC_CASES["register-copy"]
    bench = BENCH.format(
        provider=child,
        arguments=arguments,
        reset_value=expected[0],
        committed_value=expected[1],
    )
    design = _build_and_link(tmp_path, child, bench)
    lowered = subprocess.run(
        [_optimizer(), str(design), "--ac-final-to-hwir"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert lowered.returncode == 0, lowered.stderr
    text = lowered.stdout
    mutated = apply(text)
    assert mutated != text, f"{label}: mutation did not apply"

    candidate = tmp_path / f"{label}.hwir.mlir"
    candidate.write_text(mutated, encoding="utf-8")
    output = tmp_path / f"{label}.sv"
    result = subprocess.run(
        [_emitter(), str(candidate), "--target", "verilog", "--format", "source"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0, f"{label} unexpectedly emitted"
    assert diagnostic in result.stderr, result.stderr
    assert not output.exists()


SYSTEM_GENERIC = """\
from typing import Annotated
from pycircuit import rule, system

Word = Annotated[int, range(256)]

@system
def GenericUses():
    source: Word = 3
    direct_out: Word = 0
    literal_out: Word = 0

    @rule
    def direct() -> None:
        nonlocal direct_out
        direct_out = source
        return

    @rule
    def literal() -> None:
        nonlocal literal_out
        literal_out = 7
        return

    direct()
    literal()
"""


def test_system_root_generic_rules_link_and_emit(tmp_path: Path) -> None:
    """The system-root generic shape (a system whose own rules copy a register or
    assign a literal) links and emits on both backends. This makes the packet's
    third pre-fix baseline row reproducible from the repository instead of only
    from the localisation probe that found it."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "generic.py").write_text(SYSTEM_GENERIC, encoding="utf-8")
    unit = _compile(
        source_root / "generic.py",
        source_root=source_root,
        output_dir=tmp_path / "units/generic",
    )
    design = tmp_path / "generic.ac"

    linked = _link([unit], design, "demo.generic.GenericUses")
    assert linked.returncode == 0, linked.stderr
    assert design.is_file()

    for target, extension in (("cpp", "cpp"), ("verilog", "sv")):
        output = tmp_path / f"generic.{extension}"
        emitted = _emit(design, target, output)
        assert emitted.returncode == 0, f"{target}: {emitted.stderr}"
        assert output.is_file() and output.stat().st_size > 0, target
