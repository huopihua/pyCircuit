"""source preview private file bridge tests over the approved function-style value source."""

from __future__ import annotations

import os
import json
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
from pycircuit import struct, u8

@struct
class IncrementResult:
    outgoing: u8

@struct
class BlinkerResult:
    level: u8
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

TEST_INCREMENT = """\
from pycircuit import system, rule, log, report, u1, u3, u8
from .increment import Increment

@rule
def transfer_output(outgoing, child_outgoing) -> None:
    outgoing = child_outgoing

@system
def TestIncrement():
    enabled: u1 = 1
    incoming: u8 = 1
    outgoing: u8 = 0
    phase: u3 = 0
    dut = Increment(enabled, incoming)
    transfer_output(outgoing, dut.outgoing)

    @rule
    def fixture() -> None:
        nonlocal phase
        log("info", "test_complete", outgoing)
        report("completed", phase)
        phase = (phase + 1) & 7

    fixture()
"""


@dataclass(frozen=True)
class SourceUnit:
    source: Path
    body: Path
    header: Path


def _design_harness() -> Path:
    configured = os.environ.get("PYCIRCUIT_LINKER")
    if configured:
        candidate = Path(configured)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise AssertionError(
        "set PYCIRCUIT_LINKER to the private source-design file bridge; "
        "the public compiler and legacy backends are not substitutes"
    )


def _source_unit_harness() -> Path:
    candidates: list[str | None] = [os.environ.get("PYCIRCUIT_SOURCE_COMPILER")]
    toolchain = os.environ.get("PYC_TOOLCHAIN_ROOT")
    if toolchain:
        candidates.append(str(Path(toolchain) / "bin/pycircuit-source-unit"))
    candidates.append(shutil.which("pycircuit-source-unit"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise AssertionError(
        "set PYCIRCUIT_SOURCE_COMPILER or PYC_TOOLCHAIN_ROOT for source-unit tests"
    )


def _compile_source(
    source: Path,
    *,
    source_root: Path,
    output_dir: Path,
    headers: tuple[Path, ...] = (),
) -> SourceUnit:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = _capture_source_file(source, source_root=source_root)
    transport = output_dir / f"{source.stem}.transport.mlir"
    body = output_dir / f"{source.stem}.body.mlir"
    header = output_dir / f"{source.stem}.interface.mlir"
    transport.write_text(_emit_source_transport(capture), encoding="utf-8")
    command = [
        str(_source_unit_harness()),
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


def _compile_value_units(tmp_path: Path) -> tuple[Path, list[SourceUnit]]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "types.py").write_text(TYPES, encoding="utf-8")
    (source_root / "increment.py").write_text(INCREMENT, encoding="utf-8")
    (source_root / "test_increment.py").write_text(TEST_INCREMENT, encoding="utf-8")
    types = _compile_source(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    increment = _compile_source(
        source_root / "increment.py",
        source_root=source_root,
        output_dir=tmp_path / "units/increment",
        headers=(types.header,),
    )
    system = _compile_source(
        source_root / "test_increment.py",
        source_root=source_root,
        output_dir=tmp_path / "units/test_increment",
        headers=(types.header, increment.header),
    )
    return source_root, [types, increment, system]


def _link(
    units: list[SourceUnit],
    output: Path,
    *,
    top: str = "demo.test_increment.TestIncrement",
    pairs: list[tuple[Path, Path]] | None = None,
    role: str | None = None,
) -> subprocess.CompletedProcess[str]:
    selected = (
        pairs if pairs is not None else [(unit.body, unit.header) for unit in units]
    )
    command = [str(_design_harness())]
    for body, header in selected:
        command.extend(("--body", str(body), "--header", str(header)))
    command.extend(("--top", top, "--target", "final", "--output", str(output)))
    # Root role is part of the linked @module/@system contract.  Keep the
    # compatibility-only test parameter local; the native CLI no longer takes
    # a caller-supplied --role override.
    del role
    return subprocess.run(command, text=True, capture_output=True, check=False)


def _emit(
    program: Path, target: str, output: Path, role: str | None = None
) -> subprocess.CompletedProcess[str]:
    native_target = "structural-cpp" if target == "cpp" else target
    del role
    command = [str(_parts_harness()), str(program), "--target", native_target]
    if os.path.lexists(output):
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            f"error: cannot create output '{output}' without replacing an existing path\n",
        )
    with tempfile.TemporaryDirectory(prefix="source-design-bridge-") as temporary:
        hwir = Path(temporary) / "design.hwir.mlir"
        lowered = subprocess.run(
            [
                str(_optimizer_harness()),
                str(program),
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
                str(_parts_harness()),
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
            with output.open("x", encoding="utf-8") as stream:
                stream.write(emitted.stdout)
        return emitted


def _optimizer_harness() -> Path:
    configured = os.environ.get("PYCIRCUIT_OPT")
    if configured and Path(configured).is_file():
        return Path(configured).resolve()
    return _design_harness().with_name("pycircuit-opt")


def _parts_harness() -> Path:
    configured = os.environ.get("PYCIRCUIT_EMITTER")
    if configured and Path(configured).is_file():
        return Path(configured).resolve()
    return _design_harness().with_name("pycircuit-emit")


def _emit_payload(program: Path, target: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="source-design-payload-") as temporary:
        hwir = Path(temporary) / "design.hwir.mlir"
        lowered = subprocess.run(
            [str(_optimizer_harness()), str(program), "--ac-final-to-hwir", "-o", str(hwir)],
            text=True,
            capture_output=True,
            check=False,
        )
        if lowered.returncode:
            return lowered
        return subprocess.run(
            [str(_parts_harness()), str(hwir), "--target", target],
            text=True,
            capture_output=True,
            check=False,
        )


def _design_command(
    program: Path, arguments: list[str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_design_harness()), "--design", str(program), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def _required_tool(env_name: str, candidates: tuple[str, ...]) -> str:
    configured = os.environ.get(env_name)
    path = (
        configured
        if configured
        else next((found for name in candidates if (found := shutil.which(name))), None)
    )
    if not path:
        raise AssertionError(f"system backend run requires {env_name} or {candidates}")
    return path


def _run_emitted_value_models(
    tmp_path: Path, cpp_model: Path, verilog_model: Path
) -> tuple[str, str]:
    cxx = _required_tool("CXX", ("clang++", "c++"))
    iverilog = _required_tool("IVERILOG", ("iverilog",))
    vvp = _required_tool("VVP", ("vvp",))
    cpp_driver = tmp_path / "value_driver.cpp"
    cpp_binary = tmp_path / "value_cpp"
    cpp_driver.write_text(
        """\
#include "gfsim/SimSystem.h"
#include "value_model.hpp"
#include <iostream>

int main() {
  FinalSystem model;
  model.Build();
  model.Reset();
  for (int i = 0; i < 5; ++i) {
    if (model.Step() != gfsim::SimStepResult::Running) return 10 + i;
  }
  if (model.cycle() != 5) return 20;
  const auto gauges = model.Observations().Gauges();
  if (gauges.size() != 1 || gauges.front().value.bits != 1) return 21;
  std::cout << "CYCLE 5 COMPLETED 1\\n";
  return 0;
}
""",
        encoding="utf-8",
    )
    shutil.copy2(cpp_model, tmp_path / "value_model.hpp")
    cpp_compile = subprocess.run(
        [
            cxx,
            "-std=c++20",
            "-I",
            str(GFSIM_INCLUDE),
            str(cpp_driver),
            "-o",
            str(cpp_binary),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert cpp_compile.returncode == 0, cpp_compile.stderr
    cpp_run = subprocess.run(
        [str(cpp_binary)], text=True, capture_output=True, check=False
    )
    assert cpp_run.returncode == 0, cpp_run.stderr

    verilog_driver = tmp_path / "value_tb.sv"
    verilog_binary = tmp_path / "value.vvp"
    verilog_driver.write_text(
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
    $display("Q %0d", dut.dut.root_.q2);
    $finish;
  end
endmodule
""",
        encoding="utf-8",
    )
    rtl_compile = subprocess.run(
        [
            iverilog,
            "-g2012",
            "-s",
            "tb",
            "-o",
            str(verilog_binary),
            str(verilog_model),
            str(verilog_driver),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert rtl_compile.returncode == 0, rtl_compile.stderr
    rtl_run = subprocess.run(
        [vvp, str(verilog_binary)], text=True, capture_output=True, check=False
    )
    assert rtl_run.returncode == 0, rtl_run.stderr
    return cpp_run.stdout, rtl_run.stdout


def _linked_value_design(tmp_path: Path) -> tuple[list[SourceUnit], Path]:
    _design_harness()
    _, units = _compile_value_units(tmp_path)
    # The linked artifact keeps the Python source file name: test_increment.py -> test_increment.ac
    program = tmp_path / "test_increment.ac"
    completed = _link(units, program, role="testbench")
    assert completed.returncode == 0, completed.stderr
    assert program.is_file()
    return units, program


def test_value_source_design_file_is_one_input_to_both_emitters(
    tmp_path: Path,
) -> None:
    source_root, units = _compile_value_units(tmp_path)

    # Linking must use the captured body/header pairs, without reopening Python.
    assert 'path = "types.py"' in units[0].header.read_text()
    for source in source_root.glob("*.py"):
        source.unlink()
    # The linked artifact keeps the Python source file name: test_increment.py -> test_increment.ac
    program = tmp_path / "test_increment.ac"
    linked = _link(units, program, role="testbench")
    assert linked.returncode == 0, linked.stderr
    serialized = program.read_text(encoding="utf-8")
    assert 'source_owner = {package = "demo", path = "increment.py"}' in serialized
    assert 'source_owner = {package = "demo", path = "test_increment.py"}' in serialized
    assert 'package = "demo"' in serialized

    # Each new process reparses the same final file and selects only its backend.
    cpp = tmp_path / "cpp.generated"
    verilog = tmp_path / "verilog.generated"
    cpp_result = _emit(program, "cpp", cpp, role="testbench")
    assert cpp_result.returncode == 0, cpp_result.stderr
    verilog_result = _emit(program, "verilog", verilog, role="testbench")
    assert verilog_result.returncode == 0, verilog_result.stderr
    cpp_text = cpp.read_text(encoding="utf-8")
    verilog_text = verilog.read_text(encoding="utf-8")
    assert cpp_text.strip()
    assert verilog_text.strip()
    assert "public gfsim::SimModule" in cpp_text
    assert "struct pyc_observation_sink" in cpp_text
    assert "module pyc_root (" in verilog_text
    assert "module pycircuit_sim;" in verilog_text
    for retired in ("FinalSystem", "FinalModel", "FinalModelSim"):
        assert retired not in cpp_text
        assert retired not in verilog_text


@pytest.mark.parametrize("invalid_pair", ["duplicate", "unequal"])
def test_link_rejects_duplicate_source_or_unequal_unit_pair_before_output(
    tmp_path: Path, invalid_pair: str
) -> None:
    units, _ = _linked_value_design(tmp_path)
    pairs = [(unit.body, unit.header) for unit in units]
    if invalid_pair == "duplicate":
        pairs.append((units[1].body, units[1].header))
    else:
        pairs[1] = (units[1].body, units[0].header)
    output = tmp_path / f"{invalid_pair}.design.ac"
    completed = _link(units, output, pairs=pairs, role="testbench")
    expected = (
        "duplicate source link SourceOwner"
        if invalid_pair == "duplicate"
        else "body/header SourceOwner mismatch"
    )
    assert completed.returncode != 0, completed.stdout
    assert expected in completed.stderr
    assert not output.exists()


def test_link_rejects_top_that_is_not_in_the_captured_source_closure(
    tmp_path: Path,
) -> None:
    _, units = _compile_value_units(tmp_path)
    output = tmp_path / "bad-top.design.ac"
    completed = _link(units, output, top="demo.missing.Missing")
    assert completed.returncode != 0, completed.stdout
    assert "link top is not a supplied module definition" in completed.stderr
    assert not output.exists()


def test_emit_rejects_illegal_final_before_creating_output(
    tmp_path: Path,
) -> None:
    _, program = _linked_value_design(tmp_path)
    text = program.read_text(encoding="utf-8")
    assert 'ac.stage = "final"' in text
    invalid = tmp_path / "invalid_design.ac"
    invalid.write_text(text.replace('ac.stage = "final"', 'ac.stage = "source"', 1))
    output = tmp_path / "must-not-exist.cpp"

    completed = _emit(invalid, "cpp", output, role="testbench")

    assert completed.returncode != 0, completed.stdout
    assert "ac-final-to-hwir requires certified final ACIR" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("mode", ["link", "cpp", "verilog"])
@pytest.mark.parametrize(
    "existing_kind", ["file", "directory", "symlink", "dangling-symlink"]
)
def test_emit_refuses_any_existing_output_without_mutating_it(
    tmp_path: Path, existing_kind: str, mode: str
) -> None:
    units, program = _linked_value_design(tmp_path)
    output = tmp_path / f"existing-{existing_kind}.sv"
    target = tmp_path / "symlink-target"
    original = b"existing output bytes\x00"
    if existing_kind == "file":
        output.write_bytes(original)
    elif existing_kind == "directory":
        output.mkdir()
        (output / "sentinel").write_bytes(original)
    elif existing_kind == "symlink":
        target.write_bytes(original)
        output.symlink_to(target)
    else:
        output.symlink_to(tmp_path / "absent-target")

    completed = (
        _link(units, output, role="testbench")
        if mode == "link"
        else _emit(program, mode, output, role="testbench")
    )

    assert completed.returncode != 0, completed.stdout
    assert "cannot create output" in completed.stderr
    assert output.is_symlink() == (existing_kind in {"symlink", "dangling-symlink"})
    if existing_kind == "file":
        assert output.read_bytes() == original
    elif existing_kind == "directory":
        assert (output / "sentinel").read_bytes() == original
    elif existing_kind == "symlink":
        assert target.read_bytes() == original
    else:
        assert not output.resolve(strict=False).exists()


@pytest.mark.parametrize(
    "arguments,diagnostic",
    [
        (["--design", "unused", "--top", "x", "--target", "cpp"], "exactly one"),
        (["--design", "unused"], "--target and --output are required"),
        (["--design", "unused", "--target", "final"], "emit mode requires"),
        (["--body", "unused", "--target", "final"], "matching nonempty"),
        (
            ["--design", "unused", "--target", "cpp", "--target", "verilog"],
            "duplicate --target",
        ),
        (["--unknown", "unused", "--target", "cpp"], "unknown option"),
    ],
)
def test_bridge_rejects_incomplete_or_mixed_modes(tmp_path, arguments, diagnostic):
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [str(_design_harness()), *arguments, "--output", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert diagnostic in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("target", ["cpp", "verilog"])
def test_final_hardware_certification_cannot_be_replaced(tmp_path, target):
    _, program = _linked_value_design(tmp_path)
    text = program.read_text()
    assert "hwir.hardware_only" in text
    invalid = tmp_path / "wrong-stage-type.ac"
    invalid.write_text(text.replace("hwir.hardware_only", "hwir.not_certified", 1))
    output = tmp_path / "must-not-exist"
    result = _emit(invalid, target, output, role="testbench")
    assert result.returncode != 0
    assert "requires certified final ACIR" in result.stderr
    assert not output.exists()


def test_emit_splits_hardware_rtl_from_runtime_glue(tmp_path: Path) -> None:
    """The current emitter declares hardware and simulation roles in one JSON bundle."""
    _, program = _linked_value_design(tmp_path)
    emitted = _emit_payload(program, "verilog")
    assert emitted.returncode == 0, emitted.stderr
    payload = json.loads(emitted.stdout)
    assert payload["ir_contract"] == "hwir-v2"
    assert "module pyc_root (" in payload["rtl_core"]
    assert payload["rtl_source_groups"]
    assert "module pycircuit_sim;" in payload["simulation_top"]
    assert "module pycircuit_sim;" not in payload["rtl_core"]
    assert "pyc_observation_json_" not in payload["rtl_core"]


def test_glue_output_requires_verilog_emit_mode(tmp_path: Path) -> None:
    _, program = _linked_value_design(tmp_path)
    output = tmp_path / "must-not-exist.cpp"
    glue = tmp_path / "must-not-exist.glue.sv"

    result = _design_command(
        program,
        [
            "--target",
            "structural-cpp",
            "--output",
            str(output),
            "--glue-output",
            str(glue),
        ],
    )

    assert result.returncode == 2
    assert "requires emit mode with --target verilog" in result.stderr
    assert not output.exists()
    assert not glue.exists()


def test_glue_output_is_rejected_in_link_mode(tmp_path: Path) -> None:
    units, _ = _linked_value_design(tmp_path)
    output = tmp_path / "must-not-exist.ac"
    glue = tmp_path / "must-not-exist.sv"
    command = [str(_design_harness())]
    for unit in units:
        command.extend(("--body", str(unit.body), "--header", str(unit.header)))
    command.extend(
        (
            "--top",
            "demo.test_increment.TestIncrement",
            "--target",
            "final",
            "--output",
            str(output),
            "--glue-output",
            str(glue),
        )
    )

    result = subprocess.run(command, text=True, capture_output=True, check=False)

    assert result.returncode == 2
    assert "requires emit mode with --target verilog" in result.stderr
    assert not output.exists()
    assert not glue.exists()


def test_glue_output_must_differ_from_output(tmp_path: Path) -> None:
    _, program = _linked_value_design(tmp_path)
    output = tmp_path / "same.sv"

    result = _design_command(
        program,
        ["--target", "verilog", "--output", str(output), "--glue-output", str(output)],
    )

    assert result.returncode == 2
    assert "must differ from --output" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    "existing_kind", ["file", "directory", "symlink", "dangling-symlink"]
)
def test_glue_output_refuses_existing_paths_without_publishing(
    tmp_path: Path, existing_kind: str
) -> None:
    """A rejected bundle must publish neither role file, including the rollback
    path where the rtl file is created before the glue destination is refused."""
    _, program = _linked_value_design(tmp_path)
    rtl = tmp_path / "rtl.sv"
    glue = tmp_path / "glue.sv"
    target = tmp_path / "symlink-target"
    original = b"existing glue bytes\x00"
    if existing_kind == "file":
        glue.write_bytes(original)
    elif existing_kind == "directory":
        glue.mkdir()
        (glue / "sentinel").write_bytes(original)
    elif existing_kind == "symlink":
        target.write_bytes(original)
        glue.symlink_to(target)
    else:
        glue.symlink_to(tmp_path / "absent-target")

    result = _design_command(
        program,
        [
            "--target",
            "verilog",
            "--output",
            str(rtl),
            "--glue-output",
            str(glue),
        ],
    )

    assert result.returncode != 0, result.stdout
    assert "source link requires at least one complete unit" in result.stderr
    assert not rtl.exists()
    if existing_kind == "file":
        assert glue.read_bytes() == original
    elif existing_kind == "directory":
        assert (glue / "sentinel").read_bytes() == original
    elif existing_kind == "symlink":
        assert glue.is_symlink()
        assert target.read_bytes() == original
    else:
        assert glue.is_symlink()
        assert not glue.resolve(strict=False).exists()


BLINKER = """\
from pycircuit import module, rule, u8
from .types import BlinkerResult

@rule
def tick(phase, level) -> BlinkerResult:
    phase = (phase + 1) & 255
    if phase == 0:
        level = 1
    return BlinkerResult(level=level)

@module
def Blinker() -> BlinkerResult:
    phase: u8 = 0
    level: u8 = 0
    return tick(phase, level)
"""

TEST_BLINKER = """\
from pycircuit import system, rule, log, report, u8
from .blinker import Blinker

@system
def TestBlinker():
    phase: u8 = 0
    dut = Blinker()

    @rule
    def fixture() -> None:
        nonlocal phase
        log("info", "dut_started", phase)
        report("completed", phase)
        phase = (phase + 1) & 255

    fixture()
"""


def _compile_design_and_testbench(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "types.py").write_text(TYPES, encoding="utf-8")
    (source_root / "blinker.py").write_text(BLINKER, encoding="utf-8")
    (source_root / "test_blinker.py").write_text(TEST_BLINKER, encoding="utf-8")
    types = _compile_source(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    blinker = _compile_source(
        source_root / "blinker.py",
        source_root=source_root,
        output_dir=tmp_path / "units/blinker",
        headers=(types.header,),
    )
    bench = _compile_source(
        source_root / "test_blinker.py",
        source_root=source_root,
        output_dir=tmp_path / "units/test_blinker",
        headers=(types.header, blinker.header),
    )
    return types, blinker, bench


def test_design_and_testbench_are_separate_artifacts(tmp_path: Path) -> None:
    """The design artifact must carry only the DUT, and the testbench must be a
    separate artifact that consumes it. Each is named after its own source."""
    types, blinker, bench = _compile_design_and_testbench(tmp_path)
    design = tmp_path / "blinker.ac"
    testbench = tmp_path / "test_blinker.ac"

    # The design links from the DUT closure alone: the testbench source need not
    # exist, be readable, or be part of the same link.
    design_link = _link([types, blinker], design, top="demo.blinker.Blinker")
    assert design_link.returncode == 0, design_link.stderr
    bench_link = _link(
        [types, blinker, bench],
        testbench,
        top="demo.test_blinker.TestBlinker",
        role="testbench",
    )
    assert bench_link.returncode == 0, bench_link.stderr

    design_text = design.read_text()
    bench_text = testbench.read_text()
    assert "demo.blinker.Blinker" in design_text
    assert "ac.observe" not in design_text
    assert "TestBlinker" not in design_text
    assert "dut_started" not in design_text
    assert "TestBlinker" in bench_text
    assert "ac.observe" in bench_text
    assert "demo.blinker.Blinker" in bench_text
    # Built-in negative control: the observation carrier and the testbench-only
    # log message are both detectable in the testbench artifact, so the absence
    # assertions above cannot pass merely because those strings never survive.
    assert "dut_started" in bench_text

    # The design's HWIR Verilog bundle carries neither testbench observations
    # nor the retired FinalModel simulation wrapper.
    emitted = _emit_payload(design, "verilog")
    assert emitted.returncode == 0, emitted.stderr
    payload = json.loads(emitted.stdout)
    rtl = payload["rtl_core"] + "".join(
        group["text"] for group in payload["rtl_source_groups"]
    )
    assert "module pyc_root (" in rtl
    assert "FinalModel" not in rtl
    assert "dut_started" not in rtl


def test_ported_module_root_exports_the_physical_dut_io_contract(
    tmp_path: Path,
) -> None:
    _, units = _compile_value_units(tmp_path)
    output = tmp_path / "increment.ac"

    completed = _link(units[:2], output, top="demo.increment.Increment")

    assert completed.returncode == 0, completed.stderr
    emitted = _emit_payload(output, "verilog")
    assert emitted.returncode == 0, emitted.stderr
    payload = json.loads(emitted.stdout)
    ports = payload["port_availability"]
    assert [port["name"] for port in ports["inputs"][:2]] == ["enabled", "incoming"]
    assert [port["name"] for port in ports["outputs"]] == ["result"]


def test_glue_output_path_alias_still_publishes_nothing(tmp_path: Path) -> None:
    """The equal-path guard compares argv strings, so an aliased spelling that
    survives through argv reaches the second exclusive create and therefore the
    rollback. This is the only case that exercises that rollback."""
    _, program = _linked_value_design(tmp_path)
    rtl = tmp_path / "aliased.sv"
    # Keep the raw "/./" spelling: passing this string through pathlib would
    # normalize it back to the same string and the argv guard would fire first.
    aliased = str(tmp_path) + "/./aliased.sv"
    sentinel = tmp_path / "sentinel.bin"
    sentinel.write_bytes(b"keep me\x00")

    assert str(rtl) != aliased, "the two argv strings must differ"
    assert not rtl.exists()

    result = _design_command(
        program,
        [
            "--target",
            "verilog",
            "--output",
            str(rtl),
            "--glue-output",
            aliased,
        ],
    )

    # An execution failure from the second exclusive create, not the argument
    # guard: rc=2 would mean the tool never tried to publish.
    assert result.returncode == 1, (result.returncode, result.stderr)
    assert "source link requires at least one complete unit" in result.stderr
    assert "must differ" not in result.stderr, result.stderr
    # the first role file was rolled back and unrelated files are untouched
    assert not rtl.exists()
    assert not Path(aliased).exists()
    assert sentinel.read_bytes() == b"keep me\x00"


def test_system_root_is_self_describing_without_role_override(tmp_path: Path) -> None:
    _, units = _compile_value_units(tmp_path)

    undeclared = _link(units, tmp_path / "undeclared.ac")
    assert undeclared.returncode == 0, undeclared.stderr
    baseline = (tmp_path / "undeclared.ac").read_bytes()
    assert b'ac.root_kind = "system"' in baseline

    as_design = _link(units, tmp_path / "as-design.ac", role="design")
    assert as_design.returncode == 0, as_design.stderr

    as_testbench = _link(units, tmp_path / "as-testbench.ac", role="testbench")
    assert as_testbench.returncode == 0, as_testbench.stderr
    assert (tmp_path / "as-design.ac").read_bytes() == baseline
    assert (tmp_path / "as-testbench.ac").read_bytes() == baseline


def test_module_root_is_self_describing_without_role_override(tmp_path: Path) -> None:
    types, blinker, _ = _compile_design_and_testbench(tmp_path)

    mislabelled = _link(
        [types, blinker],
        tmp_path / "mislabelled.ac",
        top="demo.blinker.Blinker",
        role="testbench",
    )
    assert mislabelled.returncode == 0, mislabelled.stderr
    module_final = (tmp_path / "mislabelled.ac").read_bytes()
    assert b"entry = {callee = @demo.blinker.Blinker" in module_final
    assert b'ac.root_kind = "system"' not in module_final

    design = _link(
        [types, blinker], tmp_path / "blinker.ac", top="demo.blinker.Blinker"
    )
    assert design.returncode == 0, design.stderr


def test_emit_uses_artifact_root_kind_and_ignores_retired_role_parameter(tmp_path: Path) -> None:
    _, program = _linked_value_design(tmp_path)

    unattributed = _emit(program, "verilog", tmp_path / "unattributed.sv")
    assert unattributed.returncode == 0, unattributed.stderr

    as_design = _emit(program, "verilog", tmp_path / "as-design.sv", role="design")
    assert as_design.returncode == 0, as_design.stderr

    as_testbench = _emit(
        program, "verilog", tmp_path / "as-testbench.sv", role="testbench"
    )
    assert as_testbench.returncode == 0, as_testbench.stderr
    baseline = (tmp_path / "unattributed.sv").read_bytes()
    assert (tmp_path / "as-design.sv").read_bytes() == baseline
    assert (tmp_path / "as-testbench.sv").read_bytes() == baseline


def test_private_regression_fixtures_do_not_leak_into_shipped_surfaces() -> None:
    """Product headers, libraries and entrypoints must not know fixture models."""
    fixtures = ("TestIncrement", "TestPipeline", "ComposedFixture")
    for path in (ROOT / "compiler").rglob("*"):
        if path.suffix not in {".cpp", ".h"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for fixture in fixtures:
            assert fixture not in text, f"{fixture} leaked into {path}"


MULTI_VALUE_OBSERVATION = """\
from pycircuit import log, system, rule, u1

@system
def Multi():
    first: u1 = 1
    second: u1 = 0

    @rule
    def observe() -> None:
        log("info", "event", first, second)

    observe()
"""


def test_multi_value_observation_group_is_supported_by_both_hwir_backends(
    tmp_path: Path,
) -> None:
    """One linked observation group must survive final-to-HWIR for both backends."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "multi.py").write_text(MULTI_VALUE_OBSERVATION, encoding="utf-8")
    unit = _compile_source(
        source_root / "multi.py",
        source_root=source_root,
        output_dir=tmp_path / "units/multi",
    )
    design = tmp_path / "multi.ac"

    linked = _link([unit], design, top="demo.multi.Multi", role="testbench")
    assert linked.returncode == 0, linked.stderr
    assert design.is_file()

    for target in ("cpp", "verilog"):
        fresh = tmp_path / f"fresh.{target}"
        emitted = _emit(design, target, fresh, role="testbench")
        assert emitted.returncode == 0, emitted.stderr
        assert fresh.is_file() and fresh.stat().st_size > 0

    # An existing output must also survive a rejected emit unchanged.
    preserved = tmp_path / "preserved.verilog"
    original = b"previous backend output\x00\xff"
    preserved.write_bytes(original)
    rejected = _emit(design, "verilog", preserved, role="testbench")
    assert rejected.returncode != 0
    assert "cannot create output" in rejected.stderr
    assert preserved.read_bytes() == original
