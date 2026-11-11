"""Actual source assertion paths, aliasing and reset on generated C++/RTL."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from test_generic_assignment_roundtrip import (
    GFSIM_INCLUDE,
    _compile,
    _emit,
    _link,
    _tool,
)

pytestmark = pytest.mark.system

SOURCE = """\
from pycircuit import system, rule, log
@system
def AssertControls():
    ok: bool = True
    condition: bool = {condition}
    gate: bool = False
    sink: bool = False
    @rule
    def step() -> None:
        nonlocal gate, sink
        if ok:
            assert ok, "aliased condition and path"
        if gate and ok:
            assert ok and condition, "actual active condition"
        log("info", "observed", sink)
        sink = ok
        gate = ok
    step()
"""

CPP_DRIVER = """\
#include "gfsim/SimSystem.h"
#include "assert_model.hpp"
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
    model.Reset();
    for (int epoch = 0; epoch != 3; ++epoch) {
      dut.drive({bit(false), bit(false)});
      if (model.Step() == gfsim::SimStepResult::InvalidState) return 4;
      dut.drive({bit(true), bit(false)});
      const auto step = model.Step();
      if (step == gfsim::SimStepResult::InvalidState) return 4;
      std::cout << "ERR " << run << " " << epoch << " "
                << (step == gfsim::SimStepResult::Failed) << "\\n";
      for (const auto &event : dut.observations().Events())
        std::cout << "EV " << run << " " << event.epoch << " " << event.value.bits << "\\n";
    }
  }
}
"""

RTL_TESTBENCH = """\
module tb;
  logic clk = 0;
  logic reset = 1;
  logic [2:0] phase = 0;
  logic commit_ok = 0;
  wire check_complete, sampling_error, local_error;
  pyc_root dut(
    .pyc_7079635f636c6b(clk),
    .pyc_7079635f727374(reset),
    .pyc_phase(phase),
    .pyc_root_commit_ok(commit_ok),
    .pyc_check_complete(check_complete),
    .pyc_sampling_error(sampling_error),
    .pyc_local_error(local_error));
  task automatic reset_model;
    begin
      reset = 1; commit_ok = 1; phase = 4; #1;
      $display("RESET_ERR %0d", dut.dut.pyc_check_error_0 |
                                      dut.dut.pyc_check_error_1);
      phase = 2; #1; phase = 0; commit_ok = 0; reset = 0; #1;
    end
  endtask
  task automatic step(input logic level, input integer run, input integer epoch,
                      input logic report);
    begin
      clk = level; commit_ok = 0; phase = 0; #1; phase = 1; #1;
      if (report) $display("ERR %0d %0d %0d", run, epoch, local_error);
      if (check_complete === 1'b1 && sampling_error === 1'b0 && local_error === 1'b0) begin
        commit_ok = 1; phase = 2; #1;
        if (report && dut.dut.pyc_observation_pending_0 &&
            dut.dut.pyc_observation_path_0 && dut.dut.pyc_observation_valid_0)
          $display("EV %0d %0d", run, dut.dut.pyc_observation_value_0);
      end else begin
        phase = 3; #1;
      end
      phase = 0; commit_ok = 0; #1;
    end
  endtask
  initial begin
    for (integer run = 0; run < 2; run = run + 1) begin
      reset_model();
      for (integer epoch = 0; epoch < 3; epoch = epoch + 1) begin
        step(0, run, epoch, 0);
        step(1, run, epoch, 1);
      end
    end
    $finish;
  end
endmodule
"""


def _run(tmp_path: Path, design: Path, target: str) -> str:
    model = tmp_path / ("assert_model.hpp" if target == "cpp" else "assert_model.sv")
    emitted = _emit(design, target, model)
    assert emitted.returncode == 0, emitted.stderr
    if target == "cpp":
        driver = tmp_path / "assert_driver.cpp"
        driver.write_text(CPP_DRIVER, encoding="utf-8")
        binary = tmp_path / "assert_cpp"
        command = [
            _tool("CXX", "clang++"),
            "-std=c++20",
            "-I",
            str(GFSIM_INCLUDE),
            str(driver),
            *(str(path) for path in sorted(tmp_path.glob("source_unit_*.cpp"))),
            "-o",
            str(binary),
        ]
        execution = [str(binary)]
    else:
        driver = tmp_path / "assert_tb.sv"
        driver.write_text(RTL_TESTBENCH, encoding="utf-8")
        binary = tmp_path / "assert.vvp"
        command = [
            _tool("IVERILOG", "iverilog"),
            "-g2012",
            "-s",
            "tb",
            "-o",
            str(binary),
            str(model),
            str(driver),
            str(Path(GFSIM_INCLUDE) / "verilog/dff.v"),
            str(Path(GFSIM_INCLUDE) / "verilog/dffe.v"),
        ]
        execution = [_tool("VVP", "vvp"), str(binary)]
    compiled = subprocess.run(command, text=True, capture_output=True, check=False)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(execution, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("condition", [True, False], ids=["active-pass", "active-fail"])
@pytest.mark.parametrize("target", ["cpp", "verilog"])
def test_source_assertion_actual_paths_and_reset(
    tmp_path: Path, condition: bool, target: str
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "controls.py"
    source.write_text(SOURCE.format(condition=condition), encoding="utf-8")
    unit = _compile(source, source_root=source_root, output_dir=tmp_path / "unit")
    design = tmp_path / "controls.ac"
    linked = _link([unit], design, "demo.controls.AssertControls")
    assert linked.returncode == 0, linked.stderr
    stdout = _run(tmp_path, design, target)
    # Inactive false assertion succeeds initially. Once gate commits true,
    # the actual condition decides failure, and failure suppresses observations.
    errors = [
        int(match[2]) for match in re.findall(r"^ERR (\d+) (\d+) ([01])$", stdout, re.M)
    ]
    assert errors == ([0, 0, 0] if condition else [0, 1, 1]) * 2, stdout
    expected = [0, 1, 1] if condition else [0]
    if target == "cpp":
        for run in (0, 1):
            events = [
                (int(epoch), int(value))
                for epoch, value in re.findall(rf"^EV {run} (\d+) (\d+)$", stdout, re.M)
            ]
            if condition:
                assert [value for _, value in events] == expected, stdout
                assert len({epoch for epoch, _ in events}) == 3, stdout
            else:
                # Failed Step retains the preceding committed observation view.
                # Its epoch/value must remain unchanged rather than replaying a
                # new event or committing the proposed true sink observation.
                assert len(events) == 3 and events[0][1] == 0, stdout
                assert events == [events[0]] * 3, stdout
    else:
        assert re.findall(r"^RESET_ERR ([01])$", stdout, re.M) == ["0", "0"], stdout
        values = [int(value) for value in re.findall(r"^EV \d+ (\d+)$", stdout, re.M)]
        assert values == expected * 2, stdout
