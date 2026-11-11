"""From-source source preview preview build, runner, and design/testbench boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.system

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/integration/source-preview"
MATERIALIZER = ROOT / "tools/materialize_source_preview.py"
_ORACLE: ModuleType | None = None
_RESET_REPLAY_SOURCE = r"""#include "gfsim/SimExecutor.h"
#include "runner_metadata.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#ifdef PYCIRCUIT_RTL_RUNNER
#include "rtl_system.hpp"
using PreviewSystem = PycircuitRtlSystem;
#else
#include "pycircuit_system.hpp"
using PreviewSystem = FinalSystem;
#endif

struct ReplaySnapshot {
  std::vector<::gfsim::CommittedEventSlot> events;
  std::vector<::gfsim::GaugeSnapshot> gauges;
  std::string statistics;
  std::string error;
  bool operator==(const ReplaySnapshot &) const = default;
};

static bool collect(::gfsim::SimExecutor &executor,
                    ::gfsim::ObservationSlots &observations,
                    ReplaySnapshot &snapshot) {
  auto gauges = observations.Gauges();
  snapshot.gauges.assign(gauges.begin(), gauges.end());
  PycircuitModelBufferV1 statistics{};
  if (executor.StatisticsJson(&statistics) != PYCIRCUIT_MODEL_STATUS_V1_OK ||
      !statistics.data || !statistics.size)
    return false;
  snapshot.statistics.assign(reinterpret_cast<const char *>(statistics.data),
                              static_cast<std::size_t>(statistics.size));
  PycircuitModelBufferV1 error{};
  if (executor.LastError(&error) != PYCIRCUIT_MODEL_STATUS_V1_OK)
    return false;
  if (error.data && error.size)
    snapshot.error.assign(reinterpret_cast<const char *>(error.data),
                          static_cast<std::size_t>(error.size));
  else
    snapshot.error.clear();
  return true;
}

static bool runThree(::gfsim::SimExecutor &executor,
                     ::gfsim::ObservationSlots &observations,
                     ReplaySnapshot &snapshot) {
  snapshot.events.clear();
  for (unsigned index = 0; index < 3; ++index) {
    PycircuitModelStepResultV1 result{sizeof(PycircuitModelStepResultV1)};
    if (executor.Step(&result) != PYCIRCUIT_MODEL_STATUS_V1_OK)
      return false;
    auto events = observations.Events();
    snapshot.events.insert(snapshot.events.end(), events.begin(), events.end());
    if (index < 2 && result.state != PYCIRCUIT_MODEL_STEP_V1_RUNNING)
      return false;
    if (index == 2 && result.state != PYCIRCUIT_MODEL_STEP_V1_TERMINATED)
      return false;
  }
  if (executor.cycles() != 3 ||
      executor.state() != ::gfsim::SimExecutorState::Completed)
    return false;
  return collect(executor, observations, snapshot);
}

static bool runFailureThenRefusedStep(
    ::gfsim::SimExecutor &executor,
    ::gfsim::ObservationSlots &observations,
    ReplaySnapshot &snapshot) {
  snapshot.events.clear();
  PycircuitModelStepResultV1 first{sizeof(PycircuitModelStepResultV1)};
  if (executor.Step(&first) != PYCIRCUIT_MODEL_STATUS_V1_OK ||
      first.state != PYCIRCUIT_MODEL_STEP_V1_RUNNING || first.epoch_time != 1)
    return false;
  auto committed = observations.Events();
  snapshot.events.insert(snapshot.events.end(), committed.begin(), committed.end());

  PycircuitModelStepResultV1 failed{sizeof(PycircuitModelStepResultV1)};
  if (executor.Step(&failed) != PYCIRCUIT_MODEL_STATUS_V1_RUNTIME_FAILURE ||
      failed.state != PYCIRCUIT_MODEL_STEP_V1_FAILED || failed.epoch_time != 1 ||
      executor.cycles() != 1 || !collect(executor, observations, snapshot) ||
      snapshot.error.empty())
    return false;

  const std::string latchedError = snapshot.error;
  PycircuitModelStepResultV1 refused{sizeof(PycircuitModelStepResultV1),
                                   PYCIRCUIT_MODEL_STEP_V1_RUNNING, 77, 88, 99};
  if (executor.Step(&refused) != PYCIRCUIT_MODEL_STATUS_V1_INVALID_STATE ||
      refused.state != PYCIRCUIT_MODEL_STEP_V1_RUNNING ||
      refused.epoch_time != 77 || refused.epoch_delta != 88 ||
      refused.reserved != 99 || executor.cycles() != 1)
    return false;
  PycircuitModelBufferV1 repeatedError{};
  if (executor.LastError(&repeatedError) != PYCIRCUIT_MODEL_STATUS_V1_OK ||
      !repeatedError.data || repeatedError.size == 0 ||
      std::string(reinterpret_cast<const char *>(repeatedError.data),
                  static_cast<std::size_t>(repeatedError.size)) != latchedError)
    return false;
  return true;
}

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  std::ifstream configFile(argv[1], std::ios::binary);
  if (!configFile) return 2;
  std::string config((std::istreambuf_iterator<char>(configFile)),
                     std::istreambuf_iterator<char>());
  auto metadata = PycircuitRunnerMetadata();
  std::vector<::gfsim::ReportGaugeDescriptor> reports;
  for (const auto &entry : metadata)
    if (entry.kind == "report")
      reports.push_back({entry.stableOrdinal, entry.instance, entry.reportName});

  PreviewSystem model;
  ::gfsim::SimExecutor executor(model, model.Observations(), reports);
  if (!executor.created()) return 3;
  auto configBytes = reinterpret_cast<const std::uint8_t *>(config.data());
  if (executor.ConfigureJson(configBytes, config.size()) !=
          PYCIRCUIT_MODEL_STATUS_V1_OK ||
      executor.Reset() != PYCIRCUIT_MODEL_STATUS_V1_OK)
    return 4;

  if (std::string(argv[2]) == "success") {
    ReplaySnapshot first;
    if (!runThree(executor, model.Observations(), first)) return 5;
    if (executor.Reset() != PYCIRCUIT_MODEL_STATUS_V1_OK || executor.cycles() != 0)
      return 6;
    ReplaySnapshot second;
    if (!runThree(executor, model.Observations(), second)) return 7;
    if (!(first == second)) return 8;
    std::cout << "same-object reset replay matched\n";
    return 0;
  }
  if (std::string(argv[2]) == "failure") {
    ReplaySnapshot first;
    if (!runFailureThenRefusedStep(executor, model.Observations(), first)) return 9;
    if (executor.Reset() != PYCIRCUIT_MODEL_STATUS_V1_OK || executor.cycles() != 0 ||
        !model.Observations().Events().empty())
      return 10;
    ReplaySnapshot second;
    if (!runFailureThenRefusedStep(executor, model.Observations(), second)) return 11;
    if (!(first == second)) return 12;
    std::cout << "same-object failure reset replay matched\n";
    return 0;
  }
  return 2;
}
"""


def _oracle() -> ModuleType:
    global _ORACLE
    if _ORACLE is None:
        path = FIXTURE / "oracle.py"
        spec = importlib.util.spec_from_file_location("source_preview_oracle", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _ORACLE = module
    return _ORACLE


def _materializer_module() -> ModuleType:
    path = MATERIALIZER
    spec = importlib.util.spec_from_file_location("source_preview_materializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_build() -> Path:
    return Path(
        os.environ.get("PYCIRCUIT_NATIVE_BUILD", ROOT / ".pycircuit_out/w10-pm/build")
    ).resolve()


def _require_preview_tools() -> Path:
    native = _native_build()
    required = (
        native / "CMakeCache.txt",
        native / "bin/pycircuit-source-unit",
        native / "bin/pycircuit-link",
        native / "bin/pycircuit-emit",
    )
    if any(not path.is_file() for path in required):
        pytest.fail(
            "source preview preview needs PYCIRCUIT_NATIVE_BUILD with the three current-checkout helpers"
        )
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        pytest.fail(
            "source preview preview requires CMake and Ninja on the current platform"
        )
    if shutil.which("verilator") is None:
        pytest.fail("source preview dual-backend preview requires Verilator")
    return native


def _materializer_env(native: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYCIRCUIT_LINKER"] = str(native / "bin/pycircuit-link")
    return environment


def _parts_helper(native: Path) -> Path:
    return native / "bin/pycircuit-emit"


def _checked(
    command: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False, timeout=900
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {command!r}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _configure_preview(
    build: Path,
    *,
    native: Path,
    design: str = "design_top",
    test_consumer: Path | None = None,
) -> Path:
    command = [
        "cmake",
        "-S",
        str(FIXTURE),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DPYCIRCUIT_REPOSITORY_ROOT=" + str(ROOT),
        "-DPYCIRCUIT_NATIVE_BUILD=" + str(native),
        "-DPYCIRCUIT_COMPILER_INSTALL=" + str(_runtime_prefix()),
        "-DPYCIRCUIT_DESIGN=" + design,
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    if test_consumer is not None:
        command.append("-DPYCIRCUIT_TEST_CONSUMER=" + str(test_consumer))
    _checked(command)

    return build


def _runtime_prefix() -> Path:
    configured = os.environ.get("PYCIRCUIT_COMPILER_INSTALL")
    return Path(configured or ROOT / ".pycircuit_out/toolchain/install").resolve()


def _build_preview(
    build: Path,
    *,
    native: Path,
    design: str = "design_top",
    test_consumer: Path | None = None,
) -> Path:
    _configure_preview(
        build,
        native=native,
        design=design,
        test_consumer=test_consumer,
    )
    _checked(
        ["cmake", "--build", str(build), "--target", "preview-build", "--parallel", "2"]
    )
    return build


def _runner(build: Path, backend: str) -> Path:
    binary = build / f"model-{backend}" / "bin/pycircuit_sim"
    if os.name == "nt":
        binary = binary.with_suffix(".exe")
    assert binary.is_file(), f"CMake did not produce the {backend} runner: {binary}"
    return binary


def _materialize_final(
    final: Path, output: Path, native: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            str(final),
            str(output),
            str(_parts_helper(native)),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
        env=_materializer_env(native),
    )


def _build_materialized_models(artifacts: Path, build_root: Path) -> dict[str, Path]:
    runners: dict[str, Path] = {}
    for backend in ("cpp", "verilog"):
        model_build = build_root / f"model-{backend}"
        _checked(
            [
                "cmake",
                "-S",
                str(artifacts / backend),
                "-B",
                str(model_build),
                "-G",
                "Ninja",
                "-DPYCIRCUIT_RUNTIME_ROOT=" + str(ROOT / "runtime"),
                "-DCMAKE_PREFIX_PATH=" + str(_runtime_prefix()),
                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            ]
        )
        _checked(
            [
                "cmake",
                "--build",
                str(model_build),
                "--target",
                "pycircuit_sim",
                "--parallel",
                "2",
            ]
        )
        runner = model_build / "bin/pycircuit_sim"
        if os.name == "nt":
            runner = runner.with_suffix(".exe")
        assert runner.is_file()
        runners[backend] = runner
    return runners


def _run_model(
    build: Path,
    backend: str,
    config: Path,
    *,
    events: Path | None,
    extra: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    configuration = json.loads(config.read_text(encoding="utf-8"))
    cycles = configuration.get("max_ticks")
    assert type(cycles) is int and cycles > 0
    if backend == "cpp":
        command = [str(_runner(build, backend)), "--cycles", str(cycles)]
        if events is not None:
            command.extend(("--events", str(events)))
    else:
        command = [str(_runner(build, backend)), f"+cycles={cycles}"]
    command.extend(extra)
    result = subprocess.run(
        command, text=True, capture_output=True, check=False, timeout=60
    )
    if backend == "verilog" and events is not None and events != Path("-"):
        json_lines = []
        for line in result.stdout.splitlines():
            if not line.startswith("{"):
                break
            json_lines.append(line)
        events.write_text("\n".join(json_lines) + "\n", encoding="utf-8")
    return result


def _run_materialized_runner(
    runner: Path, backend: str, config: Path, *, events: Path
) -> subprocess.CompletedProcess[str]:
    configuration = json.loads(config.read_text(encoding="utf-8"))
    cycles = configuration.get("max_ticks")
    assert type(cycles) is int and cycles > 0
    command = (
        [str(runner), "--cycles", str(cycles), "--events", str(events)]
        if backend == "cpp"
        else [str(runner), f"+cycles={cycles}"]
    )
    result = subprocess.run(
        command, text=True, capture_output=True, check=False, timeout=60
    )
    if backend == "verilog":
        json_lines = []
        for line in result.stdout.splitlines():
            if not line.startswith("{"):
                break
            json_lines.append(line)
        events.write_text("\n".join(json_lines) + "\n", encoding="utf-8")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def _mutated_narrow_probe(text: str) -> str:
    # The independent HWIR contract is packed hardware bits: source integer
    # range declarations and signed formatting are intentionally not carried
    # into the backend.  Mutate the one authoritative u8 initializer instead
    # and retain an exact oracle for integer-vs-bool JSON classification.
    old = "#ac.math_int<7>"
    assert text.count(old) == 1, f"final mutation anchor count changed: {old}"
    return text.replace(old, "#ac.math_int<255>")


def _insert_wide_unused_constant(text: str) -> str:
    marker = "module attributes "
    assert text.count(marker) == 1
    body = text.index("{\n", text.index(marker)) + 2
    declaration = (
        '  "ac.constant"() <{sym_name = "preview.types.Huge", '
        'type = {kind = "integer"}, '
        'value = {kind = "integer", value = '
        "#ac.math_int<18446744073709551616>}}> "
        '{ac.declaration_role = "definition", '
        "ac.origin = {site = {definition = @preview.types.Huge, ast_path = []}, "
        "expansion = []}, "
        'ac.source_owner = {package = "preview", path = "types.py"}} '
        ': () -> () loc("types.py":99:1)\n'
    )
    return text[:body] + declaration + text[body:]


def _duplicate_report_name(text: str) -> str:
    marker = 'name = "second"'
    assert text.count(marker) == 1
    return text.replace(marker, 'name = "first"', 1)


def _receipt(path: Path, target: str) -> dict[str, Any]:
    value = json.loads((path / "generated.json").read_text(encoding="utf-8"))
    expected_fields = {
        "kind",
        "target",
        "entry",
        "entry_source",
        "files",
        "source_groups",
        "ir_contract",
    }
    if target == "verilog":
        expected_fields.add("rtl_mode")
    assert set(value) == expected_fields
    assert value["kind"] == "pycircuit-generated"
    assert value["target"] == target
    assert value["ir_contract"] == "hwir-v2"
    if target == "verilog":
        assert value["rtl_mode"] == "managed"
    assert value["entry"] == {
        "definition": '@"preview.design_top.DesignTop"',
        "arguments": [],
    }
    assert value["entry_source"] == {"package": "preview", "path": "design_top.py"}
    files = value["files"]
    assert [item["path"] for item in files] == sorted(item["path"] for item in files)
    assert len({item["path"] for item in files}) == len(files)
    for item in files:
        assert set(item) == {"path", "role"}
        candidate = Path(item["path"])
        assert not candidate.is_absolute() and ".." not in candidate.parts
        assert item["path"] != "generated.json"
    listed = {item["path"] for item in files}
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "generated.json"
    }
    assert actual == listed
    from pycircuit._generated_bundle import _validate_generated_bundle
    from pycircuit._publication import _publication_owner_generated

    _validate_generated_bundle(
        path,
        _publication_owner_generated(
            package="preview",
            path="design_top.py",
            definition='@"preview.design_top.DesignTop"',
            target=target,
        ),
    )
    return value


def _group_map(receipt: dict[str, Any]) -> dict[str, list[str]]:
    groups = receipt["source_groups"]
    owners = [group["source"] for group in groups]
    assert len({(owner["package"], owner["path"]) for owner in owners}) == len(owners)
    declared = {file["path"] for file in receipt["files"]}
    assigned = [path for group in groups for path in group["files"]]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) <= declared
    return {group["source"]["path"]: group["files"] for group in groups}


def _group_text(bundle: Path, files: list[str]) -> str:
    return "\n".join(
        (bundle / relative).read_text(encoding="utf-8") for relative in files
    )


def _module_body(rtl: str, name_fragment: str) -> tuple[str, str]:
    for match in re.finditer(r"\bmodule\s+(\w+)\s*\([^;]*?\);", rtl, re.DOTALL):
        name = match.group(1)
        if name_fragment not in name:
            continue
        end = rtl.find("endmodule", match.end())
        assert end >= 0, f"unterminated RTL module {name}"
        return name, rtl[match.end() : end]
    raise AssertionError(f"RTL module containing {name_fragment!r} was not emitted")


@pytest.fixture(scope="module")
def built_design(tmp_path_factory: pytest.TempPathFactory) -> Path:
    native = _require_preview_tools()
    build = tmp_path_factory.mktemp("source_preview-design")
    return _build_preview(build, native=native)


def test_from_source_compile_link_emit_and_both_runners_share_one_final(
    built_design: Path,
) -> None:
    units = built_design / "units"
    for stem in ("types", "counter", "design_top"):
        unit = units / stem
        assert (unit / f"{stem}.ac").is_file()
        assert (unit / f"{stem}.interface.ac").is_file()
        assert (unit / f"{stem}.d").is_file()
        assert (unit / "unit.json").is_file()
    final = built_design / "linked/design_top.ac"
    assert final.is_file()
    final_hash = _sha256(final)

    cpp_bundle = built_design / "artifacts/cpp"
    rtl_bundle = built_design / "artifacts/verilog"
    cpp_receipt = _receipt(cpp_bundle, "structural-cpp")
    rtl_receipt = _receipt(rtl_bundle, "verilog")
    assert cpp_receipt["source_groups"]
    cpp_groups = _group_map(cpp_receipt)
    rtl_groups = _group_map(rtl_receipt)
    assert set(cpp_groups) == {"types.py", "counter.py", "design_top.py"}
    assert set(rtl_groups) == set(cpp_groups)
    for source in cpp_groups:
        assert sum(path.endswith(".cpp") for path in cpp_groups[source]) == 1
        expected_rtl = source != "types.py"
        assert any(path.endswith(".sv") for path in rtl_groups[source]) == expected_rtl

    files = {item["path"]: item["role"] for item in cpp_receipt["files"]}
    assert files["CMakeLists.txt"] == "cmake"
    assert files["design.hpp"] == "header"
    assert files["simulation_main.cpp"] == "runtime-glue"
    rtl_files = {item["path"]: item["role"] for item in rtl_receipt["files"]}
    assert rtl_files["design_top.sv"] == "rtl"
    assert rtl_files["simulation_top.sv"] == "runtime-glue"

    cpp_compile_database = json.loads(
        (built_design / "model-cpp/compile_commands.json").read_text()
    )
    cpp_compiled_files = [Path(entry["file"]).name for entry in cpp_compile_database]
    for source in ("source_unit_0.cpp", "source_unit_1.cpp", "source_unit_2.cpp"):
        assert cpp_compiled_files.count(source) == 1
    assert cpp_compiled_files.count("simulation_main.cpp") == 1
    assert "model_api.cpp" not in cpp_compiled_files
    assert "runner_main.cpp" not in cpp_compiled_files
    assert _sha256(final) == final_hash

    configs = FIXTURE / "configs"
    cpp_events = built_design / "cpp-events.jsonl"
    rtl_events = built_design / "rtl-events.jsonl"
    cpp = _run_model(
        built_design, "cpp", configs / "three-ticks.json", events=cpp_events
    )
    rtl = _run_model(
        built_design, "verilog", configs / "three-ticks.json", events=rtl_events
    )
    assert cpp.returncode == 0, cpp.stderr
    assert rtl.returncode == 0, rtl.stderr
    cpp_trace = cpp_events.read_bytes()
    rtl_trace = rtl_events.read_bytes()
    oracle = _oracle()
    oracle.assert_design_top_run(cpp_trace)
    oracle.assert_design_top_run(rtl_trace)
    oracle.assert_repeated_run_identical(cpp_trace, rtl_trace)
    oracle_command = [
        sys.executable,
        str(FIXTURE / "oracle.py"),
        str(cpp_events),
        str(rtl_events),
    ]
    oracle_run = subprocess.run(
        oracle_command,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert oracle_run.returncode == 0, oracle_run.stderr
    assert oracle_run.stdout == "source preview preview oracle: PASS\n"
    truncated = built_design / "truncated-events.jsonl"
    truncated.write_bytes(cpp_trace[:-1])
    rejected = subprocess.run(
        [sys.executable, str(FIXTURE / "oracle.py"), str(truncated), str(rtl_events)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert rejected.returncode != 0
    assert "source preview preview oracle: FAIL" in rejected.stderr
    assert "source preview preview oracle: PASS" not in rejected.stdout
    assert (
        _sha256(final) == final_hash
    ), "both runners must consume the same linked final"

    # The final ACIR contains five DFFE sites: one in Counter and four in the
    # root. Concrete instantiation gives the child site two physical copies.
    final_text = final.read_text(encoding="utf-8")
    assert final_text.count("callee = @pycircuit.__builtins__.dffe") == 5
    assert final_text.count("callee = @preview.counter.Counter") == 2
    cpp_group_text = {
        source: _group_text(cpp_bundle, paths) for source, paths in cpp_groups.items()
    }
    header_text = (cpp_bundle / "design.hpp").read_text(encoding="utf-8")
    assert header_text.count("gfsim::dffe<gfsim::Bits<8>>") == 5

    child_rtl = _group_text(rtl_bundle, rtl_groups["counter.py"])
    root_rtl = _group_text(rtl_bundle, rtl_groups["design_top.py"])
    child_match = re.search(r"\bmodule\s+(\w+)", child_rtl)
    root_match = re.search(r"\bmodule\s+(\w+)", root_rtl)
    assert child_match and root_match
    child_name, child_body = _module_body(child_rtl, child_match.group(1))
    root_name, root_body = _module_body(root_rtl, root_match.group(1))
    child_state_blocks = len(re.findall(r"\bdffe\s*#", child_body))
    root_state_blocks = len(re.findall(r"\bdffe\s*#", root_body))
    child_instances = re.findall(rf"\b{re.escape(child_name)}\s+(\w+)\s*\(", root_body)
    assert child_state_blocks == 1 and root_state_blocks == 4
    assert len(child_instances) == 2
    assert root_state_blocks + child_state_blocks * len(child_instances) == 6
    assert root_name


@pytest.mark.parametrize(
    ("design", "config", "oracle_name"),
    [
        ("hold_top", "three-ticks.json", "assert_hold_run"),
        ("zero_rule_top", "three-ticks.json", "assert_zero_rule_run"),
    ],
)
def test_runner_distinguishes_clocked_hold_from_zero_rule_quiescence(
    tmp_path: Path, design: str, config: str, oracle_name: str
) -> None:
    native = _require_preview_tools()
    build = _build_preview(tmp_path / design, native=native, design=design)
    traces: dict[str, bytes] = {}
    oracle = _oracle()
    for backend in ("cpp", "verilog"):
        events = build / f"{backend}-events.jsonl"
        result = _run_model(
            build,
            backend,
            FIXTURE / "configs" / config,
            events=events,
        )
        assert result.returncode == 0, result.stderr
        traces[backend] = events.read_bytes()
        if oracle_name == "assert_hold_run":
            oracle.assert_hold_run(traces[backend], limit=3)
        else:
            oracle.assert_zero_rule_run(traces[backend])
    oracle.assert_repeated_run_identical(traces["cpp"], traces["verilog"])


def test_second_tick_failure_discards_failed_epoch_observations(tmp_path: Path) -> None:
    native = _require_preview_tools()
    build = _build_preview(tmp_path / "failure", native=native, design="failure_top")
    outputs: dict[str, bytes] = {}
    for backend in ("cpp", "verilog"):
        events = build / f"{backend}-failure.jsonl"
        result = _run_model(
            build,
            backend,
            FIXTURE / "configs/four-ticks.json",
            events=events,
        )
        assert result.returncode != 0
        outputs[backend] = events.read_bytes()
    oracle = _oracle()
    oracle.assert_failure_run(outputs["cpp"])
    oracle.assert_failure_events_without_result(outputs["verilog"])
    cpp_events, _result = oracle.parse_jsonl(outputs["cpp"])
    assert (
        b"\n".join(
            json.dumps(row, separators=(",", ":")).encode("utf-8")
            for row in cpp_events
        )
        + b"\n"
        == outputs["verilog"]
    )


def test_runner_without_event_sink_is_quiet(built_design: Path) -> None:
    result = _run_model(
        built_design,
        "cpp",
        FIXTURE / "configs/three-ticks.json",
        events=None,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_explicit_stdout_sink_contains_one_complete_result(built_design: Path) -> None:
    result = _run_model(
        built_design,
        "cpp",
        FIXTURE / "configs/three-ticks.json",
        events=Path("-"),
    )
    assert result.returncode == 0, result.stderr
    _oracle().assert_design_top_run(result.stdout.encode("utf-8"))

    # Event options may precede the required cycle limit.
    events = built_design / "events-before-config.jsonl"
    reordered = subprocess.run(
        [
            str(_runner(built_design, "cpp")),
            "--events",
            str(events),
            "--cycles",
            "3",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert reordered.returncode == 0, reordered.stderr
    _oracle().assert_design_top_run(events.read_bytes())


def test_fresh_runner_replays_cpp_and_verilator_observations(
    built_design: Path,
) -> None:
    config = FIXTURE / "configs/three-ticks.json"
    for backend in ("cpp", "verilog"):
        traces = []
        for index in range(2):
            events = built_design / f"replay-{backend}-{index}.jsonl"
            result = _run_model(
                built_design, backend, config, events=events
            )
            assert result.returncode == 0, result.stderr
            traces.append(events.read_bytes())
        assert traces[0] == traces[1]
        _oracle().assert_design_top_run(traces[0])


def test_fresh_runner_failure_replays_identically_in_both_backends(
    tmp_path: Path,
) -> None:
    native = _require_preview_tools()
    build = tmp_path / "failure-reset"
    _build_preview(build, native=native, design="failure_top")
    config = FIXTURE / "configs/four-ticks.json"
    for backend in ("cpp", "verilog"):
        traces = []
        diagnostics = []
        for index in range(2):
            events = build / f"failure-replay-{backend}-{index}.jsonl"
            result = _run_model(build, backend, config, events=events)
            assert result.returncode != 0
            traces.append(events.read_bytes())
            diagnostics.append(result.stderr)
        assert traces[0] == traces[1]
        assert diagnostics[0] == diagnostics[1]
        if backend == "cpp":
            _oracle().assert_failure_run(traces[0])
        else:
            _oracle().assert_failure_events_without_result(traces[0])


def test_bad_configuration_and_event_sinks_fail_before_touching_outputs(
    built_design: Path, tmp_path: Path
) -> None:
    runner = _runner(built_design, "cpp")
    invalid_configs = {
        "unbounded.json": (
            '{"deadlock_window":null,"max_domain_cycles":{},"max_ticks":null,'
            '"schema":"pycircuit-model-config","version":"1"}\n'
        ),
        "unknown.json": (
            '{"deadlock_window":null,"max_domain_cycles":{},"max_ticks":3,'
            '"schema":"pycircuit-model-config","version":"1","extra":true}\n'
        ),
        "duplicate.json": (
            '{"deadlock_window":null,"max_domain_cycles":{},"max_ticks":3,'
            '"max_ticks":4,"schema":"pycircuit-model-config","version":"1"}\n'
        ),
    }
    for name, text in invalid_configs.items():
        config = tmp_path / name
        config.write_text(text, encoding="utf-8")
        sink = tmp_path / f"{name}.jsonl"
        result = subprocess.run(
            [str(runner), "--config", str(config), "--events", str(sink)],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert result.returncode != 0
        assert not sink.exists()

    first_sink = tmp_path / "first.jsonl"
    second_sink = tmp_path / "second.jsonl"
    duplicate = subprocess.run(
        [
            str(runner),
            "--config",
            str(FIXTURE / "configs/three-ticks.json"),
            "--events",
            str(first_sink),
            "--events",
            str(second_sink),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert duplicate.returncode != 0
    assert not first_sink.exists() and not second_sink.exists()

    existing = tmp_path / "existing.jsonl"
    existing.write_bytes(b"keep me")
    valid = FIXTURE / "configs/three-ticks.json"
    refused = _run_model(built_design, "cpp", valid, events=existing)
    assert refused.returncode != 0
    assert existing.read_bytes() == b"keep me"

    target = tmp_path / "target.jsonl"
    target.write_bytes(b"keep symlink target")
    link = tmp_path / "linked.jsonl"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are unavailable")
    refused = _run_model(built_design, "cpp", valid, events=link)
    assert refused.returncode != 0
    assert link.is_symlink() and target.read_bytes() == b"keep symlink target"


def test_invalid_final_does_not_replace_existing_preview_bundles(
    built_design: Path, tmp_path: Path
) -> None:
    native = _require_preview_tools()
    artifacts = built_design / "artifacts"
    before = {
        target: _sha256(artifacts / target / "generated.json")
        for target in ("cpp", "verilog")
    }
    invalid_final = tmp_path / "invalid.ac"
    invalid_final.write_bytes(b"not a verified final hardware package\n")
    result = _materialize_final(invalid_final, artifacts, native)
    assert result.returncode != 0
    assert re.search(
        r"final|verify|design|parse|emit", result.stderr, re.I
    ), result.stderr
    after = {
        target: _sha256(artifacts / target / "generated.json")
        for target in ("cpp", "verilog")
    }
    assert after == before


def test_unmanaged_final_read_rechecks_if_publication_appears_during_snapshot(
    built_design: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _require_preview_tools()
    monkeypatch.setenv("PYCIRCUIT_LINKER", str(native / "bin/pycircuit-link"))
    source_final = built_design / "linked/design_top.ac"
    unmanaged = tmp_path / "unmanaged.ac"
    current_bytes = source_final.read_bytes()
    unmanaged.write_bytes(current_bytes)
    module = _materializer_module()

    from pycircuit._driver import _validate_published_program
    from pycircuit._publication import _publication_owner_program, _publish_file

    owner = _publication_owner_program(
        package="preview",
        path="design_top.py",
        definition='@"preview.design_top.DesignTop"',
    )
    original_read = module._read_stable_file_bytes
    injected = False

    def publish_after_snapshot_read(path: Path) -> bytes:
        nonlocal injected
        if path == unmanaged and not injected:
            injected = True

            def rewrite(stage: Path) -> None:
                stage.write_bytes(current_bytes)

            _publish_file(
                unmanaged,
                owner=owner,
                build=rewrite,
                validate=_validate_published_program,
                replace=True,
            )
            return b"stale unmanaged snapshot"
        return original_read(path)

    monkeypatch.setattr(module, "_read_stable_file_bytes", publish_after_snapshot_read)
    observed = module.final_snapshot(unmanaged)

    assert injected
    assert observed == current_bytes
    assert observed != b"stale unmanaged snapshot"


@pytest.mark.parametrize("symlink_cpp", [False, True])
def test_preview_refuses_unmanaged_existing_output_directories(
    built_design: Path, tmp_path: Path, symlink_cpp: bool
) -> None:
    native = _require_preview_tools()
    source_artifacts = built_design / "artifacts"
    output = tmp_path / "unmanaged-artifacts"
    output.mkdir()
    external_cpp = tmp_path / "copied-cpp"
    shutil.copytree(source_artifacts / "cpp", external_cpp)
    if symlink_cpp:
        try:
            (output / "cpp").symlink_to(external_cpp, target_is_directory=True)
        except (NotImplementedError, OSError):
            pytest.skip("directory symlinks are unavailable")
    else:
        shutil.copytree(source_artifacts / "cpp", output / "cpp")
    shutil.copytree(source_artifacts / "verilog", output / "verilog")
    before_cpp = _tree_bytes(external_cpp if symlink_cpp else output / "cpp")
    before_verilog = _tree_bytes(output / "verilog")
    final = built_design / "linked/design_top.ac"
    result = subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            str(final),
            str(output),
            str(_parts_helper(native)),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
        env=_materializer_env(native),
    )

    assert result.returncode != 0
    assert re.search(
        r"control|managed|publication|owner", result.stderr, re.I
    ), result.stderr
    assert _tree_bytes(external_cpp if symlink_cpp else output / "cpp") == before_cpp
    assert _tree_bytes(output / "verilog") == before_verilog
    assert not (output / ".cpp.pycircuit-publication").exists()
    assert not (output / ".verilog.pycircuit-publication").exists()
    if symlink_cpp:
        assert (output / "cpp").is_symlink()


def test_duplicate_report_name_in_one_source_instance_is_rejected_before_link_output(
    tmp_path: Path,
) -> None:
    native = _require_preview_tools()
    build = _configure_preview(
        tmp_path / "duplicate-report-source",
        native=native,
        design="duplicate_report_top",
    )
    result = subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "preview-build",
            "--parallel",
            "4",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )

    assert result.returncode != 0
    assert re.search(r"duplicate|report|gauge", result.stdout + result.stderr, re.I)
    assert not (build / "linked/duplicate_report_top.ac").exists()
    assert not (build / "artifacts/cpp/generated.json").exists()
    assert not (build / "artifacts/verilog/generated.json").exists()


def test_fresh_final_duplicate_report_name_is_rejected_by_both_emitters_and_materializer(
    tmp_path: Path,
) -> None:
    native = _require_preview_tools()
    valid = _build_preview(
        tmp_path / "valid-report-probe",
        native=native,
        design="report_probe_top",
    )
    original_final = valid / "linked/report_probe_top.ac"
    mutated = tmp_path / "duplicate-report.final.ac"
    mutated.write_text(
        _duplicate_report_name(original_final.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    design_helper = native / "bin/pycircuit-link"
    verify = subprocess.run(
        [str(design_helper), "--design", str(mutated), "--verify-only"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert verify.returncode != 0
    assert re.search(r"duplicate|report|gauge", verify.stderr, re.I)

    emitter = native / "bin/pycircuit-emit"
    for backend in ("cpp", "verilog"):
        output = tmp_path / f"duplicate-report-{backend}.out"
        native_target = "structural-cpp" if backend == "cpp" else backend
        emitted = subprocess.run(
            [
                str(emitter),
                str(mutated),
                "--target",
                native_target,
                "--format",
                "source",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        assert emitted.returncode != 0
        assert re.search(r"duplicate|report|gauge", emitted.stderr, re.I)
        assert not output.exists()

    artifacts = valid / "artifacts"
    before = {target: _tree_bytes(artifacts / target) for target in ("cpp", "verilog")}
    rejected = _materialize_final(mutated, artifacts, native)
    assert rejected.returncode != 0
    assert re.search(r"duplicate|report|gauge", rejected.stderr, re.I)
    assert {target: _tree_bytes(artifacts / target) for target in before} == before


def test_same_final_emits_narrow_integer_and_bool_with_distinct_json_kinds(
    tmp_path: Path,
) -> None:
    native = _require_preview_tools()
    baseline = _build_preview(
        tmp_path / "unsigned-value-probe",
        native=native,
        design="value_probe_top",
    )
    baseline_final = baseline / "linked/value_probe_top.ac"
    for backend in ("cpp", "verilog"):
        events = baseline / f"baseline-{backend}.jsonl"
        result = _run_model(
            baseline,
            backend,
            FIXTURE / "configs/three-ticks.json",
            events=events,
        )
        assert result.returncode == 0, result.stderr
        _oracle().assert_value_probe_run(events.read_bytes(), word=7)

    mutated_final = tmp_path / "mutated-value-probe.ac"
    mutated_final.write_text(
        _mutated_narrow_probe(baseline_final.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    verified = subprocess.run(
        [
            str(native / "bin/pycircuit-link"),
            "--design",
            str(mutated_final),
            "--verify-only",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert verified.returncode == 0, verified.stderr

    artifacts = tmp_path / "mutated-artifacts"
    artifacts.mkdir()
    materialized = _materialize_final(mutated_final, artifacts, native)
    assert materialized.returncode == 0, materialized.stderr
    runners = _build_materialized_models(artifacts, tmp_path / "mutated-build")
    traces: dict[str, bytes] = {}
    for backend, runner in runners.items():
        events = tmp_path / f"mutated-{backend}.jsonl"
        result = _run_materialized_runner(
            runner,
            backend,
            FIXTURE / "configs/three-ticks.json",
            events=events,
        )
        assert result.returncode == 0, result.stderr
        traces[backend] = events.read_bytes()
        _oracle().assert_value_probe_run(traces[backend], word=255)
    _oracle().assert_repeated_run_identical(traces["cpp"], traces["verilog"])


def test_wide_unused_integer_is_projected_out_by_both_hwir_backends(
    tmp_path: Path,
) -> None:
    native = _require_preview_tools()
    baseline = _build_preview(
        tmp_path / "wide-value-probe",
        native=native,
        design="value_probe_top",
    )
    wide_final = tmp_path / "wide-unused-constant.ac"
    wide_final.write_text(
        _insert_wide_unused_constant(
            (baseline / "linked/value_probe_top.ac").read_text(encoding="utf-8")
        ),
        encoding="utf-8",
    )
    helper = native / "bin/pycircuit-link"
    verified = subprocess.run(
        [str(helper), "--design", str(wide_final), "--verify-only"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert verified.returncode == 0, verified.stderr

    hwir = tmp_path / "wide-unused-constant.hwir.mlir"
    lowered = subprocess.run(
        [
            str(native / "bin/pycircuit-opt"),
            str(wide_final),
            "--ac-final-to-hwir",
            "-o",
            str(hwir),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert lowered.returncode == 0, lowered.stderr
    assert hwir.is_file()

    emitter = native / "bin/pycircuit-emit"
    for target in ("verilog", "structural-cpp"):
        emitted = subprocess.run(
            [
                str(emitter),
                str(hwir),
                "--target",
                target,
                "--format",
                "source",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        assert emitted.returncode == 0, emitted.stderr
        assert emitted.stdout

    artifacts = tmp_path / "wide-artifacts"
    artifacts.mkdir()
    materialized = _materialize_final(wide_final, artifacts, native)
    assert materialized.returncode == 0, materialized.stderr
    assert (artifacts / "cpp/generated.json").is_file()
    assert (artifacts / "verilog/generated.json").is_file()
