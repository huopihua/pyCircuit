"""Acceptance for the supported structural C++ host interface.

The historical shared-library C ABI (``dut.h``, ``model_api.cpp`` and
``pycircuit_model_query_v1``) is intentionally not part of the independent
HWIR backend.  The public contract is the generated ``design.hpp`` header, the
``pycircuit_modules`` CMake target, and the typed ``pyc_dut`` interface.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import test_cpp_source_parts as cpp
import test_final_value_declarations as declarations
import test_source_preview_workflow as source_preview

pytestmark = pytest.mark.system

ROOT = Path(__file__).resolve().parents[2]


def _probe_source(mode: str) -> str:
    behavior = {
        "design": """
  if (dut.system().Step() != gfsim::SimStepResult::Running) return 10;
  if (dut.system().cycle() != 1) return 11;
  dut.system().Reset();
  if (dut.system().cycle() != 0 ||
      dut.system().state() != gfsim::SimSystemState::Ready) return 12;
  if (dut.system().Step() != gfsim::SimStepResult::Running) return 13;
""",
        "hold": """
  if (dut.system().Step() != gfsim::SimStepResult::Running) return 20;
  if (dut.system().cycle() != 1) return 21;
  dut.system().Reset();
  if (dut.system().Step() != gfsim::SimStepResult::Running ||
      dut.system().cycle() != 1) return 22;
""",
        "zero": """
  if (dut.system().Step() != gfsim::SimStepResult::Running) return 30;
  if (dut.system().cycle() != 1) return 31;
  dut.system().Reset();
  dut.drive({bit(true), bit(false)});
  if (dut.system().Step() != gfsim::SimStepResult::Running ||
      dut.system().cycle() != 1) return 32;
""",
        "failure": """
  bool failed = false;
  for (unsigned edge = 0; edge != 8 && !failed; ++edge) {
    dut.drive({bit((edge & 1u) == 0), bit(false)});
    const auto result = dut.system().Step();
    if (result == gfsim::SimStepResult::Failed) failed = true;
    else if (result != gfsim::SimStepResult::Running) return 40;
  }
  if (!failed || dut.system().cycle() == 0) return 41;
  if (dut.system().failureInfo().phase != gfsim::SimFailurePhase::Check) return 42;
  if (dut.system().Step() != gfsim::SimStepResult::Failed) return 44;
  dut.system().Reset();
  if (dut.system().cycle() != 0 ||
      dut.system().state() != gfsim::SimSystemState::Ready ||
      !dut.observations().Events().empty()) return 45;
  dut.drive({bit(false), bit(false)});
  if (dut.system().Step() != gfsim::SimStepResult::Running) return 46;
  dut.drive({bit(true), bit(false)});
  if (dut.system().Step() != gfsim::SimStepResult::Running) return 47;
""",
    }[mode]
    return f"""
#include "design.hpp"
#include <type_traits>

static_assert(!std::is_copy_constructible_v<pyc_dut>);
static_assert(!std::is_copy_assignable_v<pyc_dut>);

int main() {{
  const auto bit = [](bool value) {{
    return gfsim::wire<gfsim::Bits<1>>::known(
        gfsim::Bits<1>{{value ? 1u : 0u}});
  }};
  pyc_dut dut(2);
  if (dut.system().state() != gfsim::SimSystemState::PreBuild) return 1;
  if (!dut.system().HasObservations(dut.observations())) return 2;
  (void)dut.root();
  dut.drive({{bit(false), bit(false)}});
  dut.system().Build();
  if (dut.system().state() != gfsim::SimSystemState::Built) return 3;
  dut.system().Reset();
  if (dut.system().state() != gfsim::SimSystemState::Ready ||
      dut.system().cycle() != 0) return 4;
  dut.drive({{bit(true), bit(false)}});
{behavior}
  return 0;
}}
"""


def _build_dut(build: Path, design: str, mode: str, native: Path) -> tuple[Path, Path]:
    source_preview._configure_preview(build, native=native, design=design)
    source_preview._checked(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "preview-artifacts",
            "--parallel",
            "2",
        ]
    )
    artifacts = build / "artifacts/cpp"
    header = artifacts / "design.hpp"
    assert header.is_file()
    assert not (artifacts / "dut.h").exists()
    assert not (artifacts / "model_api.cpp").exists()

    probe = artifacts / "model_abi_probe.cpp"
    probe.write_text(_probe_source(mode), encoding="utf-8")
    with (artifacts / "CMakeLists.txt").open("a", encoding="utf-8") as cmake:
        cmake.write(
            "\nadd_executable(model_abi_probe model_abi_probe.cpp)\n"
            "target_compile_features(model_abi_probe PRIVATE cxx_std_20)\n"
            "target_link_libraries(model_abi_probe PRIVATE pycircuit_modules)\n"
        )

    model_build = build / "model-cpp"
    source_preview._checked(
        [
            "cmake",
            "-S",
            str(artifacts),
            "-B",
            str(model_build),
            "-G",
            "Ninja",
            "-DPYCIRCUIT_RUNTIME_ROOT=" + str(ROOT / "runtime"),
        ]
    )
    source_preview._checked(
        [
            "cmake",
            "--build",
            str(model_build),
            "--target",
            "model_abi_probe",
            "--parallel",
            "2",
        ]
    )
    executable = model_build / "model_abi_probe"
    if not executable.is_file():
        executable = executable.with_suffix(".exe")
    assert executable.is_file()
    return artifacts, executable


@pytest.fixture(scope="module")
def generated_duts(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> dict[str, dict[str, Path]]:
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    native = source_preview._native_build()
    monkeypatch.setenv("PYCIRCUIT_NATIVE_BUILD", str(native))
    native = source_preview._require_preview_tools()
    monkeypatch.setenv(
        "PYCIRCUIT_SOURCE_COMPILER", str(native / "bin/pycircuit-source-unit")
    )
    monkeypatch.setenv("PYCIRCUIT_LINKER", str(native / "bin/pycircuit-link"))
    monkeypatch.setenv("PYCIRCUIT_EMITTER", str(native / "bin/pycircuit-emit"))
    monkeypatch.setenv("PYCIRCUIT_OPT", str(native / "bin/pycircuit-opt"))
    root = tmp_path_factory.mktemp("structural-cpp-host-api")
    result: dict[str, dict[str, Path]] = {}
    for design, mode in (
        ("design_top", "design"),
        ("hold_top", "hold"),
        ("zero_rule_top", "zero"),
        ("failure_top", "failure"),
    ):
        artifacts, executable = _build_dut(root / design, design, mode, native)
        result[design] = {"artifacts": artifacts, "executable": executable}
    return result


@pytest.mark.parametrize("design", ["design_top", "hold_top", "zero_rule_top", "failure_top"])
def test_generated_typed_cpp_host_links_and_behaves(
    generated_duts: dict[str, dict[str, Path]], design: str
) -> None:
    completed = subprocess.run(
        [str(generated_duts[design]["executable"])],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_generated_header_exposes_only_the_supported_typed_host_contract(
    generated_duts: dict[str, dict[str, Path]],
) -> None:
    text = generated_duts["design_top"]["artifacts"].joinpath("design.hpp").read_text()
    assert "class pyc_structural_system final" in text
    assert "class pyc_dut final" in text
    assert "gfsim::SimSystem &system()" in text
    assert "gfsim::ObservationSlots &observations()" in text
    assert "pycircuit_model_query_v1" not in text


def _global_declaration(name: str, kind: str) -> str:
    owner = 'ac.source_owner = {package = "", path = "__init__.py"}'
    origin = (
        f'ac.origin = {{site = {{definition = @"{name}", ast_path = []}}, '
        "expansion = []}"
    )
    if kind == "alias":
        return (
            f'    "ac.type_alias"() <{{sym_name = "{name}", '
            'target = {kind = "bool", storage = i1}}> {'
            + owner + ", " + origin
            + ', ac.declaration_role = "definition"} : () -> () '
            'loc("__init__.py":1:1)\n'
        )
    return (
        f'    "ac.constant"() <{{sym_name = "{name}", '
        'type = {kind = "bool"}, value = {kind = "bool", value = true}}> {'
        + owner + ", " + origin
        + ', ac.declaration_role = "definition"} : () -> () '
        'loc("__init__.py":1:1)\n'
    )


@pytest.mark.parametrize("kind", ["alias", "constant"])
def test_retired_c_abi_symbol_is_not_a_reserved_source_declaration(
    tmp_path: Path, kind: str
) -> None:
    final, _units = declarations._build(tmp_path / kind)
    variant = tmp_path / f"global-{kind}.ac"
    variant.write_text(
        declarations._insert_unit(
            final.read_text(),
            "__init__.py",
            _global_declaration("pycircuit_model_query_v1", kind),
            package="",
        ),
        encoding="utf-8",
    )
    emitted = cpp._emit_hwir(variant, "--target", "structural-cpp", "--format", "source")
    assert emitted.returncode == 0, emitted.stderr
    assert "pycircuit_model_query_v1" not in emitted.stdout


def _build_empty_package_module(
    directory: Path, *, path: str, source: str, definition: str, package: str = ""
) -> Path:
    source_root = directory / "source"
    source_root.mkdir(parents=True)
    source_path = source_root / path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")
    unit = cpp._compile_source(
        source_path, source_root=source_root, output_dir=directory / "unit", package=package
    )
    final = directory / "model.ac"
    linked = cpp._link([unit], final, top=definition)
    assert linked.returncode == 0, linked.stderr
    return final


@pytest.mark.parametrize(
    ("source_path", "function", "definition"),
    [
        ("pycircuit_model_query_v1.py", "Root", "pycircuit_model_query_v1.Root"),
        ("__init__.py", "PycircuitModelQueryV1", "PycircuitModelQueryV1"),
    ],
)
def test_retired_c_abi_spelling_is_valid_in_the_source_namespace(
    tmp_path: Path, source_path: str, function: str, definition: str
) -> None:
    final = _build_empty_package_module(
        tmp_path,
        path=source_path,
        source=(
            "from pycircuit import module\n\n"
            "@module\n"
            f"def {function}():\n"
            "    state: bool = True\n"
        ),
        definition=definition,
    )
    emitted = cpp._emit_hwir(final, "--target", "structural-cpp", "--format", "source")
    assert emitted.returncode == 0, emitted.stderr
    assert "pycircuit_model_query_v1" not in emitted.stdout


def test_nested_retired_c_abi_spelling_remains_a_valid_cpp_namespace(
    tmp_path: Path,
) -> None:
    final = _build_empty_package_module(
        tmp_path,
        path="pycircuit_model_query_v1.py",
        source=(
            "from pycircuit import module\n\n"
            "@module\n"
            "def Root():\n"
            "    state: bool = True\n"
        ),
        definition="demo.pycircuit_model_query_v1.Root",
        package="demo",
    )
    emitted = cpp._emit_hwir(final, "--target", "structural-cpp", "--format", "source")
    assert emitted.returncode == 0, emitted.stderr
    assert "pycircuit_model_query_v1" not in emitted.stdout
