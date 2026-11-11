"""Source-owned RTL groups are emitted from one saved final program."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
import test_cpp_source_parts as cpp
import test_public_emit as public
import test_source_preview_workflow as preview

pytestmark = pytest.mark.system

ROOT = Path(__file__).resolve().parents[2]


def _run_emit_from_final(
    final: Path, helper: Path
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="pycircuit-source-rtl-hwir-") as temporary:
        hwir = Path(temporary) / "design.hwir.mlir"
        lowered = subprocess.run(
            [
                str(helper.with_name("pycircuit-opt")),
                str(final),
                "--ac-final-to-hwir",
                "-o",
                str(hwir),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        if lowered.returncode != 0:
            return lowered
        return subprocess.run(
            [
                str(helper),
                str(hwir),
                "--target",
                "verilog",
                "--format",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )


def _emit_from_final(final: Path, helper: Path) -> dict[str, Any]:
    completed = _run_emit_from_final(final, helper)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _build_and_run_generated_target(
    generated: Path, target: str, output: Path
) -> bytes:
    install = Path(
        os.environ.get(
            "PYCIRCUIT_COMPILER_INSTALL",
            ROOT / ".pycircuit_out/source-candidate-install",
        )
    ).resolve()
    assert (install / "share/pycircuit/cmake/pycircuitConfig.cmake").is_file(), install
    build = output / f"build-{target}"
    configured = subprocess.run(
        [
            "cmake",
            "-S",
            str(generated),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_PREFIX_PATH=" + str(install),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert configured.returncode == 0, f"{configured.stdout}\n{configured.stderr}"
    compiled = subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "pycircuit_sim",
            "--parallel",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    assert compiled.returncode == 0, f"{compiled.stdout}\n{compiled.stderr}"
    runner = build / "bin/pycircuit_sim"
    if os.name == "nt":
        runner = runner.with_suffix(".exe")
    events = output / f"{target}-events.jsonl"
    arguments = (
        [str(runner), "+cycles=3"]
        if target == "verilog"
        else [
            str(runner),
            "--cycles",
            "3",
            "--events",
            str(events),
        ]
    )
    run = subprocess.run(
        arguments,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert run.returncode == 0, f"{target}: {run.stdout}\n{run.stderr}"
    if target == "verilog":
        return b"\n".join(
            line
            for line in run.stdout.encode("utf-8").splitlines()
            if line.startswith(b"{")
        ) + b"\n"
    return events.read_bytes()


def _verilate_groups(
    payload: dict[str, Any], output: Path
) -> subprocess.CompletedProcess[str]:
    verilator = shutil.which("verilator")
    if not verilator:
        pytest.fail("source RTL integration requires Verilator")
    source_paths: list[Path] = []
    for group in payload["rtl_source_groups"]:
        relative = Path(group["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(group["text"], encoding="utf-8")
        source_paths.append(path)
    core = output / "core.sv"
    core.write_text(payload["rtl_core"], encoding="utf-8")
    testbench = output / "tb.sv"
    testbench.write_text(payload["simulation_top"], encoding="utf-8")
    install = Path(os.environ["PYCIRCUIT_COMPILER_INSTALL"])
    standard_sources = [install / path for path in payload["rtl_standard_sources"]]
    assert all(path.is_file() for path in standard_sources), standard_sources
    build = output / "obj"
    command = [
        verilator,
        "--binary",
        "--timing",
        "-Wno-fatal",
        "--top-module",
        "pycircuit_sim",
        "--Mdir",
        str(build),
        *(str(path) for path in standard_sources),
        *(str(path) for path in source_paths),
        str(core),
        str(testbench),
    ]
    compiled = subprocess.run(
        command, text=True, capture_output=True, check=False, timeout=300
    )
    assert compiled.returncode == 0, (
        f"Verilator failed ({compiled.returncode}): {command!r}\n"
        f"{compiled.stdout}\n{compiled.stderr}"
    )
    executable = build / "Vpycircuit_sim"
    return subprocess.run(
        [str(executable), "+cycles=3"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def test_source_compile_link_emits_genuine_rtl_owner_files_and_matching_behavior(
    tmp_path: Path,
) -> None:
    native = preview._require_preview_tools()
    build = preview._build_preview(tmp_path / "source-design", native=native)
    final = build / "linked/design_top.ac"
    final_hash = preview._sha256(final)

    # Ask a fresh process to recover every owner from the final program alone.
    payload = _emit_from_final(final, native / "bin/pycircuit-emit")
    groups = payload["rtl_source_groups"]
    assert [
        (group["source"]["package"], group["source"]["path"]) for group in groups
    ] == [
        ("preview", "counter.py"),
        ("preview", "design_top.py"),
    ]
    assert [group["path"] for group in groups] == [
        "sources/source_unit_0_0.sv",
        "sources/source_unit_1_0.sv",
    ]
    module_names = [
        re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", group["text"])
        for group in groups
    ]
    assert all(match is not None for match in module_names)
    counter_module = module_names[0].group(1)
    root_instances = re.findall(
        rf"\b{re.escape(counter_module)}\s+(pyc_instance_\d+)", groups[1]["text"]
    )
    assert len(root_instances) == 2 and len(set(root_instances)) == 2
    # types.py is declaration-only. It has no executable RTL owner group.
    assert all(group["source"]["path"] != "types.py" for group in groups)
    assert "module pyc_root" in payload["rtl_core"]
    assert payload["root_rtl_name"] == "pyc_root"

    cpp_events = build / "cpp-source-events.jsonl"
    cpp_result = preview._run_model(
        build,
        "cpp",
        preview.FIXTURE / "configs/three-ticks.json",
        events=cpp_events,
    )
    assert cpp_result.returncode == 0, cpp_result.stderr
    preview._oracle().assert_design_top_run(cpp_events.read_bytes())

    rtl_result = _verilate_groups(payload, tmp_path / "grouped-rtl")
    assert rtl_result.returncode == 0, rtl_result.stdout + rtl_result.stderr
    assert "finish" in rtl_result.stdout.lower()
    rtl_jsonl = b"\n".join(
        line
        for line in rtl_result.stdout.encode("utf-8").splitlines()
        if line.startswith(b"{")
    ) + b"\n"
    preview._oracle().assert_design_top_run(rtl_jsonl)
    assert preview._sha256(final) == final_hash


def test_rtl_source_module_names_that_share_a_legacy_spelling_remain_distinct(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "collision-source"
    source_root.mkdir()
    (source_root / "types.py").write_text(cpp.TYPES, encoding="utf-8")
    for filename, definition in (
        ("fooBar.py", "FirstUnit"),
        ("foo_bar.py", "SecondUnit"),
    ):
        (source_root / filename).write_text(
            "from pycircuit import module, rule\n"
            "from pycircuit import u8 as Word\n\n"
            "@module\n"
            f"def {definition}(incoming: Word, outgoing: Word):\n"
            "    @rule\n"
            "    def forward():\n"
            "        nonlocal outgoing\n"
            "        outgoing = incoming\n"
            "    forward()\n",
            encoding="utf-8",
        )
    (source_root / "top.py").write_text(
        "from pycircuit import system\n"
        "from pycircuit import u8 as Word\n"
        "from .fooBar import FirstUnit\n"
        "from .foo_bar import SecondUnit\n\n"
        "@system\n"
        "def CollisionTop():\n"
        "    first_in: Word = 3\n"
        "    second_in: Word = 10\n"
        "    first_out: Word = 0\n"
        "    second_out: Word = 0\n"
        "    first = FirstUnit(first_in, first_out)\n"
        "    second = SecondUnit(second_in, second_out)\n",
        encoding="utf-8",
    )
    types = cpp._compile_source(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    first = cpp._compile_source(
        source_root / "fooBar.py",
        source_root=source_root,
        output_dir=tmp_path / "units/fooBar",
        headers=(types.header,),
    )
    second = cpp._compile_source(
        source_root / "foo_bar.py",
        source_root=source_root,
        output_dir=tmp_path / "units/foo_bar",
        headers=(types.header,),
    )
    top = cpp._compile_source(
        source_root / "top.py",
        source_root=source_root,
        output_dir=tmp_path / "units/top",
        headers=(types.header, first.header, second.header),
    )
    final = tmp_path / "collision.final.ac"
    linked = cpp._link(
        [types, first, second, top],
        final,
        top="demo.top.CollisionTop",
    )
    assert linked.returncode == 0, linked.stderr

    completed = _run_emit_from_final(
        final,
        preview._parts_helper(preview._require_preview_tools()),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    groups = {
        group["source"]["path"]: group["text"]
        for group in payload["rtl_source_groups"]
    }
    definitions = []
    for source in ("fooBar.py", "foo_bar.py"):
        match = re.search(
            r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", groups[source]
        )
        assert match is not None
        definitions.append(match.group(1))
    assert len(set(definitions)) == 2


def test_unicode_value_formals_compile_and_match_on_cpp_and_verilog(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "unicode-source"
    shutil.copytree(public.FIXTURE / "src", source_root)
    counter = source_root / "counter.py"
    text = counter.read_text(encoding="utf-8")
    counter.write_text(
        text.replace("incoming", "输入").replace("outgoing", "输出"),
        encoding="utf-8",
    )
    final = public._compile_link(source_root, tmp_path / "published", "source_public")
    cpp_output, rtl_output = tmp_path / "unicode-cpp", tmp_path / "unicode-rtl"
    public._emit(final, "structural-cpp", cpp_output)
    public._emit(final, "verilog", rtl_output)

    rtl = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(rtl_output.rglob("*.sv"))
    )
    assert "输入" not in rtl and "输出" not in rtl

    cpp_events = _build_and_run_generated_target(
        cpp_output, "cpp", tmp_path / "unicode-run"
    )
    rtl_events = _build_and_run_generated_target(
        rtl_output, "verilog", tmp_path / "unicode-run"
    )
    oracle = preview._oracle()
    oracle.assert_design_top_run(cpp_events)
    oracle.assert_design_top_run(rtl_events)


def test_verilog_formals_that_differ_by_case_remain_distinct(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "formal-collision-source"
    source_root.mkdir()
    (source_root / "types.py").write_text(cpp.TYPES, encoding="utf-8")
    (source_root / "child.py").write_text(
        "from pycircuit import module, rule\n"
        "from pycircuit import u8 as Word\n\n"
        "@module\n"
        "def Child(A: Word, a: Word):\n"
        "    @rule\n"
        "    def combine():\n"
        "        nonlocal a\n"
        "        a = A\n"
        "    combine()\n",
        encoding="utf-8",
    )
    (source_root / "top.py").write_text(
        "from pycircuit import module\n"
        "from pycircuit import u8 as Word\n"
        "from .child import Child\n\n"
        "@module\n"
        "def Top():\n"
        "    upper: Word = 3\n"
        "    lower: Word = 5\n"
        "    child = Child(upper, lower)\n",
        encoding="utf-8",
    )
    types = cpp._compile_source(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    child = cpp._compile_source(
        source_root / "child.py",
        source_root=source_root,
        output_dir=tmp_path / "units/child",
        headers=(types.header,),
    )
    top = cpp._compile_source(
        source_root / "top.py",
        source_root=source_root,
        output_dir=tmp_path / "units/top",
        headers=(types.header, child.header),
    )
    final = tmp_path / "formal-collision.final.ac"
    linked = cpp._link([types, child, top], final, top="demo.top.Top")
    assert linked.returncode == 0, linked.stderr

    output = tmp_path / "case-distinct-rtl"
    public._emit(final, "verilog", output)
    rtl = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(output.rglob("*.sv"))
    )
    assert re.search(r"\binput wire logic \[7:0\] A\b", rtl)
    assert re.search(r"\binput wire logic \[7:0\] a\b", rtl)


def test_verilog_formal_and_owned_state_use_distinct_names(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "formal-state-collision-source"
    source_root.mkdir()
    (source_root / "types.py").write_text(cpp.TYPES, encoding="utf-8")
    (source_root / "child.py").write_text(
        "from pycircuit import module, rule\n"
        "from pycircuit import u8 as Word\n\n"
        "@module\n"
        "def Child(Q0: Word):\n"
        "    state0: Word = 0\n\n"
        "    @rule\n"
        "    def advance():\n"
        "        nonlocal state0\n"
        "        state0 = Q0\n"
        "    advance()\n",
        encoding="utf-8",
    )
    (source_root / "top.py").write_text(
        "from pycircuit import module\n"
        "from pycircuit import u8 as Word\n"
        "from .child import Child\n\n"
        "@module\n"
        "def Top():\n"
        "    value: Word = 0\n"
        "    child = Child(value)\n",
        encoding="utf-8",
    )
    types = cpp._compile_source(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    child = cpp._compile_source(
        source_root / "child.py",
        source_root=source_root,
        output_dir=tmp_path / "units/child",
        headers=(types.header,),
    )
    top = cpp._compile_source(
        source_root / "top.py",
        source_root=source_root,
        output_dir=tmp_path / "units/top",
        headers=(types.header, child.header),
    )
    final = tmp_path / "formal-state-collision.final.ac"
    linked = cpp._link([types, child, top], final, top="demo.top.Top")
    assert linked.returncode == 0, linked.stderr

    output = tmp_path / "formal-state-distinct-rtl"
    public._emit(final, "verilog", output)
    rtl = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(output.rglob("*.sv"))
    )
    assert re.search(r"\binput wire logic \[7:0\] Q0\b", rtl)
    assert "dffe #( .T(logic [7:0]) ) pyc_instance_" in rtl


def test_verilog_parent_formal_and_child_result_wire_use_distinct_names(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "parent-wire-collision-source"
    source_root.mkdir()
    (source_root / "types.py").write_text(cpp.TYPES, encoding="utf-8")
    (source_root / "child.py").write_text(
        "from pycircuit import module, rule, struct\n"
        "from pycircuit import u8 as Word\n\n"
        "@struct\n"
        "class ChildResult:\n"
        "    value: Word\n\n"
        "@module\n"
        "def Child(incoming: Word) -> ChildResult:\n"
        "    @rule\n"
        "    def forward() -> ChildResult:\n"
        "        return ChildResult(value=incoming)\n"
        "    return forward()\n",
        encoding="utf-8",
    )
    (source_root / "parent.py").write_text(
        "from pycircuit import module\n"
        "from pycircuit import u8 as Word\n"
        "from .child import Child\n\n"
        "@module\n"
        "def Parent(child_0_result: Word, incoming: Word):\n"
        "    child = Child(incoming)\n",
        encoding="utf-8",
    )
    (source_root / "top.py").write_text(
        "from pycircuit import module\n"
        "from pycircuit import u8 as Word\n"
        "from .parent import Parent\n\n"
        "@module\n"
        "def Top():\n"
        "    incoming: Word = 3\n"
        "    outgoing: Word = 0\n"
        "    parent = Parent(outgoing, incoming)\n",
        encoding="utf-8",
    )
    types = cpp._compile_source(
        source_root / "types.py",
        source_root=source_root,
        output_dir=tmp_path / "units/types",
    )
    child = cpp._compile_source(
        source_root / "child.py",
        source_root=source_root,
        output_dir=tmp_path / "units/child",
        headers=(types.header,),
    )
    parent = cpp._compile_source(
        source_root / "parent.py",
        source_root=source_root,
        output_dir=tmp_path / "units/parent",
        headers=(types.header, child.header),
    )
    top = cpp._compile_source(
        source_root / "top.py",
        source_root=source_root,
        output_dir=tmp_path / "units/top",
        headers=(types.header, child.header, parent.header),
    )
    final = tmp_path / "parent-wire-collision.final.ac"
    linked = cpp._link([types, child, parent, top], final, top="demo.top.Top")
    assert linked.returncode == 0, linked.stderr

    output = tmp_path / "parent-wire-distinct-rtl"
    public._emit(final, "verilog", output)
    rtl = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(output.rglob("*.sv"))
    )
    assert re.search(r"\binput wire logic \[7:0\] child_0_result\b", rtl)
    assert ".result(" in rtl


def test_distinct_source_owners_can_each_define_a_child_named_child(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "same-short-name-source"
    for directory in (source_root / "left", source_root / "right"):
        directory.mkdir(parents=True, exist_ok=True)
    child_source = (
        "from typing import Annotated\n"
        "from pycircuit import module, rule, struct\n\n"
        "Word = Annotated[int, range(256)]\n\n"
        "@struct\n"
        "class ChildResult:\n"
        "    value: Word\n\n"
        "@module\n"
        "def Child(incoming: Word) -> ChildResult:\n"
        "    @rule\n"
        "    def forward() -> ChildResult:\n"
        "        return ChildResult(value=incoming)\n"
        "    return forward()\n"
    )
    for relative in ("left/child.py", "right/child.py"):
        (source_root / relative).write_text(child_source, encoding="utf-8")
    (source_root / "top.py").write_text(
        "from typing import Annotated\n"
        "from pycircuit import module\n"
        "from .left.child import Child as LeftChild\n"
        "from .right.child import Child as RightChild\n\n"
        "Word = Annotated[int, range(256)]\n\n"
        "@module\n"
        "def Top():\n"
        "    left_in: Word = 3\n"
        "    right_in: Word = 10\n"
        "    left_out: Word = 0\n"
        "    right_out: Word = 0\n"
        "    left = LeftChild(left_in)\n"
        "    right = RightChild(right_in)\n",
        encoding="utf-8",
    )
    left = cpp._compile_source(
        source_root / "left/child.py",
        source_root=source_root,
        output_dir=tmp_path / "units/left-child",
    )
    right = cpp._compile_source(
        source_root / "right/child.py",
        source_root=source_root,
        output_dir=tmp_path / "units/right-child",
    )
    top = cpp._compile_source(
        source_root / "top.py",
        source_root=source_root,
        output_dir=tmp_path / "units/top",
        headers=(left.header, right.header),
    )
    final = tmp_path / "same-short-name.final.ac"
    linked = cpp._link([left, right, top], final, top="demo.top.Top")
    assert linked.returncode == 0, linked.stderr

    output = tmp_path / "same-short-name-rtl"
    public._emit(final, "verilog", output)
    receipt = json.loads((output / "generated.json").read_text(encoding="utf-8"))
    groups = {group["source"]["path"]: group for group in receipt["source_groups"]}
    assert "left/child.py" in groups
    assert "right/child.py" in groups
    module_names = []
    for owner in ("left/child.py", "right/child.py"):
        source_files = [path for path in groups[owner]["files"] if path.endswith(".sv")]
        assert len(source_files) == 1
        text = (output / source_files[0]).read_text(encoding="utf-8")
        match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", text)
        assert match is not None
        module_names.append(match.group(1))
    assert len(set(module_names)) == 2
