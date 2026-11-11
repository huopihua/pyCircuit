"""Verify complete public log samples and the actual nested storage owners."""

import argparse
import hashlib
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True)
parser.add_argument("--scratch", required=True)
args = parser.parse_args()
root = Path(args.repo).resolve()
gate = Path(args.scratch).resolve()
fixture = Path(__file__).resolve().parent
vectors = root / "tests/compiler/oracles/history_slot_issue/vectors.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pack(value):
    if not isinstance(value, dict):
        return int(value)
    if set(value) == {"value"}:
        return int(value["value"])
    if set(value) == {"tag", "valid"}:
        return (int(value["tag"]) << 1) | int(value["valid"])
    result = 0
    for field, width in (
        ("valid", 1),
        ("age", 8),
        ("src0_tag", 8),
        ("src0_ready", 1),
        ("src1_tag", 8),
        ("src1_ready", 1),
    ):
        result = (result << width) | int(value[field])
    return result


def observed(events, key, value):
    if not isinstance(value, dict):
        return events[key]
    if set(value) == {"value"}:
        return events[key] if key in events else events[key + "_value"]
    return pack({field: events[key + "_" + field] for field in value})


receipt = json.loads((gate / "receipt.json").read_text())
assert sha(vectors) == receipt["before"][str(vectors)]
assert receipt["unchanged"] and len(receipt["records"]) == 27
records = []
for name, case in json.loads(vectors.read_text())["executable"].items():
    for backend in ("workers1", "workers2"):
        path = gate / (name + "-" + backend + ".stdout")
        assert path.is_file(), path
        logs = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.startswith("{")
        ]
        logs = [item for item in logs if item["kind"] == "log"]
        by_epoch = {}
        for item in logs:
            epoch = int(item["evaluation_epoch"])
            event = item["spec"]["event"]
            value = item["values"]
            assert len(value) == 1 and value[0]["kind"] in ("bool", "integer"), item
            assert event not in by_epoch.setdefault(epoch, {}), (
                name,
                backend,
                epoch,
                event,
            )
            by_epoch[epoch][event] = int(value[0]["value"])
        assert set(by_epoch) == set(range(len(case["rows"]) * 2)), (name, backend)
        for epoch, events in by_epoch.items():
            row = case["rows"][epoch // 2]
            assert events["epoch"] == epoch // 2, (name, backend, epoch)
            for key, value in row["controls"].items():
                assert observed(events, key, value) == pack(value), (
                    name,
                    backend,
                    epoch,
                    key,
                    observed(events, key, value),
                    value,
                )
        records.append(
            {
                "case": name,
                "backend": backend,
                "epochs": len(case["rows"]),
                "physical_work_samples": len(by_epoch),
                "public_logs": len(logs),
                "stdout_sha256": sha(path),
            }
        )
cpp = gate / "mailbox_original_body/structural-cpp"
header_path = cpp / "design.hpp"
header = header_path.read_text()
classes = dict(
    re.findall(
        r"class (pyc_\w+) final : public gfsim::SimModule \{(.*?)\n\};",
        header,
        re.S,
    )
)
decoded = {
    typ: bytes.fromhex(typ[4:].split("__h", 1)[0]).decode().rsplit(".", 1)[-1]
    for typ in classes
}


def concrete(name):
    found = [(typ, body) for typ, body in classes.items() if decoded[typ] == name]
    assert len(found) == 1, (name, found)
    return found[0]


nested_type, nested = concrete("nested_mailbox")
children = [
    (decoded[typ], member)
    for typ, member in re.findall(r"^\s*(pyc_\w+)\s+(pyc_\w+);$", nested, re.M)
    if decoded.get(typ) in ("valid_cell", "payload_cell")
]
assert sorted(x[0] for x in children) == ["payload_cell", "valid_cell"], children
owners = {}
for name in ("valid_cell", "payload_cell"):
    typ, text = concrete(name)
    declarations = re.findall(r"^\s*gfsim::dffe<[^;\n]+>\s+(pyc_\w+);$", text, re.M)
    assert declarations == ["pyc_q"], declarations
    interface = gate / (name + "-unit") / (name + ".interface.ac")
    assert (
        "ac.domain_inputs = {clock = 2 : i64, reset = 3 : i64}" in interface.read_text()
    )
    owners[name] = {
        "storage": declarations,
        "concrete_type": typ,
        "design_header_sha256": sha(header_path),
        "published_interface_sha256": sha(interface),
    }
rtl = gate / "mailbox_original_body/verilog/sources"
rtl_text = "\n".join(path.read_text() for path in rtl.glob("*.sv"))
for name in owners:
    module = "pyc_rtl_" + owners[name]["concrete_type"]
    body = rtl_text.split("module " + module + " (", 1)[1].split("endmodule", 1)[0]
    assert body.count("  dffe #(") == 1
    assert ".pyc_managed(1'b1)" in body
    owners[name]["rtl_module"] = module
rtl_nested = rtl_text.split("module pyc_rtl_" + nested_type + " (", 1)[1].split(
    "endmodule", 1
)[0]
for cell in ("valid_cell", "payload_cell"):
    assert rtl_nested.count("  pyc_rtl_" + owners[cell]["concrete_type"]) == 1
for port in ("pyc_7079635f636c6b", "pyc_7079635f727374"):
    assert rtl_nested.count(f".{port}(") == 2
interface = gate / "providers-unit/providers.interface.ac"
line = next(
    s
    for s in interface.read_text().splitlines()
    if 'sym_name = "history_slot_issue.providers.nested_mailbox"' in s
    and 'ac.declaration_role = "definition"' in s
)
assert "ac.domain_inputs = {clock = 3 : i64, reset = 4 : i64}" in line
for name in ("payload_types", "valid_cell", "payload_cell", "providers", "systems"):
    file = fixture / (name + ".py")
    assert not re.search(r"\bpyc_(clk|rst)\b", file.read_text()), file
output = {
    "public_observations": records,
    "nested_physical_owners": owners,
    "nested_children": children,
    "compiler_domain_propagation": True,
    "authored_hidden_pins": False,
    "vectors_sha256": sha(vectors),
    "audit_script_sha256": sha(Path(__file__)),
}
(gate / "public-observations-and-nested-owner-proof.json").write_text(
    json.dumps(output, indent=2) + "\n"
)
print("public log and nested physical owner proof passed", len(records))
