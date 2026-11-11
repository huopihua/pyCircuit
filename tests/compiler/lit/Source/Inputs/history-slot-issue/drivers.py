"""Read-only hierarchy observers and literal-input source generation for the gate."""

import ast
import re

ENTRY_FIELDS = (
    ("valid", 1),
    ("age", 8),
    ("src0_tag", 8),
    ("src0_ready", 1),
    ("src1_tag", 8),
    ("src1_ready", 1),
)
WAKE_FIELDS = (("tag", 8), ("valid", 1))


def constructor(kind, value):
    if kind == "Event":
        return f"Event(value={int(value['value'])})"
    fields = ENTRY_FIELDS if kind == "Entry" else WAKE_FIELDS
    return kind + "(" + ", ".join(f"{key}={int(value[key])}" for key, _ in fields) + ")"


def scheduled_source(template, case):
    """Only independently authored input actions become source literals."""
    kind = case["kind"]
    target = kind + "_stimulus"
    tree = ast.parse(template)
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == target
    )
    if kind == "mailbox":
        initial = "MailboxFrame(left_valid=0, left_data=Event(value=0), left_take=0, right_valid=0, right_data=Event(value=0), right_take=0)"
    else:
        initial = "IssueFrame(allocation_valid=0, allocation_data=Entry(valid=0, age=0, src0_tag=0, src0_ready=0, src1_tag=0, src1_ready=0), wakeup_valid=0, wakeup_data=Wakeup(tag=0, valid=0), issued_take=0)"
    body = [
        f"@rule\ndef {target}(epoch: u16) -> {'MailboxFrame' if kind == 'mailbox' else 'IssueFrame'}:",
        f"    frame = {initial}",
    ]
    for row in case["rows"]:
        actions = row["actions"]
        assert not actions.get(
            "inject"
        ), "exceptional historical host injection is reference-only"
        lines = []
        for queue_name, values in actions.get("offer", {}).items():
            assert len(values) == 1, "one physical offer per input per epoch"
            name = (
                queue_name.removesuffix("_input")
                if kind == "mailbox"
                else {"allocations": "allocation", "wakeups": "wakeup"}[queue_name]
            )
            carrier = (
                "Event"
                if kind == "mailbox"
                else ("Entry" if queue_name == "allocations" else "Wakeup")
            )
            lines.extend(
                (
                    f"        frame.{name}_valid = 1",
                    f"        frame.{name}_data = {constructor(carrier, values[0])}",
                )
            )
        for queue_name in (*actions.get("take", []), *actions.get("sink", [])):
            name = queue_name.removesuffix("_output") if kind == "mailbox" else "issued"
            lines.append(f"        frame.{name}_take = 1")
        if lines:
            body.append(f"    if epoch == {row['epoch']}:")
            body.extend(lines)
    body.append("    return frame\n")
    lines = template.splitlines(keepends=True)
    start = min(node.lineno, *(d.lineno for d in node.decorator_list)) - 1
    return "".join(lines[:start]) + "\n".join(body) + "".join(lines[node.end_lineno :])


def concrete_classes(header):
    return dict(
        re.findall(
            r"class (pyc_\w+) final : public gfsim::SimModule \{(.*?)\n\};",
            header,
            re.S,
        )
    )


def decoded_classes(classes):
    return {
        typ: bytes.fromhex(typ[4:].split("__h", 1)[0]).decode().rsplit(".", 1)[-1]
        for typ in classes
    }


def class_for(classes, decoded, name):
    matches = [typ for typ, actual in decoded.items() if actual == name]
    assert len(matches) == 1, (name, matches)
    return matches[0], classes[matches[0]]


def children(section, decoded):
    return [
        (decoded.get(typ), name)
        for typ, name in re.findall(r"^\s*(pyc_\w+)\s+(pyc_\w+);$", section, re.M)
        if typ in decoded
    ]


def child(section, decoded, name):
    found = [member for actual, member in children(section, decoded) if actual == name]
    assert len(found) == 1, (name, found)
    return found[0]


def queues(section):
    found = re.findall(r"^\s*gfsim::fifo<[^;\n]+>\s+(pyc_\w+);$", section, re.M)
    assert len(found) in (3, 4), found
    return found


def cpp_config(header, kind):
    """Locate actual emitted owners by their types; never supply expected data."""
    classes = concrete_classes(header)
    decoded = decoded_classes(classes)
    _root_type, section = class_for(
        classes, decoded, "slot_rule_mailbox" if kind == "mailbox" else "resident_issue"
    )
    root = "dut.root()"
    if kind == "mailbox":
        fifo = queues(section)
        left = root + "." + child(section, decoded, "explicit_mailbox")
        right = root + "." + child(section, decoded, "nested_mailbox")
        _nested_type, nested_section = class_for(classes, decoded, "nested_mailbox")
        right_valid = (
            right
            + "."
            + child(nested_section, decoded, "valid_cell")
            + ".pyc_q"
        )
        right_payload = (
            right
            + "."
            + child(nested_section, decoded, "payload_cell")
            + ".pyc_q"
        )
        image = []
        for path, incoming, outgoing in (
            (left, fifo[0], fifo[2]),
            (right, fifo[1], fifo[3]),
        ):
            image += [
                f"queue_bits({root}.{incoming}, 1)",
                f"bits({right_valid if path == right else path + '.pyc_slot_valid'}.currentState().q)",
                f"bits({right_payload if path == right else path + '.pyc_slot_payload'}.currentState().q)",
                f"queue_bits({root}.{outgoing}, 1)",
            ]
        control_wires = {}
        for side, path, incoming, outgoing in (
            ("left", left, fifo[0], fifo[2]),
            ("right", right, fifo[1], fifo[3]),
        ):
            inp, out = incoming.removesuffix("_state"), outgoing.removesuffix("_state")
            for port, base in (("in", inp), ("out", out)):
                for label, actual in (
                    ("ready", "in_ready"),
                    ("available", "out_valid"),
                    ("head", "out_data"),
                ):
                    control_wires[f"{side}_{port}_{label}"] = (
                        f"{root}.{incoming if port == 'in' else outgoing}.{actual}"
                    )
            if side == "left":
                result = f"{path}.result.packed()"
                control_wires.update(
                    {
                        f"{side}_capture": f"gfsim::wire<gfsim::Bits<1>>::fromPacked(gfsim::extract<1>({result}, 9))",
                        f"{side}_publish": f"gfsim::wire<gfsim::Bits<1>>::fromPacked(gfsim::extract<1>({result}, 8))",
                        f"{side}_publish_data": f"gfsim::wire<gfsim::Bits<8>>::fromPacked(gfsim::extract<8>({result}, 0))",
                    }
                )
            else:
                control_wires.update(
                    {
                        f"{side}_capture": f"{path}.incoming_take",
                        f"{side}_publish": f"{path}.outgoing_valid",
                        f"{side}_publish_data": f"{path}.outgoing_data",
                    }
                )

    else:
        root += "." + child(section, decoded, "issue")
        _issue_type, section = class_for(classes, decoded, "issue")
        fifo = queues(section)
        core = root + "." + child(section, decoded, "issue_core")
        image = [
            f"queue_bits({root}.{fifo[1]}, 2)",
            f"queue_bits({root}.{fifo[0]}, 2)",
            f"bits({core}.pyc_wakeup_valid.currentState().q)",
            f"bits({core}.pyc_wakeup_payload.currentState().q)",
            f"bits({core}.pyc_allocation_valid.currentState().q)",
            f"bits({core}.pyc_allocation_payload.currentState().q)",
        ]
        image += [
            f"bits({core}.pyc_entries_{i}.currentState().q)"
            for i in range(4)
        ]
        image += [f"queue_bits({root}.{fifo[2]}, 1)"]
        names = (
            "allocation_ready",
            "allocation_available",
            "allocation_head",
            "wakeup_ready",
            "wakeup_available",
            "wakeup_head",
            "issued_ready",
            "issued_available",
            "issued_head",
            "allocation_capture",
            "wakeup_capture",
            "allocation_install",
            "wakeup_release",
            "selected",
            "publish",
            "publish_data",
        )
        control_wires = {name: f"{root}.{name}" for name in names}
    controls = "\n".join(
        f'out += "{name}=" + std::to_string(unsigned_value({wire})) + ";";'
        for name, wire in control_wires.items()
    )
    return (
        "auto snapshot = [&]() { return "
        + " + ".join(image)
        + "; };\nauto controls = [&]() { std::string out; "
        + controls
        + " return out; };"
    )


def unpack(name, value, kind):
    fields = (
        (("value", 8),)
        if kind == "mailbox"
        else (WAKE_FIELDS if name.startswith("wakeup") else ENTRY_FIELDS)
    )
    result = {}
    for field, width in reversed(fields):
        result[field] = value & ((1 << width) - 1)
        value >>= width
    assert value == 0
    return result


def parse_controls(text, kind):
    result = {}
    for part in text.rstrip(";").split(";"):
        name, value = part.split("=")
        value = int(value)
        result[name] = (
            unpack(name, value, kind)
            if name.endswith(("_head", "_data"))
            else bool(value)
        )
    return result


def declarations(section, decoded):
    result = []
    pattern = re.compile(
        r"^\s*((?:pyc_\w+)|(?:gfsim::fifo<[^;\n]+>)|(?:gfsim::dffe<[^;\n]+>))\s+(pyc_\w+);$",
        re.M,
    )
    for typ, member in pattern.findall(section):
        result.append((decoded.get(typ), member, typ))
    return result


def instance(section, decoded, member):
    found = [
        f"pyc_instance_{index}"
        for index, (_actual, candidate, _typ) in enumerate(declarations(section, decoded))
        if candidate == member
    ]
    assert len(found) == 1, (member, found)
    return found[0]


def verilog_config(header, kind):
    classes = concrete_classes(header)
    decoded = decoded_classes(classes)
    root = "dut.dut"
    _root_type, section = class_for(
        classes, decoded, "slot_rule_mailbox" if kind == "mailbox" else "resident_issue"
    )

    def queue_image(path, depth, width, count_width):
        values = [f"{count_width}'({path}.count)"]
        values += [
            f"{width}'(({path}.count > {i}) ? {path}.storage[({path}.rd + {i}) % {depth}] : 0)"
            for i in range(depth)
        ]
        return "{" + ", ".join(values) + "}"

    if kind == "mailbox":
        fifo = [instance(section, decoded, q) for q in queues(section)]
        left_member = child(section, decoded, "explicit_mailbox")
        right_member = child(section, decoded, "nested_mailbox")
        left = root + "." + instance(section, decoded, left_member)
        right = root + "." + instance(section, decoded, right_member)
        _nested_type, nested_section = class_for(classes, decoded, "nested_mailbox")
        valid_member = child(nested_section, decoded, "valid_cell")
        payload_member = child(nested_section, decoded, "payload_cell")
        _valid_type, valid_section = class_for(classes, decoded, "valid_cell")
        _payload_type, payload_section = class_for(classes, decoded, "payload_cell")
        right_valid = (
            right
            + "."
            + instance(nested_section, decoded, valid_member)
            + "."
            + instance(valid_section, decoded, "pyc_q")
        )
        right_payload = (
            right
            + "."
            + instance(nested_section, decoded, payload_member)
            + "."
            + instance(payload_section, decoded, "pyc_q")
        )
        _explicit_type, explicit_section = class_for(classes, decoded, "explicit_mailbox")
        image = []
        control_wires = {}
        for side, path, incoming, outgoing in (
            ("left", left, fifo[0], fifo[2]),
            ("right", right, fifo[1], fifo[3]),
        ):
            image += [
                queue_image(f"{root}.{incoming}", 1, 8, 1),
                f"{right_valid if side == 'right' else path + '.' + instance(explicit_section, decoded, 'pyc_slot_valid')}.q_current",
                f"{right_payload if side == 'right' else path + '.' + instance(explicit_section, decoded, 'pyc_slot_payload')}.q_current",
                queue_image(f"{root}.{outgoing}", 1, 8, 1),
            ]
            for port, queue in (("in", incoming), ("out", outgoing)):
                for label, actual in (
                    ("ready", "in_ready"),
                    ("available", "out_valid"),
                    ("head", "out_data"),
                ):
                    control_wires[f"{side}_{port}_{label}"] = f"{root}.{queue}.{actual}"
            if side == "left":
                control_wires.update(
                    {
                        "left_capture": path + ".result[9]",
                        "left_publish": path + ".result[8]",
                        "left_publish_data": path + ".result[7:0]",
                    }
                )
            else:
                control_wires.update(
                    {
                        "right_capture": path + ".incoming_take",
                        "right_publish": path + ".outgoing_valid",
                        "right_publish_data": path + ".outgoing_data",
                    }
                )
        width = 54
    else:
        issue_member = child(section, decoded, "issue")
        root += "." + instance(section, decoded, issue_member)
        _issue_type, section = class_for(classes, decoded, "issue")
        fifo = [instance(section, decoded, q) for q in queues(section)]
        core_member = child(section, decoded, "issue_core")
        core = root + "." + instance(section, decoded, core_member)
        _core_type, core_section = class_for(classes, decoded, "issue_core")
        image = [
            queue_image(root + "." + fifo[1], 2, 9, 2),
            queue_image(root + "." + fifo[0], 2, 27, 2),
        ]
        image += [
            f"{core}.{instance(core_section, decoded, 'pyc_' + name)}.q_current"
            for name in (
                "wakeup_valid",
                "wakeup_payload",
                "allocation_valid",
                "allocation_payload",
            )
        ]
        image += [
            f"{core}.{instance(core_section, decoded, 'pyc_entries_' + str(i))}.q_current"
            for i in range(4)
        ]
        image += [queue_image(root + "." + fifo[2], 1, 27, 1)]
        names = (
            "allocation_ready",
            "allocation_available",
            "allocation_head",
            "wakeup_ready",
            "wakeup_available",
            "wakeup_head",
            "issued_ready",
            "issued_available",
            "issued_head",
            "allocation_capture",
            "wakeup_capture",
            "allocation_install",
            "wakeup_release",
            "selected",
            "publish",
            "publish_data",
        )
        control_wires = {name: root + "." + name for name in names}
        width = 250
    values = ", ".join(control_wires.values())
    labels = "".join(name + "=%0d;" for name in control_wires)
    return f"function automatic logic [{width - 1}:0] snapshot(); return {{{', '.join(image)}}}; endfunction\nfunction automatic string controls(); return $sformatf(\"{labels}\", {values}); endfunction"
