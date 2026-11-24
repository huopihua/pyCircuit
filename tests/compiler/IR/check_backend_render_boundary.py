#!/usr/bin/env python3
# ruff: noqa: I001
"""Reject IR traversal and source semantics in the pure render layer."""

import argparse
import sys
from pathlib import Path


FORBIDDEN = (
    "mlir::Operation",
    "mlir::Value",
    "mlir::Attribute",
    "ModuleOp",
    "HardwareAnalysis",
    "DictionaryAttr",
    "FlatSymbolRefAttr",
    "getBody(",
    "walk(",
    "dyn_cast<",
    "cast<ac::",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    files = (
        args.root / "compiler/lib/Compiler/BackendRender.cpp",
        args.root / "compiler/lib/Compiler/FinalCppSourceParts.h",
        args.root / "compiler/lib/Compiler/FinalVerilogSourceParts.h",
    )
    violations: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            for token in FORBIDDEN:
                if token in line:
                    violations.append(
                        f"{path.relative_to(args.root)}:{line_number}: "
                        f"pure render boundary forbids {token!r}"
                    )
    if violations:
        sys.stderr.write("\n".join(violations) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
