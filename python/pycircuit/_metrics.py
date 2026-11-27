"""Publication-neutral compiler metrics evidence."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path


def publish_metrics(source: Path, destination: str | Path | None) -> None:
    """Copy one native report to a fresh ordinary file outside artifact ownership."""

    if destination is None:
        return
    if not source.is_file() or source.is_symlink():
        raise ValueError("native compiler did not produce its metrics report")
    payload = source.read_bytes()
    report = json.loads(payload)
    if not isinstance(report, dict) or report.get("schema") != (
        "pycircuit-compiler-phase-metrics-v1"
    ):
        raise ValueError("native compiler metrics report has an unknown schema")
    path = Path(destination)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("metrics output must be an ordinary file")
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
