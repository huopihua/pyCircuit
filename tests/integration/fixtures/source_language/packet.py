"""Approved nominal record source; compiler input, never eagerly imported."""

from typing import Annotated

from pycircuit import struct, u1, u8

Word = Annotated[int, range(256)]


@struct
class Request:
    value: u8
    valid: u1


@struct
class ProbeResult:
    value: u8
