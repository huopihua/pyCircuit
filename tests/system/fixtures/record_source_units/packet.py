"""Approved nominal struct forms for real source-unit/header tests."""

from pycircuit import struct, u1, u8


@struct
class Request:
    value: u8
    valid: u1


@struct
class Pair:
    left: u8
    right: u8


@struct
class Required:
    value: u8


@struct
class Modes:
    positional: u8
    keyword: u8
