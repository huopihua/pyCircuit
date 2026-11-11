"""Header-only consumer for nominal struct construction and field reads."""

from pycircuit import u8 as Word
from .packet import Pair, Request


def default_request() -> Request:
    return Request(value=0, valid=0)


def keyword_request() -> Request:
    return Request(valid=1, value=7)


def read_value() -> Word:
    return Request(valid=1, value=9).value


def default_pair() -> Pair:
    return Pair(left=1, right=2)


def keyword_pair() -> Pair:
    return Pair(right=20, left=10)


def read_left() -> Word:
    return Pair(right=20, left=10).left
