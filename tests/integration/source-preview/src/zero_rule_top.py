# ruff: noqa: N802,F841
from pycircuit import system, u8 as Word


@system
def ZeroRuleTop():
    retained: Word = 9
