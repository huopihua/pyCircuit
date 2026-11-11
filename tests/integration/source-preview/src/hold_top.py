# ruff: noqa: N802,F841
from pycircuit import rule, system, u8 as Word


@system
def HoldTop():
    held: Word = 7

    @rule
    def observe_without_write() -> None:
        return

    observe_without_write()
