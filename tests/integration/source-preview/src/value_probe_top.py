# ruff: noqa: N802,F841,T201
from pycircuit import log, rule, system, u8 as Word


@system
def ValueProbeTop():
    word: Word = 7
    flag: bool = True

    @rule
    def observe_values() -> None:
        log("info", "word", word)
        log("info", "flag", flag)

    observe_values()
