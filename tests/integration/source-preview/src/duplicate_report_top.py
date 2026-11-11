# ruff: noqa: N802,F841
from pycircuit import report, rule, system, u8 as Word


@system
def DuplicateReportTop():
    value: Word = 7

    @rule
    def observe_duplicate_report() -> None:
        report("same", value)
        report("same", value)

    observe_duplicate_report()
