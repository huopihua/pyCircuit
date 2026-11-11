# ruff: noqa: N802,F841
from pycircuit import report, rule, system, u8 as Word


@system
def ReportProbeTop():
    value: Word = 7

    @rule
    def observe_reports() -> None:
        report("first", value)
        report("second", value)

    observe_reports()
