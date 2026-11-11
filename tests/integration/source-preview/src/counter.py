# ruff: noqa: N802,F841
from pycircuit import log, module, report, rule, u8 as Word

from .types import CounterResult


@rule
def tick(count, incoming, outgoing) -> CounterResult:
    result = CounterResult(count=count)
    log("info", "counter_tick", outgoing)
    report("count", count)
    count = incoming
    return result


@module
def Counter(incoming: Word, outgoing: Word) -> CounterResult:
    count: Word = 0
    return tick(count, incoming, outgoing)
