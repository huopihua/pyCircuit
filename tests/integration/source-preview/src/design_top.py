# ruff: noqa: N802,F841
from pycircuit import rule, system, u8 as Word

from .counter import Counter


@rule
def transfer_outputs(left_output, right_output, left_count, right_count) -> None:
    left_output = left_count
    right_output = right_count


@system
def DesignTop():
    left_input: Word = 3
    right_input: Word = 10
    left_output: Word = 100
    right_output: Word = 200

    left = Counter(left_input, left_output)
    right = Counter(right_input, right_output)
    transfer_outputs(left_output, right_output, left.count, right.count)
