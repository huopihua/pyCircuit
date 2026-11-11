from pycircuit import struct, u8


@struct
class CounterResult:
    count: u8
