# Issue #431 compiler baseline

Batch 0 establishes measurement and regression gates before changing the
compiler's semantic plans. Metrics are diagnostic evidence only: they do not
enter IR, change admission, or raise a budget.

The locked reproducer is `benchmarks/issue-431/wse40.lock.json`. It fixes the
consumer repository and revision, the materialized timed-machine source digest,
40 nodes, 2 early slots, 56 seen/dedup slots, a 64-byte beat, `O1`, and build
parallelism 1. `tools/measure_issue_431.py` validates the lock by default;
fetching and running require explicit flags and user-selected cache/output
directories. A run requests both C++ and Verilog emission.

## Phase report

`pycircuit compile`, `pycircuit link`, and `pycircuit emit` accept
`--metrics-out PATH`. The fresh JSON report records import, transaction
planning, routing, independent proof, final verification, and emission
planning. Each phase can report Work, Slots, Occurrences, definition/call/
instance counts, IR walks and visited nodes, plan entries and estimated bytes,
status, and wall time. Compile/link/C++ emit/Verilog emit timings are also
carried at report level. Failed invocations preserve a partial report whenever
the native helper reached the metrics channel.

Current fixed budget ceilings remain unchanged:

- import table slots: 262,144;
- transaction planning: 1,048,576 work units;
- routing/replay: 1,048,576 work units;
- native validation: 4,194,304 work units.

The issue report's routing experiment measured 360,478 preparation units,
407,522 generation units, and 912,737 independent-proof units: 1,680,737 total.
Its transaction baseline was 4,817,601 units and the prototype was 766,032.
Those experiments may overlap and must not be added as if they were disjoint.

## Regression gates

The native emitter test prepares one verified context once and renders both
C++ and Verilog from it, checking source-check count and source-owner inventory.
The example catalog gate requires exactly 14 published generated receipts and
requires every receipt's verification input set to include both
`build/cpp/generated.json` and `build/verilog/generated.json`.

Pass `--rtl-closure` to compile and execute the locked 40-node RTL with
Verilator. Its receipt records the backend, compile time, simulation time,
whether simulation actually ran, and the number of independently checked
scenarios. Icarus remains the four-state backend for the small collection and
managed-memory gates; the WSE closure does not raise its elaboration timeout.
