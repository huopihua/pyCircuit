# ValueProbeTop

This portless, no-write probe emits separate one-value observations for a
finite integer state initialized to 7 and a boolean state initialized to true.
The system test mutates the authoritative u8 initializer in a saved valid final
snapshot and verifies that both independent-HWIR backends retain distinct
integer and boolean JSON kinds.
