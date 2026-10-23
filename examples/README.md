# Examples

Worked calculations with every input file, from DFT to LORRAX, beside the
BerkeleyGW inputs of the same calculation. Each directory's README lists the
run order and the exact commands. Decks name only the keys that differ from the
defaults ([input reference](../docs/input_reference.md)).

| example | system | shows |
|---|---|---|
| [`Si_soc/`](Si_soc/README.md) | bulk Si, spin–orbit (two-component spinors) | QE → BerkeleyGW and LORRAX side by side: G₀W₀ (HL plasmon pole), QP band structure, BSE |
| [`MoS2_SOC_2026/`](MoS2_SOC_2026/README.md) | monolayer MoS2, two-component SOC | pinned BerkeleyGW 2026 workshop download → parallel fully relativistic GW/BSE tracks, executed outputs and format checks |
