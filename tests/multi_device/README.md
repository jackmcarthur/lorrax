# Multi-device gates (not in the default pytest suite)

Tier-2 device-count-invariance gate: runs the gnppm + bispinor e2e fixtures at
P=1 (1 GPU) and P=4 (4 GPUs, one process per device) and compares ζ / Σ_X /
minimax node counts / invalid census / off-pole eqp against the tolerances
from `reports/device_invariance_2026-07-08/ROOT_CAUSE.md` (lorrax_sandbox).

- `run_tier2.sh` — Perlmutter driver (needs `module load lorrax_X
  lorrax_agent` + a GPU allocation): `bash tests/multi_device/run_tier2.sh`.
- `eqp_invariance_cross_p.py compare <case> <p1_dir> <p4_dir>` — the
  launcher-agnostic compare step (tolerances + rationale in its docstring).

The sharp fixed-P pad-extent gate (Tier 1) IS in the default suite:
`tests/test_mu_pad_invariance.py`.
