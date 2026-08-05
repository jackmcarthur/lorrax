# Multi-device gates (not in the default pytest suite)

Tier-2 device-count-invariance gate: runs the gnppm + bispinor e2e fixtures at
P=1 (1 GPU) and P=4 (4 GPUs, one process per device) and compares ζ / Σ_X /
minimax node counts / invalid census / off-pole eqp against the tolerances
from `reports/device_invariance_2026-07-08/ROOT_CAUSE.md` (lorrax_sandbox).

- `run_tier2.sh` — Perlmutter driver (needs `module load lorrax_X
  lorrax_agent` + a GPU allocation): `bash tests/multi_device/run_tier2.sh`.
- `eqp_invariance_cross_p.py compare <case> <p1_dir> <p4_dir>` — the
  launcher-agnostic compare step (tolerances + rationale in its docstring).

Padded-rank collective-write gate: `phdf5_padded_rank_write.py`, one case per
process launch (`PADRANK_CASE=repro|control|exact`), any square P>1. The
`repro` case makes at least one rank own a μ block that is entirely padding —
the shape that killed two 32-node bispinor legs on 2026-08-02 by rejecting an
empty selection before it could join the collective `H5Dwrite`. It needs a
real `liblorrax_ffi_host.so`, so it cannot run under plain pytest.

    PADRANK_CASE=repro srun -n 4 python3 -m phdf5_padded_rank_write

If it ever fails, read each rank's OWN stderr file: the writer's diagnostic is
lost from the merged log under srun+apptainer at teardown, which is precisely
why the original failure went unattributed for a day.

Sharded-U ψ rotation gate: `wfn_rotate_gate.py`, any square P (2×2 is the
standard leg). Certifies `gw.wavefunction_bundle.rotate_wavefunctions` against
the replicated-U path it replaces AND against an explicit host rotation with a
transposed-U negative control — the transpose of a unitary is also unitary and
also mixes only within the occupied block, so no invariance check can see it.
Also reports the per-rank U residency and a collective census (kind, result
bytes, replica-group size) of the compiled kernel, which is where the claim
"the band sum reduces along ONE mesh axis" is actually checked.

    srun -n 4 python3 -m wfn_rotate_gate      # WR_NK/NB/NS/NMU/NACT/NOCC

Harness: `/scratch2/08271/jackmc/wfn_rotate/wrgate.sbatch` (N=2, n=4, live
tree). Green at job 7889407.
