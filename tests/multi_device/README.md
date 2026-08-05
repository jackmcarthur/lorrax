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

Σ_c(ω) layout A/B gate: `sigma_omega_layout_ab.py`, the launcher-agnostic
compare step for two full driver runs of the SAME deck that differ only in
`sigma_omega_layout = replicated | sharded`. Compares eqp0/eqp1.dat,
sigma_diag.dat, all four sigma_mnk.h5 tensors and (when present) the WFN_qp.h5
QP eigenvalues at 1e-12 relative, and checks that `sigma.host_gather` is
ELIDED under `sharded` rather than renamed. The layout had no regression
coverage before this (`grep -rln sigma_omega_layout tests/` returned nothing);
its only prior certification was two batch jobs named in commit 712a866.

    python3 sigma_omega_layout_ab.py compare <replicated_dir> <sharded_dir>

Harness: `/scratch2/08271/jackmc/omegacube_ab/{one_shot,sc}.sbatch` +
`ab_common.sh` (N=2, n=4, P=4, frozen source tree under the same directory).

Sharded-Σ_c rotation probe: `sigma_omega_rotate_probe.py`, any square P. Reads
the COMPILED (already SPMD-partitioned) HLO of a per-ω rotation of a
P(None,None,'x','y')-sharded Σ_c(ω,k,m,n) by a replicated (nk,nb,nb) W — the
contraction a same-basis `sigma_mnk.h5` / `WFN_qp.h5` write would need — and
reports the collective census and peak transient against the full cube. Four
formulations; a small-shape leg executes them all against a host reference.
Run at TWO mesh sizes: at P=4 four tiles and one cube are the same byte count.

    srun -n 4 python3 -m sigma_omega_rotate_probe   # SOR_NW/NK/NB

Harness: `/scratch2/08271/jackmc/omegacube_ab/rotprobe.sbatch` (N=8, legs at
P=4 and P=16, frozen tree). Green at job 7889790.
