# Multi-device gates (not in the default pytest suite)

Tier-2 device-count-invariance gate: runs the gnppm + bispinor e2e fixtures at
P=1 (1 GPU) and P=4 (4 GPUs, one process per device) and compares ζ / Σ_X /
minimax node counts / invalid census / off-pole eqp against the tolerances
from `reports/device_invariance_2026-07-08/ROOT_CAUSE.md`, committed in this
repo since fdc48ae.  It was previously attributed to lorrax_sandbox, which
never held it — the set lived untracked in a sibling checkout.

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

Band-rotation collective probe: `rot_probe.py`, any square P. Answers one
question — does the two-step constraint remove the global all-reduce in
`gw.qsgw_density.rotate_bands`? — by printing every collective in the LOWERED
HLO with its byte size, so the 3.36 GiB `c128[nk,nb,ns,ngkmax]` all-reduce is
visible if it is still there. Shapes come from `RP_NK`/`RP_NB`/`RP_NS`/`RP_NGK`
(defaults 16/640/2/11008); the arrays are zero-filled, so this is a compile-time
instrument and costs no real work.

    srun -n 4 python3 -m rot_probe        # RP_NK/NB/NS/NGK

It is the DIAGNOSTIC counterpart to the gate below: the probe shows you which
collectives a spelling emits, the gate pins that the spelling is correct. Added
to this roster 2026-08-09 — it had been in the directory since `7887e61d`
(2026-08-05) and was the one file here that no README row and no `run_tier2.sh`
leg mentioned, which is the only reason a completeness audit found it rather
than a reader. Its imports still resolve, so it is runnable as written.

Band-rotation primitive gate: `band_rotate_gate.py`, any square P (1, 2×2 and
4×4 are the standard legs). Certifies `gw.qsgw_density.rotate_band_axis` /
`rotate_band_matrix` — the one rotation primitive, `U` kept at
`band_rotation_spec` — against the replicated-U kernel it replaces at the two
matrix call sites (`sc_iteration._rotate_to_dft_basis` and `sigma_dispatch`'s
V_H basis change), in both directions, plus an explicit host rotation. Two
negative controls: the transposed reference — `U A U†` and `U† A U` share
hermiticity, trace and Frobenius norm, so no invariance check can separate
them — and a `conj_u` that is NOT flipped between the two axis rotations, which
does break hermiticity but which nothing in the pipeline looks at.
Also reports per-rank U residency, module argument/temp/output bytes and a
collective census of both compiled modules, and pins `rotate_bands` (which now
routes through the same primitive) bit-identical to its old spelling.

    srun -n 4 python3 -m band_rotate_gate     # BR_NK/NB/NS/NG

Harness: `/scratch2/08271/jackmc/bandrot/brgate.sbatch` (N=4, legs at P=1, 4,
16, frozen tree). Green at job 7889851: worst delta against the replicated-U
kernel 5.4e-16 relative, per-rank U 0.1250 → 0.0312 (2×2) → 0.0078 MiB (4×4).

`restart_q_storage` full-vs-wedge A/B: `restart_q_storage_ab.sh` (the driver)
and `restart_q_storage_ab.py compare <full_dir> <auto_dir>` (the
launcher-agnostic compare step). This is the gate `4e8cfd70` names as its own,
and the reason it is here rather than in the pytest suite is that SlabIO needs
the phdf5 FFI `.so` pair: every restart-writer cell in the tree is red on a box
without it, so until this ran the q_irr format's BYTES had never been measured
anywhere. Three arms off the committed `si_bse_debug` fixture, differing by one
deck line each — `restart_q_storage = full` (the control; it never asks the
closure question, so it writes the bytes the deck wrote before the format
existed), `= auto` (which resolves to the IBZ q wedge on this deck, whose
centroid set became orbit-closed at `fb046e0c`), and `write_restart_tensors =
false` (the `67eda567` suppress key, whose default is an open owner decision).

    LXRUN=/path/to/lxrun-wrapper bash tests/multi_device/restart_q_storage_ab.sh

Two things about it are not obvious and both are load-bearing. **The physics
A/B runs at P=1 on both arms**, because the sharded reader REFUSES a wedge file
by design (`bse_io._MunuSlabPlan` — a per-rank (μ,ν) hyperslab cannot unfold,
since the unfold gathers across the very axes it shards on) and only the serial
h5py reader unfolds; the P=4 leg runs anyway, on `full` because that is the
configuration the frozen reference was cut at and on `auto` because the refusal
is itself an assertion. **The frozen `bse_eigenvalues_ref.dat` is a sanity line
and not the criterion**, because it is pinned to an unconverged Lanczos spectrum
that moves 7.32 meV over 200→400 iterations and 5.51 meV over 1→4 processes with
no code change at all (`NOTE_vcoul_head_refreeze.md`); only a matched A/B cancels
that, which is why every solver setting is held fixed across the arms.

Green at lx job 56499811 (1 node × 4 A100, branch
`svc/symmetry_maps-followup-2026-08-08` @ 54d25712, BUILD_NOTES merge_ckpt `.so`
pair). `auto` resolved to the wedge at a worst closure residual of 4.596e-16;
V_qmunu and W0_qmunu went (64,480,480) → (8,480,480) and the restart file
541 335 584 → 130 297 888 B (4.155×, not 8×, because `psi_full_y` does not
shrink); `unfold(wedge)` came back **bit-identical** to the full-BZ arrays on
every dataset; the BSE eigenvalues agreed at **max |Δ| = 0.000000 meV over all
20 states**; and the `full` arm at P=4 reproduced the `fb046e0c` re-cut
reference **bit-identically** (max |Δ| = 0.000e+00 eV), which is what proves the
deck key's `full` path inert. Walls over 5 round-robin reps: the `persist_w0`
timer goes 0.325 → 0.096 → 0.000 s across the three arms, while the TOTAL step
does not resolve either change on this deck (0.695 s of separation against
~1.0 s of scatter). The numbers and how to read them are in the driver's header.

One corollary fell out of that same run and belongs with the reference rather
than with this script: `tests/test_bse_bgw_regression.py::test_bse_matches_frozen_and_bgw`
fails on Perlmutter at max |Δ| = 4.4887 meV, 12 of 20 cells over its 1e-6 eV
pin — and fails **identically** with `restart_q_storage = full` pinned in the
deck. Run at the gate's own configuration (one GPU, px=py=2) the two arms are
not merely equally wrong but `array_equal`: max |Δ| = 0.000000 meV between them,
both 4.4887 meV from the reference. So it is neither the wedge nor anything
else on this branch. `conftest.py`
pins the pytest process to one GPU, while `fb046e0c` cut the reference from two
4-process script runs and recorded, as its own honest limit, that it could not
run the gate; the reference has never been seen by the gate that pins it. The
campaign leaves it re-frozen by nobody and measured at both ends — 0.000 meV at
P=4, 4.4887 meV at P=1, same binary and same restart file — so the re-cut
decision, which is the owner's, can be made on numbers.
