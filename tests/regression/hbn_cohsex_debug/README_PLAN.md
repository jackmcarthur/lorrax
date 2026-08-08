# hBN 3x3x2 — the non-cubic e2e fixture: groundwork + reference-generation plan

Prepared 2026-08-07 by the vcoul orchestrator (registered row 8 of the vcoul
land-readiness report).  NOTHING here is frozen; freezing a reference is the
owner's act.  This directory is the runnable groundwork: QE inputs, the
LORRAX deck, and the plan below.  Work area for the runs:
`/pscratch/sd/j/jackm/svc_vcoul/hbn_fixture_prep/` (copy of this dir).

## Why, in one paragraph

Every 3D gate fixture is Si FCC, and Si FCC is PROVABLY blind to the
mini-BZ draw-convention bug class (`bvec.T = P·bvec`, P cyclic ⇒ the
transposed draw is a pure reseed — the algebra behind the 358bb0b fix).
The synthetic hexagonal z-test (`tests/test_vcoul_minibz_head_draw.py`) is
the only guard today, and it tests the FUNCTION, not the pipeline.  A small
hexagonal e2e deck makes the whole chain — WFN header → geometry →
`build_v_head_miniBZ_avg_3d` → head-slot injection → V_q → Σ — sensitive to
the bug class, plus (a bonus no current deck provides) it exercises the
native q→0 head ladder end-to-end, because it pins no `vhead`.

## System choice, and what was rejected

**Bulk hexagonal BN (AA', P6₃/mmc), 4 atoms, a=2.504 Å, c=6.661 Å,
kgrid 3×3×2 (18 k full BZ, nosym), 40 bands, ecutwfc 50 Ry, scalar ONCV
pseudos (B_ONCV_PBE-1.2, N_ONCV_PBE-1.2 — SG15).**  Rationale: the
smallest wide-gap hexagonal BULK crystal with light elements and no SOC —
4 atoms / 16 valence electrons vs 6 atoms + semicore d for 2H-MoS₂ bulk;
wurtzite AlN is comparable in size but heavier in cutoff; graphite is a
semimetal (bad COHSEX fixture).  hBN is also a standard GW benchmark, so a
future BGW anchor phase has literature to lean on.  Fixture philosophy is
the `cohsex_si_fast` one: pinned and reproducible, NOT converged — 50 Ry /
40 bands / 320 centroids are fixture-scale, and the README that freezes the
reference must say so.

Bias-sensitivity is a precondition, not an assumption: step 0 of the run
plan re-runs the signed-permutation search on the REAL WFN header and the
head-table A/B at production nmc.  On the synthetic hBN-class cell the
shipped-vs-fixed delta was ~6.5 % mean / 16 % max of the head (z=376); the
real cell must reproduce that class or the fixture choice is wrong.

## Run plan (Perlmutter; QE from the modules the sandbox MoS2 workflow used)

0. **Sensitivity precheck** (login node, pure numpy, ~seconds): after step
   3 produces WFN.h5, run the probe pair from the survey
   (`/pscratch/sd/j/jackm/tmp_vcoul_survey/vcoul_bug_probe{,2}.py`) on it.
   REQUIRED: no signed permutation P with `bvec.T = P·bvec`; seed-band z in
   the tens-to-hundreds.  Record both numbers in the fixture README.
1. **Pseudos**: fetch `B_ONCV_PBE-1.2.upf` / `N_ONCV_PBE-1.2.upf` (SG15)
   into `qe/`; record md5s.
2. **QE**: `pw.x < scf.in` (6×6×4 shifted-free SCF), then `pw.x < nscf.in`
   — nscf is `nosym=.true., noinv=.true.` on the EXPLICIT full 3×3×2 grid,
   40 bands, matching the Si fixture's nosym-full-BZ convention.
   Minutes on one CPU node.
3. **Convert**: `pw2bgw.x < pw2bgw.in` (WFN + vxc.dat + kih.dat, 1..40),
   then BGW `wfn2hdf5.x WFN WFN.h5`.  Sanity: `mf_header/kpoints/kgrid` =
   [3,3,2], 18 k, `mf_header/symmetry/ntran` = 1.
4. **LORRAX prerequisites** (mirror what `si_cohsex_debug` carries):
   `kin_ion.h5` via the `gw.kin_ion_io` CLI against WFN.h5 + kih.dat;
   `dipole.h5` via the dipole tool the Si fixture used (locate its exact
   invocation in the Si sandbox run dirs at generation time — the tool
   names live in `runs/Si/08_*/`); `centroids_frac_320.txt` via the kmeans
   centroid tool at 320 points (=8×40, Si-fast-deck ratio; orbit-closure
   note: prefer an orbit-closed set — the Si production 960 set's
   non-closure is a known 2.6 meV star-spread defect, do not replicate).
5. **The deck**: `cohsex_hbn_test.in` (this dir) on 4 GPUs via `lx run`,
   BUILD_NOTES pins.  Two runs for bit-reproducibility (byte-diff of
   eqp_hbn_test.dat data lines).
6. **Perturbation arms** (each one deck-key flip, each must move σ):
   `mc_average_vcoul_body` true↔false — the head table's liveness, e2e, on
   a cell where the average is bias-sensitive; and (cheap, high-value)
   re-run at one different `build_v_head_miniBZ_avg_3d` seed via a
   one-line local patch to measure the MC seed-width of σ — that number
   calibrates the frozen reference's meaningful digits.
7. **Candidate**: `eqp_hbn_ref.candidate.dat` + a README recording every
   number above.  STOP.  Adoption/freezing = owner.  Optional phase 2
   (listed, not planned): BGW ε+Σ on the same WFN with
   `cell_average_cutoff` pinned, giving this fixture an external anchor
   and head scalars for a `vhead`-pinned variant deck.

## Cost

QE ≈ minutes (1 CPU node); conversions trivial; each LORRAX run ≈ Si-gate
scale (tens of seconds on 4 GPUs); total including arms < 1 GPU-hour.

## 2026-08-07 — EXECUTED through step 8; candidate produced (NOT frozen)

All artifacts: `/pscratch/sd/j/jackm/svc_vcoul/hbn_fixture_prep/` —
`eqp_hbn_ref.candidate.dat` (md5 `14035d12ca40a45e392b54528ee3c76c`; data
lines `d4a7e450…`, byte-identical across run1/run2) + `candidate.README`
(381 lines: pins, jobids, pseudo md5s, all numbers).  Key evidence,
verified independently against the artifacts by the orchestrator:

- Step-0 sensitivity, REAL WFN header: no signed permutation exists;
  pre-fix draw bias = 8.27e-2 mean / 1.61e-1 max rel of the head =
  55.8 % of the whole mc-average correction, z = 293.7 (Si contrast:
  z = 3.0 = noise).  The cell guards the bug class as designed.
- PBE gap 4.66 eV; 18 k, ntran = 1, kgrid [3,3,2] confirmed in the header.
- Bit-reproducibility: two full runs, every sigma column 0.000000 meV.
- Perturbation arms: `mc_average_vcoul_body` true→false moves sigTOT by
  MAE 13.995 / max 49.732 meV; the MC seed width (seed 42→43) is
  MAE 0.396 / max 1.127 meV — the knob resolves at 35.4× the seed noise.
  On Si this knob's absence was invisible to the draw-bug class; here it
  is decisively live.
- Deviations, accepted and documented in the deck: nspinor=2
  regeneration (see the registered defect below); 330 orbit-closed
  centroids (28 orbits, rank gate 28/28) instead of the literal 320;
  launcher `-G 4 -n 4`; the retired `output_file` key dropped.

ADOPTION/FREEZING REMAINS THE OWNER'S ACT: copy the candidate in as
`eqp_hbn_ref.dat`, add WFN.h5/kin_ion.h5/dipole.h5/centroids from the
prep area, and wire the gate.

## REGISTERED DEFECT (wfn_loader / symmetry_maps seam — NOT fixed here)

LORRAX cannot read an nspinor=1 WFN.h5: `WfnLoader._eager_build`
allocates the ψ slab with the file's ns=1 (`loader.py:1474`) but the
full-BZ unfold calls `symmetry_maps.unfold_psi`, whose spinor rotation
`U_eff` is unconditionally 2×2 (`maps.py:907`), so the rotated block
comes back (nb, 2, ngk) and the slab write raises
`ValueError: could not broadcast (8,2,1457) into (8,1,1457)`.
`maps.py:855`'s docstring FALSELY claims the ns=1 case is a no-op.
Evidence preserved: the failed attempt at
`/pscratch/sd/j/jackm/svc_vcoul/hbn_fixture_prep/qe_scalar_nspinor1_ATTEMPT/`
(+ `WFN_nspinor1_ATTEMPT.h5`).  Consequence: every fixture in the tree
is nspinor=2, and any scalar-DFT WFN produced by a standard QE run is
unreadable — a silent onboarding trap.  Owned by the wfn_loader /
symmetry_maps services; registered, not touched, per file-ownership
ruling 1.
