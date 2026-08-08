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
