hBN 3x3x2 COHSEX regression fixture — the NON-CUBIC cell
=========================================================

**FROZEN 2026-08-07, OWNER-AUTHORIZED.**  `eqp_hbn_ref.dat` is the frozen
reference.  It is a byte copy of the candidate produced by the
reference-generation run of 2026-08-07/08
(`/pscratch/sd/j/jackm/svc_vcoul/hbn_fixture_prep/eqp_hbn_ref.candidate.dat`,
whole-file md5 `14035d12ca40a45e392b54528ee3c76c`, data-lines md5
`d4a7e4502a277e4aa203303042e792ec`).  The full 381-line evidence record is
`candidate.README` in that directory; the plan and its execution log are
`README_PLAN.md` here.

Why this fixture exists, in one paragraph
-----------------------------------------

Every other 3D end-to-end deck in this tree is bulk Si FCC, and Si FCC is
**provably blind** to the mini-BZ draw-convention bug class fixed in
`358bb0b`.  Si's reciprocal lattice satisfies `bvec.T = P·bvec` for a cyclic
signed permutation `P`, so transposing the draw is a pure *reseed* of the
Monte-Carlo head average — the wrong code and the right code differ only by
noise there, forever.  hBN's hexagonal `bvec` admits no such `P`, so the same
defect becomes a *bias* on this cell.  This deck therefore turns the synthetic
function-level z-test (`tests/test_vcoul_minibz_head_draw.py`) into end-to-end
pipeline coverage: WFN header → geometry → `build_v_head_miniBZ_avg_3d` →
head-slot injection → V_q → Σ.  As a bonus no other pinned deck provides, it
pins no `vhead`/`whead_0freq`, so the native q→0 head ladder
(`wcoul0_source s_tensor` → `gw.vcoul.compute_q0_averages` → the vcoul
kernels) runs end to end here and nowhere else.

### The precondition, MEASURED on this cell's real WFN header

Both step-0 requirements of the plan were checked on the actual `WFN.h5`
header (login node, pure numpy; logs `probe1_hbn.log`, `probe2_real_hbn.log`,
`probe2_builtin.log` in the prep directory):

1. **No signed permutation exists.**  All 48 signed permutations were searched
   for `bvec.T == P·bvec`; result `None`.  `max|bvec − bvec.T| = 7.666e-01`
   (zero would make the bug a no-op here).
2. **The pre-fix draw is a bias, not a reseed.**  Against the 12-seed
   corrected-rule band: single-seed `max z = 83.09 / mean z = 48.71`; on the
   noise-beaten-down mean-over-12-seeds bias test **`max z = 293.67 / mean
   z = 171.60`**, i.e. `1.61e-01 max / 8.27e-02 mean` relative error of the
   head — **55.8 % of the entire mc-average correction on this cell**.

The contrast is the whole argument:

| lattice | signed perm `P` | bias-test verdict |
|---|---|---|
| **Si FCC** (every other 3D gate) | **FOUND** → the bug is a reseed | max z = 3.03 — *consistent with noise* |
| **hBN (this fixture)** | **NONE** | **max z = 293.7 — BIAS** |
| hexagonal (MoS2-class, synthetic) | NONE | max z = 375.9 — BIAS |
| triclinic (synthetic) | NONE | max z = 71.2 — BIAS |

Fixture scale — READ THIS BEFORE QUOTING ANY NUMBER FROM IT
------------------------------------------------------------

**50 Ry / 80 spinor bands / 330 centroids on a 3x3x2 grid is FIXTURE SCALE,
NOT CONVERGED.**  This is a pinned, reproducible deck in the
`cohsex_si_fast.in` tradition, not a physics result.  It has **no external
anchor**: no BerkeleyGW ε+Σ run exists on this WFN, so nothing here says
LORRAX agrees with BerkeleyGW on hBN.  The suite's one external check remains
`test_si_production_matches_berkeleygw`.  Do not describe this fixture as an
anchor and do not cite its quasiparticle energies as hBN physics.

What is in this directory
-------------------------

| file | md5 | role |
|---|---|---|
| `WFN.h5` | `ff4f745bea20ee2892db0ea852452e06` | mean field, 67.6 MB (see "Why the binaries are tracked") |
| `kin_ion.h5` | `cb5e7f7c0276087c14f5a73a824941f4` | kinetic + ionic + V_H |
| `dipole.h5` | `a38de4bae19713cb1765570f2bc816c6` | dipole matrix elements |
| `centroids_frac_330.txt` | `2bbfee1219eec8091d269a876ba04eed` | the ISDF centroid set |
| `cohsex_hbn_test.in` | — | the deck (as run, minus the retired `output_file` key) |
| `eqp_hbn_ref.dat` | `14035d12ca40a45e392b54528ee3c76c` | **the frozen reference** |
| `qe/{scf,nscf,pw2bgw}.in` | — | the mean-field inputs AS RUN |
| `README_PLAN.md` | — | the plan + the 2026-08-07 execution record |

`vxc.dat` (md5 `e2b738da250028a8d560664988828bf8`) and `kih.dat` (md5
`61e87af77a531d4c749c992357590871`) are **not** shipped, exactly as
`si_cohsex_debug` does not ship them: they are inputs to the *generation* of
`kin_ion.h5` (`gw.kin_ion_io`), not to the deck.  They live in the prep
directory if `kin_ion.h5` ever needs regenerating.

### Why the binaries are tracked in git

Same reason `si_cohsex_debug`, `gnppm_debug` and `bispinor_debug` track
theirs.  The tree already carries a 55.1 MB `bispinor_debug/WFN.h5` and a
46.5 MB `gnppm_debug/WFN.h5`; 67.6 MB is the same class and is under the
100 MB hard limit.  Truncating a fixture WFN to save repo space is a move
this tree has already ruled against — `gnppm_debug/README.md`: *"WFN not
truncated — runtime reads only the requested window, truncation would only
shrink the repo at the cost of a new blob in history."*  Symlinking to
`/pscratch` is worse still: `harness.protect_fixtures` skips symlinks, so a
symlinked fixture would be the one unprotected fixture in the tree, and that
is exactly the shape of the 2026-07-25 incident it was written for.

Provenance — every number below is measured, not assumed
---------------------------------------------------------

| item | value |
|---|---|
| generated | 2026-08-07/08, Perlmutter |
| worktree at generation | `/pscratch/sd/j/jackm/svc_vcoul/lorrax`, detached at `a31ec236` |
| frozen at | `hbn/fixture-prep-2026-08-07`, on top of `main` @ `21d68e06` |
| **OWNER AUTHORIZATION** | **2026-08-07** — *"you can freeze hbn"* |
| allocations | `56478279` (`urgent_milan_ss11`, QE/BGW legs), `56478524` (`urgent_gpu_ss11`, all lorrax legs) |
| lx steps | scf `lx-Xg1-192833-526233-8030`, nscf `lx-Xg1-192858-528914-4147`, pw2bgw `lx-Xg1-192924-531464-2182`, wfn2hdf `lx-Xg1-192930-532052-2294`, kmeans `lx-Xg1-193022-536757-6689`, kin_ion `lx-Xg1-193119-540206-7244`, dipole `lx-Xg1-203630-800892-4731`, deck run1 `lx-Xg4-203742-806888-1087`, run2 `lx-Xg4-203837-812302-1290`, arm A `lx-Xg4-204012-818130-6083`, arm B `lx-Xg4-204157-827196-5211` |
| freeze-time 1-GPU reproduction | `lx-Xg1-224004-2009716-2551`, 32.8 s wall |

### Mean field

* **QE module**: `espresso/7.3.1-libxc-6.2.2-cpu`
  (`/global/common/software/nersc9/espresso/7.3.1-libxc-6.2.2-cpu/bin/{pw.x,pw2bgw.x}`)
* **BGW module**: `berkeleygw/4.0-gcc-12.3`.  The converter binary is
  **`wfn2hdf.x`, not `wfn2hdf5.x`**; usage `wfn2hdf.x BIN WFN WFN.h5`.
* **Pseudopotentials**: SG15 ONCV, fetched from
  `http://www.quantum-simulation.org/potentials/sg15_oncv/upf/` (none existed
  on the system).  Both 90153 bytes:

      B_ONCV_PBE-1.2.upf   md5 eb3cbeacc2d324d57d903df00f564d27   z_valence 3.00
      N_ONCV_PBE-1.2.upf   md5 c233fa8df51455e992258b7c3dee8a9f   z_valence 5.00

  → 16 valence electrons, which is what the deck assumes.

QE results: scf 6x6x4 converged in 10 iterations; nscf on the explicit
18-point 3x3x2 full grid (`nosym`/`noinv`), 80 spinor bands, zero
`c_bands ... not converged` lines.  **PBE gap 9.3628 − 4.7006 = 4.6622 eV**,
independently recomputed from the WFN.h5 eigenvalue array.  Header sanity,
all three plan requirements met:

    kgrid = [3, 3, 2]    nrk = 18    ntran = 1    nat = 4    mnband = 80
    nspin = 1  nspinor = 2   ifmax = [16 16 16 16]   ecutwfc = 50.0 Ry
    cos(b1,b2) = +0.4999999   angle 60.000004 deg   <- hexagonal
    cos(b1,b3) = cos(b2,b3) = 0                     <- and 90 deg to c

### The mean field is nspinor = 2, and that is not a choice

The plan's QE inputs described a scalar (nspinor=1) calculation.  **LORRAX
cannot read an nspinor=1 WFN.h5** — see "Registered defect" below — so the
mean field was regenerated the way every fixture WFN in this tree is made:
`noncolin = .true., lspinorb = .false., no_t_rev = .true.`, a 2-spinor
representation of a *non-relativistic* calculation, exactly as
`si_cohsex_debug`'s own QE inputs do it.  Every band count doubles
(`nbnd` 12→24 / 40→80, `vxc_diag_nmax` 40→80, deck `nval 8/ncond 32/nband 40`
→ `16/64/80`).  The physics is unchanged and that is checkable: scalar and
noncollinear SCF give the same eigenvalues (`4.7007 / 9.0101`) and the same
gap, and `bvec`/`kgrid`/`celvol` are bit-identical between the two WFNs, so
every step-0 probe number above holds under either.  The `qe/*.in` files in
this directory are the **as-run** ones.

### How the 330-centroid set was made — orbit-closed, 330 not 320

    python3 -u -m centroid.kmeans_cli 320 --qe-save <prep>/qe/hbn.save

run from the prep directory.  The generator recovered a **12-op symmorphic
point group from the charge density** (`[orbit] WFN stores 1 sym op(s)`,
because the nscf is `nosym`), unfolded 40 representatives to 468 distinct
centroids, and — selecting whole ORBITS, which is the shipping default —
picked **28 orbits → 330 unfolded centroids (orbit-closed)**, rank gate
`28/28 directions certified (floor 28, tol 0.01) — PASS`.

The request is 320; the number written is the number that survives orbit
unfolding, exactly as `centroids_frac_144.txt` is a request for 120 in
`si_cohsex_debug`.  Forcing a literal 320 would require `--no-orbit`, i.e.
reproducing the non-closure defect that costs `centroids_frac_960.txt` a
2.611 meV star spread.  **Closure wins over the round number**, and the
filename records the true count.

Other generator invocations (all exit 0):

    python3 -u -m gw.kin_ion_io -i cohsex_hbn_test.in --sys_dim 3
    python3 -u -m psp.get_dipole_mtxels -i cohsex_hbn_test.in

`kin_ion` reported `∫ρ d³r = 16.000000 e` (rel err 4.4e-16) and
`degeneracy-consistent — 539 degenerate manifolds, max split 0.0006 meV`.

Reproducibility — why the reference supports a tight pin
---------------------------------------------------------

**THREE independent runs, all byte-identical on the data lines**, at
data-lines md5 `d4a7e4502a277e4aa203303042e792ec`:

| run | mesh | processes | step | data-lines md5 |
|---|---|---|---|---|
| run1 (**the reference**) | 2x2 | 4 | `lx-Xg4-203742-806888-1087` | `d4a7e450…` |
| run2 | 2x2 | 4 | `lx-Xg4-203837-812302-1290` | `d4a7e450…` |
| freeze-time reproduction | 1x1 | 1 | `lx-Xg1-224004-2009716-2551` | `d4a7e450…` |

`delta_run2_vs_run1.txt`: every column MAE, max|Δ| and rms **exactly 0.000000
meV** over all 1440 rows.  The third run matters on its own: the pytest
harness pins **one** GPU per process (`tests/conftest.py::pytest_configure`),
so the gate runs this deck on a 1-device mesh while the reference was made on
a 2x2 one.  It was verified at the freeze that the two agree byte for byte,
so the gate is comparing like with like.  Whole-file md5s differ between runs
for exactly one reason — line 1 is
`# Generated by LORRAX unknown at <UTC timestamp>`.

### THE PERTURBATION ARMS — what the frozen digits mean

| arm | what moved | sigTOT MAE | sigTOT max\|Δ\| |
|---|---|---|---|
| **A** | `mc_average_vcoul_body` true → false | **13.995 meV** | **49.732 meV** |
| **B** | head-draw MC seed 42 → 43 | 0.396 meV | 1.127 meV |

Arm A per column: sigSX 7.836 / 41.797, sigCOH 17.217 / 50.333, sigTOT
13.995 / 49.732 meV; **`VH` and `Eo` exactly 0.000000**, which is the
expected signature — the knob acts through the Coulomb head table and not
through the mean field.  Arm B was a one-line *local, uncommitted* edit to
`services/vcoul/src/vcoul/minibz.py` (`seed: int = 42` → `43`), reverted from
a backup kept outside the worktree; `git status` was empty before and after.

**The knob this fixture exists to watch resolves at 35.4x the Monte-Carlo
seed width.**  That ratio is the fixture's whole justification, and it is why
the negative-control gate below can be pinned loosely and still be decisive.

The gates
---------

Both live in `tests/test_gw_jax_regression.py` and share nothing but the
fixture directory; each runs the deck once, fresh.

### `test_hbn_matches_frozen_reference` — the self-freeze, `atol = 1e-5 eV`

Compares kpt/band/sigSX/sigCOH/sigTOT/VH against `eqp_hbn_ref.dat`.

**Why 1e-5 eV.**  Byte-identity is the *primary* check and it is what three
independent runs deliver — `_assert_matches_reference` returns early on an
exact text match and reports that it did.  The numeric floor exists only to
absorb GPU-nondeterministic and cross-platform last-ULP drift in a text file
that prints **six decimals in eV**.  1e-6 was rejected deliberately: it is
the *exact* magnitude of the documented cross-machine ULP that forced the
2026-08-07 `_XMACHINE_ATOL_EV` ruling ("a 6-decimal `.dat` at `atol=1e-6` has
no room for a cross-platform ULP"), so pinning there would re-open a
ping-pong this tree has already closed.  1e-5 eV is therefore the same
constant the two Si freezes use, chosen for the same reason.

It is tight where tightness is the point.  **1e-5 eV = 0.01 meV is 40x
below the MC seed width of this deck (arm B, 0.396 meV MAE) and 1400x below
the knob the fixture guards (arm A, 13.995 meV MAE).**  Nothing this fixture
was built to see can hide under it, and a real code change that moves any
digit of the printed output fails the byte-identity path first.

### `test_hbn_mc_average_vcoul_body_moves_sigma` — THE NEGATIVE CONTROL

The cell this fixture exists for.  It reruns the deck with the single key
`mc_average_vcoul_body = false` and asserts sigTOT moves by **more than
5 meV MAE**.

A frozen-reference gate alone cannot make this fixture do its job.  A freeze
says "the numbers did not change"; it does not say "the head table is still
LIVE and still reachable on a non-cubic cell".  If the mini-BZ head average
were silently disconnected — the transpose-bug class, or any future edit that
routes around `build_v_head_miniBZ_avg_3d` — a re-freeze would happily pin
the wrong numbers and go green forever.  This cell is what makes that
impossible: it constructs the case where the check comes out FALSE.

**Why 5 meV and not 13.995.**  Measured effect 13.995 meV MAE; measured MC
seed width 0.396 meV MAE.  5 meV sits 12.6x above the seed noise (so seed or
default drift can never flake it) and 2.8x below the measured effect (so the
knob going dead cannot sneak past).  It is a liveness pin, not a value pin;
if you want the value, that is what the frozen reference is for.  It also
asserts `VH` is unmoved to 1e-5 eV — measured exactly 0.000000 — which is
the falsification that the effect is the Coulomb head and not a global
perturbation of the run.

Both cells skip, loudly and with a reason, if `WFN.h5` is missing from this
directory.  On a full checkout it never is; the guard exists so a partial
checkout says so instead of erroring in the driver.

Registered defect (NOT fixed here)
----------------------------------

**LORRAX cannot read an nspinor = 1 WFN.h5.**  `WfnLoader._eager_build` sizes
the ψ slab with the file's `ns = 1` (`services/wfn_loader/src/wfn_loader/loader.py:1474`)
but the full-BZ unfold calls `symmetry_maps.unfold_psi`, whose spinor
rotation `U_eff` is unconditionally 2x2 (`services/symmetry_maps/src/symmetry_maps/maps.py:907`;
`SymMaps.U_spinor` is `np.eye(2)[None,:,:]`).  numpy broadcasts the size-1
spinor axis, so the rotated block returns `(nb, 2, ngk)` and the slab write
raises

    ValueError: could not broadcast input array from shape (8,2,1457)
                into shape (8,1,1457)

The docstring at `maps.py:855` **falsely** claims the ns=1 case is a no-op.
Consequence: every fixture in the tree is nspinor=2, and any scalar-DFT WFN
from a standard QE run is unreadable — a silent onboarding trap.  Evidence
preserved at
`/pscratch/sd/j/jackm/svc_vcoul/hbn_fixture_prep/qe_scalar_nspinor1_ATTEMPT/`
(+ `WFN_nspinor1_ATTEMPT.h5`).  Owned by the wfn_loader / symmetry_maps
services; registered, not touched.

What this fixture still needs
-----------------------------

* **An external anchor.**  Phase 2 of the plan — BerkeleyGW ε+Σ on this same
  WFN with `cell_average_cutoff` pinned — was never run.  It would give this
  deck a `bgw_sigma_hp_*.dat` of its own *and* head scalars for a
  `vhead`-pinned variant deck beside this one.  **Do not convert this deck to
  a `vhead`-pinned one**: its unpinned q→0 ladder is unique coverage.
* The nspinor=1 loader defect above.

Reproducing by hand
-------------------

```bash
cp -r tests/regression/hbn_cohsex_debug /tmp/hbn && chmod -R u+w /tmp/hbn && cd /tmp/hbn
python -m gw.gw_jax -i cohsex_hbn_test.in          # ~33 s on 1 A100
diff <(tail -n +2 eqp_hbn_test.dat) <(tail -n +2 eqp_hbn_ref.dat)   # expect empty
```

Copy, never symlink — the driver writes its outputs into the run directory,
and a write through a stray symlink destroyed a checked-in fixture on
2026-07-25.  That is why `harness.protect_fixtures` keeps everything here
`a-w` at rest.
