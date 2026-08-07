Si 4x4x4 COHSEX regression fixture — TWO runs, one directory
=============================================================

This directory holds the project's only `sys_dim = 3` end-to-end COHSEX case
and the project's only EXTERNAL check against BerkeleyGW.  It ships **two
decks over one set of inputs**:

| | `cohsex_si_test.in` | `cohsex_si_fast.in` |
|---|---|---|
| role | **PRODUCTION — the BerkeleyGW anchor** | **TEST — a fast self-freeze** |
| bands | 60 (nval 8, ncond 52) | 20 (nval 8, ncond 12) |
| centroids | `centroids_frac_960.txt` | `centroids_frac_144.txt` |
| wall time | 34.6 s cold / 28 s in pytest | 15.7 s cold / ~12 s warm |
| own reference | `eqp_si_ref.dat` | `eqp_si_fast_ref.dat` *(not frozen yet)* |
| BGW reference | `bgw_sigma_hp_noavg.dat` | **none — see below** |
| agreement with BGW | sigTOT **0.64 meV** MAE | sigTOT **2109 meV** MAE |
| gates it feeds | `test_si_production_matches_frozen_reference`, `test_si_production_matches_berkeleygw` | `test_si_fast_matches_frozen_reference` |

Both run bulk Si on a 4x4x4 grid with all 48 symmetry operations, from the same
`WFN.h5` (8 IBZ k-points; Sigma is evaluated full-BZ-direct on all 64).  They
differ in exactly two knobs — band count and centroid count — so an A/B between
them is interpretable.

**One directory, not two.**  `WFN.h5` (9 MB), `kin_ion.h5` (3.7 MB) and
`dipole.h5` (12.9 MB) are 26 MB of shared input.  A second fixture directory
would have to duplicate or symlink them; symlinks are specifically what
`harness.protect_fixtures` exists to defend against (it skips symlinks, so a
symlinked fixture would be unprotected).  Two decks in one directory is the
consolidation.

Shared inputs: `WFN.h5`, `kin_ion.h5`, `dipole.h5`
(nspinor = 2, nrk = 8, mnband = 62, ng = 4573, **ecutwfc = 25.0 Ry**, ifmax = 8,
FFT grid 24x24x24, ntran = 48).

> The `100` in the file is `ecutrho`, not the wavefunction cutoff.  The deck's
> `bare_coulomb_cutoff = 25.0` matches `ecutwfc`.  Because `ecutwfc` is already
> 25 Ry there is nothing to gain from a smaller-cutoff calculation, and the fast
> deck needs **no new Quantum ESPRESSO run** — it reuses this WFN.  MEASURED:
> WFN load is 0.138 s of a 34.6 s production run, so it is not a cost worth
> optimising.


PRODUCTION deck — `cohsex_si_test.in`
-------------------------------------

This is the anchor.  **Do not shrink or re-freeze it.**  If you want a quick Si
run, use the fast deck; that is what it is for.

It is anchored to BerkeleyGW run `D_bgw_cohsex_noavg`
(`/pscratch/sd/j/jackm/lorrax_sandbox_pre_august/runs/Si/06_si_4x4x4_nosoc/`),
whose Sigma columns are committed here as `bgw_sigma_hp_noavg.dat` and gated by
`test_si_production_matches_berkeleygw`.

### MEASURED agreement with BerkeleyGW

2026-08-07, tree at HEAD `04b8bba`, single GPU, jax 0.7 image
(`LX_BASE_MODULE=lorrax_J070`), FFI pinned from `lorrax_P`.  Deviations in meV:

| column | IBZ only (8 k x 16 bands) | full BZ (64 k x 16 bands) |
|---|---|---|
| | MAE / max | MAE / max |
| bare `x_bare` vs BGW `X` | 0.351 / 1.122 | 0.330 / 1.139 |
| `sigSX` vs `X + (SX-X)` | 0.151 / 0.303 | 0.142 / 0.397 |
| `sigCOH` vs `CH'` | 0.351 / 1.214 | 0.587 / 3.104 |
| `sigTOT` vs `Sig'` | 0.433 / 1.253 | 0.644 / 2.945 |

The gate uses the **full-BZ** numbers.  The IBZ-only column is shown because it
is what earlier reports quoted, and because the two differ — see "Known
defects" below.

> **The 0.12 meV figure that used to appear here is retired.**  It came from
> `reports/cohsex_si_444_gamma_agreement_2026-05-02/`, which used a full BGW
> `vcoul` body overlay (a 185 MB dump that no longer exists on disk).  This
> fixture uses LORRAX's native finite-q Coulomb body plus BGW's q->0 head
> injected as two scalars, and its measured agreement is the table above.  Do
> not quote 0.12 meV for this configuration.

### Column conventions — read before comparing anything

BerkeleyGW's 14-column `sigma_hp.log` block
(`Sigma/write_result_hp.f90:88-100`) writes

    CH  = ach + achcor        Sig  = asig + achcor
    CH' = ach                 Sig' = asig

where `achcor` is the **static remainder** correction.  LORRAX computes no
static remainder, so the comparable BGW columns are the **primed** ones:

    LORRAX sigSX   ==  X + (SX-X)
    LORRAX sigCOH  ==  CH'          (NOT CH — they differ by ~367 meV)
    LORRAX sigTOT  ==  Sig'         (== X + (SX-X) + CH', identically)

No offset is applied to either side.  Earlier notes here spoke of "rigid
per-column offsets from BGW's self-energy-column conventions"; that was an
artifact of comparing against the unprimed columns.  Against the primed ones
there is no offset to explain away.

### `mc_average_vcoul_body = false` is a MATCHING CONVENTION

`D_bgw_cohsex_noavg` runs BGW with `cell_average_cutoff 1d-12`, under which BGW
cell-averages **only** the literal q+G=0 element and uses the point
8*pi/|q+G|^2 everywhere else (`Common/vcoul_generator.f90:101-103`).  LORRAX's
default (`true`) MC-averages v(q, G=0) at **every** q != 0 — one q-shell too
many.  MEASURED, single-variable A/B over 128 (k,band) pairs:

    mc_average_vcoul_body = true    sigX MAE 136.202 meV   max 282.961
    mc_average_vcoul_body = false   sigX MAE   0.351 meV   max   1.122

Pair BGW default (`avgcut = 1e12`) with `true`; BGW `noavg` with `false`.  This
fixture is anchored to noavg, so it sets `false`.  This is the single largest
lever in the whole fixture and the gate's main job is to keep it correct.

### `zeta_rcond = 1e-10` is pinned, deliberately

The default moved 1e-10 -> 1e-6 -> 1e-8 *after* this reference was frozen.  Si
4x4x4 at 960 centroids is NOT rank-truncation-free — its charge CCT spectrum
extends below the cut — so the value has to be pinned or the gate drifts with a
default.  Measured drift of sigTOT vs the 1e-10 freeze
(`reports/gw_conduction_postfix_2026-07-21`, `si_rcond_sweep.sh`):

    rcond    1e-9    1e-8    1e-7    1e-6    1e-5    1e-4
    max|d|  0.001   0.054   0.417   1.021   2.918  37.218   meV

(The fast deck is different — at 144 centroids it is measurably insensitive and
leaves the key unset.  See its header.)


TEST deck — `cohsex_si_fast.in`
-------------------------------

A pure self-freeze for fast iteration.  **It is not a BerkeleyGW anchor and
must never be described as one.**  MEASURED against `bgw_sigma_hp_noavg.dat`:

| | sigSX MAE | sigCOH MAE | sigTOT MAE |
|---|---|---|---|
| fast deck (20 bands, 144 centroids) | 386.4 meV | 2495.6 meV | 2109.2 meV |
| production (60 bands, 960 centroids) | 0.142 meV | 0.587 meV | 0.644 meV |

**The gap is the band cut, not the centroid count.**  Single-variable A/B, the
fast deck's 20 bands run with the PRODUCTION 960-centroid set:

    960 centroids, 20 bands   sigTOT MAE 2037.1 meV
    144 centroids, 20 bands   sigTOT MAE 2109.2 meV

Restoring 6.7x the centroids recovers 72 meV of a 2100 meV deficit.  The
Coulomb-hole sum runs over all `nband` states, so cutting 60 -> 20 removes 40
bands from Sigma_COH and shifts it by ~+2.46 eV.  That is a different physical
approximation, not a numerical error, and no centroid count repairs it.  Any
Si deck at 20 bands is a self-freeze; that is a property of the band cut.

`nband = 20` is a **clean** cut: bands 20 and 21 are non-degenerate at all 8 IBZ
k-points (min gap 228.3 meV, at k0).  Verified cuts in this WFN (tol 1e-4 eV):

    CLEAN: 8 (= nval), 16, 20, 28
    SPLIT: 10, 12, 14, 18, 22, 24, 26

Cutting inside a multiplet would make the frozen numbers depend on eigenvector
mixing within the degenerate block.  20 also retains the 4-fold at 14.01 eV.

`zeta_rcond` is **unset**, and here that is measured-safe rather than hopeful.
Sweep on this exact configuration, max|d| vs the 1e-10 arm:

    rcond     1e-10   1e-9   1e-8   default   1e-7    1e-6    1e-5     1e-4
    sigTOT    0.000   0.000  0.000   0.000    2.094  19.931  43.645  237.539  meV

1e-10 / 1e-9 / 1e-8 / key-omitted are **byte-identical**, which also confirms
the shipping default is 1e-8.  So the fast deck exercises the shipping code
path and spends no gate margin.


How the centroid sets were made
-------------------------------

`centroids_frac_144.txt` (2026-08-07), run from a directory containing
`WFN.h5` — the generator hardcodes that filename and has no `--wfn` flag:

    python3 -m centroid.kmeans_cli 120 --prune-n-val 8 --prune-n-cond 12

The request is 120; the tool writes the count that **survives** orbit unfolding
and pivoted-Cholesky pruning, here 144 (3 orbits x 48 ops, orbit-closed).  Its
rank gate reported `3/3 directions certified (floor 3, tol 0.01) — PASS`.  The
prune window is matched to this deck's own sigma window (nval 8, ncond 12) so
the ISDF basis is selected on exactly the pair densities its Sigma consumes.

`centroids_frac_960.txt` (2026-05-01) **cannot be reproduced exactly** — the
invocation was not preserved, and four of the generator's defaults have moved
since.  What is established from evidence: the weight was the occupied
ground-state charge density (`--centroid-weight band_range` did not exist until
2026-07-21), orbit mode was **off** (the set is measurably not closed under the
point group), the prune window was the legacy `v_x_c` with `n_cond` clamped to
8, and `--oversample` was 1.5.  The two sets are therefore made by *related but
not identical* procedures; the 144 set follows today's shipping defaults, which
is the right choice for a new fixture because the old `n_cond` clamp is a
documented bug.


Known defects, measured and unfixed
-----------------------------------

1. **Symmetry-equivalent k-points do not carry identical Sigma in the
   production run.**  Worst per-band spread within a star: **2.611 meV**
   (sigTOT).  It should be exactly 0.  The fast deck, whose 144-point centroid
   set *is* orbit-closed, measures exactly **0.000**.  The production 960-point
   set is a literal (non-orbit-closed) point set, so the ISDF quadrature itself
   breaks the 48-op point group.  This is why the IBZ-representative and
   full-BZ numbers in the table above differ, and why the gate uses the full BZ
   — an IBZ-only comparison hides it and depends on which representative you
   pick.  Not fixed here: fixing it means regenerating the production centroid
   set, which means re-freezing the anchor.

2. **`kin_ion` is not degeneracy-symmetrised where BGW symmetrises it.**
   Within-multiplet spread is exactly 0.000 for sigX/sigSX/sigCOH/V_H but
   ~18.5 meV for `kin_ion`.  This lands on Eqp0 (1.527 meV MAE), not on the
   Sigma columns, which is why the BGW gate compares Sigma and not Eqp0.
   `kin_ion.h5` is an input artifact this fixture does not regenerate.

3. **The residual is not decomposed** into native v(q+G) error vs ISDF error.
   The 185 MB BGW `vcoul` dump that would separate them no longer exists.


What each gate can and cannot see
---------------------------------

`test_si_production_matches_frozen_reference` — bit-identity against
`eqp_si_ref.dat`.  Sees any code change that moves the numbers.  **Cannot see
drift away from BerkeleyGW**, because BGW is not in that comparison.

`test_si_production_matches_berkeleygw` — the suite's ONE external check.  It
would fail if `mc_average_vcoul_body` reverted to the LORRAX default (141.65
meV MAE, ~95x the 1.5 meV budget), if v(q+G) broke, if the ISDF rank collapsed,
or if the q->0 head went wrong.  It would **not** catch anything below ~1.5 meV
MAE, and it says nothing about `kin_ion` or `V_H`, which are not BGW-compared
columns.

`test_si_fast_matches_frozen_reference` — "did the code change?", in ~12 s.
Says nothing whatever about BerkeleyGW.


Reproducing by hand
-------------------

```bash
cp -r tests/regression/si_cohsex_debug /tmp/si && chmod -R u+w /tmp/si && cd /tmp/si
python -m gw.gw_jax -i cohsex_si_test.in     # production, ~35 s
python -m gw.gw_jax -i cohsex_si_fast.in     # fast, ~12-16 s

python3 tools/bgw_sigma_hp_to_fixture.py compare \
    bgw_sigma_hp_noavg.dat eqp_si_test.dat
```

Copy, never symlink: the driver writes its outputs into the run directory, and
a write through a stray symlink destroyed a checked-in fixture on 2026-07-25.
That is why `harness.protect_fixtures` keeps everything here `a-w` at rest.
