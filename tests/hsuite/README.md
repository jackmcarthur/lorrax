# Driver suite on the magnetic H2⁻ spinor and bcc Na fixtures

The test suite is the production drivers run end to end on two tiny systems,
a magnetic H2⁻ spinor cell and a bcc Na metal, at P4 on one node, checked
against stored outputs.

## Fixture

`fixture/`: two H atoms with three electrons (`tot_charge = -1`),
noncollinear with spin-orbit, magnetized along y (total moment 1.000 μB), in
an 8 × 9 × 10 bohr monoclinic cell with a y screw axis (rotation
diag(−1, 1, −1), translation (0, ½, 0)). LDA (`input_dft = 'pz'`) on the PBE
`H.upf`, 8 Ry cutoff, 9 two-component bands, 5 × 5 × 1 k-grid (25 full, 9
stored). The indirect gap is 0.072 eV, the occupied set is three bands and
time reversal is broken. At Γ each band density of bands 1–7 is invariant
under the header's four operations to 3e-10. Built 2026-09-30 by QE 7.4.1
SCF → NSCF → pw2bgw with the SOC branch-selection patch, then wfn2hdf
(`runs/H2/07_fixturefix_20260930` in the sandbox); `scf_receipt.json` is the
registered SCF parser's receipt. It is a software fixture, not a converged
physical reference.

Why not H2⁺, and why not GGA: with one electron |m| = ρ at every grid point
to round-off. QE's spin-polarized gradient correlation (`gcc_spin`,
`XClib/qe_drivers_gga.f90:1078-1082`) skips a point where |ζ| > 1 and clamps
ζ to 1 − 10⁻⁶ elsewhere, so round-off picks the points, and the potential
does not have the cell's symmetry. QE's general noncollinear GGA branch takes
the same path. The H2⁺ PBE fixture it replaces had Γ band densities that
moved by 5e-3 to 9e-2 under the screw axis and the inversion, and a
symmetry-reduced run printed two q↔−q reciprocity `SANITY FAILURE` lines.
LDA on H2⁺ still leaves a (1 − ζ)^(1/3) round-off floor in the minority
potential (5e-8 on bands 3–4). With three electrons the minority density is
finite.

`fixture_na/`: bcc Na, one atom, PseudoDojo nc-sr-04 PBE (9 electrons:
2s at −64 eV and 2p at −22 eV lie 20 eV below the 3s band), 25 Ry cutoff,
Fermi-Dirac k_BT = 0.01 Ry, 3 × 3 × 3 k-grid (27 full, 4 stored), 14 scalar
bands, E_F = 1.99 eV. Built 2026-09-29 by QE 7.5 SCF → NSCF → pw2bgw →
wfn2hdf (`runs/Na/31_hsuite_fixture_20260929` in the sandbox). It is the
smallest metal with a ≥ 4 eV-gapped semicore manifold. The WFN has one band
above the deck's 13: the ζ fit's right window must have a measured upper gap,
and the SC energy ladder (13 padded to 14 at P4) must hold the velocity's
bands. The Na stages run in `run/na/`.

## The chain (`chain.py`)

| stage | driver | option covered |
|---|---|---|
| kmeans | `centroid.kmeans_cli` | orbit-closed charge centroids, forced sharding |
| kin_ion | `gw.kin_ion_io` | spinor kinetic + ionic matrix elements |
| dipole | `psp.get_dipole_mtxels` | dipole with analytic V_NL velocity |
| gnppm | `gw.gw_jax` | fresh ζ fit, GN-PPM one-shot, `head_correction = full`, spectral_shell band extrapolation, local linalg |
| bse | `bse.bse_jax` | TDA Davidson on the GN-PPM restart bundle, screened direct term (D + V − W) |
| htransform | `bandstructure.htransform` | Galerkin band interpolation on a G–X–M path |
| exciton_bands | `bse.exciton_bands` | fH interpolation, then finite-Q TDA (1v × 1c) on the same path |
| cohsex | `gw.gw_jax` | static COHSEX one-shot, restarted |
| sp_export | `gw.gw_jax` | shared-pole one-shot with `write_w`/`write_poles` (the file-model path), restarted; the W bank is compared by value, the pole model by member shapes |
| kin_ion_bisp | `gw.kin_ion_io` | four-component kinetic-balance kinetic + ionic matrix elements |
| dipole_bisp | `psp.get_dipole_mtxels` | the four-component dipole the direct head authenticates |
| bisp_sc | `gw.gw_jax` | `bispinor_gw = full_shared_pole` through the SC driver, 2 maps: fresh charge + transverse ζ (both on the charge centroid set), ordered CC/CT/TC/TT sector poles, direct four-current Γ head, `sc_head_update = dft_velocity`, live four-current density, the default band extrapolation (on the CC class); map 0 is the one-shot |
| bse_bisp | `bse.bse_jax` | TDA Davidson after `bisp_sc` on the four-component restart: the final map's W0 = V + W_c,CC(0) (charge sector only) |
| na_kin_ion | `gw.kin_ion_io` | bcc Na kinetic + ionic matrix elements; the 56 orbit-closed centroids (fitted on the ζ legs, 8 × 13) are stored in `fixture_na/` |
| na_dipole | `psp.get_dipole_mtxels` | the default dipole: q→0 velocity plus the parallel-transport link artifact |
| na_sc | `gw.gw_jax` | metal shared-pole QSGW, 2 maps to a 1.5 eV criterion, production defaults: Fermi-Dirac fixed-N occupations and metal head, `number_bands_protected = 8` (QP matrix 1–8, tail 9–13), 2s and 2p read on two coarse windows at η 5 eV, `sc_semicore = dft`, the rigid tail with min(Z, 1/Z) weights, spectral_shell extrapolation, held Σ windows re-planned on escape, `write_qsgw_datasets`, the unnamed `sc_head_update`, which resolves to `parallel_transport` (the 3³ links fail `GATE pt_head_window_hybridized`, so the map runs with the Σ term D_kΔH zeroed) with its per-map head block, distributed linalg |

All stages run in one Python process per rank (`chain.run_stage` calls each
driver's `main` in sequence): one `jax.distributed` world, one FFI load, one
compile cache. No driver code changes were needed for re-entry. Between
drivers the runner runs `gc.collect()` and `jax.clear_caches()`. The
restarted steps read the `tmp/` state (ζ, V(q), W0) the chain has written.

The `na_sc` rank-0 log must also show, by name, each default the stage
covers (`chain.REQUIRED_LINES`): the partition line, the two coarse windows
at η 5 eV, the `sc_semicore = dft` pin, the Z-weighted tail at map 1, the
unnamed head's `parallel_transport` map with the Σ term zeroed, the Fermi-Dirac
metal head, the map-1 head block with a 0.0000 eV gap, and the
map-1 Σ window re-plan.

Every stage is then checked on its outputs, not its exit code: the
eqp0/eqp1 columns, every numeric member of the h5 files it writes, and its
eigenvalue tables, against `reference/`. Tolerances are in `chain.ATOL`.

Every rank scans its own stage log for failure signatures
(`chain.FAILURE_SIGNATURES`, and a GATE refusal carried by an exception,
`chain.REFUSAL`) and joins the others with its exit status, hits, GATE
names and, on a failure, the log's last 60 lines. A GATE name that a rank
r > 0 logs and rank 0 does not is a failure too. Any failed rank fails the
test on every rank; the assertion names the stage and rank and carries the
log tail, which pytest prints. `summary.json` holds every rank's records
(`rank_records`). A failed rank waits `chain.FAIL_JOIN_S` for the others;
if they are blocked in a collective it writes `summary.rank<r>.json`, fails
alone, and `conftest.py` ends the process with `os._exit` after pytest's
report, so srun ends the step. The per-rank stage logs stay in
`run/<stage>.rank<r>.log`. `HSUITE_INJECT_REFUSAL=stage:rank[:before]` (test
only) makes one rank refuse after (or before) a stage's driver, to check
this path.

Not covered:
- GN-PPM through the SC driver: the shared-pole stages are the SC runs;
- `screening_diagrams = w_bse`: it refuses on a time-reversal-broken
  reference (`GATE w_bse_requires_measured_trs`);
- the planners' chunked paths: at μ = 6 every object is KB-sized, and a
  `memory_per_device_gb` small enough to chunk leaves the live set no room
  (at 0.05 GB the planners warn over budget and take their smallest sizes);
- the `parallel_transport` head itself: the 3³ Na links fail the window
  gate, so the unnamed default runs `dft_velocity`;
- the W line-site re-plan: it first fires at Na map 2 (and runs there,
  with every multiplet degenerate), but `na_sc` converges at its 1.5 eV
  criterion after map 1, and the two more maps a tighter criterion needs
  would take the suite past its wall budget;
- the bispinor charge route (`bare_transverse`), `full_static_cohsex`,
  and the 2-D slab head.

The bispinor stages run last: their fresh ζ fit rewrites the `tmp/` restart
the scalar restarted steps read.

## Commands

[Contributing](../../docs/contributing.md#the-test-suite) owns the canonical
invocation (`lx run -N 1 -G 4 -n 4 -- python -m pytest tests/hsuite`) and the
regenerate command.

## Wall time and caches

On one node at P4 the fixtures' arithmetic takes seconds; most of a run's wall
is the compile path. Each rank makes 3167 compile requests (claim 4168), and a
warm run still traces and lowers every program
([compilation §4](../../docs/architecture/compilation.md#4-where-the-time-goes)
has the cold and warm splits). Two caches make a second run cheaper. Both live
under `$SCRATCH/.cache/lorrax/`, at the same path on every rank:

- **JAX's persistent compile cache.** With `HSUITE_CACHE_DIR` unset, the suite
  uses the runtime's cache: `ISDF_JAX_CACHE_DIR` if exported, else the
  runtime's namespace for `np4`
  ([env_vars §2e](../../docs/reference/env_vars.md#2e-compile-cache)).
  Programs that carry a host callback compile again in every run, because JAX
  never stores them (`uncacheable` below).
- **The mathdx cubin cache** (`kconv_mathdx/`). A release links its prebuilt
  images into it from `<source root>/cubin_store`; NVRTC builds any other
  image the first time a process meets its key, at about 7 s each. The two
  fixtures need 25 images (claim 4126;
  [kconv §14](../../docs/architecture/kconv.md#build-and-cache)). The cache's
  path is part of every k-convolution program's JAX key, so a run is warm only
  under the `SCRATCH` of the run that filled the JAX cache.

`HSUITE_CACHE_DIR=<empty dir>` gives a cold JAX cache with warm cubins, the
state a release's first run meets. A fresh `SCRATCH` as well makes both
caches cold.

`summary.json` `compile_s` splits each stage's wall, as seen from rank 0:

- `trace`, `lower` and `compile` are the union of JAX's own compile-path spans
  (`jax._src.dispatch` events). `compile` covers `compile_or_get_cached`: a
  cache read, or an XLA compile. `compile_path` is the union of all three.
- `cache_lookups`, `read_s`, `fingerprint_s` and `agreement_s` come from
  `common.jax_compile_cache`. The last two are the cross-rank compile
  agreement: the module hash, and the exchange.
- `xla_compiles`, `xla_s`, `cache_hits` and `uncacheable` are the sums of the
  compile receipts in the stage log. Each stage ends with its own receipt,
  which also closes the receipt window, so an SC driver's map-0 receipt counts
  that driver only.
- `nvrtc_builds` and `nvrtc_s` count the cubins the stage built.

What remains of the wall, `wall - compile_path - nvrtc_s`, is execution, host
work and I/O.

## Steady compiles (`bisp_sc3`)

The opt-in stage `bisp_sc3` runs the `bisp_sc` deck on the face route for
three maps (it ends unconverged by design and has no reference). Its check
reads the SC maps' compile receipts: from map 2 on nothing compiles, and no
program carries a host callback (which JAX's persistent cache never stores).
Run it a second time on the same `--cache-dir` and its receipts show what a
warm process still compiles.

    lx run -N 1 -G 4 -n 4 -- python3 -m tests.hsuite.chain --out DIR --cache-dir CACHE \
        --only kmeans kin_ion kin_ion_bisp dipole_bisp bisp_sc3
