# Driver suite on the magnetic H2+ spinor fixture

The test suite is the production drivers run end to end on one tiny
magnetic system, at P4 on one node, checked against stored outputs.

## Fixture

`fixture/`: two H atoms with one electron (`tot_charge = 1`), noncollinear
with spin-orbit, magnetized along y (total moment 1.000 μB), in an 8 × 9 × 10
bohr monoclinic cell with a y screw axis (rotation diag(−1, 1, −1),
translation (0, ½, 0)). 8 Ry cutoff, 9 two-component bands, 5 × 5 × 1 k-grid
(25 full, 9 stored). The gap is 0.048 eV between the exchange-split bonding
states, so the occupied set is one band and time reversal is broken. Built
2026-09-05 by QE 7.3.1 SCF → NSCF → pw2bgw → wfn2hdf; `scf_receipt.json` is
the registered SCF parser's receipt. It is a software fixture, not a
converged physical reference.

## The chain (`chain.py`)

| stage | driver | option covered |
|---|---|---|
| kmeans | `centroid.kmeans_cli` | orbit-closed charge centroids, forced sharding |
| kin_ion | `gw.kin_ion_io` | spinor kinetic + ionic matrix elements |
| dipole | `psp.get_dipole_mtxels` | dipole with analytic V_NL velocity |
| gnppm | `gw.gw_jax` | fresh ζ fit, GN-PPM one-shot, `head_correction = full`, spectral_shell band extrapolation, local linalg |
| shared_pole_sc | `gw.gw_jax` | shared-pole full-frequency QSGW, 2 maps to a 1.5 eV criterion, ordered (TR-broken) store with the direct head, ζ restart, distributed linalg |
| bse | `bse.bse_jax` | TDA Davidson on the GN-PPM restart bundle, screened direct term (D + V − W) |
| htransform | `bandstructure.htransform` | Galerkin band interpolation on a G–X–M path |
| exciton_bands | `bse.exciton_bands` | fH interpolation, then finite-Q TDA (1v × 1c) on the same path |
| cohsex | `gw.gw_jax` | static COHSEX one-shot, restarted |
| sp_export | `gw.gw_jax` | shared-pole one-shot with `write_w`/`write_poles` (the file-model path), restarted; the W bank is compared by value, the pole model by member shapes |
| kin_ion_bisp | `gw.kin_ion_io` | four-component kinetic-balance kinetic + ionic matrix elements |
| dipole_bisp | `psp.get_dipole_mtxels` | the four-component dipole the direct head authenticates |
| bisp_sc | `gw.gw_jax` | `bispinor_gw = full_shared_pole` through the SC driver, 2 maps: fresh charge + transverse ζ (both on the charge centroid set), ordered CC/CT/TC/TT sector poles, direct four-current Γ head, `sc_head_update = dft_velocity`, live four-current density; map 0 is the one-shot |
| bse_bisp | `bse.bse_jax` | TDA Davidson after `bisp_sc` on the four-component restart: the final map's W0 = V + W_c,CC(0) (charge sector only) |

All stages run in one Python process per rank (`chain.run_stage` calls each
driver's `main` in sequence): one `jax.distributed` world, one FFI load, one
compile cache. No driver code changes were needed for re-entry. Between
drivers the runner runs `gc.collect()` and `jax.clear_caches()`. The
restarted steps read the `tmp/` state (ζ, V(q), W0) the chain has written.

Every stage is then checked on its outputs, not its exit code: the
eqp0/eqp1 columns, every numeric member of the h5 files it writes, and its
eigenvalue tables, against `reference/`. Every rank log is also scanned for
failure signatures (`chain.FAILURE_SIGNATURES`). Tolerances are in
`chain.ATOL`. A driver that fails on one rank exits its process, so srun
ends the step instead of leaving the other ranks in a collective.

Not covered:
- GN-PPM through the SC driver: the shared-pole stages are the SC runs;
- `screening_diagrams = w_bse`: it refuses on a time-reversal-broken
  reference (`GATE w_bse_requires_measured_trs`);
- the planners' chunked paths: at μ = 6 every object is KB-sized, and a
  `memory_per_device_gb` small enough to chunk leaves the live set no room
  (`GATE gn_ppm_fit_capacity` at 0.05 GB);
- semicore (none in H2+: no semicore window, η_semi read or semicore pin);
- the SC partition beyond "all Σ bands protected" (three Σ bands, one
  occupied, so no rotating class);
- metals, the bispinor charge route (`bare_transverse`), `full_static_cohsex`,
  and the 2-D slab head.

The bispinor stages run last: their fresh ζ fit rewrites the `tmp/` restart
the scalar restarted steps read.

## Commands

[Contributing](../../docs/contributing.md#the-test-suite) owns the canonical
invocation (`lx run -N 1 -G 4 -n 4 -- python -m pytest tests/hsuite`) and the
regenerate command.
