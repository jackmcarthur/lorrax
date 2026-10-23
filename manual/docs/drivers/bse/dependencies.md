---
driver: bse
page: deps
---

# Dependencies

`bse.bse_jax` What the BSE driver needs and what reads its output.
{: .page-summary }

<div class="workflow-map" data-driver="bse" data-page="deps"></div>

## Position in the workflow {: #chain }

<div class="updown" markdown>

<div markdown>

#### Reads, produced by

ISDF tensor file `isdf_tensors_<N_mu>.h5`
:   gw (`write_restart_tensors = true`, the default), or downfold

`WFN.h5`
:   nscf, or DFT + pw2bgw; `WFN_qp.h5` from a QSGW gw run

`centroids_frac_<N>.txt`
:   centroids

`eqp1.dat`
:   gw

`dipole.h5`
:   dipole; `dipole_qsgw.h5` from a QSGW gw run

`zeta_q.h5`
:   gw (only for a $W(0)$ rebuild or `head_minibz_average`)

</div>

<div markdown>

#### Read by

BerkeleyGW-format readers
:   `eigenvectors.h5` (band and k layout; the header groups are empty)

`absorption_common`
:   `eigenvectors.h5`: `load_eigenvectors_h5` (TDA files only), `load_exciton_dipoles_h5`; library functions, no entry point

user
:   `absorption_haydock.h5`, `absorption_haydock_b{1,2,3}_eh.dat`, `bse_dos_kpm.npz`

</div>

</div>

The BSE is the end of the workflow: no LORRAX driver reads its output. `bse.absorption_haydock` reads the same inputs as `bse.bse_jax`, never `eigenvectors.h5`. The sum-over-states $\varepsilon_2$ from stored dipoles is `absorption_common.eps2_from_exciton_dipoles`, which has no command line. Under `kgrid_fine` the band interpolation of the [htransform](../../utilities/htransform.md#how) utility runs in-process; no htransform file is read. File formats are on the [Files](files.md) page.

## Caches {: #caches }

The driver writes no cache. It reuses the GW run's bundle and the runtime's compilation cache.

!!! cache "isdf_tensors_&lt;N_mu&gt;.h5"
    **Reused when** always: it is the ISDF tensor file from GW, read and never written.<br>
    **and matching** centroid hash, band window, $N_\mu$, parent rows and WFN record (table above).<br>
    <span class="refuse">Refuses on mismatch.</span>

!!! cache "W0_qmunu (in the bundle)"
    **Reused when** `W0_ready` is true.<br>
    **and matching** the bundle's authentication; nothing else is compared.<br>
    <span class="refuse">Rebuilt in memory on every call by a static Dyson solve from the input file's WFN, centroids, `zeta_q.h5` and head inputs; never written back. About twice the wall time.</span>

!!! cache "JAX compilation cache"
    **Reused when** a program of the same shapes recurs (same mesh, window and k-grid).<br>
    **and matching** the JAX, jaxlib and native-library versions, which name the cache namespace.<br>
    <span class="refuse">Compiles. `ISDF_JAX_CACHE_DIR` sets the location; an empty value disables it ([Environment](../../using/environment.md#isdf_jax_cache_dir)).</span>

## Runtime {: #runtime }

- `bse.bse_jax` brings up the runtime at import, before argument parsing: `runtime.initialize_communicator_stack` (64-bit JAX, the distributed runtime, the square device mesh, the compilation cache). `--help` and argument refusals come after it.
- The device mesh is square, $(x, y)$ over the conduction and valence axes, and uses every device; `--px`/`--py` may only restate it. Band axes are padded to mesh multiples.
- `bse.absorption_haydock` calls `runtime.bootstrap()` at import, builds the same mesh and refuses on fewer than two devices. It writes no run report.
- Bundle tensors are read per rank through SlabIO (parallel HDF5); no rank gathers $V$ or $W$.
- The direct-term kernel is chosen at trace time from what the device supports (fused FFI kernel or XLA) and printed once (`[bse] W term:`).
- Launch lines are on the [Overview](index.md#run).

