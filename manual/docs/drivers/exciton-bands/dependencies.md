---
driver: exciton-bands
page: deps
---

# Dependencies

`bse.exciton_bands` What exciton_bands needs and what reads its output.
{: .page-summary }

<div class="chain-map" data-driver="exciton-bands" data-page="deps"></div>

## Position in the chain {: #chain }

<div class="updown" markdown>

<div markdown>

#### Reads, produced by

`isdf_tensors_<N_mu>.h5`
:   gw, or [downfold](../../utilities/downfold.md#bundle)

`zeta_q.h5`
:   gw, or [downfold](../../utilities/downfold.md#zeta)

`WFN.h5`
:   DFT + pw2bgw; `WFN_qp.h5` from a QSGW gw run

`centroids_frac_<N>.txt`
:   centroids

`eqp1.dat`
:   gw (optional)

`dipole.h5`
:   dipole (optional)

</div>

<div markdown>

#### Read by

no driver
:   `<prefix>.dat` and `<prefix>.png` are end products

</div>

</div>

The H-transform is not a separate run. The driver builds the Galerkin basis and $f(H)$ in process, with the [htransform](../../utilities/htransform.md#how) machinery, from `WFN.h5` and the centroid table ([Theory §2](theory.md#bands)). It reads no htransform output: not `galerkin_dft.h5`, not `bandstructure.dat`, not `qp_wfn_rotations.h5`.

## Inputs by route {: #inputs }

| Input | Read when | Used for |
|---|---|---|
| deck with `K_POINTS` | always | path corners, H-transform window `nval`/`ncond`, `wfn_file`, `centroids_file` |
| `isdf_tensors_<N_mu>.h5` | always | $\psi_{n\bk}(\br_\mu)$ and $E_{n\bk}$ on the grid, $W_\bq(\mu,\nu;0)$, $V_{\bq=0}$ and the heads; the stored $V_\bq$ tiles on every route |
| `WFN.h5` | always | the H-transform; under `refit` a second time, with `--refit-guard-bands` more bands, for the ζ′ pair densities |
| centroid table | always | the points $\br_\mu$ at which $\psi_{c\,\bk+\bQ}$ is evaluated |
| `zeta_q.h5` | `--vq-mode` `interp`, `both`, `refit` | `interp` and `both`: every stored ζ tile, full BZ; `refit`: metadata and `fit_provenance` only, so IBZ storage is enough |
| `eqp1.dat` | `eqp_corrections` | QP energies on both legs ([Theory §2.3](theory.md#bands-qp)) |
| `dipole.h5` | `--head-minibz-average` or `head_minibz_average`, on `interp` and `both` | the head tensor of [Theory §3.4](theory.md#ex-head); `dipole_file` names it, default beside the deck |

`ongrid` reads no ζ. File formats are on the [Files](files.md#inputs) page.

## Shared with `bse.bse_jax` {: #shared }

The driver loads its $\bQ$-independent data through the optical BSE's own code, so these behave as the [`bse` Dependencies](../bse/dependencies.md) page describes:

- restart discovery and its one-bundle refusal;
- the loader: band window and degeneracy guard, ζ-training check, $\bq\to0$ head injection with the deck's `vhead` and `whead_0freq` overrides, and the rebuild of a missing static $W$ from the restart;
- `kgrid_fine` densification with `w_head_densify` ([Theory §4.4](theory.md#w-fine));
- the `eqp1.dat` reader and its re-slice of the window;
- the `dipole.h5` reader and its basis check;
- the production stack matvec and the device mesh.

`bse.bse_jax` needs none of the centroid table, `WFN.h5` for interpolation, or `zeta_q.h5`. `dipole_file` names a different use in each driver: exciton dipoles there, the head tensor here. This driver has no `highest_occupied_band`; the occupied count is the WFN's `ifmax`.

## Caches {: #caches }

The driver writes no restart and no cache, and reads none of its own. The Galerkin basis, ζ′ and the conduction states at $\bk+\bQ$ are rebuilt on every run.

!!! cache "$SCRATCH/.cache/lorrax/jax_compile/&lt;namespace&gt;/np&lt;P&gt;"
    **Reused when** the JAX persistent compile cache is enabled (the default)<br>
    **and matching** the lowered program. The whole path is one scan program, so the $\bQ$ count, window, grid and mesh fix the key.<br>
    <span class="refuse">Recompiles on a miss. `ISDF_JAX_CACHE_DIR` moves the cache; an empty value turns it off.</span>

## Runtime {: #runtime }

- `runtime.initialize_communicator_stack` runs at module import, before argument parsing, so `--help` also brings up the runtime.
- The device mesh is the run's square startup mesh; an explicit `--px`/`--py` must equal it.
- Rank 0 writes every file. Host-side setup (the path, the $\{\bk+\bQ\}$ list, the mini-BZ head cubature) runs identically on every process.
- The native FFI pair is required, as for every driver; sharded restart and ζ reads go through it.
- Launch geometry is owned by the machine pages under [Installation](../../using/installation.md#routes).

