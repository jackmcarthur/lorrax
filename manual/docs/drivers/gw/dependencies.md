---
driver: gw
page: deps
---

# Dependencies

`gw.gw_jax` What gw_jax needs and what reads its output.
{: .page-summary }

<div class="chain-map" data-driver="gw" data-page="deps"></div>

## Position in the chain {: #chain }

<div class="updown" markdown>

<div markdown>

#### Reads, produced by

`WFN.h5` (`wfn_file`)
:   QE + `pw2bgw`, or [nscf](../nscf/index.md)

`centroids_frac_<N>.txt` (`centroids_file`)
:   [centroids](../../utilities/centroids.md)

`kin_ion.h5` (`kin_ion_file`)
:   [kin-ion](../../utilities/kin-ion.md)

`dipole.h5` (fixed name)
:   [dipole](../../utilities/dipole.md); not read with `head_correction = off`

`parallel_transport.h5` (`parallel_transport_file`)
:   [dipole](../../utilities/dipole.md); QSGW velocity heads only

restart bundle, `qp_wfn_rotations.h5`
:   an earlier gw run (`restart = true`, `sc_initial_qp_rotations_file`)

</div>

<div markdown>

#### Read by

[bse](../bse/index.md)
:   restart bundle; `eqp1.dat` (`eqp_corrections`) or `WFN_qp.h5` as its WFN; `tmp/zeta_q.h5` for a $W(0)$ rebuild

[exciton-bands](../exciton-bands/index.md)
:   restart bundle; `tmp/zeta_q.h5` for $V_\bq$ interpolation

[htransform](../../utilities/htransform.md)
:   `qp_wfn_rotations.h5` or `eqp1.dat`

[downfold](../../utilities/downfold.md)
:   restart bundle; `tmp/zeta_q.h5` when present

`gw.gw_jax`
:   restart bundle; `qp_wfn_rotations.h5` as a QSGW seed; `WFN_qp.h5` as `wfn_file`

`gw.eqp_bgw`
:   `sigma_mnk.h5`, `kin_ion.h5`: rebuilds the eqp pair from files

</div>

</div>

File formats are on the [Files](files.md) page.

## Caches {: #caches }

!!! cache "tmp/isdf_tensors_&lt;N_mu&gt;.h5"
    **Reused when** `restart = true` (never automatic)<br>
    **and matching** the restart checks above: WFN fingerprint, windows, $N_\mu$, centroid hashes, parent rows.<br>
    <span class="refuse">Refuses on mismatch; set `restart = false` to rebuild. $W_0$ and the head scalars are refreshed in place.</span>

!!! cache "tmp/zeta_q.h5"
    **Reused when** `restart = false`, `zeta_is_done` is true and `fit_provenance` is present<br>
    **and matching** this run's provenance JSON (WFN path spelling, a q-wedge equivalence and legacy keys at their legacy values excepted), the stored centroid table element by element, and the ζ dataset's $N_\mu$. The WFN is identified by resolved path and byte size. Device count is not in the provenance, so a fit at one process count is reused at another.<br>
    <span class="refuse">Refits on miss, per channel. `LORRAX_FORCE_REFIT=1` forces a refit.</span>

!!! cache "tmp/mpa/&lt;label&gt;_tangential/"
    **Reused when** a QSGW map finds its own committed `model.h5`, or a complete response bank of the same identity and the same $V_\bq$ digest<br>
    **and matching** the map identity: Hamiltonian, wavefunctions, energies, occupations, centroids, recipe and WFN.<br>
    <span class="refuse">Rebuilds a partial directory. A complete one-shot model refuses (`GATE tangential_output`): use its restart member or a fresh run directory.</span>

!!! cache "tmp/sigma_checkpoint_oneshot_on_shell.h5, tmp/sigma_checkpoint_oneshot.h5"
    **Reused when** `restart = true` on a one-shot tangential run (the `_on_shell` file; the other under `sigma_omega_grid_debug = true`)<br>
    **and matching** the identity hash of the $W$ digest, head, energies, options and bands, and on shell the evaluation tables and covered range.<br>
    <span class="refuse">Deleted and recomputed.</span>

## Runtime {: #runtime }

- `--help` and argument errors are answered first, by `runtime.cli_seam`, before any runtime starts. The deck's existence is checked only after.
- One module-top call brings up the runtime: `runtime.initialize_communicator_stack` (64-bit JAX, the distributed runtime at more than one process, one GPU per process, the $(x, y)$ device mesh, the compilation cache).
- The native FFI pair is required at every process count, and the parallel-HDF5 leg is initialized before the first file is read; see [Installation](../../using/installation.md#ffi) and [Startup refusals](../../using/installation.md#refusals).
- Every large array is read and written per rank through SlabIO; no rank gathers $V_\bq$, $W$ or ζ.
- Perlmutter, one node: `source config/perlmutter/gpu_env.sh`, then `srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh .venv/bin/python -u -m gw.gw_jax -i gw.in`.

