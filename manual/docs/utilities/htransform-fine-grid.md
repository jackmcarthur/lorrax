# Fine-grid interpolation

`bandstructure.bse_setup.compute_wfns_fi` States and energies on a fine k grid, interpolated from the GW grid for the BSE.
{: .page-summary }

The BSE and exciton-bands drivers call the [H-transform](htransform.md) in memory to carry $\psi_{n\bk}(\br_\mu)$ and $E^{\mathrm{QP}}_{n\bk}$ from the coarse grid $\bk_{\mathrm{co}}$ of the GW run to a fine grid $\bk_{\mathrm{fi}}$. Under [`kgrid_fine`](../drivers/bse/keywords.md#kgrid_fine) this fills the BSE fine grid, and for [exciton bands](../drivers/exciton-bands/index.md) it gives the shifted conduction states $\psi_{c\,\bk+\bQ}$. No DFT calculation on the fine grid is needed.
{: .lead }

## Band window {: #window }

The transformed Hamiltonian is built from every band from the lowest valence band up to the input file's band count, and the bands the BSE uses are returned from inside it. Inside the BSE drivers the H-transform adds no guard bands of its own, so the input file's band count must leave at least four bands above the highest conduction band the BSE uses. The exciton-bands route warns when there are fewer than four, or more than sixteen, since a very wide window asks the shared basis to represent states the kernel never uses. The method and the shared keys are on the [overview](htransform.md).

When the BSE loader fills the `kgrid_fine` grid with QP energies from `eqp_corrections` or `eqp_co_corrections`, a window that reaches above the eqp file's top band gets the same rigid conduction scissor on the bands above it as [`--eqp-file`](htransform.md#-eqp-file), with the same `[QP scissor]` log line.

## Memory {: #memory }

The fitted basis and transformed Hamiltonian are sized by the **fitted** band/basis window, not just the valence and conduction states returned to the BSE. The fine-q chunk is padded for the mesh, and the returned centroid carriers are padded separately from the physical interpolation points. These zero slots affect storage, not the requested grid or band window.

The fine-grid engine retains the interpolated wavefunctions and energies. Coefficients used to project those wavefunctions stay inside the compiled chunk unless its caller explicitly requests the coefficient output; the normal no-coefficient path omits that unused export and reshard automatically. Read the selected q-batched or FACE route and its compiled memory price; reducing the returned band window alone need not reduce the Hamiltonian or eigensolver workspace. General sizing guidance is on [Running on clusters](../using/clusters.md#resident-memory).

## Keywords {: #keywords }

Defaults are read from the keyword schema (`src/gw/gw_config.py`).

Optional
{: .kw-group }

**Fine grids**

### `kgrid_fine`, `kgrid_fine_shift` {: .kw data-type="int[3], float[3]" data-default="empty, 0 0 0" }
The fine grid $\bk_{\mathrm{fi}}$ and kgrid.x's offset $\delta\bk$ in fractions of one fine step, $\bk_i = (i + \delta\bk)/N$. Shared by the [BSE driver](../drivers/bse/keywords.md#fine-grid) and the `get_centroids_fi` route below; a shift breaks the grid's symmetry and samples the zone evenly (a DOS). Exciton bands refuse a shift.

### `number_cond_bands_coarse` {: .kw data-type="int" data-default="ncond" }
Conduction bands of the coarse grid that the H-transform fits. In the BSE it may exceed `ncond` (the extra bands are guard bands read from the WFN); in the standalone driver it replaces `--guard-bands` (fitted conduction bands = `ncond` + guard bands). `number_val_bands_coarse` is accepted and ignored: every valence band of the window is fitted.

### `wfn_fi_q_chunk` {: .kw data-type="int" data-default="0 → N_k(co)" }
$\bk_{\mathrm{fi}}$ points per $f(H_{\bk_{\mathrm{fi}}})$ build and eigensolve on the BSE routes, rounded up to a multiple of the device count.

Read by `kgrid_fine` and exciton bands only; a negative value refuses. The standalone `get_centroids_fi` recovery always uses the default.
{: .kw-details }

### `linalg` {: .kw data-type="enum" data-default="local" }
Eigensolver of the fine-grid route; `distributed` shards each $f(H_{\bk_{\mathrm{fi}}})$ when one matrix does not fit on a device.

The path always uses the native batched `eigvalsh`/`eigh`. Also sets the batch route of the basis Cholesky.
{: .kw-details }

`local` `distributed`
{: .kw-values }

On the fine-grid route the returned bands are the ascending-eigenvalue slice of the window.

`bse.exciton_bands` passes `--a-band`, `--refit-guard-bands`, `--q-per-segment` and `eqp_corrections` to this engine; they are on the [exciton-bands keyword page](../drivers/exciton-bands/keywords.md).

**Standalone fine grid**

With `get_centroids_fi = true` the standalone driver also interpolates onto `kgrid_fine` (shifted by `kgrid_fine_shift`) after the path and writes only the fine window's bands to `<output stem>_fine.h5`: `kpoints` (crystal), `energies_ev` (k, band), `bands` (1-based), and the attributes `kgrid`, `kgrid_shift`.

### `get_centroids_fi` {: .kw data-type="bool" data-default="false" }
Runs the fine-grid interpolation after the path and writes `<stem>_fine.h5`. The BSE drivers do not read this key.

### `number_val_bands_fine`, `number_cond_bands_fine` {: .kw data-type="int" data-default="nval, ncond" }
The fine window written: valence bands below and conduction bands above the occupied boundary (BerkeleyGW's counting). Fewer than four fitted bands above the window warns. `wfn_fi_min` and `wfn_fi_max` refuse with these names.

## Refusals {: #refusals }

!!! refuses
    - A BSE conduction window that reaches above the band count of the transformed Hamiltonian. Raise `number_bands` (`nband`), or lower the BSE conduction count.
    - A negative `wfn_fi_q_chunk`.

