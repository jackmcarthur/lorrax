# Fine-grid interpolation

`bandstructure.bse_setup.compute_wfns_fi` States and energies on a fine k grid, interpolated from the GW grid for the BSE.
{: .page-summary }

The BSE and exciton-bands drivers call the [H-transform](htransform.md) in memory to carry $\psi_{n\bk}(\br_\mu)$ and $E^{\mathrm{QP}}_{n\bk}$ from the coarse grid $\bk_{\mathrm{co}}$ of the GW run to a fine grid $\bk_{\mathrm{fi}}$. Under [`bse_k_grid`](../drivers/bse/keywords.md#bse_k_grid) this fills the BSE fine grid, and for [exciton bands](../drivers/exciton-bands/index.md) it gives the shifted conduction states $\psi_{c\,\bk+\bQ}$. No DFT calculation on the fine grid is needed.
{: .lead }

## Band window {: #window }

The transformed Hamiltonian is built from every band from the lowest valence band up to the deck's band count, and the bands the BSE uses are returned from inside it. Inside the BSE drivers the H-transform adds no guard bands of its own, so the deck's band count must leave at least four bands above the highest conduction band the BSE uses. The exciton-bands route warns when there are fewer than four, or more than sixteen, since a very wide window asks the shared basis to represent states the kernel never uses. The method and the shared keys are on the [overview](htransform.md).

## Keywords {: #keywords }

Defaults are read from the keyword schema (`src/gw/gw_config.py`).

Optional
{: .kw-group }

**Fine grids**

### `bse_k_grid` {: .kw data-type="str" data-default="empty" }
`"NX NY NZ"`: the Γ-centred BSE fine grid that this engine fills. Owned by the [BSE driver](../drivers/bse/keywords.md#bse_k_grid).

### `wfn_fi_q_chunk` {: .kw data-type="int" data-default="0 → N_k(co)" }
$\bk_{\mathrm{fi}}$ points per $f(H_{\bk_{\mathrm{fi}}})$ build and eigensolve on the BSE routes, rounded up to a multiple of the device count.

Read by `bse_k_grid` and exciton bands only; a negative value refuses. The standalone `get_centroids_fi` recovery always uses the default.
{: .kw-details }

### `linalg` {: .kw data-type="enum" data-default="local" }
Eigensolver of the fine-grid route; `distributed` shards each $f(H_{\bk_{\mathrm{fi}}})$ when one matrix does not fit on a device.

The path always uses the native batched `eigvalsh`/`eigh`. Also sets the batch route of the basis Cholesky.
{: .kw-details }

`local` `distributed`
{: .kw-values }

On the fine-grid route the returned bands are the ascending-eigenvalue slice of the window.

`bse.exciton_bands` passes `--a-band`, `--refit-guard-bands`, `--q-per-segment` and `--eqp` to this engine; they are on the [exciton-bands keyword page](../drivers/exciton-bands/keywords.md).

**Standalone recovery (diagnostic)**

The standalone driver can also run the fine-grid recovery after the path. Its result is logged and not written.

### `get_centroids_fi` {: .kw data-type="bool" data-default="false" }
After the path, also computes $\psi_{n\bk_{\mathrm{fi}}}(\br_\mu)$ and $E_{n\bk_{\mathrm{fi}}}$ on `kgrid_fi`.

Diagnostic: the result is logged (shapes) and not written. The BSE drivers do not read this key.
{: .kw-details }

### `kgrid_fi` {: .kw data-type="str" data-default="empty" }
`"nx ny nz"` (or comma-separated) fine grid of the `get_centroids_fi` recovery.

### `wfn_fi_min`, `wfn_fi_max` {: .kw data-type="int" data-default="0, 0" }
Window-relative sub-window $[$`wfn_fi_min`$,$ `wfn_fi_max`$)$ returned by the `get_centroids_fi` recovery.

`wfn_fi_max = 0` means the whole fitted window, guard bands included.
{: .kw-details }

## Refusals {: #refusals }

!!! refuses
    - A BSE conduction window that reaches above the band count of the transformed Hamiltonian. Raise `number_bands` (`nband`), or lower the BSE conduction count.
    - A negative `wfn_fi_q_chunk`.

