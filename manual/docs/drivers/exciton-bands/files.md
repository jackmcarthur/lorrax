---
driver: exciton-bands
page: files
---

# Files

`bse.exciton_bands` Formats of the files exciton_bands reads and writes.
{: .page-summary }

<div class="chain-map" data-driver="exciton-bands" data-page="files"></div>

## Inputs {: #inputs }

| File | Deck key | Produced by | Note |
|---|---|---|---|
| deck | `-i` | user | The GW deck plus a `K_POINTS` block. Its directory locates the restart. |
| `isdf_tensors_<N_mu>.h5` | — | gw | Exactly one, beside the deck or in its `tmp/`. ψ at the centroids, energies, $V_\bq$ tiles, static $W_\bq$, heads. |
| `zeta_q.h5` | — | gw | Beside the restart. Every route but `ongrid`; `interp` and `both` need it on the full BZ. |
| `WFN.h5` | `wfn_file` | DFT + pw2bgw | The H-transform input; also gives the occupied count. |
| `centroids_frac_<N>.txt` | `centroids_file` | centroids | Required. On a downfolded bundle, the parent table recorded in the bundle. |
| `eqp1.dat` | `--eqp` | gw | Diagonal QP energies on the wedge, unfolded on load. |
| `dipole.h5` | `--dipole` | dipole | Only with the mini-BZ head average. Default: beside the deck. |

## Outputs {: #outputs }

<div class="file-list" markdown>

`<prefix>.dat`
:   Exciton levels $\Omega_S(\bQ)$ in eV, one row per $\bQ$, with a header naming the run. Default prefix `exciton_bands`.

`<prefix>.png`
:   The lowest `--n-eig` levels along the path, corner labels on the axis, refit spot checks as markers and the certification stamp at the foot.

`<prefix>.out`
:   Run report (`--report-file`): routes, environment, band windows, the path corners, the resolved kernels, the levels at each corner, the Hermiticity check and the files read and written.

standard output
:   Progress, the gate lines (`[gate]`, `[refit-null]`, `[refit-cert]`), the interp − refit table under `both`, and a two-column stage timing table that sums to `TOTAL`.

</div>

Rank 0 writes every file. No eigenvectors, no restart and no cache are written.

## exciton_bands.dat format {: #dat }

Plain text. Header lines start with `#`; then one row per path point, then one per `--extra-q` point, then one per `both` spot check.

```text
# Exciton bandstructure E_S(Q), TDA, LORRAX
# <provenance lines: LORRAX version, commit, host, date>
# input: /scratch/si/cohsex.in
# window: n_val=4 n_cond=4; n_eig=6; kgrid 4x4x4; vq_mode=refit
# refit: window=zeta (producer's zeta-fit window)
# refit-fh-window: absolute bands [0, 20) = zeta window + 4 guard band(s); Galerkin residual 2.1e-04 (fH) / 1.8e-04 (zeta window)
# refit-cert: CERTIFIED reference grade: worst max|dE_S| 0.00412 meV against the 0.01 meV reference gate, contracted on-grid gate (refit route vs stored V_qmunu route) over 3 Q
#   Q#16 tile (2, 0, 2): max|dE_S| = 0.00412 meV
# conventions: |v k, c k+Q>; exchange tile at wrap(-Q) keeps G=0 at finite Q (energy_loss); Gamma uses the production q=0 head-body tile; energies in eV
# nodes: 0:Γ 16:X 32:K 48:Γ
# iQ  s_path  Qx  Qy  Qz  mode  E_1..E_neig (eV)
   0 0.000000  0.000000  0.000000  0.000000 interp 3.381204 3.381204 3.381204 3.412877 3.412877 3.412877
   1 0.072551  0.031250  0.000000  0.031250 interp 3.389410 3.389410 3.395022 3.415630 3.420115 3.420115
 ...
```

The values shown are a format sample, not reference numbers. The `refit`, `refit-fh-window` and `refit-cert` lines appear only on the routes that produce them; a `visualization` grade adds a `refit-cert-grade` line saying the levels are not reference numbers.

| Column | Content |
|---|---|
| `iQ` | point index, 0-based, along path then extras |
| `s_path` | cumulative path length, bohr⁻¹ (extras repeat the last path value) |
| `Qx Qy Qz` | $\bQ$ in crystal coordinates |
| `mode` | `interp` on every path row whatever `--vq-mode` is; `extra` for `--extra-q`; `refit` for `both` spot checks |
| `E_1..E_neig` | $\Omega_S(\bQ)$, ascending, eV |

