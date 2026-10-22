# BerkeleyGW export

Write BerkeleyGW 4.0 files from a finished LORRAX run, so that BerkeleyGW's own codes can use LORRAX's screening and kernel.
{: .page-summary }

LORRAX keeps the screened interaction in the ISDF basis, as $V_\bq(\mu,\nu)$ and the static $W_\bq(\mu,\nu;0)$ in the restart bundle `tmp/isdf_tensors_<N_mu>.h5`, with the interpolation vectors $\zeta_{\bq\mu}$ beside it. Two utilities turn that bundle into the plane-wave files BerkeleyGW reads. Both run after the GW driver, on the same `WFN.h5`.

| Utility | Writes | Read by |
|---|---|---|
| `gw.epsmat_cli` | `eps0mat.h5`, `epsmat.h5` | `sigma.x`, `kernel.x` |
| `bse.bsemat_cli` | `bsemat.h5` | `absorption.x` |

## Inverse dielectric matrix {: #epsmat }

```
python -m gw.epsmat_cli -i epsmat.in
```

```ini
[epsmat]
source_restart    = ../gw          # the finished GW run directory
parent_input_file = ../gw/gw.in    # the deck that produced it
output_dir        = eps
```

The files are static and complex, one $\bq$ at a time on BerkeleyGW's G order. With $Z_{\bG\mu}=\tilde\zeta_{\bq\mu}(\bG)$ and $v=v(\bq+\bG)$, the written matrix is

$$
\varepsilon^{-1}_{\bG\bG'}-\delta_{\bG\bG'}=v(\bq+\bG)\left[Z\,\chi^{\mu}Z^\dagger\right]_{\bG\bG'},\qquad
\chi^{\mu}=V^{+}\,(W-V)\,V^{+}
$$

This is not a fit. It is the plane-wave RPA inverse dielectric matrix of the polarizability the GW run used, recovered from the stored $V$ and $W(0)$. Its rank is at most $N_\mu$ per $\bq$: outside the span of $\zeta$, $\varepsilon^{-1}=1$.

- **G cutoff.** `epsmat_ecut` (Ry) sets $|\bq+\bG|^2\le E_\varepsilon$; `0`, the default, keeps the full Coulomb sphere. A smaller cutoff writes the leading block of the full inverse; BerkeleyGW's epsilon at a smaller cutoff instead inverts the smaller block of $\varepsilon$. The two differ by the local fields beyond the cut.
- **BerkeleyGW reference.** `bgw_reference`, a directory holding BerkeleyGW's own `eps0mat.h5`/`epsmat.h5` on the same WFN, supplies the $\bq$ list, $\bq_0$, cutoffs and G order, and the export checks them.
- **$\bq\to0$.** The head comes from the bundle's $q\to0$ response tensor $S$ and the wings from its dipole wings (`q0_head = s_tensor`, `q0_wings = dipole`, the defaults).
- **The reverse direction.** `mode = project` reads BerkeleyGW's files and writes a copy of the bundle whose $W_\bq(\mu,\nu;0)$ is BerkeleyGW's screening projected on $\zeta$, so LORRAX's BSE can run on BerkeleyGW's $\varepsilon^{-1}$.

Every key, with its default, is listed by `python -m gw.epsmat_cli --print-schema`.

## BSE kernel {: #bsemat }

```
# deck keys in the -i deck: number_val_bands_fine = 4, number_cond_bands_fine = 4
python -m bse.bsemat_cli -i ../gw/gw.in --wfn-co WFN_co.h5 -o bsemat.h5
```

The file holds the direct and exchange kernels of the static BSE on the coarse grid, in BerkeleyGW 4.0 layout. `absorption.x` reads it in place of `kernel.x`'s output. `--wfn-co` is the explicit full-grid coarse WFN that absorption reads (`no_symmetries_coarse_grid`). The kernel is written in that file's coefficient gauge and k order, so BerkeleyGW's interpolation to the fine grid applies unchanged.

- `--bgw-eps epsmat.h5` places the $\bq=0$ head exactly as `kernel.x` does at zone-boundary ties.
- `extended_kernel = true` writes every band block, for absorption beyond the Tamm–Dancoff approximation.
- `exciton_Q_shift Q1 Q2 Q3` writes the finite-momentum kernel of the pairs $|v\,\mathbf k-\mathbf Q,\,c\,\mathbf k\rangle$.
- Several values of `--kernel`, `--gauge`, `--q0-body` or `--mu-small` write one file each from a single load.

## What carries over {: #interop }

- **One mean field.** Both codes must read the same `WFN.h5`, and BerkeleyGW's coarse grid must be the explicit, unreduced `WFN_co.h5` passed to `--wfn-co`.
- **Static screening only.** The exported $\varepsilon^{-1}$ is at $\omega=0$. It serves BerkeleyGW's plasmon-pole Σ (`sigma.x` builds its own pole model from it) and the static BSE kernel, but not a full-frequency Σ.
- **The same Coulomb interaction.** The export rebuilds $V$ from $\zeta$ and refuses unless it equals the bundle's $V$ to $10^{-9}$, so a GW run made with [`use_bgw_vcoul`](gw/keywords.md#use_bgw_vcoul) or a Monte Carlo average placement cannot be exported.
- **Spinors.** Two-component (spin–orbit) wavefunctions are supported, with bands counted as spinor bands, as in BerkeleyGW.

A worked comparison of the two codes on the same mean field, with every input file, is in the repository's [`examples/Si_soc`](https://github.com/jackmcarthur/lorrax/tree/main/examples/Si_soc).
