# BerkeleyGW export and import

Write BerkeleyGW 4.0 files from a finished LORRAX run, and run LORRAX's Σ or BSE on BerkeleyGW's screened interaction.
{: .page-summary }

LORRAX keeps the screened interaction in the ISDF basis, as $V_\bq(\mu,\nu)$ and the static $W_\bq(\mu,\nu;0)$ in the ISDF tensor file `tmp/isdf_tensors_<N_mu>.h5`, with the interpolation vectors $\zeta_{\bq\mu}$ beside it. Two utilities turn that file into the plane-wave files BerkeleyGW reads, and the first of them also reads BerkeleyGW's screening back into it ([Importing BerkeleyGW's W](#import)). All of them work on one `WFN.h5` shared by the two codes.

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
parent_input_file = ../gw/gw.in    # the input file that produced it
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
- **$\bq\to0$.** The head comes from the ISDF tensor file's $q\to0$ response tensor $S$ and the wings from its dipole wings (`q0_head = s_tensor`, `q0_wings = dipole`, the defaults).
- **The reverse direction.** `mode = project` reads BerkeleyGW's files instead ([Importing BerkeleyGW's W](#import)).

Every key, with its default, is listed by `python -m gw.epsmat_cli --print-schema`.

## Importing BerkeleyGW's W {: #import }

`gw.epsmat_cli` with `mode = project` reads BerkeleyGW's `eps0mat.h5` and `epsmat.h5`, computed on the same `WFN.h5`, and projects the screened interaction they describe onto the ISDF vectors of a finished LORRAX run. LORRAX's GW driver can then evaluate Σ on BerkeleyGW's $W$, and its BSE can use BerkeleyGW's static $W$, without building screening of its own.

```ini
[epsmat]
mode              = project
source_restart    = ../lorrax_run          # a finished LORRAX run, or its ISDF tensor file
parent_input_file = ../lorrax_run/gw.in    # the input file that produced it
output_dir        = .                      # the directory of the Σ or BSE run; never the producer's
bgw_reference     = ../bgw/epsilon         # holds eps0mat.h5 and epsmat.h5
```

The producer needs only $V$ and $\zeta$, so a run with `compute_mode = x_only` and `write_restart_tensors = true` is enough. The projection copies its ISDF tensor file to `output_dir/tmp/` and writes into the copy:

| Dataset | Content | Written when |
|---|---|---|
| `W0_qmunu` | $W_\bq(\mu,\nu;0)$ | always |
| `Wp_qmunu` | $W_\bq(\mu,\nu;i\omega_p)$ | the epsilon ran with `frequency_dependence 3` (Godby–Needs) |
| `bgw_epsinv_head` | BerkeleyGW's $\varepsilon^{-1}_{00}(\bq_0,\omega)$ at each frequency | always |

BerkeleyGW's irreducible $\bq$ list is accepted as it is. Each stored LORRAX $\bq$ is matched to a $\bq$ of `epsmat.h5` up to a reciprocal lattice vector, or else to a symmetry image under the operations of the WFN. Γ comes from `eps0mat.h5` at BerkeleyGW's $\bq_0$, without its $\bG=0$ row and column, since the head is applied separately. `output_dir/epsmat_project.json` records the row used for every $\bq$ and, when the producer has its own $W(0)$, the relative difference between the two codes' $W^c$.

**Σ on BerkeleyGW's W.** A GW input file in `output_dir` with `restart = true` and [`screening_source`](gw/keywords.md#screening_source) `= bgw_epsmat` runs Σ with no $\chi^0$ and no Dyson solve. `compute_mode = gn_ppm` fits the plasmon pole to $W(0)$ and $W(i\omega_p)$, and `compute_mode = cohsex` uses $W(0)$. [`ppm_omega_p`](gw/keywords.md#ppm_omega_p) must equal the epsilon's imaginary frequency, 2 Ry by default in both codes. The $\bq\to0$ head at each frequency is BerkeleyGW's $\varepsilon^{-1}_{00}(\bq_0,\omega)$, averaged over the Γ cell as under `wcoul0_source = epshead`.

**The BSE on BerkeleyGW's W.** `bse.bse_jax` in `output_dir` uses BerkeleyGW's $W(0)$ for the direct term, with no new key. Its $\bq\to0$ head remains the producer's.

```
gw.gw_jax -i gw.in              # producer, any run with write_restart_tensors = true
gw.epsmat_cli -i epsmat.in      # mode = project
gw.gw_jax -i gw.in              # in output_dir: restart = true, screening_source = bgw_epsmat
bse.bse_jax -i gw.in            # in output_dir: the BSE on BerkeleyGW's W(0)
```

On Si 4×4×4 with spin–orbit coupling, GN-PPM $\Sigma^c$ on BerkeleyGW's $W$ differs from $\Sigma^c$ on LORRAX's own $W$ by 0.66 meV on average over 128 states, and COHSEX by 0.70 meV. An export followed by an import returns LORRAX's $W$ to $10^{-15}$ at every $\bq\neq0$, and to $10^{-12}$ through symmetry images.

!!! refuses
    - The projection, by `GATE epsmat_project_*`: no `eps0mat.h5`; a file that is not $\varepsilon^{-1}$ with one spin matrix; an unfinished $\bq$ row; a $\bq$ grid, cell volume or Coulomb truncation that differs from the producer's; `frequency_dependence` other than 0 or 3; frequency tables that disagree between the two files; a stored $\bq$ with no image in `epsmat.h5`; BerkeleyGW G vectors outside the producer's $\zeta$ sphere; a projected correlation outside $[-1,0]$.
    - The GW driver, by `GATE screening_source_*`: a `compute_mode` other than `cohsex` or `gn_ppm`; `restart = false`; a self-consistent run; `screening_diagrams` other than `w_rpa`; `bispinor = true`; an ISDF tensor file without the projected $W$ the treatment needs, so GN-PPM from a static epsilon refuses; `ppm_omega_p` different from the file's frequency; GN-PPM on a time-reversal-broken WFN; a projected $W$ whose centroid order does not match the run's $V$.
    - Full-frequency epsilons, the Hybertsen–Louie and tangential models, and self-consistent runs are not supported.

## BSE kernel {: #bsemat }

```
# input keys in the -i input file: number_val_bands_fine = 4, number_cond_bands_fine = 4
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
- **The same Coulomb interaction.** The export rebuilds $V$ from $\zeta$ and refuses unless it equals the stored $V$ to $10^{-9}$, so a GW run made with [`use_bgw_vcoul`](gw/keywords.md#use_bgw_vcoul) or a Monte Carlo average placement cannot be exported.
- **Spinors.** Two-component (spin–orbit) wavefunctions are supported, with bands counted as spinor bands, as in BerkeleyGW.

A worked comparison of the two codes on the same mean field, with every input file, is in the repository's [`examples/Si_soc`](https://github.com/jackmcarthur/lorrax/tree/main/examples/Si_soc).
