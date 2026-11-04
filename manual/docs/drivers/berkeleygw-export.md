# BerkeleyGW export and import

Write BerkeleyGW 4.0 files from a finished LORRAX run, and run LORRAX's Σ or BSE on BerkeleyGW's screened interaction.
{: .page-summary }

LORRAX keeps the screened interaction in the ISDF basis, as $V_\bq(\mu,\nu)$ and the static $W_\bq(\mu,\nu;0)$ in the ISDF tensor file `tmp/isdf_tensors_<N_mu>.h5`, with the interpolation vectors $\zeta_{\bq\mu}$ beside it. Two utilities turn that file into the plane-wave files BerkeleyGW reads, and the first of them also converts BerkeleyGW's screening back into it ([Running Σ or the BSE on BerkeleyGW's W](#import)). All of them work on one `WFN.h5` shared by the two codes.

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
- **The reverse direction.** `mode = project` reads BerkeleyGW's files instead ([Running Σ or the BSE on BerkeleyGW's W](#import)).

Every key, with its default, is listed by `python -m gw.epsmat_cli --print-schema`.

## Running Σ or the BSE on BerkeleyGW's W {: #import }

BerkeleyGW and LORRAX describe the screened interaction in two different bases. BerkeleyGW's `epsilon.x` stores $\varepsilon^{-1}_\bq(\bG,\bG')$ as a dense matrix over plane waves, one for each $\bq$, out to `epsilon_cutoff`; for silicon at 25 Ry that is a matrix of about 570 × 570 per $\bq$. LORRAX leaves plane waves behind once the ISDF fit is done. Its self-energy and BSE code expect $W_\bq(\mu,\nu)$, an $N_\mu\times N_\mu$ matrix in the basis of the interpolation vectors $\zeta_{\bq\mu}$, so a BerkeleyGW file cannot be handed to them as it is. It has to be converted first, by contracting the plane-wave matrix with the Fourier components $\tilde\zeta_{\bq\mu}(\bG)$:

$$
W_\bq(\mu,\nu) = V_\bq(\mu,\nu) + \sum_{\bG\bG'} \tilde\zeta^{\,*}_{\bq\mu}(\bG)\,\big[\varepsilon^{-1}_\bq(\bG,\bG')-\delta_{\bG\bG'}\big]\,v(\bq+\bG')\,\tilde\zeta_{\bq\nu}(\bG').
$$

Nothing that Σ needs is lost in this step. LORRAX expands every pair density $\psi^*_{m\bk-\bq}\psi_{n\bk}$ in the same $\zeta_{\bq\mu}$, so the self-energy only ever uses $W$ between two of those vectors, and the sum above computes exactly that. What the conversion has to get right is bookkeeping: the Coulomb potential $v(\bq+\bG)$ that LORRAX itself uses, BerkeleyGW's ordering and orientation of the $\bG$ vectors, and the $\bq$-points that BerkeleyGW did not compute directly.

`gw.epsmat_cli` with `mode = project` performs the conversion one $\bq$ at a time, spread over all ranks, and writes the result into a copy of a LORRAX ISDF tensor file. A GW run restarted from that copy evaluates Σ on BerkeleyGW's $W$ without computing any screening of its own. A BSE run started in the same place uses BerkeleyGW's static $W$ in its direct term.

Three inputs are needed: a `WFN.h5` shared by both codes, a BerkeleyGW epsilon run on it, and a LORRAX run on it that saved its ISDF tensor file. The LORRAX run does not have to compute screening. A run with `compute_mode = x_only` and `write_restart_tensors = true` fits $\zeta$ and builds $V$, which is all the conversion uses.

```ini
# epsmat.in, run in the directory where the Σ or BSE calculation will run
[epsmat]
mode              = project
source_restart    = ../lorrax_x          # the LORRAX run, or its tmp/isdf_tensors_<N_mu>.h5
parent_input_file = ../lorrax_x/gw.in    # the input file of that run
output_dir        = .
bgw_reference     = ../bgw/epsilon       # the directory with eps0mat.h5 and epsmat.h5
```

```
gw.gw_jax -i gw.in           # in ../lorrax_x: x_only, saves its ISDF tensor file
gw.epsmat_cli -i epsmat.in   # here: converts BerkeleyGW's W into tmp/
gw.gw_jax -i gw.in           # here: restart = true, screening_source = bgw_epsmat
bse.bse_jax -i gw.in         # here (optional): BSE on BerkeleyGW's W(0)
```

The GW input file for the third step differs from an ordinary one in two keys, `restart = true` and [`screening_source`](gw/keywords.md#screening_source) `= bgw_epsmat`. With `compute_mode = gn_ppm`, LORRAX fits its plasmon pole to BerkeleyGW's $W(0)$ and $W(i\omega_p)$; with `compute_mode = cohsex` it uses $W(0)$ alone.

The silicon test below shows what to expect: Si 4 × 4 × 4 with spin–orbit coupling, 28 bands, $N_\mu = 960$, the same `WFN.h5` for both codes, run on four GPUs.

| Comparison | Result |
|---|---|
| BerkeleyGW's $W^c(0)$ against LORRAX's own, at $\bq\neq0$ | relative difference $3\times10^{-6}$ to $8\times10^{-6}$ |
| LORRAX's $W$ written by the export, then read back by the import | equal to $10^{-15}$; $10^{-12}$ when the $\bq$ is reached by a symmetry operation |
| GN-PPM $\Sigma^c$ on BerkeleyGW's $W$ against $\Sigma^c$ on LORRAX's $W$, 128 states | 0.66 meV on average, 0.78 meV at most |
| COHSEX Σ, same comparison | 0.70 meV on average, 0.82 meV at most |
| Lowest four BSE excitation energies, same comparison | equal to $10^{-5}$ eV |
| Time for the conversion (8 $\bq$, two frequencies) | 28 s, including start-up |

### Questions about running it {: #import-questions }

**Why would I use BerkeleyGW's W instead of LORRAX's?** Two situations come up most. One is a comparison between the codes: with the screening fixed to BerkeleyGW's, any remaining difference in the quasiparticle or exciton energies comes from Σ or the BSE alone. The other is reuse, when a converged BerkeleyGW epsilon calculation already exists for the system.

**Which BerkeleyGW epsilon runs can be used?** Runs with `frequency_dependence 0` (static) or `frequency_dependence 3` (Godby–Needs, which stores $\omega=0$ and $\omega=i\omega_p$). A static file supports COHSEX and the BSE; a Godby–Needs file supports those and GN-PPM as well. Full-frequency files (`frequency_dependence 2`) are refused.

**Does the BerkeleyGW $\bq$ list have to match LORRAX's?** No. The irreducible list that `kgrid.x` writes for the shared `WFN.h5` works as it is. Each $\bq$ that LORRAX stores is matched to a BerkeleyGW $\bq$ either directly, up to a reciprocal lattice vector, or through a symmetry operation of the crystal, including the phase that a fractional translation contributes. If some $\bq$ cannot be reached either way, the projection stops and lists every missing one.

**What has to be the same in the two calculations?** The `WFN.h5` file and its $\bk$ grid, and the Coulomb truncation: no truncation in BerkeleyGW for [`sys_dim`](gw/keywords.md#sys_dim) `= 3`, and `cell_slab_truncation` for `sys_dim = 2`. BerkeleyGW's `epsilon_cutoff` cannot exceed LORRAX's [`bare_coulomb_cutoff`](gw/keywords.md#bare_coulomb_cutoff), which defaults to the wavefunction cutoff. For GN-PPM, [`ppm_omega_p`](gw/keywords.md#ppm_omega_p) has to equal the imaginary frequency in BerkeleyGW's file. Both codes default to 2 Ry, so this only matters when one of them was changed.

**How is $\bq\to0$ handled?** The $\bG=\bG'=0$ element of $W$ at $\bq\to0$, the head, cannot be stored as an ordinary matrix element because $v(\bq)$ is singular there, so both codes treat it separately. For Σ, LORRAX uses the value of $\varepsilon^{-1}_{00}$ that BerkeleyGW computed at its small $\bq_0$, separately for each frequency, and averages the resulting $W$ over the small cell around Γ, as `sigma.x` also does. The BSE keeps the head that the LORRAX run computed.

**Will the quasiparticle energies then match `sigma.x`?** Closer, but not exactly. Once both codes use the same $W$, the remaining difference comes from how each code evaluates Σ, for instance the frequency broadening and the ISDF representation of the pair densities. In the silicon test, $\Sigma^c$ near the gap (bands 7 to 10) differed from `sigma.x` by 8.1 meV on average with BerkeleyGW's $W$, against 8.7 meV with LORRAX's own.

**Is spin–orbit coupling supported?** Yes. The silicon test above uses two-component wavefunctions.

**How much does the conversion cost?** Far less than the screening it replaces. For each $\bq$ and frequency it performs a few dense matrix products between $N_\mu$ and BerkeleyGW's number of $\bG$ vectors, plus one $N_\mu\times N_\mu$ eigenvalue solve for the passivity check described below; no χ⁰ and no Dyson solve are needed. The memory needed per GPU is printed before the first $\bq$; in the silicon test it was 0.15 GiB.

**How can I check that the conversion worked?** `output_dir/epsmat_project.json` lists, for every stored $\bq$, which BerkeleyGW $\bq$ and which symmetry operation produced it. When the producer run computed its own $W(0)$, the file also gives the relative difference between the two codes' $W^c$; values near $10^{-5}$ at $\bq\neq0$ are normal, while Γ differs by a few percent because BerkeleyGW's Γ matrix includes the head–wing coupling that LORRAX treats separately. The run also checks every converted matrix for passivity: at each $\bq$ and frequency, the screening may only reduce the bare interaction.

**Are BerkeleyGW's files modified?** No. The projection reads `eps0mat.h5` and `epsmat.h5` and writes only to `output_dir`. The producer's ISDF tensor file is copied, never changed.

!!! refuses
    The projection stops with `GATE epsmat_project_*` when `eps0mat.h5` is missing; when a file holds something other than $\varepsilon^{-1}$ for one spin matrix, or has an unfinished $\bq$ row; when the $\bq$ grid, cell volume or Coulomb truncation differs from the LORRAX run; when `frequency_dependence` is not 0 or 3, or the two files list different frequencies; when a stored $\bq$ has no BerkeleyGW counterpart; when BerkeleyGW's $\bG$ vectors reach past LORRAX's Coulomb cutoff; or when a converted matrix fails the passivity check.

    The GW run stops with `GATE screening_source_*` when `compute_mode` is not `cohsex` or `gn_ppm`, when `restart = false`, for a self-consistent run, for `screening_diagrams` other than `w_rpa`, for `bispinor = true`, when the ISDF tensor file lacks the converted $W$ it needs (GN-PPM from a static file, for example), when `ppm_omega_p` differs from the file's frequency, for GN-PPM on a WFN without time-reversal symmetry, and when the converted $W$ is not in the same centroid order as the run's $V$.

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
