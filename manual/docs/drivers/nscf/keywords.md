---
driver: nscf
page: keywords
---

# Input keywords

`psp.run_nscf` Every deck key and command-line flag run_nscf reads, grouped by what it controls.
{: .page-summary }

<div class="chain-map" data-driver="nscf" data-page="keywords"></div>

The deck is an INI file with one `[nscf]` section, keys written `key = value`; paths resolve relative to the deck. Defaults are read from the keyword schema (`src/psp/nscf_input.py`) and the driver argparse (`src/psp/run_nscf.py`). A command-line flag of the same meaning overrides the deck value. Without `-i`, `--save` is required and the same defaults apply.

Every Hamiltonian parameter not listed here comes from the QE `.save` and cannot be changed from the deck: the cutoffs `ecutwfc` and `ecutrho`, the FFT grid, the structure and symmetry, `nelec`, the spinor representation (`noncolin`, `lspinorb`) and `assume_isolated`.

Required
{: .kw-group }

**Files**

### `save_dir` {: .kw data-type="path" data-default="required" }
QE `<prefix>.save` directory: structure, symmetry, cutoffs, FFT grid and $\rho_{\mathrm{val}}$. QE `outdir`/`prefix`; CLI `--save`.

Without a deck it is given as `--save`.
{: .kw-details }

**Bands**

### `nbnd` {: .kw data-type="int" data-default="required" }
Bands $N_b$ solved and written per k-point. QE `nbnd`; CLI `--nbnd`.

Every band must converge, unlike QE's unconverged buffer bands. Set it above the GW `number_bands` and off a degenerate multiplet.
{: .kw-details }

**k-points**

### `kgrid` {: .kw data-type="int[3]" data-default="required" }
Γ-centred, unshifted Monkhorst–Pack grid. QE `K_POINTS automatic N1 N2 N3 0 0 0`; CLI `--nk`.

Reduced to the IBZ with the `.save` symmetry operations and time reversal, and stored as `mf_header/kpoints/kgrid`. Use the unshifted GW grid.
{: .kw-details }

Optional
{: .kw-group }

**Files**

### `-i` {: .kw data-type="path" data-default="—" }
CLI only. The deck. Alias `--input`.

### `output` {: .kw data-type="path" data-default="WFN.h5" }
Output wavefunction file. pw2bgw `wfng_file`; CLI `-o`, `--output`.

### `--pseudo_dir` {: .kw data-type="path" data-default="save_dir" }
CLI only. Directory searched for `*.upf`, then `*.UPF`. QE `pseudo_dir`.

Every file found is loaded and keyed by its UPF element symbol.
{: .kw-details }

**k-points**

### `nosym` {: .kw data-type="bool" data-default="false" }
Use the full grid: identity only, no time-reversal folding. QE `nosym = .true.` with `noinv = .true.`; CLI `--nosym`.

The `WFN.h5` symmetry block is then the identity alone. The CLI flag can switch it on, not off.
{: .kw-details }

### `--ref_wfn` {: .kw data-type="path" data-default="—" }
CLI only. Take the k-points and weights from `mf_header/kpoints/{rk,w}` of this `WFN.h5` instead of `kgrid`.

`kgrid` is still written to the header. Use it to reproduce an existing k set exactly.
{: .kw-details }

**Pseudobands**

### `pseudobands` {: .kw data-type="bool" data-default="false" }
Fill the spectrum above the Davidson bands with pseudobands, written to `<output stem>_pseudobands.h5`. CLI `--pseudobands`.

Comparable in purpose to BerkeleyGW's pseudobands utility, with a different construction. The CLI flag can switch it on, not off.
{: .kw-details }

### `pb_n_prot` {: .kw data-type="int" data-default="0" }
Lowest Davidson bands kept exactly. `0` derives the count from the crossover energy.

The automatic count compares absolute QE energies with a Fermi level fixed at 0 Ry, so set it explicitly. v2 only.
{: .kw-details }

### `pb_n_windows` {: .kw data-type="int" data-default="50" }
Target number of DOS-weighted spectral windows.

### `pb_k` {: .kw data-type="int" data-default="6" }
Pseudobands per spectral window (the Galerkin block size).

### `pb_M_max` {: .kw data-type="int" data-default="1500" }
Chebyshev filter order $M$.

Sets the filter resolution $\pi B/M$, with $B$ the spectral width of $H_\bk$.
{: .kw-details }

### `pb_F` {: .kw data-type="float" data-default="0.10" }
Window ratio $F$.

In v2 it sets only the crossover energy $\pi B/(MF)$ that fixes the protected count when `pb_n_prot = 0`; in v1 it also sets the geometric window spacing.
{: .kw-details }

### `pb_version` {: .kw data-type="int" data-default="2" }
Pseudoband algorithm.

`2` is Galerkin–Ritz windows with Gauss-quadrature energies; any other value selects v1, stochastic plus Ritz.
{: .kw-details }

### `--pb-seed` {: .kw data-type="int" data-default="0" }
CLI only. Seed of the stochastic start vectors; k-point $i$ uses `i + 10000·seed`.

Vary it to average the stochastic noise.
{: .kw-details }

**Solver**

### `tol` {: .kw data-type="float" data-default="1e-8" }
Davidson residual threshold, Ry. CLI `--tol`.

Every band must satisfy $\|(H_\bk - E_n)x_n\| < \texttt{tol}\cdot\max(1,|E_n|)$; closer in role to QE `diago_thr_init` than to `conv_thr`. `--tol 0` falls back to the deck value.
{: .kw-details }

These solver parameters are fixed in code:

| Parameter | Value | Source |
|---|---|---|
| subspace capacity before restart | $10\,N_b$ | `src/solvers/davidson.py:23`, `src/psp/run_nscf.py:311` |
| block size | $N_b$ | `src/psp/run_nscf.py:309-312` |
| maximum iterations | 100 | `src/psp/run_nscf.py:343,354` |
| stall patience (iterations without a 1 % residual decrease) | 20 | `src/psp/run_nscf.py:343,354` |
| preconditioner floor on $\lvert h_\bG - E_n\rvert$ | 0.01 Ry | `src/psp/dft_precond.py:29` |
| initial plane waves | $2N_b$ per spinor component | `src/psp/dft_precond.py:59` |
| $H_\bk$ application batch | 32 vectors | `src/psp/dft_operators.py:892` |

**Pseudobands-only mode**

These keys reuse deterministic bands from an earlier run and only build the pseudobands.

### `--no-davidson` {: .kw data-type="flag" data-default="off" }
CLI only. Skip Davidson, read the deterministic bands from `--pb-wfn` (or from `wfn_file` when `rho_from_wfn = true`), and run only the pseudobands stage.

Refuses without `pseudobands`, or with neither file named.
{: .kw-details }

### `--pb-wfn` {: .kw data-type="path" data-default="—" }
CLI only. Deterministic-band `WFN.h5` for pseudobands-only mode; overrides `wfn_file`.

### `rho_from_wfn` {: .kw data-type="bool" data-default="false" }
Make `wfn_file` the deterministic-band input of pseudobands-only mode.

Despite the name, the density always comes from the `.save`. It has an effect only with `--no-davidson`.
{: .kw-details }

### `wfn_file` {: .kw data-type="path" data-default="deck directory" }
Existing `WFN.h5` with deterministic bands, read when `rho_from_wfn = true`.

It must hold more bands than occupied states and the same G spheres as this run.
{: .kw-details }

