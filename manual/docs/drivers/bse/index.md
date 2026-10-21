---
driver: bse
page: overview
---

# BSE driver

`bse.bse_jax` Optical excitons from the Bethe–Salpeter equation on the GW restart bundle.
{: .page-summary }

<div class="chain-map" data-driver="bse" data-page="overview"></div>

Solves the Bethe–Salpeter equation for $\bQ = 0$ excitons in the transition basis $|cv\bk\rangle$, with the static screened interaction from the GW run. It is the last stage of the chain; the absorption spectrum $\varepsilon_2(\omega)$ comes from its companion entry point `bse.absorption_haydock`.
{: .lead }

<div class="glance" markdown>

Invoke
:   `python -m bse.bse_jax -i gw.in --tda --bse`; spectrum: `python -m bse.absorption_haydock`

Reads
:   GW deck, restart bundle `isdf_tensors_<N_mu>.h5`, `WFN.h5`, centroid table, `eqp1.dat`, `dipole.h5`

Writes
:   `bse.out`, `eigenvectors.h5`; `absorption_haydock.h5`, `absorption_haydock_b{1,2,3}_eh.dat`

Hamiltonian
:   Tamm–Dancoff (`--tda --bse`, production) or full, with the coupling block (`--bse`); spin singlet for a scalar WFN, spinors handled directly; no triplet kernel ([Kernel and Hamiltonian](#kernel))

Solver
:   Davidson · Lanczos · thick restart · dense full BSE · Haydock

</div>

## What it computes {: #what }

<div class="stages" markdown>

1. **Optional interpolation of bands and interactions to the BSE k-grid**
   Read $\psi_{n\bk}(\br_\mu)$, $E^{\mathrm{QP}}_{n\bk}$, $V_\bq(\mu,\nu)$ and $W_\bq(\mu,\nu;0)$ from the bundle and add the $\bq \to 0$ head. Under `bse_k_grid` the bands are carried to the fine grid $\bk_{\mathrm{fi}}$ by H-transform interpolation, and $W$ is split into a short-range part, smooth in $\bq$ and Fourier-interpolated, and a long-range part carried by the plane-wave channels $\bq+\bG_a\to0$: their dressed form factors and polarizability are fitted in $\bq$, screened, switched off smoothly inside the zone, and evaluated at every fine $\bq$ (`w_head_densify = srlr`; [Theory](theory.md)).
2. **Matrix-free product $HX$**
   The Hamiltonian, Tamm–Dancoff or full ([below](#kernel)), is never stored. Each trial vector gets the diagonal $E^{\mathrm{QP}}_{c\bk} - E^{\mathrm{QP}}_{v\bk}$, the exchange term through $V_{\bq=0}$ and the direct term as a k-convolution with $W$.
3. **Eigensolve**
   The route selected on the command line returns exciton energies $\Omega_S$ and, on the default eigensolve route, amplitudes $A^S_{cv\bk}$ in `eigenvectors.h5`.
4. **Dipoles and spectrum**
   Velocity matrix elements from `dipole.h5` are put in position form. `--dipole` stores per-state dipoles $d^\alpha_S$, and `bse.absorption_haydock` computes $\varepsilon_2(\omega)$ by a continued fraction, without eigenvectors.

</div>

<figure class="flowchart" markdown>
[![BSE driver overview](../../assets/figures/b1.png)](../../assets/figures/b1.png)
<figcaption><strong>Fig. B1.</strong> The production route (<code>--tda --bse</code>) in four stages: the bands optionally carried to the BSE k-grid by the H-transform; $W$ optionally carried there in two branches, the short-range part Fourier-interpolated through $\bR$ and the long-range part built on G-space plane-wave channels ($\bq+\bG_a\to0$) fitted in $\bq$, screened and windowed, then added and assembled into the Tamm–Dancoff Hamiltonian; the matrix-free product $HX$, with the exchange (top) and direct (bottom) branches; the eigensolve and the spectrum. The Objects cards name the file each quantity comes from or goes to.</figcaption>
</figure>

!!! note
    Production runs set `--tda --bse`. `bse.bse_jax` never writes $\varepsilon_2$, and the spectrum comes from `bse.absorption_haydock`.

## Running {: #run }

Run in the GW run directory, against the GW deck. Exactly one `isdf_tensors_*.h5` may sit there or in its `tmp/`.

=== "Local"
    ```bash
    uv run python -m bse.bse_jax -i gw.in --tda --bse --n-val 4 --n-cond 8 --n-eig 10
    ```
=== "Perlmutter"
    ```bash
    srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh python -u -m bse.bse_jax -i gw.in --tda --bse --n-val 4 --n-cond 8 --n-eig 10
    ```
=== "Frontera"
    ```bash
    # copy config/frontera/templates/gw_dev.sbatch; change its python line to -m bse.bse_jax with the flags above
    sbatch bse_dev.sbatch
    ```

A minimal production run: QP energies, eigenvectors with their dipoles, then the spectrum.

```bash
python -m bse.bse_jax -i gw.in --tda --bse --n-val 4 --n-cond 8 \
    --eqp eqp1.dat --n-eig 20 --write-eigs --dipole dipole.h5
python -m bse.absorption_haydock -i gw.in --n-val 4 --n-cond 8 --n-occ 26 \
    --eqp eqp1.dat --dipole dipole.h5 --V-cell 1270.5 --eta-eV 0.05
```

The deck is the GW deck, unchanged. To solve on a finer grid, add the BSE k-grid lines:

```ini
wfn_file        = WFN.h5
centroids_file  = centroids_frac_960.txt
nval            = 8
ncond           = 16
bse_k_grid      = 12 12 1
w_head_densify  = c1
```

Under `bse_k_grid` the dipoles are carried to the fine grid with the bands, so the run above works unchanged on that deck ([`bse_k_grid`](keywords.md#bse_k_grid)).

## Choosing a solver {: #solver }

Production runs use the default route, the eigensolve, and `--kpm-dos` computes only the exciton density of states. On the eigensolve route `--tda` selects the iterative solvers and `--solver` picks one, Davidson by default. Without `--tda` the dense full-BSE solver runs. Expand a row for the method, its flags and its limits.

???+ treatment production "davidson · Block Davidson · TDA eigenpairs"
    Residual corrections preconditioned by the exact diagonal of $H$, on a block of the `--n-eig` lowest roots plus `--davidson-margin` margin roots. The solve stops when every reported root's residual is below `--davidson-tol-mev` (0.01 meV) and refuses if the iteration cap comes first. The default solver since 2026-10-20.

    ```bash
    python -m bse.bse_jax -i gw.in --tda --bse --n-eig 10
    ```

??? treatment supported "lanczos · Block Lanczos · TDA eigenpairs"
    Lanczos on the Tamm–Dancoff block $A$ with full CGS2 reorthogonalization. The lowest Ritz pairs of the tridiagonal (block-tridiagonal) projection are the excitons. It runs a fixed Krylov dimension and has no residual test.

    ```bash
    python -m bse.bse_jax -i gw.in --tda --bse --solver lanczos --n-eig 10
    ```

    **Note** Keep `--n-reorth -1` (full) for degenerate spinor spectra.

??? treatment supported "trlan · Thick-restart Lanczos · TDA eigenpairs"
    The same Krylov convergence from a basis capped at `--trlan-m-max` slots, restarted from the lowest Ritz vectors. For production sizes where the full Krylov basis does not fit in memory.

    ```bash
    python -m bse.bse_jax -i gw.in --tda --bse --solver trlan --n-eig 20
    ```

??? treatment supported "dense · Full BSE, definite pencil · non-TDA eigenpairs"
    Builds $A$ and $B$ densely from the matvec and solves the Hermitian-definite pencil, returning paired $(X, Y)$. Limited to a transition dimension of 4096; reads only `--n-eig` and the kernel flags.

    ```bash
    python -m bse.bse_jax -i gw.in --bse --n-eig 10
    ```

    **Note** Refuses a $W$ that fails the $\bq \leftrightarrow -\bq$ reciprocity preflight.

??? treatment production "haydock · Continued fraction · absorption spectrum"
    `bse.absorption_haydock`: Lanczos from the dipole vector $|d_\alpha\rangle$, one recursion per Cartesian direction, gives $\varepsilon_2(\omega)$ as a continued fraction. Always the TDA kernel $D + w_x V - W$; needs no eigenvectors and at least two devices.

    ```bash
    python -m bse.absorption_haydock -i gw.in --n-val 4 --n-cond 8 --n-occ 26 --V-cell 1270.5
    ```

??? treatment supported "kpm · Kernel polynomial DOS · exciton density of states"
    `--kpm-dos`: Chebyshev moments of $H$ with Jackson damping, written to `bse_dos_kpm.png` and `bse_dos_kpm.npz`. No eigenpairs.

    ```bash
    python -m bse.bse_jax -i gw.in --kpm-dos --tda --bse
    ```

## Kernel and Hamiltonian {: #kernel }

Two flags choose the operator on every route. Both defaults differ from BerkeleyGW's.

| Flags | Hamiltonian | BerkeleyGW |
|---|---|---|
| `--tda --bse` | $A = D + w_x V - W$ | default TDA, `spin_singlet` (spinors: spin kernel 3) |
| `--bse` | $\begin{pmatrix}A & B\\ -B^* & -A^*\end{pmatrix}$ | full BSE |
| `--tda` | $D + w_x V$ (RPA kernel, $W$ never built) | `local_fields` |
| none | full problem with the RPA kernel | — |

`--rpa` forces the RPA kernel even with `--bse`. Spin: a scalar WFN gives the singlet excitons, with exchange weight $w_x = 2$; there is no triplet kernel. A spinor WFN needs no spin channel: the spinor states carry it, and $w_x = 1$. The equations are on the [Theory](theory.md) page.

## Excitons on QSGW states {: #qsgw }

`--eqp` replaces the diagonal energies only. ψ always comes from the restart bundle, which holds the states of the GW deck's WFN. Two routes follow:

- **QSGW energies, DFT states.** Run the BSE on the self-consistent run's own bundle with `--eqp eqp1.dat`. A converged run on a route that stores W0 writes the final map's $W(0)$ there; a QSGW bundle without it is not rebuilt (`GATE bse_static_w_sc_state`).
- **QSGW states.** Run GW on the self-consistent run's `WFN_qp.h5` (`wfn_file = WFN_qp.h5`) and solve the BSE on that bundle without `--eqp`, with `--dipole dipole_qsgw.h5`. `--eqp` on a stamped `WFN_qp.h5` refuses, and so does a DFT `dipole.h5` with it (`GATE dipole_basis`).

A `WFN_qp.h5` written before 2026-09-30 from a WFN that stores both $\bk$ and $-\bk$ has broken rows; rewrite it with `python -m postprocess.rotate_wfn_to_qp WFN.h5 qp_wfn_rotations.h5`.

## Converging a spectrum {: #converge }

Measured accuracy and cost of the choices a production run makes:

| Choice | Measured |
|---|---|
| Haydock steps, `--n-iter` | CrI₃ 8×8×1 with SOC (bare $V$ as $W$) against a 500-state sum over states: 200 steps (the default) agree to $10^{-5}$ of max $\varepsilon_2$; 100 steps are 0.2–0.5 % off. |
| Sum over states from `eigenvectors.h5` | Converges slowly in peak height with the number of states. Compare spectra through `bse.absorption_haydock`. |
| Stored or rebuilt $W(0)$ | Against a stored shared-pole W0 on one bundle, the rebuild moved the ten lowest $\Omega_S$ by at most 0.0078 meV (MoS₂ 3×3) and 0.00002 meV (Si 4×4×4), and doubled the wall time (9.8 → 19.3 s, 10.5 → 20.3 s). |
| Matvec cost | CrI₃ 8×8×1 with SOC, 8 valence × 14 conduction bands, four A100-40GB: one Haydock step (three trials) 13.3 ms, 100 steps 1.28 s, 2159 MiB peak per rank. |

A downfolded bundle is validated by solving the same BSE on the parent and on the child ([downfold](../../utilities/downfold.md#mu-small)).

## Refusals {: #refusals }

!!! refuses
    - No `-i`; an unknown flag; the retired `--ring-timing`.
    - A flag set away from its default on a route that does not read it (for example `--eqp`, `--n-eig` or `--n-occ` with `--kpm-dos`), or two route flags together.
    - `--dipole` without `--write-eigs` and `--tda`.
    - `--px`/`--py` not given together, not square, or not using every device.
    - No `isdf_tensors_*.h5`, or more than one, in the run directory and its `tmp/` (`GATE bse_restart_ambiguous`).
    - A bundle whose centroid hash, band window, $N_\mu$, parent rows or QP-state stamp does not match the deck and its WFN.
    - A bundle without stored W0 that cannot be rebuilt: no `G0_mu_nu`, a four-current bundle, a QSGW deck, or no WFN/centroid receipt.
    - A window edge inside a degenerate multiplet under `--band-degeneracy strict` (the default).
    - A window band outside the ζ-fit legs of the GW run (`BseWindowOutsideZetaTrainingError`).
    - `--eqp` on a stamped QP WFN, or an `eqp1.dat` whose k blocks are not the WFN's irreducible wedge.
    - A `dipole.h5` whose `basis` stamp differs from the WFN's.
    - `bse_k_grid` with an axis smaller than the coarse grid; `w_head_densify = legacy`; `srlr` without `W0_lr` in the bundle.
    - Full BSE above transition dimension 4096, or with a non-reciprocal $W$.
    - `bse.absorption_haydock` on fewer than two devices.
    - The environment variable `LORRAX_LANCZOS_REORTH` (retired).
