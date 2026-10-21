---
driver: exciton-bands
page: theory
---

# Theory

`bse.exciton_bands` The equations each stage evaluates.
{: .page-summary }

<div class="chain-map" data-driver="exciton-bands" data-page="theory"></div>

<div class="section-cards"></div>

## <span class="n">1</span> The finite-Q BSE {: #bse-q }

An exciton of momentum $\bQ$ is built from transitions that remove an electron at $\bk$ and add one at $\bk+\bQ$. The driver solves the Tamm–Dancoff problem in that basis at every point of a path.
{: .intro }

### <span class="n">1.1</span> The Q path {: #path }

`fh_interp.generate_kpath_from_qe_segments`
{: .in-code }

The deck's `K_POINTS` block lists corners $\bQ_0,\dots,\bQ_m$ in crystal coordinates, each with a count $n_i$. Segment $i$ runs from $\bQ_{i-1}$ to $\bQ_i$ and takes the count on the line of the corner where it ends, raised to the floor `--q-per-segment`:

$$
\bQ_{i,t} = \Big(1-\frac{t}{n_i}\Big)\bQ_{i-1} + \frac{t}{n_i}\,\bQ_i, \qquad t = 1,\dots,n_i, \qquad n_i \leftarrow \max(n_i,\, n_\mathrm{floor}) \tag{1.1}
$$

The path starts at $\bQ_0$ and the count on its line is unused. The path coordinate is the cumulative Cartesian length, in bohr⁻¹. `--extra-q` points are appended after the path and solved in the same scan.

### <span class="n">1.2</span> Pair basis and Hamiltonian {: #ham }

`bse.exciton_bands`
{: .in-code }

The pair state $|cv\bk\rangle$ holds a hole in valence band $v$ at $\bk$ and an electron in conduction band $c$ at $\bk+\bQ$; the electron leg carries the shift. An exciton is $|S\bQ\rangle = \sum_{cv\bk} A^{S\bQ}_{cv\bk}\,|cv\bk\rangle$, and in the Tamm–Dancoff approximation:

$$
\langle cv\bk|H_\bQ|c'v'\bk'\rangle = \big(E^{\mathrm{QP}}_{c\,\bk+\bQ}-E^{\mathrm{QP}}_{v\bk}\big)\,\delta_{cc'}\delta_{vv'}\delta_{\bk\bk'} + \langle cv\bk|V_\bQ - W|c'v'\bk'\rangle, \qquad H_\bQ A^{S\bQ} = \Omega_S(\bQ)\,A^{S\bQ} \tag{1.2}
$$

$v$ runs over the `--n-val` highest occupied bands and $c$ over the `--n-cond` lowest empty ones, on the $N_k$ points of the restart's $\bk$ grid. At $\bQ = 0$ this is the optical TDA Hamiltonian of the [`bse`](../bse/theory.md) driver. There is no non-TDA route.

### <span class="n">1.3</span> Exchange and direct kernels at Q {: #kernels }

Both kernels are contracted on the ISDF points $\br_\mu$. The exchange pairs the two transition densities through the bare Coulomb at $\bQ$; the direct term pairs electron with electron and hole with hole through the static screened interaction:

$$
\begin{aligned}
\langle cv\bk|V_\bQ|c'v'\bk'\rangle &= \frac{w_x}{N_k}\sum_{\mu\nu}\psi^*_{c\,\bk+\bQ}(\br_\mu)\psi_{v\bk}(\br_\mu)\,V_\bQ(\mu,\nu)\,\psi_{c'\,\bk'+\bQ}(\br_\nu)\psi^*_{v'\bk'}(\br_\nu),\\
\langle cv\bk|W|c'v'\bk'\rangle &= \frac{1}{N_k}\sum_{\mu\nu}\psi^*_{c\,\bk+\bQ}(\br_\mu)\psi_{c'\,\bk'+\bQ}(\br_\mu)\,W_{\bk-\bk'}(\mu,\nu;0)\,\psi_{v\bk}(\br_\nu)\psi^*_{v'\bk'}(\br_\nu).
\end{aligned} \tag{1.3}
$$

The electron pair carries momentum $(\bk+\bQ)-(\bk'+\bQ) = \bk-\bk'$, the same as the hole pair. So $W$ is needed only at grid momenta, exactly as at $\bQ=0$, while the exchange needs $V_\bQ$ at the off-grid $\bQ$ (§3). $w_x = 2$ for a spin-restricted scalar run (singlet) and $1$ for spinors.

## <span class="n">2</span> Conduction states at k + Q {: #bands }

The conduction leg is needed at $N_k \times N_Q$ points that are not on the DFT grid. They come from one H-transform of the GW run's states; no shifted DFT calculation is run.
{: .intro }

### <span class="n">2.1</span> H-transform at the shifted points {: #bands-ht }

`bse_setup.compute_wfns_fi`
{: .in-code }

The whole-state Galerkin basis $B_\alpha$ and the smoothed Hamiltonian $f(H)$ are built once on the coarse grid, as in the [htransform](../../utilities/htransform.md#how) utility. At each point of the list $\{\bk+\bQ\}$, $f(H)$ is Fourier-interpolated and diagonalized, and the state at a centroid follows from its coefficients:

$$
f(H)_{\bk+\bQ} = \sum_\bR e^{-2\pi i(\bk+\bQ)\cdot\bR} f(H)_\bR, \qquad \psi_{c\,\bk+\bQ}(\br_\mu) = \sum_\alpha c_c(\bk+\bQ)[\alpha]\,B_\alpha(\br_\mu) \tag{2.1}
$$

The eigenvalues give $E_{c\,\bk+\bQ}$ through the inverse of $f$. The basis is fitted in memory on every run; `galerkin_dft.h5` is neither read nor written.

### <span class="n">2.2</span> Window and guard bands {: #bands-window }

`bse_setup.resolve_conduction_window`
{: .in-code }

$f(H)$ is built over the deck's full window, $[n_\mathrm{elec}-n_\mathrm{val},\, n_\mathrm{elec}+n_\mathrm{cond})$ with the deck's `nval` and `ncond`, and the BSE conduction bands are returned from its interior. With $n_c$ = `--n-cond` the number of conduction guard bands above them is:

$$
n_\mathrm{guard} = n_\mathrm{cond} - n_c \tag{2.2}
$$

Fewer than four guards let the window edge ring off the grid; more than 16 widen the basis without benefit. Both warn; $n_\mathrm{guard} < 0$ refuses. A narrow conduction-only window makes the interpolated states ring by 100–1000 meV.

### <span class="n">2.3</span> Quasiparticle energies {: #bands-qp }

`bse_window.apply_eqp_to_bse_window`
{: .in-code }

Without `--eqp` the energies are the mean-field $E_{n\bk}$ of the restart and $E^{\mathrm{QP}} = E$ in (1.2). With `--eqp eqp1.dat` the wedge corrections are unfolded by symmetry and applied to both legs: the stored valence and conduction energies, re-sliced on the corrected spectrum, and the energies $f(H)$ is built from, so $E^{\mathrm{QP}}_{c\,\bk+\bQ}$ is an interpolated QP band. Off-diagonal QP mixing is not represented. A full QP exciton band structure needs a restart built from `WFN_qp.h5`.

### <span class="n">2.4</span> The check at Γ {: #bands-gate }

`gate_htransform_vs_stored`
{: .in-code }

At the first path point on $\Gamma$, $\bk+\bQ$ is a grid point and the interpolated conduction states must reproduce the stored ones. The driver compares the energies, order-matched, and the smallest singular value $s_\mathrm{min}$ of the conduction-subspace overlap $\langle\psi^{\mathrm{ht}}_{c\bk}|\psi^{\mathrm{stored}}_{c'\bk}\rangle$ at each $\bk$:

$$
\max|\Delta E_{c\bk}| < 0.05\ \mathrm{Ry}, \qquad s_\mathrm{min} > 0.5 \tag{2.3}
$$

A violation refuses; an energy error above 20 meV warns. Agreement on the grid is not evidence of accuracy between grid points; `--extra-q` with symmetry images of off-grid points is.

## <span class="n">3</span> Exchange at Q {: #exchange }

The exchange is the one kernel that needs an interaction at a momentum the GW run never computed. `--vq-mode` selects how $V_\bQ(\mu,\nu)$ is obtained.
{: .intro }

### <span class="n">3.1</span> The tile at Q and the G = 0 term {: #ex-g0 }

`bse.exchange_path.exchange_tiles`
{: .in-code }

The exchange tile is the bare Coulomb in the ζ basis at the exciton momentum. At finite $\bQ$ the $\bG = 0$ term of $v(\bQ+\bG)$ is kept (BerkeleyGW's `energy_loss` convention); at $\Gamma$ the optical $\bq = 0$ tile is used, whose $\bG=0$ term is replaced by the cell-averaged head:

$$
V_\bQ(\mu,\nu) = \sum_\bG \zeta^*_{\bQ\mu}(\bG)\,v(\bQ+\bG)\,\zeta_{\bQ\nu}(\bG), \qquad V_{\Gamma}(\mu,\nu) = V_{\bq=0}(\mu,\nu) + \frac{v^{\mathrm{head}}}{\Omega}\,g_0^*(\mu)\,g_0(\nu) \tag{3.1}
$$

$g_0(\mu)$ is the $\bG=0$ component of $\zeta_{\bq=0,\mu}$. The kept $\bG=0$ term is the non-analytic long-range exchange, so the longitudinal branch need not join $\Omega_S(\Gamma)$ as $\bQ\to0$ ([q → 0 head](../../theory/q0-head.md)). Every tile is Hermitized, $\tfrac12(V+V^\dagger)$. In code the pair density $\psi^*_{c\,\bk+\bQ}\psi_{v\bk}$ meets the stored tile labelled $\bq = \mathrm{wrap}(-\bQ)$.

### <span class="n">3.2</span> Exchange routes {: #ex-routes }

`bse.vq_interp`
{: .in-code }

| Route | $V_\bQ$ from | Scope | Cost per $\bQ$ | Certified by |
|---|---|---|---|---|
| interp | short-range stencil + long-range model of the coarse tiles | slab only; full-BZ `zeta_q.h5` | one dispatch | — |
| refit | a new ζ fit at $\bQ$ from the H-transform states | any dimension | one ζ solve | stored tiles and on-grid eigenvalues (§3.3) |
| both | interp on the path, refit at `--refit-points` | slab only | interp + spot refits | printed interp − refit differences |
| ongrid | the stored tile | $\bQ$ on the exchange-tile grid | none | exact |

=== "interp"
    **Closed-form interpolant** <span class="status production">Production</span>

    The default, for slabs. Each coarse tile is Tikhonov-filtered and split by a Gaussian into short- and long-range parts. The short-range tiles are interpolated by a truncated real-space stencil; the long-range form factors are fitted once by a polynomial per $|G_z|$ channel and evaluated at any $\bQ$.

    $$
    V_\bQ = \sum_j w_j(\bQ)\,V^{\mathrm{SR}}(\bq_j) + \bar A A^{T}, \qquad A_\mu(\bG) = e^{-2\pi i(\bQ+\bG)\cdot\mathbf{s}_\mu} M_\mu(\bQ+\bG)\sqrt{v^{\mathrm{LR}}(\bQ+\bG)}, \qquad v^{\mathrm{LR}}(\mathbf K) = v(\mathbf K)\,e^{-K^2/4\alpha^2} \tag{3.2}
    $$

    Stencil
    :   $w_j(\bQ) = e^{-2\pi i\bQ\cdot\bR}\,\mathrm{pinv}\big(e^{-2\pi i\bq_j\cdot\bR}\big)$, $\mathbf{s}_\mu$ the centroid in crystal coordinates

    Controls
    :   `--alpha` ($\alpha$, 0.30 bohr⁻¹), `--eps-tik` ($10^{-4}$)

    ```ini
    --vq-mode interp
    ```

=== "refit"
    **Per-Q ISDF refit** <span class="status production">Production</span>

    The bulk route. At each $\bQ$ the ζ fit is repeated from the pair densities of the H-transform states, with the same solve and `zeta_rcond` the GW run recorded, and contracted with the GW run's own Coulomb kernel, $\bG = 0$ included.

    $$
    \zeta'_\bQ = Z_\bQ C_\bQ^{\dagger}\big(C_\bQ C_\bQ^{\dagger}\big)^{-1}, \qquad V_\bQ(\mu,\nu) = \sum_\bG \zeta'^{\,*}_{\bQ\mu}(\bG)\,v(\bQ+\bG)\,\zeta'_{\bQ\nu}(\bG) \tag{3.2}
    $$

    Window
    :   `--refit-window zeta` (the GW ζ-fit window) or `bse` (the deck's window), plus `--refit-guard-bands` (4)

    ```ini
    --vq-mode refit
    ```

=== "both"
    **Interpolant with refit spot checks** <span class="status supported">Supported</span>

    `interp` on every path point and `refit` at `--refit-points` (default five evenly spaced), as extra rows of the same scan. The difference per level, in meV, is printed and the refit levels go to the table as `refit` rows and to the plot as markers.

    $$
    \Delta_S(\bQ_i) = \Omega^{\mathrm{interp}}_S(\bQ_i) - \Omega^{\mathrm{refit}}_S(\bQ_i) \tag{3.2}
    $$

    ```ini
    --vq-mode both --refit-points 0,8,15,22,29
    ```

=== "ongrid"
    **Stored tiles** <span class="status supported">Supported</span>

    No model and no ζ: the tile the GW run stored at the grid momentum. Exact, but every path point must lie on the exchange-tile grid, so the path floor must be lifted with `--q-per-segment 1`.

    $$
    V_\bQ(\mu,\nu) = V_{\mathrm{wrap}(-\bQ)}(\mu,\nu), \qquad \bQ \in \text{grid} \tag{3.2}
    $$

    ```ini
    --vq-mode ongrid --q-per-segment 1
    ```

### <span class="n">3.3</span> Certifying the refit {: #ex-cert }

`vq_interp.refit_ongrid_null`, `_certify_refit_against_stored`
{: .in-code }

Before any off-grid tile is used, the refit is run at grid momenta and compared with the stored tiles: at every distinct grid tile the path passes through, plus $\Gamma$ and the three tiles farthest from it. After the scan, every finite path point on the grid is solved a second time with the stored tile, and the two level sets are compared:

$$
\frac{\lVert V^{\mathrm{refit}}_\bq - V_\bq\rVert}{\lVert V_\bq\rVert} \le 5\times10^{-2}, \qquad \max_S\big|\Omega^{\mathrm{refit}}_S(\bQ) - \Omega^{\mathrm{stored}}_S(\bQ)\big| \le 0.01\ \mathrm{meV}\ \ (1.0\ \mathrm{meV}) \tag{3.3}
$$

The tile check applies only under `--refit-window zeta`, where ζ′ is the GW run's ζ. Under `--refit-window bse` the eigenvalue check is the only one, and a path with no finite grid point refuses. The tolerance is 0.01 meV (`--cert-grade reference`) or 1.0 meV (`visualization`); either failure refuses before anything is written.

### <span class="n">3.4</span> Mini-BZ head average {: #ex-head }

`vq_interp.minibz_head_vlr`
{: .in-code }

On the `interp` route the head of $V_\bQ$, the point value $v(\bQ+\bG^*)$ at the slot nearest $\bQ$, can be replaced by its cell average over the mini-BZ around $\bQ$. The $\bG^*$ column is then dropped from the tile and the head enters as a rank-three term built from the velocity dipoles $d^a_{cv\bk}$ of `dipole.h5` and the cell moment $M_{ab}(\bQ)$:

$$
K^{\mathrm{head}}_{cv\bk,\,c'v'\bk'} = \frac{w_x}{N_k}\, d^{a*}_{cv\bk}\,M_{ab}(\bQ)\,d^{b}_{c'v'\bk'} \tag{3.4}
$$

Set by `--head-minibz-average` or the deck's `head_minibz_average`. The dipoles are exact matrix elements, not ISDF-reconstructed, so a run with the head average is not bit-comparable to one without it.

## <span class="n">4</span> The screened interaction {: #screening }

$W$ enters only at grid momenta, so on the native grid nothing is interpolated. Where $W$ is moved to a finer grid, its short-range and long-range parts are treated separately.
{: .intro }

### <span class="n">4.1</span> W on the grid {: #w-grid }

`bse_loading.load_bse_data_from_restart_sharded`
{: .in-code }

The loader reads the static $W_\bq(\mu,\nu;0)$ of the restart and adds the cell-averaged head at $\bq=0$. The driver transforms it to real space once; $W_\bR$ is a constant of the whole $\bQ$ scan:

$$
W_{\bq=0}(\mu,\nu;0) \leftarrow W_{\bq=0}(\mu,\nu;0) + \frac{w^{\mathrm{head}}}{\Omega}\,g_0^*(\mu)\,g_0(\nu), \qquad W_\bR(\mu,\nu) = \mathrm{IFFT}_{\bq\to\bR}\,W_\bq(\mu,\nu;0) \tag{4.1}
$$

### <span class="n">4.2</span> Short-range and long-range parts {: #w-srlr }

Near $\Gamma$ the screened head diverges as $v/\varepsilon$ ($1/q^2$ in bulk) and the slab kernel is non-analytic in $|\bq_\parallel|$. A band-limited interpolant reproduces neither. $W$ is therefore split into a smooth short-range part, which is interpolated, and a long-range part, which is evaluated analytically at every target $\bq$:

$$
W_\bq(\mu,\nu;0) = W^{\mathrm{SR}}_\bq(\mu,\nu) + W^{\mathrm{LR}}_\bq(\mu,\nu) \tag{4.2}
$$

On the native grid of §4.1, $W^{\mathrm{LR}}$ is just the injected $\Gamma$ head and no split is applied. The split acts in the two densification routes below, and its theory is on [Short- and long-range W](../../theory/w-range-splitting.md).

### <span class="n">4.3</span> Coarse W with `--w-coarse-grid` {: #w-coarse }

`gw.head_densify.attach_head_channel`
{: .in-code }

`--w-coarse-grid` samples the native $W$ on a sub-grid that divides the BSE grid and interpolates it back, to test a cheap coarse $W$ against a fine exciton grid. Only `c1` is implemented: the $\Gamma$ head is subtracted, the head-free body is decimated and zero-padded in $\bR$ (the trigonometric interpolant $T$), and a head channel is re-attached at every fine $\bq$ inside the coarse $\Gamma$ cell $\Gamma_c$:

$$
W^{\mathrm{fine}}_\bq = T\big[W - W^{\mathrm{LR}}\big]_\bq + \frac{S(\bq)}{\Omega}\,g_0^*\,g_0, \qquad S(\bq) \propto \frac{v(\bq)}{1 - v(\bq)\,\bq^{T}\mathsf{S}\,\bq}\ \ (\bq\in\Gamma_c), \qquad S(\bq) = 0\ \ (\bq\notin\Gamma_c) \tag{4.3}
$$

$\mathsf S$ is the $\bq\to0$ response tensor; $S(\Gamma)$ is the average over the fine mini-BZ cell (`--w-head-gamma-cell fine`), and the scale is pinned to the restart's $w^{\mathrm{head}}$. On equal grids the result is the native $W$ bit for bit. The route is compared with `srlr` on [Short- and long-range W §3.3](../../theory/w-range-splitting.md#c1).

### <span class="n">4.4</span> Fine BSE grid with `bse_k_grid` {: #w-fine }

`bse_densify._interpolate_bse_data_to_grid`
{: .in-code }

With the deck key `bse_k_grid` the loader moves the whole problem to the finer grid before the driver starts: states and energies by H-transform, $W$ by the same interpolant with `w_head_densify = c1` (as in 4.3) or `srlr`, which subtracts a fitted long-range model $\mathrm{LR}_m$ with window $w(|\bq|)$ from every coarse tile and adds it back at every fine $\bq$:

$$
W_{\bq_f} = T\big[W - w\,\mathrm{LR}_m\big]_{\bq_f} + w\,\mathrm{LR}_m(\bq_f) \tag{4.4}
$$

The exchange tiles stay on the coarse restart grid, so `ongrid` and the refit certificates use the coarse grid. The densification is described on the [`bse`](../bse/theory.md) Theory page (Fig. B4) and derived on [Short- and long-range W](../../theory/w-range-splitting.md).

## <span class="n">5</span> Solver and output {: #solver }

Every path point is one Tamm–Dancoff eigenproblem. The driver solves all of them in one compiled program and writes eigenvalues only.
{: .intro }

### <span class="n">5.1</span> The matrix-free product {: #solve-matvec }

`bse_stack_matvec.build_bse_stack_matvec`
{: .in-code }

$H_\bQ$ is never stored. The product with a trial vector $X_{cv\bk}$ is the production BSE matvec of Fig. B2, with $\psi_{c\,\bk+\bQ}$ in every conduction slot and $V_\bQ$ in place of $V_{\bq=0}$:

$$
(H_\bQ X)_{cv\bk} = \big(E^{\mathrm{QP}}_{c\,\bk+\bQ}-E^{\mathrm{QP}}_{v\bk}\big)X_{cv\bk} + (V_\bQ X)_{cv\bk} - (W X)_{cv\bk} \tag{5.1}
$$

The exchange goes through the transition density $\rho(\br_\mu)$ and $V_\bQ$; the direct term is a $\bk$-convolution, $\mathrm{IFFT}_{\bk\to\bR}$, a pointwise product with $W_\bR(\mu,\nu)$ and $\mathrm{FFT}_{\bR\to\bk}$, with a $1/\sqrt{N_k}$ on each transform.

### <span class="n">5.2</span> Block Lanczos over the path {: #solve-lanczos }

`build_path_solver`
{: .in-code }

For each $\bQ$ the pair amplitude $\psi^*_{c\,\bk+\bQ}(\br_\mu)\psi_{v\bk}(\br_\mu)$ is formed once, then a fixed-depth block Lanczos with full reorthogonalization returns the lowest `--n-eig` Ritz values:

$$
H_\bQ A^{S\bQ} = \Omega_S(\bQ)\,A^{S\bQ}, \qquad S = 1,\dots,n_\mathrm{eig}, \qquad \dim\mathcal K = b\cdot\min\!\big(n_\mathrm{iter},\ \lfloor n_v n_c N_k / b\rfloor\big) \tag{5.2}
$$

$b$ is `--block-size` and $n_\mathrm{iter}$ is `--max-iter`. The start block has no weight on mesh pad slots. The whole $\bQ$ list, with certification and spot-check rows, is one `lax.scan`: one compile serves every point, and extra points cost one scan row each.

### <span class="n">5.3</span> Output {: #solve-out }

The levels are converted from Ry to eV and written by rank 0 to `<prefix>.dat`, one row per $\bQ$, with the window, grid, route, certification and conventions in the header ([Files](files.md)). `<prefix>.png` plots the path bands. No eigenvectors and no spectrum are written.

## Notation {: #notation }

| Symbol | Meaning |
|---|---|
| $\bQ$, $\bQ_i$, $n_i$ | exciton momentum; path corners; points per segment |
| $\lvert cv\bk\rangle$ | hole in $v$ at $\bk$, electron in $c$ at $\bk+\bQ$ |
| $A^{S\bQ}$, $\Omega_S(\bQ)$ | exciton amplitudes and energies |
| $E_{n\bk}$, $E^{\mathrm{QP}}_{n\bk}$ | mean-field and quasiparticle energies |
| $N_k$, $N_\mu$, $\br_\mu$ | grid size; ISDF point count; the points |
| $\zeta_{\bq\mu}(\bG)$, $\zeta'_{\bQ\mu}$ | interpolation vectors; refitted at $\bQ$ |
| $V_\bQ(\mu,\nu)$, $v$ | bare exchange tile at $\bQ$; Coulomb kernel |
| $W_\bq(\mu,\nu;0)$, $W_\bR$ | static screened interaction; in real space |
| $W^{\mathrm{SR}}$, $W^{\mathrm{LR}}$ | short- and long-range parts of $W$ |
| $v^{\mathrm{head}}$, $w^{\mathrm{head}}$, $g_0$ | Coulomb and screened heads; $\bG=0$ component of $\zeta_{\bq=0}$ |
| $B_\alpha$, $f(H)$ | Galerkin basis; smoothed Hamiltonian of the H-transform |
| $w_x$ | exchange weight, 2 scalar, 1 spinor |
| $\alpha$, $v^{\mathrm{LR}}$ | Gaussian split width; long-range Coulomb |
| $d^a_{cv\bk}$, $M_{ab}(\bQ)$ | velocity dipoles; mini-BZ cell moment |
| $T$, $S(\bq)$, $\Gamma_c$ | trigonometric interpolant; head channel; coarse $\Gamma$ cell |

## References {: #refs }

1. M. Rohlfing and S. G. Louie, Phys. Rev. B 62, 4927 (2000).
2. M. Gatti and F. Sottile, Phys. Rev. B 88, 155113 (2013).
3. D. Y. Qiu, T. Cao and S. G. Louie, Phys. Rev. Lett. 115, 176801 (2015).
4. A. Baldereschi and E. Tosatti, Phys. Rev. B 17, 4710 (1978).
5. J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
6. J. Lu and L. Ying, J. Comput. Phys. 302, 329 (2015).
