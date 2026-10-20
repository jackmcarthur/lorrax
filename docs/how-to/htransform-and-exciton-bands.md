# Band interpolation (htransform) and exciton bands

This page explains how LORRAX puts single-particle bands and exciton bands on
an arbitrary k path from a calculation done on a coarse uniform grid. It is for
a user who has a finished DFT or GW run and wants band structures, band
character, spin and orbital moments, or the exciton dispersion $E_S(Q)$. Read
the [quickstart](../quickstart.md) first for the run chain; the GW outputs it
consumes are described in [outputs and BerkeleyGW](berkeleygw-users.md#outputs).

Two drivers share one interpolation engine:

| driver | computes | writes |
|---|---|---|
| `python -m bandstructure.htransform` | band energies $\varepsilon_n(q)$ on the deck's k path; optional band character and Brillouin-zone moments | `bandstructure.dat`, `htransform.out`, `galerkin_dft.h5`; optional `bands_<color>.png`, `band_operators_path.npz`, `moments.txt` |
| `python -m bse.exciton_bands` | the lowest exciton energies $E_S(Q)$ of the finite-momentum Bethe–Salpeter equation along the deck's Q path | `<prefix>.dat`, `<prefix>.png`, `<prefix>.out` |

The same engine also supplies the fine-grid wavefunctions of BSE
densification (`bse_k_grid`, [BSE](../architecture/bse.md)).

## 1. The problem

A GW calculation is done on a uniform grid of $N_k$ points (for example
$4\times4\times4$). Its energies $\varepsilon_{nk}$ and states $\psi_{nk}$
exist only there. A band structure needs $\varepsilon_n(q)$ at points $q$
between the grid points. Wannier interpolation solves this with localized
orbitals that the user constructs. htransform needs no such orbitals: it
expands every state in one real-space basis that does not depend on $k$, and
Fourier-interpolates a function of the Hamiltonian in that basis.

## 2. The nodal basis

Stack the states of the fitted band window $[b_0, b_1)$ at every coarse $k$
as the rows of one matrix

$$\Psi_{(k,n),(s,r)} = \psi_{nk}(s, r), \qquad n \in [b_0, b_1),$$

with $s$ the spinor component and $r$ the points of the real-space FFT grid.
Each row is the full Bloch state, phase $e^{ik\cdot r}$ included. The row
space of $\Psi$ contains every state LORRAX will interpolate, so it is the
natural space, but it is too large to use directly. `isdf.galerkin.fit_galerkin_basis`
represents every state by its values at a set of nodes $r_\nu$, sampled once
through the GW centroid-face path
(`common.wfn_transforms.load_centroids_band_chunked`):

$$\Phi_k[(s,\nu), n] = \psi_{nk}(s, r_\nu).$$

**Nodes.** $N_\mathrm{node} = \min(N_\mu, 22\,(b_1-b_0))$
(`fh_interp.htransform_nodes`, `isdf.galerkin.NODES_PER_FITTED_BAND`), so the
fit's cost follows the fitted window, not the ISDF bundle. At $N_\mu$ the nodes
are the run's ISDF centroids. Below it they are an orbit-closed,
charge-density-weighted k-means selection on the fitted bands
(`centroid.kmeans_cli.orbit_kmeans_points`: `--orbit`, rho-power 1, fit window
$b_0{:}b_1$ on both legs), written beside the basis artifact as
`galerkin_nodes_<N>_b<b0>-<b1>_<wfn>.txt` and reused while the WFN fingerprint
and window match. The states at the ISDF centroids, which the BSE consumers
need, are then back-map targets (§3) rather than nodes. Orbit closure is
mandatory for both sets.

Nodal values carry no metric. The fit adds the diagonal quadrature metric
$W = \operatorname{diag}(w_\nu)$ of the nodes: one non-negative weight per
node orbit, the least-squares (NNLS) solution of
$\Phi_k^\dagger W \Phi_k = 1$ over all coarse $k$, folded onto the orbits a
block of nodes at a time (a solve that does not converge refuses,
`GATE quadrature_metric_nnls`). $W$ is invariant under the node permutation of
every crystal operation. A per-$k$ Löwdin step then makes
the coefficients exactly orthonormal:

$$G_k = \Phi_k^\dagger W \Phi_k, \qquad c_{k} = W^{1/2}\Phi_k G_k^{-1/2},
\qquad c_k^\dagger c_k = 1,$$

shape $(N_k, b_1-b_0, N_B)$ with $N_B = N_s N_\mathrm{node}$. The log
prints $\varepsilon = \max_k \|G_k - 1\|_2$ (Si 4³ at 1100 nodes:
$1.0\times10^{-3}$; MoS₂ 9×9 at 1198: $1.7\times10^{-2}$), the number of
zero-weight nodes (they drop out of $c$), the orbit count and the fit wall
(seconds). The fit itself has no pivot or tolerance. A node or centroid set
that is not orbit-closed (`symmetry_maps.centroid_source_map_and_wrap`) refuses,
naming the remedy.

**Node count.** The off-grid error follows $\varepsilon$, not the rank. On Si
4³ (36 fitted spinor bands, orbit-closed k-means with `--fit-window 0:36,0:36`)
$\varepsilon$ = 0.37 / 0.12 / 1.6×10⁻² / 1.9×10⁻³ / 1.0×10⁻³ at 312 / 432 /
588 / 780 / 1100 nodes, and the 4³→8³ off-grid DFT MAE is 8.03 meV (max 62.7)
at 312 nodes against 6.76 meV (max 27.4) at 780. The fit warns when
$\varepsilon > 10^{-2}$ and names the count $N_\mathrm{node} \ge 22\,(b_1-b_0)$
that reached it on Si; MoS₂ 9×9 at 22.2 nodes per band (1198/54) reads
$\varepsilon = 1.7\times10^{-2}$.

**Fine-k solve.** $f(H)_q$ is numerically of rank $b_1-b_0$ ($f \le 0$ on the
fitted states, $\approx 0$ on the rest of the $N_B$ directions), so every
fine-k, path and moment-grid solve refines only its most negative block:
the fitted bands plus 32 columns, started from $c$ of the nearest coarse k
plus fixed random columns (an operand of the program, not a constant). Two
subspace steps $X \leftarrow \mathrm{qr}(-f(H)_q X)$ and a Rayleigh–Ritz
eigensolve of the block follow; further steps run until the relative
residual $\|f(H)_q v - \lambda v\|/|\lambda|$ of every band the stage's
consumer reads (the fine-k stage: the returned window; the path and the
moment grid: their states) is $\le 10^{-10}$ ($|\lambda|$ floored at
$10^{-2}$ of the block's largest), at most 16. The topmost guards sit at
$f = 0$ beside the null space; they are not read and need not converge.
The log line `[top-block]` prints the iteration count and the residual; a
residual above $10^{-6}$ refuses (`GATE top_block_not_converged`). The cost
per q is block products with the $N_B \times N_B$ operand instead of a full
eigensolve. Each such stage has two routes, admitted by the memory model
([memory model](../architecture/memory-model.md)) from the compiled program's
price: `q-batched` (whole $N_B \times N_B$ matrices of a rank's own q rows
after one face→q exchange) and `face` (one q at a time, tiled over the mesh;
`linalg = distributed` pins it).

The fit writes $c$, $\Phi$, $G^{-1/2}$, the node FFT indices and the states
at the targets to `galerkin_dft.h5` beside the deck (format 6). A later run
reuses the file only when its provenance matches exactly (WFN fingerprint,
node and target sets, band window, grids, spinor mode, node version);
otherwise it refits in memory and leaves the file unchanged. Files of an
older format are refitted. The log says `REUSED`, `REFIT` or
`FITTED and published`.

## 3. The f-transform and the interpolation

The direct choice, Fourier-interpolating
$H_k = \sum_n \varepsilon_{nk}\, c_{nk} c_{nk}^\dagger$, fails at the top of
the window. The window is a set of band indices, and the top band disperses,
so near the window edge states enter and leave $H_k$ as $k$ moves. $H_k$ is
then not smooth in $k$, its lattice transform $H_R$ is long-ranged, and the
interpolant rings between grid points.

htransform interpolates $f(H)$ instead, where $f$ removes the window edge.
Let $\varepsilon_\mathrm{top} = \max_k \varepsilon_{b_1-1,k}$ be the highest
energy of the top fitted band, $y = \varepsilon - \varepsilon_\mathrm{top}$,
and $a = 4 W_a$ with $W_a$ the bandwidth of one band (by default the top
fitted band; `--a-band` names another). Then

$$f(y) = \begin{cases} y + a/2, & y \le -a, \\ \int_0^{y} f'\,dy, & -a < y < 0, \\ 0, & y \ge 0, \end{cases}
\qquad f'(y) = \tfrac12 - \frac{\operatorname{erf}\!\big(n(\tfrac12 + y/a)\big)}{2\operatorname{erf}(n/2)},\; n = 3.$$

$f$ is the identity (shifted) deep in the window, exactly zero at and above
$\varepsilon_\mathrm{top}$, and joined by a shoulder whose slope falls
smoothly from 1 to 0 (`bandstructure.fh_interp._fun_jit`). Because every
state near the edge is mapped to zero, the matrix

$$f(H)_k = \sum_n f(\varepsilon_{nk})\, c_{nk} c_{nk}^\dagger$$

no longer cares which states sit at the edge; it is smooth in $k$, and
invariant under any unitary mixing of the window's states. In the nodal basis
it is a node-pair kernel interpolated per node pair:

$$f(H)_R = \frac{1}{N_k}\sum_k e^{2\pi i k\cdot R} f(H)_k, \qquad
f(H)_q[\nu\nu'] = \sum_{R} w^{\nu\nu'}_R\, e^{-2\pi i q\cdot R} f(H)_R[\nu\nu'],$$

where, per node pair, $R$ runs over the Wigner--Seitz images of
$r_\nu + R - r_{\nu'}$ in the coarse supercell, ties shared equally
(`common.ws_images`, `fh_interp.node_pair_table`, `fh_interp.node_fourier`).
Every crystal operation maps the images of a node pair onto those of the image
pair and leaves $W$ invariant, so the interpolant is covariant: star-related
$q$ give the same energies to rounding, provided the fitted window ends in a
gap at every coarse $k$ (a window top that cuts a degenerate multiplet makes
the coarse kernels gauge-dependent). At each path point the driver
diagonalizes $f(H)_q$: the eigenvalues are $f(\varepsilon_n(q))$ and the
eigenvectors $c_n(q)$ are the states in the orthonormal nodal frame. Newton
iteration inverts $f$ (`fh_interp.newton_inv`, converged to
$\max|f(x)-y| \le 10^{-12}$ Ry in at most 50 steps, else refused). The BSE
consumers take the states at the ISDF centroids $r_\mu$ (the targets),
$\psi_n(q; s, r_\mu) = [L_q c_n(q)]_\mu / (-f_n(q))$, from the back-map
kernel $L_k = \Phi^T_k (-f(\varepsilon_k)) c_k^\dagger$ with
$\Phi^T_k[(s,\mu), n] = \psi_{nk}(s, r_\mu)$ (equal to $\Phi_k$ when the
nodes are the centroids). It is built in the same program as $f(H)_R$
(`fh_interp.build_fH_R(..., backmap=)`) and interpolated with the per-pair
images of each target–node pair (`node_pair_table`, `target_code`): exact on
the coarse grid, and the division touches only returned window bands, which
the guard bands keep away from $f = 0$.

**Guard bands.** Inside the shoulder $f' < 1$, and at the top $f' = 0$: a band
there is compressed and, at the $k$ where it reaches $\varepsilon_\mathrm{top}$,
not represented at all, so the eigensolver returns null-space directions in
its place. htransform therefore fits `--guard-bands` (default 4, the measured
shoulder depth) extra conduction bands above the returned window and returns
only the bands below them. A guard band the WFN does not contain refuses,
because it would arrive as zeros.

**Windows.** Let $n_\mathrm{occ}$ be the number of occupied bands (the WFN's
largest `ifmax`). The returned window is the deck's
$[n_\mathrm{occ} - $`nval`$,\ n_\mathrm{occ} + $`ncond`$)$; the fitted window
adds the guard bands. Standalone htransform requires `nval` equal to the
number of occupied bands: a window whose lower edge cuts the occupied
manifold reproduces the grid energies exactly and still rings between grid
points, because the omitted bands mix with the kept ones off the grid.

**Locality receipt.** `htransform.out` prints how much of $\|f(H)_R\|$ sits
on the outermost $R$ shell of the supercell, relative to the total and to
$R = 0$ (`fh_interp._fh_locality_metrics`). A large outer-shell weight means
the coarse grid does not resolve $f(H)_R$, and the interpolation between grid
points is not controlled. Reproduction of the coarse-grid energies is
printed too, but it is not evidence of accuracy between grid points: at a
grid point the interpolant returns its input by construction. Acceptance
needs an independent check, a fine-grid DFT calculation or the symmetry test
of §7.

## 4. Quasiparticle bands

Two routes put GW energies on the path. They are mutually exclusive.

- **`--qp-rotations qp_wfn_rotations.h5`, the full QP Hamiltonian.** The GW
  run's QP eigensystem $H_\mathrm{QP} = U\,\mathrm{diag}(E_\mathrm{QP})\,U^\dagger$
  (in the DFT basis) rotates the compact rows,
  $C_\mathrm{QP}[k] = U[k]^T C[k]$, so the same build gives
  $f(H_\mathrm{QP})$ and the same DFT basis serves both
  (`fh_interp.resolve_qp_hamiltonian_state`). The QP block must lie wholly
  inside the fitted window, since a cut through it would slice $U$ and break
  its unitarity. Bands of the fitted window outside the block keep their DFT
  orbitals, with the self-consistent energy ladder when the artifact carries
  one. Where DFT guard bands remain above the block, the returned states are
  chosen by their weight in the interpolated QP-block projector
  $P_A(q) = \sum_{n \in A} c_n c_n^\dagger$
  (`fh_interp.select_active_eigenpairs`), and a path point where that choice
  is numerically ambiguous refuses. The artifact's source-WFN fingerprint,
  band range, k grid and k coordinates must match the deck's WFN.
- **`--eqp-file eqp1.dat`, the diagonal approximation.** The energies of the
  DFT bands are replaced by the GW run's wedge `eqp1.dat`, unfolded through
  the symmetry service. Off-diagonal QP mixing is not represented. The file's
  k-block coordinates must be the deck's wedge, in order.

A WFN that is itself a QP wavefunction (`WFN_qp.h5`, via `--wfn-file`) is a
third description of the same state; combining it with either flag refuses
(`file_io.qp_wfn.refuse_conflicting_qp_state_sources`).

## 5. Output: `bandstructure.dat`

Rank 0 writes the table (`htransform.write_bands_to_file`):

```text
# idx_k idx_b kx ky kz s energy_eV
# absolute_band_window=[b0,b1) fit_bands=NF guard_bands=NG
idx_k idx_b kx ky kz s energy    (format: %4d %4d % .8f % .8f % .8f % .8f % .8f)
```

`idx_k` is the path index and `idx_b` the band index counted from the bottom
of the returned window. Both `idx_b` and the header's `b0` are 0-based: the
band is absolute band `b0 + idx_b` counted from 0, which is BerkeleyGW band
`b0 + idx_b + 1`. `kx ky kz`
are crystal coordinates, `s` the cumulative path length, and the energy is in
eV with the valence-band maximum along the path at 0. The path comes from the
deck's `K_POINTS {crystal_b}` block: a count of corners, then one line per
corner with three crystal coordinates, the number of points to the next
corner and an optional `#label`; the last corner takes 1.

```text
K_POINTS {crystal_b}
3
  0.0000 0.0000 0.0000 20  #G
  0.5000 0.0000 0.0000 20  #X
  0.5000 0.5000 0.0000 1   #M
```

## 6. Band character and moments

These options attach physical observables to the bands of a spinor WFN
(`bandstructure.orbital`).

**Operators.** On the coarse full Brillouin zone the driver forms the band
matrices $\langle\psi_{nk}|O|\psi_{mk}\rangle$ of the Pauli matrices
$\sigma_{x,y,z}$ and, per requested channel, of the projector on one
element's $l$ shell (Löwdin-orthogonalized `PP_PSWFC` of the `*.upf` beside
the deck, j-averaged radial functions: QE `projwfc`'s projector). With
`--velocity` it also reads the stored velocity $v^a_{nm} = \langle
nk|\partial_{k_a}H|mk\rangle$ of the WFN's states. Each operator is carried
into the nodal frame as $C^T O\, C^*$ and interpolated with the same node-pair
sum as $f(H)$. On the `--qp-rotations` route only the eigenvectors change: $U$
is unitary on the fitted window, so the operator image is the same.

**Velocity.** `dipole.h5` (`psp.get_dipole_mtxels`, $p + i[V_\mathrm{NL}, r]$)
belongs to a DFT `WFN.h5`; for QSGW, run on the SC run's `WFN_qp.h5` with its
`dipole_qsgw.h5`, $U^\dagger(v_\mathrm{DFT} + D_k\Delta H)U$ between the same
states with their `band_energies`. The file's basis and WFN fingerprint are
checked and its `velocity` attribute is printed. A QP velocity with no
$\Sigma$ term refuses, and so does `--velocity` with `--qp-rotations` or
`--eqp-file`: the DFT velocity with QP energies is not $\partial_k
H_\mathrm{QP}$ (on a scissored toy model it gives a moment 22 % low).

**Magnetization axis.** $\hat n$ is read from the QE schema that
authenticates the WFN (`orbital.magnetization_axis`): the SCF output's
`total_vec` when it is nonzero; otherwise the input moment direction
(`angle1`/`angle2`, stored as `spin_teta`/`spin_phi`) of the magnetic species;
otherwise $z$. The report names the source.

**`--color spin|orbital|orbital:[EL:]l`** (repeatable) colors the path bands
by $\langle\sigma\cdot\hat n\rangle \in [-1, 1]$, by the modern-theory
wavepacket moment $\hat n\cdot\mathbf m_n$ (`orbital`, needs `--velocity`), or
by a channel's character $\in [0, 1]$ (`orbital:d`, `orbital:Fe:d`), and
writes `bands_<color>.png` (colon replaced by underscore) with the energy axis
$E - E_F$, $E_F$ from the coarse grid. With $v$ on the path,

$$\frac{\mathbf m_n}{\mu_B} = -\tfrac12\,\mathrm{Im}\sum_{m\ne n}
\frac{\mathbf v_{nm}\times\mathbf v_{mn}}{E_n - E_m}$$

(`psp.orbital_response.orbital_moments`; pairs closer than 1.4 meV are
omitted and degenerate multiplets averaged, because an individual moment
inside one is gauge dependent). The color scale is linear and clipped at the
95th percentile of $|\hat n\cdot\mathbf m_n|$; the colorbar states the clip.
`band_operators_path.npz` holds every path operator matrix (`path_operators`,
shape (path point, operator, band, band)), `operator_names`, `kpath_frac`,
`x_path`, `spin_axis`, `energies_ev` (relative to the path VBM),
`energy_reference_ev` (the path VBM relative to the coarse $E_F$) and, for
`orbital`, `orbital_moment_path` (path point, 3, band) and `orbital_color`.

**Totals** (`--velocity`, in `htransform.out`) are physical moments in
$\mu_B$ per cell along $\hat n$ on the coarse grid. The orbital moment is
`orbital_response.orbital_magnetization` at each parent k, star-weighted and
averaged over the group's axial time-odd action: $T = 0$ at midgap without
`occ_smearing_width_ry`, else the fixed-N Fermi–Dirac $\mu$. Without a
width, the rounded electron count must sit below a gap on the coarse grid;
overlapping bands or a fractional count refuse
(`GATE orbital_totals_t0_gap`; `bandstructure.orbital.orbital_totals`). It
sums every stored band of the velocity file; when the highest stored band
is not wholly above $\mu$ + 10 kT, `GATE orbital_totals_empty_band` asks
for more `ncond`. The spin and character matrices are formed at the WFN's
own k-points, as the dipole driver forms the velocity, and unfolded by
symmetry; no full-BZ ψ is loaded.
The spin moment $-\tfrac{g_e}{2}\mu_B\sum_\mathrm{occ}\langle\boldsymbol
\sigma\rangle$ ($g_e = 2.00232$) and the spin + orbital sum follow on their own lines.

**`--moments-grid NX NY NZ`** interpolates $f(H)$ and every $\sigma_a$ to
that uniform grid, finds $E_F$ there by Fermi–Dirac occupation of the
window's electrons at `occ_smearing_width_ry` ($10^{-4}$ Ry when unset), and
writes $\sum_{q,n} f_{qn}\langle qn|\boldsymbol\sigma|qn\rangle/N_q$ to
`moments.txt`, with its projection on $\hat n$, the grid $E_F$ and the
occupation of the top returned band. Above $10^{-6}$ that occupation biases
the fixed-N sums and refuses (`GATE htransform_moments_window`; raise
`ncond`). With no width, the coarse and grid $E_F$ use the rounded electron
count (`bandstructure.htransform`). The same sum taken directly on the coarse grid goes to
`htransform.out`. Signs follow QE ($n_\uparrow - n_\downarrow$). The grid
runs one $q_z$ plane per pass, so one dense $(N_k, N_B, N_B)$ operator image
is resident at a time.

*Example.* bcc Fe with spin–orbit coupling on a $20^3$ grid, DFT states:
interpolated to QE's own $8^3$ SCF grid, $\sum\langle\sigma\rangle = 2.30679$
against QE's total magnetization $2.30974\,\mu_B$, and $E_F$ 18.4859 against
18.4833 eV; on a $40^3$ grid, 2.28693, in 92–106 s on 9 A100 GPUs (a 3 × 3
mesh) at a device peak of 17.2 GiB.

## 7. Exciton bands

`bse.exciton_bands` solves the Tamm–Dancoff BSE at nonzero exciton momentum
$Q$ along the deck's `K_POINTS {crystal_b}` path. The pair basis is
$|vk,\, c\,k{+}Q\rangle$: the electron leg is shifted by $+Q$. The Hamiltonian
is

$$H_Q = D_Q + V_Q - W, \qquad (D_Q)_{vck} = \varepsilon_{c}(k+Q) - \varepsilon_{vk},$$

with the exchange $V_Q$ weighted as in `bse.bse_jax`
([BSE](../architecture/bse.md)). Each piece comes from a different place:

- **$\psi_c(k+Q)$ and $\varepsilon_c(k+Q)$** are eigenpairs of the
  interpolated $f(H)$ at the shifted points (`bandstructure.bse_setup.compute_wfns_fi`
  on the list $\{k+Q\}$). No shifted DFT calculation is needed. $f(H)$ is
  built over the full loaded window, all valence and conduction bands of the
  deck, and the BSE conduction bands must sit inside it with guard bands
  above: a narrow conduction-only window makes the interpolated conduction
  states ring by 100–1000 meV.
- **The direct term $W$** is the coarse-grid convolution of the GW restart,
  unchanged. When every conduction leg is shifted by the same $Q$, every
  momentum difference $k-k'$ stays on the coarse grid, so $W$ needs no
  interpolation.
- **The exchange $V_Q$** needs the bare interaction at the off-grid
  momentum. The tile is taken at $\mathrm{wrap}(-Q)$. That is an open defect: the
  pair density $\psi^*_{vk}\psi_{c,k+Q}$ has momentum $+Q$ and needs the tile at
  $\mathrm{wrap}(+Q)$ ([bsemat E12](../theory/bgw-bsemat-export.md#finite-q)). On
  MoS₂ 9×9 at $Q=(1/9,0,0)$ bright states move by up to 6.6 meV (0.38 % of the
  1.732 eV gap); $Q=0$ is unaffected. At finite $Q$ the
  $G = 0$ term of $v(Q+G)$ is kept (BerkeleyGW's `energy_loss` convention):
  this is the nonanalytic long-range exchange, so $E_S(Q\to0)$ need not equal
  $E_S(\Gamma)$ for longitudinal excitons. At exactly $\Gamma$ the production
  $q = 0$ tile and head are used. Why:
  [long-range exchange head](../theory/lt-exchange-head.md). `--vq-mode`
  picks the source (`bse.vq_interp`):

| `--vq-mode` | exchange source | scope |
|---|---|---|
| `interp` (default) | a closed-form interpolant of the coarse tiles: Tikhonov-cleaned short-range tiles on a lattice stencil plus one least-squares model of the long-range form factors | slab decks only (`sys_dim = 2`, $q_z = 0$ grids; refuses a bulk deck); needs full-BZ `zeta_q.h5` |
| `refit` | a new ISDF fit $\zeta'$ at each $Q$ from the htransform states, contracted with the producer's own Coulomb kernel | any dimension; the arbitrary-$Q$ route for bulk crystals; expensive |
| `both` | `interp` on the whole path plus `refit` at `--refit-points`, solved in the same scan | for measuring the interpolation error |
| `ongrid` | the stored tile $V_{\mathrm{wrap}(-Q)}$ | exact, but only at $Q$ on the coarse grid; refuses off-grid $Q$ |

**The refit window and its certificate.** `--refit-window zeta` (default)
fits $\zeta'$ on the producer's own ζ-fit window. It then reproduces the
stored tiles, and that identity certifies the run. It needs the parent ISDF
basis to span the window, $N_\mu n_s \ge N_k n_b$ ($n_s$ spinor components,
$n_b$ bands of the ζ window), which a wide GW window can exceed.
`--refit-window bse` fits on the BSE window plus its guards, which carries
every pair density the exchange contracts and lowers the bound; the stored
tiles are then not reproduced, so the certificate moves to the contracted
object: at every path $Q$ on the coarse grid the BSE is solved twice, with
the refit and with the stored tile, and the eigenvalues must agree within
`--cert-grade` (`reference` 0.01 meV, default; `visualization` 1.0 meV). A
path with no point on the coarse grid refuses under `bse`. The grade and the
worst difference are written into the output header.

**One compile.** The per-$Q$ solve (pair amplitudes and block Lanczos over
the BSE stack matvec) is one jitted function under one `lax.scan` over the
whole $Q$ list. A Python loop over $Q$ would recompile per point. Extra points
cost one scan row each, so `--q-per-segment` (default 16) raises each
segment's point count to at least that floor; `--q-per-segment 1` keeps the
deck's counts. The cost that grows with the number of $Q$ is the htransform
of $\psi_c(k+Q)$.

**The symmetry test.** $E_S(Q)$ must be equal at $Q$ and at its point-group
images, and an interpolation error breaks this. `--extra-q "x,y,z;x,y,z"`
appends points to the same scan (written to the `.dat` with mode `extra`, not
plotted). Passing the images of a few off-grid path points measures the
interpolation error without an external reference. Gates evaluated only on
grid points cannot see off-grid errors, for the reason given in §3.

**Quasiparticle energies.** `--eqp eqp1.dat` corrects both legs: the stored
valence and conduction energies and the htransform energies. There is no
`--qp-rotations` route; a full-QP exciton band structure needs a GW restart
built from `WFN_qp.h5`.

**Output.** `<prefix>.dat` (default `exciton_bands.dat`) has a header naming
the window, grid, `--vq-mode`, refit window and certificate, the conventions
and the path nodes, then one row per $Q$:

```text
# iQ  s_path  Qx  Qy  Qz  mode  E_1..E_neig (eV)
```

with $Q$ in crystal coordinates and $E_S$ in eV. Refit spot checks follow as
rows with mode `refit`. `<prefix>.png` plots the bands with the refit points
as markers.

| flag | default | meaning |
|---|---|---|
| `--n-val` / `--n-cond` | 4 / 4 | BSE window, in bands |
| `--n-eig` / `--block-size` / `--max-iter` | 6 / 8 / 40 | per-$Q$ block Lanczos |
| `--band-degeneracy` | `strict` | a window edge inside a multiplet refuses (`snap` widens it, `off` proceeds); checked on the loader window and the htransform conduction window |
| `--refit-guard-bands` | 4 | guard bands of the refit's $f(H)$ window above the ζ window; 0 must fail the tile identity |
| `--a-band` | top band | band whose bandwidth sets $a$ in $f$ (§3) |
| `--head-minibz-average` | the deck's `head_minibz_average` | per-$Q$ mini-BZ average of the finite-$Q$ exchange head |
| `--w-coarse-grid` / `--w-head-densify` / `--w-head-gamma-cell` | unset / `c1` / `fine` | sample W on a coarse sub-grid and trigonometrically interpolate it to the BSE grid; `c1` splits the Γ head off and re-attaches it analytically |
| `--rerun-check` | off | a second warm solve pass: a reproducibility assert and per-$Q$ timing |

A downfolded GW bundle ([downfold](../downfold.md)) works here too: the driver
takes the parent centroid table from the bundle's provenance.

## 8. htransform flags and keys

Invoke: `python -m bandstructure.htransform -i ht.in [--qp-rotations
qp_wfn_rotations.h5 | --eqp-file eqp1.dat] [--color spin] [--color orbital:d]
[--moments-grid 40 40 40]`. The deck is a GW deck with a `K_POINTS
{crystal_b}` block; relative output paths resolve beside it.

| key / flag | default | meaning |
|---|---|---|
| `nval` / `ncond` | 5 / 5 | returned window (§3); standalone runs need `nval` = occupied bands |
| `--guard-bands` | 4 | fitted bands above the returned window |
| `--a-band` | top fitted band | band whose bandwidth sets $f$'s shoulder width |
| `htransform_qr_eps` | 1e-3 | not a fit tolerance (§2 has none): the relative pseudo-inverse cutoff of the refit's node Gram (`vq_interp` refit) |
| `htransform_rank_multiplier`, `htransform_qrcp_seed` | removed | refused by name (`GATE htransform_qrcp_retired`): the nodal basis has no rank search; delete the lines |
| `-wfn` / `--wfn-file` | the deck's `wfn_file` | another WFN, for example `WFN_qp.h5` |
| `-o` / `--output-file`, `--report-file` | `bandstructure.dat`, `htransform.out` | outputs |
| `linalg` | `local` | `distributed` pins the `face` route of the top-block solves; otherwise each stage's route is admitted from its compiled price (§2) |
| `get_centroids_fi`, `kgrid_fi`, `wfn_fi_min`/`wfn_fi_max`, `wfn_fi_q_chunk` | off | BSE handoff: fine-grid ψ at the coarse centroids (`bse_setup.compute_wfns_fi`); keep at least 4 bands between `wfn_fi_max` and the top of the window |
| `--plot` | off | show the band plot |

## 9. Refusals and limits

| refusal | cause | fix |
|---|---|---|
| the nodal fit refuses: centroids not orbit-closed | the centroid file was not made in orbit-aware mode | regenerate it with `centroid.kmeans_cli` (orbit mode) |
| `[warn] nodal metric eps` | $\varepsilon > 10^{-2}$: too few nodes for the fitted window | $N_\mathrm{node} \ge 22\,(b_1-b_0)$; with fewer ISDF centroids than that the nodes are the centroids |
| `top_block_not_converged` | a top-block window residual above $10^{-6}$ after 16 steps | raise $N_\mathrm{node}$ or check the metric warning |
| `quadrature_metric_nnls` | the orbit-tied NNLS did not converge | check orbit closure and the node count |
| `htransform_qrcp_retired` | `htransform_rank_multiplier` or `htransform_qrcp_seed` in the deck | delete the line |
| `f-shoulder` | a returned band is absent from $f(H)$ at some coarse $k$ | add guard bands |
| Newton not converged | $\max|f(x) - y| > 10^{-12}$ Ry after 50 steps | a returned band lies in the flat part of $f$; add guard bands or set `--a-band` |
| standalone window cuts the occupied bands | `nval` ≠ occupied bands | set `nval` to the occupied count |
| QP block not inside the fitted window | `--qp-rotations` block extends past $[b_0, b_1)$ | widen `ncond` |
| conflicting QP sources | `WFN_qp.h5` with `--eqp-file` or `--qp-rotations` | give one |
| `orbital_totals_t0_gap` | `--velocity` totals at $T = 0$ with overlapping bands or a fractional count | set `occ_smearing_width_ry` |
| `orbital_totals_empty_band`, `htransform_moments_window` | no returned band lies wholly above $\mu$ + 10 kT; the top returned band holds more than $10^{-6}$ occupation | raise `ncond` |
| exciton bands: no `K_POINTS {crystal_b}` block; `interp` on a bulk deck or on IBZ-only ζ; `ongrid` at an off-grid $Q$; failed `bse`-window certificate | §7 gives each condition | add the path block; use `refit` on a bulk deck, or a full-BZ `zeta_q.h5` for `interp`; use `interp` or `refit` off the grid; widen the refit's guard bands or use `--refit-window zeta` |

The method interpolates within the fitted window only; a band that leaves the
window's energy range between grid points is not represented. The coarse grid
must resolve $f(H)_R$ (the locality receipt). The orbital moment omits the
itinerant term.
