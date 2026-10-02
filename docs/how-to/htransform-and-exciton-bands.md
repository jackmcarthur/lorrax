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

## 2. The whole-state Galerkin basis

Stack the states of the fitted band window $[b_0, b_1)$ at every coarse $k$
as the rows of one matrix

$$\Psi_{(k,n),(s,r)} = \psi_{nk}(s, r), \qquad n \in [b_0, b_1),$$

with $s$ the spinor component and $r$ the points of the real-space FFT grid.
Each row is the full Bloch state, phase $e^{ik\cdot r}$ included. The row
space of $\Psi$ contains every state LORRAX will interpolate, so it is the
natural basis. It is too large to use directly ($N_k (b_1-b_0)$ rows), so
`isdf.galerkin.fit_galerkin_basis` selects a subset of rows:

1. A deterministic Gaussian sketch of a random candidate subset of states,
   followed by pivoted Cholesky of the candidates' Gram matrix, reproduces the
   column choice of randomized QR with column pivoting (QRCP). The pivots
   $X$ are physical states, rows of $\Psi$.
2. The pivots are orthonormalized, $XX^\dagger = LL^\dagger$,
   $B = L^{-1}X$. $B_\alpha(s,r)$, $\alpha = 1 \dots N_B$, is the basis.
3. Every state is projected once: $C = \Psi B^\dagger$, so
   $c_{nk}[\alpha] = \langle B_\alpha | \psi_{nk}\rangle$, shape
   $(N_k, b_1-b_0, N_B)$.

$N_B$ is set by one tolerance, `htransform_qr_eps` (default $10^{-3}$, the
relative QR-diagonal threshold at which pivoting stops). The search is capped
at $\lceil$`htransform_rank_multiplier`$\cdot (b_1-b_0)\rceil$ directions
(default 20). A fit whose delivered rank exceeds 90 % of the cap refuses
(unless the cap already spans every state), because then the cap and not the
tolerance set the rank (`galerkin.fit_galerkin_basis`). `htransform_qrcp_seed`
seeds the candidate shuffle and sketch, and the report prints the candidate
and pivot hashes. Pivots whose sketched residuals agree to $10^{-8}$ are
broken by the lowest index, so the basis does not depend on the process
count.

$B$ is never stored on the full grid. The fit keeps $C$, the basis values at
the ISDF centroids $B_\alpha(s, r_\mu)$ (which the BSE consumers need) and the
pivot factor, and writes them to `galerkin_dft.h5` beside the deck. A later
run reuses the file only when its provenance matches exactly (WFN fingerprint,
centroid SHA-256, band window, grids, spinor mode, QRCP controls); otherwise it
refits in memory and leaves the file unchanged. The log says `REUSED`,
`REFIT` or `FITTED and published`.

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
invariant under any unitary mixing of the window's states. The interpolation
is then:

$$f(H)_R = \frac{1}{N_k}\sum_k e^{2\pi i k\cdot R} f(H)_k, \qquad
f(H)_q = \sum_R e^{-2\pi i q\cdot R} f(H)_R,$$

with $R$ on the coarse grid's supercell, centred on 0
(`fh_interp.build_fH_R`, `fh_interp.build_R_grid_np`). At each path point
the driver diagonalizes $f(H)_q$: the eigenvalues are $f(\varepsilon_n(q))$
and the eigenvectors $c_n(q)$ are the states in the basis $B$. Newton
iteration inverts $f$ (`fh_interp.newton_inv`, converged to
$\max|f(x)-y| \le 10^{-12}$ Ry in at most 50 steps, else refused). The state
at a centroid follows as $\psi_n(q; s, r_\mu) = \sum_\alpha c_n(q)[\alpha]\,
B_\alpha(s, r_\mu)$, which is what the BSE consumers take.

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

Two options, for spinor WFNs, attach physical observables to the bands
(`bandstructure.orbital`). Both need the `*.upf` files beside the deck.

**Operators.** On the coarse full Brillouin zone the driver forms the band
matrices $\langle\psi_{nk}|O|\psi_{mk}\rangle$ of three kinds of operator:
the Pauli matrices $\sigma_{x,y,z}$; for each atom $I$, the atomic-sphere
angular momentum $L^I_{x,y,z}$, built from the Löwdin-orthogonalized atomic
wavefunctions (`PP_PSWFC`) of the pseudopotential, with j-averaged radial
functions (the projector of QE's `projwfc`); and, per requested channel, the
projector on one element's $l$ shell. Each operator is carried into the
Galerkin basis as $C^T O\, C^*$ and Fourier-interpolated exactly as $f(H)$ is.
On the QP route only the eigenvectors change: $U$ is unitary on the fitted
window, so the operator image is the same.

**Magnetization axis.** $\hat n$ is read from the QE schema that
authenticates the WFN (`orbital.magnetization_axis`): the SCF output's
`total_vec` when it is nonzero; otherwise the input moment direction
(`angle1`/`angle2`, stored as `spin_teta`/`spin_phi`) of the magnetic species;
otherwise $z$. The report names the source.

**`--color spin` and `--color orbital:[EL:]l`** (repeatable; for example
`orbital:d` or `orbital:Fe:d`) color the path bands by $\langle\sigma\cdot\hat
n\rangle \in [-1, 1]$ or by the channel's character $\in [0, 1]$, and write
`bands_<color>.png` (colon replaced by underscore) with the energy axis
$E - E_F$, $E_F$ from the coarse grid. `band_operators_path.npz` holds every
path operator matrix (`path_operators`, shape (path point, operator, band,
band)), `operator_names`, `kpath_frac`, `x_path`, `spin_axis`, `energies_ev`
(relative to the path VBM) and `energy_reference_ev` (the path VBM relative to
the coarse $E_F$).

**`--moments-grid NX NY NZ`** interpolates $f(H)$ and every $\sigma_a$ and
$L^I_a$ to that uniform grid, finds $E_F$ there by Fermi–Dirac occupation of
the window's electrons at `occ_smearing_width_ry` ($10^{-4}$ Ry when unset),
and sums

$$m_\mathrm{spin} = \frac{1}{N_q}\sum_{q,n} f_{qn}\langle qn|\boldsymbol\sigma|qn\rangle,
\qquad m^I_\mathrm{orb} = \frac{1}{N_q}\sum_{q,n} f_{qn}\langle qn|\mathbf L^I|qn\rangle .$$

`moments.txt` gives the three components and the projection on $\hat n$ of
each, the grid $E_F$, and the occupation of the top returned band (a warning
to raise `ncond` above $10^{-6}$). The same sums taken directly on the coarse
grid, with no interpolation, go to `htransform.out` as a check. Units and
sign follow QE: $\mu_B$ per cell, $m_\mathrm{spin} = n_\uparrow - n_\downarrow$
(QE's "total magnetization"), and $m_\mathrm{orb}$ with the same sign flip.
The physical magnetic moments are therefore
$\boldsymbol\mu_\mathrm{spin} = -\tfrac{g}{2}\mu_B\, m_\mathrm{spin}$ and
$\boldsymbol\mu^I_\mathrm{orb} = -\mu_B\, m^I_\mathrm{orb}$ ($\hbar = 1$,
$g \approx 2$), and $m_\mathrm{orb}/m_\mathrm{spin} > 0$ means
$\mathbf L \parallel \mathbf S$. The grid runs one $q_z$ plane per pass, so
one dense $(N_k, N_B, N_B)$ operator image is resident at a time.

The orbital moment is the atomic-sphere part only. The itinerant
(modern-theory) orbital moment needs the Berry connection of the states and is
not formed.

*Example.* bcc Fe with spin–orbit coupling on a $20^3$ grid, DFT states:
interpolated to QE's own $8^3$ SCF grid, $m_\mathrm{spin} = 2.30679\,\mu_B$
against QE's total magnetization $2.30974\,\mu_B$, and $E_F$ 18.4859 against
18.4833 eV. On a $40^3$ grid, $m_\mathrm{spin} = 2.28693\,\mu_B$ and the Fe
$d$ orbital moment along $\hat n$ is $0.05254\,\mu_B$ ($\mathbf L \parallel
\mathbf S$); the run takes 92–106 s on 9 A100 GPUs (a 3 × 3 mesh) at a device peak of
17.2 GiB.

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
  momentum. The tile is taken at $\mathrm{wrap}(-Q)$, and at finite $Q$ the
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
| `htransform_qr_eps` / `htransform_rank_multiplier` / `htransform_qrcp_seed` | 1e-3 / 20 / 0 | Galerkin basis rank tolerance, search cap, seed (§2) |
| `-wfn` / `--wfn-file` | the deck's `wfn_file` | another WFN, for example `WFN_qp.h5` |
| `-o` / `--output-file`, `--report-file` | `bandstructure.dat`, `htransform.out` | outputs |
| `linalg`, `--eigh-backend` | `local` | layout of the $f(H)_q$ eigensolve; the flag is a debug override |
| `get_centroids_fi`, `kgrid_fi`, `wfn_fi_min`/`wfn_fi_max`, `wfn_fi_q_chunk` | off | BSE handoff: fine-grid ψ at the coarse centroids (`bse_setup.compute_wfns_fi`); keep at least 4 bands between `wfn_fi_max` and the top of the window |
| `--plot` | off | show the band plot |

## 9. Refusals and limits

| refusal | cause | fix |
|---|---|---|
| QRCP search saturates its cap | the band window needs more basis directions than `htransform_rank_multiplier` allows | read the projection receipts in the report, then raise the multiplier |
| `f-shoulder` | a returned band is absent from $f(H)$ at some coarse $k$ | add guard bands |
| Newton not converged | $\max|f(x) - y| > 10^{-12}$ Ry after 50 steps | a returned band lies in the flat part of $f$; add guard bands or set `--a-band` |
| standalone window cuts the occupied bands | `nval` ≠ occupied bands | set `nval` to the occupied count |
| QP block not inside the fitted window | `--qp-rotations` block extends past $[b_0, b_1)$ | widen `ncond` |
| conflicting QP sources | `WFN_qp.h5` with `--eqp-file` or `--qp-rotations` | give one |
| exciton bands: no `K_POINTS {crystal_b}` block; `interp` on a bulk deck or on IBZ-only ζ; `ongrid` at an off-grid $Q$; failed `bse`-window certificate | §7 gives each condition | add the path block; use `refit` on a bulk deck, or a full-BZ `zeta_q.h5` for `interp`; use `interp` or `refit` off the grid; widen the refit's guard bands or use `--refit-window zeta` |

The method interpolates within the fitted window only; a band that leaves the
window's energy range between grid points is not represented. The coarse grid
must resolve $f(H)_R$ (the locality receipt). The orbital moment omits the
itinerant term.
