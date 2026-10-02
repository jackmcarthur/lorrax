# The velocity operator and parallel-transport links

This page builds up the velocity operator that LORRAX uses for the $q \to 0$
head of the screened interaction and for optical dipoles: the DFT velocity
with its nonlocal pseudopotential term, the extra term a quasiparticle (QSGW)
Hamiltonian adds, why that term needs parallel-transport links on a discrete
k grid, how the links and their stencil are built, and what their error
means. It is for a reader running or changing self-consistent GW. The head
that consumes the velocity is on [four-current heads §3](four-current-head-corrections.md#charge-head)
and, for metals, [the metallic q→0 head](metal-q0-head.md); the
self-consistent loop is [self-consistency](../self_consistency.md).

## 1. Why the head needs a velocity

At small $q$ the density response is governed by matrix elements of
$e^{iq\cdot r}$ between Bloch states, which to first order in $q$ are matrix
elements of the position operator $r$. Position is ill-defined in a periodic
solid, but its interband matrix elements follow from the velocity
$\hat v = i[H, \hat r]$ (atomic units with $\hbar = 1$):

$$v_{mn} = i(E_m - E_n)\, r_{mn} \quad\Longrightarrow\quad
r_{mn} = \frac{v_{mn}}{i(E_m - E_n)}, \qquad m \ne n .$$

The head builds its response tensor $S_{ab}(\omega)$ from $v_{mn}$ over every
energy-ordered pair ([four-current heads §3.1](four-current-head-corrections.md#charge-head-objects)),
and a metal's Drude term from the band diagonal $v_{nn}$. Every error in
$\hat v$ goes straight into the head of $W$ and from there into every $\Sigma$
diagonal. In the Bloch representation $H(k) = e^{-ik\cdot r} H e^{ik\cdot r}$
the velocity is the $k$ derivative, $\hat v = \partial_k H(k)$.

## 2. The DFT velocity

Units are Rydberg atomic units: $\hbar = 1$, $m = 1/2$, so the kinetic energy
is $|k+G|^2$ and its derivative $2(k+G)$; $k$ is in bohr⁻¹ and $v$ in Ry·bohr.
On a plane-wave state the DFT velocity is (`common.mtxel_sweep.dipole_operator`)

$$v_a = 2(k+G)_a \;+\; s\,\frac{\partial V_\mathrm{NL}(k)}{\partial k_a}
\;\Big[+\;\frac{\partial V_U(k)}{\partial k_a}\Big].$$

- **Kinetic term.** $2(k+G)_a$, applied on every spinor component
  (`psp.dft_operators.apply_kinetic_velocity_to_ket`).
- **Nonlocal term.** A local potential commutes with $r$, but the
  pseudopotential's projectors depend on $k+G$ through their radial functions,
  spherical harmonics and atomic phases, so $V_\mathrm{NL}$ does not. With
  $V_\mathrm{NL} = Z E Z^\dagger$ ($Z$ the projector values, $E$ the coupling
  matrix, spin–orbit included), the derivative is analytic,
  $\partial_k V_\mathrm{NL} = (\partial Z) E Z^\dagger + Z E (\partial Z)^\dagger$
  (`psp.vnl_ops.vnl_velocity_block_from_coefficients`).
- **Sign.** $s$ is `vnl_velocity_sign` (deck key, or `--vnl-velocity-sign`
  of the dipole step). The default $s = +1$ gives $\hat v = \partial_k H$,
  the physical velocity. $s = -1$ exists only to reproduce velocity files made
  with that sign; it is stamped as `prov_vnl_velocity_sign`, and every consumer
  refuses a file whose sign differs from the deck's, so the two never mix. The
  term matters: on a bulk Si test deck (100 bands), against BerkeleyGW's head of
  the dielectric matrix without local fields on the same mean field, $s = +1$
  agrees to $10^{-5}$ relative; omitting the term overestimates that head by
  15 %, and $s = -1$ by 31 %.
- **DFT+U.** With `hubbard_input` and `hubbard_occupations`, the derivative of
  QE's ortho-atomic $V_U(k)$ is added (`psp.hubbard_ops`); it refuses $s = -1$.
- **Momentum only.** `--skip-vnl` writes $2(k+G)$ alone (BerkeleyGW's
  `use_momentum`). It is for comparison; the GW head refuses such a file
  (`GATE dft_head_dipole_provenance`).

`python -m psp.get_dipole_mtxels` writes the velocity as `dipole.h5`,
`dipole_cart` $(3, n_k, n_b, n_b)$ on the full k grid
([outputs](../how-to/berkeleygw-users.md#outputs)). The velocity is computed,
not differenced: it is exact at each $k$ for the given states.

**Bispinors.** With `bispinor = true` the state is the kinetic-balance
four-spinor $\Psi = (\Psi_L, (\alpha_\mathrm{fs}/2)\,\sigma\cdot p\,\Psi_L)$
($\alpha_\mathrm{fs}$ the fine-structure constant). The kinetic velocity acts on all four
components and $V_\mathrm{NL}$ on the two large ones
(`common.mtxel_sweep.dipole_operator`).

## 3. The quasiparticle Hamiltonian adds a nonlocal term

A QSGW map works with $H = H_\mathrm{DFT} + \Delta H$. Let $H^\mathrm{QP}$ be
the map's quasiparticle Hamiltonian, $T + V_\mathrm{ion} + V_H + \Sigma^\mathrm{QSGW}$
with $\Sigma^\mathrm{QSGW}$ the Hermitian static self-energy of the map, written as
a matrix in the DFT band basis; then

$$\Delta H = H^\mathrm{QP} - \mathrm{diag}(E^\mathrm{DFT})
= \Delta V_H + \Sigma^\mathrm{QSGW} - V_{xc}$$

(`gw.sc_iteration._head_delta_h_parts`; the band blocks are defined in
[self-consistency §2](../self_consistency.md#2-band-treatment)). $\Sigma$ is
nonlocal, so $\Delta H$ does not commute with $r$, and the quasiparticle
velocity has a second term:

$$\hat v^\mathrm{QSGW} = v_\mathrm{DFT} + i[\Delta H, \hat r] .$$

In the band basis the second term is the **covariant derivative** of the
matrix $\Delta H(k)$:

$$\big(i[\Delta H, \hat r]\big)_{mn} = (D_k \Delta H)_{mn}
= \partial_k \Delta H_{mn} - i\,[A, \Delta H]_{mn},
\qquad A_{mn} = i\langle u_{mk}|\nabla_k u_{nk}\rangle ,$$

with $u_{nk}$ the cell-periodic states and $A$ the Berry connection. The
velocity of the quasiparticle states is then
$v^\mathrm{QP} = U^\dagger(v_\mathrm{DFT} + D_k\Delta H)\,U$, with $U$ the
rotation from DFT to QP states (`gw.qsgw_head.qp_velocity`, the one owner of
the SC velocity).

$\Delta H$ is known only at the grid points, as a matrix in a band basis whose
phases (more generally, whose unitary frame within degenerate sets) are chosen
independently at each $k$. A plain difference
$\Delta H(k+b) - \Delta H(k)$ compares matrices in two unrelated frames and is
meaningless; both terms of $D_k\Delta H$ are gauge-dependent and only their sum
is not. The derivative therefore needs the overlap of the frames at
neighbouring points: links.

## 4. Links and the covariant derivative

**The overlap.** For a step $b$ from $k$ to a neighbouring grid point,

$$M_{mn}(k, b) = \langle u_{mk} | u_{n,k+b}\rangle ,$$

computed from the plane-wave coefficients with the reciprocal-vector shift
that brings $k+b$ back into the zone
(`common.parallel_transport.make_cross_k_overlap`).

**The link** is the unitary closest to $M$: its polar factor. With the
singular-value decomposition $M = P\,s\,Q^\dagger$, $L = P\,Q^\dagger$, keeping
singular values $s_i$ above
$10^{-10}s_\mathrm{max}$ (`--parallel-transport-rcond`;
`distrib_la` polar factor). $L_b(k)$ maps the frame at $k+b$ onto the frame at
$k$ as well as a unitary can: it is the discrete parallel transport.

**Transport and difference.** An operator at a neighbour is carried into the
frame at $k$ before it is differenced:

$$T_{+b}O = L_b(k)\,O(k+b)\,L_b(k)^\dagger, \qquad
T_{-b}O = L_b(k-b)^\dagger\,O(k-b)\,L_b(k-b),$$

$$D_b O = \tfrac12\,(T_{+b} - T_{-b})\,O \quad\text{(second order)},\qquad
D_b O = \tfrac1{12}(-T_{+2b} + 8T_{+b} - 8T_{-b} + T_{-2b})\,O \quad\text{(fourth order)},$$

with the double steps formed from composed links
(`common.parallel_transport.link_covariant_derivative`). To first order
$T_{+b}O \approx O + b\cdot\partial_k O - i[b\cdot A, O]$, so $D_b O \to
b\cdot D_k O$: the transported difference is the covariant derivative, and no
two large gauge-dependent terms have to cancel on the finite grid. The
Cartesian gradient combines the steps,
$\nabla_k O = \sum_d 2 w_d\, b_d\, D_{b_d} O$, with the weights of §5.

**The outer band set.** At the top of a band window, states leave the window
between $k$ and $k+b$, $M$ loses rank, and its smallest singular values
collapse; the link there is not a transport of the same states. The links are
therefore built on an outer band set, by default
$\min(N_\mathrm{WFN}, \lceil 1.25\, N_\mathrm{head}\rceil)$ bands
(`--parallel-transport-bands`), where $N_\mathrm{head}$ is the deck's band
window. $D_k\Delta H$ is taken on the outer set, with $\Delta H$ continued past
the head by its diagonal tail, and only the head bands are used. A run checks
that the outer link's top $N_\mathrm{head}$ singular values all exceed 0.5;
otherwise the head window is hybridized with bands outside it
(`GATE pt_head_window_hybridized`) and the links cannot serve the term (§6).

## 5. The Marzari–Vanderbilt stencil

A finite-difference gradient from a set of steps $b$ with weights $w_b$ is
exact for linear functions, with error $O(b^2)$, when

$$\sum_b w_b\, b_a b_c = \delta_{ac}$$

over the sampled directions (Marzari and Vanderbilt's completeness
condition). `common.parallel_transport.link_stencil`, the one owner of the
step set, finds it from the lattice:

1. Candidates are mesh vectors $b = (n/N)\cdot B$ with integer $|n_i| \le 3$
   ($B$ the reciprocal lattice vectors), grouped into shells of equal length.
2. Shells are taken shortest first. A shell is skipped when one of its
   vectors is parallel to an accepted one, when one of its mesh lines has
   fewer than 3 points (no centred difference), or when it adds no rank.
3. One weight per shell, by least squares on the completeness condition; the
   search stops when the residual falls below $10^{-5}$, else it refuses
   (`GATE pt_link_shell_incomplete`).

Whole shells are closed under the lattice point group, so the $O(b^2)$ error
has the crystal's symmetry: a cubic crystal's head stays isotropic, and
symmetry-forbidden elements stay zero. On bcc Fe ($4^3$ grid) the closed shell
keeps the forbidden off-diagonal elements of the head below $1.6\times10^{-7}$;
three axis steps alone leave 0.5–1.3 % of $\omega_p^2$ there.

| lattice | stored steps (one per ± pair) | vectors |
|---|---|---|
| orthogonal | the three axes | 6 |
| bcc | 6: $(100), (010), (001), (10\bar1), (1\bar10), (01\bar1)$ in reduced mesh units | 12 |
| fcc | 4: the three axes and $(111)$ | 8 |
| hexagonal slab | 3: $(100), (010), (1\bar10)$; $c$ collapsed | 6 |
| hexagonal bulk | 4: the slab's three and the $c$ axis | 8 |

Each shell uses the fourth-order ($\pm2$) form when every one of its mesh
lines has at least 5 points, else the second-order ($\pm1$) form at 3–4
points.

**A collapsed axis** (one k point: the normal of a slab, the transverse axes of
a wire) has no neighbours, and the system is not periodic along it, so the
position operator itself is used: $D_a O = -i[Z_a, O]$ with
$Z_a = 2\pi\langle m|\,\mathrm{wrap}(f_a - f^0_a)\,|n\rangle$, $f_a$ the
fractional coordinate and $f^0_a$ a branch cut placed at the centre of the
largest vacuum gap (`common.parallel_transport.collapsed_axis_vacuum_gap`).
The cut must lie in vacuum: an occupied charge above $10^{-3}$ in the middle
10 % of the gap refuses (`GATE pt_collapsed_axis_cut_density`).

**A two-point axis** has neither a centred difference nor vacuum, so no link
stencil exists. The dipole step writes no link artifact on such a grid; an
unnamed `sc_head_update` then resolves to `dft_velocity` (§7). A deck that
names `parallel_transport` stops when the head opens the missing artifact;
given an artifact written elsewhere (for example velocity only), the head
runs with $D_k\Delta H = 0$ (§6).

## 6. The link error, and what it means

The links are tested where the answer is known exactly. Since
$H_\mathrm{DFT}$ is diagonal in its own basis, $v_\mathrm{DFT} =
D_k\,\mathrm{diag}(E^\mathrm{DFT})$; the dipole step rebuilds
$v_\mathrm{DFT}$ from the links this way and compares it with the computed
velocity on the elements the head reads, the transitions with
$|f_m - f_n| > 10^{-10}$ and the diagonal of partly filled states
(`file_io.parallel_transport.complete_velocity_validation`):

$$\epsilon_\mathrm{link} = \frac{\lVert v^\mathrm{links} - v_\mathrm{DFT}\rVert}{\lVert v_\mathrm{DFT}\rVert}.$$

Above `--parallel-transport-validation-rtol` (default $5\times10^{-3}$) the
dipole step warns and still writes the artifact.

Only $D_k\Delta H$ goes through the links; $v_\mathrm{DFT}$ is exact. Each SC
map therefore bounds the head's link error by

$$\mathrm{bound} = \epsilon_\mathrm{link}\;\frac{\lVert D_k\Delta H\rVert}{\lVert v_\mathrm{DFT}\rVert}$$

on the same elements (`gw.qsgw_head.link_correction_bound`; 0 on a map that
starts from DFT, where $\Delta H = 0$). The second factor is small: at the
SC fixed point of a scalar Fe $4^3$ run (35 bands, 26 protected),
$\lVert D_k\Delta H\rVert/\lVert v_\mathrm{DFT}\rVert = 6.05\times10^{-2}$, so a
link error of a few percent bounds the head's error at a few $10^{-3}$.

$\epsilon_\mathrm{link}$ is a finite-difference error in $k$: it falls as the
grid is refined, and a large value means the k grid is too coarse for the
velocity, which is underconvergence of the calculation, not a defect of the
links. Complete links therefore always serve $D_k\Delta H$, and the bound is
printed as information and gates nothing. On monolayer MoS₂ with spin–orbit
coupling on a 3×3 grid, $\epsilon_\mathrm{link} = 9.2\,\%$, and serving the
term moves the SC gap from 5.167 to 4.483 eV; the grid, not the term, is the
problem to fix.

Links that cannot serve the term at all set $D_k\Delta H = 0$ for the whole
run, with one log line naming the reason: a two-point axis, an incomplete
artifact (connection or validation stage missing), or a hybridized window
edge (§4). The head then runs on $U^\dagger v_\mathrm{DFT} U$, and the run ends
with one line saying whether the term was served on every map. There is no
switch to another head mid-run.

## 7. The heads that use the velocity

`sc_head_update` selects how each SC map rebuilds the head
([input reference](../input_reference.md)):

| `sc_head_update` | velocity each map | reads | admitted on |
|---|---|---|---|
| `off` | none: the fixed DFT response | `dipole.h5` | every deck |
| `dft_velocity` | $U^\dagger v_\mathrm{DFT}\,U$ | `dipole.h5`, or the artifact's velocity stage | insulators; metals with the shared-pole W and `head_correction = no_local_fields` (or `full` on a scalar deck) |
| `parallel_transport` | $U^\dagger(v_\mathrm{DFT} + D_k\Delta H)\,U$ | the full link artifact | insulators; metals with the shared-pole W on a scalar or `full_shared_pole` deck |
| `interband_commutator` | $U^\dagger(v_\mathrm{DFT} + [\Delta H, \mathcal W])\,U$ | the artifact's velocity stage (`vnl_included = 1`) | insulators only |

When `sc_head_update` is not named on an SC deck with the shared-pole W and
the head on (scalar or `full_shared_pole`), it resolves to `parallel_transport`
if `parallel_transport_file` exists (the dipole step writes it by default),
else `dft_velocity` if `dipole.h5` exists, else `off` (a `full_shared_pole`
deck takes `dft_velocity` even then, because its four-current bank reads
`dipole.h5` on every route); the choice is printed under
`[config provenance]`. Other combinations refuse by name
(`GATE metal_sc_head_update_disabled`,
`GATE sc_head_interband_commutator_insulator_only`). One head serves the whole
run.

**The interband commutator** approximates $i[\Delta H, r]$ without links, on any
grid, by keeping only the position operator's cross-gap block. With
$v < n_\mathrm{occ} \le c$,

$$\mathcal W_{vc} = \frac{v_{vc}}{E_v - E_c} = i\,r_{vc}, \qquad
\mathcal W_{cv} = -\mathcal W_{vc}^*, \qquad \mathcal W = 0 \text{ within each class},$$

so $\mathcal W = i r^{VC}$ is anti-Hermitian and
$i[\Delta H, r^{VC}] = [\Delta H, \mathcal W]$
(`gw.qsgw_head.interband_commutator_velocity`). It omits the intraband
connection, so it is exact when $\Delta H$ does not mix valence and
conduction states, and its error is first order in the cross-gap mixing
$\theta = \max_k\lVert U_{VC}\rVert_F$, where $U_{VC}$ is the block of the
DFT-to-QP rotation $U$ that takes DFT valence states into QP conduction
states; each map prints $\theta$. On monolayer MoS₂ with bispinors at
$\theta \approx 0.055$ the head's $S_{zz}$ sits 1.3 % from the exact
position-operator head, about $\theta/4$. Every same-class pair is excluded,
degenerate or not: inside a class $\mathcal W$ has no gap in its denominator,
near-degenerate pairs make it arbitrarily large, and the $\partial_k\Delta H$
that would cancel it has no stencil-free form (excluding only exact
multiplets gives a Si 6×6×6 spin–orbit head 8.8 times the link head). The
artifact must stamp `vnl_included = 1`, because
$r_{mn} = -i v_{mn}/(E_m - E_n)$ holds only for the velocity of the
Hamiltonian whose energies divide it. A cross-gap pair
closer than $10^{-6}$ Ry refuses (`GATE sc_head_interband_commutator_gap`).
On a collapsed axis it uses the exact $Z_a$ of §5.

**The record.** Every map with a velocity head prints one block to
`gwjax.out`, `SC head velocity, map N`: each term's share (`p`, `V_NL`,
`Sigma`, and the total) of $\omega_p^2$ for a metal (eV², Fermi-surface
weights) or of $S_{aa}(0)$ for an insulator (%), then the link bound and the
band gap.

**QSGW dipoles.** The accepted final map writes $U^\dagger v\,U$ with the
same velocity to `dipole_qsgw.h5`; what it holds and what it pairs with:
[self-consistency](../self_consistency.md#qsgw-dipoles).

## 8. The four-current (bispinor) case

On `bispinor_gw = full_shared_pole` (the route that screens the charge and
current channels together; [bispinor GW](bispinor-gw.md)) the four-current
bank builds a direct Γ head, the first-order $q \to 0$ head assembled from
velocity matrix elements without a local-field fold
([four-current heads §5](four-current-head-corrections.md#direct-bulk-head)),
from the same velocity: a charge vertex $v_{nm}/\Delta_{nm}$ and a current
vertex $(\alpha_\mathrm{fs}/2)\,v_{nm}$ (`gw.photon_direct_head`). `off`
builds it on the DFT state, `dft_velocity` on each map's state with the
rotated `dipole.h5` velocity, `parallel_transport` with each map's
$v_\mathrm{DFT} + D_k\Delta H$ (`gw.sc_iteration.photon_head_state`);
`interband_commutator` refuses (`GATE full_shared_pole_head_update`). On
`bare_transverse` decks (the default route: charge screening plus the bare
transverse interaction) `parallel_transport` refuses on metals. The static
Hall current of the packed static photon head
([four-current heads §4](four-current-head-corrections.md#coupled-gamma-solve))
uses the same signed nonlocal term,
$\Gamma_a = \alpha^\mathrm{D}_a + s(\alpha_\mathrm{fs}/2)\,\partial_{k_a} V_\mathrm{NL}$
with $\alpha^\mathrm{D}_a$ the Dirac velocity matrices
(`common.mtxel_sweep.uniform_gauge_operator`), and its artifact is stamped
and checked like `dipole.h5`.

## 9. The artifact: `parallel_transport.h5`

The dipole step writes it beside `dipole.h5` by default
(`--parallel-transport-out`; `--no-parallel-transport` skips it). It is not
written with `--skip-vnl`, `--with-finite-q`, `--w-av-only`,
`--static-gauge-hall-only`, `--vnl-mode numeric`, or on a grid with a
two-point axis. The file is written as `.partial` and renamed only after its
links complete. Writer and reader: `file_io.parallel_transport`; schema
version 4 (`SCHEMA_VERSION`).

| content | datasets |
|---|---|
| velocity stage | `velocity_dft_cart` $(3, n_k, n_b, n_b)$ on the outer band set; `velocity_kinetic_cart` ($p$ alone); `dft_energies_ry_full`, `dft_occupations_full`; `collapsed_position_reduced` ($Z_a$) on a collapsed grid |
| links | `links_ibz` $(n_k, n_d, n_b, n_b)$, orientation $L_d(k)\,X(k+b_d)\,L_d(k)^\dagger$; `singular_values_ibz`; `source_steps` and the neighbour tables |
| stencil | `link_stencil_orders`, `link_stencil_step_orders`, `link_stencil_coefficients`, `collapsed_axis_flags`, `collapsed_axis_center_frac` |
| connection | `berry_connection_reduced`, `berry_connection_cart` |
| stamps | `schema_version`, `band_stop` (the outer count), `vnl_velocity_sign`, `vnl_included`, the WFN fingerprint, `kgrid`, unit stamps, `connection_complete`, `velocity_validation_complete`, and the validation results (`velocity_validation_head_set_relative_l2` is $\epsilon_\mathrm{link}$) |

Links are stored for the irreducible wedge only when the steps are the three
axes and the point group acts on them as signed permutations; otherwise
(bcc, fcc, hexagonal) they are stored for the full grid.
`--parallel-transport-velocity-only` writes the velocity stage alone, which is
all `dft_velocity` and `interband_commutator` need; under
`parallel_transport` such a file counts as incomplete links. A schema-3 file,
or one whose `source_steps` differ from the current stencil, refuses under
`parallel_transport` (`GATE pt_link_stencil`); rerun the dipole step.
