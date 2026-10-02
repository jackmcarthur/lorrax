# Self-consistent GW (QSGW)

This page explains how LORRAX solves quasiparticle self-consistent GW (QSGW)
when a deck sets `qp_solver = self_consistent`. It covers what one iteration
computes, how each band is treated, how the q → 0 head follows the iteration,
how the Σ(ω) frequency grid is held, how iterates are mixed and judged
converged, and what the run writes and how it restarts. It assumes GW and the
ISDF formulation; read [production QSGW](how-to/production-qsgw.md) first for
the recipe and its error budget, and the [input reference](input_reference.md)
for every key named here.

Other pages own the neighbouring material: the Σ(ω) quadrature rules
([the Σ(ω) quadrature problem](theory/sigma-quadrature-problem.md)), the
shared-pole W ([theory](theory/shared-pole-w-model.md),
[implementation](architecture/shared_pole_model.md)), the metallic head
([the metallic q → 0 head](theory/metal-q0-head.md)) and the Hartree rebuild
([direct Hartree field](theory/hartree.md)).

**Notation and terms.**

- $k$ runs over the loop's k-set (§1) and $m, n$ over bands.
  $E^{\rm DFT}_{nk}$ and $\psi^{\rm DFT}_{nk}$ are the DFT eigenpairs read
  from `WFN.h5`. $\mu$ is the Fermi level of the current map. Energies are in
  Ry inside the code and in eV in decks and logs.
- nelec is the WFN's occupied-band count (`wfn.nelec`, the largest occupied
  band index over k), so it counts bands, not electrons, and already includes
  the spin or spinor degeneracy of each band.
- The Σ band window is $[b_0, b_3)$ with $N_b = b_3 - b_0$ bands; the loop
  requires $b_0 = 0$ because its occupations count bands from band 0.
- The **carry** is the state passed from one map to the next
  (`sc_iteration.SCState`): the Hamiltonian $H$ of §1 plus the small tables
  that ride with it (the previous map's $Z$ per state, the occupation state).
- ζ is the ISDF interpolation basis fitted to pair densities, and τ the
  imaginary-time variable of the Σ(ω) quadrature
  ([physics](theory/physics.md),
  [the Σ(ω) quadrature problem](theory/sigma-quadrature-problem.md)).
- GN-PPM and HL-PPM are the Godby–Needs and Hybertsen–Louie plasmon-pole
  models of W; MPA is the multipole route (`compute_mode = mpa`), whose
  production W is the shared-pole model.
- A bispinor (four-current) deck screens the charge and the Dirac current
  together; its W has charge–charge (CC), charge–current (CT/TC) and
  current–current (TT) **sectors** ([bispinor GW](theory/bispinor-gw.md)).
- An **ordered store** is a W model of a time-reversal-broken system (a
  magnet), stored with both particle–hole orientations.
- The **near grid** is the uniform Σ(ω) grid around $E_F$ at the deck
  broadening η; coarse windows (§2) are separate grids below it.
- A step **refuses** when it stops the run with an error naming a gate,
  `GATE name`, the label to search for in the code. An **authenticated** file
  is one whose provenance stamps (source WFN fingerprint, band and k tables,
  operator settings) were checked against this run before use.

## 1 The map {#1-the-map}

QSGW replaces the energy-dependent self-energy $\Sigma(\omega)$ by a static
Hermitian operator built from it, and looks for the Hamiltonian that
reproduces itself. LORRAX carries that Hamiltonian in the fixed DFT basis,

$$
H_{k,mn} = \langle \psi^{\rm DFT}_{mk} | H | \psi^{\rm DFT}_{nk} \rangle ,
\qquad H:\ (n_k, N_b, N_b)\ \text{complex}.
$$

A fixed basis gives every iterate the same coordinates, so differences and
linear combinations of iterates, which the mixer forms (§5), are meaningful.
Each state keeps a DFT label: the DFT band it came from, assigned on every map
by overlap (§5), never by sorted position.

One map $F: H \mapsto H'$ (`sc_iteration.gw_iteration_map`) does six things:

1. Diagonalize $H_k = U_k\,\mathrm{diag}(E_k)\,U_k^\dagger$, with
   $U_{k,mn} = \langle \psi^{\rm DFT}_m | \psi^{\rm QP}_n \rangle$.
2. Rotate the DFT orbitals of $[b_0, b_3)$:
   $\psi^{\rm QP}_n = \sum_m U_{mn}\,\psi^{\rm DFT}_m$. The rotation always
   starts from the DFT orbitals, so no product of rotations accumulates
   round-off.
3. Solve the occupations of the new spectrum (μ, below) and rebuild the
   Hartree potential $V_H$ from the rotated occupied orbitals.
   `density_self_consistent` is on by default under `self_consistent`. An
   explicit `false` keeps the DFT $V_H$ as an announced comparison mode, and a
   bispinor deck refuses it, because the Dirac current changes with the
   orbitals as the charge does.
4. Build $\chi_0 \to W \to \Sigma_c(\omega)$ and $\Sigma_x$ from the rotated
   orbitals and energies through the run's screening and Σ route
   (`sigma_dispatch.compute_sigma_xc`).
5. Form the QSGW operator in the QP basis
   (`qsgw_utils.build_qsgw_sigma_xc`):

   $$
   \Sigma^{\rm QSGW}_{ij}(k) = \tfrac12\big[\Sigma_{ij}(k, \tilde E_{ik})
   + \Sigma_{ij}(k, \tilde E_{jk})\big]^{\rm h},
   \qquad \Sigma = \Sigma_x + \Sigma_c ,
   $$

   where $[\cdot]^{\rm h}$ is the Hermitian part, $\Sigma(\tilde E)$ is read by
   linear interpolation on the sampled ω grid (§4), and $\tilde E$ is the
   state's current energy (a pinned semicore state reads at its DFT energy,
   §2).
6. Rotate back and set
   $H' = T + V_{\rm ion} + V_H + U\,\Sigma^{\rm QSGW}U^\dagger$, where
   $T + V_{\rm ion}$ is the `kin_ion.h5` operator; then apply the frozen-core
   and semicore blocks of §2.

No Z-factor enters the iteration. Σ is read at the current energies, which at
the fixed point are the QP energies, so the map's own output solves the QP
equation there.

**Fermi level.** On an insulator μ is midway between the highest occupied
and the lowest empty level. By default (`density_self_consistent`) the
occupied set comes from a k-weighted step fill (`efermi.fermi_level_step`);
with it off, the occupied set is the lowest nelec bands at every k. Both give
the same μ when the fill is gapped. On a metal each map solves fixed-N
Fermi–Dirac occupations of its own input spectrum (§7). μ is a function of H
and is never mixed.

**Purity.** $F$ reads only $H$ and quantities fixed for the run, so evaluating
one $H$ twice returns the same $H'$ bit for bit, and every evaluated pair
$(H, F(H) - H)$ is valid secant data for the mixer.

**Map 0 is the one-shot.** The initial carry is $\mathrm{diag}(E^{\rm DFT})$.
Map 0 takes $E = E^{\rm DFT}$ and $U = I$ exactly instead of diagonalizing,
so it reproduces the one-shot calculation (`qp_solver = one_shot_dft`) bit for
bit, and an SC run computes no separate one-shot. Diagonalizing would return
the eigenvalues to about one ulp, re-sort them and pick an arbitrary gauge
inside degenerate multiplets; the GN-PPM fit amplifies ulp noise in the
energies (MoS2 3×3, +1 ulp on every energy: max $|\Delta\Sigma_c|$ = 1.28 eV).

**k-sets.** With `sc_on_ibz = true` (the default) $H$, $E$, $U$ and the
carried state live on the star wedge, one k per orbit of the symmetry group.
Σ is a convolution over the k grid evaluated by FFT, so steps 2–5 run on the
full grid; one seam at the end of the map
(`sc_iteration._sc_output_tables_on_loop_kset`) selects the wedge rows of Σ
together with the $U$ that defines their basis. Every k sum in the loop (the
mixer's metric, the tail fit, the electron count) weights a wedge row by its
star size.

**Cost.** Each map costs one full $\chi_0 \to W \to \Sigma$ evaluation. One
line per map, `SC map N stages (s)`, splits its wall into W response, Σ rule
refit, Σ rule plan, Σ τ sweep, Σ exchange, Σ Hartree and the rest.

## 2 Band treatment {#2-band-treatment}

**The QP matrix** is $[b_0, b_3)$ with $b_3$ = nelec + `ncond`. Every state in
it keeps its full Σ row and column and rotates among the others; no band
above $b_3$ mixes into it. Two exclusive forms request it, and giving both
refuses (`GATE band_request_forms`):

- `number_bands_protected = N`: every occupied band plus conduction bands up
  to N bands in total. It is resolved against the WFN to nval = the occupied
  count and ncond = the rest.
- `nval` / `ncond`: `ncond` sets $b_3$; `nval` sets the bottom of the ISDF
  pair-density window and the semicore floor below.

$b_3$ must lie inside the ζ fit's left band range
(`gw_init.assert_qp_matrix_fitted`, `GATE qp_matrix_zeta_left`): a state
outside it would carry Σ built on pair densities the fit never saw. $b_3$ is
a convergence parameter; its measured error is in
[production QSGW](how-to/production-qsgw.md#error-budget).

Each band belongs to one class:

| class | bands | on every map | reason |
|---|---|---|---|
| frozen core (opt-in) | the lowest `sc_frozen_core_bands` (default 0) | block held at $\mathrm{diag}(E^{\rm DFT})$ with no coupling to other bands; stays in every band sum | removes deep bands from the loop at no Σ cost |
| semicore (coarse) | occupied states below the coarse floor (below) | full Σ row, read on held coarse windows at $\eta_{\rm semi}$ = 5 eV; with `sc_semicore = dft` its own block stays at DFT | keeps semicore mixing without stretching the deck-η grid to semicore depth |
| protected | every other QP-matrix state | full Σ row, read at its own energy on the near grid at the deck η | the states the calculation is for |
| scissored tail | $[b_3,$ `number_bands`$)$ | DFT orbitals; energies shifted rigidly by β each map; enter G and χ₀ only, with no Σ and no mixing | completes the band sums without the cost of rotating them |

Production decks do not freeze semicore: a frozen block drops both the
semicore correction and its mixing with the valence, and both matter on Fe
3s/3p and the CrI3 I 5s levels. They use the semicore class instead.

### The semicore class

The class changes where a state's $\Sigma_c(\omega)$ is read, not which bands
are counted. It is built once from the DFT ladder
(`gw_init.coarse_class_for_deck`, `band_partition.semicore_floor`) and kept on
`meta.coarse_class`:

- Under `number_bands_protected`: the occupied bands below the lowest band
  gap of at least 4 eV over all k (`band_partition.SEMICORE_GAP_EV`) that lies
  under every requested band within μ ± 10 eV; none when no such gap exists. That
  request protects every occupied band, so the class needs its own boundary;
  4 eV is 16 deck broadenings at the default η = 0.25 eV, so a coarse window
  never reads a band the near grid resolves.
- Under `nval` / `ncond`: every occupied state below the minimum energy of the
  lowest requested valence band (nelec − nval at each k; `sigma_omega_min_ev`
  can only lower the floor). The rule is on energy, so a coarse state inside
  the near grid's lower pad reads the near grid.

Only the MPA and shared-pole Σ, scalar or four-current sector, read coarse
windows (`qp_support.semicore_patch_route`); a PPM or static Σ has no coarse
class.

**Coarse windows** (`qp_support.semicore_patches_ev`). The quadrature cost of
a Σ window grows with bandwidth/η, so a deck-η grid stretched down to the
semicore would multiply it. The semicore is instead read on windows of its
own at $\eta_{\rm semi}$ = 5 eV (`qp_support.SEMICORE_ETA_EV`), sampled every
$\eta_{\rm semi}/2$. There is one automatic window per coarse manifold
(coarse levels separated by a gap wider than twice the 2 eV pad), padded 2 eV
around the DFT energies, planned at map 0 and held; a coarse state whose read
support leaves it extends it, and it never shrinks. `sigma_omega_patches_ev`
`lo:hi:eta` triples (eV about $E_F$) are user windows: coarse states inside
one read it at its own η. `GATE sigma_coarse_window` refuses a malformed,
overlapping or sub-deck-η triple, and any triple in a run without a coarse
class. The Σ rules certify coarse windows at
max(`sigma_quadrature_eps`, 3e-3) (`qp_support.SEMICORE_EPS`): the crossing
node count falls with $\ln(1/\epsilon)$, and the Fe 4³ coarse window takes
188 nodes at 3e-3 against 253 at 1e-4. Adjacent automatic windows of one η
share a rule window when that lowers the summed node count
(`sigma_box_plan._coarse_runs`, decided at map 0 and held).

The broadening is also what makes the semicore converge: at the deck η the Fe
3s quasiparticle weight $Z$ leaves $(0, 1]$ from map 1 and the loop stalls,
while at 5 eV every coarse $Z$ stays inside (one `SC semicore Z` line per
map). Its cost is a systematic error reported apart from the 1 meV budget of
the controllable errors: converged $E_F \pm 1$ eV std/max 3.6/15.9 meV on
Fe 4³ (against a comparison build with `qp_support.SEMICORE_ETA_EV` set to
1 eV) and 2.9/20.0 meV on MoS2 3×3 (against the semicore read at the deck
η).

**Semicore pin** (`sc_semicore = dft`, the default). The pseudopotentials are
fitted to DFT, so the semicore levels stay at their DFT energies while their
mixing with the protected states is kept. With $P_S$ the projector on the
coarse states' DFT orbitals (labels fixed at map 0), each map sets

$$
P_S H P_S = \mathrm{diag}(E^{\rm DFT}_s)
$$

(`sc_iteration._pin_semicore_block_to_dft`) and keeps every
protected–semicore element. In step 5 of the map a QP state whose DFT label
(§5) is a coarse label s reads Σ at $\tilde E = E^{\rm DFT}_s$, so its end of
every $\Sigma^{\rm QSGW}_{ps}$ is $\Sigma_{ps}(E^{\rm DFT}_s)$ and the coarse
windows, planned on DFT energies, never need to follow it. The semicore QP
energies still move by the level repulsion of the kept mixing,
$-\sum_p |H_{ps}|^2/(E_p - E_s)$ to second order. `sc_semicore = qp` lets the
class move with its own Σ, read at its own QP energy. On Fe 4³ and MoS2 3×3
the pinned semicore converges within 27 meV of DFT, while under `qp` it lands
0.1–6 eV deeper, in the same number of maps (Fe 14, MoS2 8). A run without a
coarse class logs that there is nothing to pin.

### The scissored tail

Bands $[b_3,$ `number_bands`$)$ keep their DFT orbitals and take one rigid
shift (`sc_iteration._fit_sum_band_tail`, `scissor.fit_scissor`):

$$
E_{nk} = E^{\rm DFT}_{nk} + \beta, \qquad
\beta = \frac{\sum_{kn} w_k u_{nk}\,(E^{\rm QP}_{nk} - E^{\rm DFT}_{nk})}
{\sum_{kn} w_k u_{nk}}, \qquad
u_{nk} = \begin{cases} \min(Z_{nk}, Z_{nk}^{-1}) & Z_{nk} > 0 \\ 0 & Z_{nk} \le 0 \end{cases}
$$

The sum runs over the conduction states of the QP matrix whose Σ was read on
the sampled grid. $w_k$ is the star weight and
$Z_{nk} = (1 - \partial_\omega \mathrm{Re}\,\Sigma_{nn}(\omega)|_{E_{nk}})^{-1}$.

- The tail is not in the rotated subspace, so it can only follow the
  window's correction rigidly. The mean over every trusted conduction state,
  rather than the lowest multiplet, keeps one localized band from setting it
  (CrI3: the lowest multiplet moves +7.0 eV, the window mean +3.5 eV).
- The weight $u$ keeps a state on a satellite or near a pole of Σ, which has
  small $Z$, from dragging the tail. With
  $s = \partial_\omega \mathrm{Re}\,\Sigma$, $u = Z$ for $s \le 0$, $1 - s$
  for $0 < s < 1$ and 0 for $s \ge 1$, so $u$ is continuous in $s$ and the
  map has no jump where a $Z$ crosses 1.
- A state off the sampled grid is excluded, because its energy did not come
  from its own $\Sigma(E)$.
- $Z$ comes from the previous map and rides the carry (`SCState.tail_z_kn`),
  because this map's Σ exists only after the tail has fed χ₀ and W. Map 0
  uses unit weights. With no qualifying state the tail stays at $E^{\rm DFT}$
  and the log says so.
- On a metal the bands that cross $E_F$ enter neither the valence nor the
  conduction class (`scissor.classify_scissor_bands`).

### Degenerate multiplets

$H$ keeps the full operator inside a degenerate multiplet. Averaging only the
diagonal of a degenerate block would depend on the arbitrary basis inside it
and break the symmetry of the next map. BerkeleyGW's degeneracy averaging
(`no_degen_averaging`) applies only to reported diagonals.
`sc_exact_degeneracy_tol_ev` (at most and by default 1e-4 eV) groups levels
into multiplets for the label assignment of §5; it is not a convergence knob.

## 3 The head and the velocity on each map

The $q \to 0$ element (head) of $\chi_0$ and $W$ diverges like $1/q^2$ and
cannot be sampled on a finite k grid; LORRAX completes it from the velocity
matrix elements and the energies. In QSGW the Hamiltonian changes, so its
velocity changes: $v^{\rm QP} = U^\dagger(v^{\rm DFT} + D_k\Delta H)\,U$, with
$D_k\Delta H = i[\Delta H, r]$ the covariant k derivative of the map's
$\Delta H = H - \mathrm{diag}(E^{\rm DFT})$. The operator, the
parallel-transport links that form $D_k\Delta H$, their stencil and error,
and the artifact are owned by [the velocity operator](theory/qp-velocity.md).
`sc_head_update` chooses how much of this each map uses. One owner,
`qsgw_head.qp_velocity`, forms the velocity for the head and for
`dipole_qsgw.h5`:

| `sc_head_update` | velocity on each map | needs |
|---|---|---|
| `off` | none: the DFT head, fixed for the run (under `head_correction = full` folded through each map's W) | — |
| `dft_velocity` | $U^\dagger v^{\rm DFT} U$ | the velocity stage of `parallel_transport.h5` when the file exists, else `dipole.h5`; on a metal always `dipole.h5` |
| `parallel_transport` | $U^\dagger (v^{\rm DFT} + D_k\Delta H) U$, with $D_k$ from finite links | `parallel_transport.h5` with links |
| `interband_commutator` | $U^\dagger (v^{\rm DFT} + [\Delta H, \mathcal W]) U$, no links; insulators only | the velocity stage of `parallel_transport.h5`, stamped `vnl_included = 1` |

**Default.** An unnamed `sc_head_update` on a shared-pole SC deck (scalar, or
`bispinor_gw = full_shared_pole`) with the head on takes the best velocity
the run directory supports: `parallel_transport` when `parallel_transport_file`
exists (the dipole step writes it by default), else `dft_velocity` from
`dipole.h5`, else `off`. A `full_shared_pole` deck with neither file takes
`dft_velocity`: its four-current bank reads `dipole.h5` on every head route
and refuses without it, so `off` would gain nothing. Other decks default to `off`. The choice is made once, at
parse time, and holds for the whole run, because a velocity treatment that
changed between maps would change $F$ under the mixer.

**Links on each map.** Complete links serve $D_k\Delta H$ on every map, and
each map logs their error bound; links that cannot serve the term (an
incomplete artifact, a two-point k axis, a hybridized window edge) set
$D_k\Delta H = 0$ for the whole run, with one line naming the reason
([link error](theory/qp-velocity.md#6-the-link-error-and-what-it-means)). A
link artifact whose steps differ from the current stencil refuses
(`GATE pt_link_stencil`); rerun the dipole step. Before the loop, a head
window whose top edge splits a degenerate multiplet refuses
(`sc_iteration._refuse_degenerate_window_edge`): an edge inside a multiplet
makes the head depend on the arbitrary basis of that multiplet.

**Head block.** Each map writes one block to `gwjax.out`: each term's
contribution (p, $V_{\rm NL}$, Σ) to $\omega_p^2$ on a metal (eV²) or its
share of the static head $S_{aa}(0)$ on an insulator (%; $S_{ab}(\omega)$ is
the $q \to 0$ head tensor of the response, $a, b$ Cartesian), the link
bound and
the band gap (0 on a metal). Metal routes are in §7.

### Interband-commutator head {#interband-commutator-head}

`interband_commutator` replaces $D_k\Delta H$ by $[\Delta H, \mathcal W]$
with the cross-gap $\mathcal W = i r^{VC}$, so it needs no links and runs on
any grid (`qsgw_head.interband_commutator_velocity`; definition, accuracy and
the collapsed-axis form in
[the velocity operator §7](theory/qp-velocity.md#7-the-heads-that-use-the-velocity)).
Each map prints the cross-gap mixing $\theta = \max_k \|U_{VC}\|_F$, to which
its error is first order; no threshold on θ refuses. A metal refuses
(`GATE sc_head_interband_commutator_insulator_only`), and so do a cross-gap
pair within $10^{-6}$ Ry (`GATE sc_head_interband_commutator_gap`) and a
velocity artifact not stamped `vnl_included = 1`
(`GATE sc_head_interband_commutator_velocity_operator`).

### QSGW dipoles {#qsgw-dipoles}

Every velocity head the SC driver forms writes the accepted final map's
$U^\dagger v\,U$, with the QP energies of the same states, to
`dipole_qsgw.h5` beside the deck (`qsgw_head.write_qsgw_dipole`, the
`dipole.h5` layout, `basis = "qp"`). $v$ is the head's own velocity from
`qsgw_head.qp_velocity`, and the file's `velocity` attribute names its Σ term.
Its states are the ones the final `WFN_qp.h5` holds. (A `full_shared_pole`
deck on `dft_velocity` writes none: its four-current bank reads `dipole.h5`
itself, §7.) The absorption consumers
read it through `load_dipole_h5` and form $d_{cv} = v_{cv}/(E_c - E_v)$.

The file holds velocities between QP states, so it pairs only with a QP WFN: a
BSE reads it only on a restart built from `WFN_qp.h5` and refuses it beside a
DFT WFN (`GATE dipole_basis`, [BSE inputs](architecture/bse.md)). When the
run also writes `WFN_qp.h5`, the file carries the `dipole.h5` provenance
stamps bound to that WFN (`qsgw_head.stamp_qsgw_dipole_provenance`): the
`WFN_qp.h5` fingerprint and the deck's window, $V_{\rm NL}$ mode and sign,
representation and DFT+U stamps. A GW run on `WFN_qp.h5` with the same deck,
given the file as its `dipole.h5`, builds its head on the SC velocity; any
other WFN refuses it by fingerprint.

## 4 The Σ grid across maps {#sigma-grid-and-quadrature}

$\Sigma_c(\omega)$ is sampled on a uniform grid of step `sigma_omega_step_ev`,
measured from the $E_F$ of the Σ frame: `efermi.resolve_sigma_efermi_ry` for
MPA, and the current spectrum's VBM or midgap for GN/HL-PPM
(`ppm_sigma.ppm_fermi_frame`). Coverage is judged in that same frame
(`efermi.sigma_frame_mu_ev`). Each sampled frequency costs quadrature work,
and a changed grid recompiles every Σ executable, so the loop plans the grid
once and holds it (`gw/qp_support.py`).

**Requested states.** The grid exists to converge the requested set R
(`qp_support.requested_states`): the QP-matrix states, minus frozen-core and
semicore states, that the W model treats as active
($\max_k E^{\rm DFT}_{nk} \ge E_F - 15$ eV,
`shared_pole_recipe.active_band_mask`, evaluated once on the DFT ladder) and
that had a quasiparticle at the previous map, $Z \in (0, 1]$. A state with
$Z \notin (0, 1]$ sits within about η of a pole cluster of its $\Sigma_{nn}$
and has no quasiparticle. Its energy never moves the grid, and off the grid
it reads the out-of-grid rule below and is named in an
`SC window no-quasiparticle` line. This keeps runaway states (Na 8³ band 63
at $Z \approx -382$, Fe 4³ k 4 band 26 at $Z = 2.82$) from stretching the
grid.

**Plan.** At map 0, and in the one-shot, the grid is the deck's request D
(`sigma_omega_min_ev` / `sigma_omega_max_ev` or the patch list; an unset edge
gives the sample next to $E_F$) joined with
$[\min_R E - P,\ \max_R E + P]$, where $P$ = 2 eV
(`qp_support.SUPPORT_PAD_EV`) and E is the DFT energy. Because the one-shot
uses the same rule, SC map 0 has the one-shot's grid, rules and off-grid set.

**Hold.** Every later map keeps the grid while each requested state's read
support $[E - 0.5, E + 0.5]$ eV lies inside it. E is the map's input energy,
and 0.5 eV is the half-width of the $Z$ stencil
(`eqp_bgw.Z_FINITE_DIFFERENCE_EV`). When a state is about to leave, only the
crossed edge grows, to $E \pm P$, which leaves 1.5 eV of motion before it can
trigger again; one `SC window extension` line names the band, k, $E - \mu$,
the new edge and the run's extension count. Old samples never move, and an
interior hole of a patched grid refuses. The grid never leaves D joined with
the union over maps of $[\min_R E - P,\ \max_R E + P]$
(`GATE sigma_support_envelope`). Input energies are DFT at map 0 and the
carried QP eigenvalues afterwards; eqp0, eqp1 and Z never set the grid. A
grid that reaches far above $E_F$ means the deck requested states there (Na
8³ with `ncond` = 81 requests every band, up to +96 eV). The grid is not
re-planned at map 1, because a new grid would recompile every Σ executable
while the held rules already cover the map-0 grid.

**Out-of-grid rule** (`sigma_out_of_grid`, `qsgw_utils.sigma_eval_omega`).
One classification, `qsgw_utils.omega_coverage`, decides which energies are
on the grid; the Σ build, the grid growth and the tail mask all read it.

| policy | an off-grid $\Sigma(E)$ reads | error past the edge, median / p90 (eV) |
|---|---|---|
| `cover` (default) | the grid grows over every requested quasiparticle, so none is off grid; W-inactive, frozen-core and no-quasiparticle states read $\Sigma(\omega = 0)$ | 0 for requested states |
| `clamp` | $\Sigma$ at the nearest edge | Fe 0.3–0.7 / 0.8–4.7; CrI3 0.05–0.12 / 0.5–2.2; MoS2 0.2–0.3 / 0.6–0.8 |
| `static` | $\Sigma(\omega = 0)$ | Fe 1.6–4.4; CrI3 0.4–0.5; MoS2 0.6–0.8 (median) |

The errors compare against the sampled Σ 2–6 eV beyond a truncated edge.
`clamp` is continuous at the edge, but an edge on a GN-PPM pole gives errors
of order $10^3$ eV; `static` has two fixed points near an edge (§5). States
deeper than the active depth keep $\Sigma(0)$ because the W model carries no
plasma charge for them: covering Fe 4³'s 3s/3p stretched the grid to −98 eV,
ran 5× slower and moved those states 16 eV. The active depth is evaluated
once, so a state never switches between $\Sigma(E)$ and $\Sigma(0)$. No tail
$C_n/(\omega - \bar\omega_n)$ matched at the edge is offered: Σ at a grid edge
is far from its $1/\omega$ asymptote (Fe: −5 to −7 eV at +28 eV), so the
matched pole falls inside the extrapolated range for 10–92 % of the states.

**Coarse windows** below the near grid follow the same plan-and-hold rule
(§2).

**Held Σ rules.** The quadrature rules that integrate each product window are
certified at map 0 over the map-0 grid on padded boxes, reused while each
window's current box stays inside its certificate, and rebuilt one window at
a time on an escape, with one `SC fixed quadrature recompute:` line per
rebuilt window. The padding and the escape rules are owned by
[Σ quadrature §9](theory/sigma-quadrature-problem.md#9-self-consistent-maps).

**Held W shapes.** From map 1 the shared-pole model keeps one pole-column
extent per sector (`file_io.shared_pole_store._k_extent`: map 1's largest
pole count plus 3 %, grown with the same headroom only when a live count
exceeds it), and the four-current CT round keeps each of CC and TT at its
largest retained-span width (`shared_pole_sectors.cross_span_widths`, never
shrunk; the extra columns are exact zeros). Each growth is logged. A drifting
pole count or rank therefore does not change a compiled shape.

## 5 Mixing, state identity and convergence

### Anderson mixing

`mixing.acceleration.anderson_nojit` implements Anderson type II (Pulay) with
one map evaluation per iteration. The history holds the newest $m + 1$
evaluated pairs $(x_i, f_i = F(x_i) - x_i)$, $m$ = `sc_history_depth`
(default 20), and the next and only evaluation is at

$$
x_{n+1} = \sum_i \alpha_i\,(x_i + f_i), \qquad
\alpha = \arg\min_{\sum_i \alpha_i = 1} \Big\| \sum_i \alpha_i f_i \Big\|_P .
$$

α is real, because Hermitian matrices form a real vector space. The metric
$P$ weights each k row by the square root of its star size, so the squared
norm is the sum over the full uniform k grid whatever wedge the loop runs on.
Two safeguards cost no evaluation and carry no tunable constant:

- **Conditioning filter.** The oldest differences are dropped until the
  unit-column Gram has condition number at most $10^{12}$.
- **Nonmonotone fallback.** An evaluation worse than every residual in the
  window means the multisecant model failed there; the next point is the
  two-point secant between the best pair and the rejected one. It never fires
  twice in a row, and the rejected pair stays in the history.

A discrete change of the map, a Σ rule rebuild or a grid extension, is logged
(`SC map event`) and does not restart the history: early maps grow the grid
on most calls, and restarting there reduces the method to plain steps, which
diverge on an expansive map.

Plain iteration is refused (`sc_accelerator` accepts only `anderson`,
`GATE sc_accelerator_anderson_only`). On dense band manifolds the QSGW
Jacobian has eigenvalues of about −3 or below along cycle directions on
GN-PPM decks with many bands near the gap, so a plain fixed point 2-cycles
and damping only shrinks the cycle; undamped linear mixing also amplifies
the input's time-reversal-reality error 6–8× per map (Si 4×4×4, scalar
shared pole).

**Residency.** The history holds $2(m + 1)$ copies of the carry, stacked on a
leading axis that is never sharded, with bra bands on mesh axis X, ket bands
on Y and k replicated (`qsgw_density.band_rotation_spec`). One copy is
$16\,n_k N_b^2$ bytes: 21 MB on CrI3 8×8 (144 bands), 9.2 GB at $n_k = 144$,
$N_b = 2000$, where $m = 20$ takes 387 GB in total, 3.9 GB per rank at
$P = 100$. Each iteration issues one $(m+1)\times(m+1)$ Gram reduction. The
map needs a replicated carry, so each call gathers one $(n_k, N_b, N_b)$
matrix.

### State identity

The carry is in the DFT basis, so every per-state table (the tail weights,
the coarse pin, the convergence test) is indexed by $(k,$ DFT label$)$, not by
sorted position; a level crossing must not relabel a state.
`sc_state_identity.assign_qp_identity` assigns, at each k, QP columns to DFT
labels by maximizing the summed projector overlap $\sum |\langle \psi^{\rm
DFT}_m | \psi^{\rm QP}_n \rangle|^2$ (a linear assignment). Levels within
`sc_exact_degeneracy_tol_ev` form multiplets; a multiplet is one capacity
block scored by its summed overlap, so its internal gauge does not enter, and
its members report the block-mean energy. A label set that cuts a DFT
multiplet refuses. The convergence readout matches each map against the
labels of the map-0 output (`sc_iteration._sc_identity_for_call`).

### Convergence and stop rules

The mixer has no stopping authority; the driver decides on every evaluated
input:

| verdict | rule |
|---|---|
| **CONVERGED** | $\max \lvert E_{\rm out} - E_{\rm in}\rvert$ over the QP-matrix labels is below `sc_tol_ev` (default 1e-4 eV). $E_{\rm in}$ are the eigenvalues of the input $H$, $E_{\rm out}$ those of $F(H)$, matched by label. The loop returns that input with its own Σ, W and head. |
| **STALLED at floor, not converged** | the label-free residual $r_n = \max_k \lVert P\,(F(H_n) - H_n)\,P\rVert_2$ (logged as `SC matrix residual`) has not improved by 10 % over the last 12 maps. |
| budget | `sc_max_iter = N` (default 30): N ≥ 2 runs map 0 plus N accelerated maps; N = 1 is a special case that runs map 0 only, as a labelled one-map diagnostic. |

The test compares a map's output with its own input, not successive
iterates: a mixed iterate can barely move while $F$ still has no fixed point.
It is a maximum, not an RMS, because "every state moved less than the
tolerance" is a statement about the worst state. $r_n$ bounds every
sorted-eigenvalue residual (Weyl) and also sees eigenvector error, so
relabelling a hybridized pair cannot move it; the 10 % and the 12 maps are
fixed, not deck keys.

A stalled or budget-exhausted run refuses with
`GATE sc_fixed_point_not_converged` and writes no terminal QP result. Before
the refusal the record prints one block for the last map: the median
$\lvert\Delta E\rvert$ over all states, then the 1-based bands (no k) with a
state moving more than 20 meV, and with a state moving at least `sc_tol_ev`.
The per-map files remain (§8).

**Map gain.** From the second map on, the log prints
`SC map gain: max |dSigma_on-shell| / max |dE_in|` over adjacent maps, and the
eqp comments carry the same figure. A value above 1 means the sampled map is
not locally contracting. It is a diagnostic and controls nothing. On metals
it can stay above 1 at Fermi-crossing states, where exchange responds to an
occupation flip within the smearing width; that is the physics of the map,
not a failure of the mixer.

### Where the map is not smooth

**The grid-edge switch** (`sigma_out_of_grid = static` only). $\Sigma(0)$
makes $F$ discontinuous at each grid edge $\omega_e$ by
$\Delta_n = \mathrm{Re}\,\Sigma_{c,nn}(0) - \mathrm{Re}\,\Sigma_{c,nn}(\omega_e)$.
In the diagonal model $E = A + \Sigma(E)$, a jump that points outward
($\Delta_n > 0$ at the top edge, $< 0$ at the bottom) gives every state
within $|\Delta_n|$ of the edge a self-consistent partner on the other side:
two fixed points, and the path decides which one the loop reaches. An inward
jump cannot do this. `qsgw_utils.sigma_grid_edge_ambiguity` flags such states
every map, with no threshold because the band is the jump itself, and the
verdict appends `fixed point NOT UNIQUE`; it is not a refusal. On Fe 4³
bispinor the H-point states converge at $E - \mu$ = +9.8 eV (inside) under
one trajectory and +11.7 eV (outside) under another, around a +10 eV edge
with Δ = 1.87 eV. Under `cover` both trajectories reach one branch within
2 meV, and `clamp` has no jump.

**The elementwise MPA pole refit** (`compute_mode = mpa`,
`sigma_w_model = mpa`). The per-element pole fit from 16 samples is not
identifiable: a $10^{-7}$ change in the samples selects a different, equally
good pole set, and $\Sigma_c(\omega)$ jumps by 10–20 meV somewhere on the real
axis. No damping schedule converges this jump; resolution makes it harmless.
On Si at 24 bands and 192 centroids the map gain is 14–18 and the loop
plateaus; at 80 bands and 504 centroids it is 0.3 and the loop contracts. The
shared-pole W has no such refit (§6).

## 6 Shared-pole W across maps {#shared-pole-w-with-retained-quadrature}

With `sigma_w_model = shared_pole` every map rebuilds the whole model from its
rotated wavefunctions, energies and occupations: response samples, moments,
directions, poles and factors (`shared_pole_screening.screen_shared_poles`).
An SC map's model belongs to that map; `restart = true` restores only the
invariant ISDF basis and V(q) and never skips a map's response or W. The
model is fitted to samples of $W_c$ at fixed **supports**: line sites
$z = \omega + ih$ on a damped line at height $h$ up to a top line endpoint,
and an imaginary ladder $z = iu$ on $[u_{\min}, u_{\max}]$
([shared-pole theory §5.1](theory/shared-pole-w-model.md#51-supports-and-directions)).
What persists across maps is this sampling geometry and the quadrature
rules, held so that each map is the same function of its input:

- **Σ rules**: held as in §4. For a four-current sector model the certificate
  also covers the pole treatment ceiling, twice the χ transition span
  (`shared_pole_recipe._sector_treatment_ceiling`). The ceiling is held while
  the current span stays inside it and re-planned at twice the current span
  otherwise.
- **Response rules** (`response_bank.response_quadrature`): the shared
  complex-time rule that evaluates χ₀ at every support from the
  occupation-weighted transitions. It is planned on the transition-energy
  interval padded by 4 eV (`response_bank.RESPONSE_HOLD_PAD_EV`) and on the
  envelope that bounds a metal's occupation products,
  $|f_n(1 - f_m)| \le A\,\min(1, e^{\beta(E_m - E_n)})$ with amplitude $A$
  and decay rate $\beta$ (`response_bank.response_occupation_envelope`). It
  is reused while the current interval, decay rate, amplitude, metallicity
  and supports stay within the held plan. Otherwise they are rebuilt,
  with one line `Response rule rebuilt at a held map; failed reuse test: …`
  naming each failed field.
- **Support enclosure** (`shared_pole_recipe._support_envelope`). Map 0 uses
  its own supports. From map 1 the loop keeps the running maximum of the top
  line endpoint and of $u_{\max}$ and the running minimum of $u_{\min}$, so
  the smaller DFT gap of map 0, which lowers $u_{\min}$, is never locked in. The enclosure fixes sampling geometry
  only; it is not an interpolation-error certificate.
- **Line sites.** The line sites, placed from the band structure by the
  support rule of the theory page, are held while the support rule
  would place every site within
  max(3 meV, 0.1 × the previous map's max|dE|) of the held one
  (`shared_pole_recipe.LINE_SITE_HOLD_EV`), and re-placed from the current
  bands otherwise (`SC W line sites re-planned`). Sites frozen for the whole
  run move the end point (Fe 4³ bispinor from DFT: 99 meV in the
  $E_F \pm 10$ eV states), and a hold far below the iterate's own error
  re-plans on most maps, so each map is a different function and Anderson
  mixes residuals of different maps (Fe 4³ scalar `parallel_transport`: 10
  re-plans in 15 maps, stalled).

`head_correction = full` folds the head wings through each map's own W
(`shared_pole_head.build_shared_pole_head`); `sc_head_update = off` keeps the
DFT direct response. The ordered-store and four-component head refusals are
in [shared-pole model §8](architecture/shared_pole_model.md).

**Symmetry of the model.** The model is
$W_c(q, s) = \Pi_{G_q}\big[\sum_j b_j b_j^\dagger/(s - \Lambda_j)\big]$, where
$\Pi_{G_q}$ averages over the authenticated magnetic little group of $q$
(recipe `operator_realization = little-group-reynolds-v1`; operations from
`symmetry_maps.project_little_group_operator` through `gw/qgrid_symmetry.py`).
Each transformed residue is a unitary or conjugate-unitary congruence of a
positive residue, so the average keeps residues positive and poles real. At
complex $s$ an antiunitary operation acts as the same-time transpose of the
residue endpoints, not as a conjugation of the whole value. The projector
streams one operation at a time into a fixed accumulator, so no factor gains
a symmetry axis.

## 7 Metals {#7-metals}

- Use `compute_mode = mpa` with `sigma_w_model = shared_pole`. GN-PPM refuses
  metals (`GATE gn_ppm_refuses_metals`): its Σ splits bands by a 0/1 step at
  a derived Fermi level.
- Occupations are Fermi–Dirac only (`GATE metal_occupations_fermi_dirac`), and
  `occ_smearing_width_ry` is $k_BT$. At startup the occupations re-solved
  from the WFN's own energies must reproduce the WFN's stored occupations, so
  the deck's smearing matches the DFT run. Each map then solves μ at fixed
  electron count from its input spectrum (`sc_iteration._solve_occupation_state`);
  χ₀, the head and Σ all read that one state.
- A band is in a Σ or χ branch iff its weight ($f$ or $1 - f$) is at least
  $10^{-5}$ (`efermi.OCCUPATION_WEIGHT_FLOOR`; $|E - \mu| \le 11.5\,k_BT$ for
  Fermi–Dirac), in the one-shot and every map alike. A state that crosses the
  cut between maps switches one term by about $10^{-5}$ of its size, about
  0.01 meV.
- The scissored tail has exact-zero occupations: it enters G and the response
  at its shifted energies, and the QP matrix alone sets μ. A tail state that
  enters the fractional manifold refuses (`GATE sc_empty_tail`); enlarge the
  QP matrix.
- The local gain of the diagonal map $E \leftarrow A + \Sigma(E)$ is
  $s = \partial_\omega \mathrm{Re}\,\Sigma = 1 - 1/Z$. States far above μ,
  where $\mathrm{Re}\,\Sigma$ rises on shell, have $Z > 1$ and $s$ near 1, and
  converge slowly. End the QP matrix below them (`ncond`); convergence is
  judged on every QP-matrix state.

### Metals: direct Drude head {#metals-direct-drude-head}

On a metal the head adds the intraband (Drude) term, a Fermi-surface integral
of the velocity computed with tetrahedron weights, and a Thomas–Fermi static
slot ([the metallic q → 0 head](theory/metal-q0-head.md)). The velocity comes
from `qsgw_head.qp_velocity` (§3); the one-shot and the fixed DFT head
(`qsgw_head.build_dft_head_response`) read it at $\Delta H = 0$, $U = I$.

| `sc_head_update` on a metal | head on each map | admitted on |
|---|---|---|
| `off` | the fixed DFT response on the DFT fixed-N Fermi–Dirac state, with its Drude term and Thomas–Fermi slot (`sc_iteration._fixed_dft_head_occupation_state`) | every metal deck |
| `dft_velocity` | the `dipole.h5` velocity rotated into each map's QP basis, the current μ and tetrahedron weights; the dynamic Drude tensor at $\omega \ne 0$, Thomas–Fermi at $\omega = 0$; `full` folds it through intraband wings and the static Γ body | shared-pole decks with `head_correction = no_local_fields` (scalar, `bare_transverse`, `full_shared_pole`), or `full` on a scalar deck |
| `parallel_transport` | the same head on $U^\dagger(v^{\rm DFT} + D_k\Delta H)U$, so the Drude term sees the QP Fermi velocity | shared-pole scalar decks with `no_local_fields` or `full`, and `full_shared_pole` |
| `interband_commutator` | refused (`GATE sc_head_interband_commutator_insulator_only`) | — |

Any other metal combination refuses (`GATE metal_sc_head_update_disabled`,
`gw_config.uses_metal_direct_drude_head`): a folded bispinor metal head has no
derived completion, and the `bare_transverse` head has no link consumer.
`full` on an ordered (time-reversal-broken) store refuses on every route
(`GATE shared_pole_head_ordered`), and so does `occ_broadening > 0` next to a
metal width (`GATE metal_sc_head_update_disabled`).

On `bispinor_gw = full_shared_pole` the four-current bank builds its own
direct Γ head (`response_bank.compute_photon_bank`) from the same velocity:
on `parallel_transport` it takes this map's `qp_velocity` through
`photon_head_state`; on `dft_velocity` it rotates the authenticated
`dipole.h5` velocity, the same object at $\Delta H = 0$.

## 8 Seeding, restart and outputs {#8-seeding-restart-and-outputs}

**There is no checkpoint of the loop.** The mixer history is never written,
so an interrupted run cannot continue where it stopped. A rerun in the same
directory recomputes from map 0. It reuses a map's shared-pole scratch only
when that scratch authenticates for the same map identity (WFN fingerprint,
energies, occupations, centroids, recipe): a committed scalar `model.h5`, a
four-current `sectors.json` with every model file it names, or a complete
response bank, which resumes the constructor. Any other partial
directory is removed and rebuilt, with a WARNING line naming the files
(`shared_pole_screening.screen_shared_poles`). Only the newest map's scratch
generation is kept (`shared_pole_screening.retain_iteration_scratch`).

**Warm seed.** After every map that does not converge, the loop writes
`sc_seed/qp_wfn_rotations.h5` (`sc_iteration._write_sc_seed`): that map's
output eigensystem, the scissored tail energies and the fixed-N occupations.
It holds no wavefunctions and no mixer history.

**Seeding a new run.** `sc_initial_qp_rotations_file` imports an
authenticated eigensystem as $H = U\,\mathrm{diag}(E)\,U^\dagger$ in the
original DFT basis (`sc_iteration.make_initial_state_from_qp_rotations`). The
source WFN fingerprint, k table, band range, finite $E$/$U$ and unitarity
are checked; keep the original WFN and reference operators. At import $H$ is
averaged over each kept k's little group,
$H \leftarrow |G_k|^{-1}\sum_L A_L(H)$ with the band representations of the
operations from the WFN, so a seed from another run or code version cannot
break this run's symmetry. The `SC initial Hamiltonian` line prints the
largest change, and a seed whose Σ window cuts a multiplet refuses
(`GATE little_group_band_representation`). A seed file that records a
partial band partition or an active-window scissor refuses, because the loop
keeps every QP-matrix state's full Σ and has nothing to apply such a law to.
Occupations, the tail fit, the quadrature plan and the mixer history start
fresh, so a seeded run is a new run, not a continuation. A charge SC seeds a
bispinor SC on the same WFN
([production QSGW](how-to/production-qsgw.md#the-route)).

**`restart = true`** reads the ISDF basis and V(q) from a finished run's
restart store instead of refitting them; every map still rebuilds χ₀, W and
Σ.

**Terminal files**, written only by a converged run (or a one-map
diagnostic):

- `eqp0.dat` and `eqp1.dat` hold the SC eigenvalues: the accepted map's output
  $\mathrm{eigvalsh}\,F(H_{\rm in})$ with its tail scissor and semicore pin,
  sorted per k, on the WFN file wedge (`sc_iteration._write_sc_result_eqp`).
  Both equal the body of that map's `eqp0_iterNNNN.dat` bit for bit, and the
  header says so. No Z linearization enters: at the fixed point the map output
  solves the QP equation. The gap report reads the same array.
- `sigma_diag.dat` holds the diagonal of the final Σ on the fixed DFT states,
  a diagnostic, not the SC spectrum. `sigma_mnk.h5` carries no eqp receipt,
  and `python -m gw.eqp_bgw` refuses an SC file
  (`GATE eqp_bgw_self_consistent_run`).
- `qp_wfn_rotations.h5` (always) and `WFN_qp.h5` (when `write_wfn_h5`, the
  default) hold the accepted map's input QP states, the basis its Σ, W and
  head were built in; they agree with the eqp files to `sc_tol_ev` on the
  QP matrix. Each holds the complete energy ladder (QP matrix plus tail), the
  occupation table, μ, the smearing and the table hash. `WFN_qp.h5` is
  written collectively: each rank reads, rotates and writes its own G slab of
  every k through `file_io.slab_io`. Each file is written to a private
  sibling, validated through its format owner and made visible by
  `os.replace`; the run-completion manifest requires both names.
- `dipole_qsgw.h5` on a velocity head (§3).
- A converged run on a route that persists W0 writes it once, from the
  accepted map, so BSE can run on the SC result.

**WFNs that store both k and −k.** A WFN may store two k of one orbit (MoS2
3×3 stores all 9). `SymMaps` then builds the full-BZ row of such a stored k
from another stored row, so the QP $U$ there is in the gauge of the unfolded
parent, not of the stored orbitals. `file_io.qp_wfn.write_qp_wfn_h5` rotates
each such row by $D\cdot U$,
$D = \langle\psi_{\rm stored}|\psi_{\rm unfolded}\rangle$ on $[b_0, b_3)$,
and a non-unitary $D$ (the window cuts a multiplet) refuses
(`GATE qp_wfn_orphan_gauge`). `postprocess.rotate_wfn_to_qp` rewrites a
`WFN_qp.h5` from its `qp_wfn_rotations.h5` through the same writer; it
reapplies the stored ladder and table and neither rebuilds the tail nor
re-solves occupations.

**Per-map files are diagnostics, not restart state.** `eqp0_iterNNNN.dat`
holds each map's output energies, with the convergence criterion, the map
gain, the tail law and the label table in its comments; there is no per-map
eqp1. With `sc_dump_dir`, `rotation_iterNNNN.npy` is each map's input $U$ and
`e_history_kn_ev.npy` the output-energy history. With
`sigma_lorentz_debug_output = true`, four-current maps write
`sigma_lorentz_iterNNNN.h5`: the CC, CT+TC and TT sectors in the map's input
QP basis. A new run clears the previous run's managed snapshots.

## 9 Implementation

| concern | owner |
|---|---|
| driver entry, final rotation to the DFT basis, terminal writers | `gw.sc_iteration.run_sc_driver` |
| one map $F$ | `gw.sc_iteration.gw_iteration_map` |
| Anderson loop, stop rules | `gw.sc_iteration._run_anderson`, `mixing.acceleration.anderson_nojit` |
| state identity | `gw.sc_state_identity.assign_qp_identity`, `gw.sc_iteration._sc_identity_for_call` |
| convergence verdict | `gw.sc_iteration.protected_band_convergence` |
| QSGW operator | `gw.qsgw_utils.build_qsgw_sigma_xc`, `sigma_eval_omega`, `omega_coverage` |
| semicore class and windows | `gw.band_partition.semicore_floor`, `gw.qp_support` |
| ω grid plan and hold | `gw.qp_support`, `gw.sc_iteration._sc_sampled_support` |
| held Σ rules | `gw.sigma_box_plan._fit_fixed_sc_rules`, `_sc_padded_box_spec` |
| tail scissor | `gw.sc_iteration._fit_sum_band_tail`, `gw.scissor` |
| velocity and head | `gw.qsgw_head.qp_velocity`, `gw.sc_iteration.load_head_velocity_source` |
| seeds and QP files | `gw.sc_iteration.make_initial_state_from_qp_rotations`, `dump_qp_wfn_artifacts`, `file_io.qp_wfn` |
