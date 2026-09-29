# Self-consistent GW (QSGW)

`qp_solver = self_consistent` finds the quasiparticle self-consistent GW fixed
point by accelerating the GW map with one-evaluation Anderson mixing. This page
covers the map, how each band is treated, the acceleration and its stop rules,
and the Σ grid and quadrature across maps. Other pages own the production
recipe and its error budget ([production QSGW](how-to/production-qsgw.md)),
the deck keys and defaults ([input reference](input_reference.md)),
metallic occupations and heads
([metallic MPA screening §6–7](theory/metallic-mpa-screening.md)), the Σ(ω)
rules ([the Σ(ω) quadrature problem](theory/sigma-quadrature-problem.md)) and
the Hartree rebuild ([direct Hartree field](theory/hartree.md)).

## 1 The map

The carry is the QP Hamiltonian in the fixed DFT basis, $H_k$ of shape
`(nk, nb, nb)`. It spans the Σ band window `[b0, b3)`, the `nval + ncond`
bands, on the loop's k-set. One map $F: H \to H'$ does five things:

1. diagonalize $H_k = U_k\,\mathrm{diag}(E_k)\,U_k^\dagger$;
2. rotate the original DFT orbitals by $U$. There is no cumulative product,
   so there is no drift;
3. rebuild $V_H$ from the rotated occupied orbitals (`density_self_consistent`,
   which is true whenever the key is omitted);
4. build $\chi_0 \to W \to \Sigma_c(\omega)$ with the configured Σ scheme
   (`sigma_dispatch.compute_sigma_xc`);
5. form the static Hermitian QSGW operator (`qsgw_utils.build_qsgw_sigma_xc`)

$$
\Sigma^{\rm QSGW}_{ij}(k) = \tfrac12\,\mathrm{herm}\!\left[\Sigma_{ij}(k,E_{ik}) + \Sigma_{ij}(k,E_{jk})\right],
\qquad \Sigma = \Sigma_x + \Sigma_c ,
$$

   rotate it to the DFT basis, and set
   $H' = T + V_{\rm ion} + V_H + \Sigma^{\rm QSGW}$ under the band treatment
   of §2.

$F$ is a pure function of $H$: re-evaluating the same input returns a
bitwise-identical output (CLAIMS 2678). Every evaluated pair
$(H, F(H) - H)$ is therefore valid secant data. Map 0 takes $U = I$ exactly
instead of calling `eigh` on $\mathrm{diag}(E_{\rm DFT})$, so SC map 0 equals the
one-shot G0W0 bit for bit. Each
map costs one full $\chi_0 \to W \to \Sigma$ evaluation. Σ rule planning is
paid on map 0 (the one-shot plan) and map 1 (the held plan), and after that
only for a window a state crosses (§4).

The loop is driven by eqp0, which is Σ at the current energies. No Z-factor
enters the iteration. Each map also writes the BerkeleyGW-shaped linearization
$\mathrm{eqp1} = E_{\rm in} + Z\,(\mathrm{eqp0} - E_{\rm in})$, which equals
eqp0 at a fixed point.

With `sc_on_ibz = true` (the default), $H$, $E$, $U$, every retained k-indexed
`SigmaResult` table and the density-SC Hartree components carry the star
wedge. The full BZ exists only inside a map, while the k-grid FFT builds Σ.
One seam at the map boundary selects the retained result together with its
defining $U$.

### Interband-commutator head {#interband-commutator-head}

A head update (`sc_head_update`) rebuilds the $q \to 0$ head each map from the
QP velocity $v^{\rm QP} = U^\dagger (v + D\Delta H)\,U$. Here $v$ is the DFT
velocity $p + i[V_{\rm NL}, r]$ (plus SOC), $\Delta H = H - \mathrm{diag}(E^{\rm DFT})$,
and $D$ is the covariant $k$ derivative. `parallel_transport` forms $D\Delta H$
from finite links and needs a derivative rule on every $k$ axis.
`interband_commutator` forms it without links, on any grid:

$$
D\Delta H \approx [\Delta H, W], \qquad
W_{vc} = \frac{v_{vc}}{E_v - E_c}, \quad W_{cv} = W_{vc}^*,
$$

with $v < n_{\rm occ} \le c$, and $W = 0$ inside each occupation class.
$W = i\,r^{VC}$, so $[H^{\rm DFT}, W] = v$ on the valence–conduction blocks.
Split the covariant derivative by class, $D\Delta H = -i[A^{VC}, \Delta H] + D^{\rm class}\Delta H$.
The valence–conduction block of $D^{\rm class}\Delta H$ holds only the cross-gap
block $\Delta H_{VC}$. So the head is exact for any $\Delta H$ that does not mix
valence and conduction; a band-diagonal $\Delta H$ gives
$v_{vc}(E^{\rm QP}_v - E^{\rm QP}_c)/(E_v - E_c)$. Its error is first order in the
cross-gap mixing, and each map prints $\max_k \lVert U_{VC} \rVert_F$.
$\Delta H$ is the active block plus a diagonal tail inside the head manifold, so
no sum over states is truncated.

On a collapsed (one-point) $k$ axis the cell is not periodic, and the
connection is the stored position operator $Z_a$ of the velocity artifact
(`sc_head_update = parallel_transport` uses the same operator there). There
the reduced component of $W$ is $i Z_a$, full and exact, and the class rule
applies to the periodic axes only: $W_{\rm cart} = B^{-1}\,[B\,W^{VC}$ with row
$a$ replaced by $i Z_a]$.

**Accuracy.** On MoS2 bispinor at $\theta_{\max} \approx 0.055$ the class rule
alone put $S_{zz}$ 1.3 % from the exact position-operator head, so the
periodic-axis error is about $\theta/4$ in $S_{aa}$. A $5\times10^{-3}$ head
tolerance is out of reach at $\theta \approx 0.05$ without within-class $k$
information (links). The per-map line prints $\theta$; there is no refusal
threshold. Measurements: sandbox claim 2815.

Every same-class pair is excluded, degenerate or not. Inside a class, $W$ has
no gap in its denominator. Near-degenerate pairs make it arbitrarily large, and
the $\partial_k \Delta H$ that would cancel it has no stencil-free form.
Excluding only exact multiplets (BerkeleyGW's $10^{-6}$ Ry) left a Si SOC head
8.8 times the link head. The class rule has no tolerance, and every
denominator is at least the direct gap.

A metal refuses (`GATE sc_head_interband_commutator_insulator_only`), and so does
a cross-gap pair within $10^{-6}$ Ry (`GATE sc_head_interband_commutator_gap`).
The velocity artifact must stamp `vnl_included = 1`
(`GATE sc_head_interband_commutator_velocity_operator`). The kernel is
`qsgw_head.interband_commutator_velocity`.

**QSGW dipoles.** On this head the driver writes the accepted final map's
$U^\dagger(v + [\Delta H, W])\,U$, with the QP energies of the same states, to
`dipole_qsgw.h5` beside the deck (`qsgw_head.write_qsgw_dipole`, the
`dipole.h5` layout). Its states are the ones the map's $W$ and the final
`WFN_qp.h5` are built from. The absorption consumers read it through
`load_dipole_h5` and form $d_{cv} = v_{cv}/(E_c - E_v)$, which is then
$(U^\dagger r^{VC} U)_{cv}$ with the collapsed-axis $Z_a$ in place of
$r^{VC}$: the QSGW term $-i[r^{VC},\Delta H]$ is included; the intraband
$D^{\rm class}\Delta H$ is not. With a band-diagonal $\Delta H$ the position
form equals the DFT one, and the whole QSGW change of $|d|^2$ is the mixing
$U$ (the velocity form scales by the QP-to-DFT transition-energy ratio). A
BSE on the self-consistent restart uses its DFT parent $\psi$ with `--eqp`
energies, so it takes these dipoles at the same diagonal approximation.

## 2 Band treatment

| bands | block of $H'$ |
|---|---|
| the QP matrix `[b0, b3)`, b3 = nelec + `ncond` as counted on main | full $\Sigma^{\rm QSGW}$, off-diagonals kept within the matrix |
| its coarse (semicore) states (below) | full rows, read at their own energy on held coarse windows at $\eta_{\rm semi}$ = 5 eV |
| the lowest `sc_frozen_core_bands` | held at the DFT block $\mathrm{diag}(E_{\rm DFT})$; they stay in the $\Sigma_x$ and $\chi_0$ sums |
| the scissored tail `[b3, number_bands)` | DFT orbitals with an energy-only rigid shift, refit every map, in G and $\chi_0$ only: the $Z$-weighted mean QP correction of the matrix's conduction states that read their own $\Sigma(E)$ (below) |

**b3 counts bands** (owner 2026-09-29: "b3 will count bands as on main yes,
and only bands between b0 and b3 will be rotated amongst each other"). b3 =
nelec + `ncond` (`number_bands_protected` resolves to the same count); only
[b0, b3) rotates, and the ζ fit is untouched. A `zeta_nband` below b3
refuses on every route (`GATE qp_matrix_zeta_left`): a state outside the
fit's left range would carry Σ on unfitted pairs. The classes below change
only where each QP-matrix state's $\Sigma_c(\omega)$ is read
(`band_partition.semicore_floor`, `qp_support`).

**The request and the coarse (semicore) class.** Two exclusive forms name the
QP request; giving both refuses (`GATE band_request_forms`).

- `number_bands_protected = N` (the documented form, owner 2026-09-29): every
  occupied band plus conduction bands up to N in total. The coarse class is
  every occupied band below a band gap of at least
  `band_partition.SEMICORE_GAP_EV` = 4 eV (all k) under the requested bands
  within μ ± 10 eV; none without such a gap.
- `nval` / `ncond`: the fine window, read at the deck η, runs down to the
  minimum energy of the lowest requested valence band (`nelec − nval` at each
  k, the count below μ on a metal; `sigma_omega_min_ev` only lowers it), and
  every occupied state below that energy is coarse. A deck whose `nval`
  covers every occupied band has none. The rule is energy-based, so a coarse
  state inside the fine grid's lower pad reads the fine grid.

A coarse state stays in the matrix and mixes fully, but its Σ is read at its
own energy on coarse windows of the grid below the near support. The
automatic windows are one per coarse manifold (levels separated by a global
gap wider than twice the 2 eV plan pad), at $\eta_{\rm semi}$ = 5 eV, sampled
at $\eta_{\rm semi}/2$ (`qp_support.SEMICORE_*`), planned at map 0 over the
coarse DFT energies and held; a coarse read that leaves them extends them, as
the near support is extended. `sigma_omega_patches_ev` triples `lo:hi:eta`
(eV about E_F) are user windows: the coarse states inside one read it at its
own η instead (`GATE sigma_coarse_window` refuses a malformed, overlapping or
sub-deck-η triple, or one without a coarse class). In the Σ plan the
crossing windows that own coarse samples serve them at the window's η and
max(`sigma_quadrature_eps`, `qp_support.SEMICORE_EPS` = 3e-3) (owner
2026-09-29; against 1e-4 it moves states within E_F ± 10 eV by ≤ 0.06 meV
at maps 0–1 and semicore QP by ≤ 4.3 meV at map 0, for 10–12 % fewer map
pairs on MoS2 and Fe); adjacent automatic windows of one η share a rule
window when that lowers the summed closed-form node count of the boxes the
runs are built on (`sigma_box_plan._coarse_runs`, decided at map 0 and
held); a user window is never grouped; sign-definite windows serve coarse
samples at the deck η. Broadening flattens $d\Sigma/d\omega$: at the deck η the
Fe 3s $Z$ leaves $(0, 1]$ from map 1 and the loop stalls; at 5 eV every coarse
$Z$ stays inside (one `SC semicore Z` receipt per map). The η_semi systematic
is reported apart from the 1 meV budget of the controllable errors: converged
E_F ± 1 eV std/max 3.6/15.9 meV (Fe 4³, against η_semi 1 eV) and 2.9/20.0 meV
(MoS2 3×3, against the deck η), growing about 1.2 and 0.6 meV std per eV
(claim 2960).

**Conduction states** (owner 2026-09-29: "i'd rather default be all cond bands
have the same broadening"): every protected conduction state is read on the
near grid at the deck η; there is no far-conduction class. On Fe 4³ prot
(`number_bands_protected = 26`) the matrix-top band 26 at k 4/11 then sits
among the scissored tail's levels (+20.6 to +21 eV) with a resonant Σ, and
the SC map is bistable; `number_bands_protected = 35`, which closes the
manifold, removes the tail there (claim 2960).
A QP matrix [b0, b3) that ends inside a band manifold (band b3's maximum over
k above band b3+1's minimum) is warned about in the log, which names the
`number_bands_protected` that closes the manifold; it is not refused. Only the
scalar MPA/shared-pole Σ reads coarse windows; a sector (bispinor) route
keeps the rule below.

**Pinned semicore** (`sc_semicore = dft`, owner 2026-09-29: the pseudopotentials
are fitted to DFT, so the semicore stays at its DFT energies while its mixing
with the protected states is kept). H is carried in the fixed DFT basis, and
the pin is on the projector $P_S$ onto the coarse labels' DFT orbitals:

$$P_S H P_S = P_S H^{\rm DFT} P_S = {\rm diag}(E^{\rm DFT}_s),\qquad
H_{ps} = \tfrac12\big[\Sigma_{ps}(E^{\rm QP}_p) + \Sigma_{ps}(E^{\rm DFT}_s)\big]^{\rm h} - V^{\rm xc}_{ps}.$$

The semicore–semicore block carries no $\Sigma - V^{\rm xc}$; every other
element is the QSGW one. The Hermitian average is formed in the current QP
eigenbasis, where a QP column reads at the $E^{\rm DFT}$ of the coarse label
`sc_state_identity.assign_qp_identity` gives it (largest $|U|^2$ overlap; a DFT
multiplet is one capacity block with one energy, so its internal gauge does
not enter), and the rotation back to the DFT basis carries it. The labels are
the map-0 coarse class, fixed for the run; the coarse windows are planned on
their DFT energies and are held (only a drift of μ past their pad would extend
them). The semicore QP energies still move by the level repulsion of the kept
mixing, $-\sum_p |H_{ps}|^2/(E_p - E_s)$ to second order. `qp` (the default)
lets the class move with its own Σ.

Every other matrix band is protected: under the default `sigma_out_of_grid =
cover` it reads Σ at its own energy, and the grid grows over it; on a route
without the patch a band deeper than the W model's active depth reads
$\Sigma(\omega = 0)$ (§4). Map 0, or an authenticated seed, classifies the
band set once, and the set stays frozen: no band enters or leaves it later.

**Tail law.** The rigid shift is

$$
\beta = \frac{\sum_{kn} w_k Z_{nk}\,(E^{\rm QP}_{nk} - E^{\rm DFT}_{nk})}{\sum_{kn} w_k Z_{nk}},
\qquad Z_{nk} = \bigl(1 - \partial_\omega \mathrm{Re}\,\Sigma_{nn}(\omega)\rvert_{E_{nk}}\bigr)^{-1},
$$

over the window's conduction states that are on the sampled grid and have
$0 < Z \le 1$. A state on a satellite or near a pole of Σ has small $Z$ and
cannot drag the tail: a state moved 3 eV onto a satellite with $Z = 0.1$
shifts β by 21 meV, where the plain mean moves 188 meV (CLAIMS 2710). An
off-grid state is excluded because its energy did not come from its own
$\Sigma(E)$; on Fe 4³ three such states moved β by 14.9 meV between two fixed
points (CLAIMS 2703). The map's Σ exists only after the tail has fed
$\chi_0$ and $W$, so $Z$ comes from the previous map and rides the carry
(`SCState.tail_z_kn`); map 0 uses unit weights. No qualifying state leaves
the tail at $E_{\rm DFT}$.

The masks are indexed by `(k, DFT identity)` because the carry is in the DFT
basis. On every map, `sc_state_identity.assign_qp_identity` assigns each
sorted QP column to a DFT identity by projector overlap with the reference
multiplets. Motion and the criterion use these identities, not sorted
positions, so a level crossing cannot relabel a state.

In a degenerate subspace the full operator is kept. Averaging only the
diagonal of a degenerate block would depend on the arbitrary DFT basis within
it and would break symmetry in the next map. BerkeleyGW degeneracy averaging
(`no_degen_averaging`) applies only to the reported diagonals.
`sc_exact_degeneracy_tol_ev` groups identities; it refuses values above
0.1 meV and is not a convergence knob.

## 3 Acceleration and stop rules

`mixing.acceleration.anderson_nojit` implements Anderson type II (Pulay) with
one map evaluation per iteration. The history holds the newest $m + 1$
evaluated pairs $(x_i, f_i = F(x_i) - x_i)$, where $m$ = `sc_history_depth`
(default 20). The next and only evaluation is at

$$
x_{n+1} = \sum_i \alpha_i\,(x_i + f_i), \qquad
\alpha = \arg\min \Big\| \sum_i \alpha_i f_i \Big\|_P ,\quad \sum_i \alpha_i = 1 ,
$$

with $\alpha$ real, because Hermitian matrices form a real vector space. The
metric multiplies each residual by the per-k outer product of the window's
identity mask and the square root of its star multiplicity; padding is zero.
Its squared norm is therefore the full uniform k-grid sum, independent of
the computational wedge. Full-grid rows retain unit weight. Two safeguards
cost no evaluation and have no tunable constant:

- **Conditioning filter.** The oldest differences are dropped until the
  unit-column Gram has condition number at most $10^{12}$.
- **Nonmonotone fallback.** An evaluation worse than every residual in the
  window steps next along the two-point secant between the best pair and the
  rejected one. It never fires twice in a row, and the rejected pair stays in
  the history.

A discrete map event does not restart the history. Such an event is a Σ rule
rebuild or a grid extension, logged as `SC map
event`. Restarting there would reduce the method to Picard steps, which
diverge on an expansive map.

Plain iteration is refused. On dense band manifolds the QSGW Jacobian has
cycle-direction eigenvalues of about −3 or below, so a plain fixed point
2-cycles, and damping only shrinks the cycle. Undamped linear mixing also
amplifies the input's time-reversal-reality error 6–8× per map (CLAIMS 2391).
`sc_accelerator` therefore accepts only `anderson`, and any other value
refuses (`GATE sc_accelerator_anderson_only`).

The solver has no stopping authority of its own. The driver decides on every
evaluated input:

| verdict | rule |
|---|---|
| **CONVERGED** | $\max \lvert E_{\rm out} - E_{\rm in} \rvert$ over the window identities is below `sc_tol_ev`. The loop returns that input, together with its own Σ, W and head. The rule compares output with input, not successive iterates: a mixed iterate can barely move while $F$ still has no fixed point. |
| **STALLED at floor, not converged** | $r_n = \max_k \lVert P\,(F(H_n) - H_n)\,P \rVert_2$, logged as `SC matrix residual`, has not improved by 10 % over the last 12 maps. $r_n$ is label-free: by Weyl it bounds every sorted-eigenvalue residual, and it also sees eigenvector error. The 10 % and the 12 maps are fixed, not deck keys. |
| budget | `sc_max_iter` (default 30) accelerated evaluations after map 0. `sc_max_iter = 1` is a one-map diagnostic. |
| **fixed point NOT UNIQUE** | appended to CONVERGED when states lie within the Σ(E)/Σ(0) jump of a grid edge (§5). It is not a refusal. |

A stalled or budget-exhausted run refuses with
`GATE sc_fixed_point_not_converged`. The per-map `eqp0_iterNNNN.dat` files
remain, and no terminal QP result is reported. Just before the refusal the
record prints one block for the last map: the median $\lvert\Delta E\rvert$
over all states, then the 1-based bands (no k) of the non-scissored set with
a state moving more than 20 meV, and with a state moving at least `sc_tol_ev`.

**Cost.** The history is $2(m+1)$ copies of the carry, stacked on a leading,
never-sharded axis. Bra bands sit on `x` and ket bands on `y`
(`qsgw_density.band_rotation_spec`), with k replicated. One copy is
$16\,n_k n_b^2$ bytes: 21 MB on CrI3 8×8 (144 bands), and 9.2 GB at
$n_k = 144$, $n_b = 2000$, where $m = 20$ takes 387 GB globally, or 3.9 GB per
rank at $P = 100$. Each iteration issues one $(m+1)\times(m+1)$ Gram
reduction. The map itself needs a replicated carry, so each call gathers one
$(n_k, n_b, n_b)$ matrix. Each map output $F(H_n)$ is diagonalised once, on
device; the identity readout and the warm seed share that eigensystem, and
$r_n$ is reduced on device, so no $O(n_k n_b^3)$ eigensolve runs on the host.
For scale, CrI3 8×8 GN-PPM reaches
$\max\lvert dE\rvert < 0.1$ meV in 13 map calls (CLAIMS 2686).

**Map gain.** From map 2 the log prints
`SC map gain: max |dSigma_on-shell| / max |dE_in|` over adjacent maps, and the
eqp comments carry the same figure. A value above 1 means the sampled map is
not locally contracting. The gain is a diagnostic and controls nothing. On
metals it can stay above 1 at Fermi-crossing states, where exchange responds to
an occupation flip within the smearing width. That is the physics of the map,
not a failure of the accelerator.

## 4 Σ grid and quadrature across maps {#sigma-grid-and-quadrature}

The ω grid is measured from $E_F$. The one-shot and every SC map grow the
requested grid by one rule (`qp_support`, below), so SC map 0
is the one-shot calculation: the same grid, the same rules and the same
out-of-grid set (owner, 2026-09-24). Under `cover` the one-shot therefore reads
$\Sigma(E)$, not $\Sigma(0)$, for an active state outside the requested grid.

- **Out-of-grid policy** (`sigma_out_of_grid`, owner 2026-09-24). One
  classification, `qsgw_utils.omega_coverage`, decides which energies are
  on the sampled grid. The Σ build (`qsgw_utils.sigma_eval_omega`), the
  window plan below and the tail mask all read it.

  | policy | an off-grid $\Sigma(E)$ reads | error past the edge, median / p90 (eV) | risk | cost |
  |---|---|---|---|---|
  | `cover` (default) | no requested quasiparticle is off-grid: the grid grows over each (window plan below); deeper identities and states without a quasiparticle read $\Sigma(\omega = 0)$ | 0 for active states | shallow semicore near the active-depth line is covered and stiff: MoS2 S 3s (depth 13.9–15.1 eV) takes 17–18 maps against static's 8 | Fe 4³ with no frozen core: grid to +31 eV (not −98), SC driver 1.33× static on one map |
  | `clamp` | $\Sigma(\omega_{\rm edge})$ | Fe 0.3–0.7 / 0.8–4.7; CrI3 0.05–0.12 / 0.5–2.2; MoS2 0.2–0.3 / 0.6–0.8 | continuous at the edge, but every clamped state inherits Σ there; an edge on a GN-PPM pole gives errors of order $10^3$ eV | none |
  | `static` | $\Sigma(\omega = 0)$ | Fe 1.6–4.4; CrI3 0.4–0.5; MoS2 0.6–0.8 (median) | two fixed points for states within the edge jump (§5) | none |

  The numbers are from CLAIMS 2710: truth is the sampled Σ over the 2–6 eV
  beyond a truncated edge. `clamp` and `static` are there to second-guess a
  hard system. **Active** is the shared-pole census rule,
  $\max_k E^{\rm DFT}_{nk} \ge E_F - 15$ eV (`shared_pole_recipe.active_band_mask`),
  evaluated once on the DFT ladder, so a state never switches between
  $\Sigma(E)$ and $\Sigma(0)$. Deeper states are the ones the W model carries no
  plasma charge for; covering Fe 4³'s 3s/3p stretched the grid to −98 eV,
  ran 5× slower and moved them 16 eV (CLAIMS 2739). An energy-only law for
  them would sit 11–16 eV from $\Sigma(0)$, which is why they keep it.
  A tail matched to the edge, $C_n/(\omega - \bar\omega_n)$ with the sum
  rule $C_n > 0$, is not offered: Σ at a grid edge is far from its $1/\omega$
  asymptote (Fe: −5 to −7 eV at +28 eV), so the matched pole falls inside
  the extrapolated range for 10–92 % of the states (CLAIMS 2710).
- **Window plan** (owner 2026-09-25, 2026-09-27; `gw/qp_support.py`). The
  sampled grid is chosen to converge only the requested states and never
  leaves
  $D \cup [\min_{n\in R} E^{\rm in}_n - P,\ \max_{n\in R} E^{\rm in}_n + P]$,
  a refusal (`GATE sigma_support_envelope`) if it would. $D$ is the deck's
  `sigma_omega_min_ev`/`sigma_omega_max_ev` (or patch list), fixed. $R$ is the
  QP window's identities (`nval`, `ncond`) that the W model treats as active,
  outside `sc_frozen_core_bands`, and quasiparticles at the previous map,
  $Z \in (0, 1]$; under `clamp` and `static` only those inside the padded
  window (`scissor.sc_padded_window_ev`). $E^{\rm in}$ is DFT at map 0 and
  the carried QP eigenvalue after; eqp0, eqp1 and $Z$ are never an input.
  The pad is flat, $P$ = 2 eV, and there is one plan: the one-shot and SC
  map 0 (owner 2026-09-28). Every later map holds the grid while every
  requested state's read support
  $[E - 0.5, E + 0.5]$ eV (the $Z$ stencil, `eqp_bgw.Z_FINITE_DIFFERENCE_EV`)
  lies inside it; when one is about to cross, only that edge grows, to
  $E \pm P$, the edge the plan would set for that state, and one
  `SC window extension` line names the band, k, $E - \mu$, the edge and the
  run's extension count. A map-1 re-plan at 1 eV was deleted: it changed the
  grid, so every Σ executable recompiled at map 1, while the held rules
  already paid for the map-0 grid. A requested state with $Z \notin (0, 1]$ has no
  quasiparticle: its energy never moves the grid, and off the grid it reads
  the out-of-grid rule and is named in an `SC window no-quasiparticle`
  line. A grid that reaches far above $E_F$ therefore means the deck
  requested states there: the Na 8³ deck with `ncond` = 81 requests every
  band, up to +96 eV. Old samples do not move on an extension and an interior
  hole refuses. The Σ rule certificates below pad the band-sum states'
  outer edge by $\max(2\ \mathrm{eV}, 10\%\,\lvert E - \mu\rvert)$ and
  a crossing window's inner edge by $2\eta$.
  Coverage is judged in the frame the Σ build measures from: the current
  spectrum's VBM or midgap for GN/HL-PPM (`ppm_sigma.ppm_fermi_frame`),
  `efermi.resolve_sigma_efermi_ry` for MPA. On MoS2 3×3 the PPM frame sat
  1.4 eV above the DFT midgap, and judging coverage in the wrong one left a
  "covered" state on $\Sigma(0)$ until a later growth switched it, a 2.8 eV
  jump of its map output (CLAIMS 2736).
- **Width.** The crossing-rule node count grows linearly in bandwidth$/\eta$,
  and Σ far from $E_F$ is not smoother: on Fe the curvature at
  $\lvert\omega\rvert \ge 15$ eV is 13× that near $E_F$.
  `sc_frozen_core_bands` holds a band at its DFT block at no Σ cost, but that
  law fails on Fe 3s/3p and CrI3 I 5s (CLAIMS 2859), so production decks do
  not freeze semicore.
  `sigma_regularization_ev` is the literal broadening η of every ansatz and
  is not a speed knob. The quadrature page owns η and
  `sigma_quadrature_eps`.
- **Held rules** (`sigma_box_plan._fit_fixed_sc_rules`). Map 0 is served
  by the one-shot planner's rules, and the same balanced pass certifies the
  plan, the only one: one rule per product window on its box over the map-0
  grid. In the branch's own coordinate ($E - \mu$ on a conduction branch,
  $\mu - E$ on a valence one) the outer state edge is padded by
  `scissor.sc_window_pad_ev` = max(2 eV, 10 % of $\lvert E - \mu\rvert$).
  Near $E_F$ that is the owner's 2 eV; far away the 10 % is the QP stretch
  (Na 8³: top state +96 → +101 eV at map 1), which a flat pad cannot hold: a
  flat 1 eV map-1 plan refit Na's 1265- and 1669-node crossing windows in
  553 s. The inner edge of a crossing window is padded by
  `scissor.SC_WINDOW_INNER_PAD_ETA` = $2\eta$ and, on a metal, never past
  $-X$, the occupation floor's reach (`efermi.occupation_floor_reach_ry`:
  $X = k_BT \ln(1/10^{-5} - 1)$ = 11.5 $k_BT$ for Fermi-Dirac). That edge
  sets the crossing short side $\lvert\omega\rvert_{\max} + x -
  \Omega_{\min}$ and so the node count; a 2 eV pad there cost 18–26 nodes
  per crossing window. The edge is deliberately tight, and only a metal's
  clip is a bound (no branch state passes $-X$). The old 2 eV slack also
  absorbed motion that the tight edge now rebuilds: an inward move of the
  inner state (MoS2 3×3 map 2, the gap edge back 1.5 eV after its map-1
  overshoot), a grid extension on the crossing half (Fe 4³ map 2) and a
  near-pole drop past its 10 % pad (Fe 4³ map 1, 0.28 → 0.19 eV) each
  escape and rebuild (Fe 4³ charge SC-3: 12 windows against 9; claim 2936).
  A sign-definite window keeps the outer pad on
  both edges. Both edges stop at the window's own selector interval. Poles are padded by 10 % at the near edges and
  widths, and 2× at the far edge of an unbounded selector (deep and bulk
  windows), because the highest shared-pole mode moves 10–30 % per map and a
  sign-definite relative rule pays about one node for it. From map 1 the
  rules are held: a map reuses a rule by containment
  (`rule_source` `hit:sc-fixed`). A window whose box leaves its rule, a new window,
  or a sign change is an escape: it is rebuilt alone by the same plan rule
  around its current states (`rebuild:sc-fixed`) and held again; one
  `SC fixed quadrature recompute:` line in the report names the window and
  its reason, the state (k, band, $E - \mu$) and the certified edge, the pole
  extent or the grid edge that crossed, and the receipt counts the maps
  with an escape (`escape_maps_total`) and the windows rebuilt
  (`rebuilds_total`) over the run. The zero-side edge of a sign-definite box
  stops at 5 % of its distance to zero, so the box stays sign-definite. The
  window executables keep the session's largest node count, so a refit
  recompiles them only when it raises it. A change of material class
  re-initializes the plan. Every path accepts a rule only if its certified
  sup error is at most `sigma_quadrature_eps`; otherwise it refuses, naming
  the window, box, sup and node count. There is no retry; η and ε are fixed
  for the session.
- **Held W carriers.** From map 1 the shared-pole model keeps one
  pole-column extent per sector (`shared_pole_store._k_extent`: map 1's
  Kmax + 3 %, grown with headroom only when a live Kmax exceeds it, logged),
  the bispinor CT round keeps each of CC and TT at its largest retained-span
  width (`shared_pole_sectors.cross_span_widths`: the extent ladder of the
  retained rank, grown only when a live width exceeds it, logged, never
  shrunk; the extra columns are inactive zeros), and the density scan keeps
  its largest rotated-band count, so a drifting Kmax, retained rank or
  occupation does not change a compiled shape.

## 5 Where the map is not smooth

**The grid-edge switch** (`sigma_out_of_grid = static` only). $\Sigma(0)$
makes $F$ discontinuous at each edge $\omega_e$ by

$$
\Delta_n = \mathrm{Re}\,\Sigma_{c,nn}(0) - \mathrm{Re}\,\Sigma_{c,nn}(\omega_e).
$$

In the diagonal model $E = A + \Sigma(E)$, a jump that points outward
($\Delta_n > 0$ at the top edge, $\Delta_n < 0$ at the bottom) gives every
state within $\lvert\Delta_n\rvert$ of the edge a self-consistent partner on
the other side. There are then two fixed points, and the path decides which
one the loop reaches. An inward jump cannot do this.
`qsgw_utils.sigma_grid_edge_ambiguity` flags these states every map. It has no
threshold, because the band is the jump itself. It takes the effective edge as
the outer of the sampled-grid edge and the padded-window edge, since a state
inside the padded window grows the grid rather than leaving it. Frozen-core
bands are excluded. On Fe 4³ bispinor, the H-point states converge at
$E - \mu = +9.8$ eV (inside) under one trajectory and at $+11.7$ eV (outside)
under another, around a +10 eV edge with $\Delta = 1.87$ eV (CLAIMS 2688).
Under `cover` both seeds reach one branch within 2 meV (CLAIMS 2703); `clamp`
has no jump, so the verdict flags edge states only under `static`.

**The elementwise MPA pole refit** (`compute_mode = mpa`,
`sigma_w_model = mpa`). The imaginary-axis samples of $\chi_0$ and $W$ respond
linearly to a kick. The per-element Loewner/Padé solve from 16 samples is not
identifiable, though: a $10^{-7}$ change in the samples selects a different,
equally good pole set, and $\Sigma_c(\omega)$ jumps by 10–20 meV somewhere on
the real axis (CLAIMS 661). No damping or mixing schedule converges this jump.
Resolution makes it harmless. At 24 bands and 192 centroids on Si the map gain
is 14–18 and the loop plateaus; at 80 bands and 504 centroids it is 0.3 and
the loop contracts.

**Discrete events.** A rule rebuild or grid growth is a small jump of $F$. It
is logged, and the history keeps it (§3).

## 6 Shared-pole W with retained quadrature {#shared-pole-w-with-retained-quadrature}

With `sigma_w_model = shared_pole`, every map rebuilds the whole model from its
rotated wavefunctions, energies and occupations: the response samples, exact
moments, directions, Ritz poles and factors. SC W models are local to one map
and are never published as reusable bundle members. `restart = true` restores
only the invariant ISDF basis. Three things persist across maps:

- **Σ rules.** The frozen set of §4 keeps its nodes and weights. Each map
  recomputes the masks, pole selectors, reference energies and $W(\tau)$.
  For a sector (bispinor) model the frozen certificate also covers the pole
  ceiling $[0, \Omega_{\rm ceil}]$, fixed at map 0 as twice the χ transition
  span (`shared_pole_recipe._sector_treatment_ceiling`), so a pole that moves
  within it keeps the same nodes.
- **Response rules** (`response_bank.response_quadrature`). These are planned
  on the transition interval padded by 4 eV. They are reused while the
  current interval, decay and amplitude bounds, metallicity and sample points
  are contained in the plan, and otherwise rebuilt warm-started from the
  previous rules.
- **Support enclosure** (`shared_pole_recipe._support_envelope`). The DFT
  reference map uses its own support. From the first interacting map on, the
  loop keeps the running maximum line endpoint and the extreme imaginary
  endpoints, so the DFT gap's extra imaginary support is never locked in. The
  enclosure fixes sampling geometry only; it is not an interpolation-error
  certificate.

`head_correction = full` evaluates the current shared-pole Γ body on each map,
one frequency at a time on both mesh axes, and folds the head wings through
the total W. The head fit is bound to that map's body digest.
`sc_head_update = off` keeps the DFT direct response. The ordered-store and
four-component head refusals are covered in
[shared-pole model §8](architecture/shared_pole_model.md).

**Physical realization.** The model is
$W_c(q,s) = \Pi_{G_q}\big[\sum_k b_k b_k^\dagger/(s - \Lambda_k)\big]$, where
$\Pi_{G_q}$ averages the authenticated magnetic little group of $q$ (recipe
`operator_realization = little-group-reynolds-v1`, adapter
`gw/qgrid_symmetry.py`, operations from
`symmetry_maps.project_little_group_operator`). Each transformed residue is a
unitary or conjugate-unitary congruence of a positive residue, so the average
keeps residues positive and poles real. At complex $s$, an antiunitary
operation acts as the same-time transpose of the residue endpoints, not as a
conjugation of the whole value. The projector streams one operation at a time
into a fixed accumulator, so no factor gains a symmetry axis. The raw-model
receipts certify the unprojected model only.

## 7 Metals

- Use `compute_mode = mpa` with `sigma_w_model = shared_pole`. GN-PPM refuses
  metals (`GATE gn_ppm_refuses_metals`).
- Occupations are Fermi-Dirac only, and `occ_smearing_width_ry` is $k_BT$.
  Each map solves μ at a fixed electron count from its input spectrum
  (`_solve_occupation_state`); μ is never mixed.
- A band is in a Σ or χ branch iff its weight ($f$, or $1-f$) is at least
  $10^{-5}$ (`gw.efermi.band_in_occupation_window`; $|E-\mu|\le11.5\,k_BT$ for
  Fermi–Dirac), for the one-shot and every map alike. A state that crosses
  the cut between maps switches one term by about $10^{-5}$ of its size,
  0.01 meV; the retired 0.005 floor switched it by 5.4 meV on Fe
  (CLAIMS 2793).
- The energy-only tail above the QP window has exact-zero occupations. It
  still enters G and the response at its current shifted energies, and the
  window alone sets μ. A tail state that enters the fractional manifold
  (`FRACTIONAL_TOL`) refuses.
- The rate of convergence is set by the largest quasiparticle weight Z in the
  window. States more than a plasmon energy above μ, where Re Σ(ω) is flat or
  rising on shell, have $Z \gtrsim 1$ and walk at map gain ≈ 1. End the window
  (`ncond`) below them, and judge convergence by Fermi-window observables.

### Metals: direct Drude head {#metals-direct-drude-head}

| status on a metal | what | where |
|---|---|---|
| default | `sc_head_update = off`: the fixed DFT response on the DFT fixed-N Fermi-Dirac state, with the tetrahedron Drude term and the Thomas–Fermi static slot | `qsgw_head.build_dft_head_response`, `sc_iteration._fixed_dft_head_occupation_state` |
| admitted: shared-pole, `head_correction = no_local_fields`, or `full` on a scalar deck | `dft_velocity`: the `dipole.h5` velocity rotated into each map's QP basis, the current fixed-N μ and tetrahedron weights; the dynamic Drude tensor at $\omega \ne 0$, Thomas–Fermi at $\omega = 0$; `full` folds it through intraband wings and the static Γ body | `qsgw_head.build_iteration_head_response`, `sc_iteration._solve_head_occupations`, `gw_config.uses_metal_direct_drude_head` |
| refused (`GATE metal_sc_head_update_disabled`) | `parallel_transport`; `dft_velocity` with `full` on a bispinor deck | `gw_config.validate_material_inputs` |
| refused (`GATE shared_pole_head_ordered`) | `full` on an ordered (time-reversal-broken) store, on every route | `shared_pole_head._refuse_head_representation` |
| refused (`GATE metal_sc_head_update_disabled`) | `occ_broadening > 0` next to a metal width | `gw_config._validate_occupation_smearing` |

Insulators keep `parallel_transport` and `dft_velocity`.

## 8 Seeding, restart and outputs

- **Warm seed.** After every completed map that does not converge, the loop
  publishes `sc_seed/qp_wfn_rotations.h5` atomically. It holds that map's
  output Hamiltonian, the scissored tail energies, the fixed-N occupations and
  the frozen band policy. It holds no wavefunctions and no accelerator
  history.
- **Seeding a new run.** `sc_initial_qp_rotations_file` imports an
  authenticated eigensystem as $H = U\,\mathrm{diag}(E)\,U^\dagger$ in the
  original DFT basis. A seed whose band policy is not all-protected, or that
  carries an active-window scissor (both written only before the 2026-09-22
  all-protected rule), refuses. Keep the original WFN and
  reference operators. Occupations and the tail fit are recomputed, and the
  quadrature and the accelerator history start empty, so this is a new run,
  not a continuation.
- **`restart = true`** reuses the ISDF/W tensors of a finished run. It
  overwrites MPA pole stores in the same directory, so point it at a copy.
- **Terminal files.** `qp_wfn_rotations.h5` is always written;
  `WFN_qp.h5` is written when `write_wfn_h5` is set (the default). Both come
  from the accepted final map. They hold the complete energy ladder (window
  plus tail), the occupation table, μ, the smearing and the table hash.
  `WFN_qp.h5` is written collectively: each rank reads, rotates and writes
  its own G-slab of every k through `file_io.slab_io`, so no rank holds more
  than $\approx 5\,N_b N_s \lceil N_G^{\max}/P\rceil \cdot 16$ B of ψ. Each
  file is written to a private sibling, validated through its format owner,
  and made visible by `os.replace`. The run-completion manifest requires both
  names. `postprocess.rotate_wfn_to_qp` reapplies the stored ladder and table;
  it neither rebuilds the tail nor re-solves occupations.
- **Per-map files are diagnostics, not restart state.** `eqp0_iterNNNN.dat`
  holds the map output; at the fixed point it is the root of the QP equation,
  so no per-map eqp1 is written. `rotation_iterNNNN.npy`
  (`sc_dump_dir`) is the map's input $U$. With
  `sigma_lorentz_debug_output = true`, four-current maps write
  `sigma_lorentz_iterNNNN.h5`: the CC, CT+TC and TT sectors in the map's input
  QP basis.
