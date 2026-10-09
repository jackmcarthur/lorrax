# The dynamic Σ(ω) quadrature

Every dynamic self-energy reaches one planner and one executor through
`gw.mpa.sigma.compute_sigma_c_mpa_omega_grid`: GN/HL-PPM (written as a
one-pole in-memory store), elementwise MPA, the shared-pole W and its photon
sectors. The planner is `gw.sigma_box_plan`, the rule builder the derived
`minimax.analytic_box_rule` (§7), the τ kernel `gw.ppm_tau_kernel`. Pole models
belong to [Multipole frequency integration](THEORY_mpa_implementation.md),
[Metallic MPA screening](metallic-mpa-screening.md) and the
[shared-pole model](../architecture/shared_pole_model.md); this page owns the
frequency integral.

## 1. What is computed

W^c has poles Ω_p (Re Ω_p > 0, Im Ω_p ≤ 0) and residue matrices B_p(q) on the
ISDF centroids r_μ. After the ω′ integral each causal branch b contributes

$$
\Sigma^b_{ij}(\mathbf k,\omega)=\sum_{\mu\nu}\psi^*_{i\mathbf k}(r_\mu)\,S^b_{\mu\nu}(\mathbf k,\omega)\,\psi_{j\mathbf k}(r_\nu),
\qquad
S^b_{\mu\nu}=-\frac1{N_k}\sum_{\mathbf q}\sum_{A\in b}\sum_p
\frac{w_A\,\psi_{A,\mathbf k-\mathbf q}(r_\mu)\,\psi^*_{A,\mathbf k-\mathbf q}(r_\nu)\,B_{p,\mu\nu}(\mathbf q)}
{d_b(\omega;E_A,\Omega_p)},
$$

$$
d_b=\omega-\sigma_b(E_A+\Omega_p)+i\sigma_b\eta .
$$

On the empty branch σ_b = +1 and E_A = ε_A − μ; on the occupied branch
σ_b = −1 and E_A = μ − ε_A is the hole energy. The weight w_A is 1 on an
insulator and 1 − f_A or f_A on a metal. η (`sigma_regularization_ev`) is the
literal retarded broadening: one kernel and one η for every ansatz, both
frequency halves and every branch. The four branches are
(ω ≥ 0, ω < 0) × (empty, occupied). For E_A ≥ 0, Re d_b changes sign only on
the empty branch at ω ≥ 0 and the occupied branch at ω < 0; the planner reads
the actual sign topology from each window's box (§5).

Summed as written, this is a (q, A, p) sum at every (k, ω). The quadrature
below removes the state–pole product.

## 2. Separation: one τ node is one convolution

For Im d > 0, 1/d = −i∫₀^∞ e^{itd} dt. A rule Q(d) = Σ_l w_l e^{i t_l d}
with complex times factors each term into an external-frequency scalar, a
state factor and a pole factor. In executor time τ = σ_b t, restricted to a
product window w = (states S_w) × (poles P_w):

$$
S^{b,w}_{\mathbf k}(\omega)\approx-\sum_l w_l\,e^{-\eta\tau_l}\,
e^{-i(E^{\rm ref}_A+E^{\rm ref}_B-\sigma_b\omega)\tau_l}\,
\big[G^w\circledast W^w\big]_{\mathbf k}(\tau_l),
$$

$$
G^w_{\mathbf k}(\tau)=\sum_{A\in S_w}\psi_{A\mathbf k}\,w_A\,e^{-i\tau(E_A-E^{\rm ref}_A)}\,\psi^\dagger_{A\mathbf k},
\qquad
W^w_{\mathbf q}(\tau)=\sum_{p\in P_w}B_p(\mathbf q)\,e^{-i\tau(\Omega_p-E^{\rm ref}_B)},
$$

with [G ⊛ W]_k = N_k⁻¹ Σ_q G_{k−q} ⊙ W_q, a convolution over the k grid done
by FFT. The rule builder works in the upper half plane; the occupied branch's
lower-half-plane box is served by 1/d̄ = conj(1/d), i.e. t → −t̄, w → w̄.
The references sit at the bounded end of each factor (crossing window:
E^ref_A = min E_A, E^ref_B = 0; sign-definite window: the state and pole
endpoints on the decaying side), and their sum returns in the scalar
phase.

## 3. Cost

Per (window, τ) pair on P ranks, with N_k k-points (N_k^par symmetry
parents), N_μ centroids, n_s spinor components, N_b^w live bands in the
window and N_σ projected bands:

| step | operation | cost |
|---|---|---|
| W synthesis | Σ_{p∈P_w} B_p e^{−iτ(Ω_p−E^ref_B)} over (q, μ, ν) | O(\|P_w\| N_k N_μ² / P) |
| G build | one complex GEMM per parent, (N_μn_s × N_b^w)(N_b^w × N_μn_s); a second on the conjugated faces when an antiunitary row meets complex weights. The Green stays on the parents | 8 N_k^par (N_μn_s)² N_b^w / P flops |
| k-convolution | inverse FFT of W (once per node); then, per centroid pair, the typed unfold and spin action U G U† applied on the load of the parent Green, inverse FFT, product in R, forward FFT | O((N_μn_s)² N_k log N_k / P) |
| projection | ψ† S ψ on parents, band-block reshard | O(N_k^par N_μn_s N_σ (N_μn_s + N_σ) / P) |
| fold | the scalar above times S into each of the window's frequencies | O(N_ω^w N_k^par N_σ² / P) |

Only the fold sees the output frequencies. The sweep costs
(Σ_w N_τ^w) × (one GEMM + one k-convolution + one projection), so the
**(window, τ) pair count is the currency**: a plan is judged by its pairs
first and its planning time second. Planning is host scalar work on boxes and
never touches a spatial array. The live set per node is the parent Green (and
its partner on an antiunitary plan), one full-k convolution output of
N_k(N_μn_s)² and one W tile of N_k N_μ² complex numbers, over P; the unfolded
full-k Green is never written ([unfold on load](../architecture/kconv.md#unfold-on-load),
mode 7). W never carries a pole axis.

On the resident-pole route (GN/HL one-pole store, elementwise MPA) poles are
read in batches of `mpa_pole_batch_size` (1–8, default 4), and a window runs
once per batch that holds any of its poles, so a window over N_p poles pays
⌈N_p/b⌉ G builds and convolutions per node. The one-pole store is one batch.
The shared-pole route synthesizes W(τ) for all poles inside each node and
pays each node once.

## 4. Product windows

With a = f_e η (f_e = `sigma_window_edge_factor`, default 1.5),
x = max(0, −min_A E_A), Λ_h = max_h|ω| + a + x over the branches of ω half h,
and ν = a + x, each branch is partitioned into at most four Cartesian products
(states × poles × |ω|); `gw.mpa.sigma_windows.sigma_pole_edges` owns Λ_h and ν:

| branch | window | states | poles (Re Ω) | \|ω\| |
|---|---|---|---|---|
| crossing | resonant | E ≤ Λ_h | (0, Λ_h] | all |
| crossing | state tail | E > Λ_h | (0, Λ_h] | all |
| crossing | pole tail | all | > Λ_h | all |
| sign-definite | bulk | E > a | all | all |
| sign-definite | resonant | E ≤ a | (0, ν] | < ν |
| sign-definite | pole tail | E ≤ a | > ν | < ν |
| sign-definite | ω tail | E ≤ a | all | ≥ ν |

- **Products**, because only a product set factors into G^w(τ) ⊙ W^w(τ). A
  selector coupling one state to one pole reinstates the state–pole sum.
  A window owns a subset of its branch's frequencies; the executor scatters
  it by index.
- **A partition**: each causal (state, pole, ω) tuple has one owner, so
  the error bound of §6 carries no window-count factor.
- **These cuts** keep far states and far poles out of the crossing box, whose
  rule is linear in its width, and put them in sign-definite boxes, whose
  rules are logarithmic. On the sign-definite branch the states within a of
  zero (a small gap, an inverted band, a metal's Fermi surface) are split off
  so the bulk box stays sign-definite.
- **The ω cut ν** of the sign-definite branch: there |d| = |ω| + E + Re Ω
  with E ≥ −x, so above ν every denominator clears a and the ω tail is one
  relative box. Below ν only the excursion sliver crosses zero, over a box of
  size ~ν rather than ω_max + Ω_max (Na 8³ map 0: 1682 → 39 pairs).
  A branch with no state within a of zero (an insulator) keeps one bulk window.
- **One Λ per half**, because the SC cover grows the upper half only; a global
  Λ put 7.9 Ry of poles into the lower half's crossing box on Na.

Empty windows are dropped. Windows are never merged: a whole-branch rule
widens cheap sign-definite tails into one expensive crossing box. On a metal
the branch supports carry the occupation weights: a band belongs to the
occupied branch at weight f when |f| clears the occupation window and to the
empty branch at weight 1 − f when |1 − f| clears it, so partially occupied
bands sit in both. A state on the wrong side of μ widens Λ through x.

## 5. Denominator boxes

The real support of window w is the extent of the eight corners
ω − σ_b(E + Re Ω), over the window's extreme frequencies, the window's
extreme live states and its extreme live poles. The imaginary support is
[γ_min + η, γ_max + η] with γ = −Im Ω. Pole extrema come from a distributed
census that keeps, per pole and selector, the extrema over live entries only;
a residue decides whether an entry is live and never weights anything. The
real interval is padded by 2% of max(width, η), and a sign-definite edge moves
at most 30% of its distance toward zero, so padding never turns a
sign-definite box into a crossing one.

No histogram, sampled lattice or error apportionment enters. The same box
gives the same rule on every deck, which lets one run's plans share rules and certifies a
low-mass state at the Fermi level exactly as it certifies a heavy one; a
mass-weighted fit is what loses such a state.

## 6. Error currencies and the delivered bound

| box | currency | certificate |
|---|---|---|
| crossing (Re d spans 0) | peak-relative | sup_box η_min \|Q(d) − 1/d\| ≤ ε, η_min = min Im d |
| sign-definite | relative | sup_box \|d\| \|Q(d) − 1/d\| ≤ ε |

ε is `sigma_quadrature_eps`, the only accuracy dial, read the same way on
every Σ route (GN/HL-PPM, MPA, shared pole). The default is 10⁻⁴ (the
`relaxed` shared-pole tier defaults to 5·10⁻⁴); the
[input reference](../input_reference.md) owns both.

Because the windows partition the tuples, a state's delivered error is one
factor of its own matrix elements M_np times the certificate:

$$
|\delta\Sigma_n(\omega)|\le\varepsilon\sum_{p\in\text{crossing}}\frac{|M_{np}|}{\eta_{\min}}
+\varepsilon\sum_{p\in\text{sign-definite}}\frac{|M_{np}|}{|d_{np}|}.
$$

The two currencies follow from that bound. A term's contribution scales as
1/|d|. On a sign-definite tail a peak-relative ε would be a relative error of
ε|d|/η, large on semicore terms at |d| ≫ η, while a relative rule costs
nothing extra because exponential sums for 1/x on [a, b] are uniformly
relative at O(log(b/a)) terms. On a crossing box 1/d is bounded by its peak,
and a relative criterion would over-resolve the far edges by |d|/η for terms
the peak already dominates.

## 7. The rule and its node laws

`minimax.analytic_box_rule(box, ε)` places every node by formula and gets
the weights from one linear least-squares solve; nothing is optimized. A
crossing box gets the bent contour: a trapezoid line in complex time whose
endpoint error is exactly two Laplace integrals on the imaginary time axes,
carried by two Gauss image sets. A sign-definite box gets the elliptic
time-Ritz sector rule with the local extremal-length count. The derivation, the count laws and the fallback are
[minimax quadrature §7](minimax-quadrature.md#sigma-box-rules).

A crossing box costs about γ∫B(σ)dσ/2π nodes on the line, B(σ) the largest
live |Re d|, plus O(log(M/η)·ln(1/ε)) image nodes; no construction goes
below the band-limit floor (bandwidth × horizon/π), so the remaining savings
are in η, ε and the window geometry. A sign-definite box costs
O(log R · log(1/ε)). Neither rule reads a clock, so a rule is a function of
(box, ε) only, and a rule that cannot be certified is refused in planning,
before the sweep starts. The rule family `sigma-box-ry-v7` salts each rule's
digest, which orders equal-count candidates when a plan serves a window.

## 8. Acceptance

The planner accepts a rule for a window only if all three hold:

1. **Certificate.** Every node and weight is finite and the certified sup
   error is ≤ ε in the box's currency.
2. **Runtime noise.** With a per-term relative perturbation ε_rt = 6·10⁻⁸,
   ε_rt · max_{d∈∂box} ρ(d) Σ_l |w_l e^{i t_l d}| ≤ 5·10⁻⁶, where ρ = |d| on a
   sign-definite box and ρ = η_min on a crossing one. The budget is absolute
   (0.05 × 10⁻⁴): roundoff is set by the executor's arithmetic,
   not by ε, so a tighter ε does not tighten it. The noise mass is
   subharmonic, so its maximum is on the boundary, sampled at the rule's own
   horizon. Sign-definite rules are built under the cancellation cap that
   implies this bound.
3. **Factored growth.** Over the window's states and pole corners, neither
   separately factored exponential grows by more than e³⁰.

A failure is final. There is no retry at a tighter ε and no second quadrature
family, and the (window, τ) pair count is reported, never refused on.

## 9. Self-consistent maps

A multi-map QSGW run plans its rules once and holds them, because a rebuilt
rule costs planning time and, when it raises the node count, a recompile of
the window executables. The session lives in
`sigma_box_plan._fit_fixed_sc_rules`; the ω grid it is planned over is held by
the rule of [self-consistency §4](../self_consistency.md#sigma-grid-and-quadrature).

**Map 0** is served by the ordinary one-shot rules, so SC map 0 equals the
one-shot calculation bit for bit when no W-active state is semicore
([self-consistency §2](../self_consistency.md#2-band-treatment)). In the same parallel planning pass (§10)
the planner certifies one held rule per product window on a padded box over
the map-0 grid (`sigma_box_plan._sc_padded_box_spec`):

- **States, outer edge** (farthest from μ in the branch's own coordinate,
  E − μ on a conduction branch and μ − E on a valence one): padded by
  max(2 eV, 10 % of |E − μ|) (`scissor.sc_window_pad_ev`). Near E_F the 2 eV
  covers map-to-map motion; far from it QP corrections stretch the spectrum
  by about 10 % (Na 8³: top state +96 → +101 eV at map 1), which a flat pad
  cannot hold. The outer edge sets only a box's long side, so its pad costs
  almost no nodes.
- **States, inner edge of a crossing window** (nearest μ): padded by 2η
  (`scissor.SC_WINDOW_INNER_PAD_ETA`) and, on a metal, never past −X, the
  occupation floor's reach (X = k_BT ln(1/10⁻⁵ − 1) = 11.5 k_BT,
  `efermi.occupation_floor_reach_ry`), which no branch state passes. This
  edge sets the crossing short side |ω|_max + x − min Re Ω and so the node
  count; a 2 eV pad there cost 18–26 nodes per crossing window on the
  Fe 4³, MoS2 3×3 and Na 8³ SC decks. A
  sign-definite window takes the outer pad on both edges. Both edges stop at
  the window's own selector interval, because a state past it belongs to the
  neighbouring window, whose certificate covers it.
- **Poles**: near edges and widths padded by 10 %; the far edge of a window
  whose pole selector has no upper bound (pole tail, bulk and ω tail, §4) by
  a factor 2, because the highest shared-pole mode moves 10–30 % per map
  (Fe 4³ charge SC, map 2: 24.2 → 28.6 Ry) and a sign-definite relative rule
  pays about one node for the doubled edge. A four-current sector's pole
  treatment ceiling is included in the box.
- **Sign topology**: a sign-definite box's zero-side edge may move toward
  zero until it sits at 5 % of its map-0 distance, so the box stays
  sign-definite. Where the selectors guarantee a sign gap (the bulk and tail
  windows on positive real poles), the zero-side edge is that gap, 0.7 a
  with a the state edge of §4, which covers a state that joins the window
  on a later map.

**Later maps** reuse each window's rule while its current box lies inside the
rule's box (`rule_source` `hit:sc-fixed`). The tight inner edge lets some
motion escape: an inward move of the inner state, a grid extension on the
crossing half, or a near-pole drop past its 10 % pad. Four events end a
rule's hold:

- a window whose box leaves its rule's box, whose error currency changes
  (crossing ↔ sign-definite), or that did not exist at map 0 is rebuilt
  alone, by the same padding around its current states, and held again
  (`rebuild:sc-fixed`);
- a factored-growth failure (§8) on reuse rebuilds that window the same way;
- a metal ↔ insulator flip re-initializes the plan;
- a change of η or ε within the session refuses.

Each rebuild prints one `SC fixed quadrature recompute:` line naming the
window, the reason and what crossed (the state's k, band and E − μ, the pole
extent or the grid edge, against the certified interval). The planner's
receipt, the per-map record of what it served and rebuilt that the Σ caller
writes to the run record, counts the maps with an escape and the windows
rebuilt over the run. The window
executables keep the session's largest node count, so a rebuild recompiles
them only when it raises it.

## 10. In-run reuse and parallel planning

**No stored rules.** No quadrature rule outlives its process (owner,
2026-09-28). Every plan builds its rules cold: the builder places nodes by
formula and solves one weight system, and the sampled term matrices of the
weight solve and the certificate are evaluated in row blocks on the rank's
cores (`minimax.uniform_rule._map_rows`), which changes no bit. The widest
crossing window of the gate decks (Na 8³, [−15, 19] eV, 1148 nodes) builds
in about 1 s (fit rows follow the fastest live term; Qᴴf from the reflectors). The builder pins numpy's and scipy's
OpenBLAS at min(16, CPUs of the mask), so a rule's bytes are a function of
(box, ε) on one machine class for every mask of 16 or more CPUs.

**Request scope.** Within one run a rule is reused only through an in-process
scope. The shared-pole route keys the scope by the map's physical identity
(energies, occupations and recipe, with the SC map label stripped), η, ε and
the pole census; the PPM and MPA routes use one scope per run. A sector Σ
call scopes by the union census of its CC, TT and CT_C models, so the four
sector calls of a map reuse each other's fits; a changed spectrum never
inherits another map's plan. A lookup serves the smallest rule whose box
contains the request at the same ε and currency, with noise amplification
under the cap and at most the closed-form node count of the request's own
build. The build box widens only real edges farther than 3η from zero (by 1%
of max(width, η)) and the far imaginary edge (by 1%), so sector calls reuse
by containment without widening a crossing rank.

**Parallel planning.** Windows are fit across processes, longest predicted
fit first, and only the small rules and receipts are gathered. Each window is
then served the smallest compatible rule of the whole plan, ranked by (node
count, digest), which does not depend on rank timing. A refusal on one rank
travels as data and raises on every rank.

## 11. Execution

- **One executable per window.** Each window's node loop runs on device and
  folds each Σ(τ_l) into the window's frequencies; nothing returns to the
  host between nodes. The resident-pole route builds W(τ_l) from the pole
  batch's residues.
- **Shared-pole and sector routes.** The same window executable also
  synthesizes W(τ_l) = b d(τ_l) b† from factors read once per Σ call: the
  scalar route at the irreducible parents, then unfolded to the full q grid;
  a sector from its endpoint factors on the full grid.
- **Band brackets.** For band-convergence extrapolation, one W preparation per
  node is shared by one G build, convolution and projection per bracket; a
  bracket with no live band is skipped
  ([fixed-shape kernels](../dev/gw_fixed_shape_kernels.md)).

A per-stage split of a node is a `jax.profiler` trace of the window executable.

## 12. Refusals

| refusal | fix |
|---|---|
| rule not certified at ε, or not finite (names window, box, kind) | a sign-preserving or split product window; never a looser `sigma_quadrature_eps` |
| runtime-noise bound above 5·10⁻⁶ | the same; the box is too ill-conditioned for its currency |
| factored log growth above 30 | the same |
| live pole with Re Ω ≤ 0 or Im Ω > 0, or a nonfinite residue | refit the pole model |
| a branch with no live states | the Σ band window has no band on that side; widen `number_bands_sigma` |
| η ≤ 0, ε ∉ (0, 1), edge factor < 0 | fix the deck |
| η or ε changed inside one SC session | one currency per run |
