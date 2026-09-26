# The dynamic Σ(ω) quadrature

Every dynamic self-energy reaches one planner and one executor through
`gw.mpa.sigma.compute_sigma_c_mpa_omega_grid`: GN/HL-PPM (written as a
one-pole in-memory store), elementwise MPA, the shared-pole W and its photon
sectors. The planner is `gw.sigma_box_plan`, the rule builder
`minimax.build_uniform_rule`, the τ kernel `gw.ppm_tau_kernel`. Pole models
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
full-k Green is never written ([k-convolution router](../architecture/ffi_layout.md#k-convolution-router-and-the-mathdx-family),
mode 7). W never carries a pole axis.

On the resident-pole route (GN/HL one-pole store, elementwise MPA) poles are
read in batches of `mpa_pole_batch_size` (1–8, default 4), and a window runs
once per batch that holds any of its poles, so a window over N_p poles pays
⌈N_p/b⌉ G builds and convolutions per node. The one-pole store is one batch.
The shared-pole route synthesizes W(τ) for all poles inside each node and
pays each node once.

## 4. Product windows

With a = f_e η (f_e = `sigma_window_edge_factor`, default 1.5),
x = max(0, −min_A E_A) and Λ = max|ω| + a + x, each branch is partitioned into
at most three Cartesian products:

| branch | window | states | poles (Re Ω) |
|---|---|---|---|
| crossing | resonant | E ≤ Λ | (0, Λ] |
| crossing | state tail | E > Λ | (0, Λ] |
| crossing | pole tail | all | > Λ |
| sign-definite | bulk | E > a | all |
| sign-definite | resonant | E ≤ a | (0, Λ] |
| sign-definite | pole tail | E ≤ a | > Λ |

- **Products**, because only a product set factors into G^w(τ) ⊙ W^w(τ). A
  selector coupling one state to one pole reinstates the state–pole sum.
- **A partition**: each causal (state, pole, ω-sign) tuple has one owner, so
  the error bound of §6 carries no window-count factor.
- **These cuts** keep far states and far poles out of the crossing box, whose
  rule is linear in its width, and put them in sign-definite boxes, whose
  rules are logarithmic. On the sign-definite branch the states within a of
  zero (a small gap, an inverted band, a metal's Fermi surface) are split off
  so the bulk box stays sign-definite.

Empty windows are dropped. Windows are never merged: a whole-branch rule
widens cheap sign-definite tails into one expensive crossing box. On a metal
the branch supports carry the occupation weights: a band belongs to the
occupied branch at weight f when |f| clears the occupation window and to the
empty branch at weight 1 − f when |1 − f| clears it, so partially occupied
bands sit in both. A state on the wrong side of μ widens Λ through x.

## 5. Denominator boxes

The real support of window w is the extent of the eight corners
ω − σ_b(E + Re Ω), over the branch's extreme frequencies, the window's
extreme live states and its extreme live poles. The imaginary support is
[γ_min + η, γ_max + η] with γ = −Im Ω. Pole extrema come from a distributed
census that keeps, per pole and selector, the extrema over live entries only;
a residue decides whether an entry is live and never weights anything. The
real interval is padded by 2% of max(width, η), and a sign-definite edge moves
at most 30% of its distance toward zero, so padding never turns a
sign-definite box into a crossing one.

No histogram, sampled lattice or error apportionment enters. The same box
gives the same rule on every deck, which makes rules cacheable and certifies a
low-mass state at the Fermi level exactly as it certifies a heavy one; a
mass-weighted fit is what loses such a state.

## 6. Error currencies and the delivered bound

| box | currency | certificate |
|---|---|---|
| crossing (Re d spans 0) | peak-relative | sup_box η_min \|Q(d) − 1/d\| ≤ ε, η_min = min Im d |
| sign-definite | relative | sup_box \|d\| \|Q(d) − 1/d\| ≤ ε |

ε is `sigma_quadrature_eps`, the only accuracy dial; the shared-pole W takes
it from its `sigma_w_accuracy` tier. Defaults are in the
[input reference](../input_reference.md).

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

`build_uniform_rule(box, ε)` discretizes 1/d = −i∫₀^∞ e^{itd} dt along a ray
t = s e^{−iθ}. The angle θ is scanned over the interval where every member
decays on the box, and the smallest numerical rank wins: symmetric crossing
boxes get real time; sign-definite boxes rotate toward imaginary time, the
Laplace family.

**Crossing box: linear in A/η.** 1/d is band-limited. Resolving the peak
needs time support T = ln(c/ε)/η_min, and covering an effective real width
W_eff at the Nyquist density needs N ≈ W_eff T/2π nodes. W_eff = 2m plus a
saturating share of the long side's excess over the short side m, because the
nodes that cover a long side leave the real axis and damp it. For a symmetric
box of half-width A,

$$
N\approx1.04\,\frac{A}{\eta_{\min}}\,\frac{\ln(0.086/\varepsilon)}{\pi}\approx2.2\,\frac{A}{\eta_{\min}}\quad(\varepsilon=10^{-4}),
$$

which is within a constant of the band-limit floor (bandwidth × horizon/π).
No construction removes the A/η law; the remaining node savings are in η, ε
and the window geometry. The builder does not search for the count:
`minimax.fixed_n_start.predict_nodes` sets it and `start_param` places the
nodes (live-band Nyquist density, Im s damping the wider real edge within the
off-ray cap, the last node at the amplitude floor). One variable-projection
Levenberg–Marquardt polish follows, then the certificate. A failure raises
the count by 10%, up to eight rungs; past that bracket the interpolatory
ray-rank rule is polished once and then accepted or refused.

**Sign-definite box: logarithmic.** 1/d there is a Braess–Hackbusch
exponential sum with N = O(log R · log(1/ε)), R the corner dynamic range. The
builder starts from the interpolatory rule at the ray rank (pivoted QR of the
ray family's SVD basis) and removes nodes, in batches while far above the
target. The survivors are re-solved by variable-projection LM on the sampled
residual. A removal is kept while the sup on a finer check cloud stays ≤ ε
and the cancellation ratio stays under its cap. The builder stops when no
removal is accepted.

Neither construction reads a clock, so the rule is a function of (box, ε)
only, and a rule that cannot be certified is refused in planning, before the
sweep starts.

## 8. Acceptance

The planner accepts a rule for a window only if all three hold:

1. **Certificate.** Every node and weight is finite and the certified sup
   error is ≤ ε in the box's currency.
2. **Runtime noise.** With a per-term relative perturbation ε_rt = 6·10⁻⁸,
   ε_rt · max_{d∈∂box} ρ(d) Σ_l |w_l e^{i t_l d}| ≤ 0.05 ε, where ρ = |d| on a
   sign-definite box and ρ = η_min on a crossing one. The noise mass is
   subharmonic, so its maximum is on the boundary, sampled at the rule's own
   horizon. Sign-definite rules are built under the cancellation cap that
   implies this bound.
3. **Factored growth.** Over the window's states and pole corners, neither
   separately factored exponential grows by more than e³⁰.

A failure is final. There is no retry at a tighter ε and no second quadrature
family, and the (window, τ) pair count is reported, never refused on.

## 9. Self-consistent maps

A multi-map QSGW run carries a fixed-quadrature session. The first two
planner calls (maps 0 and 1) use one-shot rules. The third freezes a rule set
on its own boxes, each padded:

- each real edge by the classification pad of the state that sets it,
  0.5 eV + 0.10 |E − μ|, never across the window's own selector bound;
- every pole extent by 10%;
- a sign-definite edge toward zero at most to 5% of its distance from zero;
- the zero-side edge of a tail window out to the selector's guaranteed gap
  0.7a, which covers a state that enters the tail on a later map.

Later maps reuse the frozen nodes while the current box is contained. Four
events change that:

- A window that escapes its box, changes currency, or did not exist at the
  freeze is refit alone.
- A factored-growth failure refits that window alone.
- A metal ↔ insulator flip reinitializes the set.
- A change of η or ε refuses.

The receipt names every refit window and its reason.

## 10. Cache, request scope and parallel planning

**Cache.** Rules are immutable certificates keyed by (box, ε, currency) and
authenticated by a digest. A lookup serves the smallest rule whose certified
box contains the request at the same ε and currency with noise amplification
under the cap. On a miss the build box widens only real edges farther than 3η
from zero (by 1% of max(width, η)) and the far imaginary edge (by 1%), so
nearby maps hit without widening a crossing rank.
`sigma_quadrature_cache_dir = auto` places the cache at
`<input_dir>/tmp/sigma_quadrature_rules`; `off` disables it; a relative path
resolves against the deck. The cache accelerates; it is never a second
correctness path, and a failed write only warns.

**Rule table.** Below the cache, every builder call is memoized in one
run-independent table, `$SCRATCH/.cache/lorrax/sigma_box_rules`. It is keyed
exactly by the snapped build box, ε, the currency, the κ cap, the rule schema
and the solver identity (the builder, the minimax sources, numerics backend, CPU model and
pinned BLAS threads). The builder reads no clock and pins its threads, so a
hit is the rule a cold build returns, bit for bit (claim 2737). A warm run is
therefore the cold run that wrote the table, and no run depends on which
other decks wrote it; the table never serves by containment. The first writer
of a key wins; an entry whose schema, key or digest does not authenticate is a
named miss and is replaced. `off` disables the table too.

**Request scope.** The shared-pole route scopes the cache to a subdirectory
keyed by the map's physical identity (energies, occupations and recipe, with
the SC map label stripped), η, ε and the pole census. Equal physical inputs
share rules across restarts and maps, and a changed spectrum never inherits
another map's plan. There is one scope per map: a sector Σ call scopes by the
union census of its CC, TT and CT_C models, so the four sector calls of a map
reuse each other's fits.

**Parallel planning.** Windows are fit round-robin across processes, and only
the small rules and receipts are gathered. Each window is then served the
smallest compatible rule of the whole plan, ranked by (node count, digest),
which is what a warm rerun would pick and does not depend on rank timing. A
refusal on one rank travels as data and raises on every rank.

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
  node is shared by one G build, convolution and projection per bracket.

A per-stage split of a node is a `jax.profiler` trace of the window executable.

## 12. Refusals

| refusal | fix |
|---|---|
| rule not certified at ε, or not finite (names window, box, kind) | a sign-preserving or split product window; never a looser `sigma_quadrature_eps` |
| runtime-noise bound above 0.05 ε | the same; the box is too ill-conditioned for its currency |
| factored log growth above 30 | the same |
| live pole with Re Ω ≤ 0 or Im Ω > 0, or a nonfinite residue | refit the pole model |
| a branch with no live states | the Σ band window has no band on that side; widen `number_bands_sigma` or `occupation_window_threshold` |
| η ≤ 0, ε ∉ (0, 1), edge factor < 0 | fix the deck |
| η or ε changed inside one SC session | one currency per run |
| shared-pole W with `sigma_quadrature_eps` unequal to its `sigma_w_accuracy` tier's tolerance | drop the key; the recipe owns ε |
