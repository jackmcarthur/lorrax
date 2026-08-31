# The Σ(ω) quadrature problem and the constraints on any method

Specification for the delivered-Σ planner: the problem it solves, and the rules
any method must obey. Implementation: `gw/mpa/delivered_windows.py`,
`minimax/roq_fit.py`. Surrounding machinery:
[Minimax quadrature](minimax-quadrature.md),
[Multipole frequency integration](THEORY_mpa_implementation.md).

## 1. Problem statement

`W` arrives as a causal fitted pole model (`Im Ω_p < 0`): poles `Ω_p`, residues
`B_p`. `G` arrives as energies `E_n` with occupations `f_n`. One causal branch
of the correlation self-energy on a requested frequency set `{ω_j}` is

```
Σ_b(ω_j) = Σ_{n,p} R_np / (ω_j − E_n − σ_b Ω_p),      σ_b = ±1.
```

**Find the cheapest quadrature delivering Σ_b(ω_j) to a stated accuracy on that
discrete support.** Cheapest is §3, accuracy §5. The `{ω_j}` occupy a window;
several disjoint windows may be requested, each with its own broadening.

## 2. Separability — the constraint that makes the method O(N³)

Replace the reciprocal by a short sum of exponentials in an auxiliary time,

```
1/d ≈ Σ_l w_l exp(i t_l d),        d = ω_j − E_n − σ_b Ω_p,
```

so that each atom factors exactly,

```
exp(i t d) = exp(i t ω_j) · exp(−i t E_n) · exp(−i t σ_b Ω_p).
```

The electronic and screening sums then form **independently**, meeting once per
node: `N_band · N_pole` work becomes `n_τ · (N_band + N_pole)`.

**Every basis atom must therefore be a pure exponential in the constituent
energies `(E ± Ω)` paired with `t_i`.** A basis function that does not factor
this way destroys the scaling whatever its accuracy. The constraint binds the
`E` and `Ω` dependence only: the external `ω_j` phase is a scalar coefficient
folded after the spatial sweep. A per-window broadening does not change this —
`η` enters through the pole positions, not the `ω_j` phase — provided each
window is planned as its own delivery. The same family has served `ω_j` well in
practice — convenience, not requirement.

## 3. Cost — count transforms, not tuples

The unit of cost is the **number of (window, τ) pairs**: each pays a Fourier
transform of the windowed Green's function `G_window,k(τ) → G_window,R(τ)` and
of the screened interaction `W_q(τ) → W_R(τ)`. Those transforms cost at least
as much as the band contractions they enable.

Two consequences; the second is easy to get wrong:

- **Node count is the objective.** Halving nodes halves the dominant cost.
- **Windowing buys less than it appears to.** Splitting a branch reduces only
  the *contraction* work; every window pays its own transforms at every node.
  Two windows at 30 nodes cost the same transforms as one at 60 whenever their
  nodes are distinct, and buy nothing unless the split genuinely reduces total
  pairs. Windows sharing a pole interval could in principle share `W_q(τ)` at a
  coincident `τ`; production does not deduplicate across windows, because their
  spectral selectors differ, so the pair count is the honest cost today. **Windowing is permitted but
  discouraged**, and a split must justify itself in total pairs.

## 4. Window geometry — products only

A window is an energy interval crossed with a pole interval, nothing else:

```
window = [E_min, E_max] × [Ω_min, Ω_max].
```

§2 forces this: `G_window(τ)` and `W_window(τ)` are built separately, so the
window must factor into an `E` condition and an `Ω` condition. A predicate
coupling the two — "include this E only when Ω is below some cut", any diagonal
cut in `E + Ω` — cannot be expressed by two independently windowed transforms
and is prohibited. The geometric consequence: the resonance `ω = E + Ω` is a
*diagonal* band, so a product window covering it also covers material away from
it, and its denominator radius is set by the window's own extent, not the
band's width.

## 5. What accuracy means, and what it is measured against

The objective is **quasiparticle energies**, not matrix elements of
`Σ(r, r′, ω)`. An objective function must weight errors by their effect on the
QP solution:

- Weight each `(n, p)` contribution by its **delivered mass** — residue
  magnitude times occupation factor times state amplitude — since that is what
  reaches the QP equation. Measured pole histograms of `G` and `W` tell the
  fitter where accuracy is worth buying; a rule uniformly accurate over a box
  wastes nodes on regions carrying no mass.
- Report the **achieved** error on an independent refined lattice, never a
  nominal target.
- Accept a rule only if it passes a roundoff-amplification budget:
  cancellation between large weights can meet an error target and still destroy
  the result at runtime precision.

## 6. Deterministic node placement

Prefer nodes selected from an eigenvalue or matrix-pencil problem on a subspace
of the measured integrand family over gradient descent, annealing, or any
initialisation-dependent search. Determinism makes plans reproducible across
process counts and machines; the plan receipt and the P-independence tests both
rely on it. Weights may be solved iteratively; that solve is convex-ish and
reproducible.

## 7. Cost laws a method must meet

Let `A` be the window's denominator **radius** — `max |Re d|` over its support,
so the crossing bandwidth is `2A` — and `η` the broadening (physical width plus
any deliberate regularisation).

- **Crossing windows** — the support contains `Re d = 0`. A method needing more
  than about one node per `η` of crossing bandwidth — `2A/η` — is **subpar and
  should be rejected**. This is the binding case and where essentially all the
  cost lives. The measured production law is `n ≈ 2.02·(A/η)` at `ε = 1e-4`,
  i.e. the method sits at the threshold, not inside it.
- **Non-crossing windows** — the support is sign-definite. Node count should
  grow like `log(A/η)` (a Laplace-type rule on the imaginary axis). Certified
  tables already serve these at 8–12 nodes; they deserve little attention.

Contour orientation follows from the split: sign-definite support wants the
imaginary axis, crossing support a contour rotated close to it but not onto it.
Real-time contours are the wrong operating point for crossing support and should
not be tried first.

## 8. Split the branches

Treat each causal branch separately: `ω ≥ 0` and `ω < 0` (the negative half
carried as `|ω|`), conduction versus valence, i.e. `E_c + Ω` versus `E_v − Ω`.
Their supports differ in sign structure and in where their mass sits, so one
rule spanning several branches is both harder to fit and harder to certify. A
signed `ω` request costs about twice a one-sided one; that is expected, not a
defect.

## 9. Plan for metals

The metallic case is the hard one and the design target.

- Occupations are fractional, so a state can contribute to both the particle and
  hole branch with complementary weights.
- Intraband transitions bring `E_n → E_F`, so denominators approach zero
  *inside* the requested window — exactly the terms a reciprocal fit finds
  hardest.
- Those same terms carry little weight, but by a different route than the one
  they enter: the occupation *difference* `f_{mk+q} − f_{nk}` suppresses the
  intraband contribution to `χ`, hence to the residues `B_p` of `W`, while `Σ`
  weights each state by its own occupation factor (`f` on a hole branch,
  `1 − f` on a particle branch). Either way the delivered mass near the
  resonance is small while the kernel is large. A method weighting by kernel magnitude rather than
  delivered mass spends its whole budget there for nothing; a method ignoring
  them gets the metallic QP shifts wrong.
- The regularisation `η` floors the denominators and makes the problem finite.
  It is a physical broadening choice, not a numerical patch, and one of only two
  dials the planner exposes; the other is the error target.

## 10. Practical notes for extending this

- Cost scales as `A/η` with `A` set by the window's own extent, so a delivery's
  price is fixed by its **width and its own broadening**, not by its distance
  from `E_F`. Several narrow windows at appropriate `η` are cheaper than one
  wide window at the finest `η`.
- Measure-adapted fitting beats measure-independent tables by 2–3× in nodes: a
  table must be uniformly accurate over a whole geometry, a fit only where the
  mass is. Tables remain the right tool where they are already cheap (§7,
  non-crossing) and as a fallback.
- A window's error allowance is apportioned by delivered mass. Crossing windows
  typically carry a few percent of the mass and therefore receive the *loosest*
  allowance; costing them at the deck-global target badly overstates a plan.
- Consolidating windows is an optimisation to attempt, never a commitment: it
  must reduce total nodes, must not raise the branch's absolute error, and must
  be abandonable when the global budget cannot then be met.
- Bound the search from the measured support, not from a user resource dial.
  Production estimates `2A/eta` nodes for each crossing window and 20 for each
  sign-definite window, doubles their sum, and applies a floor of 32. A method
  that cannot fit under that honest-cost ceiling refuses.
- Fit the cheap candidates once. Production tries the measure-adapted crossing
  rules plus lookup-served sign-definite rules first, then adds tighter shipped
  candidates only when the exact global selector cannot close the budget. It
  reuses both adapted fits and consolidation trials across the retry.
- Refusal is terminal. There is no direct state--pole evaluator, capped or
  otherwise, and no coupled selector hidden behind a small-problem threshold.
