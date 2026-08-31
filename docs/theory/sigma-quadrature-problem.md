# The Σ(ω) quadrature problem, and the constraints a method must satisfy

This page states the problem the delivered-Σ planner solves and the rules any
proposed method must obey. It is the specification; the implementation lives in
`gw/mpa/delivered_windows.py` and `minimax/roq_fit.py`, and the surrounding
machinery in [Minimax quadrature](minimax-quadrature.md) and
[Multipole frequency integration](THEORY_mpa_implementation.md).

## 1. Problem statement

The screened interaction is supplied as a fitted pole model: complex poles
`Ω_p` with residues `B_p`, causal (`Im Ω_p < 0`). The Green's function is
supplied as electronic energies `E_n` with occupations `f_n`. One causal
branch of the correlation self-energy on a requested discrete frequency set
`{ω_j}` is

```
Σ_b(ω_j) = Σ_{n,p} R_np / (ω_j − E_n − σ_b Ω_p),      σ_b = ±1.
```

**Find the cheapest quadrature that delivers Σ_b(ω_j) to a stated accuracy on
that discrete support.** Cheapest is defined in §3. Accuracy is defined in §5.
The requested `{ω_j}` occupy a window; several disjoint windows may be
requested, each with its own broadening.

## 2. Separability — the constraint that makes the method O(N³)

Replace the reciprocal by a short sum of exponentials in an auxiliary time,

```
1/d ≈ Σ_l w_l exp(i t_l d),        d = ω_j − E_n − σ_b Ω_p,
```

so that each atom factors exactly,

```
exp(i t d) = exp(i t ω_j) · exp(−i t E_n) · exp(−i t σ_b Ω_p).
```

The electronic and screening sums then form **independently** and meet once per
node, turning `N_band · N_pole` work into `n_τ · (N_band + N_pole)`.

**Therefore: every basis atom must be a pure exponential in the constituent
energies `(E ± Ω)` paired with `t_i`.** A basis function that does not factor
this way destroys the scaling regardless of its accuracy. This applies to the
`E` and `Ω` dependence only — the external `ω_j` phase is a scalar coefficient
folded after the spatial sweep, so it is *not* subject to this constraint. In
practice the same exponential family has served the `ω_j` dependence well, but
that is an empirical convenience, not a requirement.

## 3. Cost — count transforms, not tuples

The unit of cost is the **number of (window, τ) pairs**, because each one pays a
Fourier transform of the windowed Green's function `G_window,k(τ) → G_window,R(τ)`
and of the screened interaction `W_q(τ) → W_R(τ)`. Those transforms are as
expensive as, or more expensive than, the band contractions they enable.

Two consequences follow, and the second is easy to get wrong:

- **Node count is the objective.** Halving nodes halves the dominant cost.
- **Windowing is less attractive than it first appears.** Splitting a branch
  into more windows reduces only the *contraction* work; every window pays its
  own transforms at every one of its nodes. A plan of two windows at 30 nodes
  costs the same transforms as one window at 60 and buys nothing unless the
  split genuinely reduces total nodes. **Windowing is permitted but
  discouraged**, and a split must justify itself in total pairs.

## 4. Window geometry — products only

If a branch is windowed, each window is specified by an energy interval and a
pole interval and nothing else:

```
window = [E_min, E_max] × [Ω_min, Ω_max].
```

This is forced by §2: `G_window(τ)` and `W_window(τ)` are built separately, so
the window must factor into an `E` condition and an `Ω` condition. A predicate
coupling the two — "include this E only when Ω is below some cut", any
diagonal cut in `E + Ω` — cannot be expressed by two independently windowed
transforms and is prohibited. Note the geometric consequence: the resonance
`ω = E + Ω` is a *diagonal* band, so a product window that covers it also
covers material away from it, and its denominator radius is set by the window's
own extent rather than by the band's width.

## 5. What accuracy means, and what it is measured against

The objective is **quasiparticle energies**, not matrix elements of
`Σ(r, r′, ω)`. An objective function must therefore weight errors by their
effect on the QP solution:

- Weight each `(n, p)` contribution by its **delivered mass** — the residue
  magnitude times the occupation factor times the state amplitude — because
  that is what reaches the QP equation. Pole histograms of `G` and `W` are the
  measured input that tells the fitter where accuracy is worth buying; a rule
  uniformly accurate over a box wastes nodes on regions carrying no mass.
- Report the **achieved** error on an independent refined lattice, never a
  nominal target.
- Accept a rule only if it also passes a roundoff-amplification budget:
  cancellation between large weights can meet an error target while destroying
  the result at runtime precision.

## 6. Node placement should be deterministic

Prefer constructions whose nodes come from an eigenvalue or matrix-pencil
problem — a subspace of the measured integrand family, from which nodes are
selected — over gradient descent, annealing, or any search whose result depends
on initialisation. Determinism makes plans reproducible across process counts
and machines, which the plan receipt and the P-independence tests both rely on.
Weights may be solved iteratively; that solve is convex-ish and reproducible.

## 7. Cost laws a method must meet

Let `A` be the window's denominator radius and `η` the broadening (physical
width plus any deliberate regularisation).

- **Crossing windows** — the support contains `Re d = 0`. A method requiring
  more than about `A/η` nodes to resolve the crossing is **subpar and should be
  rejected**. This is the binding case; it is where essentially all the cost
  lives.
- **Non-crossing windows** — the support is sign-definite. Node count should
  grow like `log(A/η)` (a Laplace-type rule on the imaginary axis). These are
  cheap, they are already served by certified tables at 8–12 nodes, and they do
  not deserve much attention.

The contour orientation follows from this split: sign-definite support wants
the imaginary axis; crossing support wants a contour rotated close to it but
not on it. Real-time contours are the wrong operating point for crossing
support and should not be the first thing tried.

## 8. Split the branches

Treat each causal branch separately: `ω ≥ 0` and `ω < 0` (the negative half is
carried as `|ω|`), and conduction versus valence, i.e. `E_c + Ω` versus
`E_v − Ω`. Their supports differ in sign structure and in where their mass
sits, so one rule spanning several branches is both harder to fit and harder to
certify. A signed `ω` request therefore costs about twice a one-sided one; that
is expected, not a defect.

## 9. Plan for metals

The metallic case is the hard one and the design target.

- Occupations are fractional, so a state can contribute to both the particle
  and hole branch with complementary weights.
- Intraband transitions bring `E_n → E_F` and therefore denominators
  approaching zero *inside* the requested window. These are exactly the terms a
  reciprocal fit finds hardest.
- They are simultaneously **down-weighted by small occupation differences**
  `f_{mk+q} − f_{nk}`, so the delivered mass near the resonance is small even
  though the kernel is large. A method that weights by kernel magnitude rather
  than delivered mass will spend its whole budget there for nothing; a method
  that ignores them entirely will get the metallic QP shifts wrong.
- The regularisation `η` floors the denominators and makes the problem finite.
  It is a physical broadening choice, not a numerical patch, and it is one of
  only two dials the planner exposes (the other is the error target).

## 10. Practical notes for anyone extending this

- Cost scales as `A/η`, and `A` is set by a window's own extent, so a delivery's
  price is fixed by its **width and its own broadening**, not by its distance
  from `E_F`. Several narrow windows with appropriate `η` are cheaper than one
  wide window at the finest `η`.
- Measure-adapted fitting beats measure-independent tables by 2–3× in nodes,
  because a table must be uniformly accurate over a whole geometry while a fit
  need only be accurate where the mass is. Tables remain the right tool where
  they are already cheap (§7, non-crossing) and as a fallback.
- A window's error allowance is apportioned by delivered mass. Crossing windows
  typically carry a few percent of the mass and therefore receive the *loosest*
  allowance — costing them at the deck-global target overstates a plan badly.
- Consolidating windows is an optimisation to attempt, never a commitment: it
  must reduce total nodes, must not raise the branch's absolute error, and must
  be abandonable when the global budget cannot then be met.
