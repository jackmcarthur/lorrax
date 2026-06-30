# Coarse windowing strategy for Laplace-transformed (\chi) and (\Sigma)

A self-contained note on why a **coarse six-window partition** is the right choice given the new scaling, with the final routing stated in the **raw-eigenvalue convention**.

## Motivation

We want minimax quadrature rules for two denominator approximation problems that arise when rewriting GW response and self-energy expressions in a separable time-domain form. The goal is to replace explicit sums over energy denominators by sums of products of propagator-like factors, so that the resulting contractions can be evaluated in (O(N^3)) form.

There are two qualitatively different situations.

### 1. Non-crossing windows: fixed-sign denominators

When the denominator does not change sign over the frequency range of interest, we approximate
[
\frac{1}{x}\approx \sum_{\ell} w_\ell e^{-t_\ell x},
\qquad x\in [x_{\min},x_{\max}],
]
or, after rescaling,
[
\frac{1}{x}\approx \sum_{\ell} w_\ell e^{-t_\ell x},\qquad x\in[1,R],
\quad R=\frac{E^{(\mathrm{bw})}}{E^{(\mathrm{gap})}}.
]

This is a standard exponential-sum approximation problem. For fixed target accuracy, the number of nodes grows only logarithmically in the dynamic range:
[
N_{\exp}\sim O(\log R),
]
and in practice the minimax construction is very well behaved.

### 2. Crossing windows: denominators that pass through zero

When the denominator can vanish inside the target frequency interval, the singularity at (x=0) must be regularized. We use a width (\xi) and replace (1/x) by
[
F(x;\xi)=\frac{1}{\xi}\int_0^\infty e^{-\tau-\tau^2/2}\sin(x\tau/\xi),d\tau.
]
This is smooth at (x=0), and for (|x|\gg \xi) it returns to the (1/x) asymptotic regime.

Defining the dimensionless variable (u=x/\xi), the crossing problem becomes approximation of
[
G(u)=\xi F(u\xi;\xi)
]
on a finite interval (u\in[0,A]), where
[
A=\frac{E^{(\mathrm{bw})}}{\xi}.
]
We approximate (G(u)) by a sine sum
[
G(u)\approx \sum_j w_j \sin(\tau_j u).
]

The important empirical fact for our current construction is that, at fixed accuracy, the required number of crossing nodes grows approximately **linearly** with the bandwidth:
[
N_{\sin}\sim O(A)=O!\left(\frac{E^{(\mathrm{bw})}}{\xi}\right),
]
rather than quadratically as in the original CTSP/HGL quadrature fit.

---

## Consequence for windowing

This scaling changes the windowing strategy.

In the original CTSP setting, the crossing quadrature was expensive enough that it made sense to partition the energy plane rather finely in order to keep each crossing window as narrow as possible. That logic weakens substantially once:

1. the non-crossing approximation is already cheap,
   [
   N_{\exp}\sim O(\log R),
   ]
2. the crossing approximation grows only linearly with bandwidth,
   [
   N_{\sin}\sim O(E^{(\mathrm{bw})}/\xi),
   ]
3. the total target quadrature budget is modest anyway (roughly (O(10^2)) terms),
4. and tighter partitioning introduces real overhead:

   * more bookkeeping,
   * more separate grids,
   * less quadrature reuse,
   * more transitions between exp and sine treatments,
   * and more opportunities for classification/pathology near window boundaries.

So once the crossing rule is no longer disastrously expensive, the right principle is:

> **Use the coarsest window decomposition that cleanly separates always-crossing, partially-crossing, and always-noncrossing regions.**

That is exactly what the six-window scheme does.

It captures the only geometric distinction that matters for the sum-type denominators:

* one block is always close enough to resonance that it should always use the sine quadrature,
* two blocks are mixed and require sine only on the upper part of the frequency range,
* the remaining blocks are safely noncrossing and should always use exponential quadrature.

Anything tighter than that is usually not worth it unless the density of states is very nonuniform or one is optimizing constants at a level below the quadrature/model error.

---

## Frequency broadening and the handoff margin

The same scale (\xi) that regularizes the crossing kernel also tells us where the kernel has already relaxed back to its (1/x) asymptotic form. We therefore define a numerical handoff margin
[
\zeta_{\mathrm{edge}}\sim \xi
\quad\text{or more conservatively}\quad
\zeta_{\mathrm{edge}}\sim 2\xi.
]

The rule is:

* use the **sine** quadrature whenever the denominator can come within (\zeta_{\mathrm{edge}}) of zero;
* use the **exponential** quadrature only outside that buffered region.

This avoids switching to the fixed-sign approximation while the denominator is still in the regularized bump regime.

---

## Conventions

Use the **raw-eigenvalue convention** throughout:

* (E_F=0),
* (E_c\ge 0) for conduction states,
* (E_v\le 0) for valence states,
* (\Omega_p\ge 0) for plasmon poles,
* (\omega\in[0,\Omega]), with (\Omega=\omega_{\max}).

The four denominator families are then

[
\chi:\quad \omega-(E_c-E_v),\qquad \omega+(E_c-E_v),
]
[
\Sigma:\quad \omega-(E_c+\Omega_p),\qquad \omega-(E_v-\Omega_p).
]

---

## Which families actually need crossing treatment?

### 1. (\chi) resonant term

[
\omega-(E_c-E_v).
]
Since (E_c-E_v>0), this can cross zero on the positive-frequency axis.

**Needs mixed sine/exp treatment.**

### 2. (\chi) antiresonant term

[
\omega+(E_c-E_v).
]
For (\omega\ge 0), this is always strictly positive.

**Exp-only everywhere.**

### 3. (\Sigma_c) term

[
\omega-(E_c+\Omega_p).
]
Since (E_c+\Omega_p\ge 0), this can also cross zero on the positive-frequency axis.

**Needs mixed sine/exp treatment.**

### 4. (\Sigma_v) term

[
\omega-(E_v-\Omega_p)=\omega-E_v+\Omega_p.
]
Because (E_v\le 0), this is always strictly positive for (\omega\ge 0).

**Exp-only everywhere.**

So in practice:

* **mixed families:**
  [
  \omega-(E_c-E_v),\qquad \omega-(E_c+\Omega_p),
  ]
* **exp-only families:**
  [
  \omega+(E_c-E_v),\qquad \omega-(E_v-\Omega_p).
  ]

---

## Generic sum-type geometry

The two mixed families both reduce to the same geometry:
[
\omega-(A+B),\qquad A\ge 0,; B\ge 0.
]

Use:

* (A=E_c,; B=-E_v) for (\chi),
* (A=E_c,; B=\Omega_p) for (\Sigma_c).

For a window pair
[
A\in[a_-,a_+],\qquad B\in[b_-,b_+],
]
the summed transition interval is
[
S\in[S_-,S_+]=[a_-+b_-,,a_++b_+].
]

That window should use the sine quadrature on
[
\omega\in [0,\Omega]\cap [S_- - \zeta_{\mathrm{edge}},, S_+ + \zeta_{\mathrm{edge}}],
]
and the exponential quadrature on the complement inside ([0,\Omega]).

---

## Final six-window partition

Define the buffered scale
[
\Omega_b=\Omega+\zeta_{\mathrm{edge}}.
]

Partition each positive axis into
[
L=[0,\Omega_b/2],\qquad
M=[\Omega_b/2,,3\Omega_b/2],\qquad
H=[3\Omega_b/2,E_{\max}].
]

Then use the following six blocks.

---

### Window 1: `LL`

[
A\in[0,\Omega_b/2],\qquad B\in[0,\Omega_b/2].
]

Then
[
A+B\in[0,\Omega_b].
]

So the entire target interval lies in the buffered crossing range.

* **sine:** (\omega\in[0,\Omega])
* **exp:** none

This is the always-crossing block.

---

### Window 2: `ML`

[
A\in[\Omega_b/2,,3\Omega_b/2],\qquad B\in[0,\Omega_b/2].
]

Then
[
A+B\in[\Omega_b/2,,2\Omega_b].
]

Hence the lower part of the frequency interval is noncrossing, while the upper part must use sine.

* **sine:**
  [
  \omega\in\left[\max!\left(0,\frac{\Omega-\zeta_{\mathrm{edge}}}{2}\right),,\Omega\right]
  ]
* **exp:**
  [
  \omega\in\left[0,\max!\left(0,\frac{\Omega-\zeta_{\mathrm{edge}}}{2}\right)\right)
  ]

This is a mixed block.

---

### Window 3: `LM`

[
A\in[0,\Omega_b/2],\qquad B\in[\Omega_b/2,,3\Omega_b/2].
]

Same transition range as `ML`, so the same routing applies:

* **sine:**
  [
  \omega\in\left[\max!\left(0,\frac{\Omega-\zeta_{\mathrm{edge}}}{2}\right),,\Omega\right]
  ]
* **exp:**
  [
  \omega\in\left[0,\max!\left(0,\frac{\Omega-\zeta_{\mathrm{edge}}}{2}\right)\right)
  ]

Also a mixed block.

---

### Window 4: `HL`

[
A\in[3\Omega_b/2,,A_{\max}],\qquad B\in[0,\Omega_b/2].
]

Then
[
A+B\ge 3\Omega_b/2.
]

Even after subtracting the handoff margin, this remains above (\Omega), so the denominator never comes near zero on the target interval.

* **sine:** none
* **exp:** (\omega\in[0,\Omega])

---

### Window 5: `LH`

[
A\in[0,\Omega_b/2],\qquad B\in[3\Omega_b/2,,B_{\max}].
]

By the same logic:

* **sine:** none
* **exp:** (\omega\in[0,\Omega])

---

### Window 6: `HH_union`

This is the union of all remaining blocks with both coordinates at least (\Omega_b/2):
[
A\in[\Omega_b/2,A_{\max}],\qquad B\in[\Omega_b/2,B_{\max}].
]

Equivalently, it is the union of `MM`, `MH`, `HM`, and `HH`.

Here
[
A+B\ge \Omega_b=\Omega+\zeta_{\mathrm{edge}},
]
so it is outside the buffered crossing region throughout the interior of the target interval.

* **sine:** none
* **exp:** (\omega\in[0,\Omega])

In implementation, this can either be kept as one logical block or split into four rectangular subblocks that all follow the same exp-only code path.

---

## How to apply this to each quantity

### (\chi) resonant

Use the six-window scheme with
[
A=E_c,\qquad B=-E_v.
]

So the axes are:

* conduction energy above (E_F),
* hole energy below (E_F), written as (-E_v\ge 0).

### (\Sigma_c)

Use the same six-window scheme with
[
A=E_c,\qquad B=\Omega_p.
]

### (\chi) antiresonant

No crossing logic needed.

Use exponential quadrature only.

### (\Sigma_v)

No crossing logic needed on the positive-frequency axis.

Use exponential quadrature only.

---

## Why this coarse partition is enough

The six-window scheme is preferred here for three reasons.

### 1. The crossing cost is only linear in bandwidth

Once the sine quadrature scales as
[
N_{\sin}\sim O(E^{(\mathrm{bw})}/\xi),
]
the penalty for keeping a crossing block somewhat larger is modest.

This is the main reason tight CTSP-style tiling is no longer necessary.

### 2. The noncrossing cost is already very cheap

The fixed-sign approximation scales only like
[
N_{\exp}\sim O(\log R),
]
so there is little to be gained by subdividing already-safe regions more finely.

### 3. Over-partitioning adds complexity without much payoff

A very fine windowing scheme means:

* more block definitions,
* more special cases,
* more different quadrature tables,
* more switching between quadrature types,
* less reuse of precomputed nodes,
* and more opportunities for numerical edge issues.

With the present scaling, those costs dominate the small constant-factor savings from squeezing each crossing window as tightly as possible.

So the guiding design principle is:

> **Keep only the minimal partition needed to separate fully crossing, partially crossing, and fully noncrossing regions.**

That is exactly what the six-window scheme achieves.

---

## Practical implementation summary

A clean implementation can proceed as follows:

1. Fix
   [
   \Omega,\qquad \xi,\qquad \zeta_{\mathrm{edge}}\in{\xi,2\xi}.
   ]

2. Route denominator families:

   * `chi_resonant` (\to) six-window mixed treatment
   * `chi_antiresonant` (\to) exp-only
   * `sigma_c` (\to) six-window mixed treatment
   * `sigma_v` (\to) exp-only

3. For mixed families, map to the generic (\omega-(A+B)) form:

   * `chi_resonant`: (A=E_c,; B=-E_v)
   * `sigma_c`: (A=E_c,; B=\Omega_p)

4. Build the six windows from
   [
   \Omega_b=\Omega+\zeta_{\mathrm{edge}}.
   ]

5. Use the fixed routing table:

   * `LL`: sine for all (\omega)
   * `ML`, `LM`: sine only on upper (\omega), exp below
   * `HL`, `LH`, `HH_union`: exp for all (\omega)

6. Precompute:

   * sine quadratures indexed by
     [
     A=\frac{E^{(\mathrm{bw})}}{\xi},
     ]
   * exponential quadratures indexed by
     [
     R=\frac{E^{(\mathrm{bw})}}{E^{(\mathrm{gap})}}.
     ]

This gives a simple, robust, and near-minimal scheme consistent with the improved quadrature scaling.
