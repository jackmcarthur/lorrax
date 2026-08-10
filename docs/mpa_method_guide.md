# The multipole method in LORRAX: a guide for someone starting from zero

*Read this before you touch anything under `src/gw/mpa/`. It is a lecture, not a
ledger: it builds the method from the physics it approximates, and at each step
it names the defect that taught us the current shape of the code. It is current
as of 2026-08-09 on `integration/mpa-table-2026-08-09`, and where it disagrees
with an older document it supersedes it.*

Two companions sit behind it. `THEORY_mpa_implementation.md` (service-phase
workspace) is the long reference — every equation with its published provenance,
the memory architecture, the error budget. `MPA_THREE_WAY_TABLE_2026-08-09.md` is
the campaign record: what was run, on which bytes, and what it measured. This
guide is the shortest path to being able to read either, and it carries the
several things that postdate both.

---

## 1. What the multipole approximation is for

The GW self-energy convolves the Green's function with the screened Coulomb
interaction, and the screened interaction depends on frequency. Every practical
scheme is a decision about how to carry that dependence. Plasmon-pole models —
Godby–Needs, Hybertsen–Louie — carry it with a single effective pole per matrix
element, fitted to two samples. That is cheap and often good, but it is a
one-parameter guess about a structured function, and there is no knob inside it
that converges.

The multipole approximation is the same idea with the pole count as the
convergence parameter. Split the screened interaction into its static bare part
and the rest, `W(z) = v + W_c(z)`, and represent the frequency-dependent part by
a small sum of complex poles, fitted independently for each matrix element at
each transferred momentum:

$$
W_c(z) \;=\; \sum_{p=1}^{n_p} B_p\left[\frac{1}{z-\Omega_p}-\frac{1}{z+\Omega_p}\right]
        \;=\; \sum_{p=1}^{n_p} \frac{2\,\Omega_p B_p}{z^2-\Omega_p^2},
\qquad
\Omega_p = a_p - i\Gamma_p,\quad a_p>0,\ \Gamma_p>0 .
$$

Three things are packed into that line, and each governs a later section.

**`W_c` is even in `z`.** Its warrant is the Lehmann representation of the
independent-particle polarizability, which is exactly a sum of
`2R_n\Omega_n/(\omega^2-\Omega_n^2)` terms; the multipole ansatz is that object
continued to `n_p ≪ N_T` effective poles, each standing for the envelope of a
bundle of transitions. Because it is even, the natural interpolation variable is
`x = z²`, not `z`. Section 3 is a Padé problem in `x`.

**The widths are physics.** `Γ_p` is the spectral width of the bundle a pole
represents, and the consumer reads it literally. The time-domain form the
self-energy integrates is `W_c(τ) = Σ_p B_p e^{-i a_p τ} e^{-Γ_p τ}`, so
`Re Ω_p > 0` and `Im Ω_p < 0` are structural: a pole leaking into the upper half
plane enters as `exp(+|Im Ω_p|τ)` and the integral diverges. That is why §3 has
explicit sign guards rather than trust in the linear algebra, and why there is no
broadening left to choose downstream — each pole carries its own, and adding an
`η` on top would count the same smearing twice.

**We fit `W_c`, never the polarizability.** Published implementations interpolate
`X` or `Y = ε⁻¹ − I` and form `W_c = Yv` afterwards. `W_c = vXv` carries Coulomb
factors that are strongly `q`-dependent and singular at the head, and multiplying
two independently fitted pole sets by two different `q`-singular numbers and
adding does not produce a pole set at all. Fitting the object the self-energy
consumes keeps pole structure and residue structure in one representation.

### 1.1 `W_c = W − v`, and never `W`

This is the costliest defect the project has hit, and the reason the store format
now has a declaration in it.

The first production store was written with the Dyson solve's output — the
**full** screened interaction — in the slot the fit reads. Nothing in the format
said which object the bytes were, so nothing could refuse them. The fit ran, the
poles came out, Σ came out finite and smooth, and the `n_p = 1` bridge against
Godby–Needs missed by **−120 143.93 meV**. Rebuilt on `W − v`, the same bridge
reads **−107.24 meV**.

Two measurements discriminated it before a byte was rewritten, and both are worth
keeping as diagnostics: `|v|` was 104–119 % of `|W|` at the probe frequency
(median ratio 1.386 — the bare interaction was not a correction to what was being
fitted, it was most of it), and the store's own head channel returned
`ε⁻¹ ≈ 1.01–1.03` at the largest `|z|`, where a correlation part must go to zero.

The fix adds no arithmetic. `src/file_io/mpa_store.py` gives the W(ω) store and
the fit store a `screening_content` declaration, **required at birth** by the
writers, and `require_correlation_part` is called at three consumer seams: the
fit driver, the MPA Σ pass, and the partial-cube route. An undeclared store is
refused rather than guessed at, because the first-light production store holds
full `W` while every synthetic fixture in the tree holds `W_c` — neither default
is even usually right, and the wrong one costs 130 eV with no other symptom. Red
twins: `tests/test_mpa_screening_content.py`.

The same defect had a second symptom, diagnosed for a day as an independent
problem: pole 7's Laplace bucket demanded 1312 quadrature leaves against a ceiling
of 512 and was refused. Those were not plasmon energies, they were the bare
Coulomb tail being fitted along with the screening. Refit on `W_c`, pole 7's
`Re Ω` maximum fell from 12 902 eV to 142 eV and its width maximum from 8031 eV
to 100 eV, and the same bucket over the same 80 million modes now splits into
**47** leaves. One finding, two symptoms.

### 1.2 The companion declaration: energy units

The fit store also declares its pole-axis unit (`mpa_fit_energy_unit`), after the
same kind of defect. The fit is solved against the W(ω) store's abscissae, which
are stamped Hartree; the Σ pass fed `Re Ω_p` straight in beside Rydberg band
energies. Every pole entered Σ at half its energy, and the width split and the
Laplace buckets were mis-sized from the same numbers.

No internal gate can see that, and the reason generalises: the multipole model is
invariant under rescaling `z`, `Ω` and `B` together, so a self-consistent unit
error is invisible from inside. It took an external oracle — the `n_p = 1` head
pole reads 18.118 eV as Hartree against BerkeleyGW's 18.009 eV, and would read
9.06 eV as Rydberg against a 16.7 eV measured plasmon.

The shape of the fix is the shape all of these take: **declared by the writer,
converted exactly once by the readers, refused loud in between.** The driver does
not type the unit; it inherits `header["omega_units"]` from the W store whose
samples it just fitted, the only seam that knows it for free. The readers
multiply `Ω_p` *and* `B_p` — both, because `B` carries one power of the frequency
unit, so the conversion leaves `W_c` invariant. `sigma_pass` has no unit logic of
its own, and that absence is what a test checks.

---

## 2. Sampling: where `W_c` is evaluated, and what those samples can see

The poles and residues are determined by interpolation: `W_c` is evaluated at
exactly `2n_p` complex frequencies and the model must reproduce them. Because
`n_p ≪ N_T`, where those points sit is not a detail.

### 2.1 The double-parallel protocol

Two lines parallel to the real axis: a near line at `ϖ₁ = 0.1 Ha` and a far line
at `ϖ₂ = 1 Ha`, each carrying `n_p` real abscissae. The near line resolves the
structure of `W_c` where its poles are; the far line sees a function smooth
enough for a few poles and carries the envelope. The near line's first sample is
exactly `z = 0` for insulators, `z = i·10⁻⁵ Ha` for metals — a stability shift,
never a physical width.

The real abscissae are a semi-homogeneous partition in powers of two,
`ω_n = f_n^α ω_m`, with `α = 1` for insulators and `α = 2` where the response has
structure near the origin. The published table stops at `n_p = 7`; LORRAX's rule
for continuing it (bisect the origin-adjacent interval down to a floor of `1/8`,
then bisect the widest remaining interval) reproduces all seven published sets
exactly and is exposed as `ladder_floor_octaves`. It is inert at or below
`n_p = 9`, which covers every silicon schedule.

Two properties matter downstream. The partition is built in exact rational
arithmetic and only ever *inserts* points, so growing `n_p` adds samples and
never moves them — an `n_p` scan is a scan of one thing. And the `n_p = 1`
insulator grid is `(0, 2i\,\mathrm{Ry})`, *exactly* the legacy Godby–Needs probe
pair. That identity is not a curiosity; it is the bridge gate of §6.

Code: `src/gw/mpa/sampling.py` for the grid, `src/gw/mpa/sample_plan.py` for each
sample's analytic character and route.

### 2.2 What the samples cannot see

> **Identifiability caution.** Samples taken off the real axis at height `h`
> carry almost no information about poles whose distance from the axis is much
> smaller than `h`. Two models agreeing on such samples to any tolerance you like
> may differ arbitrarily on the real axis, which is where Σ reads them.
> **Off-axis samples cannot veto a near-real pole.**

This is a measurement, not a worry, and it is the sharpest lesson of the
2026-08-09 campaign. The `q → 0` head was fitted with causality enforced inside
the solve; it reproduced its sixteen stored samples to `2.0 × 10⁻⁹` and the f-sum
rule to four digits — and the self-energy it produced moved the occupied state at
Γ by **+406 meV** against 0.7 meV at X, collapsing the indirect gap to 1.03 eV.
The mechanism is visible in the poles: that fit placed its narrowest pole at
`Γ = 0.027 eV`, while the sixteen samples sit at `z = 0` and on lines at
`Im z = 1.36 eV` and `27.2 eV`, where a pole that close to the axis is perfectly
smooth. Two head models agreed at sixteen points to nine digits and differed by
400 meV in Σ.

The corollary is a design rule, restated in §4.4 where it binds: a gate that
scores sample reproduction certifies *interpolation*, not *analytic
continuation*.

The limiting case is worth having explicitly. On the **pure imaginary axis** a
multipole fit returns `Γ_p = 0.000` for every pole at every `n_p`, with held-out
errors of 1 to 53. `W_c(iy)` carries no phase information about widths and the
fit correctly reports that it has none. Imaginary-axis samples are cheap and they
pin magnitude and tail; they cannot supply widths.

### 2.3 What the geometry probe measured

The owner asked whether two horizontal lines are the right shape, and proposed an
arc whose height rises with frequency plus far-real anchors past the top
transition. That was run at matched sample count on the big-continuum silicon
deck with the protocol as control (`SAMPLING_GEOMETRY_PROBE.md`):

* **Far-real anchors are the best trade in the study** — two orders of magnitude
  on the noncrossing tail for three samples, on both build routes, and they leave
  the damped-τ family entirely so they are nearly free there too.
* **The arc wins where Σ reads `W`** by a factor of 1.8 on the near-real strip,
  and reaches at `n_p = 6` an accuracy the control does not reach at `n_p = 8`.
* **The control keeps its advertised property** — pole-position stability under
  changes in pole count — by two to three and a half. That matters only if
  something downstream reads individual `Ω_p` rather than their sum.
* **A near-real line alone is the worst geometry tested**, confirmed from a second
  source: BerkeleyGW's own contour-deformation line, fitted with our kernel,
  wanders the plasmon from 41 eV to 9 eV as `n_p` grows.
* **Padé conditioning ranks the geometries in nearly the reverse order of their
  accuracy**, so it is not a figure of merit for choosing one.
* **The width-blindness worry is retired.** There is no sampling-height floor on
  the narrowest resolvable pole; first light's fitted widths of 1.6–2.7 eV at
  small `|q|` are the analytic structure of `W`, not artefacts.

Cost prices geometry differently on the two build routes: on the exact resolvent
route every sample is a scalar reweighting and geometry is free, while on the
damped-τ sweep the control is 6.5× cheaper in nodes because sixteen samples share
two lines. The recommendation — an open owner row, not a landed change — is the
conservative one: keep the double-parallel grid and move two samples to the real
axis beyond `ω_m`.

---

## 3. Fitting: from `2n_p` numbers to `n_p` causal poles

### 3.1 The solve

Since `W_c` is even, this is a Padé problem in `x = z²`: find `P` of degree
`n_p − 1` and monic `Q` of degree `n_p` matching the samples. Cross-multiplying
linearises it into a square `2n_p × 2n_p` system, solved on `\hat x = x/x_{\max}`
— not cosmetic, since the raw Vandermonde over a span of several rydberg squared
is hopeless by `n_p ≈ 8`.

Three decisions in `src/gw/mpa/pade_fit.py` are load-bearing. *Rows are
equilibrated to unit two-norm before the square solve, and never in the residue
system* — row scaling cannot change a square system's exact solution, only which
one finite precision finds, and the sampled `W_c` spans decades between the two
lines; applying it to the overdetermined residue system would silently reweight
the objective and change what is being fitted. *The solve is an explicit
truncated-SVD pseudo-inverse, not a library least-squares call* — the singular
values reach the diagnostics without a second factorisation, and the output
signature carries no data-dependent rank, which is what would break batching.
*Roots come from a companion matrix, residues from a second solve* at fixed poles
using **all** `2n_p` samples, both lines; pruned poles are handled by zeroing
their columns, whereupon the pseudo-inverse returns exactly zero for them as the
minimum-norm solution, so pruning needs no shape change and the kernel stays
`vmap`-able over a leading element axis.

### 3.2 The four guards, and the quadrant algebra

The linear algebra knows no physics and will return poles in the wrong half
plane. Four guards act in fixed order on `b = Ω²`, which is what makes them clean:

1. **Wrong-branch reflection.** `Re b < 0` is exactly `Γ > a`, a pole further
   from the real axis than from the imaginary one. Repair `b ← −\bar b`, which
   flips `Re b` and preserves `Im b`, so time ordering survives.
2. **Forced time ordering.** With `Re Ω ≥ 0`, time ordering is exactly
   `Im b ≤ 0`. Repair `b ← \bar b`, which flips `Im b` and preserves `Re b`, so
   it cannot undo guard 1.
3. **Coincident-pole pruning.** Two poles within a tolerance are one pole fitted
   twice; the later in canonical sort order is dropped.
4. **Out-of-range pruning.** `Re Ω > 0`; `|Ω|` above a numerical-zero floor;
   `Re Ω` below a multiple of the sampled span (above it the fit extrapolates and
   the residue is unconstrained); and
   `|Im Ω| ≤ \texttt{width\_ratio\_max}·Re Ω` at `width_ratio_max = 1`.

The first two are precisely the two conjugations carrying `b` into the closed
fourth quadrant, after which the principal square root lands in
`arg Ω ∈ [−π/4, 0]` with no further branch logic anywhere in the tree. The fourth
guard's constant reappears in §5.3 as the ceiling of the quadrature envelope —
the same number, not two that happen to agree.

**Whenever any guard fires the residues are re-solved at the corrected poles.**
Guards 1–2 move `b` so the pre-guard residues are stale; guards 3–4 change which
columns exist so they are over-complete. The refit is unconditional in the graph
rather than a data-dependent branch, so it survives batching. A suppression
switch exists for exactly one purpose: with the refit off the returned poles are
bit-identical and the residues differ by more than `10⁻⁶`. That is the red twin,
not a production setting.

### 3.3 Fit-then-reflect is not free, and the head measured the price

Guard 2 *repairs* an acausal pole after the fact. It is legitimate — an
upper-half-plane pole makes the τ integral diverge, so it cannot ship — but
reflecting a pole moves it off the point the interpolation chose, and the residue
refit cannot fully recover.

On the head channel this was measured three ways. The shipped fit misses its own
sixteen samples by `4.567 × 10⁻⁴`. Re-solving every identical stage in x86-64
extended precision moves that by a factor of **1.000**, so it is not
conditioning. With the guards off the same machinery interpolates at
`3.4 × 10⁻¹⁵` **in ordinary double precision**, so the eight-pole form can
represent this channel exactly and double precision suffices to find it. The
unconstrained fit places 2 of 8 poles at `Im Ω > 0`; guard 2 reflects them, the
refit re-solves, and what is left over is exactly the observed miss — a factor of
`2.3 × 10¹²` above the raw residual.

The alternative is causality **inside** the optimisation: parameterise
`Im Ω_p = −exp(s_p)` so every iterate is time-ordered, and make the residues not
free parameters but the exact least-squares solution at the current poles
(variable projection). That reaches `2.019 × 10⁻⁹` on the same channel, causality
checked by value on every pole. **It is not shipped**, and §4.4 says why. The
general lesson stands independently of that outcome: *constraining inside the
solve is a different and better-conditioned problem than repairing after it, and
the difference is measurable.*

### 3.4 The width floor: `Γ ≥ ξ`, and why it is a modelling statement

The Σ stage carries a threshold `ξ`, the two-point path's own crossing
regularisation width. **A pole whose fitted width falls below `ξ` is routed
through the two-point crossing machinery, at that `ξ`, as a real pole.**

The threshold is `ξ` exactly, and the argument is about information rather than
cost. On the two-point path `ξ` is a *broadening*: the crossing quadrature fits
`1/u` smeared over a width `ξ`, and the conditioning floor engages on every
default run and raises `ξ` from the requested 0.25 eV to 0.476 eV. A pole at
`Γ = 4 × 10⁻⁵ eV` is four orders below that; convolved with the same smearing,
`1/(u + iΓ)` and `1/(u + i0)` differ only inside a region of width `Γ` while the
kernel integrates over a region `ξ ≫ Γ` wide. Godby–Needs' crossing treatment
*is* the `Γ → 0` limit of the complex one with the regularisation put back, so
routing such a pole there is not an approximation of convenience; it is the same
number computed by the route that can compute it.

Two consequences make it cheap as well as defensible. It **bounds** the complex
route: afterwards `Γ ≥ ξ` for every pole the composite rule sees, so
`A = f_{\max}/Γ ≤ f_{\max}/ξ`, and the audited field's `A = 2.4 × 10⁵` corner
cannot recur. And it is **announced** per pass — `format_pass_report` prints the
count and `|B|` mass that took the legacy branch, so a field that is mostly
narrow poles reads as one instead of quietly becoming a plasmon-pole run.
Measured on silicon: 2.45 % of mode-passes at `n_p = 8` against 47.80 % at
`n_p = 1`, which is the difference between a field whose single pole sits near the
threshold and one with eight poles across two decades of width.

Read the floor as a statement about model class. The data's information content
and the kernel Σ reads are both smeared at `ξ`; a model with structure finer than
`ξ` claims to know something neither can tell it.

---

## 4. The head: the channel the basis does not carry

### 4.1 Why `q → 0` needs its own route

An ISDF basis indexes **positions**: `μ` labels a centroid, not a
reciprocal-lattice vector. There is therefore no row of `W_q` that *is* the
macroscopic screening, no `G = 0` element to read — and, dangerously, no norm,
rank or conditioning diagnostic that notices. The tile is a well-formed matrix
that simply does not contain the long-wavelength channel.

Three substitutes were measured and all fail at Γ. The trace invariant put its
dominant pole at 27–29 eV and moved it 14 eV across an `n_p` scan, because it
buries one collective mode under the whole particle–hole continuum. Reading
matrix elements gives a median pole at 64–73 eV, correctly reporting that a
typical centroid element is a short-wavelength local-field channel. Projecting
onto the softest Coulomb eigenvector works at finite `q` (17.85 eV, stable to
162 meV) and returns 35.38 eV at Γ — and no tie-break repairs that, because the
projection finds the long-wavelength direction via the divergence of `v(q+G)` at
`G = 0`, which at `q = 0` is exactly what has been removed.

### 4.2 The dipole route

Everything follows from one limit. The pair density at `G = 0` does not tend to 1
as `q → 0`; it tends to zero, because the two states are orthogonal, and the
first surviving term is the transition dipole,
`ρ_{vc\mathbf k}(\mathbf q, 0) = i\,\mathbf q\cdot\mathbf d + O(q^2)` with
`\mathbf d = \mathbf v/(i\Delta)`. Substituting into the same Lehmann sum gives

$$
\chi^0_{00}(\mathbf q, z) = |\mathbf q|^2\,\hat{\mathbf q}\cdot\mathsf A(z)\cdot\hat{\mathbf q},
\qquad
\mathsf A_{\alpha\beta}(z) \propto \sum_{vc\mathbf k} d^{*}_{\alpha}d_{\beta}\,K_z(\Delta),
$$

with `K_z` the *same* unified kernel §5 builds the quadrature substrate for. The
head is not a new subsystem; it is the same cell of the sampling table read with
a different vertex. The two powers of `|q|` cancel against `v(q) = 8π/|q|²`, and
`ε₀₀(\hat q, z) = 1 - 8π\,\hat q\cdot\mathsf A(z)\cdot\hat q` is finite.

It is carried as a `3 × 3` **tensor** all the way to the mini-BZ average. For
cubic silicon that is redundant, and the redundancy is the argument: silicon is
what the machinery is developed against and silicon cannot tell a tensor from a
scalar. Silicon's isotropy is then *evidence* — diagonal spread `5 × 10⁻⁷` across
an unsymmetrised sum over 26 624 transitions, the cubic point group appearing on
its own — while hBN through the identical call is anisotropic by 53 %. A scalar
head on a uniaxial crystal is an error the size of the anisotropy.

Two rules attach to the `q → 0` Coulomb average, which is the production
estimator `q0_average` and not a second convention. The mini-BZ average is taken
of the *quotient*, not of the tensor. And **the bare and screened averages must
be the same estimator**: the shipped code added the analytic-sphere term to one
and not the other, harmless for a plasmon-pole consumer that reads the two
numbers separately and fatal here, because the object fitted is `W − v` and two
estimators of the same bare integral differ by a constant that never decays with
`z`. Measured at 12.195 Ry — 0.37 % of the bare head, and 100 % of what the fit
sees at large `|z|`.

Code: `src/gw/mpa/head_dipole.py` (tensor, wings, mini-BZ head, f-sum),
`src/gw/mpa/sigma_head.py` (the head's Σ contribution, built as `n_p`
Godby–Needs heads — the reuse is exact, not an analogy).

### 4.3 Both sign conventions are carried

The relative sign on the nonlocal velocity commutator (`common/mtxel_sweep.py`)
is an **open owner decision**, so the store carries two labelled head sets and
every table has two columns. Five independent measurements point the same way:
`ε₀₀(0) = 24.2208` flipped against BerkeleyGW's 24.2205 and 31.8204 as shipped;
asymptotic `ω_p = 18.101 eV` flipped, matching BerkeleyGW, against 21.259 eV;
median relative agreement over 265 samples across the plane of `4.8 × 10⁻⁶`
flipped against 0.379 as shipped; spin–orbit eliminated by rebuild; and the f-sum
excess shown to be the pseudopotential's real contribution, seen by both codes.

What makes it the owner's call is scope, not doubt: one character moves every
`dipole.h5`, the four protected regression fixtures, the BSE absorption
references and the plasmon-pole head. So the campaign quantifies the choice
instead of making it. Because the head is injected *after* the pole passes, the
two columns come from **one** body integration and differ by the head convention
and nothing else, including nothing floating-point.

### 4.4 The gate, its state, and the principle it taught

`mpa_pipeline._inject_mpa_head` asks, before building anything, whether the
stored head poles reproduce the head samples stored beside them — the store keeps
the `2n_p` samples *with* the `n_p` poles precisely so this can be asked. The bar
is `HEAD_SAMPLE_REL_TOL = 1.0e-6`.

At `n_p = 1` both head sets pass at `10⁻¹⁶`. At `n_p = 8` both fail:
`4.567 × 10⁻⁴` as shipped, `6.162 × 10⁻⁵` flipped. **The `n_p = 8` quasiparticle
columns are therefore not published.** Three candidate causes are named in the
refusal, and all three are settled by measurement: not the store graft (identical
bytes in two independent stores), not mismatched head and body abscissae
(`max|z_head − z_body| = 0`), not a stored `W` (the stored sample gives
`ε_M = 31.8205 / 24.2207`, reproducing the cross-code comparison to four digits).
It is the fit-then-reflect price of §3.3.

Of the three ways out, two are closed by experiment. Fitting the head at fewer
poles has **no admissible selection** — no `n_p` from 8 down to 1 meets the bar
scored on the full sixteen-sample axis, over the nested prefix or over any of 162
subsets, and the descent gets monotonically worse. The causality-constrained fit
**does** meet it, at `2.019 × 10⁻⁹`, and the Σ it produces is wrong by 400 meV.

Which leaves the principle:

> **Certify where the object is consumed.** The head-sample gate certifies
> agreement at sixteen points *off* the real axis. Σ reads the head *on* the real
> axis, in τ. Two head models can agree at those points to `2 × 10⁻⁹`, agree on
> the f-sum to four digits, and differ by 400 meV in the self-energy they
> produce. The gate is **necessary and not sufficient**, and until a second gate
> exists that constrains the head where Σ reads it, no head-fitting scheme can be
> certified by sample reproduction alone.

Until then the safest head is the one whose poles sit *furthest* from the real
axis, not the one with the smallest sample residual.

---

## 5. Integration: the certified τ-quadrature in one page

### 5.1 One kernel, four cells

Every evaluation of `W_c` at a sample, and every window of the self-energy,
reduces to one integral:

$$
K_z(\Delta) = -2\int_0^\infty\! dt\; e^{izt}\sin(\Delta t)
            = \frac{1}{z-\Delta}-\frac{1}{z+\Delta},
\qquad z = \omega + i\varpi .
$$

A sample has each of `ω` and `ϖ` either zero or not, giving four analytic
characters — static, imaginary axis, real axis, and **the strip**. The legacy
code implemented three as unrelated subsystems and refused the fourth; they are
cases of one function. The sampling object carries, per point, its `z`, its
character, its *family* (which target serves it) and its *route* (which machine
evaluates it) — separate columns, because only the fourth cell needed a new
machine. The character tests `ω == 0` and `ϖ == 0` **exactly**, never against a
tolerance: the protocol constructs its zeros exactly, and a sample merely near an
axis is analytically on the strip. The metals shift `z = i·10⁻⁵ Ha` is the case
in point.

### 5.2 Positivity as a certificate, and lookup-and-refuse

The strip's route is a positive composite Gauss–Legendre rule on the damped-time
integral, truncated at `t_max = ln(2/ε)/ϖ` with panels graded by the envelope.
Its weights are `w_l(z) = −2h_l e^{izt_l}` with `h_l > 0`, so the total weight
mass is bounded by the `L¹` mass of the exact kernel *independently of node
count*: `Σ_l|w_l| ≤ 2/ϖ`, and the amplification ratio `κ₀ ≤ 1` measures 1.0000
everywhere. **Positivity is itself the bounded-amplification certificate** — why
this route needs no certification campaign, and why the campaign's target
elsewhere is *bounded-total-variation* minimax rather than unconstrained. Node
cost is about `5·A` at `10⁻⁶` and `7·A` at `10⁻⁸`, where `A = f_max/ϖ` is a
**beat frequency**, not a transition energy.

For the tabulated families the service is **lookup-and-refuse**
(`services/minimax/`). Runtime solving is not a production path, and the argument
is measurement: the same code with the same arguments on a different host
produced a different mathematical object, and one host produced two different
answers four months apart under a cache key recording neither solver version nor
machine. The door hands out artifacts it can name — table, hash, generating
commit, measured error, measured amplification — and refuses gaps by name,
offering the nearest achievable parameter and the generator invocation that would
close it. Cross-platform reproducibility is *certified, not promised*: two hosts
load the same bytes, they never compute the same rule. Each shipped entry passes
six checks re-derived from the shipped bytes, each paired with a
deliberately mis-certified twin the check must reject.

### 5.3 The `β` envelope, and where `β ≤ 1` comes from

Tabulated families are dimensionless. With `x_min` the lower edge of the interval
a family is tabulated on, the variable is `u = x/x_min`, the domain is `[1, R]`
with `R` the catalog's range index, and the complex-Laplace family `1/(u − iβ)`
has the imaginary-axis target as its real part.

`β` is a ratio, and two unrelated numerators had been forming it. The **width
clause** is `β = Γ_p/x_min`, a fitted pole width over a window edge. The **height
clause** is `β = ϖ/x_min`, a sampling line height over a band gap. A width is
small compared with the interval it sits in; a line height is not, because the
protocol puts the far line at 1 Ha whatever the material while silicon's gap is
0.0497 Ry. Two orders of separation is not an error in either number; it is what
happens when one dimensionless ratio is formed twice. The envelope therefore
carries **two clauses on one family**.

The shipped width clause is `β ≤ 1`, derived rather than fitted:

* On a **sign-definite** branch `x_min = min(E_A) + a_p ≥ Re Ω_p`, and the fit's
  fourth guard caps `|Im Ω| ≤ width_ratio_max · Re Ω` at `width_ratio_max = 1`.
  So `β ≤ 1` for any pole field this fitter can produce. **The rung and the guard
  are the same number** — the campaign is bounded by construction, not by a
  histogram that happened to fit.
* On a **crossing** branch the window floors `x_min` at `edge_factor·Γ_p`, so
  `β ≤ 1/edge_factor` identically. The deck's `edge_factor = 1.5` and the
  published `β_max = 2/3` are the same number, now a checked precondition
  (`refuse_edge_factor_below_envelope`) rather than a coincidence — which matters
  because 85–90 % of the `|B|` mass sits hard against that edge.

Measured on the audited 81-million-pole field: not one crossing pole exceeds 2/3,
while 11.52 % of sign-definite poles — carrying **62.25 % of the `|B|` mass** —
sit in `(2/3, 1]`. The corner expected to be dangerous was moderate-width poles
near the crossing boundary. It is not that; it is the branch that never crosses
at all, where the small quantity in the denominator is a band-structure gap.

**The binned width clause** (`feat/mpa-binned-width-clause-2026-08-09`, *not
merged*, flag OFF) addresses a sizing fact one level down: a crossing pane is
sized by its bucket's *widest* width while the rule's bandwidth divides by the
*narrowest*, so a bucket mixing decades of width pays `A ≈ edge·Γ_hi/Γ_lo`
however the clause arithmetic comes out. It bins widths geometrically at ratio 4
— no recursion, no predicate to chase, divergence structurally impossible.
Re-measured on the corrected `W_c` field, the *shipped* clause serves the real
silicon field with a ten-fold margin (47 leaves against a 512 ceiling,
independently confirmed at 43 by a second worker with a second harness), so the
binned clause is a **robustness asset worth about 6× in τ nodes and headroom for
wider-Γ physics, not a necessity on this deck**.

### 5.4 The pass loop

The self-energy runs **one pole at a time** through the *unchanged* two-point
device τ loop, accumulator and sharded tile sink (`src/gw/mpa/sigma_pass.py`;
routing in `src/gw/mpa/sigma_routing.py`). Nothing in the two-point core is
edited, so the bit-identity gate beside it holds by the argument it always held.

The correctness statement is the **re-association lemma**: `W_c(τ)` enters every
downstream contraction linearly, so a sum over poles is unchanged when
re-associated into one pass per pole. It is exact in exact arithmetic and *not*
bit-exact in floating point, which is why the order is **pinned ascending** and
why the combiner *measures* what the other orders would have cost instead of
asserting they do not matter. The exhibits: `3.83 × 10⁻¹⁶` (descending) and
`2.62 × 10⁻¹⁶` (shuffled) on a one-pole cube, and `6.0 × 10⁻¹⁵` / `6.2 × 10⁻¹⁵`
on the eight-cube production recombination — fifteen decades below the ±35 meV
effect under study.

A pass is a *slab*, not a pole: `(n_q, N_μ, N_μ)` poles at once with a spread in
both `Re Ω` and `Γ`. Each rule is therefore built at the **set's worst
parameters** — crossing truncation set by the smallest width, panel resolution by
the largest beat frequency; sign-definite decay by the smallest Laplace edge. One
term is new relative to a scalar router and is not optional: the crossing core's
beat frequency gains the slab's own `Re Ω` spread, because `E_ref_B` must be the
set minimum and the residual is a real phase in the integrand. The slab is
partitioned in `Re Ω` into geometric Laplace buckets at `r_max = 100`, sized to
the catalog's own grid so a future table lookup is a substitution rather than a
redesign.

Partial cubes are first-class: three deck keys (`mpa_pole_subset`,
`mpa_pass_partial_out`, `mpa_pass_partial_in`) let one process integrate some
poles and a later process sum the partials. Recombination happens before the head
injection, the at-DFT interpolation and the writer, so split and whole runs reach
the writer with the same object. Because a stack missing one pole or carrying one
twice returns a Σ that is finite, smooth, Hermitian and wrong by tens of meV, the
writer stamps a manifest and the combiner checks every field of it, with six red
twins in `tests/test_mpa_pass_partials.py`.

### 5.5 Compact-index panes

A pane's membership is an **index set** — ascending flat indices into the pole
field — not a boolean of that shape. The width clause partitions a non-crossing
branch into ~218 panes on the production deck; at 81.4 MB per full-size boolean
that was **17.8 GB of masks**, the whole of a pass's excess over the two-point
path. As index sets the same partition costs one index per live mode across every
pane together — 0.326 GB, a factor of 54.5. Same panes, same membership, same
nodes, same weights, bit-identical Σ, verified by a plan fingerprint compared
against the previous commit on the production field.

`MAX_WIDTH_SPLIT_LEAVES` was then re-derived from 512 to 8192: it had been
bounding *mask memory*, and with the masks gone it bounds *dispatch count*, so it
was refusing physics for a reason that had evaporated.

The honest caveat, which the commit states before anyone celebrates: **this is
not a cost fix for the τ loop.** The mask-dependent stage of a τ node is
4.6–11.0 ms against a node of 139–175 ms; the rest is the `G(τ)` formation, the
k-axis transforms and the band projection, none of which know a mask exists, and
the kernel measures the same wall at full occupancy as at 1/218. So the "0.4 %
utilization" quoted in the campaign table is a fraction of **modes**, not of the
machine, and the lever between this path and the two-point floor is the **pane
count** — a registered owner row that none of this touches.

### 5.6 Two seams worth knowing about

**The crossing operator-Im fix** (`27bd0984`). The crossing consumer stood in for
`sin(τu)` at a *complex* pole argument using an elementwise imaginary part. The
correct completion is the adjoint of the `(μ, ν)` **pair**,
`Im_op[cX] = (cX − (cX)^†)/2i`; the two coincide exactly only where `σ^τ` is
complex-symmetric, which under time reversal holds only at `k = −k mod G`.
Measured on silicon: Σ_c exactly Hermitian at the three TRIM `k` and
non-Hermitian by 8.8–30.3 eV at the five non-TRIM `k`, a star spread of 43.8 eV,
eqp0 splitting exact degeneracies by up to 67.6 eV. The corrected form reduces
bit-for-bit to the old one at TRIM, so every frozen TRIM reference is preserved
*by construction*.

**The crossing prefactor sign** (`ad6e1077`). The crossing core shipped with the
opposite sign convention to every other window in the tree, so its contribution
entered Σ backwards and the answer came back finite, smooth and wrong. The red
twins next door could not catch it: they score each quadrature against the
analytic object it reproduces, with no prefactor in the expression at all. The
new gate is **relative** — one window's sign against the other three.

Also: `refuse_wedge_pole_slab` refuses a symmetry-wedge pole store **by name**.
Unfolding a pole field is not the same operation as unfolding `W` — residues
transform as `W` does while pole *positions* only permute, and what time reversal
does to a pole in the closed fourth quadrant is precisely the question this tree
has got wrong four times in other guises.

---

## 6. The gate ecosystem, and why each gate exists

Every gate here was placed after something came back **finite, smooth, plausible
and wrong**. That is the selection criterion: a defect that crashes needs no gate.

**The bridge gate (`n_p = 1 ≡ GN`).** At one pole the multipole scheme is not
merely similar to Godby–Needs, it *is* Godby–Needs: the sample grid is exactly
GN's two probe points, a two-point Padé in `z²` returns `Γ_p = 0` by
construction, and `sigma_head` builds precisely the two-point head. Both arms fit
the same data with the same model, so any residual is *attributable* rather than
mysterious. This gate caught the `W`-versus-`W_c` defect: it read −120 143.93 meV
and now reads −107.24 meV, decomposing into a head pole value (108 meV apart
against the flipped set), a half-complex routing fraction (47.80 % legacy at
`n_p = 1`), and a window choice removed by construction. A 107 meV residual
between two arms differing by a 108 meV head pole is a residual; a −120 eV one
was not.

**Star and TRIM gates.** Σ must take the same value at every `k` in a symmetry
star. The measurement is taken on the **full BZ before** reduction to `k_irr`,
never after — because after the drop each star has one member left, every spread
arm reads identically zero, and the gate would be measuring nothing while
reporting success. This caught the crossing operator-Im defect at 43.8 eV, and it
is why the writer's ordering (arrays complete on the full BZ, statistic measured,
then one row per star kept) is an owner ruling implemented once in
`sigma_output.extract_and_stamp_k_irr`.

**Red twins.** Every check ships with a deliberately-wrong sibling it must
reject, because *a check that cannot fail is not evidence*. The catalog's six
certification checks each have a mis-certified twin. The residue refit has a
suppression switch existing only to exhibit the stale-residue defect. The partial
combiner has six: a missing pole, a doubled pole, two stores, two ω grids, a
foreign format version, a foreign `n_p`, and an empty directory — which must
raise, not return Σ_c = 0, itself finite, smooth and Hermitian. The
extended-precision head rebuild had one whose purpose was to *reproduce* the
shipped 4.567e-04, certifying the re-implementation was the shipped path and not
a lookalike. And the causality-constrained fit checks causality by value on every
returned pole rather than trusting its own parameterisation.

**Declarations.** `screening_content` and `energy_unit` are one idea applied
twice: the writer declares, the reader converts or refuses, and there is **no
default**, because a default is the same hole reopened wearing a declaration's
clothes. In both cases neither guess is even usually right — the production store
holds `W` where every fixture holds `W_c`, and it is Hartree where every fixture
is Rydberg. A store that cannot say what it holds cannot be refused, and both
defects survived exactly as long as the format was silent.

**The head-sample gate** (§4.4) exists because the head is the one channel with
no basis-internal diagnostic at all, and it is what taught the project the
difference between certifying interpolation and certifying continuation.

**Positivity and lookup-and-refuse** are structural rather than numerical gates:
a positive rule's amplification is bounded by its own construction at any node
count, and a table the service cannot name is a table it will not serve.

---

## 7. Current state, and what is open

**Branch.** Everything here is on `integration/mpa-table-2026-08-09`. The
campaign's numbers were computed at `133d2d11`; `edcd5d23` is a message-only
change to the head-sample refusal, which now names all three candidate causes,
gives the discriminator for each, and ends with the one-line experiment that
separates them. This guide sits on top of that.

**Landed.** The fit kernel and its guards; the sampling grid and the sampling
object; the complex-frequency evaluator and the χ⁰ resolvent route; the staged
B/Ω store and column-tile IO; the fit driver; the Σ pass loop and its routing; the
dipole head and the head's Σ contribution; both head sign sets under named
labels; partial-cube passes with pinned ascending recombination (`b2543bd3`); the
compact pane index and the re-derived leaf ceiling (`0f5da1ef`, merged at
`133d2d11`); the `screening_content` declaration and its three refusal seams
(`e7a32b03`); the fit store's energy-unit declaration (`2453074c`); the crossing
pair-adjoint completion (`27bd0984`).

**Not merged.** `feat/mpa-binned-width-clause-2026-08-09` @ `05a93e9e` —
certified on its own branch, flag OFF, retained as a robustness asset after the
shipped clause was shown to suffice on this deck.

**Not published.** The `n_p = 8` quasiparticle columns. The body is built,
verified and integrated into eight partial Σ_c cubes on certified rules; the
recombination stops at the head-sample gate.

**Open owner rows, as of 2026-08-09.**

1. **The head-sample bar, and the row above it.** Is `1 × 10⁻⁶` right at
   `n_p = 8`? It is a single module constant with no `n_p` in it, calibrated
   where the guard never fires, and by proportion against the one head
   perturbation measured end to end the miss is worth roughly 0.065 meV on eqp0 —
   three orders below the ±35 meV effect the campaign exists to resolve. The row
   that now outranks it is §4.4's: **a gate that constrains the head where Σ
   reads it, on the real axis.**
2. **The `q → 0` head sign** (`common/mtxel_sweep.py`). Five measurements point
   one way; the scope of the change is what makes it the owner's. Both columns
   are carried until it is ruled.
3. **The slab-aware `β` / pane-count row.** Withdrawn as a *blocker* — the
   1312-leaf refusal was a symptom of the `W`-vs-`W_c` defect and the real field
   asks for 47 — but open as a question about harder materials, where nothing has
   been measured. Pane count remains the lever between this path and the
   two-point cost floor.
4. **Sampling geometry.** Adopt far-real anchors (the best measured trade in the
   probe); if the arc is adopted, fix anchor-count nesting first; confirm which
   quarter of the ellipse; and decide whether anything downstream reads individual
   `Ω_p` or only their sum, since that decides whether pole-label stability is a
   figure of merit. The width-blindness worry is **retired**.
5. **Allocator regime.** The BFC-versus-`platform` A/B is not settled by anything
   in the MPA campaign, and nothing there is evidence against the BSE fleet's
   recommendation.

**Where to look.**

| For | Read |
|---|---|
| the model, the guards, the batching contract | `src/gw/mpa/pade_fit.py` |
| the sample grid and its nesting | `src/gw/mpa/sampling.py` |
| the 2×2 character table, routes, plans | `src/gw/mpa/sample_plan.py` |
| the unified `K_z` and the composite rule | `src/gw/mpa/evaluator.py` |
| which rule serves which pole on which branch | `src/gw/mpa/sigma_routing.py` |
| the pass loop, panes, buckets, partials | `src/gw/mpa/sigma_pass.py` |
| the `q → 0` head tensor, wings, f-sum | `src/gw/mpa/head_dipole.py` |
| the head's Σ contribution and the gate quantity | `src/gw/mpa/sigma_head.py` |
| the head gate itself, and injection | `src/gw/mpa_pipeline.py` |
| declarations, refusals, the store schema | `src/file_io/mpa_store.py` |
| certified tables, the door, the β axes | `services/minimax/` |
| every red twin named above | `tests/test_mpa_*.py` |

Module docstrings in this package are written to be read; several carry the
measurement that motivated the code they sit above, and where this guide
summarises, they are the authority.
