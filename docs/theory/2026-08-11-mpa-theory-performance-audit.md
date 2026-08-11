# A legible fast-full-frequency design for LORRAX

Date: 2026-08-11

Status: design audit, not yet an implementation

Scope: MPA construction of $W(z)$ and evaluation of $\Sigma_c$ in the
low-memory ISDF path. Simultaneous storage of multiple pole slabs is explicitly
out of scope.

## Correction to the first version of this report

The first version did not answer the central performance question. It proposed
transition-amplitude reuse even though materializing transition amplitudes is
incompatible with the intended scaling, and it discussed many secondary
changes before deriving why one MPA pole was causing about ten thousand tau
dispatches. That recommendation is withdrawn.

This replacement takes the four-branch GN-PPM construction as the reference:
use a small, intelligible number of windows; accept a bounded constant-factor
loss rather than multiplying windows to chase a local optimum; and never create
an object proportional to the transition space.

The main conclusion is simple:

> Eight poles do not intrinsically require 52,252 tau evaluations. The current
> planner destroys the elementwise correlation between pole energy and pole
> width, replaces the resulting cloud by hundreds of rectangular envelopes,
> and reruns the full $G$/FFT/projection kernel for every rectangle.

The preferred repair is also simple in outline:

1. preserve the GN four-branch decomposition;
2. use a complex-sector Laplace rule for every sign-definite branch, without
   width panes;
3. remove the discontinuous narrow-pole substitution: retain every fitted
   $\Gamma_p>0$ in the quasiparticle limit, and add an explicit
   $\eta_\Sigma$ continuously only when a broadened self-energy is requested;
4. construct the two sampling lines of $W(z)$ with line-shared, non-materializing
   $G_cG_v$ sweeps;
5. choose the smallest pole count that passes held-out $W$ and quasiparticle
   checks.

No production physics code or tests were changed as part of this audit.

## 1. The measured problem

The accepted GN-PPM calculation uses four conceptual branches. In the code,
the two crossing branches each contain a core, an A-stripe, and a B-slab, while
the two noncrossing branches each contain one window. The result is eight code
windows and 167 tau nodes:

| Half/space | Code windows | Tau nodes |
|---|---|---:|
| positive, conduction | core + A-stripe + B-slab | 70 |
| positive, valence | one noncrossing window | 14 |
| negative, conduction | one noncrossing window | 14 |
| negative, valence | core + A-stripe + B-slab | 69 |
| total | four branch plans, eight pieces | 167 |

The deterministic eight-pole MPA calculation executed the following plans:

| Pole | $+\omega$, cond. | $+\omega$, val. | $-\omega$, cond. | $-\omega$, val. | Total groups / nodes |
|---:|---:|---:|---:|---:|---:|
| 0 | 5 / 841 | 47 / 3,075 | 47 / 6,212 | 5 / 635 | 104 / 10,763 |
| 1 | 6 / 832 | 63 / 2,488 | 63 / 4,232 | 6 / 647 | 138 / 8,199 |
| 2 | 6 / 674 | 58 / 1,878 | 58 / 2,877 | 6 / 515 | 128 / 5,944 |
| 3 | 5 / 348 | 59 / 1,740 | 59 / 2,390 | 5 / 267 | 128 / 4,745 |
| 4 | 6 / 378 | 61 / 1,737 | 61 / 2,271 | 6 / 303 | 134 / 4,689 |
| 5 | 5 / 219 | 58 / 1,569 | 58 / 1,955 | 5 / 189 | 126 / 3,932 |
| 6 | 6 / 243 | 57 / 1,507 | 57 / 1,811 | 6 / 221 | 126 / 3,782 |
| 7 | 9 / 501 | 123 / 4,417 | 123 / 4,803 | 9 / 477 | 264 / 10,198 |
| total | 48 / 4,036 | 526 / 18,411 | 526 / 26,551 | 48 / 3,254 | 1,148 / 52,252 |

The important split is:

| Route | Groups | Tau nodes | Fraction of nodes |
|---|---:|---:|---:|
| two sign-definite branches | 1,036 | 44,842 | 85.8% |
| two complex crossing branches | 80 | 6,941 | 13.3% |
| legacy narrow-pole branches | 32 | 469 | 0.9% |

The ten-thousand-node passes are therefore not caused primarily by the
crossing core. They are caused by the two branches that never cross.

One tau node performs the expensive work: build $G(t)$, build the selected
piece of $W(t)$, transform, multiply, transform back, and project to the target
bands. Selecting a tiny pole mask costs only about 4.6--11 ms of a measured
139--175 ms node. Consequently, a pane containing 0.1% of the pole elements
does not cost 0.1% of a node. It costs essentially one full node.

This makes the correct objective

$$
C_\Sigma = \sum_{p,b,g,w} N_\tau(p,b,g,w)\,C_\tau,
$$

not the number of pole elements in a group and not the accuracy of each
quadrature considered in isolation.

## 2. Why the noncrossing planner explodes

Write one fitted pole as

$$
\Omega_i = a_i-i\Gamma_i,
\qquad a_i>0,
\qquad \Gamma_i\ge 0.
$$

On a sign-definite branch, the scalar denominator has the form

$$
d_i=x_i-i\Gamma_i,
\qquad x_i>0.
$$

The fitter's guard gives $\Gamma_i\le a_i$, and on the two globally
noncrossing branches $x_i\ge a_i$. Therefore every individual denominator
satisfies

$$
0\le \frac{\Gamma_i}{x_i}\le 1.
$$

The current planner does not certify that coupled set. Within a pole slab it
first buckets $a_i$, then forms the rectangular bound

$$
\beta_{\mathrm{pane}}
=
\frac{\max_{i\in\mathrm{pane}}\Gamma_i}
     {E_{A,\min}+\min_{i\in\mathrm{pane}}a_i}.
$$

The numerator and denominator can come from different matrix elements. Thus
every point may obey $\Gamma_i/x_i\le1$ while the enclosing rectangle has
$\beta_{\mathrm{pane}}>1$. The planner recursively bisects the width axis until
the false rectangle satisfies the clause. For the accepted Si pole field this
creates 46--122 `single` panes per sign-definite branch and pole.

For each pane, the current positive composite rule is sized using

$$
A_{\mathrm{rect}}
=
\frac{\Gamma_{\max}+(x_{\max}-x_{\min})}{x_{\min}},
$$

and at $10^{-8}$ its node count is approximately $7A_{\mathrm{rect}}$. The
full kernel is then rerun for that pane. This is a conservative certification
strategy for a rectangle that the physical pole cloud never occupied.

The width range is real, but the need for hundreds of full-kernel panes is not.

There is an intermediate option already present in the source. A read-only
reconstruction of the disabled ratio-four binned-width clause gives 11,631
nodes instead of 52,252, a 4.49-fold reduction, without changing the crossing
branches. That is a useful fallback while sector rules are being certified,
but it is not the desired endpoint: it remains about seventy GN node counts
and retains width panes that have no physical meaning.

## 3. The sign-definite replacement: one complex sector

The exact identity is

$$
\frac{1}{d}=\int_0^\infty e^{-d\tau}\,d\tau,
\qquad \Re d>0.
$$

Suppose the served denominators occupy a sector

$$
-\phi_{\max}\le\arg d\le0,
\qquad 0<\phi_{\max}\le\frac{\pi}{2}.
$$

Rotate the contour once, with $\theta=\phi_{\max}/2$:

$$
\frac{1}{d}
=
e^{i\theta}\int_0^\infty
\exp\!\left[-d e^{i\theta}s\right]ds.
$$

Now the exponent lies in the symmetric sector

$$
-\frac{\phi_{\max}}{2}
\le
\arg\!\left(d e^{i\theta}\right)
\le
+\frac{\phi_{\max}}{2},
$$

so its decay is bounded by the magnitude of the *same* denominator. No
unrelated $\Gamma_{\max}$ is paired with an unrelated $x_{\min}$.

With $\eta_\Sigma=0$ in the quasiparticle limit, or
$\delta_i=\Gamma_i+\eta_\Sigma$ when finite broadening is requested, the clean
universal choice is
$\phi_{\max}=\pi/2$ and $\theta=\pi/4$, covering the whole fourth quadrant. In
that case

$$
\Re\!\left[(x-i\delta)e^{i\pi/4}\right]
=
\frac{x+\delta}{\sqrt{2}},
$$

and

$$
\frac{
\left|\Im\!\left[(x-i\delta)e^{i\pi/4}\right]\right|
}{
\Re\!\left[(x-i\delta)e^{i\pi/4}\right]
}
=
\frac{|x-\delta|}{x+\delta}
\le1.
$$

This covers the globally noncrossing branches and the stripe/slab pieces with
one convention; no special case is needed when a pole is far from the real
axis.

This contour remains separable. The existing tau kernel receives

$$
t=-i e^{i\theta}s,
$$

and evaluates the exact $G$ and pole phases at that complex time. With energy
references chosen at the lower edges of their windows, both the $G$ factor and
the $W$ factor decay separately. There is no transition tensor and no need to
hold more than one pole slab.

On a stripe/slab with the $+\omega$ phase, the scalar energy-reference factor
and the scalar output-frequency factor must be formed as one exponential. The
former dominates by construction, so their product decays, but evaluating a
growing $e^{+\omega e^{i\theta}s}$ and a decaying reference factor separately
could overflow before multiplication. This is an implementation detail that
the scalar oracle must exercise explicitly.

A sector exponential rule can be constructed by sinc quadrature after
$s=e^y$, or by a sector minimax solve. Its rank should grow logarithmically
with the radial range, rather than linearly with the false rectangle's phase
bandwidth. That scaling is a mathematical design target, not yet a measured
production result. It must be certified on the actual denominator domain
before it replaces the current rule.

The default policy should be one rule per sign-definite branch. If the radial
range exceeds a certified table, use a very small number of logarithmic radial
windows. A split is accepted only when the *sum* of their node counts beats the
one-rule plan by a material factor. Hundreds of width panes are not an
acceptable fallback.

## 4. Crossing: separate the quasiparticle limit from finite broadening

The original MPA self-energy contains the Green-function time-ordering
parameter separately from the fitted complex screening pole. For an empty-state
denominator,

$$
\frac{1}
{\omega-E_m-\Omega_p+i\eta_\Sigma}
=
\frac{1}
{u+i(\Gamma_p+\eta_\Sigma)}.
$$

The fitted plasmon width is $\Gamma_p=-\Im\Omega_p$. The formal quasiparticle
target takes $\eta_\Sigma\to0^+$; a finite $\eta_\Sigma$ is a separately
requested broadened self-energy, not an intrinsic part of MPA and not a
quadrature-conditioning knob.

### 4.1 What is wrong now

For $\Gamma_p\ge\xi$, LORRAX retains the fitted complex pole. That is the
correct $\eta_\Sigma=0$ limit. For $\Gamma_p<\xi$, it discards $\Gamma_p$,
makes the pole real, and sends it through the legacy HGL-smoothed real-pole
target at $\xi$. Here $\xi=0.6667$ eV is derived from the requested output
window. The two sides are different analytic targets and do not meet
continuously at $\Gamma_p=\xi$.

This replacement changes the dispersive part coherently on either side of a
pole. It is therefore a credible candidate for part of the observed
$+6.594$ meV common Si correlation offset, but the residue-weighted sign and
magnitude are unknown. The decisive diagnostic is current HGL versus the full
complex fitted $\Gamma_p$ at the same $\eta_\Sigma=0$, followed by an
independent $\xi$-sensitivity sweep. A finite-$\eta_\Sigma$ run changes the
observable and is not, by itself, an attribution test.

### 4.2 What the rigid Si offset is, and is not

For bands 7--10—two valence and two conduction bands at all eight irreducible
$k$ points—the MPA--BGW-CD difference has mean $+7.661$ meV, while its
mean-aligned MAE is only 1.682 meV. It is therefore usefully described as a
near-gap common correlation shift, although the complete 84-state safe window
is less perfectly rigid.

Several explanations are already excluded or strongly constrained:

- bare $\Sigma_x$ has mean error $+0.0025$ meV and MAE 0.111 meV, so the
  residual is not ordinary exchange or upstream bare-Coulomb machinery;
- the comparison uses BGW's primed columns, which omit the 'achcor' static
  remainder, and LORRAX also has no static remainder. 'exact_static_ch=1'
  makes BGW's primed and corrected results available; it does not insert the
  remainder into the primed result;
- BGW's 'gpp_sexcutoff' and 'gpp_broadening' are GPP controls, not part of the
  contour-deformation reference used here, so a PPM SEX cutoff cannot explain
  this MPA--CD statistic;
- output-grid interpolation has about 0.69 meV RMS effect and $-0.027$ meV
  signed mean, too small and too non-rigid to supply the offset.

The remaining ranked candidates are the narrow-pole HGL substitution, general
body continuation error from an exactly determined but not held-out-certified
MPA fit, and a smaller residual dynamic-head continuation error. The Schur/LF
head's agreement with the matched BGW head rules out a gross head
normalization or static-screening error, but does not prove its complete
real-axis continuation at the meV level.

The shortest attribution sequence is:

1. legacy HGL versus full-complex fitted $\Gamma_p$ at
   $\eta_\Sigma=0$, followed by a $\xi$ sweep;
2. head-exact/head-off/body-only decompositions of $\Sigma_c$;
3. additional body $W(z)$ oracle points scored with a
   self-energy-weighted norm.

Until those tests are made, the rigid term is localized to the
correlation-frequency treatment but should not be assigned specifically to
the head or the narrow-pole route.

### 4.3 The common four-window geometry

Let $T=\omega_{\max}$. Each crossing branch is partitioned only by real
energies:

- core: $E_A\le T$ and $a_p\le T$;
- A-stripe: $E_A>T$ and $a_p\le T$;
- B-slab: all $E_A$ and $a_p>T$.

These inequalities partition the branch exactly. The stripe and slab are
strictly sign-definite and use the sector rule. Only the core needs real time,
with

$$
-2T\le u=\omega-E_A-a_p\le T.
$$

This topology is independent of $\Gamma_p$, $\eta_\Sigma$, and an
`edge_factor`. Widths choose a scalar rule inside the core; they do not create
a new energy-window tree.

### 4.4 Explicitly broadened self-energy

If a finite $\eta_\Sigma>0$ is requested, define

$$
\delta_p=\Gamma_p+\eta_\Sigma
$$

continuously for every pole. The core identity is

$$
\frac{1}{u+i\delta_p}
=
-i\int_0^\infty e^{iut}e^{-\delta_p t}\,dt.
$$

One rule can be certified on the full right-half-plane rectangle

$$
-2T\le u\le T,
\qquad
\eta_\Sigma\le\delta_p\le T+\eta_\Sigma,
$$

with logarithmic grading near $t=0$ for the short-time scale at large
$\delta_p$. The stored pole is not changed. On real time the integrator
multiplies by $e^{-\eta_\Sigma t}$; on a sector contour it may equivalently use
the temporary Sigma-only operand

$$
\Omega_p^{\mathrm{eff}}
=
a_p-i(\Gamma_p+\eta_\Sigma).
$$

The causal sign must be checked separately on occupied and empty branches.
$\eta_\Sigma$ must match a reference calculation or be part of an explicit
convergence sequence; it must never increase when the output window widens.

### 4.5 Strict quasiparticle limit

At $\eta_\Sigma=0$, every fitted $\Gamma_p>0$ stays in the exact complex
resolvent:

$$
\frac{1}{u+i\Gamma_p}
=
-i\int_0^\infty e^{iut}e^{-\Gamma_p t}\,dt.
$$

There is no positive global floor when fitted widths approach zero. The clean
options are therefore:

1. one certified multiscale core rule over the actual
   $(u,\Gamma_p)$ domain, if its total node count is acceptable;
2. a few geometric $\Gamma$ rule bands that retain each element's exact
   $e^{-\Gamma_p t}$ physics;
3. an analytic shifted-resolvent or principal-value treatment for exactly
   real poles.

The limit $\Gamma_p=0$ is a principal-value plus delta-function problem and
must be named as such. A small positive fitted width must not be turned into a
real pole merely because it is expensive. In an idealized time-bandwidth
model, the oscillatory part of dyadic width-band costs forms a geometric sum,
but near-zero resolution, tolerance terms, and per-band overhead prevent that
from being a certified bound. The scalar campaign must decide whether one
multiscale rule or a small number of bands is cheaper.

## 5. The target Sigma plan

For each pole pass, the conceptual plan should again be:

| Half/space | Plan |
|---|---|
| $+\omega$, conduction | crossing core + sector A-stripe + sector B-slab |
| $+\omega$, valence | one sector noncrossing rule |
| $-\omega$, conduction | one sector noncrossing rule |
| $-\omega$, valence | crossing core + sector A-stripe + sector B-slab |

This is the same four-branch story as GN-PPM. Complex poles change the scalar
quadrature served by a branch; they do not require a new combinatorial
hierarchy.

Without simultaneous pole storage, aggregate Sigma work cannot honestly equal
GN work. Each of $n_p$ pole slabs must pass through the contraction at least
once. The useful best-case architectural reference is therefore

$$
n_p N_\tau^{\mathrm{GN}}
=
8\times167
=
1{,}336,
$$

not $N_\tau^{\mathrm{GN}}$ itself. This is not a mathematical lower bound:
finite $\Gamma_p$ can make some MPA poles easier, while a small explicit
$\eta_\Sigma$ can make the crossing core harder than GN.

For an explicitly broadened calculation, a useful cost model is

$$
N_{\mathrm{total}}(\eta_\Sigma)
=
N_{\mathrm{sector}}+\frac{K}{\eta_\Sigma},
$$

where the second term represents the crossing-core time scale. Neither
$N_{\mathrm{sector}}$ nor $K$ is known yet; Step A must determine both from
certified scalar rules. In the strict quasiparticle limit the relevant model
is instead the measured cost of the multiscale or few-band $\Gamma_p$ plan.
The defensible commitment at this stage is narrower: eliminate the 44,842
sign-definite pane nodes and let the scalar campaign determine the remaining
core cost.

Independent pole jobs can reduce wall latency when enough GPUs are available,
while aggregate node work remains proportional to $n_p$. The achieved latency
also includes compilation, I/O, fit, and scheduler overhead and must be
measured. Reusing one $G(t)$ build across several pole slabs could remove the
aggregate factor, but it is the separate multi-pole-memory project and is not
assumed here.

## 6. Constructing $W(z)$ without transitions

### 6.1 What produced the accepted store

The accepted file

`/pscratch/sd/j/jackm/mpa_geom_0810/det6/stores/Wc_det_np8.h5`

is stamped `prov_route='exact-resolvent'` and
`prov_producer='geom_samples.stage_geom'`. For each irreducible $q$, that
producer calls `chi0_resolvent` for all 16 samples and then performs 16 Dyson
solves. The implementation forms, one $k$ block at a time,

$$
M^q_{cvk}(\mu)
=
\sum_s
\psi^*_{c,k-q}(s,\mu)\psi_{v,k}(s,\mu).
$$

It also allocates the complete 16-frequency $W_c$ array. This was a useful
correctness route for the Si campaign, but it is not the scalable production
architecture. It must not become the FF path for large band counts, dense
$k$ meshes, phonon displacements, or high-throughput work.

For this small Si case the memory-heavy route is fast: the recorded body build
costs about 0.74--1.30 s for all 16 samples at one irreducible $q$, followed by
about 0.29 s for the 16 Dyson solves. The objection is its transition-space
memory and scaling, not that this particular artifact was slow.

### 6.2 The non-materializing line sweep

For a transition energy $\Delta>0$ and sample $z=\omega+i\varpi$,

$$
K_z(\Delta)
=
\frac{1}{z-\Delta}-\frac{1}{z+\Delta}
=
-2\int_0^\infty
e^{-\varpi t}e^{i\omega t}\sin(\Delta t)\,dt.
$$

At one real-time node, the transition sine sum can be formed from the
anti-Hermitian combination of conduction and valence propagator products. In
schematic form,

$$
S_q(t)
=
\sum_{kvc}M^q_{cvk}M^{q*}_{cvk}\sin(\Delta_{cvk}t),
$$

but $M^q_{cvk}$ is never constructed. The existing ISDF pattern builds the
conduction and valence Green functions, multiplies them in real space, and
FFTs the product to all $q$.

Every sample on one horizontal line then differs only in scalar weights:

$$
\chi_0(q;\omega_j+i\varpi)
\approx
-2\sum_{\ell}h_\ell
e^{-\varpi t_\ell}e^{i\omega_jt_\ell}S_q(t_\ell).
$$

Thus one expensive $G_cG_v$/FFT sweep serves the complete line. The near and
far lines have different $\varpi$ and should have different rules.

$S_q(t_\ell)$ must be produced, accumulated into the requested frequency
outputs, and discarded one time node at a time. Storing it for every
$t_\ell$ would merely replace the forbidden transition tensor by an
$N_\tau n_qN_\mu^2$ time-history tensor.

This requires a real-time sibling of `w_isdf.compute_chi0_multi`; the current
`evaluator.evaluate_samples` expresses the scalar line algebra but is not wired
to a production Green-function sweep.

### 6.3 Why a separate grid for every $W(z)$ is usually worse

For the monolithic line representation, let

$$
F=\Delta_{\max}+\max_j|\omega_j|,
\qquad
A=F/\varpi.
$$

A positive composite rule at relative tolerance $10^{-8}$ uses about $7A$
nodes. The damping tail extends to

$$
T\simeq\frac{\log(2/\epsilon)}{\varpi},
$$

so a direct real-time discretization naturally encounters the time-bandwidth
scale $FT$. The measured $7A$ behavior describes the current positive
composite rule; it is not a lower bound on sector minimax rules, nonuniform
representations, or a formulation with smaller active transition windows.

If the line is split into point groups $G$, the expensive-node count is
approximately

$$
N_{\mathrm{split}}
\propto
\sum_G
\frac{\Delta_{\max}+\max_{j\in G}|\omega_j|}{\varpi}.
$$

Every group repays the $\Delta_{\max}$ term. Per-point grids are consequently
the wrong default. They can win only if they permit materially smaller
transition windows or if output accumulation, rather than the Green-function
build, dominates wall time.

The planner should compare the measured model

$$
C_W
=
\sum_G N_G
\left[C_G(G)+C_{\mathrm{FFT}}+|G|C_{\mathrm{acc}}\right]
$$

and choose the simplest plan within a factor of two of its minimum. It should
never customize the time grid by $q$, because one real-space transform
produces all $q$ together.

### 6.4 The same crossing/noncrossing idea applies to $W$

The two terms in $K_z$ should not be forced through one monolithic sine rule.
Their signs and quadrants must be explicit. If $x=\Delta+\omega>0$, then

$$
\frac{1}{z+\Delta}=\frac{1}{x+i\varpi}
$$

lies in the first quadrant and uses the conjugate rotation
$\theta=-\pi/4$. For the high-energy resonant tail,
$\Delta>\omega_{\max}$,

$$
\frac{1}{z-\Delta}
=
-\frac{1}{(\Delta-\omega)-i\varpi},
$$

so the fourth-quadrant rule carries the displayed minus sign. Consequently:

- $1/(z+\Delta)$ is always sign-definite and belongs on a sector-Laplace
  contour;
- $1/(z-\Delta)$ needs real time only where the sampled $\omega$ range overlaps
  the transition range;
- the high-energy resonant tail is sign-definite again.

The analogous occupied/empty signs must be derived, rather than inferred,
when applying sector rules to the Sigma stripe and slab.

The transition energy is a sum of positive conduction and hole energies, so
the same core/A-stripe/B-slab partition used by GN can separate the crossing
piece without ever naming a transition. Splitting a whole sampling line into
many real-frequency subranges can at best reduce the linear crossing bandwidth
by a modest factor while multiplying the number of crossing partitions. The
default should therefore be one near-line plan and one far-line plan, each with
a few GN-like pieces, not one plan per $W(z)$.

### 6.5 Memory is controlled by output batching

Accumulating $B$ frequencies at once costs, before sharding,

$$
M_{\mathrm{acc}}
=
16B\,n_qN_\mu^2\ \text{bytes}.
$$

This is accumulator memory, not peak memory. A production estimate must use

$$
M_{\mathrm{peak}}
=
M_{\mathrm{base}}+M_{\mathrm{acc}}+M_{\mathrm{Dyson}}+M_{\mathrm{output}},
$$

including resident Green-function/FFT arrays, Coulomb and Dyson workspace,
and at least one completed output matrix. The displayed $n_q$ is the full-$q$
case; selecting only an irreducible wedge during accumulation is a separate,
not-yet-implemented optimization.

If all points on a line fit, it is one sweep. Otherwise, repeat the same line
rule for output batches of size $B$. This is an explicit and measurable
memory--compute tradeoff; it never creates a transition tensor or a stored
time history. The batch is chosen from an announced peak-memory budget, not
hard-coded from the Si geometry.

## 7. Fitting: what to retain and what to change

The direct fit of $W_c(z)$ has the correct even rational form,

$$
W_c(z)
=
\sum_{p=1}^{n_p}
\frac{2\Omega_pB_p}{z^2-\Omega_p^2}.
$$

The Loewner denominator solve was better conditioned than the old
Vandermonde-like Padé solve for this Si campaign: the latter crossed
double-precision conditioning at $n_p=8$--10, while the Loewner form represents
the same rational type without powers of $z^2$. That result does not establish
Loewner as uniformly preferable across materials.

LORRAX is not using Yambo's fitting algorithm. It shares the even multipole
ansatz and the double-parallel sampling concept, but its fixed-support Loewner
solver, rank threshold, and repair operations differ from Yambo's
analytic/linear Padé and Padé--Thiele paths.

The following distinctions matter:

- Refitting residues after a pole has been reassigned, conjugated, or removed
  is necessary.
  Keeping residues belonging to the old poles would be wrong.
- A small training residual on exactly $2n_p$ samples is not a continuation
  certificate. It mainly says the interpolation problem was solved.
- The original MPA work publishes a fulfillment/time-ordering repair that
  enforces the required sign of $\Im\Omega_p$. Any particular conjugation or
  reassignment used by LORRAX is model-changing and must be identified; its
  post-repair residue refit and held-out behavior must be reported.
- Reducing the active pole count for a rank-deficient or unfulfilled element is
  consistent with the Yambo MPA strategy. Forcing every element to carry the
  nominal order is not required.

The robust workflow is:

1. keep one shared two-line sampling geometry;
2. evaluate additional unused points within the line envelopes;
3. use rank-revealing Loewner fits and select the smallest $n_p$ whose held-out
   $W$ error and quasiparticle error pass;
4. keep zero-residue slots for elements of lower numerical order, so storage
   stays regular;
5. report singular-value gaps, effective rank, residue norms, distance of each
   pole from the sampled domain, and sensitivity to the support partition and
   SVD cutoff;
6. report both raw and self-energy-weighted held-out errors, every pole repair,
   and the response-level analytic checks below.

Extra points require extra accumulation and Dyson solves. They require no new
$G_cG_v$ time nodes only when they remain inside the certified frequency
envelope and fit in the same output batch; otherwise the rule grows or a sweep
is repeated. The accepted 16-point store uses all 16 points for the $n_p=8$
fit, so it contains no true held-out data for that model. It needs additional
oracle points; reduced-order cross-validation is available only for
$n_p<8$.

Two exact model moments should be checked elementwise for every fit:

$$
W_c(0)=-2\sum_p\frac{B_p}{\Omega_p},
\qquad
\lim_{|z|\to\infty}z^2W_c(z)=2\sum_p\Omega_pB_p.
$$

Once the finite-band and pseudopotential conventions behind the
high-frequency moment are pinned, these can enter a scaled, regularized
residue refit. Until then they are diagnostics, not hard constraints. Complex
elementwise residues are fit coefficients, not individually positive
oscillator strengths. At matrix level the appropriate checks are the Schwarz
relation, schematically $W(q,z)^\dagger=W(q,z^*)$ with the repository's
$q$/time-reversal convention, and positive-semidefinite retarded loss on
$\omega>0$ under the repository's sign convention. $W$ at a generic complex
frequency is not simply Hermitian.

For geometry relaxation and phonons, hard model-order changes and pole repairs
can create non-smooth quasiparticle Hamiltonians. Topology, nominal zero-padded
order, and correction policy should be frozen across a displacement family.
Absolute sampling frequencies may remain fixed only when one family-wide
envelope covers every displacement; otherwise update them by a smooth,
deterministic rule. A rank change can be continuous when a zero-padded residue
vanishes smoothly, so it is a continuity diagnostic rather than an automatic
failure. Discontinuous changes must be flagged.

## 8. A structural research track: common matrix poles

Elementwise poles are flexible, but pole positions then depend on the chosen
matrix basis and vary over every $(\mu,\nu)$ element. That variation is exactly
what makes analytic convolution incompatible with the current low-scaling
separation.

A matrix-valued model with a shared pole dictionary would be

$$
W_c(q,z)
=
\sum_{p=1}^{r}
\frac{2\Omega_{p,q}B_{p,q}}{z^2-\Omega_{p,q}^2},
$$

where $\Omega_{p,q}$ is scalar over matrix elements at a fixed $q$ and
$B_{p,q}$ is a residue matrix. This has two possible consequences:

1. at a fixed $q$, every matrix element shares a pole frequency, so its local
   Sigma frequency geometry is GN-like;
2. the analytic MPA convolution can be written with shifted one-particle Green
   functions, avoiding both tau quadrature and transition amplitudes.

The direct analytic route is then based on objects such as

$$
G(z)=\Psi\,\mathrm{diag}\!\left(\frac{1}{z-E_m}\right)\Psi^\dagger,
$$

which are band contractions, not valence--conduction transition tensors.

A $q$-dependent pole dictionary is not genuinely global: its shifted-$G$
contraction may reintroduce a $q$ loop and lose the all-$q$ FFT separation. A
stronger GN-like claim would require poles shared across $q$ as well. Either
choice therefore needs an end-to-end scaling benchmark.

The shifted-$G$ cost is schematically

$$
C_{\mathrm{shifted}\,G}
\sim
r\,n_\omega\left(C_G+C_{\mathrm{convolution}}\right),
$$

with separate occupied and unoccupied terms. Common poles remove tau
quadrature but do not automatically remove the pole multiplier. They win only
if $n_\omega$ is sufficiently smaller than the time-rule rank, or if
$\Sigma(\omega)$ can be interpolated safely from fewer shifted evaluations.

This is not the first implementation step because a common denominator may
require many more poles to describe all matrix elements. The decisive
experiment is a read-only fit of the existing $W(z)$ store using MIMO Loewner
or vector fitting, with common poles chosen from small random sketches of $W$
itself. No transition data are involved. Continue only if roughly 8--16
common poles pass additional held-out $W$ and quasiparticle checks and the
$q$-dependent contraction is competitive; abandon it if the required rank
grows with $N_\mu$.

High-frequency moment constraints and matrix passivity constraints are useful
in this track, but they should be soft constraints or validation metrics until
their finite-band and pseudopotential conventions are pinned. They are not the
explanation for the present 52,252-node cost.

## 9. Performance expectations and honest lower bounds

There are three distinct multiplicative costs:

| Stage | GN-PPM | Elementwise $n_p$-pole MPA | What can remove the factor? |
|---|---:|---:|---|
| screening samples | 2 $W$ values | $2n_p$ $W$ values | line-shared $G_cG_v$ sweep removes repeated tau builds, not Dyson solves |
| Dyson solves | 2 | $2n_p$ | smaller adaptive order or a different matrix model |
| Sigma pole contractions | one pole model | at least $n_p$ pole passes without batching | multi-pole batching or shared-pole shifted-$G$ convolution |

The immediate sector/window repair removes the mechanism responsible for
44,842 of 52,252 nodes. Its resulting speedup is conditional on the certified
crossing-core rank; no single reduction factor should be quoted before Step A.
It does not repeal the honest $n_p$ aggregate-work factor imposed by
one-pole-at-a-time memory.

For the line-sampled screening stage, direct real-time quadrature encounters a
time-bandwidth scale proportional to
$(\Delta_{\mathrm{active}}+\omega_{\max})/\varpi_{\mathrm{near}}$. This is a
planning heuristic, not a universal lower bound. The first way to control it
is a few crossing/noncrossing energy windows and a well-chosen sampling range,
not per-frequency transition amplitudes.

For QSGW, quadrature families, compiled kernels, and allocations can be reused
when shapes remain unchanged. Energy bounds and sample geometry must be
revalidated at every iteration and configuration. For forces, use one
enclosing family-wide envelope or a smooth deterministic update rule. The
physics arrays themselves must be recomputed. A fast but discontinuous pole
selection is unsuitable for forces; reproducible smoothness is part of the
performance specification.

## 10. Minimal implementation sequence

### Step A: scalar certification, no production run

Using stored pole and energy extrema:

1. construct a sector exponential rule for the exact coupled denominator
   domain;
2. compare it with $1/d$ on random, boundary, and worst production tuples;
3. for an explicitly broadened target, construct the full-$\delta$ crossing
   rule and compare it with $1/[u+i(\Gamma+\eta_\Sigma)]$;
4. for the strict quasiparticle target, compare one multiscale rule and a few
   exact-$\Gamma$ geometric bands with $1/(u+i\Gamma)$, including the declared
   exact-real-pole limit;
5. print the resulting four-branch plan and node count for all eight stored
   poles.

Acceptance target: uniform scalar error at the requested tolerance, no width
pane dimension, and measured values for $N_{\mathrm{sector}}$, the broadened
core coefficient $K$, and the strict-limit core cost. This is a small analysis
tool, not a large unit-test campaign.

### Step B: one-pole Sigma A/B

Implement the new planner behind one explicit experimental switch:

- no `Gamma < xi` path;
- a declared strict-QP or finite-broadening target;
- fixed GN branch geometry;
- sector rules for every sign-definite piece;
- full plan and node census in the log.

The correctness gate is the direct analytic MPA Eq. 13 denominator on a small
state/q/element sample. Comparison with the current production pole partial
then measures the intended physics change from retaining narrow fitted widths;
it is not an equality gate because the current path substitutes a different
denominator.

### Step C: eight-pole production comparison

Only after Step B passes, run the existing four-GPU production protocol and
compare against the same BerkeleyGW primed contour-deformation columns. Retain
the existing $[-7,+7]$ eV grid and score only the safe $\pm5$ eV region.

Required outputs:

- raw and mean-aligned errors for several valence and conduction bands at all
  irreducible $k$ points;
- direct and indirect gaps;
- $\eta_\Sigma$ sweep;
- total and per-pole tau nodes and wall time;
- maximum memory.

For bands 7--10 at all eight irreducible $k$ points, the current result has a
rigid $+7.661$ meV mean offset. After subtracting that mean, its MAE is
1.682 meV and its maximum error is 2.577 meV; aligned to the VBM, its MAE is
2.120 meV and maximum error is 4.517 meV. The indirect and direct gap errors
are $-3.594$ and $-4.517$ meV. The target is at most 5 meV for the band
dispersion and gaps, while the absolute common offset is diagnosed separately
against matched head, static-remainder, and reference conventions. The new
route must preserve this accuracy before speed is credited.

### Step D: non-materializing $W(z)$ producer

Build the real-time multi-output sibling of `compute_chi0_multi`, initially for
one horizontal line and a small output batch, consuming $S_q(t)$ immediately
at each time node. Validate it against the accepted exact-resolvent $W(z)$
artifact. Then add the resonant/antiresonant GN-like window split and optimize
the explicit wall model, with one near-line and one far-line plan as the
default.

### Step E: reduce the pole count before making poles cheaper

Add unused oracle points within the line envelopes, because the existing 16
samples are all consumed by the $n_p=8$ fit. Use them to test $n_p=4,6,8$;
reduced-order cross-validation on the existing store is informative only for
$n_p<8$. The smallest order meeting the 5 meV quasiparticle target wins. Only
after the elementwise path is efficient should the
common-matrix-pole/shifted-$G$ track be judged.

## 11. Things this design explicitly does not do

- It does not materialize or cache $M_{cvk}(\mu)$.
- It does not precompute a transition tensor under another name.
- It does not assume multiple pole slabs fit in memory.
- It does not create a quadrature grid for each matrix element or each $q$.
- It does not treat hundreds of tiny masks as cheap because they contain few
  pole elements.
- It does not replace a fitted width below a window-dependent threshold.
- It does not claim that sample interpolation error certifies real-axis
  continuation.
- It does not add a large unit-test suite before the scalar plan has proved
  that the method is worth implementing.

## 12. Evidence and references

Local implementation evidence:

- `src/gw/mpa/sigma_pass.py`: recursive noncrossing width split and one full
  integration per resulting `WindowGroup`.
- `src/gw/mpa/sigma_routing.py`: current complex crossing and sign-definite
  integral identities.
- `src/gw/ppm_windows.py`: the GN four-branch/eight-piece reference geometry.
- `src/gw/ppm_sigma.py` and `src/gw/ppm_tau_kernel.py`: one full
  $G$/FFT/projection dispatch per tau node.
- `src/gw/mpa/chi0_resolvent.py`: the current blockwise transition-amplitude
  correctness producer.
- `src/gw/mpa/evaluator.py`: the unwired line-shared scalar damped-time
  algebra.
- `src/gw/w_isdf.py`: the existing non-materializing static multi-output
  Green-function kernel pattern.
- `/pscratch/sd/j/jackm/mpa_geom_0810/_reports/batchlogs/det_wide_p*.log`:
  accepted per-pole group and node census.

Primary method references:

- M. Kim, G. J. Martyna, and S. Ismail-Beigi,
  [Complex-time shredded propagator method for large-scale GW
  calculations](https://doi.org/10.1103/PhysRevB.101.035139), the CTSP
  framework underlying the GN-style decomposition used here.
- D. A. Leon et al., [Frequency dependence in GW made simple using a
  multi-pole approximation](https://arxiv.org/abs/2109.01532), especially
  Eqs. 11--13 for the pole model and analytic self-energy.
- [Yambo's MPA self-energy implementation](https://github.com/yambo-code/yambo/blob/3a0d457a24da514d673d18981ae316e467600e0d/src/qp/QP_mpa.F#L333-L388),
  which adds `QP_G_damp` to the external Green-function frequency before
  combining it with the complex MPA poles for finite-broadening evaluations;
  current Yambo also sets this damping to zero on non-real-axis evaluation
  grids, so this is evidence for an available broadening, not an intrinsic MPA
  width.
- D. A. Leon et al., [Efficient full frequency GW for metals using a
  multipole approach for the dielectric screening](https://arxiv.org/abs/2301.02282),
  for the metal sampling and intraband extension.
- [Yambo MPA tutorial](https://wiki.yambo-code.eu/wiki/index.php/Quasi-particles_and_Self-energy_within_the_Multipole_Approximation_%28MPA%29),
  for the production double-parallel sampling, Padé--Thiele option, reduced
  pole count for unfulfilled modes, and explicit Green-function damping used
  for self-energy/spectral calculations.
- D. A. Leon et al., [Multipole approximation for the self-energy in GW
  calculations](https://arxiv.org/abs/2501.09121), whose analytic expression
  omits a finite $G_0$ damping when fitted complex screening poles already
  supply the time ordering.

The papers evaluate the pole convolution analytically. LORRAX's complex-time
route is justified only if it preserves the low-scaling separability at a
small, measured constant over that analytic reference. The present
$52{,}252/167$ ratio is 313 times one GN evaluation, but
$52{,}252/(8\times167)=39.1$ times the honest eight-pole,
one-pole-at-a-time architectural reference. Neither meets that standard; the
four-branch sector design is the direct attempt to do so.
