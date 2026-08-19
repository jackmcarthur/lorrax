# Multipole frequency integration

This chapter is the authoritative description of multipole-approach (MPA)
frequency sampling, fitting, and self-energy windowing in LORRAX. It follows
the data in execution order, because every later approximation is constrained
by an object defined earlier. Exact deck defaults belong to the
[input reference](../input_reference.md); equations and validity domains belong
here.

`compute_mode = mpa` **runs**: the entry refusal owned by
`gw_config.UNIMPLEMENTED_MODES` was removed at `9c9b23dc` (2026-08-15) on
`integ/metal-mpa-qsgw-2026-08-15`, which is pushed but **not** an ancestor of
`origin/main`. The condition it waited on — one real-material calculation
traversing the disk-bounded chi-to-W-to-fit pipeline, the shared Sigma planner
and the common dynamic-Sigma finalizer end to end — was met by a three-iteration
metallic QSGW run on the Na deck. Read that narrowly: it means the mode parses
and executes, **not** that the result converged (it did not) and not that the
numbers are validated. Convergence and the BGW comparison continue under rungs
R5/R6; the per-site refusals that guard physics preconditions are untouched.

## 1. One pipeline, three approximations

The calculation has the following dependency chain:

```text
occupied and empty bands
  -> chi0(z_j) on two complex-frequency lines
  -> Wc(z_j) = [1 - V chi0(z_j)]^-1 V - V
  -> elementwise Loewner fit {Omega_p, B_p}
  -> geometry-dependent Sigma(omega) quadratures
  -> one shared G(t) W(t) spatial contraction per time node
  -> q=0 head, interpolation, QSGW operator, and output
```

The three approximations are deliberately separate. The chi quadrature
controls the noise in the samples presented to the pole fit. The rational fit
controls how accurately a finite pole model reconstructs those samples. The
Sigma quadrature controls how accurately the fitted model is integrated on a
real-frequency grid. A small scalar error in the last stage cannot repair a
poor pole fit, and increasing the pole count cannot repair noisy chi samples.

All energies in the runtime equations are in Ry. Stored pole files declare
their frequency-axis and residue units, and the reader converts at that
boundary. Frequencies in deck keys ending in `_ev` are converted once during
configuration.

## 2. Transition spectrum and the chi kernel

For a zero-temperature insulator, let the occupied and empty edges be

$$
E_{v,\max}=\max_{v\mathbf{k}}\epsilon_{v\mathbf{k}},
\qquad
E_{c,\min}=\min_{c\mathbf{k}}\epsilon_{c\mathbf{k}},
\qquad
E_g=E_{c,\min}-E_{v,\max}>0.
$$

It is useful to expose two nonnegative spectral coordinates,

$$
A_{c\mathbf{k}}=\epsilon_{c\mathbf{k}}-E_{c,\min},
\qquad
B_{v\mathbf{k}}=E_{v,\max}-\epsilon_{v\mathbf{k}},
$$

so every included transition is

$$
\Delta_{cv\mathbf{k}}=E_g+A_{c\mathbf{k}}+B_{v\mathbf{k}}>0.
$$

This factorization is conceptual, not a valence-conduction pair loop. At each
time node LORRAX forms one empty-band Green function and one occupied-band
Green function,

$$
G_c(\mathbf r,\mathbf r',t)
 \sim \sum_{c\mathbf k}\psi_{c\mathbf k}(\mathbf r)
 \psi^*_{c\mathbf k}(\mathbf r')e^{-i\epsilon_{c\mathbf k}t},
$$

$$
G_v(\mathbf r,\mathbf r',-t)
 \sim \sum_{v\mathbf k}\psi_{v\mathbf k}(\mathbf r)
 \psi^*_{v\mathbf k}(\mathbf r')e^{+i\epsilon_{v\mathbf k}t},
$$

and contracts their product in the ISDF basis. This is the two-sums rule that
preserves cubic scaling.

For a complex sample $z$ in the upper half-plane, the transition kernel is

$$
K_z(\Delta)
=\frac{1}{z-\Delta}-\frac{1}{z+\Delta}
=-\frac{2\Delta}{\Delta^2-z^2}
=-2\int_0^\infty e^{izt}\sin(\Delta t)\,dt.
$$

The last identity is the common door for all complex-frequency chi samples.
The closed form is a scalar oracle; production evaluates the time integral
because one time node replaces the explicit transition-pair sum by separable
band sums.

## 3. The double-parallel sample grid

Let

$$
\omega_m=\max_{cv\mathbf{k}}\Delta_{cv\mathbf{k}}
$$

over the bands actually included in screening. The sampling service constructs
$N_p$ nested fractions $0=s_0<\cdots<s_{N_p-1}=1$ and two lines

$$
z_n^{(1)}=\omega_m s_n^\alpha+i\varpi_1,
\qquad
z_n^{(2)}=\omega_m s_n^\alpha+i\varpi_2,
\qquad \varpi_2>\varpi_1>0.
$$

For an insulator, the first near-line point is replaced by the exact static
point $z_0^{(1)}=0$. For a metal, it is replaced by
$z_0^{(1)}=i\,2\times10^{-5}$ Ry. That displacement is the published
stability device around zero-energy transitions; it is not the Sigma
broadening $\eta$. The far-line origin remains $i\varpi_2$.

The default `nested` fractions form a powers-of-two ladder near the origin and
then bisect the widest remaining interval. Increasing $N_p$ inserts points
without moving old ones. The paper tabulates the sets through $N_p=7$.
`mpa_sampling_schedule = leon` instead executes Yambo's qPPS integer
continuation; the two constructions agree through $N_p=8$ and differ from
$N_p=9$ onward. The exponent $\alpha=1$ gives the linear grid, while
$\alpha=2$ concentrates samples near zero. The schedule is a stamped choice,
not an adaptive fit parameter.

Neither $\alpha$ nor $N_p$ is a periodic-table lookup. In particular, the
published $\alpha=1$ result for one Na calculation is not a transferable
certificate for another k mesh, screening window, broadening, or compressed
basis. Metals can put much more structure near the origin than a uniform
real-coordinate budget resolves. Converge the pair $(\alpha,N_p)$ against a
held-out $W$ or, preferably, a matched full-frequency Sigma/QP referee. A
small backward error on the sampled points cannot detect an under-resolved
sampling manifold.

The grid therefore adapts automatically when deeper valence bands or more
conduction bands increase $\omega_m$: its real extent grows to the new maximum
transition. The number of samples remains $2N_p$, so a much broader spectrum
can still require a pole-count convergence check. Changing the band window,
sample geometry, centroid table, or frequency units invalidates the stamped
sample and pole stores; the readers compare these identities rather than
assuming that equal shapes mean equal physics.

`gw.mpa.sampling` owns only this geometry. `gw.mpa.sample_plan` classifies the
points and chooses an evaluator. Metallic plans are evaluated as well as
constructed; which kernel serves each metal point, and the measured reasoning,
belong to [Metallic MPA screening](metallic-mpa-screening.md).

## 4. How each chi sample is evaluated

The analytic character of a sample, not its line label, selects the scalar
quadrature.

| sample | scalar target apart from the common factor $-2$ | current route |
|---|---|---|
| $z=0$ | $1/\Delta$ | positive-interval Laplace minimax |
| $z=i\varpi$ | $\Delta/(\Delta^2+\varpi^2)$ | imaginary-axis Laplace minimax |
| $z=\omega$ | $\Delta/(\Delta^2-\omega^2)$ | real-axis service, only outside the transition interval |
| $z=\omega+i\varpi$ | $\Delta/(\Delta^2-z^2)$ | damped real-time contour |

The standard insulating MPA grid uses one exact static point, one pure
imaginary far-line point, and damped points everywhere else. It contains no
nonzero real-axis sample. The real-axis service remains in the common table
because other callers use it, but it refuses when a requested frequency lies
inside the transition interval.

For one damped line, define

$$
F_\chi=\Delta_{\max}+\max_{z\text{ on line}}|\operatorname{Re}z|.
$$

The current correctness rule truncates at

$$
t_{\max}=\frac{\log(2/\epsilon_\chi)}{\varpi}
$$

and partitions $[0,t_{\max}]$ into wavelength-sized panels. Each panel uses
positive Gauss-Legendre weights. Its order is graded downward as
$e^{-\varpi t}$ suppresses late panels. The fastest beat $F_\chi$, not the
transition bandwidth alone, determines the oscillatory resolution. With the
standard grid, $\max|\operatorname{Re}z|=\Delta_{\max}=\omega_m$, so the
line rule is sized for approximately $2\omega_m$.

Each positive atom $(t_j,h_j)$ becomes the two executable contour atoms

$$
(\tau,s,c)=(+it_j,+1,+ih_j),
\qquad
(\tau,s,c)=(-it_j,-1,-ih_j).
$$

Together they reproduce

$$
-2h_j e^{izt_j}\sin(\Delta t_j)
$$

without assuming an adjoint relation between independently fitted matrix
elements. All points on one horizontal line share the same $t_j$ and $h_j$.
Only their scalar projection coefficients differ, so one sequence of
expensive Green-function contractions produces every chi sample on that line.
The near line usually dominates because its smaller $\varpi$ gives the longer
time interval; wider damping on the far line is cheap.

At fixed accuracy the damped-line rank scales as

$$
N_\chi=\mathcal O\!\left(
\frac{F_\chi}{\varpi}\log\frac{1}{\epsilon_\chi}
\right).
$$

It is linear, not quadratic, in the transition bandwidth. It has no positive
gap precondition as long as $\varpi>0$. The static interval rule does require
a positive transition floor, which is why a complete metal implementation
must treat occupations and intraband weight explicitly rather than invent a
gap.

## 5. Disk-bounded chi, W, and pole fitting

The frequency axis is intentionally persistent. For each $z_j$, the response
is reduced to the irreducible q wedge and written collectively as one native
`P(None,'x','y')` slab. The file stores

$$
(N_z,N_{q,\mathrm{irr}},N_\mu,N_\mu)
$$

with frequency leading. SlabIO owns collective creation, filesystem striping,
padding, and writes. A small readiness entry is committed only after the
collective close, so an allocated but incomplete slab cannot be consumed as a
plausible zero.

The Dyson stage reads one complete chi slab, solves

$$
W(z_j)=\left[1-V\chi_0(z_j)\right]^{-1}V,
\qquad
W_c(z_j)=W(z_j)-V,
$$

writes one $W_c(z_j)$ slab, and releases both. The selected local or
distributed Dyson implementation is inherited from the ordinary screening
configuration; MPA does not own a second solve policy.

The fit reads all $N_z=2N_p$ frequencies only for a bounded block of matrix
columns. Rows and columns remain sharded over the process mesh. The default
tile budget is chosen so that no process materializes an $N_\mu^2$ object.
Each block is fitted, queued collectively to the q-wedge pole store, drained,
and released before the next block.  The drain is required before the next
W-file HDF5 read; it does not close the pole-store handle.  One SlabIO payload
handle and its pre-opened dataset handles remain live for the complete
q-major walk.  After that handle closes, one rank-zero metadata transaction
publishes the complete block journal and completion bitmap.  A failed payload
session therefore certifies none of its possibly partial bytes.  This same
bounded path is valid when chi and W are written on every QSGW iteration.

## 6. The Loewner multipole model

Each q-wedge matrix element is fitted independently to

$$
W_c(z)\approx\sum_{p=1}^{N_p}
\frac{2\Omega_pB_p}{z^2-\Omega_p^2}
=\sum_{p=1}^{N_p}
\left[\frac{B_p}{z-\Omega_p}-\frac{B_p}{z+\Omega_p}\right],
$$

with

$$
\Omega_p=a_p-i\Gamma_p,
\qquad a_p>0,
\qquad \Gamma_p\geq0.
$$

The default solver is the normalized Loewner pencil. It interpolates the same
$2N_p$ samples without forming a Vandermonde system. Two published diagnostic
routes are explicit deck choices. `mpa_pole_solver = companion` is Yambo's
optional linear-algebra (`LA`) construction; `mpa_pole_solver = thiele` is its
default Padé--Thiele (`PT`) reciprocal-difference recurrence and preserves the
stored near-line-then-far-line sample order exactly. The Thiele denominator is
converted to a companion matrix and diagonalized with the LAPACK/cuSOLVER
`geev` family, matching Yambo's root step. All three routes then use the same
LORRAX guards and all-$2N_p$-sample residue refit; selecting a pole solver does
not silently select a different physical ansatz.

For the optional LA construction, write $x_j=z_j^2$, split the stored samples
into the near and far halves, and set

$$
W_m=\max_{j<N_p}|x_j|,\qquad
Y_{jk}=\left(\frac{x_j}{W_m}\right)^k,\quad k=0,\ldots,N_p-1,
$$

$$
M_{jk}=-W_c(z_j)Y_{jk},\qquad v_j=W_c(z_j)x_j^{N_p}.
$$

With subscripts 1 and 2 denoting the near and far halves, respectively, the
published numerator elimination is

$$
T=Y_2Y_1^{-1},\qquad
(TM_1-M_2)b=Tv_1-v_2.
$$

The coefficients in the original $x$ domain are
$c_k=b_k/W_m^k$. The eigenvalues of the companion matrix with unit
subdiagonal and last column $-c$ are $\Omega_p^2$. Both linear systems are
solved directly, as in the published implementation: no row equilibration,
SVD truncation, affine map, or tuned regularizer participates in the
companion solve. The stored condition and backward-error gates take the worse
of the $Y_1$ inversion and the eliminated denominator solve, and retain both
component diagnostics.

After either pole finder, the same retarded-sheet guards, canonical ordering,
and unconstrained all-$2N_p$-sample residue least-squares refit run. Exact-zero
residues denote pruned poles and are ignored when Sigma geometry is planned.

The fit computes a condition estimate, backward error, sample residual, and
valid-pole count for every element. Only the condition map is retained as a
full tensor for debugging. Condition and backward error are reduced per block
into the compact completion ledger, and finalization requires

$$
\kappa\leq 1/r_{\mathrm{cond}},
\qquad
\epsilon_{\mathrm{back}}\leq\sqrt{\epsilon_{\mathrm{mach}}},
$$

using the solve's own $r_{\mathrm{cond}}$. These are numerical-stability
guards, not an observable-accuracy proof. The sample residual participates in
the in-memory finite-value refusal but is not stored; it does not replace a
held-out W reconstruction or QP convergence test.

Pole number is not an energy ordering. Any pole index can appear anywhere in
the fourth quadrant, and all pole fields are fitted independently. The Sigma
executor groups elements by their actual $(a_p,\Gamma_p)$ bounds; batches of
four consecutive pole indices are only an HBM schedule.

More poles are not monotonically better. On the measured Si case, moving from
eight to ten poles reduced typical sample residuals but changed registered QP
energies by as much as 15 meV and raised the full sharded fit to 224.9 s. The
ten-pole solve remained backward stable, so this was model-selection
sensitivity rather than a failed Loewner solve. Eight poles remain the
validated starting point; increase $N_p$ only with held-out W and QP evidence.

## 7. Sigma branches and literal retarded broadening

For one fitted pole and one band energy, every body-Sigma denominator can be
written as a causal resolvent with

$$
\Omega_p=a_p-i\Gamma_p,
\qquad
r=x+i(\Gamma_p+\eta),
\qquad
\frac{1}{r}=-i\int_0^\infty e^{irt}\,dt.
$$

Here $\eta>0$ is `sigma_regularization_ev` converted to Ry. It is a literal
external retarded broadening, not a fitted pole width and not an HGL routing
scale. The stored $\Gamma_p$ remains in $W(t)$, while the planner multiplies
every time weight by $e^{-\eta t}$. Thus $\eta$ enters once.

Using $E_A$ for the nonnegative empty-state energy or occupied-state hole
energy, the four insulating branches are

| frequency half | band space | denominator can cross zero? |
|---|---|---|
| $\omega\geq0$ | empty | yes |
| $\omega\geq0$ | occupied | no |
| $\omega<0$ | empty | no |
| $\omega<0$ | occupied | yes |

The table determines the usual topology, not the answer by decree. The
planner computes bounds from actual signed energies and live fitted poles. A
nominally sign-definite rectangle whose lower bound reaches zero refuses
instead of silently applying a divergent Laplace representation. Small,
inverted, and fractional-occupation systems are where that happens for a
physical reason rather than a planning mistake; the occupation weights and
the split that keeps the rectangles sign definite are owned by
[Metallic MPA screening](metallic-mpa-screening.md).

## 8. The core, electronic stripe, and plasmon slab

Let

$$
\omega_* = \max(|\omega_{\min}|,|\omega_{\max}|),
\qquad
T=\omega_*+m_{\mathrm{edge}}\eta.
$$

For each of the two crossing branches, the Cartesian product of band energies
and pole frequencies is partitioned into three disjoint regions:

| region | band selector | pole selector | method |
|---|---|---|---|
| crossing core | $E_A\leq T$ | $a_p\leq T$ | positive causal crossing |
| electronic stripe | $E_A>T$ | $a_p\leq T$ | complex-sector minimax |
| plasmon slab | all $E_A$ | $a_p>T$ | complex-sector minimax |

The other two causal branches are sign-definite over the full band and pole
ranges and each uses one sector window.

The core is deliberately overinclusive: a pair can satisfy $E_A+a_p>T$ even
when both coordinates are below $T$. The rectangular selectors are retained
because they permit separate band and pole sums. A selector depending on each
$(E_A,a_p)$ pair would reintroduce the pairwise object that the cubic method
was designed to avoid. The stripe and slab remove the large, safely separated
energies that would otherwise enlarge the expensive crossing bandwidth.

The edge factor does not broaden Sigma. It moves work between two exact
representations of the same $1/[x+i(\Gamma+\eta)]$: a larger edge makes the
sector rectangles easier but widens the crossing core; a smaller edge narrows
the core but moves sector bounds toward zero. Every cell is checked after the
cut, so a cost-oriented edge sweep is safe provided the observable is gated.

Geometry uses only live elements with $|B_p|>0$. A pruned zero-residue pole at
an extreme energy or width therefore cannot enlarge a window. Nonfinite
residues, live poles with $a_p\leq0$, and live poles with $\Gamma_p<0$ refuse.

## 9. Sign-definite complex-sector rules

Suppose a denominator $d$ lies in one open half-plane over a complete window.
Choose a contour direction $c=e^{i\theta}$ such that
$\operatorname{Re}(cd)>0$ throughout the window. Then

$$
\frac{1}{d}
=c\int_0^\infty e^{-cds}\,ds
\approx\sum_{j=1}^{N_s}w_j e^{-d\tau_j},
\qquad \tau_j=cs_j.
$$

`services/minimax.fit_damped_reciprocal` fits one rule over the union of the
actual complex rectangles belonging to a physical class. Several admissible
sector angles are tried. A pivoted-QR ordering selects support from an
overresolved rotated-Laplace dictionary, then nonnegative Lawson refitting
minimizes the complex relative residual

$$
R(d)=1-dQ(d).
$$

This construction explains the diagonal-looking rays of complex time nodes:
their angle is a contour chosen to keep every denominator decaying, not an
assumed universal $45^\circ$ optimum. Different rectangles may choose
different angles. The returned error is a dense sampled bound on the recorded
rectangles unless the selected asset explicitly carries a continuum
certificate.

At fixed accuracy, sector rank grows approximately as

$$
N_s=\mathcal O\!\left(
\log\frac{|d|_{\max}}{|d|_{\min}}
\log\frac{1}{\epsilon_s}
\right).
$$

The important limitation is not a large bandwidth but a vanishing
$|d|_{\min}$. Crossing zero makes a single contour impossible and routes the
cell to the causal real-time rule.

The current sector support dictionary is regenerated from the requested
tolerance. Consequently, nearby tolerances need not produce nested supports
or monotone QP errors. The measured Si sweep was stable at
$\epsilon_s=6.5\times10^{-4}$ but crossed a support cliff at
$7\times10^{-4}$. Treat the sector tolerance as a discrete plan choice: save
the provenance, compare complete plans, and do not infer convergence from the
ordering of the input numbers alone.

## 10. Positive causal crossing rule

For the core, define the fitted rectangle

$$
|x|\leq F,
\qquad
\gamma_{\min}\leq\gamma\leq\gamma_{\max},
\qquad
\gamma=\Gamma_p+\eta>0.
$$

The exact common representation is

$$
\frac{1}{x+i\gamma}
=-i\int_0^\infty e^{ixt-\gamma t}\,dt
\approx-i\sum_{j=1}^{N_\times}h_j e^{ixt_j-\gamma t_j},
\qquad h_j>0.
$$

One node set covers every shallow pole width and both crossing branches.
There is still a separate physical $G(t)W(t)$ contraction for the empty and
occupied band spaces, but adding more strongly damped poles does not require
a separate quadrature. Their factors $e^{-\Gamma_pt_j}$ are already present
in $W(t_j)$.

The production rule is not a free-node minimax fit. It searches the order of
one positive global Gauss-Legendre rule on $[0,t_{\max}]$. Several deterministic
tail allocations choose

$$
t_{\max}\sim\frac{\log(1/\epsilon_\times)}{\gamma_{\min}}.
$$

For each allocation, the order scan starts from the oscillatory
time-bandwidth floor $Ft_{\max}/4$. It does not treat $\gamma_{\max}$ as an
oscillation frequency. Because Gauss error can wiggle with order, the final
bracket is scanned instead of assuming monotonicity.

Candidate rules are scored on all four rectangle edges using a dense grid.
The validation grid is shifted by half a cell so it is disjoint from the fit
grid. Analyticity puts the interior maximum on the boundary, and a derivative
cover is recorded as a conservative continuum bound. If the global family
does not pass below its cap, an independently constructed positive panelled
rule is scored on the same boundary and used as a fallback.

Positivity supplies the stability certificate

$$
\kappa(\gamma_{\min})
=\gamma_{\min}\sum_j h_j e^{-\gamma_{\min}t_j}\approx1.
$$

Signed complex-Chebyshev compression and bounded-total-variation fits were
tested on the same boxes. They saved only a few nodes before amplification
increased, so positive Gauss was retained. The large gains came instead from
removing sign-definite high-energy regions from the crossing box and assigning
separate observable-tested error budgets to the two rule families.

At fixed width range and accuracy,

$$
N_\times=\mathcal O\!\left(
\frac{F}{\gamma_{\min}}\log\frac{1}{\epsilon_\times}
\right).
$$

This is linear in energy bandwidth, not quadratic. Reducing $\eta$ lengthens
the time interval and can raise the cost nearly in inverse proportion when
the fitted poles themselves are narrow.

The sector and crossing tolerances bound the same dimensionless residual
$|1-dQ(d)|$, so they are numerically comparable. They are separate controls
because a uniform scalar budget is not an optimal observable-error allocation.
On the measured Si case, loosening only the crossing budget from
$6.5\times10^{-4}$ to $2\times10^{-3}$ reduced the physical census from 478
to 446 without a measurable change relative to the 478-node plan at the
$5\times10^{-5}$ meV reporting scale.

### 10.1 The Landau floor, and the omega-clustered decomposition

The linear law above is not a defect of the order search.  Measured on the
sodium semicore scan, the production rule costs $N_\times = 87F + 10$ at
$\epsilon_\times=2\times10^{-3}$, $\eta=0.25$ eV — and a Landau-density
count for ANY stable exponential-sum representation
$Q(x)=\sum_l\alpha_l e^{-ixt_l}$ that is uniformly accurate on the
sign-symmetric window gives $N \gtrsim (F/\gamma_{\min})\log(1/\epsilon)/\pi
\approx 107F$ at these parameters: the target's Laplace content fills
$t\in[0,\log(1/\epsilon)/\gamma_{\min}]$ and the window is $2F$ wide.  The
global Gauss rule sits within tens of percent of the floor.  Free complex
nodes cannot beat it either: the rotated-contour trick that makes the
sign-definite sector family logarithmic (section 9) needs the domain inside
a sector $|\arg d|\le\pi/2-\beta$, and the crossing rectangle fills the
full sector as $F/\gamma\to\infty$ — a ray that decays for $x>0$ grows like
$e^{|x|\tau\sin\psi}$ for $x<0$.  Rational nodes in $x$ do not factor
$x=\omega-e-a$ and cannot ride the separable $\tau$ kernel at all.

What IS wrong is the certified region.  For any single evaluation
frequency, only the thin shell $|\omega-e-a|\lesssim$ (margins) crosses;
the rest of the $[\omega_{\min},\omega_{\max}]\times$(transitions) product
set is sign-definite and belongs to the logarithmic family.  The planner
therefore clusters each branch's $|\omega|$ values at gaps larger than
`mpa_sigma_omega_cluster_gap_ry` and, when there is more than one cluster,
splits the core per cluster at the crossing-edge margin $m$:

* bands $e < w_{\rm lo} - a_{\rm hi} - m$: the denominator
  $\omega-e-a+i\gamma$ keeps a positive real part — a rotated-Laplace fit
  in CONJUGATE node placement $t=+i\,\bar n$ (the retarded upper-half
  denominator is the conjugate of the fit family's lower-half domain);
* the shell: the positive causal rule of this section, with $F$ set by the
  CLUSTER span and the pole bracket — independent of the dynamic range;
* bands $e > w_{\rm hi} - a_{\rm lo} + m$: the plasmon-slab orientation of
  the sector family.

One cluster — every contiguous production grid — reproduces the monolithic
plan bit for bit.  The metallic `sd_core` sliver decomposes on the same
pattern (its $x=\omega+e+a$ crosses only where BOTH $\omega$ and $a$ are
within the excursion scale).  The evaluation grid itself is gapped with
`sigma_omega_patches_ev`, since $\Sigma(\omega)\to E$ interpolation is
piecewise linear and needs no points where no QP energy lives; a solved
energy inside a grid hole is a refusal at the QSGW seam.

Measured at production $\eta$ and tolerances on the synthetic Fe-class
geometry (valence window + one semicore cluster; every rule certified):

| evaluation span | total nodes | linear law | largest rule |
|---|---|---|---|
| 52 eV  | 44 | 344  | 18 |
| 105 eV | 45 | 679  | 18 |
| 209 eV | 51 | 1349 | 18 |
| 419 eV | 52 | 2689 | 18 |

The damped total is exactly flat; only the sector-slab rank creeps
logarithmically.  The cost is set by how many places the physics evaluates
$\Sigma$, never by how far apart they are.  Derivation, executor-safety
constraints (the conjugate placement grows factored exponentials, so the
references anchor at the mask maximum and masked bands are clamped), and
the rejected alternatives: `docs/dev/crossing-rule-cost-law.md`.

## 11. Shared spatial execution

At one scalar time node, MPA constructs

$$
G_A(t),
\qquad
W(t)=\sum_{p\in\mathcal W}B_p
e^{-i(\Omega_p-E_{B,\mathrm{ref}})t},
$$

then calls the ansatz-neutral spatial kernel

$$
G_k(t)\times W_q(t)
\longrightarrow
\operatorname{project}\!\left[
\mathcal F\{\mathcal F^{-1}G_k\;\mathcal F^{-1}W_q\}
\right]_{mn\mathbf k}.
$$

The fused FFT convolution and band projection are shared with GN-PPM. MPA
owns pole synthesis and scalar frequency coefficients, not a duplicate
Green-function builder or convolution. This is also the extension seam: an
alternative two-point $G$ or $W$ may supply the same tiles, while a genuine
three-point vertex requires a different contraction rather than another flag.

Only one $G(t)$, one $W(t)$, and one Sigma tile are live per node. The tile is
folded directly into every requested real frequency and discarded; no time
history is written. The number of output frequencies therefore changes the
cheap fold and storage work approximately linearly, not the expensive node
count. Widening the output interval changes the denominator geometry and can
increase that node count.

Pole fields are stored on the q wedge, read through SlabIO in batches of at
most four, and unfolded on device. Four is a memory bound, not a spectral
classification. If a logical window touches both four-pole batches, its
spatial sweep is executed once for each batch. The physical dispatch census is
therefore

$$
N_{\mathrm{physical}}=
\sum_w N_w\,m_w,
$$

where $m_w$ is the number of pole batches touched by window $w$. The current
eight-pole Si plan has eight logical windows, 12 physical sweeps, and 446 time
dispatches.

Spatial symmetry reduces storage and all non-FFT work on the irreducible q
wedge. Inputs are unfolded before the k-grid FFT convolution: time-reversal or
q-wedge shortcuts are not used inside that convolution.

## 12. Head, output, and QSGW boundary

The complete dynamic long-wavelength model would use

$$
S_{\mathrm{eff}}(z)=S_0(z)
+\frac{Y(z)W_{\mathrm{body},\Gamma}(z)Z(z)}{V_{\mathrm{cell}}}.
$$

`gw.head_correction.fold_cartesian_head_wings_sharded` owns this contraction
without gathering the body tile.  MPA now builds one frequency plan and gives
the exact same complex `z` array to the body and to the QSGW direct-head
response.  Each Dyson slab may finalize one head sample while total
`W_body,Gamma(z)` is resident; only the 3x3 result survives.  The scalar
`Wc_head(z) = W_head(z) - v_head` is fit with the same Loewner policy, guards,
and complex sample grid as the body and is published collectively through
SlabIO beside the body poles.

The MPA path supplies independent complex left/right centroid wings at every
sample.  The direct all-band contraction keeps the two stored centroid-
sharded wavefunction copies, distributes equal band-pair tiles over all
`Px*Py` ranks, and circulates a tile only around the mesh axis matching its
output wing.  Frequencies are blocked inside each ring, so the transition
weight temporary is bounded and no band-pair-by-centroid tensor is stored.
Each sample is folded through total W while that body slab is resident.

Each QSGW map call solves its own occupation state at entry, from the
spectrum of the Hamiltonian it was handed, and evaluates the direct head on
the full MPA grid with that state's chemical potential and occupations; the
same state reaches the finite-q body and Sigma.  There is no carry between
calls, which is what makes the map a function of its Hamiltonian alone.
[Metallic MPA screening](metallic-mpa-screening.md) owns that rule and its
consequences, what each consumer does with the state, which pieces are
threaded end to end and which are not, the capability-gate status, and the
measured self-consistency behaviour.

One-shot and diagonal fixed-point QP solvers can consume a finalized external
pole store.  Fully self-consistent QSGW rebuilds the body and head models
because each iteration changes the orbitals and transition energies.  The
bounded path is `chi(z)` q-wedge store -> one-slab Dyson and head finalization
-> `Wc(z)` q-wedge store -> bounded body-column and scalar-head Loewner fits.
Sigma subsequently reads and unfolds four body pole/residue slabs at a time
and reads the small head fit collectively.  Public `compute_mode = mpa` is
no longer gated at driver entry — the row was deleted at `9c9b23dc` once a
real metallic run traversed the complete chi/W/head/Sigma/QSGW chain — and
what that lift does and does not assert is stated once, in
[Metallic MPA screening](metallic-mpa-screening.md) §6.4.  Read this section
as landed plumbing: it is not itself a capability claim.


The MPA body applies the configured $\eta$ to every pole. The current generic
complex-pole head consumer uses the stored head pole without adding the same
$\eta$. Until that convention is deliberately resolved, convergence claims
apply to the body-Sigma quadrature and to comparisons that keep the head path
fixed; they are not a proof of a uniformly broadened total Sigma.

After body and head are available, the common dynamic-Sigma finalizer owns the
remaining operations: add the diagonal head, interpolate the matrix-valued
cube at the requested energies, write `sigma_mnk.h5`, build the static
Hermitian QSGW operator, and apply the existing outside-band scissor. A
self-consistent MPA calculation must rebuild chi, W, and the body pole fit
after each orbital update. Writing q-wedge chi, W, and pole stores on every
iteration is supported by the bounded dataflow above; no model tensor belongs
in `SCState`. When `sc_head_update = parallel_transport`, the fixed-DFT fallback head is replaced per iteration by the direct QP-basis head described above.

## 13. Validated starting profile

The exact parser defaults are listed only in the
[input reference](../input_reference.md). The following is the measured Si
profile, included here because it ties numerical choices to evidence rather
than merely repeating defaults:

```ini
minimax_target_error = 1e-6
minimax_max_nodes = 64

mpa_n_poles = 8
mpa_material_class = insulator
mpa_sampling_alpha = 1
mpa_sampling_schedule = nested
mpa_pole_solver = loewner
mpa_varpi_near_ry = 0.2
mpa_varpi_far_ry = 2.0
mpa_pole_batch_size = 4

mpa_sigma_sector_target_error = 6.5e-4
mpa_sigma_crossing_target_error = 2e-3
mpa_sigma_max_nodes = 96
sigma_regularization_ev = 0.25
sigma_window_edge_factor = 1.5
sigma_omega_layout = sharded
```

The measured output grid was $[-7,7]$ eV in $0.5$ eV steps. On four A100
GPUs, job 56958426 at commit `f29e5c34` used 446 physical time dispatches and
12 sweeps, taking 58.775 s in the Sigma stage and 81.993 s wall. Relative to a
plan with both scalar budgets tightened to $2\times10^{-6}$, the maximum QP
difference over 84 registered states was 0.07265 meV, the RMS difference was
0.04378 meV, and the direct-Gamma gap changed by 0.07424 meV. This is a
body-quadrature convergence statement for that system and fixed head, not a
universal accuracy bound or a comparison with BerkeleyGW.

The two error budgets play different roles. Keep chi at roughly $10^{-6}$
before tuning the pole count: sample noise is amplified by the rational fit.
The measured Sigma sector budget should also be treated conservatively because
its current support ladder is not nested. The positive crossing rule degraded
smoothly enough to use $2\times10^{-3}$ on the measured system. A new material
still needs a tight-plan comparison at the QP level.

## 14. What changes the cost or the answer

Use these dependencies when moving beyond the validated profile.

- **More screening bands.** The code recomputes $\omega_m$, stretches both
  sampling lines, and rebuilds the chi rules. Near-line rank grows roughly
  linearly with the new transition bandwidth. Keep the old pole count only if
  held-out W and QP values remain converged.

- **More bands in the Sigma sum — and how to extrapolate them.** Measured on
  a one-shot GN-PPM Si ladder at $n_b = 28/40/68$ with a band-matched
  BerkeleyGW arm at every rung (claim 197; the MPA arm of this ladder was not
  run, and §5.3b's broadening caveat forbids reading the GN-PPM numbers as
  MPA ones). **94.6%** of the mean-square 28-vs-68-band error is a *rigid
  shift* of the whole spectrum (BGW 92.7%): aligning each run on its own
  $\mu$ cuts RMS $|\Delta E_{QP}|$ from **787 meV to 186 meV**, within 1.6%
  of the best possible uniform shift, so there is no cleverer reference than
  per-run $\mu$. The residual is not noise but a clean linear stretch,
  $\Delta E = s + \alpha (E - \bar E)$ with $\alpha \approx +0.031$, which
  with the shift explains 99.2% of the variance; the two codes agree on $s$
  to 2% and on $\alpha$ to 4%, which is why the mechanism is read as
  truncating the $\Sigma_c$ intermediate-state sum rather than as an
  implementation difference. The tail is textbook $1/n_b$: a per-$(k,n)$ fit
  $E(n_b) = E_\infty + A/n_b$ over three widely spread rungs reproduces them
  to 15 meV RMS — a 52x lever on the 787 meV error it corrects — and beats
  $1/n_b^2$ by 4-5x in both codes. Practice, therefore: align on per-run
  $\mu$, drop the outermost band pair at each window edge (they can be
  non-monotone in $n_b$: on the BGW arm they take the all-band fit residual
  from 15.17 to 42.70 meV RMS and 56.1 to 346.6 meV max), fit $E_\infty +
  A/n_b$ on three rungs, and report the extrapolated correction beside the
  number — that deck still owes $-528$ meV mean at 68 bands. Two limits of
  this result, both measured: energy-*local* differences already cancel it
  (the direct gap at $\Gamma$ moves 5.5 meV while the levels move 787 meV)
  while wide-window differences do not (valence bandwidth $+268$ meV), and
  it does **not** carry to the screening channel at all — on the Na head
  ladder a best rigid $\omega$-shift removes only ~24% of RMS
  $|\Delta \chi^{00}|$ and a best amplitude rescale 39%, so the head's
  finite-band error is a change of spectral *shape* and a shift-and-stretch
  ansatz is wrong for $W$.

- **A wider Sigma interval.** Both $T$ and the crossing beat bandwidth $F$
  grow. Crossing rank is approximately linear in the added bandwidth; sector
  ranks grow logarithmically until a window boundary changes. A finer output
  step at fixed endpoints adds fold/storage work but no new time nodes.

- **A smaller $\eta$.** This changes the retarded observable and raises the
  crossing cost. Sweep it as a physical convergence parameter. Do not
  compensate by changing fitted $\Gamma_p$ or the chi line heights.

- **A different edge factor.** This changes only the core/sector partition.
  Compare the physical dispatch census before running, then gate the complete
  Sigma result. Zero is not intrinsically invalid, but a sector rectangle that
  reaches the origin will refuse.

- **A looser scalar tolerance.** Crossing Gauss ranks are nearly smooth but not
  mathematically monotone. Sector support can change discontinuously. Always
  compare the emitted plan provenance and the QP observable; never assume that
  a numerically smaller tolerance produced a nested rule.

- **More poles.** This adds two chi samples per pole, increases fit work, can
  create another partial pole batch, and need not improve the rational model.
  Check W reconstruction, conditioning distributions, QP energies, and gaps.

- **A different pole batch size.** This changes residency and repeated physical
  sweeps, not the approximation. The default remains four; controlled runs may
  request 1--8.  Na c620 on A100 completed with eight resident poles, while
  ten required a 7.33-GiB contiguous allocation and OOMed on every rank;
  values above eight are refused before reaching that measured failure.

- **Small-gap or metallic occupations.** Owned by
  [Metallic MPA screening](metallic-mpa-screening.md). The cost model changes
  in two places: the origin row leaves the sampled-quadrature family for an
  exact direct-frequency ordered-pair tile scan, and the damped-line bandwidth is set by
  the occupation supports rather than by $\omega_m$, so it grows with the
  smearing width as well as the band window. Sigma rank grows because the
  crossing core absorbs the Fermi-surface straddle.

## 15. Ownership map

| responsibility | owner |
|---|---|
| double-parallel points and nested fractions | `gw.mpa.sampling` |
| analytic sample classification | `gw.mpa.sample_plan` |
| damped chi and positive crossing scalar rules | `gw.mpa.evaluator` |
| reusable minimax and sector fitting | `services/minimax` |
| chi/W orchestration | `gw.mpa.model` |
| Loewner fit algebra | `gw.mpa.pade_fit` |
| sharded column walk | `gw.mpa.fit_driver` |
| sample and pole bytes | `file_io.mpa_store` through SlabIO |
| Sigma geometry and scalar windows | `gw.mpa.sigma_windows` |
| shared $G\times W$ spatial kernel | `gw.ppm_tau_kernel` |
| dynamic-Sigma output and QSGW finalization | `gw.sigma_dispatch` and `gw.dynamic_sigma` |

This boundary is intentional. Scalar quadrature services know no bands,
wavefunctions, q mesh, or file format. Physics planners know no HDF5 details.
SlabIO owns large distributed bytes. The shared spatial kernel receives
already-built $G$ and $W$ tiles and contains no pole-fit policy.
