# Multipole frequency integration

This chapter is the authoritative description of multipole-approach (MPA)
frequency sampling, fitting, and self-energy windowing in LORRAX. It follows
the data in execution order, because every later approximation is constrained
by an object defined earlier. Exact deck defaults belong to the
[input reference](../input_reference.md); equations and validity domains belong
here.

`compute_mode = mpa` is declared but still refuses at driver entry. The
disk-bounded chi-to-W-to-fit pipeline, the shared Sigma planner, and the common
dynamic-Sigma finalizer exist on this feature branch. Public enablement waits
for one real-material calculation to traverse that complete path with the
fixed-head approximation and pass its numerical gate. The refusal is owned by
`gw_config.UNIMPLEMENTED_MODES`; internal component tests do not weaken it.

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

The fractions form a powers-of-two ladder near the origin and then bisect the
widest remaining interval. Increasing $N_p$ inserts points without moving old
ones. The published sets are reproduced exactly through $N_p=7$; the stated
greedy extension first matters above nine poles. The exponent $\alpha=1$
gives the linear grid, while $\alpha=2$ concentrates samples near zero.

The grid therefore adapts automatically when deeper valence bands or more
conduction bands increase $\omega_m$: its real extent grows to the new maximum
transition. The number of samples remains $2N_p$, so a much broader spectrum
can still require a pole-count convergence check. Changing the band window,
sample geometry, centroid table, or frequency units invalidates the stamped
sample and pole stores; the readers compare these identities rather than
assuming that equal shapes mean equal physics.

`gw.mpa.sampling` owns only this geometry. `gw.mpa.sample_plan` classifies the
points and chooses an evaluator. Constructing a metallic plan is supported;
evaluating it is not yet supported because occupation-weighted interband and
intraband chi and Sigma terms have not landed.

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
Each block is fitted, written collectively to the q-wedge pole store, and
released before the next block. This same path is valid when chi and W are
written on every QSGW iteration.

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
$2N_p$ samples as the earlier power-basis Padé solve without forming a
Vandermonde system. Companion roots are mapped to the retarded sheet,
canonicalized, and followed by a residue refit against all samples. Exact-zero
residues denote pruned poles and are ignored when Sigma geometry is planned.

The fit stores a condition estimate, backward error, sample residual, and
valid-pole count for every element. Finalization requires

$$
\kappa\leq 1/r_{\mathrm{cond}},
\qquad
\epsilon_{\mathrm{back}}\leq\sqrt{\epsilon_{\mathrm{mach}}},
$$

using the solve's own $r_{\mathrm{cond}}$. These are numerical-stability
guards, not an observable-accuracy proof. The sample residual is reported but
does not replace a held-out W reconstruction or QP convergence test.

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
instead of silently applying a divergent Laplace representation. Supporting
small, inverted, or fractional-occupation systems requires splitting such a
cell at the actual denominator boundary and supplying the missing occupation
physics.

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

The sharded $YWZ$ contraction exists, but the producer that rebuilds $S_0$,
$Y$, and $Z$ from the current orbitals is not connected. The staged MPA driver
therefore fits the established two-point DFT scalar head once and labels it
`fixed_dft_gn`. This approximation omits dynamic local-field head/wing
feedback. It is especially suspect near gap closure and under orbital
self-consistency.

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
in `SCState`.

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
  sweeps, not the approximation. The production interface refuses values above
  four until a larger resident batch is memory-certified.

- **Small-gap or metallic occupations.** The sample-grid functions exist, but
  the evaluator still refuses. Occupied and empty selection must come from
  actual occupations, fractional weights must enter both chi and Sigma, and any
  sign-straddling energy cell must be split before the mode can be called
  metallic.

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
