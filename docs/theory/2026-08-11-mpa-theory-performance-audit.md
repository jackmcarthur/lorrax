# MPA accuracy and performance audit

Date: 2026-08-11  
Scope: the LORRAX multipole approximation (MPA), its residual difference from
the matched BerkeleyGW contour-deformation calculation for Si, and changes that
could make the method more robust and less expensive. Pole-axis batching is
deliberately out of scope.

## Executive conclusions

1. The remaining error for bands 7--10 is mostly a nearly rigid correlation
   self-energy offset, not a gap or dispersion error. Before alignment, the 32
   states have an Eqp0 mean/MAE of $+7.661/+7.661$ meV and a maximum error of
   $10.199$ meV. After subtracting the mean, the MAE is $1.682$ meV and the
   maximum is $2.577$ meV. Thus LORRAX already reproduces these bands across
   all eight irreducible $k$ points to better than 5 meV up to one common
   energy zero.

2. The residual is not explained by the final self-energy interpolation,
   BerkeleyGW's static remainder, or exchange. Replacing linear interpolation
   by PCHIP or cubic interpolation changes the mean by only about $0.03$ meV.
   BerkeleyGW's static-remainder correction is about $-297$ meV for these
   states, and the compared primed columns deliberately omit it. The exchange
   error is only $0.143$ meV MAE.

3. The finite-field/local-field head is not the source of the common
   $+7.66$ meV shift. The accepted 1394-centroid Schur-complement head
   reproduces BerkeleyGW's two head samples at $0$ and $2i$ Ry at the
   $10^{-4}$ level using independently estimated Coulomb heads, or at the
   $10^{-5}$ level when the same bare head is used. More decisively, an MPA
   scalar head changes occupied and empty on-shell states with opposite signs.
   The observed error decomposes into a $+7.661$ meV common component and only
   a $+1.333$ meV particle--hole-antisymmetric component. The head can affect
   the latter, but cannot generate the former.

4. The residual must therefore be separated between the body rational fit and
   LORRAX's subsequent complex-time evaluation. The latter is not the analytic
   pole convolution used in the original MPA/Yambo method: 6--15% of the pole
   elements in each Si pass are classified as narrower than
   $\xi=0.6667$ eV and sent through a two-point regularized route. That route
   is a serious, inexpensive-to-test candidate for a few-meV bias. If it is
   innocent, a body continuation or high-frequency-moment bias is the leading
   hypothesis. Existing pole-count data are oscillatory and confounded by an
   older 1128-centroid geometry.

5. The present MPA Sigma path is much more expensive than it needs to be. The
   eight Si pole passes made 52,252 tau dispatches, versus 167 for GN-PPM: a
   factor of 313 in node evaluations and about 47 in critical-path wall time
   when pole passes run concurrently. The optimization target should be

   \[
   C_\Sigma \simeq
   \sum_{p,b,g,w} N_\tau(p,b,g,w)\,C_\tau,
   \]

   not the number of output frequencies and not merely the number of routing
   groups.

6. The first structural priorities are: build a direct analytic-resolvent
   oracle for a small state/q sample; remove or certify the $\Gamma<\xi$
   substitution; enforce line-batched construction of $W(z)$; reuse transition
   amplitudes across every $z$ sample; constrain the fit's high-frequency
   moment; introduce held-out continuation validation; and jointly
   optimize/merge CTSP windows by total tau-node cost. Per-element quadrature
   grids should not be introduced naively: they can save nodes for one element
   while destroying the much more valuable reuse of $G(\tau)$.

## 1. What the Si result actually says

The reference is the matched BerkeleyGW contour-deformation run with 100 bands,
25 Ry screened and bare cutoffs, `frequency_dependence 2`, and
`exact_static_ch 1`. LORRAX is compared with the **primed** BerkeleyGW columns,
which are the finite-band result without the static-remainder addition.

For the deterministic eight-pole, 1394-centroid, $[-7,+7]$ eV run, the safe
protocol region is

\[
|\epsilon^{\rm DFT}_{n\mathbf{k}}-E_F|\leq 5\ {\rm eV},
\]

leaving a 2 eV interpolation margin. It contains 84 states over all eight
irreducible $k$ points. The main metrics are:

| Quantity | MAE (meV) | Mean LORRAX-BGW (meV) | Max abs. (meV) |
|---|---:|---:|---:|
| Eqp0, all 84 safe states | 7.726 | +6.594 | 14.561 |
| Correlation, all 84 states | 7.752 | +6.600 | 14.612 |
| Exchange, all 84 states | 0.111 | +0.003 | 0.352 |
| Eqp0, bands 7--10/all $k$ | 7.661 | +7.661 | 10.199 |
| Bands 7--10 after mean alignment | 1.682 | 0 | 2.577 |

The indirect- and direct-Gamma-gap errors are $-3.594$ and $-4.517$ meV.
Those gap errors and the aligned band errors are already at the requested
5 meV level. What remains is chiefly an absolute correlation-energy zero.

For the selected bands, the raw means are

\[
\overline{\delta E}_{v}=+8.993\ {\rm meV},\qquad
\overline{\delta E}_{c}=+6.328\ {\rm meV}.
\]

It is useful to resolve them into common and particle--hole-antisymmetric parts:

\[
\delta E_{\rm even}=\frac{\overline{\delta E}_{v}+
\overline{\delta E}_{c}}{2}=+7.661\ {\rm meV},
\]

\[
\delta E_{\rm odd}=\frac{\overline{\delta E}_{v}-
\overline{\delta E}_{c}}{2}=+1.333\ {\rm meV}.
\]

This decomposition is the most informative diagnostic in the current data.

## 2. Attribution of the 7.6 meV component

### 2.1 Final frequency interpolation: excluded

The stored Sigma cube has a 0.5 eV mesh over $[-7,+7]$ eV. Re-evaluating the
same cube at the DFT energies gives, for bands 7--10:

| Interpolant | MAE (meV) | Mean (meV) | Max abs. (meV) |
|---|---:|---:|---:|
| Current linear | 7.661 | +7.661 | 10.199 |
| PCHIP | 7.635 | +7.635 | 9.883 |
| Cubic | 7.633 | +7.633 | 9.889 |

The nonlinear interpolants move individual states by at most about 1.2 meV and
the mean by only 0.03 meV. They cannot account for the common offset.

### 2.2 Static remainder: excluded

The comparison uses BerkeleyGW's primed Eqp0 and correlation columns, so both
codes contain the same explicit 100-band sum and neither side includes the
static remainder. In the same BerkeleyGW output, the unprimed-minus-primed
static-remainder shift for bands 7--10 has mean magnitude $296.8$ meV and
ranges from about $-375$ to $-211$ meV. It is roughly forty times too large,
strongly state dependent, and has the wrong interpretation to explain the
observed residual.

### 2.3 Exchange and DFT state matching: excluded

The safe-set exchange MAE is $0.111$ meV, with maximum $0.352$ meV. The DFT
energy mapping differs by at most $0.007$ meV. The offset enters through the
correlation path.

### 2.4 Finite-field/local-field head: bounded, not the common offset

The accepted head is built from the long-wavelength Schur complement. In block
form, with the reciprocal-space dielectric matrix separated into head and body,

\[
\epsilon =
\begin{pmatrix}
\epsilon_{00} & \epsilon_{0G}\\
\epsilon_{G0} & \epsilon_{GG'}
\end{pmatrix},
\qquad
\epsilon_M^{-1}=
\left(
\epsilon_{00}-\epsilon_{0G}\epsilon_{GG'}^{-1}\epsilon_{G'0}
\right)^{-1}.
\]

The 1394-centroid artifact gives

| Sample | LORRAX $W_h$ (Ry) | BerkeleyGW $W_h$ (Ry) | Relative difference |
|---|---:|---:|---:|
| $z=0$ | 150.0791 | 150.0650 | $9.4\times10^{-5}$ |
| $z=2i$ Ry | 2343.3675 | 2343.1341 | $1.0\times10^{-4}$ |

Most of that displayed difference is the independently integrated bare head.
Using the same $v_h$, the screened inverse-dielectric differences are
$6.9\times10^{-6}$ and $1.0\times10^{-6}$, respectively. The fitted head
poles are causal and the sample interpolation residual is
$5.5\times10^{-10}$.

That last number must not be overinterpreted. The accepted head's held-out
near-real-strip residual is about 0.3195 even though it interpolates the 16
training samples essentially exactly and satisfies its transition-manifold
moment at $10^{-5}$. This is a direct demonstration that training residual,
pole quadrant, and one asymptotic moment do not certify continuation where
Sigma evaluates it. The occupied/empty sign test still excludes the head as
the source of the *common* 7.66 meV component, but real-axis head validation
remains important for the smaller odd component and for other materials.

BerkeleyGW's finite-field calculation does not simply omit the dynamic head.
Its `eps0mat` contains the dynamic $G=G'=0$ slot, while its small-cell
Coulomb treatment defines the corresponding bare head. The two programs use
different machinery to obtain the same limiting object; the direct sample
comparison above is a stronger test than trying to infer equivalence from code
names.

There is also a structural sign test. For the scalar MPA head, exactly on shell,

\[
\Sigma^{h}_{c,v}\propto
+\sum_p \frac{B^h_p}{\Omega_p-i\Gamma_p},\qquad
\Sigma^{h}_{c,c}\propto
-\sum_p \frac{B^h_p}{\Omega_p-i\Gamma_p}.
\]

Changing the head therefore shifts occupied and empty states in opposite
directions to leading order. Directly substituting older and accepted head fits
changes the selected states by approximately $+5.3$ meV for valence and
$-5.3$ meV for conduction, with essentially zero common mean. The observed
common $+7.661$ meV component cannot be caused by a missing BerkeleyGW head
correction. Only the $1.333$ meV odd component remains plausibly head-like.

### 2.5 f-sum rule: no evidence of a unit bug; moment accuracy remains relevant

For

\[
W_c(z)=\sum_{p=1}^{n_p} B_p
\left[\frac{1}{z-\Omega_p}-\frac{1}{z+\Omega_p}\right],
\]

the high-frequency expansion is

\[
W_c(z)=\frac{C_2}{z^2}+\frac{C_4}{z^4}+O(z^{-6}),
\qquad
C_2=2\sum_pB_p\Omega_p,
\quad C_4=2\sum_pB_p\Omega_p^3.
\]

The transition-manifold f-sum residual stored with the accepted fit is
$1.0\times10^{-5}$. A separate log value near 0.19 compares against a
classical plasma-frequency target and is not evidence of a Hartree/Rydberg
conversion error: converting both $B_p$ and $\Omega_p$ changes
$B_p\Omega_p$ by the required factor of four.

Nevertheless, sample interpolation and a global f-sum check do not prove that
every body element has the right high-frequency moments. A small systematic
error in

\[
M_1(\mathbf q;\mu\nu)=2\sum_p
B_{p,\mathbf q}^{\mu\nu}\Omega_{p,\mathbf q}^{\mu\nu}
\]

is a credible mechanism for a common Coulomb-hole-like shift. This is the most
important untested theoretical diagnostic.

### 2.6 Unresolved body ambiguity: rational fit versus Sigma evaluation

The evidence does not yet identify one unique line of code. It does narrow the
search to the dynamic body correlation construction, but that construction has
two separate approximations.

First, the rational fit may have a continuation or moment bias. Existing
body-only pole ladders are oscillatory: on an older 1128-centroid geometry, the
indirect gap moved by $-48.8$ meV from $n_p=8$ to 10 and $+42.7$ meV from 10
to 12. A diagnostic projection of the 12-pole-minus-8-pole body correction
onto the current near-gap states reduces a 7.07 meV MAE to 4.39 meV, but that
projection mixes centroid geometries and is not a result that should be banked.

Second, LORRAX does not insert every fitted body pole into the analytic MPA
self-energy formula. For the $\pm7$ eV output window the conditioning floor
sets $\xi=0.6667$ eV. Every fitted element with $\Gamma_p<\xi$ is treated as a
real pole by the legacy two-point crossing machinery and broadened at $\xi$.
Per pole pass, 6.09--15.29% of all elements take this route; their fractions of
total $|B|$ are 1.62--6.15%. Across all eight passes the corresponding totals
are 10.50% by count and 4.03% by residue mass. This replacement is absent from
the published MPA/Yambo convolution and is large on the scale of fitted widths.
It may be numerically benign after integration, but that has not been shown by
a direct-resolvent comparison. It is presently the cheapest serious candidate
for a few-meV bias.

The appropriate conclusion is therefore:

- compare the current complex-time answer to the exact fitted-pole denominator
  before changing the fit;
- the body fit is not demonstrated to be converged monotonically;
- its error has the correct common-sign character;
- a same-geometry nested ladder is required before choosing between sample
  placement, insufficient pole order, a moment constraint, and a Sigma
  quadrature bias.

## 3. Relation to the MPA and CTSP formulations

### 3.1 The rational ansatz agrees with the published MPA

The original multipole method represents each response element as

\[
X^{\rm MPA}(z)=\sum_{p=1}^{n_p}
\frac{2\Omega_pR_p}{z^2-\Omega_p^2},
\]

using $2n_p$ complex-frequency samples. LORRAX instead fits $W_c$ directly,

\[
W_c^{\rm MPA}(z)=\sum_{p=1}^{n_p}
\frac{2\Omega_pB_p}{z^2-\Omega_p^2},
\]

which is the same scalar rational structure. In a diagonal plane-wave Coulomb
basis, the two residues differ by frequency-independent multiplication by bare
Coulomb factors. In the ISDF basis, however, an elementwise pole is an
effective basis-dependent representation, not necessarily a literal physical
oscillator. Pole-by-pole interpretations should therefore be made cautiously;
the reconstructed matrix response and its spectral constraints are the
physical tests.

The published self-energy convolution is analytic once the pole model is
known:

\[
\Sigma_c(\omega)=\sum_{m,p}\mathcal V_{mp}
\left[
\frac{f_m}{\omega-\epsilon_m+\Omega_p-i\eta}
+\frac{1-f_m}{\omega-\epsilon_m-\Omega_p+i\eta}
\right],
\]

where $\mathcal V_{mp}$ contains the state vertices and pole residues. Yambo
follows this pole-denominator route. LORRAX's scalar head uses the same algebra,
but its body uses CTSP quadratures to retain low scaling in the ISDF
representation. That is a valid algorithmic hybrid, but it is not numerically
equivalent to Yambo until its quadrature and narrow-pole substitutions are
checked against the analytic expression.

### 3.2 Sampling and fitting: common design, different algorithms

| Feature | Original papers / Yambo | LORRAX |
|---|---|---|
| Fitted object | Polarizability/dielectric response | $W_c$ directly in ISDF |
| Samples | One shared complex grid | One shared near/far two-line grid |
| Fit | Padé or Padé--Thiele | Scaled fixed-support Loewner by default |
| Pole guards | Filter/coalesce/map, then residue refit | Local reflection/pruning/coalescence/range guards, then residue refit |
| Sigma | Analytic pole convolution | Analytic head; CTSP body with a narrow-pole fallback |
| Precision | Double recommended | x64 |

The papers and Yambo use one shared two-line grid for all response elements,
with near/far heights of roughly 0.1/1 Ha and $2n_p$ total points. This shared
grid is essential: element-specific frequency grids would require the union of
all requested frequencies in the expensive response/Dyson stage.

LORRAX follows this design, and its published real-axis partition agrees with
the original table through $n_p=7$. Its greedy continuation at larger orders
is LORRAX-specific and differs from current Yambo's dyadic `lP` construction.
This is not inherently wrong, but it means high-order behavior must be
validated rather than inferred from Yambo.

The difference is already numerically material: on the older 1128-centroid
body data, a current-Yambo-style ten-pole grid changed the result by about
$-49.7$ meV relative to LORRAX's ten-pole grid. This does not establish which
grid is better; it establishes that the high-order point set, not just
$n_p$, is a convergence parameter.

The fitting algorithm is also intentionally different. LORRAX defaults to a
scaled, fixed-support Loewner pencil with `rcond=1e-13`; the papers and Yambo
use linear-algebra Padé or Padé--Thiele, with Padé--Thiele the current Yambo
default. Yambo explicitly warns that accuracy need not improve monotonically
with pole count, recommends double precision, and advises no more than about 20
poles. LORRAX's Loewner formulation is defensible and often better conditioned,
but it needs model-order sensitivity, held-out continuation, and physical
matrix checks rather than a claim of numerical equivalence to Yambo.

LORRAX's guard sequence broadly follows the published practice: map poles into
an underdamped fourth-quadrant sector, remove or coalesce unsupported poles, and
refit residues over all samples. The precise thresholds and survivor rules are
local choices. Reflecting a pole is a model-changing repair, not by itself a
causality proof; fourth-quadrant scalar poles with complex elementwise residues
also do not guarantee matrix Hermiticity, passivity, or positive spectral
weight.

One convention statement should be corrected before extending the code to new
contours. With $\Omega=a-i\Gamma$, the time-ordered even response has poles at
$+\Omega$ in the fourth quadrant and $-\Omega$ in the second quadrant. It is
therefore not analytic throughout the entire upper half-plane. Sampling in the
first quadrant remains appropriate, but `sample_plan.py` currently describes
this using a global-upper-half-plane analyticity statement that conflates the
time-ordered and retarded sheets.

### 3.3 CTSP is an additional approximation layer

LORRAX separates two approximations that should be diagnosed independently:

1. a rational approximation of each screened-interaction element $W(z)$;
2. a complex-time separated evaluation of the resulting Sigma denominators.

For a fitted pole $\Omega_p=\omega_p-i\gamma_p$, the correlation denominator
has the generic form

\[
D=\omega-\epsilon_{m,\mathbf{k-q}}\mp\Omega_p.
\]

Within a sign-definite energy window, a Laplace representation separates its
state and pole dependence,

\[
\frac{1}{D}=s\int_0^\infty d\tau\,
e^{-sD\tau},
\qquad \operatorname{Re}(sD)>0,
\]

and crossing windows use a regularized/shifted representation. The minimax
rules approximate the exponential family over a bounded spectral interval.
Consequently, cost and accuracy depend on the **window envelope**, damping, and
node reuse, not only on how accurately the rational fit matches its training
samples.

The current crossing/noncrossing distinction is physically and numerically
well motivated and should be retained. The weak point is that routing creates
many conservative pole/group/window panes and evaluates a separate tau rule for
each. A rule optimal for an individual Lorentzian is not automatically optimal
for the full calculation: if it prevents reuse of $G(\tau)$, it can increase
total work even with fewer nodes per element.

The $\Gamma<\xi$ branch is more than a routing optimization. It replaces
$1/(u+i\Gamma_p)$ by a real-pole target regularized over a width $\xi$. The
current argument is that poles much narrower than the calculation's crossing
resolution are indistinguishable after smearing. That may be a useful numerical
model, but $\xi$ depends on the requested output window and the HGL conditioning
floor, while $\Gamma_p$ came from the fitted response. Results can therefore
acquire an output-window-dependent approximation not present in the MPA ansatz.
A direct analytic-denominator oracle and a controlled $\xi$ sweep are mandatory
before this route is considered generally robust.

### 3.4 Robustness beyond insulating Si

One insulating Si calculation cannot validate the method for metals, small-gap
systems, localized excitations, strongly anisotropic screening, or semicore
states. Published metallic MPA work introduces a tiny imaginary displacement
at the origin, low-energy quadratic sampling for Al/Cu, and a separate
$q\rightarrow0$ intraband correction. These are different analytic
requirements, not tuning details.

A general production protocol should select among at least insulating,
small-gap, and metallic sample plans; include the appropriate intraband and
long-wavelength limits; and certify each q/block by held-out matrix errors,
moment errors, causality, and passivity. The same global candidate pool can
still be shared, but its low-energy density and origin treatment must depend on
the system class.

## 4. Measured cost structure

### 4.1 W sampling and fitting

The double-parallel grid uses exactly

\[
n_z=2n_p
\]

screened-interaction samples. The dominant response work scales approximately
as

\[
C_W=O\!\left(2n_p n_q n_k n_v n_c n_\mu^2\right),
\]

while storage scales as $2n_p n_qn_\mu^2$. Pair amplitudes and transition
energies should be constructed once per $(q,k\hbox{-block})$ and scanned over
all $z$; the resolvent implementation supports this sharing pattern.

The strip has $2(n_p-1)$ points on two horizontal lines. The evaluator's
correctness default is currently `batching="per-point"`; `per-line` reduces the
number of strip sweeps from 14 to 2 for $n_p=8$, and from 18 to 2 for
$n_p=10$. It does not remove the Dyson solve for each sample, but can remove
repeated quadrature/transition sweeps.

The fit performs $n_qn_\mu^2$ small rational problems, with work growing
roughly as $n_p^3$. The old diagnostic path redundantly refit each element.
The current path removed the full refit, but can still avoid a second Padé solve
by returning the already-computed solve vector with the conditioning result.

Priority depends on the W producer. In the recorded Si exact-resolvent build,
the inner work for all 16 samples and eight q points was only about 9.2 s,
while fitting took about 411 s and each Sigma pole pass took roughly $10^3$ s.
Reducing 16 samples to 12 would therefore save only about 2.3 s in that run.
Line batching and adaptive sample counts can matter much more for the damped-
time W producer or larger systems, but the present Si bottleneck is Sigma first
and fitting second.

### 4.2 Sigma

The production pole-pass logs contain:

| Pole | Tau dispatches | Dispatch wall (s) | Pass wall (s) |
|---:|---:|---:|---:|
| 0 | 10,763 | 2,522 | 2,954 |
| 1 | 8,199 | 1,928 | 2,269 |
| 2 | 5,944 | 1,393 | 1,663 |
| 3 | 4,745 | 1,113 | 1,337 |
| 4 | 4,689 | 1,099 | 1,329 |
| 5 | 3,932 | 923 | 1,122 |
| 6 | 3,782 | 887 | 1,077 |
| 7 | 10,198 | 2,534 | 2,941 |
| **Total** | **52,252** | **12,399** | **14,692** |

The matched GN-PPM run made 167 tau dispatches, taking 12.8 s of a 63.1 s
total. Although the eight MPA poles can be fanned out, the longest pass is about
49 minutes, versus about one minute for GN. Pole 0 alone used panes containing
roughly 50--342 nodes. This is the main wall-time problem.

Reducing the number of requested output $\omega$ points will not reduce this
cost proportionally: one tau evaluation accumulates the whole output-frequency
vector. Likewise, narrower pole storage is not the primary issue here.

## 5. Recommended theoretical and algorithmic program

The priorities below distinguish changes that preserve the mathematical
approximation from research changes that alter it.

### Priority 0: add measurements, not a large test suite

For every production MPA run, record:

- $n_p,n_z,n_q,n_\mu,n_k,n_v,n_c,n_\omega$;
- W route counts, line sweeps, quadrature-node totals, and Dyson solves;
- fit dispatch count, solve count, backward error, training residual, held-out
  residual, and moment residual;
- for every Sigma pole/branch/group/window: placement, spectral bounds,
  damping, $N_\tau$, tau-kernel time, and accumulated residue norm.

The scalar objective for routing changes should be measured total tau-kernel
time or $\sum N_\tau C_\tau$, with QP error as a constraint. This can be added
as production diagnostics and output parsing; it does not require a broad new
unit-test campaign.

### Priority 1: isolate the body error with an analytic oracle

Before changing the W fit, evaluate a small stratified set of body
contributions from the existing pole store using the published analytic
denominators. Include near-gap valence and conduction targets; crossing and
noncrossing branches; narrow and broad poles; and several q/intermediate-state
tiles. Compare three numbers from identical $B_p,\Omega_p$ data:

1. the direct analytic denominator;
2. the exact-complex CTSP route with the fitted $\Gamma_p$;
3. the current route, including the $\Gamma_p<\xi$ substitution.

This distinguishes fit error, complex-time quadrature error, and width-floor
error without another W calculation. Sweep $\xi$ downward in the oracle and
report both count fraction, residue-mass fraction, and the actual contribution
to Sigma. The result should decide whether the 7.66 meV common shift is already
present in the rational model or is introduced downstream.

### Priority 2: make current operations share work

1. **Make per-line W evaluation the production default.** This changes 14 strip
   sweeps to 2 at $n_p=8$. Confirm from production provenance that the external
   W producer actually reaches this seam. Expected W-stage speedup is roughly
   2--8x depending on the fraction spent in the unavoidable sample-wise Dyson
   solves.

2. **Build transition amplitudes once and scan all $z$.** Keep the
   `chi0_resolvent` sharing structure in the production path. Stream q or column
   tiles to avoid materializing an unnecessarily large
   $(2n_p,n_\mu,n_\mu)$ tensor.

3. **Cache and merge compatible Sigma rules.** Key a rule by its placement,
   envelope, damping range, and requested tolerance. Merge adjacent routing
   groups only when the merged rule lowers total $\sum N_\tau$, not merely the
   group count. This preserves the existing crossing/noncrossing logic and
   should reduce repeated launches without changing pole physics. The existing
   experimental width-binning clause has already reduced documented node totals
   from 5541 to 885 and from 1393 to 245, about sixfold. It is the first
   low-risk Sigma timing A/B, provided the catalog certificate and final-output
   equivalence gates remain active.

4. **Use the shipped damped-line minimax catalog.** The selector exists but has
   no production consumer. Catalog lookup avoids cold runtime rule solves and
   should replace composite fallback where its certified envelope covers the
   requested near/far lines.

5. **Return conditioning data from the first fit solve.** Avoid the remaining
   second Padé solve. This is likely a 15--35% fit-stage improvement, not an
   end-to-end breakthrough.

### Priority 3: constrain and validate continuation, not just interpolation

The current fit can interpolate its training samples to nearly machine
precision while still giving a biased continuation between or beyond them.
For each representative q/block, split a nested sample pool into fitting and
held-out points and report both

\[
r_{\rm train}=\max_{z\in Z_{\rm fit}}
\frac{|W_{\rm MPA}(z)-W(z)|}{\max(|W(z)|,W_{\rm floor})},
\]

\[
r_{\rm hold}=\max_{z\in Z_{\rm hold}}
\frac{|W_{\rm MPA}(z)-W(z)|}{\max(|W(z)|,W_{\rm floor})}.
\]

Add a moment constraint or penalty,

\[
\min_{B,\Omega}
\left\|D\bigl(W_{\rm MPA}(Z)-W(Z)\bigr)\right\|_2^2
+\lambda_M\left\|2\sum_pB_p\Omega_p-M_1^{\rm target}\right\|_F^2,
\]

with scaling $D$ chosen by a physically meaningful absolute/relative floor.
The target should come directly from the transition manifold or an independently
computed high-frequency response, not the classical homogeneous-electron
plasma frequency. Causality/passivity checks should remain hard gates.

The current Loewner/scaled fit is preferable to an unscaled raw Padé solve.
Potential research alternatives are constrained vector fitting, a symmetry-
preserving AAA/Loewner selection, or shared-pole block fits. They should be
judged by held-out $W$, moment error, and final QP energies rather than by
training residual alone.

A practical conditioning audit should fit a deterministic ensemble of balanced
Loewner support partitions, sweep rank thresholds over roughly
$10^{-10}$--$10^{-14}$, and choose the smallest rank stable in held-out
Sigma-weighted errors and trusted moments. After pole selection, refit residues
with a weighted constrained problem,

\[
\min_B\|D(AB-W)\|_2^2+\lambda\|LB\|_2^2,
\qquad CB=m,
\]

where $D$ reflects sample uncertainty and Sigma influence, $L$ penalizes large
cancellation-dominated residues, and $CB=m$ imposes only trustworthy static or
transition-manifold moments. Do not hard-enforce a classical all-electron
f-sum when the transition space is truncated or a nonlocal pseudopotential
changes the velocity commutator.

### Priority 4: adapt a shared sample pool, not every element independently

The near/far double-parallel construction is valuable because it is nested and
lets many matrix elements share expensive $W(z)$ evaluations. Blindly choosing
a different complex grid for every $W_{\mu\nu}(z)$ would normally require the
union of all requested frequencies and can cost more than it saves.

A better adaptive design is:

1. build a small, nested global candidate pool on the near/far lines plus a few
   far-real or arc anchors;
2. evaluate the candidate samples in line batches;
3. choose a subset and/or pole order per q-block or spectral cluster using a
   matrix norm, held-out error, and moment residual;
4. retain a shared-pole representation inside each block when it is accurate,
   allowing element-dependent residues without element-dependent sample calls;
5. enrich the common pool only where a block fails certification.

This keeps the favorable $W$-construction sharing while allowing easy blocks
to use fewer effective samples. Far-real anchors are particularly useful for
the high-frequency moment; near-line points resolve low-energy structure. Arc
points can improve the real-axis strip but tend to demand more expensive
damped-line quadrature, so their end-to-end cost must be measured rather than
assumed.

### Priority 5: optimize CTSP windows jointly

The original complex-time method chooses energy windows to balance the number
of states/poles against quadrature order. LORRAX should solve the analogous
discrete optimization using the actual fitted-pole distribution:

\[
\min_{\mathcal P}
\sum_{w\in\mathcal P}
N_\tau(A_w,R_w,\gamma_w,\varepsilon)
\,C_\tau(N_{{\rm state},w})
\]

subject to certified approximation error and with an explicit reward for node
reuse. Here $\mathcal P$ partitions state/pole pairs, $A_w,R_w$ are the
window's spectral bounds, and $\gamma_w$ is its damping envelope.

Practical steps are:

- preserve separate crossing and noncrossing families;
- use actual state density and residue-weighted pole density when proposing
  splits;
- merge bins if the wider rule costs less than the sum of the narrow rules;
- construct a shared union of tau nodes at branch/window scope and reuse
  $G(\tau)$ across poles where interpolation of the scalar pole factor is
  certified;
- treat per-Lorentzian minimax rules as candidates inside this global cost
  problem, not automatically as the execution grid.

The last item is a research change. A logarithmic master tau grid with
barycentric or generalized-Gaussian weights may enable much more reuse, but it
must be tested against direct denominators because interpolation in tau can
amplify cancellation in crossing windows.

### Priority 6: prototype common-pole matrix or channel models

The exact matrix response has common excitation energies and matrix residues.
Independent elementwise fits discard that structure and make each denominator
depend on $(\mu,\nu)$, which is why the analytic MPA sum is difficult to retain
inside the low-scaling ISDF contraction. A reduced model

\[
W_c^q(z)\approx\sum_{p=1}^{K}
\frac{2\Omega_{pq}B_{pq}}{z^2-\Omega_{pq}^2}
\]

with matrix residues $B_{pq}$ would make the pole energy common within a q
block or channel cluster. It offers several advantages:

- pole locations become basis covariant within the modeled block;
- reciprocity, Hermiticity, and retarded spectral positivity can be imposed on
  matrix residues;
- the denominator no longer varies across every ISDF matrix element;
- shifted-G/resolvent builds can evaluate the body convolution analytically,
  as the scalar head already does, eliminating tau quadrature for accepted
  blocks.

A single global eight-pole matrix model will probably be too rigid. Prototype
block/tangential Loewner fits of dominant dielectric eigenchannels, clusters of
channels that share poles, or PSD factorizations of retarded spectral residues.
If there are $C$ pole clusters and one shifted build remains batched over q, a
rough shifted-build count is $2CKN_\omega$. It can beat the present time-domain
route when

\[
2CKN_\omega < N_{\tau,\mathrm{current}}.
\]

For $K=8$ and nine target energies this is 144 q-batched builds per cluster,
versus several thousand tau dispatches per current pole pass. If q cannot be
batched, the count acquires an additional $N_q$ and may lose outright.
Communication, transform cost, and attainable q batching therefore determine
the real break-even point; begin with a small q/channel prototype and the
direct-denominator oracle rather than a wholesale rewrite.

The newer MPA-Sigma idea--fitting Sigma itself from first- and third-quadrant
samples--is complementary. It is attractive for dense spectra or many output
energies, but it does not remove the cost of producing the initial Sigma
samples and is not the first optimization for a small near-gap QP set.

### Priority 7: reduce pole count only after continuation is certified

Because the current $n_p=8,10,12$ ladder is not converged on a common
geometry, lowering $n_p$ now would trade an unknown bias for speed. Once the
moment/held-out checks are in place, a material- or q-block-specific reduction
from eight to six poles would save 25% of W samples and roughly 40--60% of the
small fit linear algebra, but perhaps only 15--35% end to end when Sigma
dominates.

Do not improve timing by artificially broadening fitted poles. Their imaginary
parts enter the physical Sigma denominator and routing envelopes; changing them
changes the answer.

## 6. Minimal decisive calculation sequence

No large test expansion is needed. The following production comparisons answer
the open questions in order.

1. **Direct-denominator and width-floor audit.** On stratified q/state/mode
   tiles, compare the analytic fitted-pole denominator, the exact-complex CTSP
   result, and the current $\Gamma<\xi$ route. Sweep $\xi$ toward zero. This
   requires no new $W(z)$ calculation and is the fastest attribution test.

2. **Same-geometry nested fit audit.** If the first test leaves a rational-fit
   residual, extend the accepted 1394-centroid nested sample pool from
   $n_p=8$ to 10 and 12, evaluating only new $W(z)$ points. Record training,
   held-out, elementwise moment, causality, and backward-error metrics.

3. **Near-gap Sigma target.** Initially evaluate only bands 7--10 at all eight
   irreducible k points, with the same $[-7,+7]$ eV safe window. Compare raw,
   mean-aligned, and valence/conduction-even/odd components to BerkeleyGW's
   primed finite-band result.

4. **Moment-constrained A/B.** At the smallest pole order that still exhibits
   the common offset, refit the identical $W(z)$ samples with and without the
   transition-manifold $M_1$ constraint. Do not change Sigma routing in this
   comparison.

5. **Routing optimization benchmark.** Only after accuracy attribution, compare
   current panes, safely merged panes, and shared-node candidates. Bank total
   tau dispatches, total nodes, tau-kernel wall, and final Eqp differences.

6. **Optional BerkeleyGW head upper bound.** If a final head check is desired,
   vary only BerkeleyGW's small-cell/head averaging or replace only the scalar
   head contribution. Given the parity and sign decomposition above, this is a
   lower priority than the body ladder.

Success criteria for the first cycle are:

- raw bands 7--10 MAE ≤5 meV across all eight k points, without an empirical
  energy shift;
- gaps and mean-aligned dispersions ≤5 meV;
- a monotone or at least certified held-out convergence trend with pole order;
- a substantial reduction from 52,252 tau dispatches, with unchanged direct-
  denominator answers at the requested tolerance.

## 7. Concrete code touchpoints

- `src/gw/mpa/sampling.py`: nested double-parallel candidate grids and possible
  anchor enrichment.
- `src/gw/mpa/evaluator.py`: production `per-line` batching and W-route
  counters.
- `src/gw/mpa/chi0_resolvent.py`: transition-amplitude reuse across $z$.
- `src/gw/mpa/pade_fit.py` and `fit_driver.py`: moment-constrained fit,
  held-out certification, and removal of the second solve.
- `src/gw/mpa/sigma_pass.py` and `sigma_routing.py`: joint pane/window cost
  model, direct-denominator oracle, $\Gamma<\xi$ attribution, compatible-rule
  merging, and node reuse.
- `src/gw/mpa/sample_plan.py`: correct the time-ordered versus retarded
  analyticity wording before introducing any new contour.
- `services/minimax/src/minimax/damped_line_selector.py`: connect the existing
  catalog to production.
- `src/gw/mpa/head_dipole.py` and `sigma_head.py`: retain the accepted
  Schur-complement head and its independent diagnostics; do not tune it to
  absorb a body offset.

## 8. Guardrails

- Keep the bare and screened head averages on the same estimator.
- Keep causality, conjugation, and passivity checks as hard gates.
- Never select a model using only its training-point interpolation residual.
- Do not infer general convergence from the current mixed-geometry pole ladder.
- Do not attribute a common correlation offset to the scalar head without
  defeating the occupied/empty sign test.
- Do not optimize group count while increasing total tau nodes.
- Do not create independent per-element frequency or tau grids unless their
  global union and lost reuse are included in the cost model.
- Keep the user's requested test budget: production parsers/diagnostics may be
  expanded, but this investigation should not grow a broad unit-test suite.

## 9. Local evidence used in this audit

- MPA production output:
  `/pscratch/sd/j/jackm/mpa_geom_0810/combine_det_wide`
- Pole-pass logs:
  `/pscratch/sd/j/jackm/mpa_geom_0810/_reports`
- Accepted head artifact:
  `/pscratch/sd/j/jackm/mpa_headprobe_0810/graft/head_lf_1394.h5`
- Matched BerkeleyGW contour-deformation run:
  `/pscratch/sd/j/jackm/bgw_repro_0811/bgw_cd_esch`
- Matched BerkeleyGW GN run:
  `/pscratch/sd/j/jackm/bgw_repro_0811/bgw_gn`
- LORRAX method guide: `docs/mpa_method_guide.md`
- Archived CTSP derivation/reference: 
  `docs/dev/archive/misc/references/Kim-2020-MARKDOWN.md`

## References

1. D. A. Leon *et al.*, "Frequency dependence in GW made simple using a
   multipole approximation," *Phys. Rev. B* **104**, 115157 (2021):
   [DOI](https://doi.org/10.1103/PhysRevB.104.115157),
   [arXiv](https://arxiv.org/abs/2109.01532).
2. M. Kim, G. J. Martyna, and S. Ismail-Beigi, "Complex-time shredded
   propagator method for large-scale GW calculations," *Phys. Rev. B* **101**,
   035139 (2020):
   [DOI](https://doi.org/10.1103/PhysRevB.101.035139).
3. D. A. Leon *et al.*, metallic-system extension of the MPA, *Phys. Rev. B*
   **107**, 155130 (2023):
   [DOI](https://doi.org/10.1103/PhysRevB.107.155130),
   [arXiv](https://arxiv.org/abs/2301.02282).
4. The official Yambo
   [MPA tutorial](https://wiki.yambo-code.eu/wiki/index.php/Quasi-particles_and_Self-energy_within_the_Multipole_Approximation_%28MPA%29),
   with source snapshots for the
   [sampling grid](https://github.com/yambo-code/yambo/blob/19c12410fd9f70aee9bcab61221567433566510d/src/common/FREQUENCIES_mpa_sampling.F),
   [fit and pole guards](https://github.com/yambo-code/yambo/blob/19c12410fd9f70aee9bcab61221567433566510d/src/modules/mod_MPA.F),
   and
   [frequency defaults](https://github.com/yambo-code/yambo/blob/19c12410fd9f70aee9bcab61221567433566510d/src/modules/mod_frequency.F).
5. The 2025 MPA-Sigma proposal, a separate fit of the self-energy useful for
   dense output-energy or spectral-function workloads:
   [arXiv:2501.09121](https://arxiv.org/abs/2501.09121).
