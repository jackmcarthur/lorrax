# Frequency integration in the multipole GW path

This note specifies the frequency-dependent path from independent-particle
screening to the correlation self-energy,

$$
\{\psi_{n\mathbf k},\epsilon_{n\mathbf k}\}
\longrightarrow \chi^0(z_j)
\longrightarrow W_c(z_j)
\longrightarrow \{B_p,\Omega_p\}
\longrightarrow \Sigma_c(\omega_i).
$$

It describes the mathematical domains actually served by each quadrature,
the band and pole windows that define those domains, and the points at which
large arrays cross an I/O boundary. Energies and times below use one
consistent reciprocal unit. The MPA sample grid and current frequency-resolved
$W_c$ store are in Hartree; the fit store declares the inherited unit and the
$\Sigma$ reader converts it once to Rydberg. Overall spin, volume, Coulomb,
and Fourier normalizations are suppressed where they do not affect a
frequency convention.

## 1. The one transition kernel used throughout screening

The input wavefunction object defines disjoint logical valence and conduction
band slices. Those slices, not a quadrature tolerance, decide which bands
enter screening. For an ISDF centroid $\mu$, define

$$
M^{\mathbf q}_{cv\mathbf k}(\mu)
=\sum_s\psi^{\ast}_{c,\mathbf{k-q},s}(\mu)\psi_{v,\mathbf k,s}(\mu),
\qquad
\Delta^\mathbf q_{cv\mathbf k}
=\epsilon_{c,\mathbf{k-q}}-\epsilon_{v,\mathbf k}>0 .
$$

The wrapped momentum is exactly $\mathbf{k-q}$. The body of the independent-
particle polarizability is

$$
\chi^0_{\mu\nu}(\mathbf q,z)
\sim \sum_{\mathbf kvc}
M^{\mathbf q}_{cv\mathbf k}(\mu)
M^{\mathbf q\ast}_{cv\mathbf k}(\nu)
K_z(\Delta^\mathbf q_{cv\mathbf k}),
$$

with the resonant plus antiresonant kernel

$$
K_z(\Delta)
=-\frac{2\Delta}{\Delta^2-z^2}
=-\frac1{\Delta-z}-\frac1{\Delta+z}.
\tag{1}
$$

All production screening evaluations are direct evaluations of (1) at the
requested complex frequencies. No imaginary-frequency sampling followed by
analytic continuation occurs anywhere in this path.

For stable factorization, let

$$
\begin{aligned}
v_{\max}&=\max_v\epsilon_v,
&c_{\min}&=\min_c\epsilon_c,
&E_g&=c_{\min}-v_{\max},\\
A_{c\mathbf k}&=\epsilon_{c\mathbf k}-c_{\min}\ge0,
&B_{v\mathbf k}&=v_{\max}-\epsilon_{v\mathbf k}\ge0,\\
\Delta&=E_g+A+B.
\end{aligned}
\tag{2}
$$

The quadrature interval is therefore

$$
\Delta_{\min}=c_{\min}-v_{\max},\qquad
\Delta_{\max}=c_{\max}-v_{\min},
\tag{3}
$$

where all four extrema are over the included logical band slices. Equations
(2)--(3) size a rule; they do not remove any transition inside the slices.
A gapless static calculation has $\Delta_{\min}=0$ and is correctly rejected
by a positive-interval Laplace rule.

At a complex time $\tau$, the two spectral factors are schematically

$$
\begin{aligned}
G^c_{\mathbf k}(\tau)
&\sim\sum_c\psi_{c\mathbf k}\psi^{\ast}_{c\mathbf k}
e^{-\tau(\epsilon_{c\mathbf k}-c_{\min})},\\
G^v_{\mathbf k}(-\tau)
&\sim\sum_v\psi_{v\mathbf k}\psi^{\ast}_{v\mathbf k}
e^{-\tau(v_{\max}-\epsilon_{v\mathbf k})}.
\end{aligned}
\tag{4}
$$

Two forward $\mathbf k$ FFTs, the local product
$G^c_R G^{v\ast}_R$, and a final forward FFT produce the $\mathbf{k-q}$
convolution and exactly the residue orientation in (1). The current array
layout conjugates the returned conduction matrix, so the raw complex-time
builder is called at $\bar\tau$ on the conduction side and at $-\tau$ on the
valence side. This is only a layout compensation: the physical object remains
$G_c(\tau)G_v(-\tau)$. One $\mu\times\mu$ time tile is live at a time; no
history of (4) is stored.

## 2. Which complex frequencies are requested for the MPA fit

An $n_p$-pole fit receives exactly $2n_p$ screening samples on two horizontal
lines in the upper-half complex plane. Let $f_j\in[0,1]$ be the nested
semi-homogeneous dyadic partition and let $\omega_m=\Delta_{\max}$ for the
included transitions. The real coordinates are

$$
\omega_j=f_j^\alpha\omega_m,
\qquad
z_j^{\rm near}=\omega_j+i\varpi_1,
\qquad
z_j^{\rm far}=\omega_j+i\varpi_2,
\tag{5}
$$

with published defaults $\varpi_1=0.1$ Ha and $\varpi_2=1$ Ha
($0.2$ and $2$ Ry). The near line resolves low-energy structure; the far line
constrains the broad frequency envelope. The default exponent is $\alpha=1$
for insulators and Na and $\alpha=2$ for metals requiring denser low-frequency
sampling. Increasing $n_p$ inserts dyadic points without moving existing
ones.

For an insulator the first near point is replaced by $z=0$ exactly. For a
metal it is replaced by $z=i10^{-5}$ Ha to avoid a singular zero-energy
intraband sample. That displacement is a conditioning device, not a physical
broadening. The fit order is near line in ascending real part followed by far
line in ascending real part.

The two lines are interpolation support for the rational model. They do not
claim uniform accuracy on the real axis. Increasing their density or moving
their heights is a model-convergence study; it is not analytic continuation.

## 3. How each $\chi^0(z)$ sample is evaluated

Every upper-half-plane point belongs to one of four analytic cases. The
domains in the last column are strict: a rule must be rebuilt when the
transition interval, line height, real-frequency span, or requested point set
moves outside its fitted domain.

| $z$ | scalar target, apart from $-2$ | quadrature | domain served |
|---|---|---|---|
| $0$ | $1/\Delta$ | positive exponential minimax | $\Delta\in[\Delta_{\min},\Delta_{\max}]$, $\Delta_{\min}>0$ |
| $i\eta$ | $\Delta/(\Delta^2+\eta^2)$ | imaginary-axis exponential minimax | the same interval at that $\eta$ |
| $\omega$ | $\Delta/(\Delta^2-\omega^2)$ | two shifted reciprocal rules | shipped route only for $\omega>\Delta_{\max}$ |
| $\omega+i\eta$, $\eta>0$ | (1) | damped real-time or shared-sine rule | the specified $\Delta$ interval and frequency line or finite point set |

### 3.1 Static and imaginary-axis exponential rules

For $z=0$, a minimax exponential sum approximates

$$
\frac1\Delta\simeq\sum_{\ell=1}^L h_\ell e^{-t_\ell\Delta},
\qquad \Delta\in[\Delta_{\min},\Delta_{\max}],
\quad h_\ell,t_\ell>0.
\tag{6}
$$

The dimensionless problem is solved on $[1,R]$ with
$R=\Delta_{\max}/\Delta_{\min}$, then rescaled. Variable projection chooses
the nonlinear log-times, linear least squares chooses weights, and Lawson
iterations drive the sampled residual toward an equioscillating
$L_\infty$ error. Rank is increased until the requested error is reached.
Its growth is logarithmic in $R$; a single such rule is normally small.

At $z=i\eta$, the same procedure fits

$$
\frac{\Delta}{\Delta^2+\eta^2}
\simeq\sum_\ell h_\ell e^{-t_\ell\Delta}.
\tag{7}
$$

The dimensionless parameters are $R$ and $\eta/\Delta_{\min}$. A rule at one
$\eta$ is not silently used at another because (7) is a different function.
The static and pure-imaginary samples use the existing real-$\tau$
$G_c(\tau)G_v(-\tau)$ kernel.

### 3.2 Pure-real samples above the transition band

If $\omega>\Delta_{\max}$, the two terms of (1) are sign definite and each
can be mapped to a positive reciprocal interval. LORRAX builds the two
shifted $1/y$ minimaxes and represents one arm by signed imaginary-time
nodes. If $\omega$ lies inside the transition interval, (1) has a true real
pole. The pure-real route then refuses; a finite $\eta$ must be requested
instead.

### 3.3 Damped real-time rule for one horizontal line

For $z=\omega+i\eta$, $\eta>0$, use the exact identity

$$
K_z(\Delta)
=-2\int_0^\infty e^{izt}\sin(\Delta t)\,dt.
\tag{8}
$$

One positive composite Gauss--Legendre rule serves every point on a line of
fixed $\eta$. If $\epsilon$ is the requested error relative to the natural
$1/\eta$ scale, the tail is cut at

$$
t_{\max}=\frac{\log(2/\epsilon)}{\eta}.
\tag{9}
$$

The fastest beat is

$$
F_{\max}=\max_j|\operatorname{Re}z_j|+\Delta_{\max}.
\tag{10}
$$

The interval $[0,t_{\max}]$ is divided into panels of a fixed number of
wavelengths $2\pi/F_{\max}$. Each panel order is the smallest one satisfying
an analytic Bernstein-ellipse Gauss--Legendre bound at that panel's damped
envelope. Nodes and positive base weights are common to the line; only
$-2h_\ell e^{iz_jt_\ell}$ changes with $j$. Thus the method is accurate for
the declared line height, real span, and transition bandwidth. Reducing
$\eta$ raises both $t_{\max}$ and the number of oscillations, so near-real
resolution has an irreducible time-bandwidth cost.

### 3.4 Shared positive sine dictionary for a finite set of $z$

When only the $2n_p$ points of (5) are required, one may optimize a common
dictionary specifically for that set rather than certify the full rectangles
between them. Candidate times are supplied, usually on a logarithmic grid.
For fixed times the code solves the complex Chebyshev problem

$$
\min_{h_\ell\ge0}\ \max_{j,\Delta}
\left|K_{z_j}(\Delta)
+2\sum_\ell h_\ell e^{iz_jt_\ell}\sin(\Delta t_\ell)\right|.
\tag{11}
$$

The complex residual disk is bounded by a regular polygon and solved as a
linear program; insignificant support is pruned, the worst dense-grid
$\Delta$ points are exchanged into the solve set, and increasing candidate
ranks are tested until the requested sampled error is met. Positivity limits
amplification. Unless a separate interval proof is supplied, the reported
maximum is a sampled validation bound, not a continuum certificate.

Each sine atom is expanded exactly,

$$
\sin(\Delta t)=\frac{e^{i\Delta t}-e^{-i\Delta t}}{2i},
\tag{12}
$$

so a rank-$L$ sine rule executes as $2L$ contour nodes $\tau=\pm it$ through
the unchanged complex-time kernel. This is not analytic continuation and it
does not create accuracy between the fitted $z_j$ values. It is a direct
quadrature optimized for a finite request set.

### 3.5 General contour representation and batching

All of the complex exponential arms use

$$
\frac1d=c\int_0^\infty e^{-cdt}\,dt
\simeq c\sum_\ell h_\ell e^{-cdt_\ell},
\qquad \operatorname{Re}(cd)>0.
\tag{13}
$$

For the two resolvents in (1), a node has $\tau_\ell=c_\ell t_\ell$, a
branch label $s_\ell=+1$ for $\Delta-z$ or $-1$ for $\Delta+z$, and a
per-output coefficient

$$
a_{j\ell}=-c_\ell h_{j\ell}
\exp[-\tau_\ell E_g+s_\ell\tau_\ell z_j].
\tag{14}
$$

The GPU computes the transition tile once per union node and folds every
nonzero row of (14) into its requested output. Several $z_j$ therefore share
one time sweep without storing time history. The static real-time
specialization remains separate so its established real arithmetic is not
perturbed by the complex route. The explicit $kvc$ resolvent is retained as a
small-system oracle, not as the production algorithm.

## 4. From $\chi^0(z)$ to stored $W_c(z)$

For each requested $z_j$ and $\mathbf q$, the centroid-basis Dyson equation is

$$
W(z_j)=[I-V\chi^0(z_j)]^{-1}V,
\qquad W_c(z_j)=W(z_j)-V.
\tag{15}
$$

The dense solve is independent for every $(z_j,\mathbf q)$. The body tensors
remain sharded as $P(\mathrm{None},x,y)$ on $(q,\mu,\nu)$; the solve may donate
the $\chi^0$ buffer. The finite-$q$ body does not contain the $q\rightarrow0$
head or wings. Those are built from dipole/velocity transition matrix
elements at the same $z_j$ and are stored and fitted on a separate head axis.
The body $\chi^0(z_j)$ is an in-memory input to (15), not another disk
intermediate; the durable frequency slab is $W_c(z_j)$.

The MPA fit consumes $W_c$, never full $W$. The bare $V$ is frequency
independent, is already responsible for exchange, and would otherwise appear
as a spurious zero-time contribution to every fitted pole.

The frequency-resolved body store has logical shape

$$
(2n_p,N_q,N_\mu,N_\mu),
$$

with no device padding on disk. It may use the irreducible $q$ wedge only
when the centroid basis closes under the stored symmetry maps; otherwise it
uses the full $q$ zone. The frequency grid and one readiness bit per slab are
stored beside the tensor.

Production can write one $z_j$ slab directly from the sharded device array
with collective SlabIO: every MPI rank contributes its $\mu\nu$ hyperslab,
SlabIO clips padding, collective close makes the bytes durable, and only then
is the slab marked ready. A host path may instead write one complete slab
with serial h5py. The two transports produce the same logical layout; they
are alternatives, not a second copy. The fit refuses a store until all
$2n_p$ slabs are ready.

## 5. Multipole fit of $W_c$

Each matrix element is represented by

$$
W_c(z)=\sum_{p=1}^{n_p}B_p
\left(\frac1{z-\Omega_p}-\frac1{z+\Omega_p}\right)
=\sum_{p=1}^{n_p}\frac{2\Omega_pB_p}{z^2-\Omega_p^2},
\tag{16}
$$

with time-ordered poles

$$
\Omega_p=a_p-i\Gamma_p,\qquad a_p>0,\quad\Gamma_p\ge0.
\tag{17}
$$

The default denominator solve is a Loewner pencil in $x=z^2$: the $2n_p$
samples are split into left and right supports, the Loewner and shifted
Loewner matrices are formed, and their generalized eigenvalues give
$b_p=\Omega_p^2$. A normalized cross-multiplied Padé solve is retained as
an alternate numerical formulation. Both modes feed the same physical
post-processing:

1. reflect nonphysical overdamped roots in the $b$ plane;
2. enforce $\operatorname{Im}\Omega_p\le0$;
3. remove numerical null roots and coincident poles;
4. remove roots with nonpositive $a_p$, unsupported magnitude, or
   $\Gamma_p>a_p$;
5. sort by increasing $(\operatorname{Re}\Omega_p,\operatorname{Im}\Omega_p)$;
6. refit every surviving $B_p$ by complex least squares over all $2n_p$
   samples whenever any root was moved or removed.

The last step is mandatory: residues belonging to pre-guard poles are not
residues of the corrected model. Conditioning, backward error, achieved
sample residual, and surviving-pole count are recorded per fitted element.
Small sample residual is necessary but does not prove real-axis accuracy;
$n_p$, the two-line geometry, and the final observable must be converged.

The fit is tiled because its output, two arrays of shape
$(n_p,N_q,N_\mu,N_\mu)$, is larger than one frequency slab. The normal driver
walks $q$ and contiguous $\nu$ columns. It reads all $2n_p$ frequencies for
one column tile, fits its $N_\mu\times N_{\nu,\mathrm{tile}}$ elements in one
batched solve, and immediately writes the corresponding $\Omega_p$ and $B_p$
blocks with serial h5py. A completion ledger makes restart possible. The file
is finalized only after every $(q,\nu)$ block is present. Pole arrays fitted
on a $q$ wedge remain on that wedge; their unfold tables travel with the file
so $\Sigma$ can unfold one pole slab on read. The head samples and head poles
are written separately and atomically.

## 6. The four $\Sigma_c(\omega)$ branches

The requested real-frequency grid is specified relative to the Fermi
reference. Define positive A-side energies

$$
E_c=\epsilon_c-E_F\ge0,\qquad
H_v=E_F-\epsilon_v\ge0,
$$

on the configured self-energy band slice, with occupation masks selecting
empty or occupied states. More precisely, the internal A-side Green function
uses `wfns.slices.full`, while the external $(m,n)$ band projection uses
`wfns.slices.sigma`; occupations split the former into the $E_c$ and $H_v$
sets. For one fitted pole let

$$
S=E_A+a_p,\qquad \Gamma_p=-\operatorname{Im}\Omega_p.
$$

Splitting the grid into $\omega\ge0$ and $\omega<0$ gives four explicit
branches:

| grid half | A space | denominator class |
|---|---|---|
| $\omega\ge0$ | empty, $E_A=E_c$ | crossing, $\omega-S+i\Gamma_p$ |
| $\omega\ge0$ | occupied, $E_A=H_v$ | sign definite, $\omega+S-i\Gamma_p$ |
| $\omega<0$ | empty, $E_A=E_c$ | sign definite at $|\omega|$ |
| $\omega<0$ | occupied, $E_A=H_v$ | crossing at $|\omega|$ |

The negative half is evaluated explicitly and carries its own overall sign;
it is not manufactured by conjugating the positive half. All window bounds
below use

$$
\omega_{\max}=\max_i|\omega_i|.
\tag{18}
$$

Consequently a plan is accurate only on the grid range used to construct it.
Changing the grid endpoints requires replanning, although evaluation is done
only at the discrete requested $\omega_i$.

## 7. First partition: crossing versus noncrossing

There are only two physical denominator classes.

**Noncrossing.** The real part of the denominator has one sign on the entire
requested grid. A rotated Laplace contour can then make every exponential
decay. These rules have logarithmic dependence on the denominator dynamic
range.

**Crossing.** The real part can vanish inside the requested grid. Its fitted
width $\Gamma_p$ makes the exact pole finite when $\Gamma_p>0$, and a real-time
rule resolves the resulting Lorentzian. Very narrow poles are optionally
served by the older HGL regularized crossing rule. HGL is therefore a
numerical implementation of the narrow part of the crossing class, not a
third denominator class.

The narrow/wide boundary is

$$
\xi=\max\!\left[\xi_{\rm user},
\frac{2\omega_{\max}}{24-2e}\right],
\qquad
\text{narrow: }\Gamma_p<\xi,
\quad
\text{wide: }\Gamma_p\ge\xi,
\tag{19}
$$

where $e$ is the crossing edge factor. The floor keeps the dimensionless HGL
core bandwidth $2\omega_{\max}/\xi+2e\le24$, where its sine fit is
well-conditioned. The equality belongs to the wide branch, which retains the
fitted width exactly. No additional broadening is applied to a wide pole.

## 8. Recommended fixed-sector windows

The `sector` planner is the compact geometry. On each crossing branch set
$T=\omega_{\max}$ and partition the $(E_A,a_p)$ plane as

```text
                         a_p <= T                 a_p > T
                 +-------------------------+-------------------+
 E_A <= T        | crossing core           | deep B slab       |
                 |                         |                   |
 E_A > T         | high-A stripe           | deep B slab       |
                 +-------------------------+-------------------+
```

The inequalities are upper-closed on core/shallow windows and strict on the
complement: $E_A\le T$ versus $E_A>T$, $a_p\le T$ versus $a_p>T$. Thus every
selected A--pole pair appears exactly once.

### 8.1 Wide noncrossing: `sector:single`

All selected wide poles of a noncrossing branch share one window. For each
actual coupled pair $(a_p,\Gamma_p)$, define endpoint real parts

$$
(x_{\rm lo},x_{\rm hi})=
\begin{cases}
(E_{A,\min}+a_p,\ E_{A,\max}+a_p+\omega_{\max}), & \text{noncrossing},\\
(E_{A,\min}+a_p-\omega_{\max},\ E_{A,\max}+a_p), & \text{crossing side piece}.
\end{cases}
\tag{20}
$$

The rule bounds the actual denominator magnitudes by

$$
r_{\min}=\min_p\sqrt{x_{\rm lo,p}^2+\Gamma_p^2},\qquad
r_{\max}=\max_p\sqrt{x_{\rm hi,p}^2+\Gamma_p^2}.
\tag{21}
$$

For $d=x-i\Gamma$ in the fourth quadrant, rotate by $c=e^{i\pi/4}$:

$$
\frac1d=c\int_0^\infty e^{-cds}\,ds.
\tag{22}
$$

The substitution $s=e^y$ followed by an equally spaced trapezoidal rule in
$y$ gives the analytic sinc-sector rule. The step and finite $y$ range are
chosen from strip, small-$s$, and large-$s$ error bounds; rank depends mainly
on $\log(r_{\max}/r_{\min})$ and $\log(1/\epsilon)$. Width does not create
extra panes because (21) preserves the coupled $(x,\Gamma)$ geometry.

### 8.2 Wide crossing core: `sector:core:g*`

The core contains $E_A\le T$ and $a_p\le T$. Its poles are sorted by width and
split into geometric bands with $\Gamma_{\max}/\Gamma_{\min}\le4$; the
boundary belongs to the lower band. For one band,

$$
F_{\max}=\omega_{\max}
+\max(T-E_{A,\min},0)
+(a_{\max}-a_{\min}).
\tag{23}
$$

The exact crossing identity is

$$
\frac1{u+i\Gamma_p}
=-i\int_0^\infty e^{iut}e^{-\Gamma_pt}\,dt.
\tag{24}
$$

A positive composite rule is sized with the band's smallest $\Gamma$ and
(23): $t_{\max}=\log(2/\epsilon)/\Gamma_{\min}$, panels resolve
$F_{\max}$, and analytic Gauss--Legendre bounds choose their order. Every
pole retains its own $e^{-\Gamma_pt}$ in the GPU operand. The rule returns
the full complex resolvent; no sine-only projection or extra $i\delta$ is
used.

### 8.3 Wide crossing sides: `sector:a_stripe` and `sector:b_slab`

The stripe has $E_A>T$, $a_p\le T$; the slab has $a_p>T$ and every selected
$E_A$. Both are sign definite by construction and use the same sector rule
(20)--(22). This rectangular partition is deliberately conservative: moving
a pair from a side into the exact core costs nodes but stays correct, while
moving a truly crossing pair into a Laplace side would change the function.

### 8.4 Narrow crossing compatibility: HGL core plus sides

For $\Gamma_p<\xi$, the compatibility route treats the pole phase as real.
It uses $T=\omega_{\max}+e\xi$ and the same core/stripe/slab partition. The
core replaces the singular $1/u$ by

$$
\frac1\xi G_{\rm HGL}\!\left(\frac{u}{\xi}\right),
$$
$$
G_{\rm HGL}(x)=\operatorname{Im}\!\left[
\sqrt{\frac\pi2}e^{-(x+i)^2/2}
\left(1+i\,\operatorname{erfi}\frac{x+i}{\sqrt2}\right)\right].
\tag{25}
$$

An odd minimax fit

$$
G_{\rm HGL}(x)\simeq\sum_\ell\hat\alpha_\ell
\sin(\hat t_\ell x)
\tag{26}
$$

is built by nonlinear variable projection, Lawson reweighting, and a final
linear minimax solve. Physical nodes are
$t_\ell=\hat t_\ell/\xi$ and
$\alpha_\ell=\hat\alpha_\ell/\xi$. The accumulated one-sided imaginary
window is anti-Hermitian-completed once after all its nodes. The stripe and
slab are ordinary positive reciprocal minimaxes over the real-pole bounds

$$
x_{\min}=\max(E_{A,\min}+a_{\min}-\omega_{\max},\ e\xi),
\qquad
x_{\max}=E_{A,\max}+a_{\max}.
\tag{27}
$$

Here $\xi$ is an intentional crossing regularization: features within
$O(\xi)$ of $u=0$ are smeared, while the tails approach $1/u$. The route is
not an approximation to the exact finite-$\Gamma_p$ Lorentzian at the pole;
it is the accepted narrow-pole compatibility model.

On either noncrossing branch, the same narrow subset needs no HGL core: it is
treated as a real pole and receives one ordinary positive reciprocal minimax
over

$$
x_{\min}=E_{A,\min}+a_{\min},\qquad
x_{\max}=E_{A,\max}+a_{\max}+\omega_{\max}.
\tag{28}
$$

Thus every narrow pole still belongs to exactly one of the two physical
classes: one Laplace window if noncrossing, or HGL core plus sign-definite
sides if crossing.

## 9. Optional width-pane planner

The `pane` planner implements the same crossing/noncrossing physics with a
finer adaptive partition. It remains available for comparison and for domains
not represented by a fixed plan.

First, wide poles are bucketed geometrically in $a_p$ so a sign-definite
window's ratio does not exceed the configured $R_{\max}$ (default 100). Each
bucket is then split in $\Gamma_p$: crossing buckets use geometric width
bands with ratio at most 4; noncrossing buckets are split until
$\beta=\Gamma_{\max}/x_{\min}\le1$, or at a separately qualified fixed
width-bin ratio.

For a noncrossing pane,

$$
x_{\min}=E_{A,\min}+a_{\min},\qquad
x_{\max}=E_{A,\max}+a_{\max}+\omega_{\max}.
\tag{29}
$$

For a crossing pane, $z_e=e\Gamma_{\max}$ and
$T=\omega_{\max}+z_e$. The core uses (23)--(24), with its own pole spread;
the side bounds are

$$
x_{\min}=\max(E_{A,\min}+a_{\min}-\omega_{\max},z_e),
\qquad x_{\max}=E_{A,\max}+a_{\max}.
\tag{30}
$$

Unlike the sector rule, sign-definite pane rules place time on the negative
imaginary axis. The width $\Gamma_p$ is then a phase, and the panel rule is
sized by the decay $x_{\min}$ and oscillation
$\Gamma_{\max}+x_{\max}-x_{\min}$. The narrow $\Gamma_p<\xi$ subset still
uses the HGL compatibility route of section 8.4.

A single noncrossing minimax is indeed $O(\log R)$ and usually has only a
handful of nodes. Large historical totals arose because the pane planner
created many independent pole/width/branch windows and paid that small rank
once per window. Fixed-sector planning and shared rules remove this
multiplicity; the total contraction count must never be interpreted as the
rank of one reciprocal approximation.

## 10. Selecting and sharing $\Sigma$ rules

There are three legitimate ways a rule reaches a window:

1. an analytic construction, such as the sinc-sector or damped-line composite
   rules, whose truncation and panel bounds choose the smallest admitted rank;
2. a tabulated minimax, generated offline by increasing rank until its
   interval error target is met;
3. an actual-domain positive Chebyshev fit, using the same fixed-node polygon
   LP, support pruning, and exchange logic as section 3.4 on the set of
   denominators served by the window.

The third option is useful when a rectangular envelope is much larger than
the coupled physical set. Its rule is valid only for the recorded window
domain; dense sampled error is not a proof unless a continuum certificate is
also supplied. Tolerances should ultimately be selected by convergence of
$\Sigma$, quasiparticle energies, and gaps, because different windows carry
very different residue weight.

Quadrature rank and expensive contraction count are also different. If
several pole groups have identical nodes, A masks, branch signs, projectors,
and reference conventions, linearity permits

$$
W_\ell=\sum_{p\in\mathcal P}B_p\,P_p
e^{-i(\Omega_p-E_{B,p}^{\rm ref})t_\ell}
\tag{31}
$$

to be formed before the unchanged $G\,W$ convolution. One spatial contraction
then serves the whole pole group at node $\ell$. Exact crossing, HGL, side,
and noncrossing rules are shared only within their own compatible classes;
different A masks, branch signs, or projection rules cannot be merged. Pole
batches bound static host memory without changing (31). A compact plan stores
the A mask, six scalar bounds selecting $(a_p,\Gamma_p)$, references, signs,
projector, nodes, and achieved error for each window. It does not store a
dense pole mask. Such plans are baked offline and read as small inputs before
the pole walk; they are not frequency-dependent calculation intermediates.

## 11. The $\Sigma$ time sweep and its I/O

For each window and node, the runtime forms schematically

$$
G_A(t_\ell)\sim\sum_{n\in A}\psi_n\psi_n^{\ast}
e^{-i(E_{A,n}-E_A^{\rm ref})t_\ell},
$$
$$
W_p(t_\ell)=B_pP_p
e^{-i(\Omega_p-E_B^{\rm ref})t_\ell},
\tag{32}
$$

performs the existing $k/q$ FFT convolution and band projection, and folds
the resulting tile with

$$
C_\ell(\omega_i)=p_{\rm branch}\,\alpha_\ell
e^{-i(E_A^{\rm ref}+E_B^{\rm ref}-s_\omega\omega_i)t_\ell}.
\tag{33}
$$

Reference energies cancel between (32) and (33); they only control numerical
phase range. The device accumulator holds one sharded
$(N_\omega,N_k,N_m,N_n)$ result and one transient $\Sigma(t_\ell)$ tile. It
donates the accumulator through each multiply-add and stores no time history.
Full complex windows accumulate directly. HGL imaginary windows use a
separate window sum only long enough to apply their anti-Hermitian completion
once.

In the ordinary single-run path, no intermediate $\Sigma$ file is written.
Pole slabs are read one at a time, unfolded from the $q$ wedge when necessary,
summed in ascending pole order, and the final sharded arrays are written by
the standard collective `sigma_mnk.h5` output path. The $q\to0$ head
contribution is evaluated from its separately fitted head poles and injected
before quasiparticle interpolation.

For a pole- or window-farmed run, each leg instead writes one complete-shaped
`sigma_c_partial` cube with serial h5py after gathering its result. Such a
file is only a partial sum. The combiner reads the partials on the host,
checks exact coverage of every pole or window group, and adds them in the
canonical ascending order before the normal final output. No rank writes a
partial concurrently with another rank to the same HDF5 file.

Finally, $\Sigma_c(\omega_i)$ is interpolated only within the computed real
grid and combined with exchange, Hartree, and the head correction to obtain
quasiparticle quantities. A wider quasiparticle energy range requires a wider
$\omega$ grid and therefore new window bounds; neither the MPA fit nor a
quadrature licenses extrapolation beyond its declared frequency domain.

## 12. Minimal convergence hierarchy

Frequency errors should be tightened in this order:

1. converge the included valence/conduction and self-energy band slices;
2. converge direct $\chi^0(z_j)$ quadrature on the exact requested points;
3. converge $n_p$ and the double-parallel sample geometry in reconstructed
   $W_c$ and in final observables;
4. converge each $\Sigma$ window family, keeping the exact/HGL regularization
   choice explicit;
5. compare final quasiparticle energies and gaps on a fixed real-frequency
   grid.

The $\chi^0$ tolerance should normally be tighter than the final $\Sigma$
quadrature tolerance: noise in $W_c(z_j)$ is processed by a nonlinear pole
fit before it reaches $\Sigma$. A scalar rule error is therefore a design
gate, while convergence of $W_c$, fitted poles, $\Sigma$, and the requested
observable is the physical acceptance gate.
