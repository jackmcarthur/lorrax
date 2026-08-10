# The long-range exchange head and LT splitting

The electron–hole exchange kernel has one term with no limit at the zone
centre, and everything interesting about exciton dispersion near Γ follows from
it. This page is the short account: what the term is, why it splits
longitudinal from transverse, what LORRAX computes, and how far the mini-BZ
cell average may be trusted. Rydberg units throughout, so $e^{2}=2$ and
$v(\mathbf{q}) = 8\pi/(\Omega q^{2})$. The derivations at length and the full
measurement tables are in the campaign document named at the end.

## The head is a 0 × ∞ at G = 0

At exciton momentum $\mathbf{Q}$, in the pair basis $t = (c,v,\mathbf{k})$,

$$
K^{x}_{t t'}(\mathbf{Q})
= \sum_{\mathbf{G}} M^{*}_{cv}(\mathbf{k},\mathbf{Q},\mathbf{G})\,
  v(\mathbf{Q}+\mathbf{G})\,
  M_{c'v'}(\mathbf{k}',\mathbf{Q},\mathbf{G}),
\qquad
M_{cv} = \big\langle c,\mathbf{k}+\mathbf{Q}\big|\,
  e^{i(\mathbf{Q}+\mathbf{G})\cdot\mathbf{r}}\,\big|v,\mathbf{k}\big\rangle .
$$

Every $\mathbf{G}\neq0$ term is smooth as $\mathbf{Q}\to0$. The whole difficulty
is the single $\mathbf{G}=0$ term — **the head** — and it is a genuine
$0\times\infty$: the Coulomb factor diverges as $1/Q^{2}$ while the pair density
vanishes *exactly*, by orthogonality, since at $\mathbf{Q}=0$ the exponential is
unity and $M_{cv}(\mathbf{k},0,0)=\langle u_{c\mathbf{k}}|u_{v\mathbf{k}}\rangle
= 0$. To first order $M_{cv}(\mathbf{k},\mathbf{Q},0) =
-i\,\mathbf{Q}\cdot\mathbf{d}_{cv\mathbf{k}}$, with $\mathbf{d}$ the transition
dipole in velocity form, nonlocal commutator $i[\mathbf{r},V_{\rm NL}]$
included. The two behaviours multiply to something finite:

$$
\boxed{\;
K^{x,\rm head}(\mathbf{Q})
= \Big[\,|\mathbf{Q}|^{2}\,v(\mathbf{Q})\,\Big]\,
  \big(\hat{q}\cdot\mathbf{d}_{cv\mathbf{k}}\big)^{*}
  \big(\hat{q}\cdot\mathbf{d}_{c'v'\mathbf{k}'}\big)
  \;+\;\mathcal{O}(|\mathbf{Q}|).}
$$

The bracket is a scalar set by dimensionality; the second factor carries all of
the direction dependence and has no limit — it depends on the *direction*
$\hat q$ from which the origin is approached, and on nothing else. In three
dimensions $|Q|^{2}v(Q) = 8\pi/\Omega$ exactly, so the head is a nonzero
constant times $\cos^{2}\theta_{\mathbf{Q}}$ and the dispersion genuinely jumps
at Γ. Under two-dimensional truncation the bracket picks up $f_{\rm 2D} =
1 - e^{-z_{c}|\mathbf{q}_{\parallel}|}\cos(q_{z}z_{c}) \simeq
z_{c}|\mathbf{q}_{\parallel}|$, so the head vanishes *linearly* in $|Q|$ — still
nonanalytic, but continuous, a V rather than a step. Only the bilinear
$\mathbf{d}^{*}\!\otimes\mathbf{d}$ enters, so a phase convention on
$\mathbf{d}$ cancels while its magnitude does not, which is why the velocity
sign below is load-bearing.

## Why this is LT splitting

Strinati writes the same object as a lattice dipolar sum, "piecewise continuous
at $Q=0$ with a rapid angular variation about this point", and names it as the
origin of longitudinal–transverse splitting; Rohlfing and Louie put the
operational consequence in a sentence, that the *magnitude* of the photon
momentum is unimportant but its *direction* is not. The mechanism predicts the
degeneracy pattern rather than merely asserting it: for each fixed $\hat q$ the
head is an **outer product**, rank one, built from the single vector
$\hat q\cdot\mathbf{d}_{t}$. Restricted to an $N$-fold degenerate bright
multiplet, a rank-one perturbation shifts exactly one linear combination — the
one whose dipole is parallel to $\hat q$, the *longitudinal* exciton — and
leaves the other $N-1$ untouched. That is Qiu's "one V-shaped branch, $N-1$
parabolic" rule, and it follows from the factorisation, not from any material
detail. The MoS₂ deck shows it directly: at the smallest ladder point the two
lowest states move +0.006 and +0.003 meV while two others rise by tens of meV,
dark states untouched to six digits.

Which branch you want depends on the question. Light couples to transverse
excitons, so for absorption the unshifted branch is the physics; in an exciton
band structure $E_{S}(\mathbf{Q})$ the longitudinal branch is a real feature of
the dispersion and has to be there.

## What LORRAX computes

LORRAX never carries the transition index in the exchange tile — it lives in an
ISDF centroid basis indexed by $\mu$, and transitions appear only where the
matvec contracts the tile against the pair amplitudes. Both halves of that
contraction carry $\mathbf{q}$: the pair coefficients through the conduction leg
$\psi_{c}(\mathbf{k}+\mathbf{Q})$, and the interpolation vectors because
$\tilde\zeta$ is fitted per $\mathbf{q}$. Neither the Bloch phase nor the
interpolated form factor is singular at $\mathbf{q}=0$, so all of the
nonanalyticity lives in $v$. At every sampled finite $\mathbf{Q}$ the driver
therefore already forms $v(\mathbf{Q})|M_{cv}(\mathbf{k},\mathbf{Q},0)|^{2} =
|\mathbf{Q}|^{2}v(\mathbf{Q})|\hat q\cdot\mathbf{d}|^{2}$, which **is** the
exact head, direction dependence and all, with no small-$\mathbf{Q}$ expansion
and no approximation beyond ISDF — the LT structure along a band path rides on
the pair amplitudes, and an injected directional head would be a no-op or a
double count. At exactly $\mathbf{Q}=0$ the head is present as a matrix and
annihilated by the vertex: the loader adds it back with coefficient
$\langle v\rangle_{\rm mBZ}/\Omega$, and $\mathcal{A}(t) =
\sum_{\mu}C_{\mu}\overline{g^{0}_{\mu}} =
\langle u_{c\mathbf{k}}|u_{v\mathbf{k}}\rangle = 0$ kills it, so LORRAX reaches
BerkeleyGW's $\bar v(G=0)=0$ answer by cancellation rather than by fiat. What
survives is the ISDF error in that orthogonality, and it is measured: on a MoS₂
3×3×1 640-centroid deck the geometric prefactor
$\langle v\rangle/(\Omega N_{k})$ is 3564 meV while $|\mathcal{A}|$ is
$4.9\times10^{-3}$ worst, so the spurious contamination is **0.085 meV on the
worst transition and 0.022 meV rms** — three orders below the deck's binding
energies.

## The moment tensor, and the dipole route

For a sampled point meant to stand for its mini-BZ *cell* rather than for the
strict optical limit, averaging the Coulomb factor alone is not enough. For any
$\mathbf{D}$ constant over the cell,

$$
\big\langle v(\mathbf{q})\,|\mathbf{q}\cdot\mathbf{D}|^{2}\big\rangle_{\rm cell}
= \overline{D_{a}}\,\mathsf{M}_{ab}\,D_{b},
\qquad
\mathsf{M}_{ab} = \big\langle v(\mathbf{q})\,q_{a}q_{b}\big\rangle_{\rm cell},
$$

**exactly** — no expansion, the only hypothesis being that the dipole is a
property of the transition and not of the integration variable. Six numbers
carry the whole average, and the kernel is

$$
\boxed{\;
K^{x,\rm head}_{t t'}\Big|_{\rm cell\text{-}avg}
= \frac{1}{N_{k}}\;\overline{d_{a}(t)}\;\mathsf{M}_{ab}\;d_{b}(t') . }
$$

A scalar $\langle v\rangle$ is wrong twice over: in *direction*, keeping the one
sampled $\hat q_{0}$ where the cell holds a whole $v$-weighted distribution of
directions, of which $\mathsf{M}_{ab}$ is exactly the second moment; and in
*radius*, since the correct weight is $\langle vq^{2}\rangle$, not
$\langle v\rangle|\mathbf{Q}|^{2}$. Fixing either alone does not help.

The tensor converges with Γ inside the cell, because
$v\,q_{a}q_{b} = (8\pi/\Omega)\hat q_{a}\hat q_{b}$ is bounded even at the
origin — which hands over an exact, free diagnostic:

$$
\operatorname{tr}\mathsf{M} = \big\langle v(\mathbf{q})\,q^{2}\big\rangle_{\rm cell}
= \frac{8\pi}{\Omega}
\qquad\text{(3D, exactly, any cell shape, any offset).}
$$

The production sampler reproduces it to eleven digits on the silicon 4×4×4 cell.
Under slab truncation the identity reads the geometry off the kernel instead:
$\operatorname{tr}\mathsf{M}\to
(8\pi z_{c}/\Omega)\langle|\mathbf{q}_{\parallel}|\rangle\propto\Delta$, so the
head vanishes linearly with the cell rather than being cell-independent, and
$\mathsf{M}_{zz}$ is *identically* zero — rank two, confined to the plane. On
the hBN slab the trace halves with the grid across 3×3, 6×6 and 12×12,
following the mean in-plane momentum as it must.

Assembling this inside the $\mu$ basis would need
$\partial_{a}\tilde\zeta(\mathbf{q},\mu,0)$, which nothing in the tree computes.
It is not needed, because the $q$-linear coefficient of the head's
pair-amplitude factor **is** the dipole,
$\partial_{q_{a}}M_{cv}(\mathbf{k},\mathbf{q},0)|_{0} = -i\,d_{a,cv\mathbf{k}}$,
and $\mathbf{d}$ is what LORRAX already ships in `dipole.h5`. The term therefore
lives on the transition index, where the matvec already carries rank-three
objects: three inner products per trial vector and a 3×3 multiply. Hermiticity
is automatic since $\mathsf{M}$ is real symmetric, positive semidefiniteness
follows from $v\geq0$ sample by sample so the head can only push bright states
up, and the rank of at most three over transitions is the algebraic form of the
$N-1$ rule.

## The measured validity domain

The dipole route linearises the pair amplitude across the cell —
$\mathcal{O}(\Delta^{2})+\mathcal{O}(s\Delta)$ in the cell's size and offset,
controlled for a Γ-centred cell and degrading as the centre moves out. A ladder
that leaves the cell measures the domain directly. For a longitudinal dipole
write $R = \mathbf{d}^{*}\mathsf{M}\mathbf{d}\,/\,v(\mathbf{Q})
|\mathbf{Q}\cdot\mathbf{d}|^{2}$, the cell-averaged head over the pointwise head
it is supposed to converge to. On the MoS₂ 3×3×1 deck the mini-BZ face sits at
$t^{*}=1/6$ in crystal units while Γ→M runs to $t=0.5$, so the deck reaches
three times its own face:

| $t/t^{*}$ | 0.06 | 0.24 | 0.60 | 1.00 | 1.26 | 1.80 | 2.52 | 3.00 |
|---|---|---|---|---|---|---|---|---|
| $R$ | 2.769 | 0.869 | 0.584 | 0.657 | 0.746 | 0.859 | 0.933 | 0.966 |

$R$ passes through one at $t/t^{*}=0.201$ — one fifth of the way to the face,
deep inside the first cell, not at the face where the two were expected to meet.
Beyond the face the approach is the predicted $\mathcal{O}(\Delta^{2})$
quadrature error, a log–log fit giving exponent $-2.05$, and the two heads still
differ by 3.4 % at the zone boundary. The averaged head also sits *below* the
point value over almost the whole ladder: the truncated kernel is not $C/q$
there, because $f_{\rm 2D}$ saturates once $z_{c}q\gtrsim1$ ($z_{c}=11.34$ bohr
here, so from $t\approx0.07$) and $v(q)q^{2}$ turns concave, which puts a cell
average under its centre value. The linear 2D law is a statement about
$|\mathbf{Q}|\ll1/z_{c}$, and a nine-point grid's mini-BZ is much wider than
that.

What the ladder does *not* show is the two code arms converging: on the full
36-state spectrum their trace difference runs 30–74 meV per state and does not
decay outside the face at all. That is not the averaging error but the
representation difference the design accepted — the tensor arm contracts exact
Cartesian dipoles, the default routes the head through the ISDF $\mu$-tile — and
it is set by the 640-centroid basis rather than by $|\mathbf{Q}|$.

So the domain is narrow, and it belongs as a rule. **The averaged head is the
right object where a sampled point is being asked to stand for its cell**, which
is the Γ endpoint of an exciton band structure and the case it was built for. It
is **not** a better evaluation of the head at finite $\mathbf{Q}$ along a band
path: there the pointwise value is already exact, and the average is an
approximation to it, worst at $t/t^{*}\approx0.6$ and still 3 % out at the zone
boundary. Hence `head_minibz_average` is an explicit opt-in and defaults off.

## Conventions, and where the rest is

The head is *quadratic* in $\mathbf{d}$, so any convention touching the dipole's
magnitude propagates at full strength — the measured move across the
nonlocal-commutator sign was $+31.4\%\to0.00\%$ in a head quantity. That is
closed: the velocity sign is $+1$ everywhere as of 2026-08-09, the tracked
dipole fixtures were re-cut on that arm, and `dipole.h5` carries a
`prov_vnl_velocity_sign` attribute saying which arm wrote it (absent means a
pre-stamp, legacy file). Read the stamp rather than assume it. $\mathsf{M}$ is
the third rank-two $q$-space object in the tree and obeys the family rule in
[The S-tensor convention](s-tensor-convention.md) — Cartesian indices, and
declare which power of $q$ you are the coefficient of. It is the coefficient of
the *dipole bilinear* rather than of $q$, which makes it a companion to the
canonical Cartesian $q^{2}$-coefficient $S$ rather than a fourth spelling of it.

The full account — derivations, the error budget, the complete tables, and the
open questions, including whether the Γ exchange head should stay at zero — is
`~/lorrax_bse_perf_2026-08-08/THEORY_LT_HEAD_TENSOR.md`, with the implementation
record and the prediction scoring in `HEAD_TENSOR_IMPL.md` beside it. The
across-the-face ladder has its repo-visible record in
[the 2026-08-10 measurement report](../reports/LT_LADDER_ACROSS_THE_CELL_2026-08-10.md).
