# The shared-pole screened interaction: physics and mathematics {#shared-pole-theory}

This page owns the physics and mathematics of LORRAX's **shared-pole**
representation of the correlation part of the screened interaction,
$W_c = W - v$, in the ISDF centroid basis: the exact spectral structure the
model reproduces, the matrix pencils its samples are reduced with, the error
theory that fixes the sampling, what the pole count of the model means, and
the empirical pole-count heuristic used to size production. The code that
implements each object — modules, carriers, shardings, receipt rows and
refusals — is [the shared-pole implementation page](../architecture/shared_pole_model.md);
deck keys are in the [input reference](../input_reference.md); the frequency
quadrature of $\Sigma$ itself is the
[Sigma quadrature problem](sigma-quadrature-problem.md), and the
self-consistent workflow around a moving $W$ is in
[self-consistency](../self_consistency.md#shared-pole-w-with-retained-quadrature).
Historical campaign measurements that are not restated here live in the
sandbox report tree and are cited by claim number.

> **Status (written 2026-09-18 against `main` = `a559896f`).**
> The route is implemented and gated for the **charge ($\mu\times\mu$) response
> on scalar and two-component decks**, on both the time-reversal-symmetric
> (TRS, "even") and the time-reversal-broken ("ordered") construction.
> Accepted self-energy measurements exist on Si (TRS), Na (TRS metal) and
> CrI₃ (SOC ferromagnet, ordered); the four-current CC/CT/TT/TC sectors, a
> metal self-energy reference above 86 bands, and production self-consistency
> are **not** part of this page's verified envelope. Section 9 states exactly
> what is certified and what is not.

## 1 The exact object

Write $n = N_\mu$ for the number of packed ISDF centroids, so that a response
object at momentum $q$ is an $n\times n$ matrix. All frequencies on this page
are in Ry unless a value is explicitly given in eV; $z$ is a complex frequency
in the upper half plane, $s = z^2$, and matrices act in the centroid basis
with the Coulomb congruence $v_q$ of the ISDF solver.

The retarded RPA response dressed by $v$ has the Lehmann form

$$
W_q(z) - v_q
  \;=\; \sum_{n\in q}\frac{R_{+,n}(q)}{z - E_n}
      \;-\; \sum_{m\in -q}\frac{R_{-,m}(q)}{z + E_m},
\qquad R_{\pm}\succeq 0,\qquad E_n>0,
\tag{W 1}
$$

where the $E_n$ are particle–hole excitation energies of the deck (a finite
set for a finite k-grid and band window; a continuum in the thermodynamic
limit) and the residues $R_{\pm,n}\succeq0$ are positive semidefinite
$n\times n$ matrices in the centroid basis, assembled from the pair-density
vectors and the Coulomb congruence $v_q$. Only their positivity and the
pairing (W 2) are used below; the overall normalisation is carried by the
factor $b$ of Section 4. Equation (W 1) is the object the shared-pole route
compresses: everything below follows from its analytic structure, not from a
fit ansatz.

**Particle–hole pairing is exact and needs no time reversal.**

$$
R_-(q) \;=\; R_+(-q)^{\mathsf T}
\qquad\Longleftrightarrow\qquad
W_q(z) \;=\; W_{-q}(-z)^{\mathsf T}.
\tag{W 2}
$$

The negative-frequency side of $W_q$ is therefore not new information: it is
the transposed positive-frequency side of the **partner parent** $-q$. This is
why a store keeps only positive poles per parent and why $-q$ must be a
declared parent (or a realized mirror of one) on every admitted deck.

**Reality of the response in time** adds conjugation identities that hold with
or without time reversal:

$$
W_q(\bar z) = W_q(z)^\dagger,\qquad
W_q(-\bar z) = \overline{W_{-q}(z)},\qquad
W_{-q}(iu) = \overline{W_q(iu)}\quad(u\in\mathbb R).
\tag{W 3}
$$

**Time-reversal symmetry is the additional statement**
$W_{-q} = W_q^{\mathsf T}$; together with (W 2) it makes $W_q$ even in $z$, so
$W_q(z)=F_q(z^2)$. Without it, (W 1) has an odd part. Define

$$
W^{\rm even}_q(z) \equiv \tfrac12\big[W_q(z)+W_q(-z)\big]
=\tfrac12\big[W_q(z)+W_{-q}(z)^{\mathsf T}\big],\qquad
W^{\rm odd}_q(z) \equiv \tfrac12\big[W_q(z)-W_{-q}(z)^{\mathsf T}\big].
\tag{W 4}
$$

Writing the negative-frequency branch with its own energies $\tilde E_k$ and
residues $\tilde R_k$,

$$
W^{\rm even}_q(z)-v_q
 =\sum_n\frac{E_nR_{+,n}}{z^2-E_n^2}+\sum_k\frac{\tilde E_k\tilde R_k}{z^2-\tilde E_k^2},
\qquad
W^{\rm odd}_q(z)
 =z\Big[\sum_n\frac{R_{+,n}}{z^2-E_n^2}-\sum_k\frac{\tilde R_k}{z^2-\tilde E_k^2}\Big].
\tag{W 5}
$$

$W^{\rm even}$ is a time-reversal-symmetric interaction built from the same
data; $W^{\rm odd}$ is the part no TRS model can produce. At a time-reversal
invariant momentum (TRIM, $q\equiv-q$) with $\tilde R=R^{\mathsf T}=\bar R$,

$$
W^{\rm odd}(z)\;=\;2iz\sum_n\frac{\operatorname{Im}R_{+,n}}{z^2-E_n^2},
\tag{W 6}
$$

i.e. **the odd channel is the imaginary part of the same Hermitian residues**.
This identity is the reason the ordered construction costs storage equal to
the TRS construction, not double: the odd channel rides on the same vectors
(Section 4).

**Units and conventions.**

| quantity | object | unit |
|---|---|---|
| $W_c$, $v$ | $\mu\times\mu$ response matrices | Ry |
| $z$, $\Omega_j$, $E_n$ | frequencies | Ry |
| $s = z^2$, $\Lambda_j=\Omega_j^2$ | squared frequency | Ry² |
| $b_j$ (stored factor) | vector, one per pole | Ry$^{3/2}$ |
| $R_{+,j} = b_jb_j^\dagger/(2\Omega_j)$ | positive residue | Ry |
| $M_k$ | high-frequency moments of $W_c$ | Ry$^{k+2}$ |

The $\pm i\eta$ that makes a response retarded is applied by the consumer, not
stored in the model: the poles are real. A model pole of width $\kappa$ would
enter every $\Sigma$ denominator as $\eta+\kappa$, so a fitted width
$\kappa\lesssim\eta$ is indistinguishable from a real node after the consumer's
own smoothing (TASTE 64). The recipe therefore stores real $\Omega_j$ and
leaves broadening to one parameter, the deck's $\eta$.

## 2 The latent particle–hole pencil and passivity

In RPA the same object is a resolvent of a linear pencil,

$$
W_c(z)\;=\;C\,(z\sigma_3-M)^{-1}C^\dagger,\qquad
\sigma_3=\begin{pmatrix}I&0\\0&-I\end{pmatrix},\quad
M=\begin{pmatrix}A_1&B\\B^\dagger&A_2\end{pmatrix}=M^\dagger,\quad
C=v\big[\rho_q,\ \bar\rho_{-q}\big],
\tag{W 7}
$$

with $M\succeq\operatorname{diag}(\Delta E)\succ0$ for positive transition
energies and a repulsive $v$ (a stable RPA). The involution
$\mathcal P=\tau_x\circ\mathrm{conj}$ obeys
$\tau_x\sigma_3\tau_x=-\sigma_3$ identically; the relations
$\tau_x\bar M\tau_x=M$ and $C\tau_x=\bar C$ hold only at $q\equiv-q$, and at
generic $q$ map the pencil at $q$ to the pencil at $-q$ — equation (W 2) again.
Under TRS the pencil reduces to a Hermitian problem in $s$,

$$
W_c(z)=2\Phi\,(z^2 I-S)^{-1}\Phi^\dagger,\qquad
S=\Delta^{1/2}\big(\Delta+2P^\dagger vP\big)\Delta^{1/2},\quad
\Phi=vP\Delta^{1/2},
\tag{W 8}
$$

with $\Delta$ the diagonal transition energies and $P$ the pair-density
matrix in a real, time-reversal-adapted transition basis. So $W_c$ is a matrix
Stieltjes function, $W_c(s)=\int_E d\mu(t)/(s-t)$ with $\mu\succeq0$ on
$E\subset[\Omega_{\min}^2,\Omega_{\max}^2]$.

**Passivity.** Let $K=v^{1/2}\chi_0v^{1/2}$ with
$\operatorname{Herm}K\preceq0$ on the imaginary axis and $T=I-K$. Then
$\operatorname{Herm}T^{-1}=T^{-1}(\operatorname{Herm}T)T^{-\dagger}\succeq0$
and $TT^\dagger-\operatorname{Herm}T=-\operatorname{Herm}K+KK^\dagger\succeq0$
gives $\preceq I$, hence

$$
0\;\preceq\;\operatorname{Herm}\!\big[-v^{-1/2}W_c(i\eta)v^{-1/2}\big]\;\preceq\;I ,
\tag{W 9}
$$

which holds **with broken time reversal**. The bound sums both signs of real
frequency; the anti-Hermitian part is the odd channel and has no pointwise
sign. Equation (W 9) is the passivity gate of the constructor: it is a
necessary condition on any admissible $W$, checked on the model, never imposed
on it.

## 3 High-frequency moments

The large-$z$ expansion of (W 1) defines the moments the model must carry:

$$
W_c(z)=\sum_{k\ge0}\frac{2M_k}{z^{k+1}},\qquad
\begin{aligned}
M_0&=\tfrac12\Big(\sum R_+-\sum\tilde R\Big),&
M_1&=\tfrac12\Big(\sum ER_++\sum\tilde E\tilde R\Big),\\
M_2&=\tfrac12\Big(\sum E^2R_+-\sum\tilde E^2\tilde R\Big),&
M_3&=\tfrac12\Big(\sum E^3R_++\sum\tilde E^3\tilde R\Big).
\end{aligned}
\tag{W 10}
$$

$M_1,M_3$ are even under $q\to-q$ and $M_0,M_2$ are odd; under TRS
$M_0=M_2=0$. In the pencil's language the same objects are
$m_k=C(\sigma_3M)^k\sigma_3C^\dagger=2M_k$. For the model of Section 4 with
one factor vector per pole,

$$
M_1=\tfrac12\,b\,b^\dagger,\qquad
M_3=\tfrac12\,b\,\Lambda\,b^\dagger,\qquad
\Lambda=\operatorname{diag}(\Omega_j^2),
\tag{W 11}
$$

so $M_1$ and $M_3$ are exactly the blocks that fix the model's high-frequency
behaviour. This is the "infinity block" of the construction, and it is the
part of the model that carries the **rigid shift** of the self-energy.

**From band sums.** Let the bare response expand as
$\chi_0(z)=\sum_{p\ge1}X_p/z^p+\cdots$, with $X_1,X_3$ odd, from the
band-summed pair densities. The Dyson series then gives the coefficients $C_p$
of $W_c=\sum_p C_p/z^p$, keeping every cross term:

$$
\begin{aligned}
C_1&=vX_1v,\qquad
C_2=vX_2v+vX_1vX_1v,\\
C_3&=vX_3v+v\big(X_1vX_2+X_2vX_1\big)v+v(X_1v)^3,\\
C_4&=vX_4v+v\big(X_1vX_3+X_3vX_1+X_2vX_2\big)v
    +v\big(X_1vX_1vX_2+X_1vX_2vX_1+X_2vX_1vX_1\big)v+v(X_1v)^4,
\end{aligned}
\tag{W 12}
$$

and $M_{p-1}=C_p/2$. Under TRS $C_2=vX_2v$ is the $f$-sum rule and
$C_4=vX_4v+vX_2vX_2v$. On a finite band set $X_1$ does not vanish even for
the charge density, because densities commute only in the complete basis; so
the dimensionless ratio

$$
\frac{\lVert m_0\rVert}{\lVert M_1\rVert}
=\frac{2\lVert M_0\rVert}{\lVert M_1\rVert}
\qquad(\text{Ry}^{-1})
\tag{W 13}
$$

is a band-truncation diagnostic, not a physical observable. It is written into
the bank receipt (a measured value on CrI₃ was
$5.9\times10^{-4}$ Ry⁻¹ at $q=0$, maximum $2.7\times10^{-3}$ Ry⁻¹ over the 36
parents) and it is never used to refuse a deck.

## 4 The model

### 4.1 Stored form

One set of real poles per parent is shared by every matrix element, and each
pole carries one complex vector:

$$
W_c(q,z)=\sum_{j=1}^{K_q}\frac{b_j(q)\,b_j(q)^\dagger}{s-\Omega_j(q)^2},
\qquad \Omega_j>0,\ b_j\in\mathbb C^{n},
\tag{W 14}
$$

stored as $16\,n\,K_q$ bytes per parent ($b$ in complex128, $\Omega_j^2$ in
float64), plus the small per-parent count. Equivalently, in the pole-sum form
used by the self-energy consumer,

$$
W_c(q,z)=\sum_j\frac{R_{+,j}(q)}{z-\Omega_j(q)}-\sum_j\frac{R_{+,j}(q)}{z+\Omega_j(q)},
\qquad
R_{+,j}=\frac{b_jb_j^\dagger}{2\Omega_j}\ \succeq 0 .
\tag{W 15}
$$

The sign of the negative-frequency term in (W 15) is the particle–hole pairing
(W 2) restricted to the TRS case; the consumer never forms it as written — it
synthesizes the time-domain kernel $W_c(\tau)=-i\,b\,
\operatorname{diag}\!\big(e^{-i(\Omega_j-E_{\rm ref})\tau}/2\Omega_j\big)b^\dagger$
with one diagonal phase per pole per $\tau$ node, and reads the hole branch
from the partner parent (Section 7).

### 4.2 Ordered (time-reversal-broken) stored form

Each parent stores its positive poles only. Its negative-frequency side is the
partner parent's positive side, transposed:

$$
W_q(z)-v_q
=\sum_{j\in q}\frac{b_jb_j^\dagger}{2\Omega_j\,(z-\Omega_j)}
-\sum_{k\in -q}\frac{\bar b_k\,b_k^{\mathsf T}}{2\Omega_k\,(z+\Omega_k)} .
\tag{W 16}
$$

Particle–hole pairing (W 2) therefore holds **exactly by storage**, including
on a magnet. At $q\equiv-q$ this is

$$
W_q(z)-v_q
=\sum_j\frac{\operatorname{Re}(b_jb_j^\dagger)}{z^2-\Omega_j^2}
+i\sum_j\frac{\operatorname{Im}(b_jb_j^\dagger)\,z}{\Omega_j\,(z^2-\Omega_j^2)},
\tag{W 17}
$$

the even and odd channels on the same vector and the same pole set: the odd
channel costs no storage and no second model (equation (W 6) in the model's
variables). Under TRS the partner's factors are $\bar b$ and (W 16) is (W 14)
exactly. The number of stored parents is the magnetic irreducible wedge —
physics, not model choice.

### 4.3 What the model guarantees

| property | how it is guaranteed |
|---|---|
| real poles | Hermitian Ritz problem after whitening; $\mathcal H\succ0$ is sufficient for the ordered route, finite poles from nonzero $\mu_j$ |
| residues $\operatorname{sign}(\Omega)\times$PSD | by construction, (W 15) |
| particle–hole pairing $R_-(q)=R_+(-q)^{\mathsf T}$ | storage, (W 16) |
| odd channel at no storage | $\operatorname{Im}(bb^\dagger)$ on the same $b$, (W 17) |
| TRS data ⇒ ordered model = even model at equal retained span | paired basis and $v$-block cut, Appendix B |
| moments $M_0\ldots M_3$ | infinity rows of the pencil (Section 5.2) |
| passivity | equation (W 9) on the whitened model, checked, never imposed |

## 5 Construction: tangential interpolation as a rational Gauss quadrature

### 5.1 Samples, supports and directions

One real-time $\chi_0$ stream per parent evaluates $W_c$ and
$\partial_sW_c$ at a fixed set of **supports** $\{z_a\}$: points on the damped
line $z=\omega+ih$ and points on the imaginary axis $z=iu$. The stream never
forms $W$ at a real frequency; remote (core-to-conduction) transitions enter
through Laplace cells with kernel $d/(d^2-z^2)$, and the high-frequency
moments are exact band sums. The production ladder is 18 fitted supports; the
held supports used for diagnostics and the $M_1/M_3$ infinity block are
additional and are not counted in the 18.

The support placement is a condenser problem, not a choice of "interesting
frequencies":

* **Imaginary ladder.** $m$ points log-spaced on
  $[u_{\min},u_{\max}]$ with
  $u_{\min}=\max(4\eta,\text{gap})$, $u_{\max}=\max(16\ \mathrm{eV},L)$,
  $L=\omega_p+3.5\ \mathrm{eV}$ and $\kappa=L/u_{\min}$. The count is the
  Zolotarev rate of Section 5.4, equation (W 23).
* **Line ladder.** the line sits at height $h=\max(2.6\ \mathrm{eV},4\eta)$,
  independently of the imaginary ladder and Sigma broadening, and its sites are the
  quantiles of the **band-structure crossing density** raised to a power
  $\alpha$ (production $\alpha=\tfrac12$), where the crossing density is the
  $\eta$-broadened density of the pole differences $|E-\epsilon_{mk}|$ that
  contour deformation actually crosses for the deck's delivered states.
  The rule reads only band energies, occupations and $\eta$; it is not fitted
  to $W$ and it carries no material parameter. The measured self-energy is
  flat over $\alpha\in[0.25,0.75]$ on Si, and $\alpha=0$ and $\alpha=0.5$ are
  indistinguishable on Na, which is why the shipped rule takes
  $\alpha=0.5$.

At each fitted support the constructor selects a narrow direction set
$Q_a\in\mathbb C^{n\times r_a}$ from the sample itself:

* line supports: the right singular vectors of $W_c(z_a)$ above a relative
  cutoff ($10^{-3}$ in production) with a per-support rank cap of
  $\lceil N_\mu/16\rceil$ before multiplet closure;
* imaginary supports: the leading eigenvectors of
  $-\operatorname{Herm}W_c(iu)$;
* infinity: the leading eigenvectors of $M_1$.

The imaginary and infinity widths are upper bounds: directions below the
dense eigensystem's $N\epsilon_{64}$ relative spectral resolution are excluded
before forming the pencil. A numerical null direction carries no resolved
response and must not be amplified by diagonal Gram equilibration; the Gram
positivity and pole gates are unchanged.

The only sample data that reach the pencil are the actions
$O_a=W_c(z_a)Q_a$ and $D_a=\partial_sW_c(z_a)Q_a$. A line support's conjugate
partner is not a new sample: $W(\bar s)=W(s)^\dagger$ gives the partner's
action as $O_a$ itself.

### 5.2 The Hermite pencil

Write the model in factor form, $F(s)=B^\dagger(sI-S)^{-1}B$ with $B=b^\dagger$,
and define the rational Krylov states
$X_a=(s_aI-S)^{-1}B^\dagger Q_a$. The resolvent identity
$(\bar s_aI-S)^{-1}-(s_bI-S)^{-1}
=(s_b-\bar s_a)(\bar s_aI-S)^{-1}(s_bI-S)^{-1}$
gives every pencil entry from sample actions alone:

$$
G_{ab}=X_a^\dagger X_b=\frac{Q_a^\dagger O_b-O_a^\dagger Q_b}{s_b-\bar s_a},
\qquad
H_{ab}=X_a^\dagger SX_b=s_b\,G_{ab}-Q_a^\dagger O_b,
\tag{W 18}
$$

with the confluent limit
$G_{aa'}=-Q_a^\dagger\,\partial_sW(\bar s_a)Q_{a'}$, evaluated on the
partner's node. The infinity state $X_\infty=BQ_\infty$ closes the pencil with
the moments:

$$
G_{\infty b}=Q_\infty^\dagger O_b,\qquad
H_{\infty b}=s_bG_{\infty b}-2\,(M_1Q_\infty)^\dagger Q_b,\qquad
G_{\infty\infty}=2Q_\infty^\dagger M_1Q_\infty,\qquad
H_{\infty\infty}=2Q_\infty^\dagger M_3Q_\infty .
\tag{W 19}
$$

**Nothing is fitted.** Equation (W 18) *is* the tangential Hermite
interpolation condition at the supports, assembled from three matrix products
per block; the only place the derivative $\partial_sW$ enters is the confluent
block. The pair $(G,H)$ is Hermitian by construction; it is not a
least-squares normal-equation matrix.

### 5.3 Reduction

Equilibrate $G$ by $1/\sqrt{\operatorname{diag}G}$, whiten the retained
subspace (keep $\gamma>c\,\gamma_{\max}$, $c=10^{-8}$, truncated to the
recipe's pole budget), correct the retained metric by coupled Newton–Schulz,
and diagonalize the Hermitian problem

$$
\operatorname{Herm}(Z^\dagger HZ)\;=\;U\Lambda U^\dagger,\qquad
t_j=\lambda_j,\qquad \Omega_j=\sqrt{\lambda_j},\qquad
b=OZU .
\tag{W 20}
$$

Every Ritz value is then real and non-negative and every residue
$b_jb_j^\dagger/(2\Omega_j)$ is positive semidefinite **with no constraint
ever imposed**. The cut and the budget set $K_q$; poles at
$\lambda\le10^{-6}$ Ry² are dropped only within a factor-weight budget of
$10^{-6}$, so a model can never silently discard mass.

### 5.4 Why this works, and what sets the sampling

Rayleigh–Ritz on the rational Krylov space of the samples is a **rational
Gauss quadrature** of the spectral measure $\mu$: poles and weights are exact
on $\operatorname{span}\{1/(\sigma_k-t)\}$ at the supports, so for a linear
consumer $f$

$$
\Big|\int f\,d(\mu-\mu_K)\Big|
\;\le\;2\lVert\mu\rVert\;
\operatorname{dist}_E\!\big(f,\ \operatorname{span}\{1/(\sigma_k-t)\}\big),
\tag{W 21}
$$

and for a Cauchy kernel at $\zeta$ the error is *exactly* $F(\zeta)-F_K(\zeta)$.
With shifts $\sigma_a$, Ritz values $\lambda_j$,
$q(x)=\prod_j(x-\lambda_j)$ and $w(x)=\prod_a(x-\sigma_a)$, the Galerkin error
is $q(s)^{-1}\int_E q\,d\mu/(s-t)$ with orthogonality
$\int_E q\,t^k\,d\mu/w=0$, so

$$
\sup_F\lVert W-W_K\rVert
\;\le\;\frac{\max_E|R|}{\min_F|R|}\;\sup_F\int_E\frac{d\mu(t)}{|s-t|},
\qquad R=\frac{q^2}{w},
\tag{W 22}
$$

and minimizing the ratio is Zolotarev's third problem,
$\min_R\max_E|R|/\min_F|R|\sim4\exp(-\pi^2n/\ln4\kappa_{\rm spec})$, i.e.

$$
n(\epsilon)=\frac{\ln(4\kappa)\ln(4/\epsilon)}{\pi^2}
\tag{W 23}
$$

distinct shifts for relative accuracy $\epsilon$ on a condenser of ratio
$\kappa$ (Beckermann–Reichel; Druskin–Knizhnerman–Zaslavsky). The recipe's
imaginary count,
$m=\max\!\big(2,\operatorname{round}[\ln(16\kappa^2)\ln(4/\epsilon)/2\pi^2]\big)$
with $\epsilon=10^{-3}$, is (W 23) character for character, since
$\ln(16\kappa^2)/2=\ln(4\kappa)$.

**The laws place the supports; they do not count the poles.** $n(\epsilon)$
counts *shifts*. The number of poles $K_q$ is the retained rank of the
matrix-valued pencil, which the ISDF port count caps (Section 8). This is the
single most important distinction for reading the model's cost: frequency
resolution is logarithmic in the accuracy target, port coverage is not.

**Why the acceptance norm is a sup-norm on $\Sigma$'s set.** The self-energy
is linear in $W$, so for $\delta W=W-W_K$ on the consumer's evaluation set
$F=F_{\rm line}\cup F_{\rm im}$,

$$
\lVert\delta\Sigma\rVert\;\le\;\lVert\rho\rVert_1\,
\sup_{\zeta\in F}\lVert\delta W(\zeta)\rVert,
\tag{W 24}
$$

with $\rho$ the near-flat delivery density of the quadrature. An $L^2$ norm
weighted by $|W|^2$ controls nothing pointwise where $|W|$ is small; the
measured Spearman correlation between the whole-line $L^2$ error and the
contour-deformation non-rigid error is $-0.12$, while restricting to
$\omega\le10$ eV gives $+0.82$. Four families of interpolatory $H^2$
reductions (real IRKA, complex IRKA, weighted $H^2$, anchored/two-ended IRKA)
were built on the same deck and **never beat the prescribed construction on
the binding self-energy columns at matched pole count**; the mechanism is
that $z=0$ lies outside the Hardy domain of the damped-line norm, so no bound
of the form $|E(0)|\le C\lVert E\rVert_{H^2}$ exists, and $H^2$ controls
neither the inverse moments $A_0,A_1$ nor the low-frequency band that sets the
quasiparticle energies. The rigid (state-independent) shift of $\Sigma$
follows $M_1,M_3$; the binding columns follow the 0–2.5 eV band of the line
error, not the 0–14 eV aggregate. The prescribed supports supply all three
regions by construction; an $H^2$ fit on the line supplies none of them at
the same cost.

## 6 The time-reversal-broken construction

### 6.1 The projected particle–hole pencil

Take states $X_a=(z_a\sigma_3-M)^{-1}C^\dagger Q_a$ at supports $z_a$, with
outputs $O_a=CX_a=W_c(z_a)Q_a$. The same resolvent identity as (W 18), now
with $\sigma_3$, gives the projected pencil from samples alone:

$$
\mathcal G_{ab}=X_a^\dagger\sigma_3X_b
=\frac{(W_aQ_a)^\dagger Q_b-Q_a^\dagger W_bQ_b}{z_b-\bar z_a},
\qquad
\mathcal H_{ab}=X_a^\dagger MX_b=z_b\,\mathcal G_{ab}-(W_aQ_a)^\dagger Q_b,
\tag{W 25}
$$

with the confluent block
$\mathcal G_{aa'}=-Q_a^\dagger W'(\bar z_a)Q_{a'}$ and
$W'(z)=2z\,\partial_sW$. The structural change from the even route is that
**$\mathcal H$, not $\mathcal G$, is the definite member**. The infinity
states $k_0=\sigma_3C^\dagger Q_\infty$ and
$k_1=\sigma_3M\sigma_3C^\dagger Q_\infty$ carry the $z$-moments
$m_k=2M_k$:

$$
\mathcal G[k_0,k_1]=m_0,m_1,m_2,\qquad
\mathcal H[k_0,k_1]=m_1,m_2,m_3,
\tag{W 26}
$$

so the ordered infinity block needs all four moments of (W 10) and is
`NOT_MEASURED` when the bank carries only the TRS pair $M_1,M_3$.

### 6.2 Reality of the poles

Factor $\mathcal H=Y^{-\dagger}Y^{-1}$ (equilibrate, eigendecompose, cut) and
diagonalize $Y^\dagger\mathcal GY=U\operatorname{diag}(\mu)U^\dagger$. The
reduced model is

$$
F_r(z)=\sum_j\frac{c_jc_j^\dagger}{z\mu_j-1},\qquad
c=OYU,\qquad \Omega_j=\mu_j^{-1},\qquad
b_j=\sqrt2\,c_j/|\mu_j| ,
\tag{W 27}
$$

so every pole is real and every residue
$c_jc_j^\dagger/\mu_j=\operatorname{sign}(\Omega_j)\times$PSD.
**Sufficient condition:** $\mathcal H\succ0$ on the retained span permits this
Hermitian whitening and gives real finite poles; a stable RPA ($M\succ0$)
guarantees it for linearly independent retained states, and the validity gate
checks it, refusing rather than repairing. Reality of a general pencil's
spectrum does **not** imply $\mathcal H\succ0$: $\mathcal G=I$,
$\mathcal H=\operatorname{diag}(1,-1)$ has two real generalized eigenvalues
with indefinite $\mathcal H$. The earlier "if and only if" statement is
withdrawn. A zero $\mu_j$ denotes an infinite pole and requires the separate
infinity-weight policy.

### 6.3 Paired basis, cut and deduplication

On TRS data the two states $\pm z$ share one direction set. Write

$$
w_b=\tfrac12\big[X(z_b)+X(-z_b)\big],\qquad
v_b=\frac{X(z_b)-X(-z_b)}{2z_b},\qquad
w_\infty=k_1,\quad v_\infty=k_0 .
\tag{W 28}
$$

Then $\mathcal H' = \operatorname{diag}(H_s,G_s)$ and $\mathcal G'$ is
off-diagonal, with $(G_s,H_s)$ the even pencil of (W 18); eliminating
$v=zw$ gives $(z^2G_s-H_s)w=0$, the even problem in $s$ (Appendix B). **The
rule** is to pair the states, cut on the $v$-block, and apply that span to
both halves. The ordered Gram keep ratio is $10^{-7}$: the Fe $8^3$ response
bank retained weak directions at $10^{-8}$ that made the projected paired
pencil indefinite, while the tighter cut preserved the held $W$ checks.
Cutting instead on $\operatorname{diag}(H_s,G_s)$ keeps a
different span and is measurably worse (2–5× larger held error on TRS data,
and spurious poles at 100–1000 Ry). At imaginary supports, or line supports
with $\operatorname{Re}z=0$, the conjugate partner brings no new tangent when
$W$ is Hermitian there: keep only the component of $O=WQ$ orthogonal to $Q$
above the direction cutoff,

$$
\lambda\big(O_\perp O_\perp^\dagger\big)>c_{\rm dir}^2\,
\lambda_{\max}\big(OO^\dagger\big),\qquad c_{\rm dir}=10^{-3},
\tag{W 29}
$$

so a TRS bank adds no partner columns and the ordered model equals the even
one at equal retained span. Line roles with $\operatorname{Re}z\ne0$ keep
their full partner.

**Measured equivalence on TRS data.** On MoS₂ (3×3, two-component), the
paired-basis ordered construction keeps **equal pole counts** at production
and at equal retained rank, agrees with the even construction's held $W$ to
$1.1\times10^{-7}$, and its error against the data is 0.9997–1.00004 times the
even construction's error against the symmetric part of the same data. The
small residual is owned by the two extra retention cuts the ordered reducer
applies; the identity of Appendix B is exact only at equal retained span with
neither cut firing. This is the gate that keeps the ordered route honest on
nonmagnetic decks.

## 7 How the self-energy consumes the model

### 7.1 Contour deformation, exact without time reversal

With
$\Sigma^c(\omega)=\frac{i}{2\pi}\int d\omega'\,G(\omega-\omega')W_c(\omega')$,
rotating the frequency integral onto the imaginary axis for a state at energy
$E$ and an intermediate state $m$ with $x_m=E-\varepsilon_m$ gives

$$
\Sigma^c_n(E)=-\frac1{2\pi}\sum_m\int_0^\infty du\,
\Big[\frac{\langle W_q(iu)\rangle_{nm}}{x_m-iu}
     +\frac{\langle W_q(iu)^\dagger\rangle_{nm}}{x_m+iu}\Big]
\;+\;\sum_{m\in\mathcal X(E)}\pm\,
\big\langle W^{\rm res}_m\big\rangle_{nm},
\tag{W 30}
$$

$$
W^{\rm res}_m=
\begin{cases}
W_q(|x_m|+i\eta), & x_m\ge0\ \text{(unoccupied crossing)},\\[2pt]
W_{-q}(|x_m|+i\eta)^{\mathsf T}, & x_m<0\ \text{(occupied crossing)},
\end{cases}
\tag{W 31}
$$

where $\langle A\rangle_{nm}=p_{nm}^\dagger Ap_{nm}$ and $\mathcal X(E)$ is
the set of Green's-function poles the rotation crosses. The two
time-reversal steps used to reach (W 31) are exactly the ones an integrator
must undo on a magnet: under TRS the two terms of (W 30) fold into
$2x/(x^2+u^2)$ acting on $W_q(iu)$ and the occupied residue reads
$W_q(|x|+i\eta)$. The imaginary-axis kernel integrates every pair, since
$\int_0^\infty w\,du/(w^2+u^2)=\pi/2$; the line term is a point value at
crossings only. On Si parent 3, crossings are 5.2 % of all pairs and 82.5 % of
them lie below 5 eV; on Na the readout window is global at $\mu\pm5$ eV and
the crossing reach is 4.907 eV.

### 7.2 Pole-sum form and hole routing

For a model with positive residues $R_{+,j}$ at $\Omega_j(q)$ and the paired
negative side,

$$
\Sigma^c_{nn}(\omega)=\sum_{q,m}\Big[
(1-f_m)\sum_j\frac{\langle R_{+,j}(q)\rangle_{nm}}
{\omega-\varepsilon_m-\Omega_j(q)+i\eta}
+f_m\sum_k\frac{\langle R_{+,k}(-q)^{\mathsf T}\rangle_{nm}}
{\omega-\varepsilon_m+\Omega_k(-q)-i\eta}\Big].
\tag{W 32}
$$

Unoccupied intermediate states couple to $W_q$'s positive-frequency weight;
occupied states couple to the transposed positive weight of the partner
parent. With stored factors the occupied branch contracts
$|p^{\mathsf T}b_k(-q)|^2$ where the unoccupied branch contracts
$|p^\dagger b_j(q)|^2$: the code gathers the ordered store at $-q$ and
transposes its endpoint faces, with no extra matrix product and no second
residue contraction. The lattice-level gate
(`tests/test_shared_pole_lattice_sigma.py`) reproduces real-space $\Sigma=iGW$
on a time-reversal-broken lattice to $10^{-10}$ relative and misses by more
than $10^{-3}$ with the two orientations swapped.

### 7.3 The odd channel in the self-energy

Because $\Sigma^c$ is linear in $W_c$ and $v$ is even, the time-reversal-odd
channel lives in $\Sigma^c$ alone:
$\Sigma^{\rm odd}_n(\omega)=\Sigma^c_n[W](\omega)-\Sigma^c_n[W^{\rm even}](\omega)$,
which measures the **particle–hole asymmetry of $W$'s spectral weight as seen
through the pair densities**, $\langle R_+\rangle$ against
$\langle\tilde R\rangle$. For a remote intermediate state
($|D|=|\omega-\varepsilon_m|\gg\Omega$),

$$
\Sigma^{\rm odd}_{nn}(\omega)\simeq\sum_{q,m}\Big[
\frac{(1-2f_m)\langle M_0\rangle_{nm}}{D}
+\frac{\langle N_1\rangle_{nm}}{D^2}
+\frac{(1-2f_m)\langle M_2\rangle_{nm}}{D^3}+\cdots\Big],
\qquad
N_1\equiv\tfrac12\Big(\sum E\,R_+-\sum\tilde E\,\tilde R\Big),
\tag{W 33}
$$

with $|\langle N_1\rangle|\le\langle M_1\rangle$. The second-order coefficient
$N_1$ is time-reversal-odd, occupation-independent, and **not among the
moments the bank stores**: the moments alone do not fix the size of the odd
channel, so it must be measured.

**Measured magnitude (CrI₃, 2400 centroids, SOC ferromagnet, ordered route).**
On bands 127–134 at all 36 k, the model-level odd channel is 1.68 meV RMS and
7.80 meV maximum on the diagonal, with a 1.20 meV RMS linearised quasiparticle
shift, a −1.11 meV mean gap shift and spin–orbit partner splittings up to
14.2 meV; the exact contour-deformation integrator, converged to
$2.9\times10^{-4}$ meV, gives 1.87 meV RMS / 9.25 meV maximum with splittings
to 16.56 meV. The channel is not a head or small-$q$ artefact (withholding
$q=0$ changes the maximum by 2.7 %), its sign is definite on each side of the
gap (occupied states $+0.72$ meV mean, unoccupied $-0.84$ meV mean), and the
two sides oppose, so the channel closes the gap by 1.11 meV rather than
averaging out of it. Against the exact reference the full ordered model
reproduces the self-energy to **0.017 meV own-energy median** (QP RMS
0.022 meV, gap 0.031 meV, 0 of 288 states above 1 meV). This is the measurement
that transfers the $W$-side verdict to the self-energy and shows that the odd
channel is a meV-scale, structurally necessary term on this magnet.

## 8 The pole count, and the $N_\mu$ heuristic {#shared-pole-pole-count}

### 8.1 Three sizes, not one

The construction has three dimensions that are routinely confused:

| symbol | meaning | what it prices |
|---|---|---|
| $n=N_\mu$ | packed centroid count | spatial size: matrix actions, dense spatial algebra |
| $J$ | distinct positive pole frequencies | denominator complexity |
| $K=\sum_j\operatorname{rank}(b_jb_j^\dagger)$ | total factor columns | residue storage, realization order in $s$ |

For the constructed models the poles are generically distinct, so $J=K$, and
the honest statement is the one that survives any degeneracy: **$K$ is a
realization order, not a frequency-resolution count.** The supports number
$\mathcal O(10)$ and their count grows only logarithmically with the accuracy
target, equation (W 23); the poles are the Ritz values of the retained
port-space directions, and their number is a rank.

### 8.2 What actually sets $K$

In the current production tier the retained rank is the smaller of two things:

1. **the natural rank** — equilibrated-Gram directions above
   $\gamma>10^{-8}\gamma_{\max}$ at the recipe's direction widths and cap; and
2. **the pole budget** — $\lceil 1.8\,N_\mu\rceil$ directions, largest first,
   per parent (the recipe's `pole_budget`).

On the decks measured with the 2026-09-17 sizing, the budget is what binds:
Si keeps $K=663=\lceil1.8\times368\rceil$ per parent, MoS₂
$K=346=\lceil1.8\times192\rceil$, and the symmetric CrI₃ 3×3 deck
$K=702=\lceil1.8\times390\rceil$ — all exactly at the cap, with the corrected
18-support Si score averaging $K=657.25$ over eight parents. On Na at 896
ports the natural rank is *below* the cap: the band-structure support rule
delivers $K=1212.6$ (1.35 $N_\mu$) and $K=1364.5$ (1.52 $N_\mu$), and the
budget never fires. So

> $K$ is the **budget** where the port-space spectrum is rich enough to exceed
> it, and the **port rank** where the spectrum cuts first. The two regimes are
> both present in the measured decks, and $K/N_\mu$ alone does not tell them
> apart.

### 8.3 The heuristic

> **Rule of thumb.** On the decks measured so far, the retained pole count
> that brings the *binding* self-energy columns to about 1 meV lies in
> $K\approx(1.4\text{–}2)\,N_\mu$. The production tier takes the top of that
> band as a hard budget, $K\le\lceil1.8\,N_\mu\rceil$, because the port-space
> rank — not the frequency resolution — is what the extra poles buy, and the
> measured error falls steeply below the port-rank knee and flattens above it.

The two halves of the statement are different kinds of claim and should be
read separately.

*The band is an observation.* It is set by the rank of the response in the
ISDF port space. The ISDF basis is itself an $N_\mu$-dimensional compression of
the density products; a fixed *relative* sup-norm accuracy on an
$N_\mu$-dimensional matrix Stieltjes function costs a fixed fraction of its
port directions, because each retained direction contributes its own residue
vector even though it shares the pole frequencies. The fraction is set by the
port-space spectrum — how many directions carry non-negligible weight on the
consumer's evaluation set — and is $O(1)$ rather than growing with system
size. The measured crossings are collected in Table 8.1.

*The cap is a cost decision.* Storage is $16\,N_\mu K$ bytes per parent, and
one time-domain synthesis of $W_c(\tau)$ costs $O(N_\mu^2K)$ per $\tau$ node
before the spatial contraction. $K\propto N_\mu$ is therefore exactly the
choice that keeps the screened-interaction algebra cubic per parent while
letting the prefactor be set by measured accuracy. The ordered pencil doubles
the pencil side, so at CrI₃ production ($N_\mu=2400$, three imaginary
supports) the even pencil side is $R\le2.75N_\mu$ and the ordered
$R\le5.5N_\mu$; eight $[R,R]$ complex128 blocks plus the eigensolver workspace
must fit one device, which is the arithmetic behind the 1.8 cap.

### 8.4 The measurements

**Table 8.1.** Measured pole counts and self-energy errors. The columns are
the campaign's reporting decomposition of complex $\Sigma_c$ against a
contour-deformation reference on the same deck: **gap** (valence/conduction
mean separation), **median** and **QP RMS** at each state's own energy (the
binding columns), **p90**, and the $\pm5$ eV window RMS; the state-independent
(rigid) shift is reported separately. All values in meV. "W-side" rows score
the model against the bank's own full-port samples. Source paths are
sandbox-relative to
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/`.

| deck | $N_\mu$ | bands; parents; $\eta$ | $K$ per parent | $K/N_\mu$ | binding columns (gap / median / QP RMS) | source |
|---|---|---:|---:|---:|---|---|
| Si 4×4×4, current tier | 368 | 34; 8; 0.25 eV | 657.25 | 1.79 | **0.157 / 0.136 / 0.802** (vs CD96); p90 0.156; $\pm5$ eV 8.204 | `runs/frequency_integration_sandbox/460_batchw_20260917/REPORT.md`; claim 2431 |
| Si 4×4×4, pre-cap store | 368 | 34; 8; 0.25 eV | 1372.5 | 3.73 | — / 0.022 / 0.069; certified diagonal RMS **0.2883** (vs CD96) | `claims/2074.md`; `reports/shared_pole_model_2026-09-15/report.md` Table 2 |
| Na bcc 8×8×8, FD | 896 | 86; 29; 0.25 eV | 1337.6 | 1.49 | 0.292 / 0.195 / 0.370; certified diagonal RMS **0.2830** (vs CD48) | `claims/2074.md`; `reports/shared_pole_model_2026-09-15/report.md` Table 3 |
| Na, 18-site rule | 896 | 86; 29; 0.25 eV | 1212.6 | 1.35 | 0.321 / 0.193 / 0.397 | `reports/shared_pole_push_2026-09-07/narule/report.md` |
| Na, 18-site rule, wide | 896 | 86; 29; 0.25 eV | 1364.5 | 1.52 | **0.289 / 0.191 / 0.366**; all six columns $\le0.5$ | `reports/shared_pole_push_2026-09-07/narule/report.md` |
| Na, 18-site rule, narrow | 896 | 86; 29; 0.25 eV | 900.1 | 1.00 | 0.446 / 0.218 / 0.396 | `reports/shared_pole_model_2026-09-15/report.md` Table 3 |
| CrI₃, ordered, pre-cap | 2400 | 256; 36; 0.25 eV | 5330 | 2.22 | own-energy median **0.017**, QP RMS 0.022 (vs exact CD, 0.00029 meV converged) | `claims/2380.md`, `claims/2388.md` |
| Si, uncapped ladder ($\Sigma$) | 368 | 34; 8; 0.25 eV | 412 / 730 / 934 | 1.12 / 1.98 / 2.54 | gap $\le1$ / QP RMS $\le1$ / $\pm5$ eV $\le1$ | `claims/2204.md`; `reports/shared_pole_push_2026-09-07/SUMMARY.md` |
| Na, early 3–12-support models | 896 | 86; 29; 0.25 eV | 953–1223 | 1.06–1.36 | diagonal RMS 20.3, own-energy RMS 21.7 | `claims/1393.md`; `reports/near_nmu_tangential_allq_2026-09-06/report.md` |
| Na 144 bands (1784 ports), matched $K$, W-side | 1784 | 144; 29; 0.25 eV | 1300 (matched) | 0.73 (1.45 at 86 bands) | model error $8.2\times10^{-4}$ vs $3.8\times10^{-6}$ at 86 bands | `reports/shared_pole_push_2026-09-07/{nabands,nports}/report.md` |

Three readings follow, and they are the whole content of the heuristic.

1. **The band is real but not tight.** At 1.79 $N_\mu$ the current Si tier is
   at 0.80 meV on the binding self-energy column; at 1.49–1.52 $N_\mu$ Na is
   at 0.37–0.40 meV; at 2.22 $N_\mu$ CrI₃ is at 0.02 meV. The same fraction
   spans two orders of magnitude in error, because what the fraction measures
   is how much of the port-space spectrum the model keeps, and the port-space
   spectrum is a property of the deck.
2. **$K/N_\mu$ alone is not the error.** The early Na models sat in the same
   band (1.06–1.36) and missed by 20 meV because they had no infinity block,
   no moment constraints and only three to twelve supports; and at matched
   $K=1300$ per parent the W-side model error rises by 217× and 535× when the
   band count grows from 86 to 144 to 354 because the *port count* caps the
   reachable rank. Support placement and port adequacy are the other two
   arguments of the error, and neither is a function of $K$.
3. **The cap is not convergence.** The pre-cap Si store reaches 0.288 meV at
   3.73 $N_\mu$; a 0.1 meV-class Si calculation is not available at the cap.
   The binding error at the cap is a *budget* statement, and the honest
   production phrasing is "about a meV at $1.4$–$2N_\mu$, on these decks",
   not "converged".

Two further quantitative facts bound the interpretation. The model error
falls steeply below the port-rank knee and slowly above it — measured log-log exponents
$\approx-4.8$ on the Si common-scale ladder and $\approx-6.4$ on the
1784-port Na ladder, with the Na breakpoint at $K/n_{\rm eff}\approx
0.96$–1.17 for that deck's effective rank — so the pole count is worth
buying up to the knee and buys little past it. And the *observable* moves the
crossing: on Si the gap column reaches 1 meV at $\approx1.1N_\mu$, the
QP-level column at $\approx2.0N_\mu$ and the $\pm5$ eV window at
$\approx2.5N_\mu$, while the current tier's $\pm5$ eV window is still
8.2 meV at 1.79 $N_\mu$. A pole count quoted without its observable is not a
convergence statement.

## 9 Verification status

The measurements quoted above were made on the sources and dates named in
their reports; the recipe's support placement and budget have changed since
some of them, and the changes are named where they matter. Read this page as
the model's physics together with the following envelope.

| item | status |
|---|---|
| TRS scalar charge $W$, bank → construction → store → $\Sigma$ | implemented and gated on Si; certified 0.2883 meV diagonal RMS against CD96 on the pre-cap store, 0.802 meV QP-argument RMS at the current 1.8 $N_\mu$ cap |
| TRS metal (Na, finite occupations) | implemented; certified 0.2830 meV diagonal RMS against CD48 at 1.49 $N_\mu$; the metal sample producers and rulings have changed since (2026-09-16/17) |
| Ordered (time-reversal-broken) charge route, scalar and two-component | implemented and gated; CrI₃ exact-reference agreement 0.017 meV own-energy median at 2.22 $N_\mu$ (pre-cap sizing) |
| odd channel in $\Sigma$ | measured on CrI₃, meV-scale, structurally necessary; carries no extra storage |
| full CC/CT/TC/TT four-current sectors | **not implemented for this route**; each needs its own pole model on the joint retained spans |
| metal $\Sigma$ reference above 86 bands | **not measured**; the NPORTS W-side floor is the current evidence |
| production self-consistency with a moving $W$ | the construction rebuilds from the current state every map; long-horizon SC stability has open refusals (2026-09-17/18) and is not certified here |
| $\pm5$ eV Si window at the cap | **not converged** (8.204 meV); the 1.8 cap is calibrated on the binding columns |
| quantum-well / 2D and slab decks | not part of the measured envelope |

## Appendix A. The resolvent identity

For the even pencil, $X_a=(s_aI-S)^{-1}BQ_a$ with $F(s)=B^\dagger(sI-S)^{-1}B$.
Using
$(\bar s_aI-S)^{-1}-(s_bI-S)^{-1}=(s_b-\bar s_a)
(\bar s_aI-S)^{-1}(s_bI-S)^{-1}$ and
$F(\bar s_a)=F(s_a)^\dagger$:

$$
g_{ab}=X_a^\dagger X_b=\frac{Q_a^\dagger[F(\bar s_a)-F(s_b)]Q_b}{s_b-\bar s_a},
\qquad
h_{ab}=X_a^\dagger SX_b=s_bg_{ab}-Q_a^\dagger F(\bar s_a)Q_b ,
$$

which is (W 18) with $O_a=F(s_a)Q_a$. The confluent limit
$s_b\to\bar s_a$ gives $g_{aa'}=-Q_a^\dagger F'(\bar s_a)Q_{a'}$. For the
infinity state $X_\infty=BQ_\infty$, one resolvent step gives
$X_\infty^\dagger X_b=Q_\infty^\dagger F(s_b)Q_b$,
$X_\infty^\dagger X_\infty=Q_\infty^\dagger B^\dagger BQ_\infty
=2Q_\infty^\dagger M_1Q_\infty$ and
$X_\infty^\dagger SX_\infty=2Q_\infty^\dagger M_3Q_\infty$, which is (W 19).
For the linear pencil, with $\mathcal R(z)=(z\sigma_3-M)^{-1}$ and
$X_a=\mathcal R(z_a)C^\dagger Q_a$,
$\mathcal R(\bar z_a)-\mathcal R(z_b)=
(z_b-\bar z_a)\mathcal R(\bar z_a)\sigma_3\mathcal R(z_b)$ gives
$X_a^\dagger\sigma_3X_b=
Q_a^\dagger[W(\bar z_a)-W(z_b)]Q_b/(z_b-\bar z_a)$ and, writing
$M=z_b\sigma_3-(z_b\sigma_3-M)$,
$X_a^\dagger MX_b=z_b\mathcal G_{ab}-Q_a^\dagger W(\bar z_a)Q_b$, which is
(W 25).

## Appendix B. The paired basis and deduplication

On TRS data $W_q(-z)=W_q(z)$, so $X(-z_b)$ and $X(z_b)$ share one direction
set and have equal outputs, $CX(-z_b)=W(z_b)Q_b$, while the $v$-outputs
$[X(z_b)-X(-z_b)]/(2z_b)$ vanish. Expanding
$\mathcal G=X^\dagger\sigma_3X$ and $\mathcal H=X^\dagger MX$ in the $(w,v)$
basis with $s=z^2$ gives
$w^\dagger\sigma_3w=v^\dagger\sigma_3v=0$,
$w^\dagger\sigma_3v=G_s$, $w^\dagger Mw=H_s$, $v^\dagger Mv=G_s$ and
$w^\dagger Mv=0$, hence

$$
z\mathcal G_z-\mathcal H_z
=\begin{pmatrix}-H_s & zG_s\\ zG_s & -G_s\end{pmatrix},
\qquad
\begin{pmatrix}-H_s & zG_s\\ zG_s & -G_s\end{pmatrix}
\begin{pmatrix}w\\ v\end{pmatrix}=0
\;\Rightarrow\;
v=zw,\quad (z^2G_s-H_s)w=0 .
$$

So any common span applied to both halves gives the even Galerkin model on
that span; choosing the span on $G_s$ (the $v$-block) reproduces the even
route's own span. Cutting on $\operatorname{diag}(H_s,G_s)$ keeps a different
span, which is the measured 2–5× loss. For the deduplication criterion (W 29),
the partner direction $O=WQ$ adds no new tangent exactly when
$O\in\operatorname{span}(Q)$; the gate keeps only
$O_\perp=O-Q(Q^\dagger O)$ above the direction cutoff.

## References

The physics of $GW$ and the screened interaction: L. Hedin,
*Phys. Rev.* **139**, A796 (1965); M. S. Hybertsen and S. G. Louie,
*Phys. Rev. B* **34**, 5390 (1986); G. Onida, L. Reining and A. Rubio,
*Rev. Mod. Phys.* **74**, 601 (2002); G. F. Giuliani and G. Vignale,
*Quantum Theory of the Electron Liquid* (Cambridge, 2005), chs. 4–5.

ISDF and its use in $GW$: J. Lu and L. Ying, *J. Chem. Phys.* **143**,
064110 (2015); M. Govoni and G. Galli, *J. Chem. Theory Comput.* **11**,
2680 (2015).

Rational interpolation, rational Gauss quadrature and Zolotarev's problem:
E. I. Zolotarev (1877), as summarized in N. I. Akhiezer, *Theory of
Approximation*; W. Gautschi, *Orthogonal Polynomials: Computation and
Approximation* (Oxford, 2004); V. Druskin, L. Knizhnerman and M. Zaslavsky,
*SIAM J. Sci. Comput.* **31**, 3766 (2009); B. Beckermann and S. Güttel,
*Numer. Math.* **122**, 1 (2012); S. Güttel, *GAMM-Mitteilungen* **36**, 51
(2013); Z. Drmač, S. Gugercin and C. Beattie, *SIAM J. Sci. Comput.* **37**,
A2257 (2015); C. Beattie and S. Gugercin, "Model reduction by rational
interpolation", in *Model Reduction and Approximation* (SIAM, 2017), and
its Theorem 3.1/Algorithm 4.1 for the interpolation conditions used in
Section 6.3.

The measured rows of Section 8 are owned by their sandbox reports and claims,
named in the table; this page states the model those measurements calibrate,
not the measurements themselves.
