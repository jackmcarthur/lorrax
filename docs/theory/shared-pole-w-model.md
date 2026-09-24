# The shared-pole screened interaction {#shared-pole-theory}

`sigma_w_model = shared_pole` represents the correlation part of the screened
interaction, $W_c=W-v$, in the ISDF centroid basis by **one set of real poles
per parent $q$, shared by every matrix element, with one factor vector per
pole**. The poles and factors are the Ritz pairs of a Hermitian pencil
assembled from samples of $W_c$ and $\partial_sW_c$ at a few complex
frequencies plus the exact high-frequency moments. Nothing is fitted: the
pencil *is* the tangential Hermite interpolation condition, so real poles,
positive residues and particle–hole pairing hold by construction, and the
number of frequency samples grows only logarithmically with the accuracy
target.

This page owns the model and its derivation. How the samples are streamed,
reduced, sharded, stored and consumed, the gate rows and the byte model are
[the shared-pole implementation page](../architecture/shared_pole_model.md);
deck keys are in the [input reference](../input_reference.md); the $\Sigma$
frequency quadrature is [the Sigma quadrature problem](sigma-quadrature-problem.md);
rebuilding the model inside a QSGW loop, and the magnetic little-group
realization of the stored model, are in
[self-consistency](../self_consistency.md#shared-pole-w-with-retained-quadrature).

Notation: $n=N_\mu$ is the packed centroid count; a response at momentum $q$
is an $n\times n$ matrix acting with the Coulomb congruence $v_q$. Frequencies
are in Ry unless given in eV; $z$ lies in the upper half plane and $s=z^2$.

## 1 The exact object

The RPA response dressed by $v$ has the Lehmann form

$$
W_q(z) - v_q
  \;=\; \sum_{n\in q}\frac{R_{+,n}(q)}{z - E_n}
      \;-\; \sum_{m\in -q}\frac{R_{-,m}(q)}{z + E_m},
\qquad R_{\pm}\succeq 0,\qquad E_n>0,
\tag{W 1}
$$

with $E_n$ the particle–hole excitation energies of the deck and $R_{\pm,n}$
positive semidefinite $n\times n$ residues built from pair densities and
$v_q$. Everything below follows from this analytic structure.

**Particle–hole pairing is exact and needs no time reversal:**

$$
R_-(q) \;=\; R_+(-q)^{\mathsf T}
\qquad\Longleftrightarrow\qquad
W_q(z) \;=\; W_{-q}(-z)^{\mathsf T}.
\tag{W 2}
$$

The negative-frequency side of $W_q$ is the transposed positive side of the
**partner parent** $-q$. A store therefore keeps only positive poles per parent,
and $-q$ must be a declared parent (or a realized mirror of one).

**Reality in time** gives, with or without time reversal,

$$
W_q(\bar z) = W_q(z)^\dagger,\qquad
W_q(-\bar z) = \overline{W_{-q}(z)},\qquad
W_{-q}(iu) = \overline{W_q(iu)}\quad(u\in\mathbb R).
\tag{W 3}
$$

**Time-reversal symmetry (TRS)** adds $W_{-q}=W_q^{\mathsf T}$, which with
(W 2) makes $W_q$ even in $z$, $W_q(z)=F_q(z^2)$. Without it (W 1) has an odd
part:

$$
W^{\rm even}_q(z) \equiv \tfrac12\big[W_q(z)+W_{-q}(z)^{\mathsf T}\big],\qquad
W^{\rm odd}_q(z) \equiv \tfrac12\big[W_q(z)-W_{-q}(z)^{\mathsf T}\big].
\tag{W 4}
$$

With the negative branch written by its own energies $\tilde E_k$ and residues
$\tilde R_k$,

$$
W^{\rm even}_q(z)-v_q
 =\sum_n\frac{E_nR_{+,n}}{z^2-E_n^2}+\sum_k\frac{\tilde E_k\tilde R_k}{z^2-\tilde E_k^2},
\qquad
W^{\rm odd}_q(z)
 =z\Big[\sum_n\frac{R_{+,n}}{z^2-E_n^2}-\sum_k\frac{\tilde R_k}{z^2-\tilde E_k^2}\Big].
\tag{W 5}
$$

At a time-reversal-invariant momentum ($q\equiv-q$, $\tilde R=R^{\mathsf T}=\bar R$),

$$
W^{\rm odd}(z)\;=\;2iz\sum_n\frac{\operatorname{Im}R_{+,n}}{z^2-E_n^2}:
\tag{W 6}
$$

**the odd channel is the imaginary part of the same Hermitian residues**, which
is why the time-reversal-broken model costs the same storage as the TRS one
(§4.2).

| quantity | object | unit |
|---|---|---|
| $W_c$, $v$ | $\mu\times\mu$ response matrices | Ry |
| $z$, $\Omega_j$, $E_n$ | frequencies | Ry |
| $s=z^2$, $\Lambda_j=\Omega_j^2$ | squared frequency | Ry² |
| $b_j$ (stored factor) | vector per pole | Ry$^{3/2}$ |
| $R_{+,j}=b_jb_j^\dagger/(2\Omega_j)$ | positive residue | Ry |
| $M_k$ | high-frequency moments of $W_c$ | Ry$^{k+2}$ |

The poles are real. The retarded $\pm i\eta$ is applied once, by the consumer,
from `sigma_regularization_ev`: a fitted width $\kappa$ would enter every
$\Sigma$ denominator as $\eta+\kappa$ and is indistinguishable from a real
node after the consumer's own smoothing.

## 2 The latent particle–hole pencil and passivity

In RPA the same object is the resolvent of a linear pencil,

$$
W_c(z)\;=\;C\,(z\sigma_3-M)^{-1}C^\dagger,\qquad
\sigma_3=\begin{pmatrix}I&0\\0&-I\end{pmatrix},\quad
M=\begin{pmatrix}A_1&B\\B^\dagger&A_2\end{pmatrix}=M^\dagger,\quad
C=v\big[\rho_q,\ \bar\rho_{-q}\big],
\tag{W 7}
$$

with $M\succeq\operatorname{diag}(\Delta E)\succ0$ for a stable RPA. The
involution $\mathcal P=\tau_x\circ\mathrm{conj}$ obeys
$\tau_x\sigma_3\tau_x=-\sigma_3$; $\tau_x\bar M\tau_x=M$ and $C\tau_x=\bar C$
hold only at $q\equiv-q$ and otherwise map the pencil at $q$ to the pencil at
$-q$, which is (W 2) again. Under TRS the pencil reduces to a Hermitian problem
in $s$,

$$
W_c(z)=2\Phi\,(z^2 I-S)^{-1}\Phi^\dagger,\qquad
S=\Delta^{1/2}\big(\Delta+2P^\dagger vP\big)\Delta^{1/2},\quad
\Phi=vP\Delta^{1/2},
\tag{W 8}
$$

with $\Delta$ the transition energies and $P$ the pair-density matrix in a real,
TR-adapted transition basis. $W_c$ is then a matrix Stieltjes function,
$W_c(s)=\int_E d\mu(t)/(s-t)$ with $\mu\succeq0$ on
$E\subset[\Omega_{\min}^2,\Omega_{\max}^2]$.

**Passivity.** Let $K=v^{1/2}\chi_0v^{1/2}$ with $\operatorname{Herm}K\preceq0$
on the imaginary axis and $T=I-K$. Then
$\operatorname{Herm}T^{-1}=T^{-1}(\operatorname{Herm}T)T^{-\dagger}\succeq0$, and
$TT^\dagger-\operatorname{Herm}T=-\operatorname{Herm}K+KK^\dagger\succeq0$ gives
the upper bound:

$$
0\;\preceq\;\operatorname{Herm}\!\big[-v^{-1/2}W_c(i\eta)v^{-1/2}\big]\;\preceq\;I .
\tag{W 9}
$$

(W 9) holds **with broken time reversal**; the anti-Hermitian part is the odd
channel and has no pointwise sign. It is a necessary condition, checked on the
model and never imposed on it.

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

$M_1,M_3$ are even under $q\to-q$ and $M_0,M_2$ odd; under TRS
$M_0=M_2=0$. In pencil language $m_k=C(\sigma_3M)^k\sigma_3C^\dagger=2M_k$. For
the model of §4,

$$
M_1=\tfrac12\,b\,b^\dagger,\qquad
M_3=\tfrac12\,b\,\Lambda\,b^\dagger,\qquad
\Lambda=\operatorname{diag}(\Omega_j^2),
\tag{W 11}
$$

so $M_1$ and $M_3$ fix the model's high-frequency behaviour — the "infinity
block" of the construction, which carries the **rigid shift** of $\Sigma$.

**From band sums.** With $\chi_0(z)=\sum_{p\ge1}X_p/z^p$ from band-summed pair
densities ($X_1,X_3$ odd), the Dyson series gives $W_c=\sum_pC_p/z^p$ with
every cross term:

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
$C_4=vX_4v+vX_2vX_2v$. On a finite band set $X_1\neq0$ even for the charge
density, because densities commute only in the complete basis, so

$$
\frac{\lVert m_0\rVert}{\lVert M_1\rVert}
=\frac{2\lVert M_0\rVert}{\lVert M_1\rVert}
\qquad(\text{Ry}^{-1})
\tag{W 13}
$$

is a band-truncation diagnostic, reported in the bank receipt and never used
to refuse a deck.

## 4 The model

### 4.1 Stored form

$$
W_c(q,z)=\sum_{j=1}^{K_q}\frac{b_j(q)\,b_j(q)^\dagger}{s-\Omega_j(q)^2},
\qquad \Omega_j>0,\ b_j\in\mathbb C^{n},
\tag{W 14}
$$

stored as $16\,nK_q$ bytes per parent. In pole-sum form,

$$
W_c(q,z)=\sum_j\frac{R_{+,j}(q)}{z-\Omega_j(q)}-\sum_j\frac{R_{+,j}(q)}{z+\Omega_j(q)},
\qquad
R_{+,j}=\frac{b_jb_j^\dagger}{2\Omega_j}\ \succeq 0 .
\tag{W 15}
$$

The consumer never forms (W 15): it synthesizes the time-domain kernel
$W_c(\tau)=-i\,b\,\operatorname{diag}\!\big(e^{-i(\Omega_j-E_{\rm ref})\tau}/2\Omega_j\big)b^\dagger$,
one diagonal phase per pole per $\tau$ node, and reads the hole branch from the
partner parent (§7).

### 4.2 Ordered (time-reversal-broken) form

Each parent stores positive poles only; its negative side is the partner's
positive side, transposed:

$$
W_q(z)-v_q
=\sum_{j\in q}\frac{b_jb_j^\dagger}{2\Omega_j\,(z-\Omega_j)}
-\sum_{k\in -q}\frac{\bar b_k\,b_k^{\mathsf T}}{2\Omega_k\,(z+\Omega_k)} .
\tag{W 16}
$$

(W 2) therefore holds **exactly by storage**, including on a magnet. At
$q\equiv-q$,

$$
W_q(z)-v_q
=\sum_j\frac{\operatorname{Re}(b_jb_j^\dagger)}{z^2-\Omega_j^2}
+i\sum_j\frac{\operatorname{Im}(b_jb_j^\dagger)\,z}{\Omega_j\,(z^2-\Omega_j^2)},
\tag{W 17}
$$

even and odd channels on the same vector and pole set: (W 6) in the model's
variables. Under TRS the partner's factors are $\bar b$ and (W 16) reduces to
(W 14). The number of stored parents is the magnetic irreducible wedge.

### 4.3 What the model guarantees

| property | mechanism |
|---|---|
| real poles | Hermitian Ritz problem after whitening (§5.3); for the ordered route $\mathcal H\succ0$ on the retained span (§6.2) |
| residues $\operatorname{sign}(\Omega)\times$PSD | by construction, (W 15), (W 27) |
| particle–hole pairing | storage, (W 16) |
| odd channel at no storage | $\operatorname{Im}(bb^\dagger)$ on the same $b$, (W 17) |
| ordered model = even model on TRS data at equal retained span | paired basis and $v$-block cut, Appendix B |
| moments $M_0\ldots M_3$ | infinity rows of the pencil, (W 19), (W 26) |
| passivity | (W 9), checked on the whitened model |

## 5 Construction: tangential interpolation as rational Gauss quadrature

### 5.1 Supports and directions

One real-time $\chi_0$ stream per parent evaluates $W_c$ and $\partial_sW_c$ at
a fixed set of **supports** $z_a$: points on the damped line $z=\omega+ih$ and
on the imaginary axis $z=iu$. $W$ is never formed at a real frequency; remote
transitions enter through Laplace cells ([response Laplace](response-laplace.md))
and the moments are exact band sums. Production uses 18 fitted supports; held
diagnostic supports and the $M_1/M_3$ block are additional.

Placement is a condenser problem, not a choice of interesting frequencies:

* **Imaginary ladder.** Log-spaced on $[u_{\min},u_{\max}]$ with
  $u_{\min}=\max(4\eta,E_g)$, $u_{\max}=\max(16\ \mathrm{eV},L)$,
  $L=\omega_p+3.5\ \mathrm{eV}$ ($\omega_p$ from the active electron density),
  $\kappa=L/u_{\min}$, and the count of (W 23).
* **Line ladder.** Height $h=\max(2.6\ \mathrm{eV},4\eta)$; the remaining
  $18-m$ sites are equal quantiles of $\rho^{1/2}$ on
  $[\max(h,\text{first spacing}),\,\omega_{\rm reach}]$, where $\rho$ is the
  $\eta$-broadened density of the crossings $|E-\epsilon_{mk}|$ that contour
  deformation actually meets for states delivered within $\pm5$ eV of $\mu$.
  The rule reads band energies, $\mu$ and $\eta$ only; it is not fitted to $W$.

At each fitted support a narrow direction set $Q_a\in\mathbb C^{n\times r_a}$
is selected from the sample itself:

* line: right singular vectors of $W_c(z_a)$ above a relative cutoff $10^{-3}$,
  at most $\lceil n/16\rceil$, whole multiplets;
* imaginary: the leading $\lceil n/4\rceil$ eigenvectors of
  $-\operatorname{Herm}W_c(iu)$;
* infinity: the leading $\lceil n/8\rceil$ eigenvectors of $M_1$.

Directions below the dense eigensolver's $n\epsilon_{64}$ relative resolution
are excluded: a numerical null direction carries no resolved response and must
not be amplified by Gram equilibration. Only the actions
$O_a=W_c(z_a)Q_a$ and $D_a=\partial_sW_c(z_a)Q_a$ reach the pencil. A line
support's conjugate partner is not a new sample: $W(\bar s)=W(s)^\dagger$ gives
its action from $O_a$.

### 5.2 The Hermite pencil

Write the model as $F(s)=B^\dagger(sI-S)^{-1}B$ with $B=b^\dagger$, and take
rational Krylov states $X_a=(s_aI-S)^{-1}B^\dagger Q_a$. The resolvent identity
gives every pencil entry from sample actions alone (Appendix A):

$$
G_{ab}=X_a^\dagger X_b=\frac{Q_a^\dagger O_b-O_a^\dagger Q_b}{s_b-\bar s_a},
\qquad
H_{ab}=X_a^\dagger SX_b=s_b\,G_{ab}-Q_a^\dagger O_b,
\tag{W 18}
$$

with the confluent limit $G_{aa'}=-Q_a^\dagger\,\partial_sW(\bar s_a)Q_{a'}$.
The infinity state $X_\infty=BQ_\infty$ closes the pencil with the moments:

$$
G_{\infty b}=Q_\infty^\dagger O_b,\qquad
H_{\infty b}=s_bG_{\infty b}-2\,(M_1Q_\infty)^\dagger Q_b,\qquad
G_{\infty\infty}=2Q_\infty^\dagger M_1Q_\infty,\qquad
H_{\infty\infty}=2Q_\infty^\dagger M_3Q_\infty .
\tag{W 19}
$$

(W 18) *is* the tangential Hermite interpolation condition at the supports,
three matrix products per block; $\partial_sW$ enters only the confluent block.
$(G,H)$ is Hermitian by construction, not a least-squares normal matrix.

### 5.3 Reduction

Equilibrate $G$ by $1/\sqrt{\operatorname{diag}G}$, keep eigenvalues
$\gamma>10^{-8}\gamma_{\max}$ truncated to the pole budget, correct the
retained metric to $Z^\dagger GZ=I$ by coupled Newton–Schulz, and diagonalize

$$
\operatorname{Herm}(Z^\dagger HZ)\;=\;U\Lambda U^\dagger,\qquad
\Omega_j=\sqrt{\lambda_j},\qquad
b=OZU .
\tag{W 20}
$$

Every Ritz value is real and non-negative and every residue
$b_jb_j^\dagger/(2\Omega_j)$ is positive semidefinite **with no constraint
imposed**. Poles at $\lambda\le10^{-6}$ Ry² are dropped only within a
factor-weight budget of $10^{-6}$, so a model never silently discards mass.

### 5.4 Why this works, and what sets the sampling

Rayleigh–Ritz on the rational Krylov space of the samples is a **rational Gauss
quadrature** of the spectral measure $\mu$: poles and weights are exact on
$\operatorname{span}\{1/(\sigma_k-t)\}$ at the supports, so for a linear
consumer $f$

$$
\Big|\int f\,d(\mu-\mu_K)\Big|
\;\le\;2\lVert\mu\rVert\;
\operatorname{dist}_E\!\big(f,\ \operatorname{span}\{1/(\sigma_k-t)\}\big),
\tag{W 21}
$$

and for a Cauchy kernel at $\zeta$ the error is exactly $F(\zeta)-F_K(\zeta)$.
With shifts $\sigma_a$, Ritz values $\lambda_j$, $q(x)=\prod_j(x-\lambda_j)$ and
$w(x)=\prod_a(x-\sigma_a)$, the Galerkin error is
$q(s)^{-1}\int_E q\,d\mu/(s-t)$ with $\int_E q\,t^k\,d\mu/w=0$, so

$$
\sup_F\lVert W-W_K\rVert
\;\le\;\frac{\max_E|R|}{\min_F|R|}\;\sup_F\int_E\frac{d\mu(t)}{|s-t|},
\qquad R=\frac{q^2}{w}.
\tag{W 22}
$$

Minimizing the ratio is Zolotarev's third problem,
$\min_R\max_E|R|/\min_F|R|\sim4\exp(-\pi^2n/\ln4\kappa)$:

$$
n(\epsilon)=\frac{\ln(4\kappa)\ln(4/\epsilon)}{\pi^2}
\tag{W 23}
$$

distinct shifts for relative accuracy $\epsilon$ on a condenser of ratio
$\kappa$ (Beckermann–Reichel; Druskin–Knizhnerman–Zaslavsky). The imaginary
count $m=\max\!\big(2,\operatorname{round}[\ln(16\kappa^2)\ln(4/\epsilon)/2\pi^2]\big)$
at $\epsilon=10^{-3}$ is (W 23), since $\ln(16\kappa^2)/2=\ln(4\kappa)$.

**The laws place the supports; they do not count the poles.** $n(\epsilon)$
counts shifts. The pole count $K_q$ is the retained rank of the matrix-valued
pencil, which the ISDF port count caps (§8): frequency resolution is
logarithmic in the accuracy target, port coverage is not.

**Why the acceptance norm is a sup-norm on $\Sigma$'s evaluation set.** $\Sigma$
is linear in $W$, so on $F=F_{\rm line}\cup F_{\rm im}$

$$
\lVert\delta\Sigma\rVert\;\le\;\lVert\rho\rVert_1\,
\sup_{\zeta\in F}\lVert\delta W(\zeta)\rVert,
\tag{W 24}
$$

with $\rho$ the near-flat delivery density of the quadrature. An $L^2$ norm
weighted by $|W|^2$ controls nothing where $|W|$ is small, and an $H^2$ norm on
the damped line cannot control the quasiparticle energies: $z=0$ lies outside
its Hardy domain, so no bound $|E(0)|\le C\lVert E\rVert_{H^2}$ exists, and
$H^2$ controls neither the inverse moments nor the low-frequency band that sets
the QP energies. This is why the model is interpolatory at prescribed supports
rather than an IRKA-type $H^2$ reduction. The rigid shift of $\Sigma$ follows
$M_1,M_3$; the QP energies follow the low-energy (0–2.5 eV) part of the line
error. The prescribed supports cover both by construction.

## 6 The time-reversal-broken construction

### 6.1 The projected particle–hole pencil

Take $X_a=(z_a\sigma_3-M)^{-1}C^\dagger Q_a$ with outputs $O_a=W_c(z_a)Q_a$. The
same resolvent identity, now with $\sigma_3$, gives

$$
\mathcal G_{ab}=X_a^\dagger\sigma_3X_b
=\frac{(W_aQ_a)^\dagger Q_b-Q_a^\dagger W_bQ_b}{z_b-\bar z_a},
\qquad
\mathcal H_{ab}=X_a^\dagger MX_b=z_b\,\mathcal G_{ab}-(W_aQ_a)^\dagger Q_b,
\tag{W 25}
$$

with confluent block $\mathcal G_{aa'}=-Q_a^\dagger W'(\bar z_a)Q_{a'}$ and
$W'(z)=2z\,\partial_sW$. The nodes are $\{z,\bar z,-\bar z,-z\}$; each state
$X(z)$ is followed by its mirror $X(-z)$ on the same $Q$, and
$W_q(-\bar z)=\overline{W_{-q}(z)}$ supplies the mirror from parent $-q$.
**$\mathcal H$, not $\mathcal G$, is the definite member.** The infinity states
$k_0=\sigma_3C^\dagger Q_\infty$ and $k_1=\sigma_3M\sigma_3C^\dagger Q_\infty$
carry $m_k=2M_k$:

$$
\mathcal G[k_0,k_1]=m_0,m_1,m_2,\qquad
\mathcal H[k_0,k_1]=m_1,m_2,m_3,
\tag{W 26}
$$

so the ordered infinity block needs all four moments of (W 10); a bank with only
$M_1,M_3$ builds finite states and records the block `NOT_MEASURED`.

### 6.2 Reality of the poles

Factor $\mathcal H=Y^{-\dagger}Y^{-1}$ (equilibrate, eigendecompose, cut) and
diagonalize $Y^\dagger\mathcal GY=U\operatorname{diag}(\mu)U^\dagger$:

$$
F_r(z)=\sum_j\frac{c_jc_j^\dagger}{z\mu_j-1},\qquad
c=OYU,\qquad \Omega_j=\mu_j^{-1},\qquad
b_j=\sqrt2\,c_j/|\mu_j| .
\tag{W 27}
$$

Every pole is real and every residue $c_jc_j^\dagger/\mu_j$ is
$\operatorname{sign}(\Omega_j)\times$PSD. **$\mathcal H\succ0$ on the retained
span is sufficient**; a stable RPA ($M\succ0$) guarantees it for linearly
independent retained states, and the gate refuses rather than repairs. It is not
necessary for a real spectrum ($\mathcal G=I$, $\mathcal H=\operatorname{diag}(1,-1)$
has real eigenvalues), so a real spectrum does not certify stability. $\mu_j=0$
is a pole at infinity; its output weight is reported.

### 6.3 Paired basis, cut and deduplication

On TRS data the states at $\pm z$ share one direction set. In the basis

$$
w_b=\tfrac12\big[X(z_b)+X(-z_b)\big],\qquad
v_b=\frac{X(z_b)-X(-z_b)}{2z_b},\qquad
w_\infty=k_1,\quad v_\infty=k_0,
\tag{W 28}
$$

$\mathcal H'=\operatorname{diag}(H_s,G_s)$ and $\mathcal G'$ is off-diagonal,
$(G_s,H_s)$ being the even pencil (W 18); eliminating $v=zw$ gives
$(z^2G_s-H_s)w=0$, the even problem in $s$ (Appendix B). **The rule**: pair the
states, cut on the $v$-block, and apply that span to both halves. The ordered
Gram keep ratio is $10^{-7}$ (spans normalized separately for a charge–current
cross pencil use $10^{-5}$), because weak directions retained at $10^{-8}$ can
make the projected paired pencil indefinite. Cutting on
$\operatorname{diag}(H_s,G_s)$ keeps a different span and admits spurious poles
far above the spectrum.

At imaginary supports, and line supports with $\operatorname{Re}z=0$, a
conjugate partner brings no new tangent when $W$ is Hermitian there; only the
component of $O=WQ$ orthogonal to $Q$ survives,

$$
\lambda\big(O_\perp O_\perp^\dagger\big)>c_{\rm dir}^2\,
\lambda_{\max}\big(OO^\dagger\big),\qquad c_{\rm dir}=10^{-3},
\tag{W 29}
$$

so a TRS bank adds no partner columns and **the ordered model equals the even
model at equal retained span** (`tests/test_shared_pole_ordered.py`).

## 7 How the self-energy consumes the model

### 7.1 Contour deformation without time reversal

With $\Sigma^c(\omega)=\frac{i}{2\pi}\int d\omega'\,G(\omega-\omega')W_c(\omega')$,
rotating onto the imaginary axis for a state at energy $E$ and intermediate
state $m$ with $x_m=E-\varepsilon_m$ gives

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

with $\langle A\rangle_{nm}=p_{nm}^\dagger Ap_{nm}$ and $\mathcal X(E)$ the
Green's-function poles the rotation crosses. Under TRS the two terms of (W 30)
fold into $2x/(x^2+u^2)$ acting on $W_q(iu)$ and the occupied residue reads
$W_q(|x|+i\eta)$; those are exactly the two steps an integrator must undo on a
magnet. The imaginary-axis term integrates every pair; the line term is a point
value at crossings only, which is why the line supports follow the crossing
density (§5.1).

### 7.2 Pole-sum form and hole routing

$$
\Sigma^c_{nn}(\omega)=\sum_{q,m}\Big[
(1-f_m)\sum_j\frac{\langle R_{+,j}(q)\rangle_{nm}}
{\omega-\varepsilon_m-\Omega_j(q)+i\eta}
+f_m\sum_k\frac{\langle R_{+,k}(-q)^{\mathsf T}\rangle_{nm}}
{\omega-\varepsilon_m+\Omega_k(-q)-i\eta}\Big].
\tag{W 32}
$$

Unoccupied intermediate states couple to $W_q$'s positive-frequency weight,
occupied states to the transposed positive weight of the partner parent: the
occupied branch contracts $|p^{\mathsf T}b_k(-q)|^2$ where the unoccupied branch
contracts $|p^\dagger b_j(q)|^2$, so the hole branch costs a gather at $-q$ and a
transpose, no extra product. `tests/test_shared_pole_lattice_sigma.py`
reproduces real-space $\Sigma=iGW$ on a time-reversal-broken lattice to
$10^{-10}$ relative and misses by more than $10^{-3}$ with the orientations
swapped.

### 7.3 The odd channel in the self-energy

$\Sigma^c$ is linear in $W_c$ and $v$ is even, so the odd channel lives in
$\Sigma^c$ alone:
$\Sigma^{\rm odd}_n=\Sigma^c_n[W]-\Sigma^c_n[W^{\rm even}]$, the particle–hole
asymmetry of $W$'s spectral weight seen through the pair densities. For a
remote intermediate state ($|D|=|\omega-\varepsilon_m|\gg\Omega$),

$$
\Sigma^{\rm odd}_{nn}(\omega)\simeq\sum_{q,m}\Big[
\frac{(1-2f_m)\langle M_0\rangle_{nm}}{D}
+\frac{\langle N_1\rangle_{nm}}{D^2}
+\frac{(1-2f_m)\langle M_2\rangle_{nm}}{D^3}+\cdots\Big],
\qquad
N_1\equiv\tfrac12\Big(\sum E\,R_+-\sum\tilde E\,\tilde R\Big),
\tag{W 33}
$$

with $|\langle N_1\rangle|\le\langle M_1\rangle$. $N_1$ is time-reversal-odd,
occupation-independent, and not among the stored moments, so the moments alone
do not fix the size of the odd channel. On the CrI₃ SOC ferromagnet it is
meV-scale — 1.68 meV RMS on the frontier bands, splitting spin–orbit partners
by up to 14 meV and closing the gap by 0.77 meV, with no head or small-$q$
origin (claim 2371) — and the full ordered model reproduces an exact
contour-deformation reference to 0.017 meV at the own energy (claim 2388).

## 8 The pole count {#shared-pole-pole-count}

### 8.1 Three sizes

| symbol | meaning | prices |
|---|---|---|
| $n=N_\mu$ | packed centroid count | matrix actions, dense spatial algebra |
| $J$ | distinct positive pole frequencies | denominator complexity |
| $K=\sum_j\operatorname{rank}(b_jb_j^\dagger)$ | total factor columns | residue storage, realization order in $s$ |

The constructed poles are generically distinct, so $J=K$. **$K$ is a
realization order, not a frequency-resolution count**: the supports number
$\mathcal O(10)$ and grow logarithmically with the accuracy target (W 23),
while the poles are Ritz values of the retained port-space directions and
their number is a rank.

### 8.2 What sets $K$

The retained rank is the smaller of

1. the **natural rank**: equilibrated-Gram directions with
   $\gamma>10^{-8}\gamma_{\max}$ at the recipe's direction widths; and
2. the **pole budget** $\lceil1.8\,N_\mu\rceil$, largest first, per parent.

Where the port-space spectrum is rich the budget binds; where it cuts first
the natural rank does. $K/N_\mu$ alone does not say which.

### 8.3 The heuristic, and why the cap is where it is

On the decks measured, the pole count that brings the self-energy at each
state's own energy to about 1 meV lies in $K\approx(1.4\text{–}2)\,N_\mu$.
The ISDF basis is an $N_\mu$-dimensional compression of the pair densities; a
fixed relative sup-norm accuracy on an $N_\mu$-dimensional matrix Stieltjes
function costs a fixed fraction of its port directions, because each retained
direction carries its own residue vector even though the pole frequencies are
shared. The fraction is set by the port-space spectrum on the consumer's
evaluation set; it is $O(1)$, not growing with system size. The model error
falls steeply until the port-rank knee and slowly after it, so poles are worth
buying up to the knee.

The cap is a cost decision. Storage is $16N_\mu K$ bytes per parent and one
synthesis of $W_c(\tau)$ costs $O(N_\mu^2K)$ per $\tau$ node, so
$K\propto N_\mu$ keeps the screened-interaction algebra cubic per parent. The
pencil side is bounded by the direction widths of §5.1,
$R\le\big[2(18-m)/16+m/4+1/8\big]N_\mu$ with conjugate line partners counted
($2.75N_\mu$ at $m=3$), and twice that for the ordered route; eight $[R,R]$
complex128 blocks plus the eigensolver workspace must fit one device.

The cap is a budget, not convergence. At the cap the Si QP-energy RMS against
a contour-deformation reference is 0.80 meV (claim 2431); the uncapped store at
$3.7N_\mu$ reaches 0.29 meV (claim 2074). The observable moves the crossing:
the gap converges at lower $K$ than individual QP energies, and a wide
($\pm5$ eV) window needs more. A pole count quoted without its observable is not
a convergence statement.

## 9 Signed interactions: the photon sectors

The four-current route (`bispinor_gw = full_shared_pole`) builds the same model
per sector (CC, TT, and the joint CT span) from the paramagnetic response
$\chi_p(z)=C(zJ-H_0)^{-1}C^\dagger$, with particle/hole signature $J$,
positive transition energies in $H_0$, and the square roots of positive
occupation differences and the Cartesian current vertices in $C$. With the
diamagnetic contact $D$, $\chi=\chi_p-D$ and $W=(I-V\chi)^{-1}V$. The bare
photon $V$ is Hermitian but **signed**. If the contact solve exists,
$U=W_\infty=(I+VD)^{-1}V$ is Hermitian and

$$
W(z)-U = U C\,\big[zJ-(H_0+C^\dagger U C)\big]^{-1} C^\dagger U .
\tag{W 34}
$$

Positivity of $H_0+C^\dagger UC$ is therefore a sufficient stable-realization
condition even for indefinite $U$: the ordered residue at $\Omega$ is
$\operatorname{sign}(\Omega)$ times a PSD matrix. The scalar passivity bound
(W 9) does not apply to a signed $V$; positive retained $\mathcal H$ is the
stability gate. The constant $U-V$ enters $\Sigma$ separately from the pole
model of $W-U$. Because CC, TT and CT are reduced independently, positive
retained $\mathcal H$ certifies each projected sector, not positive residues of
the assembled photon matrix; the assembled model is judged by its integrated
$\Sigma$. Which channel carries which frequency model is
[four-current heads and frequency](four-current-head-corrections.md).

## Appendix A. The resolvent identity

For the even pencil, $X_a=(s_aI-S)^{-1}BQ_a$ with $F(s)=B^\dagger(sI-S)^{-1}B$.
Using
$(\bar s_aI-S)^{-1}-(s_bI-S)^{-1}=(s_b-\bar s_a)(\bar s_aI-S)^{-1}(s_bI-S)^{-1}$
and $F(\bar s_a)=F(s_a)^\dagger$:

$$
g_{ab}=X_a^\dagger X_b=\frac{Q_a^\dagger[F(\bar s_a)-F(s_b)]Q_b}{s_b-\bar s_a},
\qquad
h_{ab}=X_a^\dagger SX_b=s_bg_{ab}-Q_a^\dagger F(\bar s_a)Q_b ,
$$

which is (W 18) with $O_a=F(s_a)Q_a$. The confluent limit $s_b\to\bar s_a$ gives
$g_{aa'}=-Q_a^\dagger F'(\bar s_a)Q_{a'}$. For $X_\infty=BQ_\infty$,
$X_\infty^\dagger X_b=Q_\infty^\dagger F(s_b)Q_b$,
$X_\infty^\dagger X_\infty=Q_\infty^\dagger B^\dagger BQ_\infty=2Q_\infty^\dagger M_1Q_\infty$
and $X_\infty^\dagger SX_\infty=2Q_\infty^\dagger M_3Q_\infty$, which is (W 19).
For the linear pencil, with $\mathcal R(z)=(z\sigma_3-M)^{-1}$ and
$X_a=\mathcal R(z_a)C^\dagger Q_a$,
$\mathcal R(\bar z_a)-\mathcal R(z_b)=(z_b-\bar z_a)\mathcal R(\bar z_a)\sigma_3\mathcal R(z_b)$
gives $X_a^\dagger\sigma_3X_b=Q_a^\dagger[W(\bar z_a)-W(z_b)]Q_b/(z_b-\bar z_a)$,
and writing $M=z_b\sigma_3-(z_b\sigma_3-M)$ gives
$X_a^\dagger MX_b=z_b\mathcal G_{ab}-Q_a^\dagger W(\bar z_a)Q_b$, which is (W 25).

## Appendix B. The paired basis and deduplication

On TRS data $W_q(-z)=W_q(z)$, so $X(\pm z_b)$ share one direction set and have
equal outputs, $CX(-z_b)=W(z_b)Q_b$, while the $v$-outputs
$[X(z_b)-X(-z_b)]/(2z_b)$ vanish. In the $(w,v)$ basis with $s=z^2$:
$w^\dagger\sigma_3w=v^\dagger\sigma_3v=0$, $w^\dagger\sigma_3v=G_s$,
$w^\dagger Mw=H_s$, $v^\dagger Mv=G_s$ and $w^\dagger Mv=0$, hence

$$
\begin{pmatrix}-H_s & zG_s\\ zG_s & -G_s\end{pmatrix}
\begin{pmatrix}w\\ v\end{pmatrix}=0
\;\Rightarrow\;
v=zw,\quad (z^2G_s-H_s)w=0 .
$$

Any common span applied to both halves gives the even Galerkin model on that
span; choosing it on $G_s$ (the $v$-block) reproduces the even route's own span.
The partner direction $O=WQ$ adds no tangent exactly when
$O\in\operatorname{span}(Q)$, which is the test (W 29) on
$O_\perp=O-Q(Q^\dagger O)$.

## References

- L. Hedin, *Phys. Rev.* **139**, A796 (1965); M. S. Hybertsen and S. G. Louie,
  *Phys. Rev. B* **34**, 5390 (1986); G. Onida, L. Reining and A. Rubio,
  *Rev. Mod. Phys.* **74**, 601 (2002).
- J. Lu and L. Ying, *J. Chem. Phys.* **143**, 064110 (2015) (ISDF).
- V. Druskin, L. Knizhnerman and M. Zaslavsky, *SIAM J. Sci. Comput.* **31**,
  3766 (2009); B. Beckermann and S. Güttel, *Numer. Math.* **122**, 1 (2012);
  S. Güttel, *GAMM-Mitteilungen* **36**, 51 (2013) (rational Krylov, Zolotarev).
- W. Gautschi, *Orthogonal Polynomials: Computation and Approximation*
  (Oxford, 2004) (rational Gauss quadrature).
- C. Beattie and S. Gugercin, "Model reduction by rational interpolation", in
  *Model Reduction and Approximation* (SIAM, 2017), Theorem 3.1 and
  Algorithm 4.1 (tangential interpolation conditions).
