# The GN plasmon-pole model from a non-Hermitian $W(i\omega_p)$

**Scope.** Derivation memo for the time-reversal-broken (magnetic) GN-PPM as
implemented at `34228021`. The model uses the Hermitian and anti-Hermitian
$i\omega_p$ components. Sources:
`src/gw/w_isdf.py` (χ₀ kernels), `src/gw/minimax_screening.py` (the fit),
`src/gw/ppm_sigma.py` / `ppm_accumulators.py` (the Σ windows), and
`DERIVATION_channel_hermiticity.md` §1.3 (the crossing-closure premise).
Every identity below is pinned by `tests/test_gnppm_ordered_orientations.py`.

## 1. Exact pole structure without time reversal

Summed over both ordered particle-hole orientations (Adler–Wiser), at
transfer $q$ and complex frequency $z$,

$$
\chi^0_q(z)=\sum_{vck}\Big[\frac{P^{q}_{vck}}{z-\Delta}
-\frac{\overline{P^{-q}_{vck}}}{z+\Delta}\Big],
\qquad P^{q}_{vck}=|\rho\rangle\langle\rho|,\ \ \rho(\mu)=\overline{\psi_{vk}(\mu)}\,\psi_{c,k+q}(\mu),
$$

with $\Delta=\epsilon_{c,k+q}-\epsilon_{vk}>0$ and the bar the elementwise
complex conjugate.  The $+\Delta$ pole carries the object built from
$\overline{\psi_v}\psi_c$; the $-\Delta$ pole carries the object built from
$\overline{\psi_c}\psi_v$, which is the conjugate of the $-q$ forward object.
Both residues are Hermitian in $(\mu,\nu)$ and the poles are real, so for any
system

$$
\chi^0_q(z)^\dagger=\chi^0_q(\bar z),\qquad
\chi^0_{-q}(z)=\overline{\chi^0_q(-\bar z)} .
$$

On the imaginary axis $z=i\omega$ these give $\chi^0_q(i\omega)^\dagger=\chi^0_q(-i\omega)$
and $\chi^0_{-q}(i\omega)=\overline{\chi^0_q(i\omega)}$: **the Hermitian part is
the even-in-$\omega$ part, the anti-Hermitian part is the odd-in-$\omega$
part, and $q$-conjugate reciprocity holds with no time-reversal assumption.**
Writing the two poles out,

$$
\chi^0_q(i\omega)=-\sum\frac{(P^q+\overline{P^{-q}})\,\Delta+i\omega\,(P^q-\overline{P^{-q}})}{\omega^2+\Delta^2}.
$$

Under $\Theta$ ($\psi_{n,-k}=\overline{\psi_{nk}}$) the second set equals the
first, the odd bracket vanishes, and $\chi^0(i\omega)$ is Hermitian.  On a
magnet the odd bracket is the time-reversal-odd channel: anti-Hermitian, odd
in the magnetisation, zero at $\omega=0$.  $W_q(z)=[1-V_q\chi^0_q(z)]^{-1}V_q$
inherits every statement above ($V_q$ Hermitian, $V_{-q}=\overline{V_q}$).

## 2. What the even route computes, and the ordered production route

The Laplace kernel (`w_isdf._get_chi_minimax_kernel`, real τ) forms one
orientation per node, $A_q(\tau)=\sum P^{\rm kern}_{q}\,e^{-\tau(\Delta-E_{\rm gap})}$,
where the kernel's own object is $P^{\rm kern}=|\overline{\psi_c}\psi_v\rangle\langle\cdot|$
(the oracle `_direct_node_sum` in `tests/test_chi_contour_kernel.py`), i.e. the
$-\Delta$-pole orientation, and completes it as $A_q+\overline{A_{-q}}$
(`_complete_static_vertex_orientations`) before weighting with the EVEN
kernel $\alpha_l\approx x/(x^2+\omega_p^2)$. Exact at $\omega=0$ and under
$\Theta$; it deletes the odd bracket otherwise.

The `complex_contour` kernel (`compute_chi0_contour`, what MPA consumes)
applies its two resolvent rows $-1/(\Delta\mp z)$ to the SAME single
orientation $P^{\rm kern}$, so it is exactly the $\Theta$-symmetric form
$\sum P^{\rm kern}\,2\Delta/(z^2-\Delta^2)$ as well.  **It does not carry the
odd channel either**.

The production non-TRS object is a linear combination of the same two carriers with
independent complex weights.  With $\gamma_l\approx-1/(x+i\omega_p)$ fitted on
$[x_{\min},x_{\max}]$ by real nodes,

$$
\boxed{\ \chi^0_q(i\omega_p)=F_q+\overline{F_{-q}},\qquad
F_q=\sum_l \gamma_l\,e^{-\tau_l E_{\rm gap}}\,A_q(\tau_l),\qquad
\gamma_l=-(\alpha_l-i\beta_l)\ }
$$

with $\sum_l\alpha_l e^{-\tau_l x}\approx x/(x^2+\omega_p^2)$ (the incumbent
even rule, unchanged) and $\sum_l\beta_l e^{-\tau_l x}\approx\omega_p/(x^2+\omega_p^2)$
(the odd rule: the same nodes plus a few greedily added ones, weights-only
Lawson fits, `minimax_screening.solve_laplace_minimax_imag_interval(with_odd_kernel=True)`).
The sign of $i\beta$ is fixed by $P^{\rm kern}$ being the $-\Delta$-pole
orientation; the conjugate partner then receives $\overline{\gamma_l}\approx-1/(x-i\omega_p)$
automatically.  $F_q$ is one `complex_contour` sweep with real nodes and
complex weights (`w_isdf.compute_chi0_imag_ordered`); the completion is a
$q$-negation gather plus a conjugate, sharding-preserving.  On a $\Theta$ deck
$\overline{A_{-q}}=A_q$ and the formula reduces to the incumbent; the code
keeps the incumbent path there (bit-identity) and routes only when
`SymMaps.trs_allowed` is false.

## 3. The two-point model with an odd residue

Per element $(q,\mu,\nu)$ the GN ansatz that has the pole structure of §1 is a
pair of poles at $\pm\Omega$ with **two Hermitian residues**:

$$
W^c(z)=\frac{R_+}{z-\Omega}-\frac{R_-}{z+\Omega},\qquad
W^c(i\omega)=-\frac{(R_++R_-)\,\Omega+i\omega\,(R_+-R_-)}{\omega^2+\Omega^2}.
$$

Data: $W^c(0)$ (Hermitian) and $W^c(i\omega_p)=h_p+a_p$ split elementwise into
its Hermitian and anti-Hermitian halves.  The even part is the incumbent fit:

$$
\Omega^2=\omega_p^2\,\frac{h_p}{W^c(0)-h_p}\ (\text{elementwise, Re taken as today}),\qquad
B\equiv\tfrac12(R_++R_-)=-\tfrac12 W^c(0)\,\Omega .
$$

The odd part at $\omega_p$ fixes the half-difference:

$$
\boxed{\ D\equiv\tfrac12(R_+-R_-)=\frac{i\,a_p\,(\omega_p^2+\Omega^2)}{2\,\omega_p},\qquad
R_\pm=B\pm D\ }
$$

$a_p$ is anti-Hermitian and $\Omega$ real symmetric, so $D$ is Hermitian and
each of $R_\pm$ is Hermitian even though $W^c(i\omega_p)$ is not.  Under a
magnetisation flip ($\psi\to\bar\psi$) $a_p\to-a_p$, $B,\Omega$ invariant,
$D\to-D$: the two residues swap.  $\Theta$ deck: $a_p=0$, $D=0$, incumbent.
Elementwise, $D$ vanishes wherever $\Omega$ is dead (pads, invalid modes:
$D$ is computed AFTER the tail policy from the final $\Omega$).

## 4. Which residue each Σ branch consumes

$\Sigma(1,2)=iG(1,2)W(1^+,2)$ gives $\Sigma_c(E)=\frac{i}{2\pi}\int d\omega'\,e^{-i\omega'0^+}G(E-\omega')W^c(\omega')$.
Time-ordered $G$ has occupied poles in the upper and empty poles in the lower
half plane; the time-ordered $W^c$ has $+\Omega-i\eta$ (residue $R_+$) below
and $-\Omega+i\eta$ (residue $-R_-$) above.  Closing below:

$$
\boxed{\ \Sigma_c(E)=\sum_{m\ \rm occ}\frac{\psi_m\psi_m^\dagger\odot R_-}{E-\epsilon_m+\Omega}
+\sum_{m\ \rm empty}\frac{\psi_m\psi_m^\dagger\odot R_+}{E-\epsilon_m-\Omega}\ }
$$

($\odot$ elementwise in $(\mu,\nu)$, then band-projected).  So the
**conduction branches consume $R_+=B+D$ and the valence branches $R_-=B-D$**;
with $D=0$ this is the incumbent GPP formula.  Verified independently by the
imaginary-axis contour $\Sigma_c(E)=-\frac1{2\pi}\int d\nu\,G(E-i\nu)W^c(i\nu)$
at midgap (test cell), whose red twin (residues swapped) fails.  The static
limit picks up $D/\Omega$ beyond COHSEX, the model's image of the odd channel
that the exact $\Sigma_c$ also carries at $E$ in the gap.

## 5. The crossing closure is unchanged; its premise is supplied per branch

`ppm_accumulators._complete_one_sided_tau` closes each crossing window as
$(Z-Z^\dagger)/2i$, which requires the residue fed to THAT window to be
Hermitian and $\Omega$ real symmetric (`DERIVATION_channel_hermiticity.md`
§1.3).  With §3 both branches receive a Hermitian residue ($R_+$ or $R_-$), so
the sine-sum closure stays valid.  What was wrong is not the closure but the
incumbent's single $B$ for both branches, and — had the corrected χ₀ been fed
to the incumbent elementwise fit — a non-Hermitian $B$ and non-symmetric
$\Omega$ from the raw $W^c(i\omega_p)$, which breaks the pair-adjoint identity
(red twin in the tests).  The Laplace windows need only bilinearity.

## 6. The charge head, HL, MPA

*Head.* The odd bracket of the Cartesian head tensor $S_{ab}(i\omega)$ is
$\propto\omega\,(P^{ab}-P^{ba})$ with $P^{ab}=\overline{v^a}v^b$: **antisymmetric
in $ab$**.  The scalar charge head is $\langle\hat q_aS_{ab}\hat q_b\rangle$, which
annihilates every antisymmetric tensor, so the scalar GN head fit is exactly
time-reversal-even and its odd residue is identically zero; `fit_head_ppm`'s
`.real` is the Hermitian part of a $1\times1$ and is correct as it stands.
The channel lives only in the antisymmetric (Faraday-like) part of $S_{ab}$,
which no scalar head can carry.

*HL.* At a real probe $\Omega_{\rm HL}$ above all transitions $W^c$ is
Hermitian for any system ($W^c(z)^\dagger=W^c(\bar z)$ with $z$ real), so a
two-point real-axis fit cannot separate $R_+-R_-$; `hl_ppm` keeps the
incumbent single-residue fit on magnets (registered, not changed).

*MPA.* Samples on complex lines need $\overline{F_{-q}(-\bar z)}$ at the
reflected frequency, i.e. a sample set symmetric under $\omega\to-\omega$;
the identity of §2 applies with that pairing. Before the ordered MPA change,
the contour completion had been corrected but the Padé model was still a
function of $z^2$ only.  It therefore deleted the odd channel downstream even
though the samples contained it.  The ordered fit and its Sigma assignment are
derived next.

## 7. Multipole fit from an ordered complex-frequency pair

For each sampled upper-half-plane point $z_j$, one response sweep produces the
two native orientations at $z_j$ and $-\bar z_j$.  Completing them in both
orders gives the independently sampled pair

$$
\chi_q(z_j)=F_q(z_j)+\overline{F_{-q}(-\bar z_j)},\qquad
\chi_q(-\bar z_j)=F_q(-\bar z_j)+\overline{F_{-q}(z_j)}.
$$

The two Dyson solves preserve the causal adjoint identity
$W_q(z)^\dagger=W_q(\bar z)$.  Thus the value needed to separate frequency
parity at the SAME ordered element is

$$
W_q^c(-z_j)=W_q^c(-\bar z_j)^\dagger.
$$

This adjoint is taken on the independently evaluated partner; it is **not**
$[W(z_j)+W(z_j)^\dagger]/2$ and imposes no Hermiticity on $W(z_j)$.  At a
pure-imaginary point the two upper-plane coordinates coincide, but their lower
partner is still the matrix adjoint, so the anti-Hermitian part remains the
odd-frequency datum.

The general $n_p$-pole extension of §3 is

$$
\boxed{
W^c(z)=\sum_p\left[\frac{R_{p,+}}{z-\Omega_p}
-\frac{R_{p,-}}{z+\Omega_p}\right]
=\sum_p\frac{2\Omega_p B_p+2zD_p}{z^2-\Omega_p^2},
\quad B_p=\frac{R_{p,+}+R_{p,-}}2,\quad
D_p=\frac{R_{p,+}-R_{p,-}}2 .}
$$

The ordered sample pair supplies

$$
W_{\rm even}(z_j)=\frac{W(z_j)+W(-z_j)}2,
\qquad
W_{\rm odd}(z_j)=\frac{W(z_j)-W(-z_j)}2.
$$

The existing Padé/Loewner/Thiele implementation is used once, element by
element in $(q,\mu,\nu)$, on $W_{\rm even}$ to determine the shared
$\Omega_p$ and $B_p$.  There is no eigenchannel decomposition, Hermitian
factorisation, lower-triangle fill, or real projection.  After the existing
pole guards, one fixed-pole complex least-squares solve determines $D_p$ from

$$
W_{\rm odd}(z_j)=\sum_p\frac{2z_jD_p}{z_j^2-\Omega_p^2}.
$$

The staged fit store carries $B_p$ and $D_p$ on the same row-sharded layout.
A TRS fit has no $D_p$ dataset and follows the incumbent arithmetic and I/O
path.  A measured-broken-TR fit carries the explicit ordered-residue stamp;
an absent dataset with a true stamp, or a present dataset with a false stamp,
is a refused partial schema.

## 8. Multipole Sigma consumes the two residues per causal branch

The contour closure in §4 applies pole by pole, so no new MPA Sigma formula is
needed.  Every conduction Green's-function branch consumes
$R_{p,+}=B_p+D_p$ and every valence branch consumes
$R_{p,-}=B_p-D_p$, through the same `_residue_for_space` selector used by
GN-PPM.  Pole-window planning uses the union of the live support of both
residues; the delivered-error planner measures the selected residue separately
for each branch.  The scalar charge head remains the even object of §6 and
therefore retains the single-residue fit.

`tests/test_gnppm_ordered_orientations.py` plants a two-pole non-Hermitian
$W$, fits the ordered samples, reconstructs a dense imaginary-axis grid, and
compares the branch-selected $\Sigma_c$ with an independent imaginary-axis
contour.  Its red twin Hermitises the samples before fitting and is required to
move the answer macroscopically.

For observability, ordered MPA executes that same planned contraction once
more with $D=0$, exactly as GN-PPM does, and records
`sigC_odd = Sigma_c[B,D] - Sigma_c[B,0]`.  The debug-off arm therefore emits
an exactly zero `sigC_odd` column.  This is deliberately not reconstructed
from a fitted scalar or from an on-shell QP difference: it is the difference
of the two production Sigma cubes on the same omega grid.
