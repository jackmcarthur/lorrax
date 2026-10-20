# The metallic q→0 head

This page owns the long-wavelength response of a metal as it enters the
Γ-cell head: the scalar charge head (`gw.qsgw_head`) and the four-current
direct head of `full_shared_pole` (`gw.photon_direct_head`). Both take one
Fermi-surface model, one intraband/interband split and one cell average from
the same owners. Nothing in it assumes a spherical Fermi surface or a
spherical cell. Occupations, the frequency plan and the finite-q body are
owned by [metallic screening](metallic-mpa-screening.md); the four-current
routes and their Dyson solve by
[four-current heads](four-current-head-corrections.md).

Units are Rydberg atomic units. $C=2/(n_{\rm spin}n_{\rm spinor})$ counts the
physical states of one band, $\Omega$ is the cell volume, $N_k$ the full-zone
k count, $f_{nk}$ the map's fixed-N Fermi–Dirac occupation, $v^a_{nm}(k)$ the
velocity matrix including the nonlocal pseudopotential, and $\alpha/2$ the
Breit current factor (`HALFALPHA`).

## 1. The response tensor at small q

The head needs $\Pi^{IJ}(\mathbf q,z)$ for $\mathbf q$ in the q = 0 cell,
$I,J\in\{0,x,y,z\}$ (charge, current). Its band pairs fall into two classes
(§2); each class has its own exact small-q form.

**Intraband (Fermi surface).** For a state on the Fermi surface,
$f_{nk}-f_{n,k+q}\simeq-f'\,\mathbf q\cdot\mathbf u_{nk}$, and the pair sum
becomes the anisotropic Lindhard function of the actual band velocities:

$$
\Pi^{IJ}_{\rm intra}(\mathbf q,z)=\sum_s W_s\,
\frac{\mathbf q\cdot\mathbf u_s}{z-\mathbf q\cdot\mathbf u_s}\,
\gamma^I_s\gamma^J_s,\qquad \gamma_s=(1,\tfrac{\alpha}{2}\mathbf u_s),
\qquad W_s=\frac{C\,w_s}{\Omega N_k}.
$$

$w_s$ is the star-covariant tetrahedron weight of $\delta(E-\mu)$ times $N_k$
(`fermi_surface.metal_head_surface_weights`), so $\sum_sW_s=N(E_F)/\Omega\equiv N_0$.
The tetrahedron partition averages the grid-preserving reciprocal-lattice
point group supplied by `symmetry_maps.lattice_grid_point_group`. It is the
same quadrature for a reduced or identity-only WFN. This averages integration
partitions, without asserting extra symmetries of the magnetic state or
projecting its response tensor.
The limits are exact for any Fermi surface:

| limit | CC | CT | TT |
|---|---|---|---|
| $z\to0$ at fixed $q$ (static) | $-N_0$ (Thomas–Fermi, $\kappa^2=8\pi N_0$) | $-\tfrac{\alpha}{2}\sum W\mathbf u=0$ | $-(\tfrac{\alpha}{2})^2D$ |
| $\lvert z\rvert\gg q\,u$ (Drude) | $\mathbf q\cdot D\cdot\mathbf q/z^2$ | $\tfrac{\alpha}{2}\,\mathbf q\cdot D/z$ | $O(q/z)$ |

with the Drude tensor $D=\sum_sW_s\,\mathbf u_s\mathbf u_s^{\sf T}$ and
$\omega_p^2(\hat q)=8\pi\,\hat q\cdot D\cdot\hat q$. Between the limits the
function carries the true shape of the Fermi surface along every $\hat q$:
the particle–hole continuum $|z|<\max_s|\mathbf q\cdot\mathbf u_s|$ and its
edge are those of the computed bands, not of a sphere with
$\bar v^2=3\hat q\cdot D\cdot\hat q/N_0$.

**The velocity atoms $(W_s,\mathbf u_s)$** (`fermi_surface.FermiSurfaceIntraband`).
A state without Fermi-surface partners keeps its Hellmann–Feynman velocity
$\mathbf u=\operatorname{Re}v_{nn}$. A state whose pairs carry a
Fermi-surface share $\phi_{nm}$ (§2; $\phi=1$ inside a degenerate multiplet)
has no single velocity: its Drude content is the trace
$\sum_{m}\phi_{nm}\bar w_{nm}v^{a*}_{nm}v^b_{nm}$, $\bar w_{nm}=(w_n+w_m)/2$,
which `qsgw_head.head_drude_tensor_sharded` sums and which is invariant under
rotations inside a multiplet. Its off-diagonal part enters as the state's
own spread $L_sL_s^{\sf T}=\sum_{m\ne n}\phi_{nm}\operatorname{Re}v^*_{nm}v^{\sf T}_{nm}$:
the atom splits into four, $\mathbf u_s+\sqrt3\,L_s\mathbf t_j$ with weight
$W_s/4$, where $\mathbf t_j$ are the unit vertices of a regular tetrahedron
($\sum_j\mathbf t_j=0$, $\tfrac14\sum_j\mathbf t_j\mathbf t_j^{\sf T}=I/3$).
Because $\sum_{\rm ordered}\bar w_{nm}R_{nm}=\sum_{\rm ordered}w_nR_{nm}$ for
the symmetric $R_{nm}=\phi_{nm}\operatorname{Re}v^*_{nm}v^{\sf T}_{nm}$, the split
preserves the zeroth, first and second velocity moments state by state, so
$\sum_sW_s\mathbf u_s\mathbf u_s^{\sf T}=\operatorname{Re}D$ and both limits
above hold to rounding (a refusal checks it); only moments $p\ge3$ inside a
state's partner space are modelled, and the anisotropy stays on the state
that carries it.

**Interband.** A pair keeps $1-\phi_{nm}$ of its first-order
long-wavelength vertex: the charge jet $\mathbf q\cdot v_{nm}/(E_m-E_n)$ and
the uniform current $(\alpha/2)v_{nm}$. For the scalar head this is the
Adler–Wiser tensor

$$
S_{ab}(z)=\frac{2C}{\Omega N_k}\sum_{k}\sum_{E_n>E_m}
\frac{(f_m-f_n)\,v^{a*}_{nm}v^b_{nm}}{\Delta_{nm}\,(z^2-\Delta_{nm}^2)},
\qquad \chi_{\rm inter}=\mathbf q\cdot S\cdot\mathbf q,
$$

and for the four-current head the ordered $6\times6$ tensor over directed
pairs (`photon_direct_head.direct_photon_interband_tensors`), which keeps
the time-reversal-odd part. For one unordered pair with
$T_{ab}=v^{a*}_{nm}v^b_{nm}$ and $\Delta=E_m-E_n$, the current–current term is

$$
(f_n-f_m)\,\frac{2\Delta\operatorname{Re}T_{ab}+2iz\operatorname{Im}T_{ab}}{z^2-\Delta^2},
$$

whose antisymmetric (Hall) part $\operatorname{Im}T_{ab}$ is the Berry
curvature of the pair: $\sigma^{\rm H}_{ab}\propto\sum(f_n-f_m)\operatorname{Im}T_{ab}/\Delta^2$
at $z\to0$. The charge block annihilates it
($\mathbf q\cdot A\cdot\mathbf q=0$ for antisymmetric $A$), so the scalar
head has no Hall content; the CT and TT blocks carry it.

## 2. The intraband/interband split

The interband form is the small-q end of a pair's two-band overlap. For two
bands at $k+\mathbf q$ with $H=\operatorname{diag}(E_n,E_m)+\mathbf q\cdot v$,
the charge vertex $|\langle nk|m\,k+q\rangle|^2=\sin^2\theta(\mathbf q)$ grows
as $x/4$ and saturates at $1/2$, with

$$
x(\mathbf q)=\frac{(\mathbf q\cdot\delta v)^2+4|\mathbf q\cdot v_{nm}|^2}{\Delta_{nm}^2},
\qquad \delta v=v_{mm}-v_{nn}.
$$

The first-order vertex keeps the $x/4$ term only; it is valid inside the
Taylor disk $x<1$ (the two-band exceptional point) and overshoots the
bound $1/2$ outside it. The Padé form $(x/4)/(1+x/2)$ is exact at both ends.
Read at the cell's Coulomb-weighted second moment

$$
Q_{ab}=\frac{\int_{\mathcal C}d^3q\,v(q)\,q_aq_b}{\int_{\mathcal C}d^3q\,v(q)}
=\frac{\int d\Omega\,\hat q_a\hat q_b\,R(\hat q)^3/3}{\int d\Omega\,R(\hat q)}
$$

(`vcoul.minibz_coulomb_moment`; $R(\hat q)$ is the distance to the Voronoi
boundary, so the true cell shape enters and no sphere does), it splits each
pair continuously:

$$
\phi_{nm}=\frac{\epsilon^2_{nm}}{\Delta^2_{nm}+\epsilon^2_{nm}},\qquad
\epsilon^2_{nm}=\operatorname{tr}Q\,\big[\delta v\,\delta v^{\sf T}+4\operatorname{Re}v^*_{nm}v^{\sf T}_{nm}\big].
$$

The interband tensor keeps $1-\phi$ of the pair and the Fermi-surface term
takes $\phi$: $D$ gains $\phi\,\bar w_{nm}v^*_{nm}v_{nm}^{\sf T}$ and the state's
atoms spread by $\sum_m\phi_{nm}\operatorname{Re}v^*_{nm}v^{\sf T}_{nm}$
(`gw.fermi_surface.intraband_pair_fraction`). A pair moves when either
occupation measure gives it Fermi-surface weight: the tetrahedron table
($\bar w_{nm}>0$) or the smeared occupation difference that weights it in the
interband sum ($f_m-f_n\neq0$, with $(f_m-f_n)/\Delta\to-f'$). It enters the
Fermi-surface term once, with $\bar w_{nm}$, so a pair seen only by the
smearing (both states off the $T=0$ surface) leaves the interband sum and adds
nothing: its thermal weight copies surface content the table already counts,
and kept whole it grows as $f'/\Delta^2$ at $z=0$. Exact degeneracies ($|\Delta|<10^{-6}$ Ry, BGW's
TOL_Degeneracy, a rounding tolerance) have $\phi=1$. On an insulator every
$w$ vanishes, $\phi$ reduces to the degeneracy rule, and nothing changes.
The scale is the cell's own ($\epsilon\sim|v|\,q_{\rm cell}$, falling as
$N_k^{-1/3}$); $k_BT$ enters only through the occupations. No constant is
tuned.

**What it bounds.** A pair's static interband weight becomes

$$
\frac{C}{\Omega N_k}\,\frac{|f_n-f_m|\,|\hat q\cdot v_{nm}|^2}{\Delta\,(\Delta^2+\epsilon^2)}
\;\xrightarrow{\Delta\to0}\;\frac{C}{\Omega N_k}\,\frac{|f'|\,|\hat q\cdot v_{nm}|^2}{\epsilon^2},
$$

finite, where the first-order form diverged as $\Delta^{-2}$. Near-crossing
pairs at the Fermi level (Fe $4^3$ has splittings below $10^{-4}$ Ry against
$\epsilon\sim10^{-1}$ Ry) have $\phi\approx1$, so their static weight no longer
hides the Thomas–Fermi term (§3). The same weight bounds the four-current
charge jets. On a coarse grid $\epsilon$ is large enough that eV-scale pairs
share their weight too, and the share falls as the cell shrinks.

**Physical and cell-effective Drude weight.** Two tensors carry the name.
The physical Drude tensor $D_{\rm phys}$ is the $\phi\to0$ limit: only exact
multiplets are intraband, and $\omega_p^2(\hat q)=8\pi\,\hat q\cdot D_{\rm phys}\cdot\hat q$
is the metal's plasma frequency on this k grid. The cell-effective tensor
$D_{\mathcal C}=D_{\rm phys}+\sum\phi_{nm}\bar w_{nm}v^*_{nm}v^{\sf T}_{nm}$ is
what the q = 0 cell uses. It adds the Fermi-surface share of near-degenerate
interband pairs, whose weight $S$ loses, so the f-sum over $S+D$ is unchanged.
Since $\epsilon^2\propto\operatorname{tr}Q\propto N_k^{-2/3}$,
$D_{\mathcal C}\to D_{\rm phys}$ as the cell shrinks. The logs print both.
On Fe $4^3$ (frozen DFT head) they are 2.09/2.31 eV (physical) and
7.57/7.62 eV (cell-effective). With the same bands and $Q$ scaled to the
$8^3$, $16^3$ and $32^3$ cells the cell-effective value falls to 5.17/5.69,
3.89/4.69 and 3.20/3.89 eV (claim 2862). The excess $\omega_p^2$ roughly halves
per doubling of the grid, i.e. it scales as $\epsilon\propto N_k^{-1/3}$, not
as $\epsilon^2$: near-crossing pairs have a continuum of small $\Delta$. A cell-effective $\omega_p$ is
not a plasma frequency to compare with experiment or with DFT.

**The Hall part of a pair's Fermi-surface share.** The share $\phi$ moves
only symmetric content; its antisymmetric part is Berry curvature, which no
Fermi-surface velocity carries. The four-current head keeps it: the
directed-pair $6\times6$ tensor is accumulated a second time with weight
$\phi$ and only its block-wise Cartesian antisymmetric part is added, with
the jet–jet block set to zero (it is annihilated by
$\mathbf q\cdot A\cdot\mathbf q$). That part is regular as $\Delta\to0$: the
CT block is $(\alpha/2)(f_n-f_m)\,2i\operatorname{Im}T_{ab}/(z^2-\Delta^2)$ and the
TT block $(\alpha/2)^2(f_n-f_m)\,2iz\operatorname{Im}T_{ab}/(z^2-\Delta^2)$, both
finite at every bank frequency and zero at $z=0$. The Hall content of every
pair is therefore unchanged by the split.

## 3. The cell average

The head is the average over the mini-BZ Voronoi cell $\mathcal C$, sample by
sample, with no spherical reduction. For the scalar head at each dynamic
plan frequency $z$,

$$
W^c_{\rm head}(z)=\Big\langle\frac{v(q)}{1-v(q)\,\mathbf q\cdot S(z)\cdot\mathbf q}\Big\rangle_{\mathcal C}-\langle v\rangle_{\mathcal C},
\qquad v(q)=\frac{8\pi}{q^2},\qquad
S(z)=S_{\rm inter}(z)+\frac{D\,r(z)}{z^2},
$$

on the scrambled-Sobol Voronoi draws of `vcoul` (with the analytic sphere
where the deck asks for it; `gw.vcoul.compute_q0_averages_screened`). The
intraband term is the generalized Drude term with a Fermi-liquid memory
function (`qsgw_head.drude_memory_damping`):

$$
\epsilon_{\rm intra}(z)=-\frac{\omega_p^2}{z\,(z+iM(z))},\qquad
M(z)=\gamma_0\,\frac{-iz}{\Gamma-iz},\qquad
r(z)=\frac{z}{z+iM}=1-\frac{\gamma_0}{\Gamma+\gamma_0-iz}.
$$

$M$ is analytic in the upper half plane (pole at $z=-i\Gamma$); on the real
axis $\operatorname{Re}M(\omega)=\gamma_0\omega^2/(\omega^2+\Gamma^2)\ge0$, a rate
that vanishes as $\omega^2$ at the Fermi surface and saturates at
$\gamma_0$; on the imaginary axis $M(i\xi)=\gamma_0\xi/(\xi+\Gamma)$ is real.
Deck `head_drude_gamma0_ev` ($\gamma_0$, default 0: the undamped Drude term
$D/z^2$) and `head_drude_gamma_width_ev` ($\Gamma$). The per-sample
anisotropic Lindhard function of the atoms (the Landau continuum at
$|z|\lesssim q\cdot u$) is not part of the dynamic head: its moment $M_1$
(section 4) grows as $\log$ of the cell's smallest $|q\cdot u|$, so it would
set $Z$ by the Sobol draw. A constant rate ($\Gamma\to0$) is rejected for the
same reason: $\epsilon(i\xi)\sim\omega_p^2/(\gamma_0\xi)$ and $M_1$ diverges.

The static slot is the $z\to0$ limit at fixed $q$, Thomas-Fermi at every
sample, with the interband $S(0)$ kept:

$$
W_{\rm head}(0)=\Big\langle\frac{8\pi}{\mathbf q\cdot\epsilon_\infty\cdot\mathbf q+\kappa^2}\Big\rangle_{\mathcal C},
\qquad \epsilon_\infty=1-8\pi S(0),\quad \kappa^2=8\pi N_0 .
$$

A folded (`full`) head uses the folded $S(0)$ and the static fold's $\kappa^2$
(`qsgw_head._metal_static_head`). The metal plan's shifted origin
$z=i\cdot2\times10^{-5}$ Ry (`mpa.sampling.metal_origin_shift`) is this static
row (`qsgw_head.metal_static_rows`): there $|z|/(q_{\rm cell}u_F)\approx4\times10^{-4}$,
every atom is static, and the builders evaluate $S$ and the wings at $z=0$
(no $D/z^2$, no dynamic intraband wing). Left dynamic, that row folded a
$(Y_DWZ_D/\Omega)/z^2$ remainder that drove $W^c_{\rm head}$ to $-\langle v\rangle$. On Fe $4^3$ map 0 the origin
$8\pi\,\hat q\cdot S(0)\cdot\hat q$ is $-151/-91$ with this split, against
$-2242/-1551$ without it, so $q^2\epsilon_\infty$ at the cell's
$\operatorname{tr}Q=0.017$ bohr$^{-2}$ is comparable with $\kappa^2=2.33$ rather than
12–17 times larger (claim 2862).
The four-current head solves the $4\times4$ Dyson equation
$W_h=[1-\mathcal D(\mathbf q)(\Pi(\mathbf q,z)-C)]^{-1}\mathcal D(\mathbf q)$
at every Sobol sample of the cell exterior and every point of the screened
sphere rule, with $\Pi=\Pi_{\rm inter}+\Pi_{\rm intra}$ and its
$z^2$-derivative from the same atoms
($\partial_{z^2}\,x/(z-x)=-x/[2z(z-x)^2]$); the static slot keeps
CC $=-N_0$, CT $=0$, TT $=-(\alpha/2)^2D$.

Cost: $O(N_{\rm samples}\,N_{\rm atoms}\,N_z)$, with
$N_{\rm atoms}\le4\,N_{\rm FS}$ and $N_{\rm FS}$, the states with nonzero
tetrahedron weight, growing as $N_k^{2/3}$. The atoms and the diagonal
velocities are replicated; their bytes are $O(N_kN_b)$, never $O(N_b^2)$.

## 4. How the head enters Σ

The scalar samples $W^c_{\rm head}(z)$ are fitted by the head pole model
with two moments pinned (`gw.mpa.model._pin_static_head_sample`): the origin
sample, and

$$
M_1=\sum_p\frac{R_p}{\Omega_p^2}=\frac1\pi\int_0^\infty d\xi\,
\frac{W^c_{\rm head}(i\xi)-W^c_{\rm head}(0^+)}{\xi^2},
$$

by Gauss-Legendre quadrature in $t$, $\xi=\xi_0t/(1-t)$, on dedicated
imaginary-axis head samples (`qsgw_head.head_quadrature_rule`, $N_\xi=24$,
$\xi_0=0.5$ Ry; each node is one head build with the fold through
$W_{\rm body}(\Gamma,i\xi)$). $W^c_{\rm head}(0^+)$ is the limit of the
dynamic model the rule samples: $-\langle v\rangle$, because
$\epsilon_M(i\xi)\to\omega_p^2\Gamma/((\Gamma+\gamma_0)\xi^2)$; the
Thomas-Fermi row $-\langle v\rangle+t$ is the static limit at fixed $q$, and
subtracting it would add $-t/\xi^2$ (`qsgw_head.head_moment_m1`). The integrand
$[W^c+\langle v\rangle]/\xi^2$ is then finite at $\xi=0$. M₁ is stamped in the
head store (`fit_moment_m1`).

$$
\Sigma^{\rm head}_{nk}(\omega)=\frac{1}{\Omega N_k}\sum_pR_p
\Big[\frac{f_{nk}}{\delta_{nk}+\Omega_p}+\frac{1-f_{nk}}{\delta_{nk}-\Omega_p}\Big],
\qquad\delta_{nk}=\omega-(E_{nk}-E_F),
$$

which is $(1/2-f_{nk})W^c_{\rm head}(0)/(\Omega N_k)$ on shell
([metallic screening §5.3](metallic-mpa-screening.md)), with slope
$\partial_\omega\Sigma^{\rm head}_{nk}=-M_1/(\Omega N_k)$. On-shell energies
therefore see the static cell average, $Z$ sees the quadrature $M_1$ of the
generalized Drude model, and only off-shell values are shaped by the fit. The four-current head's cell averages
($W_h-W_\infty$, its $z^2$ derivative, the ordered $-\mathbf q$ mirror,
$\langle W_\infty-\mathcal D\rangle$ and the large-$z$ coefficients) enter the
ordered sector bank as rank-4 updates on the literal-Γ vectors, and Σ is the
bank's consumer ([four-current heads §5](four-current-head-corrections.md#direct-bulk-head)).

## 5. Limits of the model

- **One reading point.** $\phi$ is the Padé share at the cell's
  Coulomb-weighted $Q$, one number per pair. The q-resolved version, the
  saturated vertex $(x/4)/(1+x/2)$ at every cell sample, is the next
  refinement; it needs the pair list beside the atoms.
- **The head plasmon follows $D_{\mathcal C}$.** The cell's dynamic head puts
  the shared weight at $\omega=0$, so its plasmon sits at the cell-effective,
  not the physical, $\omega_p$ until the grid converges. On Fe $4^3$ the
  bispinor map-0 eqp0/eqp1 tails (64 and 119 meV) come entirely from the CC
  head block, on states whose $|E-E_F|$ lies 3.6–4.8 eV from the Fermi
  level (claim 2862).
- **Wings (closed 2026-10-20).** The `full` scalar route folds head/body
  wings (`qsgw_head.head_wings_sharded`) split like $S$ and $D$: each pair
  keeps $1-\phi$ of its Lehmann weight and the intraband wing
  $-\tfrac{C}{2}\,\phi\,\bar w_{nm}\,v^*_{nm}\rho_{nm}/z$ takes the rest, the
  pair weight's own $|z|\gg\Delta$, $\Delta\ll k_BT$ limit. A pair's wing is then
  independent of $\phi$ for $|z|\gg\Delta$, and a multiplet's is the
  rotation-invariant trace. The previous diagonal-only surface term had the
  opposite sign to that limit; time reversal cancels the intraband wing over
  the zone, so only an ordered metal saw it, and that store refused `full`
  until both seams closed. The intraband wing stays the Drude limit at
  $z\ne0$ (q-independent in the cell, error confined to near-line samples
  $|z|\lesssim q_{\rm cell}u_F$, i.e. to $Z$). The static row's fold takes the
  row's own Γ body; only an exact $z=0$ row also builds the static Matsubara
  body. Certified by `tests/test_shared_pole_head_ordered_metal.py`.
- **Damping and the wings.** $r(z)$ multiplies the $D/z^2$ term of $S$. The
  ordered metal's intraband wing ($\propto1/z$) keeps the undamped Drude
  form at $\gamma_0>0$; time reversal cancels it on a non-magnetic metal.
- **Estimators.** The intraband term uses tetrahedron weights and the
  interband term FD occupations; at coarse grids the two describe slightly
  different Fermi surfaces.
- **Scope.** The scalar head is bulk (`sys_dim = 3`) only; a slab metal
  refuses in `gw.vcoul.compute_q0_averages`. The four-current direct head
  also runs on a slab ([four-current heads §5.1](four-current-head-corrections.md#direct-slab-head)).

## 6. Code owners

| object | owner |
|---|---|
| tetrahedron Fermi-surface table, multiplet weights | `gw.fermi_surface.metal_head_surface_weights` |
| two-band pair split $\phi$ | `gw.fermi_surface.intraband_pair_fraction` |
| velocity atoms, $\Pi_{\rm intra}$ ($4\times4$) | `gw.fermi_surface.FermiSurfaceIntraband` |
| $D$, $S(z)$, static slot, $r(z)$, the $M_1$ rule | `gw.qsgw_head` |
| two-moment pin | `gw.mpa.model._pin_static_head_sample` |
| cell moment $Q$ and cell draws | `vcoul.minibz` (`minibz_coulomb_moment`) |
| four-current interband tensor, Hall part, Γ-cell Dyson | `gw.photon_direct_head` |
