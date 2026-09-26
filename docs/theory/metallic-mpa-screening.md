# Metallic MPA screening

A metal's response and self-energy carry fractional occupations: every band
near the Fermi level is both a hole and an electron state, transition energies
run continuously through zero, and the long-wavelength limit depends on the
order in which $\omega\to0$ and $q\to0$ are taken. This page owns how
`compute_mode = mpa` treats that: the occupation-weighted response and why its
weighting is safe to discretize, the metal frequency plan, the two $q\to0$
limits, the occupation-weighted $\Sigma$, and the one occupation state per map.

The production metal model is `sigma_w_model = shared_pole`, whose bank carries
signed metallic weights through its occupation envelope
([bank](../architecture/shared_pole_model.md#2-the-response-bank));
`sigma_w_model = mpa` (elementwise poles) is the comparison and uses the plan of
§2. Sample geometry, the pole fit and the disk pipeline are
[multipole frequency integration](THEORY_mpa_implementation.md); the distributed
kernels are the [fractional χ₀ response face](../architecture/fractional_chi0_response_face.md);
the $\Sigma$ quadrature is [the Σ quadrature problem](sigma-quadrature-problem.md);
the self-consistent loop on a metal, its head routes and its scissor rules are
[self-consistency](../self_consistency.md#metals-direct-drude-head); the
`chi00 = q.S.q` convention is [S-tensor convention](s-tensor-convention.md).
All equations are in Ry.

**What a metal deck refuses.** GN-PPM and HL-PPM (`GATE gn_ppm_refuses_metals`,
`GATE fractional_occupations_require_mpa`); a missing `occ_smearing_width_ry`;
any occupation family but Fermi–Dirac (`GATE metal_occupations_fermi_dirac`);
`fermi_reference` other than `mp1_fixed_n`; the ladder `wc_source`
(`GATE w_bse_insulators_only`); elementwise MPA with time reversal measured
broken (`GATE mpa_ordered_metal`: that route fits one residue with no odd
channel); and every velocity-head route except the admitted direct Drude head
(`GATE metal_sc_head_update_disabled`, §4).

## 1. The finite-occupation response and its cancellation structure

### 1.1 The identity and the implemented form

With $a=(n,\mathbf k)$, $b=(m,\mathbf k-\mathbf q)$, $\Delta_{ab}=E_b-E_a$ and the
density-vertex outer product $X_{ab}(q)$, the independent-particle response at
$\operatorname{Im}z>0$ is the Adler–Wiser sum over **all ordered band pairs**,

$$
\chi_{0,q}(z)=C\sum_{ab}\frac{(f_a-f_b)\,X_{ab}(q)}{z-\Delta_{ab}} .
$$

The implementation rests on $f_a-f_b=f_a(1-f_b)-(1-f_a)f_b$, exact for any real
$f$. With one positive-time product

$$
A_q(t)=\sum_{ab}f_a(1-f_b)\,e^{-i\Delta_{ab}t}\,X_{ab}(q),\qquad t\ge0,
$$

Hermiticity of the density vertex, with the $q$ and $-q$ orientations kept
explicit (an orientation identity, not time reversal), gives the partner
without a second pair build:

$$
\chi_{0,q}(z)=-i\int_0^\infty dt\,e^{izt}\left[A_q(t)-A_{-q}(-t)^{\mathsf T}\right].
$$

One time node therefore costs **two weighted single-band Green sums**, one with
`band_weight = f` and one with `band_weight = 1-f`, which meet only after both
band axes are summed; no band-pair-by-node object exists
(`gw.w_isdf.compute_chi0_contour_fractional`; the weight enters
`build_G_tau(band_weight=)` linearly and is never clipped or square-rooted).
The band supports are the smallest contiguous ranges whose weight clears
$|w|>1-$`occupation_window_threshold` (default 0.995, floor 0.005); a partially
occupied band belongs to both.

### 1.2 Where the cancellation lives

Call the implemented weighting **form A**: $f_a(1-f_b)$ on one positive-time
product minus its mirror. The textbook **form B** writes the same $f_a-f_b$ as a
term weighted by $f_a$ alone minus its mirror weighted by $f_b$. Both are exact;
they differ in where the two terms cancel, which decides what a quadrature may
discretize.

- **Form B.** Every occupied–occupied pair enters both terms with weight
  $\sim1$ and must cancel between them: an $\mathcal O(N_{\rm occ}^2)$ block of
  order-unity contributions, including degenerate pairs at $\Delta=0$, exactly
  the slowly decaying content a damped-time rule resolves worst. The analytic
  parts cancel; the discretization errors do not.
- **Form A.** $f_a(1-f_b)$ vanishes pointwise on the occupied–occupied and
  empty–empty blocks. The only residual cancellation is the Fermi shell, where
  both weights are fractional and each product is $\le1/4$. In the insulating
  limit form A cancels nothing.

On a shell pair the surviving fraction is $|f_a-f_b|/f_a(1-f_b)\ge4|f_a-f_b|$,
and on the dynamic samples $1/|z-\Delta|\le1/\varpi$ bounds the kernel, so the
line heights of the plan stay well inside float64. The $z\to0$ row loses that
bound, which is why it takes a different producer (§1.4, §2.3).

### 1.3 Relation to the shredded-propagator method

Kim, Martyna and Ismail-Beigi (CTSP) evaluate the gapless static
polarizability per energy-window pair as $D^{lm}E^{lm}-F^{lm}G^{lm}$ with the
occupation carried asymmetrically — valence side by $f(E_v)$, conduction side by
$f(E_c)$ — which is form B. It is safe there because the windows partition the
valence and conduction ranges separately, so the occupied–occupied block is
never formed; the $\Delta\to0$ limit inside one window pair is handled by a
manual gap. LORRAX's scan has no windows: the $f(1-f)$ weights exclude the same
block **pointwise**, keep same-side Fermi-shell pairs that a
valence-list × conduction-list structure cannot form, and evaluate the
degenerate limit exactly (§1.4).

### 1.4 The static row

At $z=0$ the degenerate limit
$[f(E)-f(E')]/(E'-E)\to-df/dE$ is reached as $0/0$. Static $\chi_0$ on
fractional occupations has one producer, `w_isdf.compute_chi0_matsubara` at
$n=0$ (Fermi–Dirac only): its imaginary-time factors $fe^{(E-\mu)\tau}$ and
$(1-f)e^{-(E-\mu)\tau}$ lie in $(0,1]$ on $[0,\beta]$, so there is no subtraction
and no divided difference; a degenerate pair carries
$\beta f(1-f)=-df/dE$ after the transform. The ordered-pair scan refuses
$z=0$ by name. The metal MPA plan has no $z=0$ sample (§2.1).

## 2. The metal frequency plan

### 2.1 The shifted origin

The metal grid is the insulating double-parallel plan with the near line's
first point moved from $z=0$ to $z_0=i\varpi_0$, $\varpi_0=10^{-5}$ Ha
$=2\times10^{-5}$ Ry (`mpa_metal_origin_shift_ry`, refused on an insulator):
Leon *et al.*'s metals protocol, a stability displacement around zero-energy
intraband transitions, not a broadening. `mpa_sampling_alpha` defaults to 2 on
a metal, concentrating samples near zero. The plan classifies the origin as
`imag`, not `static`; that character is what routes it (§2.3).

### 2.2 Rule bandwidth from the occupation supports

The damped-line rule must resolve $F_\chi=\max|\operatorname{Re}z|+\Delta_{\max}$.
On a metal there is no valence/conduction cut, so

$$
\Delta_{\max}
=\max_{b\in\mathrm{supp}(1-f)}E_b-\min_{a\in\mathrm{supp}(f)}E_a
$$

over the same supports as the kernel (`w_isdf.occupation_support_bandwidth`).
It exceeds any gapped-cut estimate — the $f$ support reaches above $\mu$ and the
$1-f$ support below it, more so with a wider smearing — and it never sizes the
rule for transitions whose weight is zero. Metal line calls pass the rule's
positive nodes only: the fractional kernel supplies both Keldysh terms, and the
insulating $\pm\tau$ doubling would count them twice.

### 2.3 The origin row

A damped-line rule truncates at $t_{\max}=\log(2/\epsilon)/\varpi$; at
$\varpi_0=2\times10^{-5}$ Ry and a few-Ry bandwidth that is
$\sim7\times10^5$ Ry$^{-1}$ of oscillating integrand and about $10^6$ nodes, against
tens on the far line. The origin row is therefore evaluated exactly, at its
literal nonzero coordinate, by the ordered band-pair scan
`w_isdf.compute_chi0_direct_fractional`, dispatched from
`gw.mpa.model._evaluate_samples`; the far pure-imaginary point and every damped
point use the fractional contour. The file's ordinate and value describe the
same analytic function. The scan's band-pair work is quadratic, so it serves
this one sample per fit and never a frequency grid. No metal point reaches the
insulating kernels, whose positive-gap split a metal breaks
(`tests/test_chi_contour_kernel.py` pins the dispatch).

The plan is one trade: the one unaffordable sampled row becomes an exact
evaluation that is both cheaper and more accurate, and every other row keeps a
cheap certified rule. The origin sample sits four decades below $\varpi_1$ and
gives the Loewner pencil a near-isolated row, which the fit's conditioning
guards watch. It cannot be moved per consumer: the scalar head is fitted on the
**identical** complex grid as the body (`build_mpa_fit` refuses otherwise),
because head and body residues are summed in one $\Sigma$.

## 3. The finite-q body

Every stored wedge row of every dynamic sample is the fractional contour; the
origin row is the ordered-pair scan. For wedge row $j$ every $b$-side operand
(both centroid wavefunction faces, energies, occupations) is rolled by the flat
$k\to k-q_j$ map (`model._metal_kminq_rows`, which asserts that the Γ row's map
is the identity). The long-wavelength limit belongs to the head (§4). Dyson,
wedge storage and the column fit are the insulating pipeline's; the metal enters
the body only through its sample values.

## 4. The two heads, and the order of limits

A metal's long-wavelength limit is order-sensitive. The **static** limit takes
$\omega\to0$ at fixed small $q$, then $q\to0$, and gives Thomas–Fermi screening.
The **dynamic** head takes $q\to0$ first and stays $\omega$-dependent, with an
intraband (Drude) term that diverges as $1/\omega^2$ toward zero frequency. They
are different objects, and the head kernel keeps them apart by construction:

- **Dynamic.** The interband Kubo tensor $S(z)$ (`qsgw_head.head_s_tensor_sharded`,
  convention of [S-tensor convention](s-tensor-convention.md)) plus the Drude
  tensor, entering as $D/z^2$ at $z=\omega+i\eta$:

  $$
  D_{ab}=\frac{C}{\Omega N_k}\sum_k\sum_{nm\in\mathcal M}
  \bar w_{k,nm}\,v^{a*}_{k,nm}v^{b}_{k,nm},\qquad
  \omega_p^2(\hat q)=8\pi\,\hat q\cdot D\cdot\hat q,
  $$

  with $C=2/(n_{\rm spin}n_{\rm spinor})$, $w$ the star-covariant tetrahedron
  weight of $\delta(E-\mu)$ times $N_k$ (`fermi_surface.metal_head_surface_weights`),
  and the sum over each degenerate multiplet $\mathcal M$ (BGW's
  TOL_Degeneracy, $10^{-6}$ Ry) with one weight per multiplet. The multiplet
  trace is invariant under rotations inside the multiplet; its pairs are
  excluded from $S$. Free electrons give $D=2n$, i.e. $\omega_p^2=16\pi n$ Ry$^2$.
  Measured: Na bcc $8^3$ 5.95 eV (free electron at this density 6.05 eV);
  Fe bcc $4^3$ 2.09/2.31 eV, where $4^3$ does not converge the Fermi surface.
- **Static.** The exact $z=0$ slot is not the $\omega\to0$ value of the dynamic
  expression; it takes

  $$
  \kappa_{\rm TF}^2=\frac{8\pi N(E_F)}{V_{\rm cell}},
  $$

  with $N(E_F)$ the same tetrahedron weight sum, in the mini-BZ average, and
  through the static wing/body fold when the head is folded
  (`qsgw_head._metal_static_head`). The metal MPA and shared-pole plans have
  no exact-zero sample: their origin is $i\varpi_0$, where the $q$-first
  Drude head screens the whole cell ($W\to0$).

**Every metallic head carries the metal's state.** The one-shot head, the
frozen head of `sc_head_update = off` and the per-map head of `dft_velocity`
take the fixed-N Fermi–Dirac state that the body and $\Sigma$ take, and the
intraband term above; `dft_velocity` rebuilds it each map from the DFT
velocity rotated into the map's basis. A 0/1 table by band index is never a
metallic head occupation: it cuts degenerate multiplets and, before the
multiplet rule, put $1/(\Delta E\,z^2)$ with $\Delta E\sim10^{-14}$ Ry into $S$
(Na $8^3$: $W=0$ at every sample up to 11 Ry). The route table is owned by
[self-consistency](../self_consistency.md#metals-direct-drude-head).

**What the head contributes on shell.** For any pole model the band-diagonal
head at the state's own energy is

$$
\Sigma^{\rm head}_{nk}(\epsilon_{nk})=\frac{1/2-f_{nk}}{\Omega N_k}\,W^c_{\rm head}(\omega\to0),
$$

so QSGW maps and on-shell energies see the static head only; the Drude
term moves them through $W^c(0)$ alone and moves $Z$ and off-shell values
through the rest. The scalar head fit therefore reproduces its static sample
exactly (§5.3). With $W^c(0)=-\langle v\rangle_{\rm cell}$ (perfect screening
of the cell) the head adds the uniform $-\langle v\rangle/(2\Omega N_k)$ to
exchange-plus-correlation and nothing band dependent; the Thomas–Fermi value
adds $(1/2-f)\langle 8\pi/(q^2\epsilon_\infty+\kappa^2)\rangle/(\Omega N_k)$,
$\pm1.9$ meV on Na $8^3$, falling as $1/N_k$.

**Accuracy of the static anchor.** $\kappa_{\rm TF}^2$ inherits the DOS
estimator. On an $8\times8\times8$ sodium mesh five estimators of $N(E_F)$ from
the same eigenvalues spread by about ±40 %; the tetrahedron estimator is the
anchor as the most stable discrete one, and the spread is a mesh-convergence
uncertainty on absolute QP energies, not a sign or scale error (claim 182).

The phenomenological alternatives of the published metallic MPA — a Drude pole
with free $\omega_D,\gamma$, or $W(q=0)$ copied from the nearest $q$ — are not
used.

## 5. Sigma with finite occupations

### 5.1 Body branches: occupation becomes weight, energy stays signed

The four causal branches keep their topology; membership changes.
`gw.mpa.sigma._branches` sums the occupied branch over every band with
$|f|$ above the floor, at weight $f$, and the empty branch over every band with
$|1-f|$ above it, at weight $1-f$. Energies stay signed against the state's
`mu_ry`; a caller Fermi level inconsistent with it refuses (one chemical
potential per map). The executor folds support mask × weight into one selector
on the same `build_G_tau(band_weight=)` seam the $\chi$ kernel uses: one
Green-function builder, no metal copy.

### 5.2 The Fermi-window split

Fractional weights give the crossing branches a negative-$E_A$ shell a few
smearing widths wide, so a nominally sign-definite deep-pole rectangle can
reach zero, and un-split it would mis-evaluate exactly the Fermi-surface states
a metal run is for. The planner deepens the shallow/deep pole edge by the
branches' worst negative-$E_A$ excursion, keeping every deep rectangle
sign-definite and routing the straddle into the crossing core; an insulator has
zero excursion and unchanged geometry. `tests/test_sigma_fermi_split.py` pins
the weighted split against an exact fractional reference and the $f>0.5$ mask
semantics as a failing control. Box construction and its dynamic-range cost are
[the Σ quadrature problem](sigma-quadrature-problem.md).

### 5.3 Head residues

The head injection carries the same split
(`head_correction.compute_complex_pole_head_sigma_diag`):

$$
\Sigma^{\mathrm{head}}_{nk}(\omega)=\frac{1}{V_{\mathrm{cell}}N_k}\sum_p R_p
\left[\frac{f_{nk}}{\delta_{nk}+\Omega_p}
+\frac{1-f_{nk}}{\delta_{nk}-\Omega_p}\right],
\qquad \delta_{nk}=\omega-(E_{nk}-E_F),
$$

with per-$(k,n)$ occupations, so a window straddling $E_F$ stays valid.
`sigma_dispatch` asserts that the head fit's stamped occupations equal the
body's live state (`gw.mpa.sigma.assert_head_body_occupation_match`). At
$\delta_{nk}=0$ the sum is $(1/2-f_{nk})W^c(0)/(\Omega N_k)$ with
$W^c(0)=-2\sum_pR_p/\Omega_p$, so the scalar head fit keeps its poles and
re-solves the residues with the sample nearest the origin as an equality
constraint (`gw.mpa.model._pin_static_head_sample`); the unconstrained refit
had missed that sample by 5.8 Ry bohr$^3$ on Fe $4^3$, $\pm7.7$ meV on shell.

### 5.4 Exchange, SX and Hartree

`build_Gij(occupation_state=...)` uses $G_{ij}=\operatorname{diag}(f)$ for
exchange and SX and refuses an electron-count mismatch above $10^{-8}$; step
occupations recover the insulating projector. Hartree takes the same state over
the complete density ([Hartree](hartree.md)).

### 5.5 Provenance stamps

Fit and head stores carry `occ_hash, mu_ry, smearing_family,
smearing_width_ry, occ_nelec` (`file_io.mpa_store._OCC_STAMP_ORDER`), with
`occ_hash` bound to the bytes of $f_{kn}$. The stamps gate **reuse**:
`assert_occupation_stamps` refuses an unstamped store under metallic reuse and
names a mismatched field.

## 6. Occupations and the QSGW map

The material class is read from the WFN: any occupation farther than $10^{-6}$
from an integer makes a metal. A metal's occupations are Fermi–Dirac with
$k_BT$ = `occ_smearing_width_ry` (the WFN does not store it), solved at fixed
$N$ by safeguarded bisection with capacity $2/(n_{\rm spin}n_{\rm spinor})$ per
band; the fixed-$N$ invariant holds to $10^{-10}$, and at startup the solved
table must reproduce the WFN's own table ($N_e$ to $10^{-8}$, $\max|\Delta f|$ to
$10^{-6}$). `fermi_reference = mp1_fixed_n` names this fixed-$N$ $\mu$ for every
family.

**One state per map, solved at entry.** Each QSGW map solves its occupation
state from the spectrum of the $H$ it was handed, and that one state reaches
$\chi$, the head, the shared-pole recipe and $\Sigma$. Occupations are then a
function of the iterate, $\mathrm{occ}=\mathrm{occ}(H)$, so the map
$F(H)=H_{\rm KIH}+\Delta H[H,\mathrm{occ}(H)]$ depends on $H$ alone — the
property an accelerated iteration needs, since every evaluation sees
occupations consistent with its own input. Fixed points are unchanged, and the
entry `eigh` already exists, so the rule costs nothing.

**One frequency reference.** $\Sigma_c(\omega)$'s grid is measured from that
fixed-$N$ $\mu$, and every consumer reads it from the stamp:
`sigma_mnk.h5` carries `omega_reference_ev` with its provenance and
`sigma_eval_rel_ev` with `sigma_eval_provenance`, and a from-disk reassembly of
an unstamped metallic cube refuses. A state whose energy the grid never sampled
takes $\Sigma(\omega=0)$ and is counted (`dynamic_sigma.OmegaCoverage`).

The metallic self-consistent loop — the stop rule, the scissor classes for
Fermi-crossing bands, the window around the Fermi-surface manifold and the
validated defaults — is [self-consistency](../self_consistency.md).

## References

- D. A. Leon, A. Ferretti, D. Varsano, E. Molinari and C. Cardoso,
  *Phys. Rev. B* **107**, 155130 (2023) (metallic MPA).
- M. Kim, G. J. Martyna and S. Ismail-Beigi, *Phys. Rev. B* **101**, 035139
  (2020) (CTSP).
- H. N. Rojas, R. W. Godby and R. J. Needs, *Phys. Rev. Lett.* **74**, 1827
  (1995) (space-time method).
