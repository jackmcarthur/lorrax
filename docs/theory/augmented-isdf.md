# ISDF on atom-reconstructed orbital samples

The ordinary ISDF factorization survives a linear atomic reconstruction. Its
band-pair coefficient remains a product of two sampled orbitals. The sample
values and the fitted Coulomb metric change together; the Green, screening and
self-energy contractions consume the same two-index carriers.

## One orbital map and one interpolation metric

Let the authenticated atomic partial waves define

\[
T\psi=\psi+\sum_{Ai}c_{Ai}[\phi^{AE}_{Ai}-\phi^{PS}_{Ai}],\qquad
c_{Ai}=\langle p_{Ai}|\psi\rangle.
\]

The reconstruction dual is formed from the local pseudo-wave overlap, not from
the Kleinman–Bylander factors of the nonlocal pseudopotential. The AE and PS
optimal partial waves use the same principal-component coefficients. Source
operator authentication is independent of reconstruction accuracy: matching the
regenerated UPF does not certify the held-out scattering waves.

For four components the required orbital is

\[
\Psi=UT\psi,\qquad U=\begin{bmatrix}I\\X\end{bmatrix}
 (I+X^\dagger X)^{-1/2},\qquad X={\alpha_{FS}\over2}\sigma\cdot p.
\]

Linearity gives

\[
UT\psi=U\psi+\sum_{Ai}c_{Ai}U(\phi^{AE}_{Ai}-\phi^{PS}_{Ai}).
\]

Every atomic difference therefore receives the same Fourier multiplier
\(r(K)=(1+\alpha_{FS}^2K^2/4)^{-1/2}\) and the same \(\sigma\cdot K\)
as the smooth orbital. An unnormalized small-component derivative added after
the smooth lift is a different operator. The normalized atomic differences have
short noncompact tails, so any buffered sphere needs a measured tail bound.
The production field uses a compact normalized graph. If H denotes the served
large-component Hermite interpolant, it defines

\[
\delta L_i=wH[R\delta\phi_i],\qquad
\delta S_i=X\delta L_i,\qquad
\delta\chi_i=R^{-1}\delta L_i.
\]

The radial C² taper is one through `cache.taper_start` and zero at the declared
support radius. Its start must be at or beyond every authenticated native
atomic radial bound. The lower field is derived from that **same** large
interpolant, including the taper derivative and the regular origin limits.
The resulting orbital is exactly in the prescribed free graph,
\(\Psi=U(\psi+\sum_i c_i\delta\chi_i)\). Its implicit Pauli correction differs
from the unwindowed raw correction above; it is not a pointwise normalization.
The generic uncut Hankel utility remains an operator control. A hard cutoff
on separately interpolated upper/lower fields would add a derivative surface
term and is not the production model.

The explicit `ae_large_preserved_free_graph` target instead prepares the
large field from a matched native Dirac large wave and the normalized free
pseudo reference. Its declared native window gives
\(\Delta L=W(P_{AE}/r-L_{PS,free})\). The finite-momentum preparation forms
\(\Delta\chi=R^{-1}\Delta L\), so one canonical unfolding preserves
\(\Delta L\) and gives \(\Delta S=X\Delta L\). The final compact field
uses the same Hermite/taper derivative above. Native Dirac Q is retained as
an atomic diagnostic; it is not independently inserted into the free graph.
The native matching window, pseudo exterior completion, final compact taper
and Coulomb compensation radius have separate meanings and identities.
The [atomic-cache contract](../reference/augmentation_cache.md) binds them.

The lift is isometric, \(U^\dagger U=I\), but reconstruction need not be. The
explicit manifest mode `full_wfn_lowdin` measures the **actual served
four-spinor** overlap on every available WFN band. With C the unrotated atomic
coefficients, D the served-field/source overlaps and B the served local Gram,
this metric is sourceGram plus the sum of D†C+C†D+C†BC. In this equation C,D
have function-by-band orientation; the API stores band-by-function rows. Its
symmetric inverse square root A rotates the full smooth Fourier rows, atomic
coefficients and served D before selecting the public fit window. Existing
FFT faces receive the same rotation. Fractional faces are sampled from the
rotated public Fourier rows, which is algebraically equivalent to sampling
every full-band row first and then rotating those samples. Padded bands remain
exactly zero. Native ideal-Pauli overlap is not substituted for this changed
field. The raw mode `none` remains diagnostic.

This convention retains the original DFT energy labels in the existing GW
contractions. It therefore defines an effective reconstructed-field model;
it does not claim that the mixed orbitals are exact AE eigenstates. Changing
the available WFN window changes A and requires an observable convergence
control. Frozen-core orbitals are not added to the occupied GW manifold.

A fresh invocation authenticates the complete atomic manifest once and passes
that loaded artifact to overlap construction and fitting. Its fitted identity
also stamps the restart; no process-global file cache is used. Every restart
invocation independently authenticates the requested manifest before accepting
stored tensors and samples.

For Coulomb-only GW, exchange, screening and correlation receive the same
reconstructed sample bundle and fitted Coulomb tensor. There is no separate
smooth screening density. The optical q-to-zero derivative remains an
independent consistency requirement; common finite-q samples alone do not certify that derivative.
The reconstructed occupied Hartree source is captured during fitting, before
donation. Its public receiving hook evaluates the same served states with
the independent full-FFT Poisson owner. Hartree omits G=0, whereas exchange
has its canonical finite-cell head and explicit Fourier-body cutoff. Their
operator domains require separate comparisons.

## Charge Hartree through the same ISDF factor

The numerical seam `isdf.atomic_hartree` contracts reconstructed occupied
sources and receiving pair densities; `gw.augmentation_hartree` and
`gw.augmentation_hartree_receiving` own its public handoff. Physical source
occupations \(f_{nk}\) and normalized full-zone
weights define \(\rho=\sum_{nk}w_kf_{nk}\Psi_{nk}^\dagger\Psi_{nk}\);
they are independent of the endpoint weights used in the fitting loss.
Typed scalar point transport sums the occupation trace after spin and Bloch
phases cancel. Source samples, smooth Fourier rows and exact local monopoles
must use the same full-WFN factor and served-field identity.

Let \(\eta^g_{ij}=(\Omega/N_{FFT})\Psi_i^\dagger\Psi_j\) denote the
grid-normalized receiving pair. The existing charge normal equations fit
\(\eta^g_{ij}(r)=\sum_\mu\eta^g_{ij}(r_\mu)\zeta_\mu(r)\). Thus

\[
J_{ij}=\sum_\mu\eta^g_{ij}(r_\mu)F_\mu,\qquad
F=C_0^{+}H,
\]

where H contracts the **raw** smooth and local normal-equation RHSs with
the physical Hartree source potential. There is no extra conjugation of
\(\eta^g\). The smooth contribution is a full FFT-grid sum before the
route-G transform, with no \(1/N_{FFT}\) factor. The local scalar dual
acts on the provider's existing delta, PS and exact-Y00 columns, with the
single \(N_{FFT}/\Omega\) conversion for their grid units. Smooth and
local scalar RHSs are added before applying the existing charge factor,
including its actual conditioning transformation. No new Gram or full-grid
zeta artifact is needed.

For the bulk operator, the compensated source uses ordinary periodic 3D Poisson on the full FFT
grid with G=0 zero. Its local dual retains delta--delta minus
compensation--compensation, both PS--neutral adjoints, coherent exact-M0
enrichment and both periodic neutral-potential means. The source neutral
mean shifts the smooth potential and exact receiving charge; the receiving
neutral mean acts on the delta radial field. This matrix has no factor
one-half. A receiving-functional identity alone does not establish actual
Hartree ISDF accuracy: physical-source capture and matrix convergence must
also be measured. An augmented direct field is not a complete AE
Hamiltonian or quasiparticle prediction without consistent ionic, frozen-core
and exchange-correlation reference treatment.

For the aligned slab operator, the smooth and neutral cross terms use the
public two-dimensional truncated Coulomb kernel on the full FFT grid. Write
\(\delta=D+\epsilon g_0\),
\(C=C(M_{\rm rad}[D])+\epsilon g_0\), and
\(N=D-C(M_{\rm rad}[D])\). The exact-M0 enrichment cancels in \(N\).
With \(S\) the smooth density, the source and receiving bilinear form is

\[
J_{ut}=(S_u+C_u|v_{2D}|S_t+C_t)
       +(S_u|v_{2D}|N_t)+(N_u|v_{2D}|S_t)
       +\sum_A[(\delta_{uA}|v_{\rm free}|\delta_{tA})
               -(C_{uA}|v_{\rm free}|C_{tA})].
\]

The unchanged free on-site difference requires the compact-pair support
bound stated below. The source-neutral Fourier field contributes only to
the smooth potential; the receiving-neutral radial adjoint contracts only
the smooth source. In the existing local-response layout the latter occupies
`PS_delta`, while `delta_PS` is zero because that contribution is already in
the smooth potential. Exact-M0 compensation and local enrichment remain
coherent. No bulk Gamma neutral-mean term is added. The slab operator excludes
only total \(G=0\), using the same actual kernel for all remaining modes.
Its radial Fourier tables cover the full FFT corner and use the prepared
`fourier_points` convergence control, defaulting to 4097.

This private operator is labelled
`ordinary_2D_truncated_full_FFT_G0_zero`, with
`neutral_mean_policy=none_direct_smooth_neutral_fft` and an authenticated
slab-kernel binding. Independent compact-source/smooth-receiving and
smooth-source/compact-receiving CPU controls check both mixed adjoints,
complex conjugations and exact-M0 cancellation. Public slab GW and the
private full reconstructed-slab preparation path remain guarded pending
native action and actual source/artifact validation; these mathematical
controls do not certify a CrI3 Hartree or quasiparticle result.

At interpolation points define

\[
A_{mnk,q,\mu}=\Psi^*_{mk}(r_\mu)\Psi_{n,k+q}(r_\mu).
\]

Atomic samples may be genuine fractional coordinates. The centroid file declares
`# centroid coordinate kind: fractional`; an absent declaration retains the
legacy FFT-index interpretation and snapping. The typed basis carries the exact
float64 positions, including lattice images, through symmetry transport, fitting,
fingerprints and restart receipts. A fractional ζ header stores `r_mu_crystal`
and its coordinate kind and omits `r_mu_fft_idx`. Integer Bloch wraps refer to
the original physical positions. The smooth samples are evaluated from the
resident Fourier rows at those positions. In full-WFN mode the complete factor
A acts on the reciprocal rows and atomic coefficients first, so only the
public band columns need the sample DFT. All available bands still enter the
overlap and rotation. A rounded FFT gather would consume a different orbital map.
Fractional charge fitting is admitted. The static-current extension
below uses the same typed geometry; an unsupported FFT-only refitting consumer
refuses this basis explicitly.

The normal equations of the [existing conjugation-closed pair
set](isdf-zeta-vq.md) are

\[
C_q=A_q^\dagger A_q,\qquad Z_q=A_q^\dagger\rho_q,
\qquad\zeta_q=C_q^+Z_q.
\]

Split the physical density into a smooth term and local corrections,
\(\rho=\rho^s+\sum_A\Delta\rho_A\). The same factor yields
\(\zeta^s=C^+Z^s\) and \(\Delta\zeta_A=C^+\Delta Z_A\).
The raw-parent band contractions and k correlation form both RHS terms without
an explicit band-pair carrier. Route G must receive the augmented samples when
forming \(Z^s\); resampling the smooth plane waves there would solve a different
normal equation from the augmented \(C\).

The optional charge input `zeta_occupied_weight` changes the fitting loss,
not the occupied density or the orbital map. At weight \(w\geq1\), an
endpoint below the authenticated integer occupied boundary receives weight
\(w\), while every other endpoint in its original fitting window retains
weight one. Thus a pair has the product of its two endpoint weights. The
same weights enter \(C\), the smooth Fourier RHS, the local AE-minus-PS
and PS RHSs, and the auxiliary monopole RHS before ordered LR+RL completion.
All original pairs remain covered. Unit weight preserves the default arrays
and provenance; a nonunit policy binds its weight and occupied boundary into
the charge-fit and restart identity. A smaller exchange residual alone does
not certify screening or correlation accuracy under this changed loss.

An explicit manifest control `charge_fit.conditioning=unit_diagonal` may
equilibrate the sampled Gram. For \(D_{\mu\mu}=C_{\mu\mu}^{-1/2}\), factor
\(C'=DCD\) with the existing rank policy and use the physical factor
\(DB'\), where \(B'B'^\dagger=C'^+\), for both RHSs. This changes the
rank-truncation metric and is authenticated as a fitting choice. It does not
relax the condition-number ceiling, add ridge regularization or certify the
sample span. The default factor and its provenance remain unchanged.

## Compensated local Coulomb integrals

For ordinary three-dimensional Coulomb, expand each atom-local interpolation
correction in orthonormal complex spherical harmonics,

\[
\Delta\zeta_A(r)=\sum_{LM}\Delta\zeta_{A,LM}(r)Y_{LM}(\hat r).
\]

The radial Poisson kernel is

\[
v_{LM}(r)={4\pi\over2L+1}
\left[r^{-L-1}\int_0^r t^{L+2}\rho_{LM}(t)dt
+r^L\int_r^\infty t^{1-L}\rho_{LM}(t)dt\right].
\]

Two cumulative integrations apply this kernel in linear radial work. Quadrature
weights integrate \(dr\); the \(r^2\) volume factor is explicit. The radial
Coulomb bilinear form is Hermitian before any numerical symmetrization.

Choose a smooth compact compensation \(g_A\) with exactly the same retained
moments \(Q_{LM}=\int r^{L+2}\Delta\zeta_{A,LM}(r)dr\), and put
\(b_A=\Delta\zeta_A-g_A\). With nonoverlapping spheres, \(b_A\)'s potential
vanishes outside its sphere at the represented angular degrees. Thus residual
interactions between distinct atoms vanish, while the compensation charges
carry interatomic electrostatics through the periodic Fourier solve.

Write \(s=\zeta^s\), \(g=\sum_Ag_A\). The complete metric is

\[
V=(s+g|v|s+g)
+\sum_A[(\Delta\zeta_A|v|\Delta\zeta_A)-(g_A|v|g_A)]
+(s|v|\sum_Ab_A)+(\sum_Ab_A|v|s).
\]

The final two terms survive because the smooth density penetrates the spheres.
Zero external multipoles do not make its potential harmonic inside them. With
a complete Fourier representation the accumulated reciprocal expression is
equivalently

\[
V_G=s^*vs+s^*v\Delta+\Delta^*vs+g^*vg.
\]

The default provider uses this reciprocal expression over the configured finite
Coulomb body cutoff. At finite cutoff it omits the smooth–neutral cross terms
above that cutoff, so it requires a separate cutoff-convergence control. The
source FFT cutoff `ecutrho` is not the body cutoff: the latter comes from the
canonical Coulomb geometry and must be recorded in every reference comparison.
A different body cutoff gives a combined operator and fitting difference.

An exact local treatment of the smooth–neutral cross terms instead evaluates
\((s_A|v|b_A)+(b_A|v|s_A)\) inside each sphere, where \(s_A\) is the restriction
of the same fitted smooth density. It uses the same interpolation factor and
local pseudo-density normal equation. The explicit manifest policy
`charge_metric.smooth_neutral_cross=onsite` selects this expression. The
reciprocal body then contracts only the compensated density \(s+g\); the
physical head and written Fourier density still contain \(s+\Delta\). The
local pseudo-density RHS shares the same factor and units as the correction
RHS. The default policy retains its existing arithmetic. At finite body cutoff
these policies define different operators and must be authenticated separately
in an accuracy comparison. The finite-cutoff on-site difference is signed;
positive semidefiniteness requires a physical-span check, not eigenvalue clipping.

The explicit `charge_metric.body_metric=physical_low_local_high` policy
instead completes the periodic correction before subtracting its low-G part:

\[
V_+=B_{\mathrm{low}}^{\mathrm{actual}}(s+\Delta,s+\Delta)
    +B_{\mathrm{per}}^{\mathrm{bare}}(\Delta,\Delta)
    -B_{\mathrm{low}}^{\mathrm{bare}}(\Delta,\Delta).
\]

For a positive low-G kernel this is its action on the physical density plus
the positive bare high-G correction. Positivity belongs to the represented
functional; small floating-point negative eigenvalues are retained and
compared with backward error. No clipping is part of fitting or screening.
The complete bare periodic correction is

\[
B_{\mathrm{per}}^{\mathrm{bare}}(\Delta,\Delta)
=\sum_A[B_{\mathrm{free}}(\Delta_A,\Delta_A)
        -B_{\mathrm{free}}(g_A,g_A)]
 +B_{\mathrm{per}}^{\mathrm{bare}}(g,g)
 -{2\over\Omega}(Q_\Delta^*\Phi+\Phi^*Q_\Delta).
\]

Here the bracket is the bilinear form in Ry, \(Q_\Delta\) is the correction
charge, and \(\Phi\) is the integral of the neutral free potential. Both
adjoints are necessary. The global compensation metric excludes Gamma G=0;
the physical low-G density keeps the existing head convention. This policy
requires the same exact served monopoles used by its local and Fourier
pieces. Its normal equations solve only correction and exact-M0 columns;
smooth cross terms are already in the physical low-G action, so no local PS
RHS is assembled or solved.

`isdf.positive_charge_metric` retains the global moment Gram on
`P(None,'x','y')`. The solved moment rows move from their existing q owners
to that face once, and two public N,N GEMMs complete the metric before the
usual centroid unpack. The cache is an explicitly pinned geometry artifact;
it has no orbital-frame dependence or imported runtime code. Its
[cache contract](../reference/augmentation_cache.md) specifies finite-cutoff
evidence and unit conversion. The same ordinary ISDF factor and plane-wave
pass supply all these columns. Downstream charge screening and Sigma
contractions consume the resulting tensor unchanged.

For the public aligned slab kernel, the positive completion uses the same
kernel in the physical low-G action, its correction-only subtraction and
the global compensation Gram. Let \(z_c=L_z/2\) be the public truncation
half-height. If the unwrapped atomic layer satisfies
\(\max_{AB}|z_A-z_B|+2R<z_c\), all compact density pairs and their in-plane
images see the ordinary \(1/r\) kernel; every nonzero out-of-plane image is
completely excluded. The existing disjoint-sphere condition still applies.
The free on-site difference consequently remains unchanged, and

\[
B_{\mathrm{slab}}(\Delta,\Delta)
=\sum_A[B_{\mathrm{free}}(\Delta_A,\Delta_A)
        -B_{\mathrm{free}}(g_A,g_A)]
 +B_{\mathrm{slab}}(g,g).
\]

The bulk Gamma neutral-mean adjoints are absent here. Truncation produces
boundary strips whose potential integral cancels the free neutral mean;
compact layer densities do not sample those strips. At Gamma only total
\(K=0\) is excluded. The \(K_\parallel=0, G_z\ne0\) modes remain under the
public slab kernel, including its nonzero odd vertical harmonics.

This compact-support identity does not certify the separate occupied
Hartree source: its smooth plane-wave density can sample the boundary
strips. The kernel-aware full-FFT Hartree construction above evaluates both
smooth-neutral cross terms with the actual slab kernel, without a bulk
neutral-mean correction. Public scalar slab preparation binds a complete
common compact target and its paired fields through one manifest request;
the request declares either the normalized RKB charge carrier or the explicit
zero-small Pauli control on the same common frame. Carrier resolution precedes
source sampling and basis receipts, and restart admission retains that choice.
The same bound artifact enters fitting, occupied-source preparation and
restart identity checks. The live source Gram, C projections, common A and
physical WFN identity must still pass their existing owners. The first
public envelope is fixed one-shot, unsmeared and headless. The slab
correction owner alone does not enable a slab head correction, current
screening, self-consistency or a physical fitting-accuracy certificate.

This completion does not restore omitted high-G smooth self and cross terms.
Their cutoff must converge separately. A smooth source cut at energy
\(E_{\rm wfc}\) has pair-density support through \(4E_{\rm wfc}\). Covering
that sphere also requires transfer-dependent physical Miller representatives
at the Nyquist boundary of an even FFT grid. A fixed `fftfreq` table may fold
an allowed boundary coefficient onto a reciprocal vector outside the sphere;
cutoff convergence on that fixed table alone does not establish complete
plane-wave support. Reconstruction, radial/angular
representation, interpolation rank, normalization window and optical-head
consistency retain separate error controls.

An optional served-field monopole enrichment addresses radial sampling error
without imposing an overlap identity. Let \(C\) be atomic reconstruction
coefficients, \(D\) the overlap of the same served four-spinor difference with
the smooth source, and \(B\) the exact served-field atomic overlap. The
finite-q pair moment is

\[
M=D_o^\dagger C_t+C_o^\dagger D_t+C_o^\dagger B C_t.
\]

Here the subscripts denote independent outgoing and incoming states, including
their actual lattice phases. The overlap is integrated on every served cubic
Hermite cell; a native partial-wave overlap or a forced identity would define
a different field. The same full-WFN factor rotates \(C\) and \(D\) before
the public band crop. Prepared species overlaps and raw-parent \(D\) artifacts
are mandatory and source-bound, so recurring fitting never rebuilds them.

The signed block form \(H=\begin{bmatrix}B&I\\I&0\end{bmatrix}\) represents this
moment as auxiliary charge functionals on a fixed true-point quadrature. Those
fields are typed functionals, not additional reconstructed wavefunctions. The
existing canonical pair-compression kernel forms their RHS with the same
symmetry transport and interpolation factor. If its fitted moment differs from
the radial interpolant by \(\epsilon\), set
\(\Delta'=\Delta+\epsilon g_0\) and \(g'=g+\epsilon g_0\), with the existing
unit-monopole compact profile \(g_0\). Their neutral difference remains unchanged.
The local correction gains \(h^\dagger vb+b^\dagger vh\),
\(h=\epsilon g_0\); the quadratic term cancels. Both physical and compensation
Fourier transforms receive the same \(\epsilon F_{g_0}\). This is an explicit
density model requiring radial, angular and exchange convergence controls.

The on-site difference is added after the Fourier stream. Local Fourier values
follow directly from the Fourier–Bessel transform of \(\Delta\zeta_{LM}\)
and \(g_{LM}\); the hard local density never requires a global hard FFT grid.
All terms are contracted after the common \(C\) solve. Forming
\(\overline{C^+}M\overline{C^+}\) would square the conditioning amplification.
The physical head and any written reciprocal zeta tile contain \(s+\Delta\).
A reciprocal zeta file alone does not contain the high-momentum local metric.
When a charge file is requested, its header is created during fitting and its
physical payload is written in the same streamed pass that forms V, after the
local provider is attached. The completion and provenance receipts follow the
collective writer's close. A partial fit or failed stream keeps the header
incomplete. The complete GW restart separately stores the mixed Coulomb metric
and remains bound to the augmentation manifest.

## Units and supported electrostatics

LORRAX samples a normalized plane-wave orbital with \(1/\sqrt{N_r}\), whereas
physical atomic samples use \(1/\sqrt{\Omega}\). Therefore atomic wavefunctions
enter the stored sample carrier multiplied by \(\sqrt{\Omega/N_r}\).
The smooth zeta transform is an unnormalized grid sum. An atomic Fourier
integral \(F_\Delta=\int\Delta\zeta(r)e^{-i(q+G)r}d^3r\) enters it with
\(N_r/\Omega\). The existing reciprocal Coulomb table already contains
\(8\pi/(\Omega|q+G|^2)\) in Rydberg. Consequently a physical Hartree-unit
on-site radial metric receives \(2(N_r/\Omega)^2\), once.

The bulk compensation identity assumes ordinary 3D Coulomb and disjoint
supports. The aligned-slab completion additionally requires the compact-pair
support bound above and its own kernel-bound cache. The static
transverse construction below also cancels the inverse bi-Laplacian exterior
field; its grouped-current contraction refuses a scalar local provider.

## Static reconstructed current

The public extension of `atomic_reconstruction_dir` admits
`bispinor_gw=bare_transverse` only for `compute_mode=x_only`,
`qp_solver=one_shot_dft`, fixed DFT density, three-dimensional periodic Coulomb,
`head_correction=off` and no transverse head overlay. Occupation smearing is
unset, response occupation broadening is zero, and the actual WFN occupations
must describe an insulator before either fresh or restart I/O. Ordinary public fresh/file/restart reuses the same complete tensors and
samples. This extension supplies bare
static transverse exchange; it does not admit reconstructed screened/dynamic
photon models. Coulomb-only GW retains its existing charge screening and
correlation contractions.

Charge and current use one reconstructed four-spinor and one full-WFN factor
A. In the AgI validation this is the full physical 152-band factor before the
public 120-band crop. The physical pair densities are

\[
\Gamma^0=I_4,\qquad
\Gamma^i=\alpha_i=\begin{bmatrix}0&\sigma_i\\\sigma_i&0\end{bmatrix},
\qquad
j^i_{mn}=\Psi_m^\dagger\alpha_i\Psi_n
=L_m^\dagger\sigma_iS_n+S_m^\dagger\sigma_iL_n.
\]

The served lower field is still \(S=X L\), including the derivative of the
served large-field taper. Each current component uses its own existing Gram
factor for both smooth and local RHSs, with unit endpoint loss; the charge
factor and any charge loss weights are separate. Both families may use distinct
symmetry-closed fractional point sets. The same current matrices and conjugate
endpoints enter the Gram, RHS, Coulomb tensor and self-energy contraction.
The small components already contain \(1/(2c)\); multiplying the transverse
kernel by another \(c^{-2}\) would count the relativistic suppression twice.

For \(K=q+G\ne0\), the static kernel in Rydberg units is

\[
v^{TT}_{ij}(K)=-\frac{8\pi}{\Omega |K|^2}
 \left(\delta_{ij}-\frac{K_iK_j}{|K|^2}\right).
\]

The real-space Breit factor one-half is accounted for by the transverse
projector under Fourier transformation; no additional one-half multiplies
this Fourier tensor. This headless static model has no transverse \(\Gamma\)-cell head
at \(K=0\). The declared physical body mask
and mini-cell averaging policy remain part of the operator comparison.

For each Cartesian local correction and spherical harmonic, compensation
matches both radial moments

\[
Q^{(s)}_{A,LM,i}=\int_0^R r^{L+2+s}
 \Delta\zeta^i_{A,LM}(r)\,dr,\qquad s=0,2.
\]

Matching \(Q^{(0)}\) cancels the Poisson exterior field, while matching
\(Q^{(2)}\) also cancels the exterior field from the inverse bi-Laplacian.
The provider retains the local correction-minus-compensation bilinear forms
and the smooth–neutral adjoints. Physical Fourier publication contains
\(s+\Delta\), while compensation \(g\) belongs to the electrostatic
completion. Writing \(s+g\) as the physical current would change the current matrix element.

Ordinary restart stores the charge tensor and six independent transverse
blocks, with the lower off-diagonal blocks supplied by Hermitian symmetry.
These seven complete V tensors, both sample families and their typed
coordinate/source receipts feed the existing self-energy contraction. A
complete restart reuses them without atomic reconstruction, local-provider
attachment or another fit. Fresh/restart equality validates the handoff;
independent full \(\Sigma_X\) matrix comparisons validate fitting accuracy.
Neither establishes atomic transferability, continuum convergence, screened
frequency accuracy or calibrated quasiparticle band splittings.

## Implementation and accuracy

The [raw-parent fitting stage](../architecture/zeta_fit_face_psi_cct.md) remains
the owner of \(C\) and its factor. `isdf.local_rhs` forms the rectangular local
normal equation using the same four Dirac-half contractions and typed symmetry
transport as the ordinary Gram. `isdf.augmentation` owns the radial Poisson and
mixed-metric equations; `isdf.atomic_coulomb` supplies streamed radial callbacks
to `ZetaG.contract_v`. Tables are small host precomputations. Coefficients,
Fourier tiles and the Coulomb accumulator retain their q-owner sharding.

Normalized atomic radial caches are immutable, optional prepared artifacts.
Their loader binds atomic payload and metadata, quadrature and support controls,
and the exact normalized-lift owner sources. An explicit missing or mismatched
cache refuses; it never silently rebuilds. One-time atomic preparation and
cache generation are reported separately from recurring fitting time.

Normalized species caches use the explicit compact-field descriptor and
source-bound v2 schema. Raw Hankel arrays remain preparation/tail evidence;
the served lower field comes from the upper gradient. Served overlap caches
also use v2: GL10 integrates the Hermite/taper products, with cell splits at
the taper and support boundaries, and the lower Fourier row follows the same
upper row through σ·K. Old or mismatched artifacts refuse. Prepared served
species and raw-parent overlaps are required for full-WFN normalization even
when auxiliary monopole enrichment is disabled.

Preparation uses two allocated-compute commands. First,
`tools/generate_augmentation_cache.py --manifest DIR --output NEW_DIR
--served-moments` creates normalized and species served artifacts. Merge its
emitted patches into the existing `cache` and `served_moments` dictionaries.
Then `tools/generate_raw_parent_moments.py --manifest DIR --wfn WFN.h5
--output NEW_DIR` computes unrotated served overlaps D and pseudo-dual
projections C on every physical band, with raw parent blocks distributed
across the processes, and emits the raw-parent file/hash patch. Preparation
requires authenticated atomic Fourier dual caches. The v2 raw artifact binds
the complete source, reciprocal and atomic geometry, full band window,
atomic payloads, dual files and projection owners. Fitting loads these exact
C rows instead of rebuilding the projection, then applies the same full-WFN
factor before the public crop. A v2 artifact with missing C refuses; an
explicit older v1 overlap artifact retains the original live projection.
The preparation reader skips only the not-yet-created raw artifact; fitting
always uses the complete strict reader. The public raw producer checks
bounded independent host C and D contractions and strict reload.
Preparation costs are separate from the recurring fitting stage.

An independent optional pair, `cache.local_coulomb_fourier_file` and
`cache.local_coulomb_fourier_sha256`, supplies a prepared reciprocal table for
the chosen radial density interpolant. This artifact preserves the existing
cubic Fourier spline coefficients and knots exactly. Its source, quadrature,
origin treatment, angular degrees, compensation and payload are authenticated;
the consuming momentum range must fit its declared extent. Reload repeats the
ordinary direct-quadrature and zero-momentum pins. Missing or mismatched
explicit artifacts refuse. These keys do not change the separate normalized
atomic-cache controls. The rows are independent of cell volume and FFT size;
the provider applies its usual unit conversion after evaluation.

`tools/generate_local_coulomb_fourier_cache.py --manifest DIR
--maximum-wavevector KMAX --output NEW_DIR` prepares this table on one allocated
CPU rank and writes an immutable receipt. The ordinary factory remains the
default. Preparation time and complete fitting time are reported separately;
a fast reload alone does not meet a whole-stage performance target.

The route-G planner distinguishes the source centroid batch from the smaller
plane workspace it streams. When a complete owner plane axis does not fit, it
prices larger source batches with the existing streamed plane stage and the
same compiled-memory admission. Packing previews use the canonical orbit
assignment and omit expensive transport tables; the selected batch builds
those tables once. A preview refuses transport consumption. This changes
planning and allocation, not the sampled pair products or normal equations.

The higher-order charge metric uses an analytically integrated piecewise density
interpolant and the positive Poisson field-energy identity. Its moments and
Fourier transforms use the same interpolant, including regular behavior at the
origin. More accurate field integration does not repair undersampling of the
physical density; radial and angular refinement must still be measured.

Five errors must be distinguished: atomic transferability, normalized-RKB
spectral/tail convergence, local angular/radial and compensation-Fourier
convergence, Coulomb body-cutoff convergence, and the interpolation error on the
chosen orbital sample set. A fitting comparison must hold the operator cutoff
fixed, in addition to the source, reconstructed model and public band windows.
Their acceptance observable is the full held \(\Sigma_X\) matrix, with the
relativistic difference converged on its own meV scale. Point counts and atomic
PCA discarded weights are diagnostics; neither is an exchange certificate.
The complete fitting wall includes reconstruction, local fitting, contraction
and cold orchestration when compared with the incumbent.
