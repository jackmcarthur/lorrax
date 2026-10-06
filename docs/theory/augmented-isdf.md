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
The lift is isometric, \(U^\dagger U=I\), but reconstruction need not be:
\(\langle UT\psi_m|UT\psi_n\rangle=\langle T\psi_m|T\psi_n\rangle\).
The full reconstructed overlap, including smooth–atomic cross terms, is an
independent diagnostic. The explicit manifest mode `full_wfn_lowdin` measures
this overlap on every available WFN band, forms its symmetric inverse square
root A, and uses the carrier U(Tψ A). It rotates the smooth Fourier rows, atomic
coefficients and sample faces with the same A before selecting the public fit
window. Padded bands remain exactly zero. The raw mode `none` is diagnostic.

This convention retains the original DFT energy labels in the existing GW
contractions. It therefore defines an effective reconstructed-vertex model;
it does not claim that the mixed orbitals are exact AE eigenstates. Changing
the available WFN window changes A and requires an observable convergence
control. Frozen-core orbitals are not added to the occupied GW manifold.

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
resident Fourier rows at those positions before reconstruction and the common
full-WFN factor A. A rounded FFT gather would consume a different orbital map.
The current admission is augmented charge fitting; an unsupported FFT-only
refitting consumer refuses this basis explicitly.

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

The compensation identity above assumes ordinary 3D Coulomb and disjoint
supports. Dimensional truncation requires a separate boundary proof. Transverse
Breit electrostatics also contains the inverse bi-Laplacian: ordinary charge
multipole cancellation is insufficient. A Cartesian current construction needs
both its ordinary and second radial moments, or an explicitly transverse
vector-harmonic construction. The grouped-current contraction refuses a scalar
local provider rather than applying this charge identity to it.

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
