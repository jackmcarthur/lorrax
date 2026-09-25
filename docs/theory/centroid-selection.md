# Centroid selection: coverage, conditioning and cost

The production selector is periodic weighted Lloyd clustering, followed by
grid snapping, symmetry-orbit closure and optional candidate-Gram pruning.
This page distinguishes its objective from possible replacements. The fit
and its conditioning are owned by [ISDF](isdf-zeta-vq.md) and the
[rank policy](../dev/rank_truncation_policy.md); the selected band windows
must satisfy [basis adequacy](../dev/isdf_basis_adequacy_at_large_nband.md).
Full-grid pivoting and continuous point refinement below are designs, not
implemented production routes or measured speedups.

## What is being optimized?

Write the feature at position r as

\[
 f_{kmni}(r)=\sqrt{w_k}\,
       \psi^\dagger_{nk}(r)\Gamma_i\psi_{mk}(r),\qquad
 K(r,r')=\sum_{kmni}f_{kmni}(r)f_{kmni}(r')^*.
\]

The windows m in L and n in R are explicit; this uses the R-dagger/L
orientation of `isdf.core._gram_q0_fold_local`. Charge has one vertex I;
current candidate selection stacks the three Hermitian vertices alpha_i.
This **positive feature Gram** is not the signed transverse fit matrix.
Nor is it the frequency-sample Gram of the W interpolation pencil.
Their eigenvalues are different diagnostics.

`centroid.sampling_metric` computes s(r)=K(r,r) for charge and
s(r)=K(r,r)/alpha_fs^2 for current on the FFT grid. This global positive
scale changes neither Lloyd minimizers nor relative pivot/rank criteria;
absolute Gram magnitudes must retain the distinction.
`centroid.kmeans_cli` uses w(r)=sqrt(s(r)) as Lloyd's mass (before any
explicit `rho_power`). Thus the objective is

\[
 J(R)=\sum_r w(r)\min_\mu\min_{n\in\mathbb Z^3}
 (r-r_\mu+n)^T G(r-r_\mu+n),\quad G=AA^T.
\]

Positions here are fractional, and A has lattice vectors as rows. Lloyd
updates average the winning **unwrapped displacements** around each old
centroid; averaging fractional coordinates across the cell boundary is
wrong. For skew cells the winning image need not be componentwise rounding.
The same metric must serve assignment, movement and convergence.

Lloyd decreases spatial quantization error, not pair-density fit error.
For a smooth normalized feature u(r), its local representation error has
the tensor Re[(du/dr_a)^dagger (I-uu^dagger) (du/dr_b)], rather than a scalar
charge density times the Cartesian identity. Rapid changes of orbital
character therefore need not coincide with large s(r). This explains why
good geometric coverage and good linear independence can disagree; it does
not prove that replacing the weight alone improves GW.

In a three-dimensional high-resolution continuum approximation, scalar
weighted quantization gives point density proportional to w^(3/5), hence
s^(3/10) for the default weight. This asymptotic statement need not hold
after orbit closure, snapping or pruning, or for a thin slab/vacuum grid.

## What full-grid pivoting would fix, and what it would not

In exact arithmetic, pivoted Cholesky of K updates

\[
 d_j(r)=K(r,r)-\sum_{l<j}|L_l(r)|^2,\quad
 p_j=\operatorname{argmax}_r d_j(r).
\]

This is the remaining squared feature norm. Global pivots can visit a
region missing from a geometric candidate pool; candidate-only pruning
cannot. The trace sum(d_j) measures representation loss in this training
metric. It is not a bound in meV on QP energies, a uniform spatial covering
radius, or an assurance that every q-dependent fit is well conditioned.
Very small feature weight can leave a geometrical hole intentionally.

Greedy pivots maximize successive Schur complements, not the final minimum
eigenvalue or condition number. Once the sampled feature span is exhausted,
adding more points cannot restore full rank. Use the retained rank and
discarded weight under the existing solve policy; do not reject useful
overcomplete sets merely because the untruncated Gram is singular.
Here "global" means access to all candidate positions, not the globally
optimal subset or a globally optimal continuous placement. Both Lloyd and
greedy PC can settle on a poorer set than a different initialization/order.

For q != 0, the right projector belongs to k+q. A sum of selected-q Grams
is a valid positive training metric, but a healthy sum can hide a singular
individual q. A concatenated feature objective also asks for a common
representation across q, stronger than the separate zeta_q fits require.
Any multi-q selection must therefore be judged on individual downstream
fits and Sigma, with the same band windows, not only on the sum's spectrum.

## Exact column oracle and its real cost

Define the open-spin projector D_X,k(r,r')=sum_(n in X)
psi_nk(r) psi_nk(r')^dagger. Then the feature Gram above is

\[
 K(r,r')=\sum_{ki} w_k\,
 \operatorname{Tr}[\Gamma_i D_{L,k}(r,r')
                    \Gamma_i^\dagger D_{R,k}(r',r)].
\]

For scalar wavefunctions this reduces to D_L(r,r') D_R(r,r')*.
For spinors it is an open-spin contraction, not the product of two scalar
densities. The imaginary current vertex needs the conjugated endpoint;
`isdf.core` already owns that positive-Gram contraction. Do not substitute
the signed transverse C_q contraction or fold spin into the band sum.

For a block of chosen points, `isdf.pair_kernels.pair_projectors_lr` forms
D_X,k(r_p,G) from centroid faces and conjugated reciprocal coefficients.
The common Fourier and typed symmetry services can produce the column in
real space. This factorization avoids the full pair-feature matrix, but
does not make the initial diagonal free or avoid Fourier transforms of
projectors. Repeated blocks must reuse loaded wavefunctions.

With N_r FFT points and M selected points, unblocked exact selection costs
approximately O(M N_k N_r (N_b + log N_r)) for dense-grid projector columns,
plus O(N_r M^2) for residual projection, and stores O(N_r M) factors.
Actual plane-wave counts and spin multiplicity refine the first term;
orbit transport reduces raw-parent GEMM work but not all full-zone work.
N_G occupied sphere slots and N_r FFT points must not be conflated.

For illustration, N_r=10^6 and M=20,000 need 320 GB for one complex128 L,
or 20 GB per rank on P16, before projectors, FFT workspace or wavefunctions.
Scalar pivots repeatedly read that factor and require M dependent global
choices. Sharing the formal cubic scaling of zeta fitting is insufficient
evidence for a five-minute runtime.

An exact separable Euclidean distance transform is not a drop-in Lloyd
replacement: the current centroids move off grid between iterations, and a
skew-cell metric has cross terms. Snapping seeds before every assignment
changes the optimization. Jump flooding also changes the assignment unless
its candidate owners are checked; neither is an equivalent acceleration
merely because the input positions lie on a regular FFT grid.

Blocked selection makes residual updates GEMMs and amortizes communication.
Taking the top B old diagonals does not reproduce sequential pivoting:
nearby redundant features can fill the whole block. Residualize a larger
proposal panel, pivot within it, and update the **global** residual before
refilling. This remains an approximate block strategy unless selected pivots
are compared with the outside residual bound; it should not be called exact
global greedy PC. Complete symmetry orbits consume the point budget too.

## Off-grid points

The Fourier expression and its derivatives are well defined at continuous r:

\[
 \psi_{nk}(r)=N_r^{-1/2}\sum_G c_{nk}(G)e^{2\pi i(k+G)\cdot r},\quad
 \partial_{r_a}\psi_{nk}=N_r^{-1/2}\sum_G
 2\pi i(k+G)_a c_{nk}(G)e^{2\pi i(k+G)\cdot r}.
\]

Continuous points could refine coverage or a regularized volume objective.
Neither grants a global optimum or ensures pencil stability. More basically,
the current centroid contract uses integer grid indices: grid-residue phases,
grid pullbacks, orbit packing and C_q sampling share that contract. Wrapped
FFT-box G indices are equivalent to physical signed G **only on the grid**;
inserting fractional positions into the existing residue phase is wrong.
Off-grid support must use the loader's paired (k,G) convention and change the
canonical geometry/transport once, including nonsymmorphic translations.
Simply saving unsnapped Lloyd points is not a supported implementation.
The face loaders already use direct Fourier evaluation, but off-grid C_q
must also come from those directly sampled projectors, rather than grid
extraction from Z_q. This is a change to the shared representation contract,
not just to the selector's output format.

## A coverage-preserving alternative to full-grid factors

A useful design objective is to minimize the worst normalized feature-fit
residual over representative q, subject to a point budget, complete symmetry
orbits, and a covering-radius bound on the physically weighted support:

\[
 E_q(S)=\frac{\operatorname{tr}\left(K_q-
 K_{q,:,S}K_{q,S,S}^{+}K_{q,S,:}\right)}{\operatorname{tr}K_q}.
\]

The pseudoinverse uses the existing conditioning policy. This objective
states separately what coverage protects and what feature fitting optimizes;
it does not promise a well-conditioned W pencil. A practical approximation
could retain a CVT scaffold, enrich candidates from omitted high-residual
regions, and exchange redundant whole orbits without breaking coverage.
Rescoring outside the original candidate pool is essential: pruning a fixed
pool cannot discover a missing region. The residual is an assessment target,
not a proposal to materialize K_q or evaluate every q at every swap.

Even this smaller search must price wavefunction faces, not just its Gram.
For example, 30,000 candidates give a 14.4 GB complex128 Gram in total, but
at 64 full-zone k points, 1024 bands and four spin components, a face sharded
only on one axis of a 4x4 mesh occupies 31.5 GB per rank. Two such faces
already exceed a 40 GB GPU. Streaming or a different all-rank layout is
necessary before claiming that this candidate count fits.

## Decision criteria

First separate input reads, diagonal construction, Lloyd assignment/update,
candidate Gram and pruning in a real selector timing. Benchmark proposed
arithmetic against that breakdown. Exact periodic distance and a cheaper
equivalent assignment are useful independently of a new physics objective.
Then compare global blocked selection or continuous refinement at fixed M
and fixed windows by fit residual/discarded weight at several q, effective
conditioning, and a QP reference. Keep geometrical coverage as a diagnostic,
not a substitute for those observables. Do not make an unmeasured method the
default, or claim the 20,000-point target from a small-fixture timing.
