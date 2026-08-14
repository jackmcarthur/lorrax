# G-flat interpolation vectors and \(V_q\)

This page describes the current interpolation-vector and Coulomb-matrix
pipeline. Its purpose is to connect the ISDF normal equations to the few
layout choices that preserve their scaling. Historical real-space zeta files,
benchmark campaigns, and fixed source-line inventories are intentionally
omitted.

## 1. Open-spin pair density

For channel-dependent left and right band spaces, define

$$
P_{\mathbf k,\alpha\beta}(\mu,r)
=\sum_n
\psi^*_{n\mathbf k\alpha}(\mathbf r_\mu)
\psi_{n\mathbf k\beta}(\mathbf r).
$$

The charge channel traces the appropriate spin components. Bispinor channels
retain the open indices and apply their gamma-matrix factors after the lattice
transform. The same pair-density builder therefore serves charge and
transverse response without duplicating the wavefunction transform.

At fixed momentum transfer, the Galerkin matrices are lattice convolutions:

$$
C_q(\mu,\nu)
=\mathcal F_{\mathbf R\to\mathbf q}
\left[
\widetilde\gamma_L\widetilde\gamma_R\,
\overline{\mathcal F^{-1}_{\mathbf k\to\mathbf R}P_L(\mu,\nu)}
\mathcal F^{-1}_{\mathbf k\to\mathbf R}P_R(\mu,\nu)
\right],
$$

$$
Z_q(\mu,r)
=\mathcal F_{\mathbf R\to\mathbf q}
\left[
\widetilde\gamma_L\widetilde\gamma_R\,
\overline{\mathcal F^{-1}P_L(\mu,r)}
\mathcal F^{-1}P_R(\mu,r)
\right].
$$

The interpolation vectors solve

$$
C_q\,\zeta_q=Z_q
$$

independently for each \(q\) and response channel. The charge channel uses a
Hermitian positive-semidefinite factorization; more general channel blocks use
the certified pivoted solve selected by the same driver.

## 2. Why the solve is real-space chunked

The right-hand side has a full real-space axis and cannot be retained at
production size. The driver therefore:

1. loads band slabs of \(\psi(\mathbf G)\) into a bounded host cache;
2. evaluates \(\psi(\mathbf r_\mu)\) once for the persistent centroid legs;
3. builds and factors \(C_q\) once;
4. scans over bounded real-space chunks \(r_c\);
5. builds \(Z_q(\mu,r_c)\), solves for \(\zeta_q(\mu,r_c)\), and immediately
   accumulates that chunk into reciprocal space.

The band loop remains inside one compiled sharded kernel. Moving it outside
would repeatedly materialize the FFT box and turn a bounded workspace into a
sequence of large host/device transfers.

On a mesh `('x','y')`, the centroid axes of \(C_q\) are
`P(None,'x','y')`. During the \(Z_q\) build, the output centroid and real-space
chunk axes occupy different mesh axes, so neither becomes replicated. A
solver may stage through another sharding, but it must return to this native
matrix layout before the next physical owner consumes the result.

## 3. Direct accumulation into G-flat storage

Define the cell-periodic interpolation vector

$$
z_{q\mu}(\mathbf r)
=e^{-2\pi i\mathbf q\cdot\mathbf r}\zeta_{q\mu}(\mathbf r).
$$

For nonoverlapping real-space chunks,

$$
\widetilde z_{q\mu}(\mathbf G)
=\sum_c
\mathcal F\!\left[
\mathbf 1_{r\in c}\,
e^{-2\pi i\mathbf q\cdot\mathbf r}
\zeta_{q\mu}(\mathbf r)
\right]_{\mathbf G}.
$$

Linearity makes each chunk an additive update to one persistent G-flat
accumulator. Only the \(\mathbf q+\mathbf G\) sphere used by the Coulomb
contraction is retained. The final on-disk object is therefore

$$
\widetilde z[q_{\mathrm{irr}},\mu,G_{\mathrm{sphere}}],
$$

not a full real-space image and not a full FFT box. Padding outside each
logical sphere is exactly zero.

SlabIO creates and writes the distributed dataset collectively. The
interpolation-point axis remains flat-sharded across both mesh axes; no rank
gathers a full \(\mu\) slab for I/O.

## 4. Coulomb contraction

For the scalar charge channel,

$$
V_{q,\mu\nu}
=\sum_{\mathbf G\in\mathrm{sphere}(q)}
\widetilde z^*_{q\mu}(\mathbf G)\,
v(\mathbf q+\mathbf G)\,
\widetilde z_{q\nu}(\mathbf G).
$$

The G axis is reduced in bounded chunks. With left centroids on `'x'` and
right centroids on `'y'`, each chunk is a local matrix multiplication and the
result lands directly as `P(None,'x','y')`. The Coulomb service supplies
\(v(\mathbf q+\mathbf G)\), including dimensional truncation and the
long-wavelength slot; this pipeline does not reproduce those formulas.

For bispinor response,

$$
V_{q,\mu\nu}^{ij}
=\sum_{\mathbf G}
\widetilde z^{i*}_{q\mu}(\mathbf G)\,
v(\mathbf q+\mathbf G)\,
t^{ij}(\mathbf q+\mathbf G)\,
\widetilde z^j_{q\nu}(\mathbf G),
$$

where \(t^{ij}\) is the channel tensor. The driver evaluates only the unique
channel tiles and restores their Cartesian mixing after symmetry unfolding.

## 5. Irreducible-q cascade

The G-flat file uses the irreducible q wedge only when the centroid set closes
under the complete spatial-plus-time-reversal table. For a symmetry operation
\(s\), the transformed interpolation points satisfy

$$
\mathbf r_{\pi_s(\mu)}
=S_s\mathbf r_\mu+\boldsymbol\tau_s+\mathbf L_{s\mu},
$$

where \(\pi_s\) is a permutation and \(\mathbf L_{s\mu}\) is a lattice wrap.
These two tables determine the phase used to unfold \(V_q\):

$$
V_{Sq,\mu'\nu'}
=e^{2\pi i\mathbf q\cdot
(\mathbf L_{s\mu'}-\mathbf L_{s\nu'})}
V_{q,\pi_s(\mu'),\pi_s(\nu')}.
$$

Time-reversal rows apply the corresponding complex conjugation. The full
convention is owned by [Symmetry](symmetry.md).

Orbit closure is a correctness gate. If it fails, the driver computes and
stores all q points; it does not apply a partial or approximate unfold.

## 6. Lifetimes and scaling

The important live objects, in order, are:

| object | lifetime | distribution |
|---|---|---|
| wavefunction G-slab cache | complete zeta fit | host, bounded band slabs |
| \(\psi(\mathbf r_\mu)\) | complete zeta fit | centroid sharded |
| factorized \(C_q\) | all real-space chunks | both matrix axes sharded |
| G-flat accumulator | all real-space chunks | q wedge, \(\mu\) sharded |
| \(\zeta_q(\mu,r_c)\) | one real-space chunk | \(\mu\) and \(r_c\) sharded |
| \(V_q(\mu,\nu)\) | screening/self-energy stage | both matrix axes sharded |

The method stores \(\mathcal O(N_qN_\mu N_G)\) interpolation data and
\(\mathcal O(N_qN_\mu^2)\) Coulomb data, never
\(\mathcal O(N_qN_\mu N_r)\) zeta plus an additional replicated copy.
Chunk sizes change peak memory and launch count but not the equations.

Three implementation invariants follow:

1. no full zeta tensor exists between the real-space solve and G-flat
   accumulation;
2. no \(N_\mu^2\) object is gathered merely for I/O;
3. symmetry reduction precedes storage, but full-zone unfolding precedes any
   lattice FFT convolution that requires it.

The current owners are `gw.isdf_fitting` for \(C/Z\) and the solve,
`wfn_transforms` for G-flat accumulation, `gw.v_q_g_flat` for the Coulomb
contraction, `symmetry_maps` for orbit closure and unfolding, and
`file_io.SlabIO` for distributed bytes.
