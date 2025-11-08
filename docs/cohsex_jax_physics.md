# COHSEX in the JAX ISDF Pipeline

This note expands on `docs/formalism.md` with the full set of working equations
implemented in `src/isdf/gw_isdf/cohsex_jax.py` and `src/isdf/gw_isdf/w_isdf.py`.
It is organized to match the major stages of the driver: ISDF interpolation,
Coulomb metric construction, CTSP screening, self-energy evaluation, and the
(unfinished) self-consistency loop.  Throughout we emphasize why each
transformation is used in the JAX port and the numerical bottlenecks it
introduces.

## 1. Notation and wavefunctions

- Bloch spinors: \(\psi_{n\mathbf{k}}(\mathbf{r}, s)\) with band index \(n\),
  k-point \(\mathbf{k}\), and spinor component \(s\).
- FFT grid vectors \(\mathbf{G}\) relate to real-space values via orthonormal
  FFTs, so \(\psi_{n\mathbf{k}}(\mathbf{r})\) and \(c_{n\mathbf{k}}(\mathbf{G})\)
  are connected with no additional prefactors.
- Interpolation points \(\{\mathbf{r}_\mu\}\) ("centroids") are selected from
  the real-space grid using charge-density–weighted k-means; the code stores the
  wavefunctions sampled at these points as arrays
  `psi_*_rmu[k, n, s, μ]`.【F:src/isdf/gw_isdf/cohsex_jax.py†L572-L605】【F:src/isdf/gw_isdf/cohsex_jax.py†L640-L669】

The driver keeps both real-space values on the full FFT grid `psi_*_rtot` and the
centroid-restricted values `psi_*_rmu` because the least-squares reconstruction
requires access to both bases; this duplication is the first major memory
pressure point (see Section 7).

## 2. Interpolative separable density fitting (ISDF)

For a transferred crystal momentum \(\mathbf{q}\) we approximate the product of
Bloch orbitals by expanding onto the centroid basis:
\[
\rho_{mn\mathbf{k}}^{\mathbf{q}}(\mathbf{r}) =
\psi^*_{m\,\mathbf{k}-\mathbf{q}}(\mathbf{r})\psi_{n\mathbf{k}}(\mathbf{r})
\approx \sum_\mu \zeta_{\mathbf{q},\mu}(\mathbf{r})
    P_{mn\mathbf{k}}^{\mu}(\mathbf{q}),
\]
where the coefficients are the band overlaps sampled at centroids,
\(P_{mn\mathbf{k}}^{\mu}(\mathbf{q}) =
\psi^*_{m\,\mathbf{k}-\mathbf{q}}(\mathbf{r}_\mu)
\psi_{n\mathbf{k}}(\mathbf{r}_\mu)\).  The unknown interpolation vectors
\(\zeta_{\mathbf{q},\mu}(\mathbf{r})\) satisfy a linear system derived by
minimizing the residual in the full-grid metric.  Defining the centroid matrices
\[
C_{\mathbf{q},\mu\nu} = \sum_{\mathbf{k}} P^{\mu}_{mn\mathbf{k}}(
        \mathbf{q})^{*} P^{\nu}_{mn\mathbf{k}}(\mathbf{q}),\qquad
Z_{\mathbf{q},\mu}(\mathbf{r}) = \sum_{\mathbf{k}} P^{\mu}_{mn\mathbf{k}}(
        \mathbf{q})^{*}\psi_{n\mathbf{k}}(\mathbf{r}),
\]
the code forms the accumulators
`CCT` and `ZCT` (for "centroid-centroid" and "centroid-total"), and solves
\(C_{\mathbf{q}}\,\zeta_{\mathbf{q}} = Z_{\mathbf{q}}\) using a Cholesky
factorization with a small diagonal regularizer.【F:src/isdf/gw_isdf/cohsex_jax.py†L640-L707】  The
solution is stored as an \(n_{\mu} \times n_{\text{FFT}}\) matrix; reshaping to
\(\zeta_{\mathbf{q},\mu}(\mathbf{r})\) gives the real-space interpolation
vectors used everywhere downstream.

The overlap metric of the ISDF basis is the Hermitian matrix
\(S_{\mathbf{q},\mu\nu} = \langle \zeta_{\mathbf{q},\mu} | \zeta_{\mathbf{q},\nu} \rangle\),
implemented as \(S = \zeta^\dagger \zeta\).【F:src/isdf/gw_isdf/cohsex_jax.py†L708-L733】  The code caches
`S_qmunu` because it whitens the basis before inverting the dielectric matrix.

## 3. Coulomb matrix elements in the ISDF basis

For each \(\mathbf{q}\) the real-space vectors are phase shifted by the wrapped
momentum and FFT-transformed.  The bare Coulomb interaction is then projected as
\[
V_{\mathbf{q},\mu\nu} =
\sum_{\mathbf{G}} \zeta_{\mathbf{q},\mu}(\mathbf{G})^{*}
    v_{\mathbf{q}}(\mathbf{G})\, \zeta_{\mathbf{q},\nu}(\mathbf{G}),
\]
where \(v_{\mathbf{q}}(\mathbf{G})\) is the symmetrized kernel from
`compute_V_qfullG_for_q`.【F:src/isdf/gw_isdf/cohsex_jax.py†L708-L756】  The implementation multiplies the
FFT coefficients by \(\sqrt{v_{\mathbf{q}}}\) so the contraction reduces to a
Hermitian outer product, which avoids explicit dense matrix multiplications.

## 4. CTSP polarization (\(\chi^0\))

The complex-time shredded propagator (CTSP) method evaluates the static
polarizability via a quadrature over complex time slices instead of the direct
frequency summation.  For each energy window pair \((v, c)\) produced by
`get_window_info`, the routine `get_chi_lm_Yt_jax` builds the valence and
conduction Green’s functions on the centroid grid:
\[
G^{v}_{\mu\nu}(\tau_i) =
\sum_{n\mathbf{k} \in v} e^{-z_{\ell m} \tau_i (E_v^{\max} - \varepsilon_{n\mathbf{k}})}
    \psi_{n\mathbf{k}}(\mathbf{r}_\mu)\psi_{n\mathbf{k}}^{*}(\mathbf{r}_\nu),
\]
\[
G^{c}_{\mu\nu}(\tau_i) =
\sum_{m\mathbf{k} \in c} e^{-z_{\ell m} \tau_i (
        \varepsilon_{m\mathbf{k}} - E_c^{\min})}
    \psi_{m\mathbf{k}}(\mathbf{r}_\mu)\psi_{m\mathbf{k}}^{*}(\mathbf{r}_\nu),
\]
using the energy-window masks and exponential factors visible in the code.  The
polarization at each \(\tau_i\) is the spin trace of their convolution in real
space, and the final screened polarizability follows the CTSP quadrature rule:
\[
\chi^{0}_{\mathbf{q},\mu\nu} =
-\frac{2}{\sqrt{N_k}\, n_{\text{spin}} n_{\text{spinor}}}
    \sum_i \Big[2 z_{\ell m} w_i
    e^{-(z_{\ell m}(E_c^{\min}-E_v^{\max})-1)\tau_i}\Big]
    \tilde{G}^{c}_{\mu\nu}(\tau_i;\mathbf{q})
    \tilde{G}^{v}_{\nu\mu}(\tau_i;\mathbf{q}),
\]
where the prefactor matches the normalization enforced inside
`get_static_w_q_jax` and the weight in brackets is exactly `quad_w`.【F:src/isdf/gw_isdf/w_isdf.py†L16-L132】  The
FFT back-and-forth in `k_to_R` implements the \(\mathbf{k}\)-space convolution
required by the CTSP derivation.

## 5. Static screening and dielectric inversion

The screened interaction in the ISDF basis solves
\[(\mathbb{1} - V\chi^{0}) W = V\] (or its whitened variant
\((\mathbb{1} - \bar{V}\bar{\chi}^{0}) \bar{W} = \bar{V}\) when the overlap
metric \(S\) is supplied).  The routine `get_static_w_q_jax` performs this in
three steps:
1. If \(S\) is available it computes a Cholesky factor \(S = R^{\dagger}R\) and
   forms \(\bar{V} = R^{-\dagger} V R^{-1}\) and
   \(\bar{\chi}^{0} = R^{-\dagger} \chi^{0} R^{-1}\) by successive triangular
   solves.【F:src/isdf/gw_isdf/w_isdf.py†L240-L299】
2. For each \(\mathbf{q}\) it applies LU factorization to
   \(\mathbb{1} - \bar{V}\bar{\chi}^{0}\) and solves for \(\bar{W}\).【F:src/isdf/gw_isdf/w_isdf.py†L299-L310】
3. It restores \(W = R^{\dagger} \bar{W} R\) if whitening was used and reshapes
   back to \(W_{\mathbf{q},\mu\nu}\), enforcing the 2D sharding layout adopted
   throughout the pipeline.【F:src/isdf/gw_isdf/w_isdf.py†L310-L326】

The prefactor passed into the JIT, \(2/(\sqrt{N_k} n_{\text{spin}} n_{\text{spinor}})\),
encodes the CTSP normalization so that \(W\) can be plugged directly into the
self-energy kernels.【F:src/isdf/gw_isdf/w_isdf.py†L327-L337】  Inverting
\(\mathbb{1} - V\chi^{0}\) is the most communication-intensive linear-algebra
stage because each \(\mathbf{q}\)-block couples the entire \(\mu\)-space.

## 6. Static COHSEX self-energy

With \(G\) and \(V\) (or \(W\)) expressed in the centroid basis, the exchange
self-energy for a single k-point is
\[
\Sigma^{\text{SX}}_{\mu\nu}(\mathbf{k}) =
-\frac{1}{\sqrt{N_k}} \sum_{\mathbf{R}}
    e^{i\mathbf{k}\cdot\mathbf{R}}
    G_{\mu\nu}(\mathbf{R}) V_{\mu\nu}(\mathbf{R}),
\]
implemented literally as a pointwise product in real space followed by an FFT
back to \(\mathbf{k}\).【F:src/isdf/gw_isdf/cohsex_jax.py†L770-L804】  Projection to the band basis is the
centroid overlap
\(\Sigma_{ij}(\mathbf{k}) = \sum_{\mu\nu}\psi^{*}_{i\mathbf{k}}(\mathbf{r}_\mu)
    \Sigma^{\text{SX}}_{\mu\nu}(\mathbf{k})\psi_{j\mathbf{k}}(\mathbf{r}_\nu)\),
performed in two einsums by `get_sigma_x_kij_jax`.【F:src/isdf/gw_isdf/cohsex_jax.py†L804-L820】

The Hartree correction is accumulated from the \(\mathbf{q}=0\) Coulomb block by
contracting \(V_{0}\) with the centroid density \(\rho_{\mu} =
\sum_{n\mathbf{k}} |\psi_{n\mathbf{k}}(\mathbf{r}_\mu)|^{2}/N_k\).  The code
explicitly forms \(V_{0}\rho\) and projects it with the same centroids used for
\(\Sigma^{\text{SX}}\).【F:src/isdf/gw_isdf/cohsex_jax.py†L820-L834】  A dynamical Coulomb-hole contribution will
reuse the same machinery once the frequency dependence of \(W\) is reinstated.

## 7. Fixed-point self-consistency (prototype)

The current driver contains a fixed-point prototype that would iterate the total
self-energy until convergence.  It defines
\(H_{\text{QP}} = H_{\text{kin+ion}} + \Sigma\) and repeatedly re-evaluates the
exchange/Hartree pipeline given an updated \(\Sigma\).  The iteration is wrapped
in `crop_family_fixed_history_map`, which provides Anderson-like mixing and a
residual history for monitoring.  At present the loop only updates the static
exchange block and has not been validated for full self-consistency; the entry
point toggled by `self_consistent` simply calls the prototype before writing the
final matrices.【F:src/isdf/gw_isdf/cohsex_jax.py†L1230-L1372】  Extending this to a true
GW cycle would require reloading wavefunctions with updated occupancies and
adding the dynamical \(\Sigma^{\text{COH}}\) contribution.

## 8. Memory and parallel-scaling considerations

1. **Centroid and full-grid wavefunctions.**  The tensors
   `psi_*_rmu[k, n, s, μ]` and `psi_*_rtot[k, n, s, r]` dominate memory usage,
   particularly because both left and right band windows must be resident to
   assemble `CCT` and `ZCT` for each \(\mathbf{q}\).  Each scan over k-pairs in
   `compute_CCT_ZCT_for_q` touches \(\mathcal{O}(n_\mu^2)\) slices, so the arrays
   are pinned on device to avoid host shuttling.【F:src/isdf/gw_isdf/cohsex_jax.py†L640-L707】
2. **Interpolation vectors.**  `zeta_q` is an \(n_\mu \times n_{\text{FFT}}\)
   matrix per \(\mathbf{q}\); even though only one \(\mathbf{q}\) is stored at a
   time, the reshape to the FFT grid requires allocating a contiguous buffer the
   size of the entire real-space mesh.  This staging, plus the subsequent FFT and
   Coulomb contraction, make `zeta_q` a significant temporary.
3. **Dielectric inversion.**  The matrices \(\mathbb{1} - V\chi^{0}\) are dense in
   \(\mu\), so the LU factorization inside `get_static_w_q_jax` forces global
   communication when sharded over a 2D device mesh.【F:src/isdf/gw_isdf/w_isdf.py†L299-L310】
   The optional whitening path adds two triangular solves per \(\mathbf{q}\), each
   of which is difficult to shard efficiently because the solve operates on
   replicated \(n_\mu \times n_\mu\) blocks.
4. **Cholesky solve for \(\zeta\).**  Solving
   \(C_{\mathbf{q}}\,\zeta_{\mathbf{q}} = Z_{\mathbf{q}}\) requires the Cholesky
   factor to be broadcast across the sharding mesh; `jax.scipy.linalg.cho_solve`
   therefore becomes a synchronization bottleneck when \(n_\mu\) exceeds the per-
   device memory budget.【F:src/isdf/gw_isdf/cohsex_jax.py†L688-L733】
5. **Future \(\Sigma\) updates.**  Once \(W\) is frequency dependent, accumulating
   \(\Sigma^{\text{COH}}\) will involve large tensor products over \(\mu\),
   \(\nu\), and frequency slices.  These contractions will demand either further
   tiling (e.g., block-wise \(\mathbf{q}\) streaming) or distributed linear solves
   beyond the single-process `lu_factor` currently used.

The net effect is that the q-loop scales well with device count up to the point
where \(C_{\mathbf{q}}\) and the dielectric solves no longer fit comfortably on
individual devices.  Distributing the centroid dimension would mitigate this but
requires custom collective patterns for both the Cholesky and LU solves.

## 9. Outstanding challenges

- **Streaming and checkpointing.**  The TODO list calls for streaming q-blocks of
  \(V\) and \(\Sigma\) to restart files so that the on-device footprint remains
  bounded.【F:src/isdf/gw_isdf/TODO.md†L1-L15】  This is essential for scaling to large
  meshes where `V_qmunu` cannot be replicated.
- **True multi-host sharding.**  Coulomb metadata is still replicated across the
  mesh; distributing it with `device_put_sharded` is required for multi-rack
  runs.【F:src/isdf/gw_isdf/TODO.md†L1-L4】
- **(1 − Vχ) inversion.**  Achieving scalable LU or iterative inverses over
  \(\mu\) will likely need domain decomposition or custom solver support, since
  the dense solves dominate both flop count and memory traffic.
- **Self-consistency.**  The present prototype does not update wavefunctions or
  \(W\); completing a GW0 or fully self-consistent loop would reintroduce the
  CTSP construction inside each iteration, so checkpointing intermediate
  quantities (\(\zeta\), \(W\), \(\Sigma\)) becomes mandatory.
