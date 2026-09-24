# Interpolation vectors \(\zeta_q(\mathbf G)\) and \(V_q\)

ISDF replaces every band-pair density with its values at \(N_\mu\) centroids
\(\mathbf r_\mu\) times interpolation vectors \(\zeta_{q\mu}\). This page
covers four things: the least-squares problem that defines \(\zeta\), the
identity that lets LORRAX fit \(\zeta\) directly in G space, the Coulomb
matrix built from \(\zeta\), and what each stage costs. The data movement is
on [route G](../architecture/zeta_fit_mubatch.md) and in
[raw-parent ζ fitting](../architecture/zeta_fit_face_psi_cct.md).

## 1. The least-squares fit

For momentum transfer \(q\), the pair products of left states \(m\) at
\(\mathbf k\) and right states \(n\) at \(\mathbf k+\mathbf q\) are
approximated as

$$
\psi^*_{m\mathbf k}(\mathbf r)\,\psi_{n\mathbf k+\mathbf q}(\mathbf r)
\approx\sum_\mu
\psi^*_{m\mathbf k}(\mathbf r_\mu)\,\psi_{n\mathbf k+\mathbf q}(\mathbf r_\mu)\,
\zeta_{q\mu}(\mathbf r).
$$

Minimizing the squared residual over every \(\mathbf k\), every \(m\) in the
left window \(L\) and every \(n\) in the right window \(R\) gives the normal
equations \(C_q\zeta_q=Z_q\). The band sums factor through one projector per
window,

$$
D^X_{\mathbf k,ab}(\mu,\mathbf r)=\sum_{n\in X}
\psi_{n\mathbf k a}(\mathbf r_\mu)\,\psi^*_{n\mathbf k b}(\mathbf r),
\qquad X=L,R,
$$

where \(a,b\) are open spinor indices. The right-hand side and the metric are
then

$$
Z_q(\mu,\mathbf r)=\sum_{\mathbf k}\sum_{ab}
D^L_{\mathbf k,ab}(\mu,\mathbf r)\,
\overline{D^R_{\mathbf k+\mathbf q,ab}(\mu,\mathbf r)},
\qquad
C_q(\mu,\nu)=Z_q(\mu,\mathbf r_\nu).
$$

The k sum is a cross-correlation on the periodic \(N_k\) grid, so the code
evaluates it by FFTs over \(\mathbf k\), at \(O(N_k\log N_k)\) per
\((\mu,\mathbf r)\) point instead of \(O(N_k^2)\). The projectors are built
only at the raw WFN k-points and transported to the full zone by the symmetry
action ([symmetry §3–4](symmetry.md)). \(C_q\) is the Gram matrix of the
training pair products sampled at the centroids, so it is Hermitian positive
semidefinite.

A current (bispinor) channel inserts its vertex \(\tilde\gamma^i\) on the
output spinor indices after transport. Its \(C_q\) is then a Hermitian
indefinite, signed Gram
([four-current wiring](../architecture/four_current_wiring.md)).

**Conjugation closure.** The charge windows are asymmetric: \(L\) holds every
occupied state plus the \(\Sigma\) conduction window, and \(R\) holds the
\(\Sigma\) occupied window plus every empty state. Complex conjugation swaps
the ordered endpoints, and relabelling \((m,n,\mathbf k)\) gives

$$
N_{RL}(q)=\overline{N_{LR}(-q)}
$$

for both \(C\) and \(Z\). The fit therefore solves the normal equations of
the conjugation-closed set,
\(C_q+\overline{C_{-q}}\) and \(Z_q+\overline{Z_{-q}}\). This completes the
training set; it is not a projection of \(\zeta\), \(V\) or \(W\).

## 2. Conditioning

\(C_q\) becomes nearly singular when \(N_\mu\) over-completes the rank of the
pair products, and then an exact inverse amplifies round-off without bound.
The charge channel uses a rank-truncated pseudo-inverse instead. From
\(C_q=V\Lambda V^\dagger\) it keeps \(\lambda>\epsilon_\zeta\lambda_{\max}\)
(`zeta_rcond`, default \(10^{-8}\)) and sets

$$
B=V_{\mathrm{keep}}\Lambda_{\mathrm{keep}}^{-1/2},\qquad
C_q^{+}=BB^\dagger,\qquad
\kappa_{\mathrm{eff}}\le 1/\epsilon_\zeta .
$$

The cut is never placed inside a degenerate multiplet: it moves down to
drop the whole block. \(C_q\) commutes with the point group when the centroid
set is orbit-closed, so a cut between blocks keeps the retained span
invariant, and \(C_{Sq}=\Pi C_q\Pi^\dagger\) survives truncation. A cut
through a block would break the k-star identity of \(W\) and \(\Sigma\). The
criterion and its certification are in the
[rank-truncation policy](../dev/rank_truncation_policy.md).

## 3. Fit \(Z\) in G space, apply \(C^+\) afterwards

The Coulomb contraction needs \(\zeta\) only on the sphere
\(|\mathbf q+\mathbf G|^2\le E_\zeta\) (`zeta_cutoff`), expressed through the
cell-periodic vector

$$
\tilde\zeta_{q\mu}(\mathbf G)=\mathcal F_{\mathbf r\to\mathbf G}
\!\left[e^{-i\mathbf q\cdot\mathbf r}\zeta_{q\mu}(\mathbf r)\right].
$$

\(C_q^+\) acts on the centroid index and \(\mathcal F\) acts on
\(\mathbf r\), so the two commute:

$$
\tilde\zeta_q(\mu,\mathbf G)=\sum_\nu C^+_q(\mu,\nu)\,\tilde Z_q(\nu,\mathbf G),
\qquad
\tilde Z_q(\mu,\mathbf G)=\mathcal F\!\left[e^{-i\mathbf q\cdot\mathbf r}
Z_q(\mu,\mathbf r)\right]_{\mathbf G\in\mathrm{sphere}(q)}.
$$

Three consequences shape the implementation:

- **Batches need no factor.** \(\tilde Z_q(\mu,\cdot)\) of one centroid needs
  nothing from any other centroid, so the fit runs in independent μ batches,
  and each owner forms whole rows.
- **The pair GEMM runs in G space.** The projector's band contraction runs
  against the stored plane-wave coefficients \(c_{n\mathbf k}(\mathbf G)\).
  Only the k-correlation needs real space, and it runs on grid planes, one
  batch at a time.
- **ζ(r) never exists.** What the fit stores is
  \(\tilde Z(q,\mu,\mathbf G_{\mathrm{sphere}})\), of size
  \(N_qN_\mu N_G\), never \(N_qN_\mu N_r\). \(C^+\) is applied afterwards,
  one G tile at a time. Entries outside each q's sphere are exactly zero.

## 4. Coulomb matrix

For the charge channel,

$$
V_{q,\mu\nu}=\sum_{\mathbf G\in\mathrm{sphere}(q)}
\overline{\tilde\zeta_{q\mu}(\mathbf G)}\,v(\mathbf q+\mathbf G)\,
\tilde\zeta_{q\nu}(\mathbf G).
$$

V is accumulated over G tiles as each tile of \(\tilde\zeta\) is formed. The
equal expression \(\overline{C^+}\,M_q\,\overline{C^+}\), with
\(M_q=\overline{\tilde Z}\,\mathrm{diag}(v)\,\tilde Z^{T}\), is not used,
because its rounding error grows as \(\epsilon\,\kappa(C)^2\) instead of
\(\epsilon\,\kappa(C)\). The [`vcoul`](../services/vcoul.md) service supplies
\(v(\mathbf q+\mathbf G)\), including dimensional truncation and the
long-wavelength slot. The four-current channels contract their own
\(\tilde\zeta^i\) with the channel tensor
([four-current wiring](../architecture/four_current_wiring.md)).

## 5. Irreducible q

When the centroid set is closed under the full space group with time
reversal, only the irreducible q wedge is fitted and stored. \(V_q\) is then
unfolded to the full zone by the centroid permutation and lattice-wrap phase
([symmetry §4–5](symmetry.md)). Closure is a correctness gate, never an
approximation. A nonclosed set runs with identity symmetry, with every k a
parent and every q stored
([unreduced admission](../architecture/zeta_fit_face_psi_cct.md#unreduced-admission-for-nonclosed-centroid-sets)).
Symmetry reduction precedes storage, while full-zone unfolding precedes any
lattice convolution that needs it.

## 6. Cost

Symbols: \(n_p\) raw parents, \(N_k\) full-zone k, \(Q\) stored q,
\(n_b\) fit bands, \(n_s\) spinor components, \(N_{G\psi}\) and \(N_G\) the
ψ and ζ spheres, \(N_r\) grid points, \(P\) ranks.

| stage | arithmetic per rank | memory per rank |
|---|---|---|
| \(C_q\) | \(2n_pn_s^2N_\mu^2n_b/P\) GEMM, plus \(O(N_\mu^2n_s^2N_k\log N_k/P)\) correlation | \(N_kN_\mu^2\cdot16/P\) before the IBZ slice |
| factor | \(O(QN_\mu^3)\), divided by \(\min(P,Q)\) when q-parallel | a replicated q batch of at most 4 GiB, then \(\lceil Q/P\rceil N_\mu^2\cdot16\) or \(QN_\mu^2\cdot16\) |
| \(\tilde Z_q(\mathbf G)\) | \(2n_pn_s^2N_\mu n_bN_{G\psi}/P\) GEMM, plus \(O(N_\mu N_kn_s^2N_r(\log N_r+\log N_k)/P)\) transforms | the batch working set ([route G](../architecture/zeta_fit_mubatch.md#memory-per-rank-and-the-planner)) |
| \(\tilde\zeta\) and \(V_q\) | \(O(QN_\mu^2N_G/P)\) | \(\tilde Z\) store \(QN_\mu N_G\cdot16/P\) on host or disk; \(V\) \(\lceil Q/P\rceil N_\mu^2\cdot16\) (partial sums \(QN_\mu^2\cdot16\) on the replicated tier) |

At a fixed k grid, \(N_\mu\), \(n_b\) and \(N_G\) all grow linearly with
system size, so the fit is cubic. The all-to-all moves
\(2n_pn_s^2N_\mu N_{G\psi}\cdot16/P\) bytes per rank over the whole fit. No
object of size \(N_\mu^2\) is gathered for I/O.

Code owners:

- `gw.isdf_fitting`: the driver;
- `isdf.core`: \(C_q\), the factor tiers and the k-correlation seam;
- `isdf.cplus`: conditioning;
- `isdf.zeta_mubatch`: \(\tilde Z\), the store and \(V\);
- `gw.v_q_g_flat`: the V consumers and the unfold;
- `symmetry_maps`: closure and transport;
- `file_io.SlabIO`: distributed bytes.
