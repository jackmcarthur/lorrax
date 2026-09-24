# Finite-occupation χ₀ on the two-face carrier

This page covers the distributed kernels that evaluate the finite-occupation
response from the two persistent ψ faces: the contour kernel and the
ordered-pair scan, both in `gw.w_isdf`. The physics, including the weight
factorization, the contour identity, the metal frequency plan and which
sample takes which kernel, is owned by
[Metallic MPA screening](../theory/metallic-mpa-screening.md). The static
finite-occupation χ₀ is `compute_chi0_matsubara` at ν = 0, not either kernel
here.

## Carrier

| array | shape | face layout | axis layout |
|---|---|---|---|
| `psi_mun` | (N_k^in, n_s, N_μ, N_b) | `P(None, None, 'x', 'y')` | `P(None, None, 'x', None)` |
| `psi_nmu` | (N_k^in, N_b, n_s, N_μ) | `P(None, 'x', None, 'y')` | `P(None, None, None, 'y')` |
| energies, occupations | (N_k, N_b) | replicated | replicated |

The face layout distributes the band axis over the mesh axis that does not
carry μ; the axis layout replicates bands. N_k^in is the full grid, or the
N_k^par symmetry parents when the bundle carries raw parents
(`wfns.green_parent`). In that case a typed unfold plan transports each Green
tile or ψ tile to full k on its own rank. No kernel here constructs a
band-replicated single-axis ψ on the face layout.

## Why two kernels

The Kubo weight (f_a − f_b)/(ε_a − ε_b + z) couples two band indices through
its denominator, so no choice of u_a and v_b factors it as u_a v_b, and no
GEMM over one band index produces it. Its time representation does factor:
f_a − f_b = f_a(1 − f_b) − (1 − f_a)f_b, and each term's occupation product
and phase e^{−i(ε_b − ε_a)t} separate into a per-a and a per-b factor. Hence:

- **Contour kernel.** Time nodes, one pair of Green's functions per node,
  O(N³) per node. Used wherever a time rule reaches the sample.
- **Ordered-pair scan.** Explicit band pairs, O(N⁴). Used only for the MPA
  metal near-origin sample: a time rule's node count grows as the bandwidth
  over Im z, which is prohibitive there.

## Contour kernel

`_get_chi_fractional_contour_kernel_face` builds, per node t_l, an
occupied-weighted and an empty-weighted Green's function with
`greens_function_kernel.build_G_tau` (band weights f̃ and ũ, phases
e^{±iε t}). It transforms both to R by flat-k FFT and forms the spin-traced
Hadamard product

$$
A_{\mathbf q}(t)=\sum_{ab}f_a(1-f_b)\,e^{-i(\varepsilon_b-\varepsilon_a)t}\,X_{ab}(\mathbf q),
$$

a k-convolution done in R. The retarded response is accumulated as

$$
\chi(\mathbf q,z)\approx-i\sum_l h_l\,e^{izt_l}\big[A_{\mathbf q}(t_l)-A_{\mathbf q}(t_l)^*\big],
\qquad \operatorname{Im}z>0,
$$

with positive times t_l and weights h_l from a damped-line rule, followed by
one FFT R → q.

**Band weights.** `_occupation_support_slices` returns the smallest
contiguous band ranges on which |f|, respectively |1 − f|, exceeds the floor
1 − `occupation_window_threshold`. MP overshoot is kept by magnitude, and a
partially occupied band belongs to both. The weights are f̃ = f·1_f and
ũ = (1 − f)·1_u, inverted before masking so that a band outside the empty
support carries exactly zero. On the face layout a support is applied as a
zero weight and the GEMM contracts all N_b bands, because the band axis is
distributed and cannot be sliced. The bank's axis-layout direct stream
instead prepares active band ranges and contracts only the supports.
`occupation_support_bandwidth` reads the same slices to size the damped-line
rule, so the rule and the bands it must resolve cannot disagree.

**Orientation.** With 𝓕_q[f](μ, ν) = Σ_R f(r_μ, r_ν + R) e^{iq·R}, the
default trace returns 𝓕_q[χᵀ]. `ordered=True` is set when time reversal is
measured broken. It builds G from the conjugated weight and time, gathers the
rows at −q, and returns 𝓕_q[χ], the orientation Σ's G_{k−q}W_q contraction
assumes. Under time reversal the two are equal and the default trace is kept.

**Modes in use.**

| `pair_mode` | caller | output |
|---|---|---|
| `retarded` | `compute_chi0_contour_fractional` (MPA metal samples on damped lines) | n_z arrays (N_k, N_μ, N_μ) at `P(None, 'x', 'y')` |
| `direct` | the shared-pole response bank | selected q rows at `P(None, None, 'x', 'y')`; the donated accumulator [sample, q, μ_X, μ_Y] is carried across sample groups |
| `laplace_ordered`, `kms_static` | the photon bank's static reference | one selected q row |

In `direct` mode each complex node t is one Green pair
A(t) = G_u(t) ⊙ G_f(t̄)^*, and it serves every member of a shared-node sample
group (rules from `minimax.response_group_rules`). The forward rows take A(t) at q; the reverse orientation at time t̄ is
conj(A(t)), read from the −q rows of the same transform. With current
vertices (`vertex=True`, photon carriers) the empty Green's function uses the
vertex-applied endpoints and the occupied one the bare endpoints, and the
spin pairs are scanned one at a time, so no full spin Green's function pair
is resident.

**Cost per node** (P ranks, N_μ centroids, n_s spinor components):

- two complex GEMMs, each 8 N_k^in (N_μn_s)² N_b / P flops (N_b becomes the
  support width with active ranges);
- two flat-k FFTs, O((N_μn_s)² N_k log N_k / P);
- the product and spin trace, O(N_k (N_μn_s)² / P).

The live set is two full-k Green tiles at `P(None, 'x', None, 'y', None)`,
2 N_k (N_μn_s)² complex numbers over P, plus the accumulator.

## Ordered-pair scan

`compute_chi0_direct_fractional` evaluates

$$
\chi_{\mu\nu}(\mathbf q,z)=\frac1{\sqrt{N_k}}\sum_{\mathbf k}\sum_{ab}
\frac{f_a(\mathbf k)-f_b(\mathbf k-\mathbf q)}{\varepsilon_a(\mathbf k)-\varepsilon_b(\mathbf k-\mathbf q)+z}\,
\rho_{ab}(r_\mu)\,\rho^*_{ab}(r_\nu),
\qquad \rho_{ab}=\psi_{a\mathbf k}\,\psi^*_{b,\mathbf k-\mathbf q},
$$

at nonzero z, for each q row of the caller's k − q maps. The host loops over
q rows; each row is one `shard_map` program with output (n_z, N_μ, N_μ) at
`P(None, 'x', 'y')`, stacked to (n_z, N_q, N_μ, N_μ).

**Schedule.** The band axis is cut into tiles of `_FRACTIONAL_PAIR_TILE` = 32
bands, N_t = ⌈N_b^log/32⌉ of them, and two nested `lax.scan`s run over tile
pairs:

- The outer scan reconstructs tile a in both orientations once per step.
  From `psi_mun`, a clamped `take`, masked by band ownership and followed by
  `psum('y')`, gives (N_k, n_s, N_μ/P_x, 32). From `psi_nmu`, the same with
  `psum('x')` and one local transpose of the 32-band tile gives
  (N_k, n_s, N_μ/P_y, 32).
- The inner scan reconstructs tile b the same way every step and rolls it to
  k − q.
- With raw parents each gathered tile is unfolded to full k on its rank, so
  the tile, not a face, is the largest full-k ψ object.

Per tile pair, the kernel forms the pair weights (n_z, N_k, 32, 32), masks
bands at or above the logical count, builds the two pair densities and
contracts them into the (n_z, μ_X, μ_Y) accumulator.

Energies and occupations are zero-padded to a whole number of tiles. A
phantom band's owner index matches no rank, so its gathered ψ is exactly
zero, and the logical-band mask removes its weight independently.

**Why a masked gather and not a GEMM or an all-gather.** A GEMM contracts one
shared index, and this weight is a function of the pair, so no GEMM computes
it. An all-gather of a face rebuilds the band-replicated single-axis
residency, 2S/P_x instead of 2S/P for ψ of size S. The masked gather plus
`psum` is a broadcast from the owner of one bounded tile. The outer tile is
reused across the whole inner sweep, so communication is 2N_t + 2N_t² tile
broadcasts per q row, each O(N_k n_s N_μ · 32/√P). The resident working set
is O(32) bands per operand.

**Cost per q row:** O(n_z N_k N_b² N_μ² / P) for the contraction, plus
O(N_k n_s N_b² N_μ / √P) for the densities. This is the O(N⁴) Kubo sum,
confined to the one near-origin sample.

## Refusals

| refusal | cause and fix |
|---|---|
| `GATE direct_fractional_needs_nonzero_z` | z = 0 in the pair scan; static χ₀ is `compute_chi0_matsubara` at ν = 0 |
| `GATE direct_fractional_ordered_rows` | an ordered call whose k − q row is not a permutation of the full grid |
| contour `Im(z) > 0` | the retarded contour needs upper-half-plane z; static is the Matsubara route |
| no band clearing the occupation window on one side | raise `occupation_window_threshold` toward 1, or check the occupation table |
| face carrier narrower than the energy table | load at least as many bands as the table names |
| `GATE response_vertex` | current vertices need ordered full-k endpoints, applied after the symmetry unfold |
| negative or nonfinite time nodes, or weights not shaped (n_z, n_t) | caller error |

## Gates

`tests/test_chi0_fractional_face_parity.py` compares both kernels with
independent NumPy band-pair sums at n_s = 1 and 2 on a 2 × 2 mesh. The
occupation table has an exact degeneracy and deep occupied and empty tails
(±40–50 Ry), so both supports are strictly narrower than the band range and
asymmetric; a wrong support weight fails it. The native four-rank run covers
the vendor FFT that CPU workers omit. `tests/multi_device/fractional_chi_gate.py`
is the P = 4 dense Kubo gate for the contour kernel.
