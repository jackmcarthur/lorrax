# Interpolative Separable Density Fitting: Theory and Implementation

**Consolidates**: `isdf_context.md`, `isdf_spin_galerkin_derivation.md`, `ZETA_FITTING_ALGORITHM.md`, `cohsex_jax_physics.md`. The detailed ζ + V_q algorithm/sharding doc that previously occupied §11 has been split out to [`ZETA_V_Q_ALGORITHMS.md`](isdf-zeta-vq.md); the symmetry conventions and unfold procedures (IBZ tables, `mtrx`/τ actions, `unfold_psi`, `unfold_v_q`, `compute_centroid_sym_perm`) now live in [`SYMMETRY_COMPREHENSIVE.md`](symmetry.md); the memory model (per-stage formulas, G-flat planner, AOT chooser) is in [`MEMORY_MODEL.md`](../architecture/memory-model.md).

**Status (2026-05-15)**: describes the current implementation in `src/common/isdf_fitting.py` (zeta pipeline), `src/gw/gw_jax.py` (driver), `src/gw/w_isdf.py` (χ₀ + W), `src/gw/ppm_sigma.py` (GN-PPM Σ^c(ω)), `src/gw/head_correction.py` (q→0 head), and `src/gw/greens_function_kernel.py` + `src/gw/wavefunction_bundle.py` (leaf kernels). All physics arrays use the flat-k / flat-q convention (`(nk_tot, …)`); 3-D k-grid layout only appears inside `common/fft_helpers.py`. The detailed ζ + V_q algorithms, sharding map, and IBZ cascade are in [`ZETA_V_Q_ALGORITHMS.md`](isdf-zeta-vq.md); §11 below is now a short pointer with the bispinor / G-flat quick-reference table preserved as an at-a-glance summary.

---

## Overview

ISDF reduces GW computational cost by approximating pair-product densities with separable interpolation vectors. For systems with $n_k$ k-points, $n_b$ bands, and $n_r$ real-space grid points, storing the full pair-product tensor $M_{mn}(\mathbf{k}, \mathbf{q}, \mathbf{r})$ requires $O(n_k \times n_b^2 \times n_r)$ ~ TB of memory. ISDF reduces this to $O(n_\mu \times n_r)$ where $n_\mu \approx 10 \times n_b$ interpolation points, making real-space GW tractable for small systems.

---

> **Note (2026-05-15).** §3–5 below describe the historical **r-space
> ζ-on-disk** path and the **spin-traced rank-3** pair-density form.
> The current driver runs a **single rank-5 open-spin pair-density**
> path (charge μ_L=0 via identity γ̃ short-circuit, transverse μ_L≠0 via
> γ̃·γ̃ post-IFFT reduction) and the ζ-on-disk image lives in **G-flat**
> layout on the per-q `(q+G)` sphere with **IBZ-only** q-axis (factor
> ~`ntran` disk reduction). Per-r-chunk FFT-and-accumulate replaces the
> per-r-chunk on-disk write. The detailed math, shardings, communication
> and code references for the current pipeline now live in
> [`ZETA_V_Q_ALGORITHMS.md`](isdf-zeta-vq.md); the BGW symmetry
> convention and the IBZ → full-BZ unfolds (`unfold_psi`, `unfold_v_q`,
> centroid permutation tables) live in
> [`SYMMETRY_COMPREHENSIVE.md`](symmetry.md); per-stage
> memory formulas and the G-flat planner are in
> [`MEMORY_MODEL.md`](../architecture/memory-model.md). §3–5 here remain accurate for
> the conceptual / scalar / r-space narrative; §11 below is a pointer
> stub with a quick-reference table.

## 1. Wavefunctions and Notation

### 1.1 Bloch Spinors

Wavefunctions with spin-orbit coupling:

$$\psi_{n\mathbf{k},s}(\mathbf{r})$$

- $n$: band index
- $\mathbf{k}$: crystal momentum (Bloch vector)
- $s \in \{\uparrow, \downarrow\}$: spinor component
- $\mathbf{r}$: position in unit cell

Reciprocal-space representation via orthonormal FFT:

$$\psi_{n\mathbf{k},s}(\mathbf{r}) = \frac{1}{\sqrt{N_r}} \sum_{\mathbf{G}} c_{n\mathbf{k},s}(\mathbf{G}) \, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$$

where $N_r$ is the real-space FFT grid size and the $1/\sqrt{N_r}$ ensures orthonormality with `norm='ortho'` convention.

### 1.2 Interpolation Points (Centroids)

A set of $\{\mathbf{r}_\mu\}$ points ($\mu = 1, \ldots, n_\mu$) selected via **k-means clustering** weighted by the valence + conduction charge density:

$$\rho(\mathbf{r}) = \sum_{n \in \text{bands}} \sum_{\mathbf{k}} \sum_s |\psi_{n\mathbf{k},s}(\mathbf{r})|^2$$

**K-means algorithm**:
1. Initialize $n_\mu$ cluster centers randomly (k-means++ for better initialization)
2. Assign each grid point $\mathbf{r}_i$ to nearest center $\mu^*$, weighted by $\rho(\mathbf{r}_i)$
3. Update centers: $\mathbf{r}_\mu \leftarrow \sum_i w_i \mathbf{r}_i / \sum_i w_i$ where $w_i = \rho(\mathbf{r}_i) \cdot \delta_{\mu^*, \text{cluster}(i)}$
4. Repeat until convergence (typically 10-20 iterations)

This produces centroids concentrated in regions of **high electronic density** (bonds, lone pairs), which is optimal for interpolating pair products.

Typical ratio: $n_\mu \approx 10 \times n_{\text{bands}}$ for convergence.

**Files**:
- Generation: `src/centroid/kmeans_cli.py` (CLI) backed by `src/centroid/kmeans_isdf.py` (algorithm)
- Storage: `centroids_frac.h5` (fractional coordinates)
- Loading: `src/file_io/centroids.py`

---

## 2. ISDF Theory: Pair Products and Physical Charge Density

### 2.1 Band Pair-Product Tensor

For momentum transfer $\mathbf{q}$, define the **band pair-product**:

$$M_{mn,\mathbf{k}}^{ab}(\mathbf{q}; \mathbf{r}) = \psi^*_{m,\mathbf{k}-\mathbf{q},a}(\mathbf{r}) \, \psi_{n,\mathbf{k},b}(\mathbf{r})$$

where $a, b \in \{\uparrow, \downarrow\}$ are spinor indices. This tensor has size $O(n_m \times n_n \times n_k \times 4 \times n_r)$ where the factor 4 accounts for all spin combinations.

### 2.2 Physical Charge Density (Spin-Traced Product)

For GW calculations, we need the **physical charge density** formed by tracing over spins:

$$\rho_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r}) = \sum_{s} \psi^*_{m,\mathbf{k}-\mathbf{q},s}(\mathbf{r}) \, \psi_{n,\mathbf{k},s}(\mathbf{r}) = M^{\uparrow\uparrow} + M^{\downarrow\downarrow}$$

This is the spin-diagonal sum.

### 2.3 ISDF Factorization

Approximate the physical charge density using interpolation vectors $\zeta_{q,\mu}(\mathbf{r})$:

$$\rho_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r}) \approx \sum_\mu \zeta_{q,\mu}(\mathbf{r}) \cdot \rho_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r}_\mu)$$

Expanding the right-hand side:

$$\rho_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r}) \approx \sum_\mu \zeta_{q,\mu}(\mathbf{r}) \sum_s \psi^*_{m,\mathbf{k}-\mathbf{q},s}(\mathbf{r}_\mu) \, \psi_{n,\mathbf{k},s}(\mathbf{r}_\mu)$$

**Key properties**:
1. $\zeta_{q,\mu}(\mathbf{r})$ is **spin-independent** (universal for all spin channels)
2. $\zeta$ depends on $\mathbf{q}$ (different for each momentum transfer)
3. Storage: $O(n_q \times n_\mu \times n_r)$ instead of $O(n_k \times n_b^2 \times n_r)$

---

## 3. Galerkin System for Interpolation Vectors

### 3.1 Least-Squares Formulation

Find $\zeta_{q,\mu}(\mathbf{r})$ that minimizes the squared error summed over all band pairs and k-points:

$$\mathcal{E}[\zeta] = \sum_{m,n,\mathbf{k}} \int_{\Omega} \left| \rho_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r}) - \sum_\mu \zeta_{q,\mu}(\mathbf{r}) \, \rho_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r}_\mu) \right|^2 d\mathbf{r}$$

Setting the functional derivative to zero $\delta \mathcal{E} / \delta \zeta_\nu = 0$ yields the **Galerkin normal equations**:

$$\sum_\mu C_{q,\nu\mu} \, \zeta_{q,\mu}(\mathbf{r}) = Z_{q,\nu}(\mathbf{r})$$

### 3.2 Deriving CCT and ZCT Matrices

**CCT (Centroid-Centroid Term)**:

$$C_{q,\nu\mu} = \sum_{m,n,\mathbf{k}} \rho^*_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r}_\nu) \, \rho_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r}_\mu)$$

**ZCT (Centroid-Total Term)**:

$$Z_{q,\nu}(\mathbf{r}) = \sum_{m,n,\mathbf{k}} \rho^*_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r}_\nu) \, \rho_{mn,\mathbf{k}}(\mathbf{q}; \mathbf{r})$$

Substituting the spin-traced definition and factorizing band sums, both expressions reduce to products of **pair density matrices**:

$$P_{\mathbf{k},ss'}(\mathbf{r}_\nu, \mathbf{r}_\mu) = \sum_n \psi^*_{n\mathbf{k},s}(\mathbf{r}_\nu) \, \psi_{n\mathbf{k},s'}(\mathbf{r}_\mu)$$

This gives the **parallel structure**:

$$\boxed{C_{q,\nu\mu} = \sum_{\mathbf{k}} \sum_{s,s'} P^*_{\mathbf{k}-\mathbf{q},s's}(\mathbf{r}_\nu, \mathbf{r}_\mu) \, P_{\mathbf{k},ss'}(\mathbf{r}_\nu, \mathbf{r}_\mu)}$$

$$\boxed{Z_{q,\nu}(\mathbf{r}) = \sum_{\mathbf{k}} \sum_{s,s'} P^*_{\mathbf{k}-\mathbf{q},s's}(\mathbf{r}_\nu, \mathbf{r}) \, P_{\mathbf{k},ss'}(\mathbf{r}_\nu, \mathbf{r})}$$

**Key insight**: Even though we fit only the spin-diagonal sum $M^{\uparrow\uparrow} + M^{\downarrow\downarrow}$, the Galerkin error couples **all four spin channels** $P_{ss'}$ including off-diagonal $P_{\uparrow\downarrow}, P_{\downarrow\uparrow}$ due to the $\sum_{m,n} (\cdot)^* (\cdot)$ structure.

**Derivation**: See `docs/isdf_spin_galerkin_derivation.md` for detailed proof that this is the correct CCT formula (not $\sum_{ab} |P_{aa}|^2$ which would be a different fitting target).

### 3.3 Lattice Fourier Transform (k-Space Convolution)

To avoid explicitly storing the enormous pair-product tensors $P_{\mathbf{k}}(\mathbf{r}_\nu, \mathbf{r})$ or $P_{\mathbf{k}}(\mathbf{r}_\nu, \mathbf{r}_\mu)$, we use the **lattice Fourier transform** from k-space to R-space (crystal lattice vectors):

$$P_{\mathbf{R},ss'}(\mathbf{r}_\nu, \mathbf{r}_\mu) = \frac{1}{\sqrt{N_k}} \sum_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{R}} \, P_{\mathbf{k},ss'}(\mathbf{r}_\nu, \mathbf{r}_\mu)$$

Then the CCT and ZCT sums become **convolutions in R-space**:

$$C_{q,\nu\mu} = \sum_{\mathbf{R}} e^{i\mathbf{q}\cdot\mathbf{R}} \sum_{s,s'} P^*_{\mathbf{R},s's}(\mathbf{r}_\nu, \mathbf{r}_\mu) \, P_{\mathbf{R},ss'}(\mathbf{r}_\nu, \mathbf{r}_\mu)$$

$$Z_{q,\nu}(\mathbf{r}) = \sum_{\mathbf{R}} e^{i\mathbf{q}\cdot\mathbf{R}} \sum_{s,s'} P^*_{\mathbf{R},s's}(\mathbf{r}_\nu, \mathbf{r}) \, P_{\mathbf{R},ss'}(\mathbf{r}_\nu, \mathbf{r})$$

**Why FFT convolution?** Direct summation $\sum_{\mathbf{k}, \mathbf{k}'} (\cdots)$ scales as $O(n_k^2)$. Using FFTs to perform the convolution in R-space reduces this to $O(n_k \log n_k)$, making the procedure tractable for large k-grids.

**Implementation**: Compute $P_{\mathbf{k}}$ on k-grid, FFT to R-grid using `make_flat_k_ifftn` from `common/fft_helpers.py` (per-device local FFT on replicated k-axes via `custom_partitioning`), form spin-traced products, FFT back to q-space. See `compute_CCT_from_left_right()` and `compute_ZCT_from_left_right_zchunk()` in `common/isdf_fitting.py`. `norm='forward'` is used for the convolution identity: $C_q = \text{FFT}(\overline{\text{IFFT}(A)} \odot \text{IFFT}(B)) = \sum_{\mathbf k} A^*_k B_{k+q}$.

---

## 4. Complete Zeta Fitting and V_q Computation

### 4.1 Algorithm Overview (Memory-Efficient Path)

The full zeta-fitting procedure consists of five stages:

1. **Load centroids**: FFT wavefunctions from G-space to r-space, extract $\psi_{n\mathbf{k},s}(\mathbf{r}_\mu)$
2. **Build CCT matrices**: For each $\mathbf{q}$, compute $C_{q,\nu\mu}$ via k-space convolution
3. **Cholesky factorization**: $C_q = L_q L_q^\dagger$ (2D blocked for sharding)
4. **Build ZCT and solve**: For each z-chunk of real-space grid, compute $Z_q(\nu, \mathbf{r}_{\text{chunk}})$ and solve $L_q L_q^\dagger \zeta_q = Z_q$
5. **Compute Coulomb matrices**: Load $\zeta_{q,\mu}(\mathbf{r})$ from file, compute $V_{q,\mu\nu} = \langle z_{q,\mu} | v_q | z_{q,\nu} \rangle$ in G-space

Stages 1-4 produce `zeta_q.h5` with shape $(n_{qx}, n_{qy}, n_{qz}, n_\mu, n_r)$. Stage 5 produces `V_qmunu.h5` with shape $(n_q, n_\mu, n_\mu)$.

### 4.2 Stage 1: Centroid Extraction

Load G-space wavefunctions from `WFN.h5`, FFT to real-space, sample at interpolation points:

$$\psi_{n\mathbf{k},s}(\mathbf{r}_\mu) = \frac{1}{\sqrt{N_r}} \sum_{\mathbf{G}} c_{n\mathbf{k},s}(\mathbf{G}) \, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}_\mu}$$

**Memory issue**: Cannot hold $\psi_{n\mathbf{k},s}(\mathbf{r})$ for all bands simultaneously ($\sim$ 100s GB).

**Solution**: Band-chunked FFT loop (see §5.1).

**Output arrays** (persistent for remainder of calculation):
- $\psi^{(L)}_{\text{rmu}}(k, n_L, s, \mu_Y)$: left-window bands, centroid-sharded on Y
- $\psi^{(R)}_{\text{rmu}}(k, n_R, s, \mu_Y)$: right-window bands
- $\psi^{(L)}_{\text{rmuT}}(k, \mu_X, n_L, s)$: transposed for aligned matmul
- $\psi^{(R)}_{\text{rmuT}}(k, \mu_X, n_R, s)$: transposed

Here and below, subscripts $X, Y$ denote sharding on the 2D processor mesh with axes `('x', 'y')`.

### 4.3 Stage 2: CCT Matrix Construction

For each $\mathbf{q}$, build $C_q(\mu_X, \nu_Y)$ via lattice convolution:

**Step 2a**: Form pair density matrices on k-grid:

$$P^{(L)}_{\mathbf{k},ss'}(\mu, \nu) = \sum_{n \in \text{left}} \psi^{*(L)}_{n\mathbf{k},s}(\mathbf{r}_\mu) \, \psi^{(L)}_{n\mathbf{k},s'}(\mathbf{r}_\nu)$$

$$P^{(R)}_{\mathbf{k},ss'}(\mu, \nu) = \sum_{n \in \text{right}} \psi^{*(R)}_{n\mathbf{k},s}(\mathbf{r}_\mu) \, \psi^{(R)}_{n\mathbf{k},s'}(\mathbf{r}_\nu)$$

**Step 2b**: FFT to R-space (orthonormal):

$$P^{(L)}_{\mathbf{R}} = \text{IFFT}_{\mathbf{k}}\left[P^{(L)}_{\mathbf{k}}\right], \quad P^{(R)}_{\mathbf{R}} = \text{IFFT}_{\mathbf{k}}\left[P^{(R)}_{\mathbf{k}}\right]$$

**Step 2c**: Spin-trace and multiply:

$$C_{\mathbf{R}}(\mu, \nu) = \sum_{s,s'} \left(P^{(L)}_{\mathbf{R},s's}(\mu, \nu)\right)^* \cdot P^{(R)}_{\mathbf{R},ss'}(\mu, \nu)$$

This is the **Frobenius inner product** of the $2 \times 2$ spin matrix: $|\!|P^{(L)\dagger} P^{(R)}|\!|_F^2$.

**Step 2d**: FFT to q-space:

$$C_q(\mu, \nu) = \text{FFT}_{\mathbf{R}}\left[C_{\mathbf{R}}(\mu, \nu)\right]$$

**Sharding**: $C_q(\mu_X, \nu_Y)$ is 2D tiled for blocked Cholesky.

**Implementation**: `compute_CCT_from_left_right()` (spin-traced) in `common/isdf_fitting.py`.  Only the spin-traced path is currently wired to the GW driver; the explicit $P_{ab}$-channel variant lives behind `accumulate_pair_density_spin_traced` for the bispinor four-density work but is not yet exposed via a `cohsex.in` flag.

### 4.4 Stage 3: Cholesky Factorization

Compute $C_q = L_q L_q^\dagger$ for each $\mathbf{q}$ using 2D blocked algorithm:

$$L_q(\mu_X, \nu_Y) = \text{cholesky\_2d\_batched}(C_q)$$

**Algorithm**: Operates on tiles without gathering full matrix. Panel broadcasts and triangular updates use `lax.psum` for column communication.

**Complexity**:
- Compute: $O(n_\mu^3 / P)$
- Communication: $O(n_\mu^2 / \sqrt{P})$

**Sharding**: $L_q(\mu_X, \nu_Y)$ — 2D tiles `P(None, 'x', 'y', None, None)` internally; dense form `P(None, 'x', 'y')` on return.

**Implementation**: `compute_L_q_from_CCT()` in `common/isdf_fitting.py`, which calls `cholesky_2d_batched()` in `common/cholesky_2d.py`. On a 1×1 mesh (single GPU), falls back to dense `jnp.linalg.cholesky` with a small trace-proportional ridge to regularize the rank-deficient pair-density matrix.

### 4.5 Stage 4: ZCT and Triangular Solve (Z-Chunked)

**Outer loop**: Iterate over z-chunks of real-space grid (see §5.2 for chunking strategy).

For each z-chunk $\mathbf{r}_{\text{chunk}}$ (typically $x_{\text{chunk}} \times n_y \times n_z$ points):

**Step 4a**: Load this z-slice with band-chunked FFT:

$$\psi^{(L)}_{\text{chunk}}(k, n_L, s, r_Y), \quad \psi^{(R)}_{\text{chunk}}(k, n_R, s, r_Y)$$

**Step 4b**: Build ZCT for this chunk (parallel structure to CCT):

$$P^{(L)}_{\mathbf{k}}(\mu, r) = \sum_{n \in \text{left}} \psi^{*(L)}_{n\mathbf{k},s}(\mathbf{r}_\mu) \, \psi^{(L)}_{n\mathbf{k},s}(\mathbf{r})$$

$$P^{(L)}_{\mathbf{R}} = \text{IFFT}_{\mathbf{k}}[P^{(L)}_{\mathbf{k}}], \quad P^{(R)}_{\mathbf{R}} = \text{IFFT}_{\mathbf{k}}[P^{(R)}_{\mathbf{k}}]$$

$$Z_{\mathbf{R}}(\mu, r) = \sum_{s,s'} (P^{(L)}_{\mathbf{R},s's})^* \cdot P^{(R)}_{\mathbf{R},ss'}$$

$$Z_q(\mu_X, r_{XY}) = \text{FFT}_{\mathbf{R}}[Z_{\mathbf{R}}]$$

**Step 4c**: Triangular solve. `solve_zeta_from_L_q()` in `common/isdf_fitting.py` loops over q-batches of size `q_chunk_size` (default 1) and, inside a `shard_map` over `('x','y')` with the r-column axis scattered, runs a vmapped Cholesky back-solve:

```python
# shard_map in_specs = (P(None, None, None), P(None, None, ('x','y')))
# out_specs          =  P(None, None, ('x','y'))
def _sharded_cho_solve_batch(L_batch, Z_batch):
    def solve_single(L, Z):
        y = jax.scipy.linalg.solve_triangular(L, Z, lower=True)
        return jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
    return jax.vmap(solve_single)(L_batch, Z_batch)
```

L is gathered to replicated inside the shard_map, so each device solves its r-column shard. The Python-level outer loop with `donate_argnums` forces sequential GPU execution: `fori_loop` SPMD-replicates the sharded carry (88 GB OOM at Si 10³), and `scan(unroll=8)` pipelines adjacent iterations (18.9 GB preallocated temp).

**Step 4d**: Write to HDF5 via `SlabIO.write_slab` (phdf5 FFI when `use_ffi_io=true`, rank-0 allgather fallback otherwise). Layout is flat-q `(nq, n_rtot, n_rmu)` with the μ axis innermost so per-r-chunk writes are contiguous; per-q reads for V_q stay contiguous too.

**Sharding**:
- $Z_q(\mu_X, r_Y)$ built in ZCT: `P(None, 'x', 'y')`; resharded once to `P(None, None, ('x','y'))` for the solve
- $\zeta_q(\mu, r_{XY})$: `P(None, None, ('x','y'))` — r-column distributed

**Implementation**: `fit_zeta_chunked_to_h5()` in `common/isdf_fitting.py`.

### 4.6 Stage 5: Coulomb Matrix Elements

After all z-chunks are written, compute $V_{q,\mu\nu}$ from stored $\zeta_{q,\mu}(\mathbf{r})$:

**Step 5a**: Define cell-periodic part:

$$z_{q,\mu}(\mathbf{r}) = e^{-i\mathbf{q}\cdot\mathbf{r}} \, \zeta_{q,\mu}(\mathbf{r})$$

**Step 5b**: FFT to G-space:

$$z_{q,\mu}(\mathbf{G}) = \text{FFT}_{\mathbf{r}}[z_{q,\mu}(\mathbf{r})]$$

**Step 5c**: Coulomb kernel (BerkeleyGW conventions):

$$v_{\mathbf{q}}(\mathbf{G}) = \frac{4\pi}{|\mathbf{q}+\mathbf{G}|^2} \times T_{\text{cell}}(\mathbf{q}+\mathbf{G})$$

where $T_{\text{cell}}$ is the truncation factor (1 for 3D, analytical for 2D slab/wire, see `compute_vcoul.py`).

**Divergence at q=G=0**: Handled via Monte Carlo integration over Voronoi cell (see `compute_q0_averages` in `vcoul.py`).

**Step 5d**: Weighted FFT coefficients:

$$\tilde{z}_{q,\mu}(\mathbf{G}) = \sqrt{v_{\mathbf{q}}(\mathbf{G})} \cdot z_{q,\mu}(\mathbf{G})$$

**Step 5e**: Hermitian outer product:

$$V_{q,\mu\nu} = \sum_{\mathbf{G}} \tilde{z}^*_{q,\mu}(\mathbf{G}) \, \tilde{z}_{q,\nu}(\mathbf{G}) = \langle \tilde{z}_{q,\mu} | \tilde{z}_{q,\nu} \rangle$$

**Memory**: $\tilde{z}$ already lives on the per-q WFN.h5-style G-sphere on disk (G-flat layout). The orchestrator pre-reads all IBZ $\tilde{z}$ slabs in one batched call, then runs a sync per-q loop; the contract chunks over $\mathbf{G}$ (`g_chunk`), so the working set is bounded by one G-chunk plus the mesh-sharded $\zeta$ slabs — no FFT, no $\mu\times\nu$ tiling.

**Output**: `V_qmunu` array, shape $(n_{qx}, n_{qy}, n_{qz}, n_\mu, n_\mu)$, sharded `P(None, None, None, 'x', 'y')`. Used directly in memory by the GW pipeline; not routinely persisted to disk.

**Disk bottleneck**: The file `zeta_q.h5` has flat-q layout $(n_q, n_r, n_\mu)$ and is typically **10–100 GB**. Dataset layout puts $n_\mu$ innermost so per-r-chunk writes are contiguous (the earlier `(n_q, n_\mu, n_r)` layout was 8× slower on Perlmutter pscratch).

**Implementation**: `compute_all_V_q()` dispatcher in `gw/compute_vcoul.py` → `gw/v_q_g_flat.py:compute_all_V_q_g_flat` (charge) / `gw/v_q_bispinor.py:compute_V_q_bispinor_g_flat_to_h5` (bispinor); per-q $v(q+G)$ built by `compute_v_q_per_G`. Q=0 divergence handled via Voronoi Monte Carlo in `gw/vcoul.py:compute_q0_averages`. Box truncation (0-D molecules) via `gw/compute_vcoul_0d.py`.

---

## 5. Chunking Strategy (Memory-Constrained Algorithm)

The idealized algorithm in §4 assumes all arrays fit in memory. For production systems, we chunk in three dimensions:

### 5.1 Band Chunks (FFT Stage)

**Problem**: $\psi_{n\mathbf{k},s}(\mathbf{r})$ for all bands is $\sim$ 100s GB.

**Solution**: Loop over band chunks $B_b$:

```python
ψ_rmu = zeros(n_k, n_bands, n_s, n_μ)   # output fits in memory

for b_start in range(0, n_bands, B_b):
    ψ_G = load_from_HDF5(bands=slice(b_start, b_start+B_b))  # (n_k, B_b, n_s, n_G)
    ψ_r = IFFT(ψ_G)                                          # (n_k, B_b, n_s, n_r) - transient!
    ψ_rmu[b_start:b_start+B_b] = extract_centroids(ψ_r)     # sample at r_μ
    del ψ_r                                                   # free immediately
```

**Peak memory**:

$$M_{\text{FFT}} = M_{\text{full}} + M_{\text{phase}} + 2 \times 16 \times n_k \times \frac{B_b}{P} \times n_s \times n_r$$

where $M_{\text{full}}$ is persistent centroids, $M_{\text{phase}}$ is the $e^{i\mathbf{k}\cdot\mathbf{r}}$ array.

**Constraint**:

$$B_b \leq \frac{(M_{\text{budget}} - M_{\text{full}} - M_{\text{phase}}) \times P}{2 \times 16 \times n_k \times n_s \times n_r}$$

### 5.2 R-Chunks (ZCT Stage)

**Problem**: $Z_q(\mu, \mathbf{r})$ for full grid is $\sim$ 10-100 GB.

**Solution**: Process z-slices (contiguous chunks: $x_{\text{chunk}} \times n_y \times n_z$):

```python
for x_start in range(0, n_x, x_chunk):
    ψ_chunk = band_chunked_FFT(x_slice)     # already uses B_b internally
    P_k = pair_density(ψ_rmu, ψ_chunk)      # (n_k, n_μ, r_chunk)
    Z_q_chunk = k_to_R_convolution(P_k)     # (n_q, n_μ, r_chunk)
    solve_and_write(L_q, Z_q_chunk)         # per-q triangular solve
```

**Three stages must all fit**:

1. **Pair density**:

$$M_{\text{pair}} = M_{\text{base}} + 16 \times \frac{B_r}{p_y} \times [n_k n_b n_s + 2 n_k \tfrac{n_\mu}{p_x}] \leq M_{\text{budget}}$$

2. **ZCT pipeline**:

$$M_{\text{ZCT}} = M_{\text{base}} + 16 \times \frac{B_r}{p_y} \times [(2n_k + n_q) \tfrac{n_\mu}{p_x}] \leq M_{\text{budget}}$$

3. **Solve** (with $B_q = 1$):

$$M_{\text{solve}} = M_{\text{base}} + 2 \times 16 \times n_q n_\mu \frac{B_r}{P} + 16 \times n_\mu^2 \leq M_{\text{budget}}$$

where $M_{\text{base}} = M_{\text{cent}} + M_{L_q} + M_{\text{cache}}$ includes persistent arrays.

### 5.3 Q-Chunks (Solve Stage)

**Problem**: Triangular solve requires replicating $L_q$ (size $n_\mu^2$) on all devices.

**Solution**: Loop over q-chunks $B_q$:

```python
for q_start in range(0, n_q, B_q):
    L_rep = all_gather(L_q[q_start:q_start+B_q])  # replicate (n_μ, n_μ) × B_q
    ζ_chunk = L_rep^{-H}(L_rep^{-1} Z_chunk)      # column-parallel solve
    write_to_h5(ζ_chunk)
```

**Constraint**:

$$B_q \leq \frac{M_{\text{budget}} - M_{\text{base}} - 2 \times M_{Z_{\text{col}}}}{16 \times n_\mu^2}$$

**Automatic sizing**: Function `compute_optimal_chunks()` in `gw/gw_init.py` solves this constraint system analytically, iteratively reducing chunk sizes until all stages fit, using `common.gpu_utils.get_device_memory_info()` to probe the per-device budget.

**See**: `docs/architecture/memory-model.md` for detailed formulas and bottleneck arrays.

### 5.4 Key Sharding Techniques

**Staged resharding**: Avoid "involuntary full rematerialization" by resharding in two stages:

```python
# Bad: directly reshard from P(None, ('x','y'), ...) to P(None, None, ..., 'y')
# → XLA replicates full array before repartitioning

# Good: two-stage via intermediate
ψ = with_sharding_constraint(ψ, P(None, 'y', ...))    # gather over X only
ψ = with_sharding_constraint(ψ, P(None, None, ..., 'y'))  # then shard r on Y
```

**`shard_map` for FFT**: When transform axes are not sharded, run FFT independently on each device:

```python
@shard_map(mesh=mesh_xy,
           in_specs=P(None, ('x','y'), None, None, None, None),
           out_specs=P(None, ('x','y'), None, None, None, None))
def sharded_ifftn(x):
    return jnp.fft.ifftn(x, axes=(-3, -2, -1))  # local on each device
```

**2D blocked Cholesky**: Operates on tiles $C_q(\mu_X, \nu_Y)$ without gathering. Column broadcasts use `lax.psum` over mesh axis `'x'`.

---

## 6. Green's Functions and Self-Energy

### 6.1 Green's Function on ISDF Grid

The retarded Green's function with explicit spin indices:

$$G_{\mathbf{k},ab}(\mathbf{r}_\mu, \mathbf{r}_\nu; \omega) = \sum_n \frac{\psi_{n\mathbf{k},a}(\mathbf{r}_\mu) \, \psi^*_{n\mathbf{k},b}(\mathbf{r}_\nu)}{\omega - E_{n\mathbf{k}} + i\eta \, \text{sgn}(E_F - E_{n\mathbf{k}})}$$

where $a, b \in \{\uparrow, \downarrow\}$, and the sign of $i\eta$ depends on occupancy.

**Static limit** (COHSEX approximation, $\omega \to 0$):

$$G^{\text{occ}}_{\mathbf{k},ab}(\mu, \nu) = \sum_{n \in \text{occ}} \psi_{n\mathbf{k},a}(\mathbf{r}_\mu) \, \psi^*_{n\mathbf{k},b}(\mathbf{r}_\nu)$$

$$G^{\text{all}}_{\mathbf{k},ab}(\mu, \nu) = \sum_{n \in \text{all}} \psi_{n\mathbf{k},a}(\mathbf{r}_\mu) \, \psi^*_{n\mathbf{k},b}(\mathbf{r}_\nu)$$

These are computed directly on the centroid grid (no full r-space needed).

**Sharding**: $G_{\mathbf{k}}(s_a, \mu_X, s_b, \nu_Y)$ with $2 \times 2$ spin structure and 2D centroid tiling.

### 6.2 RPA Polarizability via CTSP

The independent-particle polarizability in the ISDF basis:

$$\chi^0_{q,\mu\nu}(\omega) = -\frac{1}{N_k} \int_0^\infty d\tau \, e^{i\omega\tau} \sum_{ab} G^{\text{occ}}_{\mathbf{R},ab}(\mu,\nu;\tau) \, G^{\text{empty}}_{\mathbf{R},ba}(\nu,\mu;-\tau)$$

where $G_{\mathbf{R}} = (1/\sqrt{N_k}) \sum_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{R}} G_{\mathbf{k}}$ is the lattice Fourier transform (convolution in R-space).

**Static limit** ($\omega = 0$):

$$\chi^0_{q,\mu\nu} = -\frac{1}{N_k} \int_0^\infty d\tau \sum_{ab} G^{\text{occ}}_{\mathbf{R},ab}(\mu,\nu;\tau) \, G^{\text{empty}}_{\mathbf{R},ba}(\nu,\mu;\tau)$$

The $\sum_{ab}$ performs the spin trace (contracts spin indices).

**Explicit R-space form** (what we actually compute):

$$\chi^0_{\mathbf{R},\mu\nu} = -\frac{1}{N_k} \int_0^\infty d\tau \sum_{ab} G^{\text{occ}}_{\mathbf{R},ab}(\mu,\nu;\tau) \, G^{\text{empty}}_{\mathbf{R},ba}(\nu,\mu;\tau)$$

Then Fourier transform to q-space:

$$\chi^0_{q,\mu\nu} = \sum_{\mathbf{R}} e^{i\mathbf{q}\cdot\mathbf{R}} \chi^0_{\mathbf{R},\mu\nu}$$

**Sharding**: $\chi^0_q(\mu_X, \nu_Y)$ same 2D tiling as $C_q$.

### 6.3 Energy Window Shredding and Quadrature

**Problem**: Direct summation $\sum_{n \in \text{occ}} \sum_{m \in \text{empty}} (\cdots)$ is $O(n_{\text{occ}} \times n_{\text{empty}})$ per k-point.

**Solution**: Divide bands into energy windows and evaluate integral via **Gauss-Laguerre quadrature**:

$$\int_0^\infty d\tau \, e^{-\gamma\tau \Delta E} (\cdots) \approx \sum_i w_i \, e^{-\gamma\tau_i \Delta E} (\cdots)$$

where $\gamma = 1/\sqrt{E_{\text{gap}} \times E_{\text{bw}}}$ is the energy scale, and $\{\tau_i, w_i\}$ are Gauss-Laguerre nodes/weights.

**Energy windows**: Partition valence into windows $\ell$ and conduction into windows $m$. For each pair $(\ell, m)$:

$$G^v_{\mathbf{k},ab}(\mu,\nu;\tau_i) = \sum_{n \in \ell} e^{-\gamma\tau_i(E_{\ell}^{\max} - E_{n\mathbf{k}})} \, \psi_{n\mathbf{k},a}(\mathbf{r}_\mu) \, \psi^*_{n\mathbf{k},b}(\mathbf{r}_\nu)$$

$$G^c_{\mathbf{k},ab}(\mu,\nu;\tau_i) = \sum_{m \in m} e^{-\gamma\tau_i(E_{m\mathbf{k}} - E_m^{\min})} \, \psi_{m\mathbf{k},a}(\mathbf{r}_\mu) \, \psi^*_{m\mathbf{k},b}(\mathbf{r}_\nu)$$

Then:

$$\chi^0_{\ell m} = -\frac{2\gamma}{\sqrt{N_k} \, n_{\text{spin}} n_{\text{spinor}}} \sum_i w_i e^{-(\gamma E_{\text{gap}} - 1)\tau_i} \sum_{ab} \left[\sum_{\mathbf{R}} e^{i\mathbf{q}\cdot\mathbf{R}} G^c_{\mathbf{R},ab}(\tau_i) \, G^v_{\mathbf{R},ba}(\tau_i)\right]$$

**Quadrature size**: $N_\tau = \alpha(0.4 - 0.3\ln\epsilon)$ where $\alpha = \sqrt{E_{\text{bw}}/E_{\text{gap}}}$ and $\epsilon$ is target error.

**Implementation**: `compute_chi0_minimax()` in `gw/w_isdf.py`. The cached `_get_chi_minimax_kernel()` (flat-k, compiled once per mesh × kgrid) builds $G^v(\tau)$ and $G^c(\tau)$ via `greens_function_kernel.build_G` with Laplace phases, IFFTs each to R-space using `make_flat_k_ifftn` (see §7.2 below for the swapped μ/ν ↔ 'x'/'y' assignment that keeps the output naturally sharded), contracts `einsum('Rambn,Rbnam->Rmn')`, accumulates `χ_R` over τ nodes in a donated fori-loop, and FFTs back to q. The outer `compute_chi0()` (`gw/w_isdf.py`) pulls the τ nodes and minimax weights from a `LaplaceMinimaxQuadrature` built by `common.minimax` (or loaded from `src/common/minimax_assets/` when `regenerate_minimax_tables = false`).

**Reference**: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020). Full derivation in `docs/theory/minimax-quadrature.md`. Windowing strategy in `docs/dev/notes/NEW_WINDOW_MINIMAX_GUIDELINES.md`.

### 6.4 Screened Interaction (Dyson Equation)

Solve for $W$ in the ISDF basis:

$$(1 - V\chi^0) W = V$$

**Algorithm** (direct solve, no whitening):
1. LU factorization: $(1 - V\chi^0) = L U$
2. Solve: $W = (LU)^{-1} V$

**Note**: The whitening step (orthogonalizing via overlap matrix $S = \langle \zeta_\mu | \zeta_\nu \rangle$) is **not used** in the current implementation. We solve the Dyson equation directly in the original ISDF basis.

**Sharding**: both V and χ₀ arrive as `P(None, 'x', 'y')` on a flat-q `(nq, μ, μ)` layout. `_get_w_solve_fn` pads to a multiple of `mesh.size`, reshapes the sharding via two `with_sharding_constraint` stages (replicate → `P(('x','y'), None, None)`, i.e. q-parallel with μ and ν replicated per-q), and runs a `shard_map` that loops per-q via `fori_loop` calling `jsp_linalg.lu_factor` and `lu_solve`. The result is resharded back to replicated for downstream FFT convolutions.

**Implementation**: `solve_w()` in `gw/w_isdf.py` (public) / `_get_w_solve_fn()` cached factory.

### 6.5 Head Correction for q=0, G=0 Divergence

The bare Coulomb $v_{\mathbf{q}}(\mathbf{G}) = 4\pi/|\mathbf{q}+\mathbf{G}|^2$ diverges as $\mathbf{q}, \mathbf{G} \to 0$. We handle this via:

**Step 1**: Build $V_{q,\mu\nu}$ with $v_q(G=0)$ **zeroed** (done in `make_v_munu_kernel`).

**Step 2**: Add back the cell-averaged **head** as a rank-1 correction:

$$V_{q=0,\mu\nu} \leftarrow V_{q=0,\mu\nu} + \frac{\bar{v}_0}{\Omega} \, \zeta^*_\mu(G=0) \, \zeta_\nu(G=0)$$

where $\bar{v}_0$ is the Voronoi-cell average of $4\pi/q^2$ (computed via Monte Carlo or analytical formula), $\Omega$ is the unit cell volume, and $\zeta_\mu(G=0)$ is the G=0 Fourier component of the interpolation vector.

**Dipole matrix elements**: For the screened $W$, we also correct using the **head** (later: and **wings**) based on the head of $\chi$ computed from dipole matrix elements $\mathbf{v}_{cv,\mathbf{k}} = \langle c\mathbf{k} | \mathbf{p} + i[\mathbf{r}, V_{\text{NL}}] | v\mathbf{k} \rangle$:

$$S_{\alpha\beta}(\omega) = \frac{4}{\Omega N_k n_{\text{spin}} n_{\text{spinor}}} \sum_{cv\mathbf{k}} \frac{f_v - f_c}{\Delta E_{cv} (\omega^2 - \Delta E_{cv}^2)} \, v^*_{\alpha,cv\mathbf{k}} \, v_{\beta,cv\mathbf{k}}$$

$$\chi_{00}(q,\omega) = \sum_{\alpha\beta} q_{\alpha} S_{\alpha\beta}(\omega) q_{\beta}$$

Then the head correction to $W$ is the cell-averaged $W_00(q)$

$$W_{00}(\omega) = \bar{v}_0 + \bar{v}_0 \, \chi_{00}(\omega) \, W_{00}(\omega)$$

Solving: $W_{00}(\omega) = \bar{v}_0 / (1 - \bar{v}_0 \chi_{00}(\omega))$.

The full $W$ at $q=0$ includes this head contribution added in the same way as to $V$.

**Implementation**: `gw/head_correction.py` centralizes the head path. `resolve_head_sample()` selects the source (`vhead`/`whead_0freq`/`whead_imfreq` overrides → `epshead` from `eps0mat.h5` → `s_tensor` from `dipole.h5`) and returns a scalar `HeadSample(v_c0, W_c0, source, ω)`. `compute_static_head_terms_from_sample()` builds **exact** band-diagonal shifts for Σ^X, Σ^SX, Σ^{SX−X}, and Σ^COH (all in Ry); `static_head_terms_to_kij()` broadcasts them to dense `(nk, nb, nb)` matrices that are added to the sigma matrices in `gw_jax.main` (`_add_head`). Dipole $S_{\alpha\beta}(\omega)$ is computed in `common/chi_from_dipole.py : compute_S_omega`. The GN-PPM path uses a separate scalar `HeadGNParams` (`fit_head_gn`) for dynamic head contributions.

### 6.6 Self-Energy Matrix Elements

**Exchange (SEX)**: Sum over occupied states only:

$$\Sigma^X_{\mathbf{R},ab}(\mu, \nu) = -G^{\text{occ}}_{\mathbf{R},ab}(\mu, \nu) \, W_{\mathbf{R}}(\mu, \nu)$$

where $W_{\mathbf{R}} = (1/\sqrt{N_k}) \sum_{\mathbf{q}} e^{i\mathbf{q}\cdot\mathbf{R}} W_{\mathbf{q}}$ and the product is **elementwise** in R-space. Note $W_{\mathbf{R}}$ has no spin indices (it's the spin-traced screened interaction).

Transform back to k-space:

$$\Sigma^X_{\mathbf{k},ab}(\mu, \nu) = \frac{1}{\sqrt{N_k}} \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} \, \Sigma^X_{\mathbf{R},ab}(\mu, \nu)$$

**Coulomb-hole (COH)**: Sum over all states:

$$\Sigma^{\text{COH}}_{\mathbf{R},ab}(\mu, \nu) = G^{\text{all}}_{\mathbf{R},ab}(\mu, \nu) \, \left(W_{\mathbf{R}}(\mu, \nu) - V_{\mathbf{R}}(\mu, \nu)\right)$$

where $V_{\mathbf{R}}$ is the lattice transform of bare Coulomb ($V_{\mathbf{R}} = \text{IFFT}_{\mathbf{q}}[V_{\mathbf{q}}]$, also spin-independent).

**Total static COHSEX**:

$$\Sigma^{\text{COHSEX}}_{\mathbf{k},ab}(\mu, \nu) = \Sigma^X_{\mathbf{k},ab}(\mu, \nu) + \Sigma^{\text{COH}}_{\mathbf{k},ab}(\mu, \nu)$$

**Sharding**: $\Sigma_{\mathbf{k}}(s_a, \mu_X, s_b, \nu_Y)$ with spin structure explicit.

### 6.7 Projection to Band Basis

Contract with wavefunctions to get band-diagonal elements:

$$\Sigma_{ij,\mathbf{k}} = \sum_{a,b,\mu,\nu} \psi^*_{i\mathbf{k},a}(\mathbf{r}_\mu) \, \Sigma_{\mathbf{k},ab}(\mu, \nu) \, \psi_{j\mathbf{k},b}(\mathbf{r}_\nu)$$

**Efficient implementation** (two einsums to avoid intermediate $O(n_b^2 \times n_s^2 \times n_\mu^2)$):

```python
# ψ(k, i, a, μ_Y), Σ(k, a, μ_X, b, ν_Y), ψ(k, j, b, ν_Y)
temp = einsum('kiaμ, kaμbν -> kibν', ψ.conj(), Σ)  # (k, i, b, ν)
Σ_ij = einsum('kibν, kjbν -> kij', temp, ψ)        # (k, i, j)
```

**Sharding**: $\Sigma_{ij,\mathbf{k}}(k, i, j)$ replicated or batch-sharded over k.

**Output**: The Σ_ij k-matrices are post-processed on the host: `H_QP = kin_ion + Σ_SX + Σ_COH + V_H`, Hermitianized, diagonalized by `jax.vmap(jnp.linalg.eigh)` → `eqp.dat` / `eqp1.dat` (BGW-compatible text, written by `file_io.sigma_output.write_eqp_g0w0` / `write_eqp1`). The full Σ^c(ω) (when `use_ppm_sigma=true`) is written to `sigma_mnk.h5` with datasets `sigma_c_kij_ev` / `sigma_sx_kij_ev` / `hartree_kij_ev` / `omega_ev` via `write_sigma_omega_h5` (phdf5-capable).

**Implementation**:
- Static kernels: `sigma_sx` / `sigma_coh` / `hartree` in `gw/gw_jax.py` (local `@jax.jit` closures that wrap `build_G` + `_convolve` + `project`).
- `build_G`: `gw/greens_function_kernel.py` — unified builder for `G_ii^{occ}`, `G^{all}`, and phased `G(τ)`.
- `project` / `project_ri`: `gw/wavefunction_bundle.py` — the static path uses `project`; the GN-PPM σ^τ path uses a `shard_map`'d reduce-scatter variant (`_make_project_ri_reduce_scatter` in `gw/ppm_sigma.py`) that lands the output `(m_X, n_Y)`-sharded so downstream coeff·σ multiplies stay local.

**Important frequency-dependent caveat**: for the windowed GN-PPM $\Sigma^c(\omega)$ pipeline, the per-window `Re`/`Im` projection must be taken before band projection. In general,

$$K[\operatorname{Re} X] \neq \operatorname{Re} K[X], \qquad K[\operatorname{Im} X] \neq \operatorname{Im} K[X],$$

for the band-projection map $K[X]_{ij,\mathbf{k}} = \sum_{ab\mu\nu}\psi^*_{i,a}(\mu) X_{\mathbf{k},ab}(\mu,\nu)\psi_{j,b}(\nu)$.

The correct reduced-storage implementation is to contract two band-space channels for each $\tau$ node,

$$S_u^{(R)} = K[\operatorname{Re} X_u], \qquad S_u^{(I)} = K[\operatorname{Im} X_u],$$

and then assemble each frequency point from scalar coefficients. If

$$c_u(\omega)=p_u(\omega)+i q_u(\omega),$$

then

$$K[\operatorname{Re}(c_u X_u)] = p_u S_u^{(R)} - q_u S_u^{(I)}, \qquad K[\operatorname{Im}(c_u X_u)] = p_u S_u^{(I)} + q_u S_u^{(R)}.$$

This preserves the exact window algebra while storing only band-space objects. Taking `Re` or `Im` of an already projected complex $\Sigma_{ij,\mathbf{k}}(\tau)$ is not equivalent in general and can make the answer depend on how contributions are partitioned between windows.

### 6.8 Self-Consistency Loop

The quasiparticle (QP) Hamiltonian is:

$$H_{\text{QP}} = H_{\text{KS}} - V_{xc} + \Sigma(\omega \approx E_n)$$

where $H_{\text{KS}} = K + I + V_H + V_{xc}$ is the Kohn-Sham DFT Hamiltonian.

**Implementation detail**: the on-disk `kin_ion` matrix elements are
`H_DFT - V_xc` (kinetic + ionic; Hartree only if explicitly added when the
`kin_ion` file is generated). Therefore the code should **not** subtract
$V_{xc}$ a second time; it forms

$$H_{QP} = (H_{DFT} - V_{xc}) + V_H + \Sigma_{xc}(\omega).$$

**Self-consistent GW** iteratively updates $\Sigma$ until wavefunctions and energies converge:

1. Start with DFT $\psi_n^{(0)}, E_n^{(0)}$
2. Compute $\Sigma^{(i)}[\psi^{(i)}]$ using current wavefunctions
3. Diagonalize $H_{\text{QP}}^{(i)} = H_{\text{KS}} - V_{xc} + \Sigma^{(i)}$ → new $\psi_n^{(i+1)}, E_n^{(i+1)}$
4. Repeat until $|\!|E^{(i+1)} - E^{(i)}|\!| < \epsilon$

**Current status**: A fixed-point COHSEX self-consistency path exists behind `config.self_consistent=True` in `gw_jax.main`. It uses Anderson mixing (`mixing/acceleration.py : rcrop_nojit`, history `m=3`, maxit=40, tol 1e-5) on the flattened upper-Hermitian of Σ_total. Each step:

1. Unflatten Σ, form $H_{QP} = (H_{DFT} - V_{xc}) + \Sigma$, Hermitianize.
2. `jax.vmap(jnp.linalg.eigh)` → `U_k`.
3. Form new occupation projector $G_{ij} = U_k\, f\, U_k^\dagger$ with fixed-count occupation.
4. Recompute $\Sigma_{SX}$, $\Sigma_{COH}$, $V_H$ with the new $G_{ij}$; return flattened upper-Hermitian.

The default `G_0 W_0` path (`self_consistent=False`) performs **one-shot** static COHSEX. A separate diagonal-Σ_xc fixed-point (`gw/qsgw_utils.py : solve_diagonal_sigma_fixed_point`) runs post-hoc when `use_ppm_sigma=true` to evaluate $E = \text{diag}(H_0) + \text{Re}\,\Sigma_{xc}(E)$, with a scissor fit (`gw/scissor.py`) extrapolating out-of-ω-grid bands.

**Implementation**: `gw/gw_jax.py : main` — the `if config.self_consistent:` block builds `_sc_step()` and passes it to `rcrop_nojit`; QSGW Σ^xc reconstruction via `build_qsgw_sigma_xc_from_h5` in `gw/qsgw_utils.py`.

### 6.9 GN-PPM dynamic self-energy Σ^c(ω)

Static COHSEX neglects the ω-dependence of $W$. For a full $\Sigma^c(\omega)$ we use the Godby–Needs plasmon-pole model: every $(μ,ν,q)$ matrix element of the correlated screening $W^c = W - V$ is approximated by a single pole,

$$W^c_{q,μν}(\omega) \approx \frac{B_{q,μν}}{\omega^2 - \Omega_{q,μν}^2},$$

with two parameters $(B, \Omega)$ fitted at each $(μ,ν,q)$ from two known samples — $W^c(0)$ and $W^c(i\omega_p)$ — via `gw/minimax_screening.py : fit_gn_ppm_from_wc_pair`. The driver computes $W(0)$ from the static minimax χ₀, then rebuilds $χ₀(i\omega_p)$ via a second `compute_chi0_minimax` call with an imaginary-frequency quadrature (`build_imag_quadrature`), solves the Dyson equation again for $W(i\omega_p)$, and hands both to `fit_gn_ppm` (`gw/ppm_sigma.py`).

#### 6.9.1 Time-domain integrand

Starting from the standard time-ordered expression $\Sigma^c(\omega) = \frac{i}{2\pi} \int d\omega' G(\omega + \omega') W^c(\omega')$, closing the contour in the upper / lower half-plane and substituting the PPM ansatz gives (per band):

$$\Sigma^c_{n m \mathbf{k}}(\omega) = \sum_q \sum_{a \in \{\mathrm{v}, \mathrm{c}\}} \text{sign}_a \int_0^\infty d\tau\ e^{i\,\text{sign}_\omega \omega \tau}\ \text{project}\!\left[ G_a(\tau) \cdot W_{\text{PPM}}^a(\tau) / \sqrt{N_k} \right]$$

where
- $G_\mathrm{c}(\tau) = \sum_{m \in \text{cond}} e^{-i (E_{m\mathbf{k}} - E_F)\tau}\, \psi_{m\mathbf{k}}(\mu) \psi^*_{m\mathbf{k}}(\nu)$ uses $E_A = E_c - E_F \ge 0$,
- $G_\mathrm{v}(\tau) = \sum_{m \in \text{val}} e^{-i (E_F - E_{m\mathbf{k}})\tau}\, \psi_{m\mathbf{k}}(\mu) \psi^*_{m\mathbf{k}}(\nu)$ uses $H_A = E_F - E_v \ge 0$ with a sign-flipped kernel,
- $W^a_\mathrm{PPM}(\tau) = \sum_{μν} B_{q,μν}\, e^{-i (\Omega_{q,μν} - E_{\mathrm{ref}_B}) \tau}$ (static-pole time transform of the PPM).

Four branches cover $\omega \in \mathbb R$:

| Branch | A-space | `kernel_sign` | `scale` | ω |
|---|---|---|---|---|
| (+ω, cond) | $E_c - E_F$ | +1 | +1 | $\omega \ge 0$ |
| (+ω, val) | $E_F - E_v$ | −1 | +1 | $\omega \ge 0$ |
| (−ω, cond) | $E_c - E_F$ | −1 | −1 | $\omega < 0$, evaluated at $\lvert\omega\rvert$ |
| (−ω, val) | $E_F - E_v$ | +1 | −1 | $\omega < 0$, evaluated at $\lvert\omega\rvert$ |

Enumerated in `_iter_branches` (`gw/ppm_sigma.py`).

#### 6.9.2 Minimax window decomposition

For a given branch the τ integrand decomposes into three regimes of the combined energy $E_A + \Omega$:

1. **Laplace core**: $E_A + \Omega \gg \omega$ — smooth decay, one Laplace-minimax window covers it.
2. **Crossing stripe**: $E_A + \Omega \approx \omega$ — resonance. Uses a phase-minimax (HGL / `solve_phase_minimax_bandwidth`) quadrature and stores only $\text{Im}[\text{coeff} \cdot \sigma^\tau]$ (see `_combine_coeff_with_sigma_tau`, `project_code = "imag"`).
3. **Tail slab**: $E_A + \Omega \ll \omega$ — wide energy bandwidth, covered by a second Laplace minimax with a tighter target error.

`_build_three_sigma_windows` (host-side) builds the three `_SigmaWindow` specs per +ω branch; val and −ω branches use `_build_single_sigma_window`. See the developer notes under `docs/dev/notes/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` for the full derivation of the window edges and error model.

#### 6.9.3 Per-τ kernel

For each τ node the device-side kernel (`_get_sigma_tau_kernel` in `gw/ppm_sigma.py`) builds:

- `Gij[k, i, j] = δ_ij · exp(-i (E_A[k,i] - E_ref_A) τ) · mask_A[k,i]` — band-diagonal occupation projector with Laplace phase.
- `W_t_q[q, μ, ν] = B_q[q,μ,ν] · exp(-i (Ω_q[q,μ,ν] - E_ref_B) τ) · mask_B[q,μ,ν]` — per-(μ,ν) PPM time transform.

and calls the static-shape kernel

$$\sigma^\tau_{k,m,n} = \text{project\_ri}\!\left[ \text{FFT}\!\left[ G_k(\tau) \odot W^\tau_R / \sqrt{N_k} \right] \right]$$

which reuses the same flat-k pipeline as static COHSEX but with the *frequency-integrated* projection variant: `_make_project_ri_reduce_scatter` lands the output `(m_X, n_Y)`-sharded and carries **real + imaginary** channels separately so the crossing window can keep only $\text{Im}[\text{coeff} \cdot \sigma^\tau]$ without materializing a complex σ^τ.

All τ nodes of a single window run inside one `jax.lax.scan` (`_get_sigma_tau_scan_kernel`) so XLA can pipeline NCCL across iterations — this is what makes the GN-PPM path competitive with static COHSEX wall-time for small ω grids.

#### 6.9.4 Projecting τ onto ω

Within each window, the minimax quadrature carries weights $\alpha_\ell$ and nodes $\tau_\ell$. The ω-dependence is *linear in τ* — every τ contribution feeds all ω:

$$c_u(\omega) = \alpha_u\, e^{-i(E_{\mathrm{ref}_A} + E_{\mathrm{ref}_B})\tau_u}\, e^{i\,\text{sign}_\omega \omega \tau_u}$$

Then `_project_tau_onto_omega` adds

$$\Delta \Sigma^c_{kmn}(\omega) = \text{pref} \cdot \text{scale} \cdot P\left[ c_u(\omega) \cdot \sigma^{\tau_u}_{kmn} \right]$$

where $P \in \{\text{full}, \text{imag}\}$ is the window's `project_code`. For crossing windows,

$$P_\text{imag}[c \cdot \sigma] = c_{\Re} \sigma_{\Im} + c_{\Im} \sigma_{\Re} = \text{Im}[c \sigma],$$

which is the correct reduced storage: see the "important frequency-dependent caveat" note at §6.7 — taking `Re`/`Im` of an already-band-projected complex Σ is **not** equivalent, so σ^τ must carry both channels from the kernel.

#### 6.9.5 Accumulation

The $(n_\omega, n_k, m_X, n_Y)$ accumulator is chosen automatically (`omega_accumulation = auto | kij | kij_stream`):

- `_ReduceScatterGpuAccumulator` — accumulate directly on-device, keep the `(m_X, n_Y)` sharding, gather once to host at the end. Multi-process safe. Default for typical ω grids.
- `_StreamedH5Accumulator` — single-process only. Reads / modifies / writes `sigma_c_kij_ry` via rank-0 h5py on every (τ × ω-batch) dispatch. Hundreds of round-trips — currently falls back to accum mode under multi-process. Useful only for very large ω grids that blow the device budget.

Implementation: `compute_sigma_c_ppm_omega_grid` in `gw/ppm_sigma.py`. Output written to `sigma_mnk.h5` (`omega_ev`, `sigma_c_kij_ev`, `sigma_sx_kij_ev`, `hartree_kij_ev`) via `file_io.sigma_output.write_sigma_omega_h5`.

---

## 7. JAX Sharding Summary

Everything runs on a single 2-D mesh `Mesh(devices, ('x', 'y'))` built in `gw_jax._build_mesh` as a most-square factorization of `jax.process_count() * jax.local_device_count()`. There is no `'bands'` axis. Flat-k / flat-q: the 3-D `(nkx, nky, nkz)` form only appears inside `common/fft_helpers.make_flat_k_{fftn,ifftn}`.

### 7.1 Zeta Fitting Pipeline

| Array | Sharding spec | Shape | Notes |
|-------|---------------|-------|-------|
| ψ_yr | `P(None, None, None, 'y')` | `(nk, n, s, μ_Y)` | `wavefunction_bundle.PSI_YR_SPEC` |
| ψ_xr | `P(None, None, None, 'x')` | `(nk, n, s, μ_X)` | Σ-projection LHS |
| ψ_xn | `P(None, None, 'x', None)` | `(nk, s, μ_X, n)` | G/χ₀ LHS |
| ψ_yn | `P(None, None, 'y', None)` | `(nk, s, μ_Y, n)` | Σ-projection RHS |
| P_k (spin-traced) | `P(None, 'x', 'y')` | `(nk, μ_X, ν_Y)` | `compute_pair_density_spin_traced` |
| P_k (spin-matrix) | `P(None, None, None, 'x', 'y')` | `(nk, s, s', μ_X, ν_Y)` | spin-matrix Frobenius mode |
| C_q | `P(None, 'x', 'y')` | `(nq, μ_X, ν_Y)` | CCT, flat-q |
| L_q (tiles) | `P(None, 'x', 'y', None, None)` | `(nq, J_X, J_Y, b, b)` | internal to `cholesky_2d_batched` |
| L_q (dense) | `P(None, 'x', 'y')` | `(nq, μ_X, ν_Y)` | on return |
| Z_q (ZCT) | `P(None, 'x', 'y')` | `(nq, μ_X, r_Y)` | z-chunked build |
| ζ_q (solve out) | `P(None, None, ('x','y'))` | `(nq, μ, r_{XY})` | r-columns distributed |
| `zeta_q.h5` on disk | — | `(nq_flat, n_rtot, n_rmu)` | μ innermost for contiguous r-chunk writes |

### 7.2 Screening + static Σ

| Array | Sharding spec | Shape | Notes |
|-------|---------------|-------|-------|
| G(k) 5-D flat-k | `P(None, None, 'x', None, 'y')` | `(nk, s, μ_X, s', ν_Y)` | `build_G` output |
| G (χ₀ kernel, valence) | `P(None, None, 'x', None, 'y')` | `(nk, s, μ_X, s', ν_Y)` | `_Gv_spec` — μ on x, ν on y |
| G (χ₀ kernel, conduction) | `P(None, None, 'y', None, 'x')` | `(nk, s, μ_Y, s', ν_X)` | `_Gc_spec` — **swapped** to leave einsum output sharded |
| χ₀_R accumulator | `P(None, 'y', 'x')` | `(nR, μ_Y, ν_X)` | `_chi_R_spec` — **y then x** to match `einsum('Rambn,Rbnam->Rmn')` output |
| χ₀_q | `P(None, 'x', 'y')` | `(nq, μ_X, ν_Y)` | after FFT to q |
| V_q, W_q (physics) | `P(None, 'x', 'y')` | `(nq, μ_X, ν_Y)` | flat-q Coulomb / screened |
| V / W inside `_convolve` | `P(None, None, None, 'x', 'y')` | `(nkx, nky, nkz, μ_X, ν_Y)` | 3-D k reshape internal to FFT helper |
| W solve input | `P(('x','y'), None, None)` | `(nq_padded/P, μ, ν)` | q-parallel shard for per-q LU |
| W solve output | replicated | `(nq, μ, ν)` | `with_sharding_constraint` to `rep_3d` |
| Σ_k (static/σ^τ) | `P(None, None, 'x', None, 'y')` | `(nk, s, μ_X, s', ν_Y)` | before band projection |
| Σ_{k,ij} (static) | `P(None, None, None)` | `(nk, i, j)` | replicated after project |
| Σ_{k,ij} (reduce-scatter) | `P(None, 'x', 'y')` | `(nk, m_X, n_Y)` | σ^τ path: `_make_project_ri_reduce_scatter` |
| kin_ion, enk, occ | replicated | `(nk, nb, nb)` / `(nk, nb)` | host-loaded |

### 7.3 GN-PPM parameters

| Array | Sharding spec | Shape | Notes |
|-------|---------------|-------|-------|
| B_q, Ω_q, valid_mask_q | `P(None, 'x', 'y')` | `(nq, μ_X, ν_Y)` | `fit_gn_ppm` output (constrained) |
| W_τ_q (single τ) | `P(None, 'x', 'y')` | `(nq, μ_X, ν_Y)` | `_build_tau_operands` |
| Σ^τ_kmn real / imag | `P(None, 'x', 'y')` | `(nk, m_X, n_Y)` | carried as re/im pair |
| Σ^c(ω) accumulator | `P(None, None, 'x', 'y')` | `(nω, nk, m_X, n_Y)` | `_ReduceScatterGpuAccumulator` |

### 7.4 Key collective patterns

- **χ₀ kernel**: `Gv` and `Gc` use *swapped* μ/ν ↔ 'x'/'y' so that `einsum('Rambn,Rbnam->Rmn')` contracts over the two local axes and leaves output naturally sharded `P(None,'y','x')`. The explicit `with_sharding_constraint(…, _chi_R_spec)` prevents XLA from materializing a replicated 23 GB buffer at Si 4×4×4 60 Ry μ=2400.
- **Dyson W solve**: `shard_map` over `'x'` and `'y'` flattened into a q-parallel axis `P(('x','y'), None, None)`; per-q `fori_loop` with `lu_factor/lu_solve`. Replicated on return.
- **Σ projection** (frequency-dependent path): `shard_map` with **two `psum_scatter`** calls — one over `'x'` scattering `m`, one over `'y'` scattering `n`. Same NCCL byte volume as two plain `psum`s but outputs arrive `(m_X, n_Y)`-sharded so `coeff·σ` in `_project_tau_onto_omega` stays device-local.
- **Cholesky**: `cholesky_2d_batched` operates on tile layout `P(None,'x','y',None,None)`; panel broadcasts + triangular updates via `lax.psum` over axis `'x'`. Falls back to dense `jnp.linalg.cholesky` on 1×1 meshes.
- **Triangular ζ solve**: per-q batch gathered to replicated inside a `shard_map` over `'x'`/`'y'` flattened on the r-column axis; `vmap(solve_triangular × 2)` for L⁻¹ and L⁻H. Python outer loop with `donate_argnums` for sequential GPU execution.

---

## 8. File Organization

### Core implementation

| File | Purpose |
|------|---------|
| `gw/gw_jax.py` | Driver `main()`: mesh, config, ISDF, χ₀/W, static Σ, head, GN-PPM, QSGW |
| `gw/gw_config.py` | `LorraxConfig` (parsed from `cohsex.in`) |
| `gw/gw_init.py` | `compute_optimal_chunks`, `prepare_isdf_and_wavefunctions` |
| `gw/gw_driver_helpers.py` | Config → runtime-option translators (PPM, screening) |
| `gw/w_isdf.py` | χ₀ minimax kernel + W Dyson solve (flat-q) |
| `gw/ppm_sigma.py` | GN-PPM fit + Σ^c(ω) branch/window/τ pipeline |
| `gw/minimax_screening.py` | Window construction + shipped-table lookup + PPM fit |
| `gw/minimax_config.py` | `MinimaxConfig`, `SigmaQuadratureConfig` |
| `gw/greens_function_kernel.py` | `build_G` unified Green's-function builder |
| `gw/head_correction.py` | q→0 head sample + exact static head terms |
| `gw/vcoul.py`, `gw/compute_vcoul.py`, `gw/compute_vcoul_0d.py` | Coulomb kernel + V_q build |
| `gw/qsgw_utils.py` | Diagonal Σ fixed-point + QSGW Σ^xc |
| `gw/scissor.py` | Valence/conduction scissor extrapolation |
| `gw/wavefunction_bundle.py` | `BandSlices`, `Wavefunctions` (4 sharded ψ copies); `project`, `project_ri` band-basis contractions |
| `common/isdf_fitting.py` | CCT/ZCT kernels, Cholesky, ζ solve, full pipeline |
| `common/cholesky_2d.py` | 2D blocked Cholesky |
| `common/fft_helpers.py` | Flat-k ↔ 3-D FFT helpers (custom_partitioning) |
| `common/meta.py` | `Meta` system dataclass |
| `common/symmetry_maps.py` | IBZ → full BZ unfolding, spinor rotations (TRS-augmented `SymMaps`; see [`SYMMETRY_COMPREHENSIVE.md`](symmetry.md) for the BGW `mtrx` / τ convention and the `unfold_psi` / `unfold_v_q` / `compute_centroid_sym_perm` procedures) |
| `common/minimax.py`, `common/minimax_assets/` | Quadrature solvers + shipped tables |
| `common/chi_from_dipole.py` | $S_{\alpha\beta}(\omega)$ from dipole mtxels |
| `file_io/slab_io.py` | `SlabIO` — phdf5 writer (FFI + allgather backends) |
| `file_io/sigma_output.py` | eqp.dat / eqp1.dat / sigma_mnk.h5 |
| `mixing/acceleration.py` | Anderson mixing for self-consistent COHSEX |

### Documentation

| Doc | Focus |
|-----|-------|
| **This file** | Theory + implementation + sharding map. §3–5 is the scalar / r-space narrative; §11 is a short pointer + quick-reference for the current G-flat ζ + bispinor V_q pipeline |
| [`ZETA_V_Q_ALGORITHMS.md`](isdf-zeta-vq.md) | **Current source of truth** for the rank-5 open-spin pair density, scan-INSIDE-shard_map r-chunk loop, G-flat on-disk ζ̃, per-q G-chunked V_q kernel, IBZ cascade, and bispinor Lorentz-tile sectorization |
| [`SYMMETRY_COMPREHENSIVE.md`](symmetry.md) | BGW `mtrx`/τ conventions, `SymMaps`/TRS-augmented table, `unfold_psi`, `compute_centroid_sym_perm`, `unfold_v_q`, symmorphic failure modes, sym-vs-nosym recipe |
| [`CODEBASE_COMPREHENSIVE.md`](../architecture/codebase.md) | Module map, call hierarchy, file formats |
| [`MEMORY_MODEL.md`](../architecture/memory-model.md) | Per-stage memory formulas, G-flat planner, AOT chooser, bottleneck arrays |
| [`MINIMAX_QUADRATURE.md`](minimax-quadrature.md) | CTSP theory, quadrature derivations |
| `docs/dev/notes/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` (developer notes) | GN-PPM Σ^c(ω) window derivations |
| `docs/dev/notes/NEW_WINDOW_MINIMAX_GUIDELINES.md` (developer notes) | Minimax window placement rules |
| `docs/dev/progress/SIGMA_FREQ_AUDIT_STATUS.md` (developer notes) | Current BGW-vs-LORRAX comparison status |

---

## 9. Typical Workflow

### Preparation

1. DFT wavefunctions: `pw2bgw.x` → `WFN.h5`, `WFNq.h5`
2. Centroid selection: `lxpre cohsex.in 640` (runs `centroid.kmeans_cli`, `psp.get_dipole_mtxels`, `gw.kin_ion_io` in sequence) → `centroids_frac.h5`, `dipole.h5`, `kin_ion.h5`
3. Input file: `cohsex.in` with band ranges, memory budget, head source, ISDF pair mode, GN-PPM flags

### GW calculation (one-shot)

```bash
# Perlmutter
lxrun python3 -u -m gw.gw_jax -i cohsex.in           # 4-GPU

# Local
uv run python -m gw.gw_jax -i cohsex.in
```

**Produces**:
- `eqp.dat`, `eqp1.dat` — BGW-compatible QP energy tables
- `sigma_mnk.h5` (when `use_ppm_sigma=true`) — datasets `omega_ev`, `sigma_c_kij_ev`, `sigma_sx_kij_ev`, `hartree_kij_ev`
- `qp_rotations.h5` — U matrices + eigenvalues
- `zeta_q.h5` — cached on disk under `<input_dir>/tmp/isdf_tensors_{n_rmu}.h5` for restarts (unless `isdf_restart=false`)

### Restart

Set `isdf_restart=true` in `cohsex.in` and the driver loads the cached `isdf_tensors_*.h5` instead of re-running the zeta fit.

### Sandbox / BGW comparisons

See the `lorrax_sandbox` superproject `skills/execute_workflow/SKILL.md` for the QE → BGW → LORRAX pipeline, and `skills/compare/SKILL.md` for output parsers.

---

## 10. Known Issues / Active Work

1. **Dyson solve** is q-parallel but not μ-distributed within a q — each device holds the full `(μ, ν)` matrix for its q slab. Acceptable at Si 4×4×4 / MoS2 3×3 scale; bottleneck for >= 10³ k-grids. Future work: SLATE distributed LU (FFI scaffold already in place).
2. **Σ^c(ω) streamed H5 accumulator** falls back to in-GPU accum under multi-process (rank-0 round-trip per (τ × ω-batch) is too expensive). Not urgent unless the kij accumulator blows the GPU budget.
3. **Self-consistent COHSEX** (`self_consistent=true`) converges on tested systems but doesn't compose with `use_ppm_sigma=true` yet (driver raises). A proper QSGW with full Σ^c(ω) is a separate path.
4. **Pseudobands normalization** for non-unit-norm coefficients: ISDF fit divides ψ by `max(1, band_norm)`, but the diag-SC fixed-point uses `eigvalsh(H_qp)` which is unreliable for compressed states — the driver substitutes DFT energies for out-of-grid bands via `scissor.fit_scissor`.
5. **FFI path opt-in**: `use_ffi_io=true` uses phdf5 for zeta/V/sigma I/O. `LORRAX_MPI_TYPE=pmix` required only for legacy OpenMPI paths; the default `cray_shasta` / unified MPICH stack covers all three FFI targets.

---

## 11. Bispinor-aware G-flat pipeline (pointer)

> **Note (2026-05-15).** The detailed algorithms, shardings, donations,
> communication patterns, and code references that previously occupied
> §11.1 – §11.12 have been **moved** out of this file. Maintaining the
> single rank-5 open-spin pair density, the scan-INSIDE-shard_map
> r-chunk loop, the G-flat on-disk ζ̃ layout, the per-q G-chunked V_q
> kernel, the IBZ cascade, and the bispinor Lorentz-tile sectorization
> in three separate documents proved redundant — the new layout is:
>
> | Topic | Document |
> |---|---|
> | Pair-density rank-5 construction, CCT/ZCT/Gram on the full BZ, L_q solve dispatch (Cholesky vs pivoted-LU + ridge), r-chunk loop, G-flat accumulator, per-q V_q kernel with G-chunking, async-prefetch status, bispinor tile sectorization, end-to-end shardings and donations | [`ZETA_V_Q_ALGORITHMS.md`](isdf-zeta-vq.md) |
> | BGW `mtrx`/τ conventions, `SymMaps` and the TRS-augmented index table, `unfold_psi`, `compute_centroid_sym_perm` and the L-wrap, `unfold_v_q` (scalar / pure-permute and TT off-diagonal Cartesian mixing), symmorphic failure modes, sym-vs-nosym recipe, verified gates | [`SYMMETRY_COMPREHENSIVE.md`](symmetry.md) |
> | Per-stage HBM footprints, G-flat planner (`plan_gflat_chunks`), heuristic legacy chooser, AOT NNLS chooser, binding-peak diagnostics, accumulator-FFT-box transient | [`MEMORY_MODEL.md`](../architecture/memory-model.md) |
>
> §3–5 above remain the conceptual / scalar / r-space narrative; §7
> remains the sharding map (static-Σ pipeline); §8 remains the file
> table. The bispinor canonical math reference (γ̃-matrix conventions,
> kinetic-balance lift, Σ^B derivation, open-spin definiteness analysis,
> α_FS² scaling, CrI3 residual status) lives in
> `reports/bispinor_theory_2026-05-09/report.md`.

### 11.1 Summary (1-paragraph context)

The bispinor-aware G-flat pipeline replaces the historical r-space
ζ-on-disk path of §3–5 with: (i) a **single rank-5 open-spin pair
density** `P_{k,αβ}(μ, col) = Σ_n ψ*_{n,k,α}(r_μ) ψ_{n,k,β}(r_col)`
that handles the charge channel (μ_L=0 via γ̃=I short-circuit) and the
three transverse channels (μ_L∈{1,2,3} via γ̃·γ̃ post-IFFT reduction)
through one code path, (ii) a **scan-INSIDE-shard_map r-chunk loop**
that streams ψ(G) from a host cache via `io_callback`, evaluates ζ on
each r-slab, multiplies by the per-q Bloch phase, FFTs once, and
accumulates the per-q `(q+G)`-sphere gather into a μ-sharded
`gflat_acc`; (iii) **G-flat on-disk** layout `ζ̃_{q,μ}(K=q+G)` with the
on-disk q-axis restricted to the **IBZ** subset (factor ~`ntran`
reduction) when centroid orbits are closed under the space group; (iv)
a **per-q G-chunked V_q kernel** that contracts
`V_q^{μ_L ν_L}[μ,ν] = Σ_G ζ̃_L*(q,μ,G) v(q+G) t^{μ_L ν_L}(q+G) ζ̃_R(q,ν,G)`
for the **seven unique** bispinor `(μ_L, ν_L)` tiles (CC + TT-diag-3 +
TT-off-3); and (v) a **post-loop IBZ → full BZ unfold** that is a pure
centroid double-permute for CC and a centroid double-permute composed
with Cartesian double-rotation `R^{ia}(S) R^{jb}(S)` on the (i,j)
Lorentz indices for TT-off. The 21× combined disk-size reduction
(r → G-sphere ≈ 2×, full-BZ → IBZ q-axis up to `ntran`) lands all of
MoS2 3×3 / CrI3 3×3 / CrI3 6×6 80 Ry on within-budget zeta caches.
The full math, sharding, donation, and code-pointer detail is in
[`ZETA_V_Q_ALGORITHMS.md`](isdf-zeta-vq.md); the ζ/V_q
symmetry transformation derivations and unfold implementation choices
are in [`SYMMETRY_COMPREHENSIVE.md`](symmetry.md).

### 11.2 Quick reference (bispinor / G-flat)

| Quantity | Formula |
|---|---|
| Bispinor lift | `ψ_S = (α_FS/2) (σ · (k+G)_cart) ψ_L` |
| γ̃ monomial | `γ̃^μ_{αβ} = phase_μ[α] · δ_{β, perm_μ[α]}` |
| Pair density (open-spin) | `P_{k,αβ}(μ, col) = Σ_n ψ*_{n,k,α}(r_μ) ψ_{n,k,β}(r_col)` |
| γ̃·γ̃ contraction | `Σ_{αβ} phase_L[α] phase_R[β] P_l_conj[α,β] P_r[perm_L[α], perm_R[β]]` |
| CCT (lattice convolution) | `C_q = FFT_{R→q}[ γ̃·γ̃·conj(IFFT(P_l)) · IFFT(P_r) ]` |
| ZCT | same with `col = r_chunk` instead of `col = ν` |
| ISDF normal equations | `C_q ζ_q = Z_q` per q, per μ_L |
| Cell-periodic ζ | `z_{q,μ}(r) = e^{-2πi q·r} ζ_{q,μ}(r)` |
| G-flat ζ (on-disk) | `ζ̃_{q,μ}(K) = FFT_{r→G}[z_{q,μ}(r)](K=q+G)` |
| r-chunk additive identity | `ζ̃(q,μ,K) = Σ_c FFT[pad(phase·ζ_chunk_c)](K)` |
| ζ sym transform (eq. 1) | `ζ̃_{Sq, π_s(μ)}(SG) = e^{-i(Sq+SG)·τ} ζ̃_{q,μ}(G)` |
| V_q (G-chunked) | `V_q^{μ_L ν_L}[μ,ν] = Σ_G conj(ζ̃_L^{μ_L}(q,μ,G)) v(q+G) t^{μ_L ν_L}(q+G) ζ̃_R^{ν_L}(q,ν,G)` |
| V_q transverse weight | `t^{ij}(K) = δ^{ij} − K̂_i K̂_j` |
| V_q scalar unfold (charge) | `V_full[q,μ',ν'] = exp(2π i q_irr·(L_{s,μ'}−L_{s,ν'})) · V_irr[i(q), α_s(μ'), α_s(ν')]` with `α = sym_perm`, `L = L_table` (forward source-map; user-spec inverse form). |
| V_q TT-off unfold (transverse, R_cart mixing) | `V_{Sq}^{ij}(α_s μ, α_s ν) = exp(2π i q_irr·(L_{s,μ}−L_{s,ν})) · Σ_{ab} R^{ia}(S) R^{jb}(S) V_q^{ab}(μ,ν)` |

See [`ZETA_V_Q_ALGORITHMS.md`](isdf-zeta-vq.md) for the
derivations behind each line, the source-of-truth `file.py:line`
citations, and the per-array sharding specs; see
[`SYMMETRY_COMPREHENSIVE.md`](symmetry.md) for the BGW
`mtrx`/τ conventions that fix the meaning of `π_s`, `R(S)`, and the
TRS-augmented index used by `unfold_v_q`.

---

## References

1. **ISDF**: Lu & Ying, *JCTC* 11, 3131 (2015)
2. **CTSP**: Kim, Martyna & Ismail-Beigi, *PRB* 101, 035139 (2020)
3. **BerkeleyGW**: Deslippe et al., *CPC* 183, 1269 (2012)
4. **2D Cholesky**: Golub & Van Loan, *Matrix Computations* (2013)

---

## Appendix: Quick Reference

| Equation | Formula |
|----------|---------|
| Physical charge density | $\rho_{mn,\mathbf{k}} = \sum_s \psi^*_{m,\mathbf{k}-\mathbf{q},s} \psi_{n,\mathbf{k},s}$ |
| ISDF ansatz | $\rho_{mn,\mathbf{k}}(\mathbf{r}) \approx \sum_\mu \zeta_{q,\mu}(\mathbf{r}) \rho_{mn,\mathbf{k}}(\mathbf{r}_\mu)$ |
| CCT matrix | $C_{q,\nu\mu} = \sum_{\mathbf{k},s,s'} P^*_{\mathbf{k}-\mathbf{q},s's}(\nu,\mu) P_{\mathbf{k},ss'}(\nu,\mu)$ |
| ZCT matrix | $Z_{q,\nu}(r) = \sum_{\mathbf{k},s,s'} P^*_{\mathbf{k}-\mathbf{q},s's}(\nu,r) P_{\mathbf{k},ss'}(\nu,r)$ |
| Galerkin solve | $C_q \zeta_q = Z_q$ (via Cholesky) |
| Coulomb ISDF | $V_{q,\mu\nu} = \sum_{\mathbf{G}} z^*_{q,\mu}(\mathbf{G}) v_q(\mathbf{G}) z_{q,\nu}(\mathbf{G})$ |
| CTSP $\chi^0$ | $\chi^0 \sim -\int_0^\infty d\tau \sum_{ab} G^c_{\mathbf{R},ab} G^v_{\mathbf{R},ba}$ |
| Dyson | $(1 - V\chi^0)W = V$ |
| SEX | $\Sigma^X_{\mathbf{R}} = -G^{\text{occ}}_{\mathbf{R}} \circ W_{\mathbf{R}}$ (elementwise) |
| COH | $\Sigma^{\text{COH}}_{\mathbf{R}} = G^{\text{all}}_{\mathbf{R}} \circ (W_{\mathbf{R}} - V_{\mathbf{R}})$ |
| Band projection | $\Sigma_{ij,\mathbf{k}} = \sum_{ab\mu\nu} \psi^*_{i,a}(\mu) \Sigma_{\mathbf{k},ab}(\mu,\nu) \psi_{j,b}(\nu)$ |
