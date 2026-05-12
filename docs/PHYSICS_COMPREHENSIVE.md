# Interpolative Separable Density Fitting: Theory and Implementation

**Consolidates**: `formalism.md`, `isdf_context.md`, `isdf_spin_galerkin_derivation.md`, `ZETA_FITTING_ALGORITHM.md`, `cohsex_jax_physics.md`

**Status**: describes the current implementation in `src/common/isdf_fitting.py` (zeta pipeline), `src/gw/gw_jax.py` (driver), `src/gw/w_isdf.py` (χ₀ + W), `src/gw/ppm_sigma.py` (GN-PPM Σ^c(ω)), `src/gw/head_correction.py` (q→0 head), and `src/gw/greens_function_kernel.py` + `src/gw/projection_kernel.py` (leaf kernels). All physics arrays use the flat-k / flat-q convention (`(nk_tot, …)`); 3-D k-grid layout only appears inside `common/fft_helpers.py`.

---

## Overview

ISDF reduces GW computational cost by approximating pair-product densities with separable interpolation vectors. For systems with $n_k$ k-points, $n_b$ bands, and $n_r$ real-space grid points, storing the full pair-product tensor $M_{mn}(\mathbf{k}, \mathbf{q}, \mathbf{r})$ requires $O(n_k \times n_b^2 \times n_r)$ ~ TB of memory. ISDF reduces this to $O(n_\mu \times n_r)$ where $n_\mu \approx 10 \times n_b$ interpolation points, making real-space GW tractable for small systems.

---

> **Note (2026-05-12).** §3–5 below describe the historical **r-space
> ζ-on-disk** path and the **spin-traced rank-3** pair-density form.
> The current driver runs a **single rank-5 open-spin pair-density**
> path (charge μ_L=0 via identity γ̃ short-circuit, transverse μ_L≠0 via
> γ̃·γ̃ post-IFFT reduction) and the ζ-on-disk image lives in **G-flat**
> layout on the per-q `(q+G)` sphere with **IBZ-only** q-axis (factor
> ~`ntran` disk reduction). Per-r-chunk FFT-and-accumulate replaces the
> per-r-chunk on-disk write. See **§11 below** for the math, shardings,
> and code references of the current pipeline. §3–5 here remain accurate
> for the conceptual / scalar / r-space narrative.

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
- Generation: `src/centroid/kmeans_isdf.py`
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

**Memory**: $\zeta$ is loaded $\mu$-chunked and each q-batch is read from `zeta_q.h5` into a background thread while the previous batch computes on GPU (overlapped I/O, `ThreadPoolExecutor` in `compute_all_V_q_from_zeta_h5`). A single-chunk path vmaps the whole q-batch through one JIT.

**Output**: `V_qmunu` array, shape $(n_{qx}, n_{qy}, n_{qz}, n_\mu, n_\mu)$, sharded `P(None, None, None, 'x', 'y')`. Used directly in memory by the GW pipeline; not routinely persisted to disk.

**Disk bottleneck**: The file `zeta_q.h5` has flat-q layout $(n_q, n_r, n_\mu)$ and is typically **10–100 GB**. Dataset layout puts $n_\mu$ innermost so per-r-chunk writes are contiguous (the earlier `(n_q, n_\mu, n_r)` layout was 8× slower on Perlmutter pscratch).

**Implementation**: `compute_all_V_q_from_zeta_h5()` in `gw/compute_vcoul.py`. Q=0 divergence handled via Voronoi Monte Carlo in `gw/vcoul.py:compute_q0_averages`. Box truncation (0-D molecules) via `gw/compute_vcoul_0d.py`.

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

**See**: `docs/MEMORY_MODEL.md` for detailed formulas and bottleneck arrays.

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

**Reference**: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020). Full derivation in `docs/MINIMAX_QUADRATURE.md`. Windowing strategy in `docs/NEW_WINDOW_MINIMAX_GUIDELINES.md`.

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
- `project` / `project_ri`: `gw/projection_kernel.py` — the static path uses `project`; the GN-PPM σ^τ path uses a `shard_map`'d reduce-scatter variant (`_make_project_ri_reduce_scatter` in `gw/ppm_sigma.py`) that lands the output `(m_X, n_Y)`-sharded so downstream coeff·σ multiplies stay local.

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

`_build_three_sigma_windows` (host-side) builds the three `_SigmaWindow` specs per +ω branch; val and −ω branches use `_build_single_sigma_window`. See [`GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`](GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md) for the full derivation of the window edges and error model.

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
| `gw/projection_kernel.py` | `project`, `project_ri` band-basis contractions |
| `gw/head_correction.py` | q→0 head sample + exact static head terms |
| `gw/vcoul.py`, `gw/compute_vcoul.py`, `gw/compute_vcoul_0d.py` | Coulomb kernel + V_q build |
| `gw/qsgw_utils.py` | Diagonal Σ fixed-point + QSGW Σ^xc |
| `gw/scissor.py` | Valence/conduction scissor extrapolation |
| `gw/wavefunction_bundle.py` | `BandSlices`, `Wavefunctions` (4 sharded ψ copies) |
| `common/isdf_fitting.py` | CCT/ZCT kernels, Cholesky, ζ solve, full pipeline |
| `common/cholesky_2d.py` | 2D blocked Cholesky |
| `common/fft_helpers.py` | Flat-k ↔ 3-D FFT helpers (custom_partitioning) |
| `common/meta.py` | `Meta` system dataclass |
| `common/symmetry_maps.py` | IBZ → full BZ unfolding, spinor rotations |
| `common/minimax.py`, `common/minimax_assets/` | Quadrature solvers + shipped tables |
| `common/chi_from_dipole.py` | $S_{\alpha\beta}(\omega)$ from dipole mtxels |
| `file_io/slab_io.py` | `SlabIO` — phdf5 writer (FFI + allgather backends) |
| `file_io/sigma_output.py` | eqp.dat / eqp1.dat / sigma_mnk.h5 |
| `mixing/acceleration.py` | Anderson mixing for self-consistent COHSEX |

### Documentation

| Doc | Focus |
|-----|-------|
| **This file** | Theory + implementation + sharding map. §3–5 is the scalar / r-space narrative; §11 is the current rank-5 open-spin + G-flat ζ + bispinor Lorentz-tile pipeline |
| [`CODEBASE_COMPREHENSIVE.md`](CODEBASE_COMPREHENSIVE.md) | Module map, call hierarchy, file formats |
| [`MEMORY_MODEL.md`](MEMORY_MODEL.md) | Per-stage memory formulas, bottleneck arrays |
| [`MINIMAX_QUADRATURE.md`](MINIMAX_QUADRATURE.md) | CTSP theory, quadrature derivations |
| [`GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`](GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md) | GN-PPM Σ^c(ω) window derivations |
| [`NEW_WINDOW_MINIMAX_GUIDELINES.md`](NEW_WINDOW_MINIMAX_GUIDELINES.md) | Minimax window placement rules |
| [`SIGMA_FREQ_AUDIT_STATUS.md`](SIGMA_FREQ_AUDIT_STATUS.md) | Current BGW-vs-LORRAX comparison status |

---

## 9. Typical Workflow

### Preparation

1. DFT wavefunctions: `pw2bgw.x` → `WFN.h5`, `WFNq.h5`
2. Centroid selection: `lxpre cohsex.in 640` (runs `centroid.kmeans_isdf`, `psp.get_dipole_mtxels`, `gw.kin_ion_io` in sequence) → `centroids_frac.h5`, `dipole.h5`, `kin_ion.h5`
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

## 11. Bispinor-aware G-flat pipeline (current)

Companion to §3–5 above. §3–5 describes the historical **r-space
ζ-on-disk** path and the **spin-traced rank-3** pair density. This
section is the current source of truth for:

* the **single rank-5 open-spin pair-density** path that runs for every
  Lorentz channel (charge μ_L=0 and the three transverse μ_L∈{1,2,3});
* the **G-flat ζ-on-disk** layout — ζ̃(q,μ,G) on the per-q `(q+G)` sphere —
  built by `accumulate_rchunk_to_gflat` during the r-chunk loop;
* the **IBZ-only on-disk q-axis** and the post-V_q symmetry unfold;
* the **per-q G-chunked V_q kernel** and its bispinor extension to seven
  unique `(μ_L, ν_L)` tiles.

**Scope**: Phase-1 DHF + bare-Breit (the channels currently wired through
`gw.gw_jax`). Σ^B (transverse bare-Breit) is documented for completeness
in §11.10 but its detailed regression status lives in
`reports/bispinor_theory_2026-05-09/report.md`.

Source-of-truth citations are inlined as `file.py:line`. Cited shapes
match the current code; cited shardings use the standard mesh
`Mesh(devices, ('x','y'))` with `P = p_x · p_y`.

### 11.1 Conventions

#### γ̃-matrix convention (absorbed γ⁰)

Following the absorbed convention (`gamma_matrices.py:16`), LORRAX uses

$$\boxed{\tilde\gamma^\mu \equiv \gamma^0 \gamma^\mu, \qquad \tilde\gamma^0 = I_4,\quad \tilde\gamma^i = \alpha^i}$$

so the bispinor pair density is `ρ^{μ_L} = Ψ† γ̃^{μ_L} Ψ` (no explicit
`Ψ̄`). Every `γ̃^μ` is **monomial** — exactly one nonzero per row/column,
value ∈ {±1, ±i}. The (perm, phase) decomposition

$$\boxed{\tilde\gamma^\mu_{\alpha\beta} = \mathrm{phase}_\mu[\alpha] \cdot \delta_{\beta,\, \mathrm{perm}_\mu[\alpha]}}$$

is precomputed at `gamma_matrices.py:85-87` (arrays `gammas_perm`,
`gammas_phase`, both `(4, 4)`). Explicit tables:

| μ | perm        | phase             | spectrum of γ̃^μ |
|---|-------------|-------------------|------------------|
| 0 | (0,1,2,3)   | (+1,+1,+1,+1)     | {+1,+1,+1,+1}    |
| 1 | (3,2,1,0)   | (+1,+1,+1,+1)     | {+1,+1,−1,−1}    |
| 2 | (3,2,1,0)   | (−i,+i,−i,+i)     | {+1,+1,−1,−1}    |
| 3 | (2,3,0,1)   | (+1,−1,+1,−1)     | {+1,+1,−1,−1}    |

#### Kinetic-balance lift

`bispinor_init.py:12-41`. From the large-component wavefunction
ψ_L,n,k(G) the small component is

$$\boxed{\psi_{S,n,k}(G) = \tfrac{\alpha_{\mathrm{FS}}}{2}\,\big(\boldsymbol\sigma\cdot(k+G)_{\mathrm{cart}}\big)\,\psi_{L,n,k}(G)}$$

with `halfalpha = α_FS / 2 = 1/(2·137.036) = 0.00364867628215`
(`bispinor_init.py:30`). The bispinor is assembled L-then-S:

$$\Psi(r) = \begin{pmatrix} \psi_L^\uparrow \\ \psi_L^\downarrow \\ \psi_S^\uparrow \\ \psi_S^\downarrow \end{pmatrix} \in \mathbb{C}^4, \qquad \mathtt{nspinor} = 4.$$

Norm: `‖Ψ‖² = ‖ψ_L‖² · (1 + O(α_FS²))`. The lift is *not* renormalized
post-construction — the O(α²) correction is below the Phase-1 truncation.

#### Bispinor pair density and the ISDF approximation

For vertex μ_L the pair density between (n_l, k) and (n_r, k+q) is

$$\boxed{\rho^{\mu_L}_{n_l n_r, k, q}(r) = \sum_{\alpha\beta=1}^{4} \psi^*_{l,n_l,k,\alpha}(r)\,\tilde\gamma^{\mu_L}_{\alpha\beta}\,\psi_{r,n_r,k+q,\beta}(r)}$$

Special cases: μ_L=0 (γ̃⁰=I_4) reduces to the **spin-traced charge
density** `Σ_α ψ*_l ψ_r`; μ_L=i (γ̃^i=α^i) is the *i*-th component of the
Dirac current `Ψ†α^iΨ` and is O(α_FS) by the kinetic-balance lift.

ISDF (Dong–Hu–Lin) replaces the band-pair product by a centroid expansion
**separable in (n_l, n_r)**:

$$\boxed{\rho^{\mu_L}_{n_l n_r, k, q}(r) \;\approx\; \sum_{a=1}^{n_{r\mu}}\,\zeta^{\mu_L}_{q,a}(r)\,\rho^{\mu_L}_{n_l n_r, k, q}(r_a)}$$

ζ depends on q and μ_L but **not** on the band pair. Two centroid files
in production: charge (k-means on `Σ_n|ψ|²`) for μ_L=0, current
(k-means on `Σ_i|j^{\mathrm{Gordon}}_i|²` with
`j^{\mathrm{Gordon}} = \mathrm{Im}[\psi_L^\dagger\nabla\psi_L] +
\tfrac12\nabla\times(\psi_L^\dagger\boldsymbol\sigma\psi_L)`) for μ_L=1,2,3.
The three transverse channels share **one** current-centroid file.
Counts `n_rmu_C` (charge) and `n_rmu_T` (current) are independent.

#### Lattice conventions, units, FFT norms

* **Bloch phase**: `ψ_{nk}(r) = Σ_G c_{nk}(G) e^{i(k+G)·r}`, with the
  cell-periodic part `u_{nk}(r) = e^{-ik·r}ψ_{nk}(r) = Σ_G c_{nk}(G) e^{iG·r}`.
  ζ is stored as the Bloch form ζ_{q,μ}(r) on the r-space path; on the
  G-flat path the on-disk array is the **cell-periodic FFT** of the
  Bloch ζ, i.e. ζ̃_{q,μ}(K=q+G) (see §11.6 for the explicit conversion).
* **FFT-grid Miller indices**: `np.fft.fftfreq(N)*N` order. The flat-r
  layout is `r_flat = rx·ny·nz + ry·nz + rz`.
* **q-axis on disk**: `kgrid_int` BGW-wrapped (`q > kg/2 → q − kg`),
  divided by kgrid; matches the writer's `q_irr_frac`
  (`v_q_g_flat.py:192-197`).
* **FFT norms**:
  * Pair-density k-convolution (`c_q_from_pair`, `z_q_from_pair`):
    `norm='forward'` on both IFFT_{k→R} and FFT_{R→q}
    (`isdf_fitting.py:227-228`). This makes the convolution-theorem
    identity `C_q = FFT(conj(IFFT(P_l)) ⊙ IFFT(P_r)) = Σ_k P_l^*_k P_{r,k+q}`
    reproduce the direct sum without an extra prefactor.
  * G-flat accumulator (`accumulate_rchunk_to_gflat`): `norm='backward'`
    on the r→G forward FFT (`wfn_transforms.py:609`). Forward FFT applies
    no 1/N factor; downstream V_q kernel absorbs the normalization
    constant into its definition.
* **Units**: Hartree atomic units internally; eV at output. Bohr⁻¹ for
  reciprocal vectors `(k+G)·b`.

#### Mesh, shardings, sizes

`Mesh(devices.reshape(p_x, p_y), ('x','y'))`, `P = p_x · p_y`. Spec
notation: `μ_X` = sharded on `'x'`, `μ_XY` = sharded on flat `('x','y')`,
`μ_` = replicated.

Typical → extreme:

| Symbol | Range | Where it sits |
|---|---|---|
| `n_rtot = nx·ny·nz` | 5k–2M+ | FFT-box bottleneck; everything that materialises one must be chunkable |
| `n_band` | 50–10k | per-channel left/right windows |
| `n_rmu` | 500–100k (≈10·n_band) | centroid count; rounded up to `mesh.size` as `n_rmu_padded` |
| `n_G_sph ≈ 0.05·n_rtot` | 1k–100k | per-q `(q+G)` sphere; padded to `ngkmax = max_q ngk[q]` |
| `n_k, n_q` | 1–10000 | flat-k / flat-q |
| `n_q_disk` | 1–n_q | IBZ subset (factor `≤ ntran` reduction) |

### 11.2 Open-spin rank-5 pair density

#### Definition

`isdf_fitting.py:82-148`. The pair-density tensor with both spinor axes
kept **open** is

$$\boxed{P_{k,\alpha\beta}^{(\mathrm{open})}(\mu,\,\mathrm{col}) = \sum_n \psi^*_{n,k,\alpha}(r_\mu)\,\psi_{n,k,\beta}(r_{\mathrm{col}})}$$

Einsum `'kmna,knbr->kabmr'` (`isdf_fitting.py:112, 145`).

| Array | Shape | Sharding | Notes |
|---|---|---|---|
| `psi_rmuT_X` (left) | `(nk, n_rmu, nb, ns)` | `P(None, 'x', None, None)` | rank-4 |
| `psi_rcol_Y` (right) | `(nk, nb, ns, n_col)` | `P(None, None, None, 'y')` | rank-4; `n_col = n_rmu` for CCT, `n_col = r_chunk` for ZCT |
| `P_k,αβ` | `(nk, ns, ns, n_rmu, n_col)` | `P(None, None, None, 'x', 'y')` | rank-5 |

For ns=4 (bispinor) the spin-pair index `(α, β)` runs over 16
combinations. Memory cost is **16× the historical spin-traced rank-3
form**, paid at the (k, a, b, μ, col) carrier. Band-chunked accumulation
streams over n so peak memory is bounded; see `accum_pair_density`
(`isdf_fitting.py:119-148`, donates `P_in`).

#### Why rank-5 open-spin, not rank-3 spin-traced + γ̃-weighted

The Schur form folds γ̃ into the spin-trace at construction time,
`P^{\mathrm{Schur}}_k(\mu,\nu) = Σ_{n,\alpha\beta} ψ^*_{n,k,\alpha}(r_\mu)
\tilde\gamma_{\alpha\beta} ψ_{n,k,\beta}(r_\nu)`. The resulting `C_q` is
**indefinite** for γ̃^i (eigenvalues of `γ̃^* ⊗ γ̃` are products of
γ̃-eigenvalues; for γ̃^i = α^i these are ±1 each, giving 8 modes of each
sign on the 16-dim spin-pair index). Cholesky NaNs; naive LU blows up on
near-null modes (1e17 trace on CrI3, pre-fix).

The open-spin form keeps the γ̃ contraction at the spatial-point
reduction (§11.3 below): γ̃ is contracted only **inside** the band-pair
sum of magnitudes,

$$\langle v | K_q | v \rangle = \sum_{n_l, n_r, k}\bigg|\sum_{\mu_c} v_{\mu_c}\,\tilde\rho^{\mu_L}_{n_l n_r, k, q}(r_{\mu_c})\bigg|^2 \;\geq\; 0,$$

manifestly PSD for any μ_L. (The code still dispatches μ_L≠0 to pivoted
LU + ridge — see §11.4.2; the conservatism is an open question
documented in `reports/bispinor_theory_2026-05-09 §4.5`.)

#### γ̃·γ̃ double contraction (`gamma_double_contract`)

`gamma_matrices.py:128-164`. Collapses the rank-5 P pair to rank 3 by

$$\boxed{[\tilde\gamma_L \cdot \tilde\gamma_R \cdot P_l^* P_r](\mu, \mathrm{col}) = \sum_{\alpha\beta}\mathrm{phase}_L[\alpha]\,\mathrm{phase}_R[\beta]\,P^*_{l,\alpha\beta}\,P_{r,\,\mathrm{perm}_L[\alpha],\,\mathrm{perm}_R[\beta]}}$$

implemented as a `take`+`take` gather followed by an element-wise phase
multiply and a sum on the two spin axes — no 4×4 matmul. `perm_L=None`
(charge / left-identity short-circuit) skips the gather on that side;
both None → pure Frobenius `Σ_{αβ} P_l^*·P_r` (charge channel; identical
to the historical spin-traced form).

### 11.3 CCT, ZCT, q=0 Gram on the full BZ

#### `c_q_from_pair` — CCT matrix

`isdf_fitting.py:168-279`. For each q,

$$\boxed{C^{\mu_L \nu_L}_q(\mu,\,\nu) = \sum_k\, \big[\tilde\gamma_L \cdot \tilde\gamma_R \cdot P_l^* P_r\big]_{k,\,\mu,\,\nu}\,e^{-i\,q\cdot R(k)}}$$

evaluated via FFT (k-convolution, `norm='forward'`):

| Step | Output | Shape | Sharding |
|---|---|---|---|
| `_ifft_conj(P_l)` | `P_l^* in R` | `(nk, s, s', μ, ν)` | `P(None, None, None, 'x', 'y')` |
| `local_ifftn_spin(P_r)` | `P_r in R` | same | same |
| `gamma_double_contract(P_l*, P_r; γ̃_L, γ̃_R)` | rank-3 `C_R` | `(nk, μ, ν)` | `P(None, 'x', 'y')` |
| `local_fftn_scalar(C_R)` | `C_q` | `(nq, μ, ν)` | `P(None, 'x', 'y')` |

Donates `P_l` and `P_r` (`donate_argnums=(0,1)` at line 246) so each
ifft + contract is in-place. Both pair-density inputs are at the same
sharding `P(None, None, None, 'x', 'y')`. The k→R IFFT uses
`make_flat_k_ifftn` (flat-k with `custom_partitioning`); the R→q FFT
uses `make_flat_k_fftn`. q-axis on output is the full BZ — the IBZ
subset is taken at write time (§11.6).

#### `z_q_from_pair` — ZCT matrix on (μ, r-chunk)

Same structure as above with `col = r_chunk` instead of `col = ν`.
Output `Z_q^{μ_L ν_L}(μ, r)` shape `(nq, n_rmu, r_chunk)` at sharding
`P(None, 'x', 'y')`.

#### `gram_q0_from_pair` — q=0 valence-conduction Gram

`isdf_fitting.py:282-370`. Used by `centroid.pivoted_cholesky` to pick
centroids. At q=0 the k-sum **is** the answer (no convolution); the same
γ̃·γ̃ reduction applies. Output `(n_rmu, n_rmu)` Hermitian PSD at
`P('x','y')`. Symmetrized explicitly to suppress fp roundoff
(`G = 0.5·(G + G^H)` at line 353).

#### Channel dispatch in the fit driver

`isdf_fitting.py:1862-1888` (legacy r-space) and `:1480` (G-flat).
Single rank-5 path for both charge and transverse:

```python
gamma_perm_phase  = None if vertex_mu_L == 0 else gamma_perm_phase_mu(vertex_mu_L)
gamma_L = gamma_R = gamma_perm_phase   # same vertex on both legs
P_acc = pair_density_accum(P_acc, ψ_rmuT_X, ψ_rcol_Y, mesh_xy)   # band-chunk loop
C_q   = c_q_from_pair(P_l_k_ab, P_r_k_ab, gamma_L, gamma_R, kgrid=…, mesh_xy=…)
Z_q   = z_q_from_pair(P_l_k_ab_muz, P_r_k_ab_muz, gamma_L, gamma_R, kgrid=…, mesh_xy=…)
```

For μ_L=0 the `gamma_L/R = None` branches short-circuit the gather and
phase mul at trace time — the kernel is byte-identical to a hypothetical
"charge-only" path through the spin-traced rank-3 helpers. The 16× spin
factor is a price paid uniformly.

### 11.4 L_q factorization and per-q triangular / LU solve

#### Cholesky (μ_L = 0)

`isdf_fitting.py:921-1103`. `compute_L_q_from_CCT` runs 2D-blocked
batched Cholesky (`cholesky_2d.cholesky_2d_batched`) on the tile layout
`L_q[q, J_X, J_Y, b, b]` with sharding `P(None, 'x', 'y', None, None)`,
returning the dense view at `P(None, 'x', 'y')`. Falls back to dense
`jnp.linalg.cholesky` with a trace-proportional ridge `1e-14·|tr|/n` on
1×1 meshes (line 1060). Per-q triangular solve in
`solve_zeta_from_L_q` (`isdf_fitting.py:1110-1230`):

```python
# shard_map(in_specs=(P(None,None,None), P(None,None,('x','y'))),
#           out_specs=P(None,None,('x','y')))
y    = solve_triangular(L,        Z_cols, lower=True)
zeta = solve_triangular(L.conj().T, y,    lower=False)
```

L is gathered to replicated inside the shard_map; each device solves
its r-column shard. Python outer loop with `donate_argnums` keeps
sequential GPU execution (fori_loop SPMD-replicates the sharded carry;
scan(unroll=8) preallocates a 19 GB temp).

#### Pivoted LU + ridge (μ_L ≠ 0)

`isdf_fitting.py:1183-1202`. For transverse channels the code treats
C_q as Hermitian indefinite (overcautious per §11.2 analysis; see
`reports/bispinor_theory_2026-05-09 §4.5` for the open question):

```python
n = L.shape[-1]
ridge = LU_RIDGE * jnp.abs(jnp.trace(L)) / n     # LU_RIDGE = 1e-12
L_reg = L + ridge * jnp.eye(n, dtype=L.dtype)
return jnp.linalg.solve(L_reg, Z)
```

Ridge `1e-12 · |tr(C_q)| / n_rmu` sits well below physically meaningful
eigenvalues and above the partial-pivoting stability floor. JAX exposes
no Bunch-Kaufman LDLᵀ, so pivoted LU (`jnp.linalg.solve` → `lu_solve`)
is the practical Hermitian-indefinite path.

LU is ~10% slower per chunk than Cholesky at CrI3 6×6 scale (16 GPUs,
`r_chunk=25000`): Cholesky 9.7 s/chunk vs LU 9.8–10.7 s/chunk
(`reports/bispinor_pipeline_2026-05-04 §3`).

#### `solve_zeta` reshard discipline

Solver native output → target sharding `P(None, ('x','y'), None)`
(μ_XY, r_):

| Solver | Native out spec | Reshard path |
|---|---|---|
| `cusolvermp_{cholesky,lu}` | `P(None, 'x', 'y')` | one step, single mesh axis 'y' moves r→μ |
| `sharded_{cholesky,lu}` (shard_map fallback) | `P(None, None, ('x','y'))` | two steps via `P(None, 'x', 'y')`: 'x' then 'y' |

Each step is a clean single-mesh-axis all-to-all
(`_reshard_zeta_mu_X_r_Y_to_mu_XY` at `isdf_fitting.py` and the two-step
variant). The reshard is the only collective in `solve_zeta`'s output
path; ~3 ms/call. Trailing `('x','y')` placement is what
`accumulate_rchunk_to_gflat` requires.

### 11.5 r-chunk loop and the G-flat accumulator

This is the part that has changed since
`reports/bispinor_theory_2026-05-09`. The r-chunk loop no longer writes
ζ in r-space to disk; instead each r-chunk's contribution is
**FFT'd-and-accumulated** directly into a μ-sharded G-flat buffer.

#### Math: linearity over r-chunks

The cell-periodic-on-disk form is

$$\boxed{\tilde\zeta_{q,\mu}(K) = \frac{1}{N_r}\sum_r e^{-iK\cdot r}\zeta_{q,\mu}(r) = \mathrm{FFT}_{r\to G}\!\Big[e^{-2\pi i\,q\cdot r}\,\zeta_{q,\mu}(r)\Big](G),\quad K = q + G}$$

Since `FFT_{r→G}` is linear in the r-input, splitting r into disjoint
chunks `r ∈ R_c = [r₀^c, r₀^c + r_len)` and summing,

$$\tilde\zeta_{q,\mu}(K) = \sum_c \mathrm{FFT}_{r\to G}\!\Big[\mathrm{pad}_{R_c\to[0,n_{\mathrm{rtot}})}\!\big(e^{-2\pi i\,q\cdot r}\,\zeta_{q,\mu}(r)|_{r\in R_c}\big)\Big](G)$$

Each chunk contributes additively. The r-chunk loop in
`fit_zeta_to_h5` evaluates `ζ_{q,μ}(r)` only on `R_c` (the back-solve
runs at r-chunk extent), multiplies by the per-q phase on the slab,
zero-pads to the full FFT box, FFTs once, and adds the per-q sphere
gather into `gflat_acc`. The r-space ζ-on-disk image is **never
materialised** — its on-disk size (which was the bottleneck) is paid in
FFT work instead: `n_rchunks×` chunked FFT boxes.

#### `accumulate_rchunk_to_gflat`

`common/wfn_transforms.py:439-626`. One `shard_map` over `('x','y')`; no
cross-rank collectives in the body.

```
in  : rchunk    (n_q_disk, n_rmu_padded, r_len)     P(None, ('x','y'), None)
      gflat_acc (n_q_disk, n_rmu_padded, ngkmax)    P(None, ('x','y'), None)   ← donated
out : (n_q_disk, n_rmu_padded, ngkmax)              P(None, ('x','y'), None)
```

Per-rank body inside the shard_map (`wfn_transforms.py:584-620`):

```python
N = n_q · n_mu_local;   n_mu_local = n_rmu_padded / P
rch_flat = rchunk.reshape(N, r_len)
acc_flat = acc.reshape(N, ngkmax)
zero-pad both to ⌈N/cs⌉ · cs along axis 0          # cs = chunk_size

for i in range(n_chunks):                          # lax.scan, donated acc carry
    sub      = dynamic_slice(rch_flat, i·cs, cs, axis=0)
    q_row[cs] = clip((i·cs + arange(cs)) // n_mu_local, 0, n_q-1)
    box      = zeros(cs, n_rtot).update_slice(sub, r0, axis=-1).reshape(cs, nx, ny, nz)
    if qvec_frac:
        box *= phx[q_row] · phy[q_row] · phz[q_row]    # separable e^{-2πi q·r}
    G_box    = jnp.fft.fftn(box, axes=(-3,-2,-1), norm='backward')      # local cuFFT
    contrib  = take_along_axis(G_box.reshape(cs, n_rtot), sphere_c[q_row], axis=-1)
    acc_flat = dynamic_update_slice(acc_flat, slice(acc_flat,i·cs,cs) + contrib, i·cs)

return acc_flat[:N].reshape(n_q, n_mu_local, ngkmax)
```

**Key invariants**:

1. **Flat-axis chunking on `(q · μ_local)`**: cs is a free integer; no
   divisibility constraint on n_q or n_mu_local. Pad rows
   (`pad_N = ⌈N/cs⌉·cs − N`) are zero-padded → contribute zero, no
   contamination of `acc`. `q_row ≥ n_q` from pad rows is clipped to
   `n_q-1` but the slab is zero so the take is harmless.
2. **Sharding contract**: μ is the only sharded axis on both `rchunk`
   and `gflat_acc`. The r-axis and the G-sphere axis are replicated.
   The FFT axes are entirely *per-rank-local*; `jnp.fft.fftn` inside the
   shard_map dispatches local cuFFT, no resharding.
3. **Per-q tables baked at trace time** (closure-replicated per rank):
   * `sphere_c (n_q, ngkmax) int32` — per-q sphere into the flat FFT
     box. `sphere_c[q, 0] == 0` (G=(0,0,0)) by construction
     (`coulomb_sphere.py:197-204`); pad slots use the BZ-corner
     sentinel `(nx/2, ny/2, nz/2)` which the writer zeroes post-loop
     via `jnp.where` (see §11.5.4).
   * `phx (n_q, nx)`, `phy (n_q, ny)`, `phz (n_q, nz)` — separable
     `exp(-2πi q·r)` Bloch-phase factors.
4. **Donation**: `gflat_acc` is `donate_argnums=(1,)` on the outer jit
   (line 622); inside the shard_map's `lax.scan`, `acc_flat` is the
   carry → donated by scan semantics.

#### Chunker parameter

`chunk_size` (rows per scan iter):
* Default `None` ⇒ one-shot (`cs = N`, scan compiles to 1 iter that XLA
  folds away).
* Env override: `LORRAX_GFLAT_CHUNK_SIZE`.
* Memory bound per rank: `chunk_size · n_rtot · 16 B` for the per-iter
  FFT box (the only transient).
* Suggested set: `chunk_size ≈ memory_budget_bytes / (n_rtot · 16) /
  6` (~6× live-copy slack). Per-rank ~1 GB budget ⇒ `cs ≈ 8e7/n_rtot`.
  At MoS2 3×3 (n_rtot=46k) one-shot fits; at CrI3 J_3x3 (n_rtot=243k)
  `cs ≈ 300`; at CrI3 6×6 80 Ry (n_rtot=1.125 M) `cs ≈ 64`.

#### Outer driver wiring (`fit_zeta_to_h5`)

`isdf_fitting.py:2191-2273` (chunk loop), `:2330-…` (post-loop write).

For each r-chunk:

1. `psi_G_store.begin_rchunk(r_start, r_end)` — bring host ψ(G) tiles
   into scope (file-reread mode) or no-op (host-cache mode).
2. `fit_one_rchunk(…)` — one fused jit over: ψ(G) → ψ(r-slab) (band
   chunked), pair-density accumulation (rank-5), C_q-product k→R→q
   convolution (§11.3), `solve_zeta` (§11.4). Returns
   `zeta_chunk (n_q_disk, n_rmu_padded, r_chunk)` at
   `P(None, ('x','y'), None)`. (IBZ slicing happens *inside*
   `fit_one_rchunk` before the solve — Phase B; the legacy full-BZ slice
   path is still wired for the q_irr_full_idx=None case.)
3. `gflat_acc = accumulate_rchunk_to_gflat(zeta_chunk_ibz, gflat_acc,
   r0=r_start, sphere_idx=…, qvec_frac=…, chunk_size=…, mesh=…)` ←
   donates `gflat_acc`, returns the updated buffer.
4. `del zeta_chunk_ibz` — the only persistent ζ object is `gflat_acc`.

Post-loop write (line ~2350):

```python
# Mask pad slots: per-q sphere has length ngk[q] ≤ ngkmax with sentinel pad.
mask = (arange(ngkmax)[None, :] < ngk_per_q[:, None])         # (n_q_disk, ngkmax)
gflat_acc = jnp.where(mask[:, None, :], gflat_acc, 0)         # zero-out pad
zeta_io.write_slab('zeta_q_G', gflat_acc,
                   valid_shape=(n_q_disk, n_rmu_logical, ngkmax))   # μ-pad clipped
```

`SlabIO.valid_shape=` clips the μ-pad on the write so on-disk extent is
**logical** `n_rmu`. phdf5 FFI path when `use_ffi_io=true`; allgather
fallback otherwise.

#### Cost model (CrI3 6×6 80 Ry, 16 GPUs, 4×4 mesh)

Per-rank c128 = 16 B, n_q_disk = 8 IBZ, n_rmu = 1504, n_rtot = 1.125 M,
chunk_size = 64:

| Object | Logical shape | Per-rank | Bytes/rank |
|---|---|---|---|
| `gflat_acc` (persistent) | `(8, 1504, ngkmax≈55k)` | `(8, 1504/16, 55k)` | 0.66 GB |
| `rchunk` (per iter, donated) | `(8, 1504, r_chunk)` | `(8, 1504/16, r_chunk/4)` | depends on r_chunk |
| Per-iter FFT box | `(cs, n_rtot)` | same (replicated G) | `cs · n_rtot · 16 = 1.15 GB` at cs=64 |

The accumulator's per-iter peak (1.15 GB) is the new memory term added
by the G-flat refactor. The r-space ζ-on-disk image it replaces was
much larger (`n_q · n_rtot · n_rmu · 16` ≈ 80 GB for MoS2 3×3, ~ TB for
CrI3 6×6) and lived entirely on disk; the new path keeps no r-space
on-disk image at all (`zeta_q.h5` becomes a 1D sphere store).

### 11.6 IBZ-only on-disk layout and ζ symmetry transformation

`reports/zeta_ibz_2026-05-11/report.md` is the design document; this
section captures the conventions and identities used by the writer and
the unfold path.

#### Bloch ↔ cell-periodic ↔ G-sphere

Three forms of ζ, all equivalent up to a Bloch phase and an FFT:

| Form | Symbol | Domain |
|---|---|---|
| Bloch (real-space, on-disk legacy) | `ζ_{q,μ}(r)` | `r ∈ [0, n_rtot)` |
| Cell-periodic (transient) | `z_{q,μ}(r) = e^{-2πi q·r}\,ζ_{q,μ}(r)` | `r ∈ [0, n_rtot)` |
| G-sphere (on-disk current) | `ζ̃_{q,μ}(K) = (1/N_r)\,\mathrm{FFT}_{r\to G}[z_{q,μ}(r)](K)` | `K ∈ \mathrm{sphere}(q)` |

with `K = q + G`. The `accumulate_rchunk_to_gflat` body realises the
last identity per r-chunk (linearity in r) up to the `norm='backward'`
factor — the kernel that consumes ζ̃ in V_q absorbs the missing 1/N_r
into the v(q+G) overlay.

#### Symmetry transformation of ζ

`WFN.h5` sym ops `{S | τ}`: `S = wfn.sym_matrices[s]` integer rotation
in crystal coords (BGW `mtrx`); `τ = wfn.translations[s] / (2π)`
fractional translation. Pair density transforms by

$$\rho_{Sq}(SG) = e^{-i(Sq + SG)\cdot\tau}\,\rho_q(G).$$

If centroids are closed under `{S|τ}` (orbit-aware k-means + grid-snap
ensures `S r_μ + τ ≡ r_{π_s(μ)}` mod 1, with `π_s` the centroid
permutation), the same transformation law lifts to ζ with a centroid
permutation on the μ leg:

$$\boxed{\tilde\zeta_{Sq,\,\pi_s(\mu)}(SG) = e^{-i(Sq + SG)\cdot\tau}\,\tilde\zeta_{q,\mu}(G)}\qquad\text{(eq. 1)}$$

Inverse form (unfold IBZ → full BZ for a single q_full = S · q_irr):

$$\tilde\zeta_{q_{\mathrm{full}},\,\nu}(G_{\mathrm{target}}) = e^{-i(q_{\mathrm{full}} + G_{\mathrm{target}})\cdot\tau_s}\,\tilde\zeta_{q_{\mathrm{irr}},\,\pi_{s}^{-1}(\nu)}(S^{-1} G_{\mathrm{target}}).$$

The unfold needed for *V_q* (not ζ itself) avoids the τ-phase entirely
because V is bilinear in ζ — see §11.9.

#### IBZ resolution at runtime

`v_q_g_flat.py:_resolve_ibz_q_list` (line 148-199). Inputs:
`sym = wfn.symmetry` table; `centroid_indices = r_mu_fft_idx`. Steps:

1. Try `centroid.orbit_syms.compute_centroid_sym_perm(centroid_indices,
   sym_matrices, translations, fft_grid)` — validates that every sym op
   permutes the centroid set; raises if not orbit-closed.
2. If success: call `sym.find_irreducible_qpoints()` to get
   `(q_irr_kgrid_int, full_to_irr_idx, full_to_irr_sym)` and set
   `use_ibz = True`.
3. If failure (orbit-closure violation, e.g. kmeans was run with
   `--no-orbit`): fall back to `q_irr = full BZ`, `use_ibz = False`,
   `sym_perm = None`. The post-V_q unfold becomes a no-op.

The IBZ subset is `n_q_disk` rows on the on-disk q-axis of
`zeta_q_G.h5`. Disk size shrinks by `n_q_full / n_qpt_irr` (e.g. 9× for
CrI3 6×6×1 with `ntran=12`; 21× total combined with the r → G-sphere
saving — see `gflat_e2e_bispinor_mos2_3x3_2026-05-11/report.md`).

### 11.7 V_q kernel: per-q, G-chunked, with optional async prefetch

#### Inner kernel

`gw/v_q_g_flat.py::_make_per_q_kernel` (line 55-140). For one IBZ q:

$$\boxed{V_q^{\mu_L \nu_L}[\mu_L^{(c)}, \nu_R^{(c)}] = \sum_{G \in \mathrm{sphere}(q)}\,\overline{\tilde\zeta_L^{\mu_L}(q, \mu_L^{(c)}, G)}\,v(q + G)\,t^{\mu_L \nu_L}(q + G)\,\tilde\zeta_R^{\nu_L}(q, \nu_R^{(c)}, G)}$$

where `μ_L^{(c)}`, `ν_R^{(c)}` are the centroid indices (left and right
side of the bilinear, distinct in the bispinor off-diagonal case where
the two ζ-files have different centroid counts). `v(q+G)` is the
scalar Coulomb (3-D: `4π/|q+G|²`; 2-D slab / 0-D box: dimension-aware
truncation factor); `t^{μ_L ν_L}(q+G)` is the bispinor weight (§11.8).

The kernel reads one (or two distinct) ζ-slabs per q, accumulates the
GEMM in G-chunks of size `g_chunk` into a `(n_rmu_L, n_rmu_R)` block,
and `dynamic_update_slice`s the result into the persistent
`(n_q_ibz, μ, μ)` V-accumulator. Pseudocode (`v_q_g_flat.py:89-137`):

```python
@jax.jit(donate_argnums=(0, 1))
def fn(V_acc, g0_acc, zeta_L_q, zeta_R_q, v_q, q_idx):
    zeta_L_3d = with_sharding_constraint(zeta_L_q, P(('x','y'), None))   # disk read → q-flat
    zeta_R_3d = zeta_L_3d if same_zeta else with_sharding_constraint(zeta_R_q, P(('x','y'), None))
    zeta_L    = with_sharding_constraint(zeta_L_3d[0], P('x', None))     # μ_X
    zeta_R    = with_sharding_constraint(zeta_R_3d[0], P('y', None))     # μ_Y

    V_q = jnp.zeros((n_rmu_L, n_rmu_R), dtype=c128); V_q = wsc(V_q, P('x','y'))
    for i in range(n_chunks):                                            # ngkmax // g_chunk
        L_chunk = dynamic_slice(zeta_L, i·g_chunk, g_chunk, axis=-1)     # (μ_L/p_x, g_chunk)
        R_chunk = dynamic_slice(zeta_R, i·g_chunk, g_chunk, axis=-1)     # (μ_R/p_y, g_chunk)
        v_chunk = dynamic_slice(v_q,    i·g_chunk, g_chunk, axis=0)      # (g_chunk,) replicated
        L_w     = conj(L_chunk) * v_chunk[None, :]
        V_q     = V_q + L_w @ R_chunk.T

    V_new = dynamic_update_slice(V_acc, V_q[None, :, :], (q_idx, 0, 0))
    if write_g0:
        g0_q = zeta_L[:, 0]                                              # ζ̃[μ, G=0] from sphere convention
        g0_new = dynamic_update_slice(g0_acc, g0_q[None, :], (q_idx, 0))
```

#### Shardings

| Array | Shape | Spec | Notes |
|---|---|---|---|
| `zeta_L_q` (post-read) | `(1, n_rmu_L, ngkmax)` | `P(None, ('x','y'), None)` | disk → device, μ-flat |
| `zeta_L` (post-reshard) | `(n_rmu_L, ngkmax)` | `P('x', None)` | left leg of einsum |
| `zeta_R` (post-reshard) | `(n_rmu_R, ngkmax)` | `P('y', None)` | right leg |
| `v_q` per-q row | `(ngkmax,)` | `P(None)` | replicated per rank |
| `V_q` (per-q tile) | `(n_rmu_L, n_rmu_R)` | `P('x', 'y')` | μ × ν tile |
| `V_acc` | `(n_q_ibz, n_rmu_L, n_rmu_R)` | `P(None, 'x', 'y')` | persistent, donated |
| `g0_acc` (CC only) | `(n_q_ibz, n_rmu_L)` | `P(None, 'x')` | persistent, donated |

The two `with_sharding_constraint` calls on `zeta_L_q`, `zeta_R_q` move
each ζ-slab from the disk-read sharding to the einsum-natural single-axis
sharding. XLA emits an "Involuntary full rematerialization" warning on
this reshard (`gflat_e2e_bispinor_mos2_3x3_2026-05-11 §"XLA SPMD"`); the
cost is `ngkmax · n_rmu · 16 B` per q, dwarfed by the kernel.

#### G-chunk parameter

`g_chunk` (G-axis chunk size inside the inner kernel):
* Default: `_pick_g_chunk(ngkmax, target=4096)` = largest divisor of
  `ngkmax` that is ≤ 4096 (`v_q_g_flat.py:202-207`).
* Constraint: `ngkmax % g_chunk == 0` (`:325`).
* `n_chunks = ngkmax / g_chunk` is small (1 for MoS2 3×3 `ngkmax=1963`;
  ~14 for CrI3 6×6 80 Ry `ngkmax ≈ 55k`).

#### Async prefetch

Documented but **opt-out** in production: `LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH=1`
enables a worker thread that reads `ζ̃_{q+1}` while compute thread
contracts ζ̃_q. Deadlocks against the NCCL collective in the kernel under
heavy mesh contention (see CHANGELOG 2026-05-11 "G-flat shakedown"). The
sync per-q loop is already ~6× faster than the legacy μ × ν tile driver,
so the prefetch optimization is shelved.

### 11.8 V_q^{μ_L,ν_L} sectorization (Lorentz)

#### Bare 4×4 photon propagator in Coulomb gauge

`v_q_bispinor.py:9-14`. Coulomb gauge makes the bare-photon propagator
**block-diagonal** in Lorentz indices:

$$D^{\mu_L \nu_L}(K) = \begin{pmatrix} v(K) & 0 \\ 0 & v(K)\,t^{ij}(K) \end{pmatrix},\qquad t^{ij}(K) = \delta^{ij} - \hat K_i \hat K_j,\;\hat K = K / |K|.$$

so out of the 16 = 4×4 blocks, 6 vanish identically by gauge.

#### Block-by-block weight `t^{μ_L,ν_L}`

`v_q_bispinor.py:127-174` (`_make_v_per_G_for_tile`):

| Count | Sector | `same_zeta` | `t^{μ_L,ν_L}(K)` | Dataset |
|---:|---|---|---|---|
| 6 | (0,i), (i,0)    | — (not computed) | 0 | gauge-zero |
| 1 | (0,0) CC        | True (n_rmu_C, n_rmu_C) | 1 (BGW v(q+G) overlay applies) | `V_qmunu_CC` |
| 3 | (i,i) TT-diag   | True (n_rmu_T, n_rmu_T) | `1 − K̂_i²` | `V_qmunu_TT_ii` |
| 3 | (i<j) TT-off    | False (n_rmu_T, n_rmu_T; same centroids, distinct ζ files) | `−K̂_iK̂_j` | `V_qmunu_TT_ij` |
| 3 | (i>j) TT-Herm   | — (read as `conj(swap(V[i<j], μ,ν))`) | (Hermitian-redundant) | not stored |

`UNIQUE_TILES` (`v_q_bispinor.py:57-61`) enumerates the 7 unique kernel
calls; `HERMITIAN_PAIRS` (`:70-74`) maps `(j,i) → (i,j)` for the reader.
The CC tile is the only one that materialises a `g0_acc` head term;
transverse heads are killed by the projector at q=0, G=0 (axial limit).

#### Per-tile `v_per_G_fn` closure

`v_q_bispinor.py:161-172`:

```python
def v_per_G_fn(qvec_np_batch):
    qvec_arr = jnp.asarray(qvec_np_batch, dtype=jnp.float64)
    v        = base_v_per_G_fn(qvec_arr)                  # (Q, n_G_sph) c128
    K_cart   = K_cart_batch_fn(qvec_arr)                  # (Q, n_G_sph, 3) f64
    K2       = jnp.sum(K_cart * K_cart, axis=-1)
    K2_safe  = jnp.where(K2 > eps_K2, K2, 1.0)            # guard q=0,G=0
    Khat_ij  = K_cart[..., i] * K_cart[..., j] / K2_safe
    t        = (1.0 - Khat_ij) if i == j else (-Khat_ij)
    return v * t.astype(v.dtype)
```

Each tile bakes in its `(i, j)` indices at closure construction; the
inner V_q kernel is `μ_L,ν_L`-agnostic and consumes `v_per_G_fn(q)` as
an opaque `(Q, n_G_sph)` weight table.

#### Identity reductions

* **μ_L = ν_L = 0** (charge / CC tile): `t^{0,0} = 1`, the kernel is
  byte-identical to the scalar charge-only V_q. The `gamma_L = gamma_R
  = None` short-circuit in `c_q_from_pair`/`z_q_from_pair` means the
  ζ-fit reduces to the historical spin-traced path (modulo the 16×
  rank-5 carrier in the pair-density accumulator).
* **α_FS → 0**: `ψ_S → 0`, so every γ̃^i contraction (off-block-diagonal
  in L/S) vanishes. Σ^B → 0. The TT V_q^{i,j} tiles are nonzero but
  multiply zero pair-density on the Σ side.

### 11.9 IBZ → full BZ V_q unfold (post-loop)

#### Bilinearity in ζ ⇒ no τ-phase

Apply eq. 1 to both ζ-legs in the V_q definition (§11.7):

```
V_{Sq, π_s(μ), π_s(ν)} = Σ_{G_new} ζ̃*_{Sq, π_s(μ)}(G_new) v(Sq+G_new) t(Sq+G_new) ζ̃_{Sq, π_s(ν)}(G_new)
                       = Σ_{G_new} [e^{-i(Sq+G_new)·τ} ζ̃_{q,μ}(S⁻¹G_new)]*
                                    · v(Sq+G_new) · t(Sq+G_new)
                                    · [e^{-i(Sq+G_new)·τ} ζ̃_{q,ν}(S⁻¹G_new)]
                       = Σ_{G'}  ζ̃*_{q,μ}(G') · v(q+G') · t̃(q+G') · ζ̃_{q,ν}(G')        (rename G' = S⁻¹G_new)
```

τ-phases cancel `(+i)(−i)` exactly (V is bilinear in ζ). `v(q+G)` is
rotation-invariant (depends only on `|K|²`); the survival of the
transverse weight `t` under the change of variable depends on which
tile we're in — see §11.9.2 vs §11.9.3.

#### Scalar / CC and TT-diagonal: centroid double-permute only

For these tiles the kernel weight `t(K)` is **rotation-invariant** in
the scalar/CC sense:
* CC: `t ≡ 1`.
* TT-diagonal (i,i): `t(K) = 1 − K̂_i²` is **not** invariant under the
  full point group when (i,i) is held fixed (a rotation about an axis
  other than ı̂ mixes ı̂ with ĵ, k̂). The unfold for TT-diagonal is
  therefore not the simple "pure centroid permute" case — it is a
  **special case** of the off-diagonal unfold (§11.9.3) with i = j on
  the rotated indices.

For the scalar / CC tile (where `t ≡ 1`), eq. 3 holds without further
restriction:

$$\boxed{V_{q_{\mathrm{full}},\,\mu',\,\nu'} = V_{q_{\mathrm{irr}}[i(q_{\mathrm{full}})],\,\pi_{s(q_{\mathrm{full}})^{-1}}(\mu'),\,\pi_{s(q_{\mathrm{full}})^{-1}}(\nu')}}\qquad\text{(eq. 3)}$$

Implementation: `v_q_tile.py::_unfold_v_q_ibz_to_full` (lines 1454-1559).
Two `take_along_axis`'s on the (μ, ν) axes, `mode='promise_in_bounds'`
(skips XLA's OOB bounds-check `select` which trips a `s32/s64` HLO
verifier failure on shard_map+x64 — see commit 49b7f84 history).

#### TT off-diagonal: Cartesian rotation on the (i,j) indices

`v_q_tile.py::_unfold_v_q_ij_ibz_to_full` (lines 1562-…). Under sym
op `R(S)` (the Cartesian rotation, **no** τ part — τ-phase already
cancelled by bilinearity), the transverse projector transforms as a
rank-2 Cartesian tensor:

$$t^{ij}(R K) = R^{ia}(S)\,R^{jb}(S)\,t^{ab}(K).$$

The bilinear V_q^{ij} therefore unfolds as

$$\boxed{V_{Sq}^{ij}\!\big(\pi_S\mu,\,\pi_S\nu\big) = \sum_{a,b} R^{ia}(S)\,R^{jb}(S)\,V_q^{ab}(\mu,\,\nu)}\qquad\text{(eq. 4)}$$

i.e. *Cartesian double-rotate* on the (i,j) outer indices **plus**
centroid double-permute on (μ, ν). For TT-diagonal (i = j held fixed
under R) the result is in general a **mixture** of the diagonal and
off-diagonal IBZ tiles via `R^{ia} R^{ib} V^{ab}`; this is why the
diagonal cannot in general be unfolded with the pure-permute identity
of §11.9.2 unless the system has axis-aligned symmetries (point group
generated by `σ_h ⊕ σ_v` etc.).

**Implementation note**: in `gw.v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5`
the current code uses `_unfold_v_q_ibz_to_full` (the scalar / pure-permute
unfold) for **every** tile including TT-diag and TT-off. This is
correct only when the sym ops are axis-permuting on Cartesian
coordinates (e.g. C_4 about ẑ for MoS2 3×3×1 — which the matched
runs do satisfy by ntran-2 mirror only). For general symmetries
(C_3v in MoS2, C_3 in CrI3) the polarization-mixing path
`_unfold_v_q_ij_ibz_to_full` should be wired in — see §11.11 followups.

#### R_cart materialisation

`R_cart = sym.sym_matrices_cart` is built from the crystal-frame integer
rotation `S = sym_matrices` by `R_cart = bvec.T @ S @ inv(bvec.T)` (or
equivalently `R_cart = lattice^{-1} S lattice`, with the convention
`R_cart`'s rows act on r-space cartesian column vectors). For
non-orthogonal lattices (e.g. CrI3 hexagonal) `R_cart` is a true 3×3
orthogonal matrix, not a permutation — this is the regime where
§11.9.3's Cartesian mixing matters.

`R_cart[s][:, :2]` for s = identity is `I_2`; for the C_3 generator
about ẑ it is the 2-D rotation by 120° on (x, y) — fully entangling
V_q^{11}, V_q^{12}, V_q^{22} on the unfold.

### 11.10 Bispinor Σ glue (γ̃-fold into ψ)

Out of scope for this section; see
`reports/bispinor_theory_2026-05-09 §7` for the full Σ^B derivation
and the γ̃-fold-into-ψ Hermitian identity that lets the unmodified
scalar `sigma_sx_k(wfns_ij, G, V^{ij})` evaluate the bispinor bare-Breit
matrix element. Brief recap:

* **Σ^X^total** = Σ^X[CC] + Σ^B[Breit-9-tiles]:
  * Σ^X[CC] reuses `cohsex_sigma.compute_cohsex_sigma(...,
    compute_bare_x=True)` with V_q = V_qmunu_CC and the un-rewritten
    bispinor wavefunctions (`cohsex_sigma.py:229-240`).
  * Σ^B loops the 9 transverse `(i, j)` tiles
    (`sigma_x_bispinor.py:189-205`); each iteration folds
    `γ̃^i ψ_xn` and `γ̃^j ψ_yr` on the spin axis
    (`_apply_gamma_left_to_xn`, `_apply_gamma_left_to_yr` at lines
    62-88), reads V^{i,j}_q (with Hermitian-fill for i > j), and calls
    the unmodified `sigma_sx_k(wfns_ij, Gij, V_ij)`.

* **Identity-vertex regression**: replacing the (i,j) loop with a
  single (0,0) call and γ̃=I_4 gives Σ^B byte-identical to the scalar
  Σ_X (`BISPINOR_DHFB_DESIGN.md §7.1`).

* **α_FS² scaling**: each (γ̃^i ψ_xn)_β is O(α_FS) entrywise (γ̃^i is
  L↔S off-block-diagonal, ψ_S = O(α_FS) ψ_L), so Σ^B is O(α_FS²) — for
  MoS2 valence (Σ_X[CC] ≈ −37 eV) the expected Σ^B per band is
  ~−2 meV. CrI3 (heavy elements: Zα for Cr ≈ 0.18) scales to
  ~−10 to −50 meV per band. See
  `reports/bispinor_theory_2026-05-09 §10` for the current numerical
  regression status.

### 11.11 Open questions / followups

1. **TT-diagonal and TT-offdiagonal unfold**: the current
   `compute_V_q_bispinor_g_flat_to_h5` uses the scalar/pure-permute
   `_unfold_v_q_ibz_to_full` for every tile. For general point groups
   (C_3 in CrI3) the off-diagonal unfold needs
   `_unfold_v_q_ij_ibz_to_full` (eq. 4 in §11.9.3) — currently
   implemented but not wired into the bispinor orchestrator. The
   TT-diagonal case is a special case of the same Cartesian-rotation
   unfold and likewise needs to thread through.

2. **Open-spin C_q^{i,i} definiteness**: §11.2 argues the open-spin Gram
   is PSD by construction; the code dispatches all μ_L≠0 channels to
   pivoted LU + ridge `1e-12 |tr|/n` "to be safe". A deterministic
   check (eig(C_q^{1,1}) on MoS2 3×3) would settle whether the LU
   branch is overcautious or whether there's a still-subtle indefinite
   path. See `reports/bispinor_theory_2026-05-09 §4.5 / §12.1`.

3. **CrI3 transverse residual blowup**: post LU+ridge, CrI3 Σ^B is
   still ~10⁵× larger than the α_FS²·Σ_X[CC] expectation; the
   2026-05-06 audit narrowed the cause to current-centroid ISDF basis
   conditioning. Untested fixes: (a) rebuild current centroids from a
   joint ρ_charge + W_curr weighting, (b) use the same centroid set
   for all 4 channels.
   `reports/bispinor_theory_2026-05-09 §10.4 / §12.3`.

4. **Disk-read sharding**: the bispinor V_q kernel emits 8 "Involuntary
   full rematerialization" SPMD warnings per tile-compile when the
   disk-read shape (`devices=[P,1,1]`) lands at the kernel's preferred
   `P(('x','y'), None)`. Functionally correct; performance loss is
   ~20 MB / q / rank. Followup: have `ZetaReader.read_zeta_G_slab`
   expose a `P(('x','y'), None)`-direct read variant.

5. **G-flat accumulator under `band_chunk_size = 4` (CrI3 6×6 80 Ry)**:
   the per-rank `n_rtot · 16 B` FFT-box transient is independent of
   r_chunk, but XLA materialises the band-chunk FFT box unsharded in
   `to_rmu` (`load_wfns.py:657` "16× safety margin"). Real fix is a
   `with_sharding_constraint(box, P(None, ('x','y'), None, None, None, None))`
   inside the per-bc FFT path; not yet wired. Documented in
   `reports/zeta_v_q_g_flat_reference_2026-05-12 §10` (CrI3 validation
   log).

### 11.12 File pointers

#### Source

| File | Symbol | What |
|---|---|---|
| `src/common/isdf_fitting.py:82-148` | `pair_density`, `accum_pair_density` | rank-5 open-spin P_k,αβ(μ, col) |
| `src/common/isdf_fitting.py:168-279` | `c_q_from_pair` | CCT with optional γ̃ insertions |
| `src/common/isdf_fitting.py:373-…`   | `z_q_from_pair` | ZCT on (μ, r-chunk) with optional γ̃ |
| `src/common/isdf_fitting.py:282-370` | `gram_q0_from_pair` | q=0 valence-conduction Gram for centroid selection |
| `src/common/isdf_fitting.py:921-1103`| `compute_L_q_from_CCT` | Cholesky (μ_L=0) / passthrough (μ_L≠0) |
| `src/common/isdf_fitting.py:1110-1230` | `solve_zeta_from_L_q` | per-q triangular solve (Cholesky) / `jnp.linalg.solve` (LU) |
| `src/common/isdf_fitting.py:1480-…`  | `fit_zeta_to_h5` | top-level driver, r-chunk loop, G-flat accumulator |
| `src/common/wfn_transforms.py:439-626` | `accumulate_rchunk_to_gflat` | r-chunk → G-sphere FFT-and-accumulate |
| `src/common/wfn_transforms.py:649-686` | `apply_bloch_phase` | separable `exp(±2πi k·r)` |
| `src/common/wfn_transforms.py:689-…`   | `apply_bloch_phase_on_slice` | same on a flat-r slab |
| `src/common/gamma_matrices.py:18-87` | γ̃^μ tables, `gammas_perm`, `gammas_phase` |
| `src/common/gamma_matrices.py:90-99` | `gamma_perm_phase(μ_L)` | (perm, phase) accessor |
| `src/common/gamma_matrices.py:128-164` | `gamma_double_contract` | γ̃·γ̃ rank-5 → rank-3 reduction |
| `src/common/bispinor_init.py:12-41`  | `get_small_psi_component` | ψ_S kinetic-balance lift |
| `src/common/coulomb_sphere.py:120-247` | `compute_per_q_bare_coulomb_components` | per-q `(q+G)` sphere + sentinel pad |
| `src/gw/v_q_g_flat.py:55-140`        | `_make_per_q_kernel` | G-chunked per-q V_q GEMM |
| `src/gw/v_q_g_flat.py:148-199`       | `_resolve_ibz_q_list` | IBZ list + centroid orbit closure |
| `src/gw/v_q_g_flat.py:247-408`       | `_compute_V_q_g_flat_one_tile` | per-tile end-to-end (read + kernel loop + unfold) |
| `src/gw/v_q_g_flat.py:415-…`         | `compute_all_V_q_g_flat` | charge-channel public entry point |
| `src/gw/v_q_bispinor.py:57-74`       | `UNIQUE_TILES`, `ZERO_TILES`, `HERMITIAN_PAIRS` |
| `src/gw/v_q_bispinor.py:127-174`     | `_make_v_per_G_for_tile` | per-tile `v(K)·t^{i,j}(K)` closure |
| `src/gw/v_q_bispinor.py:482-…`       | `compute_V_q_bispinor_g_flat_to_h5` | bispinor orchestrator |
| `src/gw/v_q_bispinor.py:660-…`       | `BispinorVqReader` | tile reader (gauge-zero, Hermitian-redundant) |
| `src/gw/v_q_tile.py:1454-1559`       | `_unfold_v_q_ibz_to_full` | scalar / pure centroid double-permute |
| `src/gw/v_q_tile.py:1562-…`          | `_unfold_v_q_ij_ibz_to_full` | TT off-diagonal: + Cartesian R_iaR_jb mixing |
| `src/gw/v_q_tile.py:1671-…`          | `_unfold_g0_ibz_to_full` | g0 head unfold |
| `src/file_io/zeta_reader.py`         | `ZetaReader` | G-flat on-disk reader, per-q slab |
| `src/file_io/slab_io.py`             | `SlabIO` | phdf5 writer with `valid_shape=` μ-pad clip |
| `src/gw/sigma_x_bispinor.py:62-111`  | `_apply_gamma_left_to_xn`, `_apply_gamma_left_to_yr` | γ̃-fold into ψ |
| `src/gw/sigma_x_bispinor.py:114-210` | `compute_sigma_x_bispinor` | 9-tile Σ^B orchestrator |

#### Cross-refs

| Doc | Focus |
|---|---|
| §3–5 above | Historical r-space ζ-on-disk + spin-traced rank-3 narrative |
| §7 above | Sharding map (scalar / static-Σ shardings) |
| [`MEMORY_MODEL.md`](MEMORY_MODEL.md) | Per-stage memory formulas (current memory model is stale w.r.t. G-flat refactor) |
| `reports/zeta_v_q_g_flat_reference_2026-05-12/report.md` | Living engineering reference (donations, shardings, chunker envs, CrI3 validation log) |
| `reports/bispinor_theory_2026-05-09/report.md` | Bispinor canonical math reference (conventions, Σ^B, definiteness analysis, numerical regime) |
| `reports/zeta_ibz_2026-05-11/report.md` | IBZ-only ζ-on-disk schema design + symmetry derivations (eq. 1 + eq. 3 of §11.6 / §11.9) |
| `reports/v_q_bispinor_plan_2026-05-08/report.md` | Lorentz tile sectorization + V_q_bispinor container layout |
| `reports/bispinor_pipeline_2026-05-04/report.md` | MoS2 / CrI3 reference traces; LU branch milestone |
| `reports/gflat_e2e_bispinor_mos2_3x3_2026-05-11/report.md` | First G-flat end-to-end bispinor; 21× disk-size win |

### 11.13 Quick reference (bispinor / G-flat)

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
| V_q scalar unfold (eq. 3) | `V_full[q,μ',ν'] = V_irr[i(q), π_{s(q)}⁻¹(μ'), π_{s(q)}⁻¹(ν')]` |
| V_q TT-off unfold (eq. 4) | `V_{Sq}^{ij}(π_S μ, π_S ν) = Σ_{ab} R^{ia}(S) R^{jb}(S) V_q^{ab}(μ,ν)` |

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
