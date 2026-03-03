# Interpolative Separable Density Fitting: Theory and Implementation

**Consolidates**: `formalism.md`, `isdf_context.md`, `isdf_spin_galerkin_derivation.md`, `ZETA_FITTING_ALGORITHM.md`, `cohsex_jax_physics.md`

**Status**: Describes current implementation in `src/isdf/common/load_wfns.py`, `src/gw_isdf/gw_jax.py`, `src/gw_isdf/w_isdf.py`.

---

## Overview

ISDF reduces GW computational cost by approximating pair-product densities with separable interpolation vectors. For systems with $n_k$ k-points, $n_b$ bands, and $n_r$ real-space grid points, storing the full pair-product tensor $M_{mn}(\mathbf{k}, \mathbf{q}, \mathbf{r})$ requires $O(n_k \times n_b^2 \times n_r)$ ~ TB of memory. ISDF reduces this to $O(n_\mu \times n_r)$ where $n_\mu \approx 10 \times n_b$ interpolation points, making real-space GW tractable for small systems.

---

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
- Generation: `src/isdf/isdf_init/kmeans_isdf.py`
- Storage: `centroids_frac.h5` (fractional coordinates)
- Loading: `src/isdf/io/centroids.py`

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

**Implementation**: Compute $P_{\mathbf{k}}$ on k-grid, FFT to R-grid via `jnp.fft.ifftn(..., norm='ortho')`, form spin-traced products, FFT back to q-space. See `compute_CCT_from_left_right()` and `compute_ZCT_from_left_right()` in `load_wfns.py`.

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

**Implementation**: `compute_CCT_from_left_right()` in `load_wfns.py:942`.

### 4.4 Stage 3: Cholesky Factorization

Compute $C_q = L_q L_q^\dagger$ for each $\mathbf{q}$ using 2D blocked algorithm:

$$L_q(\mu_X, \nu_Y) = \text{cholesky\_2d\_batched}(C_q)$$

**Algorithm**: Operates on tiles without gathering full matrix. Panel broadcasts and triangular updates use `lax.psum` for column communication.

**Complexity**:
- Compute: $O(n_\mu^3 / P)$
- Communication: $O(n_\mu^2 / \sqrt{P})$

**Sharding**: $L_q(\mu_X, \nu_Y)$ same 2D tiles as $C_q$.

**Implementation**: `cholesky_2d_batched()` in `cholesky_2d.py`.

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

**Step 4c**: Triangular solve (per-q gather to avoid full replication):

```python
for q in range(n_q):
    L_rep = all_gather(L_q[q])              # (n_μ, n_μ) replicated on all devices
    y = triangular_solve(L_rep, Z_q[q])     # Forward: L y = Z
    ζ_q[q] = triangular_solve(L_rep.T.conj(), y)  # Backward: L† ζ = y
```

Each device solves for its column shard: $\zeta_q[q](\mu, r_{XY})$.

**Step 4d**: Write to HDF5 immediately (per-q to avoid host OOM):

```python
for q in range(n_q):
    ζ_host = np.asarray(ζ_q[q])  # gather (n_μ, n_r_chunk) to host
    f['zeta_q'][qx, qy, qz, :, r_start:r_end] = ζ_host
```

**Sharding**:
- $Z_q(\mu_X, r_{XY})$: columns distributed across all $P$ devices
- $\zeta_q(\mu, r_{XY})$: same column sharding

**Implementation**: `fit_zeta_chunked_to_h5()` in `load_wfns.py:1720`.

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

**Memory**: $\zeta$ is loaded $\mu$-chunked (typically $\mu_{\text{chunk}} \approx M_{\text{budget}} / (3 \times 16 \times n_r)$ to hold r-space, G-space, and $\nu$-block).

**Output**: `V_qmunu.h5` with shape $(n_q, n_\mu, n_\mu)$, used as input to GW pipeline.

**Disk bottleneck**: The file `zeta_q.h5` with shape $(n_{qx}, n_{qy}, n_{qz}, n_\mu, n_r)$ is often **10-100 GB** and represents a significant disk storage constraint. For systems with $n_q = 128$, $n_\mu = 40000$, $n_r = 10^6$, this is $\sim$80 GB in complex128 format.

**Implementation**: `compute_all_V_q_from_zeta_h5()` in `compute_vcoul.py`.

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

**Automatic sizing**: Function `compute_optimal_chunks()` in `gw_init.py` solves this constraint system analytically, iteratively reducing chunk sizes until all stages fit.

**See**: `docs/MEMORY_MODEL.md` and `docs/CHUNK_BUDGETS.md` for detailed formulas.

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

**Implementation**: Universal chi kernel with energy masking in `_get_chi_kernel()` (`w_isdf.py:60`). Single JIT compilation, window pairs selected via boolean masks.

**Reference**: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020). Full derivation in `docs/chi_omega_quadrature.md`.

### 6.4 Screened Interaction (Dyson Equation)

Solve for $W$ in the ISDF basis:

$$(1 - V\chi^0) W = V$$

**Algorithm** (direct solve, no whitening):
1. LU factorization: $(1 - V\chi^0) = L U$
2. Solve: $W = (LU)^{-1} V$

**Note**: The whitening step (orthogonalizing via overlap matrix $S = \langle \zeta_\mu | \zeta_\nu \rangle$) is **not used** in the current implementation. We solve the Dyson equation directly in the original ISDF basis.

**Sharding**: During solve, reshard from $V_q(\mu_X, \nu_Y)$ to $V_q(q_{XY}, \mu, \nu)$ for per-q LU (q-point parallelism).

**Implementation**: `get_static_w_q_jax()` in `w_isdf.py:240`.

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

**Implementation**: Head added in `gw_jax.py:1948`, dipole $S_{\alpha\beta}$ computed in `chi_from_dipole.py`.

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

**Output**: Written to `sigma.h5` and `eqp.dat` (quasiparticle energies).

**Implementation**: `get_sigma_x_kij_jax()` in `gw_jax.py:804`.

### 6.8 Self-Consistency Loop

The quasiparticle (QP) Hamiltonian is:

$$H_{\text{QP}} = H_{\text{KS}} - V_{xc} + \Sigma(\omega \approx E_n)$$

where $H_{\text{KS}} = K + I + V_H + V_{xc}$ is the Kohn-Sham DFT Hamiltonian.

**Self-consistent GW** iteratively updates $\Sigma$ until wavefunctions and energies converge:

1. Start with DFT $\psi_n^{(0)}, E_n^{(0)}$
2. Compute $\Sigma^{(i)}[\psi^{(i)}]$ using current wavefunctions
3. Diagonalize $H_{\text{QP}}^{(i)} = H_{\text{KS}} - V_{xc} + \Sigma^{(i)}$ → new $\psi_n^{(i+1)}, E_n^{(i+1)}$
4. Repeat until $|\!|E^{(i+1)} - E^{(i)}|\!| < \epsilon$

**Current status**: A fixed-point iteration prototype exists in `gw_jax.py` using Anderson mixing from `mixing/acceleration.py`, but is **not yet validated**. The code currently performs **one-shot GW** (G₀W₀): compute $\Sigma$ once from DFT wavefunctions without iterating.

**Implementation**: `gw_jax.py:1230` (prototype, disabled by default).

---

## 7. JAX Sharding Summary

### 7.1 Zeta Fitting Pipeline

| Array | Sharding | Size | Notes |
|-------|----------|------|-------|
| $\psi_{\text{rmu}}$ | $(k, n, s, \mu_Y)$ | $n_k \times n_b \times 2 \times n_\mu$ | Centroids on Y |
| $\psi_{\text{rmuT}}$ | $(k, \mu_X, n, s)$ | Same | Transposed for matmul |
| $P_{\mathbf{k}}$ | $(k, s, \mu_X, s, \mu_Y)$ | $n_k \times 4 \times n_\mu^2$ | Pair density (2D tiles) |
| $C_q$ | $(q, \mu_X, \nu_Y)$ | $n_q \times n_\mu^2$ | CCT matrix |
| $L_q$ | $(q, \mu_X, \nu_Y)$ | Same | Cholesky factor |
| $Z_q$ | $(q, \mu_X, r_{XY})$ | $n_q \times n_\mu \times n_r$ | ZCT matrix |
| $\zeta_q$ | $(q, \mu, r_{XY})$ | Same | Interpolation vectors |

### 7.2 GW Pipeline

| Array | Sharding | Size | Notes |
|-------|----------|------|-------|
| $G_{\mathbf{k}}$ | $(k, s_a, \mu_X, s_b, \nu_Y)$ | $n_k \times 4 \times n_\mu^2$ | Green's function |
| $\chi^0_q$ | $(q, \mu_X, \nu_Y)$ | $n_q \times n_\mu^2$ | Polarizability |
| $V_q$ | $(q, \mu_X, \nu_Y)$ | Same | Bare Coulomb |
| $W_q$ (solve) | $(q_{XY}, \mu, \nu)$ | Same | Resharded for per-q LU |
| $W_q$ (final) | $(q, \mu_X, \nu_Y)$ | Same | Resharded back |
| $\Sigma_{\mathbf{k}}$ | $(k, s_a, \mu_X, s_b, \nu_Y)$ | $n_k \times 4 \times n_\mu^2$ | Self-energy |
| $\Sigma_{ij,\mathbf{k}}$ | $(k, i, j)$ | $n_k \times n_b^2$ | Band basis |

**Notation**: Subscripts $X, Y, XY$ denote sharding axes on 2D mesh `Mesh(devices, ('x', 'y'))`.

---

## 8. File Organization

### Core Implementation

| File | Lines | Purpose |
|------|-------|---------|
| `load_wfns.py` | 1900 | Zeta pipeline: CCT/ZCT, Cholesky, chunking |
| `gw_jax.py` | 1400 | Main driver: wfn loading, $\Sigma$ calculation |
| `w_isdf.py` | 350 | $\chi^0$ and $W$ via CTSP, Dyson solve |
| `cholesky_2d.py` | 600 | 2D blocked Cholesky for sharded CCT |
| `compute_vcoul.py` | 800 | $V_q$ from zeta HDF5 |
| `gw_init.py` | 650 | Input parsing, automatic chunk sizing |
| `meta.py` | 200 | System metadata (k/q-grids, cell) |

### Documentation

| Doc | Focus |
|-----|-------|
| **This file** | Theory + implementation |
| `MEMORY_MODEL.md` | Detailed memory formulas |
| `CHUNK_BUDGETS.md` | Quick reference constraints |
| `chi_omega_quadrature.md` | CTSP theory, quadrature derivations |

---

## 9. Typical Workflow

### Preparation

1. DFT wavefunctions: `pw2bgw.x` → `WFN.h5`, `WFNq.h5`
2. Centroid selection: `uv run kmeans_isdf -i kmeans.in` → `centroids_frac.h5`
3. Input file: `cohsex.in` with band ranges, memory budget

### Zeta Fitting

```bash
uv run python -m gw_isdf.gw_jax -i cohsex.in --fit-zeta-only
```

**Produces**:
- `zeta_q.h5`: $(n_{qx}, n_{qy}, n_{qz}, n_\mu, n_r)$
- `V_qmunu.h5`: $(n_q, n_\mu, n_\mu)$

### GW Calculation

```bash
uv run python -m gw_isdf.gw_jax -i cohsex.in
```

**Uses**: Cached `zeta_q.h5`, `V_qmunu.h5`

**Produces**: `sigma.h5`, `eqp.dat` (quasiparticle energies)

---

## 10. Known Issues

1. **Dyson solve**: LU for $(1-V\chi^0)^{-1}$ not distributed over $\mu$ (communication bottleneck)
2. **Self-consistency**: Fixed-point prototype exists but not validated

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
