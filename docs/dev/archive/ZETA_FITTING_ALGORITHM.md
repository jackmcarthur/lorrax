# Memory-Efficient ISDF Zeta Fitting Algorithm

This document describes the chunked wavefunction loading and zeta fitting pipeline implemented in `load_wfns.py` and tested in `test_chunked_wfn_loading.py`.

## Overview

The goal is to compute the ISDF interpolation vectors $\zeta_{q,\mu}(r)$ satisfying:

$$\sum_{ab} \psi^*_{m,k-q,a}(r)\psi_{n,k,b}(r) \approx \sum_\mu \zeta_{q,\mu}(r) \cdot \sum_{ab} \psi^*_{m,k-q,a}(r_\mu)\psi_{n,k,b}(r_\mu)$$

This requires solving a Galerkin problem for each $q$-point, which is memory-intensive for large systems.

---

## Algorithm Steps

The pipeline uses a 2D processor mesh $(P_x, P_y)$ with axes named `x` and `y`.

**Sharding notation:**
- $\psi_{n,k}(r_{\mu,Y})$ means the $\mu$ axis is sharded on mesh axis `y`
- $P(\cdot, \cdot, x, y)$ means axes 2,3 are sharded on `x`,`y` respectively

| Step | Operation | Equation | Sharding |
|------|-----------|----------|----------|
| **1** | Load G-space (band-chunked) | $u_{n_{\text{bc}},k}(G) \leftarrow \text{HDF5}$ | bands on $(x,y)$ |
| **2** | FFT to real-space | $u_{n_{\text{bc}},k}(r) \leftarrow \text{IFFT}[u(G)]$ | per-device `shard_map` |
| **3** | Bloch phase | $\psi_{n,k}(r) \leftarrow e^{ik\cdot r} \, u_{n,k}(r)$ | same |
| **4a** | Extract centroids | $\psi_{n,k,a}(r_{\mu,Y})$ | $P(\cdot,\cdot,\cdot,y)$ |
| **4b** | Transpose copy | $\psi_{n,k,a}(r_{\mu,X})^T$ | $P(\cdot,x,\cdot,\cdot)$ |
| **5** | Extract z-chunk | $\psi_{n,k,a}(r_{\text{chunk},Y})$ | $P(\cdot,\cdot,\cdot,y)$ |
| **6** | Pair density | $P_{k,ab}(\mu_X, \nu_Y) \leftarrow \sum_n \psi^*_{n,k,a}(\mu) \cdot \psi_{n,k,b}(\nu)$ | $P(\cdot,\cdot,\cdot,x,y)$ |
| **7** | IFFT k→R | $P_{R,ab}(\mu_X, \nu_Y) \leftarrow \text{ortho-IFFT}_k[P_k]$ | same |
| **8** | Spin-squared | $C_R(\mu_X, \nu_Y) \leftarrow \sum_{ab} \|P_{R,ab}(\mu,\nu)\|^2$ | same |
| **9** | FFT R→q | $C_q(\mu_X, \nu_Y) \leftarrow \text{ortho-FFT}_R[C_R]$ | $P(\cdot,\cdot,\cdot,x,y)$ |
| **10** | 2D blocked Cholesky | $L_q(\mu_X, \nu_Y) \leftarrow \text{chol}(C_q)$ | 2D tiled on $(x,y)$ |
| **11** | ZCT for z-chunk | $Z_q(\mu_X, r_Y) \leftarrow \text{ortho-FFT}_R[\sum_{ab}\|P_{R,ab}(\mu,r)\|^2]$ | $P(\cdot,x,y)$ |
| **12** | Triangular solve | **for** $q$: $L_q^{\text{rep}} \leftarrow \text{allgather}(L_q[q])$; $\zeta_q[q](\mu, r_{XY}) \leftarrow (L_q^{\text{rep}})^{-H}((L_q^{\text{rep}})^{-1} Z_q[q])$ | $\zeta$: $P(\cdot,\cdot,(x,y))$ |
| **13** | Write to HDF5 | **for** $q$: gather $\zeta_q[q](\mu, r_{\text{chunk}})$ to host, write | per-$q$ gather |

---

## Detailed Equations

### Pair Density (Step 6)

$$P_{k,ab}(\mu, \nu) = \sum_{n \in \text{bands}} \psi^*_{n,k,a}(r_\mu) \cdot \psi_{n,k,b}(r_\nu)$$

where $a, b \in \{\uparrow, \downarrow\}$ are spin indices. This produces a $(N_k, 2, 2, N_\mu, N_\nu)$ tensor.

### Galerkin CCT Matrix (Steps 7-9)

The CCT matrix for fitting is formed by summing over all four spin combinations:

$$C_R(\mu, \nu) = \sum_{a,b \in \{\uparrow,\downarrow\}} \left| P_{R,ab}(\mu, \nu) \right|^2$$

This is derived from the Galerkin condition for spin-traced pair densities. See `docs/isdf_spin_galerkin_derivation.md` for the full derivation.

### ZCT Matrix (Step 11)

Similarly, for each z-chunk:

$$Z_q(\mu, r) = \text{ortho-FFT}_R \left[ \sum_{a,b} \left| P_{R,ab}(\mu, r) \right|^2 \right]$$

### Zeta Solution (Step 12)

The solve loops over $q$-points to limit memory:

```
L_q ← chol(C_q)   # 2D blocked, L_q(μ_X, ν_Y)

for q in 0..Nq:
    L_rep ← all_gather(L_q[q])   # replicate L for this q: (μ, ν)
    Z_cols ← Z_q[q]              # column-sharded: (μ, r_XY)
    y ← L_rep^{-1} Z_cols        # forward substitution (column-parallel)
    zeta_q[q] ← L_rep^{-H} y     # backward substitution (column-parallel)
```

Memory per solve: $O(N_\mu^2)$ for $L_{\text{rep}}$ plus $O(N_\mu \cdot N_{\text{zchunk}} / P)$ for local columns.

---

## Chunking Strategy

### Band Chunking (for FFT memory)

The FFT step (2-3) requires holding $\psi_{n,k}(r_{\text{tot}})$ which is size $(N_k \times N_b \times N_s \times N_r)$. For large systems this exceeds device memory.

**Solution:** Loop over band chunks $n_{\text{bc}}$:

```
psi_zchunk ← zeros(Nk, Nb_total, Ns, N_zchunk)  # output fits in memory

for bc in band_chunks:
    psi_G[bc] ← load_HDF5(bc)           # (Nk, B, Ns, Ng)
    psi_r[bc] ← FFT(psi_G[bc])          # (Nk, B, Ns, Nr) - TRANSIENT
    psi_zchunk[bc] ← slice_z(psi_r[bc]) # extract z-slice, store
    del psi_r[bc]                        # free immediately
```

Peak FFT memory: $O(N_k \cdot B \cdot N_s \cdot N_r)$ where $B \ll N_b$.

### Z-Chunking (for ZCT memory)

$P_k(\mu, r_{\text{tot}})$ is too large to hold. Instead, process z-slices:

```
L_q ← chol(C_q)   # computed once from P_k(μ, μ)

for z_chunk in z_chunks:
    psi_zchunk ← load_and_FFT(z_slice)   # band-chunked internally
    P_k ← pair_density(psi_rmuT, psi_zchunk)
    Z_q ← FFT_pipeline(P_k)
    zeta_chunk ← solve(L_q, Z_q)         # loop over q internally
    write_to_h5(zeta_chunk)              # loop over q internally
```

Memory per z-chunk: $O(N_k \cdot N_s^2 \cdot N_\mu \cdot N_{\text{zchunk}})$ for $P_k$.

---

## Key Algorithmic Innovations

### 1. Sharded FFT via `shard_map`

JAX's FFT requires contiguous memory along transform axes. With bands sharded across devices, standard `jnp.fft.ifftn` fails on CPU backend.

**Solution:** Use `shard_map` to run FFT independently on each device's local data:

```python
@partial(shard_map, mesh=mesh_xy, 
         in_specs=P(None, ('x','y'), None, None, None, None),
         out_specs=P(None, ('x','y'), None, None, None, None))
def sharded_ifftn(x):
    return jnp.fft.ifftn(x, axes=(-3, -2, -1))
```

This achieves zero-communication FFT when spatial dimensions are not sharded.

### 2. Staged Resharding

Direct resharding from $P(\cdot, (x,y), \cdot, \cdot)$ to $P(\cdot, \cdot, \cdot, y)$ causes XLA to replicate the full tensor before repartitioning ("involuntary full rematerialization").

**Solution:** Two-stage reshard via intermediate sharding:

```python
# Stage 1: all-gather bands over X only
psi = with_sharding_constraint(psi, P(None, 'y', None, None))
# Stage 2: all-gather bands over Y, shard zchunk on Y
psi = with_sharding_constraint(psi, P(None, None, None, 'y'))
```

Each stage is a simple collective that XLA handles efficiently.

### 3. 2D Blocked Cholesky

For large $N_\mu$, the CCT matrix $C_q(\mu, \nu)$ is sharded as $C_q(\mu_X, \nu_Y)$. Standard `jnp.linalg.cholesky` would require gathering to a single device.

**Solution:** 2D blocked Cholesky (`cholesky_2d.py`) that operates on tiles:

- Tiles distributed as $(P_x, P_y)$ blocks
- Column broadcasts and panel updates use `lax.psum`
- Asymptotic complexity: $O(N_\mu^3 / P)$ compute, $O(N_\mu^2 / \sqrt{P})$ communication

### 4. Column-Parallel Triangular Solve with Per-q Gather

$Z_q$ is sharded as $Z_q(\mu, r_{XY})$—columns distributed across all devices. The solve:

1. **Loop over $q$**: For each $q$-point, all-gather $L_q[q]$ to replicate on all devices
2. **Column-parallel solve**: Each device solves for its local columns of $\zeta$
3. **Memory bound**: Only one $L_q[q]$ replicated at a time: $O(N_\mu^2)$

### 5. Per-q HDF5 Writing

Gathering full $\zeta_q(N_q \times N_\mu \times N_{\text{zchunk}})$ to host would exceed memory for large systems.

**Solution:** Write one $q$-point at a time:

```python
for q in range(Nq):
    zeta_q_host = np.asarray(zeta_chunk[q])  # (N_mu, N_zchunk) only
    f['zeta_q'][qx, qy, qz, :, r_start:r_end] = zeta_q_host
```

Peak host memory per write: $O(N_\mu \times N_{\text{zchunk}}) \approx 160$ MB for typical sizes.

---

## Design Constraints

The algorithm was designed around these memory constraints:

| Constraint | Implication |
|------------|-------------|
| $\psi_{nk}(r_{\text{tot}})$ too large | Cannot store full real-space wavefunctions. Use band-chunked FFT with transient $\psi(r)$. |
| $P_k(\mu, r_{\text{tot}})$ too large | Cannot compute ZCT for full grid at once. Use z-chunk loop. |
| $N_\mu \times N_\mu$ too large per device | Cannot replicate full $C_q$ on each device. Use 2D blocked Cholesky, per-$q$ all-gather for solve. |
| $N_q \times N_\mu \times N_{\text{zchunk}}$ too large for host | Cannot gather full $\zeta$ to host. Write per-$q$ to HDF5. |
| Avoid all-to-all on large arrays | Use staged resharding, column-parallel solve. |

### Typical Size Ranges

| Quantity | Range | Notes |
|----------|-------|-------|
| $N_k$ | 1 – 2,000 | k-points in BZ |
| $N_q$ | 1 – 200 | q-points (may differ from $N_k$ with NUFFT) |
| $N_b$ | 20 – 5,000 | bands |
| $N_\mu$ | 200 – 50,000 | ISDF centroids (~10× $N_b$) |
| $N_r$ | 20,000 – 2M | real-space grid |
| $N_G$ | ~$N_r/4$ | G-vectors |

---

## Input Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `band_chunk_size` | 16 | Bands per FFT during z-chunk loop |
| `z_chunk_size` | auto | Z-slices per chunk (auto: 16× $N_\mu$) |
| `max_wfn_chunk_mb` | 0 | Max memory for $P_k$ chunk in MB (overrides z_chunk_size) |

---

## File Organization

- `load_wfns.py` — Core implementation
- `cholesky_2d.py` — 2D blocked Cholesky
- `test_chunked_wfn_loading.py` — Validation tests
- `gw_init.py` — Input parsing and chunk size calculation
