## Executive Summary

The memory model is **overengineered for the zeta-fitting stage** but **completely ignores the sigma-computation stage**, currently the dominant memory consumer. Several arrays are held unnecessarily or have inefficient reshardings.

### Key Findings

1. **`psi_coh_rtot_Y` is UNUSED** — ~20 GB wasted per call
2. **`psi_v_rtot_Y` and `psi_c_rtot_Y` are UNUSED** — chi0/W only needs `rmu` 
3. **Memory model accounts for zeta fitting but not sigma wavefunction loading** (model should prioritize zeta, but the persistent costs of the X/Y wfn copies should be saved)
4. **Sharding transitions in `_finalize()` cause OOM on ≤8 GPUs**
5. **L_q replication during solve is a hidden sqrt(P) bottleneck**
6. **FFT_BUFFERS=2 may be too optimistic; original value of 4 was closer**

---

## 1. Array Lifecycle Analysis

### Arrays by Memory Category

| Category | Arrays | Typical Size | When Needed |
|----------|--------|--------------|-------------|
| **(4) Chunked rtot** | `psi_nk(rtot)`, `P_k(μ,rtot)`, `Z_q(μ,rtot)`, `zeta_q(μ,rtot)` | 10-100 GB global | ISDF fit only |
| **(3) Full XY-sharded** | `psi_nk(rmu)_Y`, `psi_nk(rmu)T_X` | 0.5-5 GB/device | sigma, chi0 |
| **(2) sqrt(P) replicated** | `L_q[iq]` during solve | `n_rmu² × 16B` = 0.1-4 GB | solve stage |
| **(1) Single device** | `Gij_static`, `rho_mu`, `V0_munu` | 10-100 MB | sigma |

### Critical Finding: Wasted rtot Arrays

The following `rtot` arrays are computed but **never used**:

```python
# In cohsex_jax.py, lines 1427-1448:
psi_v_rtot_Y, psi_v_rmu_Y, psi_v_rmuT_X = get_sharded_wfns(...)  # psi_v_rtot_Y UNUSED
psi_c_rtot_Y, psi_c_rmu_Y, psi_c_rmuT_X = get_sharded_wfns(...)  # psi_c_rtot_Y UNUSED  
psi_coh_rtot_Y, psi_coh_rmu_Y, psi_coh_rmuT_X = get_sharded_wfns(...)  # psi_coh_rtot_Y UNUSED
```

**Memory waste per call**: `3 × nk × nb × ns × n_rtot × 16B`

For 6×6 k-grid (nk=36, nb=80, ns=2, n_rtot=7.5M):
- Per array: ~19.9 GB global → 1.24 GB/device on 16 GPUs
- Total waste: ~60 GB global

**Fix**: The COH wfns are from the very lowest band b0 to b4 whereas psi_v and psi_c contain b0:b3 and b1:b4. totally eliminate the calls to all of these sharded wavefunctions (unless it's the non-chunked pipeline) and inherit the ones from the zeta fitting which correspond to the same bands; the COH function can be called twice with slice psi_v[b0:b1] and another time with all of psi_c replacing the COH wfns. be careful to retain the correct physics!

---

## 2. Computational Graph: What Must Coexist?

### Phase 1: ISDF Fitting (zeta construction)

```
Persistent:
  psi_l_rmu_Y, psi_l_rmuT_X     (left centroids)
  psi_r_rmu_Y, psi_r_rmuT_X     (right centroids)  
  L_q                           (Cholesky factor)

Per x-chunk:
  psi_nk(r_chunk)               (FFT workspace, discarded after P)
  P_l_k(μ, r_chunk)             (left pair density)
  P_r_k(μ, r_chunk)             (right pair density)
  Z_q(μ, r_chunk)               (cross-correlation)
  zeta_q(μ, r_chunk)            (solution, written to disk)

Notably it should be possible to delete P_l_k and P_r_k after Z_q is obtained so they don't need to overlap with zeta, unless it seems better for performance to donate all the buffers.

This description is quite incomplete because we need to choose three chunk sizes for three different bottlenecks: band chunks for reading psi_nk(rtot) (to obtain rchunks), q chunks for solving Lq^-H Lq^-1 Zq, and mu(/nu) chunks for calculating <zeta_qmu|Vq|zeta_qnu>
```

**Memory model covers this correctly.**

### Phase 2: chi0/W Computation

```
Persistent (from Phase 1):
  psi_l_rmu_Y, psi_l_rmuT_X
  psi_r_rmu_Y, psi_r_rmuT_X
  V_qmunu                        (bare Coulomb)

Additionally loaded:
  psi_v_rmu_Y, psi_v_rmuT_X     (valence for chi0)
  psi_c_rmu_Y, psi_c_rmuT_X     (conduction for chi0)

Computed:
  chi_q(μ,ν)                    (polarizability)
  W_q(μ,ν)                      (screened Coulomb)
```

**Memory model does NOT account for psi_v and psi_c loading here, but it would be helpful to reverse engineer from everything except chi_q and W_q how many copies for different omega's of chi_q(mu,nu,omega) we can store at once. This will set a chunk size in a later step**

### Phase 3: Sigma Computation

```
Persistent:
  psi_l_rmu, psi_l_rmuT          (for SX Green's function)
  psi_coh_rmu, psi_coh_rmuT      (for COH resolution of identity) 
  W_munu, V_munu                 (interactions)
  
Computed:
  G_munu(k)                      (Green's function)
  G_RI_munu(k)                   (RI sum)
  sigma_munu(k)                  (self-energy)

COH wavefunctions do not need to be calculated at all (see above) and all wfns should be unchanged from the zeta fitting step! no further psi(rtot)'s should be calculated and all wfn calculations should be removed!

Furthermore note that Sigma_mnk at the end is obtained in the mnk basis on every device and all collected at the end (Nb^2*Nk memory cost on each device)
```

---

## 3. Memory Model Errors

### Error 1: Missing sigma-stage accounting, see above

### Error 2: FFT_BUFFERS may be wrong

```python
# cohsex_init.py line 136
FFT_BUFFERS = 2  # Reduced from 4
```

The gw_8gpu_v2.out log shows:
```
Can't reduce memory use below 16.67GiB by rematerialization; 
only reduced to 37.14GiB
```

37.14 GB ≈ 2× the array size, suggesting FFT does need ~2 intermediate copies.
But the OOM at 27.81 GB suggests the sharding transition, not FFT, is the issue. This is suspect and we should tread carefully to determine that we have the number of copies right and are not detecting a different problem. It would be very strange for JAX to use an array 4x the original size

### Error 3: L_q replication not correctly modeled

During `solve_zeta_from_L_q()`, L_q is gathered to all devices within a q-chunk: (NOTE: it SHOULD be the case that L_qchunk is gathered to every proc with qchunk chosen to be >=1, largest that can fit)

```python
# load_wfns.py line 1232
L_rep_shard = NamedSharding(mesh_xy, P(None, None))  # REPLICATED!
```

For q_chunk=1, this means L_q (`n_rmu × n_rmu × 16B`) is replicated on **ALL P devices**.

If n_rmu = 3600, L_q = 207 MB per device — manageable.
If n_rmu = 50000 (your scaling range), L_q = 40 GB per device — **catastrophic**. Most obvious bottleneck for very large calculations, I believe.

The memory model accounts for `L_rep_per_q_gb` but this scales as `n_rmu²`, which dominates at large n_rmu.

---

## 4. Sharding Issues

### Issue 1: `_finalize()` transition causes OOM

```python
# load_wfns.py lines 359-372
# Step 1: Reshard from (b_XY, ns, n_rtot) to (b_X, ns, n_rtot)
psi_rtot = jax.lax.with_sharding_constraint(psi_rtot, x1_4)  # ← PROBLEM

# Step 2: Apply final output shardings
psi_rtot = jax.lax.with_sharding_constraint(psi_rtot, y3_4)
psi_rmu = jax.lax.with_sharding_constraint(psi_rmu, y3_4)
psi_rmuT = jax.lax.with_sharding_constraint(psi_rmuT, x1_4)
```

**Transition**: `P(None, ('x','y'), None, None)` → `P(None, 'x', None, None)` → `P(None, None, None, 'y')`

On non-square meshes (2×4 for 8 GPUs), this requires cross-mesh communication.
XLA cannot find a memory-efficient decomposition and rematerializes the full array.
(It is probably not actually related to nonsquare arrays since sharding should be able to handle that, check as a general issue! but if we frequently materialize in one direction we should probably make it the Y direction if we have the option/assume Y is the larger or the two)

**XLA warning from gw_8gpu_v2.out**:
```
Involuntary full rematerialization. The compiler was not able to go from 
sharding {devices=[1,2,1,1,4]<=[8] last_tile_dim_replicate} to 
{devices=[1,1,1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}
```

**Fix**: The user's earlier insight is correct — use an intermediate sharding:
```python
# b_XY → b_X (keep Y replicated on rtot) → b m_X n_Y
intermediate_1 = P(None, 'x', None, None)  # bands on X, rtot replicated
intermediate_2 = P(None, None, None, 'y')  # bands replicated, rtot on Y
```

### Issue 2: W solve reshards q-parallel → replicated

```python
# w_isdf.py lines 333-336
def solve_body(iq, W_acc):
    V_iq = jax.lax.with_sharding_constraint(V_flat[iq], rep_shard)  # all-gather
    chi_iq = jax.lax.with_sharding_constraint(chi_flat[iq], rep_shard)  # all-gather
```

For each q-point, V and chi matrices (`n_rmu × n_rmu`) are gathered to ALL devices.
This is O(nq × n_rmu² × P) communication, which dominates at large n_rmu.

**Alternative**: Solve should be batched across q with proper 2D parallelism, not serial fori_loop.

(we should be able to use the same functionality that batches L_q by qchunks above!)

### Issue 3: chi0 reshards within fori_loop

```python
# w_isdf.py line 162 (inside _chi_kernel)
chi_q = chi_q + jnp.fft.fftn(...).reshape(...)
```

Each window iteration does FFT + reshape + accumulate.
XLA may rematerialize chi_q if the sharding changes during reshape.

---

## 5. Bottleneck Scaling Analysis

Given: `Nmu = 10×Nb`, `Nb ∈ [50, 10000]`, `Nk ∈ [1, 2000]`, `Nrtot >> Nmu`

| Array | Size (bytes) | Category | Bottleneck At |
|-------|--------------|----------|---------------|
| `psi_nk(rmu)` | `16 × Nk × Nb × Ns × Nmu` | sqrt(P) | Nb=5000, Nk=500 → 4 TB |
| `L_q` | `16 × Nq × Nmu²` | sqrt(P) per q | Nmu=50000 → 40 GB/device |
| `Z_q(μ,rchunk)` | `16 × Nq × Nmu × rchunk` | full P | rchunk large |
| `P_k(μ,rchunk)` | `16 × Nk × Nmu × rchunk` | full P | rchunk large |
| `psi_nk(rtot)` | `16 × Nk × Nb × Ns × Nrtot` | chunked | Always huge |

**Primary bottlenecks**:
1. **Nmu ≥ 10000**: L_q replication becomes prohibitive
2. **Nk × Nb large**: psi_rmu arrays dominate
3. **Current code**: Loading rtot when only rmu needed

---

## 6. Recommended Changes

### High Priority (Fix OOM)

1. **Remove psi_COH and all wfn sharding at that step**
   - Skip rtot computation entirely, make do with unions of existing psi_L and R copies, pay attention to slicing

3. **Fix `_finalize()` resharding**
   - Use two-step transition through compatible intermediate
   - Or eliminate rtot path entirely (see #1)

### Medium Priority (Improve scaling)

4. **Add sigma-stage memory accounting**
   - Include psi_v, psi_c in memory budget
   - Account for W_q workspace

5. **Batch W solve across q**
   - Replace fori_loop with vmapped batched solve
   - Keep L_q sharded during solve (avoid replication) (not sure what this means, just investigate Lq replication which I accept as a necessary evil to solve Lq^-H Lq^-1 Zq, the most comp. expensive step)

6. **L_q scaling safeguard**
   - Add warning when `n_rmu² × 16 > 1 GB`
   - Consider blocked triangular solve to avoid full replication

### Low Priority (Cleanup)

7. **Remove unused code paths**
   - `psi_rtot` outputs from chi0/COH wfn loading
   - Old non-chunked ISDF path

8. **Simplify FFT_BUFFERS**
   - The issue is sharding transitions, not FFT
   - Document that FFT preserves sharding and is memory-neutral

---

## 7. Summary Table: Arrays That Should NOT Coexist
This is quite important, balance these assessments with whether or not it is better to donate a buffer! Don't suggest anything without assessing if it is already chunked and or maximally a performance bottleneck and whether we should delete or donate without deleting; note also many arrays are of the same size

| Phase | Can Delete After |
|-------|-----------------|
| ISDF fit | `psi_nk(r_chunk)` after P built |
| ISDF fit | `P_l, P_r` after Z built |
| ISDF fit | `Z_q` after zeta solved |
| ISDF fit | `C_q` after L built |
| chi0/W | `chi_q` after W computed |
| sigma | `G_R` after sigma_munu computed |

**Current violations**:
- `psi_coh` loaded separately when it's just union(psi_l, psi_r), keeping psi_rtot's. should be eliminated entirely
- L_q replicated per q instead of blocked solve (again check on this?)

---

## 8. Proposed Memory Model Update

Replace the current ZCT-focused model with a three-phase model:

(evaluate this yourself I can't be sure about it; it is too simple and not well commented and doesn't identify eg specific reshardings as the problem. and the current one does a probably overengineered but reasonable job accounting for comm/workspace buffers, also i note we should account for a tradeoff between rchunk size and band chunk size, because while chunking over bands in the rchunk loop, we need to have an active buffer psi_(ntot)k(r_chunk) at the same time as loading psi_(n_chunk)k(rtot) to fill that buffer, and shrinking the size of rtot will make the band loading possibly faster/easier in the presence of mem constraints. hard problem to brainstorm here)

```python
def compute_memory_requirements(meta, n_devices, mesh_shape):
    """
    Phase 1: ISDF fitting (current model, mostly correct)
    Phase 2: chi0/W computation (NEW)  
    Phase 3: Sigma computation (NEW)
    
    Returns the MAXIMUM across all phases.
    """
    # Phase 1: Current model (zeta fitting)
    phase1_peak = compute_zeta_fit_memory(...)
    
    # Phase 2: chi0/W (currently missing!)
    # Need: psi_l_rmu + psi_r_rmu + psi_v_rmu + psi_c_rmu + V_qmunu + chi_q + W_q
    m_centroids = 4 * 16 * n_k * n_b * n_s * n_rmu * (1/p_x + 1/p_y)
    m_V_W = 2 * 16 * n_q * n_rmu * n_rmu / p  # V and W
    m_chi = 16 * n_q * n_rmu * n_rmu / p
    phase2_peak = m_centroids + m_V_W + m_chi
    
    # Phase 3: Sigma (currently missing!)
    # Need: same centroids + W_munu + G_munu + sigma_munu
    m_G = 16 * n_k * n_s * n_rmu * n_s * n_rmu / p
    m_sigma = 16 * n_k * n_s * n_rmu * n_s * n_rmu / p
    phase3_peak = m_centroids + m_V_W + m_G + m_sigma
    
    return max(phase1_peak, phase2_peak, phase3_peak)
```

---

## Appendix: XLA Rematerialization Warnings Observed

From `gw_8gpu_v2.out`:
```
[spmd] Involuntary full rematerialization from 
  {devices=[1,2,1,1,4]<=[8] last_tile_dim_replicate} to 
  {devices=[1,1,1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}
for HLO: %copy.3 = c128[36,13,2,216000] copy(%reshape.110)
source_file="load_wfns.py" source_line=361
```

From `gw_16gpu_v2.out` (succeeded but warned):
```
[spmd] Involuntary full rematerialization from 
  {devices=[1,4,4]<=[16]} to {devices=[16,1,1]<=[16]}
for HLO: %reshape.7 = c128[36,150,150] reshape(%param)
source_file="w_isdf.py" source_line=324
```

The 16-GPU case succeeded because per-device memory was low enough, but the same inefficiency exists.

These are extremely high priority to fix in a way that identifies exactly what arrays are being resharded!