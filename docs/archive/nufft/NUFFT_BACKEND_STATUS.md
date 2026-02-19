# NUFFT Backend Implementation Status

**Branch**: `nufft_backend`  
**Status**: Infrastructure complete, full transforms pending

## Overview

Added experimental support for non-uniform FFT (NUFFT) q-grid in ISDF fitting pipeline. This allows using a coarser/different q-grid than k-grid for memory efficiency in C_q, Z_q, and V_q arrays.

## What's Implemented ✅

### 1. Input Parameter Parsing (`cohsex_init.py`)
- Added `q_grid` parameter to input file format: `"nqx nqy nqz"` (e.g., `"6 6 1"`)
- Comprehensive documentation of NUFFT feature in parameter comments
- Defaults to `None` (standard uniform FFT, q-grid == k-grid)

### 2. Meta Class Q-Grid Support (`meta.py`)
- Added `nqx`, `nqy`, `nqz`, `nq_tot` fields to `Meta` dataclass
- Auto-defaults to k-grid dimensions if not specified
- Added `use_nufft` boolean flag: True when q-grid ≠ k-grid
- Created cached `qgrid`, `qgrid_np`, `qgrid_jax` arrays
- Updated `from_system()` to accept optional `q_grid` parameter

### 3. NUFFT Wrapper Functions (`load_wfns.py`)
- `generate_k_points_crystal()`: Generate uniform k-point grid in [0,1)³
- `generate_q_points_crystal()`: Generate q-point grid in [0,1)³
- `nufft_k_to_R()`: Core NUFFT transform using `jax_finufft.nufft1`
  - Type-1 NUFFT: uniform k-grid → non-uniform q-grid
  - K-points in crystal coords [0,1)³, converted to radians
  - Returns P_R on q-grid (nqx×nqy×nqz) from P_k on k-grid
- `flexible_k_to_R_transform()`: Universal wrapper auto-selecting NUFFT vs FFT
- `get_cached_nufft_points()`: Efficient point caching to avoid regeneration

### 4. Pipeline Integration (`load_wfns.py`)
- Updated `compute_CCT_from_left_right()`:
  - Accepts optional `meta` parameter
  - Output C_q sized to q-grid when `meta.use_nufft=True`
  - Cache keys include `use_nufft` flag for proper JIT
- Updated `compute_ZCT_from_left_right_zchunk()`:
  - Same q-grid sizing as CCT
  - Accepts optional `meta` parameter
- Updated `fit_zeta_chunked_to_h5()`:
  - Passes `meta` to CCT/ZCT functions
  - Output zeta_q.h5 arrays sized to q-grid

### 5. Main Driver Integration (`cohsex_jax.py`)
- Parse `q_grid` from input file
- Pass to `Meta.from_system()`
- Use `meta.qgrid` and `meta.nq_tot` for array sizing
- Informative startup messages:
  - Show k-grid vs q-grid dimensions
  - Print NUFFT enabled/disabled status
  - Warn that GW pipeline not yet updated

## What's NOT Implemented ⚠️

### Full NUFFT Transforms in CCT/ZCT
Currently, the CCT and ZCT functions have **stubs** that raise `NotImplementedError`:

```python
if use_nufft:
    raise NotImplementedError(
        "[NUFFT BACKEND] Full NUFFT path for CCT/ZCT not yet implemented. "
        "Currently only standard FFT (q-grid == k-grid) is supported."
    )
```

**What needs to be done**:
1. Replace the IFFT(P_k) step with NUFFT k→R transform to q-grid
2. Perform cross-product in R-space on q-grid
3. Forward FFT from q-grid back to q-space (uniform FFT on q-grid)

**Pseudocode for full NUFFT CCT**:
```python
# Current (uniform FFT):
P_l_R = jnp.fft.ifftn(P_l_k, axes=(0,1,2), norm='forward')  # k-grid → k-grid R
P_r_R = jnp.fft.ifftn(P_r_k, axes=(0,1,2), norm='forward')
C_R = jnp.conj(P_l_R) * P_r_R
C_q = jnp.fft.fftn(C_R, axes=(0,1,2), norm='forward')

# NUFFT path:
P_l_R = nufft_k_to_R(P_l_k, kgrid, qgrid)  # k-grid → q-grid R (NUFFT!)
P_r_R = nufft_k_to_R(P_r_k, kgrid, qgrid)
C_R = jnp.conj(P_l_R) * P_r_R  # (nqx, nqy, nqz, n_rmu, n_rmu)
C_q = jnp.fft.fftn(C_R, axes=(0,1,2), norm='forward')  # uniform FFT on q-grid
```

### GW Pipeline Updates
The chi0/W/sigma pipeline in `cohsex_jax.py` and `w_isdf.py` is **not yet updated** for NUFFT:
- V_qmunu arrays still assume q-grid == k-grid in many places
- Coulomb interaction v(q) uses k-grid BZ sampling
- Screening W(q) computed on k-grid
- Self-energy Σ(k) needs q-grid convolution

**This will break GW calculations if NUFFT is enabled!**

## How to Use (Current State)

### To enable NUFFT backend (will fail until full transforms implemented):
```ini
[cohsex]
q_grid = 6 6 1  # Coarser q-grid than k-grid
# ... other parameters
```

### To use standard FFT (works now):
```ini
[cohsex]
# Omit q_grid parameter, or set equal to k-grid
# q_grid = 16 16 1  # If k-grid is also 16×16×1
```

## Testing Strategy

Once full NUFFT transforms are implemented:

1. **Convergence test**: Run with q-grid == k-grid (should match standard FFT exactly)
2. **Memory test**: Run with coarser q-grid (e.g., 6×6×1 q-grid on 16×16×1 k-grid)
3. **Physics test**: Check C_q matrix properties (Hermitian, positive definite)
4. **Zeta test**: Verify ζ_q quality with coarser q-grid

## Code Locations

All changes marked with `[NUFFT BACKEND]` comments:
- Input parsing: `src/isdf/gw_isdf/cohsex_init.py`
- Meta class: `src/isdf/common/meta.py`
- NUFFT wrappers: `src/isdf/common/load_wfns.py` (lines ~95-265)
- Pipeline integration: `src/isdf/common/load_wfns.py` (CCT/ZCT functions)
- Main driver: `src/isdf/gw_isdf/cohsex_jax.py` (startup section)

## Git History

```
d221d65 [NUFFT Backend 5/5] Wire up q_grid in cohsex_jax main pipeline
0339eaf [NUFFT Backend 4/5] Integrate NUFFT into CCT/ZCT pipeline
5f5e764 [NUFFT Backend 3/5] Add NUFFT wrapper functions to load_wfns
022433f [NUFFT Backend 2/5] Add q-grid support to Meta class
0ae8ce4 [NUFFT Backend 1/5] Add q_grid input parameter parsing
```

## Dependencies

- `jax-finufft` with GPU support (compiled with CUDA 13.0 patches)
- Currently optional (only checked if NUFFT enabled)

## Next Steps

1. **Implement full NUFFT transforms in CCT/ZCT**:
   - Replace IFFT with `nufft_k_to_R()` calls
   - Test convergence (q==k should match FFT)
   
2. **Update GW pipeline for q-grid**:
   - Modify V_q computation for non-uniform q sampling
   - Update chi0/W calculations
   - Fix Σ(k) convolution sums
   
3. **Performance optimization**:
   - Batch NUFFT calls for better GPU utilization
   - Cache k-points and q-points arrays
   - Profile NUFFT vs FFT overhead
   
4. **Documentation**:
   - Add user guide for choosing q-grid size
   - Memory savings vs accuracy tradeoffs
   - Example input files

## Summary

✅ **Infrastructure complete**: Input parsing, Meta class, NUFFT wrappers all working  
⚠️ **Transforms pending**: CCT/ZCT need full NUFFT implementation (currently stubs)  
❌ **GW not updated**: Pipeline will break if NUFFT enabled  
📝 **Well documented**: All changes marked with [NUFFT BACKEND] comments  
🔧 **Ready for development**: Clean structure to complete transforms

