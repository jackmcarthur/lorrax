# jax-finufft Integration Notes

## Summary

NUFFT backend implemented and working on CPU. GPU build from source fails on CUDA 13.0 due to multiple C++/CUDA compatibility issues. Use conda-forge for GPU support or stick with standard FFT.

## Implementation

Added NUFFT support for different k-grid / q-grid dimensions in ISDF fitting:
- Type 1 NUFFT (non-uniform k-points → uniform R-grid)
- Batched API usage (all transforms share k-grid)
- Q-grid support in Meta class, CCT/ZCT functions
- Input: `q_grid = 2 2 1` in cohsex input file

Files modified:
- `src/isdf/common/meta.py` - qgrid/use_nufft properties
- `src/isdf/common/load_wfns.py` - NUFFT in compute_CCT/ZCT functions
- `src/gw_isdf/gw_init.py` - q_grid parameter parsing
- `src/gw_isdf/gw_jax.py` - Meta initialization with q_grid

## CUDA 13.0 Build Issues

### Issue 1: Missing CUFFT error codes
**File**: `vendor/finufft/include/cufinufft/contrib/helper_cuda.h`

These error codes don't exist in CUDA 13.0:
```c
case CUFFT_INCOMPLETE_PARAMETER_LIST:
case CUFFT_PARSE_ERROR:
case CUFFT_LICENSE_ERROR:
```

**Solution**: Add `#ifdef` guards
```c
#ifdef CUFFT_PARSE_ERROR
case CUFFT_PARSE_ERROR:
    return "CUFFT_PARSE_ERROR";
#endif
```

**Status**: Patched successfully (see `finufft_cuda13_compat.patch`)

### Issue 2: thrust::binary_function removed
**File**: `vendor/finufft/src/cuda/1d/spread1d_wrapper.cu`

```cpp
template<typename T> struct cmp : public thrust::binary_function<int, int, bool>
```

`thrust::binary_function` was deprecated in C++11, removed in C++17. CUDA 13.0 defaults to C++17.

**Solution**: Would need to refactor to use std::function or lambdas throughout FINUFFT CUDA code.

**Status**: Not attempted - too invasive for a dependency

### Issue 3: Likely more compatibility issues beyond thrust

Building from source with CUDA 13.0 is not practical.

## Performance

**CPU-only** (current):
- 360k transforms: 25s (~14k/sec)
- Batched API working correctly
- Too slow for production

**Expected GPU** (10^9 pts/sec per docs):
- 360k transforms: ~0.25s
- Would be practical

## Workarounds

### Option A: conda-forge (GPU)
```bash
conda install -c conda-forge jax-finufft cufinufft
```
Pre-built for CUDA 13. Skip uv/pip entirely for jax-finufft.

### Option B: CPU-only (current)
Already configured. Fine for testing/validation, too slow for production.

### Option C: Standard FFT
Don't use q_grid parameter. Matched k/q grids use standard FFT - much faster without GPU.

## When NUFFT Makes Sense

With GPU only:
- Large k-grids (12×12×1 → 4×4×1) where memory savings matter
- Downsampling saves significant I/O

Not worth it:
- Small grids (3×3×1 → 2×2×1) - overhead exceeds benefit
- CPU-only - slower than standard FFT by ~10-100x

## Testing

CPU-only benchmark:
```bash
cd tests_isdf/cohsex_prod
source env.sh
uv run python test_nufft_speed.py
```

Full run (not recommended CPU-only):
```bash
uv run python -m gw_isdf.gw_jax -i test_nufft_2x2x1.in
```

## Recommendation

Implementation is correct but not practical without GPU. Use conda-forge if GPU NUFFT is needed, otherwise standard FFT is fine for typical use cases.

## Files

- `finufft_cuda13_compat.patch` - CUFFT error code fixes
- `rebuild_finufft_gpu_patched.sh` - Build script with patch
- `test_nufft_speed.py` - Performance benchmark
- `test_nufft_2x2x1.in` - Example input
- `NUFFT_STATUS.md` - Detailed notes (verbose)

