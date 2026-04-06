# psp: unified DFT operator backend — status & next steps

## What's built (this branch: agent/H_DFT_matvec)

### dft_operators.py — fused JIT kernels

  * `apply_H_k`: H|psi> matvec, FFT-box→sparse-G, **2.5 ms/k**
  * `build_matrix_k`: full H_mn (nb×nb), **2.6 ms/k** (37x faster
    than old separate-dispatch path)
  * `compute_kin_ion_all`: multi-GPU with pre-placed data,
    **34 ms total for 9 k on 4 GPUs** (GPU compute at floor;
    host prep ~1.2s dominated by HDF5 reads)
  * `compute_dipole_all`: full velocity/dipole via autodiff VNL
  * `vnl_velocity_autodiff`: jax.jacfwd through V_NL(k) — correct
    at Gamma (old manual code was wrong by 0.03–0.13)
  * JAX B-spline evaluator (`splev_jax`): bit-identical to scipy
  * JAX solid harmonics: polynomial, no trig, autodiff-safe at K=0
  * `@custom_jvp` for stable G_l(q)*S_lm(K) derivative at K=0

### vnl_ops.py — fast dense VNL backend

  * All channels × atoms × betas in one dense Z (total_R, nG)
  * Table-lookup radial evaluation (linear interp, 50k grid): ~100x
    faster than per-scalar B-spline
  * `apply_vnl`: **0.57 ms** (matvec, 80 bands)
  * `vnl_matrix`: **0.38 ms** (nb×nb)
  * `vnl_velocity_matrix`: **0.48 ms** (dipole)
  * `build_vnl_kdata`: **38 ms** without dZ, **255 ms** with dZ

### hamiltonian_matvec.py — multi-GPU H|psi>

  * Single-k fused sparse-G default (2.5 ms)
  * 2-D mesh (k, g) batched path with sharding constraints
  * Validated on 1,4 GPUs in (1×1), (2×2), (4×1) configs

## What to do next

### Wire vnl_ops into dft_operators (easy, high impact)

Replace the per-channel VNL in `dft_operators.apply_H_k` and
`build_matrix_k` with `vnl_ops.apply_vnl` / `vnl_ops.vnl_matrix`.
This would bring the fused H matvec from 2.5 ms to ~2 ms (the
VNL portion drops from ~0.5 ms across 6 dispatches to ~0.57 ms
in one dispatch — wash for MoS2, but wins for heavier atoms with
more channels).

### Migrate callers to dft_operators / vnl_ops

  * `gw/kin_ion_io_chunked.py` → call `dft_operators.compute_kin_ion_all`
  * `get_dipole_mtxels_chunked.py` → call `dft_operators.compute_dipole_all`
    (or directly `vnl_ops.vnl_velocity_matrix`)
  * `get_DFT_mtxels.py` → thin wrapper over `dft_operators.build_matrix_k`
  * `gw/gw_jax.py` uses `load_kin_ion_submatrix` from h5 — could call
    `dft_operators.compute_kin_ion_all` directly to skip the h5 round-trip

### V_xc (not yet built)

Shares the FFT-multiply-FFT pattern with V_loc.  Needs:
  1. XC functional evaluation on real-space grid (libxc or hand-coded LDA/PBE)
  2. Store V_xc_r alongside V_loc_r in OperatorSetup
  3. The operator kernel is identical to V_loc (just a different potential)

### Performance micro-optimisations (diminishing returns)

| Idea | Est. savings | Effort |
|------|-------------|--------|
| Explicit solid-harmonic gradients (replace jax.jvp in dZ build) | ~115 ms/k on dZ build (3x on that component) | low — l≤3 polynomials, ~30 lines |
| Precompute G_l, G'_l arrays at setup time (they only depend on |G|, not k) | most of the 38 ms build-without-dZ | medium — need to store (nbeta, nG) per k |
| Batch `_table_interp` across G_l + G'_l in one call | ~10 ms | low |
| Fuse T+V_loc+V_NL into a single XLA graph (currently 3 JIT calls in dft_operators) | ~0.5 ms dispatch overhead | already done in `_apply_H_k_fused` |
| Replace `at[].add` FFT-box scatter with dense IFFT (keep psi in box form) | eliminates scatter entirely for V_loc matvec | medium — changes external interface |

### Processor grid

The 2-D mesh (k, g) infrastructure is in `hamiltonian_matvec.py`.
For production: default to (nk_devices=ndevices, ng_devices=1).
G-sharding is only useful when nG is too large for one device's
memory (>100k G-vectors, i.e. large unit cells).
