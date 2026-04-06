# psp: unified dft_operators backend

## Goal

Consolidate all plane-wave DFT operator logic into a single
`psp/dft_operators.py` module that provides **fast, fused, shardable
operator kernels** for the PW Hamiltonian components.  Every module
that needs DFT matrix elements or matvecs sources its core
functionality from here:

  * `hamiltonian_matvec.py` — H|psi> for iterative diagonalisation
  * `gw/kin_ion_io_chunked.py` — kin+ion matrix element writer
  * `get_DFT_mtxels.py` — full H_DFT matrix builder
  * `get_dipole_mtxels_chunked.py` — dipole / velocity matrix elements
  * (future) V_xc operator (currently absent from all of the above)

## Operator inventory

All operators take wavefunctions in **sparse-G** representation
`(nvec, nspinor, nG)` and return the same, unless noted.

| Operator | Formula | Implementation | Notes |
|----------|---------|----------------|-------|
| T (kinetic) | T_G psi_G | diagonal multiply | trivially parallel |
| V_loc (local ionic) | FFT(V_r IFFT(psi)) | scatter->IFFT->mult->FFT->gather | needs FFT box scratch |
| V_NL (nonlocal KB) | Z E Z^dag psi | project->D->unproject einsums | allreduce if G-sharded |
| V_H (Hartree) | same form as V_loc | shares V_loc kernel with V_H_r | optional, from rho_val |
| V_xc (XC potential) | same form as V_loc | shares V_loc kernel with V_xc_r | **TODO: not yet built** |

## Processor grid

2-D JAX device mesh `('k', 'g')`:

  * **k-axis** — batch independent k-points (1 k per device by default)
  * **g-axis** — optional G-vector sharding within each k-point

For the `g`-axis:
  * T, V_NL unproject: trivially G-local
  * V_NL project (Z^dag psi): partial sum + allreduce over g
  * V_loc / V_H / V_xc: rematerialise full FFT on each g-device
    (allgather psi along g, do IFFT*V*FFT, keep local g-shard)

Default: `(nk_devices=ndevices, ng_devices=1)` — pure k-batching.

## Design

```
dft_operators.py
  OperatorSetup       — k-independent data (V_loc_r, vnl_plan, V_H_r, ...)
  KPointOperators     — per-k precomputed data (T_diag, Z projectors, G-indices)
  build_operator_setup(wfn, sym, meta, pseudos, ...) -> OperatorSetup
  build_kpoint_operators(k_idx, setup, ...) -> KPointOperators

  # Core fused kernels — single JIT, sparse-G in/out
  apply_T_k(psi_G, kops)            -> H_G      (diagonal)
  apply_Vloc_k(psi_G, kops)         -> H_G      (FFT-based)
  apply_VNL_k(psi_G, kops)          -> H_G      (KB einsums)
  apply_H_k(psi_G, kops)            -> H_G      (T + Vloc + VNL)

  # Matrix element builders (for kin_ion, get_DFT_mtxels)
  build_matrix_k(psi_G, kops)       -> H_mn     (nb x nb)

  # Batched (all k, 2-D mesh)
  BatchedOperators    — stacked/padded for mesh execution
  apply_H_batched(psi_G, bops, mesh) -> H_G
  build_matrix_batched(psi_G, bops, mesh) -> H_mn_k
```

## Migration path

1. Build `dft_operators.py` with core kernels reusing existing
   `build_local_ionic_potential_on_G_total`, `build_vnl_plan`, etc.
2. Rewrite `hamiltonian_matvec.py` to thin wrappers over dft_operators.
3. Rewrite `kin_ion_io_chunked.py` to use `build_matrix_k` from
   dft_operators (currently reimplements all the scaffolding).
4. Add V_xc support (needs XC functional evaluation on real-space grid).
5. Migrate `get_DFT_mtxels.py` internals to dft_operators.
6. Migrate dipole code where applicable (note: dipole uses k-space
   derivatives d/dk, not the standard V operators, so only the
   scaffolding / V_NL velocity parts overlap).

## Performance targets

Per-k matvec on A100 (MoS2 3x3, 80 bands, nG~2000):
  * Current fused path: ~2.5 ms
  * Target: ~2 ms (fuse VNL channels, reduce dispatch)
  * kin_ion full matrix (80x80): ~5 ms (currently ~130 ms in chunked)
