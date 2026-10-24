# JAX features deferred to after 1.0

LORRAX 1.0 ships on the JAX 0.12 series. It keeps its own native code for the
NCCL communicators and its own collective schedule: there is no compute and
communication overlap beyond what XLA's latency-hiding scheduler already does.
The items below use features that arrived between JAX 0.9.1 and 0.12.0.1. They
are wanted soon after 1.0 and are not 1.0 blockers. Each Python code site
carries a `TODO(after-1.0-jax)` comment (the C++ sites are named in the table
only, because any change under `src/ffi/cpp/` forces a rebuild of both legs), so

```bash
git grep -n "TODO(after-1.0-jax)"
```

lists them.

The findings come from the JAX changelog and a source diff of jax-v0.9.1 and
jax-v0.12.0 (and the XLA commits each pins), made 2026-10-23. None has been
measured on a GPU yet. Re-read the current changelog before starting an item:
the async collectives and the FFI Record API are marked unstable upstream.

Not deferred: the replacement of the removed private `all_gather_invariant` by
`lax.all_gather(x, axis, to='invarying')` in `lxkit` collectives belongs to the
0.12 port itself, because without it `ordered_psum` silently pays an extra
whole-slab `psum` on every call.

## Items

| # | Feature (JAX version) | LORRAX site | Expected gain | Validate by |
|---|---|---|---|---|
| 1 | XLA-owned NCCL communicator for FFI handlers: request it in the Prepare stage, `get_communicator` returns an `ncclComm_t` (`collectives.h`, jaxlib 0.11.2) | `src/ffi/cpp/cusolvermp/context.cc` (`ncclCommInitRank`) and the cuSOLVERMp/cuBLASMp backends of `distrib_la` | one set of communicators instead of two: less device memory, no deadlock between LORRAX's and XLA's communicators, no unique-id broadcast | the multi-node eigh/LU acceptance runs at P = 4, 16; compare memory high-water and time per call |
| 2 | Compute and communication overlap: async collectives (`psum_scatter_start` public in 0.12; the other `*_start` forms are internal), and the XLA flags `xla_gpu_experimental_enable_collective_multi_streaming`, `xla_gpu_executable_num_communication_streams`, `xla_gpu_experimental_collective_start_as_early_as_possible`, `xla_gpu_pipeline_all_gather` | `distrib_la.panel_matmul` (gather the next band panel during the current GEMM), the BSE ring `ppermute` stages (`src/bse/bse_ring_comm.py`), the band-projection reduce-scatter, `gw.ppm_accumulators.WINDOW_OVERLAP`, `gw.tangential_execution.FACE_COMPILER_OPTIONS` | time per Green build and per Σ node at P ≥ 16, where the panel exchange is most of a build | A/B through `compiler_options` first (no code change), then async collectives where the flags are not enough; compare TIMING receipts |
| 3 | CPU LAPACK parallel along the batch axis (0.10.0) | the CPU `batch_reshard` eigh and LU, `distrib_la.eigh_local` and `solve_local` | CPU (Frontera) eigensolve time | measure with the BLAS thread count pinned; it can oversubscribe cores together with MKL or LibSci threads |
| 4 | Hang diagnosis: `xla_gpu_device_execution_terminate_timeout`, `xla_gpu_execution_progress_tracking` (reports the stuck thunk) | a debug dial next to the runtime's `XLA_FLAGS` merge | names the operation a multi-node NCCL, libfabric or cuSOLVERMp hang is stuck in | provoke a known hang on two nodes and check the report |
| 5 | FFI Record API (`Ffi::BindRecord`, execution stage RECORD; `record.h`, jaxlib 0.11.2), so handlers can join CUDA command buffers | the k-convolution and active-range GEMM handlers inside `lax.scan` bodies; they declare no command-buffer trait, so each splits the CUDA graph | lower launch overhead per scan trip | per-node time of the χ⁰ and Σ scans; the header warns of frequent breaking changes |
| 6 | `jax.profiler.ProfileSession` and `jax.profiler.export_profile_data` (0.12) | `src/common/jax_profile.py` | an in-memory profile, simpler than the trace-directory handling | the profiled hsuite run |
| 7 | `jax.experimental.overlap.program_order` (0.11.1; `exclude_argnames` 0.11.2; `strict_in_out` 0.12) | ordering FFI collective custom calls against XLA collectives inside one `jit` | replaces hand-placed optimization barriers | any site that orders a native collective against an XLA one |

## Considered and not pursued

- **Uneven sharding.** `shard_map` in 0.12 still requires every sharded axis to
  divide by its mesh axis, so the padded carriers stay.
- **Pallas Mosaic GPU.** It gained Ampere `mma` and `cp.async` in 0.11.0, but
  nothing shows f64 or complex support; the NVRTC/cuFFTDx kernels stay.
- **Generalized `jax.scipy.linalg.eigh` (0.12).** It needs a definite metric;
  the tangential reduction's metric is only semidefinite.
- **`jax.sharding.no_collectives` (0.12).** It works in explicit-sharding mode
  only, and LORRAX is built on `shard_map`.

Changelog: <https://docs.jax.dev/en/latest/changelog.html>.
