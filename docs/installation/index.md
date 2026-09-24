# Installation

LORRAX needs two things: Python ≥ 3.12 with the JAX/JAXLIB 0.9 series, and the
native FFI pair. The pair is `liblorrax_ffi_host.so` (host leg: ScaLAPACK,
SLATE-CPU, CBLAS, FFTW, parallel HDF5) and `liblorrax_ffi.so` (CUDA leg:
cuSOLVERMp, cuBLASMp, SLATE-CUDA, parallel HDF5, the mathdx k-convolution
router). The native layer is required at every process count, P = 1 included,
because the JAX fallbacks were deleted
([decisions](../architecture/decisions.md)). A missing or mismatched library
refuses at startup, before the first `jit`. `pyproject.toml` owns the Python
pins; [ffi_layout.md](../architecture/ffi_layout.md) owns the native design.

## One route per platform

| platform | route | page |
|---|---|---|
| NERSC Perlmutter (A100, CUDA 13.2, JAX 0.9.1) | the `lorrax_A` base module, which supplies the runtime and one sealed FFI bundle; launch with `lx` | [Perlmutter](perlmutter.md) |
| TACC Frontera (CPU) | build the host leg with `config/frontera/build_ffi_host.sh` | [Frontera](../environment/machines/frontera.md), `config/frontera/README.md` |
| another site | build both legs from the stage scripts and seal them | [Building the FFI libraries](../building_ffi.md), [FFI native libraries](ffi-native-libs.md), `src/ffi/PORTING.md` |

## Startup refusals and their fix

| refusal | cause | fix |
|---|---|---|
| `Could not locate liblorrax_ffi_host.so (platform=cpu)` or `… liblorrax_ffi.so (platform=CUDA)`, with the paths searched | no library for this platform | Perlmutter: set `LX_BASE_MODULE=lorrax_A` and launch through `lx`. Elsewhere: build the leg the message names. |
| `HANDLER ABI MISMATCH` | the library and the Python tree disagree on `LORRAX_FFI_ABI_VERSION` | rebase onto current main, or rebuild and reseal both legs from the same tree |
| `partial sealed-bundle override refused` | only one of `LORRAX_FFI_SO` / `LORRAX_FFI_HOST_SO` is set | unset both, or pin both legs of one sealed bundle |
| `mixed native providers` | the two selected legs claim different bundle manifests | select both legs from one bundle |
| the JAX version check (`tools/require_jax09.py` before driver import, `runtime.jax_support.enforce()` before the first physics `jit`) | JAX or JAXLIB outside the 0.9 series | use the 0.9 series; there is no override |

An unsealed build-tree library loads with a `LEGACY-UNSEALED` announcement
and its hash, as a developer path; it is not a production provider.

## See also

- [Environment](../environment/overview.md): the runtime stack, JAX
  configuration and the startup block.
- [Quickstart](../quickstart.md): the bundled COHSEX fixture, end to end.
