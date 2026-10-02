# Installation

This page states what a LORRAX installation consists of, which installation
route is the default on which machine, and what refuses at startup when a
piece is missing. Read it before the page for your machine
([Perlmutter](perlmutter.md), or [Building the FFI libraries](ffi-build.md)
on any other site); the [Quickstart](../quickstart.md) assumes it is done.

## What an installation is

A LORRAX process needs three things.

1. **Python ≥ 3.12 with the JAX/JAXLIB 0.9 series** and the packages
   `pyproject.toml` pins. On NVIDIA GPUs the CUDA extra (`cuda12` or
   `cuda13`) also installs `nvidia-mathdx`, whose headers the k-convolution
   kernels are compiled from at run time ([k-convolution](../architecture/kconv.md)).
2. **The native FFI pair**: `liblorrax_ffi_host.so` (the host leg: ScaLAPACK,
   SLATE, CBLAS, an FFTW3-ABI FFT and parallel HDF5) and `liblorrax_ffi.so`
   (the CUDA leg: cuSOLVERMp, cuBLASMp, cuBLAS, cuFFT, the NVRTC-built
   mathdx k-convolution and Fourier-plan kernels, and parallel HDF5). The
   pair is required at every process count, P = 1 included, because there is
   no pure-JAX implementation of the k-convolution, the distributed linear
   algebra or the slab I/O to fall back to
   ([decisions](../architecture/decisions.md)). How the pair is built,
   verified and sealed is [Building the FFI libraries](ffi-build.md); the
   design of the layer is [the FFI layer](../architecture/ffi_layout.md).
3. **A launcher that gives each rank one device** and the machine's MPI and
   network settings; the machine pages own this
   ([Perlmutter](../environment/machines/perlmutter.md),
   [Frontera](../environment/machines/frontera.md)).

## The JAX contract {#jax}

LORRAX supports exactly the JAX/JAXLIB 0.9 series:

```text
jax     >= 0.9.0, < 0.10.0
jaxlib  >= 0.9.0, < 0.10.0
```

Patch upgrades inside the series are allowed; any other minor generation of
either package refuses, and there is no override. The window is narrow
because `common/jax_compile_cache.py` patches four `jax._src` private
functions whose arity changes between generations: a mismatched JAX does not
fail cleanly, it dies at the first compile. Three surfaces enforce the same
window:

1. `pyproject.toml` constrains the base, `cuda12`, `cuda13` and development
   installs;
2. `tools/require_jax09.py` checks the installed package metadata before a
   launcher's first JAX import, without importing JAX;
3. `runtime.jax_support.enforce()`, step 5b of
   `runtime.initialize_communicator_stack` (after the first `jax.devices()`,
   before the first `jit`), checks both live packages and the arity of each
   patched private. The version string alone is not trusted: a date-stamped
   container build can report an allowed version over a different
   `jax._src`.

JAX 0.9 does not certify a native library: the library's CUDA major, its
handler ABI and its dependency closure are checked separately when it loads
([the ABI rule](ffi-build.md#abi)).

## Runtime defaults {#defaults}

| platform | default route | page |
|---|---|---|
| NERSC Perlmutter (A100, CUDA 13.2, JAX 0.9.1) | the module stack: one venv, one source snapshot and one sealed FFI bundle, loaded with `module load`; a development clone builds its own pair and runs with plain `srun` | [Perlmutter](perlmutter.md) |
| TACC Frontera (CPU) | a clone with the host leg from `config/frontera/build_ffi_host.sh`, run inside the site's apptainer image | [Frontera](../environment/machines/frontera.md), `config/frontera/README.md` |
| any other machine | a clone: `uv sync` with the platform's JAX extra, both legs built and sealed from the stage scripts; `config/cloud/` is the all-wheel CUDA route for a machine with only a driver | [Building the FFI libraries](ffi-build.md), `config/cloud/README.md` |

A site with a tuned stack keeps its recipes in one `config/<machine>/`
directory (`config/perlmutter/`, `config/frontera/`); `config/README.md`
explains how to port one.

## Startup refusals and their fix

| refusal | cause | fix |
|---|---|---|
| `Could not locate liblorrax_ffi_host.so (platform=cpu)` or `… liblorrax_ffi.so (platform=CUDA)`, with the paths searched | no library for this platform | build the leg the message names ([Perlmutter](perlmutter.md#build), [other sites](ffi-build.md)), or load a module that supplies the pair |
| `HANDLER ABI MISMATCH` | the library and the Python tree disagree on `LORRAX_FFI_ABI_VERSION` | rebuild and reseal both legs from the same tree ([the ABI rule](ffi-build.md#abi)) |
| `partial sealed-bundle override refused` | only one of `LORRAX_FFI_SO` / `LORRAX_FFI_HOST_SO` is set | unset both, or pin both legs of one sealed bundle |
| `mixed native providers` | the two selected legs claim different bundle manifests | select both legs from one bundle |
| `JAX09_ENV_REFUSED` (from `tools/require_jax09.py`), or the `runtime.jax_support` refusal before the first `jit` | JAX or JAXLIB outside the 0.9 series, or a `jax._src` of the wrong shape | install the 0.9 series (§ [JAX contract](#jax)) |
| `GATE mathdx-headers` | a CUDA run without `nvidia-mathdx` in the venv | `uv sync --extra cuda13` (or `cuda12`) |

An unsealed library from a build tree loads with a `LEGACY-UNSEALED`
announcement and its hash. That is the normal state of a private development
build; only a sealed pair is a production provider
([sealing](ffi-build.md#seal)).
