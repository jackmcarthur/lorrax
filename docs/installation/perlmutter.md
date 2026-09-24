# Perlmutter (NERSC)

Perlmutter (4×A100 GPU nodes) is the reference platform. There is one route:
the `lorrax_A` base module, launched through `lx`. Launch geometry and machine
behaviour are owned by [machines/perlmutter.md](../environment/machines/perlmutter.md),
and the module and site descriptor by [`config/README.md`](../../config/README.md).

## What the module provides

`lorrax_A` is a site descriptor that `lx` reads; it is not a launcher. It
supplies the CUDA-13.2 / JAX-0.9.1 runtime, the MPI capabilities and one sealed
FFI bundle: both `LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO` point into a single
[sealed pair](../architecture/ffi_layout.md#2a-the-deployable-unit-is-one-sealed-pair),
so an ordinary run builds nothing. LORRAX's `runtime` package owns the JAX,
GPU-allocator, HDF5 and compile-cache policy ([env_vars.md](../dev/env_vars.md)).
Run scripts export none of it.

## Every run

```bash
export LX_BASE_MODULE=lorrax_A
lx doctor
lx run --pool POOL -N 1 -G 4 -n 4 -- python3 -u -m gw.gw_jax -i cohsex.in
```

- `lx` runs the checkout that contains the working directory. From a data
  directory outside any checkout, `LORRAX_CHECKOUT=/path/to/lorrax` selects the
  checkout; with neither, `lx` runs the module's installed source. It puts the
  selected `src/` first on the payload's `PYTHONPATH`, and startup attests that
  the core and the first-party services were imported from it.
- Use one rank per GPU: `-G` is per node, so P16 is `-N 4 -G 4 -n 16`.
- Run each preprocessing step (k-means, dipoles, kin_ion) as its own `lx run`.
- Never run a driver on a login node, and never `sbatch` an iteration.
  `lx help` documents allocations, pools, status and `lx test`.

## Native FFI stack on Perlmutter

A private build replaces the module's bundle only as a pair. Pin both legs to
one sealed bundle (`LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO`). Pinning one leg,
or mixing a private leg with the module's, refuses at startup. The build and
seal recipe is [Building the FFI libraries § Perlmutter](../building_ffi.md#perlmutter).

The nvcc translation units of the CUDA leg are compiled for
`CMAKE_CUDA_ARCHITECTURES=80` (A100) unless the build sets another
architecture. The mathdx k-convolution router compiles its kernels per k-grid
with NVRTC at run time.
