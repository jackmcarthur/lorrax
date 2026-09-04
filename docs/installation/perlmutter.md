# Perlmutter (NERSC)

Perlmutter is LORRAX's reference 4×A100 GPU-node platform. The maintained
sources are:

- [`config/README.md`](../../config/README.md) for installing the environment
  descriptor and native stages;
- [`docs/environment/machines/perlmutter.md`](../environment/machines/perlmutter.md)
  for launch geometry and verified machine behavior;
- [`docs/architecture/ffi_layout.md`](../architecture/ffi_layout.md) for native
  binding and ABI design.

## One-time environment install

```bash
vi config/perlmutter/site_config.sh
LORRAX_MODULE_NAME=lorrax_A bash config/perlmutter/install.sh
LX_BASE_MODULE=lorrax_A lx doctor --refresh
```

The base module is a site descriptor used by `lx`; it is not a launcher. It
contains the container image, native bind mounts, MPI capabilities, and
supplemental third-party Python path. LORRAX runtime owns JAX, allocator,
HDF5, compilation, and profiling defaults.

## Every run

Never run a driver on a login node and never submit an iteration with
`sbatch`. Pin the source when starting from a data directory:

```bash
export LX_BASE_MODULE=lorrax_A
export LORRAX_CHECKOUT=/path/to/lorrax
lx doctor
lx run -N 1 -G 4 -n 4 -- \
  env PYTHONPATH="$LORRAX_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -u -m gw.gw_jax -i cohsex.in
```

Use one rank per GPU. Run preprocessing as separate `lx run` steps; do not use
the retired `lxpre` wrapper. The source path belongs after `lx run`, because the
container replaces an outer `PYTHONPATH`; the runtime closure receipt must name
the checkout and its first-party services. `lx help` is the authority for
allocations, concurrent batches, status, and tests.

## Native FFI stack

Stage one compatible cuSOLVERMp, parallel-HDF5, SLATE, and FFTW closure, then
build in the selected environment:

```bash
export LORRAX_FFI_IMAGE=<verified-jax-0.9-image>
src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh
```

The embedded convolution kernels currently carry an A100 `sm_80` image and
fall back to NVRTC on other architectures. That fallback is functional, but
does not certify another GPU architecture as production-ready.
