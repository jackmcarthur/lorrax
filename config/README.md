# LORRAX site configuration

This directory describes site capabilities. It does not own allocation,
launcher, JAX, HDF5, compile-cache, or profiling policy.

On Perlmutter, `lx` is the only supported launcher:

```bash
export LX_BASE_MODULE=lorrax_A
export LORRAX_CHECKOUT=/path/to/lorrax
lx doctor --refresh
lx run -N 1 -G 4 -n 4 python3 -u -m gw.gw_jax -i cohsex.in
```

Set `LORRAX_CHECKOUT` when launching from a run/data directory. LORRAX startup
then attests that the imported core and first-party services came from that
checkout. The base module supplies the machine environment independently.

## What the Perlmutter module owns

`config/modulefiles/lorrax/0.1.0.lua` exports two values consumed by `lx`:

- `LORRAX_ROOT`: the base checkout used as a fallback and to locate helpers;
- `LORRAX_SHIFTER`: one container capability string containing the image,
  native bind mounts, MPI settings, native library order, and supplemental
  third-party Python path.

It also exposes the resolved native stage roots for build diagnostics. The
module deliberately defines no `lxalloc`, `lxrun`, `lxshell`, or `lxpre`
functions. It does not set JAX allocator, preallocation, HDF5, or compile-cache
variables; [`runtime`](../src/runtime/__init__.py) is their single owner.

The native search order contains exactly one selected cuSOLVERMp path and one
MPI implementation. Do not add a second versioned vendor tree as a fallback:
duplicate SONAMEs make runtime selection depend on path order.

## Install a base module

Edit `config/perlmutter/site_config.sh`, then install:

```bash
LORRAX_MODULE_NAME=lorrax_A bash config/perlmutter/install.sh
LX_BASE_MODULE=lorrax_A lx doctor --refresh
```

The installer requires an explicitly selected, current JAX/JAXLIB image. It
refuses the retired pre-0.9 images rather than silently reinstalling one.

The site configuration contains only:

- the checkout, module destination, container image, and supplemental
  third-party Python directory;
- the Shifter module set and native container layout;
- immutable host stage roots for NVHPC, parallel HDF5, SLATE, and FFTW;
- the CPU MPIwrapper build inputs.

First-party Python services are package dependencies in the root
`pyproject.toml`; do not enumerate their `src` directories in a launcher.

## Native closure

The Perlmutter container mounts these host stages at stable paths:

| Host setting | Container path | Contents |
|---|---|---|
| `LORRAX_FFI_NVHPC_DIR` | `/lorrax_nvhpc` | selected cuSOLVERMp/cuBLASMp tree |
| `LORRAX_FFI_PHDF5_DIR` | `/lorrax_phdf5` | one Cray parallel-HDF5 build |
| `LORRAX_FFI_SLATE_DIR` | `/lorrax_slate` | SLATE support and Cray GPU transport |
| `LORRAX_FFI_FFTW_DIR` | `/lorrax_fftw` | double-precision serial FFTW engine |

The corresponding `*_DEFAULT` values in `site_config.sh` are used unless an
operator overrides a stage before the base module is loaded. A stage change
invalidates the native build; rebuild and run the FFI gates named in
[`docs/architecture/ffi_layout.md`](../docs/architecture/ffi_layout.md).
Never bundle or prepend a second MPI implementation.

The standalone build wrapper uses the same policy:

```bash
export LORRAX_FFI_IMAGE=<verified-jax-0.9-image>
src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh
```

It sets only container/native/MPI capabilities. LORRAX startup sets runtime
policy before JAX or HDF5 import.

## CPU multi-process runs

The CPU MPI adapter is a pinned MPIwrapper build against Cray MPICH:

```bash
export LORRAX_CHECKOUT=/path/to/lorrax
lx run --cpu -N 1 -n 1 -- "$LORRAX_CHECKOUT/config/perlmutter/build_mpiwrapper.sh" --fresh
```

For the complete CPU rank prelude and validated geometry, follow
[`docs/environment/machines/perlmutter.md`](../docs/environment/machines/perlmutter.md).

## Ownership and portability

- `config/perlmutter/site_config.sh` owns Perlmutter paths and vendor stack.
- `docs/architecture/ffi_layout.md` owns native binding and ABI design.
- `docs/dev/env_vars.md` owns runtime environment variables.
- `docs/environment/machines/perlmutter.md` owns launch geometry and current
  machine verification.

For another Shifter site, copy the Perlmutter config directory and replace
only the site capability values. A non-Shifter site needs a different
environment descriptor; do not emulate `lx` with copied shell functions.
