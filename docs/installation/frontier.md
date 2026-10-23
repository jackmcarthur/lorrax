# Frontier (OLCF)

This page installs LORRAX from a clone on Frontier (MI250X, `gfx90a`): the
ROCm leg `liblorrax_ffi_rocm.so` and the host leg `liblorrax_ffi_host.so`,
built bare-metal against the Cray programming environment and the venv's JAX
ROCm wheels. [Installation](index.md) states what an installation consists of;
[Building the FFI libraries](ffi-build.md) owns the build contract. The site
pins live in `config/frontier/`.

## 1. Environment {#environment}

| what | pinned in |
|---|---|
| `PrgEnv-gnu/8.7.0`, `cray-mpich/9.0.1`, `cray-libsci/25.09.0`, `cray-hdf5-parallel/1.14.3.7`, `cray-fftw/3.3.10.11` | `config/frontier/ffi_mpi.sh` |
| the venv (JAX 0.9.1 with the `rocm-sdk` 7.14.1 wheels), default `$HOME/envs/lx-jax091-rocm7141` | `LORRAX_OLCF_VENV` in the build scripts |
| `rocm/7.14.1` (compile only: the same release as the venv's wheels) | `config/frontier/build_ffi_rocm.sh` |

The JAX ROCm plugin maps its HIP runtime, rocBLAS, rocSOLVER, hipBLASLt,
RCCL and rocFFT from the venv's `rocm-sdk` wheels. The ROCm leg and ELPA
are compiled against module `rocm/7.14.1`, the same release with the
same SONAMEs, and at run time they bind the wheel copies. So a process maps
one ROCm stack.

## 2. Build the native pair {#build}

Run once per clone, in order:

```bash
bash config/frontier/build_openblas_seq.sh   # the sequential, locking OpenBLAS ELPA links
bash config/frontier/build_ffi_host.sh       # src/ffi/cpp/build_host/liblorrax_ffi_host.so
bash config/frontier/build_ffi_rocm.sh       # src/ffi/cpp/build_rocm/liblorrax_ffi_rocm.so
```

OpenBLAS installs under `$HOME/software/openblas_seq/install`
(`LORRAX_OPENBLAS_ROOT`) and ELPA under `$HOME/software/elpa_builds`
(`build_ffi_rocm.sh` builds it when its prefix is incomplete); both are shared
by every clone. Rebuild both legs after a pull that changes
`src/ffi/cpp/` or `config/frontier/`.

**The build records, and every run enforces, each leg's dependency closure**
([the dependency closure](ffi-build.md#closure)). Each leg's build writes
`<leg>.closure.json` beside it and stamps its `closure_sha256` into
`PROVENANCE`. The record names every library the leg maps:

- **ROCm leg:** the venv's `rocm-sdk` HIP runtime, rocBLAS and rocSOLVER (and
  their dependencies); ELPA and OpenBLAS from its RUNPATH; the pinned MPI
  and HDF5.
- **Host leg:** the pinned MPI, HDF5 and LibSci.

At start-up the loader opens exactly those files, then refuses the process if
any other copy of one of them is mapped. A module ROCm library on
`LD_LIBRARY_PATH`, an `LD_PRELOAD`, or a site LibSci that maps another
cray-mpich cannot replace a recorded file, and the two legs' records may not
disagree. A refusal reads

```text
GATE native-closure: got <wrong file>; want <recorded file>; fix: <remedy>
```

The remedy is to remove what maps the other copy, or to rebuild the leg in the
environment you run in. Setting `LD_LIBRARY_PATH` neither causes nor fixes it.
A record goes stale when a recorded file changes, for example when the venv's
`rocm-sdk` wheels are reinstalled or OpenBLAS or ELPA is rebuilt. The next run
then refuses until the legs are rebuilt.

## 3. Run {#run}

```bash
source config/frontier/ffi_env.sh                 # the host leg's pins and machine directories
source $HOME/envs/lx-jax091-rocm7141/bin/activate
source config/frontier/rocm_env.sh                # libhsa-runtime64.so -> the wheel's, for libfabric
export PYTHONPATH=$PWD/src:$(ls -d $PWD/services/*/src | tr '\n' ':')
srun -N 1 -n 4 -c 7 --gpus-per-task=1 --gpu-bind=closest \
  python -u -m gw.gw_jax -i deck.in
```

The legs are found in-tree (`src/ffi/cpp/build_rocm/`, `build_host/`); no
library variable is needed. P = 4 (one half node, four GCDs) is the
verification geometry. The first load
of each leg prints `LEGACY-UNSEALED` with the file hash (a private build), and
then one `[lorrax native] closure ROCM enforced: …` / `closure cpu enforced: …`
line with the record and identity hashes.

## 4. Reusing builds across worktrees {#build-reuse}

Site tooling that shares one build between checkouts with the same FFI
sources (a source key, [startup environment](../architecture/startup_environment.md#ffi-source-key))
must also require the build's record to verify on this machine:

```bash
python -m lxkit.closure verify <leg.so>   # exit 0 and the identity, or 3 and the refusal
```

A build whose recorded libraries have changed then fails to match its key and
is rebuilt. It is never reused.
