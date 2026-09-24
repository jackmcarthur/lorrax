# Building the FFI libraries

LORRAX calls vendor libraries through two shared objects built from the one
C++ tree `src/ffi/cpp/` ([ffi_layout.md §2](architecture/ffi_layout.md#2-the-one-c-tree)).
They are deployed as one sealed pair ([§2a](architecture/ffi_layout.md#2c-the-deployable-unit-is-one-sealed-pair)).
This page owns how a pair is built, verified and sealed on each site, and the
ABI rule that pairs it with a Python tree.

The main hazard is a build that links, loads and registers every target while
delivering less than was asked for: a missing backend, a second MPI or BLAS, or
the wrong HDF5. The verifier below exists to make each of those a build failure.

## The two legs

- `liblorrax_ffi_host.so` (host leg) is CUDA-free and built bare-metal against
  the site compiler environment. It carries the ScaLAPACK, SLATE-CPU, CBLAS
  GEMM, FFTW and parallel-HDF5 handlers.
- `liblorrax_ffi.so` (CUDA leg) carries cuSOLVERMp, cuBLASMp, SLATE-CUDA,
  parallel HDF5 and the mathdx k-convolution router.

In a GPU run both legs are dlopened `RTLD_GLOBAL` into one process. They
therefore share SONAMEs (`libslate.so.2`, `libblaspp.so.2`), must link the same
MPI, and the HDF5 the host leg links must be the HDF5 the runtime provides. A
leg can be correct on its own and still unloadable beside its partner.

## Seal the deployable pair

Seal from the checkout that built both legs. Each leg's `PROVENANCE` must say
`git_dirty=no` at that checkout's `HEAD`:

```bash
python src/ffi/cpp/stage/seal_bundle.py \
  --cuda path/to/liblorrax_ffi.so \
  --host path/to/liblorrax_ffi_host.so \
  --private-lib path/to/libblaspp.so.2 \
  --private-lib path/to/libslate.so.2 \
  --output path/to/new-bundle-dir
```

Repeat `--private-lib`, in dependency-first order, for every private engine
provider in either leg's `DT_NEEDED` closure. Never list MPI, site HDF5, the
CUDA runtime or driver, NCCL, or system and compiler libraries: the machine
runtime owns those. The sealer validates source and artifact hashes and the
canonical SONAMEs, refuses an incomplete private closure and an existing output
directory, and publishes both legs with `lorrax_ffi_bundle.json` atomically. The loaders find that
manifest beside the libraries; no environment variable names it.

## The verify contract

Every build path ends at `scripts/verify_ffi_build.sh`, and
`services/distrib_la/tests/test_so_acceptance.py` runs the same script as
pytest cells, so the suite and the build agree on what a good library is.

```bash
scripts/verify_ffi_build.sh --leg host build_host/liblorrax_ffi_host.so
```

| gate | property |
|---|---|
| 0 | the backends the build was declared to contain are exported, by stamp and by symbol |
| 1 | exactly one MPI runtime in the resolved closure (`libmpi`, `libmpi_gnu`, `libmpi_gnu_<N>`) |
| 2 | one BLAS and one threading flavour |
| 3 | the host leg links nothing from the CUDA stack |
| 4 | the dependency closure resolves at load time in this environment |
| 5 | the run-time-resolved FFT engine is not a load-time dependency |
| 6 | the OpenMP runtime is an OpenMP runtime |
| 7 | one HDF5, and it is the one the runtime provides |
| 8 | the FFT engine that binds is the intended one (needs a live process) |
| 11 | the handler-signature ABI matches this source tree |

Two further gates are properties of the pair, not of one artifact:

| gate | property | where it runs |
|---|---|---|
| 9 | nothing LORRAX-owned is on the dynamic table, and every shared `lrx_*` entry point carries its leg's suffix | at link time in `config/perlmutter/build_ffi_host.sh` and `src/ffi/cpp/build.sh`; check 6 of `test_so_acceptance.py` intersects the two libraries |
| 10 | a CUDA-capable process with both libraries open does host phdf5 work | `src/ffi/cpp/gate_one_odr.py`, inside a GPU allocation |

A gate that cannot run reports `COULD NOT RUN` and is counted separately; GATE
8 cannot run on a login node. `LORRAX_FFI_VERIFY_STRICT=1` turns `COULD NOT RUN`
into a failure, for certification inside an allocation.
`LORRAX_FFI_EXPECT_BACKENDS` defaults to the leg's full backend set, so a build
that loses a backend fails, and a site that builds fewer backends must say so.
`LORRAX_FFI_VERIFY=off` disables the verifier and announces that on every
invocation; an unverified library is not deployable.

## Perlmutter

Both legs link one MPI, pinned in `config/perlmutter/ffi_mpi.sh`: cray-mpich
9.0.1 (`libmpi_gnu_123.so.12`, the MPI that the phdf5 stage and the SLATE host
install already require), cray-libsci 25.09.0, with darshan unloaded. Change
those values there and nowhere else.

- Host leg: `bash config/perlmutter/build_ffi_host.sh --fresh`.
- CUDA leg: the CUDA-13 runtime recipe
  ([Perlmutter §2](environment/machines/perlmutter.md#2-the-lorrax_a-module-and-the-ffi-bundle)),
  which sources `ffi_mpi.sh` from the checkout it builds.
- Build from a clean checkout in a zero-GPU compute step
  (`lx run --pool POOL -N 1 -G 0 -n 1 -- …`). On a login node the default HDF5 module links
  a second MPI and GATE 1 fails.
- Seal the two legs as above.

The site recipe carries this machine's answers: the explicit LibSci ScaLAPACK
link line (CMake's probe expects an MKL layout), the `_mp` threading flavour
that matches the `gpu_backend=none` SLATE install, LibSci and FFTW prefixes
captured and their modules unloaded before CMake runs, and the phdf5 stage to
compare HDF5 against. The generic `src/ffi/cpp/build_host.sh` hands over to it
on Perlmutter and Frontera; `LORRAX_FFI_GENERIC_BUILD=1` builds generically
instead.

## Frontera

```bash
LORRAX_ROOT=$PWD \
LORRAX_SLATE_HOST_INSTALL_DIR=$WORK/slate_builds/cpu/install \
  config/frontera/build_ffi_host.sh --fresh
```

MKL supplies ScaLAPACK, CBLAS and DFTI. `libmkl_blacs_intelmpi_lp64` must
match the MPI: the wrong BLACS links and fails only inside the first
`blacs_gridinit`. Without `LORRAX_SLATE_HOST_INSTALL_DIR` the recipe builds the
phdf5-only library and declares that reduced set to the verifier.

## Porting to a new site

Copy the closest site recipe and change values, not structure; the two recipes
are parallel, so `diff` shows only values.

| lever | what it selects | the trap |
|---|---|---|
| `LORRAX_FFI_EXPECT_BACKENDS` | what the build must contain | omitted, it is the full set, so a silent reduction fails |
| the BLAS/ScaLAPACK link line | ScaLAPACK + C-BLACS | CMake's probe expects an MKL layout; elsewhere pass `-DLORRAX_SCALAPACK_LIBRARIES` as a whole link line |
| the BLAS module | which BLAS links | left loaded, the compiler wrapper injects a second flavour (GATE 2) |
| the FFTW module | where the FFT engine lives | it must reach CMake as a dlopen hint only; on the link line it becomes `DT_NEEDED` (GATE 5) |
| the HDF5 module (`LORRAX_PM_HDF5` on Cray) | the SOVERSION the host leg links | this, not the stage variable, sets the link |
| the phdf5 stage (`LORRAX_FFI_PHDF5_DIR`) | the SOVERSION the runtime provides | it feeds only GATE 7's comparison; set it and the HDF5 module to the same version |
| the MPI module | which MPI both legs link | both legs must name the same `libmpi` (GATE 1, gate 10) |
| the XLA FFI headers | the jaxlib ABI compiled against | take them from the JAX that will load the library |
| the SLATE install | SLATE + blaspp + lapackpp | the host leg needs `gpu_backend=none`; a CUDA blaspp makes `get_device_count()` disagree across the legs |
| the cuSOLVERMp stage (CUDA leg) | comm path and correctness | every stage exports the same SONAME; 0.6.0 returns wrong `getrf`/`getrs` on any mesh with both P_x > 1 and P_y > 1; ≥ 0.7 is NCCL-native and needs `-DLORRAX_FFI_HAVE_CAL=OFF` |
| the MPI include/lib dirs (CUDA leg) | which MPI the library requests | unset, CMake falls back to HPC-X OpenMPI and requests `libmpi.so.40` |
| `CMAKE_CUDA_ARCHITECTURES` (CUDA leg) | the SASS/PTX of the nvcc translation units | defaults to `80` (A100) |

## The ABI pairing rule

`src/ffi/cpp/common/lorrax_ffi_abi.h` holds one number,
`LORRAX_FFI_ABI_VERSION`. It is compiled into both legs and mirrored by both
Python loaders (the live value: `LORRAX_FFI_ABI_VERSION` in
`src/ffi/common/ffi_loader.py`), and a drift test compares all three.

Bump it in the same commit as any handler-signature change: adding, removing
or reordering an `Arg` or `Ret`; moving a value between `Attr` and `Arg`;
changing a dtype or rank; or changing the meaning of a positional value while
keeping its type. A new handler needs no bump: an older library does not export
it, and `probe_target` reports that precisely.

At `dlopen` the loaders read the stamp. A different number refuses with
`HANDLER ABI MISMATCH`, naming both versions and the rebuild command. A library
with no stamp refuses when it belongs to a sealed bundle or when
`LORRAX_FFI_ABI_STRICT=1`; otherwise it loads as `LEGACY-UNSEALED` with its
hash. The bundle manifest's source revision is provenance, not a compatibility
key: the ABI number and the build contract decide compatibility.
