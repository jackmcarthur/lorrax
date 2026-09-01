# Building the FFI libraries

LORRAX is a JAX program, and on one node with one GPU it needs nothing on this
page. What it cannot do in JAX is the handful of operations that are only fast
when a specialist library owns them across many processes: a distributed
eigensolver, a distributed Cholesky and triangular solve, sharded parallel-HDF5
reads and writes, and a batched FFT over the flat-k layout. Those live behind
`src/ffi/`, compiled into two shared objects that JAX dlopens and calls through
the XLA FFI.

The libraries are not hard to build. What has been hard is knowing whether the
one you built is *correct*, because almost every way of getting it wrong
produces an artifact that links cleanly, loads cleanly, and passes every check
anyone thought to run.

## The failures this page exists because of

On **2026-08-07** a host library was deployed after being built with the generic
`src/ffi/cpp/build_host.sh` on Perlmutter. That script configures no ScaLAPACK —
CMake's built-in probe looks for an MKL directory layout, which does not
describe Cray LibSci — so `ScaLAPACK/BLACS not found` printed as a **warning**,
the link succeeded, and the library shipped with `scalapack=0` and zero
`Scalapack*` handlers. The same build left `cray-libsci` loaded while CMake ran,
so the compiler wrapper auto-injected the sequential LibSci beside the threaded
pair SLATE needs, and the finished object carries both. Two defects, one
command, nothing failed. The symptom arrived days later as nineteen contract
tests going red in a way that needed a person to diagnose.

On **2026-08-06** a host library carried a `DT_NEEDED` on cray-fftw. `DT_NEEDED`
is resolved before any of the library's own code runs, and the Shifter container
does not mount `/opt/cray/pe/fftw`, so the whole object failed to `dlopen`
in-container. Nineteen ScaLAPACK/SLATE/GEMM tests — none of which perform an FFT
— reported as **skipped**, and the suite was green at zero failures. A lost FFT
optimisation had silently become a lost linear-algebra test suite.

On **2026-08-08** a rebuild of the host leg set `LORRAX_FFI_PHDF5_DIR` to name
the HDF5 stage it had to agree with, and got a library linked against a
different HDF5 anyway. That variable does not choose the host leg's HDF5 — the
Cray compiler wrappers do, through the module named in `LORRAX_PM_HDF5` — so it
only fed the comparison, never the link. The result requested SOVERSION 310
beside a device library requesting 200, which resolves to `not found` and takes
the entire library down at load.

And twice in two days — **96a6399**, then the kchunk conversion merged as
**a16a241c** — an FFI handler signature changed. A library built before the
change and Python from after it link, load, register every target and pass every
probe, then fail at the first call that crosses the changed signature with
`INVALID_ARGUMENT: Wrong number of arguments: expected 3 but got 4`, which names
no library, no version and no fix. In between, the pairing was carried by a
hand-maintained table in a notes file whose own warning was the problem: a
mispaired run stays green until something calls `read_slabs`.

The common shape: **a build that quietly delivers less than you asked for, and
a check that cannot fail.** Everything below is the structural answer.

## The two legs

There are two libraries, built in two different places, and they must agree.

`liblorrax_ffi_host.so` — the **host leg** — is CUDA-free and built bare-metal
against the site's compiler environment. It carries the ScaLAPACK, SLATE-CPU,
CBLAS GEMM, FFT and parallel-HDF5 handlers. `liblorrax_ffi.so` — the **device
leg** — is built inside the container against cuSOLVERMp, cuBLASMp, cuFFT,
SLATE-CUDA and the same parallel HDF5, from whatever is bind-mounted there.

Two build environments and one process at run time is the whole difficulty. In a
GPU run both are dlopened `RTLD_GLOBAL`, so they share SONAMEs (`libslate.so.2`,
`libblaspp.so.2`) and sixteen symbol names, and the HDF5 the host leg linked
bare-metal has to be the HDF5 the container mounts. A host library can be
perfectly correct on its own and unloadable beside its partner.

## The verify contract

Every build path in this repository ends at `scripts/verify_ffi_build.sh`. The
site recipes call it, the generic scripts call it, and
`services/distrib_la/tests/test_so_acceptance.py` runs the same file as pytest
cells — so a user validates a build by running the test suite, and there is no
way for the suite and the build to disagree about what a good library is.

```bash
scripts/verify_ffi_build.sh --leg host build_host/liblorrax_ffi_host.so
```

The ten gates it runs, and the property each one guards:

| gate | property |
|---|---|
| 0 | the backends this build was **declared** to contain are the ones it exports, by stamp *and* by symbol |
| 1 | exactly one MPI runtime in the resolved closure — two means `MPI_COMM_WORLD` differs between frames |
| 2 | one BLAS, one threading flavour — two lets ELF load order pick which one runs |
| 3 | the host leg links nothing from the CUDA stack |
| 4 | the dependency closure resolves at load time in this environment |
| 5 | the run-time-resolved FFT engine is not a **load**-time dependency |
| 6 | the OpenMP runtime is really an OpenMP runtime |
| 7 | one HDF5, and it is the one the runtime will mount |
| 8 | the FFT engine that actually binds is the intended one (needs a live process) |
| 11 | the handler-signature ABI matches this source tree |

The two numbers missing from that list are the gates of the cross-`.so` ODR
fix, and they are not in this file because neither one is a property of a
single artifact inspected on its own:

| gate | property | where it lives |
|---|---|---|
| 9 | nothing LORRAX-owned is on the dynamic table, and every shared `lrx_*` entry point carries its leg's suffix | `config/perlmutter/build_ffi_host.sh` and `src/ffi/cpp/build.sh`, at link time; `test_so_acceptance.py` check 6 intersects the two libraries |
| 10 | a CUDA-capable process really can do host phdf5 work with both libraries open | `src/ffi/cpp/gate_one_odr.py`, inside a real GPU allocation |

Two rules make these hard to fool. **A gate that cannot run is not a gate that
passed**: GATE 8 needs a real process and cannot run on a login node, so it
reports `COULD NOT RUN` and is counted separately, and `LORRAX_FFI_VERIFY_STRICT=1`
turns that into a refusal for a certification run inside an allocation. And
**expectations are stated, not inferred**: `LORRAX_FFI_EXPECT_BACKENDS` defaults
to the *full* set for the leg, so a build that quietly loses one fails, and a
site that genuinely builds fewer says so. That default is the entire lesson of
the Aug-7 library.

The verifier can be disabled with `LORRAX_FFI_VERIFY=off`, which announces
itself loudly on every invocation. An unverified library must not be deployed.

## Perlmutter

```bash
bash config/perlmutter/build_ffi_host.sh --fresh          # host leg
src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh      # device leg
```

Running the *generic* `src/ffi/cpp/build_host.sh` here now hands over to the
site recipe rather than building a reduced library — that hand-off is the direct
fix for the Aug-7 accident. `LORRAX_FFI_GENERIC_BUILD=1` overrides it.

The site recipe knows the machine's answers: the explicit LibSci ScaLAPACK link
line (CMake's MKL-shaped probe cannot find it), the `_mp` threading flavour that
matches the `gpu_backend=none` SLATE install, capturing the LibSci and FFTW
prefixes and then *unloading* both modules before invoking CMake, and the phdf5
stage to compare the HDF5 against.

## Frontera

```bash
LORRAX_ROOT=$PWD \
LORRAX_SLATE_HOST_INSTALL_DIR=$WORK/slate_builds/cpu/install \
  config/frontera/build_ffi_host.sh --fresh
```

Same gates, different vendors: MKL supplies ScaLAPACK, CBLAS and DFTI;
`libmkl_blacs_intelmpi_lp64` must match the MPI, because the wrong BLACS links
perfectly and only fails inside the first `blacs_gridinit`. Without
`LORRAX_SLATE_HOST_INSTALL_DIR` this recipe builds the phdf5-only library, and
it declares that to the verifier — a reduced build that says so is fine; a
reduced build that does not is the defect.

## Porting to a new site

Copy the closest site recipe and change values, never structure. The two
existing ones are deliberately parallel so `diff` shows only values. Every lever
below has cost somebody a build.

| lever | what it selects | the trap |
|---|---|---|
| `LORRAX_FFI_EXPECT_BACKENDS` | what the build must contain | omit it and you inherit the full set — deliberately, so a silent reduction fails |
| the BLAS/ScaLAPACK link line | ScaLAPACK + C-BLACS | CMake's probe expects MKL; another `-DLORRAX_SCALAPACK_LIBRARIES` line must use LP64 (32-bit Fortran `INTEGER`) and conventional underscore symbols. ILP64 may link under the same names and corrupt descriptors. |
| the BLAS module | which BLAS is linked | leaving it loaded lets the compiler wrapper inject a *second* flavour on top of yours — GATE 2 |
| the FFTW module | where the FFT engine lives | it must reach CMake as a **dlopen hint** only; on the link line it becomes `DT_NEEDED` and the library stops loading anywhere that SONAME is absent — GATE 5 |
| the HDF5 module (`LORRAX_PM_HDF5` on Cray) | the SOVERSION the host leg **links** | this, not the stage variable, is the real lever |
| the phdf5 stage (`LORRAX_FFI_PHDF5_DIR`) | the SOVERSION the runtime **mounts** | it feeds GATE 7's comparison only. **Set both, and set them to the same version** |
| the XLA FFI headers | the jaxlib ABI compiled against | they must come from the image the library will be *loaded* under, not whatever python is first on `PATH` |
| the SLATE install | SLATE + blaspp + lapackpp | the host leg needs the `gpu_backend=none` build; a CUDA blaspp here makes `get_device_count()` disagree across the two legs |
| the cuSOLVERMp stage (device leg) | comm path and correctness | every stage exports the same SONAME; 0.6.0 returns **wrong** `getrf`/`getrs` on any $P_x>1$ *and* $P_y>1$ mesh, and ships `cal.h` where ≥0.7 does not |
| the MPI include/lib dirs (device leg) | which MPI the library asks for | unset, CMake falls back to HPC-X OpenMPI and the library requests `libmpi.so.40`, which is the wrong name for Cray MPICH |

## The ABI pairing rule

A Python tree and a `.so` must agree about what crosses the FFI boundary, and
nothing in the type system says so. `src/ffi/cpp/common/lorrax_ffi_abi.h` holds
one number, `LORRAX_FFI_ABI_VERSION`, baked into both libraries and mirrored by
both Python loaders (a drift test compares all three).

**Bump it in the same commit as any handler-signature change**: adding,
removing or reordering an `Arg` or `Ret`; moving a value between `Attr` and
`Arg` — both recent bumps were this; changing a dtype or rank; or changing the
meaning of a positional value while keeping its type, which is the silent one.
Do **not** bump for a new handler, which an old library simply does not export
and `probe_target` already reports precisely.

At `dlopen` the loaders read the stamp. A different number is a refusal naming
both versions and the rebuild command. A library with *no* stamp was built
before 2026-08-08 and is announced once rather than refused — unstamped is not
evidence of wrong, and a great many pinned libraries predate the mechanism.
`LORRAX_FFI_ABI_STRICT=1` closes that ratchet where the fleet has caught up.
