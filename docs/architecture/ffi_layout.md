# The FFI layer

How LORRAX reaches vendor libraries: what the layers are, what each machine
provides, which knobs decide correctness rather than speed, and how to tell
the failure modes apart.

> **Verification scope.** Every `file:line` and every default below was read
> from source on **Perlmutter**, on **2026-08-06**. The first pass read
> `886139f`; the page was then rebased onto `8789131` and the statements in
> §§3–8 that a commit in between could have moved were **re-read at
> `8789131`** the same day — the stripe defaults, the CAL option default, the
> `run_shifter.sh` library path, the DFTI/FFTW3 counts, the `phdf5` context
> defaults and the three shim call-site counts. Statements marked *(Frontera,
> unverified 2026-08-06)* were not checked on Frontera and must be re-read
> there before being relied on. Line numbers drift: they are provided so you
> can find the code, not so you can quote them. **Read the file.**
>
> This page owns the *native boundary*. It does not own owner rulings
> ([`decisions.md`](decisions.md)), the SlabIO tiers and their measurements
> ([`slab_io.md`](slab_io.md)), or knob spellings and defaults
> ([`../dev/env_vars.md`](../dev/env_vars.md)). See the
> [register](../index.md#register).

---

## 1. The layers

Five, outermost first. Each one can refuse; none silently substitutes.

| # | Layer | Lives in | Job |
|---|---|---|---|
| 1 | **Consumer** | `src/gw/`, `src/file_io/`, `src/bse/`, … | states *logical* intent — shapes, not strides |
| 2 | **Service facade** (Python) | `src/ffi/io.py`, `fft.py`, `gemm.py`, `linalg/` | owns the gate grammar, builds the descriptor, picks a backend |
| 3 | **Gate** | `src/ffi/gate.py` | announce-or-refuse; an explicit request that cannot be honoured **refuses**, never downgrades |
| 4 | **XLA FFI custom call** | `src/ffi/common/ffi_loader.py` | resolves handler symbols out of the built `.so` |
| 5 | **C++ handler + vendor library** | `src/ffi/cpp/<vendor>/` | the actual MPI-IO / BLAS / FFT / solver call |

The rule that makes this tractable: **a vendor dependency enters only
through a facade with runtime resolution and an announced refusal.** Since
the 2026-08-01 ruling (`decisions.md`) the FFI is *required*, so a missing
library is a startup refusal naming the `.so` — not a demotion to a slower
Python path. Vendor-portability fallbacks *inside* a handler (FFTW-vs-MKL
symbol resolution, plain-loop CBLAS) are a different thing and stay: they
are how the required layer remains buildable everywhere.

### Python-side module map

`src/ffi/io.py` (the parallel-HDF5 service), `fft.py` (flat-k FFT),
`gemm.py` (batched vendor GEMM), `gate.py`, and the `linalg/` facade are the
real modules. `phdf5/`, `slate/`, `scalapack/`, `mklfft/`, `mklblas/`,
`cufft/` survive as **re-export shims** — `src/ffi/phdf5/` is 40 lines
across four files.

Deleting a shim is gated on its consumers moving, and **none has moved
yet**: outside `src/ffi/` there are still 10 `ffi.phdf5` references (chiefly
`file_io/_slab_io_ffi.py`, `file_io/wfn_loader.py`), 6 `ffi.mklblas`
(`common/contract_bands.py`), and 4 `ffi.mklfft` (`common/fft_helpers.py`,
`gw/ppm_tau_kernel.py`) — counted at `886139f` and **re-counted unchanged at
`8789131`, 2026-08-06**. The gate is
`grep -rn "ffi\.mklfft" src/ tests/ | grep -v '^src/ffi/'` returning empty.
Run the grep; do not trust a count written down here.

---

## 2. The one C++ tree

`src/ffi/cpp/` — one directory, one `CMakeLists.txt`, both platform legs
behind an explicit selector.

```
src/ffi/cpp/
├── CMakeLists.txt      -DLORRAX_FFI_PLATFORM=cuda|host; FATAL_ERROR when unset
├── build.sh            CUDA leg (inside Shifter)
├── build_host.sh       host leg
├── run_shifter.sh  in_container.sh  select_gpu.sh  gate_one_mpi.sh
├── stage/              vendor stage scripts (phdf5_stage_cray.sh, …)
├── common/  mklfft/  cufft/  mklblas/  scalapack/  slate/  cusolvermp/  cublasmp/
└── phdf5/   api.cc  context.cc  ctx.h  read_ffi.cc  write_ffi.cc
            phdf5_interface.h  platform_seam.h  shard_index.h
```

Two legs, two libraries:

* `-DLORRAX_FFI_PLATFORM=host` → `liblorrax_ffi_host.so`. CUDA-free by
  construction. Feature options `LORRAX_FFI_HAVE_PHDF5`,
  `LORRAX_HOST_HAVE_SCALAPACK`, `LORRAX_HOST_HAVE_SLATE`,
  `LORRAX_HOST_HAVE_FFTW3`.
* `-DLORRAX_FFI_PLATFORM=cuda` → `liblorrax_ffi.so`. Feature options
  `LORRAX_FFI_HAVE_CAL`, `LORRAX_FFI_HAVE_CUBLASMP`, `LORRAX_FFI_HAVE_CUFFT`,
  `LORRAX_FFI_HAVE_PHDF5`, plus a SLATE probe.
* unset → `FATAL_ERROR` naming both legs (`CMakeLists.txt:1399`). "Which
  directory you pointed cmake at" was an implicit dial; it is gone.

`LORRAX_FFI_PLATFORM` is a **CMake cache variable, not an environment
variable** — it is never read from the environment by any Python module.

Vendor subdirectory names are kept deliberately: every filename stays unique
(`slate/ctx.h` vs `phdf5/ctx.h` vs `cusolvermp/ctx.h`), and every historical
reference maps mechanically `src/ffi/<v>/cpp/X` → `src/ffi/cpp/<v>/X`. Docs
and commit messages written before 2026-07-31 use the old spelling.

---

## 3. What each machine provides

| Service | Perlmutter (GPU leg + host leg) | Frontera *(unverified 2026-08-06)* |
|---|---|---|
| FFT (CPU) | `cray-fftw/3.3.10.11` | MKL's native FFTW3 export |
| FFT (GPU) | cuFFT (`cufftPlanMany64`) | n/a |
| GEMM | Cray LibSci CBLAS | MKL CBLAS |
| Dense solvers | SLATE (GPU + host), cuSOLVERMp | ScaLAPACK (MKL), SLATE |
| Parallel HDF5 | `cray-hdf5-parallel` + Cray MPICH | HDF5 + Intel MPI |
| Container | Shifter (`run_shifter.sh`) | apptainer |

**One FFT source serves both.** The CPU flat-k translation unit was
source-locked to MKL's DFTI descriptor API until 2026-08-05; it is not any
more. `src/ffi/cpp/mklfft/fft_flat_k_ffi.cc` now contains **zero**
`DftiCreateDescriptor` calls and four `fftw_plan_many_dft` calls. Entry
points are resolved at *run* time by `dlsym` (`RTLD_DEFAULT` → `RTLD_NEXT`),
so the same object links MKL's native FFTW3 export on Frontera and
`cray-fftw` on Perlmutter. No environment variable names the engine — **the
engine is named by what the `.so` links.**

That design produces a signature worth recognising, because it looks like a
defect and is not: `nm -D --undefined-only | grep -c fftw_` → **0** while
`libfftw3.so.mpi31.3` sits in `DT_NEEDED`. The `DT_NEEDED` entry exists so
the library is *loaded* for `dlsym` to resolve against; nothing binds at
link time. That is the `-Wl,--no-as-needed` idiom, not a dangling
dependency.

> **That count is not a gate, and it was used as one.** The host-FFI leg was
> certified partly on `nm -D --undefined-only | grep -c fftw_` → 0. The check
> is **necessary but not sufficient**: once entry points are resolved by
> `dlsym`, the count is driven to 0 *by construction* — it would read 0 for a
> build with no FFTW anywhere near it. It never inspects `DT_NEEDED`, which
> is the part that actually decides whether the library loads, and the next
> paragraph is what happens when only the count is checked. A check that
> cannot fail is not evidence. The sufficient form is `readelf -d` for the
> `DT_NEEDED` entries plus `ldd` **inside the container on a compute node**
> (§7c on why the login node cannot answer this).

**But `DT_NEEDED` is still load-bearing at load time**, and that is a live
failure today. Measured 2026-08-06, in-container on compute node
nid001644: the host `.so` carries the right directory in its **RPATH**
(`/opt/cray/pe/fftw/3.3.10.11/x86_milan/lib`), so on the bare host it
resolves — but inside the Shifter image `/opt/cray/pe` **does not exist at
all**, so all three FFTW entries report `not found` and the entire
`liblorrax_ffi_host.so` fails to load. Tier-1 on the CUDA leg is
consequently 33 passed / **19 skipped**, every skip reading
`liblorrax_ffi_host.so unavailable: libfftw3.so.mpi31.3: cannot open
shared object file`.

No `LD_LIBRARY_PATH` value repairs this — the files are not in the
container's mount namespace. **The containerized host leg had never been
green**; the "35/35 on both vendors" certification was a bare-host run.
A skip is not a pass.

> **Fixed — on a branch this one does not contain.** `411e257`
> ("the run-time-resolved FFT engine stops being a load-time dependency", on
> `fix/host-ffi-fftw-dt-needed-2026-08-06`) removes the link-time dependency
> so the engine is `dlopen`'d rather than `DT_NEEDED`, and adds the
> sufficient invariant — **zero `fftw` in `DT_NEEDED`** — as a build gate in
> `config/perlmutter/build_ffi_host.sh`. In-container Tier-1 host then reads
> **49 passed / 2 skipped / 1 failed** against the 33/19/0 above; the 2 skips
> are the known SLATE `heev` L-2 and the 1 failure is the FFT cell honestly
> reporting that the container ships no FFTW3.
>
> **Verified 2026-08-06: `411e257` is NOT an ancestor of `8789131`**
> (`merge-base --is-ancestor`). So the failure described above is live *in
> this branch* and the numbers above are this branch's numbers. One closure
> hole survives the fix and is not in its scope:
> `libhdf5_parallel_gnu.so.310` against the staged HDF5 1.12 — owned by the
> phdf5 stage.
>
> The earlier repair suggestion here — bind-mount `/opt/cray/pe` — is
> withdrawn. It treats a load-time dependency that should not exist as a
> mount problem, and `411e257` is the better shape.

The GPU flat-k mirror (`cufftPlanMany64`, advanced layout) was **not**
re-certified in the 2026-08-05 Perlmutter campaign. Treat a CUDA-leg FFT
number as carried forward, not measured.

---

## 4. Picking an nvhpc stage picks a communication path

**This is the section that has caused real skew. Read it before touching
the CUDA leg.**

The staged NVIDIA HPC SDK trees under `/lorrax_nvhpc` are *not* the same
library at different version numbers. They differ in how cuSOLVERMp
communicates:

| Stage | cuSOLVERMp | Ships `cal.h` / `libcal`? | Comm path | Build flag |
|---|---|---|---|---|
| `25.5_cuda12.9` | 0.6.0 | **yes** | CAL | `-DLORRAX_FFI_HAVE_CAL=ON` (the CMake default, `CMakeLists.txt:51`) |
| `0.7.2_cuda12.9` | 0.7.2 | **no** | NCCL-native | `-DLORRAX_FFI_HAVE_CAL=OFF` — required |

So a default here does not merely guess a version, it **silently picks a
communication path**. That is why `build.sh` refuses rather than choosing:
`src/ffi/cpp/build.sh:54-91` exits 2 when neither `LORRAX_NVHPC_ROOT` nor
`LORRAX_NVHPC_SUBPATH` is set, and enumerates the stages actually present,
probing each for `cal.h` and printing which flag it needs
(`build.sh:80-85`).

Three further facts, each of which has bitten:

1. **Every stage exports the same SONAME**, `libcusolverMp.so.0`. Building
   against one and running against another links cleanly and warns about
   nothing.
2. **`25.5_cuda12.9` (0.6.0) is racy on any mesh with `Px>1` *and*
   `Py>1`.** MEASURED 2026-08-06: the failure signature is
   **nondeterminism, not a stable wrong answer** — at 2×2 it trips a
   rerun-bit-determinism assert before the residual is ever compared,
   consistent with `config/perlmutter/site_config.sh` crediting 0.7.2 with
   "the race fix". `0.7.2_cuda12.9` carries both the CAL→NCCL ABI fix and
   the race fix, and is what `site_config.sh` selects; `run_shifter.sh:171`
   defaults `LORRAX_NVHPC_SUBPATH` to
   `0.7.2_cuda12.9/math_libs/12.9/lib64`.
   *Note for whoever edits the source:* the comments at
   `run_shifter.sh:165-166` and `build.sh:27-29,65-66` describe this as
   "returns WRONG getrf/getrs answers", which overstates a race as a
   deterministic result. A rerun that agrees proves nothing here.
3. **Both stages are on `LD_LIBRARY_PATH` at runtime.** `run_shifter.sh:202`
   places the *selected* stage first and `25.5_cuda12.9` after it, on
   purpose: only that tree ships `libcal.so.0`, which a CAL-built `.so`
   carries in `DT_NEEDED`. It means the ordering, not the mount, is what
   decides which `libcusolverMp` you get.
   *Still true at `8789131`, re-read 2026-08-06.* Commit `b2df35f`
   ("the 25.5 libcal fallback is vestigial after the 0.7.2 rebuild") is
   **comment-only** — it adds sixteen lines above an unchanged `LDLIB=`
   and says so: "The entry is left in place — an older CAL-linked .so still
   needs it … No behaviour change." Do not read that commit subject as a
   removal.

The single source of truth is `LORRAX_NVHPC_SUBPATH`. `run_shifter.sh`
exports both it and a `LORRAX_NVHPC_ROOT` derived from its first component
(`run_shifter.sh:240-241`), so a build launched through `run_shifter.sh`
agrees with the run it is built for **by construction** rather than by two
people remembering the same string. Launch builds that way.

**Verified end to end on the machine, 2026-08-06** (Perlmutter, compute
node, in-container build):

* `NVHPC_ROOT` resolved to `/lorrax_nvhpc/0.7.2_cuda12.9` from the single
  `LORRAX_NVHPC_SUBPATH` string.
* The rebuilt `.so` links cuSOLVERMp **0.7.2 with no CAL**: `libcal` absent
  from `DT_NEEDED`, zero hits in the build log, `nm -D | grep cal_` → 0
  symbols. The previous `.so` has `U cal_comm_create`. The runtime banner
  reads `library 0.7.2, comm path: NCCL`.

So the stage genuinely selects the comm path — this is measured, not
inferred from the CMake option.

> **Open skew, not yet resolved in code.** The CMake default is
> `LORRAX_FFI_HAVE_CAL=ON` (`CMakeLists.txt:51`) while the runtime default
> stage is `0.7.2_cuda12.9`, which ships no `libcal` and needs `OFF`. The
> two defaults disagree. `build.sh` refusing an unstated stage is what
> currently prevents the mismatch from being silent; nothing else does.

---

## 5. Parallel HDF5 — what the FFI side of it is

**[`slab_io.md`](slab_io.md) owns this subsystem**: the three tiers, the
`slab_io` deck key that routes between them, the striping campaign, the
launcher requirements and the multi-node certification. Only the two
FFI-side facts belong here.

**The C++ handler is one source serving both legs.** The same `phdf5/`
sources compile into `liblorrax_ffi.so` and into the CUDA-free
`liblorrax_ffi_host.so`, where the D2H staging into a pinned buffer degrades
to an in-place read of the XLA host buffer. Tier 1 (`PHDF5_FFI`) is that
handler; tier 2 (`phdf5_host`) drives the same MPI-IO from Python and needs
no `.so` at all.

**One boolean grammar spans both writers**, so
`LORRAX_PHDF5_COLLECTIVE_WRITES=0` means "independent" in each of them — §6.

One thing is worth repeating from `slab_io.md`, because getting it wrong
sends you hunting for a demotion that no longer exists: **`H5PY_ALLGATHER`
is a refusal above one process, not a fallback** (owner ruling 2026-08-05,
implemented `0d8e50c`). It is not a tier the system may choose; it is a tier
the system refuses to choose.

> **A gap in that refusal, found 2026-08-06.** The rule is enforced in
> `gw/gw_config.py` — `_refuse_explicit_h5py_allgather` and
> `_refuse_slab_io_no_parallel_writer`, both of which always raise, and both
> placed after the whole precedence chain so no deck branch escapes them.
> **But it is deck-level only.** The library entry point
> `file_io/slab_io.py:88-90` still returns `SlabIOBackend.H5PY_ALLGATHER`
> when `use_ffi_io is None`, at any process count, and `:97-98` maps
> `use_ffi_io=False` the same way. A caller who constructs `SlabIO` directly
> instead of going through a parsed deck still reaches the tier at P>1.
> Anywhere you read "both routes raise at parse time" — including on this
> page before today — that is a statement about the deck router, which is
> the only place it is true.

---

## 6. phdf5 defaults — read them here, then read the file

**The struct initialisers in `ctx.h:155-160` are not the effective
defaults.** Every field is reassigned from the environment at `open_file`
time in `context.cc:352-373`. Two of the six differ between the two places.
Read `context.cc`, not the header — quoting a declaration and stopping is
exactly how the stale `use_collective_write=false` claim survived for ten
days.

| Field | `ctx.h` decl | **Effective** | Env override | Notes |
|---|---|---|---|---|
| `use_collective_read` | `true` | `true` | `LORRAX_PHDF5_INDEPENDENT=1` → independent **reads** | `context.cc:352,355` |
| `use_collective_write` | `true` | **`true`** | `LORRAX_PHDF5_COLLECTIVE_WRITES=0` | flipped `false`→`true` on **2026-07-27**, commit `d40e7fd` |
| `coll_metadata` | `false` | `false` | `LORRAX_PHDF5_COLL_META=1` | non-collective metadata lets `H5Dcreate`/extend bypass the collective driver |
| `dedup_replicas` | `true` | `true` | `LORRAX_PHDF5_DEDUP_REPLICAS=0` | drops all-but-one writer of a replica group's identical hyperslab |
| `align_threshold` | 1 MiB | **4 MiB** | `LORRAX_PHDF5_ALIGN_MB` (default `4`) | `context.cc:367,372` |
| `align_length` | 1 MiB | **4 MiB** | same knob — set together | `context.cc:373` |

Alignment is deliberately *not* tied to the striping unit, and is measured
non-load-bearing on this filesystem: at 16 × 1 MiB striping, `ALIGN_MB` of
4 / 1 / 0 gave 0.830 / 0.809 / 0.813 GiB/s at 1 node and 2.975 / 2.883 /
2.915 at 4 nodes — all inside ±1.5 % repeat noise (job 56389339). It stays
at 4 rather than becoming a knob that must be kept in sync with a value it
does not depend on.

**Boolean grammar.** All the flags parse through `env_flag`
(`context.cc`), which mirrors Python's
`file_io/_slab_io_mpi_host._env_flag` exactly: unset or exactly-empty →
the default; otherwise trimmed, lowercased, and **true only for
`1` / `true` / `yes` / `on`**. Everything else is false — including `off`,
`no`, and any typo. There is no "unrecognised value" diagnostic, so
`LORRAX_PHDF5_COLLECTIVE_WRITES=ture` silently disables collective writes.
One grammar, every writer.

Two of these are correctness, not tuning:

* **`dedup_replicas`** is *required* under collective writes. Overlapping
  hyperslab selections are undefined in HDF5. Under independent writes the
  same flag is pure waste-removal.
* **`use_collective_write`** decides whether a PMI-mismatched launch fails
  loudly or corrupts silently — see §7.

History of the write default, since prose elsewhere in the tree still
carries the old value: introduced `3a7f2e5` (2026-04-17); set `false` in
`d37c47a` (2026-04-20, "independent writes by default; Cray MPICH now
works"); set `true` in `d40e7fd` (2026-07-27).

> **Known stale comments in source (reported, not edited — those files are
> owned elsewhere).** `src/ffi/cpp/stage/phdf5_stage_cray.sh:13-19` and
> `src/ffi/cpp/run_shifter.sh:100-104` both still assert that the phdf5
> default is *independent* writes with non-collective metadata. That has
> been false since 2026-07-27. The first of the two is actively harmful —
> see §7.

---

## 7. Failure modes, and how to tell them apart

### 7a. The PMI-flavour mismatch — the one that gives wrong answers

**Measured**, job 56389339, 4 nodes / 16 ranks, launched `srun --mpi=pmi2`
(the wrong PMI flavour for Cray MPICH; the right one is `cray_shasta`):

```
MPI_Comm_size(MPI_COMM_WORLD) == 1   on every rank
jax.process_count()           == 16
→ 8 hostile geometries written and read back BIT-EXACT, rc=0,
  file 16-striped and fully populated, no warning anywhere.
```

Nothing in the stack noticed, and the reason is worth internalising:
`ffi.io.open_file` checks `p*q == jax.process_count()`, and
`shard_index.h::validate_shard_encoding` checks
`prod(mesh_shape) == ctx->world_size` — but `ctx->world_size` *is*
`jax.process_count()`, passed down from Python. **Both checks compare JAX to
JAX and agree.** The MPI communicator `H5Dwrite` actually collects on was
never consulted.

It "worked" only because the hyperslabs happened to be disjoint, so there
was no collective handshake left to fail. Change the geometry so two ranks
touch one HDF5 chunk, or let one rank's metadata update race another's, and
it is silent corruption with rc=0.

The guard is in `file_io/_slab_io_ffi.py`: ask MPI once, at the first
collective open. The verdict is rank-invariant by construction, so it
refuses everywhere or nowhere — the only kind of refusal a collective
tolerates.

### 7b. The ROMIO OOM — and the documented remedy that makes it worse

With collective writes ON (the current default) that same mismatched launch
does **not** survive. But it does not diagnose either. It dies as:

```
Out of memory in .../ad_cray/ad_cray_write_coll.c, line 669
… MPI_Abort … "HDF5: infinite loop closing library"
```

That is the *same* line `stage/phdf5_stage_cray.sh:13-19` documents as a
known Cray-MPICH `≥1 GB/rank` collective-buffer OOM — whose documented
remedy is `LORRAX_PHDF5_COLLECTIVE_WRITES=0` / `LORRAX_PHDF5_INDEPENDENT=1`.

**That documented remedy is wrong in both halves.**

* `LORRAX_PHDF5_COLLECTIVE_WRITES=0` does exactly what it says — and
  converts the loud crash into the silent-wrong-answer regime of §7a.
* `LORRAX_PHDF5_INDEPENDENT=1` forces independent **reads**
  (`context.cc:352,355`). It does nothing to the write path at all, so
  against a write-side OOM it is simply inert.

The misdiagnosis is not hypothetical — the tree points straight at it, and
the half of the advice that *does* something is the half that hides the
bug.

How to tell the two apart before reaching for the knob:

| | genuine Cray collective-buffer OOM | PMI-flavour mismatch |
|---|---|---|
| `MPI_Comm_size(MPI_COMM_WORLD)` | == `jax.process_count()` | **1**, on every rank |
| per-rank aggregate | ≳ 1 GB | any size |
| independent writes | genuinely fixes it | **hides it** |

Check the world size **first**. `LORRAX_PHDF5_REQUIRE_MPI_WORLD=1` makes an
unprobeable world a refusal rather than a warning;
`LORRAX_PHDF5_SKIP_MPI_WORLD_CHECK` disables the guard entirely and should
be treated as a debugging-only escape hatch, never a remedy.

Not reproduced at 512 MiB/rank in the 2026-08-05 Perlmutter campaign, and
the collective default was revalidated there (job 56389339): keep it at `1`.

### 7c. SONAME aliases that look like two ABIs and are not

`libmpi_gnu_123.so.12` is **a deliberate symlink, not a second MPI**.
`src/ffi/cpp/stage/phdf5_stage_cray.sh:128-130` creates one shim per
cray-pe compiler-specific SONAME — `libmpi_gnu_{91,110,123}.so.12` — all
pointing at the container's generic MPICH-ABI library,
`/opt/udiImage/modules/mpich/libmpi.so.12` (`SHIM_TARGET`, line 92). The
loader follows the symlink at container startup; every variant is MPICH 4.x
`libmpi.so.12` underneath. **One object, not two.**

A 2026-08-05 report of a two-MPI-ABI defect on the CUDA leg was **retracted
on 2026-08-06** for this reason. The `ldd` behind it was run on a **login
node**, where the closure is incomplete and four dependencies show
`not found`.

> **Method note, because this cost real time.** `ldd` on a login node does
> not describe what a container run loads. Run link-closure checks inside
> the container, on a compute node, through `lx run`. A `not found` in a
> login-node `ldd` is evidence of nothing.

### 7d. Bounds-check asymmetry — the hang with no traceback

Bounds are tested once, on the *logical* slab `offset + valid_shape`, which
is a replicated quantity, so every rank reaches the same verdict. Testing a
rank-local advanced offset splits the ranks into those that refuse and those
that enter the collective, stranding the communicator with no HDF5 error and
no traceback (**measured**: 306 s hang at P=4; silent 420 s timeout on the
read path). No rank may skip a collective because of its own error: record
it, participate in the teardown, then raise. See `decisions.md` 2026-08-04.

---

## 8. Hard invariants

Checked at `886139f` and re-checked at `8789131` on 2026-08-06; not aspirational.

1. **Registered FFI custom-call target names do not change.** The full set
   is in `src/ffi/common/ffi_loader.py` (`_CUDA_TARGET_SYMBOLS`,
   `_HOST_TARGET_SYMBOLS`). Refactors move files; they never edit a target
   string or a C++ handler symbol.
2. **Env knob spellings do not change.** Aliases only.
3. **Built `.so` names and consumed paths stay stable, or their consumers
   are updated in the same commit.** `liblorrax_ffi.so` /
   `liblorrax_ffi_host.so`.
4. **A stage script refuses an unstated environment fact rather than
   guessing it.** `phdf5_stage_cray.sh` refuses an unset `HDF5_DIR`
   (lines 57-70) and an unset `MPICH_DIR` (73-84); `build.sh` refuses an
   unset nvhpc stage (54-91). Each of those was a hardcoded guess until
   2026-08-05/06, and each guess had gone stale: the HDF5 fallback named
   1.12.2.9 while the host build uses `cray-hdf5-parallel/1.14.3.7`. What
   is staged is what every later build *links against*, and a wrong guess
   does not fail at link time — it fails much later, as a wrong answer or a
   hang, with nothing on disk recording the substitution.

Invariant 4 is the generalisation of §4 and §7b: **in this layer, a
substituted default is a wrong answer with a long fuse.**

---

## 9. Deletion candidates and open work

* **cusolvermp** (~2800 LOC): 11 import sites outside `src/ffi` at
  `886139f`. The distributed CPU story is ScaLAPACK; the GPU story is
  SLATE. Deletion removes the `auto|cusolvermp` spelling from the linalg
  backend grammar — an input-deck surface, so it needs the deprecation
  window plus a GPU run proving SLATE covers the eigh/LU tiers cusolvermp
  served.
* **cublasmp** (~1450 LOC): 4 import sites (`bse/vq_interp.py`,
  `bandstructure/htransform.py`, tests). The fused W-solve path has no
  measured replacement, so this leg stays until a GPU gate exists.
* **Shim deletion**: blocked on consumer migration (§1).
* **FFT, remaining items**: `fftw_init_threads` / `plan_with_nthreads` on
  non-MKL engines under the existing `LORRAX_FFT_FFI_THREADS` grammar; the
  `fftwf_` twin table if BSE adoption wants c64; and only then gating the
  shard_map-interior `local_*fftn3` entry so the FFI can back
  `make_sharded_ifftn_3d`. That last flip is a **measurement**, not a move,
  and until it happens that layer stays XLA by ruling.
* **CUDA-leg re-certification**: outstanding. See §3.

Parity gate for any engine swap, stated once with its class: value-level,
**relative 1e-12** (the Σ-path class, `flat_k_fft_service.md` §7). Not
bit-exactness — swapping engines changes the arithmetic ordering, where bit
equality is not promised. And not the 1e-16 figures: those are *measured*
unit residuals sitting at the c128 ULP, where a threshold tests nothing.
