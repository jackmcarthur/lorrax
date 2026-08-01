# Porting LORRAX's FFI subpackages to a new cluster

Covers `ffi.cusolvermp`, `ffi.phdf5`, and `ffi.slate`. All three link
into one `liblorrax_ffi.so` and share a build system
(`src/ffi/cpp/CMakeLists.txt`).

The host-platform families (`ffi.scalapack`, `ffi.mklblas`, `ffi.mklfft`
and the host half of `ffi.phdf5` / `ffi.slate`) link into
`liblorrax_ffi_host.so` from `src/ffi/cpp/CMakeLists.txt`,
built by `config/frontera/build_ffi_host.sh`. Read §0 next before you
start hunting for a vendor library.

## 0. Which of these are vendor-swappable, and which are not

**The thing to know before porting: most of these families call a
published API, so the port is a link line, not a code change.** Only two
are tied to one implementation.

| Family | API the LORRAX handler calls | Implementations of that API | Swappable? |
|---|---|---|---|
| `scalapack` | ScaLAPACK + C-BLACS Fortran ABI — **exactly eleven symbols**, hand-declared in [`cpp/scalapack/blacs_grid.h`](cpp/scalapack/blacs_grid.h): `pzheevd_`, `pdsyevd_`, `pzgetrf_`, `pdgetrf_`, `pzgetrs_`, `pdgetrs_`, `numroc_`, `descinit_`, `Csys2blacs_handle`, `Cblacs_gridinit`, `Cblacs_gridinfo`.  That list is not a summary: it is the complete non-libc undefined-symbol set of `solve_lu_ffi.cc` + `eigh_ffi.cc`, measured. | Intel MKL (`libmkl_scalapack_lp64` **+** `libmkl_blacs_<mpi>_lp64` — measured 8 + 3 = 11), Cray LibSci (`libsci_*_mpi_*`), netlib ScaLAPACK, AMD AOCL.  **Not SLATE** — see §0b. | **Yes.** Pass `-DLORRAX_SCALAPACK_LIBRARIES="<whole link line>"` and the MKL probe is skipped entirely. No source change. |
| `mklblas` | CBLAS — `cblas_dgemm` / `cblas_sgemm` / `cblas_zgemm` / `cblas_cgemm`, plus the OPTIONAL `cblas_?gemm_batch` extension | Intel MKL, Cray LibSci, OpenBLAS, BLIS, ATLAS | **Yes.** Point `LORRAX_CBLAS_DIR` (or `CRAY_LIBSCI_PREFIX_DIR`) at a prefix with `cblas.h`. The batched extension is looked up at run time, per precision; a BLAS without it silently takes the portable plain-GEMM loop and says so once. |
| `phdf5` | HDF5 C API + MPI-IO (`H5Pset_fapl_mpio`, `H5Dwrite`, …) | any parallel HDF5, over any MPI | **Yes.** `-DHDF5_ROOT=<prefix>`; see §"phdf5 stack choice". |
| `mklfft` | Intel **DFTI** descriptor API (`DftiCreateDescriptor`/`SetValue`/`Compute*`) | **Intel MKL only.** | **No.** DFTI is Intel's. A LibSci-only site simply gets no host FFT handler: `LORRAX_FFT_FFI` refuses at run time and the default XLA lowering stands — a lost optimisation, never a wrong answer. The portable target if this ever needs fixing is the FFTW3 *guru* interface (`fftw_plan_guru_dft`), which takes the arbitrary input/output strides this handler depends on and is implemented by FFTW, cray-fftw, AOCL — and by MKL itself, which exports `fftw_plan_many_dft`/`fftw_execute` from the same library as the `Dfti*` entry points. That is a rewrite of the descriptor construction, not a link-line change, and it is unmeasured. |
| `slate` | `slate::` C++ templates (`slate::potrf`, `slate::trsm`, `slate::heev` over `slate::HermitianMatrix`) | **ICL SLATE only.** | **No** — a C++ template library has no ABI a second vendor could implement. |
| `cusolvermp` | `cusolverMp*` — opaque handle + grid + descriptor objects, `int64_t` dimensions, caller-supplied device *and* host workspace, an NCCL communicator | **NVIDIA only.** | **No.** Despite solving the same problems, it shares no symbol with ScaLAPACK (`nm -D libcusolverMp.so.0` finds none of the eleven names above). It is a different API for the same operation, not a second ScaLAPACK. |
| `cublasmp` | `cublasMp*` — same shape of API as cuSOLVERMp | **NVIDIA only.** | **No.** Likewise exports no PBLAS symbol (`pzgemm_`, `pztrsm_`). |
| `cufft` | cuFFT plan API | **NVIDIA only.** | **No.** |

## 0b. SLATE's ScaLAPACK overlay — the portability question, measured

**The question this section answers.** Is SLATE a ScaLAPACK front end that is
*agnostic to what is underneath* — netlib/MKL on a CPU cluster, CUDA on an
NVIDIA machine, ROCm on an AMD one — so that LORRAX can write its distributed
linear algebra once against ScaLAPACK names and have it follow us to a very
different platform?

**Short answer: the ABI half of that is real; the layout half is not, yet.**
The overlay exports exactly the names LORRAX calls, and exports the *same*
names whether SLATE was built `gpu_backend=none` or `=cuda`. But on the mesh
shapes LORRAX actually runs (`p>1, q>1`) it returns answers that are wrong by
~15% with no error, because of a fixable mismatch between how LORRAX assigns
shards to ranks and what the shim assumes. The fix is on LORRAX's side and
needs no upstream patch. Until it lands, the provenance guard in
`cpp/scalapack/blacs_grid.h` refuses the overlay by default.

SLATE ships this compatibility layer as `scalapack_api/` in its source tree,
built as `libslate_scalapack_api.so`; it re-defines the ScaLAPACK entry
points and forwards them to `slate::`, and its README documents `LD_PRELOAD`
interception.

Build it with [`cpp/stage/slate_build_scalapack_api.sh`](cpp/stage/slate_build_scalapack_api.sh)
and `nm -D` the result (SLATE v2025.05.28, measured 2026-07-31 against the
eleven names above):

| | LORRAX's eleven |
|---|---|
| **DEFINED (6)** | `pzheevd_` `pdsyevd_` `pzgetrf_` `pdgetrf_` `pzgetrs_` `pdgetrs_` |
| **UNDEF (2)** — it *calls* them | `numroc_` `Cblacs_gridinfo` |
| **absent (3)** | `descinit_` `Csys2blacs_handle` `Cblacs_gridinit` |

Plus four more it consumes: `Cblacs_get`, `Cblacs_pcoord`, `Cblacs_pinfo`,
`indxl2g_`.  So the overlap is **100 % of the operations LORRAX asks
ScaLAPACK to perform and 0 % of the grid/descriptor infrastructure**: it is
a layer that must sit ON TOP of a real ScaLAPACK+BLACS (upstream's own link
line is `-lslate_scalapack_api -lslate -lmkl_scalapack_lp64 …`), never a
provider you can name in `LORRAX_SCALAPACK_LIBRARIES` by itself.

**That "overlay, not provider" property is not a portability problem.**  The
five names it does not define are grid/descriptor bookkeeping
(`descinit_`, `numroc_`, `Cblacs_gridinit`, `Cblacs_gridinfo`,
`Csys2blacs_handle`), and every ScaLAPACK-bearing platform ships them: Intel
MKL (measured here — `libmkl_scalapack_lp64` defines 8 of LORRAX's 11 and
`libmkl_blacs_<mpi>_lp64` the other 3), Cray LibSci, AMD AOCL, netlib.  Any
machine that can host LORRAX at all already has that layer.  The overlay
just goes in front of it.

**It runs, and that is the problem.**  Measured end to end (job 7883874/
7883880; the overlay in `LD_PRELOAD` under LORRAX's *unmodified* ScaLAPACK
handler, 1×1 CPU mesh, waiver set):

| n | eigh `max\|Δλ\|` | `‖AZ−ZΛ‖/‖A‖` | `‖ZᴴZ−I‖` | LU `‖AX−B‖/‖B‖` |
|---|---|---|---|---|
| 32 | 1.24e-14 | 1.44e-15 | 8.43e-15 | 1.23e-14 |
| 64 | 2.49e-14 | 1.79e-15 | 1.49e-14 | 3.95e-14 |
| 128 | 1.21e-13 | 2.17e-15 | 2.65e-14 | 4.45e-14 |
| 512 | 5.76e-13 | 3.21e-15 | 7.99e-14 | 7.38e-13 |

Machine precision, every size, both operations, no crash — but every one of
those numbers is from a 1×1 mesh, which is the one geometry that cannot
expose the defect below.

### The 2-D result, and the fix

Job 7883978: 4 ranks, 2×2 mesh, n=64, `LD_PRELOAD` overlay vs plain MKL, and
the **mesh's device order as the only other variable**.

| mesh device order | provider | eigh `‖AZ−ZΛ‖/‖A‖` | `‖ZᴴZ−I‖` | LU `‖AX−B‖/‖B‖` |
|---|---|---|---|---|
| **C** — what LORRAX ships | MKL | 9.6e-16 ✅ | 1.1e-14 | 4.3e-16 ✅ |
| **C** | SLATE overlay | 1.518e-01 ❌ | 6.978 | 1.547e-01 ❌ |
| **F** — Fortran | MKL | 1.518e-01 ❌ | 6.978 | 1.547e-01 ❌ |
| **F** | SLATE overlay | 1.4e-15 ✅ | 1.6e-14 | 4.3e-16 ✅ |

The two failing rows agree to four digits.  That is what makes this a proof
rather than a coincidence: **the entire defect is one permutation**, applied
once too few times or once too many.  The shims hard-code `MPI_COMM_WORLD`
and want shard `(mx,my)` on rank `mx+my*p` (`GridOrder::Col`); LORRAX puts it
on `mx*q+my` and compensates inside `SlateCtx`.  On a 4×1 mesh the remap is
the identity and the overlay's LU is correct (4.33e-16), while its eigh
refuses loudly — SLATE's `heev` requires a **square process grid**
(`heev.cc:102`).

So the honest status is neither "it works" nor "do not use it":

* **the overlay is correct on a 2-D mesh**, at machine precision, once the
  rank↔shard mapping matches;
* **the fix is entirely on LORRAX's side** — no patched dependency, which
  matters given the hard rule against customised deps;
* **it is not free**: the same reordering that makes the overlay right makes
  MKL wrong, so the device order becomes a *provider-dependent* choice and
  the two cannot share a mesh as things stand.

The clean version is not to flip the mesh order but to stop *assuming* it:
`blacs_ctxt_for` hard-codes the C-order remap, so deriving the permutation
from the mesh's actual device order would make MKL correct under both orders
and turn F-order into a free knob that enables the overlay.  That touches
every consumer of the mesh and needs its own correctness gate — it is a
design decision, not a drive-by.  Reproducer, with MKL controls on both
orders: `wk_REL/harness/slalias_mesh.sbatch`.

### The second defect, which the fix above does not touch

**`*info` is hard-wired to 0** ("todo: extract the real info from
getrf/heevd" — `scalapack_getrf.cc:147`, `scalapack_heevd.cc:113`), so a
singular pivot or a non-converged block is reported as a clean success.  How
much that costs depends on the caller: LORRAX's ζ-fit already adds a
per-q ridge `1e-12·|tr(L)|/n` before every distributed LU
(`isdf/core.py` `_dist_ridged_lu`) precisely to keep near-null modes above
the LU stability floor, so the singular case has an independent guard.  A
non-converged `pXheevd` has none.

**A third constraint, structural.**  `slate::heev` requires
`GridOrder::Col` — asserted in five places (`he2hb.cc:83`,
`stedc_{deflate,merge,secular,sort}.cc`) — and a square process grid.
`getrf`, `getrs`, `potrf` and `trsm` carry no `GridOrder` constraint at all
(zero mentions in their sources), so they are the portable subset; eigh is
the constrained one.

A third reason was expected and **did not hold** — it is worth recording
because it points somewhere useful.  `slate_pheevd` is a direct call to
`slate::heev`, the routine bug L-2 refuses.  Through this path it was
correct at every size above; in the *same job, same mesh, same sizes*,
`ffi.slate`'s own host handler SIGSEGVed (rc 139) at n=32 and n=64.  Same
`libslate.so.2`, same MKL, same process image ⇒ **L-2 is in LORRAX's
`cpp/slate/host_ffi.cc` call path, not in SLATE.**  See
`docs/dev/linalg_ffi.md` and the reproducer
`wk_REL/harness/slalias_l2.sbatch`.

Because nothing on the Python side can observe an interposition —
`ffi_loader` keys only on LORRAX's own handler symbols, so
`resolve_backend('eigh','scalapack')` still returns `scalapack`, and
`has_target('lorrax_scalapack_eigh')` was measured still answering `True`
with the overlay live — the detection is in C++:
`cpp/scalapack/blacs_grid.h` resolves the provider of the routine it is
about to call with `dlsym` + `dladdr` and **refuses** when it is the
overlay, naming all three defects.  `LORRAX_SCALAPACK_ALLOW_SLATE_API=1`
downgrades the refusal to one loud line, for deliberately measuring it.

**Upstream does not build this target from CMake.**  SLATE v2025.05.28's
`CMakeLists.txt` has the whole `scalapack_api` block commented out with
`# todo: requires ScaLAPACK` — so it is missing from a CMake install
*regardless* of `-DSCALAPACK_LIBRARIES`, and the empty setting LORRAX's
SLATE build uses is not what excludes it.  Only the GNUmakefile route (or
the script above) produces it.

### The device is an env var, and its default is a silent demotion

This is the part of the portability claim that actually holds cleanly, and
the part with a trap in it.

`slate::` dispatches CPU / CUDA / ROCm from the **build**
(`-Dgpu_backend=none|cuda|hip`), and the *caller* selects the execution
target at run time.  Every shim reads `TargetConfig::value()`
(`scalapack_slate.hh:144-188`) and forwards it as
`{slate::Option::Target, target}` into the `slate::` call —
`scalapack_heevd.cc:105`, `getrf.cc:94`, `potrf.cc:100`, and the rest.  So
`pzheevd_` really is one symbol over three backends, with no source change
at the call site.  That is the thing worth having.

The trap: `TargetConfig`'s constructor sets `Target::HostTask` and only
overrides it if `SLATE_SCALAPACK_TARGET` is set (`devices` / `hosttask` /
`hostnest` / `hostbatch`).  **A `gpu_backend=cuda` SLATE therefore runs on
the CPU by default and says nothing.**  Under this project's doctrine that
is a demotion, and a demotion must announce itself, so
`cpp/scalapack/blacs_grid.h` prints what the variable resolves to alongside
the provider whenever the overlay is live — including the unset case, which
is the one that looks like success and is not.  If LORRAX ever adopts this
route for real, it must *set* the target per platform rather than inherit
it.

### What was measured about the GPU half, and what was not

Frontera's shipped SLATE installs are `gpu_backend=none` (`nm -D -C` on
`slate_builds/{cpu,cpu_seq}/install/lib64/libslate.so.2` finds zero
CUDA/HIP/cuBLAS/rocBLAS symbols), so `devices` had nothing to dispatch to.
`wk_REL/harness/slalias_gpu.sbatch` therefore builds one:
`gpu_backend=cuda`, sm_75, on an `rtx` node (needs `module load gcc/12.2.0`
— gcc 8.3 cannot compile SLATE 2025.05's `set_lambdas.cc`).  Results, job
7884073:

**Measured, and it is the core of the portability claim.**

* The **same source** gives **0** CUDA/HIP/cuBLAS/rocBLAS symbols at
  `gpu_backend=none` and **15** at `=cuda` (`cudaMalloc`,
  `cudaLaunchKernel`, `cudaMemcpyAsync`, …).  The backend really is a build
  switch.
* The `scalapack_api` overlay built against the **CUDA** SLATE exports the
  **identical** 11-name classification (6 DEFINED / 2 UNDEF / 3 absent) and
  a full exported ABI differing from the CPU build's by only `_init` and
  `_fini` — linker artifacts, not API.  **One ABI, two backends.**
* A caller that mentions only ScaLAPACK names (`Cblacs_*`, `descinit_`,
  `numroc_`, `pzgetrf_`, `pzgetrs_`, `pzheevd_`) links and runs unchanged
  against MKL and against the overlay, at the same accuracy
  (`‖AX−B‖/‖B‖` 1.041e-15 vs 1.040e-15).

**NOT measured: that kernels executed on the device.**  `hosttask` and
`devices` came out indistinguishable:

| cell | `pzgetrf_`+`pzgetrs_` | residual | `pzheevd_` | GPU samples |
|---|---|---|---|---|
| MKL | 0.414 s | 1.041e-15 | 1.27 s | **0** |
| overlay, `TARGET=hosttask` | 0.999 s | 1.040e-15 | 462 s | 1560 |
| overlay, `TARGET=devices` | 0.955 s | 1.043e-15 | 465 s | 1582 |

Two things went wrong with this cell and both are worth knowing:

* **The residency instrument cannot discriminate.**  The MKL zero-control
  held, but the *`hosttask`* run also showed the process resident on the
  GPU (116–218 MiB) — a CUDA-built SLATE takes a device context and
  workspace whatever the target.  Presence is not evidence of device
  *compute*.  Discriminate on `utilization.gpu` next time.
* **The problem size was wrong for the question.**  The caller uses
  `nb = n`, i.e. **one tile**, on a 1×1 grid.  SLATE offloads per tile, so
  there was nothing to distribute and no reason to expect a difference.
  The `devices` run is not evidence *against* GPU dispatch either — the
  experiment simply could not see it.  Redo with `nb ≈ n/8` and a larger n.

**So: treat "SLATE dispatches to the GPU" as a source reading**
(`TargetConfig` → `Option::Target` → `slate::`, cited above) **backed by a
proven backend-independent ABI — not as an end-to-end measurement.**  What
*is* measured end to end is that the same ScaLAPACK-only caller works
identically against MKL and against a CUDA-built SLATE.

**A cost warning from that same job.**  `pzheevd_` at n=1024 on **one**
rank: MKL 1.27 s, SLATE `hosttask` **462 s**.  A 1×1 grid is SLATE's worst
case (its heev gathers the band to rank 0 and runs `hb2st`/`stedc` there),
so this is not a scaling claim — but it is a hard reminder that SLATE's
eigh is not a drop-in at small rank counts, on top of the `nprow == npcol`
requirement.

### What LORRAX would have to change to be "ScaLAPACK names only"

Today, per operation:

| op | LORRAX calls | already ScaLAPACK names? |
|---|---|---|
| eigh | `pzheevd_` / `pdsyevd_` | **yes** |
| solve_lu | `pzgetrf_` + `pzgetrs_` | **yes** |
| cholesky | `slate::potrf` (`ffi.slate`) | no — would be `pzpotrf_`/`pdpotrf_` |
| trsm | `slate::trsm` | no — would be `pztrsm_`/`pdtrsm_` |
| batched potrf/trsm | `slate::` in a C++ loop | no — ScaLAPACK has no batched API; the loop is the same either way |

The two that are not yet ScaLAPACK-named cost **no new link dependency**:
`libmkl_scalapack_lp64` already defines `pzpotrf_`, `pdpotrf_`, `pztrsm_`,
`pdtrsm_` and `pzpotrs_` (measured with `nm -D`), and it is already on this
library's link line for the eigh/LU handlers.  Writing those two handlers in
`cpp/scalapack/` — the same shape as `solve_lu_ffi.cc`, reusing the BLACS
grid and descriptor code already in `blacs_grid.h` — would leave LORRAX
calling **one API for all four distributed operations**, which is the state
in which "move to an AMD machine" is a link line plus an env var.

Current production exposure is small: the only host use of `ffi.slate` is
the opt-in `distributed_cholesky = slate` route (`isdf/core.py`), and its
trsm back-solve is not even wired yet ("a perf follow-up", same file).

**On CUDA this is a genuine choice, not a free win.**  `cusolvermp` and
`cublasmp` export *no* ScaLAPACK symbol (§0), so the ScaLAPACK-names-only
world reaches NVIDIA through SLATE, not through them — trading NVIDIA's
native libraries for one portable API.

**The tax, measured through the same entry points.**  Job 7884073 ran one
caller that mentions only ScaLAPACK names, n=1024, 1 rank, same node, and
swapped only which library answered:

| provider | `pzgetrf_`+`pzgetrs_` | residual |
|---|---|---|
| MKL | **0.414 s** | 1.041e-15 |
| SLATE overlay, `TARGET=hosttask` | **0.999 s** | 1.040e-15 |

Same answer, **0.41× the speed** — which corroborates the ~0.47× CPU figure
quoted for SLATE against LORRAX's native paths, and is the cleanest form of
the trade: identical API, identical accuracy, ~2.4× the time.  (The eigh
residual printed by that standalone caller is not usable — it does not
symmetrise its matrix before calling `pzheevd_`; the eigh accuracy numbers
in this document come from `probes/scalapack_alias_run.py`, which does.)

Routing through `scalapack_api` adds nothing measurable on top of SLATE
itself: the shim wraps the caller's buffer with `fromScaLAPACK` (no copy),
does ~4 BLACS queries, and for `getrs` broadcasts the pivot array once per
tile-row — O(1) and O(n/nb) against an O(n³/P) factorisation.  That last
sentence is a source reading, not a measurement; the 0.41× above is the
whole SLATE+shim stack against MKL, so it bounds both together.

The ζ-fit comparison in [`docs/dev/linalg_ffi.md`](../../docs/dev/linalg_ffi.md)
is the other half of the picture and points the other way for LU: ScaLAPACK
beats the native replicated path 6.1×–11.5× on a multi-node 4×4 mesh, while
`cholesky` via SLATE is 0.6× native inside the μ=3000 chain and 0.14× on a
single node.  There is no single ratio — it depends on the operation and on
rank spread.  Halving throughput to gain portability is a real trade and
belongs to the owner, not to this file.

## 0c. What actually builds on YOUR machine — the four cells, measured

The point of calling one published API is that a user with a different set of
packages still gets a working library, minus exactly the pieces their
packages cannot supply, **announced at configure time**.  These four cells are
run as a matrix by `wk_REL/harness/slalias_gate.sbatch` cell 4; the rows below
are its output (job 7883900), not a design intent.

| Your machine has… | Handlers you get | What you lose | To get it back |
|---|---|---|---|
| **SLATE + MKL** (the Frontera production cell) | all 15: phdf5×4, slate×5 (+2 lifecycle), scalapack eigh + fused LU, MKL FFT×2, GEMM | — | — |
| **MKL, no SLATE** | 10: phdf5×4, scalapack eigh + fused LU, FFT×2, GEMM, and no `cpp/slate/host_ffi.cc` | `ffi.slate` potrf/trsm/batched → `distributed_cholesky = slate` refuses at resolve time | build a `gpu_backend=none` SLATE, point `-DLORRAX_SLATE_HOST_INSTALL_DIR` at the prefix containing `lib64/cmake/slate` |
| **Some other ScaLAPACK** (`-DLORRAX_SCALAPACK_LIBRARIES=…`; LibSci / AOCL / netlib), no MKL headers | 6: phdf5×4, scalapack eigh + fused LU | MKL's DFTI FFT and the CBLAS GEMM handler. `LORRAX_FFT_FFI=1` / `LORRAX_BANDS_GEMM_FFI=1` **refuse at run time**; the default XLA lowering stands, so this is a lost optimisation, never a wrong answer | GEMM: any CBLAS — set `LORRAX_CBLAS_DIR` or `CRAY_LIBSCI_PREFIX_DIR` to a prefix with `cblas.h`. FFT: `-DLORRAX_MKL_ROOT` **in addition to** your ScaLAPACK line — DFTI is Intel-only and has no second vendor |
| **Neither** | 4: phdf5 only | all distributed linalg + FFT + GEMM | install any ScaLAPACK+BLACS (see §0) — that one dependency restores eigh, LU **and** the FFT/GEMM handlers that ride its link line |

Three things a user should know about that table:

* **The groups are independent.**  Before 2026-07-31 the ScaLAPACK, GEMM and
  FFT resolutions were nested inside `if(slate_FOUND)`, so "no SLATE" silently
  cost all three even though none of them calls SLATE.  Verified per cell by
  which `.cc` files the generated build actually contains.
* **Every skip announces itself at configure time**, with the variable that
  fixes it, and again at run time as a refusal.  Read the configure log first.
* **Nothing at run time can observe which vendor supplied ScaLAPACK or CBLAS**
  — `ffi_loader` keys only on LORRAX's own handler symbols.  That is what makes
  the API-level aliasing real, and it is also why an interposed SLATE overlay
  is invisible without the `dlsym`/`dladdr` provenance check in
  `cpp/scalapack/blacs_grid.h` (§0b).

Two consequences worth internalising before you port:

* **`mklblas` and `mklfft` are historical names, not vendor
  assertions.** `mklblas` builds and runs against any CBLAS; and the
  `mklfft` *Python* module also drives the **cuFFT** handlers on the CUDA
  platform, under the same `jax.ffi` target names. Do not read a vendor
  out of a directory name.
* **Nothing at run time can observe which vendor supplied ScaLAPACK or
  CBLAS.** `ffi/common/ffi_loader.py` keys only on LORRAX's own handler
  symbols (`ScalapackEighHostFfi`, …). A ScaLAPACK vendor named in a
  docstring is decoration; the build is the only authority.

The mapping from "unresolved symbol" back to "dependency that was not
resolved", and the CMake variable that fixes each one, is kept in the
**"HOST NUMERICAL LIBRARIES"** comment block in
[`cpp/CMakeLists.txt`](cpp/CMakeLists.txt) and
restated in the header of `config/frontera/build_ffi_host.sh`. Read the
configure log first — every group announces what it resolved or why it
was skipped.

## Hard requirements

| Requirement             | Minimum | Notes |
|-------------------------|---------|-------|
| NVIDIA GPU              | CC ≥ 7.0 | A100 / H100 tested. |
| CUDA toolkit            | 12.0    | 12.9 is what we link against. |
| NCCL                    | 2.18    | Ships with CUDA 12.4+; JAX bundles it. |
| JAX with `jax.ffi`      | 0.5     | Must match CUDA major. `nvcr.io/nvidia/jax:25.04-py3` is our container. |
| NVHPC SDK (cuSOLVERMp)  | 22.7    | 25.5 validated. Only the `libcusolverMp` + `libcal` subset is needed. |
| Parallel HDF5           | 1.12    | Either Cray HDF5 (MPICH ABI) or a MPI-linked conda-forge build. |
| SLATE (+ blaspp/lapackpp)| any     | Built from source against the target MPI + libsci/BLAS. |

MPI is required only for `ffi.phdf5` and `ffi.slate`; `ffi.cusolvermp`
bootstraps via JAX's KV store + NCCL, no MPI.

## Build system ([`cpp/CMakeLists.txt`](cpp/CMakeLists.txt))

Autodetection probes for each dep, overridable with CMake `-D...` or
env vars:

| Dep          | Auto-probe                                                   | Override                                                 |
|--------------|--------------------------------------------------------------|----------------------------------------------------------|
| cuSOLVERMp   | `$NVHPC_ROOT`, `$HPCSDK_ROOT`, `/opt/nvidia/hpc_sdk/…`, `/lorrax_nvhpc/…`, `/usr/local/cuda-12.X/…` | `-DNVHPC_ROOT=…` or `-DCUSOLVERMP_{INCLUDE,LIB}_DIR=…` |
| Parallel HDF5| `$HDF5_ROOT`, `$HDF5_DIR`, `/lorrax_phdf5`                   | `-DHDF5_ROOT=…`                                          |
| MPI          | `$LORRAX_MPI_INCLUDE_DIR` / `$LORRAX_MPICH_LIB_DIR`          | `-DLORRAX_MPI_INCLUDE_DIR=…`, `-DLORRAX_MPICH_LIB_DIR=…` |
| SLATE        | `/global/homes/<u>/software/slate/install`                   | `-DLORRAX_SLATE_INSTALL_DIR=…` or env                    |
| NCCL header  | `/usr/include/nccl.h`, then `$NVHPC_ROOT/comm_libs/.../nccl` | `-DNCCL_INCLUDE=…`                                       |

Build: `bash src/ffi/cpp/build.sh` (calls CMake + ninja). Rebuild
from scratch with `--fresh`.

## Staging vendor deps (one-time per cluster)

Many clusters put system libs under `/opt/...`, which containers can't
bind-mount freely. We copy the minimal subsets into `$SCRATCH` (or
equivalent bindable location) via scripts under `src/ffi/cpp/stage/`:

| Script                             | Produces                                     | Bind-mounted to |
|------------------------------------|----------------------------------------------|-----------------|
| [`cpp/stage/cusolvermp_stage_nvhpc.sh`](cpp/stage/cusolvermp_stage_nvhpc.sh) | `libcusolverMp.so.0`, `libcal.so.0`, cuSOLVERMp+NCCL headers (~80 MB) | `/lorrax_nvhpc` |
| [`cpp/stage/phdf5_stage_cray.sh`](cpp/stage/phdf5_stage_cray.sh)         | Cray HDF5 1.12 + MPICH-ABI shim | `/lorrax_phdf5` |
| [`cpp/stage/phdf5_stage_openmpi.sh`](cpp/stage/phdf5_stage_openmpi.sh)   | conda-forge HDF5 (OpenMPI)      | `/lorrax_phdf5` |
| [`cpp/stage/slate_stage_cray.sh`](cpp/stage/slate_stage_cray.sh)         | Cray libsci + `libmpi_gtl_cuda.so.0` + xpmem + lustreapi | `/lorrax_slate` |

SLATE itself is built from source; `stage_cray.sh` only copies the
runtime libs SLATE links against.

## Runtime

The LORRAX Lmod module wires the full `srun + select_gpu + shifter
+ in_container` invocation. On a ported cluster, install the module
via `config/<cluster>/install.sh` (see [`config/README.md`](../../config/README.md)),
then `lxrun <cmd>` — no per-call env juggling.

What the module sets:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false     # share VRAM with NCCL/cuSOLVERMp
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async  # cudaMallocAsync — see note below
HDF5_USE_FILE_LOCKING=FALSE             # Lustre compatibility
MPICH_GPU_SUPPORT_ENABLED=1             # GPU-Direct RDMA
LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0   # CUDA-12 shim; see §Slate-specific below
```

> **`platform` is not cudaMallocAsync**, and `TF_GPU_ALLOCATOR` does
> nothing.  This block used to read
> `XLA_PYTHON_CLIENT_ALLOCATOR=platform  # = cudaMallocAsync (via
> TF_GPU_ALLOCATOR)`; both halves are false.  The CUDA plugin carries three
> distinct allocators — BFC (unset/`default`/`bfc`), plain `cudaMalloc`
> (`platform`), and `cudaMallocAsync` (`cuda_async`) — and `TF_GPU_ALLOCATOR`
> is a TensorFlow variable that JAX never reads (a cell setting only it was
> byte-identical to the unset cell, job 7882442).  `platform` also zeroes
> `memory_stats()`, which every LORRAX memory report reads.  Porting a
> cluster: pick `cuda_async`, and on sm_75 pair it with the command-buffer
> `XLA_FLAGS` restriction (`config/frontera/ffi_env.sh`).  Full table in
> [`docs/environment/overview.md`](../../docs/environment/overview.md) §2.1.

`CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` is set per-rank by
`select_gpu.sh` (invoked from `lxrun`). JAX callers must pass
`local_device_ids=[0]` to `jax.distributed.initialize()` when
`process_count > 1` — the sandbox tests auto-detect via the length
of `CUDA_VISIBLE_DEVICES`.

## Checklist for a new cluster

1. **NVHPC SDK**: `module spider nvhpc`. Run
   [`stage_nvhpc.sh`](cpp/stage/cusolvermp_stage_nvhpc.sh) to copy the
   `libcusolverMp`+`libcal` subset into `$SCRATCH`.
2. **Parallel HDF5**: pick a stack.
   - Cray MPICH + cray-hdf5-parallel → [`stage_cray.sh`](cpp/stage/phdf5_stage_cray.sh).
   - Anything else MPI → [`stage_openmpi.sh`](cpp/stage/phdf5_stage_openmpi.sh)
     (edit the conda-forge URL to match your MPI).
3. **SLATE**: clone [icl-utk-edu/slate](https://github.com/icl-utk-edu/slate),
   build against the target MPI + BLAS, install under
   `$HOME/software/slate/install`. Stage Cray runtime libs via
   [`cpp/stage/slate_stage_cray.sh`](cpp/stage/slate_stage_cray.sh).
4. **Configure and install the module**: copy
   `config/perlmutter/` → `config/<cluster>/`, edit `site_config.sh`
   (especially `LORRAX_SLURM_{ACCOUNT,QOS,CONSTRAINT}`,
   `LORRAX_SHIFTER_MODULES`, `LORRAX_MPI_TYPE_DEFAULT`,
   `LORRAX_NVHPC_SUBPATH`, `LORRAX_MPICH_CONTAINER_DIR`), run
   `bash config/<cluster>/install.sh`.
5. **Build the .so**: `bash src/ffi/cpp/build.sh` (inside shifter
   via `src/ffi/cpp/run_shifter.sh`).
6. **Verify**:
   ```bash
   lxalloc
   lxrun python3 -u -m common.cusolvermp_eigh_test
   lxrun python3 -u -m common.slate_cholesky_trsm_test -n 256 --dtype c128
   lxrun python3 -u -m common.phdf5_multi_offset_test
   ```
   Expected: all PASS at machine precision (~1e-13 for C128 eigh).

For non-Shifter runtimes (Singularity/Apptainer): swap the
`shifter ...` invocation inside the module's shell functions for
`apptainer exec --nv --bind ... image.sif ...`. Everything else
(SLURM flags, `select_gpu.sh`, `in_container.sh`, LD_LIBRARY_PATH
composition) is runtime-agnostic.

## Cluster-specific: NERSC Perlmutter

Shifter forbids `--volume` sources outside `/pscratch` and a handful
of other paths, which is why every "stage" script copies to
`$SCRATCH` first. The `nvcr.io/nvidia/jax:25.04-py3` container does
**not** ship NVHPC SDK, Cray MPICH, Cray HDF5, or SLATE — all four
come in via bind-mount.

`lxrun` uses `--mpi=cray_shasta` (not `pmi2` or `pmix`) — both of
those silently give singleton `MPI_COMM_WORLD` with
`shifter --module=mpich`.

`libmpi_gtl_cuda.so.0` is the CUDA GPU-Direct RDMA transport for Cray
MPICH. Shifter's `--module=mpich` bind-mounts a copy built against
CUDA 11, needing `libcudart.so.11.0` not in our container.
`stage_cray.sh` for slate also copies the CUDA-12 version, and
`lxrun` `LD_PRELOAD`s it so the loader binds that one first.

## phdf5 stack choice

The unified default is **Cray MPICH** on Perlmutter (and any other
Cray site). Historical context: we ran on OpenMPI / HPC-X for the
first few months because Cray MPICH's collective-write path
(`ad_cray_write_coll.c:669`) OOMs at ≥ 1 GB/rank. The 2026-04-20 fix
was to flip the FFI's default writes to `H5FD_MPIO_INDEPENDENT` and
disable collective metadata ops, which bypasses the Cray collective
write driver entirely. Result at 4 GPU / 4.29 GB C128:
**3.79 GB/s Cray vs 3.06 GB/s OpenMPI**, and small-write latency
within noise. OpenMPI path is still viable — select via
`LORRAX_PHDF5_MPI_STACK=openmpi` (affects build-time and
`run_shifter.sh`); requires `--mpi=pmix` + container's HPC-X OpenMPI.

## phdf5 tuning knobs

Env vars read at `open_file` time:

| Var                              | Default      | Effect |
|----------------------------------|--------------|--------|
| `LORRAX_PHDF5_CB_WRITE`          | _unset_ (ROMIO auto) | Forwarded to `romio_cb_write` when set. |
| `LORRAX_PHDF5_DS_WRITE`          | _unset_ (ROMIO auto) | Forwarded to `romio_ds_write` when set. |
| `LORRAX_PHDF5_CB_BUFFER_SIZE`    | _unset_ (ROMIO auto) | Per-aggregator CB buffer size (bytes) when set. |
| `LORRAX_PHDF5_CB_NODES`          | _unset_ (ROMIO auto) | ROMIO aggregator count when set. |
| `LORRAX_PHDF5_CB_PER_NODE`       | _unset_      | Cray MPICH: aggregators/node (`cb_config_list=*:N`). |
| `LORRAX_PHDF5_STRIPE_COUNT`      | `16`         | Lustre `striping_factor` hint. |
| `LORRAX_PHDF5_STRIPE_SIZE_FS`    | `4M`         | Lustre `striping_unit` hint (`lfs -S` spelling; legacy byte-valued `LORRAX_PHDF5_STRIPE_SIZE` also honoured). |
| `LORRAX_PHDF5_ALIGN_MB`          | `4`          | `H5Pset_alignment` threshold (MiB). |
| `LORRAX_PHDF5_INDEPENDENT`       | `0`          | 1 → force **reads** to independent. |
| `LORRAX_PHDF5_COLLECTIVE_WRITES` | `1`          | 0 → independent writes (pre-AI behaviour; the historical Cray `ad_cray_write_coll.c` OOM caution lives in context.cc). |
| `LORRAX_PHDF5_DEDUP_REPLICAS`    | `1`          | 0 → every replica rank writes its identical copy (UB under collective; debug only). |
| `LORRAX_PHDF5_COLL_META`         | `0`          | 1 → re-enable collective metadata ops. |

The ROMIO `cb_*`/`ds_*` rows were FORCED defaults (`enable`/`disable`/
64 MiB/`world_size`) until 2026-07-27 — a Perlmutter/OpenMPI-era tuning.
On Frontera/Intel-MPI forcing `romio_cb_write=enable` measured *slower*
than ROMIO's automatic policy under collective transfers (scorecard AI),
so all four are now pure pass-throughs, matching the Python
`_slab_io_mpi_host` writer.  Rule of thumb: bump `STRIPE_COUNT` to 32-64
for writes > 10 GB. If the enclosing directory has an explicit
`lfs setstripe` layout, the `striping_*` hints are no-ops (directory
wins), and an EXISTING file keeps its inode's layout (mode='w' unlinks
for exactly this reason).

Baked-in DCPL: `H5D_FILL_TIME_NEVER` + `H5D_ALLOC_TIME_EARLY` +
`H5F_LIBVER_LATEST`.

## Gotchas

- **`Failed to parse ib device list`**: harmless libcal warning on
  Perlmutter; libcal probes InfiniBand transports it won't use
  (we route CAL through NCCL).
- **`NCCL error 1 unhandled cuda error` → `cusolverMpSyevd status=7`**:
  NCCL starved of VRAM. Check `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  is set (the module sets it; don't override with `true` + a fixed
  `MEM_FRACTION`).
- **`MPI_COMM_WORLD` size 1 inside `--module=mpich`**: wrong
  `--mpi=` flavour. Use `cray_shasta`.
- **CUDA driver vs toolkit mismatch**: `nvidia-smi`'s "CUDA Version"
  is the driver's maximum supported toolkit; must be ≥ what we
  linked (12.9 currently).
- **Multi-node NCCL**: needs cluster-specific NCCL env (e.g.
  `NCCL_NET_PLUGIN=ofi` on Perlmutter, `NCCL_IB_HCA=...` on IB
  fabrics). Not validated here.

## References

- [NVIDIA cuSOLVERMp](https://docs.nvidia.com/cuda/cusolvermp/)
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)
- [JAX FFI](https://jax.readthedocs.io/en/latest/ffi.html)
- [NERSC parallel HDF5 tuning](https://docs.nersc.gov/performance/io/library/)
- [ROMIO hints](https://wordpress.cels.anl.gov/romio/2008/09/26/system-hints/)
- [NERSC Shifter mpich module](https://docs.nersc.gov/development/shifter/how-to-use/)
