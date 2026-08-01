# Adding a new distributed-LA FFI subpackage

Worked example: `src/ffi/cusolvermp/` → `src/ffi/elpa/`.

The `cusolvermp` and (hypothetical) `elpa` subpackages share ~90% of
their structure.  This doc walks the 10% that changes when porting.

## Anatomy of an FFI subpackage

```
src/ffi/<lib>/              python side
├── __init__.py            re-exports the public op(s)
├── context.py             Python: lazy per-process singleton
└── eigh.py                Python: shard_map(P('x','y')) wrapper
src/ffi/cpp/<lib>/          C++ side (the one C++ tree)
├── ctx.h                  per-process state struct
├── context.cc             comm + library handle setup (collective)
├── <lib>_interface.h      per-dtype thin wrappers over the library
└── eigh_ffi.cc            Impl<T> / Dispatch / XLA_FFI_DEFINE_HANDLER_SYMBOL
```

## What's shared across subpackages (do not reinvent)

Under `src/ffi/cpp/common/` (C++) and `src/ffi/common/` (python):

- **`cpp/common/ffi_helpers.h`** — `FFI_RETURN_IF_ERROR`, `LORRAX_CUDA_CHECK`,
  `LORRAX_LIB_CHECK`, `cuda_error`.
- **`cpp/common/scalapack_descriptor.h`** — `numroc(...)`,
  `local_tile_shape(...)`, `one_tile_per_rank(...)`.  ScaLAPACK
  block-cyclic math that cuSOLVERMp and ELPA both need.
- **`cpp/CMakeLists.txt`** — THE one build entry point (both platform
  legs, `-DLORRAX_FFI_PLATFORM=cuda|host`); autodetects NVHPC on the
  CUDA leg.  Add your new .cc to `LORRAX_FFI_SOURCES`.
- **`broadcast.py`** — `broadcast_bytes(buf, key=...)`: byte-exact
  broadcast from rank 0 via the JAX distributed KV store.  Works for
  any opaque library handle (ncclUniqueId, MPI_Comm split, PMI).
- **`ffi_loader.py`** — `get_lib()` loads the .so and registers all
  FFI targets.  Add your target name + symbol to `_FFI_TARGET_SYMBOLS`.

## Checklist for a new subpackage

### (1) Decide the public signature

Start with a docstring and signature.  For a new solver
`distributed_potrf(A, *, mesh, lower=True)`, mirror the structure of
`cusolvermp/eigh.py:distributed_eigh`.  The required pieces:

- Validate mesh has axes `('x', 'y')`.
- Validate matrix shape, dtype, divisibility.
- Call `get_lib()` once (idempotent).
- Call `get_or_init_context(mesh)` once (idempotent per mesh).
- Wrap `jax.ffi.ffi_call(<target>, <shapes>)` in `shard_map` with the
  appropriate `in_specs` and `out_specs`.

### (2) Pick your `PartitionSpec`s

| Op family | `in_specs` | `out_specs` |
|---|---|---|
| eigh (A → W, Q) | `P('x','y')` | `(P(), P('x','y'))` — W replicated, Q sharded |
| potrf (A → A lower-triangular) | `P('x','y')` | `P('x','y')` — in-place |
| solve (A, B → X) | `(P('x','y'), P('x',None))` | `P('x',None)` — RHS replicated on 'y' |
| qr (A → Q, R) | `P('x','y')` | `(P('x','y'), P('x','y'))` |

Pick the out_specs that match the library's natural output layout, not
what the caller wants — callers can reshard downstream.

### (3) Pick the library's target name

One string per FFI target.  Convention:
`lorrax_<lib>_<op>`.  Examples:

- `lorrax_cusolvermp_eigh`
- `lorrax_elpa_eigh`
- `lorrax_cusolvermp_potrf`

Register in `common/ffi_loader.py`'s `_FFI_TARGET_SYMBOLS` dict,
mapping the target name to the C++ symbol (`EighMpFfi`, `EighElpaFfi`,
etc.).

### (4) C++ Impl/Dispatch/Ffi

Lift the structure from `cpp/cusolvermp/eigh_ffi.cc`:

```cpp
template <typename T>
static ffi::Error PotrfImpl(int64_t n, int64_t mb, int64_t nb,
                            bool lower,
                            cudaStream_t xla_stream,
                            ffi::ScratchAllocator& scratch,
                            LorraxElpaCtx* ctx,        // <-- library-specific
                            T* d_A)                     // in-place
{
    // --- cross-stream wait, workspace scratch.Allocate, run the solve ---
}

static ffi::Error PotrfDispatch(cudaStream_t stream,
                                ffi::ScratchAllocator scratch,
                                ffi::AnyBuffer A,
                                ffi::Result<ffi::AnyBuffer> A_out,
                                int64_t n, int64_t mb, int64_t nb,
                                int64_t ctx_handle, bool lower)
{
    auto* ctx = reinterpret_cast<LorraxElpaCtx*>(ctx_handle);
    // ... dtype switch ...
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PotrfElpaFfi, PotrfDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::AnyBuffer>()            // A
        .Ret<ffi::AnyBuffer>()            // A_out (alias of A for in-place)
        .Attr<int64_t>("n")
        .Attr<int64_t>("mb")
        .Attr<int64_t>("nb")
        .Attr<int64_t>("ctx_handle")
        .Attr<bool>("lower"));
```

### (5) Per-dtype wrapper header (`<lib>_interface.h`)

```cpp
namespace lorrax_ffi::elpa::lib {
    template <typename T> struct RealOf                    { using type = T; };
    template <> struct RealOf<std::complex<float>>         { using type = float; };
    template <> struct RealOf<std::complex<double>>        { using type = double; };
    template <typename T> using RealOf_t = typename RealOf<T>::type;

    template <typename T> inline int
    PotrfInplace(elpa_t h, T* d_A, int64_t n, int64_t lda);
    template <> inline int PotrfInplace<double>(elpa_t h, double* A, int64_t n, int64_t lda) {
        int err; elpa_cholesky_d(h, A, &err); return err;
    }
    template <> inline int PotrfInplace<std::complex<double>>(...) { ... }
}
```

### (6) Context / comm bootstrap (the only meaningfully different piece)

For ELPA you'll write `context.cc` that:

1. Initializes MPI if not already (mpi4py's `MPI_Init_thread` covers it
   from Python — or call it from C++ with `MPI_THREAD_MULTIPLE`).
2. Builds an `MPI_Comm` covering all JAX processes — ELPA takes one.
3. Calls `elpa_init()`, `elpa_allocate()`, `elpa_set("mpi_comm_rows",
   ...)`, `elpa_set("mpi_comm_cols", ...)`, `elpa_setup()`.
4. Stores the `elpa_t` handle on `LorraxElpaCtx`.

For the unique-handle broadcast (if needed), reuse:

```python
from ..common.broadcast import broadcast_bytes
uid = np.zeros(size, dtype=np.uint8)
if jax.process_index() == 0:
    fill_uid(uid.ctypes.data)
uid = broadcast_bytes(uid, key="lorrax_ffi/elpa/handle/v0")
```

### (7) Link the library in CMake

In `src/ffi/cpp/CMakeLists.txt` (CUDA leg), next to `find_library(CUSOLVERMP_...)`:

```cmake
find_library(ELPA_LIBRARY elpa
    PATHS "${ELPA_ROOT}/lib" "${ELPA_ROOT}/lib64"
    NO_DEFAULT_PATH)
```

Add an `ELPA_ROOT` cache var with autodetection from `$ELPA_DIR` /
`$ELPA_ROOT` env vars (same pattern as `NVHPC_ROOT`).

### (8) Python test

Mirror `src/common/cusolvermp_eigh_test.py`:

- Build a random matrix on rank 0, shard it.
- Call your new op.
- Compare eigenvalues (or residuals for solve/potrf) to
  `np.linalg.<op>` on the gathered matrix.

### (9) Runtime env

Document the required env vars (ELPA likely wants
`OMP_NUM_THREADS=<cores-per-gpu>`, maybe `ELPA_DEFAULT_2stage_cpu=yes`,
and whatever MPI launcher your cluster uses).  Add them as a row in
`src/ffi/AGENTS.md`'s "Run the <lib> test" section.

## Don't factor prematurely

The instinct to share code between two library subpackages is a
trap when only two exist.  Keep each library's Python files flat
(~100 lines of copy-with-edits); that lets a reviewer diff the two to
see exactly what the libraries differ on, rather than unwinding a
callback-based factory.  Extract into `common/` only when a third
library would copy the same code a third time.
