# Plan — unified sharded-slab I/O for `gw_jax` sigma writes

> **2026-07-31 note:** every reference to `_accumulate_kij_stream` /
> `kij_stream` below is moot — the streamed accumulation mode was removed
> (owner-approved deprecation sweep); only the `write_sigma_omega_h5`
> targets remain relevant. Variable names in the snippets are historical;
> current Hartree APIs are documented in `docs/theory/hartree.md`.

## Scope (narrowed after the second audit)

Two write sites in `gw_jax` — both already routed through
[`file_io/sigma_output.py:write_sigma_omega_h5`](/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_C/src/file_io/sigma_output.py#L350).
That's the entire target of v1:

1. **Static COHSEX** — three `(nk, nb, nb)` C128 arrays (`sig_sx`,
   `sig_coh`, `sig_h`), produced sharded under `mesh_xy` and then flattened
   to host via `np.array(...)` at
   [`gw_jax.py:500-501`](/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_C/src/gw/gw_jax.py#L500).
2. **Dynamic Σ_c(ω)** — one `(n_omega, nk, nb, nb)` C128 tensor
   (`sigma_c_omega`), produced sharded, then either:
   - **accum mode**: accumulated on host, written in one shot
     ([`ppm_sigma.py:828`](/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_C/src/gw/ppm_sigma.py#L828));
   - **stream mode**: read-modify-write into an open HDF5 dataset per
     ω-batch, single-process only today
     ([`ppm_sigma.py:862-868`](/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_C/src/gw/ppm_sigma.py#L862)).

Other call sites surveyed earlier (zeta_q, tagged_arrays, kin_ion, BSE
eigenvectors) will migrate onto the same API later but are deferred —
`gw_jax` is the driver we care about now.

## Backends

A single `use_ffi_io: bool = False` flag selects between two backends
behind one identical Python surface:

| Backend | Default? | What it does |
|---|---|---|
| **allgather+rank0** | yes | `jax.experimental.multihost_utils.process_allgather(A, tiled=False)` → rank 0 `h5py.File` write.  Today's pattern, consolidated. |
| **ffi** | opt-in | `ffi.phdf5.{open_file,write_sharded_slab,read_sharded_slab,close_file}` — collective MPI-IO, N-D hyperslabs. |

Default `False` is byte-identical to today's code — just pulled out of
six files into one helper.  `True` routes through the already-shipped
FFI read/write primitives (extended to N-D).

## File organisation

```
src/
├── file_io/
│   ├── slab_io.py            ← NEW.  Public: SlabIO, write_slab,
│   │                            read_slab, accumulate_slab.
│   │                            Dispatches on use_ffi_io.
│   ├── _slab_io_allgather.py ← NEW.  Default-path impl: allgather +
│   │                            rank-0 h5py.  No mpi4py / FFI deps.
│   ├── _slab_io_ffi.py       ← NEW.  FFI-path impl: thin wrapper over
│   │                            ffi.phdf5 (+ N-D offset computation
│   │                            from jax.sharding).  Lazy-imported.
│   └── sigma_output.py       ← EDIT.  write_sigma_omega_h5 becomes a
│                                thin caller into SlabIO; no h5py
│                                imports, no np.array gathers.
└── ffi/phdf5/                 ← unchanged public surface; only
    cpp/{write,read}_ffi.cc     generalised to N-D hyperslabs.
```

Rationale:

- `src/file_io/` already owns HDF5 I/O helpers; putting `SlabIO` there
  keeps the domain separation clean.
- The FFI stays an implementation detail — `file_io.slab_io` imports
  `ffi.phdf5` lazily, only when `use_ffi_io=True`.  A user whose
  `liblorrax_ffi.so` isn't built can still run the default path.
- Splitting the two backends into separate modules keeps each file
  small and readable; `slab_io.py` itself is just the public surface
  + dispatch.

## Public API

```python
from file_io.slab_io import SlabIO, write_slab, read_slab, accumulate_slab
```

### Single-shot writes / reads

For call sites that do exactly one dataset op:

```python
write_slab(path, ds_name, A, *,
           offset=None,             # default: (0,...,0)
           global_shape=None,       # default: A.shape
           mesh=None,               # required only if A is sharded
           mode="a",                # file open mode
           use_ffi_io=False)

read_slab(path, ds_name, *,
          shape, dtype,
          offset=None,
          mesh=None,
          use_ffi_io=False) -> jax.Array
```

`A` is an N-D `jax.Array`; its sharding is introspected from
`A.sharding.spec`.  Rank-replicated arrays degrade cleanly to a
rank-0-only write (no allgather needed).

### Multi-write lifecycles

For call sites that open a file once and hit it repeatedly
(the dynamic-sigma stream path, future zeta_q migration):

```python
with SlabIO(path, mode="w", mesh=mesh, use_ffi_io=use_ffi_io) as io:
    io.create_dataset("sigma_c_kij_ev",
                      shape=(n_omega, nk, nb, nb),
                      dtype=jnp.complex128,
                      chunks=(batch, min(4, nk), nb, nb),
                      attrs={"layout": "omega,k,i,j"})
    for omega_batch in batches:
        io.write_slab("sigma_c_kij_ev", contrib,
                      offset=(omega_batch.start, 0, 0, 0))
```

### Accumulation (stream mode)

For the Σ_c(ω) stream path specifically:

```python
io.accumulate_slab("sigma_c_kij_ry", contrib, offset=(omega_idx, 0, 0, 0))
```

Semantics: `dset[offset : offset+count] += contrib`.  Implementation:

- **allgather+rank0**: `buf = f[ds][...]`, `buf += allgather(contrib)`,
  `f[ds][...] = buf` — same as what `_accumulate_kij_stream` does
  today at [`ppm_sigma.py:862-868`](/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_C/src/gw/ppm_sigma.py#L862).
- **ffi**: `read_sharded_slab → jnp.add → write_sharded_slab` at the
  same offset.  Collective, multi-process-safe.  Lifts the
  single-process restriction that prevents parallel stream mode
  today.

## Migration of `gw_jax` (what changes)

### `file_io/sigma_output.py` — `write_sigma_omega_h5`

Becomes a ~15-line wrapper over `SlabIO`:

```python
def write_sigma_omega_h5(
    filepath, omega_ev, sigma_total_kij_ev=None,
    *, sigma_c_kij_ev=None, sigma_sx_kij_ev=None,
    hartree_kij_ev=None, mesh=None,
    use_ffi_io: bool = False,
):
    with SlabIO(filepath, mode="w", mesh=mesh, use_ffi_io=use_ffi_io) as io:
        io.write_attr("omega_ev", omega_ev)           # tiny; rank-0 direct
        for name, A in [("sigma_total_kij_ev", sigma_total_kij_ev),
                        ("sigma_c_kij_ev",     sigma_c_kij_ev),
                        ("sigma_sx_kij_ev",    sigma_sx_kij_ev),
                        ("hartree_kij_ev",     hartree_kij_ev)]:
            if A is not None:
                io.write_slab(name, A, global_shape=A.shape)
```

The k-chunked loop + `np.asarray` gathers disappear: either the
allgather backend handles them internally (once, in one place), or the
FFI streams the shards straight to Lustre.

### `gw/gw_jax.py` — caller

```python
# before (gw_jax.py:486-490)
write_sigma_omega_h5(
    sigma_omega_h5_path, ppm_options.omega_grid_ev, None,
    sigma_c_kij_ev=ryd2ev * sigma_c_omega,
    sigma_sx_kij_ev=ryd2ev * sig_sx,
    hartree_kij_ev=ryd2ev * sig_h)

# after — adds use_ffi_io + mesh, removes the `if meta.rank == 0:` guard
#         (the helper handles rank-0 itself)
write_sigma_omega_h5(
    sigma_omega_h5_path, ppm_options.omega_grid_ev, None,
    sigma_c_kij_ev=ryd2ev * sigma_c_omega,
    sigma_sx_kij_ev=ryd2ev * sig_sx,
    hartree_kij_ev=ryd2ev * sig_h,
    mesh=mesh_xy,
    use_ffi_io=config.use_ffi_io)
```

### `gw/ppm_sigma.py` — stream mode

`_accumulate_kij_stream` (which currently force-falls-back to accum
when `process_count() > 1`) becomes:

```python
def _accumulate_kij_stream(global_idx, contrib_batch):
    accumulate_slab(kij_stream_path, "sigma_c_kij_ry",
                    contrib_batch, offset=(global_idx[0], 0, 0, 0),
                    mesh=mesh_xy, use_ffi_io=use_ffi_io)
```

Removes the single-process restriction at
[`ppm_sigma.py:835-838`](/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_C/src/gw/ppm_sigma.py#L835).

### Flag plumbing

One edit to the `GwJaxConfig` / options bundle that `gw_jax.main`
already reads:

```python
class GwJaxConfig:
    ...
    use_ffi_io: bool = False
```

Threaded into `write_sigma_omega_h5` and `compute_sigma_omega`.  The
CLI flag becomes `--use-ffi-io`; default is `False` so existing runs
are unaffected.

## FFI generalisation (what the C++ side needs)

Current [`write_ffi.cc`](/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_C/src/ffi/cpp/phdf5/write_ffi.cc) hardcodes 2-D
with `(rank/q, rank%q)` offsets.  To support N-D:

- Replace `Attr<int64_t>("n_rows")` + `Attr<int64_t>("n_cols")` with
  two `Attr<Span<int64_t>>` arrays: `offset[]` and `count[]`.  Both
  N-long, caller-computed.
- Replace the two stack `hsize_t[2]` arrays with caller-sized
  heap/stack buffers.  `H5Sselect_hyperslab` already takes arbitrary
  rank.
- `(rank/q, rank%q)` rank-offset logic moves from C++ into Python
  (`_slab_io_ffi.py` derives per-rank offsets from
  `A.sharding.spec` + caller-provided `offset`).  Cleaner anyway —
  JAX knows its sharding; C++ shouldn't re-derive it.

Same for `read_ffi.cc`.  The existing 2-D `write_sharded_slab` /
`read_sharded_slab` either get deprecated in favour of the N-D
primitive, or kept as thin wrappers that set `N=2` offsets.

## What's in / out of scope

**v1 (this refactor)**

- `file_io/slab_io.py` + the two backend modules.
- `write_sigma_omega_h5` rewritten on top of it.
- `_accumulate_kij_stream` rewritten on top of `accumulate_slab`.
- `use_ffi_io` flag threaded from `GwJaxConfig` to the two call sites.
- N-D FFI extension on `write_ffi.cc` / `read_ffi.cc`.
- Round-trip test at n=128 (rank-0 h5py + FFI parity check).

**Deferred (explicitly)**

- zeta_q writer in `isdf_fitting.py` and its async writer thread.
- tagged_arrays checkpoint / restart paths.
- kin_ion, BSE eigenvector, get_DFT_mtxels, gw_init sigma writes.
  All migrate onto the same API in a follow-up once v1 is proven.
- Ragged WFN / epsilon G-vector reads — separate design.

## Risks

- **Rank-offset computation from sharding spec.**  The N-D extension
  assumes a block-partitioned `NamedSharding` whose shard positions
  can be derived from `A.sharding.spec` + the global shape.  Any
  exotic (e.g. strided) sharding would error.  We add a
  `sharding_to_block_offsets(A)` helper with a clear error message
  for non-block layouts.  `sig_sx` / `sigma_c_omega` in gw_jax are
  straight block-partitioned so this isn't hit in v1.
- **Accumulate collective under MPI-IO.**  `H5Dread` →
  `jnp.add` → `H5Dwrite` at the same hyperslab must be fenced
  correctly between ranks.  The existing FFI
  `read_sharded_slab` already does `sync_global_devices` around the
  H5Dread; composing it with `write_sharded_slab` gives us the same
  ordering guarantee for free.
- **FFI path instability.**  Today the Cray MPICH stack is still
  unstable (see PORTING.md); OpenMPI is the verified default.  We
  keep that default.  `use_ffi_io=True` must be a user-made opt-in.
- **Allgather memory peak on rank 0.**  The default path still
  replicates the array on rank 0 once per write.  No worse than
  today.  A single `write_sigma_omega_h5` call gathers at most
  `n_omega × nk × nb² × 16 B` — for typical `(1024, 64, 100, 100)`
  this is 100 GB and already needs to be chunked; `SlabIO` exposes
  `k_chunk_size` (matches the existing `write_sigma_omega_h5`
  kwarg) and will do per-chunk allgathers internally.

## Rough effort

| Step | Time | GPU alloc? |
|---|---|---|
| `slab_io.py` + `_slab_io_allgather.py` | 2 h | no |
| N-D FFI extension + `_slab_io_ffi.py`   | 3 h | no (login-node build) |
| Rewrite `write_sigma_omega_h5` + `_accumulate_kij_stream` | 2 h | no |
| `use_ffi_io` flag in `GwJaxConfig` + CLI | 1 h | no |
| Round-trip test (rank-0 vs FFI parity) at n=128 | 1 h | yes (30 min, 4 GPU) |
| Full-size bench on real gw_jax run  | 1 h | yes (2 h, 16 GPU) |

One day of focused work, one 30-min GPU alloc for correctness, one
longer alloc for the performance number.  No changes needed in the
existing 2-D FFI call sites (phdf5_write_test etc.) — they stay
working through the deprecated-but-kept wrappers.
