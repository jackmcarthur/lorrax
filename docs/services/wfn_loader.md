# wfn_loader — ψ(G) loading behind one door

`services/wfn_loader/` is the single entry point for reading a BerkeleyGW
`WFN.h5`: the header, G-vector and FFT-box index tables, and ψ(G) itself,
band-sharded (`load`), per rank (`load_process_local`) or band-chunked
(`bands`). It is independently installable (src-layout) and depends on
`lxkit`, JAX, NumPy and h5py. Collective I/O is reached only through the
`slab_io` door at load time; the loader holds no FFI target name, context
handle or phdf5 call.

There are two supported spellings: `import wfn_loader` and top-level names
(after runtime startup, or `ffi._services.ensure_on_path()` for a
direct-library caller, has selected the service set), or
`from file_io import WfnLoader` / `WFNReader` (an alias of the same class
object). `from wfn_loader.loader import …` fails `tests/test_layering.py`, and
`src/file_io/wfn_loader.py` must not exist
(`tests/test_service_path_bootstrap.py::test_the_retired_shim_files_are_gone`).

## API

| name | contract |
|---|---|
| `WfnLoader(path, *, mesh=None, backend='auto', qe_schema=None)` | Open the file, pick the backend (§ Backends), and run the 2c DFT time-reversal check. `qe_schema=None` discovers the QE schema at the first `symmetry()`; an explicit schema must authenticate or the call refuses. |
| `load(*, bands, k='full_bz', sharding=None, bispinor=False, bispinor_lift='raw')` | ψ for a band window and k-set: `(n_k, nb_padded, ns, ngkmax)` complex128, the band axis padded to the mesh and, at P > 1, sharded `P(None, ('x','y'), None, None)`. Every rank must request the same window. |
| `load_process_local(*, bands, k='full_bz', bispinor=False, bispinor_lift='raw')` | This process's window only, on one device: `nb = b_hi − b_lo` exactly, no padding, no collective; ranks may request different windows. |
| `bands(b_lo, b_hi, *, chunk, ...)` | Band-chunked iterator over `load`. |
| `KSpec` | `'ibz'`, `'full_bz'`, an explicit list of full-BZ rows, or `IBZRows(rows)` (raw file rows, no unfolding). |
| `kvecs(k=...)` | Fractional k paired row-for-row with `gvecs`: the file's `kpoints` for raw rows, `SymMaps.unfolded_kpts` in request order for full-BZ rows. Form `k+G` and Bloch phases from these two tables only; rebuilding k from the integer grid can pick another reciprocal-lattice image. |
| `gvecs(k=...)`, `ngk_valid(k=...)` | `(n_k, ngkmax, 3)` int32 Miller indices and the valid lengths. Pad rows hold the FFT-box pad sentinel, never zeros; the pair is the contract. |
| `box_index(k=...)`, `box_index_dev(k=..., mesh=...)` | The one ψ(G)↔box table, `(n_k, ngkmax)` int32 (`common.gvec_fft_box.build_sphere_box_index`): entry `[k, g]` is the flat C-order box cell of slot `g`, and pad slot `g` holds the distinct out-of-box value `n_rtot + g`, so a `mode='drop'` scatter ignores it and a `mode='fill'` gather returns zero. Cost `n_k·ngkmax·4` bytes. Cached per (k-set, `fft_grid`); the device copy is replicated and placed once per (k, mesh). |
| `full_k_parent_groups(full_k=None)` | Stable O(n_k) grouping of requested full-BZ rows by raw IBZ parent. |
| `unfold_parent_to_full_k(parent_psi, *, parent, full_k, bispinor=False)` | The canonical unitary/antiunitary action applied to one already-loaded parent row, so a star is realized with one child workspace and no parent re-read. |
| `full_k_box_index_one_dev(full_k)`, `ibz_box_index_one_dev(parent)` | One `(1, ngkmax)` replicated sphere index, built on device from the current parent G row, for strict one-k streams. |
| `symmetry()` | The loader's `SymMaps`; it consumes `trs_holds` (§ Contract). |
| `trs_holds`, `trs_reference` | The 2c occupied-subspace TRS verdict and its receipt. `density_symmetry` is a compatibility alias of the receipt. |
| `qe_symmetry_binding`, `qe_symmetry_diagnostic` | Authenticated per-operation unitary/antiunitary provenance, or the reason `SymMaps` uses the announced WFN-only fallback. |
| `occupations_are_exact_integer`, `occupation_state_capacity`, `physical_density_band_stop`, `physical_density_occupations(*, k)` | The occupation table's integer test (smearing tails count), electrons per unit occupation, the exclusive band stop, and the `(n_k, n_b)` occupation operand for a physical density. |
| `adopt_mesh(mesh)` | Late mesh binding, § Contract. May raise. |
| `release_read_staging()` | Close the collective read handle, freeing its host staging buffer (about ψ(G)/P per rank after the parent read). |
| `close()`, context manager | Releases both file handles and propagates close failures; only destructor cleanup suppresses them, with a diagnostic. |
| header surface, `path`, `kpt_starts` | The `MfHeader` fields (`nkpts`, `nbands`, `nspinor`, `kgrid`, `fft_grid`, `bvec`, `sym_matrices`, `translations`, …) and derived `nelec`, `vbm`, `cbm`, `efermi`, `atom_crys`. |
| `get_gvec_nk(ik)` | Unpadded `(ngk, 3)` G list of one k, read from the raw slab without the padding logic; no `src` caller (a test oracle and `scripts/checks` use it). |
| `WfnProvenance`, `read_wfn_provenance(path)` | Header-only identity and occupation view; no G or ψ payload. |
| `uniform_band_windows(b_lo, b_hi, width)` | Fixed-width `(lo, mask)` windows covering a band range once; the last window overlaps and its 0/1 mask removes the overlap, so every consumer compiles one FFT shape. |

`common.psi_G_store.load_parent_psi_G` is the one-read consumer: it reads each
band chunk once through `load`, moves it from band shards to G-slot shards
with one all-to-all, and samples the centroid faces from the local G slots, so
the G-slot store (`P(None, None, None, ('x','y'))`) and the faces come from
one pass.

`kweights` has one value per raw WFN k row. The loader validates only shape
`(nkpts,)`, finite nonnegative values and a positive sum, and never spreads
weights over stars or decides whether the rows are an IBZ wedge or the full
grid; consumers decide that against authenticated symmetry metadata
(`centroid.sampling_metric.full_k_quadrature_weights`).

## Contract

* **Padding is a conjunction.** Band-axis pad rows of ψ are zero. G-axis pad
  columns of ψ are zero **and** the matching `gvecs` rows hold the pad
  sentinel (the Nyquist-corner Miller index, which no physical G occupies).
  The zero makes a slot inert; the sentinel makes a dropped mask detectable
  (`common.gvec_fft_box.refuse_padded_gvecs_without_mask`) instead of
  aliasing pad slots onto Γ. Consumers must still carry `ngk_valid`.
* **Backends are byte-identical** for the same request (`np.array_equal`, no
  tolerance, on hostile geometry and on both platforms). That is why
  `LORRAX_WFN_BACKEND` (owned by [`env_vars.md`](../dev/env_vars.md)) may
  force one.
* **Refusals.**
  - `backend='phdf5_host'` is deleted and refuses; an unknown backend refuses.
  - `phdf5` without a mesh refuses.
  - `auto` at P > 1 with a mesh and no phdf5-capable library on either
    platform refuses, quoting each platform's probe reason and naming
    `LORRAX_WFN_BACKEND=eager`. It never demotes silently.
  - `flavor != 2` (real WFN) and `nspin != 1` refuse at construction, before
    the coefficient dataset is read: the coefficient slicing hard-codes the
    complex axis, and treats axis 1 as spinor only, so an `nspin = 2` file
    would silently read spin up. `nspinor` may be 1 or 2.
* **Time reversal.** The constructor measures the 2c occupied-subspace
  residual (`trs_holds`, `trs_reference`); `symmetry()` passes the verdict to
  `SymMaps`, which refuses when it is missing. [`symmetry_maps`](symmetry_maps.md#contract)
  owns how the verdict is consumed.
* **QE schema discovery is bounded.** `qe_schema=None` calls
  `symmetry_maps.discover_qe_schema_paths(wfn_path)`: a
  `data-file-schema.xml` beside the WFN, or one inside a `*.save/` under `.`,
  `scf`, `nscf`, `qe/scf` or `qe/nscf`, anchored at the WFN directory (given
  and resolved) and at most two directories above it. Failing to find one is
  not a refusal: `SymMaps` falls back to the conservative all-spatial header
  interpretation plus the global TRS verdict and prints a
  `SYMMETRY PROVENANCE WARNING` naming `qe_symmetry_diagnostic`.
* **Parent/star streaming holds one k.** Each full-k star loads its raw parent
  once per band tile and realizes children serially through
  `unfold_parent_to_full_k`. One parent G row and its star's FFT indices live
  across those tiles and are then released; a nonzero child phase is one
  device vector, and zero translations skip it. No dense full-k phase,
  G-vector or FFT-index table exists on this path, and the four-component lift
  derives child G on device from the same parent row.
* **Late mesh binding is narrow.** `adopt_mesh` binds only an auto-picked,
  currently eager loader, at any P; an explicit backend is never overridden.
* **The instance accepts attributes.** `psp.get_DFT_mtxels` attaches
  `grid_rho` and `gw.kin_ion_io` reads it, so `__slots__` or a strict
  `__setattr__` on `WfnLoader` breaks `psp`.

## Backends

| backend | transport | picked by `auto` when |
|---|---|---|
| `eager` | per-rank h5py read of the rank's band block, host unfold | no mesh, one process, or forced |
| `phdf5` | `SlabIO.read_slabs`: the union of k windows, read by independent MPI-IO in chunks of band rows ([slab_io tuning](../architecture/slab_io.md#tuning)), unfold on device | several processes, a 2-D mesh, and a phdf5-capable CUDA or host library |

`read_slabs` takes n windows of one slab shape with per-window valid shapes
and returns a window axis. The per-rank band clamp,
`max(0, min(slab, logical − offset))` per dimension, lives only in
`file_io._slab_io_ffi._derive_window_counts` → `_derive_valid_shape`. One
union read beats n separate `read_slab` calls because each collective
`H5Dread` has a fixed overhead and the loop adds a `jnp.stack`; n is the
request's IBZ k-count, the axis production decks grow along. Each rank
reads its band block straight from the file's stripes, so a
`stripe_count = 1` file serves every rank from one OST; rank 0 announces the
file's stripe layout at open.

## Tests

Markers `services` and `wfn_loader`: `pytest services/wfn_loader/tests`, or
`pytest -m wfn_loader` in the monorepo (deselect with `--no-services` /
`--only-service=NAME`, never a second `-m`).

| tier | file | needs |
|---|---|---|
| contract | `test_wfn_loader_contract.py`, `test_wfn_loader_close.py` | the checked-in fixtures |
| emulated 2×2 | `test_wfn_loader_emulated_mesh.py` | four emulated devices (service conftest); skips below four |
| real multi-process | `test_wfn_loader_multiproc.py` (`check_*` bodies plus `_CLI_CELLS`) | one process per device |
| skip honesty | `test_wfn_loader_skip_honesty.py` | a machine profile |
| import isolation | `test_wfn_loader_import_isolation.py` | `python -S`; asserts `sys.modules` and `sys.path` |
| layering and bootstrap | `tests/test_layering.py`, `tests/test_service_path_bootstrap.py` | AST and subprocesses |

Hostile geometry runs on real checked-in decks (gnppm: `mnband = 82`, so
82 mod 4 = 2, ragged `ngk` 1917–1963) with self-assertions that the band
window does not divide the world and pad slots exist. The real-process cells
assert the padding conjunction and eager/phdf5 bit identity, and log per-rank
clamped band counts (`[3,3,3,1]` on the hostile window) next to a perturbed
negative control. `tests/test_service_path_bootstrap.py` walks the AST of
`src/` and requires every module-scope importer of the door to have a runtime
or compatibility seal on a line above the import, with a red twin.

## Antipatterns

* **A second copy of the band-clamp arithmetic.** It lives only in
  `_derive_window_counts` → `_derive_valid_shape`; a local copy put real file
  bands into pad rows on every non-divisible geometry.
* **Consuming `gvecs()` without `ngk_valid()`.**
* **An n × `read_slab` loop for a multi-window read.** Use
  `SlabIO.read_slabs`.
* **FFI knowledge in the loader or its consumers.** No `ffi.phdf5`, target
  strings or context handles outside `slab_io`.
* **`jax.device_put(numpy_array, multi_process_sharding)` in loader paths.**
  It triggers JAX's hidden `assert_equal` all-gather (6.45 GB/rank at P = 64
  for the index table); use `device_put_process_local`.
* **Private spellings** (`._filename`, `._ensure_sym()`, `._kpt_starts`) in
  new code: use `path`, `symmetry()`, `kpt_starts`.
* **`__slots__` or strict `__setattr__` on `WfnLoader`** (§ Contract).
