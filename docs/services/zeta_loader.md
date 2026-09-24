# zeta_loader

`services/zeta_loader/` is the reader and format-contract owner of
`zeta_q.h5` and its bispinor siblings `zeta_q_mu{1,2,3}.h5`: the header
surface, the collective slab read that feeds V_q, the non-collective local
read, and a probe that never raises. It owns no mathematics: `zeta_rcond`,
the fit and the solver tiers belong to the producer (`isdf`,
`gw.isdf_fitting`).

The file is written once by `gw.isdf_fitting.fit_zeta_to_h5` and read by V_q
(`gw.v_q_g_flat`, `gw.v_q_bispinor`), BSE interpolation (`bse.vq_interp`),
basis projection (`common.zeta_projection`) and the fit-reuse gates
(`gw.gw_init`). All of them read through this door.

## API

```python
import zeta_loader                                        # no JAX until first ZetaLoader access

zeta_loader.ZetaLoader(path, *, mesh=None, mode='r')
    .read_zeta_G_slab(*, q_offset, q_count, mu_offset, mu_count, mesh=None)
    .read_zeta_G_local(key)
    .load(*, q='ibz' | seq[int], mu=None, sharding=None)
    .gvecs(q='ibz'), .ngk_valid(q='ibz')
    .slab_io, .close(), context manager
    # plus the mf_header / isdf_header attribute surface (§ Contract)
zeta_loader.probe_zeta_file(path) -> ZetaFileProbe       # never raises
```

In a LORRAX application the service set is selected by runtime startup;
a direct-library caller uses `ffi._services.ensure_on_path()`, which
delegates to the same seal.

| method | contract |
|---|---|
| `read_zeta_G_slab` | The production V_q read: one collective `read_slab` of the `(Q, μ, ngkmax)` window, returning `(q_count, μ_per_rank, ngkmax)` complex128 at `P(None, ('x','y'), None)`. The per-q FFT-box phase is already in the stored tensor. A `mu_count` past the on-disk extent comes back zero-filled, so a caller padding μ to the mesh states the extent it consumes. |
| `read_zeta_G_local(key)` | `zeta_q_G[key]` as host NumPy on this rank, through a serial h5py handle; any h5py key. Non-collective by contract, and works at `mesh=None`. |
| `load(*, q, mu, sharding)` | Rows `q` (`'ibz'` or disk indices) and a μ range `(lo, hi)`, `slice` or `None`; default sharding `P(None, ('x','y'), None)`. `q='full_bz'` on an IBZ file and a strided μ refuse. |
| `gvecs(q)`, `ngk_valid(q)` | Padded Miller indices (pad slots hold the FFT-box sentinel) and valid lengths; `gvecs` revalidates the stored components against the header FFT grid and refuses a mismatch. |
| `probe_zeta_file(path)` | One open, no JAX, no LORRAX. Any input, including `None`, a directory, a foreign or truncated file, returns `ZetaFileProbe` with `readable=False` and `error` rather than raising, because its callers are pre-fit guards about to overwrite the file. |

## Contract

* **One data layout.** Every data method reads G-flat `zeta_q_G` and refuses
  an r-space `zeta_q` by name. The layout dispatch
  `(('zeta_q_G', 1), ('zeta_q', 2))` lives once, in `format.py`. The
  constructor opens either layout, because the header surface is
  layout-independent.
* **`mesh=None` is header-only.** Header attributes, `gvecs`, the probe and
  the local read work without the phdf5 FFI; collective reads refuse, naming
  the missing mesh. With a mesh, one `SlabIO` handle is opened at
  construction and held for the loader's lifetime, amortizing the phdf5
  context. The context frees a synchronous read buffer above 32 MiB after its
  host-to-device copy, so a loader held open keeps no large staging between
  read phases.
* **Checked at open:**
  - completeness: `isdf_header/zeta_is_done = False` refuses
    (`LORRAX_ALLOW_PARTIAL_ZETA=1` overrides, for debugging; see
    [`env_vars.md`](../dev/env_vars.md));
  - μ: the ζ dataset's μ extent must be at least the header's `n_rmu` for
    `zeta_q_G` and equal to it for `zeta_q`, else the header and ζ block came
    from different runs;
  - G: the header `ngkmax_zeta` must equal the `zeta_q_G` G axis, because the
    collective plan sizes from the header and the local plan from the dataset.
* **Refusal order: request before stack.** A bad request (wrong layout,
  `full_bz`, strided μ) is reported even on a stack with no transport; the
  transport refusal fires only for an otherwise servable request.
* **The two plans are byte-identical** where they overlap: same on-disk
  elements, no reduction.
* **G = 0 is stored once.** `zeta_q_G[q, :, 0]` is the G = (0,0,0) coefficient
  of each stored parent q in the canonical sphere order. A full-BZ G = 0 view
  is derived through the symmetry service, not a second dataset.
* **Header surface** (pinned by the contract test): `nspin`, `kgrid`,
  `fft_grid`, `sym_matrices`, `ntran`, `bvec`, `adot`, `blat`, `cell_volume`,
  `ifmax`, `kpoints`, `vertex_mu_L`, `r_mu_fft_idx`, `n_rmu`, `zeta_layout`,
  `gvec_components`, `ngk_per_q`, `ngkmax_zeta`, `zeta_cutoff_ry`,
  `zeta_is_done`, and derived `n_q_on_disk`, `n_rtot_disk`, `n_rmu_disk`,
  `n_G_sph_disk`, `n_q_full`, `q_layout`, `n_rtot`.

## Backends

The collective plan has one transport, `SlabIO` over the phdf5 FFI; headers,
the probe and the local plan use serial h5py. There is no backend switch and
nothing to demote to. The FFI carries a per-process MPI context, so a mesh
must satisfy `Px·Py == jax.process_count()`. The one exception is an emulated
mesh (one process, more cells), which `SlabIO` serves through
`file_io._slab_io_serial` (single-process h5py); its numbers are real but are
not a P = 4 result. Declared dependencies are `lxkit`, JAX, NumPy and h5py;
`file_io.slab_io`, `file_io.mf_header`, `file_io.isdf_header` and
`common.gvec_fft_box` are call-time imports that refuse by name outside a
sealed LORRAX application. The probe works standalone.

**Cost.** Holding the handle open is worth 2.3–9.4× read bandwidth on CPU
meshes and 1.8–5.9× on CUDA meshes against opening per read, depending on
shape; the `held_open` and `open_close` rows in
`services/zeta_loader/bench/baselines/` converging is the regression signal.
`read_zeta_G_local` runs at 4.4–11 GB/s from host memory.

## Tests

`services/zeta_loader/tests/`, markers `services` and `zeta_loader` (select
with `-m zeta_loader` or `--only-service=zeta_loader`, deselect with
`--no-services`).

* **Contract:** probe truth table, every refusal with its red twin, the header
  surface pin.
* **AST guards** (`test_zeta_loader_ast_guards.py`): the production writer
  creates the file collectively (`SlabIO(mode='w')` before `copy_mf_header`)
  so striping applies, and every `zeta_rcond` default names
  `gw.gw_config.ZETA_RCOND_DEFAULT` with no literal.
* **Real multi-process** (`test_zeta_loader_multiproc.py`, shared check
  bodies plus a CLI): μ windows interior to and straddling rank tiles, q
  windows at `n_q = 74`, ragged `ngk` with a non-vacuity assertion, bispinor
  handles with `n_rmu_C ≠ n_rmu_T`, and local-versus-collective byte
  identity. Its report records `hostile_mu_boundary` / `hostile_q_axis`,
  false at 1×1, so a 1×1 run cannot be read as a 2×2 result.
* **Import isolation** (the format surface works without JAX or LORRAX) and
  **skip honesty** (the phdf5 row through `file_io.slab_io.probe_availability`).

## Antipatterns

* **A second reader.** No `h5py.File(zeta_path)` outside this package for data
  or layout facts; use `read_zeta_G_local`.
* **Making the local read collective**, or wrapping it in a `SlabIO` read: a
  rank-0 diagnostic becomes a hang.
* **Persisting a `g0_mu` mirror.** It duplicates the stored G = 0 slice.
* **Consuming `gvec_components` raw** instead of `gvecs()`, which validates it.
* **Writing padded extents to disk.** Files store logical extents.
* **Re-deriving the layout dispatch** outside `format.py`.
* **A `zeta_rcond` default literal.** Import `ZETA_RCOND_DEFAULT` from
  `gw.gw_config`.
* **Quoting an emulated multi-device mesh as a P = 4 result.** Build the mesh
  from the process count and take scaling and transport claims from a real
  multi-process run.
