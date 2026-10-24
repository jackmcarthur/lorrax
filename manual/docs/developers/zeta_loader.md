# zeta_loader

`import zeta_loader` The reader of `zeta_q.h5`: the ISDF interpolating vectors $\zeta_{\bq\mu}(\bG)$, their per-$\bq$ G spheres and headers, read collectively, locally or header-only.
{: .page-summary }

LORRAX writes every pair density as $\psi^*_{m\bk}(\br)\psi_{n\bk+\bq}(\br)\approx\sum_\mu\zeta_{\bq\mu}(\br)\,\psi^*_{m\bk}(\br_\mu)\psi_{n\bk+\bq}(\br_\mu)$ ([ISDF §1.1](../theory/isdf.md#isdf-ansatz)). The interpolating vectors $\zeta_{\bq\mu}$ are fitted once per centroid set and stored in G space: `zeta_q_G[q, μ, g]` is the coefficient of $e^{i(\bq+\bG_g)\cdot\br}$ in $\zeta_{\bq\mu}(\br)$ on the per-$\bq$ sphere $|\bq+\bG|^2\le$ `zeta_cutoff` ([files](../drivers/gw/files.md#zeta)). Everything that turns ζ into an interaction contracts that block with a Coulomb kernel, $V_\bq(\mu,\nu)=\sum_\bG\zeta^*_{\bq\mu}(\bG)\,v(\bq+\bG)\,\zeta_{\bq\nu}(\bG)$ with $v$ from [`vcoul`](vcoul.md): the bare $V_\bq$, the long-range channel rows of the W split, the BSE exchange tiles, basis projections and exports. `zeta_loader` is the one reader of that file. It checks at open that the file is complete and that its header and data agree, serves the sharded read that feeds $V_\bq$, a non-collective host read for diagnostics, and a probe that never raises for the guards that decide whether to refit. It owns no mathematics: the fit and its conditioning belong to the producer (`gw.isdf_fitting`). The ISDF tensor file (`tmp/isdf_tensors_<N_mu>.h5`, holding $V_\bq$, $W_0$ and the head scalars) is a different file and is not read here; its $\bq$-wedge storage format belongs to [`symmetry_maps`](symmetry_maps.md#qirr).
{: .lead }

<div class="glance" markdown>

Import
:   `import zeta_loader`. `ZetaLoader` is loaded lazily on first access (it imports JAX); `probe_zeta_file` and `ZetaFileProbe` need only numpy and h5py.

Depends on
:   lxkit, JAX, numpy, h5py. The data path also needs LORRAX's `file_io.slab_io`, `file_io.mf_header`, `file_io.isdf_header` and `common.gvec_fft_box` at call time, which a LORRAX run puts on the path; a direct-library caller runs `ffi._services.ensure_on_path()` first.

Devices
:   Collective reads return a `jax.Array` on the mesh, `P(None, ('x','y'), None)`. Header, probe and local reads are host numpy.

Used by
:   `gw.v_q_g_flat`, `gw.lr_channels`, `gw.gw_init`, `gw.epsmat_export`, `bse.bsemat_export`, `tdagw.model` ([below](#used-by))

</div>

## The file {: #file }

| Dataset | Shape, dtype | Meaning |
|---|---|---|
| `zeta_q_G` | `(n_q_on_disk, n_mu_disk, ngkmax)` complex128 | $\zeta_{\bq\mu}(\bG)$; zero for `g >= ngk[q]`. The per-$\bq$ FFT-box phase is already applied by the writer |
| `isdf_header/gvec_components` | `(n_q, 3, ngkmax)` int32 | Miller indices of each slot; pad slots hold the FFT-box sentinel |
| `isdf_header/ngk` | `(n_q,)` int32 | logical sphere size per $\bq$ |
| `isdf_header/centroids/r_mu_fft_idx` | `(N_mu, 3)` int32 | centroid FFT-grid indices |
| `isdf_header/zeta_is_done` | bool | false until the last block lands |
| `mf_header/` | | copied from WFN.h5 |

`n_q_on_disk` is either the irreducible $\bq$ wedge or the full grid. On a wedge file, row `i` is the irreducible $\bq$ `SymMaps.q_irr_full_idx[i]`, with crystal coordinates `bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, kgrid)[i]` ([`symmetry_maps`](symmetry_maps.md#grid-algebra)). $\zeta$ is never unfolded on read: a full-grid consumer contracts on the wedge and unfolds the bilinear result with `symmetry_maps.unfold_isdf_operator`. The bispinor siblings `zeta_q_mu1.h5`, `zeta_q_mu2.h5`, `zeta_q_mu3.h5` have the same layout and are read the same way.

## ZetaLoader {: #zetaloader }

```python
ZetaLoader(path, *, mesh=None, mode="r")
```

| Argument | Meaning |
|---|---|
| `path` | `zeta_q.h5` or a bispinor sibling |
| `mesh` | 2-D `('x', 'y')` `jax.sharding.Mesh` with `Px·Py == jax.process_count()` (or a one-process emulated mesh). `None` opens the file **header-only** |
| `mode` | open mode of the collective handle |

With a mesh, one collective `SlabIO` handle is opened at construction and held until `close()`; holding it is worth 2–9× read bandwidth over reopening. Use the loader as a context manager or call `close()`, which releases both the collective and the local handle and is idempotent.

Raises `ValueError` at open when:

- `isdf_header/zeta_is_done` is false: the fit that wrote the file did not finish. `LORRAX_ALLOW_PARTIAL_ZETA=1` overrides, for debugging only.
- `zeta_q_G` has fewer μ rows than the header's `n_rmu` (the header and the block come from different runs).
- the header `ngkmax` differs from the `zeta_q_G` G axis.

### Header attributes {: #header }

All available at `mesh=None`, as plain attributes:

| Attribute | Meaning, units |
|---|---|
| `nspin`, `nspinor`, `kgrid`, `fft_grid` | from `mf_header` |
| `bvec`, `blat`, `adot`, `cell_volume` | reciprocal rows in units of `blat` (bohr⁻¹), real-space metric, Ω in bohr³ |
| `kpoints`, `ifmax` | WFN k rows (crystal) and highest occupied band |
| `sym_matrices`, `ntran` | BerkeleyGW `mtrx` and the operation count |
| `n_rmu`, `r_mu_fft_idx`, `vertex_mu_L` | centroid count, `(N_mu, 3)` grid indices, vertex label (0 charge, 1–3 current) |
| `zeta_layout` | `'G_flat'` (every file the current writer produces) or the legacy `'r_space'` |
| `gvec_components`, `ngk_per_q`, `ngkmax_zeta` | the raw sphere tables (prefer `gvecs()`, which validates them) |
| `zeta_cutoff_ry` | `zeta_cutoff` in Ry |
| `zeta_is_done` | completeness flag |
| `n_q_on_disk`, `n_q_full`, `q_layout` | stored $\bq$ rows, `prod(kgrid)`, and `'ibz'` or `'full_bz'` |
| `n_rmu_disk`, `n_G_sph_disk`, `n_rtot` | stored (possibly padded) μ extent, `ngkmax`, `prod(fft_grid)` |

### gvecs, ngk_valid {: #gvecs }

```python
loader.gvecs(*, q="ibz") -> np.ndarray       # (n_q, ngkmax, 3) int32
loader.ngk_valid(*, q="ibz") -> np.ndarray   # (n_q,) int32
```

`q` is `'ibz'` (every stored row), `'full_bz'` (the same rows; refused later by `load` on a wedge file) or a 1-D sequence of stored row indices. `gvecs` is shaped and padded exactly like [`WfnLoader.gvecs`](wfn_loader.md#tables): pad rows hold the FFT-box sentinel, not zeros. It re-checks that the stored pad rows are the sentinel of `mf_header`'s FFT grid and raises `ValueError` on a mismatch. Both raise on an `r_space` file, which carries no per-$\bq$ sphere.

### read_zeta_G_slab {: #slab }

```python
loader.read_zeta_G_slab(*, q_offset, q_count, mu_offset, mu_count, mesh=None) -> jax.Array
```

The production read. **Collective**: every process calls it with the same window. Returns `(q_count, mu_count, ngkmax)` complex128 sharded `P(None, ('x','y'), None)` (the μ axis split over all devices). `q_offset` indexes the stored $\bq$ axis (a wedge index on a wedge file). A `mu_count` past the stored μ extent comes back zero-filled, so a caller padding μ to a multiple of the device count passes the padded extent and nothing else. `mesh` overrides the loader's mesh. Raises `RuntimeError` on a header-only or closed loader and `ValueError` on an `r_space` file.

From `src/gw/v_q_g_flat.py`, one read per $\bq$ tile:

```python
def read_q_tile(q_offset: int, q_count: int) -> jax.Array:
    return zeta_loader.read_zeta_G_slab(
        q_offset=int(q_offset), q_count=int(q_count),
        mu_offset=0, mu_count=int(n_rmu_padded),
        mesh=mesh_xy,
    )
```

### read_zeta_G_local {: #local }

```python
loader.read_zeta_G_local(key) -> np.ndarray
```

`zeta_q_G[key]` as host numpy on this process only, through a serial h5py handle opened on first use. `key` is any h5py key (int, slice, tuple, index list). **Non-collective by contract**: processes may read different windows or none, and it works at `mesh=None`. Where it overlaps `read_zeta_G_slab` it returns the same bytes. Raises `RuntimeError` after `close()`.

From `src/gw/lr_channels.py`, reading the ζ columns of the long-range channels at one $\bq$:

```python
cols = zl.read_zeta_G_local((qi, slice(None), list(np.asarray(slots)[order])))
```

### load {: #load }

```python
loader.load(*, q="ibz", mu=None, sharding=None) -> jax.Array
```

A [`WfnLoader`](wfn_loader.md)-shaped collective window: `q` as for `gvecs`, `mu` a half-open `(lo, hi)`, a unit-step `slice` or `None` (all stored rows), `sharding` a `PartitionSpec` (default `P(None, ('x','y'), None)`). Returns `(n_q, n_mu, ngkmax)` complex128. Refusals come in a fixed order, request before transport: an `r_space` file (`ValueError`), `q='full_bz'` on a wedge file (`NotImplementedError`), a strided μ slice (`ValueError`), then a missing mesh (`RuntimeError`).

`loader.slab_io` exposes the underlying `SlabIO` handle for callers that still need its raw `read_slab` contract.

## probe_zeta_file {: #probe }

```python
probe_zeta_file(path) -> ZetaFileProbe
```

One open, no JAX, no LORRAX, and it **never raises**, for any input: `None`, a directory, a zero-byte file, a foreign HDF5 file, a ζ truncated mid-write. Its callers are the guards that run before a fit that may overwrite the file.

| `ZetaFileProbe` field | Meaning |
|---|---|
| `exists` | `os.path.exists(path)` |
| `readable` | the file opened as HDF5 and every field below was read |
| `error` | `"<ExceptionType>: <message>"` when a read was attempted and failed, else `None` |
| `dataset_name` | `'zeta_q_G'`, `'zeta_q'` (legacy) or `None` (header written, no block yet) |
| `mu_extent` | the μ extent of the ζ block (axis 1 G-flat, axis 2 r-space) |
| `zeta_done` | `isdf_header/zeta_is_done` |
| `r_mu_fft_idx` | the header's centroid table |

`ZetaFileProbe` has no `__eq__` (it holds an array). From `src/gw/gw_init.py`:

```python
probe = probe_zeta_file(zeta_h5_path)
if not probe.readable:
    log.info(f"  [zeta guard] could not read {zeta_h5_path} ({probe.error}); continuing.")
    return
existing = probe.mu_extent
```

## Invariants and gotchas {: #gotchas }

- **One reader.** Do not open `zeta_q.h5` with h5py for data or layout facts outside this package; use `read_zeta_G_local` or the header attributes.
- **Never make the local read collective.** Wrapping `read_zeta_G_local` in a `SlabIO` read turns a rank-0 diagnostic into a hang.
- **Carry `ngk_valid` with `gvecs`.** Pad slots hold the sentinel Miller index and zero coefficients; a consumer that drops the mask reads garbage slots.
- **Use `gvecs()`, not `gvec_components`.** Only `gvecs()` checks the stored table against the FFT grid. The raw table is also transposed: `(n_q, 3, ngkmax)` on disk, `(n_q, ngkmax, 3)` from `gvecs()`.
- **The μ axis on disk may be padded.** `n_rmu_disk >= n_rmu`; physical centroids are the first `n_rmu` rows.
- **Wedge rows are not grid rows.** On a wedge file `q_offset` and `q=[...]` index the stored wedge; map to the full grid through `SymMaps.q_irr_full_idx`.
- **$\bG = 0$ is stored once.** `zeta_q_G[q, :, 0]` is the $\bG = 0$ coefficient of each stored $\bq$. A full-grid $\bG=0$ view comes from the symmetry service, not from a second dataset.
- **Emulated meshes are not multi-process results.** A one-process mesh with several devices is served by a serial path; take bandwidth and scaling numbers from a real multi-process run.

## Where LORRAX uses it {: #used-by }

| File | Purpose |
|---|---|
| `src/gw/v_q_g_flat.py` | the bare $V_\bq$ from G-flat ζ (`read_zeta_G_slab` per $\bq$ tile) |
| `src/gw/lr_channels.py` | ζ columns of the long-range channels for the SR/LR W split (`read_zeta_G_local`) |
| `src/gw/gw_init.py` | pre-fit guards (`probe_zeta_file`) and ζ reuse |
| `src/file_io/restart_bundle.py` | opens the loader for reuse paths |
| `src/gw/epsmat_export.py` | per-$\bq$ ζ for BerkeleyGW `epsmat` export |
| `src/bse/bsemat_export.py` | head columns for BSE kernel export |
| `src/tdagw/model.py` | per-$\bq$ ζ for the time-dependent model |
