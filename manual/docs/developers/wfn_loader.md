# wfn_loader

`import wfn_loader` The reader of BerkeleyGW `WFN.h5`: the mean-field header, the per-k G tables, and the plane-wave coefficients $\psi_{n\bk}(\bG)$ on the full k grid, sharded over bands.
{: .page-summary }

Every LORRAX calculation starts from Kohn–Sham (or quasiparticle) states $\psi_{n\bk}(\br)=\sum_{\bG}c_{n\bk}(\bG)\,e^{i(\bk+\bG)\cdot\br}$ stored in a BerkeleyGW `WFN.h5` on the k rows the DFT code kept. `WfnLoader` is the single entry point for that file. A caller names a band window and a k set, and receives the coefficients in one layout, `(n_k, n_b, n_spinor, ngkmax)` complex128 with each k's G sphere padded to a common length, either as a global array sharded over bands across the device mesh, or as a process-local array. Full-grid k that are not stored are produced on the fly from their stored parent by the symmetry operation that maps one to the other, including the nonsymmorphic phase, the spinor rotation and, on time-reversed rows, complex conjugation with the negated G list ([Symmetry §3.1](../theory/symmetry.md#sym-psi-pw)): $c_{\bk}(S\bG-\mathbf g_0)=\mathcal U\,e^{-i(S\bG)\cdot\mathbf t}\,c_{\bar\bk}(\bG)$. The loader also owns the tables that must travel with ψ (padded Miller indices, valid lengths, the paired fractional k, the sphere-to-FFT-box index), the occupation metadata, and the time-reversal verdict that the symmetry service consumes. Everything downstream (the centroid faces $\psi_{n\bk}(\br_\mu)$, the ζ fit, matrix elements, the BSE) reads states through it; a new driver that needs wavefunctions or band energies should too.
{: .lead }

<div class="glance" markdown>

Import
:   `import wfn_loader` and use top-level names, or `from file_io import WfnLoader` (the same class). A direct-library caller runs `ffi._services.ensure_on_path()` first.

Depends on
:   lxkit, JAX, numpy, h5py; [`symmetry_maps`](symmetry_maps.md) for the unfold; LORRAX's `file_io.slab_io` for the collective read.

Devices
:   `load` returns a global `jax.Array`, band axis on `('x','y')`; `load_process_local` returns a single-device array. Header and index tables are host numpy.

Used by
:   `common.psi_G_store`, `common.wfn_transforms`, `gw.gw_jax`, `gw.kin_ion_io`, `file_io.qp_wfn`, `psp.*`, `bse.*`, `centroid.*` ([below](#used-by))

</div>

## WfnLoader {: #wfnloader }

```python
WfnLoader(path, *, mesh=None, backend="auto", qe_schema=None)
```

| Argument | Meaning |
|---|---|
| `path` | `WFN.h5`. Complex flavor and `nspin = 1` only; `nspinor` 1 or 2 |
| `mesh` | 2-D `('x', 'y')` `jax.sharding.Mesh`, or `None`. Can be bound later with `adopt_mesh` |
| `backend` | `"auto"`, `"eager"` (per-rank h5py) or `"phdf5"` (collective MPI-IO). See [backends](#backends) |
| `qe_schema` | path to QE's `data-file-schema.xml`. `None` searches near the WFN; an explicit path must authenticate against the WFN or the call raises |

Construction reads the header, checks the layout, and runs the two-component time-reversal check (cached across processes and runs; [Symmetry §2](../theory/symmetry.md#sym-trs)). Raises `ValueError` for `flavor != 2` (real wavefunctions), `nspin != 1` (collinear spin-polarized), an unknown backend, the deleted `"phdf5_host"`, and `"phdf5"` without a mesh. Use as a context manager or call `close()`; `close()` releases both file handles and propagates a close failure.

### Header attributes {: #header }

Every `mf_header` field is bound as an attribute:

| Attribute | Shape, units |
|---|---|
| `nkpts`, `nbands`, `nspinor`, `nspin`, `ngkmax`, `ntran` | int |
| `kgrid`, `shift`, `fft_grid` | `(3,)` |
| `kpoints` | `(nkpts, 3)` crystal |
| `kweights` | `(nkpts,)`, one per stored row; not redistributed over stars |
| `ngk`, `ifmin`, `ifmax` | per stored k |
| `energies`, `occs` | `(nspin, nkpts, nbands)`; energies in Ry |
| `ecutwfc`, `ecutrho` | Ry |
| `avec`, `alat`, `adot` | lattice rows in units of `alat` (bohr) |
| `bvec`, `blat`, `bdot` | reciprocal rows in units of `blat` (bohr⁻¹; Cartesian bohr⁻¹ is `blat * bvec`) |
| `cell_volume` | bohr³ |
| `sym_matrices`, `translations` | BerkeleyGW `mtrx` (padded to 48; the first `ntran` are active) and `tnp` = 2π·τ |
| `atom_types`, `atom_positions` | Cartesian, units of `alat` |

Derived: `nelec` (`max(ifmax)`), `vbm`, `cbm`, `efermi` (gap midpoint), all Ry; `num_electrons`; `atom_crys` (crystal positions); `path`; `kpt_starts` (offsets into the flat G axis of `wfns/coeffs`); `backend` (the resolved transport). The instance accepts new attributes; some drivers attach their own.

### Loading ψ {: #load }

```python
loader.load(*, bands, k="full_bz", sharding=None, bispinor=False, bispinor_lift="raw") -> jax.Array
loader.load_process_local(*, bands, k="full_bz", bispinor=False, bispinor_lift="raw") -> jax.Array
loader.bands(b_lo, b_hi, *, chunk, k="full_bz", sharding=None, bispinor=False, bispinor_lift="raw")
```

| Argument | Meaning |
|---|---|
| `bands` | half-open `(b_lo, b_hi)` within `[0, nbands)`; empty or out-of-range raises `ValueError` |
| `k` | a [`KSpec`](#kspec) |
| `sharding` | output `PartitionSpec`. Default: `P(None, ('x','y'), None, None)` on a multi-device mesh, replicated otherwise |
| `bispinor` | append the small components, $\psi_S=(\alpha/2)\,\boldsymbol\sigma\cdot(\bk+\bG)\,\psi_L$; requires `nspinor == 2`, output spinor axis 4 |
| `bispinor_lift` | `"raw"` or `"normalized_rkb"`; anything but `"raw"` requires `bispinor=True` |

**`load`** returns the global array `(n_k, nb_padded, n_s, ngkmax)` complex128. **Collective**: every process requests the same window. The band axis is padded to a multiple of the device count on it; pad rows are zero. G pad columns past `ngk_valid` are zero.

**`load_process_local`** returns `(n_k, b_hi − b_lo, n_s, ngkmax)` on `jax.local_devices()[0]`: no band padding, no collective, and each process may request a different window. Use it for work parallel over k; combine results explicitly.

**`bands`** yields `((bc_lo, bc_hi), psi)` from `load` in chunks of `chunk` bands.

From `src/file_io/parallel_transport.py`:

```python
center_xy = wfn.load(
    bands=(0, nb), k=center_ids, sharding=band_sphere_spec(),
    bispinor=bool(bispinor))
g_center = wfn.gvecs(k=center_ids)[0]
ngk_center = int(wfn.ngk_valid(k=center_ids)[0])
```

From `src/common/wfn_transforms.py`, one k on this process:

```python
psi = loader.load_process_local(
    bands=(int(b_lo), int(nb)), k=[int(k_idx)], bispinor=bool(bispinor),
    bispinor_lift=bispinor_lift)
```

### KSpec {: #kspec }

| Value | Rows | Unfold |
|---|---|---|
| `"ibz"` | every stored WFN row, file order | no |
| `"full_bz"` | every full-grid row, `SymMaps.unfolded_kpts` order | yes |
| sequence of int | those full-grid rows, in the given order | yes |
| `IBZRows(rows)` | those stored WFN rows | no |

`IBZRows` is a frozen dataclass with one field, `rows: tuple[int, ...]`.

### G tables {: #tables }

```python
loader.kvecs(*, k="full_bz") -> np.ndarray                  # (n_k, 3) crystal
loader.gvecs(*, k="full_bz", cache=True) -> np.ndarray      # (n_k, ngkmax, 3) int32
loader.ngk_valid(*, k="full_bz", cache=True) -> np.ndarray  # (n_k,) int32
loader.box_index(*, k="full_bz", cache=True) -> np.ndarray  # (n_k, ngkmax) int32
loader.box_index_dev(*, k="full_bz", mesh=None, sharding=None, cache=True) -> jax.Array
loader.get_gvec_nk(ik) -> np.ndarray                         # (ngk[ik], 3), stored row ik
```

`kvecs` and `gvecs` are paired row for row: form $\bk+\bG$ and Bloch phases from these two tables only, since rebuilding k from the integer grid can pick another reciprocal-lattice image. `gvecs` pad rows hold the FFT-box **pad sentinel** (the Nyquist-corner Miller index), never zeros, so a consumer that drops `ngk_valid` is detectable instead of silently aliasing pad slots onto Γ. `box_index[k, g]` is the flat C-order FFT-box cell of slot `g`; pad slot `g` holds the out-of-box value `n_rtot + g`, so a `mode='drop'` scatter ignores it and a `mode='fill'` gather returns zero. `box_index_dev` places that table once per `(k, mesh)`, replicated, and returns the same device buffer to every caller. `cache=False` bypasses the loader's caches for bounded requests.

### Streaming one star at a time {: #stars }

```python
loader.full_k_parent_groups(full_k=None) -> tuple[tuple[int, np.ndarray], ...]
loader.unfold_parent_to_full_k(parent_psi, *, parent, full_k, bispinor=False, bispinor_lift="raw") -> jax.Array
loader.little_group_rows(parent) -> np.ndarray
loader.unfold_parent_by_row(parent_psi, *, parent, row) -> (child, gvecs)
loader.full_k_box_index_one_dev(full_k) -> jax.Array        # (1, ngkmax)
loader.ibz_box_index_one_dev(parent) -> jax.Array           # (1, ngkmax)
```

For memory-bounded work over the full grid, read each stored parent once and realize its star from it. `full_k_parent_groups` returns `(parent, children)` pairs in first-occurrence order (duplicates raise). `parent_psi` is the one-row result of `load(..., k=IBZRows((parent,)))`; `unfold_parent_to_full_k` applies the child's operation and returns a one-k array with the same sharding, validating that `full_k` belongs to `parent`. `little_group_rows` and `unfold_parent_by_row` apply any operation of the parent's little group. No full-grid G, phase or index table is built on this path.

From `src/file_io/qp_wfn.py`:

```python
parent_psi = wfn.load(bands=bands, k=IBZRows((p,)), sharding=band_spec)
child = wfn.unfold_parent_to_full_k(parent_psi, parent=p, full_k=f)
```

### Symmetry, time reversal, occupations {: #symmetry }

| Member | Meaning |
|---|---|
| `symmetry()` | the loader's [`SymMaps`](symmetry_maps.md#symmaps), built on first call and then the same object |
| `trs_holds` | bool, the measured two-component time-reversal verdict; [`SymMaps.trs_allowed`](symmetry_maps.md#trs) is taken from it |
| `trs_reference` | the `DensitySymmetryReport` behind the verdict (`density_symmetry` is an older name for it) |
| `qe_symmetry_binding`, `qe_symmetry_diagnostic` | the authenticated QE schema receipt, or why there is none |
| `resolve_qe_symmetry()`, `spinorbit` | the receipt, resolved once; QE's `spinorbit` flag or `None` |
| `occupations_are_exact_integer` | the stored occupation table is exactly `[1…1, 0…0]` |
| `occupation_state_capacity` | electrons per unit occupation, $2/(n_{\mathrm{spin}}n_{\mathrm{spinor}})$ |
| `physical_density_band_stop` | exclusive band stop of every nonzero occupation |
| `physical_density_occupations(*, k, unit_as_none=False)` | `(n_k, n_b)` occupations on a k set; `None` for an exact insulator when `unit_as_none` |

### Lifecycle {: #lifecycle }

| Member | Meaning |
|---|---|
| `adopt_mesh(mesh)` | bind a mesh after construction and re-run the `"auto"` pick; returns the backend now in force. Only an `"auto"` loader that resolved to `"eager"` changes. May raise |
| `release_read_staging()` | close the collective read handle and free its host staging buffer (about ψ/P per rank). Collective |
| `WfnLoader.sync_read_staging()` | drain posted native reads before a direct h5py access; opens nothing |
| `close()`, `with WfnLoader(...) as w:` | release both file handles |

## Module functions {: #functions }

```python
read_wfn_provenance(path) -> WfnProvenance
uniform_band_windows(b_lo, b_hi, width) -> list[tuple[int, np.ndarray]]
```

`WfnProvenance` is a header-only view (`path`, `energies`, `kpoints`, `nelec`, `nspinor`, `nbands`, `num_electrons`, `occupations_are_exact_integer`, `physical_density_band_stop`); it reads no G or ψ. `uniform_band_windows` returns fixed-width `(lo, mask)` windows covering `[b_lo, b_hi)` once: the last window overlaps the previous one instead of shrinking, and its 0/1 mask removes the overlap, so every consumer compiles one FFT shape.

`_phdf5_unfold_kernel`, `_get_bispinor_lift_jit` and `_bispinor_lift_kernel` are exported for in-tree tests that pin the device unfold and the bispinor lift; they are not a stable interface.

## Backends {: #backends }

| Backend | Transport | Chosen by `"auto"` when |
|---|---|---|
| `eager` | per-rank h5py read of the rank's band block; unfold on host | no mesh, one process, or `LORRAX_WFN_BACKEND=eager` |
| `phdf5` | one collective MPI-IO read of the union of requested k windows; unfold on device | several processes, a 2-D mesh, and a phdf5-capable CUDA or host library |

The two are byte-identical for the same request. At several processes with a mesh and no phdf5-capable library, `"auto"` raises (naming `LORRAX_WFN_BACKEND=eager`) instead of demoting silently. Rank 0 logs the choice once.

## Invariants and gotchas {: #gotchas }

- **Padding is a pair.** ψ pad slots are zero and the matching `gvecs` rows hold the sentinel. Always carry `ngk_valid` with `gvecs`.
- **Pair k with G from the loader.** `kvecs(k)` and `gvecs(k)` are consistent images; a k rebuilt from `kgrid` may not be.
- **`load` is collective, `load_process_local` is not.** Mixing them up either hangs or builds a global array whose shards are different physical objects.
- **`nspinor = 2` means noncollinear.** It does not imply spin–orbit coupling; collinear `nspin = 2` files refuse.
- **No hidden all-gathers.** Do not `jax.device_put` a host numpy array onto a multi-process sharding in loader paths; it triggers a hidden equality all-gather. Use `lxkit.device_put_process_local` ([lxkit](lxkit.md)).
- **One star, one parent read.** For full-grid sweeps under memory pressure use `full_k_parent_groups` + `unfold_parent_to_full_k`, not `load(k="full_bz")`.
- **Weights are per stored row.** `kweights` is not spread over stars; decide full-grid quadrature weights against the symmetry tables.
- **Use the public names.** `path`, `symmetry()`, `kpt_starts`; not `_filename`, `_ensure_sym()`, `_kpt_starts`.

## Where LORRAX uses it {: #used-by }

| File | Purpose |
|---|---|
| `src/common/psi_G_store.py` | the one-read ψ(G) store: band chunks via `load`, then an all-to-all to G-slot shards and the centroid faces |
| `src/common/wfn_transforms.py` | ψ to the FFT box and to centroids; parent-star streaming |
| `src/gw/gw_jax.py` | the GW driver's loader and symmetry |
| `src/gw/kin_ion_io.py`, `src/psp/` | kinetic, ionic, dipole and DFT operator matrix elements |
| `src/file_io/qp_wfn.py`, `src/postprocess/rotate_wfn_to_qp.py` | writing `WFN_qp.h5` |
| `src/file_io/parallel_transport.py` | overlaps between neighbouring k for the head |
| `src/file_io/restart_bundle.py` | authenticates the WFN and takes its symmetry when unfolding parent faces read back from the ISDF tensor file |
| `src/centroid/` | centroid selection from the charge density and pair sampling |
| `src/bse/`, `src/tdagw/` | BSE and time-dependent drivers |
