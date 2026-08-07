# zeta_loader

The reader and format-contract owner of `zeta_q.h5` (and the bispinor
siblings `zeta_q_mu{1,2,3}.h5`). One door: the header surface, the
collective slab read that feeds V_q, the sanctioned local plan, the
never-raising probe, and the one post-close append. It owns no
mathematics — `zeta_rcond`, the fit and the solver tiers are
producer-side (`isdf` / `gw.isdf_fitting`).

## Purpose

`zeta_q.h5` is written once by `gw.isdf_fitting.fit_zeta_to_h5` and read
many times — by V_q (`gw/v_q_g_flat`, `gw/v_q_bispinor`), by the BSE
interpolation (`bse/vq_interp`), by basis projection
(`common/zeta_projection`) and by the fit-reuse gates (`gw/gw_init`).
Before this service, four sites read or wrote the file around the loader
(raw h5py; one of them a WRITE with no door), and the layout dispatch
`(('zeta_q_G',1),('zeta_q',2))` existed in three copies — one of which
probed only the legacy dataset name for months and silently passed on
exactly the production files it guarded. The service is that surface,
owned once. The r-space/full-BZ read paths were deleted at extraction
(2026-08-07): no writer in the tree has emitted that layout since the
G-flat migration, and removing them took the loader's coupling to the
concurrently-moving wave-1 services to zero.

## API

```python
from ffi import _services; _services.ensure_on_path()   # transitional
import zeta_loader                                       # jax-free import

zeta_loader.ZetaLoader(path, *, mesh=None, mode='r')     # first access imports jax
    .read_zeta_G_slab(*, q_offset, q_count, mu_offset, mu_count, mesh=None)
    .read_zeta_G_local(key)
    .load(*, q='ibz'|seq[int], mu=None, sharding=None)
    .gvecs(q='ibz') / .ngk_valid(q='ibz')
    .close() / context manager
    # + the bound mf_header/isdf_header attribute surface (see Contract)
zeta_loader.probe_zeta_file(path) -> ZetaFileProbe   # NEVER raises
zeta_loader.write_g0_mu(path, g0_logical, *, n_rmu_expected=None)
```

`mesh=None` is HEADER-ONLY mode: every header attribute and the local
plan work with no phdf5 FFI anywhere; the collective reads refuse naming
the missing mesh. With a mesh, ONE SlabIO handle is opened eagerly and
held for the loader's lifetime (amortises the phdf5 ctx — measured, see
Performance). The door itself is lazy on jax (PEP 562): the format
surface is pure h5py+numpy and runs on a jax-free stack (login-node
`python3` was the case that forced this); `ZetaLoader` pays the jax
import on first attribute access.

## Contract

- **One data layout.** Every data method reads G-flat `zeta_q_G` and
  refuses anything else by name. `__init__` still opens r-space files:
  the header surface is layout-independent.
- **Agreement is checked at open, not assumed.** Header μ vs dataset μ;
  header `ngkmax` vs the `zeta_q_G` G axis (the collective plan sizes
  from the header, the local plan from the dataset — a disagreement
  would have the two plans silently reading different extents).
- **Completeness is checked at open.** `isdf_header/zeta_is_done=False`
  refuses (`LORRAX_ALLOW_PARTIAL_ZETA=1` overrides, for debugging).
- **Refusal order: request facts before stack facts.** A bad request
  (wrong layout, `full_bz`, a strided μ) is reported even on a stack
  with no transport; the transport refusal fires only for a request
  that was otherwise servable.
- **Padding.** A `mu_count` past the on-disk extent comes back
  zero-filled (SlabIO's logical-extent contract); pad G-slots carry the
  FFT-box sentinel Miller index in `gvecs()` — masks are detectable,
  not optional. `gvecs()` re-validates the components table against the
  mf_header FFT grid at read time and refuses a mismatch.
- **The two plans are byte-identical where they overlap** (same on-disk
  elements, no reduction) — asserted at real 2×2 by the L-c identity
  cell.
- **`read_zeta_G_local` is non-collective BY CONTRACT.** Making it
  collective turns a rank-0 diagnostic into a hang. It works at
  `mesh=None`.
- **`write_g0_mu` is the one sanctioned post-close serial append.** The
  caller keeps the rank-0 gate and the barriers (its docstring gives the
  exact sequence); the logical-extent guard refuses padded arrays —
  files store LOGICAL extents so they re-read identically at any
  process count.
- **Header surface** (pinned by the contract test): nspin, kgrid,
  fft_grid, sym_matrices, ntran, bvec, adot, blat, cell_volume, ifmax,
  kpoints, vertex_mu_L, r_mu_fft_idx, n_rmu, zeta_layout,
  gvec_components, ngk_per_q, ngkmax_zeta, zeta_cutoff_ry, zeta_is_done,
  plus derived n_q_on_disk, n_rtot_disk, n_rmu_disk, n_G_sph_disk,
  n_q_full, q_layout, n_rtot.

## Backends

One transport: SlabIO (`_FfiBackend`, phdf5 via the LORRAX FFI pair) for
the collective plan; serial h5py for the headers, the probe, the local
plan and the g0_mu append. There is no backend switch and nothing to
demote to: a stack without the phdf5 FFI serves the whole header/format
surface and refuses collective reads by name. The transport carries a
per-PROCESS MPI context, so a mesh handed to the loader must satisfy
`p*q == jax.process_count()` — emulated multi-device meshes are refused
by the FFI itself. Declared package deps are lxkit, jax, numpy, h5py;
`file_io.slab_io`, `file_io.mf_header`, `file_io.isdf_header` and
`common.gvec_fft_box` are reached through call-time imports that refuse
by name — the wave-1b seam (they become package deps when
slab_io/file_formats extract).

## Tests

`services/zeta_loader/tests/`, markers `services` + `zeta_loader`,
staged into the main suite (deselect with `--no-services`; select with
`-m zeta_loader` or `--only-service=zeta_loader`).

- L-a: probe truth table, every refusal with its red twin, the header
  surface pin, the striping AST guard over `fit_zeta_to_h5` (the
  production writer's create order — `SlabIO(mode='w')` before
  `copy_mf_header`, no `dst_mode='w'`), the ZETA_RCOND_DEFAULT
  five-signatures-agree guard + literal-default ratchet.
- L-b: the emulated 4-device tier serves the mesh-free cells; the
  transport cells are process-bound (see Backends) and run 1×1 under
  plain pytest.
- L-c: real `srun -n 4` via shared check bodies + CLI — the hostile
  geometry, first executed 2026-08-07 (JID 56447670, `ran=5
  failures=0`): μ 300-in-304 interior to a rank tile, 3-in-4 and
  1-in-4 on rank edges; q windows at n_q=74 (74/47/1); ragged ngk with
  the non-vacuity assertion; bispinor n_rmu_C=4 ≠ n_rmu_T=3 across four
  handles; local-vs-collective byte identity. The report JSON records
  `hostile_mu_boundary`/`hostile_q_axis`, false at 1×1 by design, so a
  1×1 record can never be read as a 2×2 result.
- Import isolation (`lxkit.testing.import_isolation`) + red twin; the
  format surface is asserted functional AND jax-free with lorrax off
  sys.path.
- Skip-honesty: the machine profile's phdf5 MUST row is asserted through
  `file_io.slab_io.probe_availability` — zeta_loader is the slab_io
  client the lxkit profile said would supply that probe.
- End-to-end: the Si production deck's `fit_zeta_to_h5` output read back
  through the door (probe, header surface, `gvecs()` on the real ragged
  sphere — min ngk 537 < ngkmax 588 — and local-plan byte identity), and
  the striping policy measured on the fit-written file itself:
  `lfs getstripe -c` = 4 vs directory default 1 (2026-08-07, the
  measurement the in-tree striping test cannot make where `lfs` is
  absent).

## Performance

Claims-style baselines in `services/zeta_loader/bench/baselines/`
(`cpu1x1.json`, `cpu2x2.json`; op, shape, mesh, nodes, seconds, MB/s,
jobid, FFI pins recorded in-file; jobid 56447670, 2026-08-07). The
held-open-handle claim is measured as the `held_open` vs `open_close`
row pair — the amortisation is worth 2.3–9.4× depending on shape
(e.g. 1×1 `[8,64,512]`: 3098 vs 328 MB/s; 2×2 `[8,512,2048]`: 2886 vs
1667 MB/s), and the two rows converging is the regression signal.
`read_zeta_G_local` (host numpy, no device round-trip) runs 4.4–11 GB/s
on the same files. Reference band: slab_io's 2026-08-07 calibration
(dd 725, serial h5py 967, SlabIO same-handle 953–961, open-close-per-leg
410–580 MB/s on production-scale files; the synthetic bench files are
page-cache-warm, so compare rows to rows, not bands to bands).

## Antipatterns

- **A second reader.** Do not `h5py.File(zeta_path)` outside this
  package for data or layout facts — that is exactly the class of copy
  that silently probed the wrong dataset name for months. The local plan
  you are about to hand-roll is `read_zeta_G_local`.
- **Making the local plan collective** (or wrapping it in a SlabIO read
  "for consistency"): a rank-0 diagnostic becomes a hang.
- **Calling `write_g0_mu` from every rank** or before the collective
  handles close: silent corruption on a shared filesystem. The caller
  owns the gate and barriers, deliberately.
- **Consuming `gvec_components` raw** when `gvecs()` exists: the raw
  table is unvalidated against the FFT grid; the accessor refuses the
  corrupt case.
- **Writing padded extents to disk.** Files store LOGICAL extents;
  `write_g0_mu` refuses a padded μ axis, and any new dataset must follow
  the same rule.
- **Re-deriving the layout dispatch.** `(('zeta_q_G',1),('zeta_q',2))`
  lives in `format.py`, once. A new copy is a future silent-pass.
- **Adding a `zeta_rcond` mirror.** Import `ZETA_RCOND_DEFAULT` from
  `gw.gw_config`; the AST ratchet fails a literal default.
- **Handing the transport an emulated multi-device mesh.** The FFI
  refuses `p*q != process_count` — build the mesh from the process
  count, and get 2×2 claims from a real `srun -n 4` leg.
