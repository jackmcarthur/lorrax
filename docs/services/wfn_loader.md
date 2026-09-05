# wfn_loader — ψ(G) loading behind one door

`services/wfn_loader/`. Independently installable (`pyproject.toml`,
src-layout); depends on `lxkit`, jax, numpy, h5py and nothing else in
LORRAX at import time. Collective I/O appears in exactly one dependency
edge — the `slab_io` door, reached lazily at load time — and in zero
declared dependencies. The loader contains no FFI target name, no ctx
handle and no phdf5 spelling: `grep -rn "ffi\.phdf5\|read_kchunk\|
lorrax_phdf5" services/wfn_loader/src/` is empty, and the import-isolation
suite proves the package imports with the monorepo absent.

## Purpose

One class, `WfnLoader`, is the single entry point for wavefunction
loading: header surface, G-vector tables, FFT-box index tables, and ψ(G)
itself — band-sharded global (`load`), per-rank single-device
(`load_process_local`), or chunk-iterated (`bands`). It replaced the
{WFNReader + PhdfWfnReader + SymMaps unfold helpers + load_wfns helpers}
family; `WFNReader` survives as an alias. The owner's mandate for this
service was FEWER LINES AND FEWER CONCEPTS: the union read and its
per-rank band clamp — five of the file's eighteen concepts, and the part
that drove the FFI directly — moved behind `SlabIO.read_slabs`
(one door primitive), deleting the loader's hand-copied open-guard block,
its private counts table, and every FFI import, net −125 lines, with the
promotion measured bit-identical and 0.7% faster on the production deck.

The package is the door. `import wfn_loader` and use top-level names;
`from wfn_loader.loader import …` from lorrax is a layering failure
(`tests/test_layering.py`, with a red twin).

`kweights` has one value per raw WFN k row. The loader validates only its
storage contract—shape `(nkpts,)`, finite nonnegative values, and a positive
sum—and preserves the values. It does not infer whether those rows are an IBZ
wedge or the complete grid and does not spread weights over stars. Consumers
that need full-grid quadrature make that distinction against authenticated
symmetry metadata; the centroid implementation is
`centroid.sampling_metric.full_k_quadrature_weights`.

The transitional shim that used to sit at `src/file_io/wfn_loader.py` is
gone: the phase-wide cleanup commit deleted it along with the other four
wave-1 shims, and nothing aliases it back, so `from file_io.wfn_loader
import WfnLoader` now raises `ModuleNotFoundError` rather than resolving
anywhere. Two spellings are supported and there is no third. The first is
`import wfn_loader` and then top-level names, after the metadata-derived
application seal has selected one coherent installed or source closure.
Core drivers obtain the seal from runtime startup. A direct-library caller
may use `ffi._services.ensure_on_path()`, which delegates to that same seal
and owns no independent path scan. Merely putting `<checkout>/src` on
`PYTHONPATH` does not select the service set. The second spelling is
`from file_io import WfnLoader` — or `WFNReader`, the back-compat alias
bound to the same class — which is the spelling most in-tree consumers
already use.

That second spelling is not a new shim wearing a different hat, and
`src/file_io/__init__.py` says so in its own words: "THE RE-EXPORTS
STAY, and they are not a new shim … What changed is where the object
comes from: the door, once, with no second module object in between."
`file_io` asks the service door for the class and re-exports the object
it gets; there is no intermediate module for a future reader to mistake
for the real one, which is exactly what the deleted shim was.

## API

| name | what it is |
|---|---|
| `WfnLoader(path, *, mesh=None, backend='auto', qe_schema=None)` | Open the WFN and check a 2c DFT reference before symmetry unfolding. `qe_schema=None` performs bounded WFN-anchored discovery at first `symmetry()`; an explicit schema must authenticate or the call refuses. |
| `load(*, bands, k='full_bz', sharding=None, bispinor=False, bispinor_lift="raw")` | ψ for a (band-window, k-set): `(n_k, nb_padded, ns, ngkmax)` c128, band axis mesh-padded and (by default at P>1) sharded `P(None,('x','y'),None,None)`. `bispinor_lift` names the four-spinor carrier: `raw` (σ·p, the shipped lift), `isometric` (library only), or `velocity_1/2/3` (one current carrier per Cartesian channel; branch `feat/bispinor-velocity-lift-2026-09-04`). The family name `velocity` is not a carrier and is refused by the lift. |
| `nonlocal_velocity_lift` (attribute, default `None`) | The driver-attached callable `f(psi_2, gvecs_int, kvecs_frac, ngk_valid, *, channel)` that `bispinor_lift="velocity_1/2/3"` needs: channel a's projector-velocity ket, `(n_k, nb, 2, ngkmax)`, Ry velocity units. The loader owns no projectors; a velocity request with no hook refuses by name (`GATE bispinor_velocity_lift_needs_projectors`). `gw_jax` and `centroid.kmeans_cli` attach `psp.vnl_ops.nonlocal_velocity_lift_from_pseudo_dir(...)` (which also carries `.provenance`), `psp.get_dipole_mtxels` attaches `psp.vnl_ops.nonlocal_velocity_lift(setup)`. |
| `lift(psi_2, *, k, bispinor_lift, sharding=None)` | Lift an already loaded `(n_k, nb, 2, ngkmax)` two-spinor window (same `k` request) into any four-spinor carrier without re-reading WFN.h5; how a consumer that needs several carriers of one window (three velocity channels plus the σ·p charge carrier) pays one read. |
| `spinorbit` (property) | QE's `<spinorbit>` from the authenticated schema bound at symmetry initialization, `None` when unbound. Read by `psp.vnl_ops.resolve_soc_mode`, so j-resolved/j-averaged projectors follow the DFT run's record instead of a multiplet measurement (which refuses as UNMEASURABLE on bulk Bi). |
| `load_process_local(*, bands, k, bispinor=False, bispinor_lift="raw")` | THIS process's window only, single-device, `nb = b_hi−b_lo` exactly — no mesh padding, no collective; each rank may ask for a different window. Same `bispinor_lift` selectors as `load`. |
| `bands(b_lo, b_hi, *, chunk, ...)` | Chunked iterator over `load`. |
| `full_k_parent_groups(full_k=None)` | Stable O(nk) grouping of requested full-BZ rows by raw IBZ parent. |
| `unfold_parent_to_full_k(parent_psi, *, parent, full_k, bispinor=False, bispinor_lift="raw")` | Apply the canonical typed unitary/antiunitary action to one already-loaded raw parent row. Consumers can realize a star with one child workspace and no parent re-read. Same `bispinor_lift` selectors as `load`; a `velocity_a` lift is evaluated on the child's own G list and k. |
| `full_k_box_index_one_dev(full_k)` | Build one child's replicated FFT gather index from the current parent G row; strict one-k streams avoid retained full-BZ G/index tables. |
| `ibz_box_index_one_dev(parent)` | Build the matching FFT gather index for one raw WFN parent without creating a complete IBZ index table. |
| `gvecs(k=...)` | `(n_k, ngkmax, 3)` int32, pad rows = the FFT-box **pad sentinel**, never zeros. |
| `ngk_valid(k=...)` | The mask that makes the pad rows discountable. The pair is the contract. |
| `box_index(k=...)` / `box_index_dev(...)` | FFT-box gather table, host / device-cached (the replicated-buffer-leak fix). |
| `adopt_mesh(mesh)` | Late mesh binding for drivers whose mesh cannot exist at construction; narrow by design; MAY RAISE (the refusal is the point). |
| `path`, `symmetry()`, `kpt_starts` | The public spellings of what consumers used to reach as `._filename` / `._ensure_sym()` / `._kpt_starts`. |
| `get_gvec_nk(ik)` | Deprecated one-k shim for legacy vcoul/qp_wfn; one release. |
| header surface | The `MfHeader` fields (`nkpts`, `nbands`, `nspinor`, `kgrid`, `fft_grid`, `bvec`, …) plus derived `nelec/vbm/cbm/efermi/atom_crys`, same names `WFNReader` exposed. |
| `trs_holds`, `trs_reference` | 2c occupied-subspace verdict and receipt. `density_symmetry` is a temporary compatibility alias. |
| `qe_symmetry_binding`, `qe_symmetry_diagnostic` | Authenticated per-operation unitary/antiunitary provenance, or the reason initialization must use and loudly announce the WFN-only fallback. |
| `close()` / context manager | Releases the h5py handle and the SlabIO handle. |

## Contract

* **The padding contract is a conjunction.** Band-axis pad rows of ψ are
  zero. G-axis pad columns of ψ are zero AND the matching `gvecs` rows
  beyond `ngk_valid` hold the pad sentinel (the Nyquist-corner Miller
  index — a cell no physical G occupies). Zero coefficient makes the slot
  inert; sentinel G-vector makes a dropped mask DETECTABLE instead of
  silently aliasing pad onto Γ. Mask detectable ≠ mask optional: consumers
  must carry `ngk_valid`. The L-c suite asserts the conjunction on a real
  sharded multi-rank load, both backends, hostile band counts.
* **Backend byte-identity.** eager and phdf5 produce byte-identical output
  for the same request — that is the only reason `LORRAX_WFN_BACKEND` is
  safe to expose. Verified bit-exact (`np.array_equal`, no atol) at world 4
  on hostile geometry, CPU and CUDA platforms.
* **Refusals.** A deleted spelling refuses rather than resolving elsewhere
  (`backend='phdf5_host'`, both doors). At P>1 with a mesh and no
  phdf5-capable library on either platform, `auto` REFUSES, quoting the
  door probe's reason per platform and the `LORRAX_WFN_BACKEND=eager` way
  through — no silent demotion. `phdf5` without a mesh refuses. Headers the
  coeffs slicing cannot serve refuse at construction, before anything reads
  the raw dataset: `flavor != 2` (real-flavor files; the re/im axis is
  hardcoded as 2) and `nspin != 1` (coeffs axis 1 is treated as the spinor
  axis alone, so nspin=2 would silently read spin-up only).
* **`load` vs `load_process_local`.** `load` returns one logical global
  array (every rank must request the same window); `load_process_local`
  is the k-parallel primitive (per-rank windows, combination explicit).
  They are different primitives, not a flag.
* **Parent/star streaming keeps one-k memory.** Every full-k star loads its
  raw IBZ parent once per band tile and realizes children serially through
  `unfold_parent_to_full_k`. One parent G row and its symmetry-bounded star of
  FFT indices are retained across those tiles, then released. A nonzero child
  phase is one separate device vector; zero translations skip it. No dense
  full-k phase, G-vector, or FFT-index table exists on this path. The
  four-component lift derives child G on device from the same parent row; a
  `velocity_a` lift hands that child G list, the child's k and `ngk_valid` to
  the attached `nonlocal_velocity_lift`, so the projector velocity is
  evaluated in the child's own gauge, and the resulting ket is one extra
  band-sharded operand of the lift kernel.
* **Late mesh binding is narrow.** `adopt_mesh` switches only a
  multi-process, auto-picked, currently-eager loader; explicit requests
  are never overridden.
* **The loader instance is open to attribute attachment.** `psp` attaches
  `grid_rho` at runtime; `__slots__`/strict `__setattr__` are forbidden
  forever (see Antipatterns).
* **QE schema binding is bounded discovery, not a search.**
  `qe_schema=None` calls `symmetry_maps.discover_qe_schema_paths(wfn_path)`:
  candidates are a `data-file-schema.xml` beside the WFN, or one inside any
  `*.save/` under `.`, `scf`, `nscf`, `qe/scf` or `qe/nscf`, anchored at the
  WFN's directory (given and resolved) and at most two directories above it.
  An explicit `qe_schema=` that does not authenticate against the WFN is a
  refusal; failure to find one in auto mode is not — `SymMaps` falls back to
  the **conservative legacy all-spatial header interpretation plus the global
  DFT-reference TRS verdict, and announces it** with a
  `SYMMETRY PROVENANCE WARNING` naming `qe_symmetry_diagnostic` as the reason
  (`services/symmetry_maps/src/symmetry_maps/maps.py:1958-1971`). What an
  authenticated binding pins: [`symmetry_maps`](symmetry_maps.md).

## Backends

| backend | transport | picked when |
|---|---|---|
| `eager` | per-rank serial h5py + host unfold | single-process, mesh-less, or forced; at P>1 each rank still reads only its band block |
| `phdf5` | `SlabIO.read_slabs` — ONE collective MPI-IO H5Dread over the k-window union, on-device unfold | multi-process + 2-D mesh + the door probe holds on either platform |

* The union read is a **slab_io door primitive** (`read_slabs`: n windows
  of one slab shape, per-window valid shapes, a window axis in the
  output). The per-rank band clamp — `max(0, min(slab, logical − offset))`
  per dim — lives in `_slab_io_ffi` three definitions below
  `_derive_valid_shape`, the clip it delegates to. Layout and clip in one
  file: the 22049c3 divergence class is structurally dead.
* **Why not n × `read_slab`?** Measured (step 0, P=4 CPU milan 2×2,
  JID **56446562**, both arms through the SAME `PhdfCtx`): the loop never
  wins — warm-min ratios **3.58× / 3.22×** at fixture scale and **1.44×**
  (+2.1 s) at MoS2 12×12 400b (144 IBZ windows, 15.6 GB), of which ~1.4 s
  is per-call collective `H5Dread` overhead (144 × ~42.7 ms vs one union
  call) and ~0.6 s is the extra `jnp.stack` the union path never needs.
  The cost axis is `n_reads` = the request's IBZ k-count, which is exactly
  the axis production decks grow along. Async overlap is not the story
  (~1% end-to-end, measured in-tree at `read_ffi.cc:819-829`), so these
  CPU-platform numbers carry to CUDA. The promoted door path reproduced
  the union numbers (**4.694 s** warm-min, **3331 MB/s**, JID
  **56456596**) — 0.7% faster than pre-promotion.
* Platform note: the same C++ read core serves both platforms; CUDA is
  the production path, and the L-c cells have run green on both.
* `stripe_count=1` files read through one aggregator at any rank count;
  the open announces the file's own stripe layout (rank 0) so the read
  side names its dominant term.

## Tests

Markers `services` + `wfn_loader`; standalone `pytest
services/wfn_loader/tests`; monorepo `pytest -m wfn_loader`; deselect via
`--no-services` / `--only-service=NAME`, never a second `-m`.

| tier | file | needs |
|---|---|---|
| contract (L-a) | `test_wfn_loader_contract.py` | the checked-in fixtures; refusals constructibly fire |
| L-b emulated 2×2 | `test_wfn_loader_emulated_mesh.py` | `XLA_FLAGS` via the service conftest; skips below 4 devices |
| L-c real multi-process | `test_wfn_loader_multiproc.py` | `srun -n 4`; shared `check_*` bodies + `_CLI_CELLS` |
| skip honesty | `test_wfn_loader_skip_honesty.py` | service-local (lxkit `_ARMED` is single-scope); perlmutter MUST-rows |
| import isolation | `test_wfn_loader_import_isolation.py` | `python -S`; sys.modules AND sys.path; red twin |
| layering + bootstrap (monorepo, not the service) | `tests/test_layering.py`, `tests/test_service_path_bootstrap.py` | nothing — pure AST + subprocesses |

* Hostile geometry runs on REAL checked-in decks (gnppm: mnband 82,
  82 % 4 = 2, ragged ngk 1917–1963), with the anti-tautology
  self-assertions (`(b_hi−b_lo) % world != 0`, pad slots > 0) so a future
  default change cannot quietly return the suite to the geometry that hid
  the band-pad bug for months.
* The 22049c3 negative control is a live cell: the perturbed (mnband)
  bound differs on the tail rank and matches at the divisible control.
* The graduated parity harness (`tests/bench/…parity_test.py`) shares the
  same `check_*` bodies — one implementation, defaults hostile, atol 0.0.
* Every check ships with the case where it returns FALSE.
* **A door consumer that forgets the application seal is a
  green-suite / red-cluster failure**, so the coverage is structural
  rather than sampled: `tests/test_service_path_bootstrap.py` walks the
  AST of `src/`, enumerates every module-scope importer of the door
  and asserts each has a canonical runtime or compatibility seal on a line
  STRICTLY ABOVE the import. Non-empty, ordered, and complete — a
  new consumer is a red cell until it is listed. Its red twin runs
  the same detector over a tree built to be wrong (missing bootstrap, late
  bootstrap, a function-scope lazy import, a level-1 relative import).
  The four subprocess cells that launch a real bare interpreter remain a
  SAMPLE; the AST cells are the population.
* Perlmutter floor, 2026-08-07, HEAD `c96674e3`, BUILD_NOTES pins —
  step-3 re-runs, independently re-read by the audit arm: `-m wfn_loader`
  **91 collected / 0 failed / 15 skipped**; service suite by path (with
  lxkit) **211 collected / 0 failed / 0 errors** (= step-2's 208 + the 3
  publics cells); L-c CLI cells **5/5** on CPU 2×2 (`lc_cpu_2x2.log`,
  `eager_vs_phdf5: 'bit-identical'` on both `ibz` and `full_bz`) and the
  GPU 2×2 leg green.
* The L-c log carries its own non-vacuity: per-rank clamped band counts
  `[3,3,3,1]` (rank 3 genuinely clipped), ragged `ngk (1917, 1963)`, and
  the 22049c3 negative control separating — `good [3,3,3,1]` vs
  `perturbed [3,3,3,3]`. A parity number with none of those visible is a
  parity number measured on the geometry that hid the bug.
* **Replumb evidence (step 3, two independent arms).** Old-path import
  census **45 edges / 36 files → 3 / 3**, converted delta **42**,
  reconciled exactly by both arms. Si COHSEX eqp data-line md5
  `139265eadb0fd1e96483e13d18e45fe8` before AND after — bit-identical,
  and recorded by the audit arm before the after-leg existed. Local
  full-suite set-diff vs the step-0 baseline: **empty both directions**
  (95 failing ids before, the same 95 after), 1534 collected.
* Census convention, so two counts of the same thing agree: a door edge is
  an **AST import node parsed from source**. `test_wfn_loader_import_-
  isolation.py` spells its door imports inside subprocess code STRINGS, so
  they are counted separately — a constant **+3 edges / +1 file**.

## Performance

Baselines (never slow tests) ride the step-0/1b measured runs; claims
files in `services/wfn_loader/bench/baselines/`, with
`bench/bench_wfn_loader.py` as the driver that regenerates them. **Step 6
CITES measured numbers; re-running is a future bench invocation, not a
test** — regression detection is diffing baseline files across branches,
never a threshold on a shared machine. Every row below: Perlmutter, CPU
platform, P=4 2×2 milan, complex128, BUILD_NOTES `.so` pins.

| read | deck / window | cold s | warm-min s | MB/s | jobid |
|---|---|---|---|---|---|
| `read_slabs` (door, union) | gnppm (0,10) hostile, 9 windows | 0.0246 | 0.0030 | 1862 | 56446562 |
| `read_slabs` (door, union) | gnppm (0,82), 9 windows | 0.0411 | 0.0147 | 3129 | 56446562 |
| `read_slabs` (door, union) | MoS2 12×12 400b (0,400), 144 windows, 15.6 GB | 4.98 | 4.727 | 3308 | 56446562 |
| same, AFTER the door promotion | MoS2 12×12 400b (0,400) | — | **4.694** | **3331** | 56456596 |
| n × `read_slab` loop — **REJECTED shape** | gnppm (0,10) hostile | 0.1138 | 0.0108 | 520 | 56446562 |
| n × `read_slab` loop — **REJECTED shape** | gnppm (0,82) | 0.1528 | 0.0474 | 971 | 56446562 |
| n × `read_slab` loop — **REJECTED shape** | MoS2 12×12 400b (0,400) | 6.39 | 6.804 | 2298 | 56446562 |

The rejected rows are kept on purpose: they are the fold ruling's
evidence, and a baseline table that records only the shape that won cannot
answer "was this worth it" the next time somebody proposes the loop.

Parity on the same three deck/window pairs: **bit-identical per rank**,
`np.array_equal`, no atol — including the hostile window, where the
per-rank clamped band counts are `[3,3,3,1]` summing to `nb_logical=10`.

The kchunk `ctx_handle`/`ds_id` compile-cache leak is MEASURED and
bounded: it fires per ctx-ADDRESS change, not per launch (byte-identical
relaunches at the same heap address grow zero entries; when the address
moved, the persistent cache grew by exactly the **2** modules carrying
`ctx_handle` as an FFI `Attr`; control flat at 1; `ds_id` constant). The
Attr→runtime-buffer conversion is registered with evidence and the HLO
quotes, gated on a coordinated `.so` rebuild.

**Not baselined yet** — see `bench/baselines/GAPS.md`: the GPU-platform
read timings (the L-c GPU 2×2 leg ran green but untimed) and the
large-deck CUDA leg. Both registered for post-wave.

## Antipatterns

* **A second copy of the clamp arithmetic.** The band clamp lives in
  `_slab_io_ffi._derive_window_counts` → `_derive_valid_shape` and
  NOWHERE else. The last local copy diverged silently on the band axis
  and produced real file bands in pad rows on every non-divisible
  geometry — invisible at the divisible defaults the old harness ran.
  A structural cell greps for re-inlining; do not make it fire.
* **`__slots__` or strict `__setattr__` on `WfnLoader`.**
  `psp/get_DFT_mtxels.py` attaches `grid_rho` to the instance at runtime
  and `gw/kin_ion_io.py` reads it back. Locking the class breaks `psp`
  silently.
* **Consuming `gvecs()` without `ngk_valid()`.** The pad rows are a
  sentinel, not zeros, precisely so this mistake is detectable —
  consumers refuse via `refuse_padded_gvecs_without_mask`. Dropping the
  mask used to add ngkmax−ngk copies of ψ(Γ) with no symptom.
* **An n × `read_slab` loop for a multi-window read.** Measured 1.44–3.6×
  slower (this page, Backends). `SlabIO.read_slabs` is the primitive.
* **FFI knowledge in the loader or its consumers.** No `ffi.phdf5`, no
  target strings, no ctx handles outside `slab_io`. The quarantine is
  structural; the layering door test and the empty grep are its gates.
* **`jax.device_put(numpy, multi-process sharding)` in loader paths.**
  It fires JAX's hidden `assert_equal` → allgather: 6.45 GB/rank at P=64
  for the g_index table (scorecard Y.3). `device_put_process_local` is
  the spelling.
* **Reading `._filename` / `._ensure_sym()`.** Public spellings exist
  (`path`, `symmetry()`); the privates survive for compat, not for new
  code.
* **Re-creating `src/file_io/wfn_loader.py`.** The file was deleted by
  the phase-wide cleanup and its absence is ratcheted, because a shim
  acquires a second life exactly one way: somebody hits the old import
  path, re-creates the file to green their branch, and the deletion
  silently un-happens. Putting it back turns
  `tests/test_service_path_bootstrap.py::test_the_retired_shim_files_are_gone`
  red with "wave-1 transitional shims are back". If something still
  reaches for the old spelling, migrate the caller to one of the two
  supported spellings above; the module itself lives in
  `services/wfn_loader/` and is edited there.
