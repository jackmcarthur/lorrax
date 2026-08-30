# wfn_loader — design brief (Fable, 2026-08-07)

Step-0 deliverable. Read with: surveys/w1_wfn_loader.md (evidence; line numbers
at e9340d1), WAVE1_BRIEF.md (binding rulings), CHARTER, SERVICE_FORM
(distrib_la is the mold). This document decides; the survey evidences.
Owner's constraint, verbatim: "a bloated piece of code i wish were shorter" —
FEWER LINES AND FEWER CONCEPTS; a prior consolidation was rejected as added
indirection. No new layers.

## What wfn_loader is

`services/wfn_loader/` — the single entry point for ψ(G) loading:
`WfnLoader(path, mesh=, backend=)` with `load` (global, band-sharded,
mesh-padded), `load_process_local` (single-device, no padding), `bands`
(chunked iterator), `gvecs`/`ngk_valid`/`box_index`/`box_index_dev` (the
sentinel-padded G surface), `adopt_mesh`, the 35-field mf_header surface +
derived band-fill. Drop-in for today's class; `file_io.wfn_loader` and the
`WFNReader` alias become shims that STAY until all wave-1 branches land
(coordination ruling 2).

## DECISION 1 — the fold question (the design brief's first decision)

Survey §3 established on paper that `SlabIO.read_slab` reproduces one
kchunk-union window exactly (per-rank band clamp included: the C++ `PhdfRead`
shard loop IS `_derive_valid_shape`), leaving one missing number: wall-clock
of `n_reads × read_slab` vs `1 × read_kchunk_union` at P>1 on real data.

**MEASURED 2026-08-07 (this branch's step-0 leg, CPU milan 2×2 mesh,
`lx run --cpu -n 4`, JID 56446562, BUILD_NOTES pins, both arms through the
SAME `PhdfCtx`; artifacts /pscratch/sd/j/jackm/svc_wfn_loader/_measure_fold/):**

| deck | window | arm | n_reads | cold s | warm reps s | warm-min MB/s |
|---|---|---|---|---|---|---|
| gnppm (nrk 9, mnband 82) | (0,10) hostile | U union | 9 | 0.0246 | .0043/.0031/.0030 | 1862 |
| gnppm | (0,10) hostile | S fold | 9 | 0.1138 | .0123/.0110/.0108 | 520 |
| gnppm | (0,82) | U union | 9 | 0.0411 | .0162/.0147/.0152 | 3129 |
| gnppm | (0,82) | S fold | 9 | 0.1528 | .0485/.0483/.0474 | 971 |
| MoS2 12×12 400b (nrk 144) | (0,400), 15.6 GB | U union | 144 | 4.98 | 4.727/4.759 | 3308 |
| MoS2 12×12 400b | (0,400) | S fold | 144 | 6.39 | 6.840/6.804 | 2298 |

Arm S NEVER wins: S/U warm-min = 3.58× / 3.22× / **1.44×**; at the production
deck the fold-down costs +2.1 s per ψ load at P=4, of which ~1.4 s is per-call
collective H5Dread overhead (144 × ~42.7 ms vs one union call) and ~0.6 s the
extra `jnp.stack` the union path never needs. The cost axis is n_reads = the
request's IBZ k-count — exactly the axis that grows with production decks.
The union path's async `ffi::Future` is NOT the story: the in-tree measurement
at `read_ffi.cc:819-829` records ~1% end-to-end from overlap, so these
CPU-platform numbers carry to CUDA. Parity: BIT-IDENTICAL per-rank on all
three deck/window pairs, including the hostile window (per-rank clamped band
counts [3,3,3,1] summing to nb_logical=10; pad rows exactly zero in both
arms; anti-tautology guards asserted: ragged ngk, (b_hi−b_lo)%4≠0).

**RULING: option (A) — promote the union read INTO the slab_io door as a
multi-window read primitive; fold-down (B) is REJECTED on the measurement;
(C) is dominated by (A) on both concept count and quarantine.**

Shape of (A), as taken:
- `SlabIO` gains ONE door method (`read_slabs`: n windows of a common slab
  shape, per-window valid shapes, a window axis in the output — the packed
  union-read semantics), implemented in `_FfiBackend` over the EXISTING
  `ffi.io.read_kchunk_union_sharded` machinery. The per-rank clamped-counts
  TABLE derivation moves there too, next to `_derive_valid_shape` — layout
  and clip finally live in the same file (the 22049c3 divergence class dies
  structurally). ~+80 lines in `_slab_io_ffi.py`, ~+30 in `slab_io.py`
  (wave-1b-unowned files; minimal, flagged, registered to the slab_io
  retrofit for template conformance).
- wfn_loader DELETES: `_ensure_phdf5_ctx` (42 lines — the hand-copied
  `_FfiBackend.__init__` guard block; SlabIO's constructor runs the real
  guards), `_build_phdf5_clamped_counts` (73+docstring — moves behind the
  door), the FFI half of `_phdf5_build` (~60 of 122), every `ffi.phdf5`
  import and FFI target name. Net ≈ −170 lines in the loader; survey
  concepts 11, 12, 13, 18 dissolve from it (the union read becomes a
  documented door primitive); concept 14 (the unfold kernel) stays.
  The charter's "only slab_io sees phdf5" becomes STRUCTURAL for this
  service. The file that grows is `_slab_io_ffi.py` — the file that already
  owns the C++ seam (the survey's §0.4 question, answered).
- The read-side stripe-layout announcement (the one non-duplicated thing
  `_ensure_phdf5_ctx` did) moves into the backend open-for-read path so the
  read side keeps naming its dominant term.
- Perf gate for the promotion commit: the harness re-run must reproduce the
  union-arm numbers above (same handler, same call count — any regression is
  a defect in the move).
- `read_kchunk_sharded`/`PhdfReadKchunk` (zero Python callers, survey §3.4):
  Python wrapper deleted; C++ handler deletion REGISTERED to the owner (Q2).

Kchunk leak (falsification doctrine — 96a6399 said it "could not be measured
here"; now it has been):

| run | ctx_handle | total entries | jit__per_rank | control | new |
|---|---|---|---|---|---|
| 1 | 14849664 | 12 | 4 | 1 | 12 |
| 2 | 14849664 | 12 | 4 | 1 | 0 |
| 3 | 14849664 | 12 | 4 | 1 | 0 |
| 4 (fresh dir) | 19920480 | 12 | 4 | 1 | 12 |
| 5 | 17895136 | 14 | 6 | 1 | **2** |
| 6 | 17895136 | 14 | 6 | 1 | 0 |

**The leak is REAL and REACHABLE but fires per ctx-ADDRESS change, not per
launch**: byte-identical launches reusing the same heap address show zero
growth; when the address moved (runs 4→5) the persistent cache grew by
exactly the two modules that carry `ctx_handle` as an FFI Attr — both
`lorrax_phdf5_read_kchunk_union` (xla_dump attribution, quoted HLO in
`_measure_fold/hlo_union_pid*.txt`), control flat at 1. `ds_id` was constant
across launches. Magnitude is therefore BOUNDED (entries per distinct ctx
address per geometry), unlike the 47%-dead-entry slab case 96a6399 fixed.
CONSEQUENCE: the Attr→runtime-buffer conversion (the identical mechanical
change, survey §5.2 edit table, Future-fail nuance included) is JUSTIFIED but
not urgent. It requires this branch's OWN .so pair. Decision gate: taken as
the final optional commit after step 6 IF the schedule (shared GPU pool, cert
window) permits; otherwise REGISTERED with this measurement, the HLO quotes,
and the recipe attached — the leak exists in production today and the door
promotion neither worsens nor masks it.

**EXECUTED 2026-08-08** on `fix/kchunk-cache-identity-2026-08-08` (owner
approval "kchunk conv yes"), survey §5.2 edit table applied verbatim with two
drift corrections: the table's `read_kchunk_sharded` row was already dead (the
wrapper had been deleted), and the union dispatch needed a SECOND failure
lambda, not one — `fail` announces through `ctx`, which does not exist yet at
the handle-copy, so the two pre-ctx failures use a `fail_early` that builds
its own Promise/Future pair.  One design addition the table did not carry: the
collective `H5Dopen` moved to its own `_open_dataset_memo` lru_cache, because
`read_kchunk_union_sharded` runs on EVERY `read_slabs` call — resolving `ds_id`
outside the old cache without re-memoising it would have added a collective and
leaked a `hid_t` per read.

The measurement was re-run as a CONTROLLABLE experiment rather than waiting on
ASLR: `ffi.io.open_file` caches contexts per PATH, so two paths to the same
bytes (a symlink) give two different ctx addresses on demand, at identical
geometry.  Same probe, same 4-process 2×2 CPU mesh, same private cache dir,
both arms:

| arm | tree + .so | 2nd-ctx lru miss | 2nd-ctx persistent Δ | union modules in xla_dump / carrying `ctx_handle` |
|---|---|---|---|---|
| PRE | `origin/main` 21d68e06 + deployed pair | 1 | **+1** | 4 / **4** (3 distinct addresses baked in) |
| POST | this branch + its own pair | 0 | **0** | 2 / **0** |

The PRE arm is the red twin: it shows the probe CAN see the growth, so POST's
zero means something.  ψ is bitwise identical through both contexts on all 4
ranks in both arms.  PRE's Δ is +1 rather than the +2 of the table above
because this probe loads ONE k-plan per context (`k="ibz"`); the original
loaded `ibz` + `full_bz`, i.e. two union geometries.  Per-geometry the number
is the same.

## DECISION 2 — service boundary

- IN: `src/file_io/wfn_loader.py` (the class + module kernels), nothing else.
- OUT, registered (survey Q3): `common/wfn_transforms.py` — it is the ψ
  transform layer and a CONSUMER (takes the loader as its `wfn` argument);
  absorbing it adds ~2000 lines and a layer. Its loader-touching sites are
  replumbed as consumer sites (ruling 1: unowned file). Roster question
  (own service vs file_formats) goes to the main Fable.
- OUT: `mf_header.py` (wave 1b file_formats), `kin_ion` (owner ruling: core;
  `gw/kin_ion_io.py` is already a clean client — survey §9.2 — its needs are
  exactly the public door: load / load_process_local / box_index /
  adopt_mesh / header fields).
- Sibling seams (ruling 1/2): `common/symmetry_maps.py`,
  `common/density_symmetry_check.py` belong to the symmetry_maps
  orchestrator. wfn_loader reaches the service through lazy runtime imports
  in `_ensure_sym` and `_run_density_symmetry_check`.
  REGISTERED to symmetry_maps: `density_symmetry_check` reads loader
  privates `._file`/`._kpt_starts`/`.ngk` (survey §2.1); `kpt_starts`
  becomes a public property on the door now, the `._file` read is theirs.
- `_shard_map.py` remains service-local. `_collectives.py` retains only the
  loader-specific `_local_shard_and_global_offset`; `device_put_process_local`
  has one implementation in `lxkit.placement` and is re-exported here for API
  compatibility.
- Stays imported lazily from lorrax (documented, runtime-only):
  `common.gvec_fft_box` (the sentinel contract is SHARED with zeta_loader —
  a private copy would fork the single source of truth),
  `runtime.padding.spec_divisor` (its own docstring forbids a second copy),
  `file_io.slab_io` (the door this service is a client of).
- Import-time property: stdlib + jax only; every lorrax import is inside a
  method. The import-isolation test (SERVICE_FORM) asserts exactly this.

## DECISION 3 — API v1 (drop-in; changes are subtractive or compat-preserving)

Public surface unchanged in name and semantics: `WfnLoader`, `load`,
`load_process_local`, `bands`, `gvecs`, `ngk_valid`, `get_gvec_nk`
(deprecated shim, stays one more release), `box_index`, `box_index_dev`,
`adopt_mesh`, `close`/context manager, header fields + derived
(`nelec/vbm/cbm/efermi/atom_crys` STAY here — moving them to mf_header is
1b cross-wave, registered as survey Q6).

- `path` becomes a public property; `_filename` stays as a compat attribute
  (**7** production sites at a96439c — 4 convertible: `qp_wfn:160`,
  `charge_density:122,286`, `current_density:145`; 3 held back behind
  legacy-handle fallbacks where the arrival is not a `WfnLoader` and `.path`
  would not exist: `dft_operators:160`, `get_DFT_mtxels:71,77`. Corrected
  from 6, which matched no reading of the tree — step-3 adjudication ruling
  5, B's tree count; the two defensive `getattr(wfn,'_filename',None)` reads
  in `gw_init:210`/`isdf_fitting:727` are the compat attribute's other
  documented use case, ruling 1. Zero-cost).
- `symmetry()` becomes the public accessor; `_ensure_sym` stays as an alias
  (2 production sites replumb to the public name).
- `kpt_starts` is public for the bounded raw-IBZ coefficient reader above.
- NO `__slots__`, NO strict `__setattr__`, ever: `psp/get_DFT_mtxels.py:1043`
  monkey-patches `grid_rho` onto the instance (survey §2.4.3). Documented in
  Antipatterns.
- Backend vocabulary unchanged: `auto`/`eager`/`phdf5`; both phdf5_host
  refusal doors and their tests survive VERBATIM (deleted-spelling-must-
  refuse doctrine); the four docstring history passages compress to one
  sentence + a docs pointer (~30 lines, survey §4's recommended ruling —
  the doctrine stays, the prose pointer moves to the service docs).
- Duplicate `__all__` (lines 89/1715): one deleted.
- Prose rule (survey risk 7): comments carrying measurements/incidents stay
  (charter); the deletion mandate is STRUCTURE (the fold), not prose.

## DECISION 4 — tests (step 2, SERVICE_FORM tiers)

Markers `services` + `wfn_loader`; deselection via the existing conftest
hooks; skip-honesty profile rows for h5py (always), slab_io/FFI presence
(perlmutter MUST), emulated-device floor.

- L-a (WSL, pure): k-resolution/_kplan algebra; sentinel-contract algebra
  (exists, migrates); the clamp arithmetic is now slab_io's
  `_derive_valid_shape` — the service keeps ONE structural cell asserting
  the fold didn't grow a second copy (the 22049c3 control-(A) lesson).
- L-b (emulated 4-device): eager path + padding round-trips.
- L-c (REAL srun -n 4, 2×2) — the contract's named mandatory cells, all on
  the checked-in hostile fixtures (survey §6.4: gnppm mnband=82, 82%4=2,
  ragged ngk 1917–1963):
  1. **Band-pad clamp class**: eager↔phdf5 parity at bands=(0,10)
     (nb_logical%world=2), atol=0.0, with the anti-tautology
     self-assertion `(b_hi−b_lo) % world != 0` and `min(ngk) < ngkmax`, plus
     the 22049c3 negative control (restore the mnband bound → red; the
     divisible row stays green).
  2. **Sentinel-mask conjunction**: for every (k, j) with j ≥ ngk_valid[k]:
     ψ[k,:,:,j] == 0 AND gvecs[k,j] == sentinel, on the SHARDED multi-rank
     load, both backends, hostile bands. "Mask detectable ≠ mask optional"
     — this is the cell 22049c3 lacked (survey §7.4).
  3. Parity harness graduation: `tests/bench/wfn_loader_backend_parity_test.py`
     repointed at the in-repo gnppm fixture (byte-size-identical to the
     dead-machine /pscratch file — survey §6.4; `--wfn` override kept,
     restage question registered as Q7), defaults flipped to non-divisible
     bands + atol=0.0, rebuilt on shared `check_*(mesh, ...)` bodies +
     `_CLI_CELLS` so the same functions are the pytest cells and the
     cluster legs.
- Import isolation + layering door rule cells land WITH the extraction
  commit (coverage never gaps).
- Red twins for every new refusal/guard (falsification doctrine).
- Step-1a adjudications (Fable, recorded): (i) `file_io.wfn_loader` is now a
  module-scope bootstrap consumer with NO bare-launch cell —
  `tests/test_service_path_bootstrap.py`'s parametrization is
  distrib_la-specific; step 2 generalizes it to (module, service) pairs and
  adds the wfn_loader cell (the "green suite, red cluster" class from the
  flagship's adjudication #1). (ii) `arm_skip_honesty` cannot be called by a
  second service — `lxkit/testing.py:801` `_ARMED` is a module-global a
  second caller would clobber (verified). wfn_loader's skip-honesty gate is
  implemented SERVICE-LOCALLY, consuming lxkit's machine-profile vocabulary
  read-only; per-scope arming in lxkit is REGISTERED as a change request to
  the main Fable (lxkit is frozen this wave, ruling 3).
- Registered test-side repairs OUTSIDE this service's files, fixed as
  flagged consumer-site commits: the two bare-`return` silent non-skips in
  `tests/test_sanity_gates_jax.py:849,882`.

## DECISION 5 — replumb scope (step 3, two-arm)

Consumer sites in files owned by NO wave-1 sibling: `common/wfn_transforms.py`
(`._ensure_sym` → `symmetry()`, loader-arg sites), `gw/kin_ion_io.py`
(no change needed beyond verification — already clean),
`psp/dft_operators.py`, `file_io/qp_wfn.py`, `centroid/*`, `bse/bse_io.py`,
`gw/sc_iteration.py`, etc. (survey §2 census is the site list). Sites inside
sibling-owned files (symmetry consumers in zeta_loader.py etc.) are
REGISTERED, not edited (ruling 1). Old import paths keep working via the
shims (ruling 2) — the replumb moves lorrax onto the door WITHOUT deleting
the shims; the shim-deletion gate is the phase-wide cleanup commit.
Verification floor mirrors REPLUMB_BRIEF: full-suite set-diff empty both
directions vs the step-0 baseline; Si COHSEX eqp BIT-IDENTICAL before/after
(pure plumbing); layering green; the L-c legs re-run post-replumb.

CENSUS CONVENTION (step-3 adjudication ruling 5, so the two arms stop
publishing different door numbers): a door edge is an **AST import node**
parsed from source, so `test_wfn_loader_import_isolation.py`'s door imports
— spelled inside subprocess code STRINGS, never parsed as this tree's
source — are counted SEPARATELY, a constant **+3 edges / +1 file**. That
offset is the whole difference between the two arms' door-edge counts
(7 → 49 without it, 10 → 52 with it); the OLD-PATH census both arms
publish, 45/36 → 3/3 with converted delta 42, is unaffected either way.

## Registered / owner rows (running list; final table in the land report)

- Q2: C++ kchunk handler deletion from the deployed .so pair (owner).
- Q7: parity fixture restage vs in-repo twin (owner intent).
- 26.5× kin_ion Lustre row = KNOWN_LORRAX_ISSUES "dipole/kin-ion, evidence
  7885316, legacy §dipole" (ruling 5): taken only if it falls out naturally.
- Broken consumers (survey §2.4): `w_from_eps0_0d_check.py` (dead attrs),
  `get_DFT_mtxels.py:80` dead `..io` import branch, `pivoted_cholesky.py:54`
  unused import, `bse_io.py:1734` unclosed loader, `orbital_magnetization.py:500`
  `wfn.nrk` hasattr-fallback — fixed only as separate flagged commits where
  in scope, else registered.
- `wfn_writer.py` zero tests / no round-trip (1b file_formats).
- `file_io/kin_ion.py` explicit-shape read_slab spellings (1b slab_io
  retrofit).
- KNOWN_FAILURES.md:49 stale [mos2] count row.
- Complete: `device_put_process_local` is owned once by `lxkit.placement` and
  re-exported by `_collectives`; `_local_shard_and_global_offset` remains
  loader-specific and `_shard_map.py` remains service-local.
- Observed during the leak measurement (not this service's): the
  `common/jax_compile_cache.py` agreement layer vetoes 2 entries on
  non-writer ranks (`vetoed=2, compiles=2` on ranks 1–3 vs 12/12 hits on
  rank 0) — evidence in `_measure_fold/B_leak.log`.
- slab_io retrofit note: `probe_availability` caches per-process WITHOUT a
  platform key and probes the WRITE handler as the availability proxy —
  fine on the deployed pair (handlers travel together), worth a row in the
  1b retrofit.

## Execution order

Measurement leg (done, above) → extraction+shim commit (lorrax stays green;
layering edits ride along) → fold commit (its gate: the L-c parity harness,
byte-identity) → step-2 suite → two-arm replumb → docs
(`docs/services/wfn_loader.md`, exact section order) → perf baselines
(`services/wfn_loader/bench/baselines/`, claims-style) → step-7 rebase onto
the flagship's final head + gates + land-readiness report. ONE git writer at
a time on svc/wfn_loader-2026-08-07; push freely; never main; never land.
