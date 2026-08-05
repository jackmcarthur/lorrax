# Padding audit — Scope 4: IO/FFI boundaries (2026-07-08)

Tree: `sources/lorrax_D` @ `agent/memplanner-cleanup` (HEAD 62b0365).
Uncommitted at audit time: **only `src/gw/gw_init.py`** (a g0_mu logical-clip on write, site C-11
below) plus untracked test files. NOTE: the prompt's example of in-flight work — the
`mu_logical_mask` arg in `ppm_sigma._prepare_sigma_state` — is in fact **already committed**
(62b0365 "PPM mode census + window statistics on LOGICAL modes only").

Judged against the ideal: pad born once at ingest; `(n_logical, n_padded)` carried in ONE place
(meta); consumers either structurally neutral or using ONE canonical helper.

## 0. Where the contract is stated

The disk-logical / memory-padded contract is stated **once, in three coordinated places that
reference each other** and is honored mechanically at the SlabIO boundary:

1. `src/runtime/padding.py` (module header, lines 1–29) — "arrays in memory may be padded to
   mesh-divisibility … files on disk store the logical (unpadded) extent." Canonical helpers:
   `padded_mu_extent` (THE μ round-up, honors `LORRAX_EXTRA_MU_PAD`), `pad_shape_to_mesh`,
   `pad_array_to_mesh`/`unpad_array_from_mesh` + `PadAxis`, `valid_shape_from_pad_meta`,
   `round_up_to_mesh_product`.
2. `src/file_io/slab_io.py` — `valid_shape=` on `write_slab`/`read_slab` (docstrings at
   37–42, 186–194, 206–208, 245–247): write clips the padded tail; read zero-fills it.
3. `src/file_io/_slab_io_ffi.py:196–246` — `_normalize_valid_shape`, ONE validator imported by
   **all three** backends (FFI, mpi_host, allgather).

`common/meta.py` carries the pair in one place: `n_rmu` (logical) / `n_rmu_padded` (via
`padded_mu_extent`), and `b_id_4_user` (logical) / `b_id_4` (padded) for bands, with the
consumer contract documented at meta.py:38–46 and 100–117.

## 1. Inventory (27 sites)

Classes: **N** structurally-neutral, **C** canonical-helper, **A** ad-hoc per-site logic,
**M** missing/latent bug. "~lines" = pad-specific lines incl. load-bearing comments.

### A. Contract + backend implementations (the clean core)

| # | Site | Class | ~lines |
|---|------|-------|--------|
| 1 | `runtime/padding.py` (whole module) | C (the helper itself) | 403 (≈180 code) |
| 2 | `slab_io.py` valid_shape API plumbing | C | 25 |
| 3 | `_slab_io_ffi._normalize_valid_shape` (196–246) — shared by all 3 backends | C | 50 |
| 4 | `_slab_io_ffi` write/read valid_shape plumbing (555–556, 581–583, 629, 640–641, 671–673, 716) | C | 15 |
| 5 | `ffi/phdf5/{write,read}.py` valid_shape pass-through (write:32–37, read:96–101, …) | C | 20 |
| 6 | `ffi/phdf5/cpp/write_ffi.cc:282–290,343–390` (per-rank hyperslab clip: empty selection past prefix, clipped last rank) + `read_ffi.cc:154–155,198–245,643–644` (clip + pinned-buffer `memset` 0 → pad cells exact zero) | C — ONE C++ implementation per direction | 40 |
| 7 | `_slab_io_mpi_host._clip_shard_to_valid` (94–140) + plumbing (250–260, 307–325) | C — per-transport re-expression of #3's contract | 55 |
| 8 | `_slab_io_allgather` write prefix-clip (141–170) + read zero-embed (204–252) | C — per-transport | 20 |

The tail-rank clipping question the prompt asks: it is implemented **once per transport**
(3 implementations, unavoidable — different I/O primitives), but all three validate through the
single `_normalize_valid_shape` and state "same padding contract as PHDF5" explicitly
(`_slab_io_mpi_host.py:27`, `_slab_io_allgather.py:140–141`). This is honored mechanically, not
re-derived per reader/writer.

### B. Producers/consumers over SlabIO

| # | Site | Class | ~lines |
|---|------|-------|--------|
| 9 | `gw/isdf_fitting.py` zeta_q_G create (680–694, logical shape) + FFI write (971–981, `valid_shape=` clips μ pad) | C | 12 |
| 9M | **same function, allgather branch (982–992)**: bypasses SlabIO, raw h5py `_f['zeta_q_G'][...] = _g` where `_g` is the **padded** gathered `gflat_acc` `(n_q, n_rmu_padded, ngk)` and the dataset is logical `(n_q, n_rmu, ngk)` | **M — latent bug** | 10 |
| 10 | `gw/isdf_fitting.py:955–968` per-q ngk sentinel-slot mask before write (G-axis raggedness, not μ pad) | A (necessary; different convention, see §3) | 14 |
| 11 | `gw/gw_init.py:511–517` **UNCOMMITTED**: g0_mu write clipped `np.asarray(G0_gathered)[..., :meta.n_rmu]` before raw-h5py `create_dataset` | A (correct; hand-rolled clip because it's a rank-0 h5py write, not SlabIO) | 6 |
| 12 | `file_io/zeta_reader.py` `valid_mu=` → `valid_shape=` (162–183, 237–285) | C | 14 |
| 13 | `file_io/zeta_loader.py` `valid_mu`/`valid_shape` read+write paths (194–275, 481–560) | C, but a **duplicate** of #12 (deferred merge = task #8) | 30 |
| 14 | `zeta_loader.py:122–124` comment: "G-flat: (n_q_disk, **n_rmu_padded**, ngkmax)" | **M — doc drift**: the writer (#9) clips to logical; comment contradicts the contract and invites a future reader to assume padded disk | — |
| 15 | `gw/v_q_g_flat.py:355–366` `_pad` = `padded_mu_extent`; read at padded `mu_count` with `valid_mu=n_rmu_logical` (249–261) | C (post-fix: previously computed its own round-up — the knob exposed and killed that, ROOT_CAUSE §Fixes 1) | 12 |
| 16 | `gw/v_q_bispinor.py` `_padded_shape_LR` via `padded_mu_extent` (497–516); writes/reads with logical `valid_shape` (330–395, 519–560) | C | 35 |

### C. WFN band/G padding at load

| # | Site | Class | ~lines |
|---|------|-------|--------|
| 17 | `common/meta.py:100–109` `b_id_4 = _round_up(b_id_4_user, world_size)` + contract comment | C in role (the ONE band-pad birth) but uses a **private `_round_up`** (meta.py:8) instead of runtime.padding | 12 |
| 18 | `common/meta.py:127–129` `nbnd_jax`/`n_rtot_jax`/`n_rmu_jax = _round_up(·, n_proc)` — legacy parallel pad fields with a **different divisor** (process count, not device count); only remaining writer `gw_init.py:200–215` (transverse refresh), no readers found | A — near-dead duplicate convention | 8 |
| 19 | `common/meta.py:130–136` `n_rmu_padded = padded_mu_extent(...)` | C | 7 |
| 20 | `file_io/wfn_loader.py` `_pad_to` (517–520) + `_default_sharding` p_band derivation (483–515) | A — third private round-up implementation (correct; must agree with meta's `b_id_4`) | 20 |
| 21 | `file_io/wfn_loader.py:71–118` `_build_phdf5_clamped_counts` — tail-rank band clamp for the phdf5 kchunk-union read (past-EOF ranks get count 0, pinned-buffer pre-zero = zero pad rows) | C — single well-documented helper | 48 |
| 22 | `wfn_loader.py` ragged-G handling: `ngkmax`-padded G axis, `ngk_valid` per-k logical (319–320, 357, 404–406, 767–771) | C (the WFN.h5 file-format convention; masked at gather) | 15 |
| 23 | `common/psi_G_store.py` populate loop: past-EOF bc skip/zero (207–247), short-load zero-concat (255–267), `_zero_user_band_pad_in_shard` (61–91, applied 273–278) | A — **three** distinct band-pad mechanisms in one loop, each per-site (well-commented; partially forced by `WfnLoader.load` bounds-check + b_id_4_user semantics) | 80 |
| 24 | `psi_G_store._slice_local_tile_bc` (306–345): bc tiles zero-padded to static `_bpd_max` shape for io_callback/scan; `np.zeros`-not-`np.empty` noted as load-bearing | A-but-necessary (static-shape constraint) | 20 |

### D. FFI solver divisibility (cusolvermp / cholesky_2d)

| # | Site | Class | ~lines |
|---|------|-------|--------|
| 25 | `common/cholesky_2d.py:57,125–126` — asserts `n % b == 0`, `J % Px == 0`, `J % Py == 0` (J = lcm(Px,Py)); **no padding here** — fail-loud precondition, caller pads | N (fail-loud) | 3 |
| 26 | `ffi/cusolvermp/batched.py:130–132,205,353–357`, `eigh.py:94` — raises on N/Mrhs/NRHS not divisible by Px/Py; docstring (batched.py:186) delegates padding to caller | N (fail-loud) | 10 |
| 27 | `isdf/core.py` caller-side NRHS/zchunk pads: potrs branch (1263–1270), getrs branch (1293–1315, incl. the Fix-1 logical μ slice), even-share zchunk pad (1344–1347, 1474–1476, 1529, 1553) — each hand-rolls `((n + d − 1)//d)*d` + `jnp.pad` + post-slice | A ×3 — same 4-line idiom repeated inline | 30 |

## 2. Latent-bug candidates

1. **CONFIRMED-by-reading (crash-type, not silent): `isdf_fitting.py:982–992`.** The
   H5PY_ALLGATHER zeta write gathers the **padded** `gflat_acc` and assigns it whole into the
   **logical**-shaped dataset. Whenever a μ pad exists on the allgather path (multi-device
   single-host GPU run with `n_rmu % P ≠ 0`, or any run with `LORRAX_EXTRA_MU_PAD` set — i.e.
   the Tier-1 gate itself if a fixture ever selects this backend), h5py raises a broadcast
   error. Fails loud, so no wrong numbers — but it is exactly the defect class ROOT_CAUSE names:
   a consumer that had to *individually remember* the clip, and didn't. The fix is a deletion:
   the allgather backend's own `write_slab` already implements the prefix clip
   (`_slab_io_allgather.py:140–170`), so the special-case branch should not exist.
2. **Doc drift that will seed the next bug:** `zeta_loader.py:124` documents the on-disk G-flat
   ζ as `(n_q_disk, n_rmu_padded, ngkmax)` — false since the `valid_shape` write clip; a reader
   trusting it would size logical extents off `n_rmu_disk` ≠ padded and "re-pad" wrongly.
   Similarly `slab_io.py:191` still points to "the agent/padding-refactor branch" for
   `runtime.padding`, which is long since on main.
3. **Divisor ambiguity held in Meta:** `n_rmu_jax`/`nbnd_jax`/`n_rtot_jax` round to
   `jax.process_count()` while `n_rmu_padded`/`b_id_4` round to `jax.device_count()`. On
   multi-GPU-per-process these differ. No reader found, but one writer keeps them alive
   (`gw_init.py:200–215`), so the wrong one is available to grab.

## 3. Line count and verdict

Pad-specific lines in this scope: **≈ 900 total** — 403 in the canonical `runtime/padding.py`
(≈180 code, deliberately docs-heavy), ≈ 225 in the SlabIO/phdf5 contract implementation
(sites 2–8), ≈ 280 at producer/consumer/load sites (9–24), ≈ 43 in solver divisibility +
caller pads (25–27). Of the ≈ 280 consumer-side lines, **≈ 140 are ad-hoc** (sites 9M, 10, 11,
18, 20, 23, 24, 27).

**Verdict: the IO/FFI boundary is the cleanest layer of the padding scheme — the
disk-logical/memory-padded contract is genuinely stated once and honored mechanically**
(`valid_shape` + one shared validator + one C++ clip per direction; readers derive logical
extents from the file, writers clip via the helper). Post-fix, all μ round-ups at the V_q/ζ
boundaries route through `padded_mu_extent`. The residual dirt is *around* the boundary, not
in it:

- one write branch that bypasses SlabIO and misses the clip (§2.1 — a deletion fixes it),
- round-up arithmetic written **five separate times** (`meta._round_up`,
  `wfn_loader._pad_to`, 3× inline in `core.py`) instead of one `round_up(n, d)` in
  `runtime.padding`,
- three legacy `*_jax` pad fields in Meta with the wrong divisor,
- the known twin zeta readers duplicating the `valid_mu` plumbing (~30 dup lines; task #8 —
  note its stated blocker, "needs padded-μ gate", is now **cleared** by
  `tests/test_mu_pad_invariance.py`),
- `psi_G_store`'s three-mechanism band-pad populate loop (defensible: static-shape io_callback
  + past-EOF + user-band-stop are genuinely different constraints, but a single
  "load bc, zero-padded, user-stop-masked" helper would collapse ~80 lines to ~40).

Concretely achievable now, without behavior change: **≈ −60 lines and 3 fewer conventions**
(delete allgather special-case; delete `n_rmu_jax`/`nbnd_jax`/`n_rtot_jax` + refresh; add
`round_up(n,d)` to runtime.padding and retire the 5 private copies; fix 2 doc drifts), plus
**≈ −350 lines** when the deferred zeta reader merge lands. One convention that must stay
separate and should just be *named* in padding.py's header: the ragged **G-axis** `ngkmax`
padding (WFN.h5 file format, per-k `ngk_valid`, sentinel-slot masking) is a different contract
from mesh-divisibility μ/band padding and is intentionally not `valid_shape`-shaped.
