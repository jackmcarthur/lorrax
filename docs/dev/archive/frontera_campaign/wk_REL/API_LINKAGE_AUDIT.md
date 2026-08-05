# API linkage audit — how far LORRAX is from "drivers contain only physics"

*Read-only audit of `/work2/08271/jackmc/frontera/lorrax`, branch
`fix/zq-band-gather-device-invariance` @ `49877c0`, working tree carrying
uncommitted work from five workstreams. Audit window **2026-07-30 16:47–17:35 CDT**.
Nothing under the repo was modified.*

**The question.** The owner wants `gw/gw_jax.py`, `bse/bse_jax.py`,
`gw/kin_ion_io.py` and the kmeans drivers to "appear to contain only physics on
inspection and all use unified wrapped functions for the layers below jax." This
is the map of how far away that is, and what the target should be.

**Two things changed the shape of this report:**

1. **The target architecture is already being built, right now, in the working
   tree.** `src/common/collectives.py` changed under this audit between 16:47
   and 16:53. Its new `__all__` (`common/collectives.py:68-98`) is a declaration
   of exactly the service this report would otherwise have had to invent. The
   gap is not design. The gap is **migration**, and it is currently near-total.
2. **`mkl_thread_pin.h` does not exist.** The brief's exemplar divergence was
   read out of a doc that *proposes* creating that header
   (`docs/dev/vendor_gemm_service.md:180-186`, "the fix is a new MPI-free
   `ffi/common/cpp/mkl_thread_pin.h` … Not done in this wave"). The doc is
   accurate and honest; the reading turned a proposal into an artifact. See
   §B.6.0 — and note this is standing lesson #2 running in reverse: a *positive*
   grep hit in prose was taken as evidence of a file.
   **The three-copy divergence it describes is real, and I found a worse one.**

---

> ## SECOND SNAPSHOT — 17:08 CDT. The tree moved under this audit, a lot.
>
> Between my first driver read (16:47) and my final verification pass (17:08),
> the live workstreams landed most of the migration this report recommends —
> and, in doing so, created **one new collision worth catching before it sets**.
> Full detail in **§C.7**. In one table:
>
> | | 16:47 | 17:08 |
> |---|---|---|
> | `gw/kin_ion_io.py` raw jax imports / env reads | 6 / 4 | **1 / 0** (1088 → 814 lines) |
> | `centroid/kmeans_cli.py` raw jax imports | 3 | **0** |
> | `centroid/kmeans_isdf.py` global `config.update` | yes (`:34`) | **gone** |
> | `common/collectives.py` `__all__` names undefined | 9 | **0** (1045 lines) |
> | production `Mesh(` construction sites | 21 | **8** |
> | device-layer services | 1 (declared) | **3 (built in parallel)** |
>
> **The one thing to act on today:** `single_device_mesh` now exists **twice
> with opposite contracts** — `common/collectives.py:242-256` returns a
> *per-rank* mesh and is safe at P>1; `centroid/distribution.py:148-165`
> **raises** at P>1 and returns the *global* device 0. A reader who learns one
> and then meets the other gets it exactly backwards. Rename one. Five minutes
> today; a permanent trap in a month.
>
> Everything below §C.7 describes the 16:47 state unless a line says otherwise.
> That state is still the right map of *how the tree got here* and of the
> ~60 % that has not moved (BSE, `bandstructure`, `gw_jax`'s mesh, the whole
> C++ layer, every layering violation in §A.3).

---

## Reading rules

* Every claim carries `file:line` I opened. Where I am inferring rather than
  confirming, I say so.
* A `grep`/`strings` negative is never cited as evidence of absence
  (`wk_REL/README.md` lesson #2). Absences are established by AST parse, by
  `git ls-files`, or by `nm -C` on a built `.so` — never by `strings`.
* Three agents were editing `src/common/`, `src/centroid/` and `src/psp/` while I
  read. Every driver quote in §C carries an md5 and a clock time.
* Ranking everywhere is **(value to a future reader) / (risk of the change)**.

---

# 0. Summary for the owner

## The divergence-class finding the brief asked for

**`ensure_mpi_initialized` exists in two copies, and only one of them checks
what MPI actually granted.**

| | `src/ffi/phdf5/cpp/context.cc` | `src/ffi/slate/cpp/context.cc` |
|---|---|---|
| init call | `:78` `MPI_Init_thread(…, MPI_THREAD_MULTIPLE, &provided)` | `:31` **identical call** |
| checks `provided`? | **yes** — `:200-217` `MPI_Query_thread`, rank-0 warning naming the **measured ~29 % multi-node crash rate** (scorecard AS.4b) and the fix (`MPITRAMPOLINE_LIB` → `config/frontera/build_mpiwrapper.sh`) | **no** — `provided` is captured at `:30` and **discarded**; no `MPI_Query_thread` anywhere in `slate/cpp/` |

Same process, same `libmpi`, same hazard. SLATE's own header comment
(`slate/cpp/context.cc:6`) states it needs `THREAD_MULTIPLE` for its internal
OMP+MPI. If jax's MPI collectives (or an unpatched MPIwrapper) initialise MPI
first at a weaker level, **phdf5 announces the hazard and SLATE proceeds in
silence.** This is exactly the `RTLD_DEFAULT`/`RTLD_NEXT` class — one copy
hardened, one not — except it is **not masked by anything**. The pin divergence
is currently latent (§B.6.1); this one is live.

## The three worst layering violations

**V1 — `gw/gw_config.py` is the whole project's config module, but it lives
inside the GW driver package.** Nine import sites in five modules across three
packages reach *up* into `gw` for it: `file_io/slab_io.py:66` (with the comment
`# avoid circular import at module load`), `:94`, `:127`;
`psp/get_DFT_mtxels.py:740`; `psp/get_dipole_mtxels.py:42` (**module-level** —
`psp` cannot be imported without importing `gw`); `psp/run_sternheimer.py:1492`;
`bse/bse_io.py:581`, `:625`; `bse/exciton_bands.py:395`; `bse/vq_interp.py:1296`.
`SlabIOBackend` — the enum that *types the file-IO service's own backend
parameter* — is defined in the GW driver's config file, and
`file_io/slab_io.py:93`'s own public docstring instructs users to
`from gw.gw_config import SlabIOBackend`.

**V2 — nobody owns collective warm-up, so physics kernels do it.** Two warm-up
services with different platform gates: `runtime.nccl_warmup`
(`runtime/__init__.py:607`, GPU/NCCL, called only from `gw/gw_jax.py:115` and
`gw/kin_ion_io.py:964`) and `common.collectives.warm_mesh_cliques`
(`common/collectives.py:638`, CPU/MPI, no-ops unless
`JAX_CPU_COLLECTIVES_IMPLEMENTATION == mpi`). **`gw/gw_jax.py` never calls
`warm_mesh_cliques`.** Under `impl=mpi` the GW driver's mesh is clique-warmed
only *incidentally*, by whichever physics kernels happen to run:
`common/zeta_projection.py:422, :511, :547, :596, :864` and
`common/contract_bands.py:542`. A ζ-projection factory is performing
distributed-runtime initialisation because no driver does. That is the layering
inverted, and it is the mechanism behind the recorded "`impl=mpi` upgrade is
GW-ONLY — it kills BSE" (git `b4c7bca`).

**V3 — `gw/sigma_dispatch.py:174` imports the CLI driver `gw/kin_ion_io.py`,
and that single edge is why `kin_ion_io` cannot use `bootstrap()`.**
`kin_ion_io.py:60-64` hand-rolls four `os.environ.setdefault` calls and `:73-78`
hand-rolls the three-call bootstrap header; the comment at `:66-72` states the
reason verbatim — the module must survive being lazily imported *from inside an
already-bootstrapped driver*. **The library→driver edge is what forces the only
top-of-module env writes in any audited driver.**

## The three highest-value unifications, with their gates

**U1 — `process_allgather` / barrier → the API `common/collectives.py` already
declares.** This is the biggest one, and it is bigger than the brief's "≥4
places". There are **five** hand-rolled host-gather wrappers in production, and
**four of them name another of the five in their own docstring as the thing
they are copying**: `file_io/_slab_io_allgather.py:66` (the original),
`solvers/davidson.py:54` and `bse/bse_davidson_helpers.py:48` (byte-identical
bodies, both citing the original at `:50` / `:44`), `bse/exciton_bands.py:125`
and `bse/vq_interp.py:557` (a second byte-identical pair), plus a sixth variant
at `gw/ppm_windows.py:193`. **They disagree on `tiled` (3 True / 3 False) and on
the fast-path predicate** (`is_fully_addressable` ×2, `is_fully_replicated` ×1,
`process_count()==1` ×2, unguarded ×1) — and `exciton_bands.py:117-119`
documents that taking the wrong branch **silently multiplies the leading axis by
P**. Meanwhile `common/collectives.py` exposes no gather at all today; its
`__all__` promises four (§B.3).
The barrier half is sharper still: `file_io/_slab_io_ffi.py:48-55` is
`try: multihost_utils.sync_global_devices(tag) / except Exception: pass` — the
**exact** block `common/collectives.py:37-47` quotes as the cause of the
campaign's signature hang-then-rc-0 — **in a file that already imports from that
service seven lines earlier** (`_slab_io_ffi.py:41`). Same shape at
`gw/gw_output.py:294-298`.
*Gate:* cache-**cold** HLO collective table (`ISDF_JAX_CACHE_DIR=""`, lesson #3)
before/after — `all-gather` count non-increasing, no new `jit(_identity_fn)`.
Plus a P=4 run with a deliberately broken barrier tag to prove the `except: pass`
removal actually makes it fail. The measured stake is recorded at
`common/collectives.py:341-360`: JAX's hidden `assert_equal` all-gather cost
**17.4 GB/rank** at the production mesh.

**U2 — mesh construction → `common.collectives.resolve_mesh` /
`single_device_mesh`.** 21 production sites, 5 mutually incompatible dialects
(§B.1). The API landed at 16:52 today (`common/collectives.py:232`, `:261`);
**0 of 21 sites have migrated.** Three of them (`centroid/charge_density.py:108`,
`:245`, `centroid/pivoted_cholesky.py:119`) build
`Mesh(jax.devices()[:1].reshape(1,1))` — process 0's device on *every* rank —
which the new `_require_addressable` guard (`common/collectives.py:209-229`)
names as "the usual cause" of a bare `StopIteration` at P>1.
*Gate:* a `pytest` that AST-walks every module under `src/` for an `ast.Call` to
the name `Mesh` and fails on any hit outside `common/collectives.py` and
`common/wfn_transforms.py`. Seed it with a deliberate `Mesh(` first to prove it
fires (lesson #1). Plus a P=4 CPU `kmeans_cli` run byte-compared against
`centroids_T_t134.txt`.

**U3 — one boolean grammar for env knobs.** There are **five live grammars** and
they disagree on real inputs (§B.5). The canonical one exists, is documented,
is proven against a real bug — and is **private**: `runtime/__init__.py:53`
defines `_FALSY_TOKENS` as "the canonical LORRAX boolean grammar … ONE token set,
shared by every knob in this module", and `__all__` at `:60-68` does not export
`_env_falsy`. Two consequences worth the owner's attention:
* `LORRAX_MALLOC_TRIM=OFF` / `=No` / `=FALSE` all leave the trim **ON**
  (`gw/isdf_fitting.py:907`: `not in ("0","off","false")`, no `.strip()`, no
  `.lower()`), while its sibling `LORRAX_MALLOC_TUNE=OFF` correctly turns off
  (`runtime/__init__.py:395`). `docs/dev/env_vars.md:116-117` presents them as
  one family.
* **`LORRAX_EXIT_AFTER_ZETA=0` ends the job** — `gw/gw_init.py:1007` is a bare
  presence test (`if os.environ.get(...)`) followed by `raise SystemExit(0)` at
  `:1011`. The value that reads as "disabled" truncates the run after ζ-fit,
  cleanly, **with rc=0** — QUALITY_PATTERNS §7's own failure class, armed by a
  typo.
*Gate:* one table-driven unit test — for each of ~15 boolean knobs, assert that
`{"", "0", "off", "OFF", "no", "False", " 0 "}` all disable and
`{"1","on","true","yes"}` all enable. It fails today on at least
`LORRAX_MALLOC_TRIM`, `LORRAX_RESTART_WRITE_LOG` and `LORRAX_SKIP_VQ_GATES`;
that is the proof it can fire.

## The three false-unification traps

**T1 — do NOT merge `runtime.nccl_warmup` and `collectives.warm_mesh_cliques`.**
`warm_mesh_cliques` is correct *only because* its jit is small enough that XLA
takes `ThunkExecutor::ExecuteSequential` and runs the thunk **inline on the
calling thread** (`common/collectives.py:653-660`) — that inlining *is* the
mechanism that satisfies `MPI_Is_thread_main`. `nccl_warmup` deliberately avoids
`lax.psum` and says why (`runtime/__init__.py:635-637`); its job is to pay
`ncclCommInitRank` topology discovery, which has no thread constraint. **Unify
the call site, never the bodies.**

**T2 — do NOT collapse the two *deliberate* boolean grammars.** U3 is about the
five *accidental* ones. `runtime._FALSY_TOKENS` (two-valued, `""` is falsy) and
`ffi.common.gate.MODE_SPELLINGS` (`ffi/common/gate.py:91-95`, three-valued, `""`
maps to the gate's declared *default*, never to a spelling, `:87-89`) are
different **on purpose**: `auto` must be distinguishable from an explicit `on`,
because `LORRAX_BANDS_GEMM_FFI`'s AUTO default never refuses while an explicit
`on` must announce-or-raise. Share the token table; keep the two resolvers. The
related sub-trap: any unified rank resolver must keep launcher env vars **ahead
of** `jax.process_index()` — `ffi/common/gate.py:56-61` records that the latter
goes through `get_backend()` and **initialises the XLA backend**, destroying
tier 1's before-`initialize` promise and the kernel-cache keys in
`gw/ppm_tau_kernel.py`.

**T3 — do NOT build a generic `shard_map` wrapper.** 50 call sites is a tempting
number, but `in_specs`/`out_specs` *are* the distributed algorithm; a wrapper
that takes them as arguments has abstracted nothing and has added a layer between
the reader and the only thing they came to read. The evidence that the right
abstraction is *specific* is `common/contract_bands.py`: it replaces a whole
family of `shard_map` sites not by wrapping `shard_map` but by owning **one named
pattern** with its axis-order policy, its de-promotion policy and its
divisibility refusal — which is why it can carry
`docs/dev/staged_reshard_primitive.md`. A generic wrapper could carry no such
document. The counter-example is decisive: `common/fft_helpers.py:253` and `:271`
are the only two functions in the tree written *to be* reusable `shard_map`
wrappers, and they are two of the 21 sites that omit `check_rep`. Genericity did
not make them safer.

## The single change I would make first

**Rename one of the two `single_device_mesh` functions.** As of 17:08 the name
means opposite things in two live modules: `common/collectives.py:242-256`
returns a **per-rank** mesh (`jax.local_devices()[:1]`) and is safe at P>1;
`centroid/distribution.py:148-165` **raises** at P>1 and returns the **global**
device 0. Both are individually correct and well-documented; the collision is
hours old and was created by the consolidation itself (§C.7). Suggested split:
keep `single_device_mesh` for the per-rank meaning (`collectives`) and rename
centroid's to `one_process_mesh`. Cost: a rename plus three call sites.
*Gate:* a test that calls both at `process_count()==2` and asserts they differ
in the documented way — it cannot pass vacuously, because today one of them
raises and the other does not, which is the whole point.

**The single *structural* change I would make first — move `SlabIOBackend` —
and only `SlabIOBackend` — out of `gw/gw_config.py` into `file_io/`,
re-exporting from `gw.gw_config` for compatibility.**

On (value)/(risk) it wins: it is the narrowest cut that severs the `file_io → gw`
back-edge, it deletes three lazy "avoid circular import" imports that exist purely
to paper over the violation (`file_io/slab_io.py:66, :94, :127`), and it lets the
I/O service's public docstring stop telling users to import from the GW driver
(`:93`). It is an enum move plus a re-export — no behaviour change, no collective
moves, no shape changes.
*Gate:* `python -c "import file_io"` with `src/gw` renamed away must succeed. Run
it **before** the move to prove it errors on `ModuleNotFoundError: gw`; then
`pytest tests/ -q` byte-identical.

---

# A. The layering map

## A.1 The layers

Assigned from what each module does and imports, not from its directory name.
Edges are from an AST parse of all 236 `.py` files under `src/` (Appendix 1).

| Layer | Package / module | Role |
|---|---|---|
| **0 runtime** | `runtime/` (`__init__.py`, `aot_memory.py`, `padding.py`) | env defaults, `jax.distributed`, CPU fallback, failfast hook, AOT memory model, pad arithmetic. Imports jax **only inside function bodies** — deliberate and correct (`runtime/__init__.py:22-24`). |
| **1 FFI** | `ffi/common/`, the 8 backend packages, `ffi/linalg/` | native-library location, probing, gating, dispatch. |
| **2 service** | `common/{collectives,contract_bands,timing,progress,sanity,provenance,gpu_utils,jax_compile_cache,meta}.py`, `file_io/*` | cross-process API, the reshard primitive, instrumentation, IO. |
| **3 physics kernel** | `common/{zeta_projection,wfn_transforms,symmetry_maps,fft_helpers,gvec_fft_box,cholesky_2d,psi_G_store,minimax,…}.py`, `isdf/core.py`, `solvers/`, `mixing/`, `centroid/{kmeans_isdf,pivoted_cholesky,charge_density,orbit_syms}.py`, `psp/*` (non-CLI), `gw/{w_isdf,ppm_*,screening,isdf_fitting,head_correction,…}.py`, `bse/{bse_ring_comm,bse_stack_matvec,bse_lanczos,vq_interp,…}.py`, `bandstructure/htransform.py` | the physics. |
| **4 driver** | `gw/gw_jax.py`, `bse/bse_jax.py`, `gw/kin_ion_io.py`, `centroid/kmeans_cli.py`, `psp/{run_nscf,run_sternheimer,get_dipole_mtxels,get_DFT_mtxels}.py`, `bse/{bse_feast,bse_pseudopoles,bse_w_exact,bse_kpm}.py` | CLIs. |
| **config (unplaced)** | `gw/gw_config.py` | belongs in its own layer; currently inside layer 4's package. |

Two modules resist the taxonomy, and that is itself the finding:

* **`centroid/kmeans_isdf.py`** is filed as a driver in the brief and as "the
  algorithm library, no `__main__`" in `AGENTS.md:48`. It is a layer-3 kernel
  that behaves like a layer-4 driver in one respect: it runs
  `config.update("jax_enable_x64", True)` at **module scope** (`:34`). Importing
  a kernel flips a global jax setting.
* **`gw/kin_ion_io.py`** is genuinely both — a CLI (`:57`) and a library imported
  by `gw/sigma_dispatch.py:174`. See V3.

## A.2 Cross-package edge census

```
gw       -> common     75      common  -> ffi        22      gw       -> ffi      4
bse      -> common     33      file_io -> common     14      gw       -> isdf     4
gw       -> file_io    29      psp     -> file_io    13      common   -> runtime  5
psp      -> common     23      centroid-> common     12      psp      -> runtime  4
```

Downward edges are the bulk of the graph and are healthy. The upward and
sideways edges follow.

## A.3 Every violating edge, ranked by cost to a reader

| # | Edge | Sites | Why it costs a reader | Risk to fix |
|---|---|---|---|---|
| **1** | `file_io`, `psp`, `bse` → `gw.gw_config` | `file_io/slab_io.py:66, :94, :127`; `psp/get_DFT_mtxels.py:740`; `psp/get_dipole_mtxels.py:42`; `psp/run_sternheimer.py:1492`; `bse/bse_io.py:581, :625`; `bse/exciton_bands.py:395`; `bse/vq_interp.py:1296` | To read the I/O service you must read the GW driver's 94 KB config. Three sites are lazy imports whose comment admits the cycle. One is module-level: `import psp.get_dipole_mtxels` executes `gw.gw_config`. | **Low** for `SlabIOBackend` (enum move + re-export). Medium for `read_lorrax_input`. |
| **2** | `gw.sigma_dispatch` → `gw.kin_ion_io` (driver) | `gw/sigma_dispatch.py:174` | The single cause of the only top-of-module env block in any audited driver. A reader of the cleanest GW file must know why the messiest one is imported from it. | **Low–medium**: extract `compute_hartree_matrix` + `replicate_to_mesh` to `gw/hartree_matrix.py`; the CLI imports *that*. |
| **3** | physics kernels → runtime warm-up | `common/zeta_projection.py:422, :511, :547, :596, :864`; `common/contract_bands.py:542`; `bse/bse_ring_comm.py:64` | Invisible until missing. `contract_bands.py:539-541` at least documents it as "Policy 4"; `zeta_projection.py`'s five sites do not. | **Medium.** Load-bearing (job 7881053 controls, `common/collectives.py:661-668`). Needs the driver-level call to land first, plus a control proving equivalent clique coverage. |
| **4** | `ffi` → `file_io` (**private** name) | `ffi/phdf5/read.py:76`: `from file_io._slab_io_ffi import _sharding_to_axis_info` | Layer 1 importing a private function from layer 2. The comment (`:74-75`) explains the laziness, not the direction. The encoder *is* genuinely shared — this is misplacement, not a spurious edge. | **Low.** Move `_sharding_to_axis_info` to `ffi/common/` (both sides are FFI-shaped); `file_io` then imports downward. |
| **5** | `ffi` → `common` | `ffi/linalg/plan.py:97` | Same shape as #4, better justified — see §E.4, where I argue **not** to fix it. | n/a |
| **6** | `file_io` → `centroid` | `file_io/zeta_loader.py:512` | The ζ loader reaching into the centroid package. | Low–medium; `orbit_syms` looks like a `common/` symmetry utility. |
| **7** | `common` → `psp` / `gw` | `common/density_symmetry_check.py:650, :699`; `common/w_solve_modes_test.py:52, :86` | Both are test/diagnostic modules that happen to live in `common/`. | **Move them to `tests/`.** They inflate the count without costing a production reader anything. |
| **8** | `psp.orbital_magnetization` → `psp.run_sternheimer` (driver) | `psp/orbital_magnetization.py` | Same class as #2, colder path. | Low. |
| **9** | `bse` → `gw` physics | `bse/bse_io.py:922, :1468` → `gw.head_correction`; `bse/vq_interp.py:885, :887` → `gw.coulomb.base`, `gw.vcoul` | Genuinely shared physics (Coulomb head, `V_q` truncation) filed under `gw`. | Medium — wants a shared `coulomb/` package. **Not urgent**: a BSE reader importing the Coulomb head is not surprised. |

**Non-violations, stated so they are not "fixed" by mistake:** `common → runtime`
(`common/meta.py:6`, `common/wfn_transforms.py:49`, `common/fft_helpers.py:64`)
is 2→0 and correct; `isdf/core.py:43 → ffi.linalg` is 3→1 and correct;
`gw/kin_ion_io.py:95-108 → psp` is 4→3 and correct.

## A.4 Underlayer reach (`jax._src`)

Three modules only, AST-confirmed: `common/jax_compile_cache.py` (9 sites:
`:384, :399, :433, :607, :633, :679, :680, :681, :720`), `common/timing.py:93`,
`ffi/common/broadcast.py:58`. This matches
`wk_REL/docs/JAX_UNDERLAYER_NECESSITY_AUDIT.md`. **No driver and no physics kernel
reaches into `jax._src`** — that part of the goal is already met.

---

# B. Duplicated plumbing idioms, quantified

## B.1 Mesh construction — 21 production sites, 5 dialects, 0 migrated

The brief's "19 modules" is the number written in `common/collectives.py`'s own
docstring (`:15-16`: *"19 modules build their own; these two are what they should
call"*). My enumeration finds **21 production sites** plus 19 in
`common/*_test.py` / `*_bench.py`. One is not in the brief's list:
`common/eigh_block_sweep.py:54`.

**The replacement API landed today**: `common/collectives.py:232` (`resolve_mesh`)
and `:261` (`single_device_mesh`). **Migrated sites: 0 of 21** (checked 16:47 and
again at 16:53).

### The drift, ranked by consequence

**D1 (silent, wrong at P>1) — `jax.devices()[:1]` vs `jax.local_devices()[:1]`.**
`jax.devices()` is the GLOBAL list, so `[0]` is *process 0's device on every
rank*. The codebase has written this down twice
(`common/wfn_transforms.py:1436-1442`, `common/gpu_utils.py:38-46`) and the new
guard `common/collectives.py:225-229` names it as "the usual cause" of a bare
`StopIteration`. The correct replacement (`process_local_mesh`,
`wfn_transforms.py:1443-1447`) was **never propagated**:
`centroid/charge_density.py:108`, `:245`; `centroid/pivoted_cholesky.py:119`;
and `centroid/kmeans_cli.py:201`, `:216` (where `devices = jax.devices()` at
`:196`). `kmeans_cli`'s two are partially protected — `:213` refuses the fallback
when `multi_host` (`:198`), with a comment explaining the deadlock. The three
`centroid/` sites have no guard.
*(Read 16:53:32; md5 `charge_density` `46df831b…`, `pivoted_cholesky`
`96a354ef…`, `kmeans_cli` `25d041d9…`. `centroid/` is under active edit.)*

**D2 (silent, wastes the machine) — `--px/--py` default to 1.**
`bse/bse_feast.py:781`, `bse/bse_pseudopoles.py:212`, `bse/bse_w_exact.py:85` all
do `devices[:px*py]`, and argparse defaults both to 1 (`bse_feast.py:1167-1168`,
`bse_pseudopoles.py:557-558`, `bse_w_exact.py:560-561`). 16 devices, no flags →
a 1×1 mesh on global device 0, no warning.
The sharpest instance is **inside one driver**: `bse/bse_jax.py:131` uses
`create_mesh_2d()` — all devices, plus the clique warm-up — for the production
sharded path, while `:247-248` defines `--px/--py` with `default=1` and `:388-389`
feeds them to `ring_matvec_correctness_check`. **The correctness check runs on a
1×1 mesh by default while production runs full-mesh** — QUALITY_PATTERNS §2
inverted at the CLI surface.

**D3 (crash, not silence) — three notions of "how many devices".**
`gw/gw_jax.py:84`, `bandstructure/htransform.py:35`, `psp/get_DFT_mtxels.py:~800`
use `process_count() * local_device_count()`; `gw/kin_ion_io.py:239` uses
`jax.device_count()`; **`common/eigh_block_sweep.py:51` uses
`world = jax.process_count()`** and then reshapes `jax.devices()` (length
`device_count()`) into a `(p,q)` grid derived from `world` — a `ValueError` on any
build with >1 addressable device per process. `ffi/cusolvermp/profile_batched.py:121`
`SystemExit`s unless `px*py == jax.process_count()`. Sites agree *only* because
Frontera runs one device per process; `gw/kin_ion_io.py:233-238` documents that
assumption — in the one module that does not depend on it.

**D4 (silent, degenerate axis) — `reshape(1, -1)`.**
`psp/get_dipole_mtxels.py:547` builds a 1×N mesh (`mesh.shape['x'] == 1` always)
with the comment `# JAX mesh (simple 1D default; minimal sharding for demo)` at
`:545`. Its sibling `psp/get_DFT_mtxels.py:807` builds a most-square mesh. Both
then call `read_Gvecs_to_devices(..., mesh_xy)`.

**D5 (divergent failure policy).** Not enough devices → `bse_feast.py:778-779`
raises `ValueError`; `bse_ring_comm.py:1039-1043` raises `RuntimeError` with an
`XLA_FLAGS` hint; `centroid/kmeans_cli.py:212-216` prints and falls back to
single-device. Only `kmeans_cli` reasons about the multi-host consequence
(`:210-212`) — and `bse_feast`'s default-1×1 path walks straight into the hang
that comment describes.

**D6 (the divergence class, one layer up) — the warm-up.**
`bse/bse_ring_comm.py:64` calls `warm_mesh_cliques(mesh)`; its comment (`:53-63`)
records the measurement: *"Without it the TDA Lanczos dies on every rank at P=16
(32 refusals); with it, 0."* **The other 20 mesh constructions do not call it** —
including `bse/bse_feast.py:781`, which feeds the same BSE kernels. Under
`impl=mpi`, a mesh from `bse_feast` and a mesh from `bse_ring_comm` with
*identical device arrays* are not interchangeable.

**D7 (cosmetic — flagged so it is not mistaken for drift).** `['x','y']`
(`gw/gw_jax.py:88`, `psp/get_DFT_mtxels.py:807`, `psp/get_dipole_mtxels.py:547`)
vs `('x','y')` elsewhere. `Mesh` normalises `axis_names` to a tuple, so
`mesh.axis_names` compares equal downstream (e.g.
`common/zeta_projection.py:173`). All 21 sites use the names `x`/`y`. **Not
behavioural.**

**What `resolve_mesh` already fixes:** D1 (via `_require_addressable`), D3 (one
definition of device count), D5 (one refusal). **What it does not yet fix:** D2
(no `px/py` override) and D6 (it does not warm). See §D.2 S2.

## B.2 `shard_map` — 50 files, 4 spellings, 29/21 split on `check_rep`

**Count.** 50 files under `src/` mention it (enumerated, not counted by grep);
**50 application sites**. The brief's "48 modules" is close — the difference is
import-only files and the two shim files.

| spelling | files |
|---|---|
| `from jax.experimental.shard_map import shard_map` | 38 in `src/` (27 module-level, 11 lazy) |
| `from jax import shard_map as _shard_map_fn` (try-branch) | `bse/bse_ring_comm.py:14`, `bse/bse_stack_matvec.py:64` |
| `from jax.experimental import shard_map as _shard_map_mod` (except-branch of the same shim) | same 2 files, `:16-17` / `:66-67` |
| bare `jax.shard_map`, no import | `tests/archive/test_blocked_cholesky.py:412`, `:430` |

*(Correction to the brief: **two** shim files, not three. The third candidate is
`common/collectives.py`, whose `try/ImportError` blocks are unrelated to
`shard_map`.)*

**The shim points the opposite way from the fleet.** `jax.shard_map` is the newer
canonical location; `jax.experimental.shard_map` is legacy. `pyproject.toml:9`
pins `jax[cuda12]>=0.9.0` and `uv.lock:1014-1015` resolves 0.9.1, so the shim's
`try` branch **always wins** — those two BSE files bind a *different callable*
than the other 38. Not a behaviour difference today, but it means the effective
`check_rep` default for the BSE ring kernels is decided by a `try/except
ImportError` two files away, not by anything at the call site.

**`check_rep`: 29 sites pass `check_rep=False`, 21 omit it. Every occurrence in
`src/` is `=False`; there are no other values. There is no comment anywhere in
`src/` explaining why.** The 21 that omit: `common/fft_helpers.py:253`, `:271`;
`gw/w_isdf.py:325`; all 11 `_shard_map_fn` sites in `bse/bse_ring_comm.py`
(`:320, :334, :344, :532, :546, :562, :576, :586, :606, :861, :989`);
`bse/bse_stack_matvec.py:137`.

**Lazy imports: 11 sites, zero stated reasons.** `common/zeta_projection.py:362,
:468, :523, :565, :639`; `common/collectives.py:528` *(pre-16:52 line number)*;
`common/contract_bands.py:265`; `gw/w_isdf.py:233, :421`; `gw/qsgw_utils.py:194,
:246`. Two have a visible structural motive (the collectives one sits behind an
early `return` off `impl=mpi`; the `qsgw_utils` pair sit inside kernel-cache
misses). `zeta_projection.py`'s five are in a module that already imports jax at
top level, so the laziness buys nothing.

**Near-duplicate boilerplate — 5 clusters.** `bse/bse_ring_comm.py:320` and `:334`
are character-for-character identical except the wrapped function; `:532`/`:546`
are a second verbatim copy in a different builder; `:344`, `:586`, `:606` are
three copies of the 5-spec `apply_V_ring` wrapper with identical specs — 11 sites
collapsing to ~4 shapes. `ffi/slate/trsm.py:153` (`_untranspose`) is a
byte-identical twin of `ffi/slate/cholesky.py:68` (`_local_T`). The
row-major↔col-major transpose-wrap recurs across **four** FFI backends
(`ffi/slate/cholesky.py:109`; `ffi/cusolvermp/batched.py:161`, `:301`;
`ffi/cusolvermp/eigh.py:136`; `ffi/scalapack/eigh.py:119`, `:130`).
`centroid/kmeans_isdf.py:617` and `:681` differ only in one argument name.

**The one API — narrow, not generic** (see T3 for why generic is a trap):
1. **One `check_rep` policy**, stated once — either passed from one helper, or
   the one comment explaining why 21 sites do not need it.
2. **One `transpose_wrap(kernel, mesh, spec_in, spec_out)`** in `ffi/common/`.
   `ffi/linalg/plan.py:100-111` has already made exactly this argument for the
   *result* normalisation ("it used to live inside `dispatch_eigh` only, so
   anything calling `backend_module` … had to remember it independently") but not
   for the *operand* transpose.
3. **Migrate the named clusters onto `contract_bands`** —
   `common/contract_bands.py:26-28` already lists `bse.bse_stack_matvec._w_stack`,
   the `bse_ring_comm` matvec family and `vq_interp` V_Q assembly as slated
   adopters. That is the real answer to the boilerplate.

## B.3 `process_allgather` / barrier — the API is declared but not written

`common/collectives.py`'s new `__all__` (`:68-98`) promises 21 names. An AST
parse at 16:53 (md5 `861d44ae64171a3d5a55f6f882522b22`, 724 lines) finds **9
undefined**:

```
replicate_to_mesh, psum_replicate, all_gather_processes, gather_indexed_blocks,
local_share, sweep_local_k, gather_k_blocks, sweep_lookahead, SWEEP_LOOKAHEAD_ENV
```

Defined: `COLLECTIVE_RTOL, barrier, device_count, device_put_process_local,
process_count, process_rank, process_rank_world, psum_scatter_checked,
report_collective_residual, resolve_mesh, single_device_mesh, warm_mesh_cliques`.

**This is in-flight work, not a defect to file** — but while it holds,
`from common.collectives import *` raises. It is also the most useful artifact in
this audit: the owner's declared target API, written down.

### The five hand-rolled host-gather wrappers (U1)

| wrapper | site | fast path | `tiled` |
|---|---|---|---|
| `_to_host` | `file_io/_slab_io_allgather.py:66-73` | `is_fully_replicated` | `True` |
| `_to_host` | `solvers/davidson.py:54-63` | `process_count()==1` | `False` |
| `_to_host` | `bse/bse_davidson_helpers.py:48-58` | `process_count()==1` | `False` |
| `_to_numpy` | `bse/exciton_bands.py:125-128` | `is_fully_addressable` | `True` |
| `_to_numpy` | `bse/vq_interp.py:557-560` | `is_fully_addressable` | `True` |
| (variant) | `gw/ppm_windows.py:193-200` | mutates `tiled` on `is_fully_addressable`, then `try/except` | both |

`davidson.py:50` and `bse_davidson_helpers.py:44` both say in their docstrings
that they mirror `file_io._slab_io_allgather._to_host`. **The duplication is
documented inside the copies.** `exciton_bands.py:117-119` documents that the
wrong branch silently multiplies the leading axis by P — so the `tiled` /
predicate disagreement is a correctness surface, not a style one.

Direct (non-wrapper) sites: `gw/qsgw_utils.py:289-291`; `gw/kin_ion_io.py:381-387`
(two `tiled=False` calls plus a hand-rolled index-carrying scatter);
`gw/minimax_screening.py:55` (used as a cross-rank equality probe);
`centroid/pivoted_cholesky.py:457`; `bse/davidson_absorption.py:55`;
**`psp/run_nscf.py:255-264` — three `process_allgather(tiled=False)` then
`.reshape(n_proc,…).sum(axis=0)`, i.e. an all-gather emulating an all-reduce**,
exactly the P-linear cost class `common/collectives.py:341-360` measured at
17.4 GB/rank.

### The barrier — the service exists and 5 production files bypass it

Correct users (7 sites): `file_io/tagged_arrays.py:16, :227, :260, :285, :299`;
`gw/gw_init.py:486, :644, :847, :1110`; `gw/sc_iteration.py:943`;
`gw/v_q_bispinor.py:413`; `common/wfn_transforms.py:2143`.

Bypassing, in severity order:
1. **`file_io/_slab_io_ffi.py:48-55`** — `try: sync_global_devices(tag) / except
   Exception: pass`, the verbatim anti-pattern `common/collectives.py:37-47`
   quotes, **in a file that imports `device_put_process_local` from that same
   service at `:41`**. Called at `:913`.
2. **`gw/gw_output.py:294-298`** — same swallow-everything shape.
3. `gw/isdf_fitting.py:695-696`, `:1091`; `gw/kin_ion_io.py:1066-1068` (hand-rolls
   the `process_count()<=1` guard); `psp/get_dipole_mtxels.py:748, :773`;
   `psp/run_nscf.py:265, :278`.

**`assert_equal`, `host_local_array_to_global_array` and
`global_array_to_host_local_array` have zero live call sites** — `assert_equal`
appears only in 9 comments, all documenting the *hidden* one inside
`jax.device_put`. That absence is AST/grep-confirmed across `src/`; I note it
because it means the anti-collective story is complete on that axis.

**Related, and the same "service written, never wired" pattern:**
`common/collectives.py` records that `psum_scatter_checked` has **zero call
sites** for the fifteen reduce-scatters it was written for. And
`jax.lax.all_gather` appears at 12 in-jit sites (`isdf/core.py:773, :1942, :2072,
:2073, :2806, :2807`; `gw/w_isdf.py:466, :467`; `bse/bse_ring_comm.py:328, :330,
:540, :542, :570, :572`; `bse/bse_stack_matvec.py:110, :112`) with no shared
helper — a different layer, but the same sprawl.

## B.4 Rank / process-count resolution — four resolvers, two `announce_once`

| resolver | site | chain |
|---|---|---|
| `runtime._resolve_proc_count` | `runtime/__init__.py:494` | `JAX_PROCESS_COUNT` → `JAX_NUM_PROCESSES` → `SLURM_NTASKS` → 1 |
| `runtime._resolve_proc_id` | `runtime/__init__.py:503` | `JAX_PROCESS_INDEX` → `SLURM_PROCID` → 0 |
| `ffi.common.gate.rank_id` | `ffi/common/gate.py:117-138` | `SLURM_PROCID` → `PMI_RANK` → `OMPI_COMM_WORLD_RANK` → `jax.process_index()` → `None` |
| inline copy | `gw/kin_ion_io.py:800-803` | byte-identical to `_resolve_proc_count`, re-typed |
| `common.collectives.process_count` | `common/collectives.py:100-110` | asks jax; returns 1 on any exception |

1. **`_resolve_proc_id` does not know `PMI_RANK` or `OMPI_COMM_WORLD_RANK`.**
   Under an OpenMPI/PMI launcher without SLURM env it returns **0 on every rank**,
   so the failfast banner (`runtime/__init__.py:151-152`) prints `rank 0/N` from
   every rank — destroying the attribution it exists for.
2. **`gate.rank_id` is hardened; `_resolve_proc_id` is not.** `rank_id` checks
   `raw.strip() != ""` and catches `ValueError` (`ffi/common/gate.py:129-133`).
   `_resolve_proc_id` does a bare `int(os.environ.get(...))`, so
   `JAX_PROCESS_INDEX=""` raises **inside `install_failfast_excepthook`** — inside
   the thing installed to make failures loud.
3. `gate.rank_id` deliberately does **not** call `jax.process_index()` first,
   and the reason is recorded (`ffi/common/gate.py:56-61`). Any unification must
   preserve it — see T2.

**Downstream: two functions named `announce_once`, different rank policies.**

| | `ffi/common/gate.py:147` | `runtime/aot_memory.py:51` |
|---|---|---|
| rank source | `rank_id()` (4-var, hardened) | `runtime._resolve_proc_id` (2-var) |
| who prints | rank 0 only, or the local rank tagged | **every rank, always** (`:60-64`) |
| validates scope / empty msg | yes (`:158-167`) | no |
| consumers | `ffi/common/gate.py` ×6 | `common/fft_helpers.py:117, :133`; `gw/gflat_memory_model.py:83, :167` |

At P=64 a memory-model demotion prints **64 lines**; a gate demotion prints one.
Under a non-SLURM launcher all 64 tag themselves `[proc 0]`.

## B.5 Env vars — 71 names, 5 boolean grammars, and one rc=0 trap

**Volume.** 244 raw env-read matches across `src/`. Production hot spots:
`runtime/__init__.py` (30), `gw/isdf_fitting.py` (9), `gw/kin_ion_io.py` (8),
`gw/gw_init.py` (7), `psp/get_DFT_mtxels.py` (6), `isdf/core.py` (6),
`common/jax_compile_cache.py` (6), `gw/gw_config.py` (5),
`file_io/_slab_io_ffi.py` (5). `common/`'s apparent 125 is misleading — 108 are
the JAX-preamble block copy-pasted into 18 standalone `*_test.py`/`*_bench.py`
drivers that live inside the library directory. **That preamble is the single
largest duplication in the sweep**, and it is fixed by moving those files to
`tests/` (A.3 #7), not by a new abstraction.

**67 distinct names read as string literals, plus 4 read through a name variable**
(`SLURM_JOB_NUM_NODES`, `SLURM_NNODES` at `gw/gw_config.py:280`; `PMI_RANK`,
`OMPI_COMM_WORLD_RANK` at `ffi/common/gate.py:117`) = **71**.

Read in ≥3 files: `JAX_ENABLE_X64` (31), `JAX_PLATFORMS` (23), `SLURM_NTASKS`
(21), `CUDA_VISIBLE_DEVICES` (13), `LORRAX_FORCE_FULL_BZ` (4 files / 5 sites),
`XLA_PYTHON_CLIENT_PREALLOCATE` (3), `XLA_PYTHON_CLIENT_ALLOCATOR` (3),
`JAX_PLATFORM_NAME` (3).

### Env reads inside physics kernels — the layering finding

24 sites. The ones that cost a reader most:
`gw/isdf_fitting.py:687-688` reads `LORRAX_PHDF5_STRIPE_COUNT` /
`_STRIPE_SIZE_FS` — **a ζ-fit kernel doing Lustre striping config**, duplicating
`file_io/_slab_io_ffi.py:68` and `file_io/_slab_io_mpi_host.py:119`.
`gw/isdf_fitting.py:112` reads `CUDA_VISIBLE_DEVICES`.
`gw/screening.py:217` and `gw/v_q_g_flat.py:175` both re-derive
`LORRAX_FORCE_FULL_BZ`, a routing decision the driver already made at
`gw/gw_init.py:352, :610, :719`.
`bandstructure/htransform.py:62` (`LORRAX_GALERKIN_CHUNK_GIB`) and
`solvers/sternheimer_solve.py:44` (`STERN_DEBUG`) read env at **module import
time** — the latter as `bool(int(...))`, so a word spelling crashes on import.

**The good shape exists and should be the model:** `common/contract_bands.py`
has **no** direct `os.environ` call. It reads its dial one hop away, as a typed
capability object — `contract_bands.py:178, :187, :198, :322` use
`ffi.mklblas.gemm.GATE`, declared at `ffi/mklblas/gemm.py:48-49`. Same for
`ffi/mklfft/flat_k.py:81-82` and `:128-129`. **A kernel consuming a `Gate` rather
than a string is the target state for every one of the 24 sites above.**

### The five boolean grammars

| | grammar | canonical site |
|---|---|---|
| **A** | `.strip().lower() in ("","0","false","no","off")` | `runtime/__init__.py:53, :56-58` |
| **B** | unset/`""` → default, else `.strip().lower() in ("1","true","yes","on")` | `isdf/core.py:1389, :1392-1409` |
| **C** | `.strip().lower() not in ("","0","false","no","off")` — the only one matching A | `common/jax_compile_cache.py:256-258` |
| **D** | bare presence (`if os.environ.get(X)`) — so `X=0` turns it **ON** | many |
| **E** | `bool(int(...))` — `ValueError` on any word spelling | `gw/screening.py:217` etc. |

Grammars A and B both claim in their docstrings to be *the* consolidation
(`runtime/__init__.py:46-52`, `isdf/core.py:1392-1404`) and they **disagree on
an unrecognised value**: under A, `X=maybe` is truthy; under B it is falsy. Only
`ffi/common/gate.py:250-262` handles an unrecognised value correctly — it
announces and falls back.

**Concrete drift pairs, all verified by reading both sides:**

1. **`LORRAX_RESTART_WRITE_LOG` vs `LORRAX_W_RESIDUAL_CHECK` — the audit that
   fixed one missed the other.** `gw/w_isdf.py:531-536` carries a comment saying
   the vocabulary was widened for exactly this reason ("audit fix/zq
   2026-07-28"), and parses `not in ("", "0", "false", "no", "off")` at `:537-538`.
   `file_io/tagged_arrays.py:47-48` parses `not in ("0", "", "false")` — **no
   `.strip()`, no `.lower()`, no `no`/`off`.** So `=off`, `=no`, `=False`,
   `=FALSE`, `= 0 ` all leave the log **ON**, and `docs/dev/env_vars.md:168`
   advertises `=0` — the only spelling that works.
2. **`LORRAX_MALLOC_TRIM` vs `LORRAX_MALLOC_TUNE` — sibling knobs, opposite
   rules.** `gw/isdf_fitting.py:907`: `not in ("0","off","false")`, no strip/lower,
   **and `""` is not falsy so `=""` enables it**. `runtime/__init__.py:395`:
   grammar A. `env_vars.md:116-117` presents them as one family.
3. **`LORRAX_SKIP_VQ_GATES` is exact-match only.** `bse/vq_interp.py:1103`:
   `== "1"`. So `=true`/`=on`/`=yes`/`= 1 ` are silently ignored and the V_Q gate
   battery runs anyway — **the 58 GB alloc the knob exists to avoid**
   (`vq_interp.py:1099-1102`). Its two nearest siblings
   (`gw/ppm_sigma.py:275`, `gw/ppm_tau_kernel.py:66`) use grammar B.
4. **`LORRAX_EXIT_AFTER_ZETA=0` ends the job.** `gw/gw_init.py:1007` is a bare
   presence test; `:1011` is `raise SystemExit(0)`. The value that reads as
   "disabled" truncates the run after ζ-fit **with rc=0**. Same grammar-D class:
   `LORRAX_MEM_DEBUG` (`gw/isdf_fitting.py:58, :895, :987`; `gw/gw_init.py:1028`),
   `LORRAX_RCHUNK_DEBUG` (`gw/isdf_fitting.py:924`, `isdf/core.py:3227`).
   `env_vars.md:272-273` blesses these as "all presence-test ✓" — consistent with
   each other and with neither A nor B.
5. **Numeric parsing has four policies too.** Malformed input →
   `common/density_symmetry_check.py:399-418` warns and falls back;
   `common/jax_compile_cache.py:261-265` falls back silently;
   `common/timing.py:62-71` raises with a named message;
   `gw/gw_config.py:1533-1537` catches bare `Exception` and uses 0.0;
   `file_io/_slab_io_mpi_host.py:125-131` raises.

### `docs/dev/env_vars.md` is a real registry with a layer-shaped blind spot

312 lines, three classified sections, per-var defaults and effects, and a
consistency-audit table at `:262-280`. **61 of the 67 literal-read vars are
documented.** The undocumented LORRAX-owned ones are
`LORRAX_PPM_HERM_DIAG` (`gw/ppm_sigma.py:275`), `LORRAX_SIGMA_TAU_TIMING`
(`gw/ppm_tau_kernel.py:66`), `LORRAX_GRAM_COL_BLOCK`
(`centroid/pivoted_cholesky.py:988`), `LORRAX_KIN_ION_LOOKAHEAD`
(`gw/kin_ion_io.py:404`) — plus `JAX_PLATFORM_NAME`, `XLA_FLAGS`, `SCRATCH`,
`SLURMD_NODENAME`, `HOSTNAME`, and the four indirect ones.
**All four undocumented LORRAX vars are in physics-kernel modules** — the
registry's blind spot is precisely the layer this audit is about. Cause: its own
re-run recipe (`env_vars.md:285`, `python3 tools/env_audit.py src`) is an AST
walk that cannot see `gate.py:117`'s tuple-driven reads or `gw_config.py:280`'s
loop variable. And the consistency table compares **defaults, not grammars** —
which is why `LORRAX_FORCE_FULL_BZ` passes it ("5 sites, all `'0'` ✓",
`env_vars.md:269`) while `=true` raises `ValueError` mid-run in three files.

## B.6 The C/C++ / FFI layer

### B.6.0 The ghost header — and why it matters methodologically

**`mkl_thread_pin.h` does not exist.** Verified by `find` over the tree and by
`git ls-files` filtered to `.h/.hpp/.cuh` — the repo has exactly ten headers, and
that is not one of them. The one occurrence of the string is
`docs/dev/vendor_gemm_service.md:183`, inside a passage that **proposes creating
it**:

> The pin is currently the third copy of the same ~60-line block
> (`scalapack/cpp/blacs_grid.h:157-233`, `mklfft/cpp/fft_flat_k_ffi.cc:96-159`,
> here at `:277-338`) … The copies have already DRIFTED … **the fix is a new
> MPI-free `ffi/common/cpp/mkl_thread_pin.h`** … Not done in this wave (C++ was
> out of scope); recorded so it is not rediscovered.

The doc is accurate, honest, and did exactly what it set out to do. What went
wrong is the reading: a filename appearing in prose was taken as a filename
existing on disk. **This is standing lesson #2 in reverse** — the lesson warns
that a grep *negative* is not evidence of absence; this is a grep *positive*
mistaken for evidence of presence. Both failures have the same root: the grep hit
was not opened.

### B.6.1 The pin divergence is real, and currently masked

Three copies, confirmed by reading each and by `nm -C` on the built `.so`
(`lorrax_ffi_unified/build_host_C64/liblorrax_ffi_host.so`), which shows **three
distinct symbols at three addresses**:

```
lorrax_ffi::scalapack::mkl_set_num_threads_local_ptr()::fn
lorrax_ffi::mklfft::mkl_set_num_threads_local_ptr()::fn
lorrax_ffi::mklblas::mkl_set_num_threads_local_ptr()::fn
```

| copy | site | resolution | symbols |
|---|---|---|---|
| A | `ffi/scalapack/cpp/blacs_grid.h:155-158` | bare `dlsym(RTLD_DEFAULT)` | `MKL_Set_Num_Threads_Local` **and** `MKL_Get_Max_Threads` (`:161-164`) |
| B | `ffi/mklfft/cpp/fft_flat_k_ffi.cc:98-101` | bare `dlsym(RTLD_DEFAULT)` | set only |
| C | `ffi/mklblas/cpp/gemm_batch_ffi.cc:139-144` | `dlsym(RTLD_DEFAULT)` **then `#ifdef RTLD_NEXT` → `dlsym(RTLD_NEXT)`** | set + the 4 `cblas_?gemm_batch` entries |

The RAII wrappers are three copies under **different class names** —
`MklThreadScope` (`blacs_grid.h:218-240`), `MklLocalPin`
(`fft_flat_k_ffi.cc:104-122`), `MklLocalPin` (`gemm_batch_ffi.cc:285-303`) —
which is why a same-basename search misses them. `str_ieq` is a fourth
triplicated helper (`blacs_grid.h:168`, `fft_flat_k_ffi.cc:124`,
`gemm_batch_ffi.cc:305`).

**Correction to the brief's severity:** the failure is **latent, not live.**
`ffi/common/ffi_loader.py:514` is the only dlopen of these libraries and it is
`ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)`; `readelf -d` shows
`libmkl_intel_lp64 / libmkl_gnu_thread / libmkl_core` in `DT_NEEDED`, so under
`RTLD_GLOBAL` all three copies find the symbol. `nm -C -D --undefined-only`
confirms `MKL_Set_Num_Threads_Local` is not link-referenced — i.e. this is
established by symbol table, not by `strings`. **What is at stake if the load
scope ever changes:** copies A and B have no second chance, and A's silent
degradation means inheriting `MKL_NUM_THREADS=28`, the configuration
`blacs_grid.h:110-118` documents as **measured 24× slower `pzheevd`** at the
12×12 production grid. The no-op-on-null design (correct for non-MKL BLAS) makes
the safety path and the failure path indistinguishable.

**Second-order divergence:** A resolves `MKL_Get_Max_Threads`; B and C do not,
and fall back to `omp_get_max_threads()`. So "auto" means something different in
the three copies. The structural cause is a **build** fact: `-fopenmp` is applied
**per-source-file**, not per-target — `set_source_files_properties` names
`mklfft/cpp/fft_flat_k_ffi.cc` and `mklblas/cpp/gemm_batch_ffi.cc` only — so the
scalapack TU has no OpenMP and *cannot* call `omp_get_max_threads`. That is the
reason the three copies exist, and it is written down nowhere.

### B.6.2 The MPI thread-level divergence — the live one

Covered in §0. `slate/cpp/context.cc:30-31` captures `provided` and discards it,
with no `MPI_Query_thread` anywhere in `slate/cpp/`;
`phdf5/cpp/context.cc:200-217` has the full guard with the measured crash rate
and the named fix. **This is the equivalent-class finding the brief asked for,
and unlike the pin it is not masked by anything.**

### B.6.3 The other C++ duplications

* **`cross_stream_wait_pooled` — 6 byte-identical `static` copies**
  (`cusolvermp/cpp/eigh_ffi.cc:58`, `batched_potrf_ffi.cc:51`,
  `batched_potrs_ffi.cc:49`, `batched_solve_lu_ffi.cc:69`;
  `cublasmp/cpp/batched_gemm_ffi.cc:54`, `batched_w_solve_ffi.cc:79`). Identical
  today — and exactly the shape that produced the pin divergence.
* **Error macros — three families.** `common/cpp/ffi_helpers.h` owns
  `FFI_RETURN_IF_ERROR` (`:41`), `LORRAX_CUDA_CHECK` (`:50`), `LORRAX_LIB_CHECK`
  (`:66`). `cublasmp` defines `LORRAX_CUBLASMP_CHECK` twice, byte-identically
  (`batched_gemm_ffi.cc:44`, `batched_w_solve_ffi.cc:59`), where
  `LORRAX_LIB_CHECK(expr, CUBLASMP_STATUS_SUCCESS, "cublasMp")` covers both.
  **`cufft/cpp/fft_flat_k_cuda_ffi.cc:152` defines its own `LRX_CUDA_CHECK` that
  drops `__FILE__:__LINE__` and `cudaGetErrorName`** relative to
  `ffi_helpers.h:50` — a divergence in diagnosability, not correctness.
* **`log_enabled()` ×4** (`cufft:112`, `mklfft:190`, `mklblas:340`, inline at
  `scalapack/cpp/eigh_ffi.cc:227`) — all presence-only, consistent, benign.
* **Cleared as benign name collisions** (full `diff`): `ctx.h` ×3, `api.cc` ×2,
  `context.cc` ×3, `eigh_ffi.cc` ×3, `*_interface.h` ×3. Disjoint content.

### B.6.4 Two-sided (C++ **and** Python) env duplication

**`LORRAX_PHDF5_STRIPE_SIZE_FS` has opposite failure modes on the two sides of
the same run.** C++ `phdf5/cpp/context.cc:334-347`: an unrecognised suffix sets
`mult = 0` and the comment says `// unparseable -> keep default` — **silently
4 MiB**. Python `file_io/_slab_io_mpi_host.py:119-133`: **raises `ValueError`**
naming the grammar, and its docstring explicitly says the silent-default
behaviour *was an audit finding* ("an explicit `=4MiB` A/B experiment quietly
measured the default configuration"). So `=4MiB` refuses on the host writer and
silently measures the default on the FFI writer. **The Python side cites this
defect class and fixed only its own copy.** Same split for
`LORRAX_PHDF5_STRIPE_COUNT` (C++ `context.cc:331` raw pass-through; Python
`_slab_io_ffi.py:68-77` refuses with a named grammar).

**The one that was done right, and proves the pattern is achievable:**
`phdf5/cpp/context.cc:95-105` documents that its `env_flag` grammar is
*deliberately mirrored* from `file_io/_slab_io_mpi_host._env_flag`, and the two
do match.

### B.6.5 Build flags

* **`-fvisibility=hidden`, `-Wl,--exclude-libs`, `-Bsymbolic` and version scripts
  are set in neither build.** The host lib exports **389 dynamic symbols**
  (`nm -D --defined-only`), including every internal `xla::ffi::Handler<...>` /
  `Binding<...>::To<...>` template instantiation as a weak global. Both libs are
  dlopen'd `RTLD_GLOBAL` and both export the same weak instantiations, so
  whichever loads first wins the binding for both. That is survivable only
  because they are compiled against the same jaxlib headers — which
  `host/CMakeLists.txt:57-63` warns about but nothing enforces. **This is a
  header-only-template hazard: `strings` will never show it; `nm -C` does.**
* `-Wl,--no-as-needed` is set on the host MKL branch only (mandatory there — ld10+
  drops transitive MKL from `DT_NEEDED`); the CUDA lib has no equivalent.
* `${CMAKE_DL_LIBS}` is unconditional on host, but **conditional on
  `if(LORRAX_CUFFT_ON)`** in the CUDA build — a latent link-order dependency on
  glibc ≥ 2.34 (where libdl folded into libc), not a stated invariant.
* **Whether the FFT and GEMM handlers exist at all is downstream of how
  ScaLAPACK resolved**: the mklfft/mklblas sources are appended only inside the
  `if(slate_FOUND)` branch. Confirmed empirically — of ten host build trees on
  disk, five contain `MklFft*`/`MklBlas*` handlers and five (including
  `build_host`, the default path in `build_ffi_host.sh`) contain **zero**.

### B.6.6 The gate layer — no overlap, and one genuine gap

`ffi/common/gate.py` and `ffi/linalg/resolve.py` **overlap only on
`ffi_loader.probe_target`**. `resolve.py` reads **no environment at all** — its
inputs are function arguments — so there is no shared-variable drift to find. The
split is documented and justified at `gate.py:35-49`, and
`ffi/linalg/dispatch.py:22` states the boundary in one line. This confirms the
prior assessment's refusal to fold them; see §E.1.

One soft overlap: `mesh_ffi_platform` (`gate.py:186`) and `mesh_platform`
(`resolve.py:141`) implement the same mapping and diverge on a third platform
(gate returns the raw string deliberately, `:189-191`, so a refusal can say
`'tpu'`). Cosmetic today.

**Backends using `Gate`: 2 of 8** — mklfft (`ffi/mklfft/flat_k.py:81`, `:128`)
and mklblas (`ffi/mklblas/gemm.py:48`); **cufft is gated transitively** through
mklfft's `GATE`, which declares `platforms=("cpu","CUDA")` (`flat_k.py:84`).
cusolvermp, slate and scalapack use `resolve.resolve_backend`'s six-guard ladder
(`resolve.py:126-134`). **phdf5 rolls its own** — `probe_target` at three
consumer sites in two packages (`file_io/wfn_loader.py:388-390`;
`gw/gw_config.py:221`, `:296`) rather than one `Gate`, so the three can drift
independently.

**`cublasmp` has no gate at all** — bare `get_lib()` straight to `ffi_call`
(`ffi/cublasmp/batched.py:148`, `:281`), no `_SPEC` row in `resolve.py`. Meanwhile
`build_ffi.sh` sets `-DLORRAX_FFI_HAVE_CUBLASMP=OFF` by default on Frontera, and
`nm -D` on the shipped CUDA `.so` confirms `CublasMpBatchedGemmFfi` /
`CublasMpBatchedWSolveFfi` are **absent**. So a cublasmp call on the production
build fails inside XLA at execution with a target-not-found error rather than
refusing at resolve time with the reason and the fix — the exact doctrine
`resolve.py:16-21` and `gate.py:13-16` exist to enforce. **Mitigating: the
package has no in-tree consumer** (only two `common/*_test.py` drivers), so this
is a latent gap, not a live one.

### B.6.7 One stale claim, worth fixing so it does not breed another ghost

`ffi/mklfft/flat_k.py:33-46` states *"ADOPTION STATE (2026-07-30):
`common/fft_helpers.py` still carries its own copy of the gate and the two
bodies … treat `fft_helpers` as the live path and this as its verbatim
relocation."* That is **no longer true**: `common/fft_helpers.py:297-301` now
imports `fft_ffi_enabled`, `make_gw_conv_ffi`, `make_flat_k_fft_ffi` from
`ffi.mklfft` and constructs no `Gate`. The duplication was resolved; the comment
claiming it exists was not updated — and it will send the next auditor hunting a
duplicate that isn't there, which is precisely how the `mkl_thread_pin.h` ghost
was born.

---

# C. The driver acceptance test

**All five files read 2026-07-30 16:47:07–16:48:40 CDT.** md5 and mtime recorded
so a later reader can tell whether the live agents have moved underneath this.
**None of the five had changed at the time of reading** — `common/collectives.py`
did change during the audit; these did not.

| file | mtime | md5 | lines | raw jax imports | top-of-module env reads |
|---|---|---|---|---|---|
| `gw/gw_jax.py` | 07-28 15:27:52 | `ce11099cecc660114d1cc45996ef3986` | 651 | **3** | **0** |
| `bse/bse_jax.py` | 07-28 02:44:04 | `e042fc7cfe3e89548de179ee87686ac0` | 528 | **2** | **0** |
| `gw/kin_ion_io.py` | 07-30 14:46:49 | `eac5e643995cba5b7edcd0fa23e73b61` | 1088 | **6** (4 module + 2 lazy) | **4** `setdefault` @ `:61-64` |
| `centroid/kmeans_isdf.py` | 07-27 15:36:23 | `74a469fbcf584507727bf134bf25c5dc` | 1004 | **5** | **0** |
| `centroid/kmeans_cli.py` | 07-25 23:08:24 | `25d041d932e9b7b6a9e53b5c744d4595` | 513 | **3** | **0** |

These reproduce the brief's measured starting points exactly.

Plumbing density (lines mentioning any of `jax.|jnp.|Mesh|NamedSharding|
PartitionSpec|shard_map|sharding|device_put|block_until_ready|psum|os.environ|
mpi4py|process_index|process_count|local_device`):

```
gw/gw_jax.py             25/651   3.8%
bse/bse_jax.py           27/528   5.1%
centroid/kmeans_cli.py   21/513   4.1%
gw/kin_ion_io.py         84/1088  7.7%
centroid/kmeans_isdf.py 202/1004 20.1%   <- a kernel, not a driver
```

## C.1 `gw/gw_jax.py` — closest to the goal

```python
33  from runtime import bootstrap
34  bootstrap()  # env + jax.distributed + CPU fallback — BEFORE `import jax`.
35
36  import argparse
37  import gc
38  import os
39
40  import numpy as np
41  import jax
42  import jax.numpy as jnp
43  from jax.sharding import Mesh
44  jax.config.update("jax_enable_x64", True)
45
46  from file_io import (
47      WFNReader,
48      load_kin_ion_submatrix, load_centroids,
49  )
50  from common import Meta, RYD_TO_EV, symmetry_maps
51  from common.wfn_transforms import get_enk_bandrange
52  import common.timing as timing
53  from .gw_config import ComputeMode, LorraxConfig, QPSolver
54  from .gw_init import prepare_isdf_and_wavefunctions
...  (12 more `.`-relative physics-stage imports, :55-79)
```

**Env reads: none.** `os` is used only for `os.path.join` / `os.makedirs`
(`:212-214`).

**What a physicist must already know about jax:**
1. That `bootstrap()` must precede `import jax` and why — `:34` says "BEFORE
   `import jax`" but not that jax reads its env at import time (that is in
   `runtime/__init__.py:170`).
2. What `jax.config.update("jax_enable_x64", True)` is and that omitting it
   silently halves precision (`:44`).
3. What a **`Mesh`** is, what "most-square factorization" means, and that
   `jax.devices()` is the global list — all needed to read `_build_mesh`
   (`:82-88`).
4. That `jax.process_index() == 0` is the rank-0 idiom (`:182`).
5. `jax.default_backend()`, `jax.devices()`, `jax.process_count()`,
   `mesh_xy.devices.shape` for the banner (`:200-206`).

**Verdict: 3 imports and 0 env reads is already good.** The residue is
`_build_mesh` (`:82-88`, 7 lines → `collectives.resolve_mesh()`) and
`_setup_runtime` (`:91-140`, which is *the* driver-level runtime service in
disguise — §D.2). Remove those two and `from jax.sharding import Mesh` goes with
them, leaving `import jax` + `import jax.numpy as jnp`, and `jnp` is physics.
**Distance to target: ~15 lines.**

## C.2 `bse/bse_jax.py` — the existence proof

```python
 1  """BSE JAX entry points and CLI wrappers."""
 2  from __future__ import annotations
 3
 4  import os
 5  import sys
 6
 7  # Ensure x64 + jax.distributed bootstrap before any jax-collective code
 8  # (the ring matvec uses lax.psum/ppermute on the 2D mesh, which is silent-
 9  # wrong if processes don't agree on a shared distributed runtime).
10  # Single-sourced in runtime.bootstrap() (env defaults + jax.distributed
11  # init + CPU fallback); MUST run before this module's own `import jax`.
12  from runtime import bootstrap
13  bootstrap()
14
15  import jax
16  import jax.numpy as jnp
17
18  from .bse_ring_comm import (
19      build_bse_ring_matvec, build_bse_ring_matvec_full, create_mesh_2d,
20      make_bse_shardings, ring_matvec_correctness_check, ring_matvec_smoke_test,
21  )
...
39  jax.config.update("jax_enable_x64", True)
```

**Env reads: none.**

**`bse_jax.py` is the existence proof that the target shape works.** It never
says `Mesh`, never says `sharding`, never says `shard_map` — because
`create_mesh_2d` (`bse/bse_ring_comm.py:35`) hides all of it. A physicist needs
items 1, 2 and 4 from `gw_jax` and nothing more.

Two things the reader is **not** told, both of which matter:
* `create_mesh_2d` performs a **collective** (`warm_mesh_cliques`,
  `bse_ring_comm.py:64`). The name does not suggest it; the "every rank must
  reach this together" warning lives in the function (`:38-40`), not at the call
  site.
* `--px/--py` default to 1 (`:247-248`) and reach a *different* mesh builder
  (`:388-389`) than the production path uses (`:131`) — §B.1 D2.

**Distance to target: ~0 import-block lines.** The residue is behavioural, not
structural.

## C.3 `gw/kin_ion_io.py` — the one that fails

```python
60  import os
61  os.environ.setdefault("JAX_ENABLE_X64", "1")
62  os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
63  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
64  os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
65
66  # ---- join the distributed world BEFORE anything touches XLA ------------
67  # ``jax.distributed.initialize()`` refuses to run once the XLA backend is
68  # up, and the import graph below (``psp.*``) reaches jax.  ``runtime``
69  # itself imports no jax, and every piece here is idempotent through an env
70  # sentinel — which is what makes this safe under ``gw.sigma_dispatch``'s
71  # LAZY import of this module from inside an already-bootstrapped driver:
72  # there the calls are all no-ops.
73  from runtime import (init_jax_distributed,                    # noqa: E402
74                       fallback_to_cpu_if_no_gpu_backend,
75                       install_failfast_excepthook)
76  init_jax_distributed()
77  fallback_to_cpu_if_no_gpu_backend()
78  install_failfast_excepthook()
79
80  import argparse
81  from functools import partial
82
83  import numpy as np
84  import jax
85  import jax.numpy as jnp
86  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
87  from jax.experimental.shard_map import shard_map
88  import h5py
```

Plus two lazy jax imports at `:381` and `:1067`, both
`from jax.experimental import multihost_utils as _mh`.

**All four env-read sites:**
* `:61-64` — four `os.environ.setdefault`. These are **exactly**
  `runtime.set_default_env` (`runtime/__init__.py:167-195`), re-typed.
* `:404` — `int(os.environ.get("LORRAX_KIN_ION_LOOKAHEAD", "2"))` inside
  `_default_lookahead()`, in a `try/except ValueError`. The docstring
  (`:397-402`) justifies it: `=1` restores serialised behaviour "so the overlap's
  contribution can be measured against the pipelined default rather than
  asserted". **This is capability/measurement, not policy — QUALITY_PATTERNS
  §8-legal**, and `common/collectives.py`'s `__all__` already claims
  `sweep_lookahead` / `SWEEP_LOOKAHEAD_ENV` as its migration target.
* `:800-803` — the inline `JAX_PROCESS_COUNT`/`JAX_NUM_PROCESSES`/`SLURM_NTASKS`
  chain duplicating `runtime._resolve_proc_count`. Its *use* (`:804-812`, refusing
  when the launcher advertises P tasks but jax joined a world of 1) is excellent
  and should be kept — it is the resolver, not the check, that is duplicated.

**What a physicist must know beyond `gw_jax`:** what `JAX_PLATFORMS` and the two
`XLA_PYTHON_CLIENT_*` knobs do and that they must be set before import; what
`NamedSharding` and `PartitionSpec` are; what `shard_map` is; what
`multihost_utils.process_allgather(tiled=False)` returns; and why
`jax.distributed.initialize()` "refuses to run once the XLA backend is up" — a
genuinely deep fact about jax's initialisation order, stated at the top of a file
about the kinetic + ionic Hamiltonian.

**Distance to target: `:60-78` (19 lines) collapse to
`from runtime import bootstrap; bootstrap()` the moment
`gw/sigma_dispatch.py:174` stops importing this module.** The
`Mesh`/`NamedSharding`/`P`/`shard_map` imports are the deeper problem: **this is
a driver and a library in one file.**

## C.4 `centroid/kmeans_isdf.py` — a kernel judged as a driver

```python
20  from __future__ import annotations
21
22  from functools import partial
23
24  import numpy as np
25  import jax
26  import jax.numpy as jnp
27  from jax import lax, config
28  from jax.sharding import Mesh, PartitionSpec, NamedSharding
29  from jax.experimental.shard_map import shard_map
30
31  from common import timing
32  from common.collectives import device_put_process_local
33
34  config.update("jax_enable_x64", True)
```

**Env reads: none.**

**`:34` is a layering violation of its own class** — a library module mutating
global jax configuration at import time. `AGENTS.md:48` explicitly classifies
this file as "the algorithm library, no `__main__`". Any process that imports it
— including a test that only wants `build_min_image_offsets` — silently gets x64
enabled. `gw/gw_jax.py:44` and `bse/bse_jax.py:39` do the same thing, but they
are entry points, where it is correct.

**Judged as a kernel, its 5 jax imports are appropriate** — `lax`,
`PartitionSpec`, `NamedSharding`, `shard_map` are the vocabulary of a sharded
Lloyd loop, and the docstring (`:11-12`) is honest: "Requires a mesh; the entire
Lloyd loop runs as one `lax.while_loop` inside `shard_map`". **The action is not
to reduce the imports. It is to delete `:34`** and let `kmeans_cli.py`'s
`bootstrap()` own it. Note `:32` — this is the **only** one of the five audited
files that already imports from the service layer.

## C.5 `centroid/kmeans_cli.py` — clean block, policy in the body

```python
 7  from __future__ import annotations
 8
 9  # Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap()
10  # (env defaults + jax.distributed init + CPU fallback; all idempotent).
11  # MUST precede this module's own `import jax` so JAX_ENABLE_X64 etc. take
12  # effect.  NOTE: this used to be set_default_env() + init_jax_distributed()
13  # only; bootstrap() adds fallback_to_cpu_if_no_gpu_backend(), so on a node
14  # with no usable GPU backend this CLI now falls back to CPU instead of
15  # dying at the first jax call.
16  from runtime import bootstrap
17  bootstrap()
18
19  import argparse
20  import math
21
22  import numpy as np
23  import jax
24  import jax.numpy as jnp
25  from jax.sharding import Mesh
26
27  from file_io import WfnLoader as WFNReader
28  from common import symmetry_maps, timing
29
30  from .charge_density import get_charge_density
31  from .kmeans_isdf import (...)
```

**Env reads: none.**

**What a physicist must know:** items 1–5 from `gw_jax`, plus the whole mesh
*policy* argument — `_build_mesh` here (`:190-224`) is not a factorisation, it is
35 lines of policy: `--no-shard`, `--force-shard`, a points-per-shard minimum,
and a multi-host deadlock guard (`:210-212`). That policy is real and worth
keeping; it does not belong in a driver. `:9-15` is also the only place in the
tree where a bootstrap *changelog* is recorded in a driver comment.

**Distance to target: `from jax.sharding import Mesh` disappears once
`_build_mesh` moves to `collectives.resolve_mesh(..., allow_single=...)`.**

## C.6 Scored

| criterion | gw_jax | bse_jax | kin_ion_io | kmeans_isdf | kmeans_cli |
|---|---|---|---|---|---|
| no top-of-module env writes | ✅ | ✅ | ❌ 4 | ✅ | ✅ |
| no `Mesh` in the import block | ❌ | ✅ | ❌ | (kernel) | ❌ |
| no `shard_map`/`PartitionSpec` in the import block | ✅ | ✅ | ❌ | (kernel) | ✅ |
| no global jax config mutation | ❌ `:44`\* | ❌ `:39`\* | ❌ `:61-64` | ❌ `:34` | ✅ |
| uses the service layer | ⚠ `runtime` only | ⚠ `runtime` only | ⚠ `runtime`, unbundled | ✅ `collectives` | ⚠ `runtime` only |

\* correct in an entry point; a violation only in `kmeans_isdf.py`.

## C.7 Second reading — 2026-07-30 17:08:12 CDT

Three of the five files changed between 16:48 and 17:08. Re-read in full; new
md5s and line counts below. **`gw/gw_jax.py` and `bse/bse_jax.py` did not move**
(md5s identical to §C), so C.1 and C.2 stand as written.

| file | md5 @16:47 | md5 @17:08 | lines | raw jax imports | env reads |
|---|---|---|---|---|---|
| `gw/kin_ion_io.py` | `eac5e643…` | `6f96d56b…` | 1088 → **814** | 6 → **1** | 4 → **0** |
| `centroid/kmeans_isdf.py` | `74a469fb…` | `aa705050…` | 1004 → **849** | 5 → **3** | 0 → 0 |
| `centroid/kmeans_cli.py` | `25d041d9…` | `792a6255…` | 513 → **515** | 3 → **0** | 0 → 0 |

**`gw/kin_ion_io.py` now passes the acceptance test.** The `:61-64`
`os.environ.setdefault` block and the three-call unbundled header are gone,
replaced by `bootstrap()` at `:70` — the V3 edge was cut by extracting the
library half, exactly as §A.3 #2 proposed. The whole
`Mesh`/`NamedSharding`/`PartitionSpec`/`shard_map` import line is gone; the only
raw jax import left is `import jax.numpy as jnp` (`:75`). In its place, `:76-78`:

```python
from common.collectives import (barrier, device_count, gather_k_blocks,
                                local_share, process_rank_world,
                                psum_replicate, resolve_mesh)
```

— seven of the nine names that were undefined in `__all__` at 16:53 (§B.3),
now implemented and consumed. `LORRAX_KIN_ION_LOOKAHEAD` is gone from the file.
The one surviving env read is the launcher-mismatch check at `:524-527`, which
is the site §C.3 argued should be **kept** (the check is good; only its resolver
is duplicated).

**`centroid/kmeans_cli.py` now has zero jax imports and no `Mesh(` anywhere.**
**`centroid/kmeans_isdf.py` lost `config.update("jax_enable_x64", True)`** — the
§C.4 recommendation, landed. Its docstring now reads *"Device placement, mesh
construction and the one collective this algorithm needs live in
`centroid.distribution`, not here."*

### The new risk: three device layers, built in parallel

The consolidation produced **three** services where this report argues for one:

| service | mesh builders | delegates to `collectives`? |
|---|---|---|
| `common/collectives.py` (1045 lines, `__all__` now complete) | `resolve_mesh:213`, `single_device_mesh:242` | — it *is* the service |
| `centroid/distribution.py` (**new**, 16:53) | `build_mesh:98`, `single_device_mesh:148`, `process_local_mesh:59` | only `device_put_process_local` (`:34`); its mesh builders are independent |
| `psp/_dist.py` (**new**, 16:55) | `device_mesh_xy:52` | no |

`psp/_dist.py` is self-aware and honest about it — its module docstring opens
*"**TEMPORARY ADAPTER. Delete when the distribution service lands in
`common`.**"*, files three named requests (R1 `device_mesh_xy`, R2
`shard_over_k`, R3 `allgather_sum`), states *"it is here only because
`src/common` is another agent's file"*, and **quantifies the cost it is
deliberately not fixing**: R3's `process_allgather` is P-linear —
"26 MB × P for run_nscf's coefficient buffer … 1.7 GB/rank at P=64" — kept
bit-identical on purpose because that workstream cannot gate `run_nscf` at P>1.
That is a model piece of engineering hygiene and it should be honoured by
actually deleting the file, not by letting it become permanent.

`centroid/distribution.py` is genuinely good in its own right — `:64-66`
documents the `jax.devices()[:1]` hazard as "the spelling this package used at
three call sites until 2026-07-30" (the D1 sites, now fixed), and `:117-127`
turns the `--no-shard` multi-host deadlock into a named refusal. **But it
duplicates the mesh factorisation** (`:130-131`, "same recipe as
`gw_jax._build_mesh`") rather than calling `collectives.resolve_mesh`, and it
re-defines `_P_PER_SHARD_MIN` policy locally.

**And the two `single_device_mesh` functions now contradict each other:**

| | `common/collectives.py:242-256` | `centroid/distribution.py:148-165` |
|---|---|---|
| docstring | "the 1x1 mesh over **this process's own** device… every rank gets a different mesh" | "1x1 mesh over the one device of a **ONE-PROCESS** run" |
| device | `jax.local_devices()[:1]` (via `wfn_transforms.process_local_mesh`) | `jax.devices()[:1]` — the **global** list |
| at P>1 | **works**, per-rank | **raises `ValueError`** |
| per-rank equivalent | this function | a *different* one, `process_local_mesh:59` |

Both are defensible designs. Both are correct in isolation. **The same name
meaning opposite things at P>1 is not**, and it is the exact drift class this
report catalogues — created, in a few hours, by the fix for it.

### Remaining production `Mesh(` sites: 8 (was 21)

`bandstructure/htransform.py:39`; `bse/bse_feast.py:781`;
`bse/bse_pseudopoles.py:212`; `bse/bse_ring_comm.py:52`, `:1045`, `:1103`;
`bse/bse_w_exact.py:85`; `gw/gw_jax.py:88`; `psp/_dist.py:59` — plus three
service-internal ones (`collectives.py:235`, `distribution.py:145`, `:165`) and
`wfn_transforms.py:1443`. **`bse/` and `gw/gw_jax.py` are untouched**, so every
BSE finding in §B.1 (D2, D5, D6) stands unchanged, as does §V2.

---

# D. The target architecture

## D.1 What already exists and is good — the reference shapes

| service | file | why it is the reference |
|---|---|---|
| **`ffi.common.gate.Gate`** | `ffi/common/gate.py` (401 lines) | Owns four things that used to be four drifting copies: grammar, platform, probe, announce-or-refuse. Its docstring (`:21-49`) *argues its own boundary* — it explains why a single-tier resolver cannot serve both a cache key and a lowering decision, which is why it does not fold into `ffi/linalg/resolve.py`. Doc: `docs/dev/ffi_gate_contract.md`. |
| **`common.contract_bands`** | `common/contract_bands.py` (544 lines) | Five encoded policies, each with its measurement cited inline (`:30-84`); five named refusals that fire before any collective; and an explicit statement of what was **refuted** by measurement (`:57-61`). It also **reads no env directly** — it consumes `ffi.mklblas.gemm.GATE` as a typed object. Doc: `docs/dev/staged_reshard_primitive.md`. |
| **`runtime`** | `runtime/__init__.py` | **The precedent that this exercise works.** Its docstring `:30-38` records: "Previously five different modules had their own copies of this logic, drifting apart over time (gw.gw_jax had the SLURM-coordinator fallback; psp.run_nscf and centroid.kmeans_isdf didn't…)". It solved exactly this problem once; the result is `bootstrap()`, used verbatim by 3 of the 5 audited drivers. |
| **`common.collectives`** (as declared) | `common/collectives.py:1-8` | The target statement for the whole architecture: *"A driver that needs another process's data — or needs to agree with it — calls this module and nothing below it. Nothing here requires the caller to know what a `Mesh`, a `NamedSharding`, a `shard_map` or `multihost_utils` is; that is the point."* |

## D.2 The service list

### S1. `runtime` — process bootstrap. **Exists, good, near-complete.**
Owns: env defaults before `import jax`, `jax.distributed.initialize`, CPU
fallback, failfast excepthook, glibc malloc tuning, GPU-plugin skip.
API: `bootstrap()` + the six pieces (`runtime/__init__.py:60-68`).
**Gap:** `_env_falsy` / `_FALSY_TOKENS` (`:53-58`) are private. Exporting them is
a one-line `__all__` change and is what lets S3 exist without a sixth grammar.
Secondary gap: `_resolve_proc_id` needs `PMI_RANK` / `OMPI_COMM_WORLD_RANK` and
`gate.rank_id`'s hardening (§B.4).

### S2. `common.collectives` — the cross-process layer. **~~Exists, ~55 % written.~~ As of 17:08: complete and consumed.**
API: `__all__` at `common/collectives.py:68-98` — take it as given, it is the
owner's declaration.
**Gaps, in priority order** *(gap 1 closed at 17:08; the rest stand)*:
1. ~~The 9 undefined names (§B.3).~~ **Closed** — AST re-check at 17:09 shows 0
   undefined, and `gw/kin_ion_io.py:76-78` now imports seven of them (§C.7).
   **The gap this leaves behind is bigger:** two *other* device layers were
   built in parallel (`centroid/distribution.py`, `psp/_dist.py`), one of which
   asks in its own docstring to be deleted into `common`. **Deciding whether
   `collectives` is THE device layer or one of three is now the open
   architectural question**, and it is more urgent than anything else in this
   list because `psp/_dist.py`'s three filed requests (R1/R2/R3) are a
   ready-made spec that will go stale.
2. **`resolve_mesh` does not call `warm_mesh_cliques`**, and
   `bse_ring_comm.create_mesh_2d:64` does — which is the only reason the BSE
   Lanczos survives `impl=mpi`. Either `resolve_mesh` warms (and `create_mesh_2d`
   becomes a thin alias), or the warm-up is a mandatory, *named* second call.
   The current state — whether your mesh is warm depends on which of 21 builders
   you used — cannot stand. **Highest-severity single item in the report.**
3. `resolve_mesh` has no `px/py` override, so the four BSE CLIs cannot migrate
   without losing `--px/--py`. Add `shape=None`, folding in
   `bse_feast.py:778-779`'s existing over-request check.
4. A sanctioned `gather_to_host(x, *, tiled=)` — the six copies disagree on
   `tiled` and on the fast-path predicate, and one of those disagreements is a
   documented correctness surface (`bse/exciton_bands.py:117-119`).

### S3. **`common.config`** — the deck and the knobs. **Missing.**
Owns: (a) the deck parse (`gw/gw_config.py:read_lorrax_input`, imported by four
non-`gw` modules); (b) the backend enums, starting with `SlabIOBackend`; (c) **one**
env-knob accessor built on `runtime._env_falsy` plus `gate.MODE_SPELLINGS`' token
table, so no sixth grammar is invented; (d) ownership of
`docs/dev/env_vars.md`'s generator, extended to catch tuple-driven and
loop-variable reads (§B.5).
API: `read_deck(path) -> LorraxConfig`; `SlabIOBackend`; `env_flag(name, default)`.
**`LorraxConfig` itself can stay in `gw/`** — it is genuinely GW's, and moving
94 KB of dataclasses is all risk and no reader value.

### S4. `ffi.common.gate` + `ffi.common.ffi_loader` — FFI capability. **Exists, good.**
One gap worth closing: **cublasmp has no gate** (§B.6.6). Give it a `_SPEC` row in
`resolve.py` so it refuses at resolve time with the reason and the fix, matching
`resolve.py:16-21`'s own doctrine. Low risk — no in-tree consumer.

### S5. **`common.announce`** — one announcement discipline. **Missing; two half-copies.**
Owns `rank_id`, `rank0`, `announce_once(key, msg, scope=)`. Today:
`ffi/common/gate.py:117-177` (good) and `runtime/aot_memory.py:51-64` (prints
from every rank). **Home must be `runtime/` (layer 0)** — `aot_memory` cannot
import layer 1 — with `ffi.common.gate` re-exporting. Constraint that must
survive: launcher env vars ahead of `jax.process_index()`
(`ffi/common/gate.py:56-61`).

### S6. `common.contract_bands` — the reshard primitive. **Exists, good.**
The named future adopters (`common/contract_bands.py:26-28`) are precisely the
§B.2 near-duplicate clusters. **Migrating them is the real answer to the
`shard_map` boilerplate.**

### S7. **`ffi.common.mkl_thread_pin` + `ffi.common.layout`** — the C++ extractions. **Missing.**
`mkl_thread_pin.h` is already specified in `docs/dev/vendor_gemm_service.md:180-186`
("MPI-free, **not** including `blacs_grid.h`") and is now met on the doc's own
threshold. It must expose the `RTLD_DEFAULT → RTLD_NEXT` policy (copy C) and
both symbols (copy A), and it must work in a TU without `-fopenmp` — which means
fixing the per-source-file `-fopenmp` scoping first, or making the auto policy
prefer `MKL_Get_Max_Threads` over `omp_get_max_threads` in all three.
`ffi.common.layout` replaces the four-backend transpose-wrap family (§B.2).

## D.3 Migration order, with the gate for each step

Each step is independently revertible and independently gated. **No step here
proposes a refactor whose gate I cannot name.**

*Steps 4, 8 and most of 7 landed between 16:47 and 17:08 (§C.7). Steps 0 and 0b
below are new, created by that landing.*

| # | Step | Gate (must be able to fail) |
|---|---|---|
| **0** | Rename one `single_device_mesh` (§C.7) | A test calling both at `process_count()==2`, asserting the documented difference. Cannot pass vacuously — today one raises and one does not. |
| **0b** | Decide: is `common.collectives` THE device layer? Then either delete `psp/_dist.py` into it (its R1/R2/R3 are the spec) or write down why three layers is right | `grep -c` for `def .*mesh` outside the chosen service must reach its target number — and seed it with a decoy first, because `grep -c` printing 0 while exiting 1 is a recorded void-instrument in this project (README lesson #1). |
| 1 | Move `SlabIOBackend` to `file_io/`, re-export from `gw.gw_config` | `import file_io` succeeds with `src/gw` renamed away. Run it **before** the move to prove it errors. |
| 2 | **Add the `MPI_Query_thread` guard to `slate/cpp/context.cc`**, mirroring `phdf5/cpp/context.cc:200-217` | Run SLATE under an MPI initialised at `FUNNELED`; the warning must appear. Without the deliberate downgrade the gate is void. |
| 3 | Export `_env_falsy`/`_FALSY_TOKENS`; one table-driven grammar test | The test must fail today on `LORRAX_MALLOC_TRIM`, `LORRAX_RESTART_WRITE_LOG`, `LORRAX_SKIP_VQ_GATES`. Then fix those three plus `LORRAX_EXIT_AFTER_ZETA`'s presence test. |
| 4 | Finish the 9 undefined `collectives` names; migrate `kin_ion_io.replicate_to_mesh` + `_gather_indexed_blocks`, `WfnLoader._assemble_process_local`, and the six `_to_host`/`_to_numpy` copies | Cache-**cold** HLO collective table (`ISDF_JAX_CACHE_DIR=""`) before/after: `all-gather` count non-increasing, no new `jit(_identity_fn)`. Plus `kin_ion.h5` dataset-md5 parity (**not** `h5dump` — lesson #1 records an `h5dump` md5 that diverged only because the filename is in the dump header). |
| 5 | Replace the `except: pass` barriers (`file_io/_slab_io_ffi.py:48-55`, `gw/gw_output.py:294-298`, +4) with `collectives.barrier` | P=4 run with one rank given a deliberately different barrier tag: must die loudly with rc≠0, not hang-then-rc-0. |
| 6 | Decide the warm-up contract (S2 gap 2) | The existing discriminating control (job 7881053 shape): world-only must FAIL, x+y+world must pass. Reuse `wk_REL/allmpi/bse_warm_gate.sbatch`, which already ships a control built to fail. |
| 7 | Migrate the 21 `Mesh(` sites | AST test: no `Mesh(` call outside `collectives.py` + `wfn_transforms.py`, seeded first to prove it fires. Plus a P=4 CPU `kmeans_cli` byte-compare against `centroids_T_t134.txt`. |
| 8 | Extract `compute_hartree_matrix`/`replicate_to_mesh` to `gw/hartree_matrix.py`; collapse `kin_ion_io:60-78` to `bootstrap()` | `python -m gw.kin_ion_io -i <deck>` → byte-identical `kin_ion.h5` (dataset md5s). |
| 9 | `common.announce`; retire `aot_memory.announce_once` | P=4: a forced memory-model demotion prints **once**, not 4×. |
| 10 | Extract `ffi/common/cpp/mkl_thread_pin.h`; fix `-fopenmp` scoping | `nm -C` on the rebuilt host `.so`: **one** `mkl_set_num_threads_local_ptr` symbol, not three. Plus the scalapack `pzheevd` timing at the 12×12 grid must not regress (the 24× claim at `blacs_grid.h:110-118` is the reference). |
| 11 | Fix the four `--px/--py` defaults; then migrate those CLIs | `bse_feast` on 4 devices with no flags builds a 2×2 mesh. Fails today. |
| 12 | `ffi.common.layout` | Existing FFI backend tests, byte-parity on eigenvalues **and** eigenvectors — the eigenvector convention is the thing being unified (`ffi/linalg/plan.py:101-111`). |
| 13 | Migrate `bse_ring_comm` / `bse_stack_matvec` / `vq_interp` onto `contract_bands` | Already specified as slated adopters at `common/contract_bands.py:26-28`; use that module's own gate standard. |

---

# E. What NOT to unify

A named reason not to abstract is worth as much as an abstraction. Six traps.

## E.1 Do NOT merge `runtime.nccl_warmup` and `collectives.warm_mesh_cliques`

Same signature, adjacent purposes, **different designs**.
`warm_mesh_cliques` (`common/collectives.py:638-724`) works *because* its jit is
small enough that XLA takes `ThunkExecutor::ExecuteSequential` and runs the thunk
**inline on the calling thread** (`:653-660`) — that inlining is the mechanism
that satisfies `MPI_Is_thread_main` at communicator creation. It uses `shard_map`
+ `lax.psum` explicitly and no-ops unless `impl=mpi` (`:695-697`).
`nccl_warmup` (`runtime/__init__.py:607-649`) deliberately does **not** use
`lax.psum` and says why (`:635-637`); its purpose is to pay `ncclCommInitRank`
topology discovery, and there is no thread constraint.
A merged body would have to be simultaneously small-enough-to-inline (CPU) and
shaped to emit distinct NCCL `replica_groups` (GPU), carrying two
mutually-irrelevant platform gates. **Unify the call site — one
`prepare_mesh(mesh)` the driver calls, each half still a no-op off its platform.**

*(Related and confirmed: the same refusal already holds one layer down.
`ffi/linalg/resolve.py` reads **no environment at all**, so there is no
shared-variable drift between it and `gate.py`; their only overlap is
`probe_target`. The prior assessment's refusal to fold them stands, and §B.6.6
now has the negative evidence to back it.)*

## E.2 Do NOT build a generic `shard_map` wrapper

`in_specs`/`out_specs` **are** the distributed algorithm; a wrapper that takes
them as arguments has abstracted nothing and inserted a layer between the reader
and the only thing they came to read. `common/contract_bands.py` is the proof
that the right abstraction is *specific*: it replaces a family of `shard_map`
sites by owning **one named pattern** with its axis-order policy, its
de-promotion policy and its divisibility refusal — which is why it can carry a
562-line contract document. A generic wrapper could carry no such document.
And the counter-example is decisive: `common/fft_helpers.py:253`/`:271` are the
only two functions written *to be* reusable `shard_map` wrappers, and they are
two of the 21 sites that omit `check_rep`.

## E.3 Do NOT collapse the two *deliberate* boolean grammars, or reorder `rank_id`

`runtime._FALSY_TOKENS` (`runtime/__init__.py:53`) is two-valued with `""` falsy;
`gate.MODE_SPELLINGS` (`ffi/common/gate.py:91-95`) is three-valued and maps `""`
to the gate's declared *default*, never to a spelling (`:87-89`) — because `auto`
must be distinguishable from an explicit `on`. Collapsing them silently converts
`LORRAX_BANDS_GEMM_FFI`'s auto-demote (never refuses) into a refusal or
vice-versa. **Share the token table; keep two resolvers.**
Second sub-trap: any unified rank resolver must keep launcher env vars **ahead
of** `jax.process_index()`. `ffi/common/gate.py:56-61` records that the latter
goes through `get_backend()` (`jax/_src/xla_bridge.py:1119`) and **initialises
the XLA backend**, destroying tier 1's before-`initialize` promise and the
kernel-cache keys in `gw/ppm_tau_kernel.py`. A "simplification" to
`jax.process_index()` is a correctness regression with no visible symptom until a
compile-cache miss storm.

## E.4 Do NOT remove `ffi/linalg/plan.py:97`'s upward import — and do NOT duplicate it

It is a layer-1→layer-2 edge and looks like a violation. Read `:90-96` first: it
exists because plain `jax.device_put` of host numpy onto a multi-process sharding
"fires JAX's hidden `assert_equal` all-gather at P × x.nbytes … for an (n, n)
c128 FFI operand, exactly the class of silent cost this helper exists to
prevent." The alternative — a second copy of `device_put_process_local` inside
`ffi/` — is the exact duplication this audit exists to eliminate. **Accept the
edge and document it as a sanctioned exception.** (Moving the helper down to
layer 0/1 would drag the whole topology API with it, for no reader benefit.)

## E.5 Do NOT unify the test/bench mesh sites with the production ones

19 of the 40 `Mesh(` sites are in `common/*_test.py` / `*_bench.py` /
`ffi/cusolvermp/profile_batched.py`. They construct meshes *deliberately
differently* — `profile_batched.make_mesh` parses `--mesh 2x2` and `SystemExit`s
on a mismatch (`ffi/cusolvermp/profile_batched.py:119-126`), which is correct for
a benchmark whose whole job is sweeping geometries. Forcing them through
`resolve_mesh` would delete the sweep. **Move them to `tests/`** — which also
removes 108 of `common/`'s 125 env-read matches and two of the nine layering
violations (A.3 #7) — **and leave their mesh construction alone.**

## E.6 Do NOT give the four `--px/--py` BSE CLIs one shared mesh-flag parser

The divergence there is not plumbing, it is a **defaults bug** (§B.1 D2), and
wrapping it in a shared helper freezes `default=1` into the shared layer where it
becomes harder to see. Fix the default first, in place, four times. Only then ask
whether a helper is warranted — and by then `resolve_mesh(shape=…)` will already
be it.

---

# Appendix 1 — Method, and what I did not verify

**Dependency graph.** All 236 `.py` files under `src/` parsed with Python's `ast`;
`Import`/`ImportFrom` nodes collected with line numbers; relative imports resolved
against each module's package. This is why §A.4's "three modules reach `jax._src`"
is a confirmation, not a grep negative.

**Export completeness.** `common/collectives.py`'s `__all__`-vs-defined comparison
(§B.3) is an AST parse of module-level `FunctionDef`/`ClassDef`/`Assign`/
`AnnAssign`.

**Absences.** `mkl_thread_pin.h` (§B.6.0) is established by `find` **and**
`git ls-files` filtered to header extensions — never by `strings`. The three pin
copies and the 389-symbol export surface are established by `nm -C` / `nm -D` on
built `.so` files; `DT_NEEDED` contents by `readelf -d`. Header-defined symbols
appear only in the symbol table, which is the trap lesson #2 records.

**Timestamps.** Audit start `2026-07-30 16:47:07 CDT`, end `~17:35`.
`common/collectives.py` changed mid-audit (16:47 → 16:53); all line numbers cited
for it are from md5 `861d44ae64171a3d5a55f6f882522b22` (724 lines) **except** the
one marked `(pre-16:52 line number)` in §B.2. `centroid/` and `psp/` were under
active edit; §C carries md5s for all five audited drivers, none of which moved.

**What I did NOT verify, and would need a job to settle:**
1. That `gw/gw_jax.py` survives `impl=mpi` **only** because
   `zeta_projection`/`contract_bands` warm its cliques (V2). I have the call
   sites and the platform gates; I do not have a run with those calls removed.
   The control shape exists (`wk_REL/allmpi/bse_warm_gate.sbatch`).
2. Whether `nccl_warmup`'s `jax.jit(jnp.sum)` path *also* creates CPU MPI cliques
   under `impl=mpi` (it is not gated on the impl). If it does, V2 is better than
   stated; if not, it is exactly as stated. `TF_CPP_VMODULE=cpu_cliques=3` settles
   it in one P=4 run — the same instrument `common/collectives.py:667-668` used.
3. That no local-scope `dlopen` path exists (§B.6.1). Established by grep over
   `.py`/`.sh` for `RTLD|CDLL(|LoadLibrary`; **a C-level `dlopen` from a
   third-party dependency would not appear in that sweep.** The conclusion "the
   pin divergence is currently masked" rests on that, and is therefore an
   inference, not a confirmation.
4. The SLATE `THREAD_MULTIPLE` consequence (§0, §B.6.2). The *code paths* are
   confirmed by reading both files; the runtime consequence is inferred from
   `phdf5`'s own recorded measurement (~29 % multi-node crash rate, scorecard
   AS.4b). I did not run SLATE under a downgraded MPI.
5. Runtime behaviour of anything else. This is a static audit; every performance
   and failure claim quoted is the codebase's own recorded measurement, cited to
   the comment that records it.
