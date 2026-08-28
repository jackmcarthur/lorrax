# Upgrade notes

User-visible changes, newest first. Binding rulings behind the breaking
changes live in `docs/architecture/decisions.md`.

## 2026-08-28 — startup ownership, BSE mesh flags, emulated CPU meshes

- The runtime owns `JAX_ENABLE_X64`: it applies the resolved value even when
  jax was imported before the driver, and a resolved `False` refuses at
  startup. `LORRAX_ALLOW_X64_OFF=1` continues as an announced uncertified
  run. The per-driver `jax.config.update` lines are gone.
- Drivers no longer arm the persistent compile cache; step 7 of
  `runtime.initialize_communicator_stack` owns it. With `ISDF_JAX_CACHE_DIR`
  unset and `LORRAX_RUN_DIR` set, sequential drivers share one
  workflow-local cache under `$LORRAX_RUN_DIR/.lorrax_jax_cache`.
- BSE-family drivers: omitted `--px/--py` now means the run's canonical
  square mesh (it used to mean 1×1). An explicit shape must consume the
  job's device count exactly — under- and over-requests both refuse.
- `gw_jax`, `kin_ion_io`, `downfold_cli` and `kmeans_cli` answer `--help`
  and bad argv before any runtime exists (`runtime.cli_seam`); the other
  four drivers still pay full bring-up first.
- Single-process multi-device CPU meshes
  (`XLA_FLAGS=--xla_force_host_platform_device_count=N`) now run end to
  end: `SlabIO` serves them through an announced serial tier
  (`file_io._slab_io_serial`, CPU only). The `p*q == process_count`
  refusals stand everywhere else.
- `qp_solver = self_consistent` beside a dynamic `compute_mode`
  (`gn_ppm`/`hl_ppm`/`mpa`) refuses at driver entry; pair it with `cohsex`.

## 2026-08-18 — retired HDF5 controls now refuse or are absent

- GW decks containing `slab_io` or `use_ffi_io` now refuse with a targeted
  removal message. Remove either line; SlabIO has one collective transport.
- kmeans no longer accepts `--use-phdf5`; `WfnLoader` selects its one valid
  scalable read path from runtime capability.
- SlabIO no longer accepts `chunks=`. The argument was ignored and every
  collective dataset was already contiguous. The sigma and zeta writers no
  longer request a layout the native create cannot produce.
- `LORRAX_PHDF5_CLOSE_VERBOSE` defaults to compact logging: empty/fast closes
  are quiet, while queued or slow I/O still emits one summary. Set it to `1`
  for the former per-phase diagnostics or `0` for silence.

## Changes through 2026-08-01

The remaining entries describe the earlier origin/main-to-HEAD upgrade.

## What breaks (refusals and hard errors)

**The FFI layer is REQUIRED** (`decisions.md` 2026-08-01). Where origin/main
ran everything through native XLA, GW and htransform now route their flat-k
FFTs and the large band contraction through vendor FFI handlers, and a
missing or unloadable FFI library is a **startup refusal**
(`ffi.gate.Gate.enforce`, wired into `runtime.initialize_communicator_stack`
step 6b), naming the `.so`, the env var, and `docs/environment/overview.md`
— never a silent demotion. Practical consequence: you must build the FFI
library before running — `src/ffi/cpp/build_host.sh` (generic host) or
`config/frontera/build_ffi_host.sh` (Frontera MKL/ScaLAPACK), pointed at by
`LORRAX_FFI_HOST_SO`; the CUDA library via `config/frontera/build_ffi.sh` and
`LORRAX_FFI_SO`. Per-knob semantics:

- `LORRAX_FFT_FFI=0` **refuses**: the XLA flat-k twin inside
  `make_flat_k_fft` was deleted, there is nothing to opt out to (recover the
  arm from git history for a debugging build). Handlers are c128-only.
- `LORRAX_FFT_FFI_FUSED=0` is a real, announced opt-out onto the decomposed
  three-transform chain — itself FFI-served, so a structural choice between
  two certified forms, not a native fallback.
- `LORRAX_BANDS_GEMM_FFI=0` is an announced **UNCERTIFIED** opt-out onto the
  retained XLA einsum arm (retained because `extra="minor"` structurally
  cannot ride a batched GEMM and quietly keeps the XLA plan under every mode).
- CUDA differences: the FFT dial resolves to the cuFFT strided handlers
  (same target names, both flat-k and gw_conv); the GEMM dial does not exist
  on CUDA (host symbol table only — XLA:GPU's cuBLAS dot lowering IS the
  required path there, and the startup report says so); an absent CUDA
  library refuses identically (verified rtx job 7885151). BSE is out of
  scope on both platforms: its FFTs are `local_*fftn3` = `jnp.fft` aliases
  with no FFI route, kept by the ruling.

**Square process meshes only.** `resolve_mesh` refuses a device count that
is not a perfect square, naming s² and (s+1)² to request;
`create_mesh_2d` / `create_mesh_xy` / `RuntimeStack.reshape` refuse
px ≠ py; the rectangular-mesh accommodation was deleted. Note the ruling's
letter in `decisions.md` prescribes idle-rank truncation; the implementation
deliberately refuses instead, because idle ranks deadlock under the
`impl=mpi` transport (communicator creation is collective over
MPI_COMM_WORLD — full argument in the `resolve_mesh` docstring; sandbox
CLAIMS row 33 records the deviation, owner may re-open). Launch square
counts: 4, 16, 64, ...

**`sigma_omega_accumulation = kij_stream`** raises ValueError: the
single-process streamed-h5 accumulator was removed 2026-07-31. Use `kij` or
`auto`; for cubes that do not fit, `sigma_omega_layout = sharded` (below).

**`w_dyson_solver = lstsq`** raises (two-plan W cleanup): the SVD min-norm
inner solve masked a rank-deficient A = 1 − V·χ0 — reduce n_mu or raise
`zeta_rcond` instead. `lu` deprecation-warns and resolves to `local`.

**`use_low_mem_eigh = true` with `eigh_backend = off`** refuses at parse
time (a contradiction; see the new-keys section).

**`strict_keys = true`** (new, default false) upgrades the unknown-deck-key
warning to a ValueError naming every unknown key — set it in CI decks.

## What warns (deprecations and behavior you should notice)

**Unknown deck keys now warn.** Any key not in `gw_config._DEFAULTS` and not
covered by a legacy branch is reported in ONE aggregated rank-0 warning
(key + line number) and ignored. On origin/main such keys were dropped
silently. Consequence for removed keys:

- `cusolvermp_charge` / `cusolvermp_lu` (deprecated aliases on origin/main)
  were **removed** — they now warn-and-ignore, i.e. they stop steering
  anything. Use `distributed_cholesky` / `distributed_lu`.
- `isdf_memory_mode` (auto | high_mem | low_mem) was removed with the W
  cleanup — warn-and-ignore. The W Dyson solve is selected by
  `w_dyson_solver = local | distributed`.

**Env-twin deprecations**: `LORRAX_ZETA_RCOND` / `LORRAX_ZETA_RIDGE` and the
`LORRAX_SC_*` family still win over the deck keys when non-empty, printing a
rank-0 deprecation notice; ζ-fit provenance records the EFFECTIVE
(post-override) values so dropping the env cannot silently reuse a ζ at a
different conditioning cutoff.

**FFT-FFI knob renames** (P1 wave, 2026-07-31): the C++ knobs are spelled
`LORRAX_FFT_FFI_THREADS`, `LORRAX_FFT_FFI_CHUNK`, `LORRAX_FFT_FFI_LOG` (one
spelling on both platforms; `_LOG` is rank-0-scoped, `=all` for every rank).
The old spellings `LORRAX_MKLFFT_THREADS` / `LORRAX_MKLFFT_CHUNK` /
`LORRAX_MKLFFT_LOG` / `LORRAX_CUFFT_LOG` are honored as deprecated aliases
with a one-time announcement; the new spelling wins when both are set.

**Env grammar hardening**: unrecognized values of the C++/py knobs announce
loudly and resolve to the default (grammar errors must not kill a run, and a
typo must not silently pick a known-bad policy — e.g.
`LORRAX_SCALAPACK_MKL_THREADS` garbage used to fall through `atoi()` to the
24×-slower configuration). Off-dials may refuse; typos never do.

## What changed silently-but-safely

**New deck keys** (full list: `docs/input_reference.md`, generated from
`_DEFAULTS`; load-bearing discussion in `docs/drivers.md`):

- `hartree_source = auto | stored | isdf | gspace` — the G-space vs ISDF
  V_H switch; auto resolves stored → folded → isdf.
- `distributed_zeta_solve = auto | replicated | per_q | distributed` — ζ
  back-solve tier; auto = replicated under the 4 GiB gather cap, else
  per_q; `distributed` (ScaLAPACK pzheevd factor + 2-D-sharded back-solve,
  nothing O(μ²) replicated) is a different, equally valid GAUGE (~κ·ε).
  ζ-fit provenance now records the gauge tier ('replicated' |
  'distributed'; per_q collapses to replicated — same factor bits). The
  schema was NOT bumped: a legacy `tmp/zeta_q.h5` whose stamp lacks the
  tier key is treated as a replicated-gauge fit — replicated-tier reruns
  reuse it with a one-line notice; a distributed-tier rerun refits, with
  the mismatch named. No forced refit of existing ζ files.
- `w_dyson_solver = local | distributed` — the exactly-two W Dyson plans
  (`auto` is a permanent alias of `local`); `distributed` refuses loudly
  when unavailable, never downgrades.
- `sigma_omega_layout = replicated | sharded` — Σ_c(ω,k,m,n) cube stays
  mesh-tiled end-to-end under `sharded`, for every `qp_solver`; refuses an
  indivisible window or `h5py_allgather` at P>1.  The `self_consistent`
  refusal shipped with this key was REMOVED 2026-08-05: the SC loop never
  rotates the cube, so the "rotation seam" it named does not exist, and the
  two layouts measure bit-identical under SC (jobs 7889782/7889789).
- `eigh_backend = auto | off | distributed | cusolvermp | slate |
  scalapack` — BSE/htransform distributed-eigh sites; `use_low_mem_eigh =
  true` + `auto` resolves to `distributed`.
- `strict_keys` — see above.

**Startup entry point.** All seven chain drivers (kmeans_cli,
get_dipole_mtxels, kin_ion_io, gw_jax, htransform, bse_jax, exciton_bands)
now start through ONE module-top call, `runtime.initialize_communicator_stack()`
— failfast hook, env defaults, jax.distributed, backend init, canonical
square mesh + communicator-clique warm-up (`warm_mesh_cliques`, required
under `impl=mpi`), FFI gate enforcement, and one rank-0 startup report
stating every resolved dial and demotion. Drivers no longer call
`prepare_mesh`/`bootstrap` themselves.

**Process teardown.** `gw.gw_jax`'s `__main__` ends via
`runtime.finalize_process(rc)`: ordered explicit teardown (effects barrier,
unregister jax's `clean_up` atexit, `jax.distributed.shutdown()`, run the
remaining atexit hooks, announced `os._exit`). This cures a deterministic
interpreter-teardown deadlock after fully-cold in-process compile storms
(XLA:CPU client destructor pool shutdown; jobs 7884928/7884989). If you
wrapped gw_jax in your own post-main `os._exit` harness, drop it. Note the
process does not run interpreter finalization after `main()` — atexit
duties are executed explicitly, nothing is silently skipped.

**`slab_io = auto` demotes instead of aborting on bare launches**: it now
probes MPI bootstrapability (launcher PMI env, else a throwaway-subprocess
singleton-init probe) before selecting either MPI tier, and demotes to
`h5py_allgather` with a full announcement when MPI cannot bootstrap — a
bare `python -m gw.gw_jax` no longer dies in MPI_Init_thread.

**Transport.** Production CPU collectives are
`JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` (MPItrampoline → patched
MPIwrapper → Intel MPI/mlx; recipe `docs/dev/mpi_collectives.md`,
env block `config/frontera/mpi_transport_env.sh`). gloo is banned at
distributed tiers: reproducible ReduceScatter timeouts at P=64 and ~5%
silent reduce-scatter corruption (sandbox CLAIMS rows 3-4). The startup
report warns when a multi-process CPU run lands on gloo.
`LORRAX_MPI_FINALIZE_FIX=skip_atexit` (overlay sitecustomize) is mandatory
for impl=mpi runs.

**Two behaviour changes from the distrib_la replumb, ACCEPTED as correct
rather than fixed** (2026-08-07; adjudication item 7 — recorded here because
neither is a bug and both would otherwise read as one to the next person who
finds them).

1. **`solve_zeta`'s `mu_pad` divisibility net is now UNREACHABLE for
   ScaLAPACK factors, and that is the fix, not a regression.** The net
   (`isdf/core.solve_zeta`) demoted `scalapack_lu`/`cusolvermp_lu` to the
   per-q `jnp.linalg.solve` when `n_rmu_logical` did not divide both mesh
   axes. A ScaLAPACK factor now arrives as a `distrib_la.FactorToken` and
   the token branch returns before the net, because `distrib_la.factor`
   REFUSES a non-dividing extent at FACTOR time — earlier, with the failed
   guard named, and before any collective. The supersession is strictly an
   improvement: the old solve-time demote kept the ScaLAPACK factor's own
   `ipiv` and handed it to `lax.linalg.lu_solve`, whose pivot convention is
   not ScaLAPACK's, so the "safe fallback" computed a wrong answer
   successfully. The net stays in place for the array-factor routes it is
   still correct for; its `print` (not `warnings.warn`) is deliberate —
   warning dedupe is what made the original demotion invisible in
   production logs.

2. **`use_low_mem_eigh` now threads into `compute_wfns_fi` on the two
   raw-params drivers** (`bandstructure.htransform`, `bse.exciton_bands`).
   Both used to spell the CLI-over-deck precedence inline and never call
   `gw_config.resolve_eigh_backend`, so the key parsed, defaulted, validated
   and was read by nobody on those two paths. It is live now: with
   `use_low_mem_eigh = true` and `eigh_backend = auto`, htransform's Gram-eigh
   line changes from the native description to the distributed one. Intended
   — and the consequence is that a machine which cannot serve the
   distributed eigh now REFUSES those runs where it used to run native
   silently. That refusal is armed on purpose; it is the whole point of the
   key.

**Where the docs live now**: `docs/drivers.md` (the seven drivers: flags,
outputs, failure modes), `docs/input_reference.md` (every deck key —
regenerate with `tools/gen_input_reference.py`), `docs/environment/`
(overview, transports, per-machine pages incl. `machines/frontera.md`),
`docs/dev/large_nmu_operation.md` (two-plans-per-family map, keys,
thresholds), `docs/dev/env_vars.md` (the env registry — gated by
`tests/test_env_registry.py`), `docs/architecture/decisions.md` (binding
rulings). Which page owns which fact is stated once, in the register at the
top of `docs/index.md`; certification scope lives in the sandbox `CLAIMS.md`
ledger rather than in a doc page, because a page recording it goes stale
silently.
