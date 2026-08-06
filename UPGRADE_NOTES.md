# Upgrade notes — origin/main → current HEAD (2026-08-01)

What a user continuing from origin/main hits on this branch
(`fix/zq-band-gather-device-invariance`, a fast-forward of ~210 commits).
Every claim below is verified against the code at HEAD, not against memory.
Grouped: what breaks, what warns, what changed silently-but-safely.
Binding rulings behind the breaking changes: `docs/architecture/decisions.md`.

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

**`use_ffi_io` is deprecated** (tri-state, default now unset; origin/main
default was `true`). It still works: explicit `false` forces the
`h5py_allgather` writer, `true` states a phdf5 preference, and `slab_io`
takes precedence when both are set — each case prints one deprecation
notice. Remove it from decks; the replacement is
`slab_io = auto | phdf5_ffi | phdf5_host | h5py_allgather`.

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
