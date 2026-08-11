# Small issues ledger — non-physics items to fix later

A living list for the small, non-physics-implicating things agents stumble
on. Rules: strike rows in place when fixed (cite the commit); nothing
physics-implicating belongs here (that goes to KNOWN_FAILURES or the
owner's decision list); keep this file short enough to read in one sitting.
Started 2026-08-09 from the BSE/sigma campaign's residue.

**Canonical copy: this file, `tests/known_failures/SMALL_ISSUES.md`, in the
repository — and it is the only one.** It used to live outside the tree in two
places that were kept in step by hand, the sandbox copy at
`/pscratch/sd/j/jackm/sandbox_v2_rescue_2026-08-06/SMALL_ISSUES.md` and the
mirror at `~/lorrax_service_phase/SMALL_ISSUES.md`; both are now three-line
pointer stubs. That arrangement nearly lost the ledger twice, because an agent
editing one copy had no way to know the other had moved. Edit this file on a
branch and let it land like any other change, so the ledger is versioned with
the tree it describes and two branches editing it get a merge rather than a
silent overwrite.

## Open

1. **Sternheimer G-sphere pad mask** — 23 MB at 144-k stored as f64 for a
   0/1 mask, applied as three multiplies per matvec. Lives in
   `src/psp`/`src/common` — adjacent to the protected projector
   contraction, so scope any fix carefully and touch only the pad-mask
   transport. [WINDOWED_EXP_SWEEP.md]
2. **`sternheimer_precond`'s "mask" is not a mask** — continuous TPA
   weights wearing the name; rename/doc cleanup only, no numerics.
   [WINDOWED_EXP_SWEEP.md]
3. **XLA_FLAGS loses the race with the census's first `import jax`** —
   9 tests silently skip (`contract_bands` ×7, `projection_lgemm` ×2);
   a fixture provably cannot fix it; needs a launcher-level or
   lazy-import solution. [FIX_p19_env_leak.md]
4. ~~**`lx test --help` is not a help flag**~~ **FIXED** by the
   harness-hardening lane at `0835d2b` on sandbox_v2
   `fix/lx-harness-traps-2026-08-10` (pushed, unmerged). `-h`/`--help` is
   now intercepted for every subcommand before dispatch and prints the help
   text, launching nothing; anything after `--` still belongs to the
   payload, so `lx test -- --help` remains the way to reach pytest's own
   help. Before/after measured 2026-08-10: at the pre-fix tree
   `lx test --help` built
   `... in_container.sh python3 -u -m pytest -q -n 4 --help
   /global/homes/j/jackm/software/lorrax_P/tests` — a real containerised
   whole-node run against the base module's tree — and at the fixed tree
   it prints help with zero `srun` lines and rc=0.
   [final-batch report, 2026-08-09; harness-hardening lane 2026-08-10]
5. ~~**`pytest` is not on the container PATH**~~ **FIXED** at the same
   commit. `lx run` translates a leading bare `pytest` to
   `python3 -m pytest` and says so. Measured: pre-fix,
   `lx run -G 1 pytest --version` → `in_container.sh: exec: pytest: not
   found`, exit 127, reported as 89; post-fix, the same command prints
   `pytest 9.0.3` and returns 0. (The 127→89 laundering was the same
   defect as row 26 and is fixed with it.) [final-batch report]
6. **Two malformed KNOWN_FAILURES table rows** — 6 pipes where the header
   declares 4 columns; lines 168 and 1089 at `d25d8d6a`. Pre-existing.
   [final-batch report]
7. ~~**`tests/tools/` duplicate of `tools/profile_gw_xprof.py`**~~ **FIXED**
   by deleting the whole `tests/tools/` directory on
   `chore/tools-dedupe-2026-08-10`. It held two files, not one, and both
   were orphan copies of their `tools/` equivalents. The dead-proof was
   re-run at `fc3ade69` and again at this branch's base `5eac02a5`, and
   still holds both times: `git grep -n "tests/tools"` now
   matches only this ledger row, `git grep -nE "from tests\.tools|import
   tests\.tools"` returns nothing, and pointing pytest straight at the
   directory under the census gate collects nothing, because neither file
   carries a `test_` prefix and nothing overrides `python_files`.
   `analyze_xprof_memory.py` was byte-identical to the canonical copy, and
   the canonical `tools/profile_gw_xprof.py` is now a strict superset of
   the orphan — the two comment lines recording the `gw_isdf` → `gw`
   rename were added at `83981f0c`. That commit removed the one reason the
   audit gave for not deleting blindly, which was that the orphan had
   briefly been the more current of the two. Both surviving `tools/`
   scripts compile, and `tools/profile_gw_xprof.py --help` prints its
   usage. This closes punch-list item P15.
   [COMPLETENESS_AUDIT.md orphan inventory, §P15]
8. **Ambient `lorrax_sandbox` PYTHONPATH entry from the `lx` harness** —
   sits last on `sys.path`, shadows nothing today; the no-precede gate
   guards the dangerous direction; removal is an lx/modulefile change
   (batched with the allocator owner-touch). [P21 report]
9. **Owed measurement legs** — six of them now, consolidated into this one
   row so the debt is legible in one place. The original three: the hbm80g
   fraction pair (ALLOCATOR_DECISION.md §11), the 4×4 `distributed_eigh`
   n=8192 control, and the `tr_p9`/`tr_p16` sigma traces. Three more from
   the same class, owed in reports and never in this row: the ScaLAPACK
   `ROUTE_BACKEND_BATCHED` leg (FIX_bsesetup.md O3 — it is the one eigh
   backend with a stacked FFI entry point, so it is where the batching
   seconds actually are; **blocked on transport**, because it needs a real
   multi-rank CPU mesh and both attempts ran under gloo, and there is no
   Perlmutter `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` recipe —
   `docs/dev/mpi_collectives.md` documents Frontera only); the
   crossing-window mask density; and the P=9 / P=16 sigma traces. All are
   refused for pool capacity, and none changes a standing conclusion.
   **Trigger: check this row whenever a quiet allocation appears; the
   orchestrator owns noticing.** Caveat learned 2026-08-09: "a quiet
   allocation" cannot be read off `lx status` while the exciton_bands
   never-exit hang (KNOWN_FAILURES 2026-08-09 amendment) is unfixed —
   hung steps render as busy GPUs, so verify with `sstat`/step logs
   before concluding the pool is actually in use. The 2026-08-09 batch
   pass ran the q-resolved rank check and the direction ladder from
   this class; the remaining four legs stay owed. [various;
   FIX_bsesetup.md O3; ASIDES_AUDIT.md §B4; OWED_LEGS_BATCH.md]
10. **Profiler session across the phdf5 collective close segfaults
    rank 0 on multi-node meshes** — worked around by the trace-section
    allowlist; root cause open. [SIGMA_SCALING.md §9; KNOWN_FAILURES
    2026-08-09 amendment]
11. ~~**`lx status` co-tenant rendering ambiguity**~~ **FIXED** at
    `0835d2b`, both symptoms, plus the CPU-allocation trap from
    BUILD_NOTES that belongs with them.
    (a) Co-tenants are no longer hidden: `(+1)` used to be the whole story
    a reader got about a second step on a node, and every co-tenant now
    gets its own indented line naming it.
    (b) `lx status --verify` answers the working-vs-hung question the
    table could not: two `sstat` samples six seconds apart, and a per-step
    delta of CPU time and disk I/O. Measured 2026-08-10 on one node
    holding two 1-GPU steps that render identically as `busy` — a
    `sleep 300` and a compute+write loop — the column reads
    `[idle: +0.0s cpu, +0B io]` and `[working: +6.0s cpu, +9.7kB io]`.
    Read the labels as evidence, not verdict: a collective deadlock spins
    in a barrier, so it burns CPU and moves no bytes (`cpu-only`), and so
    does a long kernel. The banner prints that caveat under the table.
    (c) The banner now names the partition kind. A CPU allocation used to
    render `1 nodes · 1/1 free`, the same shape as a GPU pool entry — the
    reading that sent a census leg to job 56530826 to die on `Invalid
    generic resource (gres) specification`. It now reads
    `1 nodes · cpu · 1/1 nodes free`, and a GPU launch pinned to a CPU
    allocation is refused `LX-CPUALLOC` by name before any `srun` runs
    (both arms measured 2026-08-10).
    [ops incidents, 2026-08-08; OWED_LEGS_BATCH.md; BUILD_NOTES 2026-08-09]
12. ~~**`windowed_exp_iEt` boundary alignment**~~ **FIXED** at `85112489`
    on `fix/windowed-exp-boundary-2026-08-10` (unmerged). The predicate
    is now `(E > E_min) & (E <= E_max)` — half-open, **closed at the top
    `(lo, hi]`**, the convention the clause-certification lane decided
    and recorded in the catalog's `bin_convention` field (branch
    aa24c099): a pane must contain its supremum, since every rule is
    built at max(Γ) over its pane. The flip also ends the mismatch
    inside Σ — `ppm_windows.window_mask_B_bounds` was already `(lo, hi]`,
    so the A and B sides used to send a threshold pole in opposite
    directions; the B side is untouched and the three comments that
    documented the mismatch as deliberate now document the agreement.
    As predicted it moves nothing numerically: no production caller
    passes `E_min`/`E_max` at all (every `build_G_tau` call site uses the
    band-identity `mask=` route), and with the pre-flip predicate
    reinstalled in the same tree the 19 non-boundary gates stay green
    while exactly the 4 boundary ones go red. Gates 21 → 23 cells: the
    old boundary twin flips direction, plus the boundary-pole twin this
    row asked for (a pole planted on each interior edge of an abutting
    cover is carried by the pane BELOW it, exactly once) and a cross-side
    twin that feeds `window_mask_B_bounds`' own bounds through the helper,
    so the two sides of Σ cannot drift apart again without a red.
    Regression sweep, both arms run: base `8bbff76d` 106 passed / 3
    failed, branch 108 passed / 3 failed, identical failure set, all
    three pre-existing (`test_g2_branch_window_tiles_are_frozen` P1b and
    two `test_pad_parity_gates` env rows, all in KNOWN_FAILURES).
    [WINDOWED_EXP_SWEEP.md; peer decision 2026-08-09]
13. **Wheel-content check** — the peer found `complex_laplace_width/*.npz`
    never declared in package-data: every wheel since that sweep shipped
    the catalog with none of its 15 tables, invisible from source
    checkouts (same failure class as the documented damped_line case).
    Sweep OUR package_data declarations vs shipped data files (services
    + main package) once; add a gate comparing wheel contents to a
    manifest. [peer fleet, 2026-08-09]
14. ~~**One `--write-eigs` confirmation leg at P=4 sharded is owed**~~
    **STRUCK — the leg was run 2026-08-09 and it COMPLETES.** P=4 sharded
    on `deck_si444` (`bse_si_test.in`, `--n-val 4 --n-cond 4 --n-eig 20
    --px 2 --py 2 --write-eigs`), tree
    `fix/multislice-cachekey-2026-08-09` @ `b7b02e36` (a descendant of
    `df361cd9`): **rc=0, `TOTAL (wall) 21.302 s`**, of which
    `bse.write_eigenvectors` is **0.042 s** — against the 24 minutes the
    original observation was killed at. `Wrote 20 eigenvectors to
    eigenvectors.h5`; the file verifies open with the right shapes —
    `exciton_data/eigenvalues (20,)`, `exciton_data/eigenvectors
    (1, 20, 64, 4, 4, 1, 2)` = (nQ, nevecs, nk, nc, nv, ns, re/im), all
    finite, plus the four BGW-style headers. So the hang WAS cured by
    `df361cd9`, as the row suspected; it is a confirmation, not a defect,
    and nothing is promoted to KNOWN_FAILURES. All four ranks reported
    `xla_compiles=0 hits=40 vetoed=0`, i.e. the write path is symmetric
    too. Evidence: `/pscratch/sd/j/jackm/perf_bse_0808/_reports/we_try3.log`;
    `~/lorrax_bse_perf_2026-08-08/FIX_multislice_cachekey.md` §5.
    [was FFT_DONATION_AUDIT.md:454-457; ASIDES_AUDIT.md §A2]
15. ~~**`exciton_bands` still carries the un-persistable host callback**~~
    **FIXED** by the exciton-bands feature lane: sink opened inside the
    scan body, three scalars per Q out as scan `ys`, worst-Q reduction on
    the host; the persistent-cache gate is green (real cache entry written,
    twin writes none). Landed on main in the `feat/exciton-bands-2026-08-09`
    merge, `824032b7`. Closes `FIX_warmcache.md` open item 3.
    [EXCITON_BANDS_FEATURES.md §3; was ASIDES_AUDIT.md §A3]
16. **`--matvec-kind` is parsed and silently ignored on `bse_jax`** —
    `src/bse/bse_jax.py:192` writes `data["matvec_kind"]` and nothing
    reads it; `bse_lanczos.py:243` says the selector is retired and builds
    `build_bse_stack_matvec` unconditionally. The only live consumer of
    the name is `absorption_haydock.py`, a different driver with its own
    argparse. It is worth a row rather than a shrug because it is an
    **A/B-voiding knob**: two arms differing only in this flag are
    structurally guaranteed to agree, so a "no difference" result means
    nothing. Fix is one of two: delete the flag from `bse_jax`'s CLI, or
    route it. [FIX_feastkey.md §5; ASIDES_AUDIT.md §A5]
17. **`tools/gen_input_reference.py` is broken, so `docs/input_reference.md`
    cannot be regenerated** — it dies on an `ast.literal_eval` of a
    non-literal `_DEFAULTS` entry (`ValueError: malformed node or string on
    line 1132`, raised from `main` at `:176`). **The line number is
    drifting**: it was a different entry on 2026-08-09, so at least one more
    non-literal default has been added since, and each one is another
    hand-edit to a file whose own header says it is generated. Fix sketch:
    make the walker fall back to `ast.unparse` for non-literal defaults
    instead of `literal_eval`. (Not P14 — that is `tools/env_audit.py`'s use
    of the removed `ast.Str`, a different instrument.) **2026-08-10
    amendment (docs lane): the fix is more than the fallback** — even
    repaired, regeneration would OVERWRITE four long hand-written
    entries (`mc_average_placement`, `restart_q_storage`,
    `write_restart_tensors`, `write_qsgw_datasets`) and re-add
    `use_ffi_io`/`slab_io`, removed 2026-08-06. The file is de facto
    hand-maintained; a real fix must merge generated stubs with
    hand-written prose (or declare the file hand-maintained and demote
    the generator to a drift-checker, which may be the honest design).
    The `[downfold]` section documents its own source of truth
    (`DOWNFOLD_DEFAULTS`) and is outside `_DEFAULTS` entirely.
    [compute-mode worker 2026-08-09; ASIDES_AUDIT.md §A6; docs lane
    2026-08-10]
18. **`chunks=` is silently dropped on every SlabIO write** — so no file
    this transport creates is chunked, including all four `sigma_mnk.h5`
    datasets and the `zeta_q_G` store. The tree is honest about the
    mechanism (`src/file_io/_slab_io_ffi.py:1751` and `:1856` both say
    `chunks=` cannot be honoured by this transport at all, and the warning
    was demoted to once-per-file when that landed) but the **consequence**
    is recorded nowhere a reader looking for open issues would find it,
    which is what this row fixes. Honouring it needs chunk dims plumbed
    through `lrx_phdf5_ensure_dataset`, which takes only name/shape/dtype
    today — a real but bounded FFI-signature change, and one to **sequence
    with the FFI build-contract gates** rather than do casually.
    [SlabIO-attrs worker 2026-08-09; ASIDES_AUDIT.md §A8]
19. **Read the `Plan.batched` bit-identity gate and record what it actually
    asserts** — this is the cheap half of a two-part item. `FIX_bsesetup.md`
    O1 measured `Plan.batched` on `ROUTE_SCAN` over cuSOLVERMp diverging
    from repeated `Plan.__call__`: 3.33e-16 Ry on the eigenvalues, 4.71e-14
    on ψ, at rank 64 on a 2×2 CUDA mesh. `plan.py:398-402` says a private
    `_route` override exists precisely so a batched-vs-serial bit-identity
    gate can run both routes on one set of operands. Nobody has gone to
    look at what that gate asserts at this shape — and **a gate claiming
    bit-identity where it does not hold is a false green**, which is the
    sharp half of the finding. This row is only "read it and write down the
    answer". **The NUMERICS RULING is the owner's** — does the batched route
    get to differ at 3e-16? — and does not belong in this file.
    [FIX_bsesetup.md §6 O1; ASIDES_AUDIT.md §A9]
20. **`lx test` is unusable whenever the pool is full** — it demands a whole
    node (`-G 4`) and refuses with `LX-POOLFULL`; the pool has been at
    0 nodes with 4 free GPUs for long stretches. Real cost, not a nit: it is
    why several censuses this campaign ran through `lx run -G 1 --wait`
    instead of the sanctioned harness, which is a different instrument.
    Sibling of rows 4-6 from the same "listed, not chased" section.
    [final-batch report 2026-08-09; ASIDES_AUDIT.md §A12]
21. **`src/psp/finite_q_head_interp.py` is an 875-line orphan** — no
    importer anywhere in the tree, and it depends on `common.chi_sos`,
    which does not exist, so it cannot even be imported. Found by the
    exciton-bands lane's LT-splitting derivation sweep. Dead-proof in
    EXCITON_BANDS_FEATURES.md §5; delete or resurrect is a judgement
    call, same class as the punch list's P7.
    [exciton-bands lane, 2026-08-09]
22. ~~**The two S-tensor builders disagree in representation**~~
    **SETTLED** by the head-tensor lane, landed `e06fc7de`: the
    Cartesian q²-coefficient is canonical (it is the one with readers,
    and the reader's arithmetic requires it); the Sternheimer builder
    was CONVERTED, not deleted — it now emits `S = ½B⁻¹HB⁻ᵀ` and
    stamps `s_tensor_convention`, keeping it as an independent
    cross-check. Documented in docs/theory/s-tensor-convention.md.
    [EXCITON_BANDS_FEATURES.md §5; HEAD_TENSOR_IMPL.md]

23. **Mesh-dependent zeroing of the top three ω slices in the ppm Σ
    stack** — pre-existing (predates the k-star completion fix, present
    at its base), does NOT reach `eqp0` per the fix lane's measurement,
    which is why it is here and not in KNOWN_FAILURES. Found during the
    k-star derivation's phase-collection sweep; details in
    FIX_sigma_kstar.md's unresolved list. Needs a root-cause pass when
    someone is next in `ppm_tau_kernel`/`ppm_accumulators`.
    [k-star fix lane, 2026-08-09]

24. ~~**`chi_from_dipole.py:22`'s documented `deltaE` sign is inverted**~~
    **FIXED** in the head-tensor landing `e06fc7de` (same file, same
    defect class, corrected alongside the tensor work).
    [LT_HEAD_PROBLEM.md appendix; HEAD_TENSOR_IMPL.md]

25. ~~**`lx run --wait` abandons its wait on a transient `lx_pool: timeout
    running scontrol`**~~ **FIXED** at `0835d2b`. Two changes: a probe that
    times out is retried (`LX_PROBE_RETRIES`, default 2 extra attempts),
    and a probe that still will not answer exits 3 rather than 2, so `lx`
    can tell "SLURM would not answer me" from "I asked and the answer was
    no". The former is now waited out inside the `--wait` budget — which
    also covers attach and probe, not just node selection as before — and
    if the budget runs out the refusal is `LX-ALLOCTIMEOUT`, whose text
    says explicitly that nothing has been asserted about the pool.
    Measured 2026-08-10 with a shim making the first `squeue` take 20 s:
    pre-fix, `lx run -G 1 --wait 120` refused `LX-ALLOCFAIL` after 15 s
    saying "no usable allocation could be attached to or created" while a
    4-node allocation with 2h48m left sat RUNNING; post-fix the same
    invocation absorbs three slow probes and launches, in 57 s.
    [OWED_LEGS_BATCH.md, 2026-08-09]
26. ~~**Two JAX-distributed steps co-placed on one node collide**~~
    **FIXED** at `0835d2b`, both halves.
    The laundering first, because it is what hid everything else: the
    remap test was `rc >= 90`, which swallowed not just the reserved
    90-99 refusal band but every code above it, so all signal deaths
    arrived as 89. It is now `90 <= rc <= 99`, and a signal death is
    reported intact and named (`payload died on signal 6 (exit 134):
    SIGABRT`). Measured: a step that `kill -ABRT`s itself returned 89
    pre-fix and 134 post-fix.
    The co-placement rule second: `lx` passes `--exclusive-distributed`
    for any P>1 launch and `lx_pool` skips nodes already hosting a
    multi-rank step, read from `scontrol show step`'s `Tasks=` field.
    Per-step coordination ports were the alternative and are not
    available to `lx`: JAX derives the coordinator port from
    SLURM_JOB_ID, so two steps in one allocation compute the same port
    and nothing inside either can avoid it — the same shape as the
    physical-GPU collision, and the launcher is again the only party who
    can see both. Measured with a live P=2 step on nid008277: pre-fix a
    second P=2 step was placed on that same node; post-fix the node is
    excluded by name and the leg goes elsewhere. `LX_ALLOW_COPLACEMENT=1`
    opts out. Narrow by construction — single-rank and whole-node
    launches never take this path. [OWED_LEGS_BATCH.md, 2026-08-09]
27. ~~**With two allocations live, `lx run --wait` sits on one while the
    other has 4/4 GPUs free**~~ **FIXED** at `0835d2b`, both halves.
    The waiter now polls every allocation that could satisfy the launch —
    filtered by the same size and kind rules `resolve_allocation` already
    used, so nothing is considered that `lx` would not have been willing
    to choose — and reports which one it took. Measured 2026-08-10 with
    two live allocations: pre-fix, `-G 4 -n 4 --wait 30` sat on the full
    one for the whole budget and refused `LX-POOLFULL` while the other
    had three 4/4-free nodes; post-fix it says "using JID 56575336
    instead of 56578991 — it is the one with room" and launches in 6 s.
    `LX_SINGLE_ALLOC=1` restores the old behaviour.
    The `SLURM_JOBID` pin second: `get_allocation` now separates "the
    controller answered and does not know this job" from "the controller
    did not answer", and only the first is `LX-EXPIRED`. Measured with a
    shim failing the first `squeue -j`: pre-fix, a pinned allocation with
    3h01m left was reported `LX-EXPIRED: JID 56575336 is no longer a
    running allocation`; post-fix the failure is named, retried inside
    the budget, and the leg launches. [OWED_LEGS_BATCH.md, 2026-08-09]
28. ~~**`-G 0` does not buy a CPU-only slot on a full pool**~~ **FIXED**
    at `0835d2b`. `-G 0` is now a zero-GPU *container* step — which is
    what the fleet actually wants, since `--cpu` allocations carry no
    image and therefore no jax: `--gres=none` so the step does not
    inherit the job's GPUs, no `select_gpu.sh`, `MPICH_GPU_SUPPORT_ENABLED=0`
    for the reason the `--cpu` path sets it, one rank per node instead of
    the `nodes × 0 = 0` that would have built `srun -n 0`, and the pool
    charges it 0 GPUs so it is placed on any node with room, emptiest
    first. Measured 2026-08-10: pre-fix, `lx run -G 0` was refused
    `LX-POOLFULL — no node in JID 56575336 has 0 GPU(s) free` because it
    fell through to the wholly-idle-node branch; post-fix it runs, rc=0
    in 3 s, with `CUDA_VISIBLE_DEVICES` unset in the payload. Set
    `JAX_PLATFORMS=cpu` yourself if the payload uses JAX.
    [OWED_LEGS_BATCH.md, 2026-08-09]

29. **Two committed `dipole.h5` fixtures have no deck that reproduces
    them** — gnppm_debug and cohsex_debug: no in-tree deck produces
    their committed shapes at all (found during the dZ-None fixture
    comparison; si's committed fixture also came from
    `cohsex_si_test.in`, not the `_fast` deck, so re-cuts must use the
    right deck). Until decks exist, those two fixtures can never be
    regenerated — owner input needed on where their generating decks
    live or whether to retire them. [FIX_dz_none_dipole.md §7]

31. ~~**`lx release --all` cancels the SHARED pool, not just your own
    allocations**~~ — measured cost 2026-08-09 night: one lane's cleanup
    script cancelled pool 56554959 and killed another lane's 23-minute
    leg (restored as 56555953). **FIXED** at `0835d2b` on sandbox_v2
    `fix/lx-harness-traps-2026-08-10` (pushed, unmerged).
    The root cause was that "this tool created it" was one step too weak
    a test. The created-ledger is a single file, `~/.lorrax/created.jsonl`,
    in a `$HOME` every lane shares, so every `lx` that ever allocated
    wrote into it and `--all` meant "everything any lane ever created".
    Ledger rows now record the creating **agent**, `--all` reaches only
    that agent's allocations, and anything else is listed, left alone, and
    reachable only by naming its jobid to `--include-pool <jid>` — naming
    it is the confirmation, and the cancellation is announced in red.
    Measured 2026-08-10 both ways, with `scancel` shimmed so the shared
    pool was never at risk: pre-fix, an agent that had created nothing ran
    `lx release --all` and `scancel 56575336` — the live shared pool — was
    invoked; post-fix the same call cancels nothing and prints the pool as
    "NOT cancelling ... this agent did not create". The positive control
    matters as much: an agent that HAD created two allocations released
    exactly those two and left both live `lx-alloc-jackm` pools running.
    `_reconcile` got the same scoping, since it could cancel another
    lane's seconds-old allocation by the same reasoning.
    **The BUILD_NOTES rule stands until the branch is deployed** — the
    fix is not in `~/lx_deploy` yet. [refreeze lane incident, 2026-08-09]

32. ~~**`bse_jax --write-eigs` crashes on any deck whose band window
    SNAPS**~~ **FIXED** at `d71f99d0` on
    `fix/writeeigs-snap-2026-08-09` (rebased onto `origin/main`
    `d9d418db`), 2026-08-09. The loader's resolved geometry is now the
    only authority on the window: `_preview_lanczos` reads
    `data['n_val']`/`data['n_cond']` on both its branches, and
    `_load_ring_subset` publishes those four keys the way the sharded
    loader already did, so the 1-device path cannot report its window
    differently. Gate, P=4 on the Si deck `bse_si_test.in` (`--n-val 4
    --n-cond 4 --n-eig 20 --px 2 --py 2 --write-eigs`), shared pool
    56555953: the snapping arm (`4 → 8` conduction, default
    `--band-degeneracy snap`) is **rc=0**, and `eigenvectors.h5`
    verifies with the POST-snap shapes — `nc=8 nv=4`, `eigenvectors
    (1, 20, 64, 8, 4, 1, 2)`, `bse_hamiltonian_size 2048`, all finite,
    per-state norm `1.000000000`. The non-snapping control
    (`--band-degeneracy off`, the 4/4 window of the row-14 recipe) is
    **rc=0 and BYTE-IDENTICAL** between the branch and `origin/main`
    (md5 `43354d6e…`). `bse.absorption_eigvecs` then runs end to end on
    the snapped file for the first time (rc=0), reading `nc=8, nv=4`
    back out and slicing the dipole over exactly those bands — the
    thing this row said had never been possible.
    **The row understated the blast radius by one driver.** The sweep
    for the same pre-snap assumption found `absorption_haydock`, which
    sliced `dipole.h5` with its own CLI `n_val`/`n_cond` after the
    loader had snapped. That one does *not* crash: measured on the same
    deck, `origin/main` builds a Hamiltonian over `8 cond × 4 val` and a
    dipole seed over `nc=4`, zero-padding the four conduction slots the
    BSE window is really using — silently wrong absorption. On the
    branch both read 8. Everything else in the sweep already reads the
    loader's counts (`bse_feast`, `bse_kpm`, `bse_w_exact`,
    `davidson_absorption`, `exciton_bands`, `bse_lanczos`), and
    `absorption_eigvecs` reads `nc`/`nv` back out of the file.
    New gate `tests/test_bse_eigenvector_window.py`: 9 cells, 7 red at
    `origin/main`, including two AST cells that read the two drivers'
    source — the defect is invisible on a non-snapping deck, so the gate
    cannot wait for a deck that snaps. Evidence:
    `/pscratch/sd/j/jackm/weigs_0809/_reports/`.
    [FIX_absorption_conjugation.md, 2026-08-09]

33. **`absorption_haydock` has no one-writer gate** — every rank reaches
    `h5py.File(..., "w")` on the same `--out-prefix` paths; the file has
    no `jax.process_index()` anywhere. MEASURED at P=4 on the Si deck,
    2026-08-09: the driver logs `[haydock] wrote hay_snap.h5` four
    times, and the step is a coin flip — eight legs, four per tree,
    failed **once on each tree** with `OSError: Unable to synchronously
    create file (file signature not found)` and passed otherwise. That
    symmetry is what makes it pre-existing rather than this lane's: it
    is QUALITY_PATTERNS #7, exactly as
    `bse_io.write_eigenvectors_stream` documents having fixed for
    itself. The physics finishes before the writer runs, so the failure
    costs the whole run for a file-creation race at the end.
    Fix: gate the writer block on rank 0, with the ungated-collective
    note the eigenvector writer already carries.
    [writeeigs-snap lane, 2026-08-09]

34. **C₃-equivalent off-grid Q points disagree on the `--extra-q` path**
    — measured by the LT ladder (head OFF, so not the head machinery):
    two Γ→M points 120.0001° apart differ by 1.2–11.8 meV at small t,
    growing to 109 meV at t=0.42, on the MoS₂ 3×3 slab deck. Plausible
    benign mechanism: the interpolated evaluator's model form-factor
    fit at off-grid Q is not symmetry-constrained, so equivalent
    directions accumulate independent fit residuals — but that is a
    hypothesis, not a measurement. Measure on a denser grid (residual
    should shrink) before treating as a defect; the standing rule says
    today's diffs are the suspect pool first, and this path predates
    today. NOTE: this reading supersedes the earlier "1.2–11.8 meV
    trigonal warping" interpretation in EXCITON_BANDS_FEATURES §6.1 —
    those two directions are symmetry-equivalent, so the spread is
    residual, not physics. [LT_LADDER_ACROSS_THE_CELL_2026-08-10.md]

35. **NOT A DEFECT — `-G=4` is not a parsing quirk.** Recorded because
    the opposite was believed for a day and cost eleven OOM-killed
    attempts. BUILD_NOTES 2026-08-09 says "`-G=4` gets you a QUARTER of
    the memory (59 GB), not four times it", read as an `lx` argument
    parsing bug. Measured 2026-08-10: `-G=4`, `-G 4`, `-G4` and
    `--gpus=4` all parse to the same thing and all emit
    `--gres=gpu:4` with the job name `lx-Xg4` — argparse splits on `=`
    for short options exactly as it does for long ones. The real
    mechanism is the GEOMETRY: `-G 4` is four ranks of one GPU each,
    because `select_gpu.sh` pins one device per rank, so each rank sees
    one device's memory. One process across all four devices is
    `-G 4 -n 1`, which `lx` already supported and already announced. The
    converse note was the one missing, and `0835d2b` adds it: every
    multi-GPU launch now prints its rank×device geometry and points at
    `-G 4 -n 1`. Nothing was changed in the parser, and refusing the `=`
    form by name would have been wrong — it works.
    [BUILD_NOTES 2026-08-09; ASIDES_AUDIT.md §A13; harness-hardening lane]
36. **SPEC, not a defect — `lx` should hint when it sees the serial
    anti-pattern.** The fleet's median duty cycle is 0.41 and 115 of 152
    evidence directories are strictly serial (`docs/warm_worker.md`), so
    the cheapest guardrail is one line of output: on a second sequential
    `lx run` from the same workdir within N minutes with no overlap, print
    `independent legs? try lx batch`. Specified, deliberately not
    implemented here, for the lane building `lx batch` —
    `docs/SPEC_lx_fanout_hint.md` on sandbox_v2
    `docs/agent-efficiency-2026-08-10`. Non-behavioural by construction:
    stderr only, never an exit code, never a refusal.
    [AGENT_PREAMBLE.md efficiency doctrine rule 1]

37. **`lx batch` v2 improvements, measured-and-specified by v1** — three
    items from the first live batch (evidence lxbatch_0810): (a) probe
    the pool ONCE per batch, not per leg (per-leg launcher overhead
    7.8→42.9 s at P=4 — the control plane, not GPUs, became binding);
    (b) claim devices once for the whole batch instead of N launchers
    racing the claim/step window (the actual mechanism of the six
    unplaced legs — capacity and device decisions read different
    snapshots inside the 180 s grace); (c) the claim refusal prints
    the claim table it just read, so the next occurrence self-diagnoses.
    [batch lane report, 2026-08-10]

39. ~~**`vq_interp.refit_vq` cannot run at P>1 unless the basis size happens
    to defeat sharding**~~ **FIXED** by the refit-sharding + cert-grade lane
    on `fix/refit-shard-and-cert-grade-2026-08-11` (pushed, unmerged): the
    fetch is `_to_host` (= `common.collectives.gather_to_host`), whose three
    arms cover all three layouts — `device_get` when fully addressable (P=1),
    `addressable_data(0)` when replicated (the odd-μ arm, still no
    collective), `process_allgather(tiled=True)` when genuinely sharded. The
    mechanism is now written down where it bit: **`Array._value` serves a
    fully REPLICATED array out of the local shard before it ever reaches the
    addressability check**, which is exactly why odd n_μ worked and even n_μ
    did not, and why no amount of P=4 coverage on the μ=191 child could have
    caught it. **Red twin, and it is an even-n_μ FOUR-PROCESS cell**:
    `tests/test_refit_vq_shard_p4.py` drives `tests/_refit_shard_twin.py`
    through `tests/mesh_launch.py`; on the pre-fix tree it dies at
    `tree_base/src/bse/vq_interp.py:2778` with the production traceback and
    on the post-fix tree it is 6 passed, both at four real A100s in four
    processes (`refitshard_0811/_logs/twin_pre.log`, `twin_post.log`). The
    twin also refuses to bank a green unless the even arm's ζ'(G) box really
    was non-addressable and non-replicated, so it cannot go vacuously green
    the way the old coverage did. **Note for anyone reproducing on a bench:
    the μ sharding survives to ζ on XLA:GPU and does NOT on XLA:CPU** — an
    emulated CPU mesh replicates at the Cholesky and shows nothing, which is
    a second reason this needed a real four-GPU leg. One import was narrowed
    in passing (`compute_wfns_fi` now imported in the htransform branch that
    uses it, not at function scope), because dragging the communicator stack
    and its required-FFI gate into the `"stored"` leg is what made the twin
    unrunnable anywhere but a compute node. Original report follows.
    `src/bse/vq_interp.py:2778` did a bare
    `jax.device_get(ztG_box[:, jnp.asarray(fi)])` on a globally sharded
    array and dies with `RuntimeError: Fetching value for jax.Array that
    spans non-addressable (non process local) devices`. It is the sibling
    of the `refit_prepare` fix on `feat/xbands-bse-window-refit-2026-08-10`
    (`_to_host`, not `device_get`) and was missed because **the failure
    depends on n_μ parity**: the μ=191 child is odd, so `sharding_fit`
    declines to shard it (`191 % 2 != 0` → replicated), the array is fully
    addressable and the call accidentally works; the μ=960 parent is even,
    shards for real, and the call fails. Every P=4 leg the refit path has
    ever had was the odd-μ child arm, so the whole existing coverage is
    blind to it by construction. Cost paid: this lane's parent control had
    to be taken at P=1. Fix is one call site plus a cell that exercises
    `refit_vq` at P>1 on an EVEN n_μ. Not physics — the P=1 arithmetic is
    unaffected — but it blocks a four-GPU control, so it is not cosmetic.
    [qsign re-measurement lane, 2026-08-11;
    `/pscratch/sd/j/jackm/qsign_recut_0811/_logs/xb_ctl_parent.log`;
    `tests/known_failures/2026-08-11-qsign-recut-verdicts.md` §4]

40. **An in-leg mesh assert that knows only one driver's banner produces
    FALSE REFUSALS** — the three drivers announce their shape in three
    dialects (`exciton_bands`: `[dist] jax.device_count()=4 ...
    mesh_xy.shape={...}`; `downfold_cli`: `mesh {'x': 2, 'y': 2} on 4
    device(s), 4 process(es)`; `bse_jax`: `This is rank 0 of 4, and it
    addresses 1 of the 4 devices`). A wrapper grepping for one of them
    killed two `bse_jax` legs that had already **completed their physics**,
    reporting exit 95 as if the shape were wrong. Two lessons worth
    carrying, both cheap: parse all three dialects, and **never let a shape
    assert overwrite a non-zero payload rc** — reporting 95 over a real
    traceback hides the defect underneath, which is exactly how row 39
    nearly went unnoticed. Reference implementation:
    `/pscratch/sd/j/jackm/qsign_recut_0811/inleg.sh`.
    [qsign re-measurement lane, 2026-08-11]

41. **`ssh -O exit perlmutter` kills every backgrounded LAUNCHER on the login
    node but NOT the Slurm step it launched — which is worse than killing
    both, and `AGENT_PREAMBLE` recommends the command without saying so.** The
    certificate row says "a working `ssh` is not evidence — `ControlPersist`
    answers past expiry; re-probe with `ssh -O exit perlmutter`". That is
    correct about certificates and expensive about work: tearing down the
    ControlMaster tears down every multiplexed session with it, and `nohup
    ... &` does **not** survive it. What is left behind is a **half-observed
    run**: the `lx batch` wrapper is gone, so no rc file, no `summary.json` and
    no completion signal ever appear, while the `srun` step keeps running on
    the compute node and finishes normally. Measured here — the wrapper for a
    `bse.exciton_bands` control leg died at 01:19, `squeue --me -s` showed only
    `.extern`, and the step nonetheless completed at 976 s and wrote its full
    certification block to the log. **The trap is what that invites**: reading
    "no rc file" as "the leg died" and relaunching, which is exactly what this
    lane did — and the duplicate **overwrote the original's
    `<logdir>/<id>.log` from byte 16461 onward**, because `lx batch` names the
    per-leg log after the leg id and a relaunch under the same id reuses it.
    The original's log survives **only** because this lane had copied it aside
    before the duplicate finished writing. The overwrite is confirmed, not
    near-missed: the live file now carries step `lx-Xg1-012101` (973 s) and the
    copy carries `lx-Xg1-010647` (976 s). It cost nothing here — the two runs
    are the same deterministic deck and agree to five decimals on all four Q,
    so the accident produced a free independent replicate — but a relaunch that
    overwrites the log of a run you have not read yet destroys the evidence.
    **Mitigations:** launch with `setsid nohup ... < /dev/null &` so the
    launcher leaves the ssh session's process group; re-probe the certificate
    with `ssh -o ControlPath=none` rather than evicting the master; and before
    relaunching anything that "died", check the **log** and `sacct` for the
    step, not just the rc file. Give a relaunch a distinct leg id so it cannot
    overwrite the original's log. **The `AGENT_PREAMBLE` line is now
    CORRECTED** — refit-sharding + cert-grade lane,
    `fix/refit-shard-and-cert-grade-2026-08-11` (pushed, unmerged): the
    certificate row's misleading clause is replaced IN PLACE with `ssh -o
    ControlPath=none perlmutter true` plus a named refusal of `ssh -O exit`
    and the reason, so the page gains no line. The rest of this row stands:
    the file could only ever carry the one-line rule, and the incident is
    here.
    [qsign re-measurement lane, 2026-08-11; corrected by the refit-sharding
    lane, 2026-08-11]
42. **`tools/gen_input_reference.py` has not been able to run for some
    time, so `docs/input_reference.md` is hand-maintained by neglect.** It
    refuses on key drift against `gw_config._DEFAULTS` and there are seven:
    five keys with no one-liner (`mc_average_placement`,
    `mc_average_placement_vcoul`, `restart_q_storage`, `sc_eigh`,
    `write_restart_tensors`) and two one-liners for keys that no longer
    exist (`slab_io`, `use_ffi_io`). Verified PRE-EXISTING on `origin/main`
    `fa86c6b8` — the generator refuses identically on a clean base
    worktree — so it is nobody's branch. The consequence is that the page
    the register calls "generated from the parser, so it is complete" is
    neither: `zeta_nband`'s row had to be hand-written in the generator's
    own format, which is exactly the drift the generator exists to prevent.
    Fixing it is seven one-line strings plus one regeneration, but the
    regeneration rewrites the whole file and would collide with every lane
    that has a docs edit in flight, so it wants its own bounded pass rather
    than a ride-along. [ζ-solve + `zeta_nband` lane, 2026-08-11]

42. **The 2-D slab Coulomb kernel exists twice, and unifying it is NOT
    free.** `bse.vq_interp.v_slab_on_set(kind="slab")` and
    `vcoul.Slab2D._v_bare_per_q` compute the same Ismail-Beigi truncation
    from the same inputs, but in different arithmetic order — the service
    spells it `(8π/K²)·f2d * fact` with `fact = 1/Ω` and is bit-compared
    against BerkeleyGW's pre-port table, while `vq_interp` spells it
    `8π/K² · f2d / celvol`. Those differ in the last ulp, and the caller's
    own on-grid gate (`makeVq_vs_disk`, 5e-6) sits above a path whose bulk
    sibling lands at 3.3e-14, so swapping one for the other is a numerics
    change, not a refactor. `vq_interp` additionally needs `slab_sr` /
    `slab_lr` (the Gaussian SR/LR split the b26p long-range model is built
    on), which the service has no equivalent for **on an explicit Miller
    set** — `vcoul.minibz._minibz_kernel_bare` carries those kinds but takes
    a Cartesian shift plus δq draws, not a `(3, nG)` Miller table. So the
    honest unification is a new service entry point (`v` on an explicit
    Miller set, all four kinds, one arithmetic order) plus a re-pin of both
    callers' references — an owner-scoped change to a bit-pinned surface,
    not a cleanup lane's. Noted here so the next lane does not either
    re-derive the finding or make the swap silently. The verdict is also
    written into `v_slab_on_set`'s own docstring, where a lane reaching for
    it will meet it. The refit's BULK kernel already goes through the door
    (`vq_interp.make_v_on_set` -> `gw.compute_vcoul.compute_v_q_per_G`), so
    this is the only Coulomb arithmetic left local to `src/bse/`.
    [xbands cleanup lane, 2026-08-11]

43. **`bse_densify` reads the deck twice, two ways, and only one of them
    can be reused.** `_read_lorrax_input_quietly` and `_resolve_bse_k_grid`
    both wrap `gw.gw_config.read_lorrax_input` in the same
    "a config read must never crash a load" try/except, and the second's
    docstring even says so ("Same tolerance as `_resolve_bse_k_grid`'s own
    read"). They are NOT interchangeable: `_resolve_bse_k_grid` keeps
    `_parse_grid_spec` INSIDE its `try`, so a malformed three-integer
    `bse_k_grid` value returns `None` (feature off) rather than raising,
    and the two print different messages. Collapsing them is therefore a
    behaviour change on a deck-error path — small, but a real one — and it
    was left alone by a lane whose mandate was behaviour-preserving. Whoever
    takes it should decide first whether a malformed `bse_k_grid` ought to
    REFUSE (which is what `2026-08-10-unknown-deck-key-refusal-spec.md`
    argues for every deck key) rather than fall back to defaults; the dedupe
    then follows from that ruling instead of preserving an accident.
    [xbands cleanup lane, 2026-08-11]
44. **The f-shoulder gate RUNS on the refit path but does not LOG there.**
    `bse.vq_interp.refit_vq` calls `compute_wfns_fi` without a `log_fn`, and
    `refit_ongrid_null` calls `refit_vq` with a silenced one, so the
    `[gate] f-shoulder over the RETURNED bands …` line — the dead-set census
    and the `min_k |f|/max|f|` a lane needs to choose a guard count — appears
    only for the driver's own BSE-window call and never for the refit's. The
    gate itself is fine and fires there (measured: `twowin_0811/_logs/
    nA_g0trip.log` refuses by name at band 50 with 16 zero slots), so this is
    observability, not correctness. The fix is not "pass `log_fn`": that would
    print the same three lines once per off-grid Q, which on a 129-point path
    is 129 copies. It wants a print-once, at `refit_prepare` time, of the
    census over the window the refit will return — which is computable there
    from `enk_sigma` alone. [two-window lane, 2026-08-11]
45. **`lx batch` co-placement: do not start a second batch against a pool that
    already has 4-GPU legs in flight.** Three legs of this lane died in the
    JAX coordination service (`ABORTED: … tried to connect with a different
    incarnation`, rc 143/134) at the two moments a second `lx batch` was fired
    against allocation 56644791 while the first still had 4-GPU legs running —
    the same failure the zsolve lane recorded for `-P 4` on one node, reached
    by a different route. `-P 2` inside ONE batch was fine for eleven legs;
    two concurrent batches at `-P 2` and `-P 1` was not. The tell is rc
    143/134 at ~30–45 s with `RegisterTask` errors in the log, and the legs
    measure nothing. [two-window lane, 2026-08-11]
46. **Killing a waiting `lx batch`'s `launch.sh` and its current `lx run`
    leaves the `lx batch` process itself alive, and it fires the next leg.**
    A lane that decides mid-wait to move its batch onto a different
    allocation naturally kills what `pgrep -af "launch.sh"` shows it: the
    wrapper, and the `lx run` currently queued for capacity. Neither is the
    batch. `lx batch` is its own long-lived `python3 .../lx batch ...`
    process, it survives both, and when a node frees it launches leg 2 of a
    manifest whose leg 1 the lane has already re-run elsewhere — a duplicate
    leg against a pool the lane no longer thinks it is using, and (row 45) a
    second concurrent batch on that pool. Found the honest way: 25 minutes
    after the relaunch, `ps -eo pid,cmd | grep <workspace>` still showed the
    orphaned `lx batch` plus a queued `lx run` for the arm that had already
    reported. Kill the `lx batch` PID first and the children after, and
    verify with `ps` against the WORKSPACE PATH rather than against
    `launch.sh` — the wrapper's name is the one thing every process in the
    tree does not share. [gnppmfft lane, 2026-08-11]
47. **A base ARM needs its own LAUNCHER, not its own manifest row: `lx`
    resolves the source tree from the AMBIENT `LORRAX_CHECKOUT`.** A batch
    leg carrying `"env": {"LORRAX_CHECKOUT": "<basetree>"}` gets that value
    inside the container — but `lx run` has already chosen which `src` to put
    on the path, from the environment the batch was FIRED in, and a lane's
    `env.sh` normally points that at the fix tree. The result is an A/B whose
    two arms run the same code. It is visible in the leg's own log and only
    there, as two adjacent lines that disagree: `[inleg] git HEAD:
    f09bec97…` (the per-leg env, so the base commit) next to `[lx] source
    tree: …/**tree**/src [LORRAX_CHECKOUT]` (the ambient one, so the fix
    tree). Measured here: the base arm of an all-64-q tile null printed the
    fix arm's 64 numbers to every digit. The fix is a second launcher that
    exports `LORRAX_CHECKOUT=<basetree>` before `lx batch`; the durable guard
    is to print the imported module's own `__file__` and a fix-marker
    (`hasattr(v, 'zeta_r_to_sphere_q')`) from inside every leg, so the two
    instruments cannot disagree silently. [sixth-wall lane, 2026-08-11]
48. **Editing an in-flight `inleg.sh`/`launch.sh` corrupts the running leg.**
    bash reads a script incrementally, so patching the wrapper while a leg is
    executing it makes the shell resume at a byte offset that is no longer a
    statement boundary: measured here as `inleg.sh: line 25: syntax error near
    unexpected token '('` on a leg whose pytest had already finished and
    reported (15 failed / 1203 passed), turning a green gate into rc 2. The
    file was syntactically valid before and after — `bash -n` passes on the
    patched copy. Patch wrappers between waves, never during one, and judge a
    leg that dies this way by its artefacts. [sixth-wall lane, 2026-08-11]
49. **The `lorrax_*` modulefiles set `XLA_PYTHON_CLIENT_ALLOCATOR=platform`,
    which is NOT the campaign default, and a long per-Q refit run dies of it.**
    `modulefiles/lorrax_J070/*.lua:189` does `setenv(
    "XLA_PYTHON_CLIENT_ALLOCATOR", "platform")` and repeats it as a
    `--env=` on the shifter line; the startup banner already says so in as
    many words — *"NOT LORRAX's canonical pair, which is preallocate=false
    with the allocator left unset (BFC); a caller overrode it"* — and nothing
    reads that line. `platform` is cudaMalloc/cudaFree per buffer with no
    pooling, so a driver that allocates and frees a large tensor once per Q
    fragments. Measured: `exciton_bands --vq-mode refit --q-per-segment 16`
    on the μ=2988 parent ran **60 off-grid Q and then died**
    `RESOURCE_EXHAUSTED: Failed to allocate request for 15.41GiB` inside
    `refit_vq`'s `cq_and_x`, on a 40 GB card at `mem_fraction 0.85` — i.e.
    with ~34 GB nominally available and the same allocation having succeeded
    sixty times. Per-leg `env` `XLA_PYTHON_CLIENT_ALLOCATOR=default` reaches
    it (lx applies the leg env inside the container, after the module).
    Whether to change the modulefile is an owner call — it moves every
    wall-time on the fleet, and every timing in the corpus was taken under
    `platform`. [sixth-wall lane, 2026-08-11]
50. **A multi-rank leg SMALLER than a whole node is not device-placed, so it
    collides with the one-GPU co-tenants it is sharing the node with.**
    `lx` places one-GPU legs on a free device and says so; multi-rank legs it
    leaves alone, which is correct when the leg owns the node and wrong when
    it does not. Measured 2026-08-11: with the pool fragmented (four nodes,
    8/16 GPUs free, no node with four free), the Si 4x4x4 production COHSEX
    deck was launched at `-N 2 -G 2 -n 4` — a legitimate P=4 geometry that
    fits the fragmentation — and every rank 0 logged
    `[select_gpu] local rank 0 -> CUDA_VISIBLE_DEVICES=0 (from
    CUDA_VISIBLE_DEVICES)` on BOTH nodes, i.e. onto the same physical device
    a placed one-GPU co-tenant already held. It died
    `RESOURCE_EXHAUSTED: Failed to allocate request for 19.42GiB` inside
    `isdf.core.z_q_from_psi_sm` — a traceback that names the zeta fit and
    says nothing about placement, which is the same tell the stale-`lx`
    incident left. So `-N 1 -G 4` is not merely the convention for a
    production-deck leg, it is the only geometry that is safe under
    co-tenancy, and a fragmented pool means WAITING rather than reshaping.
    Evidence `/pscratch/sd/j/jackm/ioaudit_0811/logs_cotenancy_oom_G2/`.
    [io-dead-code-audit lane, 2026-08-11]

## Fixed (strike-in-place graveyard — newest first)

- **Row 38** (`centroid/pivoted_cholesky.py` imports `wfn_loader` at module
  scope for annotations that are never evaluated) — **TAKEN AND FIXED** on
  `feat/downfold-orbit-floor-2026-08-10`, 2026-08-10, branch pushed and NOT
  merged. Folded into that lane because it was already editing this file
  (batch-small-items policy); it rides that lane's P=4 gate and cost no extra
  leg. The import is now under `if TYPE_CHECKING:` and the two `wfn:
  WfnLoader` annotations are quoted, so a type checker still resolves them and
  the run-time import edge is gone — which is the edge a clobbered
  `PYTHONPATH` on Perlmutter used to kill an h5py-less import of a module that
  does not need h5py. **Its sibling is deliberately UNTOUCHED and is not
  fixed**: `import symmetry_maps` two lines further down is also used only for
  the `sym: symmetry_maps.SymMaps` annotations, but it sits behind an
  `_services.ensure_on_path()` bootstrap whose removal is a different question
  from this one, and quietly widening a named row is how a small fix becomes
  an unreviewed one. Register it separately if it is wanted.

- Rows 4, 5, 11, 25, 26, 27, 28, 31 (the `lx` harness trap inventory) —
  fixed together at `0835d2b` on sandbox_v2
  `fix/lx-harness-traps-2026-08-10`, 2026-08-10, branch pushed and
  deliberately NOT merged. Struck in place above, each with its own
  before/after measurement. Two things worth carrying forward.
  **The branch is not deployed.** `~/lx_deploy` is a plain copy, not a
  checkout, so the live `lx` every lane runs is still `744591a` and every
  BUILD_NOTES interim rule — above all "never `lx release --all` while a
  shared pool is live" — stands until the owner deploys. Deploying is
  `cp` of two files from the branch; it is an owner action because it
  changes the tool under every running lane at once.
  **What makes these eight one commit rather than eight.** Six of the
  eight are the same mistake in different clothes: a check that cannot
  tell "I asked and the answer was no" from "I could not ask", or from
  "the answer was about something else". `>= 90` could not tell a refusal
  from a signal death; a timed-out `squeue` could not tell an empty pool
  from an unreachable controller; a created-ledger keyed on the tool
  could not tell one lane's allocation from another's; a `busy` column
  could not tell occupancy from progress. Where a distinction was
  missing, the fix was to make it and name both sides, which is why every
  one of them shows up as new text in a message rather than only as new
  behaviour.
  Regression evidence: the `srun` line this branch emits is BYTE-IDENTICAL
  to the pre-fix one for `-G 1`, `-G 4 -n 4`, `lx test` and `--cpu`
  (diffed with job name and nodelist normalised), and the normal-path
  smoke matrix is green — `run -G 1` rc=0; a P=4 GPU leg rc=0 with the
  four ranks on four distinct devices (`CUDA_VISIBLE_DEVICES` 0/1/2/3);
  `lx test tests/test_band_partition.py` 5 passed; `lx status`;
  `--wait 60` against a live allocation in 9 s; release-by-agent
  cancelling exactly its own two allocations.

- Row 30 (`bandstructure.htransform` exit-hang shape) — fixed at
  `80e9a319` on `chore/refreeze-and-htransform-2026-08-09`, 2026-08-09.
  The two-line pattern from FIX_exciton_exit_hang.md, applied verbatim:
  an outputs barrier and `wfn.close()` after the rank-0
  `bandstructure.dat` writer block, and `finalize_process(main())` in
  place of the bare `SystemExit`. Gate: one P=4 leg on the Si deck
  (`exb16.in`, own allocation 56555885) exits rc 0 in 18 s with
  `[runtime] process finalized` in the log, and `bandstructure.dat` is
  byte-identical to the pre-fix arm (md5 `e47351d7…`, 133 lines).
  ONE THING THE GATE FOUND that the row did not predict: the pre-fix arm
  on this deck **also** exits (rc 0, 24 s). The exciton hang needs a
  SECOND unordered collective to deadlock against — the `atexit`
  `H5Fclose` on a live phdf5 restart/zeta context — and this deck runs
  `restart = false` and opens none, so the `__del__` barrier is the only
  collective in flight and all four ranks reach it. The fix here is
  therefore PREVENTIVE on the deck it was measured on: the exposure is
  real (the loader is mesh-aware and unclosed, exactly as §6 of the
  report says) but it needs a restart-carrying deck to bite. Whoever
  wants the red twin for this driver should drive it with a deck that
  leaves a phdf5 context open at exit.

- Row 15 (exciton_bands un-persistable host callback) — fixed at
  `824032b7`, 2026-08-09, persistent-cache gate green.

- **`w_head_densify` is documented as a deck key and is not one** (found
  2026-08-10, measurement lane, densified-downfold experiment). Three places
  say it is a deck key — `bse_densify`'s module docstring,
  `resolve_w_head_densify`'s own docstring ("*else the deck's
  `w_head_densify` key*"), and
  `tests/known_failures/2026-08-10-w-densifier-head-interpolation.md`
  ("`w_head_densify = legacy` (deck key, or `--w-head-densify` on
  `exciton_bands`)"). It is absent from `gw_config`'s recognized-key set, so
  `read_lorrax_input` prints *"unrecognized deck key … ignored"* and drops
  it, and `resolve_w_head_densify(None, params)` then resolves to `c1`.
  Measured: a deck containing `w_head_densify = legacy` resolves to `c1`.
  **Consequence: the C1 branch's own A/B control arm cannot be selected from
  a deck at all** — only from the `exciton_bands` CLI flag, which does not
  reach `bse.bse_jax`'s `bse_k_grid` path. A lane that sets the key in a deck
  and reports a "legacy arm" has measured the C1 arm twice. Fix is one schema
  row (plus `w_head_gamma_cell`, same gap). This is also a live instance of
  the log-and-proceed hazard in
  `tests/known_failures/2026-08-10-unknown-deck-key-refusal-spec.md`: the
  warning was printed and the run proceeded on the wrong branch.

## `dipole.h5` provenance sanity check reads the RUN's band window as 5/5/128 (2026-08-10)

Every Si 6×6×6 GW run prints, twice:

    *** LORRAX SANITY FAILURE: .../dipole.h5 was generated from a DIFFERENT
    DFT solution or band window than this run (prov_nval: file=8 run=5;
    prov_ncond: file=52 run=5; prov_nband: file=60 run=128). ***

The **file** side is right — 8/52/60 is exactly the deck the `dipole.h5` was
generated from, minutes earlier, by `psp.get_dipole_mtxels -i <that same
deck>`. The **run** side is wrong: 5/5/128 are the `nval`/`ncond`/`nband`
defaults, so the comparison is against unresolved defaults rather than
against the run's actual window. The check therefore fires on a *correctly
matched* dipole file and would presumably stay silent on some genuinely
mismatched ones, which is the worse half.

Reproduced on three separate runs and on both windows: the original
reference (`si666_ref_0810/gw_666.log`), and both arms of the band-window
A/B (`bandwin666_0810/_logs/gw_{C,F}_*.log`, `nband` 60 and 68 — the file
side tracks the deck, the run side stays 5/5/128 in all of them). It
predates `fix/band-window-degeneracy-closure-2026-08-11` and is not caused
by it; it fires identically in both arms of that A/B, so it does not affect
the comparison. Not investigated further — noted so the next lane does not
spend a leg on a false positive, or trust a true negative.

## 47. Every GN-PPM timing table on disk before 2026-08-11 was taken under the `platform` allocator, and `sigma.tau.dispatch` is not comparable across the two arms (2026-08-11)

`symgate444_0810/arm{A,B}/gw_*.log`, `si666_ref_0810/gw_666.log`,
`si666_ref_0810/symgate/gw_fullbz.log` and the `si_gnppm_0809` family all
print `XLA allocator: platform  preallocate: off  mem_fraction: 0.85`. They
are the only Si 4×4×4 and Si 6×6×6 GN-PPM decompositions in the record, and a
lane reaching for "the last measured sigma numbers" will land on one of them.

The row that moves is `sigma.tau.dispatch`, and it moves by two orders of
magnitude. Under `platform` the dispatch call blocks on `cuMemAlloc`, so it
absorbs the tau kernel's device wall; under BFC it is an async submit and the
wait lands in `sigma.tau.d2h_wait` instead. Same deck (Si 6×6×6), same P=4,
same mesh:

| row | `platform` (`gw_666.log`) | BFC@0.85 (`gnppmdecomp_0811/U3`) |
|---|---|---|
| `sigma.tau.dispatch` | 41.277 s | **0.189 s** |
| `sigma.tau.d2h_wait` | 0.004 s | **29.902 s** |
| driver wall | 96.317 s | 95.919 s |

**The wall does not move** — 0.4 %, inside this deck's own noise. So at this
scale the allocator relocates 41 s of attribution and buys nothing, because
the device is genuinely busy. That matters because `SIGMA_PPM_CAMPAIGN.md`
reads the same collapse on its own deck (MoS2 6×6, μ=1496) as "allocator
churn masquerading as device work" worth "40 % of the Sigma stage" — true
there, and **not** a fleet-wide 40 %. Whether the collapse is a saving or a
relabelling depends on whether the device was idle, which is a per-deck
question.

Practical rule: never compare a `sigma.tau.dispatch` or `d2h_wait` number
across two logs without checking the `XLA allocator:` line in both, and read
`host_accum`/`d2h_wait` as the device wall on the BFC arm
(`src/gw/ppm_accumulators.py:425`). Full ladder and adjudication:
`tests/known_failures/2026-08-11-gnppm-sigma-performance-claims-adjudicated.md`.
