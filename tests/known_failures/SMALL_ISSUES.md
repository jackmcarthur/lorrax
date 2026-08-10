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
7. **`tests/tools/` duplicate of `tools/profile_gw_xprof.py`** — orphan
   copy (dead-proof recorded: zero importers, not collected); dedupe or
   delete. [COMPLETENESS_AUDIT.md orphan inventory]
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
12. **`windowed_exp_iEt` boundary alignment** — the convention is now
    DECIDED and landed by the clause-certification lane: half-open,
    **closed at the top `(lo, hi]`** (a pane must contain its supremum,
    since every rule is built at max(Γ) over its pane; recorded in the
    catalog's `bin_convention` field, branch aa24c099). Our helper
    landed as `[lo, hi)` and must flip to match — one line + its gate,
    with a boundary-pole red twin. Currently moves nothing (no real
    pole sits exactly on an edge — measured), so batch-class.
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

## Fixed (strike-in-place graveyard — newest first)

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
