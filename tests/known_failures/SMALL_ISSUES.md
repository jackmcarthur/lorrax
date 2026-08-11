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
4. **`lx test --help` is not a help flag** — it attempts a real full-suite
   run against `~/software/lorrax_P/tests` (the wrong tree); only a full
   pool has prevented it so far. [final-batch report, 2026-08-09]
5. **`pytest` is not on the container PATH** — `python -m pytest` works;
   `lx run ... pytest` dies with a remapped exit 89. [final-batch report]
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
11. **`lx status` co-tenant rendering ambiguity** — misreading it caused
    two kill-the-wrong-process incidents in one day; kill-by-PID
    (`pgrep -af srun`) is the adopted rule, but clearer rendering would
    remove the trap at the source. **Second, costlier symptom
    (2026-08-09): the table cannot distinguish a working step from a
    hung one** — the three exciton_bands never-exit steps (KNOWN_FAILURES
    2026-08-09 amendment) rendered as healthy occupancy all evening.
    [ops incidents, 2026-08-08; OWED_LEGS_BATCH.md]
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

25. **`lx run --wait` abandons its wait on a transient `lx_pool: timeout
    running scontrol` and refuses instantly with `LX-ALLOCFAIL`** — the
    wait loop should treat a slow scontrol as "still waiting", not as
    failure; sibling of the `lx_pool` squeue-timeout abort in
    BUILD_NOTES. Fix = retry inside the wait, or at least distinguish
    the two exit reasons. [OWED_LEGS_BATCH.md, 2026-08-09]
26. **Two JAX-distributed steps co-placed on one node collide** —
    `different incarnation` coordination-service error; one step dies
    with exit 134 laundered to rc 89 by the harness. Needs either
    per-step coordination ports or an lx-level no-coplacement rule for
    P>1 steps. [OWED_LEGS_BATCH.md, 2026-08-09]
27. **With two allocations live, `lx run --wait` sits on one while the
    other has 4/4 GPUs free** — and pinning via `SLURM_JOBID=<jid>`
    turns the wait into an outright `LX-EXPIRED`. The waiter should
    consider every live allocation it could satisfy.
    [OWED_LEGS_BATCH.md, 2026-08-09]
28. **`-G 0` does not buy a CPU-only slot on a full pool** — a
    CPU-only step should not compete for GPU capacity; today it queues
    behind GPU legs. [OWED_LEGS_BATCH.md, 2026-08-09]

29. **Two committed `dipole.h5` fixtures have no deck that reproduces
    them** — gnppm_debug and cohsex_debug: no in-tree deck produces
    their committed shapes at all (found during the dZ-None fixture
    comparison; si's committed fixture also came from
    `cohsex_si_test.in`, not the `_fast` deck, so re-cuts must use the
    right deck). Until decks exist, those two fixtures can never be
    regenerated — owner input needed on where their generating decks
    live or whether to retire them. [FIX_dz_none_dipole.md §7]

31. **`lx release --all` cancels the SHARED pool, not just your own
    allocations** — measured cost 2026-08-09 night: one lane's cleanup
    script cancelled pool 56554959 and killed another lane's 23-minute
    leg (restored as 56555953). Harness fix: `--all` should scope to
    allocations the caller created, or demand confirmation when a
    pool-tagged allocation is in the set. Until then the RULE is in
    BUILD_NOTES: never `lx release --all` while a shared pool is live —
    release by explicit job ID. [refreeze lane incident, 2026-08-09]

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

35. **`GATE wing_transition_axis` cannot see the error it names** —
    `gw.mpa.head_dipole.wing_tensors` requires its `pair_amplitudes` and
    its dipoles to be "the SAME vertical (k, c, v) set in the same
    order" (its own words) and checks only
    `M_flat.shape[0] != d_flat.shape[1]`, a COUNT. A (c, v)
    transposition leaves that count invariant whenever the transition
    manifold is a full product, which it always is: the Si
    big-continuum deck has n_c = 92 and n_v = 8, and 64·92·8 = 64·8·92 =
    47 104 either way. So the gate passes a transposed wing silently.
    The failure it would let through is not loud: a mis-ordered wing
    pairs each dipole with another transition's pair density, so the sum
    is INCOHERENT and collapses toward zero by phase cancellation —
    it looks like a correction that is merely too small, and the
    tempting repair is a scale factor, which restores the magnitude and
    not the frequency dependence. NOT the active cause of anything today:
    the 2026-08-10 LF-head lane's caller was checked and is correct
    (`_pair_amplitude(psi[:, sl.cond], psi[:, sl.val])`, logged as
    `M (64, 92, 8, 1128)`), and the tree's three producers —
    `transition_dipoles`, `_pair_amplitude` and the `delta` outer
    difference — all agree on (k, c, v) in C order. Registered as a
    latent hazard because the gate's own docstring states a requirement
    it does not test, and the next caller may not be checked by hand.
    Cheap fix: pass the (n_k, n_c, n_v) shape rather than the flattened
    count, or have `wing_tensors` do its own reshape from a named
    3-axis array instead of accepting a pre-flattened one.

## Fixed (strike-in-place graveyard — newest first)

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
