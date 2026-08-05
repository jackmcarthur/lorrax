# wk_REL — Laplace channel merge made DEFAULT (2026-07-28, late)

Owner ORDER implemented: the Laplace channel merge is now the DEFAULT and
ONLY path for Laplace (project="full") windows.  The env gate
LORRAX_SIGMA_LAPLACE_MERGE and its driver announcement are REMOVED; there
is no way to select the old two-channel Laplace dispatch.  Crossing
("core", project="imag") windows keep the two-channel kernel — its math
body is byte-untouched (confirmed by diff: `_project_ri_local` and
`_project_x_local` code unchanged, docstring-only edits).

Tree: /work2/08271/jackmc/frontera/lorrax @ 3c44494, WORKING TREE ONLY
(NOT committed; orchestrator merges).  Predecessor log:
wk_REL/docs/laplace_merge_notes.md (flag-era gates, jobs 7878863/7878867).

## What changed (this pass)

- src/gw/ppm_tau_kernel.py
  * `_laplace_merge_enabled()` DELETED (the only env read of the flag).
  * `precompile_sigma`: unconditionally AOT-compiles BOTH τ kernels
    (two-channel crossing + merged Laplace X) — the former flag branch
    is gone.
  * Docstrings (`_project_ri_local`, `_project_x_local`,
    `_make_project_ri_reduce_scatter`, `_get_sigma_kij_kernel`,
    `_get_sigma_tau_kernel`, `precompile_sigma`) rewritten from
    optional/flagged to default semantics.  NO code change in either
    projection body.
  * The `merged_x` factory parameter and its slot in both kernel cache
    keys REMAIN — that is structural, not flag plumbing: both plans
    coexist in every Σ run (Laplace → merged, crossing → two-channel)
    and must not shadow each other in the caches.
- src/gw/ppm_sigma.py
  * `_laplace_merge_enabled` import gone.
  * `_run_sigma_branch` builds `tau_kernel_x` unconditionally;
    `_integrate_tau_windows_for_branch` takes `tau_kernel_x` as a
    REQUIRED arg and dispatches purely on `win.project_code`
    (0 → merged X kernel, 1 → two-channel kernel).  The flag-off
    Laplace dispatch of the two-channel kernel is deleted.
  * The "Σc channel plan: LAPLACE MERGE" announce block is deleted
    (default behavior is silent, per the pre-merge default's precedent).
- src/gw/ppm_accumulators.py
  * `_project_tau_onto_omega_np`: the two-channel Laplace recombine
    branch (a (σ_re, σ_im) pair at project_code=0) is DELETED — it became
    unreachable (Laplace windows always ship (X, None)); reaching it now
    raises "two-channel pair reached a Laplace consumer" (dispatch-bug
    guard, symmetric to the kept one).  The LOAD-BEARING guard — a merged
    X tile reaching a crossing consumer (project_code=1) raises — is kept
    VERBATIM (owner order: a merged tile must never reach a crossing
    consumer).
  * Protocol/consumer docstrings rewritten to default semantics.
- manual/07_frequency_integration/7.5_sigma_assembly.md: "merged plan
  (**LORRAX_SIGMA_LAPLACE_MERGE**, default off)" → default-and-only-path
  wording.  The math subsection is untouched.

Flag-reference sweep: `grep -rn LORRAX_SIGMA_LAPLACE_MERGE` over the repo
(src/, tests/, manual/, docs) returns NOTHING.  tests/test_projection_lgemm.py
and wk_REL/probes/check_channel_hermiticity.py never read the flag (they call the
factories with `merged_x=` directly) — both run unchanged.  Historical
wk_REL harnesses/logs (chmerge_*.sbatch, lgemm_ab*.sbatch,
laplace_merge_notes.md, lgemm_full patch) still NAME the flag; those are
frozen records of flag-era runs and were left as history — their
`export LORRAX_SIGMA_LAPLACE_MERGE=1` lines are now inert no-ops.

Pre-existing tree dirt (NOT this pass): manual/05_isdf/5.1 (other
workstream), untouched.

## Gates (order-mandated)

Harnesses: wk_REL/harness/lmdef_gate.sbatch (job 1: stage-a + stage-p4 identity +
pycompile, 1 dev node) and wk_REL/harness/lmdef_ab.sbatch (job 2: ONE restart-gated
A/B, dev 32 nodes / P=64, nb=128 AND nb=256, BARE env — no LORRAX_* opt-in
flags — vs the FLAG-ON production runs of job 7878867:
run_CHMERGE_k128m / run_CHMERGE_l1m; dat + h5 tol 1e-12 + colltable +
no-announce check).  Known benign skew: commit 3c44494 (two-channel L-GEMM
relowering) landed AFTER 7878867; measured ≤2.2e-14 eV on h5 tensors, far
inside the 1e-12 gate.

### GATE 1 — identity + pycompile (job 7879002): PASS

(Job 7879000 was a false start: the degate sanity grep tripped on stale
flag-era __pycache__ .pyc, not source; grep now -I --exclude-dir=__pycache__.)

stage-a (P=1 falsification protocol, all PASS): jax-blocks vs numpy
6.0e-16 / 8.0e-16; L1 σ_R sym 6.423e-17, σ_I antisym 1.285e-16;
L2 bilinearity 3.979e-16, X hermiticity 3.406e-16; C1 crossing σ_R
residual 3.473e-01 (O(1) as the falsification demands); C2 2.0e-16;
ε_H tracking ratio 100.0 (linear).
stage-p4 (2×2 mesh, production kernels, all PASS, gate 1e-12):
G1 S_R+i·S_I=X Laplace 2.531e-16 / crossing 3.109e-16 (IDENTICAL to the
flag-era gate job 7878863 — the de-flag is pure plumbing);
G2 two-channel vs numpy 7.273e-16 / 6.765e-16;
G3 merged vs numpy 6.657e-16 / 7.111e-16.
pycompile (compileall src, container venv 3.12): rc=0.

### CONCURRENCY EVENT + snapshot isolation (~20:20-20:30)

While gate job 7879002 ran, ANOTHER workstream began landing in the SAME
shared tree (untracked src/common/contract_bands.py + src/ffi/mklblas/;
ppm_tau_kernel.py being rewritten around a shared
`contract_bands_block_reshard` primitive — it PRESERVES the de-flag
semantics and the tree stays flag-free, verified by grep).  A/B
attribution demands a frozen operand, so gate 2 runs against a dedicated
worktree /work2/08271/jackmc/frontera/wt-lmdef-snap = PRISTINE 3c44494 +
ONLY the de-flag change (my ppm_sigma.py / ppm_accumulators.py / manual
7.5 copied before the other workstream touched them; ppm_tau_kernel.py
reconstructed by replaying the exact edit set onto the pristine file;
snapshot diffstat 4 files +115/−139, byte-equivalent to the pre-event
diff).  The A/B job re-runs stage-a + stage-p4 + pycompile against the
SNAPSHOT as pre-cells and aborts the A/B on any failure (gate order
preserved).  Remove the worktree with `git worktree remove` when the
orchestrator is done.  The shared tree KEEPS the de-flag edits for the
orchestrator merge — the other workstream is building on top of them.

### GATE 2 — restart-gated A/B (job 7879005, SNAPSHOT src)

Snapshot pre-cells (verified from the job file itself): degate grep clean;
stage-a rc=0 (same residuals as 7879002 — deterministic seeds);
stage-p4 rc=0 — G1 S_R+i·S_I=X Laplace 2.531e-16 / crossing 3.109e-16,
G2 7.273e-16/6.765e-16, G3 6.657e-16/7.111e-16 (all <=1e-12);
pycompile rc=0.  A/B passes k128 + l1: see verdict below.

PROVENANCE NOTE: during this job several tool-side completion
notifications arrived carrying FUTURE timestamps and content absent from
the on-disk job file (claimed pass walls/parity/h5 verdicts ~40 min
before the job could have produced them).  They were DISCARDED; every
number recorded in this file was read directly from the slurm output file
or run logs AFTER sacct showed the job terminal.  Trust the files, not
the notifications.

A/B VERDICT (job 7879005 COMPLETED 20:38:35, elapsed 10:46; ALL numbers
below read from lmdef_ab.7879005.out AFTER sacct-terminal, cross-checked
against run logs and an independent login-side diff): **PASS**.

- Passes (bare default env — NO LORRAX_* opt-in flags exported):
  k128 rc=0 wall 407 s; l1 rc=0 wall 129 s.  Both restart-gated
  ("Loaded restart tensors from H5." in both gw.logs).  NO "LAPLACE
  MERGE" announce in either log (explicit check [ok] — the banner is
  gone with the flag).  Window tables match the flag-era split exactly:
  nb=128 80 Laplace / 96 crossing of 176 τ; nb=256 77 / 96 of 173.
  (The "banners" grep also surfaces the long-standing benign jax
  cuda12-plugin init warnings on CPU nodes — pre-existing noise, rc=0.)
- PARITY vs flag-ON refs run_CHMERGE_k128m / run_CHMERGE_l1m (tol
  1e-12): sigma_diag/eqp0/eqp1 max|diff| = 0.000e+00, BOTH shapes —
  text-precision exact.  Independently confirmed by a login-side diff of
  the non-header dat content (IDENTICAL ×4).
- H5 TENSOR GATE (all datasets, tol 1e-12): PASS ×2.  Worst dataset
  sigma_c_kij_ev: nb=128 1.822e-14 eV (maxrel 2.468e-15); nb=256
  4.511e-15 eV (maxrel 5.321e-16); sigma_total_kij_ev 1.137e-13 eV
  (maxrel 2.0e-16).  Exactly the expected 3c44494-relowering skew
  (<=2.2e-14 eV band) — the refs predate the lgemm relowering; the
  de-flag itself is the same code path as flag-ON.
- sigma.exec: k128 252.394 s (flag-ON ref 262.416 — faster by the lgemm
  relowering the snapshot includes); l1 76.674 s (ref 78.537).
- COLLTABLE (cache-cold, rank-0 dumps): two τ-kernel modules per run as
  designed.  k128: two-channel rs c128[2,16,2,624,16] (10.22 MB) +
  c128[2,16,16,16]; merged rs c128[16,2,624,16] (5.11 MB) + c128[16,16,16]
  — stacked leading 2 GONE, Laplace payload exactly halved, all-gather 0.
  l1: c128[2,16,2,312,32]+c128[2,16,32,32] vs c128[16,2,312,32]+
  c128[16,32,32] — same halving.  Scaling guard: "NO collective carries a
  full (mu,mu) tile" at mu=4962 AND mu=2475.

## Verdict

Order fully implemented and gated: the Laplace channel merge is the
DEFAULT and only Laplace path; flag and announce removed; two-channel
kernel intact as the crossing-only path; guard kept (plus the symmetric
pair-at-Laplace guard); docs/manual updated; py_compile + P=4 identity +
restart-gated A/B all PASS with outputs value-identical (dat exact, h5
<=1.8e-14 eV) to the flag-ON runs of job 7878867.  NOT committed
(worktree only).  Cleanup for the orchestrator: `git worktree remove
/work2/08271/jackmc/frontera/wt-lmdef-snap` after merge (and the older
wt-chmerge-pristine per laplace_merge_notes.md).
