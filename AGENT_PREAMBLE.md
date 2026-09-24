# AGENT_PREAMBLE — read once, at dispatch

The working contract for an agent lane on LORRAX: how to spend compute, how to
make a measurement mean something, and the machine. `AGENTS.md` covers the
code; the register at the foot says which page owns what.

## Efficiency

| # | rule | why | tool |
|---|---|---|---|
| 1 | **Fan out independent legs.** Serial submission of independent legs is a planning defect. | A lane waiting on one leg at a time is idle between dispatches | several `lx run --pool` submits in one turn, or `lx batch <manifest>` |
| 2 | **One combined P=4 leg per verification:** gates, driver and red twin in one dispatch. | Each extra dispatch repays the bring-up floor (`initialize_communicator_stack`) | `lx run --pool POOL -N 1 -G 4 -n 4`, `lx test` |
| 3 | **Warm worker** when one geometry runs more than twice; landing evidence still comes from a cold leg. | A warm leg skips the bring-up and the compile | `lx warm start`, `lx warm submit`, `lx batch --mode auto` |
| 4 | **Lane weight named in the report's first line.** Light (mechanical fix, one number): five one-sentence lines — changed, proof, evidence path, owed, branch. Heavy (design, investigation): a full report. | Readers triage by weight | — |
| 5 | **Ledger as you go:** evidence path in every report; supersession recorded where the superseded result is indexed. | Hand-resolved ledger conflicts and orphaned workspaces | `tests/known_failures/<date>-<slug>.md`, `tests/known_failures/SMALL_ISSUES.md` |

Rules 1 and 2 are complements: fan out across independent work, combine
within one verification. The measurements behind them are in
[`docs/warm_worker.md`](docs/warm_worker.md).

## Measurement discipline

A leg can be efficient and measure nothing. Each rule names the failure it
prevents.

| # | rule | failure it prevents |
|---|---|---|
| 1 | **Instrument check on every A/B arm:** the `[lx] source tree:` line, `git rev-parse HEAD` inside the leg, and a grep of the log for ignored or unknown deck keys | a worktree one commit behind the deck key ran both arms flag-off |
| 2 | **Both sides ran:** read passed/failed on both arms before publishing a set difference | an empty or killed base arm reported as an improvement |
| 3 | **Tail is not total:** reconcile against the run's own summary line | "4 failures" that were 13 |
| 4 | **Arm ancestry:** `git merge-base` a reused arm against every fix the current tree has | a stale arm carrying a fixed defect into a comparison |
| 5 | **Pre-registration** for selection or correlation studies: predictions and candidates committed before scoring, every candidate reported | post-hoc selection |
| 6 | **Certify where consumed:** gate the quantity at the locus that consumes it | a head fit passing its sample gate at 2e-9 while moving Σ by 400 meV |
| 7 | **Provenance travels with the claim:** an environment or infrastructure claim carries its exact configuration and log path, or it is a hypothesis | a one-datum allocator verdict that took three rounds to retract |
| 8 | **Stop at the funded line:** price before launching; over budget, stop and report the plan. Kill by PID, never by `pkill` pattern. Delete no artifact under a running comparison | — |
| 9 | **Cache symmetry is a key set, not a count:** at P>1, compare the persistent-cache key sets across ranks (`LORRAX_JAX_CACHE_KEYDUMP`) | four ranks with private programs reporting the same counts as four sharing one |
| 10 | **Verify what was collected:** assert the collected test count against the expected set, and itemize skips by name | a green run that collected the wrong tree, or one silent module-wide skip |

Claims and checks are graded against
[`docs/dev/QUALITY_PATTERNS.md`](docs/dev/QUALITY_PATTERNS.md), cited by
class number.

## The four-GPU rule

- **Every GPU verification leg runs at P=4** (`-N 1 -G 4 -n 4`, one rank per
  GPU). A P=1-only verification never lands; report it as unverified.
- **Unit and CPU cells are exempt.** A lane claiming the exemption names it.
  Emulated CPU meshes (`--xla_force_host_platform_device_count`) serve
  device-count logic, never a real GPU path.
- Co-tenancy, sharded-loader drift, collectives and the allocator are
  invisible at P=1.
- The rule governs verification. Independent one-GPU exploration legs under
  rule 1 are not verification.

## The machine (Perlmutter)

| thing | what to know |
|---|---|
| runtime | `export LX_BASE_MODULE=lorrax_A` on the login node, before `lx run`. The module selects a `git archive` source snapshot and one **sealed FFI bundle** (CUDA and host legs, handler ABI `LORRAX_FFI_ABI_VERSION` in `src/ffi/common/ffi_loader.py`); a checkout needs no `.so` of its own. The loader refuses a pinned `LORRAX_FFI_SO` whose ABI is not the source's ([Perlmutter](docs/environment/machines/perlmutter.md)) |
| allocations | agents never allocate: the coordinator runs shared pools and a leg joins one with `lx run --pool NAME` (or `--jid`). `lx` claims a free node per leg; a full pool makes the leg wait (`--wait`). `lx release` cancels only what the calling agent created |
| exit codes | 0–89 are the command's; 90–98 mean the step never ran (`LX-WRONGSITE` 90, `NOSLURM` 91, `NESTED` 92, `ALLOCFAIL` 93, `LOCKHELD` 94, `TOOSMALL` 95, `POOLFULL` 96, `SITEENV` 97, `EXPIRED` 98). An `LX-*` code is an absence, never a measurement |
| hung or working | `lx status` cannot tell; `lx status --verify` samples `sstat` twice, 6 s apart. `lx status` draws CPU allocations GPU-shaped: check `AllocTRES` in `scontrol show job <id>` for `gres/gpu` |
| certificate | 24 h; compute the minutes left from `ssh-keygen -L -f ~/.ssh/nersc-cert.pub`. A working `ssh` is no evidence (ControlPersist answers past expiry): probe with `ssh -o ControlPath=none perlmutter true`. Never `ssh -O exit`: it kills every backgrounded launcher |
| GPU memory pool | owned by the runtime: `cuda_async`, reserved, fraction 0.89; a leg script exports none of the pool variables ([overview §2.1](docs/environment/overview.md#gpu-pool)). Timings are comparable only under the same pool |
| artifacts, not rc | Judge a leg by its artifacts, not its rc. `$HOME` is 40 GiB; a full `$HOME` yields a 38-byte junitxml that parses as zero tests |
| band degeneracy | the default is `strict`. Never set `LORRAX_BAND_DEGENERACY=snap` to make a gate pass |

## Etiquette

- **Your own worktree:** `git worktree add -b <branch> <path> origin/main`.
  Never commit in another lane's tree.
- **Before a cluster submit**, add a row to the sandbox's `RUNS_INFLIGHT.md`;
  strike it when the leg ends.
- **Push feature branches freely; `main` needs the owner's approval.**
  `git merge-base --is-ancestor <commit> origin/main` is the only "landed"
  check.

## Register

| need | read |
|---|---|
| the code: modules, conventions, running | `AGENTS.md` |
| is this red already known | `tests/KNOWN_FAILURES.md` (one level above `tests/known_failures/`) |
| is this claim or check sound | `docs/dev/QUALITY_PATTERNS.md` |
| what a deck key is called and does | `docs/input_reference.md` |
| which page owns a documented fact | the register in `docs/index.md` |
| environment variables | `docs/dev/env_vars.md` |
| what a run actually resolved | that run's rank-0 startup block, which outranks every page |
