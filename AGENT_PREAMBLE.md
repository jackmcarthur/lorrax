# AGENT_PREAMBLE — read once, at dispatch

The entry document for a lane working on LORRAX. **A lane that reads only this file
should behave correctly.** Everything else is depth: `AGENTS.md` is the code,
`~/lorrax_service_phase/BUILD_NOTES.md` is the trap inventory, and the register at the
foot says who owns what. Do not re-derive any of this per leg — that is what it is for.

## Efficiency doctrine

Seven rules. Each was measured on this fleet; each has a tool that makes it the cheap
path. Numbers: `docs/warm_worker.md` (2,183 steps, 1,904 banners, 2026-08-06→10).

| # | rule | measured | tool |
|---|---|---|---|
| 1 | **FAN OUT** independent legs; serial submission of independent legs is a defect in the plan | median duty cycle **0.41**; 115 of 152 evidence dirs strictly serial; **32.4 h idle vs 17.5 h compute**; co-tenancy free and one-leg-per-node **retired** since 2026-08-10 (4 placed 1-GPU legs/node within 2 % of solo walls); whole nodes only for legs needing the full card's memory | `lx batch` manifest (landing, sandbox batch lane); until then several `lx run` submits in one turn; fan out across INDEPENDENT resources — four pytest runs on one box starved all four (Stage A lane, self-reported) |
| 2 | **ONE COMBINED P=4 LEG** does a lane's whole verification — gates + driver + red twin in one dispatch, not one leg per gate | each extra dispatch re-pays the 16 s floor for nothing | `lx run -N 1 -G 4 -n 4`; `lx test` |
| 3 | **HARVEST BEFORE MEASURING.** There are **two** report corpora and only one is indexed — quoting the index alone as if it were the corpus is how a lane concludes "no measurement exists" for something already measured | 82 reports indexed with *superseded by* markers; **60 more in `~/lorrax_service_phase/` with no index — grep it, that is the supported access** (a 2026-08-10 lane priced a stage at 5–15 min that `CENTROID_GEN_ASSESSMENT.md` had at 6 s) | `~/lorrax_bse_perf_2026-08-08/INDEX.md`, then its `EVIDENCE_MANIFEST.md`, then grep `~/lorrax_service_phase/` |
| 4 | **WARM WORKER** when the same geometry runs more than twice; landing evidence still comes from a cold leg | **2.40 s warm vs 20.94 s cold**; 16 s cold floor is `initialize_communicator_stack`, not queue/container; 3-leg light lane 63.17 s → 10.71 s | `lx warm start`, `lx warm submit` |
| 5 | **LANE WEIGHT** chosen when the lane is planned and named in the report's first line | light = mechanical fix/rename/one number → **five lines** — one sentence each, ~35 words, not five paragraphs: changed, proof, evidence path, owed, branch. heavy = design/investigation → full report + index entry | the five-line report |
| 6 | **ENVIRONMENT CONTRACT READ ONCE** — this file. If it is wrong, correct it on your branch | four different accounts of the same pool, from working around it silently | this page |
| 7 | **LEDGER AS YOU GO** — dated amendment file, in-repo small issues, evidence path in every report; a supersession goes in the INDEX row, not only the report body | 3 `KNOWN_FAILURES.md` conflicts hand-resolved in one night; 4 orphan `/pscratch` workspaces, 2 frozen refs now unregenerable | `tests/known_failures/<date>-<slug>.md`; `tests/known_failures/SMALL_ISSUES.md` (canonical, only copy); `EVIDENCE_MANIFEST.md` |

Rules 1 and 2 are complements, not exceptions: **fan out across independent work,
combine within one verification.** A lane that splits its own gates into four legs and
runs those serially breaks both, and that is the shape of most of the 32.4 idle hours.
"Two to four minutes of bring-up" is refuted — one banner in 1,904 exceeded 40 s (`docs/warm_worker.md`).
A frozen reference's provenance header names its **workspace**, not just the commit.

## Measurement discipline

Validity, not cost — a leg can be perfectly efficient and measure nothing. The
incident is the reason; it is not decoration.

| # | rule | incident |
|---|---|---|
| 1 | **A/B instrument check**: `[lx] source tree:` is necessary, not sufficient — also `git rev-parse HEAD` *inside* the leg, and grep the log for `ignored`/unknown-key lines | a worktree one commit behind the deck key ran both arms flag-off; a green A/B measured nothing. Corollary (spec, not implemented here): unknown deck keys should REFUSE, not log-and-proceed → `tests/known_failures/2026-08-10-unknown-deck-key-refusal-spec.md` |
| 2 | **BOTH SIDES RAN**: read N-passed/N-failed on both arms before publishing any set-diff | three invalid regression diffs in one day, each an empty or killed base side reported as an improvement |
| 3 | **TAIL IS NOT TOTAL**: reconcile against the run's own summary line | a "4 failures" report was 13 |
| 4 | **ARM ANCESTRY**: `git merge-base` the reused arm's tree against every fix the current tree has | a stale arm carried a since-fixed defect and 7.41 eV of impossible Im Eqp0 into a comparison — a stale arm is not a control (`SIGMA_PPM_CAMPAIGN.md`) |
| 5 | **PRE-REGISTRATION** for selection/correlation studies: predictions + candidate definitions committed before scoring, enforced by code, every candidate reported including the dead ones | `prereg.py` refuses to write a ranking if results exist (`~/lorrax_service_phase/BGW_CD_COMPARISON_DESIGN.md`) |
| 6 | **CERTIFY WHERE CONSUMED**: the gate measures the functional at the locus that consumes it, not at a convenient proxy | a head fit passed its sample gate at 2e-9 while corrupting Σ by 400 meV, via structure the samples could not see |
| 7 | **PROVENANCE TRAVELS OR THE CLAIM DOESN'T**: env/infra claims leave a lane with exact config + log path, else labelled hypothesis | "BFC is unusable": one datum under the wrong config, three cross-fleet rounds to retract (`ALLOCATOR_ARCHAEOLOGY.md`) |
| 8 | **STOP AT THE FUNDED LINE**: price before launching; over cap → stop and report the plan. Stage boundaries are resume points. Kill by PID, never `pkill` pattern. No artifact deletion under a running comparison | — |
| 9 | **CACHE SYMMETRY IS NOT A COUNT**: a P>1 leg that reports `xla_compiles=0 vetoed=0` has shown nothing until the cache-KEY SET is compared across ranks — four ranks each holding a private program report the same numbers as four sharing one | `tests/test_jax_cache_contract.py`; `LORRAX_JAX_CACHE_KEYDUMP`; the five siblings of `FIX_multislice_cachekey.md` §6.1 |
| 10 | **VERIFY WHAT WAS COLLECTED, not just what failed** — a green run whose collection you did not check is not a green run; print/assert the collected count against the expected set (three variants in one day: worktree venv importing the wrong tree, direct_cells not expanding parametrize so green read FAIL, importorskip collapsing a whole module to one silent SKIP) | collected==expected, skips itemized by name, before believing any green |

Claim/check rubric: `docs/dev/QUALITY_PATTERNS.md` (ten classes, cited by number).
Worked gate-by-gate example: **§6 of `docs/mpa_method_guide.md`**, on the MPA
integration branches, not main — `git show feat/mpa-wedge-pole-unfold-2026-08-10:docs/mpa_method_guide.md`.

## THE FOUR-GPU RULE

> **"use four gpus for 100% of all testing so that never ever do we run something on
> one GPU and then learn it doesn't generalize later"**

- **Every GPU verification leg runs at P=4.** Not P=1 with P=4 owed; not "P=1 first".
- **A P=1-only verification never lands.** Report it unverified, however green.
- **Unit and CPU cells are exempt** — a lane claiming no P=4 leg is owed names this
  clause and the reason it applies; emulated CPU meshes
  (`--xla_force_host_platform_device_count`) are fine for device-count *logic*, and do
  not substitute for the P=4 leg on a real GPU path.
- Why: co-tenancy, sharded-vs-single loader drift, collectives and allocator behaviour
  are invisible at P=1 by construction, and each was found after a P=1 leg had been
  reported as evidence.
- This governs **verification**. A farm of independent 1-GPU exploration legs under
  doctrine rule 1 is not verification and is not what this rule refuses.

## The machine

| thing | what to know |
|---|---|
| certificate | 24 h, owner refreshes (`~/bin/sshproxy.sh -u jackm`). Get the **minutes left**, never eyeball the window against a remembered clock: `echo $(( ($(date -d "$(ssh-keygen -L -f ~/.ssh/nersc-cert.pub \| sed -n 's/.*to //p')" +%s) - $(date +%s)) / 60 )) min left`. A lane that misreads this wraps up early and writes its report twice. A working `ssh` is **not** evidence — `ControlPersist` answers past expiry; re-probe with `ssh -o ControlPath=none perlmutter true`, and **never** `ssh -O exit`: it evicts the ControlMaster and kills every backgrounded launcher with it while the Slurm steps run on, so a lane reads "no rc file" as a death and relaunches into a log race (`SMALL_ISSUES` row 41) |
| Shifter gateway | when down, no container step starts anywhere while `ssh`/`squeue`/`lx doctor` all look healthy. Check `shifterimg images`; `shifterimg lookup` false-negatives on a healthy gateway |
| `lx status` | cannot tell working from hung (`exciton_bands` never-exit read as healthy occupancy for an evening) — use `lx status --verify` (two `sstat` samples 6 s apart; a deadlock burns CPU and moves no bytes, and so does a long kernel). It renders CPU-only allocations GPU-shaped — check `scontrol show job <id>` for `gres/gpu` in `AllocTRES`. Empty `squeue --me` ≠ no jobs: `scontrol ping` before believing `LX-ALLOCFAIL`. An idle allocation is a pool or a warm worker, **not** a leak |
| allocation ownership | `lx run` **allocates if nothing is live, attaches if something is** — your leg running as a step of a pool you did not create is designed co-tenancy, not an incident, and is not worth investigating. `lx release` cancels only what THIS agent created and leaves another's pool alone; there is **no positional-jobid form** (`lx release <jid>` is an argparse error), and `--include-pool <jid>` is the only way to reach someone else's. The 56554959 kill was a pre-fix `--all` |
| EXIT first | 0–89 the command's; 90–98 the step never ran (`LX-WRONGSITE` 90, `NOSLURM` 91, `NESTED` 92, `ALLOCFAIL` 93, `LOCKHELD` 94, `TOOSMALL` 95, `POOLFULL` 96, `SITEENV` 97, `EXPIRED` 98). An `LX-*` code is an absence, never a measurement |
| artifact size, not rc | `$HOME` is 40 GiB; full `$HOME` gives a 38-byte junitxml that parses as zero tests |
| traps | `~/lorrax_service_phase/BUILD_NOTES.md` **before planning any cluster leg**: `.so` pair tables + ABI stamp, `LX_BASE_MODULE=lorrax_J070` (without it: wrong jax, ~52F/32E instead of 14F) — **export it on the LOGIN NODE, before `lx run`**, because setting it inside the step's own script runs after `lx` has chosen the container image and does nothing; a **fresh worktree has no `.so` pair at all**, so any leg that runs a DRIVER (not just pytest) must link or build one first and the FFI gate refuses at `initialize_communicator_stack` if it cannot, the tell being `FfiLibraryNotBuilt: Could not locate liblorrax_ffi.so`; the WSL worktree-`PYTHONPATH` trap, `-G=4` = a **quarter** of the memory, co-tenancy + its retirement, gateway outage, suite sharding |
| WSL bench | your own worktree, always: `git worktree add -b <branch> <scratchpad>/wt_<slug> origin/main` — never `git worktree list` to decide where (sixty rows, all other lanes'). Tests run `/home/jackm/projects/lorrax/.venv/bin/python -m pytest` with `PYTHONPATH=<wt>/src` and `JAX_PLATFORMS=cpu`; the bare `python3` has no jax. Prove the pin before measuring: `python -c "import <mod>; print(<mod>.__file__)"` must print a path inside your worktree |
| fan-out verb | `lx batch <manifest>` — JSONL, one leg per line (argv, workdir, optional env); fires up to `-P` concurrently, per-leg logs, REFUSES BY NAME on any missing/failed leg; pass `--wait`; `-P 1` = the serial control, same code path; `--mode auto` submits into a live warm worker (~2.4 s/leg vs ~21 cold — iteration only, landing evidence stays cold-leg) |
| band degeneracy | defaults `strict` (owner ruling 2026-08-10). `snap` turned a 4v4c anchor into 4v8c and the gate read it as an 0.0906 eV regression no branch caused. **Never set `snap` to make a gate pass** |
| allocator | every wall-time carries its allocator in the same sentence — `BFC@0.85` is the campaign default. Unstated allocator = not comparable to any other timing, including its own re-run |

## Etiquette

- **Your own worktree.** Never `/home/jackm/projects/lorrax` (shared clone; a lane's
  commit already landed on another's ref there); never `/pscratch/sd/j/jackm/wt_int0808_bgw`.
- **`RUNS_INFLIGHT.md` row before any cluster submit** (canonical: `/pscratch/sd/j/jackm/sandbox_v2_rescue_2026-08-06/RUNS_INFLIGHT.md`) — struck when the claim
  how a peer discovers its legs can ride along with yours (rule 1).
- **Push feature branches freely; `main` needs owner approval.** Conflating those into
  "never push" cost a night. `git merge-base --is-ancestor <c> origin/main` is the only
  "landed" check.
- Before editing a file that is not obviously yours, check `ListAgents` and peer branches.

## Register

| need | read |
|---|---|
| working method + machine | this file — the whole contract |
| numbers behind rules 1 and 4 | `docs/warm_worker.md` |
| what a report already settled | `~/lorrax_bse_perf_2026-08-08/INDEX.md` |
| is this evidence path still live | `~/lorrax_bse_perf_2026-08-08/EVIDENCE_MANIFEST.md` |
| traps already sprung | `~/lorrax_service_phase/BUILD_NOTES.md` |
| is this red already known | `tests/KNOWN_FAILURES.md` — note the path: one level **above** `tests/known_failures/` |
| the code: modules, conventions, running | `AGENTS.md` |
| is this claim/check any good | `docs/dev/QUALITY_PATTERNS.md` |
| what a deck key is called and does | `docs/input_reference.md` — generated from the parser, so it is complete; never grep `src/` for key names |
| which page owns a documented fact | `docs/index.md` register table |
| what a run actually resolved | that run's rank-0 startup block — outranks every page here |
