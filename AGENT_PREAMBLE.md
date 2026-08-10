# AGENT_PREAMBLE — the environment every dispatched lane starts from

This is the standard preamble for a lane dispatched to work on LORRAX. It
exists so that the environment is stated once, in the tree, rather than
re-derived in every dispatching prompt — which is how lanes ended up running
under different assumptions about the pool, the allocator and the gate.

Read it before you plan a leg. It is about the machine and the etiquette;
`AGENTS.md` is about the code, and `~/lorrax_service_phase/BUILD_NOTES.md` is
the running inventory of traps that have already cost someone a night.

---

## THE FOUR-GPU RULE

The owner's instruction, verbatim, and it governs everything below:

> **"use four gpus for 100% of all testing so that never ever do we run
> something on one GPU and then learn it doesn't generalize later"**

So:

- **Every GPU verification leg runs at P=4.** Not P=1 with a note that P=4 is
  owed; not "P=1 first and P=4 if it looks interesting". Four.
- **A P=1-only verification is never sufficient for landing.** If the only
  evidence a branch carries is single-device, the branch is not ready, however
  green that evidence is. Report it as unverified rather than as passing.
- **Unit and CPU cells are exempt.** Pure-algebra cells, AST gates, and
  anything that runs on a CPU box were never the problem this rule solves.
  Emulated CPU meshes (`--xla_force_host_platform_device_count`) are also fine
  where the thing under test is device-count logic rather than device
  behaviour — but they do not substitute for the P=4 leg on a real GPU path.

The rationale is not conservatism about correctness at P=1. It is that the
whole class of defects this tree keeps finding — co-tenancy, sharded-vs-single
loader drift, collectives, allocator behaviour — is invisible at P=1 by
construction, and every one of them was found late, after a P=1 leg had
already been reported as evidence.

In practice: `lx test` already takes all four GPUs on the node. For a driver
leg, ask for the whole node with `-G 4` rather than accepting the one-GPU
default.

---

## Before anything: the certificate

The NERSC certificate lasts 24 hours and the owner has to refresh it
interactively (`~/bin/sshproxy.sh -u jackm`). Check it before starting any
long task:

```bash
ssh-keygen -L -f ~/.ssh/nersc-cert.pub | grep Valid
```

A working `ssh` is not evidence that the certificate is good. `ControlPersist`
keeps a master connection alive across the expiry, so every probe through that
socket keeps answering long after a fresh connection would be refused — this
has already produced both a false "cluster is up" and a false "cluster is
down". To genuinely re-probe, drop the master first with `ssh -O exit
perlmutter`.

The other one-second check worth doing before you plan a cluster leg is the
Shifter image gateway, because when it is down no container step can start on
any node and everything else still looks healthy: `ssh` answers, `squeue`
lists live allocations, and `lx doctor` says `verdict: OK`. Use
`shifterimg images`; `shifterimg lookup` false-negatives on a healthy gateway
unless you give it a full image spec.

## Pool discovery, and what `lx status` does not tell you

`lx status` is where you find out who is running where. Four caveats, each of
which has cost real time:

1. **It cannot distinguish a working step from a hung one.** The
   `exciton_bands` never-exit steps rendered as healthy occupancy for a whole
   evening. Before concluding either that the pool is busy or that it is free,
   check `sstat` and the step logs.
2. **It prints a GPU-shaped table for CPU-only allocations.** A CPU-partition
   job shows as "4 nodes · 4/4 free" exactly like a GPU pool entry, and a leg
   pinned there dies instantly on an invalid gres specification. Before pinning
   a leg to an allocation, check `scontrol show job <id>` for `gres/gpu` in
   `AllocTRES` — the shape of the `lx status` table is not evidence.
3. **Before believing any `LX-ALLOCFAIL`, run `scontrol ping`.** An empty
   `squeue --me` — no header at all — is indistinguishable from "no jobs".
4. **An idle allocation is normal, not a leak.** A warm pool is how lanes skip
   the queue tax. Never cancel one you did not create.

## Release by job ID, never `--all`

`lx release --all` cancels every allocation `lx` created, which on a shared
fleet includes allocations that are not yours. It cancelled pool 56554959
mid-leg and killed another lane's 23-minute run. **Release by explicit job ID
only, and never use `--all` while a shared pool is live.**

## Verify every step's EXIT before you read its numbers

`lx` reserves the top of the exit-code range for its own refusals: 0–89 are
the command's own, and 90–98 mean the step never ran —
`LX-WRONGSITE` 90, `LX-NOSLURM` 91, `LX-NESTED` 92, `LX-ALLOCFAIL` 93,
`LX-LOCKHELD` 94, `LX-TOOSMALL` 95, `LX-POOLFULL` 96, `LX-SITEENV` 97,
`LX-EXPIRED` 98.

A leg that comes back 96 is not a red and it is not a green; it is an absence.
Read the EXIT of each step before reading anything else, and never let an
`LX-*` code enter a results table as though it were a measurement.

Two companion habits, from the same defect class:

- **Check artifact size, not exit code.** `$HOME` is 40 GiB and a full `$HOME`
  fails runs silently: the junitxml exists, is nonzero at 38 bytes, and parses
  as zero tests.
- **Read the `[lx] source tree:` line in every log.** A stale
  `LORRAX_CHECKOUT` once ran a different worktree entirely and produced a
  convincing false red on the BSE anchor. That line is the falsification check
  for "did my code actually run".

## The trap inventory

Do not re-derive the environment's traps.
**`~/lorrax_service_phase/BUILD_NOTES.md`** is the running inventory and it is
the first thing to read before planning a cluster leg. It currently carries the
`.so` pair tables and the ABI stamp, the `LX_BASE_MODULE=lorrax_J070`
requirement (without it you land in an image with the wrong jax and the suite
reports ~52F/32E instead of 14F), the WSL trap where a worktree's pytest
silently imports the main checkout unless `PYTHONPATH` pins the worktree, the
fact that `-G=4` gets you a quarter of the memory rather than four times it,
the GPU co-tenancy trap and its one-leg-per-node interim rule, the Shifter
gateway outage, and the suite-sharding trap.

## Band degeneracy defaults to `strict`

`--band-degeneracy` has defaulted to `strict` since the owner's ruling of
2026-08-10. `snap` shipped as the default with `824032b7` and within a day had
silently turned the `si_bse_debug` BerkeleyGW anchor's requested 4v4c into
4v8c, which the gate reported as an 0.0906 eV code regression that no branch
had caused. A widened window is a different calculation rather than a repair,
so widening is something you now ask for by name. **Do not set `snap` to make a
gate pass** — if a window lands inside a degenerate multiplet, that is the
finding.

## State the allocator on every timing

Every wall-time number you report carries the allocator it was taken under, in
the same sentence as the number: **BFC at `MEM_FRACTION=0.85`**, written
`BFC@0.85`, is the campaign default and what the existing timings are against.

This is not bookkeeping. BFC-versus-`platform` comparisons are confounded when
the allocator is unstated, both regimes have OOMed in contexts where the other
did not — BFC@0.85 OOMs one deck's 24³ ISDF path where 36³, 40³ and 48³ all
complete — and the allocator has not been qualified as a cost dial. A timing
without its allocator is not comparable to any other timing, including its own
re-run.

## One combined leg is the default

Pool capacity is the binding constraint, and the pool is shared with other
lanes. **Dispatch one leg that does all of a branch's GPU verification** — the
gate battery, the driver run and the red twin together — rather than several
small legs.

Each additional dispatch pays the queue tax again, adds a co-tenancy risk, and
under the interim one-leg-per-node rule buys concurrency only by taking whole
nodes. Combine first; split a leg only when it genuinely cannot be combined,
and say why when you do.

## Name your evidence directory, in the report

**Every lane report names the directory its evidence lives in**, spelled out as
a path, in the report itself. A number quoted without the workspace that
produced it is unverifiable the moment the lane ends.

This is not a filing preference. The evidence-purge lane found four orphan
workspaces on `/pscratch` that existed only because the reports leaning on them
never wrote their paths down — and one of those sits behind two committed
frozen references whose provenance headers name the fix sha but not the
workspace, so the references cannot be regenerated from what is written down.
So the rule has a second half: **a frozen reference's provenance header names
the workspace that produced it**, not just the commit.

Register the directory in **`/pscratch/sd/j/jackm/EVIDENCE_MANIFEST.md`**,
which is the index of what is load-bearing on scratch and what is not. A path
that is not in the manifest and not cited in a report is indistinguishable from
dead, and will eventually be treated as dead.

## File-boundary etiquette

Several lanes usually run at once, so before you edit a file that is not
obviously yours, find out who else is live — check `ListAgents` and the other
lanes' worktrees, and look at what their branches touch. Three
`KNOWN_FAILURES.md` merge conflicts were hand-resolved in one night, which is
why amendments are now one dated file each under `tests/known_failures/`
(see the index at the top of `tests/KNOWN_FAILURES.md`).

- **Take your own worktree.** Do not work in `/home/jackm/projects/lorrax` —
  it is the shared clone, and one lane's commit has already landed on
  another's ref there. Do not work in
  `/pscratch/sd/j/jackm/wt_int0808_bgw` at all.
- **Append a row to `RUNS_INFLIGHT.md` before you submit**, and strike it when
  the claim lands. Other lanes read it to avoid colliding with you.
- **Push feature branches freely; `main` needs the owner's approval.** Those
  are different rules, and conflating them into "never push" once cost a full
  night. `git merge-base --is-ancestor <commit> origin/main` is the only
  "landed" check.
