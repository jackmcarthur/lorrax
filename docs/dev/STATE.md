# State of the tree — 2026-08-01

One page, dated. Supersedes the July handoffs
(`docs/dev/archive/HANDOFF_2026-07-28.md`, `.../HANDOFF_2026-07-29.md` — frozen
records, correct as of their dates). What changed for users coming from
origin/main: `UPGRADE_NOTES.md` (repo root). Binding rulings:
`docs/architecture/decisions.md`.

**Tree**: branch `fix/zq-band-gather-device-invariance`, ~210 commits ahead of
origin/main (a fast-forward), NOT pushed — the owner pushes. Measurement and
decision ledger: sandbox `/scratch2/08271/jackmc/lorrax_sandbox/CLAIMS.md`
(row numbers cited below).

## Certification scope

**CPU-Frontera: certified at P ≤ 64** under the required-FFI defaults and
`impl=mpi` transport — AQ 4962c/P=64 green (jobs 7877754/7877789, CLAIMS 3);
b300 P=16 A/B exact-0 on all five outputs under the new defaults vs both
pinned baselines (job 7885146, CLAIMS 34); full 4×4 pipeline parity
1.1e-13 eV (jobs 7884609/7884612, CLAIMS 1). gloo is banned at distributed
tiers (CLAIMS 4). **GPU (Frontera rtx): smoke-certified under required-FFI**
— committed cuFFT TU built clean, g108 defaults run rc=0 with eqp0/eqp1
exact-0 vs the CPU baseline, absent-library startup refusal verified (jobs
7885151/7885153, CLAIMS 35). **Perlmutter: unverified pending the outage**;
the head-wing-fix rescue patches exist only on the Perlmutter checkout
(data risk, parked with the owner).

## Pre-push checklist

One command: `tools/release_check.sh` (add `--with-allocation` for the
fastloop leg). Status with evidence:

| check | status | evidence |
|---|---|---|
| Five login AST suites (layering 68, crossfile 34, env_registry 9, env_grammar 46, fft_shardmap 4) | GREEN | release_check.sh run 2026-08-01 (login) |
| input-reference drift (`tools/gen_input_reference.py`) | GREEN | after the 2026-08-01 `ppm_probe_chi_reuse` generator fix |
| Origin-delta blob scan (>1 MB) + secrets grep | GREEN | release_check.sh run 2026-08-01: 0 blobs, 0 hits |
| fastloop mini-deck chain, both legs, NO gate exports | GREEN | job 7885145 (CLAIMS 34); certification 7884926/7884936 (CLAIMS 17) |
| b300 GW A/B under new defaults | GREEN, exact-0 | job 7885146 (CLAIMS 34) |
| GPU leg under required-FFI | GREEN | jobs 7885151/7885153 (CLAIMS 35) |
| Transverse-hoist merge certification | GREEN | job 7885137 (CLAIMS 32) |
| Full pytest suite triage | **PENDING** — `tests/KNOWN_FAILURES.md` (sibling workstream; pytest job 7885154 running at time of writing; this row stays DONE — tests/KNOWN_FAILURES.md (f485b5a), job 7885154: 735 pass / 24 triaged / 1 ship-listed known-fail| — |
| `git status` clean | PENDING — live workstreams hold uncommitted edits | release_check.sh reports the paths |

## Remainder and parked decisions

- **Priority-queue remainder** (agent-actionable): the open rows of sandbox
  `KNOWN_LORRAX_ISSUES.md` — kmeans/BSE single-axis μ-sharding, BSE
  ring/preview per-rank reads, eager-FFT debt, Σ_c full-frequency
  cross-check, probe-χ planner cheapening, cusolvermp hoist port, QE→pw2bgw
  certification.
- **Coulomb head slot — OPEN, needs an owner ruling, 2026-08-05.** The
  2026-08-05 consolidation was authorised to move the GW body-head slot from
  the Miller-(0,0,0) rule to `argmin_G |q+G|^2`, on the stated premise that
  the two coincide inside the first BZ so the change would be BSE-only. THE
  PREMISE IS FALSE and the change was NOT made. Measured (Frontera job
  7890613): over the full production q-list, `argmin` differs from
  Miller-(0,0,0) at **12 of 64 q** on the Si 4x4x4 fcc deck and **1 of 4** on
  the MoS2 mini deck. BGW wraps q to the fractional cube [-1/2, 1/2), which is
  the parallelepiped, NOT the Wigner-Seitz cell — on a non-orthogonal lattice
  those differ, and a wrapped q can sit outside the first BZ. So for GW this
  is a physics change, not a relabelling, and its gate ("value-level for 3D")
  could not have been met. It is also not a one-line change: the head TABLE is
  averaged AROUND the slot it is injected at, so the slot and the shift have
  to move together or the average lands on the wrong neighbourhood.
  `argmin` is retained where it was already correct and already load-bearing
  (`bse/vq_interp.py:1002`, the arbitrary-Q path, where Q is not on the grid
  at all). Deciding for GW needs a Sigma A/B of the slot+shift change on the
  Si deck; the harness for it is
  `/scratch2/08271/jackmc/coulomb_consolidation/ab/` (two frozen source arms,
  one sbatch, ~7 min/arm on one development node).

- **`bare_coulomb_cutoff` is documented three ways — REGISTERED, not fixed.**
  `docs/input_reference.md:53` says "None = ecutwfc";
  `manual/06_coulomb/6.3_cutoffs.md:17` says "Default E_cut, matching
  BerkeleyGW's `screened_coulomb_cutoff` default"; `gw_config.py:1167` has
  `None` and `gw_init.py:227` records a resolved `zeta_vcoul_cutoff` into
  provenance. These may all be consistent, but confirming that means tracing
  the resolution chain, which the 2026-08-05 Coulomb work did not touch.
  Noted rather than guessed at.

- **SUP decisions parked with the owner**: bispinor transverse rank-gate /
  rcond policy; the 512b μ-convergence non-monotonicity verdict; Perlmutter
  head-wing rescue (data risk); plus the standing owner queue at
  `/scratch2/08271/jackmc/lorrax_setup/wk_REL/OWNER_DECISIONS.md` (push,
  upstream jax PR, cublasmp package deletion, `ppm_probe_chi_reuse`
  default-flip after planner cheapening).
