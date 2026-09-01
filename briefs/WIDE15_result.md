# WIDE15 result — 2026-09-01

**REFUSED:** 12 product windows were censused; 10 were served and 2 crossing
windows refused.  No plan was selected, so selected `(window,tau)` pairs = 0,
the Sigma sweep did not start, and no QP or accuracy comparison exists.
Planning occupied approximately **63 s** from the last timestamped pre-planner
step (23:07:41) to refusal (23:08:44); the refusal path does not emit the exact
planner timer.  The full SC-map attempt was 233.1 s, including a separately
reported 59.254 s live MPA fit.

| Window | Family/status | Rank | Residual | kappa p99 |
|---|---|---:|---:|---:|
| `w>=EF cond:resonant` | null / REFUSED | n/e | 1.044684e-3 | 103.122 |
| `w>=EF cond:state_tail` | measure_adapted_roq / served | n/e | 3.597414e-5 | 1.01369 |
| `w>=EF cond:pole_tail` | measure_adapted_roq / served | n/e | 1.917843e-4 | 1.16420 |
| `w>=EF val:bulk` | measure_adapted_roq / served | n/e | 9.434173e-5 | 1.01467 |
| `w>=EF val:resonant` | measure_adapted_roq / served | n/e | 8.681514e-5 | 1.06081 |
| `w>=EF val:pole_tail` | measure_adapted_roq / served | n/e | 2.129103e-5 | 1.31299 |
| `w<EF cond:bulk` | measure_adapted_roq / served | n/e | 1.681562e-4 | 1.06219 |
| `w<EF cond:resonant` | measure_adapted_roq / served | n/e | 6.106060e-5 | 1.06396 |
| `w<EF cond:pole_tail` | measure_adapted_roq / served | n/e | 2.013582e-5 | 1.32688 |
| `w<EF val:resonant` | null / REFUSED | n/e | 1.229704e-3 | 26.9059 |
| `w<EF val:state_tail` | measure_adapted_roq / served | n/e | 1.087079e-5 | 1.11185 |
| `w<EF val:pole_tail` | measure_adapted_roq / served | n/e | 1.130164e-4 | 1.16855 |

`n/e` means not emitted: the refusal census records family/residual/kappa but
not the fitted rank.  The first refusal row was `w>=EF cond:resonant`, with
`A/gamma_min=123.494`, `A/eta=124.711`, apportioned target `1.195596e-4`,
residual `1.044684e-3`, and kappa p99 `103.122`.  A second census refusal was
`w<EF val:resonant`, with `A/gamma_min=73.4263`, target `4.931288e-4`, residual
`1.229704e-3`, and kappa p99 `26.9059`.

The prescribed CPU gate passed **136/136** in 172.02 s; the retained
on-demand factor-reference conflict cell then passed **1/1** in 3.29 s, giving
the assembled 137-cell set without a failing cell.  Conflict resolution kept
the zero-referenced/pole-centred factor rule and per-frequency envelopes,
removed every catalog path, and rejected the marginal branch's incompatible
table-walk test/code while retaining the on-demand-only route.  P=4 used four
A100-40GB ranks, BFC@0.85, exact source `289b208a`.

Evidence:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/60_sc_delivered_20260831/test_wide15_20260901/`
(`lx_launch.log`, `job_receipt.txt`, live MPA artifacts).  Branch:
`test/wide15-2026-09-01`.
