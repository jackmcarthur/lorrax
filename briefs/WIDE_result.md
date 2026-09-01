# WIDE result — all three targets refuse before execution

| Deck target | Plan? | Window census | `(window, tau)` pairs | Observed planner interval | First refusal: apportioned target / residual / p99 |
|---:|:---:|:---:|:---:|---:|:---|
| `1e-4` | no | 14 served, 5 refused | none | 78 s | `w>=EF cond:resonant`: `1.195596e-4 / 4.259036e-3 / 2.287775e4` |
| `1e-3` | no | 18 served, 1 refused | none | 57 s | `w<EF val:resonant:positive_flank[p1/2]`: `0.5 / 6.480385e-4 / 1.075077` |
| `1e-2` | no | 18 served, 1 refused | none | 52 s | `w<EF val:resonant:positive_flank[p1/2]`: `0.5 / 6.480385e-4 / 1.075077` |

**Straight answer:** no tested deck target makes the two-window `-15..-0.25, 0..+15 eV` deck execute. The `1e-3` and `1e-2` rows refuse even though their reported residual is 771.6 times smaller than the reported apportioned target. This moves the blocker away from the wide crossing fit and isolates a selector/family-acceptance refusal: the row has `candidate_family=null` despite the small residual and well-conditioned `p99=1.075077`.

The intervals above are measured from the last pre-planner `vh_matrix` stage close to the planner exception (whole-second log timestamps); full self-consistent-driver walls to refusal were 249.4, 226.4, and 222.6 s. Every arm emitted 19 census rows. No arm wrote `sc_plan.pkl`, `sigma.h5`, `eqp0.dat`, or `eqp1.dat`, so no tau sweep ran and no honest Sigma-c/control or BerkeleyGW QP error exists. The registered comparator was therefore not invoked.

Evidence: `/pscratch/sd/j/jackm/wt_wide_2026-08-31/tmp/wide_sweep_20260831/{t1e-4,t1e-3,t1e-2}/launcher_final.log`. Each useful leg was P=4 on a distinct A100 node, BFC@0.85, SHA `568204acc8f8c37f8f0094636ec7984b28692c3d`, JID 57800954. The copied archive's three obsolete/ignored target/material/layout keys were removed so the logs contain no ignored-key confound; the archive's documented diagnostic degeneracy policy was retained because its 48-band WFN has no 49th band with which strict closure could be certified.

Prerequisites merged: usable-rank cap plus five-second planner. Fixed CPU gate: **134 passed** in 79.90 s. No execution/FFT code was touched.
