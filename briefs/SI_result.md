# Si merged-planner result — heavy lane

**PASS: Si now plans and runs.** P=4/BFC@0.85 on SHA `dd335067` produced **12 windows / 147 (window,tau) pairs**, completed **255 tau dispatches in 24 sweeps**, and wrote finite Sigma/QP artifacts. Total wall was **80.247 s** and the complete Sigma stage was **69.641 s**. The planner's in-memory receipt contains `plan_seconds`, but the driver does not print or persist it; therefore the achieved planner-only wall is not recoverable from this artifact and the honest measured bound is **<=69.641 s**, not a relabelled estimate.

| window | selected rung | rank / n_tau | residual / selected allowance | kappa p99 |
|---|---|---:|---:|---:|
| >=EF cond:resonant | on-demand ROQ | 30 | 1.87180e-3 / 1.89483e-3 | 23.5940 |
| >=EF cond:state_tail | on-demand ROQ | 9 | 9.68978e-6 / 3.27270e-5 | 1.05735 |
| >=EF cond:pole_tail | on-demand ROQ | 9 | **5.41001e-5 / 7.71374e-5** | 1.09479 |
| >=EF val:bulk | on-demand ROQ | 9 | 6.78271e-6 / 2.98200e-5 | 1.06322 |
| >=EF val:resonant | on-demand ROQ | 9 | 9.68100e-6 / 3.27183e-5 | 1.05461 |
| >=EF val:pole_tail | on-demand ROQ | 9 | 6.79386e-6 / 2.98311e-5 | 1.08522 |
| <EF cond:bulk | on-demand ROQ | 9 | 5.56711e-6 / 2.86044e-5 | 1.06213 |
| <EF cond:resonant | on-demand ROQ | 9 | 6.11337e-6 / 2.91506e-5 | 1.07068 |
| <EF cond:pole_tail | on-demand ROQ | 9 | 5.79501e-6 / 2.88323e-5 | 1.08401 |
| <EF val:resonant | on-demand ROQ | 27 | 2.23452e-3 / 2.25756e-3 | 21.7446 |
| <EF val:state_tail | on-demand ROQ | 9 | 4.99201e-6 / 2.80293e-5 | 1.06032 |
| <EF val:pole_tail | on-demand ROQ | 9 | 5.89995e-5 / 8.20368e-5 | 1.16722 |

The former blocker, `>=EF cond:pole_tail`, moved from residual `1.09254e-4` at kappa `1.39058` on the refusing tree to `5.41001e-5` at kappa `1.09479`; it is now served **2.13x below** its original apportioned target `1.15338e-4`. All 12 census rows report `status=served` and `candidate_family=measure_adapted_roq`.

The run wrote 128 state rows to each of `eqp0.dat` and `eqp1.dat`. The full-matrix effective-H fundamental gap is **0.97020 eV**, versus DFT **0.67666 eV** (correction **+0.29354 eV**). At Gamma, the `eqp1.dat` QP energies include bands 7-8 at **8.381527615 eV** and bands 9-10 at **11.384653405 eV**. The driver warns that 406/1024 Sigma(E_DFT) cells lie outside the inherited [-5,+5] eV grid and use its existing clamp policy; no accuracy dial was changed.

Merge resolution: the null-family energy-reference correction and pointwise MILP selector were kept; catalog code/tests from the marginal branch were not resurrected after `on-demand-only` deleted them. Thus every selected Si rule above was fitted from the run's measure. The mandated CPU gate passed **136 tests** in **172.01 s** (base stated 134).

Evidence: `/pscratch/sd/j/jackm/wt_si_2026-09-01/tmp/si_p4_20260901`. `si_p4_retry.log` SHA-256 `8cd8a9f801d27e2bdf52216cfeef2a699fe1fa8efd0546df13598d43585075bc`; `eqp0.dat` `88580c411519145b08f919649415b98fb132b2f17d4ac51916e320a19b53199a`; `eqp1.dat` `268c395bde0c8fa89fdd175b7c722ec29df063be6046a0c475770ec074c029bb`; `sigma_mnk.h5` `eddd98c5dae73e0e93a65346e078c357ef60e721ea835428be612523dd45ae7b`. Allocation JID 57804947 was attach-only and was not released. Branch: `test/si-2026-09-01`.
