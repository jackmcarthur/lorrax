# Planner-materials result — heavy lane

**Numbers first.** Current SHA `2ed82a64` ran P=4/BFC@0.85 on gapped Si (DFT gap 0.67666 eV), 24 bands, 192 centroids, and a complete 12-pole fit. The planner formed 12 local product-window candidates in a 54.5 s Sigma stage, then **REFUSED** the global plan: best delivered-error cost `4.80478e7` versus budget `1.95049e7` (**2.463x over**), pair ceiling 500. Therefore completed-plan windows/pairs are **none/none**, the executor did not run, and this delivered arm produced no QP energies. The same input/fit's incumbent panes run had previously completed with 544/544 parsed numbers finite in each of `eqp0.dat` and `eqp1.dat`; that is a control, not a delivered-plan pass.

| window | kind | A/gamma | cells | mass share | locally served family | rank/nodes | residual / apportioned target | kappa p99 | local target |
|---|---|---:|---:|---:|---|---|---:|---:|---|
| >=EF cond:resonant | crossing | 40.019 | 641 | 0.44661% | measure-adapted ROQ | not emitted* | 5.83180e-3 / 8.60356e-3 | 22.2261 | met |
| >=EF cond:state_tail | sign-negative | — | 636 | 0.29620% | noncrossing lookup | not emitted* | 7.42641e-6 / 1.29725e-2 | 1.05241 | met |
| >=EF cond:pole_tail | sign-negative | — | 648 | 33.31495% | noncrossing lookup | not emitted* | 1.09254e-4 / 1.15338e-4 | 1.39058 | met; **global blocker** |
| >=EF val:bulk | sign-positive | — | 647 | 16.92601% | noncrossing lookup | not emitted* | 1.79534e-4 / 2.27015e-4 | 1.27512 | met |
| >=EF val:resonant | sign-positive | — | 476 | 0.00407% | noncrossing lookup | not emitted* | 1.80451e-5 / 5.0e-1 | 1.09758 | met |
| >=EF val:pole_tail | sign-positive | — | 562 | 0.21387% | noncrossing lookup | not emitted* | 6.41829e-5 / 1.79667e-2 | 1.36241 | met |
| <EF cond:bulk | sign-negative | — | 649 | 26.04170% | noncrossing lookup | not emitted* | 1.02602e-4 / 1.47550e-4 | 1.23644 | met |
| <EF cond:resonant | sign-negative | — | 471 | 0.00571% | noncrossing lookup | not emitted* | 9.82309e-6 / 5.0e-1 | 1.09259 | met |
| <EF cond:pole_tail | sign-negative | — | 561 | 0.31061% | noncrossing lookup | not emitted* | 5.66362e-5 / 1.23708e-2 | 1.35437 | met |
| <EF val:resonant | crossing | 37.093 | 643 | 0.48786% | measure-adapted ROQ | not emitted* | 4.86525e-3 / 7.87612e-3 | 21.0703 | met |
| <EF val:state_tail | sign-positive | — | 632 | 0.11784% | noncrossing lookup | not emitted* | 2.36134e-6 / 3.26062e-2 | 1.05124 | met |
| <EF val:pole_tail | sign-positive | — | 649 | 21.83457% | noncrossing lookup | not emitted* | 1.52426e-4 / 1.75981e-4 | 1.44168 | met |

\* The refusal happens before final selection and the failure receipt does not emit candidate rank/node count; reporting one would be fabrication. It also does not emit planner-only wall time, so 54.5 s is the measured Sigma-stage upper bound (full step 72 s), not relabelled as an exact planner time.

## Generalization verdict

- **Crossing law:** no valid material-specific slope can be fitted. There are only two Si crossing supports (A/gamma 37.093 and 40.019), and refusal prevented observing selected node counts. The log did load shipped fallback candidates of 48 nodes at catalog A=40 and 66 at A=60 (affine slope 0.9 with intercept 12; through-origin least-squares slope 1.131), but those catalog rules were not the locally preferred ROQ rules and are not a Si fit. Thus `2.02` is neither confirmed nor falsified here.
- **Sodium-derived constants:** the noise cut behaved sensibly locally: sign-definite kappa p99 is 1.051-1.442 and crossing is 21.07-22.23; all 12 local candidates met their apportioned targets. Rank margin 1.5 and `NODES_PER_RANK=3` cannot be judged because no selected ranks were emitted. The observed failure is instead the **global error-budget selector**, at 2.463x over despite benign conditioning; its blocker is sign-definite and carries 33.315% of mass.
- **Effective dimensionality is qualitatively different:** Si's top three windows carry 81.191% of delivered mass, top four 98.117%, with window participation ratio `1/sum(p_i^2)=3.919`. Frozen sodium p0 has top three 99.743% and participation ratio 1.320 (one window alone 86.009%). Si therefore spreads consequential mass across roughly three times as many windows. A numerical rank of Si's weighted subspace was not persisted on refusal, so no rank comparison is claimed.

CPU gate: **134 passed** in 82.63 s. Evidence: `/pscratch/sd/j/jackm/wt_planner_materials_2026-08-31/tmp/planner_materials_si` (`planner_p4.log` SHA-256 `1319262889f9eff239aa2325ab9c89a5e51b75c2dfd49990ab6424cff2972adf`; `cpu_gate.log` SHA-256 `4f6f0d2e8f91b299390e27c82846b021d63f0dc2956a253e1c08a2cf4d847dda`). Exact owned JID 57800531 was cancelled after harvest; the unavailable stated JID 57781731 was untouched. Branch: `study/planner-materials-2026-08-31`.
