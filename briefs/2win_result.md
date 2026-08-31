# 2WIN result — two-patch Sigma is not selectable on this base

**Measured verdict:** the requested `-26:-24,-5:+5 eV` deck did **not** produce a plan. A cold P=4/BFC@0.85 attempt spent **109.0 s** in the Sigma/planner stage (**127 s** `lx` wall), emitted **15** product-window census rows (**14 served, 1 refused**), and stopped before a Sigma sweep. The refusing row was `ω<E_F val:resonant`: `A/eta=206.3802`, `A/gamma_min=204.3667`, residual **8.531982e-3**, kappa_p99 **93625.5**, candidate family **none**. Consequently the achieved plan has **zero emitted `(window,tau)` pairs**, no selected ranks, no `sigma_mnk.h5`, and no finite QP energies to compare with BerkeleyGW. A third `-55:-52.5 eV` patch was not attempted.

## Census

The planner does not emit candidate rank/`n_tau` until the whole plan survives, so rank is honestly `--` for every row below rather than inferred from the cost law.

| product window | status | family | residual | kappa_p99 | A/gamma |
|---|---:|---|---:|---:|---:|
| `w>=EF cond:resonant` | served | measure_adapted_roq | 5.8949e-5 | 67.689 | 181.521 |
| `w>=EF cond:pole_tail` | served | noncrossing | 3.2399e-7 | 1.043 | -- |
| `w>=EF val:bulk` | served | noncrossing | 1.0993e-5 | 1.017 | -- |
| `w>=EF val:resonant[p1/2]` | served | measure_adapted_roq | 1.3767e-4 | 1.201 | 12.383 |
| `w>=EF val:resonant:negative_flank[p1/2]` | served | noncrossing | 1.2295e-5 | 1.145 | -- |
| `w>=EF val:resonant:negative_flank[p2/2]` | served | noncrossing | 1.8627e-6 | 1.093 | -- |
| `w>=EF val:pole_tail` | served | noncrossing | 8.1955e-7 | 1.040 | -- |
| `w<EF cond:bulk` | served | noncrossing | 3.7480e-6 | 1.038 | -- |
| `w<EF cond:resonant:negative_flank[p1/2]` | served | noncrossing | 1.1083e-6 | 1.057 | -- |
| `w<EF cond:resonant[p2/2]` | served | measure_adapted_roq | 4.5314e-5 | 1.132 | 16.385 |
| `w<EF cond:resonant:negative_flank[p2/2]` | served | noncrossing | 6.5693e-6 | 1.119 | -- |
| `w<EF cond:pole_tail` | served | noncrossing | 7.4994e-7 | 1.035 | -- |
| `w<EF val:resonant` | **refused** | **none** | **8.5320e-3** | **93625.5** | **204.367** |
| `w<EF val:state_tail` | served | noncrossing | 4.1802e-8 | 1.015 | -- |
| `w<EF val:pole_tail` | served | noncrossing | 2.4955e-4 | 1.036 | -- |

## Control and BGW consequence

The archived near-Fermi-only control (`sigma_omega=0:+5 eV`, eta/step 0.25 eV) did plan and run: **6 logical windows, 115 `(window,tau)` pairs, 146 dispatches/9 sweeps, 63.07 s Sigma, 89.22 s total**, and finite `eqp0.dat`/`eqp1.dat`. The current two-patch arm is therefore strictly worse operationally: it refuses after 109.0 s of Sigma-stage work and produces no observable. No LORRAX-vs-BGW QP-energy residual for bands 3--8 or the near-Fermi manifold exists for this arm; quoting the control's endpoint-clamped deep-band QPs as a two-window comparison would answer the wrong question.

Energy convention was fixed before the attempt: both requested patches are relative to LORRAX fixed-N `E_F = +1.64676169 eV`; the BGW and LORRAX EQP files use the shared absolute mean-field zero. The registered `tools/compare_bgw_gwjax.py` comparison is inapplicable because the candidate never wrote `sigma_diag.dat`; an attempted control-only invocation also refused the archived BGW log's 11-column rows before comparison, so no ad-hoc extraction replaced it.

## Provenance and caveat

Evidence: `/pscratch/sd/j/jackm/wt_2win_2026-08-31/tmp/two_sigma_windows_p4_20260831/{mpa_sigma.in,two_windows_p4.log,job_receipt.txt}`. JID **57781731**, step **62**, node `nid001021`, source **5f750a774985936f1e6fea0fb28c4348dd2dbca0**, 4 ranks/4 A100s, BFC@0.85. No ignored/unknown-key lines were present.

A strict-policy preflight first refused because the frozen WFN ends exactly at the named 48-band chi cutoff. The measured selector attempt inherited the supplied working control's diagnostic `LORRAX_BAND_DEGENERACY=snap`; it also warns that the 24-band Sigma edge splits a multiplet. Thus the negative selector result is useful feasibility evidence, not landing-grade physics evidence. Branch: `feat/two-sigma-windows-2026-08-31`.
