# Lane REG result — pointwise accuracy audit

**Measured first:** the 74-pair plan is honestly inside its pointwise
inverse-gap-envelope contract.  Its maximum certified spend over all served
frequencies is **0.765477 of the safety-reduced budget**, hence **0.612381 of
the unsafetied 1e-4 contract**.  `ENVELOPE_ERROR_SAFETY=0.8` is still applied
to the same combined envelope.  The exact selector installs one constraint
for every served omega; it does not select only at the combined-envelope peak.

Against `control_panes_24b`, achieved `sigma_c_kij_ev` error for the 74-pair
plan is below (max/RMS, meV).  This is the consumer-level frequency-resolved
measurement; the planner contract is explicitly not a physical-Sigma claim.

| omega (eV) | max | RMS | omega (eV) | max | RMS |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.206288 | 0.009789 | 2.75 | 0.099652 | 0.008212 |
| 0.25 | 0.138409 | 0.008143 | 3.00 | 0.106708 | 0.008779 |
| 0.50 | 0.146210 | 0.009127 | 3.25 | 0.094288 | 0.008472 |
| 0.75 | 0.131972 | 0.008069 | 3.50 | 0.121188 | 0.009390 |
| 1.00 | 0.088012 | 0.008281 | 3.75 | 0.133497 | 0.010328 |
| 1.25 | 0.139525 | 0.008619 | 4.00 | 0.099028 | 0.008772 |
| 1.50 | 0.091156 | 0.008664 | 4.25 | 0.167819 | 0.012718 |
| 1.75 | 0.075555 | 0.007775 | 4.50 | 0.150769 | 0.010670 |
| 2.00 | 0.082795 | 0.008480 | 4.75 | 0.182750 | 0.012079 |
| 2.25 | 0.086542 | 0.008076 | 5.00 | **0.235789** | **0.015113** |
| 2.50 | 0.077975 | 0.007821 | all omega | **0.235789** | **0.009575** |

The 115-pair scalar plan measured **0.195876 / 0.008834 meV** globally.  The
74-pair plan stays below that old global maximum at **19/21** omega points;
the exceptions are 0 eV (0.206288 meV) and 5 eV (0.235789 meV).  Both plans'
worst point is 5 eV.  Thus the 36% node reduction buys less over-delivery; it
does not expose an unbudgeted off-peak frequency, and changing the calibration
to force scalar-plan parity would silently redefine the existing contract.

Evidence: `/pscratch/sd/j/jackm/int_wide_sigma_20260831/regression` and
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/{control_panes_24b,scfix_regression_p4_20260831}`.
The prescribed CPU gate passed **136/136** in **86.27 s**.
The existing P=4/BFC@0.85 artifacts were reused; no new band-policy override
was introduced.  Branch: `fix/pointwise-accuracy-2026-08-31`.
