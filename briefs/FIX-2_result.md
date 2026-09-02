# FIX-2 result — SC anchor cycle and tightening rounds

Heavy fix lane. **Achieved:** the audit's six-map edge case now holds the
protected mask at `[1,1]`, the upper energy at **1.061267292017 eV**, and the
retained off-diagonal at **0.200000000000 eV** on all six maps. A genuinely
drifted band at **1.20 eV** loses protection beyond the tested **0.125 eV**
deadband (half the existing 0.25 eV omega-grid step).

The SC map still rebuilds every window from its current spectrum and current
fixed-N Fermi level. It now carries only the preceding protected-band decision
for a Schmitt boundary; the margin is derived as the larger of half the
existing omega-grid step and the existing occupation-smearing width. No deck
dial was added.

The compounding planner reached selection call **4** and accepted after two
tightened fits. Achieved allowances were **1.6e-4 -> 7.2e-5 -> 3.24e-5**.
The loop now raises only after all three advertised tightened stages have been
tried; its comment says five selection stages and three compounded re-fits.

Evidence: exact red twins in `tests/test_band_partition.py` and
`tests/test_delivered_windows.py`; targeted result **2 passed in 5.47 s**;
affected suites **37 passed in 6.33 s**; prescribed CPU gate **135 passed,
8 warnings in 91.86 s**. No GPU leg is owed: both changes and regressions are
host planner/partition logic under the four-GPU rule's CPU-cell exemption.

Branch: `fix/sc-anchor-cycle-2026-08-31`.
