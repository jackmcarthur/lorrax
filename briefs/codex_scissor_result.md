# Sodium SC scissor audit

| Sigma half-width (eV) | valence n / w | valence alpha / beta / RMSE (eV) | crossing n / w | conduction n / w | conduction alpha / beta / RMSE (eV) |
|---:|---:|---:|---:|---:|---:|
| 5  | 0 / 0 | 1.000000 / +0.000000 / 0 | 58 / 1024 | 0 / 0 | 1.000000 / +0.000000 / 0 |
| 10 | 0 / 0 | 1.000000 / +0.000000 / 0 | 58 / 1024 | 0 / 0 | 1.000000 / +0.000000 / 0 |
| 15 | 0 / 0 | 1.000000 / +0.000000 / 0 | 58 / 1024 | 174 / 3072 | 1.000125 / +0.660598 / 0.111241 |
| 20 | 0 / 0 | 1.000000 / +0.000000 / 0 | 58 / 1024 | 406 / 7168 | 1.175748 / -0.675318 / 0.397927 |

`n` is the number of star-wedge `(k,band)` rows (29 k rows); `w` is their
full-512-point-BZ weight.  The crossing pair is in range but deliberately
excluded from both regressions, so it has no alpha/beta/RMSE.

## Window required by this metal

The all-k rule first admits the crossing pair (bands 9--10) at **4.46452
eV**.  The conduction fit first becomes well determined at **11.60594 eV**,
when the complete bands 11--16 manifold enters; the 15 eV row measures that
case.  Bands 17--20 enter at 17.19267 eV and all active conduction bands
11--24 are direct at **19.27269 eV**, measured by the 20 eV row.  Therefore
this 24-band Na calculation needs a **20 eV half-width** to avoid conduction
scissoring.

No valence band enters through 20 eV.  Bands 7--8 first enter at 25.10131 eV,
but a clean complete 2p manifold (bands 3--8) needs **25.29042 eV**; bands
1--2 need 53.80460 eV.  This does not make the consumed valence energies
identity: the SC policy intentionally preserves their displacement from the
candidate Fermi level, while the logged valence regression is diagnostic.

## Pairing and three-way split

The split is correctly placed at valence 1--8 / crossing 9--10 / conduction
11--24.  At 15 eV, independent sorting changes 82 of 174 raw band-label
assignments, mostly Kramers/multiplet permutations.  Seven QP sources sit
outside the global fit mask, but six are the locally degenerate bands 17--18.
The seventh is band 9 paired to conduction rank 12 at the single touching k:
DFT bands 9--12 are degenerate within 4.3e-14 eV there, and their four QP
diagonals span only 0.0715 meV.  The same single harmless crossing-source
permutation occurs at 20 eV.

Band-index pairing at 15 eV gives alpha=1.000841, beta=+0.656373 eV and
RMSE=0.113110 eV, versus the current 1.000125, +0.660598 eV and 0.111241 eV.
Its applied tail differs from current sort-and-pair by 10.750 meV max / 6.614
meV weighted RMS.  At 20 eV the laws agree to 3.2e-6 in alpha and 0.037 meV
in beta, and the fit has no QP-energy effect because every conduction band is
direct.  The measured crossing is thus an energy-rank degeneracy, not an
incorrectly placed Fermi manifold; changing the pairing would not cure the
observed extrapolation error.

## Affine adequacy and decision

The 15 eV affine law is not adequate for the omitted bands 17--24.  Against
their direct 20 eV `eqp0` values, the actually consumed 15 eV scissored QP
energies differ by **3.001078 eV max / 1.478967 eV weighted RMS**, with
-1.269637 eV bias.  The shared direct bands 11--16 move only 0.108477 eV max /
0.011514 eV weighted RMS between the two controls.  Identity is worse on the
tail (3.664152 eV max / 2.078261 eV weighted RMS), so deleting the correction
is not a solution either.

The 20 eV conduction residual is structured by band pair: weighted biases
for bands 11--12, 13--14, 15--16, 17--18, 19--20, 21--22, and 23--24 are
+0.354, -0.020, -0.344, -0.377, -0.188, +0.050, and +0.525 eV.  A quadratic
reduces RMSE from 0.397927 to 0.191714 eV (51.8%) but worsens max residual
from 0.963100 to 1.000621 eV, and there is no direct QP control beyond 20 eV
on which to validate its extrapolation.  No higher-order law and no new dial
are justified.  **Production code is unchanged; widen this deck to 20 eV.**

## Evidence and qualification

Evidence is in
`runs/Na/02_soc48b_qsgw_mpa/60_sc_delivered_20260831/codex_scissor_audit_p4_20260831/`:
four P=4 receipts (JID 57772354), exact fit-input NPZ files, `eqp0_iter0000.dat`
controls, and `analysis.json`.  The arms copied the supplied deck, changed
only half-width and `sc_max_iter=1`, and used the existing pane/product-window
path (no direct-pair fallback).  They used the supplied diagnostic
`LORRAX_BAND_DEGENERACY=snap` because the 48-band WFN has no band 49 with
which to certify its upper multiplet edge.  Runtime source `5bc76402` includes
the supplied metal static-window prerequisite; its `src/gw/scissor.py` SHA256
is byte-identical to this branch (`80624628...`).

Focused CPU census: 64 passed, 1 expected xfail.  One inherited static
source-text test fails because base `263ab34a` removed the exact comment it
searches for; no numerical scissor test failed.
