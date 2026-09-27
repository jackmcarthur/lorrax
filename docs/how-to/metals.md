# Metals: a GW or QSGW run

This page turns a request of the form "QSGW on *metal* within $E_{\rm win}$ of
$E_F$ on an $N^3$ grid" into a deck. Key meanings are in the
[input reference](../input_reference.md); the map, its stop rules and the SC
head routes are in [self-consistency](../self_consistency.md); the screening
theory, including the $q\to0$ head, is in
[metallic MPA screening](../theory/metallic-mpa-screening.md). This page does
not restate them.

## Fixed rules for every metal

- **Occupations are Fermi–Dirac, always.** `occ_smearing_width_ry` is $k_BT$
  and must equal the DFT smearing (`smearing = 'fd'`, `degauss` = $k_BT$ in
  QE). Every other smearing family refuses (`GATE metal_occupations_fermi_dirac`).
  The code classifies the material from the `WFN.h5` occupations; no key sets
  it.
- **W is the shared-pole model.** `compute_mode = mpa`,
  `sigma_w_model = shared_pole`. GN-PPM refuses metals
  (`GATE gn_ppm_refuses_metals`). Elementwise MPA (`sigma_w_model = mpa`) is a
  comparison only, and it refuses time-reversal-broken metals
  (`GATE mpa_ordered_metal`).
- **One band support.** A band is in a χ or Σ branch iff its weight ($f$ or
  $1-f$) is resolved in float64, $|w|\ge 2^{-53}$, i.e.
  $|E-\mu|\le 53\ln2\,k_BT$. It is the same in the one-shot and every SC map,
  so SC map 0 is the one-shot. There is no key.

## Inputs

`WFN.h5` from a QE NSCF on the full $N^3$ grid with `smearing = 'fd'`
([inputs from DFT](../preprocessing.md)), `kin_ion.h5`, `dipole.h5` in the run
directory (the [dipole driver](../drivers.md); the metal head reads its
velocities, and a missing file refuses) and a centroid table. Read $E_F$ and
the band energies from the NSCF output; a one-shot's `eqp0.dat` and the
`E_F = ... (fixed-N mu)` line of `gwjax.out` give the same numbers at $k_BT$.

## What you choose

| choice | how to set it | why |
|---|---|---|
| $k_BT$ | `occ_smearing_width_ry` = the DFT `degauss`. Use 0.01 Ry at $N\ge 8$ and 0.02 Ry at $N=4$–6 (the Na 8³ and Fe 4³/8³ production decks) | the grid must resolve the Fermi surface: $k_BT$ of order the band dispersion across one k spacing (Marzari–Vanderbilt). Halve it when you double $N$ |
| QP window top | `ncond` = (highest band whose minimum over k lies below $E_F+E_{\rm win}$) − `nelec`, where `nelec` is the WFN's occupied-band boundary (`max(ifmax)`; `kmeans.out` prints it as `occupied-band boundary`) | Σ diagonals are computed for bands $[0,$ `nelec + ncond`$)$. The edge may not split a degenerate multiplet at any k (1 meV tolerance; `BandWindowDegeneracyError` refuses), so raise `ncond` to the top of the multiplet. States above the window are a rigid tail ([self-consistency §2](../self_consistency.md#2-band-treatment)) |
| pair window | `nval` = the occupied bands whose maximum over k lies above $E_F-E_{\rm win}$ | it sets the lower edge of the ISDF pair-density window, not the bottom of the QP window. Do not freeze deep bands with `sc_frozen_core_bands`: on Fe 3s/3p the frozen law is off by hundreds of meV (CLAIMS 2859) |
| band sums | `number_bands`: the bands the NSCF has | the χ0 and Σ sums; band-count convergence is a separate study |
| centroids | select on the Σ pair set, `--fit-window 0:B,0:number_bands` with `B = nelec + ncond`. First run `python3 -m centroid.kmeans_cli` with a large request and read `achieved numerical rank=r` in `kmeans.out`; then select $N_\mu$ between $0.5r$ and $1.3r$. The rank line appears only when the snapped candidates outnumber the request (`pruning: not applied` otherwise), so keep the probe request below the FFT-grid point count | the ISDF exchange error falls with $N_\mu/r$: RMS ≤ 1 meV near $0.5r$, max ≤ 1 meV near $1.3r$ (CLAIMS 2860). See [drivers](../drivers.md) |
| head | `head_correction = no_local_fields` (direct charge head $S(\omega)$ with the Drude, Thomas–Fermi and Lindhard-cell terms), or `full` on a scalar deck | [metallic MPA screening](../theory/metallic-mpa-screening.md) owns the head model |
| SC head | `sc_head_update = off` (default: fixed DFT head on the DFT Fermi–Dirac state), or `dft_velocity` with `dipole.h5` | [self-consistency §7](../self_consistency.md#metals-direct-drude-head) |
| stop rule | `sc_tol_ev = 1e-3` | 1 meV is the reproducibility the default `sigma_quadrature_eps` is chosen for; judge convergence by states near $E_F$ ([self-consistency §7](../self_consistency.md#7-metals)) |

State `sys_dim = 3`. `fermi_reference = mp1_fixed_n` is required on a metal
(the name is historical; with Fermi–Dirac it is the fixed-N μ).

## What the code derives

- μ at fixed electron count from each map's spectrum; μ is never mixed.
- The Σ(ω) grid from the protected bands (`sigma_out_of_grid = cover`, the
  default); `sigma_omega_min_ev`/`sigma_omega_max_ev` are optional minimum
  extents.
- The shared-pole support ladders, the Σ quadrature rules
  (`sigma_quadrature_eps`, default 3e-5) and the SC pad-then-hold windows.
- The q→0 head from the tetrahedron Fermi-surface weights and `dipole.h5`
  when present.

## What refuses

| deck | gate |
|---|---|
| `compute_mode = gn_ppm`, `cohsex` or `hl_ppm` | `gn_ppm_refuses_metals`, `fractional_occupations_require_mpa` |
| no `occ_smearing_width_ry`, or `occ_smearing_family` | required; `metal_occupations_fermi_dirac` |
| `occ_broadening > 0` beside `occ_smearing_width_ry` (MP1 smeared head) | `metal_sc_head_update_disabled` |
| `sc_head_update = parallel_transport`, or `dft_velocity` with `full` on a bispinor deck | `metal_sc_head_update_disabled` |
| `sc_head_update = interband_commutator` | `sc_head_interband_commutator_insulator_only` |
| `head_correction = full` on a time-reversal-broken shared-pole store | `shared_pole_head_ordered` |
| `sigma_w_model = mpa` on a time-reversal-broken metal | `mpa_ordered_metal` |
| `occupation_window_threshold` | retired |

## Worked deck: bcc Fe, 4³, QSGW within 10 eV of $E_F$

Charge-only Fe on a spinor WFN, 35 bands, $k_BT$ = 0.02 Ry. The one-shot
gives $E_F$ = 18.01 eV and `nelec` = 18. Bands 19–24 have their minimum
below $E_F$ + 10 eV and band 25 does not, but bands 24–26 are one multiplet
at Γ (0.7 meV), so `ncond` = 8. Occupied bands 9–18 reach above
$E_F$ − 10 eV, so `nval` = 10. Centroids are selected on
`--fit-window 0:26,0:35`.

```ini
[cohsex]
wfn_file = WFN.h5
centroids_file = centroids_frac.txt
kin_ion_file = kin_ion.h5
sys_dim = 3
bispinor = false
nval = 10
ncond = 8
number_bands = 35
compute_mode = mpa
sigma_w_model = shared_pole
occ_smearing_width_ry = 0.02
fermi_reference = mp1_fixed_n
head_correction = no_local_fields
qp_solver = self_consistent
sc_tol_ev = 1e-3
```

A one-shot G0W0 is the same deck with `qp_solver = one_shot_dft`.
