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

## What you choose

| choice | how to set it | why |
|---|---|---|
| $k_BT$ | `occ_smearing_width_ry` = the DFT `degauss`. Use 0.01 Ry at $N\ge 8$ and 0.02 Ry at $N=4$–6 (the Na 8³ and Fe 4³/8³ production decks) | the grid must resolve the Fermi surface: $k_BT$ of order the band dispersion across one k spacing (Marzari–Vanderbilt). Halve it when you double $N$ |
| QP window top | `ncond` = (highest band whose minimum over k lies below $E_F+E_{\rm win}$) − `nelec`, where `nelec` is the highest band occupied at any k | Σ diagonals are computed for bands $[0,$ `nelec + ncond`$)$. States above the window are a rigid tail ([self-consistency §2](../self_consistency.md#2-band-treatment)) |
| pair window | `nval` = the occupied bands whose maximum over k lies above $E_F-E_{\rm win}$ | it sets the lower edge of the ISDF pair-density window, not the bottom of the QP window. Do not freeze deep bands with `sc_frozen_core_bands`: on Fe 3s/3p the frozen law is off by hundreds of meV (CLAIMS 2859) |
| band sums | `number_bands`: all bands the NSCF has; at least 2–3× `nelec + ncond` | the χ0 and Σ sums; band-count convergence is a separate study |
| centroids | `python3 -m centroid.kmeans_cli N_mu --fit-window 0:B_sigma,0:number_bands`, with $N_\mu$ at 0.5–1.3× the numerical rank of that pair set | ISDF exchange error collapses in $N_\mu/r$ (CLAIMS 2860); `kmeans.out` prints the pool rank. See [drivers](../drivers.md) |
| head | `head_correction = no_local_fields` (direct charge head $S(\omega)$ with the Drude, Thomas–Fermi and Lindhard-cell terms), or `full` on a scalar deck | [metallic MPA screening](../theory/metallic-mpa-screening.md) owns the head model |
| SC head | `sc_head_update = off` (default: fixed DFT head on the DFT Fermi–Dirac state), or `dft_velocity` with `dipole.h5` | [self-consistency §7](../self_consistency.md#metals-direct-drude-head) |
| stop rule | `sc_tol_ev = 1e-3` for a 1 meV Fermi-window target | the default `1e-4` is below the Σ rule-set noise on metals; judge convergence by states near $E_F$ ([self-consistency §7](../self_consistency.md#7-metals)) |

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

## Worked deck: bcc Fe, 4³, QSGW within 5 eV of $E_F$

Scalar (charge-only) Fe, 35 bands, FD $k_BT$ = 0.02 Ry. The window (bands
up to `nelec + ncond`) covers $E_F$ + 5 eV.

```ini
[cohsex]
wfn_file = WFN.h5
centroids_file = centroids_frac.txt
kin_ion_file = kin_ion.h5
sys_dim = 3
bispinor = false
nval = 4
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
