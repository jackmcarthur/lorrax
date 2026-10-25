# MoS₂ with spin–orbit coupling

`6a-SOC` Hands-on GW and BSE calculations alongside the BerkeleyGW 2026 workshop.
{: .page-summary }

## Overview {: #overview }

In this exercise, we will compute the quasiparticle band structure and optical
absorption of monolayer MoS₂ with spin–orbit coupling (SOC). We use the fully
relativistic mean field from the [2026 BerkeleyGW workshop](https://workshop.berkeleygw.org/home)
and add parallel LORRAX directories beside its plane-wave calculations. The
link scripts, numbered stages and output filenames follow the original exercise.

Both tracks interpolate from a 3 × 3 × 1 coarse grid to a shifted 8 × 8 × 1
fine grid. This comparison is underconverged, which leads to some differences
in the final results between the plane-wave code and LORRAX. However, the
[figures below](#interpolation-validation) show that at converged k-grid sizes
the two are very similar.

The [example directory](https://github.com/jackmcarthur/lorrax/tree/main/examples/MoS2_SOC_2026)
describes the differences between the main plane-wave package and LORRAX.

## Goals {: #goals }

1. Follow fully relativistic, two-component DFT wavefunctions through GW and BSE.
2. Run LORRAX using a directory layout familiar from the BerkeleyGW workshop.
3. Compare band structures and absorption spectra between the two codes.
4. Follow the exciton bands from Γ to K.

## Settings {: #prerequisites }

The calculations use a [built LORRAX checkout](../using/installation.md), one
128-core CPU node for Quantum ESPRESSO and the plane-wave reference, and one
four-GPU node for LORRAX.

| Setting | Value |
|---|---|
| Mean field | Fully relativistic Mo/S ONCV pseudopotentials; two spinor components; 45 Ry wavefunction cutoff |
| Coarse GW grid | 3 × 3 × 1, four stored k-points |
| GW sums | 200 total bands; band extrapolation disabled |
| ISDF points | 1,000 requested, 996 written with complete spatial symmetry orbits |
| QP Hamiltonian | 48 bands: 26 occupied and 22 conduction |
| Frequency model | Hybertsen–Louie plasmon pole; full head correction |
| Fine BSE grid | 8 × 8 × 1, shifted by (0.47, 0.37, 0) in grid-step units |
| Fine BSE bands | Six valence and eight conduction; all 3,072 Tamm–Dancoff roots |
| Spectrum | 0.15 eV Gaussian width; 0.01 eV spacing |
| Exciton bands | Γ-centered 24 × 24 × 1 integration grid; six valence and eight conduction bands; 128 roots at 37 Q points on Γ–K |

## Steps {: #instructions }

Copy the workshop inputs and the LORRAX directories into a new working directory:

```bash
export LORRAX=/path/to/lorrax
python3 "$LORRAX/examples/MoS2_SOC_2026/perlmutter/prepare_meanfield.py" \
  --tutorial "$SCRATCH/MoS2_SOC_2026/6a-SOC" init
```

The calculation then runs in four parts. The README in each directory gives its
commands and what to check.

1. **Mean field.** Quantum ESPRESSO SCF and NSCF with the workshop inputs,
   converted to `WFN.h5` (`perlmutter/01-run-meanfield.sh`). Two extra NSCF
   files add spare bands and the full nine-point coarse grid.
2. **Plane-wave reference.** The workshop's own epsilon, sigma, inteqp, kernel
   and absorption steps (`perlmutter/03-run-reference-cpu.sh`), which the
   comparison plots read.
3. **LORRAX GW.** The ISDF points, kinetic/ionic matrix and velocity matrix
   (`2-gw/3-lorrax_fully_rel/1-epsilon`), then screening and self-energy
   together (`2-sigma`), which also exports `eps0mat.h5` and `epsmat.h5`, and
   the band structure by [`htransform`](../utilities/htransform-bands.md) (`3-bs`).
4. **LORRAX BSE.** The `bsemat.h5` export (`3-bse/3-lorrax_fully_rel/1-kernel`)
   and the BSE with its absorption spectra (`2-absorption`), followed by
   the exciton dispersion (`3-exciton-bands`).

`bash 00-run-lorrax.sh` runs parts 3 and 4 in order on a four-GPU allocation
and makes the plots. The [For BerkeleyGW users](../using/berkeleygw-users.md)
page maps the workshop's `.inp` keys to the `gw.in` inputs.

## Exciton bands {: #exciton-bands }

The final LORRAX stage follows 128 finite-momentum exciton levels from Γ to K:

```bash
cd 3-bse/3-lorrax_fully_rel/3-exciton-bands
bash 01-run_exciton_bands.run
source ../../../2-gw/3-lorrax_fully_rel/lorrax_env.sh
lorrax_python 02-plot_exciton_bands.py
```

The electron-hole sum retains six valence and eight conduction bands on a
Γ-centered 24 × 24 × 1 integration grid. H-transform supplies the electron
at k + Q; the hole remains at k. Screening and the QP Hamiltonian still
come from the same 3 × 3 × 1 coarse calculation.

The 37-point path takes eight steps from Γ to `(0.003, 0.003, 0)`, then
28 to `K = (1/3, 1/3, 0)`, in fractional reciprocal coordinates. Its largest
Q spacing is about four times smaller than on the former Γ–K segment.
These are actual finite-Q solves with a 0.25 meV Davidson residual budget;
the plot connects their ordered energies with straight segments. Arbitrary-Q
exchange refits let the denser path retain the same integration grid.
Four GPUs were tested and are recommended for this stage. The scan took
34 min 43 s on four Perlmutter A100 80 GB GPUs, including its preparation
and certification, with a largest reported stage device peak of 16.35 GB
per rank. This does not establish a minimum GPU count.

The plot shows the lowest 128 levels only on Γ–K and only up to 3.0 eV.
Its lower panel plots $E_j(Q)-E_j(Γ)$ in meV for ordered levels 3 and 4,
using 11 actual momenta through |Q| = 0.1056 Å⁻¹. Level 3 shows resolved
quadratic curvature over that window, while level 4 has a strong linear
contribution near Γ, as permitted by two-dimensional long-range exchange.
Shifts below 0.5 meV remain unresolved conservatively; energies alone do not
identify longitudinal/transverse or bright/dark character.

<figure style="width:3in; max-width:100%; margin:1em 0;" markdown="1">

[![The lowest 128 LORRAX exciton levels along Gamma-K up to 3.0 eV, and energy shifts of ordered levels 3 and 4 near Gamma.](../assets/figures/tutorial-mos2-exciton-bands.png){ width="288" }](../assets/figures/tutorial-mos2-exciton-bands.png)

<figcaption style="font-size:0.86em; line-height:1.45;" markdown="1">

The dashed line marks the matched independent-particle threshold where it
falls within the displayed range. The root count does not guarantee continuum
coverage at every Q: the highest retained level reaches it at 15 of 37
momenta. The lower panel subtracts each level's own Γ energy; dots are
computed values from the same scan, with no fitted curves.
[Download the PDF](../assets/figures/tutorial-mos2-exciton-bands.pdf).

</figcaption>
</figure>

These are illustrative results: previous matched 24 × 24 and 30 × 30 controls
changed the lowest eight levels by up to 7.65 meV at shared momenta. Denser Q
sampling improves the displayed dispersion, while coarse GW and integration-grid
convergence require further calculations. The tutorial uses the 1 meV
**visualization** exchange check; the driver's **reference** default remains
unchanged. The [stage README](https://github.com/jackmcarthur/lorrax/blob/main/examples/MoS2_SOC_2026/3-bse/3-lorrax_fully_rel/3-exciton-bands/README.md)
explains the reciprocal coordinates, residual budgets and certification.

## Results {: #results }

<div style="display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start;" markdown="1">

<figure style="width:3in; max-width:100%; margin:0;" markdown="1">

[![MoS₂ DFT bands in light gray, LORRAX GW in blue and plane-wave GW in orange on Γ–M–K–Γ, with dots at the coarse k-points.](../assets/figures/tutorial-mos2-soc-bands.png){ width="288" }](../assets/figures/tutorial-mos2-soc-bands.png)

<figcaption style="font-size:0.86em; line-height:1.45;" markdown="1">
**Band structure.** Light gray: QE DFT; blue: LORRAX GW; orange: plane-wave GW.
Dots mark the 3 × 3 × 1 coarse k-points on the path, where the QP energies are
computed directly; between them they are interpolated. Each dataset is aligned
to its own VBM.
</figcaption>
</figure>

<figure style="width:3in; max-width:100%; margin:0;" markdown="1">

[![MoS₂ optical absorption: LORRAX in blue and the plane-wave code in orange, with solid BSE and dashed independent-particle curves.](../assets/figures/tutorial-mos2-soc-absorption.png){ width="288" }](../assets/figures/tutorial-mos2-soc-absorption.png)

<figcaption style="font-size:0.86em; line-height:1.45;" markdown="1">
**Absorption (0–5 eV).** Blue: LORRAX; orange: plane-wave code. Solid: BSE;
dashed: independent particle. Both use the workshop's shifted 8 × 8 × 1 grid.
</figcaption>
</figure>

</div>

| Quantity (eV) | LORRAX | Plane wave |
|---|---:|---:|
| Coarse-grid EQP1 minimum gap | 3.208396 | 3.123690 |
| Path EQP1 indirect gap | 2.872913 | 3.071206 |
| Lowest fine-grid BSE state | 2.511020 | 1.591552 |
| Largest interacting spectral peak | 3.18 | 2.60 |

### Why the onsets differ {: #onset }

The lowest excitons differ by 0.92 eV, while the fine-grid quasiparticle gaps
differ by only 0.075 eV (3.233 eV in LORRAX, 3.159 eV in the plane-wave code).
The difference is therefore almost entirely exciton binding. On the 3 × 3 × 1
coarse grid the two codes agree, with binding energies of 1.544 and 1.536 eV.
The 2D exciton envelope cannot be resolved on so few k-points, and the coarse
binding is far too large.

LORRAX interpolates the screened interaction to the fine grid by its
[short- and long-range split](../theory/w-range-splitting.md), and its binding
falls toward its converged value as the BSE grid grows. The plane-wave dual-grid
interpolation keeps the coarse binding, 1.567 eV on the shifted 8 × 8 × 1 grid.

| BSE grid, same 3 × 3 × 1 GW | 3 × 3 | 6 × 6 | 8 × 8 shifted | 12 × 12 | 24 × 24 |
|---|---:|---:|---:|---:|---:|
| LORRAX binding (eV) | 1.544 | 0.860 | 0.722 | 0.594 | 0.526 |

A direct check on 6 × 6 × 1, with the GW and the BSE both computed on that grid,
gives a binding of 0.928 eV against 0.856 eV from the 3 × 3 × 1 interpolation
(DFT energies in both). At this very coarse starting grid the interpolated
binding is 72 meV too small, while the coarse-grid value is about 0.6 eV too
large. Converged grids remove the difference between the codes, as the next
section shows.

### Interpolation on converged grids {: #interpolation-validation }

On better-converged coarse grids, with screening and BSE also computed directly
on the fine grid and the QP energies held fixed, LORRAX's interpolated spectra
follow the direct fine-grid calculations closely.

[![Interpolation validation: MoS₂ uses a 9×9 coarse and 12×12 fine grid; Si uses 4³ coarse and 8³ fine grids. LORRAX full SR/LR interpolation follows the direct fine-grid absorption, while the plane-wave dual-grid curve deviates more strongly. Lower panels show residuals as a percentage of the spectral peak.](../assets/figures/bse-interp-comparison.png)](../assets/figures/bse-interp-comparison.png)

**Denser-grid validation.** MoS₂ (9 × 9 → 12 × 12) in panels **a, c**, and Si
(4³ → 8³) in **b, d**. Gray: direct coarse grid; green: direct fine grid; light
blue: plane-wave dual-grid interpolation; blue dashed: LORRAX. The lower panels
show residuals relative to the fine-grid spectral peak. Over the lowest 40
excitons the mean absolute offsets from the direct calculation are **2.65 meV
for MoS₂** and **1.41 meV for Si** ([details](../theory/w-range-splitting.md#example)).

The [saved tables and plots](https://github.com/jackmcarthur/lorrax/tree/main/examples/MoS2_SOC_2026/results)
are in the example directory, with a report on the exported screening and
kernel read by the plane-wave code.

**Questions.** Where are the SOC splittings visible along the band path? How do
the quasiparticle corrections affect them? Which changes in the absorption
can be associated with those transitions, and which would require a finer
k-grid to interpret?

## Credits {: #credits }

The original SOC exercise is by Bradford A. Barker (2022), distributed in the
2026 workshop; see [Barker et al., Phys. Rev. B **106**, 115127 (2022)](https://doi.org/10.1103/PhysRevB.106.115127).
