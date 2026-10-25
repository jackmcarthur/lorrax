# Executed workshop results

The supplied top-level workflow completed on four Perlmutter GPUs. The original
fully relativistic plane-wave reference was executed independently on a CPU node.
The numerical LORRAX source is `f075e4e796f996719a1c9a48ce48d6abb5291424`;
`f9ef4a6a93c5971a586bbd4b5ccac965e6d2bf05` supplies the EQP text compatibility
fix. Its rewrite preserved every numerical byte and was consumed successfully
by the plane-wave code itself.

| Quantity (eV) | LORRAX | Plane wave |
|---|---:|---:|
| Coarse-grid QP1 minimum gap | 3.208396 | 3.123690 |
| Path QP1 indirect gap | 2.872913 | 3.071206 |
| Lowest fine-grid BSE state | 2.511020 | 1.591552 |
| Largest interacting spectral peak | 3.18 | 2.60 |

Both fine-grid spectra interpolate from the 3 × 3 × 1 coarse grid to the
shifted 8 × 8 × 1 grid, using different QP and kernel interpolation methods.
Of the 0.92 eV between the two onsets, 0.075 eV is the fine-grid QP gap
(3.233 eV against 3.159 eV) and 0.845 eV the exciton binding energy
(0.722 eV against 1.567 eV). On the 3 × 3 × 1 grid itself the two codes bind
alike (1.544 eV and 1.536 eV). LORRAX's binding falls as the BSE grid
densifies, as the 2D exciton envelope is resolved: 0.860, 0.810, 0.676, 0.594,
0.539 and 0.526 eV on unshifted 6², 8², 9², 12², 18² and 24² grids. The
plane-wave dual-grid result keeps the coarse-grid binding. A direct
6 × 6 × 1 control (DFT on the 6 × 6 grid, LORRAX GW and BSE there, DFT
energies) binds by 0.928 eV, against 0.856 eV from the same BSE interpolated
from 3 × 3 × 1. No direct calculation on the shifted 8 × 8 × 1 grid was run.

The separate [9 × 9 → 12 × 12 MoS₂ benchmark](https://jackmcarthur.github.io/lorrax/theory/w-range-splitting/#example)
uses screening and BSE computed directly on the fine grid, with matched QP
energies, and finds LORRAX SR/LR closer than plane-wave dual-grid interpolation.
That accuracy test should not be confused with this workshop comparison.

Plane-wave reader controls accepted the native exported screening and
kernel. With plane-wave QP data held fixed, all 3,072 exciton energies differed
from the original plane-wave calculation by RMS 7.680 meV and at most 9.552 meV.
With native QP data, that same consumer gave an onset of 1.670295 eV. A native
coarse-grid control gave 1.664460 eV versus 1.672848 eV in its corresponding
plane-wave control; the complete 432-state coarse spectra still differ
(RMS 78.54 meV). The fine-grid consumer controls retain the plane-wave code's own
interpolation even when they read a native coarse kernel. They establish
coarse-kernel compatibility under that consumer, not an independent accuracy
test of either interpolation. The separate coarse-spectrum differences do
not identify a particular implementation defect.

[Independent validation](validation/OUTPUT_VALIDATION_REPORT.txt) separates
31 passing structural/coverage/convergence checks from descriptive scientific
comparisons. [Reader controls](validation/INTEROP_REPORT.txt) preserve the exact
inputs and scope of the comparisons. Solver convergence, complete frequency
coverage and file compatibility each have separate checks.

The plots were regenerated on 2026-10-24 from a rerun at main `639be6916`
plus the htransform changes of `claude/htransform-scissor-mos2`; it reproduces
every saved table to 6 µeV. That rerun exported `eps0mat.h5`/`epsmat.h5` at
`epsmat_ecut = 25.0`; the plane-wave reader controls above give the same
3,072 eigenvalues from the 25 Ry and the 9 Ry exports.

The text tables and plots here are small reference artifacts. Complete HDF5
screening, kernel, eigenvectors and wavefunction files are retained in the
executed Perlmutter/CFS archive recorded in the delivered report. Regenerate
those files with the supplied preparation and run scripts.
