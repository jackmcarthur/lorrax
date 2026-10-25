# Executed workshop results

The GW and optical BSE workflow completed on four Perlmutter GPUs. The original
fully relativistic BerkeleyGW reference was executed independently on a CPU node.
The numerical LORRAX source is `f075e4e796f996719a1c9a48ce48d6abb5291424`;
`f9ef4a6a93c5971a586bbd4b5ccac965e6d2bf05` supplies the EQP text compatibility
fix. Its rewrite preserved every numerical byte and was consumed successfully
by BerkeleyGW itself.

| Quantity (eV) | LORRAX | BerkeleyGW |
|---|---:|---:|
| Coarse-grid QP1 minimum gap | 3.208396 | 3.123690 |
| Path QP1 indirect gap | 2.872913 | 3.071206 |
| Lowest fine-grid BSE state | 2.511020 | 1.591552 |
| Largest interacting spectral peak | 3.18 | 2.60 |

Both fine-grid spectra interpolate from the 3 × 3 × 1 coarse grid to the
shifted 8 × 8 × 1 grid, using different QP and kernel interpolation methods.
No explicit GW+BSE calculation on the fine grid was performed for this
exercise. Their difference therefore does not establish which interpolation
is more accurate; that requires an independent fine-grid reference with
matched physical settings.

The separate [9 × 9 → 12 × 12 MoS₂ benchmark](https://jackmcarthur.github.io/lorrax/theory/w-range-splitting/#example)
uses screening and BSE computed directly on the fine grid, with matched QP
energies, and finds LORRAX SR/LR closer than BerkeleyGW dual-grid interpolation.
That accuracy test should not be confused with this workshop comparison.

Actual BerkeleyGW reader controls accepted the native exported screening and
kernel. With BerkeleyGW QP data held fixed, all 3,072 exciton energies differed
from the original BerkeleyGW calculation by RMS 7.680 meV and at most 9.552 meV.
With native QP data, that same consumer gave an onset of 1.670295 eV. A native
coarse-grid control gave 1.664460 eV versus 1.672848 eV in its corresponding
BerkeleyGW control; the complete 432-state coarse spectra still differ
(RMS 78.54 meV). The fine-grid consumer controls retain BerkeleyGW's own
interpolation even when they read a native coarse kernel. They establish
coarse-kernel compatibility under that consumer, not an independent accuracy
test of either interpolation. The separate coarse-spectrum differences do
not identify a particular implementation defect.

[Independent validation](validation/OUTPUT_VALIDATION_REPORT.txt) separates
31 passing structural/coverage/convergence checks from descriptive scientific
comparisons. [Reader controls](validation/INTEROP_REPORT.txt) preserve the exact
inputs and scope of the comparisons. Solver convergence, complete frequency
coverage and file compatibility each have separate checks.

The text tables and plots here are small reference artifacts. Complete HDF5
screening, kernel, eigenvectors and wavefunction files are retained in the
executed Perlmutter/CFS archive recorded in the delivered report. Regenerate
those files with the supplied preparation and run scripts.

The canonical [exciton-band scan](lorrax/exciton/) was executed separately
from Γ to K at 37 actual momenta, with 128 roots, an 18 × 18 × 1
integration grid and six valence/eight conduction bands. It took 24 min 11 s
on four Perlmutter A100 80 GB GPUs, including preparation and certification.
The [execution receipt](lorrax/exciton/execution.json) records numerical source
`0c93540d95541b85aa18d07aef0c87d097e52450`, the unchanged output hashes,
residuals, reciprocal-coordinate checks and continuum coverage.

The [24 × 24 reference](lorrax/exciton/reference_k24/) is retained for
integration-grid comparisons. The faster 18 × 18 scan is illustrative; the
lowest eight ordered levels differ by up to 30.3 meV from that reference.
