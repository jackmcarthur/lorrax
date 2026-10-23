# Executed workshop results

The supplied top-level workflow completed on four Perlmutter GPUs. The original
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

The native fine-grid spectrum differs substantially. These saved files provide
an executed workflow and measured comparisons; numerical equivalence between
the native fine-grid calculations remains unestablished.

Actual BerkeleyGW reader controls accepted the native exported screening and
kernel. With BerkeleyGW QP data held fixed, all 3,072 exciton energies differed
from the original BerkeleyGW calculation by RMS 7.680 meV and at most 9.552 meV.
With native QP data, that same consumer gave an onset of 1.670295 eV. A native
coarse-grid control gave 1.664460 eV versus 1.672848 eV in its corresponding
BerkeleyGW control; the complete 432-state coarse spectra still differ
(RMS 78.54 meV). These controls support further investigation of the native
coarse-to-fine treatment and establish neither full native spectral parity nor
a particular cause for the discrepancy.

[Independent validation](validation/OUTPUT_VALIDATION_REPORT.txt) separates
31 passing structural/coverage/convergence checks from descriptive scientific
comparisons. [Reader controls](validation/INTEROP_REPORT.txt) preserve the exact
inputs and scope of the comparisons. Solver convergence, complete frequency
coverage and file compatibility each have separate checks.

The text tables and plots here are small reference artifacts. Complete HDF5
screening, kernel, eigenvectors and wavefunction files are retained in the
executed Perlmutter/CFS archive recorded in the delivered report. Regenerate
those files with the supplied preparation and run scripts.
