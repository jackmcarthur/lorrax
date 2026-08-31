# Measure-compression audit: raw pairs do not reduce Na ROQ node count

Heavy offline lane. Branch `study/measure-compression-2026-08-31`, base
`1426d9f4`. No compute job was launched. The machine-readable evidence is
`measure_compression_baseline.json`; the reproducer is
`measure_compression_audit.py` in this directory.

## Method

I rebuilt two frozen Na windows directly from `g_states.csv`,
`w_histogram.csv`, and `windows.json`. Window 4 is crossing and contains
91,008 uncompressed state-by-pole pairs. Window 5 is sign-definite and
contains 76,800 pairs. For both windows, the reconstructed 25-bin fit and
50-bin validation lattices are bit-exact with
`na_reconstructed_problems_v1.npz`.

The fixed rule family is the valence configuration recorded by ROQ commit
`1426d9f4`: causal contour angle -58 degrees, horizon 27 Ry^-1, 96
Gauss-Legendre candidates, delivered-mass snapshot SVD, QDEIM placement, and
the production IRLS weight solve. I scanned every integer rank from 4 upward.
The first accepted rank had to meet the window target and the production
noise gate on the raw pairs. Thus “minimum” below means ranks 4--6 failed the
raw reference and rank 7 was the first pass; no compressed-lattice score was
used for acceptance.

## Na results

`Fit/raw` below is the achieved error on the fitting measure divided by the
same rule's error on raw pairs. A value below one is optimistic.

| Window | Fit measure | Cells | Min N on raw | Fit error | Raw error | Fit/raw | Raw kappa p99 | Audit wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| val:resonant, crossing; target 1.081295e-3 | 25-bin | 537 | 7 | 9.163515e-4 | 8.657314e-4 | 1.05847 | 3.27115 | 2.85 |
| | 50-bin | 1,840 | 7 | 8.020392e-4 | 7.729522e-4 | 1.03763 | 3.24981 | 2.45 |
| | 100-bin | 3,060 | 7 | 8.121843e-4 | 7.831301e-4 | 1.03710 | 2.32426 | 3.01 |
| | raw | 91,008 | 7 | 7.255378e-4 | 7.255378e-4 | 1.00000 | 2.85899 | 105.07 |
| val:pole_tail, sign-definite; target 6.484834e-4 | 25-bin | 550 | 7 | 4.958083e-4 | 4.918100e-4 | 1.00813 | 1.22243 | 2.12 |
| | 50-bin | 1,675 | 7 | 4.507621e-4 | 4.503718e-4 | 1.00087 | 1.19465 | 2.75 |
| | 100-bin | 2,304 | 7 | 4.571842e-4 | 4.535611e-4 | 1.00799 | 1.20222 | 2.67 |
| | raw | 76,800 | 7 | 4.513797e-4 | 4.513797e-4 | 1.00000 | 1.20304 | 81.48 |

Compression costs **0 nodes** at equal true error in both Na windows. Every
raw noise bound also passes by more than two orders of magnitude. The raw fit
is not a production alternative: it took 81--105 s here versus 2--3 s for a
compressed fit, exceeds the planner budget, and is the prohibited explicit
pair route.

The Na rules fitted in this controlled comparison are not flattered: all six
compressed `Fit/raw` ratios are 1.00087--1.05847. The frozen incumbent rule is
very slightly optimistic on sign-definite window 5 (25-bin/raw = 0.99770 and
50-bin/raw = 0.99834), a 0.23% maximum difference; its crossing-window scores
are conservative.

## DEV-80 toy crossing check

The frozen DEV-80 rules already store scores on the 25-bin fit lattice, the
50-bin validation lattice, and raw pairs. This is a useful adversarial check
because all eight branch supports cross the 0--5 eV frequency interval.

| Seed/branch | N | 25-bin | 50-bin validation | Raw | 50-bin/raw |
|---|---:|---:|---:|---:|---:|
| 0 cond | 59 | 9.46038e-4 | 2.20706e-3 | 2.54425e-3 | 0.86747 |
| 0 val | 11 | 9.60984e-4 | 1.00222e-3 | 1.04834e-3 | 0.95601 |
| 1 cond | 66 | 9.69001e-4 | 1.01194e-3 | 1.10248e-3 | 0.91787 |
| 1 val | 9 | 8.08756e-4 | 8.13160e-4 | 1.07081e-3 | 0.75938 |
| 2 cond | 37 | 9.77563e-4 | 1.08514e-3 | 1.33670e-3 | 0.81181 |
| 2 val | 7 | 9.86047e-4 | 9.92098e-4 | 1.06834e-3 | 0.92864 |
| 3 cond | 49 | 9.15222e-4 | 2.22915e-3 | 2.71468e-3 | 0.82115 |
| 3 val | 10 | 8.85426e-4 | 8.89551e-4 | 9.10646e-4 | 0.97684 |

Yes, compression can flatter a rule. At the common 1e-3 target, the 25-bin
fit lattice would falsely accept 7/8 toy rules. The independent 50-bin
validation removes five of those, but still falsely accepts seed-1 valence
and seed-2 valence. Its worst under-report is 24.1%; the worst 25-bin
under-report is 66.3%.

## Decision

No production compressor change is justified by the assigned node-count
question: the measured saving is **no material saving (0 nodes)**. I tested
the suggested small changes offline before making that decision. Pure
mass-quantile axes made frozen-rule scores range from 1.08x to 93.1x raw;
mixed count/mass axes remained up to about 61x raw; a bounded mass-quantile
centroid lattice ranged from 0.925x to 2.44x raw; and simply using a 100-bin
count lattice ranged from 1.14x to 59.4x raw. These are not faithful fixes
and would trade false optimism for uncontrolled node cost. The non-monotonic
behavior comes from moving mass for a nonlinear near-crossing kernel; total
mass conservation alone cannot bound it.

The remaining toy false-acceptance risk therefore stays explicit. Closing it
needs a certified interpolation-error bound at the consumer, or another
bounded product-aware certificate; an explicit raw pair scorer is ruled out
by both cost and the campaign's no-O(N^4) ruling. The code change in this lane
only pins the existing deterministic, roundoff-conservative, fixed-cell-count
contract in a service test.
