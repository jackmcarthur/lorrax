Hands-on Session: MoS2 Exciton Bands
===================================

Overview
--------

In this directory, we follow 128 exciton levels from Gamma to K. The electron
has momentum k + Q and the hole has momentum k. The GW screening and
single-particle Hamiltonian come from the workshop's 3 × 3 × 1 calculation;
no additional shifted DFT calculation is needed. This is an additional
LORRAX exercise alongside the BerkeleyGW workshop track.

Instructions
------------

1. Finish the native GW calculation and run `../01-link_files.sh`. Read
   `gw.in`: the electron-hole sum uses six valence and eight conduction bands
   on a Gamma-centered 24 × 24 × 1 integration grid. The 48-band H-transform
   window includes the same conduction guards as the optical calculation.

2. We tested this stage on four GPUs and recommend that allocation. Run:

   ```bash
   bash 01-run_exciton_bands.run
   ```

   Inspect `exciton_bands.out` and `.log`. The deck requests 128 roots at
   each of 37 actual Q points with a 0.25 meV Davidson residual budget:
   eight steps from Gamma to `(0.003, 0.003, 0)`, then 28 steps to
   `K = (1/3, 1/3, 0)`. The largest Q spacing is about four times smaller
   than on the former seven-interval Gamma–K segment. Every momentum is
   evaluated by the driver; no extra points are inserted by the plotter.

   The executed scan took 34 min 43 s on four Perlmutter A100 80 GB GPUs,
   including restart loading, H-transform, exchange refits, all 37 Q solves
   and the exchange certification. It excludes the preceding GW and
   preprocessing stages. The largest reported stage device peak was 16.35 GB
   per rank; host peak was 2.40 GB per rank. Four GPUs are a tested allocation,
   rather than a measured minimum.

   These coordinates use the supplied cell's reciprocal basis. Its in-plane
   basis vectors meet at 60 degrees, so `(1/3, 1/3, 0)` is a Brillouin-zone
   corner K and `(1/2, 0, 0)` is an edge midpoint M. Their energy ordering
   is not a test of the labels.

3. Source the shared tutorial environment and plot:

   ```bash
   source ../../../2-gw/3-lorrax_fully_rel/lorrax_env.sh
   lorrax_python 02-plot_exciton_bands.py
   ```

   The plot writes `MoS2_exciton_bands.png` and `.pdf`, showing the lowest
   128 computed levels only on Gamma–K and only up to 3.0 eV. The lower
   panel shows the shifts of ordered levels 3 and 4 from each level's own
   Gamma energy, in meV, through |Q| = 0.1056 Å⁻¹. Dots mark all 11 computed
   momenta in that detail. Straight segments connect actual ordered energies;
   no spline, fitted curve or eigenvector connectivity is inserted.

   The [saved tables](../../../results/lorrax/exciton/) contain the actual
   37-point output and matched threshold, with an execution receipt.

   `exciton_bands_continuum.dat` records the independent-particle threshold
   from the same QP transitions at every Q. The plot shows that threshold
   where it falls within the displayed energy range. A root count of 128
   does not by itself guarantee that the continuum is reached at every Q:
   the highest retained level reaches it at 15 of these 37 momenta.

Exchange certification
----------------------

The runner uses `--vq-mode refit --refit-window bse`: both H-transform
and exchange are evaluated at the requested arbitrary Q. A denser Q path
therefore does not require a finer integration grid. The 24 × 24 grid
controls the electron-hole sum and remains unchanged.

The tutorial explicitly selects the existing `visualization` certificate
with a 1 meV exchange-comparison gate; the completed scan measured 0.04417 meV.
All 128 roots converged at every Q, with a largest residual of 0.22844 meV. On-grid Q points compare refitted
and stored exchange eigenvalues using the same solver. This checks the
exchange construction, rather than integration-grid convergence. The
solver uses a tighter residual budget for both members of that comparison.
The driver's `reference` default and all global solver defaults are unchanged.

The script accepts `deck`, `prefix` and `grade` as optional positional
arguments. For example, `bash 01-run_exciton_bands.run gw.in reference
reference` requests the stricter grade and refuses when its gate is missed.

Interpretation
--------------

For a bright exciton doublet in two dimensions, long-range exchange can
produce a leading linear term in |Q| for the longitudinal combination,
while the transverse partner begins quadratically. The near-Gamma panel
makes this comparison legible after subtracting the different Gamma energies.
Level 3 has resolved quadratic curvature over the displayed window: a
quadratic fit has 0.145 meV RMS error, versus 1.69 meV for a linear fit.
Level 4 already rises by 1.345 meV at the first nonzero Q, where level 3
moves only 0.005 meV, illustrating a strong linear contribution. The plot
contains the computed points only, without those diagnostic fits.

The 0.25 meV residual budget applies to each energy; shifts smaller than
0.5 meV do not resolve differences conservatively. In particular, the tiny
initial shifts of level 3 do not establish its curvature alone. The wider
window above resolves it, without determining asymptotic coefficients.
Energies alone do not establish longitudinal, transverse, bright or dark
character.

These are illustrative results from a sparse coarse GW grid. Previous
matched 24 × 24 and 30 × 30 controls changed the lowest eight ordered
levels by up to 7.65 meV at their shared momenta; they do not establish
integration-grid convergence. Denser Q sampling improves the visible
dispersion without improving the underlying GW or electron-hole-sum
convergence. Completed older full-path outputs are retained as provenance,
not as inputs to this Gamma–K workflow.
