Hands-on Session: Spin-Orbit Coupling (SOC) - BSE - LORRAX Fully Rel
=================================================================

Overview
--------

In this directory, we will calculate optical absorption with spin-orbit
coupling using LORRAX. This track reads the GW result in
`../../2-gw/3-lorrax_fully_rel` and is parallel to `../2-fully_rel`.

Goals
-----

1. Export the native screened interaction as a BerkeleyGW BSE kernel.
2. Solve the native BSE and compare absorption with and without electron-hole
   interactions.
3. Compare the fully relativistic spectra with the plane-wave track.
4. Plot exciton bands along Gamma–K and inspect the dispersion
   close to Gamma.

Instructions
------------

1. Finish the LORRAX GW track, then run `bash 01-link_files.sh` here.
   Inspect `gw.in` in `1-kernel` and `2-absorption`; compare them with
   `../2-fully_rel/1-kernel/kernel.inp` and
   `../2-fully_rel/2-absorption/absorption.inp`.

2. Go to `1-kernel` and run `bash 01-run_kernel.run`. Check `kernel.out`.
   The exporter writes `bsemat.h5` using the explicit full-grid `WFN_co.h5`
   from `../../1-mf/2-fully_rel/2c-Wfn_full` to match its coefficient gauge and
   k ordering. The coarse grid is `3 3 1`; GW still uses its 200-band sums.

3. Go to `2-absorption` and run `bash 01-run_absorption.run`. LORRAX solves
   its matrix-free BSE from the same native GW restart used for the export.
   The fine window is six valence and eight conduction bands on `8 8 1`,
   with shift `(0.47, 0.37, 0)`.

4. Check `absorption.out` for convergence of all 3,072 requested roots and
   inspect the residuals. These roots cover the complete retained
   Tamm–Dancoff transition space. The outputs include `eigenvalues.dat`,
   `eigenvectors.h5`, `absorption_eh.dat` and `absorption_noeh.dat`.

5. After the adjacent fully relativistic plane-wave reference has run, plot
   the interacting and independent-particle spectra:

   ```bash
   source ../../../2-gw/3-lorrax_fully_rel/lorrax_env.sh
   lorrax_python 02-plot_absorption.py
   ```

   Compare the onset, peak positions and stage timings between the two codes.
   No scalar-relativistic calculation is needed for this comparison.

6. Go to [3-exciton-bands](3-exciton-bands/README.md) and run
   `bash 01-run_exciton_bands.run`, then plot with
   `lorrax_python 02-plot_exciton_bands.py` after sourcing the shared
   `lorrax_env.sh`. This additional LORRAX exercise uses a Gamma-centered
   18 × 18 × 1 electron-hole integration grid, retaining the same 3 × 3 × 1
   screening calculation. It requests 128 levels at 37 momenta from Gamma to K,
   rather than the complete transition space. Check every root's residual
   and whether the plotted levels reach the independent-particle threshold.

Calculation settings
--------------------

The 22 coarse conduction states are part of the 48-band interpolation
manifold, including guards. The retained fine space stays at six valence and
eight conduction bands; the exported kernel uses twelve valence and twelve
conduction bands. The saved coarse manifold has a 49.79 meV global upper DFT
gap, allowing the unweighted velocity interpolation route.

The reciprocal-basis polarization is `(1, 1, 0)`, matching the plane-wave
reference default. The plane-wave code uses finite-shift velocity matrix elements;
LORRAX uses its analytic velocity operator. The Gaussian width is 0.15 eV,
and the native spectra run from 0 to 17 eV in 0.01 eV steps. The plane-wave code chooses
each file's end point automatically.

Comparison
----------

The saved fine-grid onset is 2.511 eV in LORRAX, versus 1.592 eV in the plane-wave code.
These are different interpolations from 3 × 3 × 1 to shifted 8 × 8 × 1,
without an explicit fine-grid GW+BSE reference to rank their accuracy.
Convergence of all retained
roots is a solver check, not convergence of the physical spectrum or proof
of agreement between codes. See the [saved results](https://github.com/jackmcarthur/lorrax/blob/main/examples/MoS2_SOC_2026/results/README.md)
for the spectra and controlled comparisons. The workshop's small grids and
band counts should not be used as converged MoS2 predictions.
