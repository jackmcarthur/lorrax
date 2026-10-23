Hands-on Session: MoS2 Exciton Bands
===================================

Overview
--------

In this directory, we will follow the exciton energies away from Gamma.
The electron has momentum k + Q and the hole has momentum k. The GW
screening and single-particle Hamiltonian still come from the workshop's
3 × 3 × 1 calculation. No additional shifted DFT calculation is needed.
This is an additional LORRAX exercise; the adjacent BerkeleyGW workshop
track does not contain an exciton-dispersion step.

Instructions
------------

1. Finish the native GW calculation and run `../01-link_files.sh`. Read
   `gw.in`: it retains six valence and eight conduction bands, samples the
   electron-hole sum on a Gamma-centered 30 × 30 × 1 grid, and requests
   64 levels at each Q. The 48-band H-transform window includes the same
   conduction guards as the optical calculation.

2. On the four-GPU allocation, run:

   ```bash
   bash 01-run_exciton_bands.run
   ```

   Inspect `exciton_bands.out` and `exciton_bands.log`. Each of the 64 roots
   at every Q must satisfy the 0.01 meV Davidson residual. The path is
   Gamma–M–K–Gamma, with extra points near the first Gamma. The script also
   accepts a deck and output prefix; `gw-near-gamma.in` gives a shorter
   21-point scan for timing and integration-grid checks.

3. Source the shared tutorial environment and plot:

   ```bash
   source ../../../2-gw/3-lorrax_fully_rel/lorrax_env.sh
   lorrax_python 02-plot_exciton_bands.py
   ```

   The plot writes `MoS2_exciton_bands.png` and `.pdf`. Its upper panel
   shows all requested levels; the lower panel resolves the low-energy
   structure near Gamma. A matching `exciton_bands_continuum.dat`, when
   present, adds the independent-particle threshold computed from the
   same QP transitions at each Q. The plot never substitutes the coarse
   gap or a DFT gap for that threshold.

Calculation settings
--------------------

The fine integration grid controls the electron-hole sum. It does not
replace the 3 × 3 × 1 screening data or establish convergence of that
coarse calculation. Compare otherwise identical 18 × 18, 24 × 24 and
30 × 30 near-Gamma scans before choosing a cheaper integration grid.
Retain enough levels to reach the independent-particle threshold if the
goal is to inspect the onset of the continuum.

For a bright exciton doublet in two dimensions, long-range exchange can
produce a leading linear term in |Q| for the longitudinal combination,
while the transverse partner begins quadratically. Look for those shapes
in the near-Gamma panel. These files contain energies without exciton
eigenvectors or dipoles, so energy ordering alone does not identify a
branch as longitudinal, transverse, bright or dark.
