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
   bash 01-run_exciton_bands.run gw-near-gamma.in near_gamma
   ```

   Inspect `exciton_bands.out`/`.log` and `near_gamma.out`/`.log`. The full
   path has 25 Q points: Gamma, four steps to `(0.05, 0, 0)`, seven to M,
   six to K and seven back to Gamma. It requests 64 roots with a 0.25 meV
   Davidson residual budget. The separate scan keeps eight roots with a
   0.01 meV budget: 20 steps from Gamma to `(0.06, 0, 0)`, then K as an
   on-grid certification anchor, for 22 Q points in total. Coordinates are
   fractional reciprocal coordinates. Both use the same 30 × 30 × 1
   integration grid, six-valence/eight-conduction window and coarse QP data.

3. Source the shared tutorial environment and plot:

   ```bash
   source ../../../2-gw/3-lorrax_fully_rel/lorrax_env.sh
   lorrax_python 02-plot_exciton_bands.py --near-gamma-prefix near_gamma
   ```

   The plot writes `MoS2_exciton_bands.png` and `.pdf`. Its upper panel
   shows all requested levels; the lower panel resolves the low-energy
   structure near Gamma. A matching `exciton_bands_continuum.dat`, when
   present, adds the independent-particle threshold computed from the
   same QP transitions at each Q. The plot never substitutes the coarse
   gap or a DFT gap for that threshold.

   The upper panel retains the full-path data and its continuum threshold.
   The lower panel shows the eight near-Gamma levels within 0.12 Å⁻¹;
   the distant K certification anchor is outside that zoom. These remain
   ordered energy levels; the plot does not assign optical character.

Exchange certification
----------------------

The runner uses `--vq-mode refit --refit-window bse` and the existing
`visualization` certificate grade, whose exchange-comparison gate is 1 meV.
Matched on-grid checks on the 24 × 24 and 30 × 30 integration grids gave
0.01963 and 0.02086 meV, respectively. These pass the visualization gate
and exceed the 0.01 meV reference gate. They measure refitted versus stored
exchange eigenvalues, not the error of the exciton energies against a
converged calculation or a comparison between the two integration grids.
The solver uses a tighter absolute residual budget on both members of each
certification comparison, independently of the plotting budget.

The script accepts `deck`, `prefix` and `grade` as three optional positional
arguments. For example, `bash 01-run_exciton_bands.run gw.in reference
reference` requests the stricter grade and refuses when its gate is missed.
The driver's own default remains `reference`; the tutorial selects
`visualization` explicitly for an illustrative figure. No certificate
constant or refusal is changed.

Calculation settings
--------------------

The fine integration grid controls the electron-hole sum. It does not
replace the 3 × 3 × 1 screening data or establish convergence of that
coarse calculation. Compare otherwise identical 24 × 24 and 30 × 30 scans
before choosing a cheaper integration grid. The exchange certification above
does not establish that integration-grid convergence. At each Q, check the
highest requested level against the matched independent-particle threshold;
64 roots are not assumed to reach the continuum.

For a bright exciton doublet in two dimensions, long-range exchange can
produce a leading linear term in |Q| for the longitudinal combination,
while the transverse partner begins quadratically. Look for those shapes
in the near-Gamma panel. These files contain energies without exciton
eigenvectors or dipoles, so energy ordering alone does not identify a
branch as longitudinal, transverse, bright or dark.
