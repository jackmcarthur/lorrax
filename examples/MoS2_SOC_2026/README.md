Hands-on Session: Spin-Orbit Coupling (SOC) with LORRAX
=====================================================

Overview
--------

In this example, we will calculate the quasiparticle bandstructure and optical
absorption of MoS2 with spin-orbit coupling using LORRAX. We will use the fully
relativistic mean field from the 2026 BerkeleyGW workshop and compare the two
codes in parallel GW and BSE folders.

The workshop uses small cutoffs, k-point grids and band counts for teaching.
This comparison is underconverged, which leads to some differences in the
final results between the plane-wave code and LORRAX: both interpolate from
3 × 3 × 1 to a shifted 8 × 8 × 1 grid, and their saved BSE onsets are
2.511 eV (LORRAX) and 1.592 eV (plane-wave code). At converged k-grid sizes the
two agree closely; the manual's tutorial page shows those comparisons.

Goals
-----

1. Compute a fully relativistic GW bandstructure with LORRAX.
2. Solve the BSE and plot interacting and independent-particle absorption.
3. Compare the inputs, outputs and timings with the plane-wave track.
4. Plot the finite-momentum exciton bands with LORRAX.

Instructions
------------

1. Follow [README-LORRAX.md](README-LORRAX.md) to download and prepare a new
   working copy. The original workshop files and LICENSE stay in that copy;
   this directory supplies the additional LORRAX folders.
2. Work through [GW](2-gw/3-lorrax_fully_rel/README.md), then
   [BSE](3-bse/3-lorrax_fully_rel/README.md), in numerical order. The
   [Perlmutter preparation guide](perlmutter/README.md) gives the modules and
   mean-field commands.
3. Compare your plots and tables with the [saved results](results/README.md),
   which distinguish successful execution from agreement between the codes.

The final BSE folder adds [exciton bands](3-bse/3-lorrax_fully_rel/3-exciton-bands/README.md)
along Gamma–M–K–Gamma and a near-Gamma zoom. Its 30 × 30 × 1 integration
grid interpolates the same 3 × 3 × 1 screening calculation; it is a
separate grid from the shifted 8 × 8 × 1 optical calculation above.
