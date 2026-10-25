Hands-on Session: Spin-Orbit Coupling (SOC) with LORRAX
=====================================================

Overview
--------

In this example, we will calculate the quasiparticle bandstructure and optical
absorption of MoS2 with spin-orbit coupling using LORRAX. We will use the fully
relativistic mean field from the 2026 BerkeleyGW workshop and compare the two
codes in parallel GW and BSE folders.

The workshop uses small cutoffs, k-point grids and band counts for teaching.
These are not converged predictions. Both methods interpolate from 3 × 3 × 1
to a shifted 8 × 8 × 1 grid; their saved BSE onsets are 2.511 eV (LORRAX) and
1.592 eV (BerkeleyGW). No explicit fine-grid GW+BSE reference was computed,
so this comparison does not establish which interpolation is more accurate.

Goals
-----

1. Compute a fully relativistic GW bandstructure with LORRAX.
2. Solve the BSE and plot interacting and independent-particle absorption.
3. Compare the inputs, outputs and timings with the BerkeleyGW track.
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
along Gamma–K, showing 128 levels up to 3.0 eV and a near-Gamma detail.
Its 24 × 24 × 1 integration grid interpolates the same 3 × 3 × 1 screening calculation; it is a
separate grid from the shifted 8 × 8 × 1 optical calculation above.
