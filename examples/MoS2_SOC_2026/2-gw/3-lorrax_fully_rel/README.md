Hands-on Session: Spin-Orbit Coupling (SOC) - GW - LORRAX Fully Rel
================================================================

Overview
--------

In this directory, we will perform the GW calculation with spin-orbit
coupling using LORRAX. This track is parallel to `../2-fully_rel` and reads
the same fully relativistic, two-component Quantum ESPRESSO mean field.

Goals
-----

1. Prepare the ISDF points and mean-field matrix elements.
2. Calculate screening, self-energy and a QP bandstructure.
3. Compare the outputs and stage timings with the BerkeleyGW track.

Instructions
------------

1. Complete `../../1-mf/2-fully_rel`, including `WFN.h5` conversion and the
   two additional `2c-Wfn_full` and `2d-Wfn_guarded` stages described in
   [the preparation guide](../../perlmutter/README.md). Use a built LORRAX
   checkout on Perlmutter or Frontier and a GPU allocation; set
   `export LORRAX=/path/to/lorrax` and activate the build's Python environment.
   The shared launcher selects four CUDA GPUs or four ROCm GCDs automatically.
   For a CPU allocation with at least 64 cores, also set
   `export JAX_PLATFORMS=cpu`.

2. Run `bash 01-link_files.sh`. Inspect `gw.in` in `1-epsilon` and `2-sigma`.
   Compare their bands, screening model and Coulomb settings with
   `../2-fully_rel/1-epsilon/epsilon.inp` and `../2-fully_rel/2-sigma/sigma.inp`.

3. Go to `1-epsilon` and run `bash 01-run_epsilon.run`. It requests 1,000
   symmetry-closed ISDF points with seed 42 and fit windows `0:48,0:200`
   (996 points in the saved run), then computes the kinetic/ionic and
   velocity matrices. Check `centroids.out`, `kin_ion.out` and `dipole.out`.

4. Go to `2-sigma` and run `bash 01-run_sigma.run`. LORRAX evaluates
   screening and self-energy together. The script also writes the exported
   `eps0mat.h5` and `epsmat.h5` into `1-epsilon`. Check `sigma.out` for
   completion and `COMPLETE` on-shell coverage of all 48 requested QP bands.

5. Go to `3-bs`, run `bash 01-run_inteqp.run`, then plot the bandstructure:

   ```bash
   source ../lorrax_env.sh
   lorrax_python 02-plot_bandstructure.py
   ```

   The path is Gamma–M–K–Gamma with the workshop's 106 points and 40 plotted
   bands. Compare the DFT and QP curves and the times spent in each GW stage.
   The plot comparison needs the adjacent fully relativistic BerkeleyGW run.

6. Return to the tutorial root and continue in `3-bse/3-lorrax_fully_rel`.

Calculation settings
--------------------

The geometry, pseudopotentials and 200-band GW sums are the workshop's.
The guarded mean-field file holds 204 bands so LORRAX can check the gap above
the 200-band fit window; these spare states do not enter the sums. The QP
calculation keeps 48 bands, including eight guards beyond the plotted 40.

Both tracks use HL-PPM. The explicit native `ppm_invalid_mode = static_limit`
matches BerkeleyGW's `invalid_gpp_mode 3`: invalid modes contribute their
static COHSEX limit. The unchanged BerkeleyGW4 workshop input already uses
that default. The reference adapter records both policies and input hashes
in `perlmutter/bgw_sigma_policy.json`. Matching this policy does not make
the fitted poles identical: the native HL body uses the finite-band
high-frequency response; BerkeleyGW uses the density sum rule.

The screening bases also differ. BerkeleyGW builds and inverts its dielectric
matrix at 9 Ry. LORRAX screens in the 996-point ISDF basis with 45 Ry
bare-Coulomb and ζ spheres; 45 Ry is not a plane-wave χ cutoff.
`epsmat_ecut = 9.0` selects a block of the already-screened inverse dielectric
matrix for export. It does not rerun native GW at 9 Ry. Matching the exported
dimensions therefore does not match the internal screening basis, and the
workshop comparison does not isolate interpolation.

`sigma_omega_min_ev = -65.0` eV, relative to the Sigma Fermi reference,
includes the deep Mo semicore states. `mc_average_vcoul_body = false` uses
the workshop's point Coulomb body. The production 2D kernel still averages
the singular Gamma cell; leave `head_minibz_average` at its default.

The coarse 48-band manifold also supports the BSE velocity interpolation:
its highest DFT energy is below band 49 throughout the coarse grid, with a
49.79 meV global gap in the saved run. This selects the unweighted velocity
route. See the [saved comparison](https://github.com/jackmcarthur/lorrax/blob/main/examples/MoS2_SOC_2026/results/README.md) for numerical
differences; this small example is not a convergence study.
