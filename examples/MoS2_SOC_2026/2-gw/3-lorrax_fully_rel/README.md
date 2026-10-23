Hands-on Session: Spin-Orbit Coupling (SOC) - GW - LORRAX Fully Rel
================================================================

This is the LORRAX track parallel to `../2-fully_rel`. Both tracks read
the workshop's fully relativistic, two-component Quantum ESPRESSO states.
No scalar-relativistic calculation is required for this track.

Instructions
------------

1. Complete `../../1-mf/2-fully_rel`, including conversion to `WFN.h5`.
   Use a built LORRAX clone and a four-GPU allocation on Perlmutter:
   `export LORRAX=/path/to/lorrax`.
   A CPU allocation with at least 64 cores can run the same scripts after
   `export JAX_PLATFORMS=cpu`; GPU is the default for this tutorial.

2. Run `bash 01-link_files.sh`. In `1-epsilon`, run
   `bash 01-run_epsilon.run` to select the compressed orbital basis and
   compute the kinetic/ionic and velocity matrices.

3. In `2-sigma`, run `bash 01-run_sigma.run`. LORRAX evaluates screening
   and self-energy together in one GW run. The script also exports the
   computed screening to BerkeleyGW-format `eps0mat.h5` and `epsmat.h5`
   under `1-epsilon`; these are exports of LORRAX's result, not copies of
   the BerkeleyGW reference.

4. In `3-bs`, run `bash 01-run_inteqp.run`, then
   `"$LORRAX/.venv/bin/python" 02-plot_bandstructure.py`. The path is Gamma-M-K-Gamma with
   the same 106 points and 40 plotted bands as the workshop.

5. Continue in `../../3-bse/3-lorrax_fully_rel`.

The mean-field cutoff, geometry, pseudopotentials and 200-band sums are
the workshop's. LORRAX additionally computes eight QP guard bands, giving
48 QP bands, so the interpolation can return all 40 workshop bands.
The separate `2d-Wfn_guarded` mean-field file contains 204 bands. The
GW sums and right ISDF fit window still use the original 200 bands.
The extra four states let LORRAX verify a gap above the fit window; the
measured smallest gap is 51.69 meV. The first 200 DFT energies agree
with the original file within 8e-11 eV. Run the supplied preparation
helper before linking this track.
The two codes use their own screening representations and HL first-moment
construction; matching file formats does not imply identical energies.

The Sigma grid starts at `sigma_omega_min_ev = -65.0` eV relative to its
Fermi reference. This includes Mo semicore bands in the requested 48-band
QP table; the default near-valence window does not cover them. Check the
reported on-shell coverage before using all bands in `eqp1.dat`.

`mc_average_vcoul_body = false` matches this workshop’s point Coulomb
body. The production 2D kernel already averages the singular Gamma cell
with its Wigner–Seitz rule; leave `head_minibz_average` at its default.

The BSE velocity interpolation uses the full 48-band manifold: its highest
DFT energy is below the lowest band-49 energy on the entire coarse grid,
with a 49.79 meV gap. This selects the existing unweighted velocity route.
The plotted band path still returns the workshop’s 40 bands.
