Hands-on Session: Spin-Orbit Coupling (SOC) - BSE - LORRAX Fully Rel
=================================================================

This track uses the same fully relativistic mean field as the workshop.
It consumes the LORRAX GW result in `../../2-gw/3-lorrax_fully_rel`.

Instructions
------------

1. Run `bash 01-link_files.sh`.

2. In `1-kernel`, run `bash 01-run_kernel.run`. The native LORRAX exporter
   writes `bsemat.h5` in BerkeleyGW's format. Its explicit full-grid
   `WFN_co.h5` is the additional 3 x 3 mean-field calculation in
   `1-mf/2-fully_rel/2c-Wfn_full`; this provides the coefficient gauge and
   k ordering required by the exporter. It does not replace the workshop's
   original 200-band WFN used by GW.

3. In `2-absorption`, run `bash 01-run_absorption.run`. LORRAX solves its
   matrix-free BSE from the same native restart used to export the kernel.
   The retained bands, fine grid and shift are the workshop's: six valence,
   eight conduction, 8 x 8 x 1, shift (0.47, 0.37, 0).

4. Check the solver's residual and convergence report. The requested
   3,072 roots cover the complete retained TDA transition space; the
   Gaussian width is the workshop's 0.15 eV. The output includes
   `eigenvalues.dat`, `eigenvectors.h5`, `absorption_eh.dat` and `absorption_noeh.dat` in the
   BerkeleyGW layouts.

The explicit reciprocal-basis polarization `(1, 1, 0)` matches the
BerkeleyGW reference default, which follows its fine-grid shift. Its
velocity operator uses a finite shift; LORRAX uses its analytic velocity
operator, so a finite numerical difference is expected.

5. After the BerkeleyGW fully relativistic reference has run, execute
   `"$LORRAX/.venv/bin/python" 02-plot_absorption.py` to compare both implementations. This
   comparison needs no scalar-relativistic calculation.

The 22 coarse conduction states include interpolation guards. The retained
fine BSE space stays at six valence and eight conduction bands, and the
exported kernel retains twelve valence/twelve conduction states. The
48-band coarse manifold has a global 49.79 meV upper DFT gap; it avoids
inverting vanishing velocity weights. Spectra use a 0.01 eV grid through
17 eV; BerkeleyGW chooses each file’s end point automatically.
