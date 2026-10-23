Hands-on Session: MoS2 Spin-Orbit GW and BSE with LORRAX
=====================================================

Overview
--------

In this example, we will run a fully relativistic LORRAX calculation beside
the `2-fully_rel` tracks of the [2026 BerkeleyGW workshop](https://workshop.berkeleygw.org/home).
Both use the workshop's two-component SOC mean field. The additional folders,
`2-gw/3-lorrax_fully_rel` and `3-bse/3-lorrax_fully_rel`, follow the original
link-script, numbered-directory and run-script organization.

The original inputs, pseudopotentials, notebooks and LICENSE are downloaded
from the workshop archive. The scalar-relativistic track is not needed here.
The small grids and band counts make this a teaching example, not a converged
calculation. The saved native fine-grid BSE onset differs substantially from
BerkeleyGW; compare the results rather than assuming numerical equivalence.

Goals
-----

1. Calculate a SOC quasiparticle bandstructure with LORRAX.
2. Export screening and a BSE kernel in BerkeleyGW's file layouts.
3. Calculate and compare optical absorption with and without electron-hole
   interactions, then compare the fully relativistic results between codes.

Instructions
------------

1. Use a built LORRAX checkout with its normal Perlmutter environment. Create
   a new working copy; the destination must not already exist:

   ```bash
   export LORRAX=/path/to/lorrax
   python3 "$LORRAX/examples/MoS2_SOC_2026/perlmutter/prepare_meanfield.py" \
     --tutorial "$SCRATCH/MoS2_SOC_2026/6a-SOC" init
   cd "$SCRATCH/MoS2_SOC_2026/6a-SOC"
   ```

   The downloader verifies the workshop archive's checksum before extracting
   it. Keep its original LICENSE and attribution with the working copy.

2. Read [perlmutter/README.md](perlmutter/README.md), load its listed modules,
   and prepare the mean field on an allocated 128-core CPU node:

   ```bash
   CPU_JOB_ID=<active-allocation> CPU_NODE=<allocated-node> \
     bash perlmutter/01-run-meanfield.sh all
   ```

   This runs the SCF, coarse, shifted, fine and band-path calculations. It
   also prepares `2c-Wfn_full` (40 bands on all nine coarse points, for kernel
   export) and `2d-Wfn_guarded` (204 bands at the original four stored points).
   The latter supplies spare states for the upper-edge gap check; the GW sums
   and right ISDF fit window still stop at 200 bands.

3. Run the original fully relativistic BerkeleyGW reference on the CPU
   allocation, using the supplied links and original input decks:

   ```bash
   CPU_JOB_ID=<active-allocation> CPU_NODE=<allocated-node> \
     bash perlmutter/03-run-reference-cpu.sh
   ```

   The LORRAX comparison plots read these reference outputs.

4. Allocate four GPUs, keep `LORRAX` set to your built checkout, and run:

   ```bash
   bash 00-run-lorrax.sh
   ```

   Alternatively, go through [GW](2-gw/3-lorrax_fully_rel/README.md), then
   [BSE](3-bse/3-lorrax_fully_rel/README.md), running the directories and scripts
   in numerical order. A CPU allocation with at least 64 cores can use the
   same four-rank scripts after `export JAX_PLATFORMS=cpu`; the complete
   supplied workflow was executed on GPUs.

5. Inspect the stage logs, `lorrax_timings.tsv`, bandstructure and absorption
   plots. Compare the input decks and time spent in each stage with the
   adjacent BerkeleyGW track. LORRAX computes screening and self-energy
   together in `2-sigma`; `1-epsilon` prepares its ISDF points and operators
   and receives the exported `eps0mat.h5` and `epsmat.h5` afterward.

6. After both tracks finish, check their outputs:

   ```bash
   "$LORRAX/.venv/bin/python" validate_tutorial.py --tutorial-root . > validation.json
   ```

   Inspect `status` in `validation.json`: missing outputs are `PENDING`,
   failures are `FAIL`, and the script itself still exits normally. `PASS`
   covers execution and file checks, not numerical equality between codes.
   Check file layouts, q coverage, complete on-shell Sigma evaluation, energy
   references and convergence of all requested BSE roots. Then compare with
   the [saved tables and plots](https://github.com/jackmcarthur/lorrax/blob/main/examples/MoS2_SOC_2026/results/README.md) in the LORRAX checkout.

Calculation settings
--------------------

The GW track requests 1,000 symmetry-closed ISDF points (996 in the saved run)
with seed 42 and fit windows `0:48,0:200`. It uses 200-band HL-PPM sums and
48 QP bands, including eight guards beyond the 40 plotted bands. The explicit
`sigma_omega_min_ev = -65` eV includes the deep Mo semicore states. Check the
reported on-shell coverage before using the complete QP table.

The QE XML specifies periodic DFT. LORRAX uses that boundary for the ionic
and Hartree terms, while `sys_dim = 2` selects slab GW/BSE screening. The
workshop's point Coulomb body is selected by `mc_average_vcoul_body = false`;
the singular Gamma cell is still averaged by the production Wigner–Seitz
rule. Leave `head_minibz_average` at its default.

The band path is Gamma–M–K–Gamma with 106 points. `3-bs` keeps both native
seven-column DFT/QP tables and writes the eight-column `bandstructure.dat`
with absolute energy references and Cartesian coordinates.

The native BSE uses six valence and eight conduction bands on the shifted
`8 8 1` grid, shift `(0.47, 0.37, 0)`, polarization `(1, 1, 0)` and 0.15 eV
Gaussian broadening. All 3,072 retained Tamm–Dancoff roots are requested.

Comparison
----------

The saved native fine-grid BSE onset is 2.511 eV, compared with 1.592 eV in
BerkeleyGW. This disagreement remains unresolved; small solver residuals do
not establish agreement or physical convergence. The codes use different
screening representations, QP Hamiltonians and interpolation procedures.
The [results page](https://github.com/jackmcarthur/lorrax/blob/main/examples/MoS2_SOC_2026/results/README.md) gives the measured GW and optical
differences, reader controls and numerical provenance separately.
