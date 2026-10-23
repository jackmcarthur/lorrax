MoS2 spin-orbit GW and BSE with LORRAX
====================================

This adds a fully relativistic LORRAX track to `6a-SOC` distributed by the
[2026 BerkeleyGW workshop](https://workshop.berkeleygw.org/home).
The original inputs, pseudopotentials, notebooks and LICENSE are downloaded
from the workshop archive. They are kept in their original folders. The
additional tracks are `2-gw/3-lorrax_fully_rel` and
`3-bse/3-lorrax_fully_rel`; each follows its adjacent BerkeleyGW track's
link-script, numbered-stage, input, run-script and plot-script organization.
Only the fully relativistic, two-component SOC calculation is used.

Preparation on Perlmutter
------------------------

Use a built LORRAX checkout with its normal Perlmutter environment. Create a
new working copy (the destination must not already exist):

```bash
export LORRAX=/path/to/lorrax
python3 "$LORRAX/examples/MoS2_SOC_2026/perlmutter/prepare_meanfield.py" \
  --tutorial "$SCRATCH/MoS2_SOC_2026/6a-SOC" init
cd "$SCRATCH/MoS2_SOC_2026/6a-SOC"
```
The mean-field adapter uses the installed Quantum ESPRESSO and BerkeleyGW
converter modules. Its README lists the exact modules and launch sizes.
Run the preparation adapter in `perlmutter/README.md` to compute the SCF,
original coarse, shifted, fine and band-path states, and two additional
fixed-potential NSCF files:

* `2c-Wfn_full`: 40 bands on all nine coarse k-points, for the BerkeleyGW
  kernel exporter to identify the explicit coefficient gauge and ordering.
* `2d-Wfn_guarded`: 204 bands on the original four stored coarse k-points.
  LORRAX still uses 200 bands in its GW sums and right ISDF fit window.
  The extra states allow the strict upper-edge gap check. They do not add
  bands to the GW sum or alter the original BerkeleyGW reference files.

Allocate four GPUs, set `export LORRAX=/path/to/lorrax`, and run:

```bash
bash 00-run-lorrax.sh
```

The individual steps can instead be run using the instructions in each
track's README. On a CPU allocation, `export JAX_PLATFORMS=cpu` selects the
same four-rank scripts; allow at least 64 CPU cores. The GPU path is the
measured end-to-end tutorial. The site adapter owns the launcher; input
decks contain the scientific settings.

The QE XML records the boundary used for the mean-field Hamiltonian. This
workshop uses periodic DFT; LORRAX therefore reconstructs its ionic and Hartree
terms with that boundary, while `sys_dim = 2` selects slab screening for GW/BSE.
This distinction is automatic from authenticated QE metadata and requires no
additional input key. Regenerate the kinetic/ionic file with the documented
LORRAX version rather than reusing one made with the wrong DFT boundary.

What each step computes
-----------------------

The `1-epsilon` stage selects 1,000 requested symmetry-closed ISDF points
(the decorated crystal permits 996 here), then writes kinetic/ionic and
velocity matrix files. `2-sigma` computes native screening and self-energy
and exports its own `eps0mat.h5` and `epsmat.h5` back into `1-epsilon`.
LORRAX evaluates screening and Sigma together rather than running two
separate screening calculations. `3-bs` computes both DFT and QP paths and converts their recorded energy
references and fractional coordinates to the eight-column `bandstructure.dat`
on the same 106-point Gamma–M–K–Gamma path.

The Coulomb body uses the workshop’s point-value convention
(`mc_average_vcoul_body = false`); the production 2D kernel averages the
singular Gamma cell with its Wigner–Seitz rule.

The Sigma frequency window explicitly includes the deep Mo semicore
states (`sigma_omega_min_ev = -65` eV relative to its Fermi reference).
The two original seven-column native path tables are retained beside the
converted BerkeleyGW-layout table.

The kernel stage exports the native screened interaction as `bsemat.h5`.
The absorption stage solves the native matrix-free BSE on the workshop's
shifted 8 x 8 x 1 grid, retaining six valence and eight conduction bands.
The requested 3,072 roots cover the whole retained Tamm–Dancoff space.
Text and HDF5 output conventions follow BerkeleyGW, so the folder is
recognizable to someone who has completed the original tutorial.

Comparison and accuracy
-----------------------

The original fully relativistic BerkeleyGW reference can be run on a
128-core CPU allocation with:

```bash
CPU_JOB_ID=<active-allocation> CPU_NODE=<allocated-node> \
  bash perlmutter/03-run-reference-cpu.sh
```

The reference adapter uses the original input decks and its own relative
links. It does not overwrite the completed NSCF schemas. Its independently
reproduced DFT and GW path gaps are
1.665839754 eV and 3.071206207 eV. These are small demonstration calculations,
not convergence studies. LORRAX uses its ISDF screening representation and
finite-band HL first-moment construction, a full-matrix effective QP
Hamiltonian and its own Hamiltonian interpolation. BerkeleyGW uses its own
screening, QP correction and wavefunction-interpolation approximations.
Identical file layouts do not imply identical numerical energies. The comparison receipt records measured
differences rather than treating one code's results as the other's output.

The original archive SHA256 is
`fb84d8e26ba26c4e02772c1feb16fe6a2200087c6f3ef01198921c54c0e6c8af`.
The downloader verifies it before extracting. Keep the original archive's
LICENSE and attribution alongside the downloaded materials.

Validation and saved results
----------------------------

The numerical run used `f075e4e796f996719a1c9a48ce48d6abb5291424`;
use the checkout including `f9ef4a6a93c5971a586bbd4b5ccac965e6d2bf05`
for directly BerkeleyGW-readable EQP files. That change moves provenance
comments after the numeric blocks and leaves all numerical rows unchanged.
After both tracks complete, run the independent read-only checks with:

```bash
"$LORRAX/.venv/bin/python" validate_tutorial.py --tutorial-root . > validation.json
```

The checks cover file layout, q coverage, complete on-shell Sigma evaluation,
energy-reference conversion and all requested BSE roots. The comparison
report separately records remaining differences in GW energies and optical
spectra. Saved tables and comparison plots are in the clone at
`$LORRAX/examples/MoS2_SOC_2026/results`.
