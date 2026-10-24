Meanfield preparation for the BerkeleyGW 2026 SOC tutorial
=======================================================

These helpers run the original fully relativistic MoS2 stages and prepare the
two additional LORRAX WFN files. The upstream exercise is by Bradford A. Barker
(2022); its README, LICENSE, pseudopotentials and inputs remain in the downloaded
tree. The scalar track is not executed. This is the workshop's small educational
calculation; retain its stated convergence limitations.

Start from the pinned archive
-----------------------------

Create a **new** working tree with one command:

```bash
python3 /path/to/lorrax_overlay/perlmutter/prepare_meanfield.py \
  --tutorial /path/to/run/6a-SOC init
```

The helper downloads the official 2026 `6a-SOC` archive from its pinned Drive URL
and requires SHA256
`fb84d8e26ba26c4e02772c1feb16fe6a2200087c6f3ef01198921c54c0e6c8af`,
refuses an existing destination, extracts the original files, and copies these
helpers, the parallel LORRAX GW/BSE folders, `00-run-lorrax.sh`, and
`README-LORRAX.md` into the new tree. It preserves the upstream README/LICENSE.
For an offline initialization, add `--archive /path/to/6a-SOC.tar.gz`; that file
must have the same pinned hash. The other-cluster changes described below affect
only launch/environment settings.

Run on an allocated CPU node
----------------------------

Prerequisites: Bash, Python3 (standard library only for preparation), Slurm,
`/usr/bin/time`, and the site's QE/BGW modules. On Perlmutter the tested versions
are `espresso/7.5-libxc-7.0.0-cpu` and `berkeleygw/4.0-nvhpc-23.9`.
The CPU allocation must supply one complete 128-core CPU node. Start the script
from a shell with the normal NERSC module initialization; no QE work runs on the
login node.

```bash
cd /path/to/run/6a-SOC
CPU_JOB_ID=<active-allocation> CPU_NODE=<allocated-node> \
  bash perlmutter/01-run-meanfield.sh all
```

`all` runs SCF, the original five fully relativistic NSCF stages, and both extra
stages, keeping binary `WFN` and `WFN.h5` side by side. QE uses 32 MPI ranks,
four OpenMP threads per rank, and four k-pools; `pw2bgw.x` uses four MPI ranks
with the same four pools. `kgrid.x` and `wfn2hdf.x` run serially on the allocated
CPU node. Each Slurm step has a 15-minute limit. Timings and logs stay in the
tutorial. The NERSC BGW utility binaries link CUDA libraries, so the adapter
loads `gpu` before the BGW module even for these CPU-only utilities.

To resume, rerun `all` or name one stage (for example `2d-Wfn_guarded`). Completed
stages are skipped after checking all completion markers and expected files.
`extras` runs only the two additions after SCF has completed. A partial stage is
rerun. Stage preparation copies SCF XML only when no stage XML exists and creates
the fixed-charge HDF5 link; it preserves a completed stage's NSCF XML. Do not
rerun the upstream global `01-link_files.sh` over completed NSCF directories:
that older script overwrites their XML with the SCF schema. The LORRAX track has
its own subsequent link scripts. After `all` completes, the supplied
`03-run-reference-cpu.sh` creates the original plane-wave reference's WFN/RHO/Vxc links
directly and then its epsilon/kernel/output links. It preserves every NSCF XML
and refuses missing meanfield inputs or replacement of existing regular input
files. No upstream global link script is needed.

The reference adapter also records `bgw_sigma_policy.json` with the actual
input hashes and effective invalid-pole policies. The MoS2 reference uses
HL-PPM. BerkeleyGW4's omitted `invalid_gpp_mode` selects the same static
COHSEX branch as explicit `invalid_gpp_mode 3`, matching the native
`ppm_invalid_mode = static_limit`. The original workshop input is preserved;
making the native default explicit changes no numerical setting.

On another cluster, adapt the module functions and `srun` launcher in
`01-run-meanfield.sh` to its QE/BGW installation and scheduler. The four k-pools
require compatible MPI counts; the tested 32-rank QE and 4-rank converter choices
avoid zero-work pools on the four-point coarse grid. The input generator is
site-independent and uses the same fixed SCF density for every stage.

What the two additional files mean
----------------------------------

* `2c-Wfn_full`: 40 bands on the explicit full 3x3 coarse grid (nine equal-weight
  points). This supports full-coarse file-gauge interoperability for native
  LORRAX BSEMAT export. The original 200-band/four-stored-k WFN remains untouched.
* `2d-Wfn_guarded`: 204 bands at the **original four k-points**. Its sole NSCF input
  change is `nbnd=200` to 204; the converter stays identical, including Vxc's
  200-band range. The four spare states let strict fit-window validation inspect
  band 201. LORRAX's chi/Sigma band sums and explicit zeta fit bound remain 200.

Both additions retain the original 45 Ry basis, fully relativistic Mo/S pseudos,
cell, atomic positions and fixed SCF charge. Each new directory receives its
exact generated inputs, a short README, and `INPUT_PROVENANCE.json`.

The executed 204-band file had band 200-to-201 gaps of 119.030, 51.689, 51.689 and
146.907 meV; the first 200 energies matched the original within 7.6e-11 eV. Regenerated
files should be checked again rather than disabling the production edge guard.
The already executed original stages and additions are recorded separately in
the tutorial's meanfield and guarded-WFN receipts.
