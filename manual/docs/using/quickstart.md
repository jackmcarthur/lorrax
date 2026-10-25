# Quickstart

`WFN.h5` From a Quantum ESPRESSO calculation to quasiparticle energies and an absorption spectrum.
{: .page-summary }

The following example input files carry out a run on bulk Si with spin–orbit coupling on a 4×4×4 grid. The QE input files of a workflow that LORRAX runs in its own test suite are `tests/hsuite/fixture/{scf,nscf,pw2bgw}.in`. LORRAX must be installed first ([Installation](installation.md)).

## The workflow {: #workflow }

Each step reads files the steps before it wrote, so they run in this order:

<div class="stages" markdown>

1. **Mean field** ([§1](#mean-field))
   Quantum ESPRESSO scf and nscf, exported to `WFN.h5`, with the pseudopotentials and the QE `.save` kept beside it.
2. **Preprocessing** ([§3](#preprocessing))
   From `WFN.h5` and the GW input file: [`centroids`](../utilities/centroids.md) chooses the ISDF interpolation points, [`kin-ion`](../utilities/kin-ion.md) computes the kinetic-plus-ionic single-particle Hamiltonian, and [`dipole`](../utilities/dipole.md) the velocity matrix elements. GW needs all three.
3. **GW** ([§4](#gw))
   [`gw`](../drivers/gw/index.md) computes the quasiparticle energies and writes the ISDF tensor file the BSE reads.
4. **Excitons and spectra** ([§5](#bse))
   [`bse`](../drivers/bse/index.md) solves for the excitons from the ISDF tensor file and `eqp1.dat`, saves the eigenvectors to file and writes the absorption spectrum.

</div>

## The launch prefix {: #launch }

Every LORRAX step runs on a compute node, never a login node, with one process per device. Set the prefix once per shell:

=== "Local"
    ```bash
    RUN="uv run python -u -m"
    ```
=== "Perlmutter"
    ```bash
    source $LORRAX/config/perlmutter/gpu_env.sh
    RUN="srun -N 1 -n 4 --gpus-per-node=4 $LORRAX/src/ffi/cpp/select_gpu.sh $LORRAX/.venv/bin/python -u -m"
    ```
=== "Frontera"
    ```bash
    LORRAX_ROOT=$LORRAX LORRAX_RUN_DIR=$PWD LORRAX_INPUT=gw.in \
      sbatch $LORRAX/config/frontera/templates/gw_dev.sbatch
    ```

`$LORRAX` is the source tree. With the Perlmutter module loaded, it is `$LORRAX_ROOT` and the interpreter is plain `python`. On Frontera each step is a batch job: the template runs `gw.gw_jax` on the CPU leg inside apptainer, and another step replaces the module on its `"$LORRAX_PY" -u -m` line ([Installation](installation.md#frontera)).

## 1. Mean field {: #mean-field }

Run a QE scf, then an nscf on the unshifted GW grid with empty states. The pseudopotential is a fully relativistic norm-conserving (ONCV) UPF. `scf.in`:

```fortran
&control
   calculation = 'scf'
   prefix      = 'Si'
   pseudo_dir  = './'
   outdir      = './'
/
&system
   ibrav     = 2
   celldm(1) = 10.26
   nat       = 2
   ntyp      = 1
   ecutwfc   = 60.0
   noncolin  = .true.
   lspinorb  = .true.
/
&electrons
   conv_thr = 1.0d-10
/
ATOMIC_SPECIES
 Si 28.086 Si.upf
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 8 8 8 0 0 0
```

`nscf.in` is the same input file with `calculation = 'nscf'`, `nbnd = 80` in `&system`, and the GW grid `4 4 4 0 0 0` under `K_POINTS automatic`. LORRAX unfolds QE's reduced list itself; there is no `kgrid.x` step ([k-points](inputs-from-dft.md#kgrid)). Then export with `pw2bgw.x` right after the nscf, since each `pw.x` run overwrites `Si.save/`:

```fortran
&input_pw2bgw
   prefix          = 'Si'
   real_or_complex = 2
   wfng_flag       = .true.
   wfng_file       = 'WFN'
   wfng_kgrid      = .true.
   wfng_nk1        = 4
   wfng_nk2        = 4
   wfng_nk3        = 4
   vxc_flag        = .false.
   kih_flag        = .false.
/
```

```bash
pw.x -in scf.in > scf.out                  # SCF density
pw.x -in nscf.in > nscf.out                # bands on the GW grid
pw2bgw.x -in pw2bgw.in > pw2bgw.out        # BerkeleyGW binary WFN
wfn2hdf.x BIN WFN WFN.h5                   # to HDF5
cp Si.save/data-file-schema.xml .          # symmetry and time-reversal record
```

The QE side, and what `WFN.h5` must contain, is on [Required DFT inputs](inputs-from-dft.md).

!!! note
    The LORRAX [`nscf`](../drivers/nscf/index.md) driver replaces the QE nscf, `pw2bgw.x` and `wfn2hdf.x` for a nonmagnetic, norm-conserving PBE calculation: `$RUN psp.run_nscf -i nscf.in`, with `save_dir`, `nbnd = 80` and `kgrid = 4 4 4` in an `[nscf]` section. It writes no `data-file-schema.xml`, and the scf one is used only when its k-points are those of `WFN.h5`. This run's scf is on 8×8×8, so with its fully relativistic pseudopotential `dipole` and `kin-ion` refuse ([nscf dependencies](../drivers/nscf/dependencies.md)); use the QE route here.

## 2. The GW input file {: #deck }

In the run directory put `WFN.h5`, `Si.upf` and `data-file-schema.xml`, and write `gw.in`:

```ini
[cohsex]
wfn_file       = WFN.h5
centroids_file = centroids_frac_600.txt   # the name step 3 prints
sys_dim        = 3                        # 3 bulk, 2 slab; required
nval           = 4                        # valence bands in the QP window
ncond          = 8                        # conduction bands in the QP window
number_bands   = 60                       # top of the chi0 and Sigma band sums
compute_mode   = tangential
qp_solver      = one_shot_dft
```

The same input file drives every step below. `number_bands` is also the top of the ζ-fit window, which refuses when its edge reaches the last band of `WFN.h5` (here 80) or cuts a degenerate multiplet, and names the boundary and the k-point. The band-sum check on the same edge only logs a `LORRAX SANITY FAILURE` warning when the edge comes from `number_bands`, and refuses for a named `number_bands_chi` or `number_bands_sigma` ([Band counts](inputs-from-dft.md#bands)). Keys and defaults: [gw keywords](../drivers/gw/keywords.md).

## 3. Preprocessing {: #preprocessing }

```bash
$RUN centroid.kmeans_cli 600 -i gw.in             # ISDF points
$RUN psp.get_dipole_mtxels -i gw.in              # velocity matrix elements
$RUN gw.kin_ion_io -i gw.in                      # T + V_ion
```

- [`centroids`](../utilities/centroids.md) selects $N_\mu$ points $\br_\mu$ and writes `centroids_frac_<n>.txt`. `n` is the count after pruning and can be below the 600 requested; read it from the `Centroids :` line of `kmeans.out` and set `centroids_file` to that name. A first count of 10 × `number_bands` is widely used in the literature.
- [`dipole`](../utilities/dipole.md) writes `dipole.h5`, the velocity matrix $\langle m\bk|\hat v|n\bk\rangle$, which is used by the $\bq \to 0$ head corrections in GW and for the dipole matrix elements in the absorption spectrum. On a grid that supports links it also writes `parallel_transport.h5`, the Berry connection matrix elements, which are used to compute the commutator $[\br,\Sigma^{xc}]$.
- [`kin-ion`](../utilities/kin-ion.md) writes `kin_ion.h5`, the matrix $\langle m\bk|T + V^{\mathrm{ion}}|n\bk\rangle$ that replaces BerkeleyGW's `vxc.dat`.

## 4. GW {: #gw }

```bash
$RUN gw.gw_jax -i gw.in
```

[`gw`](../drivers/gw/index.md) fits $\zeta_{\bq\mu}(\br)$, builds $W$, evaluates $\Sigma^{xc}$ and writes, among others ([gw files](../drivers/gw/files.md)):

| File | Content |
|---|---|
| `eqp0.dat`, `eqp1.dat` | QP energies on the irreducible wedge, BerkeleyGW format, eV |
| `sigma_diag.dat` | the Σ diagonal per k-point and band |
| `qp_wfn_rotations.h5` | the QP eigensystem $U$, $E^{\mathrm{QP}}_{n\bk}$ |
| `tmp/isdf_tensors_<N_mu>.h5` | the ISDF tensor file the BSE reads |
| `gwjax.out` | the run report |

A step has finished when its report ends with `LORRAX … completed.`; a refused step prints a `REFUSED (…)` line naming the condition.

## 5. Excitons and absorption {: #bse }

Add the BSE keys to `gw.in` (BerkeleyGW's absorption.inp names):

```ini
number_val_bands_fine  = 4
number_cond_bands_fine = 4
eqp_corrections        = eqp1.dat
dipole_file            = dipole.h5
number_eigenvalues     = 200
write_eigenvectors     = 200
```

```bash
$RUN bse.bse_jax -i gw.in
```

[`bse`](../drivers/bse/index.md) reads the one ISDF tensor file in `tmp/` beside the input file (two refuse), applies the QP energies of `eqp1.dat`, and solves the Tamm–Dancoff BSE for the lowest 200 excitons. It saves their eigenvectors to `eigenvectors.h5`, writes their energies and dipoles to `eigenvalues_b1.dat` … `eigenvalues_b3.dat` and the window's non-interacting transitions to `eigenvalues_b1_noeh.dat` … `eigenvalues_b3_noeh.dat`, and writes the absorption spectrum with and without electron–hole interaction to `absorption_b1_eh.dat` … `absorption_b3_eh.dat` and `absorption_b1_noeh.dat` …, one file per Cartesian polarization, broadened by `energy_resolution` (0.1 eV). These files follow BerkeleyGW's formats. The spectrum is complete up to the energy of the highest computed exciton. Band counts include spinor bands: Si with spin–orbit coupling has 8 occupied bands. A window edge inside a degenerate multiplet refuses; `band_degeneracy = snap` widens it.

## Next {: #next }

- Self-consistent GW: set `qp_solver = self_consistent` and rerun step 4 ([QSGW](../theory/qsgw.md)).
- Metals: `compute_mode = tangential` with `occ_smearing_width_ry` ([Metals](../theory/metals.md)).
- QP band structures: [`htransform`](../utilities/htransform.md) with a `K_POINTS crystal_b` block in the input file.
- Exciton dispersion: [`exciton-bands`](../drivers/exciton-bands/index.md).

