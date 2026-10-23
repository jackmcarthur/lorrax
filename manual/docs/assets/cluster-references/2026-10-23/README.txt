Perlmutter reference calculations, 2026-10-23
============================================
Source: 2d2ca2f104a085876a573e8b1278a7874c479b98
See /using/clusters/ for measured resources, cache state and memory scopes.

Each case contains the exact GW deck, physical centroid table, original QE
SCF/NSCF input text, and a manifest with input/output hashes and measurements.
The manifests identify private NERSC evidence paths. They are archive pointers,
not public download URLs. WFN.h5, QE XML, UPF and generated HDF5 matrices are not
bundled. Regenerating DFT inputs is a new numerical run and need not produce the
same byte hashes. Obtain matching NC pseudopotentials and check the UPF hashes.

Build a coherent checkout/environment/native pair using /using/installation/;
for the exact historical source, use its installation instructions at commit
2d2ca2f104 (the current build scripts may differ).
do not mix a module's native legs with another source tree. The recorded library
hashes identify the historical measurements, not a portable prebuilt install.

Si SOC and Al2O3 (10 atoms)
-------------------------
For either case, create an empty shared parent directory with qe/ and run/.
Place the matching WFN.h5 and data-file-schema.xml in qe/. Copy gw.in and the
provided centroid table into run/, and place the named UPFs in run/ (symlinks to
shared files are sufficient). The GW deck uses wfn_file=../qe/WFN.h5.

Within a one-node/four-GPU allocation, in run/, with LORRAX_ROOT and JOBID set:

  source "$LORRAX_ROOT/config/perlmutter/gpu_env.sh"
  export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  reference_stage() {
    srun --jobid="$JOBID" -N1 -n4 -c16 --gpus-per-node=4 \
      --threads-per-core=1 --cpu-bind=threads \
      "$LORRAX_ROOT/src/ffi/cpp/select_gpu.sh" \
      "$LORRAX_ROOT/.venv/bin/python" "$@"
  }
  reference_stage -m psp.get_dipole_mtxels -i gw.in
  reference_stage -m gw.kin_ion_io -i gw.in
  reference_stage -m gw.gw_jax -i gw.in

To regenerate centroids instead of using the supplied physical point table:
  Si:     reference_stage -m centroid.kmeans_cli 528 --seed 42 \
            --fit-window 0:16,0:52 -i gw.in
  Al2O3:  reference_stage -m centroid.kmeans_cli 1584 --seed 42 \
            --fit-window 0:30,0:154 -i gw.in
Check the delivered count/complete symmetry orbits before continuing. Retain
stdout/stderr and exit status from every rank. The final Al deck deliberately
sets sigma_omega_min_ev=-24 so its entire requested QP prefix is dynamic.

Fe (ferromagnetic, 8x8x8)
------------------------
The exact deck keeps the original shared-directory arrangement:
  qe/WFN.h5 and qe/data-file-schema.xml
  preprocess/centroids_frac.txt  (copy supplied centroids_frac_1099.txt)
  preprocess/kin_ion.h5
  gpu/gw.in
Place the Fe.upf alias in preprocess/ and gpu/. Copy the exact gw.in into
both gpu/gw.in and preprocess/prep.in. With reference_stage defined as above:

  ( cd preprocess
    reference_stage -m psp.get_dipole_mtxels -i prep.in --out ../gpu/dipole.h5
    reference_stage -m gw.kin_ion_io -i prep.in -o kin_ion.h5
  )
  ( cd gpu
    reference_stage -m gw.gw_jax -i gw.in
  )

The original GW deck is preserved instead of silently changing its path bytes;
adjust paths for a new arrangement and record that change.

Centroid regeneration used target 1100, seed 0, oversample 1.5 and
--fit-window 0:22,0:110; complete-orbit selection delivered 1099 physical points.
The production r5 shared-pole recipe uses the default pole budget 1.0*Nmu.
CPU64/144 runs use the same physical points and bands but different dummy-slot
padding; compare canonical outputs, not the padded array bytes.

Cr:Al2O3 (80 atoms, SOC, 16 GPUs)
--------------------------------
Stage qe/ and run/ as for Si/Al2O3, with matching Al.upf/O.upf/Cr.upf
aliases in run/. Copy the final gw.in and centroid table into run/.
The original preprocess.in is also supplied; its CENTROIDS_PENDING marker
belongs to preparation and should not replace the final GW deck.
Use the same stage commands as the small GN-PPM cases, with the srun geometry
changed to -N4 -n16 --ntasks-per-node=4 --gpus-per-node=4 and add
--network=no_vni. The final GW deck covers the whole 1-399 QP prefix with
sigma_omega_min_ev=-75 and sigma_omega_max_ev=2.75.
To regenerate points, use 23700 --seed 0 --fit-window 0:399,0:2370 with the
ordinary spaced CLI form shown in each manifest. The measured selector
returned 23700 physical points; P16 storage adds 12 dummy slots.
The attached eqp0.dat/eqp1.dat are the actual fixed-input outputs, not
converged material predictions. See case_notes.txt for structural,
symmetry, head/model and memory qualifications from the private archive.

These are capacity/performance references, not converged material predictions.
Do not infer a smaller successful rank count from one measured peak. A lower
memory budget can choose different batches and must be measured independently.
