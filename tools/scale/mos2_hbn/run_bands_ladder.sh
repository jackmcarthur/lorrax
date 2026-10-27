#!/bin/bash
# usage: run_bands_ladder.sh JOBID  -- one GPU node: GPU-QE NSCF (nbnd 3400) -> WFN.h5 ->
# gn_ppm probe at 3264 bands, N_mu 16320 (10 x N_b), falling back to 12000. Releases JOBID on exit.
# (salloc --no-shell -q interactive -A m2651_g -C 'gpu&hbm80g' -N 1 --gpus-per-node=4 -t 04:00:00)
set -uo pipefail
J=$1; W=/global/cfs/cdirs/m4598/jackm/mos2_hbn; D=$W/2b-wfn3400
trap 'echo RELEASING; scancel $J' EXIT
mkdir -p $D/hetero.save; cd $D
if ! grep -qs 'JOB DONE' wfn.out; then
  # Only before a fresh NSCF: afterwards the save's XML is the NSCF's (3400 bands) and
  # re-copying the SCF one (816 bands) silently truncates pw2bgw's output.
  cp $W/1-scf/hetero.save/charge-density.* $W/1-scf/hetero.save/data-file-schema.xml hetero.save/
  cp $W/1-scf/hetero.save/*.upf hetero.save/ 2>/dev/null || true
fi
L=(srun --jobid=$J -N1 -n4 -c32 --cpu-bind=cores --gpus-per-node=4 --kill-on-bad-exit=1)
(
  module load espresso/7.5-libxc-7.0.0-gpu-cu13
  export OMP_NUM_THREADS=8
  if ! grep -qs 'JOB DONE' wfn.out; then
    date; /usr/bin/time -p -o wfn.time "${L[@]}" pw.x -npools 1 -ndiag 1 -in wfn.in > wfn.out 2>&1
  fi
  grep -q 'JOB DONE' wfn.out || exit 1; echo NSCF_DONE; date
  # pw2bgw re-writes distributed wfc records; at 4 GPU ranks one k is ~6 GB/rank and davcio
  # fails ("error writing file hetero.wfc1"). Run the CPU build on 32 host ranks instead.
  module load espresso/7.5-libxc-7.0.0-cpu; export OMP_NUM_THREADS=1; rm -f hetero.wfc*
  /usr/bin/time -p -o pp.time srun --jobid=$J -N1 -n32 -c2 --cpu-bind=cores --gpus=0 \
      pw2bgw.x -npools 1 -in wfn.pp.in > wfn.pp.out 2>&1
  grep -q 'JOB DONE' wfn.pp.out || exit 1
  module load gpu; module load berkeleygw/4.0-nvhpc-23.9
  /usr/bin/time -p -o hdf.time srun --jobid=$J -N1 -n1 -c32 wfn2hdf.x BIN WFN WFN.h5 > wfn2hdf.out 2>&1
  grep -q 'Conversion from binary to HDF5 file complete!' wfn2hdf.out || exit 1; echo MF_DONE; date
) || { echo MF_FAILED; exit 1; }
cd $W/3-probe
for MU in 16320 12000; do
  date; env -i HOME=$HOME PATH=/usr/bin:/bin bash -lc "WFNDIR=$D bash $W/3-probe/run_probe.sh $J $MU 3264"
  grep -q 'gnppm rc=0' mu${MU}_nb3264/stages.txt 2>/dev/null && { echo "MU $MU OK"; break; }
  echo "MU $MU FAILED"
done
echo LADDER_DONE; date
