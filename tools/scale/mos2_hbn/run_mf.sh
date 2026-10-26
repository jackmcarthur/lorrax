#!/bin/bash
# usage: run_mf.sh JOBID  -- SCF, NSCF (nbnd 1700), pw2bgw, wfn2hdf on one CPU node
# (inside an allocation: salloc --no-shell -q interactive -A m2651 -C cpu -N 1 -t 04:00:00)
set -euo pipefail
J=$1; W=/global/cfs/cdirs/m4598/jackm/mos2_hbn
export OMP_NUM_THREADS=2 OMP_PLACES=threads OMP_PROC_BIND=spread
L=(srun --jobid=$J -N1 -c4 --cpu-bind=cores --kill-on-bad-exit=1)
module load espresso/7.5-libxc-7.0.0-cpu
cd $W/1-scf
if ! grep -qs 'JOB DONE' scf.out; then
  date; /usr/bin/time -p -o scf.time "${L[@]}" -n64 pw.x -npools 2 -ndiag 16 -in scf.in > scf.out 2>&1
fi
grep -q 'JOB DONE' scf.out; echo SCF_DONE; date
cd $W/2-wfn; mkdir -p hetero.save
cp ../1-scf/hetero.save/charge-density.* ../1-scf/hetero.save/data-file-schema.xml hetero.save/
cp ../1-scf/hetero.save/*.upf hetero.save/ 2>/dev/null || true
/usr/bin/time -p -o wfn.time "${L[@]}" -n64 pw.x -npools 2 -ndiag 16 -in wfn.in > wfn.out 2>&1
grep -q 'JOB DONE' wfn.out; echo NSCF_DONE; date
/usr/bin/time -p -o pp.time "${L[@]}" -n4 pw2bgw.x -npools 2 -in wfn.pp.in > wfn.pp.out 2>&1
grep -q 'JOB DONE' wfn.pp.out
module load gpu; module load berkeleygw/4.0-nvhpc-23.9
/usr/bin/time -p -o hdf.time "${L[@]}" -n1 wfn2hdf.x BIN WFN WFN.h5 > wfn2hdf.out 2>&1
grep -q 'Conversion from binary to HDF5 file complete!' wfn2hdf.out; echo MF_ALL_DONE; date
