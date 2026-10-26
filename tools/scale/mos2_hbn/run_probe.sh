#!/bin/bash
# usage: run_probe.sh JOBID MU [NB]  -- kmeans, kin_ion, dipole, gn_ppm one-shot at P=4
# (inside: salloc --no-shell -q interactive -A m2651_g -C 'gpu&hbm80g' -N 1 --gpus-per-node=4 -t 04:00:00)
set -uo pipefail
J=$1; MU=$2; NB=${3:-1632}; W=/global/cfs/cdirs/m4598/jackm/mos2_hbn; B3=$((816+16))
export SLURM_JOB_ID=$J SLURM_JOBID=$J LORRAX=${LORRAX:-/pscratch/sd/j/jackm/htscissor/main}
D=$W/3-probe/mu${MU}_nb${NB}; mkdir -p $D; cd $D
ln -sfn $W/2-wfn/WFN.h5 WFN.h5; ln -sfn $W/2-wfn/hetero.save hetero.save 2>/dev/null
sed "s/NB/$NB/" ../gnppm.in.tmpl > gnppm.in
ln -sfn $W/pseudos/*.upf .   # kin_ion_io reads V_loc/V_NL from *.upf in the run dir
source $LORRAX/examples/MoS2_SOC_2026/2-gw/3-lorrax_fully_rel/lorrax_env.sh
git -C $LORRAX rev-parse HEAD > lorrax_head.txt
t() { local n=$1; shift; local s=$(date +%s); "$@" > $n.out 2>&1; local rc=$?; echo "$n rc=$rc wall=$(( $(date +%s)-s ))s" | tee -a stages.txt; return $rc; }
if [ ! -s centroids.txt ]; then
  t kmeans lorrax_run centroid.kmeans_cli $MU -i gnppm.in --seed 42 --fit-window 0:$B3,0:$NB || exit 1
  ln -sfn $(ls centroids_frac_*.txt | head -1) centroids.txt
fi
[ -s kin_ion.h5 ] || t kin_ion lorrax_run gw.kin_ion_io -i gnppm.in || exit 1
[ -s dipole.h5 ] || t dipole lorrax_run psp.get_dipole_mtxels -i gnppm.in || exit 1
t gnppm lorrax_run gw.gw_jax -i gnppm.in
echo PROBE_DONE
