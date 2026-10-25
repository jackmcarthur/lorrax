#!/usr/bin/env bash
# Site launch adapter; all calculations run inside the supplied CPU allocation.
set -euo pipefail
site_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tutorial_dir=${TUTORIAL_DIR:-$(cd "$site_dir/.." && pwd)}
mf_dir="$tutorial_dir/1-mf/2-fully_rel"
: "${CPU_JOB_ID:?Set CPU_JOB_ID to an active CPU allocation}"
: "${CPU_NODE:?Set CPU_NODE to one allocated CPU node}"
stage=${1:-all}
case "$stage" in
    all|extras|1-Density|2a-Wfn|2b-Wfnq|2c-Wfn_full|2d-Wfn_guarded|3a-Wfn_fi|3b-Wfnq_fi|4-Wfn_inteqp) ;;
    *) echo "Unknown meanfield stage: $stage" >&2; exit 2 ;;
esac
# Tested on one 128-core Perlmutter CPU node: 32 ranks x4 threads, 4 k-pools.
# Converter uses 4 ranks; kgrid and binary-to-HDF utilities use one rank.
export OMP_NUM_THREADS=4
python_bin=${PYTHON:-python3}
launch=(srun --jobid="$CPU_JOB_ID" --nodelist="$CPU_NODE" -N1 -c4
        --exact --exclusive --threads-per-core=1 --cpu-bind=cores
        --time=00:15:00 --kill-on-bad-exit=1)
qe_env() { module load espresso/7.5-libxc-7.0.0-cpu; }
bgw_env() {
    # NERSC's serial utilities also require their CUDA-linked runtime libraries.
    module load gpu
    module load berkeleygw/4.0-nvhpc-23.9
}
run_stage() {
    local name=$1
    if [[ $name != 1-Density ]]; then
        "$python_bin" "$site_dir/prepare_meanfield.py" --tutorial "$tutorial_dir" stage "$name"
    fi
    cd "$mf_dir/$name"
    # Resume skips only a stage with all completion markers and expected outputs.
    if [[ $name == 1-Density ]]; then
        if [[ -f scf.out && -f MoS2.save/charge-density.hdf5 && -f MoS2.save/data-file-schema.xml ]] &&
                grep -q 'JOB DONE' scf.out; then
            echo "Already complete: $name"; return
        fi
        qe_env
        /usr/bin/time -p -o "$site_dir/scf.time" "${launch[@]}" -n32 \
            pw.x -npools 4 -in scf.in >scf.out 2>&1
        grep -q 'JOB DONE' scf.out
        return
    fi
    if [[ -f wfn.out && -f wfn.pp.out && -f wfn2hdf.out && -s WFN && -s WFN.h5 ]] &&
            grep -q 'JOB DONE' wfn.out && grep -q 'JOB DONE' wfn.pp.out &&
            grep -q 'Conversion from binary to HDF5 file complete!' wfn2hdf.out; then
        echo "Already complete: $name"; return
    fi
    if [[ -f wfn.in.temp ]]; then
        bgw_env
        /usr/bin/time -p -o "$site_dir/${name}_kgrid.time" "${launch[@]}" -n1 \
            kgrid.x kgrid.inp kgrid.out kgrid.log >kgrid.stdout 2>&1
        cat wfn.in.temp kgrid.out >wfn.in
    fi
    qe_env
    /usr/bin/time -p -o "$site_dir/${name}_pw.time" "${launch[@]}" -n32 \
        pw.x -npools 4 -in wfn.in >wfn.out 2>&1
    grep -q 'JOB DONE' wfn.out
    /usr/bin/time -p -o "$site_dir/${name}_convert.time" "${launch[@]}" -n4 \
        pw2bgw.x -npools 4 -in wfn.pp.in >wfn.pp.out 2>&1
    grep -q 'JOB DONE' wfn.pp.out
    bgw_env
    /usr/bin/time -p -o "$site_dir/${name}_hdf.time" "${launch[@]}" -n1 \
        wfn2hdf.x BIN WFN WFN.h5 >wfn2hdf.out 2>&1
    grep -q 'Conversion from binary to HDF5 file complete!' wfn2hdf.out
}
if [[ $stage == all ]]; then
    run_stage 1-Density
    for stage in 2a-Wfn 2b-Wfnq 2c-Wfn_full 2d-Wfn_guarded 3a-Wfn_fi 3b-Wfnq_fi 4-Wfn_inteqp; do
        run_stage "$stage"
    done
elif [[ $stage == extras ]]; then
    run_stage 2c-Wfn_full
    run_stage 2d-Wfn_guarded
else
    run_stage "$stage"
fi
