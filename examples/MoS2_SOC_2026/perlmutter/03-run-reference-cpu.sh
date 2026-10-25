#!/usr/bin/env bash
# CPU reference using the exact same upstream inputs as the GPU launcher.
set -euo pipefail
site_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tutorial_dir=$(cd "$site_dir/.." && pwd)
: "${CPU_JOB_ID:?Set CPU_JOB_ID to an active CPU allocation}"
: "${CPU_NODE:?Set CPU_NODE to one allocated CPU node}"
module load cpu
module load berkeleygw/4.0-gcc-12.3
export OMP_NUM_THREADS=8
module list >"$site_dir/bgw_cpu_modules.txt" 2>&1
# Record the actual comparison policy without changing the workshop input.
# BerkeleyGW4's omitted/-1 invalid_gpp_mode takes the same branch as3.
python3 - "$tutorial_dir" <<'PYTHON'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
bgw = root / '2-gw/2-fully_rel/2-sigma/sigma.inp'
native = root / '2-gw/3-lorrax_fully_rel/2-sigma/gw.in'
def settings(path):
    rows = {}
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].split('!', 1)[0].replace('=', ' ').split()
        if len(line) >= 2:
            rows[line[0].lower()] = line[1]
    return rows
reference, candidate = settings(bgw), settings(native)
declared = int(reference.get('invalid_gpp_mode', '-1'))
effective = 3 if declared == -1 else declared
native_mode = candidate['ppm_invalid_mode'].lower()
policy = {'static_limit': 3, 'infinity': 3, 'zero': 0, 'skip': 0, '2ry': 2}
record = dict(bgw_version='4.0', bgw_frequency_dependence=int(reference.get('frequency_dependence', '1')),
    bgw_invalid_gpp_mode_declared=declared, bgw_invalid_gpp_mode_effective=effective,
    native_compute_mode=candidate['compute_mode'], native_ppm_invalid_mode=native_mode,
    matching_invalid_policy=policy.get(native_mode) == effective,
    reference_adapter_modified_input=False,
    input_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (bgw, native)})
(root / 'perlmutter/bgw_sigma_policy.json').write_text(json.dumps(record, indent=2) + '\n')
PYTHON
# The upstream meanfield link script also replaces every NSCF schema with
# the SCF XML. Link only reference inputs here, after meanfield has finished.
link_mf_input() {
    local stage=$1 source_name=$2 directory=$3 name=${4:-$2}
    local source="../../../1-mf/2-fully_rel/$stage/$source_name"
    local destination="$tutorial_dir/$directory/$name"
    if [[ ! -s "$tutorial_dir/$directory/$source" ]]; then
        echo "Missing meanfield reference input: $tutorial_dir/1-mf/2-fully_rel/$stage/$source_name" >&2
        return 1
    fi
    if [[ -e $destination && ! -L $destination ]]; then
        echo "Refusing to replace an existing reference input: $destination" >&2
        return 1
    fi
    ln -sfn "$source" "$destination"
}
link_mf_input 2a-Wfn WFN 2-gw/2-fully_rel/1-epsilon
link_mf_input 2b-Wfnq WFN 2-gw/2-fully_rel/1-epsilon WFNq
link_mf_input 2a-Wfn RHO 2-gw/2-fully_rel/2-sigma
link_mf_input 2a-Wfn WFN 2-gw/2-fully_rel/2-sigma WFN_inner
link_mf_input 2a-Wfn vxc.dat 2-gw/2-fully_rel/2-sigma
link_mf_input 2a-Wfn WFN 2-gw/2-fully_rel/3-bs WFN_co
link_mf_input 4-Wfn_inteqp WFN 2-gw/2-fully_rel/3-bs WFN_fi
link_mf_input 2a-Wfn WFN 3-bse/2-fully_rel/1-kernel WFN_co
link_mf_input 2a-Wfn WFN 3-bse/2-fully_rel/2-absorption WFN_co
link_mf_input 3a-Wfn_fi WFN 3-bse/2-fully_rel/2-absorption WFN_fi
link_mf_input 3b-Wfnq_fi WFN 3-bse/2-fully_rel/2-absorption WFNq_fi
(cd "$tutorial_dir/2-gw/2-fully_rel"; bash 01-link_files.sh)
(cd "$tutorial_dir/3-bse/2-fully_rel"; bash 01-link_files.sh)
run_stage() {
    local name=$1 directory=$2 executable=$3 output=$4
    cd "$tutorial_dir/$directory"
    command -v "$executable" >"$site_dir/${name}_executable.txt"
    /usr/bin/time -p -o "$site_dir/${name}_cpu.time" \
        srun --jobid="$CPU_JOB_ID" --nodelist="$CPU_NODE" -N1 -n16 -c8 \
        --exact --exclusive --threads-per-core=1 --cpu-bind=cores \
        --time=00:02:00 --kill-on-bad-exit=1 "$executable" >"$output" 2>&1
    if [[ $name == kernel ]]; then
        # BGW4.0 kernel exits normally with timing tables, without Job Done.
        [[ -s bsemat.h5 ]]
    else
        grep -qi 'JOB DONE' "$output"
    fi
    echo "BGW_CPU_STAGE_DONE $name"
}
if [[ $# == 0 ]]; then set -- epsilon sigma inteqp kernel absorption; fi
for stage in "$@"; do
    case "$stage" in
        epsilon) run_stage epsilon 2-gw/2-fully_rel/1-epsilon epsilon.cplx.x epsilon.out ;;
        sigma) run_stage sigma 2-gw/2-fully_rel/2-sigma sigma.cplx.x sigma.out ;;
        inteqp) run_stage inteqp 2-gw/2-fully_rel/3-bs inteqp.cplx.x inteqp.out ;;
        kernel) run_stage kernel 3-bse/2-fully_rel/1-kernel kernel.cplx.x kernel.out ;;
        absorption) run_stage absorption 3-bse/2-fully_rel/2-absorption absorption.cplx.x absorption.out ;;
        *) echo "Unknown reference stage: $stage" >&2; exit 2 ;;
    esac
done
echo 'Requested fully relativistic BerkeleyGW CPU reference stages finished.'
