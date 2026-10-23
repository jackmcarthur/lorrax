#!/usr/bin/env bash
# Run from any directory after preparing the fully relativistic mean field.
set -euo pipefail
cd "$(dirname "$0")"
: "${LORRAX:?Set LORRAX to your built checkout}"
source 2-gw/3-lorrax_fully_rel/lorrax_env.sh
run_stage() {
    local stage="$1" start="$SECONDS"
    bash "$stage"
    printf '%s\t%s\n' "$stage" "$((SECONDS-start))" >> lorrax_timings.tsv
}
printf 'stage\twall_seconds\n' > lorrax_timings.tsv
bash 2-gw/3-lorrax_fully_rel/01-link_files.sh
bash 3-bse/3-lorrax_fully_rel/01-link_files.sh
run_stage 2-gw/3-lorrax_fully_rel/1-epsilon/01-run_epsilon.run
run_stage 2-gw/3-lorrax_fully_rel/2-sigma/01-run_sigma.run
run_stage 2-gw/3-lorrax_fully_rel/3-bs/01-run_inteqp.run
run_stage 3-bse/3-lorrax_fully_rel/1-kernel/01-run_kernel.run
run_stage 3-bse/3-lorrax_fully_rel/2-absorption/01-run_absorption.run
run_stage 3-bse/3-lorrax_fully_rel/3-exciton-bands/01-run_exciton_bands.run
(cd 2-gw/3-lorrax_fully_rel/3-bs && lorrax_python 02-plot_bandstructure.py)
(cd 3-bse/3-lorrax_fully_rel/2-absorption && lorrax_python 02-plot_absorption.py)
(cd 3-bse/3-lorrax_fully_rel/3-exciton-bands && lorrax_python 02-plot_exciton_bands.py)
