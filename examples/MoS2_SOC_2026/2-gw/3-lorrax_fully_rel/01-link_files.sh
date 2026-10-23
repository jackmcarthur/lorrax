#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for stage in 1-epsilon 2-sigma 3-bs; do
    mkdir -p "$stage"
    ln -sfn ../../../1-mf/2-fully_rel/2d-Wfn_guarded/WFN.h5 "$stage/WFN.h5"
    ln -sfn ../../../1-mf/2-fully_rel/2d-Wfn_guarded/MoS2.save/data-file-schema.xml "$stage/data-file-schema.xml"
    ln -sfn ../../../pseudos/Mo_ONCV_PBE_FR-1.0.upf "$stage/Mo_ONCV_PBE_FR-1.0.upf"
    ln -sfn ../../../pseudos/S_ONCV_PBE_FR-1.1.upf "$stage/S_ONCV_PBE_FR-1.1.upf"
done
for stage in 2-sigma 3-bs; do
    ln -sfn ../1-epsilon/centroids.txt "$stage/centroids.txt"
    ln -sfn ../1-epsilon/kin_ion.h5 "$stage/kin_ion.h5"
    ln -sfn ../1-epsilon/dipole.h5 "$stage/dipole.h5"
done
ln -sfn ../2-sigma/eqp1.dat 3-bs/eqp_co.dat
printf 'Linked the shared fully relativistic mean field and LORRAX inputs.\n'
