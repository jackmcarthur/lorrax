#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for stage in 1-kernel 2-absorption; do
    mkdir -p "$stage"
    ln -sfn ../../../1-mf/2-fully_rel/2d-Wfn_guarded/WFN.h5 "$stage/WFN.h5"
    ln -sfn ../../../1-mf/2-fully_rel/2c-Wfn_full/WFN.h5 "$stage/WFN_co.h5"
    ln -sfn ../../../1-mf/2-fully_rel/2d-Wfn_guarded/MoS2.save/data-file-schema.xml "$stage/data-file-schema.xml"
    for pp in Mo_ONCV_PBE_FR-1.0.upf S_ONCV_PBE_FR-1.1.upf; do
        ln -sfn "../../../pseudos/$pp" "$stage/$pp"
    done
    ln -sfn ../../../2-gw/3-lorrax_fully_rel/1-epsilon/centroids.txt "$stage/centroids.txt"
    ln -sfn ../../../2-gw/3-lorrax_fully_rel/1-epsilon/dipole.h5 "$stage/dipole.h5"
    ln -sfn ../../../2-gw/3-lorrax_fully_rel/1-epsilon/kin_ion.h5 "$stage/kin_ion.h5"
    ln -sfn ../../../2-gw/3-lorrax_fully_rel/1-epsilon/epsmat.h5 "$stage/epsmat.h5"
    ln -sfn ../../../2-gw/3-lorrax_fully_rel/1-epsilon/eps0mat.h5 "$stage/eps0mat.h5"
    ln -sfn ../../../2-gw/3-lorrax_fully_rel/2-sigma/tmp "$stage/tmp"
    ln -sfn ../../../2-gw/3-lorrax_fully_rel/2-sigma/eqp1.dat "$stage/eqp_co.dat"
done
ln -sfn ../../../2-gw/3-lorrax_fully_rel/2-sigma/gw.in 1-kernel/gw.in
ln -sfn ../1-kernel/bsemat.h5 2-absorption/bsemat.h5
printf 'Linked LORRAX screening, QP energies and wavefunctions for BSE.\n'
