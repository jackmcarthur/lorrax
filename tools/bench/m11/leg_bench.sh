#!/bin/bash
# usage: leg_bench.sh JID OUT VAR1[,VAR2...] -- CASES...   (sequential, one GPU; VAR = tag[@lib]; env per var via ENV_<tag>)
set -u
export PATH=$HOME/bin:$PATH LORRAX_ACCOUNT=m2651_g LX_BASE_MODULE=lorrax_A
E=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/kc_m11
JID=$1; OUT=$2; VARS=$3; shift 4
for V in ${VARS//,/ }; do
  TAG=${V%@*}; LIB=${V#*@}
  if [ "$LIB" = base ]; then TREE=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/wt_kc_occ; else TREE=/pscratch/sd/j/jackm/cs_claude/sigma_depth2/wt_kc_m11; fi
  EV=ENV_$TAG; EXTRA=${!EV:-M11_X=1}
  cd $TREE
  LORRAX_CHECKOUT=$TREE timeout 900 lx run --jid $JID --wait 120 -N 1 -G 1 -n 1 -- env $EXTRA bash $E/tools/m11_run.sh $V $OUT "$@" > $OUT.$TAG.log 2>&1
  echo "VAR $V rc=$?"
done
echo LEG_DONE
