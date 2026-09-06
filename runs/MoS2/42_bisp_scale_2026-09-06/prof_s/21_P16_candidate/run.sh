#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
JID=${1:?Pass an explicitly authorized allocation JID}
export LX_BASE_MODULE=lorrax_A
for attempt in 1 2; do
 test ! -e "driver.$attempt.log" || { echo "Attempt exists; make a new variant."; exit 1; }
 lx run --jid "$JID" --wait 1800 -N 4 -G 4 -n 16 -- ./rankwrap.sh ./driver.sh > "driver.$attempt.log" 2>&1
 rc=$?
 if [ "$rc" = 0 ]; then
  test -s eqp0.dat && test -s eqp1.dat && test -s sigma_diag.dat && grep -Eq '\[lx\] step .* exit 0' "driver.$attempt.log"
  exit $?
 fi
 [ "$rc" = 98 ] || exit "$rc"
done
exit 98
