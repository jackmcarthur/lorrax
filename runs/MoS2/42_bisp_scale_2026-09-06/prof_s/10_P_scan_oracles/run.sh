#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
export LX_BASE_MODULE=lorrax_A
for attempt in 1 2; do
 test ! -e "driver.$attempt.log" || exit 1
 lx run --jid 57982945 --wait 1800 -N 1 -G 4 -n 4 -- ./rankwrap.sh ./driver.sh > "driver.$attempt.log" 2>&1
 rc=$?
 if [ "$rc" = 0 ]; then
  test -s combined_gate.json && grep -Eq '\[lx\] step .* exit 0' "driver.$attempt.log"
  exit $?
 fi
 [ "$rc" = 98 ] || exit "$rc"
done
exit 98
