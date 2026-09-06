#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
export LX_BASE_MODULE=lorrax_A
for attempt in 1 2; do
 test ! -e "driver.$attempt.log" || exit 1
 lx run --jid 57966610 --wait 1800 -N 1 -G 0 -n 1 -- ./rankwrap.sh bash ./driver.sh > "driver.$attempt.log" 2>&1
 rc=$?
 if [ "$rc" = 0 ]; then
  grep -q CPU_PYTEST_PASS result.txt && grep -Eq '\[lx\] step .* exit 0' "driver.$attempt.log"
  exit $?
 fi
 [ "$rc" = 98 ] || exit "$rc"
done
exit 98
