#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
JID=${1:?Pass an explicitly authorized pool JID}
test ! -e lx_attempt1.log
export LX_BASE_MODULE=lorrax_A
for attempt in 1 2; do
  set +e
  lx run --jid "$JID" --wait 1800 -N 1 -G 4 -n 4 ./rankwrap.sh ./driver.sh > "lx_attempt${attempt}.log" 2>&1
  rc=$?
  set -e
  if [ "$rc" = 0 ]; then
    test -s eqp0.dat && test -s eqp1.dat
    grep -Eq '\[lx\] step .*exit 0' "lx_attempt${attempt}.log"
    exit 0
  fi
  # Retry only a pre-launch pool-expiry refusal on the same authorized JID.
  if [ "$rc" != 98 ] || [ "$attempt" = 2 ]; then exit "$rc"; fi
done
