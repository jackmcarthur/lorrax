#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
export LX_BASE_MODULE=lorrax_A
lx run --jid 57982945 --wait 1800 -N 1 -G 0 -n 1 -- ./rankwrap.sh bash ./driver.sh > driver.1.log 2>&1
rc=$?
test -s rules.log && test -s rules_exit.txt || exit 2
exit "$rc"
