#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export LX_BASE_MODULE=lorrax_A
lx run --jid 57982945 --wait 1800 -N 1 -G 0 -n 1 -- ./rankwrap.sh bash ./driver.sh > driver.1.log 2>&1
test -s result.txt
grep -Eq '\[lx\] step .* exit 0' driver.1.log
