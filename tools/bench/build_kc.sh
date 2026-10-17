#!/bin/bash
# usage: build_kc.sh SRCROOT OUTDIR   (SRCROOT = a lorrax checkout root)
set -eo pipefail
SRC=$1; OUT=$2; mkdir -p $OUT/gen
source /opt/cray/pe/lmod/lmod/init/bash 2>/dev/null || true
module unload cudatoolkit >/dev/null 2>&1 || true
module load gcc-native/14 cudatoolkit/13.2 >/dev/null 2>&1
CUDA=${CUDA_HOME}
PY=/global/common/software/m4598/jackm/lorrax_cuda13_runtime/.venv/bin/python
JAXINC=$($PY -c "import jax; print(jax.ffi.include_dir())")
C=$SRC/src/ffi/cpp
python3 - "$C/cufft/kbox_stage.cuh" "$C/cufft/kbox_stage_src.h.in" "$OUT/gen/kbox_stage_src.h" <<'EOF'
import sys
src=open(sys.argv[1]).read(); tpl=open(sys.argv[2]).read()
open(sys.argv[3],'w').write(tpl.replace('@LRX_KBOX_SRC@', src))
EOF
g++ -std=c++17 -O2 -fPIC -shared -Wall -Wno-unused -I$C -I$C/common -I$OUT/gen -isystem $JAXINC -isystem $CUDA/include \
  -isystem $CUDA/targets/x86_64-linux/include \
  $C/cufft/kconv_mathdx_cuda_ffi.cc $C/common/rtc_build.cc -L$CUDA/lib64 -lcudart -lnvrtc -ldl -Wl,-rpath,$CUDA/lib64 -o $OUT/libkc.so
echo BUILD_OK $OUT/libkc.so
