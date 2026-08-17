#!/usr/bin/env bash
# Build the benchmark-only local cuBLAS FFI probe inside the JAX container.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/output/libvertex_gemm_tc_probe.so" >&2
  exit 2
fi

out=$1
case "$out" in
  /*) ;;
  *) echo "output path must be absolute: $out" >&2; exit 2 ;;
esac

jax_include=$(python3 -c 'import pathlib, jaxlib; print(pathlib.Path(jaxlib.__file__).parent / "include")')
cuda_root=/usr/local/cuda/targets/x86_64-linux
mkdir -p "$(dirname "$out")"

/usr/bin/c++ -O3 -DNDEBUG -std=gnu++17 -fPIC -shared \
  -isystem "$jax_include" \
  -isystem /usr/local/cuda/include \
  -isystem "$cuda_root/include" \
  "$(dirname "$0")/vertex_gemm_tc_ffi.cc" \
  -L"$cuda_root/lib" -Wl,-rpath,"$cuda_root/lib" \
  -lcublas -lcudart -o "$out"

readelf -d "$out" | grep -E 'NEEDED|RPATH|RUNPATH'
nm -D "$out" | grep -E 'LorraxVertexZgemmStridedBatchedProbeFfi'
