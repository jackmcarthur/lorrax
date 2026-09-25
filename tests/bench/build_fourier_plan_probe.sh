#!/usr/bin/env bash
# Build lorrax_fourier_plan (src/ffi/cpp/cufft/fourier_plan*.{cc,cu}) as a
# standalone probe .so for tests and benchmarks before it joins the sealed
# bundle.  Usage: build_fourier_plan_probe.sh /absolute/out/libfourier_plan_probe.so
set -euo pipefail
out=${1:?usage: $0 /absolute/output.so}
here=$(cd "$(dirname "$0")/../../src/ffi/cpp/cufft" && pwd)
cuda=${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/13.2}
math=${LORRAX_MATH_LIBS:-$(dirname "$cuda")/../math_libs/$(basename "$cuda")}
jax_inc=${JAX_FFI_INCLUDE_DIR:-$(python3 -c 'import pathlib, jaxlib; print(pathlib.Path(jaxlib.__file__).parent / "include")')}
tmp=$(mktemp -d)
"$cuda/bin/nvcc" -O3 -std=c++17 -Xcompiler -fPIC \
  -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 \
  -c "$here/fourier_plan.cu" -o "$tmp/fourier_plan.o"
g++ -O3 -std=gnu++17 -fPIC -Wall -Wextra -c "$here/fourier_plan_cuda_ffi.cc" \
  -isystem "$jax_inc" -isystem "$cuda/include" -isystem "$math/include" -o "$tmp/ffi.o"
g++ -shared -o "$out" "$tmp/fourier_plan.o" "$tmp/ffi.o" \
  -L"$math/lib64" -L"$cuda/lib64" -Wl,-rpath,"$math/lib64" -Wl,-rpath,"$cuda/lib64" \
  -lcublas -lcufft -lcudart
nm -D "$out" | grep LorraxFourierPlanCudaFfi
