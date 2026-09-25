# `src/ffi/` — the XLA FFI bridge

LORRAX's native kernels and vendor handlers, built into two libraries from
the one C++ tree `cpp/`: `liblorrax_ffi.so` (the CUDA leg, the complete
NVIDIA stack) and `liblorrax_ffi_host.so` (the CUDA-free host leg).

- **Find a kernel:** the [kernel operations](../../docs/architecture/ffi_layout.md#kernel-operations)
  table lists every core operation with its engine per hardware class, gate
  and code; the [kernel catalog](../../docs/architecture/ffi_layout.md#kernel-catalog)
  maps each family to its sources and target strings.
- **Python:** `fft.py` (the k-convolution router, the plane door, the
  Fourier-plan call and the host FFT gate), `io.py` (parallel HDF5),
  `gemm.py` (host CBLAS GEMM), `gate.py` (env dials), `common/ffi_loader.py`
  (locate, attest and register the libraries), `cublasmp/` (the fused
  W-solve). Distributed linear algebra lives in `services/distrib_la`.
- **Build, verify, seal:** [`docs/building_ffi.md`](../../docs/building_ffi.md);
  dependencies: [`docs/installation/ffi-native-libs.md`](../../docs/installation/ffi-native-libs.md).
- **Add a target:** a k-convolution mode, "A new mode is added in four
  steps" in the [router section](../../docs/architecture/ffi_layout.md#k-convolution-router-and-the-mathdx-family);
  a linear-algebra backend, [`docs/dev/linalg_ffi.md`](../../docs/dev/linalg_ffi.md#adding-a-backend);
  an env-gated rank-local handler, [`docs/dev/ffi_gate_contract.md`](../../docs/dev/ffi_gate_contract.md#adding-a-dial).
  A changed handler signature bumps `cpp/common/lorrax_ffi_abi.h`
  ([ABI rule](../../docs/building_ffi.md#the-abi-pairing-rule)).
