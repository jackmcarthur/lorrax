# `src/ffi/` — the XLA FFI bridge

LORRAX's native kernels and vendor handlers, built into two libraries from
the one C++ tree `cpp/`: `liblorrax_ffi.so` (the CUDA leg, the complete
NVIDIA stack) and `liblorrax_ffi_host.so` (the CUDA-free host leg).

- **Find a kernel:** the [kernel operations](../../docs/architecture/ffi_layout.md#kernel-operations)
  table lists every core operation with its engine per hardware class, gate
  and code; the [kernel catalog](../../docs/architecture/ffi_layout.md#kernel-catalog)
  maps each family to its sources and target strings.
- **Python:** `fft.py` (the k-convolution router: mathdx on CUDA, the XLA
  backend elsewhere; the plane factory and the Fourier-plan call), `io.py`
  (parallel HDF5), `gemm.py` (host CBLAS GEMM), `gate.py` (the binding of
  `lxkit.gate`: env dials and the vendor platform key), `contour.py` (the
  contour accumulator), `common/ffi_loader.py` (locate, attest and register
  the libraries). Distributed linear algebra lives in `services/distrib_la`.
- **XLA is the reference path** ([decisions](../../docs/architecture/decisions.md#xla-reference)):
  a kernel here stays only while it is ≥ 2× faster on a production stage or
  decisive on memory, and it is gated against the XLA path on the same device.
- **Build, verify, seal, dependencies:** [`docs/installation/ffi-build.md`](../../docs/installation/ffi-build.md).
- **Add a target:** a k-convolution mode, [Adding a mode](../../docs/architecture/kconv.md#adding-a-mode);
  a linear-algebra backend, [`docs/services/distrib_la/backends.md`](../../docs/services/distrib_la/backends.md#adding-a-backend);
  an env-gated rank-local handler, [`docs/dev/ffi_gate_contract.md`](../../docs/dev/ffi_gate_contract.md#adding-a-dial).
  A changed handler signature bumps `cpp/common/lorrax_ffi_abi.h`
  ([ABI rule](../../docs/installation/ffi-build.md#abi)).
