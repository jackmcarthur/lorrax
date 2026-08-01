# Design decisions

Dated, binding rulings from the code owner. Each entry states the decision,
its consequence for code, and what it licenses deleting. Newest first.
These override older prose anywhere in the tree.

## 2026-08-01 — FFI backends are required, not optional

The FFI layer (FFT, GEMM, distributed linear algebra, parallel HDF5) is an
essential part of the build. Where a certified FFI path exists, the native
JAX fallback path is not maintained and may be deleted; a missing or
unloadable FFI library is a refusal at startup (with the library named),
not a silent demotion to a slower path. Auto-demotion remains only where
the alternative is a different *service tier* (e.g. the h5py write route
on launches where MPI cannot bootstrap), never a duplicate compute path.

The code must still run on one process: the required libraries must build
and load for P=1 (they do — the host library has no MPI hard dependency
for the FFT/GEMM handlers, and the fastloop gate runs the full chain at
P=1 with the FFI stack on).

Licenses deleting: the XLA fft path inside the flat-k consumers and its
gate arms; the ungated-vs-gated dual factories in `fft_helpers`; every
"if FFI enabled ... else jnp" fork where the FFI side is certified.
Does NOT license deleting: the vendor-portability fallbacks *inside* the
FFI handlers (plain-loop CBLAS, FFTW-vs-MKL resolution) — those are how
the required layer stays buildable everywhere; and BSE's XLA FFTs, which
have no FFI route yet.

## 2026-08-01 — Square process meshes only; nonsquare P truncates

Only square 2-D device meshes are supported. When a run is launched with
a nonsquare process count P, the mesh resolver uses s = floor(sqrt(P)),
builds the s x s mesh, and announces that P - s^2 processes are idle;
it does not build a rectangular mesh and does not refuse. Launch scripts
should request square counts; the truncation is the safety net, not the
recommendation.

Rationale: rectangular meshes complicate ScaLAPACK grid geometry and the
divisibility contracts for no measured benefit at our scales (ruled
2026-07-27; this entry adds the truncation default, 2026-08-01).

Licenses deleting: rectangular-mesh accommodation in the mesh resolver
and any divisibility contortions that exist only to serve non-square
grids.

## Standing (recorded earlier, restated for one-page reference)

- Thousands-of-low-memory-processes is the scaling target: no N_mu^2-class
  object may be required to fit on one rank in the large-P limit
  (2026-07-27). Two plans per solve family: a local whole-tile plan and a
  distributed plan; execution schedules of a plan are not new plans.
- No time-reversal-symmetry-based reductions; no DFT-as-matmul (standing
  vetoes).
- Every performance claim carries a job id and an on-disk artifact.
