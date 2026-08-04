# Design decisions

Dated, binding rulings from the code owner. Each entry states the decision,
its consequence for code, and what it licenses deleting. Newest first.
These override older prose anywhere in the tree.

## 2026-08-04 — Padding is SlabIO's business, not the caller's

A caller states LOGICAL shapes only. Physical padding, mesh divisibility and
pad rows are implementation details SlabIO owns and must never require the
caller to reason about.

* `write_slab(name, A, offset=...)` accepts any `A`. If the backend needs a
  mesh-divisible physical extent, SlabIO pads internally. `valid_shape`
  defaults to A's logical extent clipped to the dataset and becomes an
  OVERRIDE for the ragged-chunk case, not a routine argument.
* `read_slab(name, shape=...)` returns exactly `shape`. Any padded extent the
  backend needs is read and trimmed internally; a caller never sees a pad row.
* Bounds are tested ONCE, on the logical slab `offset + valid_shape`. That is
  a replicated quantity, so every rank reaches the same verdict. Testing a
  rank-local advanced offset is forbidden: it splits the ranks into those that
  refuse and those that enter the collective, which strands the communicator
  with no HDF5 error and no traceback (measured: 306 s hang at P=4, silent
  420 s timeout on the read path).
* No rank may skip a collective because of its own error. Record the error,
  participate in the collective teardown, then raise.
* `create_dataset` on an existing dataset: identical logical shape and dtype
  ⇒ reuse (idempotent). Anything else ⇒ REFUSE, naming both shapes. Never
  silently delete-and-recreate (data loss) and never silently write into the
  previous geometry (wrong physics, no symptom).

CLIPPING IS SILENT AND THAT IS INTENDED (owner, 2026-08-04). A logical slab
that overhangs the dataset is clipped to the dataset extent and writes a
slightly smaller version of the same array. No warning. The overhang case and
the ordinary pad-row case are indistinguishable from inside SlabIO, and the
owner does not want a diagnostic for a path that is robust — a warning nobody
can act on is noise that trains people to ignore the log. An EXPLICIT
`valid_shape` override still refuses, because there the caller has stated an
intent that can be contradicted.

Rationale: every SlabIO defect found on 2026-08-02/04 — the wholly-padded
rank refusal, its residual asymmetric-bounds twin in both directions, the
mode-ignoring context cache, the ragged `valid_shape` hazards — is the same
root cause: padding leaked into the caller's contract, so correctness
depended on every call site independently getting it right.

Consequences: the host backend's absent divisibility check stops being a
backend divergence, because divisibility is no longer part of the contract.
`ensure_dataset` gains a shape/dtype check, which will newly refuse `mode='a'`
reruns at a different mu — those were silently writing into the old geometry
and were never correct.

Licenses deleting: caller-side pad/unpad arithmetic that exists only to
satisfy SlabIO, and per-call-site `valid_shape` computation that merely
restates the logical extent.

## 2026-07-30 — D10: fixed-shape ngkmax G loading (ratified 2026-08-04)

G-vector counts differ per k-point. Every per-k kernel takes the loader's
own fixed `(n_k, ngkmax, 3)` table rather than a ragged slice back to each
k's `ngk`, so every k presents identical operand shapes, the kernel lowers
ONCE instead of once per distinct `ngk`, and the k sweep becomes dispatches
of a single executable that `collectives.sweep_local_k` can pipeline behind
one host readback. Padded-vs-ragged agreement is gated at 1e-12 relative,
not bit-exactness (`tests/test_kin_ion_padded_gvectors.py`,
`tests/test_psp_padded_gvectors.py`), because appending exact zeros does not
change a sum but does change XLA's reduction blocking.

PROVENANCE NOTE, recorded because it was the reason for ratifying late: this
decision was cited as binding in code (`gw/kin_ion_io.py`,
`common/collectives.py:938`), carried a named tolerance (`RTOL_D10`) and
gated two test files, but was never written in this register — so there was
nowhere to read it in order to sign off. Owner confirmed it 2026-08-04. A
code comment must not be able to mint an "owner decision"; cite an entry
here or do not use the phrase.

Licenses deleting: ragged per-k G slicing in production consumers.
`psp.dft_operators.generate_gvectors_k` stays ONLY as the reference route
the D10 gates compare against.

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
