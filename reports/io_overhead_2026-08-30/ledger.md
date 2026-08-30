# Sigma I/O-overhead audit ledger

- 2026-08-30: branch `perf/io-overhead-2026-08-30` started from
  `af80501f9214c3abf9d6cfb02d3fbc058166c976`.
- Scope: cold-run fixed and per-batch costs outside the delivered per-dispatch sigma
  kernel chain; slab reads, finite checks, collective mesh identity, startup/restart,
  and diagnostic output.
- Evidence rule: CPU reproductions may establish cache-key behavior; delivered-path
  performance and correctness evidence must come from one combined cold P=4 `lx`
  leg on this lane's own copy of `perf_probe_io_24b` inputs.
- Historical cold delivered receipt (preserved
  `test_delivered_24b.onekernel_diagnostic_step15.log`, `af80501f`, P=4):
  468.33 s total, 449.87 s Sigma, 7.65 s runtime bring-up, 5.68 s restart.
- Slab-read census: 24 `_FfiBackend.read_slab` trace lines are six read
  signatures repeated on four ranks, not 24 distinct shapes.  Only one
  signature is the `(4-pole)` body read.  The 8-pole/4-pole-batch arm has no
  tail; its two batches, two walks, and two pole datasets reuse that one
  module and one `PoleReader` collective handle.
- CPU factory positive control: a repeated shape returns the identical cached
  read function (one hit/one miss); a deliberately shorter tail creates the
  second function (two misses total).  Focused test:
  `test_read_kernel_factory_reuses_one_shape_and_only_splits_a_tail`.
- Finite cache: two distinct but equal `NamedSharding` instances have equal
  hashes and resolve to the identical `_finite_stats_fn`; cache size remains
  one.  The receipt's 24 trace lines are five logical keys (one used twice),
  rank-expanded, and occur at stage boundaries rather than pole batches.
- Delivered psum cache: the receipt has three `jit__body` keys, each on four
  ranks, explained by payload rank `ndim={1,2,0}`.  The delivered wrapper
  forwards the supplied process mesh by identity; it does not construct a
  twin.  Keep `_PSUM_KERNELS` keyed by `id(mesh)`.
- Logging: 1,168 directly identifiable cache-diagnostic lines occupy 113,968
  of 201,424 bytes (56.6%) in the preserved cold receipt.  Patch `3382ca3f`
  leaves stage debug intact but makes explanations require the standard
  `JAX_EXPLAIN_CACHE_MISSES=1` opt-in.
- Own-arm cold baseline (`perf_probe_io_24b`, P=4, disk cache explicitly off,
  `09339ac0`): 466.33 s total, 443.68 s Sigma, 10.34 s bring-up, 6.21 s
  restart; finite-output parser PASS.
- Own-arm fixed cold leg (`3382ca3f`): cache directory absent before launch,
  runtime reported 0 advertised/agreed entries; 459.30 s total, 443.99 s
  Sigma, 5.68 s bring-up, 4.29 s restart, 94 cold compiles / 9.70 s.  Log is
  55,683 B / 597 lines with zero trace/persistent-cache miss lines.  Finite
  parser PASS; all 11 HDF5 datasets have exact-zero max absolute delta against
  baseline.
- Verification: the four focused source/invariant tests pass (4 passed in
  3.47 s), and the full private-arm cold P=4 driver passes.  A subsequent
  aggregate default-gate attempt on P=4 collected 849 passes but could not
  certify the tree in the deployed CUDA-only test environment: 42 service/
  backend failures and three Si fixture setup errors (`LORRAX_FFI_HOST_SO` and
  the documented `BUILD_NOTES.md` pin were unavailable).  None of the four
  focused tests is in the failure list.  Raw output is retained as
  `default_gate_p4.log`; no aggregate-gate PASS is claimed.
- Allocation hygiene: `lx test` created exact JID 57733660 without a lane tag.
  After evidence harvest this lane cancelled only that explicit job ID; it did
  not issue `lx release --all` or cancel any shared allocation.
