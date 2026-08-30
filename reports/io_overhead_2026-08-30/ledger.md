# Sigma I/O-overhead audit ledger

- 2026-08-30: branch `perf/io-overhead-2026-08-30` started from
  `af80501f9214c3abf9d6cfb02d3fbc058166c976`.
- Scope: cold-run fixed and per-batch costs outside the delivered per-dispatch sigma
  kernel chain; slab reads, finite checks, collective mesh identity, startup/restart,
  and diagnostic output.
- Evidence rule: CPU reproductions may establish cache-key behavior; delivered-path
  performance and correctness evidence must come from one combined cold P=4 `lx`
  leg on this lane's own copy of `perf_probe_io_24b` inputs.
- Status: source and prior-evidence census in progress; no performance conclusion yet.
