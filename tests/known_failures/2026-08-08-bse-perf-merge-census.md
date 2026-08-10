# AMENDMENT — BSE-PERF MERGE, `integration/bse-perf-merge-2026-08-08` @ `e69a867f` (2026-08-08)

**This supersedes the merge-checkpoint amendment below for the counts at THIS
head and configuration, and closes nothing.**  Five BSE performance-campaign
branches merged onto `main` @ `602e1d8b` — the feast-runner cache key, the
warm-cache alpha-gate persistability fix, the `bse_setup` scan, the
`w_omega_chain` conversion and the htransform/exciton instrument — plus the
owner-approved exciton-bands rerun-check default flip.  No FFI signatures
changed; the restage-candidate `.so` pair is unchanged and was re-verified by an
in-process loader probe before the census.

| | |
|---|---|
| machine | Perlmutter, JID 56499811, 1 node, 4xA100, Shifter, `lx test` (default xdist geometry) |
| module | `LX_BASE_MODULE=lorrax_J070`, jax `0.7.0.dev20260808` |
| tree | `/pscratch/sd/j/jackm/perf_bse_0808/wt_merge`; the `[lx] source tree:` line was read on every leg |
| `.so` pins | the restage candidate: device md5 `c680c229...`, host md5 `91f330c3...` |
| **`LORRAX_FFTW3_SO`** | **PINNED** (`lorrax_fftw_cray/.../libfftw3.so.mpi31.3.6.10`).  The checkpoint census below pinned none, which is why its FFT-engine block was red on both its legs and is green here |
| artifacts | `/pscratch/sd/j/jackm/perf_bse_0808/_reports_merge/census_e69a867f.xml` — **1996 testcases**, 341287 B; set-diff by `setdiff.py` in the same directory |
| run | one `lx test` invocation, **303.39 s** |

## The census at this head

| leg | pass | fail | skip | error | total |
|---|---|---|---|---|---|
| `tests/` | 1242 | 2 | 61 | 0 | 1305 |
| `services/distrib_la` | 137 | 0 | 32 | 0 | 169 |
| `services/lxkit` | 120 | 0 | 0 | 0 | 120 |
| `services/symmetry_maps` | 150 | 1 | 14 | 0 | 165 |
| `services/vcoul` | 33 | 1 | 0 | 0 | 34 |
| `services/wfn_loader` | 77 | 0 | 15 | 0 | 92 |
| `services/zeta_loader` | 110 | 0 | 1 | 0 | 111 |
| **ALL** | **1869** | **4** | **123** | **0** | **1996** |

The junitxml counts the single xfail among the skips; pytest printed
`4 failed, 1869 passed, 122 skipped, 1 xfailed`.

## SET-DIFF vs this document

| direction | result |
|---|---|
| **newly RED** | **ZERO.**  Every non-passing cell in every leg is listed in this file by name |
| newly GREEN | the red set is a strict SUBSET of the checkpoint amendment's 39.  **Not attributed cell by cell, and deliberately NOT closed anywhere in this file.**  The difference is pins and geometry — the `LORRAX_FFTW3_SO` row above, one node, this collection order — not code.  A census run in the checkpoint configuration would see them red again, and a row marked closed here would lie to it |
| collection delta | **+30 cells against 1966, all green** — 12 `test_bse_w_omega_chain_scan`, 8 `test_exciton_bands_rerun_default`, 7 `test_bse_feast_runner_cache`, 3 `test_bse_nontda` persistability gates |

## The four reds

| cell | class | fingerprint at this head |
|---|---|---|
| `test_bse_setup_qchunk::test_values_are_invariant_to_the_chunk_width` | **P2** | `_maxdiff = 1.3743988419548263` against `< 1e-10` |
| `test_bse_setup_qchunk::test_chunk_width_ulp_spread_is_reported` | **P2** | 5 spreads, first `(2, 2.220446049250313e-15, ...)` |
| `services/symmetry_maps ... test_the_lorentz_mixing_matches_a_dense_numpy_reference[1-1]` | cross-service conftest collision | `ValueError: Memory kinds passed to jax.jit does not match ...` |
| `services/vcoul ... test_vcoul_imports_and_computes_with_no_scipy` | cross-service conftest collision | `RuntimeError: Backend cuda is not in the list of known backends: [cpu, tpu]` |

**The P2 pair got its own A/B**, because `perf/bse-setup-scan` rewrote exactly
the chunking machinery those two cells gauge (`81891dbd`, the FFI eigh arm
walking q in one program per chunk).  Same node, same pins, one detached
worktree at pre-merge `602e1d8b` against the merge head: **2 failed / 22 passed
on both sides, with `_maxdiff` and the full spread table identical to every
printed digit.**  Inherited, not caused — which is independently what
`FIX_bsesetup.md` §4(7) found at the lane itself.
