# Bispinor GN-PPM fixture re-frozen on the packed heads-on route

**Date:** 2026-09-03  
**Gate:** `tests/test_gw_jax_regression.py::test_bispinor_gnppm_matches_reference`  
**Evidence:** Perlmutter JID 57891164, step 9
(`lx-Xg4-064604-2189732-6294`), under
`runs/DEV/321_bisp_fixture_refreeze_20260902/`.

## Ruling

The old reference is replaced. The fixture now names the one packed
`bare_transverse` owner, uses `head_correction = full`, has no scalar-head
overrides, leaves the mesh-agnostic bare solver unnamed, and parses with
`strict_keys = true`. Its newly shipped `dipole.h5` was generated from that
exact deck and WFN; SHA-256 is
`438623dbb1e2bf34d4148626e5cb638465b89132459dd7c39ef58156902d050e`.
The reference is a direct copy of the completed heads-on output and therefore
uses the current `Hdir=` schema.

This is an honest re-freeze because lane FIXTURE closed every component of
the pre-existing 2026-08-09 drift. Its report
`reports/bisp_fixture_drift_2026-09-02/report.md` records these exact
parent/fix or artifact A/B measurements over 270 rows:

| cause | measured effect |
|---|---|
| regenerated `kin_ion.h5` at `0b43d0a6` | direct field +0.274625 eV mean / 1.429902 eV max; every Sigma column exactly unchanged; stored exact Hartree replaces the old ISDF fallback |
| `e83ff7d5 -> 07451900` | WFN reciprocal scale: direct field +0.003042 eV mean / 0.004631 eV max, plus sub-meV Sigma changes |
| `9a3409ca -> 4534dc79` | physical source-state normalization: `sigC` +0.512756 eV mean / 0.803032 eV max |
| `56627937 -> 8f46b0de` | Coulomb-gauge spatial-metric sign: `sigX` +0.004243 eV mean / 0.008389 eV max |
| `d235f100 -> 27f1e5c9` | orbit-safe 0.2% GN tail policy: `sigC` -0.001734 eV mean / 0.013729 eV max |

The summed residual is at most 2e-6 eV, below this gate's 1e-5 eV tolerance.
The `07071457` q-grid repair is exactly zero here and `b0212b6b` changes only
three printed `sigXC` cells by 1e-6 eV.

The later frequency-policy interval remains separately identified:
`837ed531 -> 8576f9f9` is the certified denominator-box change (`sigC`
-0.001282 eV mean / 0.004073 eV max), and `8576f9f9 -> 7571f402` moves to the
literal `sigma_regularization_ev` owner while removing the pair ceiling.

## Packed-route and head movement

The exact 8576f9f9 direct-route gate in lane RESTART reported the old
reference mismatch as 1.434381 eV maximum (JID 57882521, step
`lx-Xg4-001317-677707-2191`,
`lane_RESTART/logs/focused_base_exact.log`). The current packed head-off
control differs from that 8576f9f9 output only in `sigC`: +0.000160578 eV
mean / 0.005851 eV max; `sigX` and `Hdir` are exactly unchanged. That interval
contains the subsequent certified frequency-policy changes. Independently,
the pre-deletion MoS2 certificate measured the packed and direct bare bodies
at 0.000 ueV with heads off (claim 581).

The controlled change on the current source is heads only. Heads-on minus
heads-off, same deck/WFN/dipole and one GPU:

| column | mean (eV) | max abs (eV) |
|---|---:|---:|
| `sigX` | -3.083476837 | 3.557968 |
| `sigC` | +0.546517241 | 1.318300 |
| `sigXC` | -2.536959615 | 3.393146 |
| `Hdir` | 0 | 0 |

The full-matrix gap moves from 3.41193 eV (DEBUG head-off) to 4.33762 eV
(production heads-on); the maximum printed `eqp1` movement is
3.393145979 eV. The head record authenticates the exact polygon rule,
`<v>=1652.678662338`, `tr<D_TT>=-3305.357324677`, and
`tr<D_TT> + 2<v> = 0` to printed precision. Full comparison output is
`runs/DEV/321_bisp_fixture_refreeze_20260902/head_comparisons.txt`.

## Gate contract

The first tip rerun must take the normalized byte-identity fast path: every
data byte is compared after `tests.harness.normalize_dat` removes only the
run timestamp and roundoff-valued star-spread diagnostic headers.
Cross-machine fallback remains label-based through
`tests.harness.parse_eqp_rows` and uses `atol = 1e-5 eV`; the current `Hdir=`
direct-field spelling is parsed rather than relying on bytes to hide an
unreadable file. Any later nonzero movement must again be attributed before
another freeze.
