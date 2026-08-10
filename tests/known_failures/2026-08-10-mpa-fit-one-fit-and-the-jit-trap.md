# AMENDMENT — THE MPA FIT STOPPED FITTING EVERY ELEMENT TWICE (2026-08-10) — **CLOSED, plus one trap that outlives it**

No red is opened or closed by this row. It is here for the same reason the
P>1 row is: the durable part is not the change, it is the two things the
measurement found on the way, and both of them are traps a later lane
would otherwise pay for again.

## What was wrong, and what it cost

`gw/mpa/fit_driver.fit_one_block` bought its backward error from
`gw.mpa.diagnostics.solve_conditioning`, which re-solves the Pade system
and then runs a COMPLETE SECOND FIT of every element to report a forward
residual beside it. The driver discarded three of that second fit's four
outputs and read the residual and `n_valid` off the first fit, which had
already returned them. Two complete fits and three Pade solves per
element, for the quantities of one fit and two solves. The driver's own
docstring had said so since the driver landed; nobody had priced it.

Measured on the production W_c store (Si 4×4×4, 1128 ISDF centroids,
n_p = 8), one A100, BFC@0.85, on the real column block the production
walk takes — 70 columns × 1128 rows = 78 960 elements:

| | shipped | after |
|---|---:|---:|
| fit kernel | 55.2 µs/element | 55.2 µs/element |
| conditioning | 64.9 µs/element | 9.7 µs/element |
| **block** | **120.1 µs/element** | **65.1 µs/element** |

`gw/mpa/fit_conditioning.py` is `solve_conditioning` with the refit
deleted and nothing else touched, so the four values it returns are
byte-identical rather than close, and the suite asserts that as byte
equality. Five cells in `tests/test_mpa_fit_driver.py` section (d2)
carry it, including the end-to-end one that compares every byte of the
finalized store against a reference computed by the old composition.

## TRAP 1 — the MPA fit is NOT bit-identical under `jax.jit`, and the poles move at 5e-9

This is the durable half. The obvious lever on a stage with no `jit`
anywhere is to add one; it compiles, it is about one per cent faster, and
**it changes the answer**. Measured on the same block:

| stage | `jit` vs eager |
|---|---|
| Pade solve coefficients | 2.1e-16 relative — one ulp |
| companion roots | 1.6e-12 relative |
| `jnp.linalg.eigvals` on identical input | **BYTE-IDENTICAL** |
| fitted Ω_p | 2.5e-11 relative |
| fitted B_p | **4.8e-9 relative** |

The eigvals is not the source; the SVD-based solve is, and the
root-finding amplifies one ulp of coefficient by about 7 600x, after
which the residue least-squares amplifies it by another 3 000x. So a
one-ulp perturbation anywhere in the Pade solve arrives in the stored
residues at 5e-9 relative. **Anything that touches the solve — a
reconditioning, a different equilibration, a `jit` — must be judged
against the residues and not against the coefficients**, and any lane
that wants the fit store byte-stable cannot have `jit` on this path at
today's arithmetic.

## TRAP 2 — the companion-root `eigvals` DOES have a GPU lowering, and it costs 28 µs, not 119

A prior study recorded "no GPU lowering", "119 µs/element" and "scales
n_p^1.0 — per-element call overhead". All three are corrected here, from
the compiled HLO and from a batch/n_p sweep:

* `jnp.linalg.eigvals` lowers to `stablehlo.custom_call
  @cuhybrid_eig_comp` (`magma = "auto"`, `num_batch_dims = 1`) and the
  same target appears in the COMPILED module. There is no
  `TransferToHost`, no callback and no `lapack_*` target in either.
* At n_p = 8 and batch ≥ 1024 it costs **28.1 µs/element**; the shipped
  block costs 120.1, which is within noise of the study's 119 and is the
  reading the study's number is most likely to be.
* It scales as **n_p^1.87** (7.8 / 28.1 / 60.8 / 105.9 µs at n_p = 4 / 8
  / 12 / 16), not n_p^1.0. The flat-in-n_p behaviour the study inferred
  is the batch-1 regime (157-292 µs/element), which the vmapped
  production path never enters.
* The CPU backend costs the same to three digits, so moving the roots to
  the host buys nothing.

Evidence for every number above:
`/pscratch/sd/j/jackm/mpa_fitperf_0810/_reports/probe_probe2.log`,
registered in `/pscratch/sd/j/jackm/EVIDENCE_MANIFEST.md`. Branch
`perf/mpa-fit-efficiency-2026-08-10`.
