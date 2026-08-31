# Certified causal crossing tables — result

Branch: `feat/causal-crossing-tables-2026-08-31` (base `f1c3f2d3`).

## Delivered

Added the `crossing_causal` / `causal_reciprocal` service family for

`Q(x+i*g) = sum_r alpha_r exp(i*t_r*(x+i*g))`, `t_r > 0`,

with the certified convention

`max_{|x|<=A, 1<=g<=100} g*|Q(x+i*g)-1/(x+i*g)| <= error_bound`.

Runtime scaling therefore gives physical absolute error at most
`error_bound/gamma_min`.  MPA lookup now requests `crossing_causal` only;
the GN-PPM HGL family is not a candidate.  The single pre-existing runtime
fallback remains unchanged.  All six payloads have complete generator,
backend, continuum-certifier, payload-SHA, and bit-identity provenance.

| A | requested eps | nodes | dense family error | certified bound | kappa0 |
|---:|---:|---:|---:|---:|---:|
| 10 | 1e-4 | 60 | 2.008817527940e-5 | 2.018836630130e-5 | 1.000000000001 |
| 10 | 1e-5 | 71 | 2.374999991228e-6 | 2.375000077625e-6 | 1.000000000000 |
| 20 | 1e-4 | 64 | 2.312500000556e-5 | 2.467896873052e-5 | 1.000000000000 |
| 20 | 1e-5 | 78 | 2.000000034363e-6 | 2.447183530883e-6 | 1.000000000000 |
| 40 | 1e-4 | 122 | 2.312499995116e-5 | 2.344319307216e-5 | 1.000000000003 |
| 40 | 1e-5 | 147 | 2.249999992432e-6 | 2.368819903358e-6 | 1.000000000000 |

## Verification

Focused CPU gate: **100 passed** in 18.93 s.  This includes
`tests/test_delivered_windows.py`, minimax door/refusal/import-isolation
tests, all six new payload pins, and a test proving MPA crossing lookup opens
`crossing_causal/causal_reciprocal` and never HGL.

## P=4 acceptance refusal

The requested Na `-10..+10 eV` deck was run at P=4/BFC@0.85 from the branch
checkout.  It does **not** reach a plan or Sigma artifact with the mandated
`A in {10,20,40}` catalog, so no checker PASS or per-patch receipt is claimed.
The production router refuses before lookup because even a one-omega-row
causal patch has `A_dim > 40`:

`crossing support cannot be served by omega product windows: even one-row patches exceed the widest shipped causal span A=40`.

This is not a table-error or conditioning failure.  The frozen Na measure
independently gives crossing radii 45.6628 and 43.2063 in eta units.  An exact
state×omega split was tested as a diagnostic: all crossing pieces became
table-servable, but exact coverage required 30 product windows and at least
1,335 `(window,tau)` pairs, versus the deck's 500 effective ceiling (420
requested).  That experiment was fully reverted; the delivered branch keeps
the original fallback and product router unchanged.

Evidence directory:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/60_sc_delivered_20260831/codex_causaltables_probe_20260831`.
The branch-correct refusal is `launcher.causal_span_refusal.log`; the bounded
split census is preserved in `launcher.minimum_1335_refusal.log`.

The remaining owner decision is therefore explicit: add a certified causal
range above A=45.663 (outside this brief's declared A set), or raise the
product-pair resource contract.  Weakening the A=40 certificate, applying it
outside its family, or restoring direct pair evaluation was not done.
