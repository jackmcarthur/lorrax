# SC delivered-planner instrumentation result

## Outcome

Commit `39f9c893` emits one rank-0 JSON line prefixed
`[delivered-planner-window]` for every product window on every live or
fit-cache planner call. Each record contains the window name and kind, cell
count, crossing radius, `gamma_min`, `A/eta`, `A/gamma_min`, denominator
scale span, delivered-envelope mass share, apportioned target, candidate
family, best residual and kappa, source, and served/refused status. Parallel
fit errors are gathered first, all window records are flushed, and only then
is the first refusal raised. The P=4 shipped-path diagnostic proves the
failure case: all 12 records appear before the traceback, including the
refusing row.

The signed ±5 eV P=4 SC run reproduced map 1 with 12 windows, 161
`(window,tau)` pairs, and `max|dE|=1.653059 eV` (not converged). The
comparison below uses its initial live census and the next-map live census.
The full source is `39f9c893`; the diagnostic receipt is JID 57766417,
step 10, four A100 ranks. `LORRAX_BAND_DEGENERACY=snap` was inherited from
the established legacy 24-band probe and is not a planner dial.

## What changed in `ω≥E_F cond:resonant`

| field | SC map 1 | SC map 2 | change |
|---|---:|---:|---:|
| crossing radius (Ry) | 0.845207443420175 | 0.856896768933017 | +0.011689325512842 (+1.3830%) |
| `gamma_min` (Ry) | 0.0185097728798629 | 0.0185562891774594 | +0.0000465162975965 (+0.2513%) |
| `A/eta` | 45.9985491129046 | 46.6347148469916 | +1.3830% |
| `A/gamma_min` | 45.6627668478684 | 46.1782396651752 | +1.1289% |
| scale span `max|d|/min|d|` | 51.4715552913198 | 52.0557354930722 | +1.1350% |
| cells | 649 | 648 | -1 (-0.1541%) |
| delivered mass share | 0.0127266767752052 | 0.0828748925544063 | 6.5119× (+551.19%) |
| apportioned target | 0.00294398427419215 | 0.000466503083806718 | 0.15846× (-84.154%) |

This is not a radius-bracket transition. Both `A/gamma_min` values are
above 40 and below 60, so shipped lookup selects the same A=60 HGL family in
both maps. The earlier “40→60 between SC iterations” explanation is false.

## What the fit is sensitive to

The large change is the measure, not its outer support: this window receives
6.512× more delivered-envelope share and therefore a 6.305× tighter relative
allowance. Their product changes only 3.19%, as expected from global budget
apportionment. The one-percent radius/gamma/span changes do not change the
catalog family.

The fit does not consume the scalar diagnostics as independent knobs. The
measure-adapted path consumes `eta`, the apportioned target, and every
frequency, internal-sum cell, and cell mass. The shipped path selects by
`A/gamma_min` and scaled target, then validates and applies factor-growth
gates on the full measure. Thus cell count, mass share, radius and span are
summaries of arrays to which the fit is sensitive, not sufficient causes.

There is direct family evidence on the same map-2 problem. The parent
branch's primary measure-adapted rule serves it at residual
`6.99058e-5`, `kappa_p99=31.3887`. A P=4 diagnostic that disables only
that primary candidate and exposes the shipped fallback refuses before the
traceback at residual `0.0110868`, `kappa_p99=2360.05`. The historical
pre-parent control also refused this window (`0.00483183`, `4197.69`).
So the map-2 measure is hostile to the shipped family, but is not intrinsically
unservable by a product window.

One parent-branch caveat matters: at `0761062d` the primary ROQ now serves
this particular window and the later global selector instead refuses
`ω<E_F val:pole_tail`; a fresh no-receipt map-1 plan likewise encounters a
later selector refusal. Those are parent behavior, not caused by this
logging-only patch. I propose no fix: the scalar diff identifies the mass
redistribution and tighter allowance, but does not establish why the shipped
residual and cancellation deteriorate.

## Evidence and gates

- P=4 SC/current-base log: `codex_scinstrument_probe_20260831/p4_scinstrument_attempt5.log`
- P=4 shipped-path failure log: `codex_scinstrument_probe_20260831/p4_scinstrument_shipped_diagnostic.log`
- Frozen map-2 fit copy SHA-256:
  `e052930b58b04e6008858d40839301db76f984a03448bd54f17a843ed7d62a4a`
- Focused CPU gate: **124 passed**, 2 warnings, 64.92 s; log:
  `results/codex_scinstrument_cpu_gates.log`
  (`test_delivered_windows.py`, `test_hybrid_wiring.py`,
  `test_delivered_executor.py`, `test_layering.py`).
