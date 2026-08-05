# `LORRAX_W_RESIDUAL_CHECK` — contamination report + harness-fix RECIPE

wk_REL scale10k, 2026-07-29.  Coordinator item 2.
**No production harness was edited.**  Other workstreams are mid-flight on
`l5_*`/`l7_*`/`sigma_*`; this file is the recipe for the coordinator to land,
in the same form the audit agent used for the Gloo-pin change.

## 1. Footprint (measured, `/scratch2/08271/jackmc`)

| quantity | value |
|---|---|
| `export LORRAX_W_RESIDUAL_CHECK=` lines | **103** |
| ... set to `1` | **103 — all of them** |
| ... set to `0` / `off` / left unset | **0** |
| distinct `*.sbatch` harnesses setting it | **24** |
| run dirs where it FIRED (`[W solve] Dyson residual` in `gw.log`) | **65** |

The 18 in `mos2_4x4_test/`:

    aq_rehearsal          diag_b512_weap        gate_sigma_reference
    l1_b256               l2_b256_c3491         l2_b256_c3500
    l3_b512_c5000         l4_b512_c7000         l5_b1024_c10000
    l6_r45_b2048          l7_b1024_bigmu        omega_512cell
    omega_ab              sigma_haccum2         sigma_hostaccum_gate
    sigma_iter            sigma_perf_ab         wres_ab_p64 (this workstream's)

**That is the entire size ladder l1..l7 plus the whole sigma-perf A/B family.**
Every `W.exec` / `gw_jax.chi0_W` wall quoted from a ladder or sigma-perf run was
therefore measured with a diagnostic the code says must be off.

## 2. Why it matters

`gw/w_isdf.py:546-569`, the function's own docstring:

> "Diagnostic-only, opt-in via `LORRAX_W_RESIDUAL_CHECK=1`; **never on in the
> traced production path, so the collective-table gate is taken with it OFF**."

Its `_res` jit is HLO modules 0656 / 0732 in the rung-5 dump
(`wk_REL/results/colltable_L5_10k.txt`): **5 all-gathers of 807.70 MB per W solve, twice
per run = ~8.1 GB/rank** of purely diagnostic collective traffic, inside the
`W[static]`/`W[probe] Dyson solve` stage timers.  It also invalidates any
collective table taken from such a dump for those two modules.

## 3. Measured cost — A/B at mu=10015, P=64 (job 7879529)

Restart-gated re-entry (AC.4) on the rung-5 tensors; both legs identical except
the flag; each killed after `Finished screening` (Sigma is irrelevant here).

| stage | ON (=1) | OFF (=0) | delta |
|---|---:|---:|---:|
| `W[static] Dyson solve` (10 q) | 55.8 s | **50.2 s** | **-5.6 s (-10.0 %)** |
| `W[probe] Dyson solve` (16 q) | 70.7 s | **66.2 s** | **-4.5 s (-6.4 %)** |
| `Finished screening` | 179 s | **168 s** | **-11 s (-6.1 %)** |
| residual lines emitted | 2 | 0 | flag verified effective |

**~10 s per run: 6 % of screening, 0.6 % of the 1811 s GW wall.**  Large
collective volume (~8.1 GB/rank), small wall.  Caveat: cross-allocation variance
on this stage is the same size as the effect (the rung-5 run recorded
`W[static] 46.3 s` WITH the check on), so this holds only as a same-allocation
delta.  **The cost to the campaign is evidence quality, not throughput:**
absolute W numbers from the 65 affected run dirs are ~6 % high, and the
collective table from those dumps is wrong for modules 0656/0732.

## 4. RECIPE — what to change

### 4a. The one-line harness change (apply to each of the 24)

The harnesses write an `inner.sh` containing an unconditional line. Replace:

```bash
export LORRAX_W_RESIDUAL_CHECK=1
```

with a parameterised default-OFF:

```bash
# Dyson-residual DIAGNOSTIC. Default OFF: it costs ~8.1 GB/rank of all-gathers
# inside W.exec and w_isdf.py's own docstring says it must not be on in traced
# runs. Turn it on ONLY for correctness gates:  sbatch --export=ALL,WRES=1 ...
export LORRAX_W_RESIDUAL_CHECK=${WRES:-0}
```

`WRES` then rides the existing `--export=ALL,...` convention the ladder already
uses for `CENTFILE`/`TAG`/`WEAPONS`/`SHARDED`/`RCHUNK`, so no harness grows a new
mechanism.

### 4b. Which harnesses should default it ON

None. The residual is the strict numerical contract of the *distributed* W plan
(a block-cyclic LU is not bit-comparable to a per-q local LU), so it belongs on
the correctness gates only:

* `gate_sigma_reference.sbatch` and any `*_gate.sbatch` → submit with `WRES=1`.
* `l1..l7`, `sigma_*`, `omega_*`, `diag_*`, `aq_rehearsal` → leave at the new
  default 0; submit one `WRES=1` leg per *new W plan or new size*, not per run.

### 4c. Guard so this cannot silently recur

Two cheap options, either is enough:

1. **Announce it.** `gw/w_isdf.py` already parses the flag; add one rank-0 line
   at resolve time — `print("  [W solve] Dyson residual diagnostic ENABLED "
   "(~5 all-gathers of <n> MB per solve) — not a production setting")`. The
   env-coupled-behaviour pattern (QUALITY_PATTERNS §8) says a capability that
   changes cost must announce itself; today it is silent except for the
   residual line itself, which reads like a health check rather than a bill.
2. **Fail the colltable gate on it.** The collective-table harness
   (`wk_AN/colltable.py` callers) should refuse when
   `LORRAX_W_RESIDUAL_CHECK` is truthy in the run's `inner.sh`, since the table
   is then not the production table. One `grep` in the reducer.

### 4d. Do NOT do

Do not delete the diagnostic or change its default *inside* `w_isdf.py` — the
in-code default is already `"0"` (`os.environ.get("LORRAX_W_RESIDUAL_CHECK",
"0")`). The defect is entirely in the harnesses, which override that default
unconditionally. The source is correct as written.

## 5. Claim-decay note for the scorecard

Any section quoting a `W.exec`, `gw_jax.chi0_W`, or `chi0_W_probe` wall from a
run in the 65-run list carries this contamination. Suggested banner, to be
placed wherever those walls are first quoted:

```markdown
> ⚠ CLAIM-DECAY (wk_REL, 2026-07-29): every ladder / sigma-perf harness exports
> `LORRAX_W_RESIDUAL_CHECK=1` (103 export lines, zero of them 0), so the quoted
> `W.exec` walls include the Dyson-residual diagnostic — ~8.1 GB/rank of
> all-gathers per run that `w_isdf.py` states must be off in traced runs.
> Measured effect at mu=10015/P=64: job 7879529. Re-measure before quoting a W
> number as production.
```
