# ARCH1: bounded legacy Dyson product, no shared-bank speedup

Production change: `3b956c2`, tested at Na-sized P16 in `4d34957`, on
`arch/sp-arch1-2026-09-09`. The product lives behind `distrib_la.panel_matmul`;
`w_isdf._a_local` only masks faces, calls the service, and subtracts from I.
The shared-bank hoist was measured twice, rejected at the unchanged Si gate,
and reverted. Its existing local/distributed algebra is unchanged.

## Actual dataflow and ruling 40 classification

Read references: `src/gw/v_q_g_flat.py:54,102-145` keeps the reshard on the
mu/G face; `src/gw/downfold.py:1496-1499` inherits two-stage psum_scatter.
This change introduces no all-to-all or driver-side layout selection. Large
matrix products enter the service with both mu axes tiled over named x/y.

Let Q be the current q chunk, N the full q count, P=Px Py, and m the padded
matrix side. Complex128 is 16 bytes. Class L means total element count at
least m²; S means at most m²/4; M is between (treated as L). Classes count
**the whole q stack**, not one parent and not one rank. Integer/bool bytes
are stated separately. Na product gate uses Q=N=29,m=896,Px=Py=4;
its matrix reference size is 802,816 elements / 12,845,056 bytes.
The live bank uses its own 960 carrier; it never calls this legacy factory.

| Object, creation order | Actual generation / layout | Shape; full bytes | Per-rank bytes | Class at Na |
|---|---|---|---|---|
| V and chi inputs | caller faces, `P(None,x,y)` | each N m²; 16N m² | each 16N m²/P =23,281,664 | L, service operands |
| Scaled chi | `_scale(chi,pref)`, donation, same face | N m²;16N m² | same; chi buffer consumed | L, elementwise |
| A accumulator | `_zeros_like(V)`, face | N m²;16N m² |23,281,664 | L, elementwise allocation |
| V/chi q blocks | slices passed to `_a_chunk`, face | each Q m²;16Q m² |16Q m²/P | L; no whole-q matrix on one rank |
| row/column indices and booleans | `_logical_tile`: local iota+broadcast predicates | global row/column length m; int32 4m each; mask m² bool | indices 4m/Px,4m/Py; mask m²/P | S indices; L distributed mask |
| masked operands | `_mask_face`, face (no host materialization) | each Q m²;16Q m² |each16Q m²/P | L; service inputs |
| product accumulator C | service `zeros([Q,1,m_X,m_Y])` | Q m²;16Q m² |23,281,664 | L, service face |
| left panel, mask, broadcast | service dynamic_slice then masked `psum(y)` | Q m b;16Q m b |16Q m b/Px =11,640,832 | L at3.625m² total; internal service workspace |
| right panel, mask, broadcast | service dynamic_slice then masked `psum(x)` | Q b m;16Q b m |16Q b m/Py =11,640,832 | L at3.625m² total; internal service workspace |
| local product/update | service `old+left@right`, update_slice C | Q m²;16Q m² logical |16Q m²/P; HLO folds beta=1 update | L, service face |
| loop indices | service scan over m/b, samples | int32 vectors;4m/b and4S |32 bytes for m/b=8; S=1 | S |
| identity tile / A block | `_identity_minus(product)` then donated update of A | m² and Q m²;16m²,16Q m² |802,816 and23,281,664 | L, elementwise faces |
| B RHS / W | `_copy_zero_pad(V)` then existing `plan(...).batched(A,B)` | each N m²;16N m² |each23,281,664, plus service LU workspace | L, existing service solve |

`src/gw/w_isdf.py:1402-1471` owns the masks, q budget and calls.
`services/distrib_la/src/distrib_la/_panel_matmul.py:45-94` owns panel sizing,
loops, GEMM and shardings. Broadcast panels are **not** justified by the
small-object exception: their all-q total is L. They are bounded, temporary
workspace inside the distributed service, not full matrix rows exposed to a
driver. Inputs, output and accumulator remain 2-D faces. This is the bounded
panel loop explicitly requested by rulings36/40.

The caller admits a pair-panel budget of one Q-face (with a tiny one-column
floor); the service chooses a divisor of gcd(m/Px,m/Py) within that budget.
For square meshes, b=m/(2 sqrt(P)) when divisible. At Na b=112; each live
operand panel is half a face/rank. At fixed Q with m² proportional to P,
face and panel residency stay constant. Retired full-row/column operands
cost16Q m²/Px and16Q m²/Py, growing as sqrt(P) in that limit. The small
integer/boolean workspace cannot change that asymptotic result.

The generic service also accepts a sample axis: input/output faces then
scale by S, the left panel is reused, and only one right sample-panel is
live. Production legacy Dyson has S=1. Hoisting into the shared bank did
not pass its matrix gate and is not shipped.

Compiled product temporaries are190,448,384→27,477,816 bytes/rank at Na
P16, including compiler/native GEMM workspace; operand/output allocations
are unchanged. These are executable memory-analysis bytes. A separate fresh-process
actual-solver gate is recorded below. The real shared-bank control's allocator
peak is separately measured and unchanged at4,286,200,816 bytes/rank.

## Communication and measured consequences

For S samples, panel loops execute (m/b)(1+S) masked all-reduces, versus
2S full all-gathers in the retired product. A ring all-reduce need not have
the same wire cost as a root broadcast: the zero-mask implementation can
pay a reduction phase as well. No wire-byte saving is claimed. J,K and the
number of bank time nodes do not enter this legacy product; frequency
sample hoisting in the real bank was rejected.

All artifact paths below are relative to
`S/runs/frequency_integration_sandbox/330_arch1_20260909`.

| Receipt | Result / scope |
|---|---|
| P4 kernel58128234.1, `02_dyson_p4/receipt_rank0.json` | Q8,m368: median1.217→0.894ms; temporaries21.529→8.529MB; product1.56e-15, poisoned-pad Dyson3.62e-16 relative |
| Si strict58128234.5, `07_si_legacy_fix_p4/artifact_check.json` | identical K; every CC†/CΛC†/W(iu) invariant and analytic Sigma array/metric difference exactly0; all4 peaks unchanged |
| Na kernel58128234.7, `08_na_bank_pair_p16/panel/receipt_rank0.json` | Q29,m896: relative2.91e-15; median71.854→103.829ms; all16 PASS; memory fix, timing regression |
| Na bank58128234.7, `08_na_bank_pair_p16/after/bank_parity.json` | all1624 W/dW/M1/M3 rows exact, amplitude red detected; all16 peaks unchanged; zero legacy-factory calls |
| Actual Dyson58128234.9/.10, `10_solver_peak_p4/artifact_check.json` and `11_solver_peak_p16/artifact_check.json` | Independent complex rank-one inverse oracle: relative9.72e-16 atP4,9.08e-16 atP16. P4 all-rank peak34,532,060→30,351,184B. P16 maximum279,382,748→270,567,380B, no rank rises; ranks0–3 unchanged. Fresh processes, production-sized synthetic faces, not physical Na W. |
| Na bank owner host bands, same leg, `before/` and `after/arch1_stages_rank0.json` |119.880→128.073s bank;19.062→19.400s moments. Same bank source; not attributed to the uncalled product |
| Five-call P4 NSYS58128234.8, `09_panel_traces_p4/p4/{before,after}/analysis/rank0.json` | NCCL10→40 kernels; summed kernel1.630→2.681ms |
| Five-call P16 NSYS58128234.7, `09_panel_traces_p4/p16/{before,after}/analysis/rank0.json` | NCCL10→80 kernels; summed kernel0.321193→0.530700s |

HLO receipts in `09_panel_traces_p4/{p4,p16}/hlo_after` pass the registered
no-all-gather/no-all-to-all check. The P16 scan holds `[29,224,112]` and
`[29,112,224]` operand panels, never `[29,224,896]` or `[29,896,224]`.
NSYS capture wall times include profiler activation/collection overhead;
use summed kernels for the quoted collective time, not capture wall.

Rejected bank-hoist trials58128234.3/.4 have identical K but invariant
maxima3.355e-9/3.405e-9 versus1e-9; sample Dyson0.4066→0.8078/0.6942s.
Second trial's Sigma-path row delta0.000348meV passes independently; it
cannot override the failed matrix gate. See claim2033 and
`05_si_wider_p4/rejected_source.patch`.

Hardware counters58128234.6 cover the actual Na P16 stream at three time
nodes: FFT DRAM20.97–25.71%, achieved occupancy56.1–56.5%, theoretical62.5%
limited by46 registers/thread; lg_throttle dominates and there is no spill.
Accumulator DRAM88.24%, occupancy95.79%. See claim2038 and
`06_stream_counters_p16/counter_summary.json`. DCGM resumed. FFT layout is
now AFFT's lane; no new kernel or attainable speedup is claimed here.

Verdict: retain the service-owned memory fix; do not credit it with a bank
speedup. The actual bank hoist is rejected. Si strict parity and Na bank
parity are measured; a new Na constructor/Sigma/CD campaign is not claimed.
The actual-solver peak test is archived verbatim as
`tests/multi_device/dyson_solver_peak.py`; its measurement harness has the
same bytes. LU and other workspace can dominate the total solver peak,
so the large product-temporary reduction is not a comparable reduction
in the full-solver peak. No physical-sample full-solver peak claim is made.
Static gate0 fails the same inherited rules/ledger lint on baseline and
candidate (`logs/gate0_control.log`, `logs/gate0.log`).
