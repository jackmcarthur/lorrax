# ARCH2: keep operator convolution; give certificates durable ownership

Restructure Σ accounting and certificate lifetime, retaining the existing full-q convolution: run-local rule reuse addresses the measured 121.01→0.26 s planning cost without increasing device residency. Name the 57.96/26.13 s cold/warm residual before optimizing it; neither those seconds nor their overnight increase are demonstrated removable overhead.

Evidence abbreviations: S is `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14`; F=`S/runs/frequency_integration_sandbox`. Cold C=`F/328_coord_assembled_20260909/01_na_p16_cold/gwjax.out`, job **58124441.0**; warm H=`F/327_pland_20260909/11_na_rule_cache_reuse/production/gwjax.out`, **58124441.2**. These supply the opening timings and 39.68/31.82 s sweeps; allocator is not recorded in their stage summaries, so differences are not causal attribution. Code references below are at **b16fe23ff2d831809ffa53f0765cc0fde3fd0f41**. No compute or implementation performed.

## Materialisation: algebra permits alternatives, not the proposed shortcut

`src/gw/ppm_tau_kernel.py:347` computes, suppressing normalization,

\[
\Sigma_k=\mathrm{FFT}_R[G_R\odot W_R],\qquad
W_R=\sum_q e^{iqR}\sum_j C_{qj}d_{qj}C_{qj}^\dagger.
\]

Transforming C and multiplying its transforms introduces independent q,q′ cross terms. The Fourier transform acts on the q-dependent outer product; it does not commute through that product. There is also no common pole-column gauge across q. An exact factor representation at R retains the compound index (q,j), of width QK, or evaluates the explicit k,q,band vertices. The latter removes W but replaces the FFT convolution with a q sum per external k and internal band. Since K exceeds μ here, neither is an attractive rank reduction. This is a cost argument, not an impossibility theorem.

Symmetry is manageable: `src/gw/mpa/sigma.py:191` already routes child factors through `symmetry_maps.unfold_endpoint_panel`; antiunitary actions conjugate factors, never causal weights. Fixed-q projection needs the partner contribution too. Retaining all child factors would violate the footprint objective. Streaming q/K panels avoids that residency but repeats transforms/projections or retains a dense spatial accumulator.

Keep the implemented dataflow (`sigma.py:249`): store owns canonical `[B,μ,1,K]`; admitted X/Y faces use `P(None,'x',None,None)` / `P(None,'y',None,None)`. `distrib_la.contract_faces` synthesizes two parent orientations, the symmetry service unfolds them, and one `[Q,μ,μ]`, `P(None,'x','y')`, enters the existing spatial owner. Release node temporaries before advancing τ. Both linalg policies retain this route; incumbent MPA receives precisely its existing spatial call.

## Bytes and collectives

Let B=Nq,irr, Q=Nq,full, P=PxPy, μ=packed extent, Sₙ=sample count, T=total window/time pairs; complex128 costs 16 bytes. These are component sizes, not compiler peaks:

| Per-rank object | Before and recommended after |
|---|---:|
| Parent factor faces | 16BμK(1/Px+1/Py) |
| Weighted X | 16BμK/Px |
| Two parent operators | 32Bμ²/P |
| One full-q W | 16Qμ²/P |
| Sample bank, per value/derivative channel | 16SₙBμ²/P, screening only |
| Pole census / complex nodes and weights | 8BK / 32T |

Na: B=29, Q=512, μ=960 (logical 896), Kmax=1589, Px=Py=4, Sₙ=27, T=251. Faces=353.90 MB, weighted X=176.95 MB, parent pair=53.45 MB, W=471.86 MB, bank/channel=721.61 MB, census=0.369 MB, rules≈8 kB. Actual live-column model storage is 556.72 MB globally; it is not the padded face residency. Shape receipt: **58120076.7**, `F/326_psigma_20260909/08_na_candidate_p16/schedule_before_rank0.json` (planned synthesis peak 1.413 GB, not measured peak).

An all-child factor pair instead costs 16QμK(1/Px+1/Py)=6.248 GB/rank: rejecting it saves memory. Existing local synthesis/unfold has zero collectives; operator writes alone total at least 16TQμ²/P≈118.44 GB/rank over the sweep. Faces require collective reads and packing once if resident; routed fallback exchanges endpoint panels. Existing G/projection communication and FFT scratch remain unchanged. No custom kernel or communication-speed claim follows from this algebra.

## What “Σ other” actually contains

`src/gw/production_report.py:575` defines it as **whole Sigma minus plan minus sweep**, including SC-driver work for SC runs. For these one-shot runs:

- Bare exchange and live Hartree: `sigma_dispatch.py:1290,1321`.
- Model validation, including complete bounded payload hashing: `mpa/sigma.py:1312`; `shared_pole_store.py:561,461`. Hashing rereads factors in q/column panels and reduces row digests; it is integrity work, not passivity validation.
- Capacity planning, branch preparation, pole census and summary adapter: `sigma.py:1324,1335,1359`.
- Finalization: interpolation, spectrum HDF5 output and QSGW assembly (`sigma_dispatch.py:563`).

Crucially, face reads, synthesis compilation, fixed-q diagnostic and actual projection are **inside sweep**, `sigma.py:1465,326,343`; constructor receipts belong to screening. They cannot explain this residual.

A narrower earlier receipt actually names Hartree 0.025/1.128/1.142 s, finalization 3.939 s, and unpartitioned exclusive Sigma 11.955 s: **58120076.7**, `08_na_candidate_p16/receipt_rank0.json`. These are not an attribution of C/H’s regression. Their subphase timings are missing.

Keep capacity, authentication, census and fixed-q projection in production. Move only repeated warning-only fixed-q *measurement* to model sealing, preserving one receipt per current model and its gate; reuse admitted faces. Do not drop payload authentication on restart or trade added residency for fewer reads. First expose named residual timers and distinguish I/O, compilation, execution and waits.

## Rule ownership and acceptance

Run-local is already default (`sigma_box_plan.py:52`): `<input_dir>/tmp/sigma_quadrature_rules`. Preserve it with restart artifacts; SC maps should resolve the stable run root. Recompute current bands/occupation selectors/pole intervals every map. Shared-pole deliberately bypasses frozen SC sessions (`sigma.py:1409`).

Reuse a scalar rule when its certified box contains the new domain, currency/tolerance match, and current factor-growth/noise checks pass (`sigma_box_plan.py:240,495`). Moving poles can satisfy containment; unchanged poles alone are insufficient. Keep map provenance separate from reusable certificates. Harden persisted identity with schema, rule-family/certifier version, Ry units, box, currency, tolerance, reduction policy, node/weight digest and certificate/noise convention; validate on load. Current filenames hash payloads (`:343`), but lookup does not authenticate that hash/version. Corrupted weights, changed η/occupations, boundary escapes and incompatible versions need refusal/rebuild twins; never freeze poles or loosen tolerance.

## Decision and decisive measurement

**Worth doing now:** accounting and durable authenticated rule reuse. **Not worth doing now:** factor-direct convolution. One combined P16 restart replay, cold then cached, with fenced named residual subphases and per-rank compile counters, would confirm whether hashing/diagnostics merit restructuring; budget **32 node-minutes** (four nodes, eight minutes), estimate only. Preserve raw Σ, eqp and peak-memory gates, add cache-corruption/domain-escape twins, and retain Na CD48/Si CD96 accuracy gates for implementation. Implementation must also pass local/distributed Si P4 twins, antiunitary and fixed-q complex-τ parity, and unchanged MPA output. Reassociation exposes summation-order error and the 1/(2Ω) normalization; the existing all-q complex frontier Σ comparison catches both. No speedup beyond existing receipts is claimed.
