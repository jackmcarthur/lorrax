# ARCH3 — reusable stage programs, safe executable persistence

**Claim.** Reuse stage callables across samples and self-consistent maps, then persist only executables whose runtime resources can be rebound safely. This targets the measured 160.6–162.2 compiler seconds, including 805 constructor compilations, rather than promising the entire 424.9 s cold/reuse difference.

Design only; source inspected at `b16fe23`, branch `arch/sp-arch3-2026-09-09`; no compute or implementation. Paths below use the brief's S/F/D; Run325=`F/325_pbank_20260909`, Run327=`F/327_pland_20260909`, Run324=`F/324_pconb_20260909`.

## Evidence and module census

The matched FD MPA reference **ran**: 58121258.2, `Run327/03_na_fd_mpa_p16/run.log`, 1082.23 s, 479 tau nodes. Its available log/output contain no module-count/compiler-time receipt. Therefore an MPA count and a measured shared/MPA module ratio are **unestablished**; the historical Si “220 modules” is not this deck's baseline.

Shared-pole 58124441.2 has **1417** backend compilations/rank, **805** inside construction and **612** outside (`Run327/na_rule_reuse_compile_summary.json`). These count compilation events, not unique physical kernels. The directory initially had no executable entries: copied rules are not a warm JAX cache. The Sigma-only pair has 179 wrapper compilations/11.38 s, a different scope (58124085.0, `S/reports/shared_pole_push_2026-09-07/perf/psigma/report.md`). No per-cause numerical census exists in these receipts.

Source explains where to attack, without inventing counts:

| Owner, file:line | Cause and treatment |
|---|---|
| `src/gw/response_bank.py:153,681` | Calls the uncached contour builder directly for each Laplace cell. Reuse by mesh/layout, mode and admitted shapes; current weights/references remain operands. PBANK's later `c099589` addresses this; integrate its owner, do not duplicate it. |
| `src/gw/response_bank.py:618,649,669` | Capacity search compiles successive widths; fresh zero closure per panel. Cache zero creation; use monotone ledger sizing before AOT, retaining final exact workspace verification. |
| `src/gw/shared_pole_local.py:79,142` | Cached reducer specializes on the whole parent-extent tuple and builds switch branches per parent. Distinct eigensolve extents are structural; duplicate extent pairs and per-call native callable identities are avoidable specialization. Deduplicate branches, reuse service plans. |
| `src/gw/shared_pole_constructor.py:76,807` | Eager diagonal builder and distributed selector recreate JITs. Move into reusable owners; direction selection, receipts and writer use the same policy. |
| `src/gw/mpa/sigma.py:287,398,408` | Per-panel unfold closures and compact `(first,last)` kernels; reuse code with map tables/offsets as operands. Keep compact widths: replacing width540 by1589 would undo measured work savings. |

The incumbent caches its fit kernel by mesh/pole count/solver (`mpa/fit_driver.py:145`), passes logical counts dynamically, and masks redundant poles without shrinking shapes (`mpa/pade_fit.py:740`). Copy that discipline, not its model. The floor is one executable per genuinely distinct shape/layout/backend/static physics mode, plus I/O/runtime programs—not literally one module per stage. Its numerical value needs the missing census.

## Small launches are not the prize

58124680.0, `Run325/18_bank_profile_final_p16/trace/analysis/rank0.json`: eight fusion families contribute 675 small launches each, 913 are small cuBLAS kernels, six are setup slices/transposes. Their total is **0.086563904 s**; making every one free gives **38.4810 s**, versus38.5676 s. GPU idle within the activity span is only0.0638 s. This does not prove zero device launch latency, but rules out a large observed host-starvation gap.

The loop already is `lax.scan(unroll=1)` (`w_isdf.py:1120`), with donated carry at:988. Fusion names indicate scalar phase/weight, slicing and loop bookkeeping; exact instruction attribution is not certified: the report's cited `trace/hlo/real_time.txt` is absent at inspection. Recover that artifact before assigning each fusion a source expression. No fusion kernel, time batching, or donation rewrite is justified by this row.

## Dataflow, bytes and communication

Keep physics drivers and existing services. Runtime owns executable lifetime; stage factories hold only immutable plan descriptors. Each map supplies fresh energies, occupations, poles, factors, indices and counts. I/O handles and resident arrays remain map-scoped. Retain sequential donated scan carries, bounded bank panels, original-extent service eigensolves, and the existing Sigma synthesis→spatial owner boundary. Reuse moments/selection/writer callables without caching their data. Both local and distributed routes remain selectable.

Before=after, complex128, P=PxPy, μ=padded extent, S=sample points, T=time nodes, Q=parents, Qf=full q:

- Bank carry `[2S,Q,μ,μ]`, `P(None,None,'x','y')`: **32SQμ²/P bytes/rank**; full-q field **16Qfμ²/P**. Panels replace S,Q by admitted capacities; no T-sized field stack.
- Existing factor views `[Q,μ,1,K]`: **16QμK(1/Px+1/Py)**; replicated poles/counts O(QK). No new views or larger padding.
- Native Green broadcasts: **16T·16Q(μ/Px)(Nb/Py)** logical bytes/rank for the square Na schedule; unchanged. Sigma synthesis remains collective-free; spatial collectives remain unchanged. Constructor retains existing bounded action reshard and inverse all-to-alls; incremental network bytes are zero.

Na: Q=29,Qf=512,μ=960 (logical896),Px=Py=4,Nb=88,S=27,T=675,Kcap=1589. Bank carry=1,443,225,600 B/rank; field=471,859,200 B; factor-view capacity=353,902,080 B; stream logical payload=26,459,136,000 B/rank. These are separate live-object terms, not an added peak or measured wire traffic. TΣ=251 likewise adds no resident time axis. Admission must show unchanged peak, including executable/native workspace.

## Cache contract and exposure

`common/jax_compile_cache.py:1441` strips handle literals only for compile agreement; `distrib_la/matmul.py:302` embeds a process-local context pointer. This is a concrete divergence mechanism, **not a proven attribution of the Na key diff**. The Sigma pair requested317keys/rank,307common,467union (58124085.0, PSIGMA report above); this does not identify which component differs. Compare differing key components/IR first. Never merely strip a pointer from the persistent key: the executable would still contain it. Services must supply runtime operands or stable resource IDs resolved in the current process; otherwise those programs remain nonpersistent.

Namespace by sealed source-content inventory, native binaries/dependencies/ABI, JAX+jaxlib/plugin build, XLA flags, accelerator target and mesh geometry; executable keys retain shapes, shardings and static semantics. Extend the existing runtime owner using PCONB's archived candidate. Use a private shared scratch sequence cache, outside iteration directories; one-shot runs get a run-owned directory. Freeze the all-rank readable intersection, atomic rank0 writes, no eviction while readers live. Keep the existing one-second threshold, not zero; bound files/bytes/write budget at run boundaries. The retained Si cache trial reached386.3→15.7 s, but its numerical gates never ran (58124662.1, `Run324/15_si_cache_cold` and `16_si_cache_warm`). The315.5 s cumulative write receipt is not additive wall time.

## Decision and confirming measurement

**Worth doing after landing**, starting with callable reuse; default-on is conditional. One future combined P16 campaign: matched MPA census, shared cold/repeat/new-process reuse, resource/source/native/version/geometry invalidation twins, and local/distributed gates. Budget approximately240 node-minutes, an estimate; no leg launched. Require identical requested key sets across16ranks, fewer compilations, bounded writes, unchanged peak, unchanged K/gauge/passivity and Na/Si Sigma rows. Padded-Gram solver failures, stale-resource reloads, changed reduction order and frozen map inputs are the main exposures. Failure of any receipt keeps persistence opt-in.
