# Sigma/screening/SC threshold audit — result

| Threshold family (source) | Quantity, units, and scale | Degenerate regime | Absorb or refuse; verdict |
|---|---|---|---|
| Laplace interval `_TINY=1e-12`, `x_max/x_min` padding, node floors (`minimax_screening.py:59,845-848,920-924,1006-1009,1110-1111`) | Positive transition denominator in Ry, dimensionless range, and rule size; scale is the nonzero occupation-difference support. | Metals make the fundamental gap nonpositive; invalid ranges/targets/node caps are turned into plausible requests. | **Silent defect THR-001/005.** Validate/refuse; derive a metal floor from the existing smearing/eta and live transitions. |
| Scaled tolerance and cache tolerance key (`minimax_screening.py:70-100`; `minimax/door.py:559-588`) | Physical absolute kernel error times `x_min`, dimensionless. | Small `x_min` legitimately produces `<5e-15`; old decimal rounding made zero. | **Fixed in parent:** significant digits survive; nonpositive refuses. |
| Offline node ceilings (`minimax/solver.py:205-216,572-581,807-824`; `door.py:591-641`) | Maximum rank; scale set by requested range/error and resource ceiling. | Target remains unmet at `N_max`. | **Fixed here:** the solvers still expose best-effort offline output, but the runtime door now refuses it instead of serving it (THR-003). |
| Remaining runtime cache rounding (`door.py:618-635`) | `log R`, `omega/x_min`, crossing `A`, and `eps_q`; all dimensionless target-function coordinates. | Values below/within `5e-13` alias a different function; cache hit has no exact-request validation. | **Silent defect THR-006.** Open; shipped lookup is unaffected. |
| Catalog selection slacks `1e-12/1e-18`, beta bands (`minimax/_catalog.py:348-385`; `beta_selector.py:525-622`) | Floating comparison slack on range/tier/eps_q; exact beta coverage is table-measured. | Only pathological sub-ulp requests; beta outside measured band can be arbitrarily large on collapsed `x_min`. | Catalog and beta selector **refuse**; slacks are numerical comparison scale and are fine. |
| `kappa0` 512-point measure and `R` floor (`door.py:356-372`) | Dimensionless cancellation amplification over `[1,R]`. | `R<=1` has no interval but is silently changed to `1+1e-12`. | **Silent defect THR-012** for invalid range; 512 samples are diagnostic only and fine after validation. |
| Solver IRLS grids/stops (`solver.py:114-202,469-581,596-824`) | Dimensionless optimizer tolerances (`1e-14`), grids (`max(1000,40N)` etc.), Lawson floors, and iteration caps. | Ill-conditioned/wide problems may stall or miss between grid points. | Offline numerics are acceptable only because achieved error is measured and the production door now refuses a miss. |
| GN-PPM `|Wc0-Wcz|>1e-14` (`minimax_screening.py:730-748`) | Absolute two-point-fit denominator in W units. | Pure rescaling/system size changes which entries get fallback poles. | Counted but silently substituted: **open THR-011**; needs a relative conditioning measure before change. |
| GN tails `1/500`; arena `8 GiB x 32 tiles` (`minimax_screening.py:61-67,520-541`) | 0.2% physics coarsening; host-selected memory chunk size. | Very few valid modes gives zero coarsening; huge systems over-chunk. | Fine: explicit owner policy and exact budget; memory constants change placement only, with a conservative measured factor. |
| MPA model equality and zero-head dummy pole (`mpa/model.py:35-76,145-180`) | Stored sample frequencies in Ry and an immaterial `-i Ry` pole with zero residue. | Changed plan/deck or nonzero head under head-off. | Exact array equality/nonzero head **refuse**; dummy cannot contribute. Fine. |
| Pole batches `[1,8]`, occupation split `0.5`, mu agreement `1e-12 Ry`, edge `1.5 eta` (`mpa/sigma.py:27-34,307-360`) | Executor resource bound; band occupation; common energy origin; product-window guard width. | Larger pole fit; fractional metal without `OccupationState`; mismatched mu. | Batch and mu **refuse**; `0.5` is insulating-only and metal passes live fractions; `1.5 eta` is the measured product geometry. Fine. |
| Legacy crossing floor 500, `f_max` lattice `0.25 Ry`, gamma cache rounding (`mpa/sigma_windows.py:46,275-292`) | Node resource ceiling; conservative bandwidth bin; pole-width rectangle coordinates in Ry. | Tiny widths alias after 12 decimals; floor can override a lower caller cap. | Upward bandwidth bin is safe; node floor is visible resource policy; gamma alias is **silent THR-007**. |
| Positive-slab exponent 600 (`mpa/sigma_windows.py:352-374`) | Dimensionless log growth; float64 overflow is near 709. | Wide/strongly damped windows consume residue headroom. | Explicit **refusal** with measured exponent; 109 log units of headroom. Fine. |
| Lattice/census `25/4096`, safety `0.8`, factor cap `30`, pair cap `200` then support-derived (`mpa/delivered_windows.py:50-69,823,2026-2220`) | Validation resolution; conservative error share; exponential growth; executor work. | Wide/many-window SC plans can exceed caps or be under-resolved. | Full refined validation, factor/pair gates **refuse**; no pairwise escape exists. Na 25/50/100 lattice evidence found zero node changes. Fine. |
| Noise `kappa_p99*6e-8 <= .05 target` (`delivered_windows.py:52-54,1668,2536`; `minimax/roq_fit.py:81-82`) | Runtime complex128 noise model and allowed target share, dimensionless. | High cancellation makes the accuracy target meaningless. | Measured on each plan and **refuses**. Binding owner acceptance gate; fine. |
| Crossing `eps_q=1e-3`, fit `12/3/.2/.9` (`delivered_windows.py:71-88,2059-2063`) | Shipped HGL tail tier and fixed offline Lawson/stall/conditioning/tail fractions. | Different eps_q would be a different target/table; hard fits can stall. | eps_q mismatch **refuses**; final delivered-error/noise validation catches fitter failure. Provenance comments present; fine. |
| Tightening `0.9`, floor `.05`, three retries (`delivered_windows.py:57-62,2280-2410`) | Multiplicative allocation update, dimensionless. | Severe shortfall may exhaust retries/ceiling. | No silent best effort: final budget check **refuses**. Fine resource convergence constants. |
| Fingerprints (4096 samples/seven digits) and cache version 6 (`delivered_windows.py:69,360-390`) | Plan-cache eligibility only. | Collision or changed support. | A non-identical fingerprint triggers live fit revalidation before reuse; invalid hit becomes miss. Fine. |
| Scissor WLS `denom<1e-30`; 0/1-sample fallbacks; fractional `1e-8` (`scissor.py:69-122,374-378`) | eV weighted variance and dimensionless occupation classification. | Constant/tiny energy span makes slope unidentified; small sample cannot identify slope. | Explicit identity/rigid-shift models encode lack of information; fractional threshold is far from float noise and MP1 partial mass. Fine. |
| Multiplet tolerance and outward promotion (`band_partition.py:117-203`) | Shared `DEGENERACY_TOL_RY` (1 meV policy), energy gap. | Degenerate boundary or protected state outside Sigma grid. | Boundary is promoted and a protected/grid leak is loud, but only a warning. Not silent; accepted owner policy. |
| SC live partition and convergence cutoff (`sc_iteration.py:141-217,2470-2487,3110-3122,3640-4027`) | Current absolute Sigma window around live mu; max band movement in eV. | Mu moved 1.352 eV on Na; frozen masks tested the wrong bands. | **Fixed here THR-004:** apply and verdict consume the same map-local partition; zero test set refuses. |
| Fixed-Sigma EVSC partition (`sc_iteration.py:1381-1392`) | Same live window/mask, but for EQP2 eigenvalue iteration. | Metallic mu/eigenvalues move while the DFT mask is frozen. | **Silent open THR-009.** Not changed without a separate real EQP2 measurement. |
| Eigh memory 1%, k-star `1e-6`, link singular value `.5`, missing outer band (`sc_iteration.py:975,2027,4198-4210,4254-4290`) | Resource route; relative star spread; overlap fidelity; edge observability. | Large systems route native; second material may not support `.5`; truncated ladder cannot inspect its edge. | Memory/star/link paths announce or refuse. `.5` is explicitly under-calibrated. Missing outer band silently returns: **THR-010**. |
| Occupation support `.995`; MP1 clamp `1e-8` capped `1e-3` (`efermi.py:169,244-270`) | Dimensionless occupation mass; MP1 extrema set safe maximum; width sets energy scale. | Very broad smearing enlarges support; clamp near lobe extremum would alter physics. | Values validate/refuse; default is derived as `~4.3` scaled widths and included inside fixed-N solve. Fine. |
| Exact fill `1e-9`, bracket 16 widths, 64 bisections, fixed-N `1e-10 e` (`efermi.py:335-343,650-665,805-844`) | Relative band count; smearing-scaled energy bracket; electron-count residual. | Huge electron counts make relative exact-fill tolerance larger; insufficient bands or degenerate frontier. | Degenerate/partial/inconsistent states **refuse**; 16 widths gives `<1e-27` tail and 64 steps float64 precision. Fine. |
| Full-ladder `E_top+1.0` (`efermi.py:556-583,650-658`) | Supposed Fermi energy in caller units; no physical scale exists without an empty band. | All provided states occupied. | **Silent THR-008:** arbitrary unit-dependent answer. Open; require a frontier band. |
| Service family records (`minimax/{family_axes,targets,records,refusals,__init__}.py`) | Declarative rounding directions, target versions and provenance enums; dimensionless metadata, not physics cutoffs. | Unknown family/axis/version or malformed declaration. | Exact lookup or typed **refusal**; no numerical default. Fine. |
| Service caches (`minimax/cache.py`) | SHA-256/version/backend identity and filesystem availability; affects reuse cost, never the quadrature target. | Old, corrupt, missing or unwritable cache. | Every fallback is announced; corrupt/missing entries re-solve, and the door rechecks achieved error. Fine for correctness; a failure can still violate the planning-time budget loudly. |
| Complex/damped catalog selectors (`minimax/{beta_selector,damped_line_selector}.py`) | Dimensionless range/span, target error, line ratio (published 10), exact sampling fraction and node ceiling; scales come from the named family axes. | Out-of-range span/tier/nodes, changed beta/line ratio, or absent fraction row. | Conservative axes round only in their declared direction; different-function axes and malformed payloads **refuse**. `1e-12` relative equality slack is float comparison scale. Fine. |
| Measure lattice (`minimax/measure_windows.py:26-129`) | 25 count-quantile bins/axis and `1e-12` mass-conservation tolerance; dimensionless compressed support measure. | Very sparse/duplicate support collapses bins; huge dynamic range can stress interpolation. | Empty/negative mass and conservation loss **refuse**; duplicates are exact merged mass, followed by full delivered-lattice validation. Fine. |
| Service ROQ/reciprocal/window fits (`minimax/{frequency_fit,reciprocal_fit,roq_fit,time_node_search,windowed_fit,sector}.py`) | Rank/angle/grid/IRLS/analytic-bound constants; dimensionless fit machinery, calibrated on frozen Na measures. | Wide windows or new systems can exhaust rank or make rung proxy diverge. | Production callers validate on the full delivered lattice plus noise gate and **refuse**; intermediate best-effort objects carry `target_met=False`, and the sector bound refuses. Fine under the current measured gate. |

This census groups repeated literals that implement one threshold; integers used only
for array axes, polynomial coefficients, unit conversion, or loop indexing are not
thresholds.  The source-tree issue register is `KNOWN_LORRAX_ISSUES.md`.

## Changes and evidence

- `c281bed3`: an uncertified minimax rule is servable only when its measured
  error is finite and strictly below the request; focused red twin passes.
- `c281bed3`: `SCOutputs` carries the map-local partition and all three SC
  convergence sites consume it.  The synthetic moved-window red twin changes
  the worst band from the frozen band 0 to live band 1 and correctly refuses
  convergence.
- Focused CPU gate: 32 SC tests pass; the two new red twins pass.  A wider
  service batch exposed three inherited collection/cache reds (stale 31-entry
  catalog count, unresolved `MeasureWindow`, and cross-test lru/cache state),
  while 70 tests passed; none is caused by these changes.
- The repository default gate selected 1100 cells but refused before running
  them because an inherited selected test mutates `HDF5_USE_FILE_LOCKING` at
  collection time.  The supplied campaign CPU environment also cannot collect
  seven selected modules because it lacks `xsdata`/`psp.normalize`.
- P=4 one-shot evidence:
  `runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/codex_threshold_audit_p4_20260831`.
  Commit `c281bed3` completed in 118.327 s wall (89.954 s Sigma), with 6
  logical windows, 115 `(window,tau)` pairs, 146 tau dispatches and zero
  undispatched terms.  `sigma_c_kij_ev` against `control_panes_24b` is finite
  and has **0.195875457964 meV max / 0.008833916923 meV RMS** error, preserving
  the supplied baseline exactly.
- P=4 signed-SC evidence:
  `runs/Na/02_soc48b_qsgw_mpa/60_sc_delivered_20260831/codex_threshold_audit_p4_20260831`.
  The first map moved mu from 1.64676169 eV by **1.35208243 eV**.  Its
  degeneracy-closed map-local partition contained 12 protected/in-range bands,
  and the convergence verdict evaluated the same 12 bands.  A second frozen
  fit correctly refused because its occupation hash described a different
  state; the live follow-up independently reproduced THR-001's collapsed
  `target=1e-18`, `log(R)=29.5592` request.  It was still inside that planner
  after **10:12**, so it was stopped before an unmet rule could contaminate the
  shared home cache.  Thus the static metallic planner does not meet the
  owner's `<5 s` requirement; THR-001 remains landing-blocking.
