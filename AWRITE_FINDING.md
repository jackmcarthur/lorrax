# AWRITE finding — reference-deck gates pass

Measured source: `20f241b3` on `lane/sp-awrite-2026-09-10`, incorporating requested AFIX base `d6dd0507`. Quadrature budget is unchanged relative to that base. Source is pushed for review, not merged to main.

## Output and format decisions

`write_poles` extends the existing v1 shared-pole store through its existing writer. The standalone file exposes `b` as an HDF5 hard link to `factor`, so there is one payload and existing readers remain compatible. `b[q,mu,spin,j]` is the factorised plasmon-pole residue, B=b b†, in Wc=b(s−Λ)⁻¹b†; spin is singleton for this supported scalar representation. `poles2_ry2[q,j]` stores Λ, the Ω² analogue, and `K[q]` selects active columns. Neither the causal τ weight nor 1/(2Ω) is included in b.

`write_w` writes the fixed physical Wc(q,z_i) sample bank. It is not W+(τ) and does not add bare V. The existing authenticated bank writer/reader also owns dWc/ds, M1 and M3, so these companions are retained rather than introducing a second serialization path. No new screening samples or pole fits are computed. Sample coordinates and roles remain in the canonical bank header.

Both options default false. They are accepted only for shared-pole MPA and write `tmp/mpa/<map>_poles.h5` and `tmp/mpa/<map>_w.h5`, where map is oneshot or the existing SC label. Existing output paths refuse overwrite. Internal model/bank stores remain necessary. Restart export authenticates the model; write_w requires its matching sibling bank and refuses a model-only restart instead of silently resampling. Full restart/SC driver output was not measured in this lane.

As with zeta_q.h5, SlabIO creates the striped inode and owns payload I/O. The existing copy_mf_header owner appends the source WFN mf_header verbatim. Centroid coordinates share write_centroid_coordinates with the zeta header writer. Object-specific descriptions live in poles_header/w_header. This is the same artifact convention for a different object, with no duplicate factor writer and no fake zeta schema.

## Storage and measured output cost

| Reference | n / parents / samples | Wc-only bytes | exported bank logical bytes | compact pole bytes | actual pole / bank file bytes | combined output band s |
|---|---|---:|---:|---:|---|---:|
| si_p4 (58166522.3) | 368 / 8 / 41 | 710705152 | 1456078848 | 94807744 | 178660510 / 1469737888 | 46.027918 |
| na_p16 (58166522.7) | 896 / 29 / 27 | 10057678848 | 20860370944 | 660986096 | 1315176119 / 20868709551 | 358.290123 |

Bank formula: 16*Nq*(2*S+2)*n² bytes. Compact poles: Nq*(16*n*Kmax+8*Kmax+8). The existing model writer stages then finalizes; its physical file can retain freed staging space. Header metadata is additional. Na one logical face is 12,845,056 bytes; 14,745,600 bytes describes the 960² padded face. The all-parent Na bank is large enough that q selection would be useful for inspection, but no third dial is introduced: write_w explicitly writes all parents. Runtime is measured with both options enabled together; isolated per-option wall times are not claimed.

File sizes/hashes and source byte comparisons: Si on/export_gate.json; Na Run350/10_na_export_postprocess/export_gate.json. Striping receipts are each on/stripe_receipt.json. Both outputs use the SlabIO stripe policy; source header and centroid equality are additionally exercised by the P4 format fixture.

## Gates and memory

All eight arms require exact all-parent b/poles/K, identity and basis metadata; physical decks differ only in the two output keys. Every fitted quadrature rule must certify. Production Sigma is compared within 2 meV, not required bit-identical. Off additionally requires every-rank JAX peak non-increase. The on arms compare every exported payload byte and inspect optimized HLO on every rank.

| Geometry / arm | job.step | max production Sigma change meV | rank peak bytes | model |
|---|---|---:|---|---|
| si_p4/control | 58166522.0 | 0.089629223 | 4166923192, 4406953808 | exact |
| si_p4/control_repeat | 58166522.1 | 0.000000000 | 4166923192, 4406953808 | exact |
| si_p4/off | 58166522.2 | 0.000000000 | 4166923192, 4406953808 | exact |
| si_p4/on | 58166522.3 | 0.000000000 | 4166923192, 4406953808 | exact |
| na_p16/control | 58166522.4 | 0.000000000 | 8517691768 | exact |
| na_p16/control_repeat | 58166522.5 | 0.037879434 | 8517691768 | exact |
| na_p16/off | 58166522.6 | 0.015627060 | 8517691768 | exact |
| na_p16/on | 58166522.7 | 0.037879434 | 8517691768 | exact |

The first control compares to the stored reference; repeat/off/on compare to that fresh control. Peaks are device allocator high-water receipts, not a claim about all host RSS. Off and on peaks are unchanged on every rank on both reference decks. Raw rank rows remain in each gate.json.

P4 contract gate 58166522.2: 84 configuration/header tests pass on each rank; exact planted exports pass; all 10 existing model/bank contracts pass. Receipts: Run350/08_afix_format_p4/config_rank0.xml through config_rank3.xml, complete_rank*.json, outputs/receipt.json and store/receipt.json. This is focused validation, not a claim that the repository's full historical suite passed.

## Layout proof

The copy loop reads one parent at P(None,x,None,y) for factors. Both matrix axes remain tiled through existing pack/write/unpack owners. W reads one parent, sample and field at P(None,None,x,y), moments at P(None,x,y). No full W face is gathered. SlabIO drains a field before the next is allocated; the ledger retains input buffers while the writer reserves its temporaries. The existing authenticator uses bounded parent batches and at most ceil(K/Py) columns before I/O: local factor panel n*ceil(K/Py)/Px, not n*K/Px. It exchanges row hashes and small pole metadata, not factor payloads.

The exact existing conversion kernels are lowered at measured factor and W shapes. Each HLO gate refuses all-gather and bounds every local complex array by the staged local panel size; X and Y redistribution occur separately through PackedCentroidBasis. Si HLO job.step is 58166522.3; Na HLO and file comparison job.step is 58166522.8. Per-rank receipts contain argument/output/temporary bytes and collective lines; raw optimized HLO is retained beside them.

| Geometry / conversion | largest local complex elements | allowed local panel elements |
|---|---:|---:|
| si_p4/factor_pack | 193152 | 193152 |
| si_p4/factor_unpack | 193152 | 193152 |
| si_p4/W_pack | 36864 | 36864 |
| si_p4/W_unpack | 36864 | 36864 |
| na_p16/factor_pack | 96000 | 96000 |
| na_p16/factor_unpack | 96000 | 96000 |
| na_p16/W_pack | 57600 | 57600 |
| na_p16/W_unpack | 57600 | 57600 |

## Band-level timing

Rank-zero GPU/NCCL interval unions and backend-compiler counters use the existing ANEST owners. Capture spans the full driver; the inherited outer label constructor means driver.main in this adapter and is not quoted as constructor cost. Only named spole bands are compared. Driver wall is context only; the published same-source spread is 12.555450169 s. Published band spreads are backend 0.340940 s, Gram 0.336710 s, direction 0.246062 s. These and the two-control observed ranges are not confidence bounds; no speedup claim is made.

| Geometry / band | control s | repeat s | off s | own control range s | off−control s |
|---|---:|---:|---:|---:|---:|
| si_p4/spole.bank | 14.579583 | 15.869198 | 12.977206 | 1.289616 | -1.602377 |
| si_p4/spole.moments | 4.511435 | 4.663675 | 4.713350 | 0.152240 | +0.201915 |
| si_p4/spole.gram_reduction | 22.102413 | 22.130378 | 22.134739 | 0.027966 | +0.032327 |
| si_p4/spole.direction_selection | 6.043268 | 6.181118 | 6.131242 | 0.137850 | +0.087973 |
| si_p4/spole.writer | 2.990147 | 3.145464 | 3.110204 | 0.155317 | +0.120058 |
| si_p4/spole.screening_finalize | 0.310676 | 0.303531 | 0.300898 | 0.007145 | -0.009778 |
| na_p16/spole.bank | 122.799671 | 122.369339 | 122.869370 | 0.430332 | +0.069699 |
| na_p16/spole.moments | 19.774378 | 20.422743 | 19.889046 | 0.648366 | +0.114668 |
| na_p16/spole.gram_reduction | 76.411910 | 76.429788 | 76.376378 | 0.017878 | -0.035532 |
| na_p16/spole.direction_selection | 13.786805 | 14.367492 | 13.888567 | 0.580687 | +0.101762 |
| na_p16/spole.writer | 7.673485 | 7.739598 | 7.674337 | 0.066113 | +0.000851 |
| na_p16/spole.screening_finalize | 1.046532 | 1.038988 | 1.040961 | 0.007544 | -0.005571 |

si_p4 output-on band: 46.027918 s wall, 1.690374 s backend compilation, 1.405519 s GPU union, 1.299949 s collective union. The uncovered interval is not labeled CPU idle or filesystem time. Driver walls control/repeat/off/on: 286.397630/290.808146/286.350289/335.666217 s.

na_p16 output-on band: 358.290123 s wall, 1.880196 s backend compilation, 37.837223 s GPU union, 37.521389 s collective union. The uncovered interval is not labeled CPU idle or filesystem time. Driver walls control/repeat/off/on: 560.596402/571.902693/542.803549/924.620406 s.

## Model census, passivity and exact-input CD reuse

Models are real-pole here: damping fraction 0. Every model passes the V-whitened passivity receipt at eta=0.25 eV. J equals active K for these exports. The complete parent census is in assessment.json and construction_receipt.json; no single-parent result is advertised as all-parent.

si_p4 (58166522.3): J/K range 1068–2010; condition range 94123581.4–99997882; passivity min -4.49778354e-17, max 0.864070905. Compact and actual storage are above.

Reused score source and its hash are embedded in each gate.json; CD was not recomputed.

si_p4 reference job 58128243.22: All64 internal q; eight external representatives, six frontier bands1..6, full6x6 block at each bra-state energy plus41 offsets -5..5eV; eta0.25eV, step occupations, 34 intermediate bands; direct CD convergence controls, not QP/band convergence

| Metric | Value |
|---|---:|
| full rms mev | 0.127414404 |
| diagonal rms mev | 0.288330821 |
| offdiagonal rms mev | 0.0534264355 |
| own rms mev | 0.121316548 |
| own max mev | 0.35007109 |
| sampled max mev | 5.94612129 |
| constant rms mev | 0.107011864 |
| shape rms mev | 0.267737041 |

na_p16 (58166522.7): J/K range 1061–1589; condition range 86701317.2–99906154.4; passivity min -8.57005437e-17, max 0.966469642. Compact and actual storage are above.

Reused score source and its hash are embedded in each gate.json; CD was not recomputed.

na_p16 reference job 58128243.23: 29parents,29external representatives,all512internalq,frontier3x3x41 global energies -5..5eV relativeFD9mu,eta0.25eV,fixedG,86logical/88carrierbands,nooffset

| Metric | Value |
|---|---:|
| full rms mev | 0.163883184 |
| diagonal rms mev | 0.282992393 |
| offdiagonal rms mev | 0.015626898 |
| sampled max mev | 2.39989056 |
| own energy count | 49 |
| own energy rms mev | 0.403174831 |
| own energy max mev | 1.49845552 |
| constant rms mev | 0.188026484 |
| remainder rms mev | 0.21149642 |
| constant squared fraction | 0.441457122 |

Si CD96 is the existing exact-pole/direct-CD comparison on its six-band block. Na CD48 is the stored correlated translation, not new direct-W integration or a new absolute-CD certificate. No fitted offset is subtracted; constant/remainder diagnostics are retained. No new fit-quality claim, damping model, frozen-pole stability trick, quadrature policy, or arbitrary-system performance guarantee is introduced.

## Evidence and review boundary

Current evidence root: `F/350_awrite_20260910/07_afix_base`; assessment: `F/350_awrite_20260910/09_afix_assessment/assessment.json`. F denotes `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox`. Each arm carries source binding, launch command, allocation receipt, exact-input gate, trace receipt and all-rank driver data. Native pin/cuda_async@0.85 and one-rank-per-GPU geometry are fixed across arms.

Old e6ac7915-base runs remain historical only. The first old profiling attempt stopped capture mid-driver and stalled in Nsight injection; its stack evidence is retained under Run350/02_si_p4/control. Fresh current-base captures run through the full driver and all drivers finish. The Na output-on postprocessing then hit a JAX rank-arrival timeout because rank zero was still comparing files; production model/Sigma/trace receipts had already passed. Run350/10_na_export_postprocess verifies those immutable files and layouts in a fresh P16 leg, with collective checks before serial file validation. Its gate receipts name the separate job.step; no driver rerun or unmeasured export claim is substituted. Completed runs were not overwritten. Source is offered for the requested second review; no main merge or pool release is performed.

Code owners: [export writer](/pscratch/sd/j/jackm/wt_sp_awrite/src/file_io/shared_pole_store.py), [driver stage](/pscratch/sd/j/jackm/wt_sp_awrite/src/gw/shared_pole_screening.py), [shared centroid header](/pscratch/sd/j/jackm/wt_sp_awrite/src/file_io/isdf_header.py), [deck keys](/pscratch/sd/j/jackm/wt_sp_awrite/src/gw/gw_config.py), [input reference](/pscratch/sd/j/jackm/wt_sp_awrite/docs/input_reference.md). The Na output-enabled driver took 924.62 s, above the campaign 12-minute target; these optional diagnostic exports are therefore a material extra cost and remain off by default. No claim is made that enabling them preserves the production timing target.

Ledger: claim 2162 records the current-base verdict; claim 2152 remains historical old-base fixture evidence.
