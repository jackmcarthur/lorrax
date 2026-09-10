# AWRITE finding — AFIX base updated; fresh production gates pending

The two outputs reuse the existing stores. `write_poles` produces a standalone v1 model through `write_shared_pole_model`; `b` is an HDF5 hard link to `factor`, not a second tensor. Lambda is `poles2_ry2`, and K selects active columns. b is the factorised plasmon-pole residue, B=b b†, in Wc=b(s−Lambda)⁻¹b†. The causal exp[−i(Omega−Eref)tau]/(2Omega) weight is absent. v1 names, normalization text and gate recipe remain compatible.

`write_w` writes the fixed physical frequency bank, not W+(tau). It includes Wc, dWc/ds, M1 and M3 because the canonical bank writer and reader own that authenticated format. This choice doubles the Wc-only disk requirement; it does not add screening solves. Both outputs are standalone, not external links into tmp. They append the verbatim source WFN mf_header using copy_mf_header after SlabIO has created the inode with MPI-IO striping. Centroid coordinate serialization is shared with zeta via write_centroid_coordinates; object-specific metadata lives in poles_header/w_header rather than pretending a b file is a zeta file.

Output paths are tmp/mpa/<map>_poles.h5 and tmp/mpa/<map>_w.h5, where map is oneshot or the SC label. Both flags default false and refuse outside shared-pole MPA. Off performs no output read, allocation, digest, or header operation. Internal model/bank files remain required. Restart exports authenticate the current model; write_w requires its matching sibling bank.h5 and refuses a model-only restart instead of silently resampling. Existing export paths refuse overwrite.

## Reference sizes (computed from stored measured geometry, not timed)

| deck / geometry receipt | n | parents | samples | Wc-only bytes | exported bank bytes | compact poles bytes |
|---|---:|---:|---:|---:|---:|---:|
| Si P4, 58137440.16 | 368 | 8 | 41 | 710705152 | 1456078848 | 94807744 |
| Na P16, 58137440.17 | 896 | 29 | 27 | 10057678848 | 20860370944 | 660986096 |

Source artifacts: F/341_aserv_20260910/10_placement_moments/{si_p4,na_p16}/production/tmp/mpa/oneshot_shared_pole/construction_receipt.json. Bank formula 16*Nq*(2*S+2)*n²; compact poles formula Nq*(16*n*Kmax+8*Kmax+8). Metadata and HDF5 free space are extra. The existing model writer stages before finalization, so its transient file can approach twice compact payload; final physical file length can retain freed staging space. Na one logical W face is 12845056 bytes (12.845 MB); 14.746 MB is the 960² padded face, not 896² logical disk data. The Na export is large enough that selective inspection would benefit from q selection, but no third dial is introduced: this option explicitly writes all parents. Runtime and every-rank on-cost remain unmeasured.

## Layout argument (source; optimized-HLO receipt pending)

Model exports read one canonical parent at P(None,x,None,y), pack with PackedCentroidBasis and invoke the existing writer, which unpacks at the I/O boundary. Both matrix axes stay tiled; poles/counts alone are replicated. W exports read one parent and one sample/field at P(None,None,x,y), or moments at P(None,x,y), then use the existing bank writer. No read_shared_pole_faces one-axis factor carrier is introduced. No payload enters h5py in production; only small metadata does. The existing PackedCentroidBasis owns staged X/Y redistribution; no new all-to-all or FFT kernel is added. The capacity ledger retains export inputs while the writer reserves its own temporaries. SlabIO drains each field before the next allocation. Direct HLO and real-rank peak gates are still required; this source argument alone is not that proof.

## Gates

- 58152292.0, F/350_awrite_20260910/01_format_p4: 77 configuration tests passed on each of four ranks; planted complete bank/pole export matched every payload byte, including odd centroid and padded pole extents, and copied header attributes. receipt: 01_format_p4/outputs/receipt.json. Exploratory dirty source at e6ac7915; final committed-source repeat is pending. No production claim.
- Pending: Si P4 and Na P16 control/repeat/off/on, exact all-parent model inputs, every-rank off peak non-increase, on peak/time cost, certified rules and <=2 meV Sigma, CD48/CD96 exact-input reuse.
- Pending: changed-band activity/compile receipts using existing ANEST instrumentation. Driver wall is context only; published same-source range 12.555450169 s, not a confidence bound. Published band spreads: backend 0.340940 s, Gram 0.336710 s, direction 0.246062 s; no saving claimed.

No source changes are claimed landed. No quadrature policy change belongs to this lane; AFIX owns it. No new model fit or new accuracy claim is introduced by serialization.

## Base update requested 2026-09-10

Current implementation incorporates AFIX d6dd0507 by a history-preserving merge. No quadrature-budget code is changed relative to that base. Earlier Si gates 58152292.2–.5 used e6ac7915 and are retained as historical evidence only. Fresh controls and candidate runs use SP-E1 58166522. The explicit output timing section exposes the existing output work to the campaign NVTX instrumentation; no arithmetic changes.

The one-parent residency statement describes the copy loop. The existing authenticator reads bounded parent batches and at most ceil(K/Py) columns before I/O, with local factor panel n*ceil(K/Py)/Px; it exchanges row hashes, not factors.
