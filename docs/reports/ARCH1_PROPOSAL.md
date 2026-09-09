# ARCH1 — persist tangential states, retain diagnostic matrices

**Claim.** Move per-sample direction selection and action formation into the bank producer’s consumption boundary, eliminating fitted full-W scratch without replaying the stream. This attacks the measured 20.5 s bank-write envelope and constructor reads, but promises neither their complete removal nor a Dyson/stream speedup.

Design only; no compute or implementation. Source inspected: `b16fe23`, branch `arch/sp-arch1-2026-09-09`. Paths below are source-relative; F/S have the dispatch meanings. Timing evidence: S/reports/shared_pole_push_2026-09-07/perf/report.md, jobs 58124441.0/.2; stream trace 58124680.0, F/325_pbank_20260909/18_bank_profile_final_p16/roofline.json. Timing allocator configuration must travel from those jobs into any comparison; no new timing is claimed.

## Why, and which full matrices

`response_bank.py:660-695` accumulates **all time nodes first**, into raw χ/dχ `[2A,b,μp,μp]`; `:707-716` performs Dyson and writes W/dW. `shared_pole_constructor.py:726-747` then reads an entire fitted-sample span, selects directions, and retains narrow states. These are separate storage and execution lifetimes.

The production receipt, 58124662.5, F/323_pcona_20260909/37_na_p16_local_checks/warm/constructor_receipt.json, resolves **21 line roles + 3 imaginary roles = 23 distinct fit samples**, plus four held samples. The shared zero-real-part sample must serve both roles. The guide’s older 15-sample/fixed-56 Na description is not this deck.

Every distinct fit W must reach the existing full spectral owner: right singular vectors for line roles, eigenvectors of Hermitian −W for imaginary roles (`constructor.py:478-491`, abbreviated hereafter). Infinity needs full M1 for selection, and M3 actions (`:716-724`). Full M1/M3 and four held W/dW pairs are also consumed by the existing Frobenius diagnostics (`:516-535`, `:832-855`). Thus this unchanged-recipe route still forms 27 W matrices per parent and two moments; derivatives need no spectral decomposition. “Never form W” would require another direction algorithm and is out of scope.

## Proposed dataflow

Keep `screen_shared_poles` as the physics outline. Refactor `_direction_states` into the single reusable per-sample owner, invoked by `produce_sample_bank` immediately after bounded Dyson output. For each role save Q, O=WQ, D=W′Q; for a nonreal line s also save W†Q and W′†Q, reusing Q. Preserve role order, multiplets, masks and physical coordinates.

**No ordering replay:** Q_a depends only on W_a. Cross blocks use O_a†Q_b and Q_a†O_b, not W_aQ_b (`constructor.py:102-115`). Select Q_a and form its actions while W_a is live, then release W_a. A global-Q-first schedule would unnecessarily replay the 38.6 s real-time stream and remote Laplace cells; the latter cost another 15.1 s in the brief, before repeated Dyson.

Replace fitted payloads in the scratch schema owned by `file_io/shared_pole_store.py:726-779` with ragged per-parent/per-role Q/O/D datasets and conjugate action datasets, logical widths, role-to-sample links, completion masks and digests. Keep the four held pairs and M1/M3 dense. The actual current filename is `<label>_shared_pole/bank.h5` (`shared_pole_screening.py:166-184`). Canonical disk coordinates remain portable; runtime panels `[b,μp,rp]` use `NamedSharding/P(None,'x','y')`. No host matrix gather. Reconstruction reads panels into the existing pencil owner; infinity selection remains there.

Only bounded producer buffers exist on device. Flush panels before the next sample; do not retain all-q actions beside raw χ. The ledger must admit raw carry + spectral workspace + writer staging against the **previous peak**, including native allocations. This overlap could defeat the design: shrinking the sample batch can force another stream pass. Retain both service-selected local and distributed routes; no new q-replicated dense cache.

Version scratch validation and transaction masks together. Final model/restart schema remains unchanged. Authenticate current Hamiltonian, bands, occupations, coordinates, recipe and gates; changed SC maps regenerate everything. Committed actions reproduce the same current-map pencil, not arbitrary future direction choices. Partial-file replay must refuse stale identities. Existing driver partial-directory deletion may remain (`screening.py:153-156`).

## Bytes and collectives

Let P=PxPy, A samples, H held samples, Lq=sum of distinct role widths (Q stored once), Rq=sum including conjugate states. Excluding metadata:

- Old scratch: B₀=16Nq(2A+2)μ² bytes.
- New scratch: B₁=16[Nq(2H+2)μ²+μΣq(Lq+2Rq)]. Logical I/O share per rank is B/P; write-plus-read traffic is 2B/P, before conversion/repeated diagnostic reads.
- Full face: 16μ²/P; action face: 16μr/P. Raw carry remains 32Abμp²/P. Narrow assembly residency remains approximately 48bμpR/P plus O(bR²/P) pencils and service workspace. Model storage remains ΣqKq(16μ+8).

Na: Nq=29, μ=896, Px=Py=4, A=27, H=4. Full face **802,816 B/rank**; r=56/224 action **50,176/200,704 B/rank**. B₀=20.860 GB including moments; samples alone 20.115 GB. Receipt role widths span 5–91 on the line and 224–225 imaginary. Therefore Lq+2Rq≤103×91+9×225, giving **B₁≤8.464 GB**, including the unchanged 3.725 GB diagnostic payload; this is an algebraic upper bound, not measured I/O. Packed extents and native workspaces must be priced separately. All-parent logical raw carry alone remains 1.257 GB/rank.

No stream collectives disappear: the traced count remains 4Nt=2700 at Nt=675, with the same payload. Selection/action GEMMs move from reader to producer; reduced I/O does not imply fewer GEMM collectives. At `_a_local`, dense gathered panels remain 16bμ²/Px and 16bμ²/Py (3.211 MB each per Na parent), until that operation itself changes.

## Dyson limit, risks, decision

The actual bank uses `response_bank.py:65-73`, not `_a_local`: X=pHχH, E=(I−X)⁻¹. With known Q, Y=solve(I−X,HQ) gives WQ=HXY and W′Q=H solve(I−X,X′Y). Dense factorization remains; χQ alone cannot determine these actions because the inverse couples omitted directions. Adjoint states need adjoint solves. Narrow derivatives could avoid a full derivative output, but cannot remove full W needed for Q. Defer that numerical change; neither all 15.7 s Dyson nor the unattributed 16.6 s NCCL is a justified saving.

**One confirming measurement:** a combined cold/repeated A/B bank-through-constructor replay, Na P16 in both linalg modes, with peak memory, stream-pass count, phase timers and bytes. Budget approximately **120 node-minutes**, an estimate. Require unchanged K, CC†/CΛC†/W(iu), Gram/passivity/moment/held receipts, stale-SC rejection and analytic Σ parity; later landing retains Na CD48/Si CD96 rows. Ill-conditioned Gram cuts amplify changed rounding; phase/gauge changes alone are not failure.

**Worth doing after landing:** substantial scratch reduction is defensible; speed and non-increasing peak remain unmeasured. A completely matrix-free bank, unchanged diagnostics and one stream pass are not jointly established.
