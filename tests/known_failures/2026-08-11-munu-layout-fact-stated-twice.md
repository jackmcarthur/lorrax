# 2026-08-11 — the (μ, ν) restart layout fact is stated twice, and the second statement cannot adopt the first

**Status:** owner row, not a failure. Nothing is wrong today. Filed because a
cleanup lane looked for the dedupe, found a real contract mismatch behind it,
and the mismatch should be on the record rather than rediscovered.

## What was looked for

The `cleanup/bse-loading-io-2026-08-11` lane was sent to hunt for routines in
`src/bse/bse_loading.py` that a service already owns — above all the
star/unfold/orbit/wedge logic belonging to `symmetry_maps`. That hunt came back
empty, and correctly so: `is_q_wedge` is a wrapper over
`symmetry_maps.dataset_q_storage`, `restart_munu_full_bz` is a wrapper over
`symmetry_maps.read_tensor`, the unfold itself is
`symmetry_maps.unfold_isdf_operator`, and there is no local Coulomb arithmetic
anywhere in the module. Three source-level gates already ratchet that state
(`test_restart_qirr_consumers::test_the_probe_has_exactly_two_named_callers`,
`test_crossfile_requests::test_bse_io_guard_comes_from_the_service`,
`test_restart_qirr_consumers::test_the_full_file_reader_goes_through_the_seam`).

## What is genuinely stated twice

The three on-disk `(…, μ, ν)` layouts — 3-D flat-q, 6-D transitional, 8-D
legacy — are described in two places that cannot share:

* `bse.bse_loading._MunuSlabPlan` (and, for the serial path,
  `_resolve_munu_reader`), and
* `file_io.tagged_arrays._munu_slab_request`, whose own docstring already names
  the BSE statement and sets the trigger for consolidating: *"If a third
  consumer appears, the layout fact should move here (L3) rather than be stated
  a third time."*

`file_io.tagged_arrays.read_munu_tensor_from_h5` is the closest thing to a
drop-in service call, and its docstring names `bse_io` explicitly as a consumer
that re-derives the same read for itself.

## Why the lane did not adopt it — three contract mismatches

1. **Different output layout.** `read_munu_tensor_from_h5` returns flat-q
   `P(None,'x','y')` after `_collapse_leading`. The BSE consumer needs μ-major
   3-D-k `(μ, ν, nkx, nky, nkz)` at `P('x','y',None,None,None)`, which is what
   `_slabio_read_munu` produces with a pinned local transpose.
2. **No single-q selection.** `_MunuSlabPlan.request(q_index=...)` exists for
   the `V_q0` route, which reads exactly one q rather than all `nq`.
   `_munu_slab_request` never selects a q and therefore never takes a kgrid;
   routing `V_q0` through it would read `nq`× the bytes it uses.
3. **Opposite wedge policy, and this is the load-bearing one.** The GW-side
   reader ALWAYS UNFOLDS (owner ruling 2026-08-08 ~13:20). `_MunuSlabPlan`
   deliberately REFUSES a wedge, on cost grounds that have never been priced on
   a real interconnect. Adopting the service call would silently enable the BSE
   wedge read — a behaviour change, and precisely the one that
   `DESIGN_restart_consolidation.md` §4 holds until a Perlmutter timing leg
   exists.

Mismatch 3 means this is not a refactor at all under a behaviour-preserving
mandate; it is the wedge-transport decision wearing a refactor's clothes.

## What would retire this row

The Perlmutter timing leg for the all_to_all unfold against the bytes the wedge
saves (`DESIGN_restart_consolidation.md` §4). If that lands and the BSE starts
unfolding, mismatch 3 disappears and 1–2 become an ordinary adapter, at which
point the layout fact should move to `file_io` per that module's own L3 note.
Until then, two statements is the honest count and both are annotated to point
at each other.
