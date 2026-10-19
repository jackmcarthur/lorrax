"""The staged sector prices against the measured CrI3 24x24 P64 peaks (run 09, main 8c1bc7153, 72 GB).

Leg: runs/CrI3/512_fm_24x24_750b_20261002/09_simple_shape_route_20261008/leg01, map 1, rank 0
(`sections_rank0_map1.txt`). Shapes (61 parents, 8x8): TT side 25856, 2c 18432, 5184 rows, infinity
640, stages of 4; CC side 19264, 2c 12288, 3328 rows, infinity 416, stages of 16; CT joint side 17408.
The upstream is the ledger's 7.44 GB plus a (61, 256, 4, 76601) complex128 stack, 1.196 GB per rank
(the current channel's N_G), live through the sector stage and absent from the ledger
(KNOWN_LORRAX_ISSUES): with it, every staged row bounds its measured section peak.
"""
GB = 1e9
RANKS, NQ = 64, 61
UPSTREAM = 7.44 * GB + 1.196 * GB
PEAKS = {  # section: measured GB
    "TT.stages": 56.61, "TT.H_vv": 53.59, "TT.Schur": 39.46, "TT.reduced": 69.33,
    "CC.stages": 61.98, "CC.H_vv": 48.64, "CC.Schur": 38.03, "CC.reduced": 51.32,
    "CT.metric": 63.46, "CT.Ritz": 63.46,
}
SHAPES = {"TT": (25856, 9216, 5184, 640, 4), "CC": (19264, 6144, 3328, 416, 16)}
per_rank = lambda elements: -(-16 * int(elements) * NQ // RANKS)


def _panels(side, rows, infinity):
    return per_rank(2 * rows * (side - 2 * infinity) + 5 * rows * infinity), per_rank(rows * (side - 2 * infinity))


def test_staged_rows_bound_the_measured_peaks():
    from types import SimpleNamespace
    from gw.shared_pole_capacity import staged_cross_bytes, staged_sector_bytes
    from gw.shared_pole_execution import face_reduction_bytes
    mesh = SimpleNamespace(shape={'x': 8, 'y': 8}, size=RANKS)
    eigh = lambda m: 8 * m * m * 16            # route (c), one whole matrix per rank
    beside, priced = UPSTREAM, {}
    for name in ("TT", "CC"):
        side, carrier, rows, infinity, width = SHAPES[name]
        held, dw = _panels(side, rows, infinity)
        stacks, boundaries = staged_sector_bytes(parents=NQ, ranks=RANKS, side=side, carrier=carrier, packed=rows)
        program = face_reduction_bytes(mesh, width, rows=rows, side=side, carrier=carrier, retain_span=True)
        priced[f"{name}.stages"] = beside + stacks + held + dw + program
        for label, (m, bound) in zip(("H_vv", "Schur", "reduced"), boundaries):
            priced[f"{name}.{label}"] = beside + bound + held + eigh(m)
        beside += held + per_rank(2 * rows * side + side * 2 * carrier)     # its held outputs, for CC
    kept = UPSTREAM + sum(per_rank(2 * rows * side) for side, _, rows, _, _ in SHAPES.values())
    _, boundaries = staged_cross_bytes(parents=NQ, ranks=RANKS, side=17408, rows=(3328, 5184))
    for label, (m, bound) in zip(("metric", "Ritz"), boundaries):
        priced[f"CT.{label}"] = kept + bound + eigh(m)
    for section, peak in PEAKS.items():
        assert priced[section] >= peak * GB, (section, priced[section] / GB, peak)


def test_stage_rows_carry_the_eigenvector_stacks():
    """The keep stage reads U of H'_vv beside the members and the restricted pencil, the paired
    stage U of the Schur complement, the output stage the Ritz rotation (review finding 5)."""
    from gw.shared_pole_capacity import staged_cross_bytes, staged_sector_bytes
    side, carrier, rows, _, _ = SHAPES["TT"]
    stacks, _ = staged_sector_bytes(parents=NQ, ranks=RANKS, side=side, carrier=carrier, packed=rows)
    hvv, two = side // 2, 2 * carrier
    members = 6 * hvv * hvv + 2 * rows * hvv
    restricted = 2 * two * two + 2 * carrier * carrier + hvv * carrier + rows * two
    assert stacks == per_rank(members + hvv * hvv + restricted)              # 2.55 GB/rank of U at P64
    assert per_rank(hvv * hvv) > 2.5 * GB
    ct, _ = staged_cross_bytes(parents=NQ, ranks=RANKS, side=17408, rows=(3328, 5184))
    assert ct == per_rank(2 * 17408 ** 2 + 8512 * 17408 + 17408 ** 2 + 2 * 17408 ** 2)
    assert per_rank(17408 ** 2) > 4.5 * GB                                    # CT's U
