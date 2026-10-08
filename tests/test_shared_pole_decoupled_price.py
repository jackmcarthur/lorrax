"""The staged sector prices at the CrI3 24x24 P64 shapes of leg 08 (main dc98076fe, 72 GB).

Leg: runs/CrI3/512_fm_24x24_750b_20261002/08_sectfast2_sigma_20261007/leg01, map 1, rank 0. Shapes
(61 parents, 8x8): TT side 25856, 2c 18432, 5184 rows, infinity 640; CC side 19264, 2c 12288,
3328 rows, infinity 416; CT joint side 17408. Per eigh section: that leg's ledger row and its
measured peak (GB). Every sector row there sat a constant 1.3 GB under its peak, outside the
sector's own arrays (KNOWN_LORRAX_ISSUES); the staged price must not fall below the old row,
and must carry the eigenvector stacks the old stage rows left out (review finding 5).
"""
GB = 1e9
RANKS, NQ = 64, 61
UPSTREAM = 7.44 * GB
LEG = {  # section: (ledger GB, peak GB)
    "TT.H_vv": (52.27, 53.59), "TT.Schur": (38.14, 39.46), "TT.reduced": (68.00, 69.33),
    "CC.H_vv": (47.33, 48.64), "CC.Schur": (36.72, 38.03), "CC.reduced": (50.00, 51.32),
    "CT.metric": (62.21, 63.51), "CT.Ritz": (62.21, 63.51),
}
SHAPES = {"TT": (25856, 9216, 5184, 640), "CC": (19264, 6144, 3328, 416)}
per_rank = lambda elements: -(-16 * int(elements) * NQ // RANKS)


def _panels(side, rows, infinity):
    return per_rank(2 * rows * (side - 2 * infinity) + 5 * rows * infinity)


def test_staged_rows_do_not_fall_below_the_leg_rows():
    from gw.shared_pole_capacity import staged_cross_bytes, staged_sector_bytes
    eigh = lambda m: 8 * m * m * 16            # route (c), one whole matrix per rank
    beside, priced = UPSTREAM, {}
    for name in ("TT", "CC"):
        side, carrier, rows, infinity = SHAPES[name]
        held = _panels(side, rows, infinity)
        _, boundaries = staged_sector_bytes(parents=NQ, ranks=RANKS, side=side, carrier=carrier, packed=rows)
        for label, (m, bound) in zip(("H_vv", "Schur", "reduced"), boundaries):
            priced[f"{name}.{label}"] = beside + bound + held + eigh(m)
        beside += held + per_rank(2 * rows * side + side * 2 * carrier)     # its held outputs, for CC
    kept = UPSTREAM + sum(per_rank(2 * rows * side) for side, _, rows, _ in SHAPES.values())
    _, boundaries = staged_cross_bytes(parents=NQ, ranks=RANKS, side=17408, rows=(3328, 5184))
    for label, (m, bound) in zip(("metric", "Ritz"), boundaries):
        priced[f"CT.{label}"] = kept + bound + eigh(m)
    for section, (row, peak) in LEG.items():
        assert priced[section] >= row * GB, (section, priced[section] / GB, row)
        assert priced[section] <= 72 * GB, (section, priced[section] / GB)


def test_stage_rows_carry_the_eigenvector_stacks():
    """The keep stage reads U of H'_vv beside the members and the restricted pencil, the paired
    stage U of the Schur complement, the output stage the Ritz rotation (finding 5)."""
    from gw.shared_pole_capacity import staged_cross_bytes, staged_sector_bytes
    side, carrier, rows, _ = SHAPES["TT"]
    stacks, _ = staged_sector_bytes(parents=NQ, ranks=RANKS, side=side, carrier=carrier, packed=rows)
    hvv, two = side // 2, 2 * carrier
    members = 6 * hvv * hvv + 2 * rows * hvv
    restricted = 2 * two * two + 2 * carrier * carrier + hvv * carrier + rows * two
    assert stacks == per_rank(members + hvv * hvv + restricted)              # 2.55 GB/rank of U at P64
    assert per_rank(hvv * hvv) > 2.5 * GB
    ct, _ = staged_cross_bytes(parents=NQ, ranks=RANKS, side=17408, rows=(3328, 5184))
    assert ct == per_rank(2 * 17408 ** 2 + 8512 * 17408 + 17408 ** 2 + 2 * 17408 ** 2)
    assert per_rank(17408 ** 2) > 4.5 * GB                                    # CT's U
