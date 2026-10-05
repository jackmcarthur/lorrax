"""The face batch is sized at the recipe bound in map 0 and at the held sides from map 1 (CPU)."""


def test_sized_sector_sides_take_the_held_sides_after_map_zero():
    from gw.shared_pole_execution import sized_sector_sides
    routes = [dict(sector="CC", conservative_pencil_side=20800, signed_side_bound=12288),
              dict(sector="TT", conservative_pencil_side=32000, signed_side_bound=18432)]
    # Map 0 and a one-shot: the recipe bound, spans at most the signed side bound.
    assert sized_sector_sides(routes, None) == ([20800, 32000], [12288, 18432])
    assert sized_sector_sides(routes, {}) == ([20800, 32000], [12288, 18432])
    # From map 1: the session's pencil sides and CT span widths from map 0.
    held = {"CC_pencil_side": 17472, "TT_pencil_side": 24832, "CC": 6144, "TT": 9216, "_events": []}
    assert sized_sector_sides(routes, held) == ([17472, 24832], [6144, 9216])
    # A held side alone (CT spans not yet recorded) bounds the span by itself.
    assert sized_sector_sides(routes, {"CC_pencil_side": 10000}) == ([10000, 32000], [10000, 18432])
