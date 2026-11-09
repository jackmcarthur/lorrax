"""kmeans_cli --pool: k-means by default; 'random' is reported in the settings."""


def test_pool_flag_default_and_settings_row():
    from centroid.kmeans_cli import build_parser
    from centroid.production_output import kmeans_settings_rows
    parser = build_parser()
    default = parser.parse_args(["10"])
    assert default.pool == "kmeans"
    assert not any(row.key == "pool" for row in
                   kmeans_settings_rows(default, source=lambda *f: "default"))
    drawn = parser.parse_args(["10", "--pool", "random"])
    rows = kmeans_settings_rows(drawn, source=lambda *f: "cli")
    assert [(row.key, row.value, row.source) for row in rows if row.key == "pool"] == [
        ("pool", "random", "cli")]


def test_random_pool_draws_whole_orbits_in_budget():
    import numpy as np
    from centroid.kmeans_cli import _weighted_random_pool
    from centroid.kmeans_isdf import grid_orbit_canonical
    side, budget = 6, 101
    weight = np.random.default_rng(0).random((side,) * 3)
    weight.reshape(-1)[:50] = 0.0
    syms = np.stack((np.eye(3, dtype=np.int32), -np.eye(3, dtype=np.int32)))
    tau = np.zeros((2, 3))
    frac = _weighted_random_pool(weight, (side,) * 3, syms, tau, True, budget, 3)
    flat = np.ravel_multi_index(np.rint(frac * side).astype(int).T, (side,) * 3)
    canonical = grid_orbit_canonical((side,) * 3, syms, tau)
    assert np.array_equal(canonical[flat], flat)          # one draw per orbit
    sizes = np.bincount(canonical, minlength=canonical.size)[flat]
    assert budget - 2 < sizes.sum() <= budget
    mass = np.bincount(canonical, weights=weight.reshape(-1),
                       minlength=canonical.size)[flat]
    assert np.all(mass > 0)
    again = _weighted_random_pool(weight, (side,) * 3, syms, tau, True, budget, 3)
    assert np.array_equal(again, frac)
