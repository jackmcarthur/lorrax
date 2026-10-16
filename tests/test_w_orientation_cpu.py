"""The per-branch W orientation rule and the identity little-group predicate (CPU, no device).

``gw.mpa.sigma._w_orientation`` decides what one Sigma branch forms; a branch whose load
rows read both orientations, or any panel whose realization is not the identity pair, must
keep the pair (the transpose exchange, exactly as before).  ``little_group_projection_is_identity``
is the host rule that lets the realization and the orientation choice skip the partner.
"""
import numpy as np
import pytest


def test_orientation_rule_truth_table():
    from gw.mpa.sigma import _w_orientation
    # identity realization: one orientation exactly when the branch reads one
    assert _w_orientation(True, True, False) == "plus"
    assert _w_orientation(True, False, True) == "transpose"
    # a branch whose tables read BOTH orientations within one window: the pair
    assert _w_orientation(True, True, True) == "pair"
    # defensive: no row at all reads anything -> the pair, never an empty choice
    assert _w_orientation(True, False, False) == "pair"
    # a little-group (non-identity) realization mixes W with its partner: always the pair
    for reads in ((True, False), (False, True), (True, True)):
        assert _w_orientation(False, *reads) == "pair"


def test_orientation_needs_weights_on_load():
    """Off the device route the synthesis GEMM forms the weighted copy b_X diag(d) itself, which
    a single-orientation kernel holds at its peak (A100 CUDA native arm, Fe V2 P=4: +11.5 MB in
    the window executable): such a branch keeps the pair whatever it reads."""
    from gw.mpa.sigma import _w_orientation
    for reads in ((True, False), (False, True), (True, True), (False, False)):
        for identity in (True, False):
            assert _w_orientation(identity, *reads, weights_on_load=False) == "pair"
    assert _w_orientation(True, True, False, weights_on_load=True) == "plus"


def test_only_the_device_route_loads_weights():
    from distrib_la import active_weights_on_load
    from distrib_la._active_local import ACTIVE_LOCAL_ROUTE
    assert active_weights_on_load("ROCM") is True
    assert not any(active_weights_on_load(p) for p in ("cpu", "tpu"))
    assert active_weights_on_load("CUDA") is (ACTIVE_LOCAL_ROUTE["CUDA"] == "device")


def _tables(nsp, m, rotations_spatial):
    rot = np.concatenate([rotations_spatial, -rotations_spatial]).astype(np.int64)
    perm = np.tile(np.arange(m, dtype=np.int32), (2 * nsp, 1))
    wraps = np.zeros((2 * nsp, m, 3), np.int32)
    return rot, perm, wraps


@pytest.mark.parametrize("kgrid", [(4, 4, 4), (2, 3, 1)])
def test_identity_predicate(kgrid):
    from symmetry_maps import little_group_projection_is_identity
    eye, inv = np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)
    rot, perm, wraps = _tables(2, 5, np.stack([eye, inv]))
    q = np.arange(int(np.prod(kgrid)))
    common = dict(kgrid=kgrid, sym_mats_k=rot, sym_perm=perm, L_table=wraps, n_sym_spatial=2)
    # identity alone authorized: every stabilizer is {E}
    assert little_group_projection_is_identity(q, active_symmetry_rows=np.array([0]), **common)
    # inversion authorized: Gamma (q = 0) is stabilized by {E, I} -> not the identity
    assert not little_group_projection_is_identity(
        q, active_symmetry_rows=np.array([0, 1]), **common)
    # rows whose stabilizer under inversion is trivial (q != -q on the mesh) stay the identity
    coords = np.stack(np.unravel_index(q, kgrid), axis=1)
    generic = q[np.any((2 * coords) % np.asarray(kgrid) != 0, axis=1)]
    if generic.size:
        assert little_group_projection_is_identity(
            generic, active_symmetry_rows=np.array([0, 1]), **common)
    # no authenticated identity row (a permuted 'identity'): never treated as the identity
    perm_bad = perm.copy(); perm_bad[0] = np.roll(perm_bad[0], 1)
    assert not little_group_projection_is_identity(
        q, active_symmetry_rows=np.array([0]), kgrid=kgrid, sym_mats_k=rot,
        sym_perm=perm_bad, L_table=wraps, n_sym_spatial=2)
