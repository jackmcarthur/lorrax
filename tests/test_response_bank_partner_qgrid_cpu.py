"""The ordered bank's partner rows belong to its bosonic output grid."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("qgrid", [(4, 4, 1), (8, 8, 1), None])
def test_ordered_bank_partners_use_bosonic_grid_before_stream_admission(monkeypatch, qgrid):
    import gw.response_bank as owner
    import symmetry_maps

    fine = (8, 8, 1)
    grid = fine if qgrid is None else qgrid
    qids = np.arange(np.prod(grid), dtype=np.int64)
    meta = SimpleNamespace(nkx=8, nky=8, nkz=1, mu_basis=SimpleNamespace(n_packed=4))
    if qgrid is not None:
        meta.qgrid = qgrid
    header = dict(ordered=True, line_panels=dict(sample_span=[0, 1]))
    receipt = {}
    monkeypatch.setattr(owner, "_bank_context", lambda *args: (header, qids, {}))
    monkeypatch.setattr(owner, "authenticate_sample_plan", lambda *args: None)
    monkeypatch.setattr(owner, "bank_points", lambda *args: np.array([1.+1.j]))
    monkeypatch.setattr(owner, "_receipt", lambda *args: receipt)
    seen = []
    original = symmetry_maps.q_negation_index

    def negate(request):
        negative = np.asarray(original(request))
        seen.append((tuple(request), negative[qids]))
        return negative

    monkeypatch.setattr(symmetry_maps, "q_negation_index", negate)

    class ReachedExecutionPlanning(Exception):
        pass

    def execution(*args, **kwargs):
        # Stop before any allocation/compile: the actual producer has already
        # constructed its partner table and stream-domain receipt.
        assert receipt["minus_q_partner"]["full_q_rows"] == len(qids)
        raise ReachedExecutionPlanning

    monkeypatch.setattr(owner, "_bank_execution", execution)
    with pytest.raises(ReachedExecutionPlanning):
        owner.produce_sample_bank(
            SimpleNamespace(green_parent=None), meta,
            SimpleNamespace(debug=SimpleNamespace(sigma_freq_debug_output=False)),
            sym=SimpleNamespace(trs_allowed=False), mesh_xy=None,
            bank_io=dict(coulomb="original", identity="current"), sample_plan={})
    assert len(seen) == 1 and seen[0][0] == grid
    expected = np.ravel_multi_index(
        tuple((-np.indices(grid).reshape(3, -1)) % np.asarray(grid)[:, None]), grid)
    np.testing.assert_array_equal(seen[0][1], expected)
    assert np.all((seen[0][1] >= 0) & (seen[0][1] < len(qids)))
    np.testing.assert_array_equal(seen[0][1][seen[0][1]], qids)
