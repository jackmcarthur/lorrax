"""Shared-pole route and resume guards on toy inputs (CPU only, seconds).

1. A complete bank built against another bare-V digest (another P, older code)
   is not resumable: the map rebuilds instead of refusing.
2. The pole-budget keep cut never splits a degenerate multiplet at its edge:
   the tied member leaves with its partner, so K <= budget and the kept span
   does not depend on the eigenbasis inside the multiplet.
3. A local CT round runs at its held spans whatever its price (warn, never
   refuse: the route was decided from the recipe shapes, and there is no rerun).
"""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_constructor_resume_rebuilds_a_bank_of_another_bare_v(tmp_path, monkeypatch):
    import file_io.shared_pole_store as store
    from gw.shared_pole_screening import _authenticated_constructor_resume

    identity = dict(label="sc_map_0001", recipe_hash="r")
    coulomb = dict(basis="canonical", sha256="aa" * 32)
    for name in ("bank", "moments"):
        (tmp_path / f"{name}_receipt.json").write_text(json.dumps(dict(
            identity=identity, completion=True, bank_complete=True, coulomb_identity=coulomb)))
    (tmp_path / "bank.h5").write_bytes(b"")
    monkeypatch.setattr(store, "validate_shared_pole_bank", lambda *a, **k: {})
    resume = lambda sha: _authenticated_constructor_resume(tmp_path, identity, {}, coulomb_sha256=sha)
    assert resume("aa" * 32)
    # The same V packed for another mesh, or a receipt from older code (a file hash).
    assert not resume("bb" * 32)


def test_budget_cut_never_splits_the_edge_multiplet():
    from gw.shared_pole_reduction import _within_budget

    def kept(gamma, budget):
        mask = np.broadcast_to(np.asarray(_within_budget(jnp.asarray([gamma]), budget)), (1, len(gamma)))
        return np.asarray(gamma)[mask[0]].tolist()
    pair = [0.1, 0.5, 1.0, 1.0 + 1e-12, 2.0, 3.0]          # a pair split only by round-off
    assert kept(pair, 3) == [2.0, 3.0]                      # the edge member leaves with its partner
    assert kept(pair, 4) == [1.0, 1.0 + 1e-12, 2.0, 3.0]    # a cut between multiplets is unchanged
    assert kept(pair, 2) == [2.0, 3.0]
    assert kept(pair, None) == pair
    assert kept([1.0] * 4, 2) == [1.0, 1.0]                  # no edge in one tied run: index cut


def test_local_ct_over_budget_still_runs(monkeypatch):
    import file_io  # noqa: F401  (service path bootstrap)
    import gw.gw_config as cfg
    import gw.shared_pole_capacity as cap
    import gw.shared_pole_execution as ex
    import gw.shared_pole_sectors as sectors

    class Ran(Exception):
        pass

    class Capacity:
        status = "FAIL"

        def __init__(self, *a, **k):
            self.face_room = None

        def preview(self, side, **_):
            return dict(device_budget_status=Capacity.status)

        def plan(self, *a, **k):
            return {}

        def eigenplan(self, side):
            return None

    def reduce(*a, **k):
        raise Ran
    monkeypatch.setattr(cap, "ConstructorCapacity", Capacity)
    monkeypatch.setattr(ex, "is_face", lambda array: False)
    monkeypatch.setattr(cfg, "linalg_resolution", lambda deck: None)
    monkeypatch.setattr(sectors, "cross_span_widths", lambda meta, s: ([2, 2], [2, 2]))
    monkeypatch.setattr(sectors, "cross_round_actions", lambda *a, **k: ())
    monkeypatch.setattr(sectors, "_pack_cross_spans", lambda s, widths, actions, **k: (
        [(None, None, None, None, np.zeros((1, 3, w))) for w in widths], ((), ())))
    monkeypatch.setattr(sectors, "reduce_cross_round", reduce)
    sector = dict(model=(np.zeros((1, 3, 2)),), signed=(), coefficients=np.zeros((1, 3, 2)),
                  infinity=(), states=(), roles=((),), recipe={}, budget=SimpleNamespace(face_room=None))
    meta = SimpleNamespace(shared_pole_capacity=SimpleNamespace(live_stages=()))
    config = SimpleNamespace(backend=SimpleNamespace(linalg="local"))
    panels = dict(Wc=None, dWc_ds=None)
    args = ((sector, sector), (panels, panels), {f"M{i}": None for i in range(4)}, meta, config)
    kwargs = dict(mesh_xy=None, sample_ids=(), line_cross=({}, {}), real=1)
    for status in ("FAIL", "PASS"):
        Capacity.status = status
        try:
            sectors.construct_cross_sector_round(*args, **kwargs)
        except Ran:
            pass
        else:
            raise AssertionError(f"a local CT round priced {status} did not run")
