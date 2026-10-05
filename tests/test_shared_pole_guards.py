"""Shared-pole route and resume guards on toy inputs (CPU only, seconds).

1. The relaxed accuracy tier (no pole budget) sizes the bispinor sector face
   batch with no Ritz carrier instead of crashing on ``int(None)``.
2. A complete bank built against another bare-V digest (another P, older code)
   is not resumable: the map rebuilds instead of refusing.
3. The pole-budget keep cut never splits a degenerate multiplet at its edge:
   the tied member leaves with its partner, so K <= budget and the kept span
   does not depend on the eigenbasis inside the multiplet.
"""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_relaxed_sector_face_batch_has_no_carrier(monkeypatch):
    import file_io  # noqa: F401  (service path bootstrap)
    import gw.shared_pole_execution as ex
    from jax.sharding import Mesh
    from gw.shared_pole_sectors import sector_recipe

    carriers = []

    def reduction_bytes(mesh, width, *, keep_budget, carrier, **shape):
        carriers.append((keep_budget, carrier))
        return 0
    # The face batch search calls program_bytes once at its first width.
    monkeypatch.setattr(ex, "face_batch_width",
                        lambda *a, program_bytes, **k: (1, dict(compiled=program_bytes(1))))
    monkeypatch.setattr(ex, "face_reduction_bytes", reduction_bytes)
    monkeypatch.setattr(ex, "face_cross_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(ex, "line_panel_count", lambda recipe: 1)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    for tier, carried in (("production", True), ("relaxed", False)):
        carriers.clear()
        rows = [dict(packed_extent=n, conservative_pencil_side=4 * n, signed_side_bound=2 * n,
                     line_width=2, infinity_width=2,
                     pole_budget=sector_recipe({"accuracy": tier}, n)["pole_budget"]) for n in (8, 24)]
        ex.sector_batch_width(SimpleNamespace(n_rmu_padded=8), None, {"fit_ids": [0, 1, 2]}, rows,
                              mesh=mesh, ledger=SimpleNamespace(live_stages=()), nq=1)
        assert len(carriers) == 2
        assert all((carrier is not None) == carried for _, carrier in carriers)


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
