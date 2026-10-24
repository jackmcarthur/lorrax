"""The internal screened head has independent placement and restart identity."""
from dataclasses import dataclass
from types import SimpleNamespace as NS

import numpy as np
import pytest


def test_shared_loader_separates_external_and_internal_head_without_changing_defaults(monkeypatch):
    import inspect
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from bse.bse_loading import load_bse_data_from_restart_sharded as load
    from bse.bse_head import _inject_q0_head
    from file_io import restart_bundle
    from common.collectives import device_put_process_local
    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    g0 = np.array([1., .8+0.1j])
    header = dict(grid=(1, 1, 1), centroid_count=2, screened_ready=True,
                  head_vector=g0, bare_head=31., screened_head=np.array([18.+0j]),
                  logical_band_count=2, energies=np.array([[-.2, .5]]))
    monkeypatch.setattr(restart_bundle, "read_metadata", lambda _p: header)
    monkeypatch.setattr(restart_bundle, "refuse_mpa_static_w", lambda _p: None)
    v = device_put_process_local(np.eye(2, dtype=complex), NamedSharding(mesh, P("x", "y")))
    w = device_put_process_local(np.ones((2, 2, 1, 1, 1), complex),
                                NamedSharding(mesh, P("x", "y", None, None, None)))
    psi = device_put_process_local(np.ones((1, 1, 2, 2), complex),
                                  NamedSharding(mesh, P(None, None, None, "x")))
    monkeypatch.setattr(restart_bundle, "read_bse_payload", lambda *a, **k: (psi, psi, v, w, None))
    def read(**kwargs):
        return load("unused", 1, 1, mesh_xy=mesh, n_occ=1, cell_volume=20., **kwargs)
    bare = read(inject_head=False)
    selected = read(inject_head=False, inject_w_head=True, q0_head="diagonal")
    np.testing.assert_array_equal(selected["V_q0"], bare["V_q0"])
    np.testing.assert_array_equal(selected["W_q"], bare["W_q"])
    assert selected["q0_head_ry"] == 18./20. and bare["q0_head_ry"] == 0.
    np.testing.assert_array_equal(selected["eps_c"], [[.5]])
    # Existing optical defaults still inject both channels through one owner.
    assert inspect.signature(load).parameters["inject_w_head"].default is None
    legacy = read()
    vg, wg, _, _ = _inject_q0_head(v, w, jnp.asarray(g0), jnp.asarray(g0),
                                  31., jnp.array([18.+0j]), 20., w0_ready=True, nk=1)
    np.testing.assert_array_equal(legacy["V_q0"], vg)
    np.testing.assert_array_equal(legacy["W_q"], wg)
    with pytest.raises(ValueError, match="bse_screened_head"):
        read(inject_head=False, inject_w_head=True, need_W=False)
    header["screened_head"] = None
    with pytest.raises(ValueError, match="bse_screened_head"):
        read(inject_head=False, inject_w_head=True, q0_head="diagonal")


def test_static_head_identity_binds_resolved_scalar_geometry_and_policy(monkeypatch, tmp_path):
    from bse.w_ladder import ladder_static_head_identity, LADDER_INTERNAL_HEAD_POLICY
    from file_io import restart_bundle
    header = dict(grid=np.array([3, 3, 1]), head_vector=np.array([1., .8+0.1j]),
                  bare_head=31., screened_head=np.array([18.+0j]))
    monkeypatch.setattr(restart_bundle, "read_head_metadata", lambda _path: header)
    def identity(volume=20., deck=None):
        return ladder_static_head_identity("unused", deck, cell_volume=volume)
    original = identity()
    assert original["policy"] == LADDER_INTERNAL_HEAD_POLICY
    assert original["q0_head"] == "diagonal"
    assert original["q0_head_ry"] == 18./(20.*9)
    # Each of these changes the physical static operator independently of W0.
    for name, changed in (("screened_head", np.array([19.+0j])),
                          ("head_vector", np.array([1., .81+0.1j])),
                          ("grid", np.array([4, 3, 1]))):
        old = header[name]
        header[name] = changed
        assert identity() != original
        header[name] = old
    assert identity(volume=21.) != original
    deck = tmp_path / "cohsex.in"
    deck.write_text("whead_0freq = 23\n")
    overridden = identity(deck=str(deck))
    assert overridden["whead_ry_bohr3"] == [23., 0.]
    assert overridden["q0_head_ry"] == 23./(20.*9)
    header["screened_head"] = None
    with pytest.raises(ValueError, match="ladder_static_head_identity"):
        identity()


def test_static_rpa_rung_is_persisted_with_its_producer_diagram(monkeypatch):
    from gw import screening_bse as owner
    from gw import gw_output
    from gw.gw_config import HeadCorrection, ScreeningDiagrams
    @dataclass(frozen=True)
    class Screening:
        diagrams: object
    @dataclass(frozen=True)
    class Config:
        screening: object
        head: object
    config = Config(Screening(ScreeningDiagrams.W_BSE), NS(correction=HeadCorrection.OFF))
    static = np.ones((1, 2, 2))
    seen = []
    monkeypatch.setattr(owner, "_refuse_metallic_mean_field", lambda *a, **k: None)
    monkeypatch.setattr(owner, "_refuse_unusable_restart", lambda *a, **k: None)
    monkeypatch.setattr(owner, "_gate_w_or_refuse", lambda *a, **k: None)
    monkeypatch.setattr(owner, "_assert_restart_is_loadable", lambda *a, **k: None)
    monkeypatch.setattr(owner, "compute_static_w", lambda *a, **k: NS(unfold=lambda _m: static))
    monkeypatch.setattr(gw_output, "persist_w0_and_head", lambda *a, **k: seen.append(k["config"]))
    result = owner.prepare_ladder_restart(None, None, quad=None, e_ref=0.,
        sym=NS(), centroid_indices=[], config=config, meta=NS(kgrid=(1, 1, 1)),
        mesh_xy=None, tensors_filename="unused", head_resolver=None,
        print_fn=lambda *a: None)
    assert result is static
    assert config.screening.diagrams is ScreeningDiagrams.W_BSE
    assert len(seen) == 1 and seen[0].screening.diagrams is ScreeningDiagrams.W_RPA
    assert seen[0].head is config.head
