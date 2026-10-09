"""Shared-pole route and resume guards on toy inputs (CPU only, seconds).

1. A complete bank built against another bare-V digest (another P, older code)
   is not resumable: the map rebuilds instead of refusing.
2. The pole-budget keep cut never splits a degenerate multiplet at its edge:
   the tied member leaves with its partner, so K <= budget and the kept span
   does not depend on the eigenbasis inside the multiplet.
3. A local CT round runs at its held spans whatever its price (warn, never
   refuse: the route was decided from the recipe shapes, and there is no rerun).
"""
import hashlib
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)



@pytest.mark.parametrize("model, charge, transverse, photon", [
    ("coulomb_only", True, False, False),
    ("bare_transverse", True, True, False),
    ("full_shared_pole", False, False, True),
    ("full_static_cohsex", False, False, False),
])
def test_four_component_charge_bank_routing(model, charge, transverse, photon):
    from gw.gw_config import (
        ComputeMode, ScreeningDiagrams, uses_bare_transverse_shared_pole,
        uses_charge_bispinor_shared_pole, uses_full_bispinor_shared_pole,
    )
    from gw.shared_pole_recipe import resolve_shared_pole_recipe

    config = SimpleNamespace(
        bispinor=True, bispinor_gw=model, compute_mode=ComputeMode.MPA,
        sigma=SimpleNamespace(w_model="shared_pole"),
        screening=SimpleNamespace(diagrams=ScreeningDiagrams.W_RPA))
    assert uses_charge_bispinor_shared_pole(config) is charge
    assert uses_bare_transverse_shared_pole(config) is transverse
    assert uses_full_bispinor_shared_pole(config) is photon
    # Exercise the resolver seam itself: admitted carriers must reach the
    # mandatory current-census check, while a static photon model refuses
    # the dynamic carrier before constructing or evaluating any response.
    expected = "shared_pole_census" if charge or photon else "shared_pole_representation"
    with pytest.raises(ValueError, match=expected):
        resolve_shared_pole_recipe(
            config, None, SimpleNamespace(nspinor=4), mesh_xy=None,
            print_fn=lambda *a, **k: None)
    config.bispinor = False
    assert not uses_charge_bispinor_shared_pole(config)


def test_shared_pole_tables_preserve_fractional_translated_and_tr_centroids():
    from gw.shared_pole_screening import _shared_pole_tables

    # The two nearby points snap to the same FFT cell. Their translated
    # partners close the physical set, including a nonzero lattice wrap.
    fractional = np.array([[.03,.1,.2], [.53,.1,.2],
                           [.031,.1,.2], [.531,.1,.2]])
    basis = SimpleNamespace(canonical_indices=fractional, coordinate_kind="fractional")
    sym = SimpleNamespace(
        sym_matrices=np.repeat(np.eye(3, dtype=np.int32)[None],2,axis=0),
        translations=np.array([[0.,0.,0.],[np.pi,0.,0.]]),
        irr_idx_q=np.zeros(4,np.int32), sym_idx_q=np.arange(4,dtype=np.int32),
        q_irr_kgrid_int=np.zeros((1,3),np.int32), q_irr_full_idx=np.array([0]))
    meta = SimpleNamespace(fft_grid=np.array([4,4,4]),nkx=1,nky=1,nkz=1)
    tables = _shared_pole_tables(meta,sym,basis)['qirr']
    expected = np.array([[0,1,2,3],[1,0,3,2]],np.int32)
    np.testing.assert_array_equal(tables.sym_perm,np.concatenate((expected,expected)))
    wraps = np.zeros((2,4,3),np.int32);wraps[1,[0,2],0] = -1
    np.testing.assert_array_equal(tables.L_table,np.concatenate((wraps,wraps)))
    k = np.array([1/7,1/11,-1/9]);g = np.array([2,-1,0])
    wave = np.exp(2j*np.pi*(fractional@(k+g)))
    shifted = fractional-np.array([.5,0,0])
    phase = np.exp(2j*np.pi*(tables.L_table[1]@k))
    pulled = phase*wave[tables.sym_perm[1]]
    np.testing.assert_allclose(pulled,np.exp(2j*np.pi*(shifted@(k+g))),atol=5e-15,rtol=0)
    # Time reversal keeps the centroid map and conjugates both Bloch factors.
    tr_pulled = phase.conj()*wave[tables.sym_perm[3]].conj()
    np.testing.assert_allclose(tr_pulled,np.exp(-2j*np.pi*(shifted@(k+g))),atol=5e-15,rtol=0)


def test_shared_pole_integer_tables_are_unchanged():
    from gw.shared_pole_screening import _shared_pole_tables
    from symmetry_maps import centroid_source_map_and_wrap

    integer = np.array([[0,1,2],[2,1,2]],np.int32)
    sym = SimpleNamespace(
        sym_matrices=np.repeat(np.eye(3,dtype=np.int32)[None],2,axis=0),
        translations=np.array([[0.,0.,0.],[np.pi,0.,0.]]),
        irr_idx_q=np.zeros(4,np.int32),sym_idx_q=np.arange(4,dtype=np.int32),
        q_irr_kgrid_int=np.zeros((1,3),np.int32),q_irr_full_idx=np.array([0]))
    meta = SimpleNamespace(fft_grid=np.array([4,4,4]),nkx=1,nky=1,nkz=1)
    perm,wraps = centroid_source_map_and_wrap(
        integer,sym.sym_matrices,sym.translations,meta.fft_grid,extend_trs=True)
    basis = SimpleNamespace(canonical_indices=integer,coordinate_kind="fft_indices")
    tables = _shared_pole_tables(meta,sym,basis)['qirr']
    assert tables.sym_perm.tobytes()==perm.tobytes()
    assert tables.L_table.tobytes()==wraps.tobytes()


def _toy_shared_pole_identity(monkeypatch, points, *, coordinate_kind="fractional",
                              source="source", label="dft", state=None):
    from common import parallel_transport
    from gw import response_bank
    from gw.shared_pole_screening import shared_pole_identity

    monkeypatch.setattr(parallel_transport, "wfn_fingerprint", lambda wfn: source)
    monkeypatch.setattr(parallel_transport, "fingerprint_from_binding", lambda binding, wfn: source)
    monkeypatch.setattr(response_bank, "response_weights", lambda wfns, meta: (
        None, dict(energy_sha256="energies", occupation_sha256="occupations")))
    basis = SimpleNamespace(canonical_indices=points, coordinate_kind=coordinate_kind)
    meta = SimpleNamespace(mu_basis=basis, shared_pole_recipe=dict(recipe_hash="recipe", gate_hash="gates"),
                           shared_pole_state_identity=state)
    identity = shared_pole_identity(None, meta, label=label, wfn=None,
                                   binding=None, centroid_indices=points)
    return identity, meta


def test_shared_pole_fractional_identity_rejects_stale_bank_and_model(tmp_path, monkeypatch):
    import file_io.shared_pole_store as store
    from file_io.wfn_basis import centroid_table_md5, centroid_table_fingerprint_scheme
    from gw.shared_pole_screening import _authenticated_constructor_resume, _published_sector_handle

    first = np.array([[.03,.1,.2], [.53,.1,.2]])
    moved = first.copy(); moved[0,0] += .001
    np.testing.assert_array_equal(first.astype(np.int64), moved.astype(np.int64))
    old, _ = _toy_shared_pole_identity(monkeypatch, first)
    current, _ = _toy_shared_pole_identity(monkeypatch, moved)
    assert old["centroids"] != current["centroids"]
    assert current["centroids"] == centroid_table_md5(moved, coordinate_kind="fractional")
    assert current["centroid_coordinate_kind"] == "fractional"
    assert current["centroid_fingerprint_scheme"] == centroid_table_fingerprint_scheme("fractional")
    # This is the common identity gate used by both the real bank and model
    # validators, including the early tensor-restart branch before V exists.
    store._check_identity(current, current)
    with pytest.raises(ValueError, match="centroids"):
        store._check_identity(old, current)
    coulomb = dict(basis="canonical", sha256="aa" * 32)
    for name in ("bank", "moments"):
        (tmp_path / f"{name}_receipt.json").write_text(json.dumps(dict(
            identity=old, completion=True, bank_complete=True, coulomb_identity=coulomb)))
    (tmp_path / "bank.h5").write_bytes(b"")
    monkeypatch.setattr(store, "validate_shared_pole_bank", lambda *a, **k: {})
    assert _authenticated_constructor_resume(tmp_path, old, {}, coulomb_sha256=coulomb["sha256"])
    assert not _authenticated_constructor_resume(tmp_path, current, {}, coulomb_sha256=coulomb["sha256"])
    (tmp_path / "sectors.json").write_text(json.dumps(dict(identity=old)))
    with pytest.raises(ValueError, match="binds another"):
        _published_sector_handle(tmp_path, current)


def test_shared_pole_integer_identity_preserves_legacy_protocol(monkeypatch):
    points = np.array([[0,1,2], [2,1,2]], np.int32)
    identity, _ = _toy_shared_pole_identity(monkeypatch, points, coordinate_kind="fft_indices")
    assert identity == dict(iteration_id="dft", wavefunctions="source",
        hamiltonian=hashlib.sha256(b"sourceenergies").hexdigest(),
        energies="energies", occupations="occupations",
        centroids=hashlib.sha256(np.asarray(points,np.int64).tobytes()).hexdigest(),
        recipe_hash="recipe", gate_hash="gates")


def test_shared_pole_identity_binds_canonical_carrier_and_sc_source(monkeypatch):
    import file_io.shared_pole_store as store
    from gw.shared_pole_screening import shared_pole_identity

    points = np.array([[.03,.1,.2], [.53,.1,.2]])
    old, meta = _toy_shared_pole_identity(monkeypatch, points)
    moved = points.copy(); moved[0,0] += .001
    with pytest.raises(ValueError, match="canonical basis"):
        shared_pole_identity(None, meta, label="dft", wfn=None, binding=None, centroid_indices=moved)
    current, _ = _toy_shared_pole_identity(monkeypatch, points, source="other-source")
    with pytest.raises(ValueError, match="wavefunctions"):
        store._check_identity(old, current)
    state = dict(hamiltonian="sc-H", wavefunctions="sc-U")
    old_sc, _ = _toy_shared_pole_identity(monkeypatch, points, label="sc_1", state=state)
    current_sc, _ = _toy_shared_pole_identity(monkeypatch, points, source="other-source",
                                             label="sc_1", state=state)
    assert old_sc["hamiltonian"] == current_sc["hamiltonian"] == "sc-H"
    assert old_sc["wavefunctions"] == current_sc["wavefunctions"] == "sc-U"
    with pytest.raises(ValueError, match="wfn"):
        store._check_identity(old_sc, current_sc)


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
