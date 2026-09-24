"""The real head orchestration refuses oversized projection before tensor I/O."""
from types import SimpleNamespace as NS

import numpy as np
import pytest


def test_ordered_store_refuses_the_time_reversal_even_gamma_head(monkeypatch):
    """The Gamma body evaluates b (s - Lambda)^-1 b^dagger, the time-reversal-even form.

    An ordered store declares the signed particle-hole model, so folding the wings
    through the even body (head_correction = full) would be silently wrong; refuse by
    name before any factor is read, and point at the direct head an ordered store does
    carry (head_correction = no_local_fields; owner scope 2026-09-21).
    """
    from file_io import slab_io
    from gw.gw_config import HeadCorrection
    from gw.mpa import sample_plan
    from gw.shared_pole_head import build_shared_pole_head
    from gw.shared_pole_recipe import CapacityLedger

    mesh = NS(size=4, shape={"x": 2, "y": 2})
    meta = NS(nk_tot=1, nspinor=1, n_rmu=368, mu_basis=NS(n_packed=368, n_logical=368))
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=1 << 30)
    meta.shared_pole_capacity.live_stages = ()
    header = dict(q_irr_full_idx=[0], n_q_full=1, n_mu_logical=368, nspinor=1,
                  representation="scalar-ordered-ph", ordered=True)
    config = NS(head=NS(uses_bgw_metal_q0shift=False, correction=HeadCorrection.FULL))
    monkeypatch.setattr(sample_plan, "plan_z", lambda _plan: np.array([1+.5j]))

    def tensor_read(*_args, **_kwargs):
        raise AssertionError("factors read before the representation was checked")

    monkeypatch.setattr(slab_io, "SlabIO", tensor_read)
    with pytest.raises(ValueError, match="GATE shared_pole_head_ordered") as caught:
        build_shared_pole_head(dict(path="not-opened.h5"), header, None, None, meta, config,
            mesh_xy=mesh, wfn=None, response=NS(omegas=(1+.5j,)),
            head_resolver=None, plan=object(), material_class="semiconductor",
            occupation_state=None)
    assert "head_correction = no_local_fields" in str(caught.value)


def test_ordered_store_carries_the_direct_head_without_a_fold(monkeypatch):
    """head_correction = no_local_fields on an ordered store: the direct response sample
    is finalized with NO Gamma body (W_body_gamma is None), no realizer is bound, no
    factor is read, and the fit is the direct model.  The ordered wing/body fold is
    deferred (owner scope 2026-09-21); this pins that nothing evaluates it by accident."""
    import gw.qsgw_head as qsgw_head
    import gw.mpa.model as mpa_model
    from file_io import slab_io
    from gw import qgrid_symmetry
    from gw.gw_config import HeadCorrection
    from gw.mpa import sample_plan
    from gw.shared_pole_head import build_shared_pole_head
    from gw.shared_pole_recipe import CapacityLedger

    mesh = NS(size=4, shape={"x": 2, "y": 2})
    meta = NS(nk_tot=1, nspinor=2, n_rmu=368, mu_basis=NS(n_packed=368, n_logical=368),
              shared_pole_recipe={"census": {"mu_ry": 0.25, "energy_span_ry": 3.0}})
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=1 << 30)
    meta.shared_pole_capacity.live_stages = ()
    header = dict(q_irr_full_idx=[0], n_q_full=1, n_mu_logical=368, nspinor=2,
                  representation="scalar-ordered-ph", ordered=True)
    config = NS(head=NS(uses_bgw_metal_q0shift=False, correction=HeadCorrection.NO_LOCAL_FIELDS),
                mpa=NS(n_poles=4, pole_solver="loewner"))
    z = np.array([1+.5j, 2+.5j])
    monkeypatch.setattr(sample_plan, "plan_z", lambda _plan: z)
    monkeypatch.setattr(slab_io, "SlabIO",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("factors read for a direct head")))
    monkeypatch.setattr(qgrid_symmetry, "shared_pole_operator_realizer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("realizer bound for a direct head")))
    seen = []

    def finalize(response, index, W_body_gamma=None, **kwargs):
        seen.append((index, W_body_gamma))
        return NS(omega=response.omegas[index], source="head_direct")

    def fit(samples, points, n_poles, *, model, solve, occupation_state):
        assert model == "dft_direct_loewner" and len(samples) == len(points) == 2
        return {"model": model}

    monkeypatch.setattr(qsgw_head, "finalize_iteration_head_sample", finalize)
    monkeypatch.setattr(mpa_model, "fit_head_samples", fit)
    wfns = NS(enk=np.zeros((1, 4)), occ=np.zeros((1, 4)), slices=NS(sigma=slice(0, 4)))
    response = NS(omegas=tuple(map(complex, z)), trs_allowed=False)
    head, iteration = build_shared_pole_head(
        dict(path="not-opened.h5", identity="id", digest="d"), header, None, wfns, meta,
        config, mesh_xy=mesh, wfn=None, response=response, head_resolver=None,
        plan=object(), material_class="metal", occupation_state=None)
    assert seen == [(0, None), (1, None)]
    assert head["model"] == "dft_direct_loewner" and head["completion"] is True
    assert tuple(iteration.omegas) == tuple(map(complex, z))


@pytest.mark.parametrize("q_count,packed,allowed", [
    (1, 384, False),  # Orbit padding exceeds logical U even on all P ranks.
    (1, 368, True),   # Equality with the individual matrix bound is valid.
    (64, 384, True),  # The actual Si geometry is unchanged by this guard.
])
def test_head_projection_logical_bound_precedes_tensor_reads(monkeypatch, q_count, packed, allowed):
    from file_io import slab_io
    from gw import qgrid_symmetry
    from gw.gw_config import HeadCorrection
    from gw.mpa import sample_plan
    from gw.shared_pole_head import build_shared_pole_head
    from gw.shared_pole_recipe import CapacityLedger

    mesh = NS(size=4, shape={"x": 2, "y": 2})
    meta = NS(nk_tot=q_count, nspinor=1, n_rmu=368,
              mu_basis=NS(n_packed=packed, n_logical=368))
    # A large device budget deliberately permits >3U aggregate reservations.
    # It must not waive the separate, individual projection-matrix bound.
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh,
                                               device_budget_bytes=1 << 30)
    meta.shared_pole_capacity.live_stages = ()
    header = dict(q_irr_full_idx=[0], n_q_full=q_count,
                  n_mu_logical=368, nspinor=1,
                  representation="scalar-trs-even-s")
    config = NS(head=NS(uses_bgw_metal_q0shift=False, correction=HeadCorrection.FULL))
    monkeypatch.setattr(sample_plan, "plan_z", lambda _plan: np.array([1+.5j]))
    reached = []
    monkeypatch.setattr(qgrid_symmetry, "shared_pole_operator_realizer",
        lambda *_args, **_kwargs: reached.append("realizer"))

    class TensorReadReached(Exception):
        pass

    def tensor_read(*_args, **_kwargs):
        reached.append("tensor_read")
        raise TensorReadReached

    monkeypatch.setattr(slab_io, "SlabIO", tensor_read)
    expected = TensorReadReached if allowed else ValueError
    match = None if allowed else "Gamma projection exceeds the all-P logical matrix bound"
    with pytest.raises(expected, match=match):
        build_shared_pole_head(dict(path="not-opened.h5"), header, None, None, meta, config,
            mesh_xy=mesh, wfn=None, response=NS(omegas=(1+.5j,)),
            head_resolver=None, plan=object(), material_class="semiconductor",
            occupation_state=None)
    assert reached == (["realizer", "tensor_read"] if allowed else [])


def _two_component_geometry(monkeypatch, *, map_nspinor, store_nspinor):
    """A 368-centroid Gamma parent whose map and store may disagree on N_spinor."""
    from file_io import slab_io
    from gw import qgrid_symmetry
    from gw.gw_config import HeadCorrection
    from gw.mpa import sample_plan
    from gw.shared_pole_recipe import CapacityLedger

    mesh = NS(size=4, shape={"x": 2, "y": 2})
    meta = NS(nk_tot=1, nspinor=map_nspinor, n_rmu=368,
              mu_basis=NS(n_packed=368, n_logical=368))
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=1 << 30)
    meta.shared_pole_capacity.live_stages = ()
    header = dict(q_irr_full_idx=[0], n_q_full=1, n_mu_logical=368, nspinor=store_nspinor,
                  representation="scalar-trs-even-s")
    config = NS(head=NS(uses_bgw_metal_q0shift=False, correction=HeadCorrection.FULL))
    monkeypatch.setattr(sample_plan, "plan_z", lambda _plan: np.array([1+.5j]))
    reached = []
    monkeypatch.setattr(qgrid_symmetry, "shared_pole_operator_realizer",
        lambda *_args, **_kwargs: reached.append("realizer"))

    class TensorReadReached(Exception):
        pass

    def tensor_read(*_args, **_kwargs):
        reached.append("tensor_read")
        raise TensorReadReached

    monkeypatch.setattr(slab_io, "SlabIO", tensor_read)
    return meta, header, config, mesh, reached, TensorReadReached


def _build(meta, header, config, mesh):
    from gw.shared_pole_head import build_shared_pole_head
    return build_shared_pole_head(dict(path="not-opened.h5"), header, None, None, meta, config,
        mesh_xy=mesh, wfn=None, response=NS(omegas=(1+.5j,)),
        head_resolver=None, plan=object(), material_class="semiconductor",
        occupation_state=None)


def test_two_component_store_reaches_the_gamma_body_with_its_spin_geometry(monkeypatch):
    """The two-component store holds the same spin-traced charge operator (its factor
    spin axis is 1, the head vertices trace the spinor index), so the head evaluates it
    like a scalar store.  The ledger's unit is U = 16 Q (N_spinor N_mu)^2 / P; the head's
    geometry identity must carry the store's N_spinor or every two-component store would
    refuse as a geometry mismatch the moment the representation door admitted it."""
    meta, header, config, mesh, reached, TensorReadReached = _two_component_geometry(
        monkeypatch, map_nspinor=2, store_nspinor=2)
    with pytest.raises(TensorReadReached):
        _build(meta, header, config, mesh)
    assert reached == ["realizer", "tensor_read"]


@pytest.mark.parametrize("map_nspinor,store_nspinor", [(1, 2), (2, 1)])
def test_store_and_map_must_agree_on_spin_geometry(monkeypatch, map_nspinor, store_nspinor):
    """A store built for another spin geometry is a geometry mismatch, not a spin rule."""
    meta, header, config, mesh, reached, _ = _two_component_geometry(
        monkeypatch, map_nspinor=map_nspinor, store_nspinor=store_nspinor)
    with pytest.raises(ValueError, match="store/current-map geometry mismatch"):
        _build(meta, header, config, mesh)
    assert reached == []


def test_bispinor_lift_store_refuses_the_scalar_charge_head_by_name(monkeypatch):
    """N_spinor = 4 is the kinetic-balance lift; its Gamma completion is the packed
    photon head, so the scalar charge head refuses it by its own name before any read."""
    meta, header, config, mesh, reached, _ = _two_component_geometry(
        monkeypatch, map_nspinor=4, store_nspinor=4)
    with pytest.raises(ValueError, match="GATE shared_pole_head_nspinor"):
        _build(meta, header, config, mesh)
    assert reached == []


@pytest.mark.parametrize("model,correction,trs,nspinor,rule", [
    ("shared_pole", "full", False, 1, "GATE shared_pole_head_ordered"),
    ("shared_pole", "full", True, 2, None),
    ("shared_pole", "full", True, 4, "GATE shared_pole_head_nspinor"),
    ("shared_pole", "full", True, 1, None),
    ("shared_pole", "off", False, 2, None),
    ("shared_pole", "no_local_fields", False, 2, None),
    ("mpa", "full", False, 2, None),
])
def test_the_head_refusal_fires_at_input_resolution(model, correction, trs, nspinor, rule):
    """Q0HEAD 2026-09-16: the refusal fired only after bank and constructor had
    committed model.h5, which then blocked a rerun. It is now an input-door owner."""
    from gw.gw_config import HeadCorrection
    from gw.shared_pole_head import refuse_unsupported_shared_pole_head

    config = NS(sigma=NS(w_model=model), head=NS(correction=HeadCorrection(correction)))
    if rule is None:
        refuse_unsupported_shared_pole_head(config, trs_allowed=trs, nspinor=nspinor)
        return
    with pytest.raises(ValueError, match=rule) as caught:
        refuse_unsupported_shared_pole_head(config, trs_allowed=trs, nspinor=nspinor)
    assert "head_correction = off" in str(caught.value)


def test_four_component_charge_full_head_refuses_before_the_bank():
    """The source WFN has two components but the hybrid CC model stores four."""
    from gw.gw_config import (BispinorGWMode, ComputeMode, HeadCorrection,
                              ScreeningDiagrams)
    from gw.shared_pole_head import refuse_unsupported_shared_pole_head

    config = NS(bispinor=True, bispinor_gw=BispinorGWMode.BARE_TRANSVERSE,
                compute_mode=ComputeMode.MPA,
                screening=NS(diagrams=ScreeningDiagrams.W_RPA),
                sigma=NS(w_model="shared_pole"),
                head=NS(correction=HeadCorrection.FULL))
    with pytest.raises(ValueError, match="GATE shared_pole_head_nspinor"):
        refuse_unsupported_shared_pole_head(
            config, trs_allowed=True, nspinor=2)
    config.head.correction = HeadCorrection.NO_LOCAL_FIELDS
    refuse_unsupported_shared_pole_head(config, trs_allowed=True, nspinor=2)


def test_the_driver_calls_the_head_door_on_the_final_symmetry_before_any_build():
    """Read from disk: importing gw.gw_jax initializes the communicator stack."""
    import ast
    from pathlib import Path
    import gw.shared_pole_head as head

    source = (Path(head.__file__).parent / "gw_jax.py").read_text()
    fn = next(node for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.FunctionDef) and node.name == "_load_system_inputs")
    body = ast.get_source_segment(source, fn)
    door = body.index("refuse_unsupported_shared_pole_head(")
    # trivial_view() restricts the group before the door reads the final verdict.
    assert body.index("sym.trivial_view()") < door
    assert door < body.index("isdf_tensors_")
