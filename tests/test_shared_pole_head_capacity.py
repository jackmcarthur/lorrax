"""The real head orchestration refuses oversized projection before tensor I/O."""
from types import SimpleNamespace as NS

import numpy as np
import pytest


def test_ordered_store_refuses_the_time_reversal_even_gamma_head(monkeypatch):
    """The Gamma body evaluates b (s - Lambda)^-1 b^dagger, the time-reversal-even form.

    An ordered store declares the signed particle-hole model, and gw_config forces
    head_correction = full for every shared-pole SC run, so evaluating one with the even
    formula would be silently wrong on the dominant Sigma term. Refuse by name instead,
    before any factor is read; the signed Gamma head belongs to the head branch.
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
    with pytest.raises(ValueError, match="GATE shared_pole_head_ordered"):
        build_shared_pole_head(dict(path="not-opened.h5"), header, None, None, meta, config,
            mesh_xy=mesh, wfn=None, response=NS(omegas=(1+.5j,)),
            head_resolver=None, plan=object(), material_class="semiconductor",
            occupation_state=None)


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


def test_two_component_store_refuses_by_its_own_name_not_as_capacity(monkeypatch):
    """N_spinor != 1 was folded into the capacity-geometry message; it is its own rule."""
    from file_io import slab_io
    from gw.gw_config import HeadCorrection
    from gw.mpa import sample_plan
    from gw.shared_pole_head import build_shared_pole_head
    from gw.shared_pole_recipe import CapacityLedger

    mesh = NS(size=4, shape={"x": 2, "y": 2})
    meta = NS(nk_tot=1, nspinor=2, n_rmu=368, mu_basis=NS(n_packed=368, n_logical=368))
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=1 << 30)
    meta.shared_pole_capacity.live_stages = ()
    header = dict(q_irr_full_idx=[0], n_q_full=1, n_mu_logical=368, nspinor=2,
                  representation="scalar-trs-even-s")
    config = NS(head=NS(uses_bgw_metal_q0shift=False, correction=HeadCorrection.FULL))
    monkeypatch.setattr(sample_plan, "plan_z", lambda _plan: np.array([1+.5j]))

    def tensor_read(*_args, **_kwargs):
        raise AssertionError("factors read before N_spinor was checked")

    monkeypatch.setattr(slab_io, "SlabIO", tensor_read)
    with pytest.raises(ValueError, match="GATE shared_pole_head_nspinor"):
        build_shared_pole_head(dict(path="not-opened.h5"), header, None, None, meta, config,
            mesh_xy=mesh, wfn=None, response=NS(omegas=(1+.5j,)),
            head_resolver=None, plan=object(), material_class="semiconductor",
            occupation_state=None)


@pytest.mark.parametrize("model,correction,trs,nspinor,rule", [
    ("shared_pole", "full", False, 1, "GATE shared_pole_head_ordered"),
    ("shared_pole", "full", True, 2, "GATE shared_pole_head_nspinor"),
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
    # trivial_view() drops TRS, which makes the store ordered: the door must see it.
    assert body.index("sym.trivial_view()") < door
    assert door < body.index("isdf_tensors_")
