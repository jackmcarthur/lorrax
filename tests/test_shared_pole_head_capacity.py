"""The real head orchestration refuses oversized projection before tensor I/O."""
from types import SimpleNamespace as NS

import numpy as np
import pytest


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
                  n_mu_logical=368, nspinor=1)
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
