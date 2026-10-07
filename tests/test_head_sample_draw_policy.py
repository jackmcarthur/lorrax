"""Head sampler forwarding and existing Coulomb draw/refusal contracts."""
from types import SimpleNamespace

import numpy as np
import pytest


def _config():
    return SimpleNamespace(head=SimpleNamespace(
        vhead=None, whead_0freq=None, whead_imfreq=None,
        head_minibz_average=False, analytic_q0_sphere=False))


@pytest.mark.parametrize("policy", [None, dict(nsamples=32, qmc_reps=3, method="sobol")])
@pytest.mark.parametrize("static", [False, True])
def test_both_head_branches_forward_literal_draw_policy(monkeypatch, policy, static):
    from gw import vcoul
    from gw.qsgw_head import head_samples_from_s
    import gw.isdf_fitting

    monkeypatch.setattr(gw.isdf_fitting, "mem_probe", lambda *a, **kw: None)
    calls = []

    def screened(wfn, meta, tensors, **kw):
        calls.append(("screened", kw))
        return 7., [3.] * len(tensors)

    def unscreened(wfn, eps, meta, **kw):
        calls.append(("static", kw))
        return 7., 2.

    monkeypatch.setattr(vcoul, "compute_q0_averages_screened", screened)
    monkeypatch.setattr(vcoul, "compute_q0_averages", unscreened)
    z = [0j, 1j] if static else [1j, 2j]
    fields = head_samples_from_s(np.zeros((2, 3, 3)), z,
        wfn=object(), meta=object(), config=_config(),
        static_kappa2_bohr2=1. if static else None, **(policy or {}))
    expected = policy or dict(nsamples=2**18, qmc_reps=10, method="auto")
    assert [row[0] for row in calls] == (["screened", "static"] if static else ["screened"])
    for _, kwargs in calls:
        assert {key: kwargs[key] for key in expected} == expected
        assert kwargs["analytic_sphere"] is False
    assert [a.vc0 for a in fields] == [7.+0j, 7.+0j]
    assert [a.wcoul0 for a in fields] == ([2.+0j, 3.+0j] if static else [3.+0j, 3.+0j])


def test_public_draw_cache_binds_count_replicates_and_generator():
    from vcoul.minibz import minibz_voronoi_batches
    geometry = np.eye(3)
    first = minibz_voronoi_batches(geometry, (8, 8, 8), nsamples=16, qmc_reps=2, method="sobol")
    repeated = minibz_voronoi_batches(geometry, (8, 8, 8), nsamples=16, qmc_reps=2, method="sobol")
    assert len(first) == 2 and all(a.shape == (16, 3) for a in first)
    assert all(a is b for a, b in zip(first, repeated))
    larger = minibz_voronoi_batches(geometry, (8, 8, 8), nsamples=32, qmc_reps=3, method="sobol")
    assert len(larger) == 3 and all(a.shape == (32, 3) for a in larger)
    uniform = minibz_voronoi_batches(geometry, (8, 8, 8), nsamples=16, qmc_reps=2, method="uniform")
    assert len(uniform) == 1 and uniform[0].shape == (16, 3)
    assert not np.array_equal(first[0], uniform[0])


def test_explicit_sobol_refuses_unavailable_generator(monkeypatch):
    from scipy.stats import qmc
    from vcoul.minibz import minibz_voronoi_batches

    def unavailable(*args, **kwargs):
        raise ImportError("planted missing Sobol generator")

    monkeypatch.setattr(qmc, "Sobol", unavailable)
    with pytest.raises(RuntimeError, match="REFUSAL"):
        minibz_voronoi_batches(np.eye(3), (7, 8, 8), nsamples=8,
                              qmc_reps=1, method="sobol", seed_offset=71)


def test_head_forwarding_preserves_analytic_extra_chi_refusal():
    from gw.qsgw_head import head_samples_from_s
    cfg = _config()
    cfg.head.analytic_q0_sphere = True
    intraband = SimpleNamespace(drude_tensor=np.zeros((3, 3)))
    wfn = SimpleNamespace(blat=1., bvec=np.eye(3), cell_volume=1.)
    meta = SimpleNamespace(nkx=8, nky=8, nkz=8, sys_dim=3)
    with pytest.raises(NotImplementedError, match="bulk_q0_sphere_extra_chi_unavailable"):
        head_samples_from_s(np.zeros((1, 3, 3)), [1j], wfn=wfn,
            meta=meta, config=cfg, intraband=intraband,
            nsamples=8, qmc_reps=1, method="sobol")
