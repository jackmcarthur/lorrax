"""Bare exchange keeps the Gamma cell and never needs dielectric files."""
from types import SimpleNamespace

import numpy as np
import pytest

from gw import head_correction, vcoul


def _resolver(**changes):
    head = dict(wcoul0_source="s_tensor", wcoul0_eta=0.0, vhead=None,
                whead_0freq=None, whead_imfreq=None,
                head_minibz_average=False, bgw_metal_q0_treatment="exact",
                correction="full")
    head.update(changes)
    config = SimpleNamespace(head=SimpleNamespace(**head), bispinor=True,
                             bispinor_gw="coulomb_only", nval=36, ncond=28,
                             nband=120, do_screened=False)
    return head_correction.HeadResolver(
        config, "/absent-dielectric-files", object(), None, object(),
        lambda *args: None)


def test_bare_head_uses_same_canonical_average_once(monkeypatch):
    calls = []

    def average(wfn, epsilon, meta, **kwargs):
        calls.append(kwargs)
        assert float(epsilon) == 1.0
        assert kwargs["S_cart"] is None
        return 3.25, 3.25

    monkeypatch.setattr(vcoul, "compute_q0_averages", average)
    monkeypatch.setattr(head_correction, "resolve_head_sample",
                        lambda *args, **kwargs: pytest.fail("unused dielectric read"))
    resolver = _resolver(head_minibz_average=True)
    head = resolver.bare_at(0.0)
    assert resolver.bare_at(0.0) is head
    assert head.vc0 == head.wcoul0 == 3.25
    assert len(calls) == 1 and calls[0]["analytic_sphere"]
    terms = head_correction.compute_static_head_terms_from_sample(
        head, occ=[1, 0], cell_volume=2.0, nk_tot=4)
    np.testing.assert_allclose(terms.sigma_x_diag, [-3.25/8, 0])
    np.testing.assert_array_equal(terms.sigma_sx_minus_x_diag, 0)
    np.testing.assert_array_equal(terms.sigma_coh_diag, 0)


def test_bare_v_override_needs_no_screened_override(monkeypatch):
    monkeypatch.setattr(vcoul, "compute_q0_averages",
                        lambda *args, **kwargs: pytest.fail("unused average"))
    head = _resolver(vhead=7.0).bare_at(0.0)
    assert head.vc0 == head.wcoul0 == 7.0
    assert head.response_kind is head_correction.HeadResponseKind.OVERRIDE


def test_off_head_does_not_reenable_bare_cell(monkeypatch):
    monkeypatch.setattr(vcoul, "compute_q0_averages",
                        lambda *args, **kwargs: pytest.fail("disabled head"))
    head = _resolver(correction="off", vhead=7.0).bare_at(0.0)
    assert head.vc0 == head.wcoul0 == 0.0
    assert head.response_kind is head_correction.HeadResponseKind.OFF


def test_screened_diagnostic_still_requires_configured_source(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("configured s_tensor")

    monkeypatch.setattr(head_correction, "resolve_head_sample", fail)
    with pytest.raises(FileNotFoundError, match="configured s_tensor"):
        _resolver().direct_at(0.0)
