"""A W bank sample with no accepted complex-time rule refuses by name.

At wide QP windows the tangential resolver raises its top line site with the
highest Sigma read, and one line sample's complex-time fit over the whole
chi0 interval stops passing the coefficient-mass gate (NiO, CONV_NIO T6:
interval [0.3162, 27.858] Ry, height 2.6 eV, rel_tol 1e-8: 67.0 eV passes,
71.89 eV, the window-152 top site, is refused; that fit takes minutes, so
the cell here forces the same gate with a tight mass bound on a short
interval). The refusal names the sample in eV, the gate that refused it and
the deck key that avoids it (``GATE tangential_response_fit``,
``sigma_w_support_sites_ev``); the same sample under the bank's own gate
builds its rule as before.

Host-side rule construction, no device code: a CPU unit cell, exempt from
the four-GPU rule under its unit-and-CPU-cell clause.
"""
import pytest

RY = 13.605693122994


def _tree(**gate):
    import numpy as np
    from ffi import _services
    _services.ensure_on_path()
    import minimax
    z = np.asarray([(10.0 + 2.6j) / RY])
    return minimax.ResponseGroupTree(0.3162, 3.0, z, rel_tol=1e-8, decay_rate=0., **gate)


def test_response_fit_refusal_names_sample_gate_and_key():
    from gw.response_bank import _gather_group_rules
    rules = _gather_group_rules([[0]], lambda members: _tree())
    assert len(rules) == 1 and rules[0]
    with pytest.raises(ValueError) as refused:
        _gather_group_rules([[0]], lambda members: _tree(max_kappa=1.0))
    text = str(refused.value)
    assert text.startswith("GATE tangential_response_fit")
    assert "10.000 + 2.600i eV" in text
    assert "coefficient-mass gate (kappa <= 1)" in text
    assert "sigma_w_support_sites_ev" in text
