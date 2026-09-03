"""Regression coverage for the live mean-field output guard."""

import numpy as np

from gw.gw_output import _warn_on_unphysical_h0


def test_unphysical_h0_warning_names_the_live_exact_source(monkeypatch):
    """A tripped warning must report the defect rather than raise NameError."""
    monkeypatch.setenv("LORRAX_SANITY", "warn")
    messages = []
    implied = _warn_on_unphysical_h0(
        e_dft_ev=np.array([[0.0]]),
        kin_ion_diag_ev=np.array([[-100.0]]),
        hartree_diag_ev=np.array([[0.0]]),
        print_fn=messages.append,
    )

    assert implied[0, 0] == 100.0
    assert any(
        "H0 = kin_ion + V_H[exact, live G-space] is UNPHYSICAL" in message
        for message in messages
    )
