"""CPU pins of ``sigma_w_time_reversal = even`` (theory (W 4), (W 35)).

1. The producer's fold (``gw.response_bank._even_fold_program``) of a magnetic
   Lehmann W with its minus-q partner W_q(-conj z) is the even Lehmann form
   (W 5) and its s-derivative, against the closed form; on the imaginary axis
   it is the Hermitian part.
2. The routing (``gw.tangential_recipe.ordered_w_model``) and the gate tables:
   the measured TRS and ordered tables keep their hashes, and the even table
   differs only in its representation row.
3. The parse gates refuse the key off the scalar tangential RPA route.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest


def _lehmann(rng, n, poles):
    """Positive residues R_n, R~_m and energies of a magnetic (R != R~) parent."""
    def psd():
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return a @ a.conj().T / n
    return ([psd() for _ in range(poles)], rng.uniform(0.3, 2.0, poles),
            [psd() for _ in range(poles)], rng.uniform(0.3, 2.0, poles))


def _w(model, z):
    """W_q(z) - v = sum R/(z - E) - sum R~/(z + E~) (W 1) and dW/dz."""
    R, E, Rt, Et = model
    w = sum(r / (z - e) for r, e in zip(R, E)) - sum(r / (z + e) for r, e in zip(Rt, Et))
    dw = -sum(r / (z - e) ** 2 for r, e in zip(R, E)) + sum(r / (z + e) ** 2 for r, e in zip(Rt, Et))
    return w, dw


def test_even_fold_is_the_even_lehmann_form():
    import jax
    from gw.response_bank import _even_fold_program

    tol = 1e-12 if jax.config.jax_enable_x64 else 2e-5
    rng = np.random.default_rng(7)
    model = _lehmann(rng, 5, 4)
    R, E, Rt, Et = model
    fold = _even_fold_program(None, False)
    for z in (0.7 + 0.2j, -1.3 + 0.19j, 0.45j):
        w, dw = _w(model, z)
        wm, dwm = _w(model, -np.conj(z))
        # The producer's slopes are d/ds = (d/dz) / (2 node) at each node.
        got_w, got_ds = (np.asarray(a) for a in fold(w[None], (dw / (2 * z))[None],
                                                     wm[None], (dwm / (-2 * np.conj(z)))[None]))
        s = z * z
        want_w = (sum(e * r / (s - e * e) for r, e in zip(R, E))
                  + sum(e * r / (s - e * e) for r, e in zip(Rt, Et)))
        want_ds = -(sum(e * r / (s - e * e) ** 2 for r, e in zip(R, E))
                    + sum(e * r / (s - e * e) ** 2 for r, e in zip(Rt, Et)))
        scale = np.abs(want_w).max()
        assert np.abs(got_w[0] - want_w).max() <= tol * scale, z
        assert np.abs(got_ds[0] - want_ds).max() <= tol * np.abs(want_ds).max(), z
        # The odd channel this drops is nonzero on this magnetic parent.
        assert np.abs(w - want_w).max() > 1e-3 * scale
        if z.real == 0:
            assert np.abs(got_w[0] - 0.5 * (w + w.conj().T)).max() <= tol * scale


def test_routing_and_gate_tables():
    from gw.tangential_recipe import (GATE_HASH, ORDERED_GATE_HASH, ordered_w_model,
                                      scalar_gate_table, shared_real_pole_gates_trs_even_v1,
                                      shared_real_pole_gates_v1_r3b, table_hash)
    broken, holds = SimpleNamespace(trs_allowed=False), SimpleNamespace(trs_allowed=True)
    even = {"time_reversal_w": "even"}
    assert ordered_w_model(broken, {}) and ordered_w_model(broken, None)
    assert not ordered_w_model(broken, even)
    assert not ordered_w_model(holds, {}) and not ordered_w_model(holds, even)
    assert table_hash(scalar_gate_table(trs_allowed=True)) == GATE_HASH
    assert table_hash(scalar_gate_table(trs_allowed=False)) == ORDERED_GATE_HASH
    assert scalar_gate_table(trs_allowed=False, recipe=even) is shared_real_pole_gates_trs_even_v1
    assert scalar_gate_table(trs_allowed=True, recipe=even) is shared_real_pole_gates_v1_r3b
    differ = {k for k in shared_real_pole_gates_v1_r3b
              if shared_real_pole_gates_v1_r3b[k] != shared_real_pole_gates_trs_even_v1[k]}
    assert differ == {"representation"}


BASE = """[cohsex]
sys_dim = 3
compute_mode = tangential
"""


@pytest.mark.parametrize("extra, gate", [
    ("sigma_w_time_reversal = even\n", None),
    ("sigma_w_time_reversal = odd\n", "GATE tangential_enum"),
    ("sigma_w_time_reversal = even\nbispinor = true\n", "GATE tangential_trs_even_route"),
    ("sigma_w_time_reversal = even\nscreening_diagrams = w_bse\n", "GATE tangential_trs_even_route"),
])
def test_parse_gates(tmp_path, extra, gate):
    from gw.gw_config import read_lorrax_input
    deck = tmp_path / "w.in"

    def read(text):
        deck.write_text(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return read_lorrax_input(str(deck))
    if gate is None:
        assert read(BASE + extra)["sigma_w_time_reversal"] == "even"
        assert read(BASE)["sigma_w_time_reversal"] == "measured"
        with pytest.raises(ValueError, match="GATE tangential_applicability"):
            read(BASE.replace("tangential", "gn_ppm") + extra)
        return
    with pytest.raises(ValueError, match=gate):
        read(BASE + extra)
