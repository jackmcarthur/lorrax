"""Diagnostic bispinor switches (``common.bispinor_diag``): inert when unset.

CPU, pytest.  Checks the grammar and the route gate, that an unset
environment leaves ``HALFALPHA``, the transverse direct field and the Sigma^B
occupation projector exactly as they are, and that each set switch changes the
one quantity it names: the lift's small block scales with alpha, the field is
exactly zero, the lowest rows of the projector are zero.  The end-to-end
statement (an unset-switch run reproduces the parent commit) is a cluster
check.
"""

import os
import subprocess
import sys

import pytest

_NAMES = ("LORRAX_DIAG_TT_EXCHANGE", "LORRAX_DIAG_TRANSVERSE_HARTREE",
          "LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS", "LORRAX_DIAG_ALPHA_FS_SCALE")


@pytest.fixture(autouse=True)
def _clean_switches(monkeypatch):
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)


def test_grammar_and_route_gate(monkeypatch):
    from common import bispinor_diag as diag

    assert diag.tt_exchange_enabled() and diag.transverse_hartree_enabled()
    assert diag.tt_exchange_skip_bands() == 0
    assert diag.alpha_fs_scale() == 1.0
    assert diag.active_switches() is None
    # Unset: the gate has nothing to refuse, whatever the route.
    diag.refuse_outside_bare_transverse(
        bispinor=False, bare_transverse=False, packed_route=True)

    monkeypatch.setenv("LORRAX_DIAG_TT_EXCHANGE", " OFF ")
    monkeypatch.setenv("LORRAX_DIAG_TRANSVERSE_HARTREE", "off")
    monkeypatch.setenv("LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS", "8")
    monkeypatch.setenv("LORRAX_DIAG_ALPHA_FS_SCALE", "0.5")
    assert not diag.tt_exchange_enabled()
    assert not diag.transverse_hartree_enabled()
    assert diag.tt_exchange_skip_bands() == 8
    assert diag.alpha_fs_scale() == 0.5
    assert diag.active_switches() == (
        "LORRAX_DIAG_TT_EXCHANGE=off; LORRAX_DIAG_TRANSVERSE_HARTREE=off; "
        "LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS=8; LORRAX_DIAG_ALPHA_FS_SCALE=0.5")
    diag.refuse_outside_bare_transverse(
        bispinor=True, bare_transverse=True, packed_route=False)
    for route in (dict(bispinor=False, bare_transverse=True, packed_route=False),
                  dict(bispinor=True, bare_transverse=False, packed_route=False),
                  dict(bispinor=True, bare_transverse=True, packed_route=True)):
        with pytest.raises(ValueError, match="bispinor_diag_switch_route"):
            diag.refuse_outside_bare_transverse(**route)

    # The Hartree and alpha switches reach every bispinor route.
    monkeypatch.delenv("LORRAX_DIAG_TT_EXCHANGE")
    monkeypatch.delenv("LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS")
    diag.refuse_outside_bare_transverse(
        bispinor=True, bare_transverse=False, packed_route=True)

    for name, bad in (("LORRAX_DIAG_TT_EXCHANGE", "0"),
                      ("LORRAX_DIAG_TRANSVERSE_HARTREE", "false"),
                      ("LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS", "-1"),
                      ("LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS", "3.5"),
                      ("LORRAX_DIAG_ALPHA_FS_SCALE", "0"),
                      ("LORRAX_DIAG_ALPHA_FS_SCALE", "nan"),
                      ("LORRAX_DIAG_ALPHA_FS_SCALE", "two")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match=name):
            diag.active_switches()
        monkeypatch.delenv(name)


def test_unset_alpha_is_the_literal():
    from common.bispinor_init import ALPHA_FS, HALFALPHA

    assert HALFALPHA == 0.00364867628215
    assert ALPHA_FS == 2.0 * 0.00364867628215


def test_alpha_scale_reaches_the_lift_only_in_the_small_block():
    """A fresh interpreter: the scale is read where the constant is defined."""
    code = (
        "import numpy as np, jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "from common.bispinor_init import (HALFALPHA, LARGE_BLOCK_LIFT,\n"
        "    RAW_KINETIC_BALANCE_LIFT, lift_to_4spinor)\n"
        "rng = np.random.default_rng(7)\n"
        "psi = rng.normal(size=(1, 2, 2, 5)) + 1j * rng.normal(size=(1, 2, 2, 5))\n"
        "g = rng.integers(-3, 4, size=(1, 5, 3)).astype(float)\n"
        "k = np.array([[0.1, 0.2, 0.3]])\n"
        "b = np.eye(3) * 1.3\n"
        "raw = np.asarray(lift_to_4spinor(psi, g, k, b,\n"
        "    representation=RAW_KINETIC_BALANCE_LIFT))\n"
        "big = np.asarray(lift_to_4spinor(psi, g, k, b,\n"
        "    representation=LARGE_BLOCK_LIFT))\n"
        "np.save(sys.argv[1], np.concatenate((raw.ravel(), big.ravel(),\n"
        "    np.array([HALFALPHA]))))\n")
    import tempfile

    import numpy as np

    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, scale in (("one", None), ("half", "0.5")):
            env = dict(os.environ)
            env.pop("LORRAX_DIAG_ALPHA_FS_SCALE", None)
            if scale is not None:
                env["LORRAX_DIAG_ALPHA_FS_SCALE"] = scale
            path = os.path.join(tmp, label + ".npy")
            subprocess.run([sys.executable, "-c", "import sys\n" + code, path],
                           check=True, env=env)
            out[label] = np.load(path)
    n = (out["one"].size - 1) // 2
    raw_one, big_one = out["one"][:n].reshape(1, 2, 4, 5), out["one"][n:2 * n]
    raw_half, big_half = out["half"][:n].reshape(1, 2, 4, 5), out["half"][n:2 * n]
    assert out["one"][-1].real == 0.00364867628215
    assert out["half"][-1].real == 0.5 * 0.00364867628215
    assert np.array_equal(big_one, big_half)                 # charge carrier
    assert np.array_equal(raw_one[:, :, :2], raw_half[:, :, :2])
    assert np.array_equal(0.5 * raw_one[:, :, 2:], raw_half[:, :, 2:])
    assert np.max(np.abs(raw_one[:, :, 2:])) > 0.0


def test_transverse_field_switch(monkeypatch):
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from psp.dft_operators import transverse_potential_from_current
    from vcoul import COULOMB_GAUGE_TT_SIGN

    rng = np.random.default_rng(11)
    current = jnp.asarray(rng.normal(size=(3, 6, 6, 6)))
    bvec = np.eye(3)
    blat = 1.1
    bdot = (bvec * blat) @ (bvec * blat).T
    args = (current, jnp.asarray(bdot), jnp.asarray(bvec), blat, False)
    live = np.asarray(transverse_potential_from_current(
        *args, tt_metric_sign=float(COULOMB_GAUGE_TT_SIGN)))
    assert live.shape == (3, 6, 6, 6) and np.max(np.abs(live)) > 0.0
    monkeypatch.setenv("LORRAX_DIAG_TRANSVERSE_HARTREE", "off")
    off = np.asarray(transverse_potential_from_current(
        *args, tt_metric_sign=float(COULOMB_GAUGE_TT_SIGN)))
    assert off.shape == live.shape and off.dtype == live.dtype
    assert not np.any(off)
    monkeypatch.delenv("LORRAX_DIAG_TRANSVERSE_HARTREE")
    again = np.asarray(transverse_potential_from_current(
        *args, tt_metric_sign=float(COULOMB_GAUGE_TT_SIGN)))
    assert np.array_equal(again, live)


def test_skip_bands_masks_the_lowest_rows_of_the_projector():
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from types import SimpleNamespace
    from gw.cohsex_sigma import _occ_diag_full
    from gw.sigma_x_bispinor import _drop_lowest_occupied

    nk, nb, nocc = 3, 7, 5
    occ = np.zeros((nk, nb, nb))
    occ[:, np.arange(nocc), np.arange(nocc)] = 1.0
    occ[1, nocc - 1, nocc - 1] = 0.25                        # a fractional row
    Gij = jnp.asarray(occ)
    masked = _drop_lowest_occupied(Gij, 2, SimpleNamespace(b0=0))
    assert masked.shape == Gij.shape and masked.dtype == Gij.dtype
    weights = np.asarray(_occ_diag_full(masked, nb, nb + 1))
    expected = np.asarray(_occ_diag_full(Gij, nb, nb + 1)).copy()
    expected[:, :2] = 0.0
    assert np.array_equal(weights, expected)
    assert np.array_equal(np.asarray(Gij), occ)              # input untouched
    for bad in (0, nb, nb + 3):
        with pytest.raises(ValueError, match="SKIP_BANDS"):
            _drop_lowest_occupied(Gij, bad, SimpleNamespace(b0=0))
