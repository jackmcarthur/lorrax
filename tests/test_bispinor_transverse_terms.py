"""The transverse-term deck keys of a bispinor run, and the one diagnostic switch.

CPU, pytest.  The keys ``bispinor_transverse_hartree``,
``bispinor_transverse_exchange``, ``bispinor_transverse_exchange_skip_bands``
and ``bispinor_semicore_fold`` (docs/input_reference.md): the parse, the
route gates with a control that makes each fire, the stamp every output
carries, and that each key changes the one quantity it names at its one
seam (the direct field's producer is an exact zero, Sigma^B is an exact zero
without a contraction, the lowest rows of Sigma^B's occupied projector are
zero, a count inside a manifold refuses).  ``LORRAX_DIAG_ALPHA_FS_SCALE``
reaches the lift's small block only.  The end-to-end statements (both terms
off equals the two-component run; the alpha scale halves both terms) are
the suite stage ``bisp_bt_null`` and the P = 4 linearity leg.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gw.gw_config import LorraxConfig

_BISPINOR = """[cohsex]
sys_dim = 3
compute_mode = tangential
qp_solver = one_shot_dft
number_bands = 30
nval = 4
ncond = 4
bispinor = true
"""


def _config(tmp_path, text):
    path = tmp_path / "gw.in"
    path.write_text(text)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                       runtime_platform="cpu",
                                       print_fn=lambda *_a: None)


def test_keys_route_gates_and_stamp(tmp_path):
    from gw.gw_config import bispinor_transverse_terms_stamp
    default = _config(tmp_path, _BISPINOR)
    assert bispinor_transverse_terms_stamp(default) == (
        "hartree=on; exchange=on; semicore=none")
    off = _config(tmp_path, _BISPINOR + "bispinor_transverse_hartree = false\n"
                  "bispinor_transverse_exchange = false\n")
    assert bispinor_transverse_terms_stamp(off) == (
        "hartree=off; exchange=off; semicore=none")
    semicore = _config(tmp_path, _BISPINOR + "bispinor_transverse_exchange_skip_bands = 8\n"
                       "bispinor_semicore_fold = true\n")
    assert semicore.bispinor_transverse_exchange_skip_bands == 8
    assert bispinor_transverse_terms_stamp(semicore).endswith(
        "semicore=excluded_bands:8+fold")
    scalar = _config(tmp_path, _BISPINOR.replace("bispinor = true", "bispinor = false"))
    assert bispinor_transverse_terms_stamp(scalar) is None

    # Controls that make each gate fire.
    for key in ("bispinor_transverse_hartree = true", "bispinor_semicore_fold = false",
                "bispinor_transverse_exchange_skip_bands = 0"):
        with pytest.raises(ValueError, match="GATE bispinor_transverse_terms_route"):
            _config(tmp_path, _BISPINOR.replace("bispinor = true", "bispinor = false")
                    + key + "\n")
    with pytest.raises(ValueError, match="GATE bispinor_transverse_exchange_skip_bands"):
        _config(tmp_path, _BISPINOR + "bispinor_transverse_exchange_skip_bands = -1\n")
    full = _BISPINOR + "bispinor_gw = full_tangential\n"
    for key in ("bispinor_transverse_exchange = false",
                "bispinor_transverse_exchange_skip_bands = 2"):
        with pytest.raises(ValueError, match="GATE bispinor_transverse_exchange_route"):
            _config(tmp_path, full + key + "\n")
    # The direct field has one producer for every route.
    assert not _config(tmp_path, full + "bispinor_transverse_hartree = false\n"
                       ).bispinor_transverse_hartree


def test_direct_field_key_is_an_exact_zero_at_the_producer():
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
    sign = float(COULOMB_GAUGE_TT_SIGN)
    live = np.asarray(transverse_potential_from_current(
        *args, tt_metric_sign=sign, transverse_hartree=True))
    assert live.shape == (3, 6, 6, 6) and np.max(np.abs(live)) > 0.0
    off = np.asarray(transverse_potential_from_current(
        *args, tt_metric_sign=sign, transverse_hartree=False))
    assert off.shape == live.shape and off.dtype == live.dtype
    assert not np.any(off)


def test_exchange_keys_act_on_sigma_b_alone():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from gw.cohsex_sigma import _occ_diag_full
    from gw.sigma_x_bispinor import (compute_sigma_x_bispinor,
                                     transverse_exchange_occupations)

    nk, nb, nocc = 3, 7, 5
    occ = np.zeros((nk, nb, nb))
    occ[:, np.arange(nocc), np.arange(nocc)] = 1.0
    occ[1, nocc - 1, nocc - 1] = 0.25                        # a fractional row
    Gij = jnp.asarray(occ)
    slices = SimpleNamespace(b0=0, nb_sigma=nb)
    assert transverse_exchange_occupations(
        Gij, SimpleNamespace(transverse_exchange_skip_bands=0), slices) is Gij
    masked = transverse_exchange_occupations(
        Gij, SimpleNamespace(transverse_exchange_skip_bands=2), slices)
    assert masked.shape == Gij.shape and masked.dtype == Gij.dtype
    weights = np.asarray(_occ_diag_full(masked, nb, nb + 1))
    expected = np.asarray(_occ_diag_full(Gij, nb, nb + 1)).copy()
    expected[:, :2] = 0.0
    assert np.array_equal(weights, expected)
    assert np.array_equal(np.asarray(Gij), occ)              # input untouched
    for bad in (nb, nb + 3):
        with pytest.raises(ValueError, match="skip_bands"):
            transverse_exchange_occupations(
                Gij, SimpleNamespace(transverse_exchange_skip_bands=bad), slices)

    # Exchange off: an exact zero of the production shape; nothing else of
    # the transverse family is touched (no plan, no V file).
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    zero = compute_sigma_x_bispinor(
        wfns_transverse=SimpleNamespace(slices=slices), Gij=Gij,
        bispinor_v_q_path=None, meta=SimpleNamespace(transverse_exchange=False,
                                                     nk_tot=nk),
        mesh_xy=mesh, mu_bases=None, verbose=False)
    assert zero.shape == (nk, nb, nb) and zero.dtype == jnp.complex128
    assert not np.any(np.asarray(zero))


def test_skip_count_must_close_a_semicore_manifold():
    from gw.gw_init import transverse_exchange_skip_bands
    from common.units import RYD_TO_EV

    # Two k-points: 2 bands at -50 eV, 3 valence bands near -5 eV, 2 empty.
    e_ev = np.array([[-50.1, -50.0, -6.0, -5.5, -5.0, 1.0, 2.0],
                     [-50.2, -49.9, -6.1, -5.4, -4.9, 1.1, 2.1]])
    wfn = SimpleNamespace(nelec=5, energies=(e_ev / RYD_TO_EV)[None])
    said = []
    take = lambda n: transverse_exchange_skip_bands(
        SimpleNamespace(bispinor_transverse_exchange_skip_bands=n), wfn, said.append)
    assert take(0) == 0 and not said
    assert take(2) == 2 and "bands 1-2" in said[-1]
    for bad in (1, 3, 5, 6):                                 # inside a manifold, or not semicore
        with pytest.raises(ValueError, match="GATE bispinor_transverse_exchange_skip_bands"):
            take(bad)


def test_alpha_switch_grammar_and_two_component_refusal(monkeypatch):
    from common import bispinor_diag as diag
    from common.bispinor_init import ALPHA_FS, HALFALPHA

    monkeypatch.delenv("LORRAX_DIAG_ALPHA_FS_SCALE", raising=False)
    assert HALFALPHA == 0.00364867628215 and ALPHA_FS == 2.0 * 0.00364867628215
    assert diag.alpha_fs_scale() == 1.0 and diag.active_switches() is None
    diag.refuse_on_two_component(bispinor=False)              # unset: nothing to refuse
    monkeypatch.setenv("LORRAX_DIAG_ALPHA_FS_SCALE", "0.5")
    assert diag.active_switches() == "LORRAX_DIAG_ALPHA_FS_SCALE=0.5"
    diag.refuse_on_two_component(bispinor=True)
    with pytest.raises(ValueError, match="GATE bispinor_diag_switch_route"):
        diag.refuse_on_two_component(bispinor=False)
    for bad in ("0", "nan", "two", "-1"):
        monkeypatch.setenv("LORRAX_DIAG_ALPHA_FS_SCALE", bad)
        with pytest.raises(ValueError, match="LORRAX_DIAG_ALPHA_FS_SCALE"):
            diag.active_switches()


def test_alpha_scale_reaches_the_lift_only_in_the_small_block(tmp_path):
    """A fresh interpreter: the scale is read where the constant is defined."""
    code = (
        "import sys\n"
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
    out = {}
    for label, scale in (("one", None), ("half", "0.5")):
        env = dict(os.environ)
        env.pop("LORRAX_DIAG_ALPHA_FS_SCALE", None)
        if scale is not None:
            env["LORRAX_DIAG_ALPHA_FS_SCALE"] = scale
        path = tmp_path / f"{label}.npy"
        subprocess.run([sys.executable, "-c", code, str(path)], check=True, env=env)
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
