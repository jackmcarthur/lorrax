"""Literal full-centroid static-control/head oracle; CPU, not GPU admission."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import runtime  # noqa: F401
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

from tests.tdagw.test_sc_native_reference_cpu import _fixture


def helper():
    path = Path(__file__).with_name("sc750_static_control.py")
    spec = importlib.util.spec_from_file_location("sc750_static_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_BSE_band_weights_do_not_assume_central_2v2c_overlap():
    path = Path(__file__).parents[1]/"cri3_full_microscopic"/"native_bse_probe.py"
    spec = importlib.util.spec_from_file_location("native_bse_weights", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    power = np.zeros((2, 4, 4))
    power[0, 0, 3] = 7.
    power[1, 3, 0] = 2.
    report = mod.band_pair_summary(power, nocc=130)
    assert report["conduction_bands_zero_based"] == [130, 131, 132, 133]
    assert report["valence_bands_zero_based"] == [126, 127, 128, 129]
    assert report["central_2v2c_weight"] == [1., 0.]
    np.testing.assert_allclose(np.asarray(report["normalized_pair_weights"]).sum((1, 2)), 1.)
    with pytest.raises(ValueError, match="nonzero"):
        mod.band_pair_summary(np.zeros((1, 2, 2)), nocc=130)


def test_static_source_grid_and_reference_join_refuse_before_synthesis():
    source = dict(mesh=None, meta=NS(kgrid=(8, 8, 1), qgrid=(4, 4, 1)))
    with pytest.raises(ValueError, match="qgrid=kgrid"):
        helper().build_static_control(source, {})
    source.update(meta=NS(kgrid=(8, 8, 1), qgrid=(8, 8, 1)), frame=dict(frame="wrong"))
    with pytest.raises(ValueError, match="identities differ"):
        helper().build_static_control(source, dict(receipts=dict(frame="accepted")))


def test_static_W_exchange_fullcoherent_head_once_bareHartree_unchanged():
    from ffi import fft
    from common.units import RYD_TO_EV
    from tdagw.tdgw_io import HA_IN_EV
    from tdagw import snapshot_runtime as sr, propagate as tp
    from symmetry_maps import QirrOperator
    prepared, psi, V, rng = _fixture()
    mesh, active = prepared["mesh"], slice(1, 3)
    prepared["qgrid"] = prepared["kgrid"]
    req = np.broadcast_to(np.diag([1., 0.]), (4, 2, 2)).astype(complex).copy()
    reqd = jax.device_put(req, NamedSharding(mesh, P(("x", "y"), None, None)))
    a = rng.normal(size=(1, 4, 2, 2))+1j*rng.normal(size=(1, 4, 2, 2))
    delta = (a+a.conj().swapaxes(-1, -2))*.03
    rho = tp.place_window_blocks(req[None]+delta, mesh, 4)
    body = .36*V
    W0 = QirrOperator.whole_zone(jax.device_put(body, NamedSharding(mesh, P(None, "x", "y"))))
    g0, volume, whead = np.array([.1, .2j, -.3, .25]), 13., .07
    head = NS(vc0=.23, wcoul0=whead)
    with fft.xla_reference():
        bare = sr.native_hf_delta_model(prepared, active, reqd,
            bare_head_ha=.013, hartree_q0=prepared["V_q"][0])
        reference = dict(prepared=prepared, model=bare)
        result = helper().static_control_from_body(reference, W0, head,
            packed_G0=g0, cell_volume=volume)
        model = result["model"]
        run = jax.jit(lambda p, r: model.apply(p, r, 0.))
        actual = np.asarray(run(model.params, rho))
        zero = np.asarray(run(model.params, tp.place_window_blocks(req[None], mesh, 4)))
    # Independent ordered finite-k sum: Hartree sees BARE V0, exchange sees
    # screened body + wcoul0 head. No vc0+wcoul0 or bare-head scalar survives.
    p = psi[:, active]
    charge = np.einsum("kasu,tkab,kbsu->tu", p, delta, p.conj())
    hartree = np.einsum("kasu,uv,tv,kbsu->tkab", p.conj(), V[0], charge, p)/4
    W = body.copy()
    W[0] += whead/volume*np.outer(g0.conj(), g0)
    exchange = np.zeros_like(hartree)
    head_x = np.zeros_like(hartree)
    for k in range(4):
        for kp in range(4):
            exchange[:, k] += np.einsum("asu,csu,uv,tcd,drv,brv->tab",
                p[k].conj(), p[kp], W[(k-kp) % 4], delta[:, kp],
                p[kp].conj(), p[k], optimize=True)/4
        head_x[:, k] = np.einsum("asu,csu,uv,tcd,drv,brv->tab",
            p[k].conj(), p[k], np.outer(g0.conj(), g0), delta[:, k],
            p[k].conj(), p[k], optimize=True)/4/volume
    expected = (hartree-exchange)*(RYD_TO_EV/HA_IN_EV)
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=3e-13)
    assert np.linalg.norm(head.vc0*head_x*(RYD_TO_EV/HA_IN_EV)) > 1e-10
    assert np.linalg.norm(.013*delta) > 1e-4
    np.testing.assert_array_equal(zero, np.zeros_like(zero))
    np.testing.assert_allclose(actual, actual.conj().swapaxes(-1, -2), atol=3e-13)
    assert model.params[3] is bare.params[3] and model.params[-1] == 0.
    assert result["receipt"]["arm"] == "A_STATIC_SEX_HARTREE_NO_COLLISION"
    assert result["receipt"]["material_admission"] == "NOT_RUN"
    with pytest.raises(ValueError, match="qgrid=kgrid"):
        helper().static_control_from_body(dict(reference, prepared=dict(prepared, qgrid=(2, 1, 1))),
            W0, head, packed_G0=g0, cell_volume=volume)
    from dataclasses import replace
    permuted = replace(W0, full_rows=np.roll(np.arange(4), 1))
    # is_whole_zone checks irr_idx/sym_idx, not representative full_rows.
    assert permuted.is_whole_zone()
    with pytest.raises(ValueError, match="canonical full-q rows"):
        helper().static_control_from_body(reference, permuted, head,
            packed_G0=g0, cell_volume=volume)
