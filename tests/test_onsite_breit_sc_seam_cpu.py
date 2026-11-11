"""On-site Breit seam into the GW direct field (``gw.sigma_dispatch`` / ``gw.sc_iteration``), CPU 2x2.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``
(CPU clause of the four-GPU rule: layout and seam logic only).  Checked without
the GW machinery: the one combiner H_T + H_OS / H_OS keeps the slot's two-axis
sharding and refuses a wrong key pairing; the SC output seam selects
``onsite_dft`` with the other exact-Hartree tables and refuses a wrong band
extent; the one-shot seam adds the static term only when the switch is on;
``sigma_mnk.h5`` is stamped, and an unstampable file refuses.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

NK, NB = 4, 6


def _mesh():
    devs = jax.devices()
    if len(devs) < 4:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=4")
    return Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))


def _herm(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(NK, NB, NB)) + 1j * rng.normal(size=(NK, NB, NB))
    return a + np.conj(np.swapaxes(a, -1, -2))


def test_combiner_keeps_slot_sharding_and_refuses_wrong_pairing():
    """H_T + H_OS lands at H_T's P(None,'x','y'); H_OS alone at V_H's; one
    compiled program per layout; a key pairing that disagrees with the slot,
    or a wrong-shape H_OS, refuses."""
    from gw.sigma_dispatch import (_ONSITE_BREIT_INTO_TRANSVERSE,
                                   onsite_breit_into_transverse)
    mesh = _mesh()
    band = NamedSharding(mesh, P(None, "x", "y"))
    repl = NamedSharding(mesh, P(None, None, None))
    h_t_host, h_os_host, v_h_host = _herm(1), _herm(2), _herm(3)
    h_t = jax.device_put(h_t_host, band)
    h_os = jax.device_put(h_os_host, repl)
    v_h = jax.device_put(v_h_host, band)

    out = onsite_breit_into_transverse(h_t, h_os, like=h_t, double_counting="paw")
    assert out.sharding == band
    np.testing.assert_array_equal(np.asarray(out), h_t_host + h_os_host)
    alone = onsite_breit_into_transverse(None, h_os, like=v_h,
                                         double_counting="onsite_only")
    assert alone.sharding == band
    np.testing.assert_array_equal(np.asarray(alone), h_os_host)

    # An equivalent reconstructed layout reuses the compiled combiner.
    n_cached = len(_ONSITE_BREIT_INTO_TRANSVERSE)
    h_t2 = jax.device_put(h_t_host, NamedSharding(mesh, P(None, "x", "y")))
    onsite_breit_into_transverse(h_t2, h_os, like=h_t2, double_counting="paw")
    assert len(_ONSITE_BREIT_INTO_TRANSVERSE) == n_cached

    with pytest.raises(RuntimeError, match="smooth transverse field is absent"):
        onsite_breit_into_transverse(None, h_os, like=v_h, double_counting="paw")
    with pytest.raises(RuntimeError, match="smooth transverse field is present"):
        onsite_breit_into_transverse(h_t, h_os, like=h_t,
                                     double_counting="onsite_only")
    with pytest.raises(ValueError, match="share the full-BZ k rows"):
        onsite_breit_into_transverse(
            h_t, jax.device_put(h_os_host[:-1], repl), like=h_t,
            double_counting="paw")


def test_sc_output_seam_selects_onsite_with_the_exact_hartree():
    """``onsite_dft`` crosses the star-wedge seam with scalar/transverse
    (same rows), and a band extent other than the SC matrix's refuses."""
    from gw.sc_iteration import SCExactHartree, _sc_output_tables_on_loop_kset
    from gw.sigma_dispatch import SigmaResult

    rows = np.array([0, 2])
    kstar = SimpleNamespace(
        is_identity=False, nk_irr=rows.size, nk_full=NK,
        select=lambda a: jnp.take(jnp.asarray(a), jnp.asarray(rows), axis=0))
    mats = [jnp.asarray(_herm(s)) for s in range(10, 16)]
    sigma = SigmaResult(v_h_kij_ry=mats[0], sigma_x_kij_ry=mats[1],
                        sigma_xc_kij_ry=mats[2])
    record = SimpleNamespace(tag="host-side record")
    exact = SCExactHartree(scalar_dft=mats[3], transverse_dft=mats[4],
                           efermi_ry=0.1, onsite_dft=mats[5],
                           onsite_record=record)
    U = jnp.asarray(_herm(20))
    _, _, exact_loop, delta, _, _ = _sc_output_tables_on_loop_kset(
        sigma, mats[2], None, U, exact, None, kstar)
    assert exact_loop.onsite_record is record
    np.testing.assert_array_equal(np.asarray(exact_loop.onsite_dft),
                                  np.asarray(mats[5])[rows])
    np.testing.assert_array_equal(np.asarray(exact_loop.transverse_dft),
                                  np.asarray(mats[4])[rows])
    # total = V_H + H_T only: H_OS is already inside H_T, never added again.
    np.testing.assert_array_equal(
        np.asarray(exact_loop.total),
        np.asarray(mats[3])[rows] + np.asarray(mats[4])[rows])

    bad = SCExactHartree(scalar_dft=mats[3], transverse_dft=mats[4],
                         efermi_ry=0.1, onsite_dft=mats[5][:, :-1, :-1])
    with pytest.raises(ValueError, match="exact Hartree onsite_dft has band matrix"):
        _sc_output_tables_on_loop_kset(sigma, mats[2], None, U, bad, None, kstar)


def test_oneshot_seam_adds_static_term_only_when_on(monkeypatch):
    """``_static_onsite_breit``: off -> slot untouched, no provenance; on ->
    H_OS[D_DFT] built on [b0, b3) with U = f = None and added to H_T (bispinor)
    or placed alone at V_H's sharding (scalar)."""
    import gw.hartree as hartree
    from gw import sigma_dispatch as sd
    mesh = _mesh()
    band = NamedSharding(mesh, P(None, "x", "y"))
    h_t_host, h_os_host, v_h_host = _herm(4), _herm(5), _herm(6)
    h_t = jax.device_put(h_t_host, band)
    v_h = jax.device_put(v_h_host, band)
    slices = SimpleNamespace(b0=2, b3=2 + NB)
    calls = []

    def field(wfn, sym, setup, *, band_range, mesh, U=None, occupations=None,
              print_fn=None):
        calls.append((band_range, U, occupations))
        return SimpleNamespace(matrix=jax.device_put(
            h_os_host, NamedSharding(mesh, P(None, None, None))))

    monkeypatch.setattr(hartree, "onsite_breit_field", field, raising=False)
    monkeypatch.setattr(hartree, "onsite_breit_setup_for",
                        lambda *a, **k: None, raising=False)
    cfg = SimpleNamespace(bispinor=True)
    out, prov = sd._static_onsite_breit(
        cfg, None, None, slices, mesh, input_dir=".", v_h=v_h, h_transverse=h_t,
        print_fn=lambda *a, **k: None)
    assert out is h_t and prov is None and not calls

    seen = {}

    def setup_for(config, wfn, *, input_dir, tt_exchange):
        seen["tt_exchange"] = tt_exchange
        dc = "paw" if config.bispinor else "onsite_only"
        return SimpleNamespace(double_counting=dc, exchange="paw" if dc == "paw" else "full",
                               projector="ortho-atomic", provenance='{"x":1}')

    monkeypatch.setattr(hartree, "onsite_breit_setup_for", setup_for, raising=False)
    out, prov = sd._static_onsite_breit(
        cfg, None, None, slices, mesh, input_dir=".", v_h=v_h, h_transverse=h_t,
        print_fn=lambda *a, **k: None)
    assert prov == '{"x":1}' and seen["tt_exchange"] is True
    assert calls[-1] == ((2, 2 + NB), None, None)
    assert out.sharding == band
    np.testing.assert_array_equal(np.asarray(out), h_t_host + h_os_host)

    out, _ = sd._static_onsite_breit(
        SimpleNamespace(bispinor=False), None, None, slices, mesh, input_dir=".",
        v_h=v_h, h_transverse=None, print_fn=lambda *a, **k: None)
    assert seen["tt_exchange"] is False and out.sharding == band
    np.testing.assert_array_equal(np.asarray(out), h_os_host)


def test_sigma_h5_stamp_and_refusal(tmp_path):
    """The transverse slot and the aggregate carry ``onsite_breit_provenance``;
    a file with no transverse dataset refuses (the term would be invisible)."""
    import h5py
    from file_io.sigma_output import (ONSITE_BREIT_PROVENANCE_ATTR,
                                      stamp_onsite_breit_provenance_h5)
    good = tmp_path / "sigma_mnk.h5"
    with h5py.File(good, "w") as h5:
        for name in ("hartree_kij_ev", "hartree_scalar_kij_ev",
                     "hartree_transverse_kij_ev"):
            h5.create_dataset(name, data=np.zeros((1, 2, 2), np.complex128))
    stamp_onsite_breit_provenance_h5(good, '{"scheme":"lorrax.onsite_breit/v1"}')
    with h5py.File(good, "r") as h5:
        for name in ("hartree_kij_ev", "hartree_transverse_kij_ev"):
            assert "onsite_breit" in str(h5[name].attrs[ONSITE_BREIT_PROVENANCE_ATTR])
        assert ONSITE_BREIT_PROVENANCE_ATTR not in h5["hartree_scalar_kij_ev"].attrs

    bare = tmp_path / "charge_only.h5"
    with h5py.File(bare, "w") as h5:
        h5.create_dataset("hartree_kij_ev", data=np.zeros((1, 2, 2), np.complex128))
    with pytest.raises(ValueError, match="no hartree_transverse_kij_ev"):
        stamp_onsite_breit_provenance_h5(bare, '{"x":1}')
