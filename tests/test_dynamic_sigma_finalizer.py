"""Contract gates for the ansatz-neutral dynamic-Sigma finalizer."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh


def test_head_diagonal_is_added_once_and_only_on_the_diagonal():
    from gw.dynamic_sigma import add_head_sigma_diag

    body = jnp.asarray(
        np.arange(2 * 1 * 3 * 3).reshape(2, 1, 3, 3),
        dtype=jnp.complex128)
    head = np.asarray([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    got = np.asarray(add_head_sigma_diag(body, head))
    want = np.asarray(body).copy()
    i = np.arange(3)
    want[:, :, i, i] += head
    np.testing.assert_array_equal(got, want)


def test_head_diagonal_refuses_a_body_mismatch():
    from gw.dynamic_sigma import add_head_sigma_diag

    with pytest.raises(ValueError, match="head shape"):
        add_head_sigma_diag(
            jnp.zeros((2, 1, 3, 3), dtype=jnp.complex128),
            np.zeros((2, 1, 2)),
        )


def test_finalizer_owns_the_dynamic_tail_and_returns_sigma_result(monkeypatch):
    import gw.dynamic_sigma as dynamic
    import gw.qsgw_utils as qsgw
    import gw.sigma_dispatch as dispatch

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    body = jnp.zeros((2, 1, 2, 2), dtype=jnp.complex128)
    head = np.ones((2, 1, 2), dtype=np.complex128)
    post_head = body + jnp.eye(2)[None, None]
    sig_x = jnp.ones((1, 2, 2), dtype=jnp.complex128)
    sig_h = 2 * sig_x
    qsgw_xc = 3 * sig_x
    calls = []

    def add(got_body, got_head):
        calls.append("head")
        assert got_body is body and got_head is head
        return post_head

    def evaluate(got_sigma, **kwargs):
        calls.append("interp")
        assert got_sigma is post_head
        return (np.full((1, 2), 4.0), np.full((1, 2), 6.0), 7.0)

    def write(got_sigma, **kwargs):
        calls.append("write")
        assert got_sigma is post_head
        return "/tmp/sigma_mnk.h5"

    def build(got_sigma, got_x, omega, energy, got_mesh):
        calls.append("qsgw")
        assert got_sigma is post_head and got_mesh is mesh
        np.testing.assert_array_equal(omega, [-1.0, 1.0])
        return qsgw_xc, {"n_clipped": 0, "frac_clipped": 0.0}

    def append(path, cube, **kwargs):
        calls.append("append")
        assert path == "/tmp/sigma_mnk.h5" and cube is qsgw_xc

    monkeypatch.setattr(dynamic, "add_head_sigma_diag", add)
    monkeypatch.setattr(dynamic, "eval_sigma_c_at_dft_energies", evaluate)
    monkeypatch.setattr(dynamic, "write_sigma_omega", write)
    monkeypatch.setattr(qsgw, "build_qsgw_sigma_xc", build)
    monkeypatch.setattr(qsgw, "write_qsgw_sigma_cube", append)
    monkeypatch.setattr(
        dispatch, "device_put_process_local", lambda value, _sharding: value)

    config = SimpleNamespace(
        omega_grid_ev=np.asarray([-1.0, 1.0]),
        omega_grid_ry=np.asarray([-0.1, 0.1]),
    )
    result = dispatch.finalize_dynamic_sigma(
        body, head,
        sig_x=sig_x, sig_h=sig_h, e_qp_ev=np.zeros((1, 2)),
        config=config, meta=object(), mesh_xy=mesh, sym=object(),
        wfn=SimpleNamespace(efermi=0.0), band_slices=object(),
        input_dir="/tmp", print_fn=lambda *_: None,
    )

    assert calls == ["head", "interp", "write", "qsgw", "append"]
    assert result.sigma_c_omega_kij_ry is post_head
    assert result.sigma_xc_kij_ry is qsgw_xc
    assert result.sigma_omega_h5_path == "/tmp/sigma_mnk.h5"
    np.testing.assert_array_equal(result.sigma_c_at_dft_diag_ev, 4.0)
    np.testing.assert_array_equal(result.omega_dft_rel_ev, 6.0)
    assert result.efermi_dft_ev == 7.0
