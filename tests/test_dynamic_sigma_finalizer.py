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

    coverage = SimpleNamespace(
        mask_kn=np.ones((1, 2), dtype=bool), n_uncovered=0,
        fraction_uncovered=0.0, omega_min_ev=-1.0, omega_max_ev=1.0,
        policy="clamp")

    def evaluate(got_sigma, **kwargs):
        calls.append("interp")
        assert got_sigma is post_head
        return (np.full((1, 2), 4.0), np.full((1, 2), 6.0), 7.0,
                "fixed-N mu", coverage)

    def write(got_sigma, **kwargs):
        calls.append("write")
        assert got_sigma is post_head
        # THE FINALIZE'S OWN ANSWER REACHES THE WRITER (audit A2): the
        # stamp is not re-derived at the write site, so the file records
        # the reference the interpolation above actually used.
        assert kwargs["omega_reference_ev"] == 7.0
        assert kwargs["omega_reference_provenance"] == "fixed-N mu"
        # AND THE SECOND REFERENCE, added 2026-08-22: where this Sigma was
        # EVALUATED.  ``e_qp_ev`` here is [[2, 3]] while E_DFT - E_F is
        # [[6, 6]], so the two are NOT equal and the writer must be told
        # ``self_consistent_qp`` -- a fact the finalize MEASURES with an
        # array comparison rather than inferring from a config key.
        np.testing.assert_array_equal(
            kwargs["eval_energies_rel_ev"], np.asarray([[2.0, 3.0]]) - 7.0)
        assert kwargs["eval_energies_provenance"] == "self_consistent_qp"
        assert kwargs["omega_coverage"] is coverage
        return "/tmp/sigma_mnk.h5"

    def build(got_sigma, got_x, omega, energy, got_mesh):
        calls.append("qsgw")
        assert got_sigma is post_head and got_mesh is mesh
        np.testing.assert_array_equal(omega, [-1.0, 1.0])
        # The QSGW ansatz is evaluated at e_qp_ev on the finalize's OWN
        # reference — one subtraction, not a second opinion about the zero.
        np.testing.assert_array_equal(energy, np.asarray([[2.0, 3.0]]) - 7.0)
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
        sig_x=sig_x, sig_h=sig_h, e_qp_ev=np.asarray([[2.0, 3.0]]),
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
    assert result.omega_reference_provenance == "fixed-N mu"
    # The energies this Σ was EVALUATED at are carried out on the result,
    # absolute eV, because that is where eqp1 has to be linearized and the
    # writer cannot re-derive them (``eqp_bgw.compute_eqp_diag``).
    np.testing.assert_array_equal(result.e_eval_ev, [[2.0, 3.0]])
    # And the coverage of the at-DFT interpolation rides out with it, so a
    # writer downstream can say which of its cells are measurements.
    assert result.omega_coverage is coverage


def test_finalizer_adds_faraday_once_and_books_it_to_ct(monkeypatch):
    """The Hall-only cube reaches total Sigma, ``sigCT_hall``, and CT once."""
    import gw.dynamic_sigma as dynamic
    import gw.qsgw_utils as qsgw
    import gw.sigma_dispatch as dispatch

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    body = jnp.zeros((2, 1, 2, 2), dtype=jnp.complex128)
    hall = jnp.full_like(body, 2.0)
    sig_x = jnp.ones((1, 2, 2), dtype=jnp.complex128)
    static_lorentz = jnp.stack((
        jnp.ones_like(sig_x), 2.0 * jnp.ones_like(sig_x),
        4.0 * jnp.ones_like(sig_x)))
    coverage = SimpleNamespace(name="covered")
    eval_maxima = []
    qsgw_x_maxima = []

    monkeypatch.setattr(
        dynamic, "add_head_sigma_diag", lambda sigma, _head: sigma)

    def evaluate(sigma, **_kwargs):
        maximum = float(np.max(np.abs(np.asarray(sigma))))
        eval_maxima.append(maximum)
        return (np.full((1, 2), maximum), np.zeros((1, 2)), 0.0,
                "midgap", coverage)

    def build(_sigma, sigma_x, *_args, **_kwargs):
        x_max = float(np.max(np.abs(np.asarray(sigma_x))))
        qsgw_x_maxima.append(x_max)
        value = 10.0 if x_max else 3.0
        return (jnp.full_like(sig_x, value),
                {"n_clipped": 0, "frac_clipped": 0.0})

    monkeypatch.setattr(dynamic, "eval_sigma_c_at_dft_energies", evaluate)
    monkeypatch.setattr(qsgw, "build_qsgw_sigma_xc", build)
    monkeypatch.setattr(
        dispatch, "device_put_process_local", lambda value, _sharding: value)

    faraday = SimpleNamespace(
        omega_h_ry=1.25,
        sigma_H_static=np.array([0.0, 0.0, 4.0e-8]),
        sigma_H_probe=np.array([0.0, 0.0, 1.5e-8]))
    config = SimpleNamespace(
        omega_grid_ev=np.asarray([-1.0, 1.0]),
        omega_grid_ry=np.asarray([-0.1, 0.1]), sc=None)
    result = dispatch.finalize_dynamic_sigma(
        body, None, sig_x=sig_x, sig_h=2.0 * sig_x,
        e_qp_ev=np.zeros((1, 2)), config=config, meta=object(),
        mesh_xy=mesh, sym=object(), wfn=SimpleNamespace(efermi=0.0),
        band_slices=object(), input_dir="/tmp", write_sigma_omega_h5=False,
        sigma_lorentz_static_skij_ry=static_lorentz,
        sigma_ct_hall_body_omega=hall, faraday_ppm=faraday,
        efermi_ry=0.0, efermi_provenance="midgap",
        print_fn=lambda *_: None)

    # Total and Hall-only interpolation see the same value because body=0;
    # there is no second addition in either path.
    assert eval_maxima == [2.0, 2.0]
    assert qsgw_x_maxima == [1.0, 0.0]
    np.testing.assert_array_equal(result.sigma_ct_hall_at_dft_diag_ev, 2.0)
    np.testing.assert_array_equal(np.asarray(result.sigma_c_omega_kij_ry), 2.0)
    np.testing.assert_array_equal(
        np.asarray(result.sigma_lorentz_skij_ry[:, 0]),
        np.stack((np.ones((2, 2)), 5.0 * np.ones((2, 2)),
                  4.0 * np.ones((2, 2)))))
    assert result.faraday_head_omega_h_ry == 1.25
    assert result.faraday_sigma_h_probe_bohr_inv == (0.0, 0.0, 1.5e-8)


def test_the_sc_driver_does_not_overwrite_the_finalize_omega_reference():
    """The reference that reaches the eqp writer is the finalize's own.

    ``run_sc_driver`` hands the driver's post-Σ seam a rebased copy of the
    last iteration's ``SigmaResult``, and ``gw_jax`` puts its
    ``efermi_dft_ev`` straight into ``GWResults.efermi_ev``, which is the
    single number ``gw_output.write_results`` forms ``e_dft_rel_ev`` (and
    the eqp1 centre) from.  That field used to be re-filled here with the
    loader's ``wfn.efermi`` — mid-gap ½(VBM+CBM) of the DFT spectrum —
    unconditionally, so a metallic run's eqp0/eqp1.dat sampled Σ_c(ω)
    2.932 eV away from the reference its own grid was built with on the
    sodium 8×8×8 deck (measured: those files reassemble from disk to
    3.6e-4 eV at wfn.efermi and to 3.7 eV at the loop's fixed-N μ).

    AST, not a run: the defect is one keyword in one call, it is invisible
    to every shape/finiteness gate, and reproducing it end to end costs a
    self-consistent QSGW.  Static modes never fill the field, which is
    what the unconditional re-fill was for — so the ``else`` arm may
    still be ``wfn.efermi``; what may not come back is dropping the test.
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "gw" / "sc_iteration.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "run_sc_driver")
    kw = [k for call in ast.walk(fn)
          if isinstance(call, ast.Call) for k in call.keywords
          if k.arg == "efermi_dft_ev"]
    assert len(kw) == 1, f"expected exactly one efermi_dft_ev=, got {len(kw)}"
    value = kw[0].value
    assert isinstance(value, ast.IfExp), (
        "run_sc_driver must keep the dynamic finalize's own omega "
        "reference and fall back to wfn.efermi only where there is none; "
        f"got an unconditional {ast.dump(value)[:90]}")
    assert (isinstance(value.body, ast.Attribute)
            and value.body.attr == "efermi_dft_ev"), ast.dump(value.body)[:90]
