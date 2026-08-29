"""Basis and convergence gates for the opt-in fixed-Sigma eqp2 ladder."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from common.collectives import single_device_mesh
from common.units import RYD_TO_EV
from gw.sc_iteration import (
    _place,
    _rotate_fixed_sigma_cube_to_qp,
    run_fixed_sigma_evsc,
)
from gw.sigma_dispatch import SigmaResult
from gw.wavefunction_bundle import BandSlices


def _config(*, tol_ev=1.0e-9, max_iter=30, accelerator="linear"):
    return SimpleNamespace(
        eqp2=SimpleNamespace(
            tol_ev=tol_ev, max_iter=max_iter, accelerator=accelerator,
            history_depth=5),
        sc=SimpleNamespace(eigh="native"),
        memory=SimpleNamespace(per_device_gb=4.0),
    )


def _band_context(e_dft_ry, *, n_occ=1):
    """Minimal canonical active-window metadata for the focused map tests."""
    nb = int(np.asarray(e_dft_ry).shape[1])
    return dict(
        meta=SimpleNamespace(nelec=int(n_occ)),
        band_slices=BandSlices.from_band_edges(
            0, 0, int(n_occ), nb, nb),
        wfn=SimpleNamespace(energies=np.asarray([e_dft_ry])),
    )


def _unitary(rng, nk, nb):
    out = []
    for _ in range(nk):
        q, r = np.linalg.qr(
            rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb)))
        phase = np.diag(r)
        q *= np.where(np.abs(phase), phase / np.abs(phase), 1.0)[None, :]
        out.append(q)
    return np.asarray(out)


def _rotate_host(cube, U):
    return np.einsum(
        "kpm,wkpq,kqn->wkmn", np.conj(U), cube, U, optimize=True)


def test_full_sigma_cube_rotation_is_u_dagger_sigma_u():
    rng = np.random.default_rng(42)
    nw, nk, nb = 4, 2, 3
    cube = (rng.normal(size=(nw, nk, nb, nb))
            + 1j * rng.normal(size=(nw, nk, nb, nb)))
    U = _unitary(rng, nk, nb)
    mesh = single_device_mesh()
    got = _rotate_fixed_sigma_cube_to_qp(
        jnp.asarray(cube), _place(U, mesh, None), mesh=mesh)
    np.testing.assert_allclose(np.asarray(got), _rotate_host(cube, U),
                               rtol=2e-13, atol=2e-13)


@pytest.mark.mesh(4)
def test_full_sigma_cube_rotation_stays_two_axis_sharded_on_p4():
    if len(jax.devices()) < 4:
        pytest.skip("requires four visible devices")
    rng = np.random.default_rng(7)
    nw, nk, nb = 3, 2, 4
    cube = (rng.normal(size=(nw, nk, nb, nb))
            + 1j * rng.normal(size=(nw, nk, nb, nb)))
    U = _unitary(rng, nk, nb)
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2),
                axis_names=("x", "y"))
    got = _rotate_fixed_sigma_cube_to_qp(
        _place(cube, mesh, P(None, None, "x", "y")),
        _place(U, mesh, P(None, "x", "y")), mesh=mesh)
    assert got.sharding.spec == P(None, None, "x", "y")
    np.testing.assert_allclose(np.asarray(got), _rotate_host(cube, U),
                               rtol=3e-13, atol=3e-13)


def _host_reference(kin, vh, sx, sigma_w, omega_ev, e_dft_ry,
                    *, tol_ev, max_iter):
    nk, nb = e_dft_ry.shape
    U = np.broadcast_to(np.eye(nb, dtype=complex), (nk, nb, nb)).copy()
    prev = np.asarray(e_dft_ry, dtype=float)
    h0 = kin + vh
    pending_final_check = False
    for iteration in range(1, max_iter + 1):
        h0_q = np.einsum(
            "kpm,kpq,kqn->kmn", np.conj(U), h0, U, optimize=True)
        sx_q = np.einsum(
            "kpm,kpq,kqn->kmn", np.conj(U), sx, U, optimize=True)
        sc_q = _rotate_host(sigma_w, U)
        M = np.empty_like(sx_q)
        for k in range(nk):
            for m in range(nb):
                for n in range(nb):
                    at_m = np.interp(
                        prev[k, m] * RYD_TO_EV, omega_ev,
                        sc_q[:, k, m, n].real)
                    at_m += 1j * np.interp(
                        prev[k, m] * RYD_TO_EV, omega_ev,
                        sc_q[:, k, m, n].imag)
                    at_n = np.interp(
                        prev[k, n] * RYD_TO_EV, omega_ev,
                        sc_q[:, k, m, n].real)
                    at_n += 1j * np.interp(
                        prev[k, n] * RYD_TO_EV, omega_ev,
                        sc_q[:, k, m, n].imag)
                    M[k, m, n] = 0.5 * (at_m + at_n) + sx_q[k, m, n]
        M = 0.5 * (M + np.swapaxes(M.conj(), -1, -2))
        new, V = np.linalg.eigh(h0_q + M)
        residual = np.max(np.abs(new - prev)) * RYD_TO_EV
        U = U @ V
        if residual <= tol_ev:
            if pending_final_check:
                return new, U, iteration, residual
            pending_final_check = True
        else:
            pending_final_check = False
        prev = new
    raise AssertionError("host reference did not converge")


def test_fixed_sigma_evsc_matches_independent_full_matrix_iteration():
    # A weak, frequency-dependent two-band model with a genuinely complex
    # off-diagonal.  Linear omega dependence makes the interpolation exact,
    # leaving only the basis convention and iteration map under test.
    omega_ev = np.linspace(-4.0, 4.0, 17)
    e_dft_ev = np.array([[-1.2, 1.1], [-1.0, 1.3]])
    e_dft_ry = e_dft_ev / RYD_TO_EV
    nk, nb = e_dft_ev.shape
    kin = np.zeros((nk, nb, nb), dtype=complex)
    kin[:, np.arange(nb), np.arange(nb)] = e_dft_ry
    kin[:, 0, 1] = np.array([0.025 + 0.012j, -0.018 + 0.009j]) / RYD_TO_EV
    kin[:, 1, 0] = np.conj(kin[:, 0, 1])
    vh = np.zeros_like(kin)
    sx = np.zeros_like(kin)
    sx[:, 0, 0] = -0.18 / RYD_TO_EV
    sx[:, 1, 1] = -0.07 / RYD_TO_EV
    sx[:, 0, 1] = (0.022 + 0.011j) / RYD_TO_EV
    sx[:, 1, 0] = np.conj(sx[:, 0, 1])
    sigma_w = np.empty((omega_ev.size, nk, nb, nb), dtype=complex)
    for iw, omega in enumerate(omega_ev):
        sigma_w[iw, :, 0, 0] = (0.09 + 0.025 * omega) / RYD_TO_EV
        sigma_w[iw, :, 1, 1] = (-0.04 + 0.018 * omega) / RYD_TO_EV
        sigma_w[iw, :, 0, 1] = (
            0.035 + 0.006 * omega + 1j * (0.014 - 0.002 * omega)
        ) / RYD_TO_EV
        sigma_w[iw, :, 1, 0] = np.conj(sigma_w[iw, :, 0, 1])

    result = SigmaResult(
        v_h_kij_ry=jnp.asarray(vh),
        sigma_x_kij_ry=jnp.asarray(sx),
        sigma_xc_kij_ry=jnp.zeros_like(jnp.asarray(sx)),
        sigma_c_omega_kij_ry=jnp.asarray(sigma_w),
        omega_grid_ev=omega_ev,
        omega_grid_ry=omega_ev / RYD_TO_EV,
        efermi_dft_ev=0.0,
    )
    cfg = _config(tol_ev=1.0e-9, max_iter=30)
    got = run_fixed_sigma_evsc(
        result, jnp.asarray(kin), e_dft_ry,
        config=cfg, **_band_context(e_dft_ry),
        mesh_xy=single_device_mesh(), print_fn=lambda *a: None)
    ref_e, ref_u, ref_n, ref_resid = _host_reference(
        kin, vh, sx, sigma_w, omega_ev, e_dft_ry,
        tol_ev=cfg.eqp2.tol_ev, max_iter=cfg.eqp2.max_iter)
    np.testing.assert_allclose(got.energies_ry, ref_e, rtol=3e-12, atol=3e-12)
    # Compare columns modulo arbitrary eigenvector phases.
    got_u = np.asarray(got.U_dft_to_qp)
    for k in range(nk):
        overlap = np.abs(ref_u[k].conj().T @ got_u[k])
        np.testing.assert_allclose(overlap, np.eye(nb),
                                   rtol=3e-10, atol=3e-10)
    assert got.iterations == ref_n
    assert got.residual_ev == pytest.approx(ref_resid, rel=2e-9, abs=1e-12)


def test_rcrop_fixed_hamiltonian_map_reaches_same_solution():
    omega_ev = np.linspace(-3.0, 3.0, 13)
    e_dft_ry = np.array([[-0.8, 0.9]]) / RYD_TO_EV
    kin = np.diag(e_dft_ry[0])[None].astype(complex)
    sx = np.array([[[-0.08, 0.025], [0.025, -0.03]]]) / RYD_TO_EV
    sigma_w = np.empty((omega_ev.size, 1, 2, 2), dtype=complex)
    for iw, omega in enumerate(omega_ev):
        sigma_w[iw, 0] = np.array([
            [0.04 + 0.012 * omega, 0.018 + 0.003 * omega],
            [0.018 + 0.003 * omega, -0.02 + 0.009 * omega],
        ]) / RYD_TO_EV
    z = jnp.zeros_like(jnp.asarray(kin))
    result = SigmaResult(
        v_h_kij_ry=z,
        sigma_x_kij_ry=jnp.asarray(sx),
        sigma_xc_kij_ry=z,
        sigma_c_omega_kij_ry=jnp.asarray(sigma_w),
        omega_grid_ev=omega_ev,
        omega_grid_ry=omega_ev / RYD_TO_EV,
        efermi_dft_ev=0.0,
    )
    linear = run_fixed_sigma_evsc(
        result, jnp.asarray(kin), e_dft_ry,
        config=_config(tol_ev=1e-8, accelerator="linear"),
        **_band_context(e_dft_ry),
        mesh_xy=single_device_mesh(), print_fn=lambda *a: None)
    rcrop = run_fixed_sigma_evsc(
        result, jnp.asarray(kin), e_dft_ry,
        config=_config(tol_ev=1e-8, accelerator="rcrop"),
        **_band_context(e_dft_ry),
        mesh_xy=single_device_mesh(), print_fn=lambda *a: None)
    np.testing.assert_allclose(rcrop.energies_ry, linear.energies_ry,
                               rtol=2e-9, atol=2e-9)
    assert rcrop.residual_ev <= 1e-8


def test_eqp2_reapplies_semicore_and_conduction_scissors_on_final_map():
    omega_ev = np.linspace(-2.0, 2.0, 9)
    e_dft_ev = np.array([[-8.0, -1.0, 1.0, 8.0]])
    e_dft_ry = e_dft_ev / RYD_TO_EV
    kin = np.diag(e_dft_ry[0])[None].astype(complex)
    sigma_x = np.zeros((1, 4, 4), dtype=complex)
    sigma_x[0, 1, 1] = -0.2 / RYD_TO_EV
    sigma_x[0, 2, 2] = +0.3 / RYD_TO_EV
    zmat = jnp.zeros((1, 4, 4), dtype=jnp.complex128)
    result = SigmaResult(
        v_h_kij_ry=zmat,
        sigma_x_kij_ry=jnp.asarray(sigma_x),
        sigma_xc_kij_ry=zmat,
        sigma_c_omega_kij_ry=jnp.zeros(
            (omega_ev.size, 1, 4, 4), dtype=jnp.complex128),
        omega_grid_ev=omega_ev,
        omega_grid_ry=omega_ev / RYD_TO_EV,
        efermi_dft_ev=0.0,
    )
    log = []
    got = run_fixed_sigma_evsc(
        result, jnp.asarray(kin), e_dft_ry,
        config=_config(tol_ev=1e-10, accelerator="linear"),
        **_band_context(e_dft_ry, n_occ=2),
        mesh_xy=single_device_mesh(), print_fn=log.append)

    # Protected states set the no-lag midgap to +0.05 eV.  The semicore
    # keeps E-E_F, while the one-point conduction fit is the +0.3 eV rigid
    # shift inferred from the protected conduction state.
    np.testing.assert_allclose(
        got.energies_ry * RYD_TO_EV,
        np.array([[-7.95, -1.2, 1.3, 8.3]]), atol=2e-10, rtol=0.0)
    assert got.iterations >= 3
    assert any("final post-rotation verification" in line for line in log)
    assert any("optional out-of-grid" in line for line in log)


def test_fixed_sigma_evsc_refuses_uncovered_energy_before_clamping():
    omega_ev = np.linspace(-1.0, 1.0, 5)
    # Both DFT states are protected initially.  The first map pushes one
    # outside the table, so the *next* evaluation must refuse rather than
    # reclassify that protected state as an optional scissored tail.
    e_dft_ry = np.array([[-0.2, 0.2]]) / RYD_TO_EV
    zmat = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    kin = jnp.asarray(np.diag(e_dft_ry[0])[None].astype(complex))
    sigma_x = np.zeros((1, 2, 2), dtype=complex)
    sigma_x[0, 0, 0] = -2.0 / RYD_TO_EV
    result = SigmaResult(
        v_h_kij_ry=zmat,
        sigma_x_kij_ry=jnp.asarray(sigma_x),
        sigma_xc_kij_ry=zmat,
        sigma_c_omega_kij_ry=jnp.zeros((5, 1, 2, 2), dtype=jnp.complex128),
        omega_grid_ev=omega_ev,
        omega_grid_ry=omega_ev / RYD_TO_EV,
        efermi_dft_ev=0.0,
    )
    with pytest.raises(ValueError, match="eqp2_omega_coverage"):
        run_fixed_sigma_evsc(
            result, kin, e_dft_ry, config=_config(),
            **_band_context(e_dft_ry),
            mesh_xy=single_device_mesh(), print_fn=lambda *a: None)
