"""
Toy-system validation for scalar-ω CTSP frequency integration.

Checks:
1) Outside crossing region (ω < E_gap or ω > E_bw): matches exact denominator sum.
2) Inside crossing region (E_gap <= ω <= E_bw): matches HGL-regularized reference.
"""

import numpy as np
from scipy.integrate import quad

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from common.meta import Meta
from gw_isdf.get_windows import WindowInfo, WindowPair
from gw_isdf.w_isdf_dynamic import frequency_integration


jax.config.update("jax_enable_x64", True)


def _build_toy_problem():
    E_v = np.array([[-0.6, -0.3]], dtype=np.float64)
    E_c = np.array([[0.2, 0.8]], dtype=np.float64)

    # Scalar (nmu=1, ns=1) wavefunction coefficients for controlled reference checks.
    psi_vTX = np.array([[[[1.2, 0.7]]]], dtype=np.float64)   # (nk, ns, nmu, nv)
    psi_vY = np.array([[[[0.9]], [[1.1]]]], dtype=np.float64)  # (nk, nv, ns, nmu)
    psi_cX = np.array([[[[1.3]], [[0.8]]]], dtype=np.float64)  # (nk, nc, ns, nmu)
    psi_cTY = np.array([[[[1.0, 0.6]]]], dtype=np.float64)     # (nk, ns, nmu, nc)

    val_win = WindowInfo(float(E_v.min()), float(E_v.max()), index=0, count=1)
    cond_win = WindowInfo(float(E_c.min()), float(E_c.max()), index=0, count=1)
    win = WindowPair(val_win, cond_win, epsq=1e-8)

    meta = Meta(
        rank=0,
        n_proc=1,
        b_id_0=0,
        b_id_1=0,
        b_id_2=0,
        b_id_3=0,
        b_id_4=0,
        fft_grid=(1, 1, 1),
        n_rtot=1,
        n_rmu=1,
        npol=1,
        nfreq=1,
        nspin=1,
        nspinor=1,
        nspinor_wfnfile=1,
        nkx=1,
        nky=1,
        nkz=1,
        nk_tot=1,
        nbnd_jax=1,
        n_rtot_jax=1,
        n_rmu_jax=1,
    )
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    return E_v, E_c, psi_vTX, psi_vY, psi_cX, psi_cTY, [win], meta, mesh


def _reference_values(E_v, E_c, psi_vTX, psi_vY, psi_cX, psi_cTY, win, omega):
    A_v = np.conj(psi_vTX[0, 0, 0, :]) * psi_vY[0, :, 0, 0]
    A_c = np.conj(psi_cTY[0, 0, 0, :]) * psi_cX[0, :, 0, 0]

    E_gap = win.cond_window.start_energy - win.val_window.end_energy
    E_bw = win.cond_window.end_energy - win.val_window.start_energy
    gamma = win.z_lm

    def F(y):
        val, _ = quad(lambda t: np.sin(y * t) * np.exp(-t - 0.5 * t * t), 0, 120, limit=300)
        return val

    exact = 0.0
    reg = 0.0
    for ic, Ec in enumerate(E_c[0]):
        for iv, Ev in enumerate(E_v[0]):
            d = Ec - Ev
            anti = -1.0 / (omega + d)
            reson_exact = 1.0 / (omega - d)
            reson_reg = gamma * F(gamma * (omega - d)) if (E_gap <= omega <= E_bw) else reson_exact
            pref = A_c[ic] * A_v[iv]
            exact += pref * (reson_exact + anti)
            reg += pref * (reson_reg + anti)
    return float(np.real(exact)), float(np.real(reg)), E_gap, E_bw


def test_frequency_integration_toy():
    E_v, E_c, psi_vTX, psi_vY, psi_cX, psi_cTY, windows, meta, mesh = _build_toy_problem()
    win = windows[0]

    psi_vTX_j = jnp.asarray(psi_vTX)
    psi_vY_j = jnp.asarray(psi_vY)
    psi_cX_j = jnp.asarray(psi_cX)
    psi_cTY_j = jnp.asarray(psi_cTY)
    E_v_j = jnp.asarray(E_v)
    E_c_j = jnp.asarray(E_c)

    test_omegas = [0.0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.6]
    for omega in test_omegas:
        chi = frequency_integration(
            psi_vTX_j,
            psi_vY_j,
            psi_cX_j,
            psi_cTY_j,
            E_v_j,
            E_c_j,
            windows,
            omega,
            meta,
            mesh,
        )
        chi_code = float(np.asarray(chi)[0, 0, 0, 0, 0, 0, 0].real)
        chi_exact, chi_reg, E_gap, E_bw = _reference_values(E_v, E_c, psi_vTX, psi_vY, psi_cX, psi_cTY, win, omega)

        if E_gap <= omega <= E_bw:
            rel = abs(chi_code - chi_reg) / max(1e-12, abs(chi_reg))
            assert rel < 1e-5, f"crossing ω={omega}: code={chi_code}, ref={chi_reg}, rel={rel}"
        else:
            rel = abs(chi_code - chi_exact) / max(1e-12, abs(chi_exact))
            assert rel < 1e-5, f"non-crossing ω={omega}: code={chi_code}, exact={chi_exact}, rel={rel}"


if __name__ == "__main__":
    test_frequency_integration_toy()
    print("Toy frequency integration test passed.")
