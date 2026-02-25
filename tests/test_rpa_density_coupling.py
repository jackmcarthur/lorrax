import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

# Ensure local package import works under pytest.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

jax.config.update("jax_enable_x64", True)


def _make_1x1_mesh() -> Mesh:
    devices = jax.devices("cpu")
    return Mesh(np.array([devices[0]]).reshape(1, 1), axis_names=("x", "y"))


def test_rpa_v_term_density_couples_k_matches_explicit():
    """RPA/Hartree V contraction should couple k when v_couples_k=True."""
    from isdf.bse_isdf.bse_ring_comm import build_bse_ring_matvec_full, make_bse_shardings

    nkx, nky, nkz = 3, 1, 1
    nk = nkx * nky * nkz
    batch = 1
    nc, nv = 3, 2
    nspinor = 2
    n_rmu = 6

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 9)

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        psi_c = jax.random.normal(keys[0], (nk, nc, nspinor, n_rmu)) + 1j * jax.random.normal(
            keys[1], (nk, nc, nspinor, n_rmu)
        )
        psi_v = jax.random.normal(keys[2], (nk, nv, nspinor, n_rmu)) + 1j * jax.random.normal(
            keys[3], (nk, nv, nspinor, n_rmu)
        )
        eps_c = jax.random.uniform(keys[4], (nk, nc), minval=0.1, maxval=0.8)
        eps_v = jax.random.uniform(keys[5], (nk, nv), minval=-0.8, maxval=-0.1)

        v = jax.random.normal(keys[6], (n_rmu, n_rmu), dtype=jnp.float64)
        V_q0 = (v + v.T) / 2.0

        X_full = jax.random.normal(keys[7], (2, batch, nc, nv, nk)) + 1j * jax.random.normal(
            keys[8], (2, batch, nc, nv, nk)
        )
        W_R = jnp.zeros((n_rmu, n_rmu, nkx, nky, nkz), dtype=jnp.complex128)

    mesh = _make_1x1_mesh()
    sh = make_bse_shardings(mesh)
    matvec = build_bse_ring_matvec_full(mesh, nkx, nky, nkz, include_W=False, v_couples_k=True)

    with mesh:
        psi_c_X = jax.lax.with_sharding_constraint(psi_c, sh.psi_x)
        psi_c_Y = jax.lax.with_sharding_constraint(psi_c, sh.psi_y)
        psi_v_X = jax.lax.with_sharding_constraint(psi_v, sh.psi_x)
        psi_v_Y = jax.lax.with_sharding_constraint(psi_v, sh.psi_y)
        eps_c_s = jax.lax.with_sharding_constraint(eps_c, sh.eps)
        eps_v_s = jax.lax.with_sharding_constraint(eps_v, sh.eps)
        V_q0_s = jax.lax.with_sharding_constraint(V_q0, sh.V)
        W_R_s = jax.lax.with_sharding_constraint(W_R, sh.W)
        X_full_s = jax.lax.with_sharding_constraint(X_full, sh.X_full)

        HX = matvec(X_full_s, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c_s, eps_v_s, W_R_s, V_q0_s)
        HX.block_until_ready()
        HX_host = np.asarray(jax.device_get(HX))

    psi_c_host = np.asarray(jax.device_get(psi_c))
    psi_v_host = np.asarray(jax.device_get(psi_v))
    eps_c_host = np.asarray(jax.device_get(eps_c))
    eps_v_host = np.asarray(jax.device_get(eps_v))
    V_q0_host = np.asarray(jax.device_get(V_q0))
    X_full_host = np.asarray(jax.device_get(X_full))

    d_host = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c_host), psi_v_host)  # (k,c,v,mu)
    ncv = nc * nv

    def apply_A(vec_kcv: np.ndarray) -> np.ndarray:
        # vec_kcv: (ncv, nk)
        rho = np.zeros((n_rmu,), dtype=np.complex128)
        for k in range(nk):
            d_k = d_host[k].reshape(ncv, n_rmu)
            rho += vec_kcv[:, k] @ d_k
        phi = V_q0_host @ rho
        out = np.zeros_like(vec_kcv)
        for k in range(nk):
            d_k = d_host[k].reshape(ncv, n_rmu)
            delta_e = (eps_c_host[k, :, None] - eps_v_host[k, None, :]).reshape(ncv)
            out[:, k] = delta_e * vec_kcv[:, k] + (d_k.conj() @ phi) / nk
        return out

    def apply_B(vec_kcv: np.ndarray) -> np.ndarray:
        rho_bar = np.zeros((n_rmu,), dtype=np.complex128)
        for k in range(nk):
            d_k = d_host[k].reshape(ncv, n_rmu)
            rho_bar += vec_kcv[:, k] @ d_k.conj()
        phi_bar = V_q0_host @ rho_bar
        out = np.zeros_like(vec_kcv)
        for k in range(nk):
            d_k = d_host[k].reshape(ncv, n_rmu)
            out[:, k] = (d_k.conj() @ phi_bar) / nk
        return out

    x = X_full_host[0, 0].reshape(ncv, nk)
    y = X_full_host[1, 0].reshape(ncv, nk)

    x_out = apply_A(x) + apply_B(y)
    a_star_y = np.conj(apply_A(np.conj(y)))
    b_star_x = np.conj(apply_B(np.conj(x)))
    y_out = -b_star_x - a_star_y

    X_ref = x_out.reshape(nc, nv, nk)
    Y_ref = y_out.reshape(nc, nv, nk)

    num_x = np.linalg.norm(HX_host[0, 0] - X_ref)
    den_x = np.linalg.norm(X_ref) + 1e-14
    num_y = np.linalg.norm(HX_host[1, 0] - Y_ref)
    den_y = np.linalg.norm(Y_ref) + 1e-14

    rel_x = num_x / den_x
    rel_y = num_y / den_y

    assert rel_x < 1e-10, f"X block rel err too large: {rel_x:.3e}"
    assert rel_y < 1e-10, f"Y block rel err too large: {rel_y:.3e}"

