"""Gate: the block-Lanczos W(omega) chain reproduces the shifted-solve oracle.

``w_omega_chain`` builds ONE structure-preserving block-Lanczos chain on the RPA
screening operator (the Casida ``Omega^2`` reduction ``S = D^{1/2}(D+2V)D^{1/2}``
extracted from the production matvec verbatim) and evaluates ``W_q(omega) - v_q``
for arbitrary complex omega with NO per-omega solve.  This gate checks — cheaply,
on the gnppm fixture at q=0 and the smallest nonzero IBZ q — that the chain
evaluator agrees with the ``bse_w_exact`` per-omega oracle (the gate reference)
at a static, an imaginary-axis, and a real-axis+i*eta frequency, that it
converges monotonically with chain length, and that it emits the established
``(mu_X, nu_Y)`` tile.  Tolerances are tied to the measured convergence at the
gate's chain length (PHASE2_LOG "W(omega) Lanczos-chain model").
"""
import numpy as np
import pytest
import jax
from jax.sharding import Mesh, PartitionSpec as P

from bse import bse_io
from bse.bse_w_exact import (
    _build_rpa_resolvent, _resolve_wc_columns, _select_compare_cols,
    build_finite_q_data, _symmetry_reduced_q_list,
)
from bse.w_omega_chain import build_w_omega_chain, eval_w_omega_chain

RY = 13.6056980659
CHAIN_LEN = 24


def _oracle_tile(cols, z, data, res, p):
    matvec, diag_h, gen, snapshot, sh = res
    W_tile, resids = _resolve_wc_columns(
        cols, z, data, matvec, diag_h, gen, snapshot, sh, max_iter=200, tol=1e-10)
    return (np.asarray(jax.device_get(W_tile)),
            np.asarray(jax.device_get(resids))[:p])


def _chain_rel(chain, data, snapshot, sh, z, w_oracle, cols, nlog, m_use=None):
    (W_tile,) = eval_w_omega_chain(chain, data, snapshot, sh, z, m_use=m_use)
    assert W_tile.sharding.spec == P("x", "y"), (
        f"chain W tile carries {W_tile.sharding.spec}, expected P('x','y')")
    wc = np.asarray(jax.device_get(W_tile))
    return float(max(
        np.linalg.norm(wc[:nlog, i] - w_oracle[:nlog, i])
        / np.linalg.norm(w_oracle[:nlog, i]) for i in range(len(cols))))


@pytest.mark.gpu
def test_w_omega_chain_matches_oracle_q0(gnppm_session):
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False)
    nlog = int(data["n_rmu"])
    T = (np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
         - np.asarray(jax.device_get(data["V_q0"])))

    res = _build_rpa_resolvent(mesh, data)
    matvec, diag_h, gen, snapshot, sh = res
    cols, _ = _select_compare_cols(T, nlog, 3, seed=0)
    chain = build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_LEN)

    # static: the chain evaluator reproduces the W(0) closure through the chain.
    z0 = 0.0 + 0.0j
    w_or, resid = _oracle_tile(cols, z0, data, res, len(cols))
    assert resid.max() < 1e-6, f"oracle GMRES not converged (resid={resid.max():.2e})"
    rel0 = _chain_rel(chain, data, snapshot, sh, z0, w_or, cols, nlog)
    assert rel0 < 5e-3, f"static chain vs oracle rel_err={rel0:.2e} (>5e-3)"
    # closure vs the on-disk (W0 - V) ground truth, through the chain evaluator.
    (W0_tile,) = eval_w_omega_chain(chain, data, snapshot, sh, z0)
    wc0 = np.asarray(jax.device_get(W0_tile))
    rel_disk = float(max(
        np.linalg.norm(wc0[:nlog, i] - T[:nlog, int(cols[i])])
        / np.linalg.norm(T[:nlog, int(cols[i])]) for i in range(len(cols))))
    assert rel_disk < 5e-3, f"static chain vs disk rel_err={rel_disk:.2e} (>5e-3)"

    # imaginary axis (the GW-relevant axis) — converges fastest.
    zi = (0.0 + 4.0j) / RY
    w_or_i, _ = _oracle_tile(cols, zi, data, res, len(cols))
    rel_i_full = _chain_rel(chain, data, snapshot, sh, zi, w_or_i, cols, nlog)
    rel_i_half = _chain_rel(chain, data, snapshot, sh, zi, w_or_i, cols, nlog,
                            m_use=CHAIN_LEN // 2)
    assert rel_i_full < 2e-3, f"imag-axis chain vs oracle rel_err={rel_i_full:.2e} (>2e-3)"
    assert rel_i_full < rel_i_half, (
        f"chain not converging on imaginary axis: m={CHAIN_LEN} rel={rel_i_full:.2e} "
        f">= m={CHAIN_LEN//2} rel={rel_i_half:.2e}")

    # real axis below the gap + small broadening.
    zr = (1.0 + 0.2j) / RY
    w_or_r, _ = _oracle_tile(cols, zr, data, res, len(cols))
    rel_r = _chain_rel(chain, data, snapshot, sh, zr, w_or_r, cols, nlog)
    assert rel_r < 1e-2, f"real-axis chain vs oracle rel_err={rel_r:.2e} (>1e-2)"


@pytest.mark.gpu
def test_w_omega_chain_matches_oracle_finite_q(gnppm_session):
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False, load_v_full=True)
    nlog = int(data["n_rmu"])

    q_list = _symmetry_reduced_q_list(input_path)
    nz = q_list[np.any(q_list != 0, axis=1)]
    q = nz[int(np.argmin((nz.astype(np.int64) ** 2).sum(axis=1)))]
    qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
    assert (qx, qy, qz) != (0, 0, 0), "no nonzero symmetry-reduced q on this fixture"

    dq = build_finite_q_data(data, (qx, qy, qz), mesh)
    res = _build_rpa_resolvent(mesh, dq)
    matvec, diag_h, gen, snapshot, sh = res
    T = (np.asarray(jax.device_get(data["W_q"][:, :, qx, qy, qz]))
         - np.asarray(jax.device_get(data["V_q_full"][:, :, qx, qy, qz])))
    cols, _ = _select_compare_cols(T, nlog, 3, seed=0)
    chain = build_w_omega_chain(dq, matvec, gen, sh, cols, CHAIN_LEN)

    # static + imaginary-axis closure vs the per-q oracle (finite-q floor is ~10x
    # above q=0, so tolerances are looser — still well inside a convergence check).
    z0 = 0.0 + 0.0j
    w_or, resid = _oracle_tile(cols, z0, dq, res, len(cols))
    assert resid.max() < 1e-6, f"q={(qx,qy,qz)} oracle not converged ({resid.max():.2e})"
    rel0 = _chain_rel(chain, dq, snapshot, sh, z0, w_or, cols, nlog)
    assert rel0 < 1e-2, f"q={(qx,qy,qz)} static chain vs oracle rel_err={rel0:.2e} (>1e-2)"

    zi = (0.0 + 4.0j) / RY
    w_or_i, _ = _oracle_tile(cols, zi, dq, res, len(cols))
    rel_i = _chain_rel(chain, dq, snapshot, sh, zi, w_or_i, cols, nlog)
    rel_i_half = _chain_rel(chain, dq, snapshot, sh, zi, w_or_i, cols, nlog,
                            m_use=CHAIN_LEN // 2)
    assert rel_i < 5e-3, f"q={(qx,qy,qz)} imag chain vs oracle rel_err={rel_i:.2e} (>5e-3)"
    assert rel_i < rel_i_half, (
        f"q={(qx,qy,qz)} chain not converging on imaginary axis "
        f"(m={CHAIN_LEN} {rel_i:.2e} >= m={CHAIN_LEN//2} {rel_i_half:.2e})")
