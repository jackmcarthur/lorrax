"""Equality + memory gates for the trial-stack BSE matvec.

The stack matvec (``bse.bse_stack_matvec.build_bse_stack_matvec``) applies the
TDA BSE (``kernel='bse'``, D+V−W) or RPA (``kernel='rpa'``, D+V) Hamiltonian to a
stack of ``n_trials`` trial vectors at once, materialising exactly ONE direct-term
``T``-tensor regardless of ``n_trials`` (scan-inside-shard_map).  These gates
assert, on the shared ``bse_dense_state`` fixture (MoS2 3×3×1, 2v2c, N=36):

1. **Equality** — per trial, ``stack(X)[b] == H @ X[b]`` (dense reference) AND
   ``== simple_matvec(X[b])``, for both kernels, to relerr < 1e-9.  The dense
   reference already carries the settled B1 dense (k,k') exchange, so this pins
   the stack's exchange + direct terms in one shot.
2. **Memory** — the compiled peak temp is FLAT in ``n_trials`` (the ``n_trials``
   axis never lands on an intermediate), unlike the legacy ring matvec whose
   ``T`` scales linearly.  Machine-checked via ``memory_analysis()``.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from test_bse_dense_reference import _build_dense_H, _relerr  # noqa: E402


def _mesh():
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))


def _random_stack(nt, nc, nv, nk):
    rng = np.random.default_rng(20260716)
    x = (rng.standard_normal((nt, nc, nv, nk))
         + 1j * rng.standard_normal((nt, nc, nv, nk)))
    return jnp.asarray(x)


def _place(data, mesh):
    """Shard the fixture arrays + build W_R for the stack/simple matvecs."""
    from bse.bse_ring_comm import make_bse_shardings
    sh = make_bse_shardings(mesh)
    with mesh:
        out = dict(
            psi_c_X=jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_x),
            psi_c_Y=jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_y),
            psi_v_X=jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_x),
            psi_v_Y=jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_y),
            W_q=jax.lax.with_sharding_constraint(data["W_q"], sh.W),
            V_q0=jax.lax.with_sharding_constraint(data["V_q0"], sh.V),
        )
        out["W_R"] = jnp.fft.ifftn(out["W_q"], axes=(2, 3, 4), norm="ortho")
    return sh, out


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["bse", "rpa"])
def test_stack_matches_dense_and_simple(bse_dense_state, kernel):
    """stack(X)[b] == H @ X[b] (dense) and == simple(X[b]), per trial."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_stack_matvec import build_bse_stack_matvec
    from bse.bse_simple import build_bse_simple_matvec

    data = bse_dense_state
    H, D, Kx, _ = _build_dense_H(data)
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    include_W = kernel == "bse"
    Href = H if include_W else (np.diag(D.astype(np.complex128)) + Kx)

    mesh = _mesh()
    sh, arr = _place(data, mesh)
    nt = 4
    X = _random_stack(nt, nc, nv, nk)
    with mesh:
        Xs = jax.lax.with_sharding_constraint(X, sh.X)
        stack_mv = build_bse_stack_matvec(mesh, nkx, nky, nkz, kernel=kernel)
        simple_mv = build_bse_simple_matvec(mesh, nkx, nky, nkz, include_W=include_W)
        HXs = np.asarray(stack_mv(
            Xs, arr["psi_c_X"], arr["psi_c_Y"], arr["psi_v_X"], arr["psi_v_Y"],
            data["eps_c"], data["eps_v"], arr["W_R"], arr["V_q0"]))
        for t in range(nt):
            xin = jax.lax.with_sharding_constraint(Xs[t:t + 1], sh.X)
            s_t = np.asarray(simple_mv(
                xin, arr["psi_c_X"], arr["psi_c_Y"], arr["psi_v_X"], arr["psi_v_Y"],
                data["eps_c"], data["eps_v"], arr["W_R"], arr["V_q0"]))[0].reshape(-1)
            got = HXs[t].reshape(-1)
            ref = Href @ np.asarray(X)[t].reshape(-1)
            assert _relerr(got, ref) < 1e-9, \
                f"{kernel} trial {t}: vs dense relerr {_relerr(got, ref):.2e}"
            assert _relerr(got, s_t) < 1e-9, \
                f"{kernel} trial {t}: vs simple relerr {_relerr(got, s_t):.2e}"


@pytest.mark.gpu
def test_stack_memory_flat_in_n_trials(bse_dense_state):
    """Compiled peak temp is FLAT in n_trials (the T axis carries no n_trials).

    Contrast with the legacy ring matvec, whose ``T[b,μ,ν,t,s,k]`` temp scales
    linearly.  Also bound the stack temp against the single-trial T size
    ``μ_pad²·ns²·nk·16 B`` (the FFT scratch is a small multiple of one T).
    """
    harness.skip_unless_gpu(pytest)
    from bse.bse_stack_matvec import build_bse_stack_matvec
    from bse.bse_ring_comm import build_bse_ring_matvec

    data = bse_dense_state
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    ns = int(data["psi_c"].shape[2])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    n_rmu_pad = int(data["n_rmu_pad"])
    one_T = n_rmu_pad * n_rmu_pad * ns * ns * nk * 16  # bytes, single-trial T

    mesh = _mesh()
    sh, arr = _place(data, mesh)

    def temp_bytes(mv, nt):
        with mesh:
            X = jax.lax.with_sharding_constraint(
                jnp.zeros((nt, nc, nv, nk), dtype=jnp.complex128), sh.X)
            rest = (arr["psi_c_X"], arr["psi_c_Y"], arr["psi_v_X"], arr["psi_v_Y"],
                    data["eps_c"], data["eps_v"], arr["W_R"], arr["V_q0"])
            compiled = mv.lower(X, *rest).compile()
        return int(compiled.memory_analysis().temp_size_in_bytes)

    with mesh:
        stack_mv = build_bse_stack_matvec(mesh, nkx, nky, nkz, kernel="bse")
        ring_mv = build_bse_ring_matvec(mesh, nkx, nky, nkz, include_W=True, low_mem=True)

    st = {nt: temp_bytes(stack_mv, nt) for nt in (1, 4, 8)}
    rt = {nt: temp_bytes(ring_mv, nt) for nt in (1, 4, 8)}
    for nt in (1, 4, 8):
        print(f"n_trials={nt}: stack temp={st[nt]/1e6:.1f} MB  "
              f"ring temp={rt[nt]/1e6:.1f} MB  (1-T bound={one_T/1e6:.1f})")

    # Flat: temp(8) within 25% of temp(1) — no linear n_trials growth.
    assert st[8] < 1.25 * st[1], f"stack temp not flat: {st}"
    # The single-T bound is respected up to the FFT-scratch multiple (~2-3×).
    assert st[8] < 5 * one_T, f"stack temp {st[8]} exceeds 5× one-T bound {one_T}"
    # And it is strictly better than the ring's linear growth at n_trials=8.
    assert st[8] < 0.5 * rt[8], f"stack {st[8]} not << ring {rt[8]}"
    # Sanity: the ring temp DOES grow with n_trials (guards the contrast).
    assert rt[8] > 2 * rt[1], f"ring temp unexpectedly flat: {rt}"
