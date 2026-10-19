"""The charge ζ factor on the face (isdf.core._factor_c_q_face): main's rank truncation, its eigh
on the face through distrib_la's checked eigh (CPU, 2x2 host mesh)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from isdf import core  # noqa: E402

RCOND = 1e-8


def _gram(n, rank, seed):
    g = np.random.default_rng(seed)
    b = g.standard_normal((rank, n)) + 1j * g.standard_normal((rank, n))
    c = b.conj().T @ b / rank
    return (c + c.conj().T) / 2


@pytest.mark.parametrize("rank", [128, 40])     # well conditioned, and over-complete (the cut binds)
def test_the_face_factor_is_main_s_pseudo_inverse(rank, capsys):
    mesh, n_log, n = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y")), 120, 128
    c = np.zeros((2, n, n), complex)
    for q in range(2):
        c[q, :n_log, :n_log] = _gram(n_log, min(rank, 2 * n_log), 3 + q)
    c[:, n_log:, n_log:] = np.eye(n - n_log)               # factor_c_q's identity pad block
    B = np.asarray(core._factor_c_q_face(jax.device_put(jnp.asarray(c), NamedSharding(mesh, P(None, "x", "y"))),
                                         mesh, n_log, zeta_rcond=RCOND))
    face = capsys.readouterr().out
    z = np.random.default_rng(9).standard_normal((n_log, 6)) + 0j
    for q in range(2):
        zeta, _ = core.solve_zeta_charge_dense(jnp.asarray(c[q, :n_log, :n_log]), jnp.asarray(z),
                                               charge_zeta_solve="rank_truncate", zeta_rcond=RCOND)
        got = B[q] @ (B[q].conj().T @ np.pad(z, ((0, n - n_log), (0, 0))))
        assert np.max(np.abs(got[:n_log] - np.asarray(zeta))) <= 1e-9 * np.max(np.abs(zeta))
        assert not np.any(got[n_log:])                    # pad rows of ζ stay zero
    keep = [int(np.sum(np.linalg.eigvalsh(c[q, :n_log, :n_log]) > RCOND * np.linalg.eigvalsh(c[q, :n_log, :n_log]).max()))
            for q in range(2)]
    assert f"n_keep/q=[{keep[0]} {keep[1]}]" in face
