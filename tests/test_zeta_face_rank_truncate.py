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
    c = np.zeros((4, n, n), complex)                         # one q per rank: the native route (_charge_eigh_local)
    for q in range(4):
        c[q, :n_log, :n_log] = _gram(n_log, min(rank, 2 * n_log), 3 + q)
    c[:, n_log:, n_log:] = np.eye(n - n_log)               # factor_c_q's identity pad block
    B = np.asarray(core._factor_c_q_face(jax.device_put(jnp.asarray(c), NamedSharding(mesh, P(None, "x", "y"))),
                                         mesh, n_log, zeta_rcond=RCOND))
    face = capsys.readouterr().out
    z = np.random.default_rng(9).standard_normal((n_log, 6)) + 0j
    for q in range(4):
        zeta, _ = core.solve_zeta_charge_dense(jnp.asarray(c[q, :n_log, :n_log]), jnp.asarray(z),
                                               charge_zeta_solve="rank_truncate", zeta_rcond=RCOND)
        got = B[q] @ (B[q].conj().T @ np.pad(z, ((0, n - n_log), (0, 0))))
        assert np.max(np.abs(got[:n_log] - np.asarray(zeta))) <= 1e-9 * np.max(np.abs(zeta))
        assert not np.any(got[n_log:])                    # pad rows of ζ stay zero
    keep = [int(np.sum(np.linalg.eigvalsh(c[q, :n_log, :n_log]) > RCOND * np.linalg.eigvalsh(c[q, :n_log, :n_log]).max()))
            for q in range(4)]
    assert f"n_keep/q=[{' '.join(map(str, keep))}]" in face


def test_place_decides_whole_matrices_per_rank_or_the_mesh():
    """distrib_la.plan.place, the one eigh placement: the faster route whose price fits, timed
    from the measured A100 walls (dla_sweep P4). Fe's charge stack (64 x 312^2) runs 16 whole
    matrices per rank; P matrices of 8192 run whole per rank (2.05 s vs 6.53 s on the mesh), one
    runs on the mesh (1.63 s vs 2.05 s); the mesh over its room is refused by name."""
    import sys
    import dataclasses
    import distrib_la
    pl = sys.modules["distrib_la.plan"]
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    fe = pl.place("eigh", mesh, 64, 312, np.complex128, int(80e9), "scan")
    assert (fe.route, fe.per_rank, fe.rounds) == (pl.ROUTE_BATCH_RESHARD, 16, 1)
    assert pl.place("eigh", mesh, 4, 8192, np.complex128, int(80e9), "scan").route == pl.ROUTE_BATCH_RESHARD
    one = pl.place("eigh", mesh, 1, 8192, np.complex128, int(80e9), "scan")
    assert one.route == "scan" and one.program_bytes == (pl.CHECKED_EIGH_TILES + 2) * 8192 ** 2 * 16 // 4
    assert pl.place("eigh", mesh, 1, 1456, np.complex128, int(80e9), "scan").route == pl.ROUTE_BATCH_RESHARD
    big = pl.place("eigh", mesh, 1, 30000, np.complex128, int(8e9), "scan")       # 32.4 GB on 2x2
    assert big.route == "scan" and big.program_bytes > big.room
    plan = dataclasses.replace(distrib_la.plan("eigh", mesh, n=30000, backend="off", batched_route="auto",
                                               budget_bytes=int(8e9)), backend="scalapack")
    with pytest.raises(RuntimeError, match="GATE distrib_la_capacity"):
        plan.stack_route((1, 30000, 30000), np.complex128)


def test_constructor_refuses_a_whole_41250_parent_per_rank(monkeypatch):
    """The plausibility deck's largest constructor eigh (one 41250 x 41250 parent, nq 1): route (c)
    would hold the whole matrix per rank; its shape price is over an 80 GB room, so admit_local
    sends it to the whole mesh (staged_eigh)."""
    import distrib_la
    from common import gpu_utils
    from gw.shared_pole_capacity import _local_eigenplan
    monkeypatch.setattr(gpu_utils, "_RUN_DEVICE_BUDGET_GB", 80.0)
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    price = distrib_la.eigh_stack_bytes(_local_eigenplan(mesh, 41250), (1, 41250, 41250), np.complex128)
    assert price > 80e9
    assert not gpu_utils.admit_local("test constructor eigh 1 x 41250^2", price, 80e9,
                                     distributed_bytes=2 * 16 * 41250 ** 2 // 16, names=("route (c)", "whole mesh"))
