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


def test_place_decides_whole_matrices_per_rank_or_the_mesh(monkeypatch):
    """distrib_la.plan.place, the one eigh placement: the faster route whose price fits, timed
    from the measured A100 walls (dla_sweep P4). Fe's charge stack (64 x 312^2) runs 16 whole
    matrices per rank; P matrices of 8192 run whole per rank (2.05 s vs 6.53 s on the mesh), one
    runs on the mesh (1.63 s vs 2.05 s); the mesh over its room is refused by name."""
    import sys
    import dataclasses
    import distrib_la
    pl = sys.modules["distrib_la.plan"]
    monkeypatch.setattr(pl, "mesh_platform", lambda mesh: "CUDA")              # the measured row
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    mesh_figure = lambda: 123456                          # the plan's own program, compiled
    place = lambda nb, n, room: pl.place("eigh", mesh, nb, n, np.complex128, int(room), "scan", mesh_figure)
    fe = place(64, 312, 80e9)
    assert (fe.route, fe.per_rank, fe.rounds) == (pl.ROUTE_BATCH_RESHARD, 16, 1)
    assert fe.program_bytes == pl._stack_bytes("eigh", mesh, 64, 312, np.complex128, 1, "all",
                                               pl._gathered_admitted(mesh, (312, 312), np.complex128, int(80e9)))
    assert place(4, 8192, 80e9).route == pl.ROUTE_BATCH_RESHARD
    one = place(1, 8192, 80e9)
    assert one.route == "scan" and one.program_bytes == 123456
    assert place(1, 1456, 80e9).route == pl.ROUTE_BATCH_RESHARD
    big = place(1, 30000, 8e9)                            # route (c)'s whole matrix: over 8 GB
    assert big.route == "scan" and big.program_bytes == 123456
    plan = dataclasses.replace(distrib_la.plan("eigh", mesh, n=30000, backend="off", batched_route="auto",
                                               budget_bytes=int(8e9)), backend="scalapack")
    assert plan.stack_route((1, 30000, 30000), np.complex128).program_bytes is None   # no lowering here


def test_place_times_the_mesh_by_p_and_platform_without_a_vendor_query(monkeypatch):
    """P16 (4x4): two 20480 matrices take the mesh (2 x 3.6 s, its time past n 16384 scaled 4/P)
    over route (c) (26.8 s; unscaled the mesh would be 2 x 14.2 s); two of 8192 keep route (c)
    (2.05 s), the mesh held at its P4 time below n 16384 (2 x 1.63 s). A platform without a
    measured row (ROCm) keeps route (c) where it fits. No vendor query is issued."""
    import sys
    from types import SimpleNamespace
    import distrib_la  # noqa: F401
    from distrib_la import loader
    pl = sys.modules["distrib_la.plan"]

    def refuse(platform):
        raise AssertionError("a vendor query while planning")
    monkeypatch.setattr(loader, "get_lib", refuse)
    p16 = SimpleNamespace(shape={"x": 4, "y": 4})
    # The time model alone: every candidate fits (the compiled figures need a real 16-device mesh).
    monkeypatch.setattr(pl, "_stack_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(pl, "_gathered_admitted", lambda *a, **k: False)
    place = lambda nb, n: pl.place("eigh", p16, nb, n, np.complex128, int(100e9), "scan", lambda: 0)
    monkeypatch.setattr(pl, "mesh_platform", lambda mesh: "CUDA")
    assert place(2, 20480).route == "scan"
    assert place(2, 8192).route == pl.ROUTE_BATCH_RESHARD
    monkeypatch.setattr(pl, "mesh_platform", lambda mesh: "ROCM")
    assert place(2, 20480).route == pl.ROUTE_BATCH_RESHARD


def test_a_stack_smaller_than_the_ranks_is_no_batch_layout_fit():
    """The head Gamma GEMM (run30k: 1 x 30000 x 15000 on one GPU for 4+ min): one matrix on 16
    ranks fits the batch layout only within GATHERED_EIGH_BYTES; the same live set without the
    stack count fits a large budget."""
    from types import SimpleNamespace
    import distrib_la
    p16 = SimpleNamespace(mesh=SimpleNamespace(devices=np.empty((4, 4))))
    shapes = ((1, 2048, 1024), (1, 1024, 2048), (1, 2048, 2048))               # 0.13 GB live
    assert not distrib_la.fits_local(p16, "gemm", shapes, np.complex128, 10 ** 10, batch=1)
    assert distrib_la.fits_local(p16, "gemm", shapes, np.complex128, 10 ** 10)


def test_the_passivity_eigh_room_hands_back_its_own_reservation():
    """run30k passivity_held: the model row already reserves the check's n x n eigh (7.38 GB), so
    its room hands it back: 15 GiB, not 8 GiB, beside the 18 GB check program."""
    from gw.shared_pole_capacity import check_eigh_room
    row = dict(available_device_bytes_per_rank=int(63.41e9), aggregate_bytes_per_rank=int(36e9),
               native_workspace=dict(eigh=int(7.38e9)))
    assert check_eigh_room(row, int(18e9)) == 15 << 30
    assert check_eigh_room(dict(row, native_workspace={}), int(18e9)) == 8 << 30


def test_constructor_refuses_a_whole_41250_parent_per_rank(monkeypatch):
    """The plausibility deck's largest constructor eigh (one 41250 x 41250 parent, nq 1): route (c)
    would hold the whole matrix per rank; its shape price is over an 80 GB room, so lxkit.memory.admit
    sends it to the whole mesh (staged_eigh)."""
    import distrib_la
    from lxkit import memory
    from gw.shared_pole_capacity import _local_eigenplan
    memory.set_budget(80e9)
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    price = distrib_la.eigh_stack_bytes(_local_eigenplan(mesh, 41250), (1, 41250, 41250), np.complex128)
    assert price > 80e9
    assert memory.admit("test constructor eigh 1 x 41250^2", (
        ("route (c)", memory.priced(price, "eigh_stack_bytes")),
        ("whole mesh", memory.priced(2 * 16 * 41250 ** 2 // 16, "faces"))), 80e9) == "whole mesh"
