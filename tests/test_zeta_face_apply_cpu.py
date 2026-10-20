"""The charge channel's face apply (zeta = B(B^H Z), isdf.zeta_mubatch._face_kernels) at Q = 1 < P:
the staged GEMM only within its room, else the face SUMMA; same values, no whole n x n per rank."""
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def test_face_apply_at_one_parent_is_distributed_and_equal(mesh, monkeypatch):
    from common import gpu_utils
    from distrib_la import matmul
    from isdf import zeta_mubatch as zm
    n, g = 256, 64
    monkeypatch.setattr(gpu_utils, "shared_device_room", lambda stage: 1 << 20)   # 1 MiB room
    zm._kernel_cache.clear()
    _, apply, _ = zm._face_kernels(mesh, 1, g, False)
    rng = np.random.default_rng(7)
    face = NamedSharding(mesh, P(None, "x", "y"))
    B = jax.device_put(rng.standard_normal((1, n, n)) + 1j * rng.standard_normal((1, n, n)), face)
    Z = jax.device_put(rng.standard_normal((1, n, g)) + 1j * rng.standard_normal((1, n, g)), face)
    staged = jax.jit(lambda B, Z: matmul(B, matmul(B, Z, transa="C", mesh=mesh), mesh=mesh))(B, Z)
    np.testing.assert_allclose(np.asarray(apply(B, Z)), np.asarray(staged), rtol=1e-12, atol=1e-12)
    text = apply.lower(B, Z).compile().as_text()
    whole = re.compile(rf"c128\[1,{n},{n}\]")
    assert not [l for l in text.splitlines() if whole.search(l.replace(" ", ""))]
    zm._kernel_cache.clear()


def test_face_pass_books_its_compiled_chain(mesh, monkeypatch):
    """The face ζ pass (run30k's V(q), 50.95 GB unpriced) books its tile chain: each program
    compiled at its operands, beside the prefetched tile; V is unchanged."""
    from types import SimpleNamespace
    from common import gpu_utils
    from lxkit import memory
    from runtime.aot_memory import compiled_new_bytes
    from isdf import zeta_mubatch as zm
    monkeypatch.setattr(gpu_utils, "shared_device_room", lambda stage: 1 << 30)
    zm._kernel_cache.clear()
    q, mu, g, nt = 1, 64, 16, 2
    rng = np.random.default_rng(3)
    mush, face = NamedSharding(mesh, P(None, ("x", "y"), None)), NamedSharding(mesh, P(None, "x", "y"))
    host = [rng.standard_normal((q, mu, g)) + 1j * rng.standard_normal((q, mu, g)) for _ in range(nt)]
    tiles = [jax.device_put(z, mush) for z in host]
    b = rng.standard_normal((q, mu, mu)) + 1j * rng.standard_normal((q, mu, mu))
    from runtime.padding import padded_axis
    store = SimpleNamespace(Q=q, g_tile=g, g_axis=padded_axis(g * nt, g, name="test_g"), n_Gt=nt, mu_pad=mu, read_tile=lambda t: tiles[t],
                            placement="device", t_read=0.0)
    fake = SimpleNamespace(store=store, mesh=mesh, solver_kind="rank_truncate", ngk_per_q=np.full(q, g * nt),
                           factor=jax.device_put(b, face), mu_basis=None, print_fn=lambda *_: None)
    V = zm.ZetaG._contract_v_face(fake, np.ones((q, g * nt)), keep=np.zeros((q, 1), np.int32), zeta_io=None,
                                  print_fn=lambda *_: None, with_v=True)
    zeta = [b @ (np.conj(np.swapaxes(b, 1, 2)) @ z) for z in host]
    np.testing.assert_allclose(np.asarray(V), sum(np.conj(z) @ np.swapaxes(z, 1, 2) for z in zeta), rtol=1e-10)
    label = next(k for k in memory._PLANNED if k.startswith("zeta face G-tile chain"))
    to_face, apply, _ = zm._face_kernels(mesh, q, g, True)
    floor = compiled_new_bytes(apply.lower(fake.factor, to_face(tiles[0])).compile()) + tiles[0].nbytes // 4
    assert memory._PLANNED[label]["parts"][label].bytes() >= floor, (memory._PLANNED[label], floor)
    zm._kernel_cache.clear()
