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
