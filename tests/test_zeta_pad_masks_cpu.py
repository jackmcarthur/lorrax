"""The ζ charge factor's pad handling holds no whole (n, n) per rank: the identity pad block is
traced on the face (isdf.core._identity_pad_block_diagonal) and the logical mask of
_factor_c_q_face is an iota, not an (n, n) host constant. P = 4 host devices."""
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def test_identity_pad_is_traced_on_the_face(mesh):
    from isdf import core
    n, n_log = 64, 61
    face = NamedSharding(mesh, P(None, "x", "y"))
    m = jax.device_put(np.random.default_rng(2).standard_normal((1, n, n)) + 0j, face)
    out = core._identity_pad_block_diagonal(m, n_rmu_logical=n_log, mesh_xy=mesh)
    ref = np.asarray(m) + np.diag((np.arange(n) >= n_log).astype(float))[None]
    np.testing.assert_array_equal(np.asarray(out), ref)
    assert out.sharding.is_equivalent_to(face, 3)
    text = core._identity_pad_program(face, n, n_log).lower(m).compile().as_text()
    assert not re.search(rf"c128\[(1,)?{n},{n}\]", text.replace(" ", ""))


def test_charge_mask_is_not_an_n_by_n_constant(mesh, monkeypatch):
    from common import gpu_utils
    from isdf import core
    monkeypatch.setattr(gpu_utils, "shared_device_room", lambda stage: 1 << 34)
    n, n_log = 64, 61
    face = NamedSharding(mesh, P(None, "x", "y"))
    a = np.random.default_rng(3).standard_normal((1, n, 2 * n))
    c = jax.device_put((a @ a.transpose(0, 2, 1)) + 0j, face)
    core._charge_face_cache.clear()
    core._factor_c_q_face(c, mesh, n_log)
    (fn,) = core._charge_face_cache.values()
    text = fn.lower(c).compile().as_text()
    assert not [l for l in text.splitlines() if " constant(" in l and re.search(rf"pred\[{n},{n}\]", l.replace(" ", ""))]
    core._charge_face_cache.clear()
