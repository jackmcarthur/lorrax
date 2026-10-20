"""contract_v_group on the face (isdf.zeta_mubatch._contract_v_group_face): the charge B and a
current channel's mesh inverse C^-1 form their ζ on the face and every pair's V accumulates
there, equal to the dense reference, with no whole (mu, mu) per rank at Q = 1 < P."""
import re
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

Q, MU, GT, NGT = 1, 32, 8, 3


@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _zeta(mesh, kind, factor, z_all, ngk):
    owner = NamedSharding(mesh, P(("x", "y"), None, None))
    q_pad = 4
    tiles = [np.pad(z_all[:, :, t * GT:(t + 1) * GT], ((0, q_pad - Q), (0, 0), (0, 0))) for t in range(NGT)]
    from runtime.padding import padded_axis
    store = types.SimpleNamespace(Q=Q, Q_pad=q_pad, mu_pad=MU, g_tile=GT, n_Gt=NGT,
                                  g_axis=padded_axis(GT * NGT, 1, name="test G"),
                                  q_axis=padded_axis(Q, q_pad, name="test q"), read_tile=lambda t: jax.device_put(tiles[t], owner), t_read=0.0)
    return types.SimpleNamespace(store=store, factor=factor, solver_kind=kind, mesh=mesh, mu_basis=None,
                                 ngk_per_q=ngk, n_rmu_solve=MU, print_fn=print, shell=None, shell_slots=None)


def test_face_group_matches_the_dense_reference(mesh, monkeypatch):
    from common import gpu_utils
    from isdf import zeta_mubatch as zm
    monkeypatch.setattr(gpu_utils, "shared_device_room", lambda stage: 1 << 10)   # face SUMMA
    zm._kernel_cache.clear()
    rng = np.random.default_rng(4)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    face = NamedSharding(mesh, P(None, "x", "y"))
    b, cinv = cplx(Q, MU, MU), cplx(Q, MU, MU)
    z1, z2 = cplx(Q, MU, GT * NGT), cplx(Q, MU, GT * NGT)
    ngk = np.array([GT * NGT - 5], np.int32)
    zetas = [_zeta(mesh, "rank_truncate", jax.device_put(b, face), z1, ngk),
             _zeta(mesh, "inverse", jax.device_put(cinv, face), z2, ngk)]
    pairs = ((0, 0), (0, 1), (1, 1))
    v_tables = [rng.uniform(.5, 1.5, (Q, GT * NGT)) for _ in pairs]
    keep = np.array([[0, 3]], np.int32)
    out = zm.contract_v_group(zetas, pairs, v_tables, keep=keep, print_fn=lambda *_: None)
    mask = (np.arange(GT * NGT) < ngk[0])[None, None, :]
    zeta = [np.where(mask, b @ np.conj(np.swapaxes(b, 1, 2)) @ z1, 0), np.where(mask, cinv @ z2, 0)]
    for (a, c), vt, V in zip(pairs, v_tables, out):
        ref = np.einsum("qmg,qg,qng->qmn", np.conj(zeta[a]), vt * mask[:, 0], zeta[c])
        np.testing.assert_allclose(np.asarray(V), ref, rtol=1e-11, atol=1e-11 * np.abs(ref).max())
    for z, ref in zip(zetas, zeta):
        np.testing.assert_allclose(np.asarray(z.shell), ref[:, :, keep[0]], rtol=1e-11, atol=1e-11)
    # Per rank at Q = 1: faces only, in the ζ formers and the pair accumulation.
    whole = re.compile(rf"c128\[(1,)?{MU},{MU}\]")
    step = zm._face_pair_kernel(mesh, GT, pairs, 2)
    zt = tuple(jax.device_put(np.zeros((Q, MU, GT), complex), face) for _ in range(2))
    args = (zt, jnp.asarray(np.stack(v_tables)), jnp.asarray(ngk), jnp.asarray(keep), jnp.int32(0),
            tuple(jax.device_put(np.zeros((Q, MU, MU), complex), face) for _ in pairs),
            tuple(jax.device_put(np.zeros((Q, MU, 2), complex), NamedSharding(mesh, P(None, ("x", "y"), None)))
                  for _ in range(2)))
    texts = [step.lower(*args).compile().as_text()]
    for inverse in (False, True):
        _, apply, _ = zm._face_kernels(mesh, Q, GT, False, inverse=inverse)
        texts.append(apply.lower(jax.device_put(b, face), zt[0]).compile().as_text())
    for text in texts:
        assert not [l for l in text.splitlines() if whole.search(l.replace(" ", ""))]
    zm._kernel_cache.clear()
