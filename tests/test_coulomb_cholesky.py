"""The Coulomb root H = L of V = L Lᴴ (gw.response_bank._coulomb_cholesky): CPU, 2x2 host mesh."""
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw import response_bank as rb  # noqa: E402


def _mesh():
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _meta():
    rows = []
    ledger = SimpleNamespace(entries=rows, live_stages=(),
                             reserve=lambda name, **kw: rows.append(name) or dict(name=name, **kw))
    return SimpleNamespace(shared_pole_capacity=ledger)


def _v(n, nq, active, seed):
    g = np.random.default_rng(seed)
    b = g.standard_normal((nq, n, 2 * n)) + 1j * g.standard_normal((nq, n, 2 * n))
    v = b @ np.conj(np.swapaxes(b, -1, -2)) / (2 * n)
    return v * active[:, None] * active[None, :]          # the packed V: pad slots are zero


def test_the_root_reproduces_v_and_its_inverse_on_the_logical_block():
    mesh, n, nq = _mesh(), 16, 2
    active = np.ones(n, bool); active[[3, 9, 15]] = False  # interleaved pad slots
    v = _v(n, nq, active, 1)
    face = NamedSharding(mesh, P(None, "x", "y"))
    h, hi = rb._coulomb_cholesky(_meta(), SimpleNamespace(active_mask=active, n_logical=13), mesh,
                                 jax.device_put(v, face))
    h, hi, a = np.asarray(h), np.asarray(hi), np.ix_(active, active)
    assert np.allclose(h @ np.conj(np.swapaxes(h, -1, -2)), v, atol=1e-12)
    assert np.allclose((hi @ h)[(slice(None),) + a], np.eye(13), atol=1e-12)
    assert not np.any(h[:, ~active]) and not np.any(hi[:, :, ~active])


def test_a_v_that_is_not_positive_definite_is_refused_by_name():
    mesh, n = _mesh(), 16
    active = np.ones(n, bool)
    v = _v(n, 1, active, 2)
    v[0] -= 2 * np.linalg.eigvalsh(v[0]).min() * np.eye(n) + 1e-3 * np.eye(n)
    with pytest.raises(ValueError, match="GATE response_coulomb_psd"):
        rb._coulomb_cholesky(_meta(), SimpleNamespace(active_mask=active, n_logical=n), mesh,
                             jax.device_put(v, NamedSharding(mesh, P(None, "x", "y"))))


def test_w_and_the_moments_are_the_same_operator_under_either_root():
    """Wc = H X (I - X)⁻¹ Hᴴ with X = Hᴴ χ H: H = L and H = V^½ differ by a unitary."""
    mesh, n, nq = _mesh(), 8, 2
    face = NamedSharding(mesh, P(None, "x", "y"))
    v = _v(n, nq, np.ones(n, bool), 3)
    lam, q = np.linalg.eigh(v)
    roots = (np.linalg.cholesky(v), (q * np.sqrt(lam)[:, None, :]) @ np.conj(np.swapaxes(q, -1, -2)))
    g = np.random.default_rng(4)
    chi = [0.05 * (lambda b: -(b @ np.conj(np.swapaxes(b, -1, -2))))(g.standard_normal((nq, n, n)) + 0j) for _ in range(2)]
    built = rb._response_programs(mesh, n, "off", "auto", 0.5, False, None)
    dyson, moments, _ = built.bind(None)
    out = []
    for h in roots:
        h, c0, c1 = (jax.device_put(x, face) for x in (h, *chi))
        out.append([np.asarray(dyson.value(h, c0))] + [np.asarray(m) for m in moments(h, c0, c1)])
    for a, b in zip(*out):
        assert np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(b))
