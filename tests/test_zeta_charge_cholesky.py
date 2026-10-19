"""The charge ζ seam (isdf.cplus): a certified face Cholesky, refused by name where the old eigh
pseudo-inverse's rank cut would have bound (CPU, 2x2 host mesh, native2d)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from isdf import cplus  # noqa: E402

RCOND = 1e-8


def _mesh():
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _face(mesh, a):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, P(None, "x", "y")))


def _gram(n, rank, seed, floor=0.0):
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((rank, n)) + 1j * rng.standard_normal((rank, n))
    c = b.conj().T @ b / rank
    return (c + c.conj().T) / 2 + floor * np.eye(n)


def test_a_well_conditioned_gram_is_factored_and_solves_on_the_face(capsys):
    mesh, n = _mesh(), 128
    c = _gram(n, 3 * n, 1)[None]
    z = (np.random.default_rng(2).standard_normal((1, n, 16)) + 0j)
    F = cplus.factor(_face(mesh, c.copy()), mesh, rcond=RCOND, n_log=n)
    zeta = np.asarray(cplus.solve(F, _face(mesh, z)))
    assert np.max(np.abs(zeta - np.linalg.solve(c[0], z[0]))) < 1e-10 * np.max(np.abs(zeta))
    assert "[zeta charge cholesky] n_log=128" in capsys.readouterr().out


def test_an_over_complete_centroid_set_is_refused_by_name():
    """Rank 40 of N_mu = 128 (the Cr:Al2O3 shape: pivoted-Cholesky rank 18,779 of 29,998)."""
    mesh, n = _mesh(), 128
    c = _gram(n, 40, 3)[None]
    with pytest.raises(ValueError) as error:
        cplus.factor(_face(mesh, c), mesh, rcond=RCOND, n_log=n)
    msg = str(error.value)
    assert "GATE zeta_charge_conditioning" in msg and "N_mu = 128" in msg
    assert "LORRAX_CENTROID_SELECT=strict" in msg and "fewer centroids" in msg and "more bands" in msg
    assert "numerical rank/q [40] (eigenvalues above rcond·λ_max)" in msg


def test_the_certificate_refuses_where_the_cut_would_bind():
    """kappa just above 1/rcond: PD, factorable, and refused (the old cut dropped modes here)."""
    mesh, n = _mesh(), 64
    q, _ = np.linalg.qr(np.random.default_rng(4).standard_normal((n, n)) + 0j)
    w = np.geomspace(1.0, 0.3 * RCOND, n)
    c = (q * w) @ q.conj().T
    with pytest.raises(ValueError, match="kappa"):
        cplus.factor(_face(mesh, ((c + c.conj().T) / 2)[None]), mesh, rcond=RCOND, n_log=n)


def test_the_dense_refit_is_the_same_certified_solve():
    n = 64
    c, z = _gram(n, 3 * n, 5), np.random.default_rng(6).standard_normal((n, 8)) + 0j
    zeta, figures = jax.jit(lambda a, b: cplus.dense(a, b, rcond=RCOND))(jnp.asarray(c), jnp.asarray(z))
    cplus.report(figures, rcond=RCOND, n_log=n)
    assert np.max(np.abs(np.asarray(zeta) - np.linalg.solve(c, z))) < 1e-10 * np.max(np.abs(zeta))
    bad = _gram(n, 20, 7)
    _, figures = jax.jit(lambda a, b: cplus.dense(a, b, rcond=RCOND))(jnp.asarray(bad), jnp.asarray(z))
    with pytest.raises(ValueError, match="GATE zeta_charge_conditioning"):
        cplus.report(figures, rcond=RCOND, n_log=n)
