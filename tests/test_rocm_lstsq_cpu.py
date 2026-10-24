"""QR-SVD minimum-norm least squares: padded/rank-deficient GMRES references."""
import runtime  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


def test_rocm_qr_svd_cutoff_minimum_norm_and_padded_hessenberg(monkeypatch):
    from distrib_la import local_svd as owner
    monkeypatch.setattr(owner, '_process_leg_key', lambda: 'ROCM')
    rng = np.random.default_rng(372)
    cases = []
    # General complex rectangular and rank-deficient (duplicate-column) input.
    a = rng.normal(size=(17, 9))+1j*rng.normal(size=(17, 9))
    cases.append((a, rng.normal(size=(17, 2))+1j*rng.normal(size=(17, 2)), None))
    a[:, 7] = a[:, 2];a[:, 8] = 0
    cases.append((a.copy(), rng.normal(size=17)+1j*rng.normal(size=17), None))
    # Padded upper-Hessenberg: inactive columns must have the min-norm0 result.
    h = np.zeros((49, 48), complex)
    h[:19, :18] = np.triu(rng.normal(size=(19, 18))+1j*rng.normal(size=(19, 18)), -1)
    h[:18, :18] += np.eye(18)*(.191203+.191096j)
    g = np.zeros(49, complex);g[0] = 1
    cases.append((h, g, None))
    # This conditioning would square beyond double precision in normal equations.
    u, _ = np.linalg.qr(rng.normal(size=(17, 9))+1j*rng.normal(size=(17, 9)))
    v, _ = np.linalg.qr(rng.normal(size=(9, 9))+1j*rng.normal(size=(9, 9)))
    bad = (u*np.geomspace(1., 1e-8, 9)) @ v.conj().T
    cases.append((bad, rng.normal(size=17)+1j*rng.normal(size=17), None))
    # A prescribed threshold exercises a truly discarded nonzero singular value.
    cases.append((np.diag([2., 1e-6, 1e-10]).astype(complex), np.ones(3, complex), 1e-8))
    for a, b, cut in cases:
        x, residuals, rank, s = jax.jit(lambda aa, bb: owner.lstsq_local(aa, bb, cut))(a, b)
        expected = np.linalg.lstsq(a, b, rcond=cut)
        tolerance = 5e-8 if np.linalg.cond(a) > 1e7 and expected[2] == a.shape[1] else 3e-11
        np.testing.assert_allclose(x, expected[0], rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(s, expected[3], rtol=2e-12, atol=3e-15)
        assert int(rank) == expected[2]
        np.testing.assert_allclose(residuals, np.linalg.norm(b-a@np.asarray(x), axis=0, keepdims=(b.ndim == 1))**2,
                                   rtol=max(3e-11, tolerance), atol=2e-13)
        if a is h:
            np.testing.assert_array_equal(np.asarray(x)[18:], 0)
    equality = owner.lstsq_local(jnp.diag(jnp.asarray([2., 1., 0.])), jnp.ones(3), rcond=.5)
    np.testing.assert_allclose(equality[0], [.5, 1., 0.], atol=1e-15)
    assert int(equality[2]) == 2  # JAX includes equality and excludes exact zeros.
    zero = owner.lstsq_local(jnp.zeros((7, 5), complex), jnp.arange(7))
    np.testing.assert_array_equal(zero[0], 0)
    assert int(zero[2]) == 0


def test_qr_algorithm_is_explicit_and_other_legs_preserve_jax(monkeypatch):
    from distrib_la import local_svd as owner
    monkeypatch.setattr(owner, '_process_leg_key', lambda: 'ROCM')
    a = np.array([[2., 1j], [-.5j, 3.], [1., .3]], complex)
    b = np.array([1., 2j, 3.])
    graph = jax.make_jaxpr(lambda x: owner.svd_local(x, full_matrices=False))(a)
    calls = [eq for eq in graph.jaxpr.eqns if eq.primitive.name == 'svd']
    assert len(calls) == 1 and calls[0].params['algorithm'].value == 'QR'
    expected = jnp.linalg.lstsq(a, b)
    for leg in ('CPU', 'CUDA'):
        monkeypatch.setattr(owner, '_process_leg_key', lambda: leg)
        actual = owner.lstsq_local(a, b)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
        for got, want in zip(owner.svd_local(a, full_matrices=False),
                             jnp.linalg.svd(a, full_matrices=False)):
            np.testing.assert_array_equal(got, want)


def test_negative_cutoff_empty_and_vector_matrix_rhs(monkeypatch):
    from distrib_la import local_svd as owner
    monkeypatch.setattr(owner, '_process_leg_key', lambda: 'ROCM')
    for a, b in ((np.diag([1., 1e-17]), np.ones(2)),
                  (np.zeros((0, 3)), np.zeros(0)),
                  (np.zeros((3, 0)), np.ones((3, 2))),
                  (np.array([[1, 2], [3, 5], [7, 8]], np.int64), np.ones(3, np.int64))):
        actual = owner.lstsq_local(a, b, rcond=-1.)
        expected = jnp.linalg.lstsq(a, b, rcond=-1.)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, rtol=2e-13, atol=2e-13)
