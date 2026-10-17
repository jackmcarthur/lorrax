"""The spectral direction cuts decide from one device word; the spectra are read only on request (CPU, 4 devices)."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

N = 16


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices")
    return jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _hermitian(rng, b, spectra):
    out = []
    for i in range(b):
        u, _ = np.linalg.qr(rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
        out.append((u * spectra[i]) @ u.conj().T)
    a = np.asarray(out)
    return (a + np.conj(np.swapaxes(a, -1, -2))) / 2


def _stacks(layout, b):
    rng = np.random.default_rng(7)
    base = np.array([1.0, 0.6, 0.3, 0.12, 0.0901, 0.09005, 0.09, 0.05, 0.02] + [1e-3] * (N - 9))
    perp = _hermitian(rng, b, [base * (1 + 0.1 * i) for i in range(b)])
    top = _hermitian(rng, b, [np.linspace(1.0, 0.1, N) * (1 + 0.1 * i) for i in range(b)])
    mesh = _mesh()
    spec = P(None, "x", "y") if layout == "face" else P(("x", "y"), None, None)
    put = lambda a: jax.device_put(a, NamedSharding(mesh, spec))
    return mesh, put(perp), put(top)


def _plan(mesh):
    import distrib_la
    return distrib_la.plan("eigh", mesh, n=N, backend="off")


def _extent(k):
    return max(2, -(-int(k) // 2) * 2)


def _host_sequence(perp, top, cutoff, real, plan, tol, real_rows):
    """The partner cut as the rounds made it before: two selections, host counts, retain."""
    import distrib_la
    _, largest = distrib_la.leading_eigenvectors(top, 1, eigh_plan=plan, column_extent=_extent,
                                                 multiplet_tol=tol, real_rows=real_rows)
    directions, spectrum = distrib_la.leading_eigenvectors(
        perp, N, eigh_plan=plan, column_extent=_extent, multiplet_tol=tol, real_rows=real_rows)
    counts = tuple(int(np.sum(np.asarray(v) > float(cutoff) ** 2 * float(np.asarray(t)[0])))
                   if i < real else 0 for i, (v, t) in enumerate(zip(spectrum, largest)))
    if max(counts) == 0:
        return None, counts
    kept, values = distrib_la.retain_leading_eigenvectors(
        directions, spectrum, counts, mesh=plan.mesh, column_extent=_extent,
        multiplet_tol=tol, real_rows=real_rows)
    return kept, tuple(int(v.size) for v in values)


@pytest.mark.parametrize("layout,real_rows", [("face", None), ("batch", 3)])
@pytest.mark.parametrize("cutoff,tol", [(0.3, 1e-3), (0.3, 0.0), (0.7, 1e-6), (2.0, 1e-6)])
def test_partner_cut_equals_the_host_sequence(layout, real_rows, cutoff, tol, monkeypatch):
    import distrib_la
    monkeypatch.setenv("LORRAX_CHECK_COUNTS", "1")
    mesh, perp, top = _stacks(layout, 4)
    plan = _plan(mesh)
    real = 3
    want_q, want = _host_sequence(perp, top, cutoff, real, plan, tol, real_rows)
    got_q, got = distrib_la.leading_eigenvectors_above(
        perp, top, cutoff, real=real, eigh_plan=plan, column_extent=_extent,
        multiplet_tol=tol, real_rows=real_rows)
    assert got == want and got[real:] == (0,) * (4 - real)
    if want_q is None:
        assert got_q is None
    else:
        assert got_q.shape == want_q.shape and got_q.sharding.is_equivalent_to(want_q.sharding, 3)
        assert np.array_equal(np.asarray(got_q), np.asarray(want_q))


def test_retained_spectra_are_read_only_on_request():
    import distrib_la
    from distrib_la.polar import RetainedSpectrum
    mesh, perp, _ = _stacks("face", 3)
    plan = _plan(mesh)
    q, spectra = distrib_la.leading_eigenvectors(perp, 4, eigh_plan=plan, column_extent=_extent,
                                                 multiplet_tol=1e-3)
    assert all(isinstance(r, RetainedSpectrum) for r in spectra)
    read = spectra[0]._read
    assert [r.size for r in spectra] == [len(r) for r in spectra] == [r.shape[-1] for r in spectra]
    assert read._host is None                      # sizes needed no transfer
    eig = np.linalg.eigvalsh(np.asarray(perp))[:, ::-1]
    for i, r in enumerate(spectra):
        np.testing.assert_allclose(np.asarray(r), eig[i, :r.size], rtol=1e-10, atol=1e-12)
        assert np.asarray(r).dtype == np.float64 and r[0] == np.asarray(r)[0]
    assert read._host is not None and read._device is None   # one read, the device copy released


def test_singular_cut_word_matches_the_host_cut(monkeypatch):
    import distrib_la
    monkeypatch.setenv("LORRAX_CHECK_COUNTS", "1")
    mesh, perp, _ = _stacks("face", 3)
    plan = _plan(mesh)
    q, spectra = distrib_la.right_singular_vectors(perp, 0.1, eigh_plan=plan, column_extent=_extent,
                                                   multiplet_tol=1e-3, max_rank=6)
    assert all(1 <= r.size <= 6 for r in spectra)
