"""Full electronic lattice -> restricted response lattice, tiny CPU only."""
from itertools import product

import numpy as np
import pytest


def _mesh():
    import jax
    from jax.sharding import Mesh
    jax.config.update("jax_enable_x64", True)
    devices = np.asarray(jax.devices("cpu"))
    return Mesh(devices[:4].reshape(2, 2) if len(devices) >= 4 else devices[:1].reshape(1, 1), ("x", "y"))


def _literal_crop(value, source, target):
    """Independent signed-cell loop, including the two even boundary images."""
    result = np.zeros((*target, *value.shape[1:]), complex)
    shaped = value.reshape(*source, *value.shape[1:])
    cells = [range(-m//2, m//2+1) if m % 2 == 0 and m != n
             else range(-(m//2), (m-1)//2+1) for n, m in zip(source, target)]
    for cell in product(*cells):
        weight = np.prod([.5 if m != n and m % 2 == 0 and abs(r) == m//2 else 1.
                          for r, n, m in zip(cell, source, target)])
        result[tuple(r % m for r, m in zip(cell, target))] += weight*shaped[tuple(r % n for r, n in zip(cell, source))]
    return result.reshape(np.prod(target), *value.shape[1:])


@pytest.mark.parametrize("source,target", [((24, 24, 1), (6, 6, 1)),
    ((8, 6, 2), (4, 3, 1)), ((5, 3, 1), (1, 3, 1))])
def test_random_complex_spinor_greens_full_fft_then_real_crop_matches_literal(source, target):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.fft_helpers import make_flat_k_ifftn, make_response_lattice_fftn
    from ffi.fft import xla_reference
    mesh = _mesh(); rng = np.random.default_rng(1421)
    nk, nq, mu, ns = int(np.prod(source)), int(np.prod(target)), 4, 2
    random = lambda: rng.normal(size=(nk, mu, ns, mu, ns))+1j*rng.normal(size=(nk, mu, ns, mu, ns))
    left, right = random(), random()
    gspec = P(None, None, None, "x", None, "y", None)
    spec = P(None, None, None, "x", "y")
    put = lambda a: jax.device_put(a, NamedSharding(mesh, P(None, "x", None, "y", None)))
    with xla_reference():
        inverse = make_flat_k_ifftn(mesh, source, gspec, norm="ortho")
        transform = make_response_lattice_fftn(mesh, source, target, spec)
    lr, rr = inverse(put(left)), inverse(put(right))
    actual_r = jax.jit(lambda a, b: jnp.einsum("kmanb,kmanb->kmn", a.conj(), b))(lr, rr)
    actual = np.asarray(transform(actual_r))
    lr_np = np.fft.ifftn(left.reshape(*source, mu, ns, mu, ns), axes=(0, 1, 2), norm="ortho").reshape(left.shape)
    rr_np = np.fft.ifftn(right.reshape(*source, mu, ns, mu, ns), axes=(0, 1, 2), norm="ortho").reshape(right.shape)
    full_r = np.einsum("kmanb,kmanb->kmn", lr_np.conj(), rr_np)
    restricted = _literal_crop(full_r, source, target)
    expected = np.fft.fftn(restricted.reshape(*target, mu, mu), axes=(0, 1, 2), norm="ortho").reshape(nq, mu, mu)*np.sqrt(nq/nk)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=3e-13)
    # The physical prefactor still refers to ELECTRONIC Nk.
    physical = actual/np.sqrt(nk)
    direct = np.fft.fftn(restricted.reshape(*target, mu, mu), axes=(0, 1, 2)).reshape(nq, mu, mu)/nk
    np.testing.assert_allclose(physical, direct, rtol=2e-12, atol=3e-13)


def test_equal_grid_preserves_legacy_fft_bits_and_default():
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.fft_helpers import make_flat_k_fftn, make_response_lattice_fftn
    from ffi.fft import xla_reference
    mesh = _mesh(); grid = (4, 6, 1); rng = np.random.default_rng(812)
    value = jax.device_put(rng.normal(size=(24, 4, 4))+1j*rng.normal(size=(24, 4, 4)),
                           NamedSharding(mesh, P(None, "x", "y")))
    spec = P(None, None, None, "x", "y")
    with xla_reference():
        legacy = make_flat_k_fftn(mesh, grid, spec, norm="ortho")
        equal = make_response_lattice_fftn(mesh, grid, grid, spec)
        default = make_response_lattice_fftn(mesh, grid, None, spec)
    expected = np.asarray(legacy(value)).view(np.uint64)
    np.testing.assert_array_equal(np.asarray(equal(value)).view(np.uint64), expected)
    np.testing.assert_array_equal(np.asarray(default(value)).view(np.uint64), expected)


def test_even_boundary_half_weights_preserve_inversion_pairs_and_not_q_patch():
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.fft_helpers import make_response_lattice_fftn
    from ffi.fft import xla_reference
    mesh = _mesh(); source, target = (8, 8, 1), (4, 4, 1)
    value = np.zeros((8, 8, 1, 4, 4), complex)
    value[2, 0, 0] = 2+3j; value[-2, 0, 0] = 2-3j
    value[0, 2, 0] = 4+5j; value[0, -2, 0] = 4-5j
    value[2, 2, 0] = 7+9j; value[-2, -2, 0] = 7-9j
    with xla_reference():
        transform = make_response_lattice_fftn(mesh, source, target, P(None, None, None, "x", "y"))
    actual = np.asarray(transform(jax.device_put(value.reshape(64, 4, 4), NamedSharding(mesh, P(None, "x", "y")))))
    expected_r = _literal_crop(value.reshape(64, 4, 4), source, target)
    assert expected_r.reshape(4, 4, 1, 4, 4)[2, 0, 0, 0, 0] == 2
    assert expected_r.reshape(4, 4, 1, 4, 4)[2, 2, 0, 0, 0] == 3.5
    assert np.max(abs(actual.imag)) < 2e-14
    # It is a new full-BZ transform, not a gather of low-q full-grid values.
    fine = np.fft.fftn(value, axes=(0, 1, 2), norm="ortho").reshape(64, 4, 4)
    assert np.linalg.norm(actual-fine[:16]) > 1


@pytest.mark.parametrize("target", [(3, 4, 1), (12, 4, 1), (4, 0, 1), (4., 4., 1.), (4, 4)])
def test_invalid_response_grid_refuses_before_any_response_array(target):
    from common.fft_helpers import make_response_lattice_fftn
    from jax.sharding import PartitionSpec as P
    with pytest.raises(ValueError, match="response_qgrid"):
        make_response_lattice_fftn(_mesh(), (8, 8, 1), target, P(None, None, None, "x", "y"))


def test_full_coherent_stream_restricts_lattice_before_coarse_q_selection():
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi.fft import xla_reference
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream
    from tests._full_microscopic_oracle import ordered_pr_time
    mesh = _mesh(); rng = np.random.default_rng(883)
    source, target = (8, 1, 1), (4, 1, 1)
    nk, nb, ns, mu = 8, 4, 2, 4
    psi = .2*(rng.normal(size=(nk, nb, ns, mu))+1j*rng.normal(size=(nk, nb, ns, mu)))
    energy = np.broadcast_to([-.8, -.2, .3, .9], (nk, nb)).copy()+rng.normal(size=(nk, nb))*.03
    rho = np.empty((nk, nb, nb), complex)
    for k in range(nk):
        unitary, _ = np.linalg.qr(rng.normal(size=(nb, nb))+1j*rng.normal(size=(nb, nb)))
        rho[k] = unitary@np.diag([.05, .18, .77, .96])@unitary.conj().T
    assert np.linalg.norm(rho*energy[:, None, :]-energy[:, :, None]*rho) > .1
    pm = jax.device_put(psi.transpose(0, 2, 3, 1), NamedSharding(mesh, P(None, None, "x", "y")))
    pn = jax.device_put(psi, NamedSharding(mesh, P(None, "x", None, "y")))
    endpoint = density_left_endpoint(mesh, pm.shape)(pm, rho)
    times = np.array([.13+.21j, -.11+.29j]); weights = np.eye(len(times), dtype=complex)
    requested = (3, 0, 1)
    with xla_reference():
        stream = full_density_response_stream(mesh, source, (nk, nb, mu, ns),
            q_ids=requested, n_outputs=len(times), qgrid=target)
        actual = np.asarray(stream(times, weights, pm, pn, endpoint, energy))
    full = np.stack([ordered_pr_time(psi, energy, rho, (np.arange(nk)-q)%nk,
                                    times, spin_weight=np.sqrt(nk)) for q in range(nk)])
    full_r = np.fft.ifft(full, axis=0, norm="ortho")
    restricted = _literal_crop(full_r, source, target)
    expected = np.fft.fft(restricted, axis=0, norm="ortho")*np.sqrt(4/8)
    np.testing.assert_allclose(actual, expected[list(requested)], rtol=3e-12, atol=3e-14)
    with xla_reference():
        legacy = full_density_response_stream(mesh, source, (nk, nb, mu, ns),
            q_ids=range(nk), n_outputs=len(times))
        equal = full_density_response_stream(mesh, source, (nk, nb, mu, ns),
            q_ids=range(nk), n_outputs=len(times), qgrid=source)
    original = np.asarray(legacy(times, weights, pm, pn, endpoint, energy))
    np.testing.assert_allclose(original, full, rtol=3e-12, atol=3e-14)
    np.testing.assert_array_equal(original.view(np.uint64),
        np.asarray(equal(times, weights, pm, pn, endpoint, energy)).view(np.uint64))
    with xla_reference(), pytest.raises(ValueError, match="invalid exact q rows"):
        full_density_response_stream(mesh, source, (nk, nb, mu, ns), q_ids=[4],
                                     n_outputs=1, qgrid=target)


@pytest.mark.parametrize("ordered", [False, True])
def test_scalar_retarded_stream_selects_coarse_rows_after_lattice_crop(ordered):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi.fft import xla_reference
    from gw.w_isdf import _get_chi_fractional_contour_kernel_face
    mesh = _mesh(); rng = np.random.default_rng(926)
    source, target = (8, 1, 1), (4, 1, 1)
    nk, nb, ns, mu = 8, 4, 2, 4
    psi = .2*(rng.normal(size=(nk, nb, ns, mu))+1j*rng.normal(size=(nk, nb, ns, mu)))
    energy = np.broadcast_to([-.8, -.2, .3, .9], (nk, nb)).copy()+rng.normal(size=(nk, nb))*.03
    occupied = np.broadcast_to([.9, .6, .2, .1], (nk, nb)).copy()
    pm = jax.device_put(psi.transpose(0, 2, 3, 1), NamedSharding(mesh, P(None, None, "x", "y")))
    pn = jax.device_put(psi, NamedSharding(mesh, P(None, "x", None, "y")))
    args = (np.array([.13, .27]), np.array([[.4+.1j, .2-.1j], [.1+.2j, -.3+.2j]]),
            pm, pn, energy, occupied, 1-occupied, np.array(0.))
    requested = (3, 0, 1)
    with xla_reference():
        legacy = _get_chi_fractional_contour_kernel_face(mesh, source, 2, (nk, nb, mu, ns),
                                                        ordered=ordered)
        coarse = _get_chi_fractional_contour_kernel_face(mesh, source, 2, (nk, nb, mu, ns),
            ordered=ordered, selected_q=requested, qgrid=target)
        full = np.stack(legacy(*args), axis=1)
        actual = np.asarray(coarse(*args))
    restricted = _literal_crop(np.fft.ifft(full, axis=0, norm="ortho"), source, target)
    expected = np.fft.fft(restricted, axis=0, norm="ortho")*np.sqrt(4/8)
    np.testing.assert_allclose(actual, expected[list(requested)], rtol=3e-12, atol=3e-14)
    with xla_reference(), pytest.raises(ValueError, match="bosonic full-grid indices"):
        _get_chi_fractional_contour_kernel_face(mesh, source, 2, (nk, nb, mu, ns),
            selected_q=(4,), qgrid=target)


def test_native_parent_direct_stream_crops_after_full_electronic_kconv(monkeypatch):
    from ffi.fft import xla_reference
    from gw.w_isdf import _get_chi_fractional_contour_kernel_face
    from tests import test_tdagw_dynsig_sigma as fixture
    mesh = _mesh(); rng = np.random.default_rng(628)
    source, target = (4, 2, 1), (4, 1, 1)
    for name, value in dict(KGRID=source, NBF=3, NB_PAD=4, NS=2, NMU=4, NOCC=2).items():
        monkeypatch.setattr(fixture, name, value)
    nk, nb, ns, mu = 8, 3, 2, 4
    psi = .2*(rng.normal(size=(nk, nb, ns, mu))+1j*rng.normal(size=(nk, nb, ns, mu)))
    energy = rng.uniform(-.8, .9, (nk, nb))
    occupied = rng.uniform(.1, .9, (nk, nb))
    kf = np.indices(source).reshape(3, -1).T/np.array(source)
    plan, perm = fixture._plan(mesh, kf, rng)
    wfns, _ = fixture._gw_bundle(mesh, psi[..., perm], energy, occupied, plan)
    parent = wfns.green_parent
    times = np.array([.13+.1j, -.07+.19j])
    projections = np.array([[[.4+.1j, .2-.1j], [.1+.2j, -.3+.2j]],
                            [[.2+.3j, -.1+.1j], [.3-.2j, .2+.1j]]])
    args = (times, projections, parent.psi_mun, parent.psi_nmu, parent.enk,
            parent.occ, 1-parent.occ, np.array([0., 0.]))
    requested = (3, 0, 1, 2)
    with xla_reference():
        full_kernel = _get_chi_fractional_contour_kernel_face(mesh, source, 2,
            (nk, 4, mu, ns), k_unfold_plan=plan, selected_q=tuple(range(nk)),
            pair_mode="direct", ordered=True)
        coarse_kernel = _get_chi_fractional_contour_kernel_face(mesh, source, 2,
            (nk, 4, mu, ns), k_unfold_plan=plan, selected_q=requested,
            pair_mode="direct", ordered=True, qgrid=target)
        full = np.asarray(full_kernel(*args))
        actual = np.asarray(coarse_kernel(*args))
    full_r = np.fft.ifftn(full.reshape(*source, 2, mu, mu), axes=(0, 1, 2), norm="ortho").reshape(full.shape)
    restricted = _literal_crop(full_r, source, target)
    expected = np.fft.fftn(restricted.reshape(*target, 2, mu, mu), axes=(0, 1, 2), norm="ortho").reshape(4, 2, mu, mu)*np.sqrt(4/8)
    np.testing.assert_allclose(actual, expected[list(requested)], rtol=3e-12, atol=3e-14)
