"""Explicit Fejer response smoothing: tiny dense and native CPU oracles."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_chi_real_space_crop_cpu import _mesh


def _literal(value, source, target):
    out = np.zeros((int(np.prod(target)), *value.shape[1:]), complex)
    for row in product(*(range(n) for n in source)):
        weight = 1.0
        for r, n, m in zip(row, source, target):
            if n != m:
                signed = r if r <= (n-1)//2 else r-n
                weight *= max(1-abs(signed)/(m/2), 0.)
        dest = np.ravel_multi_index(tuple(r % m for r, m in zip(row, target)), target)
        out[dest] += weight*value[np.ravel_multi_index(row, source)]
    return np.fft.fftn(out.reshape(*target, *out.shape[1:]), axes=(0, 1, 2),
                       norm="ortho").reshape(out.shape)*np.sqrt(np.prod(target)/np.prod(source))


@pytest.mark.parametrize("source,target", [((8, 8, 1), (4, 4, 1)),
                                          ((8, 4, 1), (4, 4, 1))])
def test_fejer_complex_lattice_values_normalization_and_positive_fourier_kernel(source, target):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.fft_helpers import make_response_lattice_fftn
    from ffi.fft import xla_reference

    mesh = _mesh(); rng = np.random.default_rng(891)
    nk, nq = int(np.prod(source)), int(np.prod(target))
    value = rng.normal(size=(nk, 4, 4))+1j*rng.normal(size=(nk, 4, 4))
    with xla_reference():
        transform = make_response_lattice_fftn(mesh, source, target,
            P(None, None, None, "x", "y"), window="fejer")
    put = lambda a: jax.device_put(a, NamedSharding(mesh, P(None, "x", "y")))
    np.testing.assert_allclose(transform(put(value)), _literal(value, source, target),
                               rtol=2e-12, atol=1e-13)
    fine = np.indices(source).reshape(3, -1).T/np.array(source)
    coarse = np.indices(target).reshape(3, -1).T/np.array(target)
    kernel = np.ones((nq, nk))
    for axis, (n, m) in enumerate(zip(source, target)):
        angle = 2*np.pi*(coarse[:, None, axis]-fine[None, :, axis])
        if n == m:
            factor = np.exp(1j*angle[..., None]*np.arange(n)).sum(axis=-1).real
        else:
            length = m//2
            factor = abs(np.exp(1j*angle[..., None]*np.arange(length)).sum(axis=-1))**2/length
        kernel *= factor
    kernel /= nk
    assert kernel.min() >= -2e-15
    np.testing.assert_allclose(kernel.sum(axis=1), 1., atol=2e-15)
    # Positive full-port spectral matrices remain positive under this
    # normalized average; no eigenvalue repair is used by the transform.
    spectrum = value@value.conj().swapaxes(-1, -2)
    real = np.fft.ifftn(spectrum.reshape(*source, 4, 4), axes=(0, 1, 2),
                        norm="ortho").reshape(spectrum.shape)
    actual = np.asarray(transform(put(real)))
    np.testing.assert_allclose(actual, np.einsum("qk,kij->qij", kernel, spectrum),
                               rtol=2e-12, atol=2e-13)
    assert np.linalg.eigvalsh(actual).min() > 0.0


def test_fejer_equal_grid_legacy_bits_and_odd_reduced_axis_refusal():
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.fft_helpers import make_flat_k_fftn, make_response_lattice_fftn
    from ffi.fft import xla_reference
    mesh = _mesh(); grid = (3, 2, 1)
    values = np.arange(6*4*4).reshape(6, 4, 4).astype(complex)*(1+.3j)
    values = jax.device_put(values, NamedSharding(mesh, P(None, "x", "y")))
    spec = P(None, None, None, "x", "y")
    with xla_reference():
        legacy = make_flat_k_fftn(mesh, grid, spec, norm="ortho")
        equal = make_response_lattice_fftn(mesh, grid, grid, spec, window="fejer")
        with pytest.raises(ValueError, match="reduced qgrid axis to be even"):
            make_response_lattice_fftn(mesh, (6, 2, 1), (3, 2, 1), spec, window="fejer")
        with pytest.raises(ValueError, match="sharp or fejer"):
            make_response_lattice_fftn(mesh, grid, grid, spec, window="unknown")
    np.testing.assert_array_equal(np.asarray(equal(values)).view(np.uint64),
                                   np.asarray(legacy(values)).view(np.uint64))


def test_fejer_scalar_and_full_density_streams_use_same_literal_window():
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi.fft import xla_reference
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream, _get_chi_fractional_contour_kernel_face
    mesh = _mesh(); rng = np.random.default_rng(93)
    source, target = (8, 1, 1), (4, 1, 1)
    nk, nb, ns, mu = 8, 4, 2, 4
    psi = .2*(rng.normal(size=(nk, nb, ns, mu))+1j*rng.normal(size=(nk, nb, ns, mu)))
    energy = rng.normal(size=(nk, nb))
    rotation, _ = np.linalg.qr(rng.normal(size=(nk, nb, nb))+1j*rng.normal(size=(nk, nb, nb)))
    rho = (rotation*np.array([0., .2, .8, 1.])[None, None])@rotation.conj().swapaxes(-1, -2)
    pm = jax.device_put(psi.transpose(0, 2, 3, 1), NamedSharding(mesh, P(None, None, "x", "y")))
    pn = jax.device_put(psi, NamedSharding(mesh, P(None, "x", None, "y")))
    endpoint = density_left_endpoint(mesh, pm.shape)(pm, rho)
    times, weights = np.array([.13+.1j, -.07+.2j]), np.eye(2, dtype=complex)
    with xla_reference():
        full = full_density_response_stream(mesh, source, (nk, nb, mu, ns),
            q_ids=range(nk), n_outputs=2)
        coarse = full_density_response_stream(mesh, source, (nk, nb, mu, ns),
            q_ids=range(4), n_outputs=2, qgrid=target, response_window="fejer")
        exact = np.asarray(full(times, weights, pm, pn, endpoint, energy))
        actual = np.asarray(coarse(times, weights, pm, pn, endpoint, energy))
        scalar_full = _get_chi_fractional_contour_kernel_face(mesh, source, 2,
            (nk, nb, mu, ns), ordered=True, selected_q=tuple(range(nk)))
        scalar_coarse = _get_chi_fractional_contour_kernel_face(mesh, source, 2,
            (nk, nb, mu, ns), ordered=True, selected_q=tuple(range(4)),
            qgrid=target, response_window="fejer")
    expected = _literal(np.fft.ifft(exact, axis=0, norm="ortho"), source, target)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)
    from gw.response_bank import full_density_bare_moments
    meta = SimpleNamespace(nkx=8, nky=1, nkz=1, qgrid=source)
    with xla_reference():
        moments = full_density_bare_moments((pm, pn, endpoint, energy), meta,
                                           mesh_xy=mesh, q_ids=range(nk))
        meta.qgrid, meta.qgrid_response_window = target, "fejer"
        smoothed = full_density_bare_moments((pm, pn, endpoint, energy), meta,
                                            mesh_xy=mesh, q_ids=range(4))
    for full_moment, actual_moment in zip(moments, smoothed):
        np.testing.assert_allclose(actual_moment,
            _literal(np.fft.ifft(np.asarray(full_moment), axis=0, norm="ortho"), source, target),
            rtol=3e-12, atol=3e-14)
    f = np.broadcast_to([1., .8, .2, 0.], (nk, nb)).copy()
    args = (np.array([.13, .2]), weights, pm, pn, energy, f, 1-f, np.array(0.))
    sf, sc = np.asarray(scalar_full(*args)), np.asarray(scalar_coarse(*args))
    np.testing.assert_allclose(sc, _literal(np.fft.ifft(sf, axis=0, norm="ortho"), source, target),
                               rtol=3e-12, atol=3e-14)


def test_fejer_moment_and_response_bank_forwarding(monkeypatch):
    import gw.w_isdf as kernel
    from gw.response_bank import full_density_bare_moments, response_stream
    seen = []
    def stream(*args, **kwargs):
        seen.append(kwargs)
        return lambda *operands: np.ones((4, 1, 4, 4))
    monkeypatch.setattr(kernel, "full_density_response_stream", stream)
    meta = SimpleNamespace(nkx=8, nky=8, nkz=1, qgrid=(4, 4, 1),
                           qgrid_response_window="fejer")
    full_density_bare_moments((np.zeros((64, 2, 4, 4)), None, None, None),
                               meta, mesh_xy=None, q_ids=range(16))
    assert len(seen) == 3
    assert all(k["response_window"] == "fejer" and k["qgrid"] == (4, 4, 1) for k in seen)
    import gw.response_bank as owner
    monkeypatch.setattr(owner, "_response_stream_kernel", stream)
    meta.nk_tot, meta.nspinor, meta.mu_basis = 64, 2, SimpleNamespace(n_packed=4)
    wfns = SimpleNamespace(green_parent=None, slices=SimpleNamespace(nb_full=4),
                           layout="face", psi_mun=None, psi_nmu=None, enk=None)
    response_stream(wfns, meta, mesh_xy=None, q_ids=range(16), n_outputs=1)
    assert seen[-1]["response_window"] == "fejer"


def test_public_fejer_config_meta_and_early_odd_grid_guard(tmp_path):
    import dataclasses
    from gw.gw_config import LorraxConfig
    from gw.gw_init import validate_qgrid_route
    from tests.test_qgrid_config_cpu import _meta
    from tests.test_gw_qdomain_cpu import _config
    path = tmp_path/"fejer.in"
    path.write_text("[cohsex]\nsys_dim=2\nqgrid=4 4 1\nqgrid_response_window=fejer\n")
    cfg = LorraxConfig.from_input_file(str(path), resolve_hardware=False, print_fn=lambda *a: None)
    assert cfg.qgrid_response_window == "fejer"
    assert dataclasses.replace(_meta((4, 4, 1)), qgrid_response_window="fejer").qgrid_response_window == "fejer"
    with pytest.raises(ValueError, match="qgrid_response_window"):
        dataclasses.replace(cfg, qgrid_response_window="floor")
    with pytest.raises(ValueError, match="reduced qgrid axis to be even"):
        dataclasses.replace(_meta((3, 6, 1)), qgrid_response_window="fejer")
    with pytest.raises(ValueError, match="reduced qgrid axis to be even"):
        validate_qgrid_route(_config(qgrid=(3, 2, 1), qgrid_response_window="fejer"), (6, 4, 1))


def test_store_and_recipe_authenticate_window_with_old_header_sharp_fallback(monkeypatch):
    from file_io.shared_pole_store import _metadata, _check_basis
    from gw.shared_pole_recipe import bind_shared_pole_census, resolve_shared_pole_recipe
    from common.units import RYD_TO_EV
    from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture
    case = _native_port_fixture(monkeypatch)
    meta, mesh, wfns = (case[k] for k in ("meta", "mesh", "wfns"))
    cfg = case["fixture"]._config(np.array([-2., 2.]))
    cfg.bispinor = False
    cfg.sigma.w_model, cfg.sigma.w_accuracy = "shared_pole", "relaxed"
    cfg.sigma.w_support_sites_ev = "1,3,5|2,4,8|1.9"
    cfg.sigma.regularization_ev = .14*RYD_TO_EV
    bind_shared_pole_census(wfns, meta, occupation_state=None,
        trs_allowed=False, state_capacity=1., kweights=np.full(3, 1/3))
    identity = dict(iteration_id="tiny", hamiltonian="H", energies="E",
                    occupations="f", wavefunctions="psi", centroids="mu")
    recipes, headers = [], []
    for window in ("sharp", "fejer"):
        meta.qgrid_response_window = window
        recipe = resolve_shared_pole_recipe(cfg, wfns, meta, mesh_xy=mesh,
            print_fn=lambda *a: None, support_reads_ev=case["energy"][:, 1:]*RYD_TO_EV)
        header = _metadata(meta, case["tables"], recipe, identity)
        assert _check_basis(meta, header) is meta.mu_basis
        recipes.append(recipe); headers.append(header)
    assert recipes[0]["recipe_hash"] != recipes[1]["recipe_hash"]
    assert recipes[0]["gate_hash"] == recipes[1]["gate_hash"]  # Same physical gates.
    assert "qgrid_response_window" not in headers[0]
    assert headers[1]["qgrid_response_window"] == "fejer"
    with pytest.raises(ValueError, match="qgrid_response_window changed"):
        _check_basis(meta, headers[0])
    meta.qgrid_response_window = "sharp"
    assert _check_basis(meta, headers[0]) is meta.mu_basis
    with pytest.raises(ValueError, match="qgrid_response_window changed"):
        _check_basis(meta, headers[1])


def test_fejer_reciprocal_bosons_coarse_q_subgroup_preserves_fresh_active_number():
    """Ordinary coarse-q event identity, not a ring-Phi or Pauli theorem."""
    from tests._full_microscopic_oracle import density_vertices, retarded_reference_pairs
    rng = np.random.default_rng(627)
    nk, nq, nb, mu = 8, 4, 2, 2
    psi = .2*(rng.normal(size=(nk, nb, 1, mu))+1j*rng.normal(size=(nk, nb, 1, mu)))
    energy = rng.normal(size=(nk, nb))
    unitary, _ = np.linalg.qr(rng.normal(size=(nk, nb, nb))+1j*rng.normal(size=(nk, nb, nb)))
    rho = (unitary*np.array([.2, .8])[None, None])@unitary.conj().swapaxes(-1, -2)
    fine = rng.normal(size=(nk, mu, mu))+1j*rng.normal(size=(nk, mu, mu))
    fine = fine@fine.conj().swapaxes(-1, -2)
    angle = 2*np.pi*(np.arange(nq)[:, None]/nq-np.arange(nk)[None]/nk)
    kernel = abs(1+np.exp(1j*angle))**2/(2*nk)  # Fejer order L=2.
    positive = np.einsum("qk,kij->qij", kernel, fine)
    gamma = np.zeros_like(rho); injection = np.zeros_like(rho)
    for q in range(nq):
        partner = (np.arange(nk)-(nk//nq)*q) % nk
        vertex = density_vertices(psi, partner, range(nb))
        poles = np.array([.7-.03j, -.7-.03j])
        residues = np.stack((positive[q], -positive[(-q) % nq].conj()))
        rp, rh = retarded_reference_pairs(vertex, energy[partner], rho[partner],
                                          energy, poles, residues, .15)
        dagger = lambda a: a.conj().swapaxes(-1, -2)
        total = rp[1]+rh[1]
        gamma += 1j*(total-dagger(total))/nq
        injection += 1j*(rh[1]-dagger(rh[1]))/nq
    collision = injection-.5*(gamma@rho+rho@gamma)
    assert np.linalg.norm(collision) > 1e-4
    np.testing.assert_allclose(np.trace(collision, axis1=-2, axis2=-1).mean(),
                               0., atol=3e-15)
