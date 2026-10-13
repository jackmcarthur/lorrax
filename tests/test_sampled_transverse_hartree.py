"""Independent literal current Hartree; no actual atomic-field claim."""
import numpy as np
import pytest


def _mesh():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))


def _alpha():
    """Literal Dirac alpha, independently of the monomial gamma owner."""
    pauli = np.asarray([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    result = np.zeros((3, 4, 4), complex)
    result[:, :2, 2:] = pauli
    result[:, 2:, :2] = pauli
    return result


def _put(mesh, value, spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(np.asarray(value), NamedSharding(mesh, P(*spec)))


def _helical():
    return np.asarray([[1, -1j, 0], [1, 1j, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)


@pytest.mark.parametrize('helical', [False, True])
def test_factored_hartree_matches_literal_q0_no_quadrature_or_half(helical):
    from gw.augmentation_current_hartree import (make_occupied_current_trace,
        make_gamma_current_block_action, make_sampled_current_potential_matrix)
    mesh = _mesh()
    rng = np.random.default_rng(31287)
    nk, nb, mu = 3, 8, 12
    points = np.arange(mu) < 10
    bands = np.arange(nb) < 6
    source = (rng.normal(size=(nk, nb, 4, mu)) + 1j*rng.normal(size=(nk, nb, 4, mu))) * .13
    receiving = (rng.normal(size=(2, nb, 4, mu)) + 1j*rng.normal(size=(2, nb, 4, mu))) * .17
    occ = np.zeros((nk, nb)); occ[:, :5] = [1., .9, .8, .4, .1]
    weights = np.asarray([.2, .3, .5])
    source[:, :, :, ~points] = np.inf
    source[:, 5:] = np.nan
    receiving[:, :, :, ~points] = np.inf
    receiving[:, ~bands] = np.nan
    clean_source = np.where((occ > 0)[:, :, None, None] & points[None, None, None], source, 0.)
    clean_receiver = np.where(bands[None, :, None, None] & points[None, None, None], receiving, 0.)
    alpha = _alpha()
    d = np.einsum('knsm,ist,kntm,kn,k->im', clean_source.conj(), alpha,
                  clean_source, occ, weights)
    # A genuine Gamma operator is real in Cartesian current coordinates
    # and Hermitian under simultaneous component/sample transpose.
    raw = rng.normal(size=(3, mu, 3, mu)) * .07
    T = (raw + raw.transpose(2, 3, 0, 1)).transpose(0, 2, 1, 3) / 2
    T *= points[None, None, :, None] * points[None, None, None, :]
    a_cart = np.einsum('abmn,bn->am', T, d)
    literal = np.einsum(
        'kmsr,ist,kntr,ir->kmn', clean_receiver.conj(), alpha, clean_receiver, a_cart)
    basis = _helical() if helical else None
    B = np.eye(3) if basis is None else basis
    T_fit = np.einsum('ai,ijmn,bj->abmn', B, T, B.conj())
    trace = make_occupied_current_trace(mesh, occ, weights, point_active=points,
                                      current_basis_rows=basis)
    got_d = trace(_put(mesh, source, (None, 'x', None, 'y')))
    np.testing.assert_allclose(np.asarray(got_d), B @ d, rtol=2e-13, atol=2e-14)
    action = make_gamma_current_block_action(mesh, points)
    a = np.zeros((3, mu), complex)
    for i in range(3):
        for j in range(3):
            # Deliberately poison ignored interaction padding too.
            block = np.where(points[:, None] & points[None], T_fit[i, j], np.inf).astype(complex)
            row = _put(mesh, np.asarray(got_d)[j], ('y',))
            a[i] += np.asarray(action(_put(mesh, block, ('x', 'y')), row))
    matrix = make_sampled_current_potential_matrix(mesh, points, bands,
                                                  current_basis_rows=basis)
    actual = np.asarray(matrix(_put(mesh, receiving, (None, 'x', None, 'y')),
                               _put(mesh, a, ())))
    np.testing.assert_allclose(actual, literal, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(actual, actual.conj().transpose(0, 2, 1), atol=3e-14, rtol=0)
    assert np.max(abs(literal)) > 1e-3
    assert not actual[:, ~bands].any() and not actual[:, :, ~bands].any()
    assert np.isfinite(actual).all()
    # A hidden exchange sign or energy functional half-factor must fail.
    assert np.max(abs(actual + literal)) > 1e-3
    assert np.max(abs(actual - literal/2)) > 1e-3


def test_actual_130_occupations_and_exact_zero_small_control():
    from gw.augmentation_current_hartree import make_occupied_current_trace
    mesh = _mesh(); nk, nb, mu = 9, 200, 4
    active = np.ones(mu, bool)
    occ = np.zeros((nk, nb)); occ[:, :130] = 1.
    weights = np.full(nk, 1/nk)
    source = np.zeros((nk, nb, 4, mu), complex)
    source[:, :, 0] = 1/np.sqrt(2); source[:, :, 2] = 1/np.sqrt(2)
    trace = make_occupied_current_trace(mesh, occ, weights, point_active=active)
    actual = np.asarray(trace(_put(mesh, source, (None, 'x', None, 'y'))))
    # Canonical physical occupation-count tolerance (not energy accuracy).
    np.testing.assert_allclose(actual[2], 130., rtol=0, atol=2e-10)
    assert np.max(abs(actual[:2])) == 0.
    source[:, :, 2:] = 0.
    assert np.max(abs(np.asarray(trace(_put(mesh, source, (None, 'x', None, 'y')))))) == 0.
    with pytest.raises(ValueError, match='physical full-zone occupations'):
        make_occupied_current_trace(mesh, occ * 4, weights, point_active=active)


def test_explicit_full_zone_preserves_polar_time_odd_current_action():
    from gw.augmentation_current_hartree import make_occupied_current_trace
    mesh = _mesh(); rng = np.random.default_rng(878)
    psi = rng.normal(size=(4, 4, 4)) + 1j*rng.normal(size=(4, 4, 4))
    # Genuine spin-one-half time reversal acts on both Pauli blocks.
    theta = np.zeros((4, 4), complex)
    theta[:2, :2] = [[0, 1], [-1, 0]]
    theta[2:, 2:] = [[0, 1], [-1, 0]]
    reversed_psi = np.einsum('st,ntm->nsm', theta, psi.conj())
    face = np.stack((psi, reversed_psi))
    trace = make_occupied_current_trace(mesh, np.ones((2, 4)), np.asarray([.5, .5]),
                                      point_active=np.ones(4, bool))
    actual = np.asarray(trace(_put(mesh, face, (None, 'x', None, 'y'))))
    np.testing.assert_allclose(actual, 0., rtol=0, atol=5e-14)
    # Proper z rotation on the spinor rotates the polar current, retaining
    # its time-odd sign under the separately tested antiunitary operation.
    angle = .37
    S = np.diag(np.tile([np.exp(-.5j*angle), np.exp(.5j*angle)], 2))
    rotated = np.einsum('st,ntm->nsm', S, psi)
    literal = np.einsum('nsm,ist,ntm->im', psi.conj(), _alpha(), psi)
    R = np.asarray([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    one = make_occupied_current_trace(mesh, np.ones((1, 4)), np.ones(1),
                                    point_active=np.ones(4, bool))
    got = np.asarray(one(_put(mesh, rotated[None], (None, 'x', None, 'y'))))
    np.testing.assert_allclose(got, R @ literal, rtol=2e-13, atol=3e-14)


def test_current_component_basis_refuses_nonunitary_rows():
    from gw.augmentation_current_hartree import make_occupied_current_trace
    with pytest.raises(ValueError, match='unitary'):
        make_occupied_current_trace(_mesh(), np.ones((1, 4)), np.ones(1),
            point_active=np.ones(4, bool), current_basis_rows=2*np.eye(3))


def test_reader_contraction_uses_distributed_complex_adjoint(monkeypatch):
    """Numerical reader seam only: independent full combined-T contraction."""
    from file_io.restart_bundle import BispinorVqReader
    from gw.augmentation_current_hartree import sampled_transverse_hartree
    import common.gpu_utils as gpu_utils
    mesh = _mesh(); rng = np.random.default_rng(54621)
    nk, nb, mu = 3, 4, 8
    points = np.arange(mu) < 6
    bands = np.arange(nb) < 3
    source = (rng.normal(size=(nk, nb, 4, mu))
              + 1j*rng.normal(size=(nk, nb, 4, mu))) * .1
    receiver = (rng.normal(size=(2, nb, 4, mu))
                + 1j*rng.normal(size=(2, nb, 4, mu))) * .1
    occ = np.zeros((nk, nb)); occ[:, :2] = 1.
    weights = np.full(nk, 1/nk)
    raw = rng.normal(size=(3*mu, 3*mu)) + 1j*rng.normal(size=(3*mu, 3*mu))
    T = ((raw + raw.conj().T) / 2).reshape(3, mu, 3, mu).transpose(0, 2, 1, 3)
    clean_source = source * points[None, None, None]
    clean_receiver = receiver * points[None, None, None] * bands[None, :, None, None]
    current = np.einsum('knsm,ist,kntm,kn,k->im', clean_source.conj(), _alpha(),
                        clean_source, occ, weights).real
    T *= points[None, None, :, None] * points[None, None, None, :]
    potential = np.einsum('ijmn,jn->im', T, current)
    literal = np.einsum('kmsr,ist,kntr,ir->kmn', clean_receiver.conj(), _alpha(),
                        clean_receiver, potential.real)
    # This synthetic reader checks numerical execution, not artifact admission.
    reader = object.__new__(BispinorVqReader)
    reader._mesh = mesh; reader._current_family_authenticated = True
    reader.kgrid = (3, 1, 1)
    reader.get_current_gamma_tile = lambda i, j: _put(mesh, T[i-1, j-1], ('x', 'y'))
    monkeypatch.setattr(gpu_utils, 'device_budget_bytes', lambda: 10**10)
    monkeypatch.setattr(gpu_utils, 'record_stage_price', lambda *args: None)
    got = sampled_transverse_hartree(reader, dict(
        source_full=_put(mesh, source, (None, 'x', None, 'y')),
        receiving=_put(mesh, receiver, (None, 'x', None, 'y')),
        point_active=points, receiving_band_active=bands),
        dict(occupations=occ, full_kweights=weights), resident_bytes=0.)
    np.testing.assert_allclose(np.asarray(got['parent_kij_ry']), literal, rtol=4e-13, atol=2e-14)
    assert got['reality_diagnostics']['raw_potential_imaginary_max'] > 1e-3
    owners = {q['owner'] for q in got['memory_receipt']['quotes']}
    assert {'Cartesian_current_row_projection', 'Gamma_current_block_adjoint'} <= owners


@pytest.mark.parametrize('helical', [False, True])
def test_complex_hermitian_energy_derivative_projects_after_cartesian_conversion(helical):
    from gw.augmentation_current_hartree import make_sampled_current_potential_matrix
    mesh = _mesh(); rng = np.random.default_rng(567)
    nb, mu = 4, 4
    psi = (rng.normal(size=(nb, 4, mu)) + 1j*rng.normal(size=(nb, 4, mu))) * .2
    F = np.einsum('msu,ist,ntu->imnu', psi.conj(), _alpha(), psi)
    D = np.diag([.8, .7, .4, .2]).astype(complex)
    eta = rng.normal(size=(nb, nb)) + 1j*rng.normal(size=(nb, nb))
    eta = (eta + eta.conj().T) / 2
    raw = (rng.normal(size=(3*mu, 3*mu)) + 1j*rng.normal(size=(3*mu, 3*mu))) * .03
    T = (raw + raw.conj().T) / 2

    def density(D):
        value = np.einsum('ab,ibau->iu', D, F)
        np.testing.assert_allclose(value.imag, 0., rtol=0, atol=2e-14)
        return value.real

    def energy(D):
        d = density(D).reshape(-1)
        value = .5*d.conj() @ T @ d
        np.testing.assert_allclose(value.imag, 0., rtol=0, atol=2e-14)
        return value.real

    d = density(D)
    a = (T @ d.reshape(-1)).reshape(3, mu)
    assert np.max(abs(a.imag)) > 1e-3
    expected = np.einsum('iu,imnu->mn', a.real, F)
    B = _helical() if helical else np.eye(3)
    contract = make_sampled_current_potential_matrix(mesh, np.ones(mu, bool), np.ones(nb, bool),
        current_basis_rows=B if helical else None)
    actual = np.asarray(contract(_put(mesh, psi[None], (None, 'x', None, 'y')),
                                 _put(mesh, B @ a, ())))[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=3e-14)
    np.testing.assert_allclose(actual, actual.conj().T, rtol=0, atol=3e-14)
    eps = 2e-5
    centered = (energy(D + eps*eta) - energy(D - eps*eta)) / (2*eps)
    derivative = np.einsum('ab,ba', eta, actual)
    np.testing.assert_allclose(derivative, centered, rtol=2e-9, atol=2e-11)
    unprojected = np.einsum('iu,imnu->mn', a, F)
    assert np.max(abs(unprojected - unprojected.conj().T)) > 1e-3
    if helical:
        prematurely_projected = np.einsum('iu,imnu->mn', B.conj().T @ (B @ a).real, F)
        assert np.max(abs(prematurely_projected - expected)) > 1e-3
