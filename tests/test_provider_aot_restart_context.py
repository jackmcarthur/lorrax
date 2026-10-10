"""Explicit restart model binding and unchanged public Coulomb AOT actions.

CPU unit scope: mode/receipt flow and tiny physical radial contractions.
These checks do not admit a production cache, SC calculation or MPI layout.
"""
from types import SimpleNamespace

import numpy as np
import pytest


def _restart(monkeypatch, *, spinors, mode, stored_tag):
    from gw import gw_init
    from file_io import restart_bundle

    points = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    digest = gw_init._centroid_table_md5(points, coordinate_kind='fft_indices')
    monkeypatch.setattr(restart_bundle, 'read_metadata', lambda _: dict(
        centroid_hashes={'charge': digest, 'current': None},
        charge_representation=stored_tag, bispinor_gw=mode.value))
    received = []

    class Receipt:
        @staticmethod
        def from_bound_source(**kwargs):
            received.append(kwargs)
            return kwargs

    meta = SimpleNamespace(nspinor=spinors, fft_grid=(4, 4, 4),
        mu_basis=SimpleNamespace(coordinate_kind='fft_indices'),
        n_rmu=2, n_rmu_padded=2)
    result = gw_init._restart_charge_basis(Receipt, (0, 8), True,
        'opaque-bound-source-token', points, None, meta, lambda *_: None,
        'metadata-only-fixture', object(), bispinor_gw=mode)
    assert len(received) == 1
    assert received[0]['band_interval'] == (0, 8)
    assert received[0]['wfn_fingerprint_binding'] == 'opaque-bound-source-token'
    return result[1]


@pytest.mark.parametrize('spinors', [2, 4])
def test_restart_keeps_shipped_source_and_rkb_lifts(monkeypatch, spinors):
    from common.bispinor_init import NORMALIZED_RKB_LIFT
    from common.four_current_model import (
        NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION, SOURCE_WFN_CHARGE_REPRESENTATION)
    from gw.gw_config import BispinorGWMode

    tag = (SOURCE_WFN_CHARGE_REPRESENTATION if spinors == 2
           else NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION)
    receipt = _restart(monkeypatch, spinors=spinors,
        mode=BispinorGWMode.COULOMB_ONLY, stored_tag=tag)
    assert receipt['bispinor_lift'] == ('raw' if spinors == 2 else NORMALIZED_RKB_LIFT)


def _install_pauli_resolver(monkeypatch):
    # Import orchestration first, so a cached generic alias would fail this
    # test. The installed representation is deliberately charge-only.
    from gw import gw_init
    from common import four_current_model
    from gw.gw_config import BispinorGWMode

    def resolve(bispinor, mode):
        if not bispinor or mode is not BispinorGWMode.COULOMB_ONLY:
            raise ValueError('Pauli charge fixture forbids current modes')
        return four_current_model.FourCurrentRepresentation(True,
            'dev_embedded_implicit_pauli', False, None, True,
            'dev_implicit_pauli_embedded_four_slot_charge_v1')

    monkeypatch.setattr(four_current_model, 'resolve_four_current_representation', resolve)


def test_restart_observes_installed_pauli_after_orchestration_import(monkeypatch):
    from gw.gw_config import BispinorGWMode

    _install_pauli_resolver(monkeypatch)
    receipt = _restart(monkeypatch, spinors=4, mode=BispinorGWMode.COULOMB_ONLY,
        stored_tag='dev_implicit_pauli_embedded_four_slot_charge_v1')
    assert receipt['bispinor_lift'] == 'dev_embedded_implicit_pauli'


def test_restart_does_not_relabel_rkb_samples_as_pauli(monkeypatch):
    from common.four_current_model import NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION
    from gw.gw_config import BispinorGWMode

    _install_pauli_resolver(monkeypatch)
    with pytest.raises(ValueError, match='restart_bispinor_charge_carrier'):
        _restart(monkeypatch, spinors=4, mode=BispinorGWMode.COULOMB_ONLY,
            stored_tag=NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION)


def test_restart_pauli_current_mode_refuses_before_basis_receipt(monkeypatch):
    from gw.gw_config import BispinorGWMode

    _install_pauli_resolver(monkeypatch)
    with pytest.raises(ValueError, match='forbids current modes'):
        _restart(monkeypatch, spinors=4, mode=BispinorGWMode.BARE_TRANSVERSE,
            stored_tag='dev_implicit_pauli_embedded_four_slot_charge_v1')


@pytest.mark.parametrize('kind', ['plain', 'paired', 'monopole', 'paired_monopole'])
def test_public_aot_free_kernel_retains_actual_callback_and_radial_action(kind):
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from isdf.atomic_coulomb import radial_coulomb_provider

    mesh = Mesh(np.asarray(jax.devices()).reshape(1, 1), ('x', 'y'))
    shape = NamedSharding(mesh, P(('x', 'y'), None, None))
    put = lambda a: device_put_process_local(np.asarray(a), shape)
    r = np.linspace(.08, 1.2, 8)
    edges = np.r_[0., (r[:-1] + r[1:]) / 2, 1.2]
    w = np.diff(edges)
    q, mu, nf = 2, 3, len(r)
    rng = np.random.default_rng(1495)
    random = lambda n: rng.normal(size=n) + 1j*rng.normal(size=n)
    stub = SimpleNamespace(mesh=mesh, solver_kind='rank_truncate',
        store=SimpleNamespace(Q=q, Q_pad=q, mu_pad=mu, g_tile=2,
                              g_axis=SimpleNamespace(logical=3)),
        ngk_per_q=np.asarray([3, 2]))
    paired = kind in ('paired', 'paired_monopole')
    enriched = kind in ('monopole', 'paired_monopole')
    rhs = put(random((q, mu, nf))*.01)
    kwargs = dict(smooth_rhs=put(random((q, mu, nf))*.02) if paired else None,
        monopole_rhs=put(random((q, mu, 1))*.03) if enriched else None)
    if paired or enriched:
        kwargs.update(interpolation_degree=3, quadrature_order=16)
    provider = radial_coulomb_provider(stub, rhs, radius=r, weights_dr=w,
        lm=np.asarray([[0, 0]]), centers_cart=np.zeros((1, 3)),
        q_plus_G_cart=np.asarray([[[0., 0., 0.], [.2, 0., 0.], [0., .2, 0.]],
                                 [[.1, 0., 0.], [.3, 0., 0.], [.1, .2, 0.]]]),
        cell_volume=1000., fft_points=64, support_radius=1.2, **kwargs)
    c_host = random(provider['rhs'].shape)*.02
    c = put(c_host)
    metadata = provider['aot_metadata']
    assert metadata['schema'] == 'lorrax.radial_coulomb_provider_aot.v1'
    handles = metadata['free_onsite']
    executable = handles['kernel'].lower(c, *handles['table_operands']).compile()
    value = np.asarray(executable(c, *handles['table_operands']))
    np.testing.assert_allclose(value, np.asarray(provider['onsite'](c)), atol=2e-13, rtol=2e-13)
    assert np.max(np.abs(value)) > 1e-9
    if kind == 'plain':
        # Independent double-shell Coulomb kernel, not the prefix-sum owner.
        K = 4*np.pi*(w*r*r)[:, None]*(w*r*r)[None, :]/np.maximum(r[:, None], r[None, :])
        moment = w*r*r
        g = (1-(r/1.2)**2)**6
        g /= g @ moment
        metric = 2*(64/1000.)**2*(K-np.outer(moment, moment)*(g @ K @ g))
        reference = np.einsum('qmr,rt,qnt->qmn', c_host.conj(), metric, c_host)
        np.testing.assert_allclose(value, reference, atol=2e-13, rtol=2e-13)
    fourier = metadata['fourier']
    assert fourier['table_shapes'] == ((q, 1, 2, len(r)), (q, 1, 1, 2), (q, 1, 2))
    # AOT lower consumes explicit operands; no distributed closure is wrapped.
    tables = tuple(jax.ShapeDtypeStruct(s, np.dtype(d), sharding=NamedSharding(mesh, p))
        for s, d, p in zip(fourier['table_shapes'], fourier['table_dtypes'], fourier['table_specs']))
    assert fourier['kernel'].lower(c, *tables, *fourier['table_operands']).compile().memory_analysis() is not None


def test_positive_aot_completion_and_exact_moments_keep_public_actions():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from runtime.padding import padded_axis
    from isdf.positive_charge_metric import positive_radial_coulomb_provider

    mesh = Mesh(np.asarray(jax.devices()).reshape(1, 1), ('x', 'y'))
    qspec = P(('x', 'y'), None, None)
    put = lambda a: device_put_process_local(np.asarray(a), NamedSharding(mesh, qspec))
    r = np.linspace(.08, 1.2, 8)
    edges = np.r_[0., (r[:-1] + r[1:])/2, 1.2]
    q = np.asarray([[0., 0., 0.], [.25, 0., 0.]])
    g = np.zeros((2, 3, 3))
    g[:, 0, 1] = 1
    g[:, 1, 2] = 1
    reciprocal = np.eye(3)*(2*np.pi/10)
    centers = np.zeros((1, 3))
    lm = np.asarray([[0, 0]])
    geometry = dict(reciprocal_rows_bohr_inverse=reciprocal,
        atom_centres_bohr=centers, operator_q_fractional=q,
        cell_volume_bohr3=1000., support_radius_bohr=1.2)
    axis = padded_axis(1, mesh, name='unit-test periodic moment',
        specs=((P(None, 'x', 'y'), 1), (P(None, 'x', 'y'), 2)))
    # Periodic V is outside this local-completion unit. This inert admitted
    # plan prevents an unrelated vendor GEMM allocation; it is never called.
    from isdf.coulomb_fourier_cache import periodic_compensation_contract
    _,schema,model=periodic_compensation_contract(geometry)
    cache = dict(metadata=dict(geometry=geometry, lm=lm,schema=schema,model=model), moment_axis=axis,
                 file_sha256='0'*64)
    plan = dict(geometry=geometry, moment_axis=axis,
        action=lambda _: pytest.fail('periodic action is outside this test'),
        receipt=dict(cache_file_sha256=cache['file_sha256']))
    stub = SimpleNamespace(mesh=mesh, solver_kind='rank_truncate',
        store=SimpleNamespace(Q=2, Q_pad=2, mu_pad=3, g_tile=2,
                              g_axis=SimpleNamespace(logical=3)),
        ngk_per_q=np.asarray([3, 3]), mu_basis=object())
    rng = np.random.default_rng(7163)
    random = lambda s: (rng.normal(size=s)+1j*rng.normal(size=s))*.02
    provider = positive_radial_coulomb_provider(stub, put(random((2, 3, 8))),
        monopole_rhs=put(random((2, 3, 1))), radius=r, weights_dr=np.diff(edges),
        lm=lm, centers_cart=centers, q_frac=q, gvec_components=g,
        q_plus_G_cart=(g.transpose(0, 2, 1)+q[:, None])@reciprocal,
        cell_volume=1000., fft_points=64, support_radius=1.2,
        minimum_atom_image_distance=10., body_cutoff_ry=320.,
        periodic_cache=cache, periodic_plan=plan,
        interpolation_degree=3, quadrature_order=16)
    host = random((2, 3, 9))
    coefficients = put(host)
    metadata = provider['aot_metadata']
    free = metadata['free_onsite']
    kernel = free['kernel'].lower(coefficients, *free['table_operands']).compile()
    value = kernel(coefficients, *free['table_operands'])
    free_host = np.asarray(value).copy()
    mean = metadata['mean_completion']
    complete = mean['kernel'].lower(value, coefficients, *mean['table_operands']).compile()
    actual = np.asarray(complete(value, coefficients, *mean['table_operands']))
    expected = np.asarray(provider['onsite'](coefficients))
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_array_equal(actual[1], free_host[1])
    assert np.max(np.abs(actual[0]-free_host[0])) > 1e-12
    moments = metadata['periodic_moments']
    moment_kernel = moments['kernel'].lower(coefficients, *moments['table_operands']).compile()
    actual_moments = np.asarray(moment_kernel(coefficients, *moments['table_operands']))
    np.testing.assert_allclose(actual_moments, host[..., 8:], atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(actual_moments,
        np.asarray(provider['periodic_moment_rows'](coefficients)), atol=2e-13, rtol=2e-13)
