"""OWN3 packet solves versus independent dense solves and incumbent provider."""
from types import SimpleNamespace
import numpy as np
import pytest


def fixture():
    import jax
    import jax.numpy as jnp
    from scipy.linalg import lu_factor
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from isdf.zeta_mubatch import ZStore, ZetaG
    from runtime.padding import padded_axis, PaddedAxis
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    qs = NamedSharding(mesh, P(('x', 'y'), None, None))
    put = lambda a: jax.device_put(np.asarray(a), qs)
    rng = np.random.default_rng(43191)
    random = lambda shape: rng.normal(size=shape) + 1j*rng.normal(size=shape)
    Q, Qp, mu, na, nh, nr = 3, 4, 4, 2, 4, 8
    basis = SimpleNamespace(n_logical=mu)
    zetas, C, rhs, expected = [], [], [], []
    for channel in range(3):
        F = random((Qp, mu, mu))
        matrix = F.conj().swapaxes(1, 2)@F + (channel+1)*np.eye(mu)
        if channel == 1:
            matrix = -matrix  # The Cartesian imaginary-alpha signed LU seam.
        factors = [lu_factor(v) for v in matrix]
        st = ZStore(mesh=mesh, q_axis=PaddedAxis('stream q', Q, Qp, 1),
                    mu_pad=mu, g_axis=padded_axis(4, 4, name='stream G'),
                    b=mu, placement='host')
        z = ZetaG(st, mesh=mesh, L_q=put(np.stack([v[0] for v in factors])),
            lu_piv=jax.device_put(np.stack([v[1] for v in factors]).astype(np.int32),
                NamedSharding(mesh, P(('x', 'y'), None))),
            solver_kind='lu', batched_route='batch_reshard', path='streamed current fixture',
            n_rmu_solve=mu, n_rmu=mu, mu_basis=basis,
            ngk_per_q=np.full(Q, 4), gvec_components=np.zeros((Q, 3, 4),np.int32))
        z.fit_vertex_mu_L = channel+1
        z.fit_augmentation_identity = 'same-field'
        z.fit_current_angular_support_identity = 'same-angular-support'
        z.fit_q_full_indices = np.asarray([0, 5, 1])
        profiles = [random((Qp, mu, na, nh, nr))*.01 for _ in range(2)]
        for values in profiles:
            values[Q:] = 1e7-3e7j
        normal_rhs = [np.einsum('qmn,qnahr->qmahr',matrix,v) for v in profiles]
        correct = [v.copy() for v in profiles]
        for values in correct:
            values[Q:] = 0.
        zetas.append(z);C.append(matrix);rhs.append(normal_rhs);expected.append(correct)
    return SimpleNamespace(mesh=mesh, put=put, zetas=tuple(zetas), C=C,
        rhs=rhs, expected=expected, Q=Q, Qp=Qp, mu=mu, na=na, nh=nh, nr=nr)


def make_stream(f):
    from isdf.current_coefficients import CurrentCoefficientStream
    return CurrentCoefficientStream(f.zetas, atoms=f.na, harmonics=f.nh,
        radial_points=f.nr, radial_packet=4,
        identity='same-field', angular_identity='same-angular-support')


def fill(stream, f):
    for first in (4, 0):  # Deliberate nonmonotone but complete radial census.
        for c in range(3):
            for kind in range(2):
                packet = f.rhs[c][kind][..., first:first+4].reshape(f.Qp, f.mu, -1)
                stream.append(c, kind, f.put(packet), first)
    return stream.finish()


def test_streamed_owned_coefficients_match_independent_signed_dense_solve():
    f = fixture();banks = fill(make_stream(f), f)
    got = banks.authenticate(f.zetas)
    for c in range(3):
        for kind in range(2):
            expected = np.stack([np.linalg.solve(f.C[c][q],
                f.rhs[c][kind][q].reshape(f.mu, -1)) for q in range(f.Q)])
            np.testing.assert_allclose(np.asarray(got[c][kind])[:f.Q], expected,
                                       atol=3e-15, rtol=3e-13)
            assert np.max(abs(np.asarray(got[c][kind])[f.Q:])) == 0.
            assert got[c][kind].shape == banks.shape


@pytest.mark.parametrize('mutation', ['factor', 'q', 'basis', 'field', 'angular'])
def test_sealed_current_coefficients_refuse_changed_fit_owner(mutation):
    import jax.numpy as jnp
    f = fixture();banks = fill(make_stream(f),f);z=f.zetas[1]
    if mutation == 'factor':
        z.L_q = jnp.array(z.L_q) + 0.  # Same values, a different factor owner.
    elif mutation == 'q':
        z.fit_q_full_indices = np.asarray([0,1,5])
    elif mutation == 'basis':
        z.mu_basis = SimpleNamespace(n_logical=f.mu)
    elif mutation == 'field':
        z.fit_augmentation_identity = 'different-field'
    else:
        z.fit_current_angular_support_identity = 'different-angular'
    with pytest.raises(ValueError, match='owner changed'):
        banks.authenticate(f.zetas)


def test_current_stream_refuses_missing_repeated_and_consumed_radial_packets():
    f=fixture();stream=make_stream(f)
    with pytest.raises(ValueError,match='complete nonoverlapping'):
        stream.finish()
    packet=f.rhs[0][0][...,:4].reshape(f.Qp,f.mu,-1)
    stream.append(0,0,f.put(packet),0)
    with pytest.raises(ValueError,match='invalid or repeated'):
        stream.append(0,0,f.put(packet),0)
    stream=make_stream(f);fill(stream,f)
    with pytest.raises(ValueError,match='consumed'):
        stream.append(0,0,f.put(packet),0)
    # A wider transport packet must be quoted before allocation, not force
    # a new unpriced solve compilation during append.
    stream=make_stream(f)
    with pytest.raises(ValueError,match='packet shape'):
        stream.append(0,0,f.put(np.pad(packet,((0,0),(0,0),(0,1)))),0)


def test_separate_current_banks_match_incumbent_radial_fourier_and_onsite():
    from isdf.atomic_breit import radial_breit_providers
    from isdf.zeta_mubatch import _local_coefficient_solve
    f=fixture();banks=fill(make_stream(f),f)
    radius=.8*(np.arange(1,f.nr+1)/f.nr)**3
    lm=np.asarray([(l,m) for l in range(2) for m in range(-l,l+1)],np.int32)
    K=np.tile(np.asarray([[.7,.1,.3],[.2,1.,.4],[-.5,.6,.3],[.1,.2,-1.]]),(f.Q,1,1))
    args=dict(body_metric='compensated',radius=radius,lm=lm,
        centers_cart=np.asarray([[0.,0.,0.],[2.,0.,0.]]),q_plus_G_cart=K,
        cell_volume=30.,fft_points=64,support_radius=.8,
        minimum_atom_image_distance=2.,reciprocal_prefactor=-8*np.pi/30,
        interpolation_degree=3,quadrature_order=8,fourier_points=257,
        field_tile=4,centroid_tile=2)
    delta=tuple(f.put(v[0].reshape(f.Qp,f.mu,-1)) for v in f.rhs)
    smooth=tuple(f.put(v[1].reshape(f.Qp,f.mu,-1)) for v in f.rhs)
    raw=radial_breit_providers(f.zetas,delta,smooth_rhs=smooth,**args)[0]['group_provider']
    streamed=radial_breit_providers(f.zetas,banks,**args)[0]['group_provider']
    paired=tuple(f.put(np.concatenate((v[0].reshape(f.Qp,f.mu,-1),
                                      v[1].reshape(f.Qp,f.mu,-1)),axis=-1)) for v in f.rhs)
    solved=tuple(_local_coefficient_solve(f.mesh,z.solver_kind,z.n_rmu_solve,f.Q)(z.factor,p)
                 for z,p in zip(f.zetas,paired))
    original=raw['prepare'](solved);separate=streamed['prepare'](None)
    for a,b in zip(original['moments'],separate['moments']):
        np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=3e-15,rtol=3e-13)
    for parts in zip(raw['fourier_tile'](0,original),streamed['fourier_tile'](0,separate)):
        for a,b in zip(*parts):
            np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=3e-13,rtol=3e-13)
    pairs=tuple((a,b) for a in range(3) for b in range(a,3))
    for a,b in zip(raw['onsite'](original,pairs),streamed['onsite'](separate,pairs)):
        np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=3e-13,rtol=3e-13)
    with pytest.raises(ValueError,match='authenticated half-bank owner'):
        streamed['prepare'](solved)


def test_deferred_current_requires_actual_charge_close_before_replay():
    from gw.augmentation_current import DeferredCurrentDensity,live_device_bytes
    f=fixture();source=f.put(np.ones((f.Qp,f.mu,4),complex))
    current=dict(identity='same-field',angular_support_identity='same-angular-support',
        q_full_indices=np.asarray([0,5,1]),plan=SimpleNamespace(mesh_xy=f.mesh),phase_prices={})
    charge_state=dict(rhs=source)
    deferred=DeferredCurrentDensity(current,charge_state,source=source,
        coefficients=(f.zetas[0].L_q,),parameters={})
    with pytest.raises(ValueError,match='charge RHS consumption'):
        deferred.after_charge_close()
    del charge_state['rhs']
    with pytest.raises(ValueError,match='actual charge zeta lifetime binding'):
        deferred.after_charge_close()
    charge=SimpleNamespace(mesh=f.mesh,L_q=source,lu_piv=None,local_augmentation={},
                           shell=None,q0_response_rhs=None)
    deferred.bind_charge_zeta(charge)
    with pytest.raises(ValueError,match='actual charge zeta to be closed'):
        deferred.after_charge_close()
    charge.L_q=charge.local_augmentation=None
    deferred.after_charge_close()
    assert current['phase_prices']['charge_closed_before_current_fit'] is True
    assert current['fit_resident_extra_bytes_per_rank']==live_device_bytes(source,f.zetas[0].L_q)
    with pytest.raises(ValueError,match='actual charge zeta once'):
        deferred.bind_charge_zeta(charge)
    current['q_full_indices']=np.asarray([0,1,5])
    with pytest.raises(ValueError,match='binding changed'):
        deferred.after_charge_close()


def test_current_close_releases_banks_held_by_provider_closures():
    f=fixture();banks=fill(make_stream(f),f)
    f.zetas[0].local_augmentation=dict(coefficient_owner=banks)
    f.zetas[0].close()
    assert banks._banks is None and banks._factors == ()
    with pytest.raises(ValueError,match='owner changed'):
        banks.authenticate(f.zetas)


def test_current_stream_refuses_closed_factors_before_compilation_or_banks():
    f=fixture();f.zetas[1].close()
    with pytest.raises(ValueError,match='actual live OWN factor leaves'):
        make_stream(f)
