"""Independent source-frozen normalization, transpose, padding and native-Sigma gates."""
import numpy as np
import pytest


def test_signed_conversion_swaps_independent_factors_and_compacts_odd_counts():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.shared_pole_reference import _signed_reference_faces
    jax.config.update('jax_enable_x64', True)
    devs = np.asarray(jax.devices('cpu'))
    mesh = Mesh(devs[:4].reshape(2, 2) if len(devs) >= 4 else devs[:1].reshape(1, 1), ('x', 'y'))
    rng = np.random.default_rng(441)
    counts, minus = np.array([3, 1, 2]), np.array([0, 2, 1])
    l = rng.normal(size=(3, 4, 1, 4))+1j*rng.normal(size=(3, 4, 1, 4))
    r = rng.normal(size=l.shape)+1j*rng.normal(size=l.shape)
    lam = np.array([[.16, .49, 1., 1.], [.25, 1., 1., 1.], [.36, .81, 1., 1.]])
    for q, k in enumerate(counts):
        l[q, ..., k:] = r[q, ..., k:] = 0
    put = lambda a, spec: jax.device_put(a, NamedSharding(mesh, spec))
    a, b, poles, k = _signed_reference_faces((put(l, P(None,'x',None,'y')),
        put(r, P(None,'y',None,'x')), put(lam, P())), counts, minus, mesh_xy=mesh, width=6)
    a, b, poles, k = map(np.asarray, (a, b, poles, k))
    np.testing.assert_array_equal(k, [6, 3, 3])
    for q in range(3):
        qm = minus[q]
        for z in [.23+.17j, -.4+.31j]:
            expected = np.zeros((4,4), complex)
            for j in range(counts[q]):
                o = np.sqrt(lam[q,j])
                expected += np.outer(l[q,:,0,j], r[q,:,0,j].conj())/(2*o*(z-o))
            for j in range(counts[qm]):
                o = np.sqrt(lam[qm,j])
                expected -= np.outer(l[qm,:,0,j], r[qm,:,0,j].conj()).T/(2*o*(z+o))
            actual = (a[q,:,0,:]/(z-poles[q]))@b[q,:,0,:].conj().T
            np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
        assert np.all(a[q,...,k[q]:] == 0) and np.all(b[q,...,k[q]:] == 0)
        # Literal greater/lesser branch amplitudes at complex time.
        t = .19-.07j
        pos = poles[q].real > 0
        neg = poles[q].real < 0
        greater = (a[q,:,0,pos]*np.exp(-1j*poles[q,pos]*t)[:,None]).T@b[q,:,0,pos].conj()
        lesser = -(a[q,:,0,neg]*np.exp(-1j*poles[q,neg]*t)[:,None]).T@b[q,:,0,neg].conj()
        for actual, parent, sign in [(greater,q,1), (lesser,qm,-1)]:
            expected = sum(np.outer(l[parent,:,0,j], r[parent,:,0,j].conj())
                *np.exp(-1j*sign*np.sqrt(lam[parent,j])*t)/(2*np.sqrt(lam[parent,j]))
                for j in range(counts[parent]))
            if sign < 0:
                expected = expected.T
            np.testing.assert_allclose(actual, expected, rtol=4e-14, atol=4e-14)


@pytest.mark.parametrize('with_head', [False, True])
def test_positive_store_adapter_native_full_rho_sigma_matches_literal(tmp_path, monkeypatch, with_head):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from dataclasses import replace
    from file_io.shared_pole_store import write_shared_pole_model
    from gw.shared_pole_reference import source_frozen_ordered_model
    from gw.shared_pole_recipe import shared_real_pole_v1_r3b
    from gw.head_correction import HeadSigmaDiag
    from common.units import RYD_TO_EV
    from ffi import fft as F
    from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture
    from tests._full_microscopic_oracle import density_vertices, retarded_reference_pairs, branch_coefficients
    from tests.tdagw.source_frozen_saved_state import evaluate_source_frozen_snapshot
    from tdagw.collision import branch_operators

    c = _native_port_fixture(monkeypatch)
    rng, mesh, nk, nmu, perm, meta = (c[k] for k in ('rng','mesh','nk','nmu','perm','meta'))
    counts = np.array([3,1,2])
    omega = np.array([[.41,.65,.93,1.], [.49,1.,1.,1.], [.51,.87,1.,1.]])
    b = .08*(rng.normal(size=(nk,nmu,1,4))+1j*rng.normal(size=(nk,nmu,1,4)))
    for q, k in enumerate(counts):
        b[q,...,k:] = 0
    put = lambda a: jax.device_put(np.ascontiguousarray(a), NamedSharding(mesh,P(None,'x',None,'y')))
    identity = dict(iteration_id='source-1', hamiltonian='source-H', energies='source-E',
                    occupations='source-rho', wavefunctions='source-psi', centroids='source-mu')
    meta.shared_pole_k_capacity = {'None':4}
    header = write_shared_pole_model(str(tmp_path/'positive.h5'), put(b[:,perm]),
        jax.device_put(omega**2,NamedSharding(mesh,P())), counts, q_span=(0,nk),
        meta=meta,tables=c['tables'], recipe=dict(operator_realization=shared_real_pole_v1_r3b['operator_realization']),
        receipts=dict(identity=identity),ordered=True)
    positive = dict(path=str(tmp_path/'positive.h5'),identity=identity,digest=header['digest'])
    ordered = source_frozen_ordered_model(positive, meta, mesh_xy=mesh,
        tables=c['tables'],path=str(tmp_path/'ordered.h5'))
    assert ordered['representation'] == 'ordered_complex'
    assert ordered['source_frozen_receipt']['refreshed_screening'] is False
    rotation, rho_a = [], []
    for _ in range(nk):
        u,_ = np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
        v,_ = np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
        rotation.append(u); rho_a.append(v@np.diag([.19,.83])@v.conj().T)
    rotation, rho_a = np.asarray(rotation), np.asarray(rho_a)
    energy = c['energy'].copy()
    energy[:,0] += .01
    energy[:,1:] += [.01,.02]
    config = c['fixture']._config(np.array([-2.,2.]))
    config.sigma.regularization_ev = .14*RYD_TO_EV
    config.sigma.quadrature_eps = 2e-7
    head = None
    if with_head:
        head = HeadSigmaDiag.build(omega_grid_ry=np.array([-2.,2.]),enk_ry=energy,
            occupations=c['occ'],efermi_ry=0.,poles_ry=np.array([.73-.09j]),
            residues_ry=np.array([.012]),cell_volume=8.,nk_tot=nk)
    source = dict(wfns=c['wfns'], meta=meta, config=config, head=head, linalg='distributed')
    prepared = dict(nb=3, nk=nk, mesh=mesh, efermi_ry=0.)
    with F.xla_reference():
        result = evaluate_source_frozen_snapshot(source, prepared,
            dict(U=rotation,e=energy[:,1:],rho=rho_a), b_lo=1, model=ordered,
            print_fn=lambda *a,**k:None)
    got = result['raw']
    psi = c['psi'].copy()
    psi[:,1:] = np.einsum('kmsu,kmi->kisu',psi[:,1:],rotation)
    rho = np.zeros((nk,3,3),complex); rho[:,0,0]=1.; rho[:,1:,1:]=rho_a
    full = {s:np.zeros((2,nk,2,2),complex) for s in ('cond','val')}
    active = {s:np.zeros_like(x) for s,x in full.items()}
    for q in range(nk):
        qm = (-q)%nk
        poles = np.r_[omega[q,:counts[q]],-omega[qm,:counts[qm]]]
        residue = [np.outer(b[q,:,0,j],b[q,:,0,j].conj())/(2*omega[q,j]) for j in range(counts[q])]
        residue += [-np.outer(b[qm,:,0,j],b[qm,:,0,j].conj()).T/(2*omega[qm,j]) for j in range(counts[qm])]
        partner = (np.arange(nk)-q)%nk
        vertex = density_vertices(psi,partner,[1,2])
        f = retarded_reference_pairs(vertex,energy[partner],rho[partner],energy[:,1:],poles,np.asarray(residue),.14)
        a = retarded_reference_pairs(vertex[:,:,1:],energy[partner,1:],rho[partner,1:,1:],energy[:,1:],poles,np.asarray(residue),.14)
        for i,s in enumerate(('cond','val')):
            full[s] += f[i]/nk; active[s] += a[i]/nk
    if with_head:
        h = replace(head,enk_ry=energy[:,1:],occupations=np.ones((nk,2)))
        for slot in range(2):
            for i in range(2):
                x = np.repeat(energy[:,i+1:i+2],2,axis=1) if slot==0 else energy[:,1:]
                val = rho_a[:,i]*np.conj(h.on_shell(x)[0])
                cond = (np.eye(2)[None]-rho_a)[:,i]*replace(h,occupations=np.zeros((nk,2))).on_shell(x)[0]
                for target in (full,active):
                    target['val'][slot,:,i] += val
                    target['cond'][slot,:,i] += cond
    for s in full:
        np.testing.assert_allclose(got['retarded_pair'][s],full[s],rtol=2e-5,atol=4e-8)
    ga,da,_ = branch_operators(got['active_branch_pair']['cond'],got['active_branch_pair']['val'])
    _,_,mh = branch_operators(got['branch_pair']['cond'],got['branch_pair']['val'])
    expected_active,expected_full = branch_coefficients(active['cond'],active['val']),branch_coefficients(full['cond'],full['val'])
    for actual,expected in [(ga,expected_active['gamma']),(da,expected_active['injection']),(mh,expected_full['m_h'])]:
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=4e-8)
    for name, expected in [('gamma',ga),('injection',da),('m_h',mh)]:
        np.testing.assert_allclose(result[name],rotation@expected@rotation.conj().swapaxes(-1,-2),
                                   rtol=2e-5,atol=4e-8)
    with pytest.raises(RuntimeError, match='FileExistsError'):
        source_frozen_ordered_model(positive,meta,mesh_xy=mesh,tables=c['tables'],path=str(tmp_path/'ordered.h5'))
