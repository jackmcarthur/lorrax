"""The streamed exchange refit must not materialize zeta or V on a host."""
from types import SimpleNamespace
import sys

import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('side,ns', [(2, 2), (3, 1)])
def test_streamed_refit_has_no_zeta_or_matrix_host_gather(monkeypatch, side, ns):
    import distrib_la
    from bse import vq_interp as V
    from common import collectives
    from ffi import fft as F
    from isdf import galerkin as G
    if jax.device_count() < side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    # The local reference uses the canonical exact JAX eigensolver; the
    # native multi-process gate separately uses the production providers.
    original_plan = distrib_la.plan
    monkeypatch.setattr(distrib_la, 'plan', lambda op, m, **kw:
        original_plan(op, m, **dict(kw, backend='off')) if op == 'eigh'
        else original_plan(op, m, **kw))
    nodes, nr, nk, logical_mu = 3, 13, 1, 13
    nb = ns*nodes
    rng = np.random.default_rng(954+side)
    q = np.linalg.qr(rng.normal(size=(nb, nb))+1j*rng.normal(size=(nb, nb)))[0]
    continuation = rng.normal(size=(nodes, nr))+1j*rng.normal(size=(nodes, nr))
    continuation[:, :nodes] = np.eye(nodes)
    physical_c = q.T[None]
    values = np.einsum('kbsn,nr->kbsr', physical_c.reshape(nk, nb, ns, nodes), continuation)
    carrier = G._rank_carrier(nb, mesh, nb)
    coeff = jax.device_put(np.pad(physical_c, ((0,0),(0,0),(0,carrier-nb))),
        NamedSharding(mesh, P(None,None,('x','y'))))
    basis = G.GalerkinBasis(coeff,
        G.NodalInterp(coeff, np.zeros((nodes,3)), np.eye(3), 3, ns),
        nb, (0,nb), tuple((i,0,0) for i in range(nodes)), refit_rcond=1e-7)
    bc = -(-nb//side)*side

    class Source:
        band_chunk_ranges = ((0,nb),)
        band_chunk_carrier = bc

        def iter_rchunk_bandwise(self, r0, r1, *, product_r_spec):
            rpad = -(-(r1-r0)//(side*side))*(side*side)
            a = np.pad(values[:,:,:,r0:r1], ((0,0),(0,bc-nb),(0,0),(0,rpad-(r1-r0))))
            yield (0,nb), jax.device_put(a, NamedSharding(mesh, product_r_spec))

    from runtime.padding import padded_mu_extent
    mu_pad = padded_mu_extent(logical_mu, mesh)
    psi_mu = np.pad(values, ((0,0),(0,0),(0,0),(0,mu_pad-logical_mu)))
    psi = jax.device_put(psi_mu, NamedSharding(mesh, P(None,None,None,('x','y'))))
    zx = dict(nk=nk,nb=nb,ns=ns,n_mu=mu_pad,n_mu_logical=logical_mu,
        nx=nr,ny=1,nz=1,bvec=np.diag([1.,100.,100.]),zeta_cutoff=36.01,psi=psi)
    rst = dict(rank=carrier,zeta_solve=('rank_truncate',1e-12),
        sym=SimpleNamespace(find_qpoint_index=lambda k:0), k_frac=np.zeros((1,3)),
        psi_source=Source(),basis=basis,meta=SimpleNamespace(nspinor=ns,n_rtot=nr),
        r_chunk_ranges=((0,4),(4,nr)),window_abs=(0,nb),galerkin_rel=0.,gflat_chunk_size=0,
        v_on_set=lambda q,gs:1.+.1*gs[0]**2)

    def no_host(*a, **kw):
        raise AssertionError('production refit attempted a zeta/V host gather')

    # The former production tail calls this exact owner. It must fail the
    # baseline even when all virtual devices are addressable by this process.
    monkeypatch.setattr(V, '_to_host', no_host)
    original_gather = collectives.gather_to_host
    original_get = jax.device_get

    def guarded(fn):
        def call(a, *args, **kw):
            shape = getattr(a, 'shape', ())
            if len(shape)>=2 and min(shape[-2:])>=logical_mu:
                no_host()
            return fn(a, *args, **kw)
        return call

    monkeypatch.setattr(collectives, 'gather_to_host', guarded(original_gather))
    monkeypatch.setattr(jax, 'device_get', guarded(original_get))
    target = 14 if side == 2 else 15
    with F.xla_reference():
        result = V.refit_vq(zx,rst,np.zeros(3),mesh,m_leg='stored',n_mu_out=target,
                            log_fn=lambda *_:None)
    assert result.sharding.is_equivalent_to(NamedSharding(mesh,P('x','y')),2)
    # Literal charge-fit equations, independent of the sharded factor/Gram:
    # C=X^H X, zeta=C^+ X^H rho, followed by its explicit Fourier sum.
    x = np.einsum('kmsu,knsu->kmnu', values.conj(), values).reshape(-1,nr)
    c = x.conj().T @ x
    zeta_r = np.linalg.pinv(c,rcond=1e-12) @ c
    gs = np.arange(-6,7)
    zeta_g = zeta_r @ np.exp(-2j*np.pi*np.arange(nr)[:,None]*gs[None,:]/nr)
    expected = np.zeros((target,target),complex)
    expected[:nr,:nr] = (zeta_g.conj()*(1.+.1*gs**2)) @ zeta_g.T
    np.testing.assert_allclose(np.asarray(result), expected, rtol=3e-8, atol=3e-8)
    np.testing.assert_array_equal(np.asarray(result)[logical_mu:],0)
    np.testing.assert_array_equal(np.asarray(result)[:,logical_mu:],0)

    # The production htransform returns physical targets while the loader
    # uses an all-rank carrier. Exercise that boundary with an exact on-grid
    # reconstruction, retaining the real Galerkin and streamed FFT owners.
    cm = np.pad(physical_c.transpose(0,2,1),((0,0),(0,carrier-nb),(0,0)))
    bundle = SimpleNamespace(psi_rmu_Y=jax.device_put(values,
        NamedSharding(mesh,P())),coeffs_fi=jax.device_put(cm,NamedSharding(mesh,P())))
    monkeypatch.setitem(sys.modules,'bandstructure.bse_setup',
        SimpleNamespace(compute_wfns_fi=lambda **kw:bundle))
    rst.update(ctilde=coeff,B_at_mu=basis.basis_at_nodes,enk_sigma=None,kgrid_co=(1,1,1))
    with F.xla_reference():
        reconstructed = V.refit_vq(zx,rst,np.zeros(3),mesh,m_leg='htransform',
                                   n_mu_out=target,log_fn=lambda *_:None)
    assert reconstructed.sharding.is_equivalent_to(NamedSharding(mesh,P('x','y')),2)
    np.testing.assert_allclose(np.asarray(reconstructed),expected,rtol=3e-8,atol=3e-8)


def test_explicit_host_diagnostics_use_exact_phase_and_refuse_before_io():
    from bse import vq_interp as V
    shape = (3,4,2)
    q = np.array([.13,-.27,.31])
    zx = dict(nx=3,ny=4,nz=2,qfr=np.array([q]))
    coords = np.stack(np.meshgrid(*(np.arange(n)/n for n in shape),indexing='ij'),axis=-1).reshape(-1,3)
    for sign in (-1,+1):
        np.testing.assert_allclose(V._diagnostic_phase(zx,0,sign),
            np.exp(sign*2j*np.pi*(coords@q)),rtol=1e-14,atol=1e-14)

    class Unread:
        def __getitem__(self,key):
            raise AssertionError('oversized diagnostic touched the basis file')
    huge = dict(n_mu=100000,n_rtot=100000,ngkmax=1,ZG=Unread())
    with pytest.raises(ValueError,match='fixed diagnostic limit'):
        V.recon(huge,0)
    with pytest.raises(ValueError,match='fixed diagnostic limit'):
        V.to_sphere(huge,SimpleNamespace(shape=(100000,100000)),0)
