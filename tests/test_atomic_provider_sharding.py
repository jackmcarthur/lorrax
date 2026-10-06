"""Actual radial provider callback ownership, including negative layouts."""
from pathlib import Path
import json
import os


def check_atomic_provider_sharding(runtime):
    from types import SimpleNamespace
    import numpy as np
    import jax
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from isdf.atomic_coulomb import radial_coulomb_provider
    from isdf.zeta_mubatch import _require_q_owned
    from runtime.padding import padded_axis

    mesh = runtime.mesh
    assert mesh.size == 4
    rng = np.random.default_rng(32044)
    Q,Qp,mu,ng,gt,nr = 3,4,8,6,4,8
    lm = np.array([(l,m) for l in range(2) for m in range(-l,l+1)],np.int32)
    rhs = rng.normal(size=(Qp,mu,len(lm)*nr))+1j*rng.normal(size=(Qp,mu,len(lm)*nr))
    rhs[Q:] = 0.
    qsh = NamedSharding(mesh,P(('x','y'),None,None))
    put = lambda a,s: device_put_process_local(np.asarray(a),NamedSharding(mesh,s))
    coeff = put(rhs,P(('x','y'),None,None))
    vectors = rng.normal(size=(Q,ng,3))
    counts = np.array([6,4,5])
    zeta = SimpleNamespace(mesh=mesh,ngk_per_q=counts,
        store=SimpleNamespace(Q=Q,Q_pad=Qp,mu_pad=mu,g_tile=gt,
                              g_axis=padded_axis(ng,gt,name='provider ownership G')))
    radius = np.geomspace(1e-5,.9,nr)
    weights = np.full(nr,.9/nr)
    signatures = []
    for degree,order in ((None,None),(3,16)):
        provider = radial_coulomb_provider(zeta,coeff,radius=radius,weights_dr=weights,lm=lm,
            centers_cart=np.array([[.1,.2,.3]]),q_plus_G_cart=vectors,
            cell_volume=80.,fft_points=512,support_radius=1.,
            interpolation_degree=degree,quadrature_order=order,fourier_points=4097)
        for t in range(2):
            delta,comp = provider['fourier_tile'](t,coeff)
            for value,label in ((delta,'delta'),(comp,'compensation')):
                _require_q_owned(value,mesh,(Qp,mu,gt),name=label)
                assert value.sharding.is_equivalent_to(qsh,ndim=3)
                host = gather_to_host(value)
                assert np.max(abs(host[Q:])) == 0.
                assert np.max(abs(host[:Q]*
                    (t*gt+np.arange(gt)[None,:] >= counts[:,None])[:,None,:])) == 0.
            signatures.append(dict(degree=degree,tile=t,shape=list(delta.shape),
                                   dtype=str(delta.dtype),sharding=str(delta.sharding)))
        onsite = provider['onsite'](coeff)
        _require_q_owned(onsite,mesh,(Qp,mu,mu),name='onsite')
    bad = put(np.ones((Qp,mu,gt),np.complex128),P(None,'x','y'))
    try:
        _require_q_owned(bad,mesh,(Qp,mu,gt),name='wrong-layout negative control')
    except ValueError as exc:
        assert 'got shape' in str(exc),str(exc)
    else:
        raise AssertionError('face-owned provider operand passed the q-owner guard')
    result = dict(callback_signatures=signatures,wrong_layout_refused=True,
                  physical_padding_exact=True,full_and_collocation_modes=True)
    if jax.process_index() == 0:
        print(json.dumps(result),flush=True)
        if os.environ.get('ATOMIC_PROVIDER_REPORT'):
            Path(os.environ['ATOMIC_PROVIDER_REPORT']).write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime = initialize_communicator_stack()
    def main():
        check_atomic_provider_sharding(runtime)
        return 0
    run_main_and_finalize(main)
