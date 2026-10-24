"""FEAST cores share programs across Hamiltonians and release caller payloads."""
import gc
import inspect
from unittest.mock import patch
import weakref

import jax
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P


def _dense_matvec(x,a,*unused):
    return a@x


def check_feast_runner_lifetime(mesh):
    import jax.numpy as jnp
    from bse import bse_feast as owner
    from common.collectives import device_put_process_local,gather_to_host
    jax.config.update('jax_enable_x64',True)
    put=lambda a,spec:device_put_process_local(np.asarray(a),NamedSharding(mesh,spec))
    h1=np.asarray([[1.,.2],[.2,2.]],complex)
    h2=np.asarray([[1.6,.35],[.35,2.8]],complex)
    x=np.asarray([[1.,-.3],[.4,.6]],complex)
    z=np.asarray([.8+.7j,2.4+.6j])
    weights=np.asarray([.2+.1j,-.1+.2j])

    def payload(h):
        return dict(psi_c_X=put(h,P('x','y')),
            **{name:put(np.ones((2,2),complex),P('x','y'))
               for name in ('psi_c_Y','psi_v_X','psi_v_Y','W_R','V_q0','M')},
            eps_c=put(np.ones(2),P('x')),eps_v=put(np.zeros(2),P('x')))

    def expected(h):
        return np.stack([sum(w*np.linalg.solve(point*np.eye(2)-h,row)
                            for point,w in zip(z,weights)) for row in x])

    with patch.object(owner,'_FEAST_RUNNER_CACHE',{}):
        a,b=payload(h1),payload(h2)
        refs=[weakref.ref(v) for data in (a,b) for v in data.values()]
        options=dict(n_quad=2,n_ritz=2,max_iter=4,tol=1e-12,ry_to_ev=1.,
                     dtype=jnp.complex128,use_conjugate_symmetry=False)
        first=owner._get_feast_runner(_dense_matvec,a,**options)
        second=owner._get_feast_runner(_dense_matvec,b,**options)
        assert first is not second and first.core is second.core
        assert len(owner._FEAST_RUNNER_CACHE)==1
        core_closure=inspect.getclosurevars(first.core._fun).nonlocals
        assert 'data' not in core_closure and 'operands' not in core_closure
        assert not any(isinstance(v,jax.Array) for v in core_closure.values())
        args=(put(x,P(None,'x')),put(z,P()),put(weights,P()))
        out1,_=first(*args,put(np.diag(h1),P('x')))
        out2,_=second(*args,put(np.diag(h2),P('x')))
        replay,_=first(*args,put(np.diag(h1),P('x')))
        jax.block_until_ready((out1,out2,replay))
        values1,values2=map(lambda v:np.asarray(gather_to_host(v)),(out1,out2))
        np.testing.assert_allclose(values1,expected(h1),atol=3e-12,rtol=3e-12)
        np.testing.assert_allclose(values2,expected(h2),atol=3e-12,rtol=3e-12)
        np.testing.assert_array_equal(gather_to_host(replay),values1)
        assert not np.allclose(values1,values2)
        del a,b,first,second
        gc.collect()
        assert all(ref() is None for ref in refs)
        return dict(shared_core=True,payloads_released=True,
            core_freevars=sorted(core_closure),
            max_abs_error=max(float(np.max(abs(values1-expected(h1)))),
                              float(np.max(abs(values2-expected(h2))))))


def test_feast_core_shares_structure_without_retaining_hamiltonians():
    for side in ((1,2) if len(jax.devices('cpu'))>=4 else (1,)):
        mesh=Mesh(np.asarray(jax.devices('cpu')[:side*side]).reshape(side,side),('x','y'))
        check_feast_runner_lifetime(mesh)
