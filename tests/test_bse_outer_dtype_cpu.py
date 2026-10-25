"""The native BSE outer plan must not capture unsupported operand dtypes."""
import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()

from pathlib import Path
import pytest


def test_c64_outer_reference_composition_remains_available():
    """The native-only restriction does not change the public XLA composition."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ffi import fft as F

    rng = np.random.default_rng(731)
    rand = lambda shape: (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(np.complex64)
    left, right, w = rand((4, 1, 3, 2)), rand((4, 2, 1, 5)), rand((3, 5, 4))
    with F.xla_reference():
        conv = F.make_local_kconv_klead_outer(None, (2, 2, 1), norm='ortho')
        got = jax.jit(lambda l, r, v: conv(l, r, v, conj_r=True))(
            jnp.asarray(left), jnp.asarray(right), jnp.asarray(w))
    t = np.einsum('kaxK,kKby->kaxby', left, right.conj()).reshape(2, 2, 1, 1, 3, 1, 5)
    wr = np.moveaxis(w, -1, 0)[:, None, :, None, :].reshape(t.shape)
    expected = np.fft.fftn(np.fft.ifftn(t, axes=(0, 1, 2), norm='ortho')*wr,
                          axes=(0, 1, 2), norm='ortho').reshape(4, 1, 3, 1, 5)
    assert got.dtype == jnp.complex64
    np.testing.assert_allclose(np.asarray(got), expected, rtol=3e-5, atol=3e-6)


def test_outer_dtype_owner_and_c64_production_stage(monkeypatch):
    import runpy
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from ffi import fft as F
    from bse import bse_stack_matvec as B
    from common.collectives import gather_to_host
    calls=[];token=object()
    route=lambda rank:(calls.append(rank) or token)
    monkeypatch.setattr(B,'klead_outer_decode_refusal',lambda *a:'storedU test')
    assert B._w_route(route,4,None,(16,16,1),4,operand_dtypes=(jnp.complex128,)*4)==('outer',token,None)
    assert calls==[4]
    for dtypes in ((jnp.complex64,)*4,(jnp.complex128,jnp.complex64,jnp.complex128,jnp.complex128),()):
        assert B._w_route(route,4,None,(16,16,1),4,operand_dtypes=dtypes)==('xla',None,None)
    assert calls==[4]
    if jax.device_count()<4:pytest.skip('four virtual devices required for C64 production stage')
    mesh=Mesh(np.asarray(jax.devices()[:4]).reshape(2,2),('x','y'))
    rng=np.random.default_rng(719);grid=(2,2,1);nk=4;nc=nv=4;ns=2;mu=12
    rand=lambda shape:(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)/np.float32(4)
    pc,pv=rand((nk,nc,ns,mu)),rand((nk,nv,ns,mu));x=rand((3,nc,nv,nk));w=rand((mu,mu)+grid)
    put=lambda a,s:jax.device_put(np.asarray(a),NamedSharding(mesh,s))
    op=(put(pc,P(None,None,None,'x')),put(pc,P(None,None,None,'y')),put(pv,P(None,None,None,'x')),put(pv,P(None,None,None,'y')),put(np.zeros((nk,nc),np.float32),P()),put(np.zeros((nk,nv),np.float32),P()),put(w,P('x','y',None,None,None)),put(np.zeros((mu,mu),np.complex64),P('x','y')),put(np.zeros((nk,nc,nv,mu),np.complex64),P(None,'x','y',None)))
    xdev=put(x,P(None,'x','y',None))
    def forbidden(rank):raise AssertionError('C64 production stage queried unsupported native outer')
    monkeypatch.setattr(B,'_make_outer_router',lambda *a:forbidden)
    refs=runpy.run_path(str(Path(__file__).with_name('test_bse_w_blocks.py')))
    a=refs['_w_reference'](x,pc,pv,w,grid)
    b=refs['_w_reference_B'](x.conj(),pc,pv,pc,pv,w,grid)
    assert np.max(abs(a))>1e-5
    with F.xla_reference():
        tda=B.build_bse_stack_matvec(mesh,*grid)
        got=tda(xdev,*op)
        assert got.dtype==jnp.complex64
        np.testing.assert_allclose(gather_to_host(got),-a,rtol=3e-5,atol=3e-6)
        for fuse in (False,True):
            pair=B.build_bse_stack_pair_matvec(mesh,*grid,fuse=fuse)
            got=pair(xdev,put(np.asarray(1,np.complex64),P()),*op)
            assert got.dtype==jnp.complex64
            np.testing.assert_allclose(gather_to_host(got),-a-b,rtol=3e-5,atol=3e-6)
