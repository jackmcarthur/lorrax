"""Coarse-grid closure for complex bra/ket and path-padding conventions."""
import numpy as np
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

from runtime import initialize_communicator_stack
from common.collectives import gather_to_host
from bandstructure.orbital import interpolate_band_operator


def test_complex_operator_recovered_on_source_grid():
    mesh = initialize_communicator_stack().mesh
    rng = np.random.default_rng(35)
    nk, nb, rank = 8, 4, 8
    c = np.stack([np.linalg.qr(rng.normal(size=(rank,nb))
                  +1j*rng.normal(size=(rank,nb)))[0].T for _ in range(nk)])
    v = rng.normal(size=(3,nk,nb,nb))+1j*rng.normal(size=(3,nk,nb,nb))
    v = (v+v.swapaxes(-1,-2).conj())/2
    k = np.stack(np.meshgrid(*([np.arange(2)/2]*3), indexing='ij'), axis=-1).reshape(-1,3)
    cp = np.pad(c.transpose(0,2,1), ((0,(-nk)%mesh.size),(0,0),(0,0)))
    with mesh:
        got = interpolate_band_operator(
            jax.device_put(v, NamedSharding(mesh,P(None,None,'x','y'))),
            jax.device_put(c, NamedSharding(mesh,P(None,None,'x'))),
            jax.device_put(cp,NamedSharding(mesh,P(('x','y'),None,None))),
            k, (2,2,2), mesh)
    np.testing.assert_allclose(gather_to_host(got)[:nk], v.transpose(1,0,2,3),
                               atol=3e-13, rtol=3e-13)
