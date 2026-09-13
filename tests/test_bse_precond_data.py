"""BSE preconditioner data remains an explicit whole-solver operand."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bse.bse_davidson_helpers import bse_diagonal_precond


@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('olsen', [False, True])
def test_bse_preconditioner_explicit_data_matches_callable(exact, olsen):
    rng=np.random.default_rng(22)
    ec=jnp.asarray([[1.,1.7],[1.1,1.8]])
    ev=jnp.asarray([[-.3,-.1],[-.4,-.2]])
    diagonal=ec.T[:,None,:]-ev.T[None,:,:]+.04 if exact else None
    pc=bse_diagonal_precond(ec,ev,diag_H=diagonal,olsen=olsen)
    r=jnp.asarray(rng.normal(size=(2,2,2,2))+1j*rng.normal(size=(2,2,2,2)))
    x=jnp.asarray(rng.normal(size=(2,2,2,2))+1j*rng.normal(size=(2,2,2,2)))
    e=jnp.asarray([.2,.4])
    exe=pc.apply.lower(pc.data,r,e,x).compile()
    np.testing.assert_array_equal(exe(pc.data,r,e,x),pc(r,e,x))
    shifted=(ec+.1,ev,None if diagonal is None else diagonal+.1)
    got=exe(shifted,r,e,x)
    reference=bse_diagonal_precond(shifted[0],shifted[1],diag_H=shifted[2],olsen=olsen)
    np.testing.assert_array_equal(got,reference(r,e,x))
    assert not np.allclose(got,pc(r,e,x))
