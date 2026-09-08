"""Independent dense full-order solve for the mandatory unweighted BT gate.

Reuse the authenticated parent reader and square-root balancing pipeline;
replace only the reduction observation with a full-state dense resolvent.
"""
import json
import time
import numpy as np
import balance
import reductions


def full_dense(a, B, C, balancing, context):
    """Evaluate Cb(sI-Ab)^-1 Bb directly against the squared-pole parent.

    Parameters
    ----------
    a : jax.Array, (d,)
        Diagonal stable generator in Ry, on the assigned single GPU.
    B, C : jax.Array
        Input (d,n) and output (n,d) physical factors.
    balancing : tuple
        Cholesky roots and singular triplet of Lq†Lp, all on that GPU.
    context : dict
        Authenticated parent factors, z samples in Ry, and output receipt.
    """
    import jax
    import jax.numpy as jnp
    lp,lq,u,h,vh=balancing
    adj=lambda x:x.conj().T
    T=(lp@adj(vh))/jnp.sqrt(h)[None,:]
    Ti=(adj(u)@adj(lq))/jnp.sqrt(h)[:,None]
    Ar=Ti@(a[:,None]*T)
    Br=Ti@B
    Cr=C@T
    jax.block_until_ready((Ar,Br,Cr))
    factor,om=context['factor'],context['omega']
    errors=[];t0=time.monotonic()
    for zi in context['z']:
        matrix=(-1j*zi-balance.ETA)*jnp.eye(len(a))-Ar
        wr=Cr@jnp.linalg.solve(matrix,Br)
        exact=(factor/(zi**2-om**2)[None,:])@adj(factor)
        error=float(jnp.linalg.norm(wr-exact)/jnp.linalg.norm(exact))
        errors.append(error)
        print(json.dumps(dict(stage='full_dense_point',q=context['receipt']['q'],point=len(errors),error=error)),flush=True)
    receipt=dict(q=context['receipt']['q'],jobid=context['receipt']['jobid'],stepid=context['receipt']['stepid'],
                 state_order=len(a),weight=context['receipt']['weight'],relative_errors=errors,maximum=max(errors),
                 threshold=1e-10,status='PASS' if max(errors)<=1e-10 else 'FAIL',
                 elapsed_seconds=time.monotonic()-t0,script_sha256=balance.sha(__file__),
                 route='explicit dense Ar=Ti diag(a) T; dense solve(sI-Ar,TiB); no inherited diagonal resolvent')
    (context['out']/'full_dense_roundtrip.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert max(errors)<=1e-10,receipt


if __name__=='__main__':
    reductions.reduce_models=full_dense
    balance.main()
