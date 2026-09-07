"""Typed parent-load convolution against local symmetry transport and direct k sums."""
if __name__ == '__main__':
    from runtime import bootstrap
    bootstrap()

from functools import partial
from itertools import product
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.shard_map import shard_map
from symmetry_maps import open_spin_block_coefficient, unfold_operator_local


def _fixture(ns):
    """Return random parents, owner-local endpoint tables and nontrivial unitary/TR rows."""
    rng = np.random.default_rng(730 + ns)
    shape = (3, ns, 4, ns, 6)
    parents = [rng.normal(size=shape) + 1j*rng.normal(size=shape) for _ in range(2)]
    irr = np.asarray([0, 1, 1, 2], np.int32)
    sym = np.asarray([0, 1, 2, 3], np.int32)
    left = np.asarray([[0,1,0,1], [1,0,1,0], [1,0,0,1], [0,1,1,0]], np.int32)
    right = np.asarray([[0,1,2,0,1,2], [2,0,1,1,2,0],
                        [1,2,0,2,0,1], [2,1,0,0,2,1]], np.int32)
    L = rng.integers(-1, 2, size=(4,4,3)).astype(np.float64)
    R = rng.integers(-1, 2, size=(4,6,3)).astype(np.float64)
    q = rng.uniform(-.5, .5, size=(3,3))
    U = np.stack([np.linalg.qr(rng.normal(size=(ns,ns)) +
                              1j*rng.normal(size=(ns,ns)))[0] for _ in range(4)])
    coef = np.stack([np.asarray(open_spin_block_coefficient(U,a,b))
                     for a in range(ns) for b in range(ns)], axis=1).reshape(4,ns*ns,ns*ns)
    return parents, (irr,sym,left,right,L,R,q,(sym>=2).astype(np.int32),coef), U


def _literal(parents, tables, U):
    """Apply U T[phase D] U† literally, then sum conjugate P(k) times P(k+q)."""
    irr,sym,left,right,L,R,q,trs,_ = tables
    ns=parents[0].shape[1]
    full=[]
    for D in parents:
        child=np.empty((4,ns,4,6,ns),np.complex128)
        for k in range(4):
            op,p=int(sym[k]),int(irr[k])
            for m in range(4):
                for n in range(6):
                    lm=(m//2)*2+left[op,m]
                    rn=(n//3)*3+right[op,n]
                    spatial=np.exp(2j*np.pi*q[p].dot(L[op,m]))*D[p,:,lm,:,rn]
                    spatial=spatial*np.exp(-2j*np.pi*q[p].dot(R[op,n]))
                    if trs[k]: spatial=spatial.conj()
                    child[k,:,m,n,:]=(U[k]@spatial@U[k].conj().T).conj()
        full.append(child)
    return full


def _direct_convolution(full, perm_l, phase_l, perm_r, phase_r):
    """Sum the circular k correlation with the fixed monomial vertices."""
    left,right=full
    out=np.zeros((4,4,6),np.complex128)
    right=right[:,perm_l][:,:,:,:,perm_r]*phase_l[None,:,None,None,None]*phase_r[None,None,None,None,:]
    for q in range(4):
        for k in range(4):
            kq=((k//2+q//2)%2)*2+(k%2+q%2)%2
            out[q]+=np.einsum('amnb,amnb->mn',left[k].conj(),right[kq])
    return out


def _decomposed(parents,tables,U,mesh):
    """Use the existing local scalar transport for every open-spin source block."""
    irr,sym,left,right,L,R,q,trs,coef=tables
    ns=parents[0].shape[1]
    spec=P(None,None,'x',None,'y')
    @partial(shard_map,mesh=mesh,in_specs=spec,out_specs=P(None,None,'x','y',None),check_vma=False)
    def unfold(D):
        blocks=[]
        for a in range(ns):
            row=[]
            for b in range(ns):
                result=jnp.zeros((4,2,3),jnp.complex128)
                for c in range(ns):
                    for d in range(ns):
                        block=unfold_operator_local(D[:,c,:,d,:],irr_idx=irr,sym_idx=sym,
                            q_irr_frac=q,left_local_perm=left,left_L_table=L,
                            right_local_perm=right,right_L_table=R,n_sym_spatial=2)
                        result+=jnp.asarray(coef[:,a*ns+b,c*ns+d])[:,None,None]*block
                row.append(jnp.conj(result))
            blocks.append(jnp.stack(row,axis=-1))
        return jnp.stack(blocks,axis=1)
    return [jax.jit(unfold)(jax.device_put(D,NamedSharding(mesh,spec))) for D in parents]


def test_parent_conv_typed_tables_match_decomposed():
    """Random ns=2/4 parent loads with spin mixing and antiunitary rows match the oracle."""
    assert len(jax.devices()) >= 4, 'requires four emulated CPU devices'
    mesh=Mesh(np.asarray(jax.devices()[:4]).reshape(2,2),('x','y'))
    for ns in (2,4):
        parents,tables,U=_fixture(ns)
        expected=_literal(parents,tables,U)
        actual=_decomposed(parents,tables,U,mesh)
        for a,b in zip(actual,expected):
            error=np.linalg.norm(np.asarray(a)-b)/np.linalg.norm(b)
            assert error < 1e-12, (ns,error)


def gpu_main():
    """Exercise both native arms at P4 against direct k sums on random parent tensors."""
    import os
    os.environ["LORRAX_CONV_KPAIR_FFI"] = "on"
    from runtime import initialize_communicator_stack
    from ffi import fft
    from common.gamma_matrices import gamma_perm_phase
    initialize_communicator_stack()
    mesh=Mesh(np.asarray(jax.devices()).reshape(2,2),('x','y'))
    spec=P(None,None,'x',None,'y')
    table_specs=(P(),P(),P(None,'x'),P(None,'y'),P(None,'x',None),
                 P(None,'y',None),P(),P(),P(),P())
    for ns in (1,2,4):
        parents,tables,U=_fixture(ns)
        for vertex in ((0,2) if ns==4 else (0,)):
            perm,phase=(gamma_perm_phase(vertex) if vertex else (np.arange(ns),np.ones(ns)))
            expected=_direct_convolution(_literal(parents,tables,U),np.asarray(perm),
                                         np.asarray(phase),np.asarray(perm),np.asarray(phase))
            for arm, centroid_major in product(('resident','two_stage'), (False, True)):
                folded=list((*tables,tables[-1]))
                pairs=(np.asarray(perm)[:,None]*ns+np.asarray(perm)[None,:]).reshape(-1)
                phases=(np.asarray(phase)[:,None]*np.asarray(phase)[None,:]).reshape(-1)
                folded[-1]=tables[-1][:,pairs,:]*np.conj(phases)[None,:,None]
                old=fft.conv_kpair_plan
                fft.conv_kpair_plan=lambda *args: (arm,'oracle forced coverage')
                try:
                    kernel=fft.make_fused_conv_kparent(mesh,(2,2,1),ns,(2,3),
                        perm_l=np.arange(ns),phase_l=np.ones(ns),
                        perm_r=np.arange(ns),phase_r=np.ones(ns),centroid_major=centroid_major)
                finally: fft.conv_kpair_plan=old
                assert kernel is not None
                @partial(shard_map,mesh=mesh,in_specs=(spec,spec,table_specs),
                         out_specs=P(None,'x','y'),check_vma=False)
                def run(a,b,t): return kernel(a,b,t)
                operands=[jax.device_put(a,NamedSharding(mesh,spec)) for a in parents]
                device_tables=tuple(jax.device_put(a,NamedSharding(mesh,p)) for a,p in zip(folded,table_specs))
                value=jax.jit(run)(*operands,device_tables)
                ref=jax.device_put(expected,NamedSharding(mesh,P(None,'x','y')))
                error=float(jax.device_get(jnp.linalg.norm(value-ref)/jnp.linalg.norm(ref)))
                assert error < 1e-12,(ns,vertex,arm,error)
                if jax.process_index()==0: print(json.dumps(dict(ns=ns,vertex=vertex,arm=arm,centroid_major=centroid_major,relative=error)),flush=True)
    return 0


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(gpu_main)
