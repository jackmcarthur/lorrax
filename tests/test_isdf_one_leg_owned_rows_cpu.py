"""One Fourier leg routes centroid rows without gathering a parent field."""
from types import SimpleNamespace
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
jax.config.update('jax_enable_x64', True)


def _tables(mesh, single, local):
    physical=65
    nmu=-(-physical//mesh.size)*mesh.size
    eye=np.eye(3,dtype=int)
    spatial=np.array([eye,-eye])
    mats=np.concatenate((spatial,-spatial))
    integers=np.array([[0,0,0]]) if single else np.array([[i,0,0] for i in range(5)])
    parents=np.array([[0,0,0]]) if single else np.array([[0,0,0],[1,0,0],[2,0,0]])
    idx=np.array([0]) if single else np.array([0,1,2,2,1])
    rows=np.array([3]) if single else np.array([3,0,0,1,2])
    grid=(1,1,1) if single else (5,1,1)
    sym=SimpleNamespace(kvecs_asints=integers,q_irr_kgrid_int=parents,irr_idx_q=idx,
        sym_matrices=spatial,sym_mats_k=mats,translations=np.array([[0.,0.,0.],[.5*np.pi,0.,0.]]))
    sym.cartesian_action=lambda selected,**_kw:mats[np.asarray(selected)]
    q=parents/np.array(grid)
    perm=np.tile(np.arange(nmu,dtype=np.int32),(4,1))
    if local:
        width=nmu//int(mesh.shape['x'])
        for lo in range(0,physical,width):
            hi=min(physical,lo+width)
            perm[1,lo:hi]=np.arange(lo,hi)[::-1]
    else:
        perm[1,:physical]=np.arange(physical)[::-1]
    perm[3]=perm[1]
    rng=np.random.default_rng(195+int(mesh.size)+int(single))
    wraps=np.zeros((4,nmu,3),int)
    wraps[1,:physical]=rng.integers(-2,3,(physical,3))
    wraps[3]=wraps[1]
    return sym,rows,q,perm,wraps,grid,physical


def check_one_leg_owned_rows(monkeypatch, mesh, *, single=False, local=False):
    """Independent phase/permutation oracle, shared by the native MPI proof."""
    from symmetry_maps import maps
    from common.collectives import device_put_process_local
    sym,rows,q,perm,wraps,grid,physical=_tables(mesh,single,local)
    nmu=perm.shape[1]
    idx=sym.irr_idx_q
    rng=np.random.default_rng(338+int(mesh.size))
    base=rng.normal(size=(len(q),nmu,3))+1j*rng.normal(size=(len(q),nmu,3))
    base[:,physical:]=0
    source_g=np.tile(np.array([[-1,0,1],[0,0,0],[0,0,0]])[None],(len(q),1,1))
    captures=[]
    traced_metadata=[]
    def concrete_metadata(owner):
        def place(*a,**kw):
            value=owner(*a,**kw)
            traced=isinstance(a[0],jax.core.Tracer)
            traced_metadata.append(traced)
            assert not isinstance(value,jax.core.Tracer) or traced, 'host metadata created traced Python shards'
            return value
        return place
    monkeypatch.setattr(maps,'device_put_process_local',concrete_metadata(maps.device_put_process_local))
    monkeypatch.setattr(maps,'device_put_process_local_cached',concrete_metadata(maps.device_put_process_local_cached))
    original=maps._get_unfold_isdf_one_leg_jit
    def observe(**kw):
        program=original(**kw)
        def execute(*args):
            if any(isinstance(a,jax.core.Tracer) for a in args):
                return program(*args)
            exe=program.lower(*args).compile()
            text=exe.as_text()
            shapes=[tuple(map(int,s.split(','))) if s else () for s in
                    re.findall(r'=\s*c128\[([0-9,]*)\]',text)]
            assert (len(q),nmu) not in shapes and (len(rows),nmu) not in shapes
            maximum=max(int(np.prod(s,dtype=np.int64)) for s in shapes)
            bound=max(len(rows),len(q))*nmu//int(mesh.shape['x'])*3
            assert maximum<=bound, 'one-leg transport materialized a global parent field'
            gathers=len(re.findall(r'\ball-gather(?:-start)?\(',text))
            if kw['preselected']:
                assert gathers==0, 'preselected Fourier leg gathered a full centroid row'
            captures.append(dict(preselected=kw['preselected'],largest_complex_elements=maximum,
                payload_bound=bound,all_gather_count=gathers,
                collective_permute_count=len(re.findall(r'\bcollective-permute(?:-start)?\(',text))))
            return exe(*args)
        return execute
    monkeypatch.setattr(maps,'_get_unfold_isdf_one_leg_jit',observe)
    worst=0.
    for preselected in (True,False):
        host=base[:,:,2] if preselected else base
        spec=P(None,'x') if preselected else P(None,('x','y'),None)
        data=device_put_process_local(host,NamedSharding(mesh,spec))
        for action in ('scalar','polar'):
            for scale in (1,2):
                runtime_wraps=scale*wraps
                kwargs=dict(source_gvec_components=np.tile([1,0,0],(len(q),1))) if preselected else dict(gvec_components=source_g)
                kwargs.update(sym=sym,sym_idx=rows,sym_perm=perm,L_table=runtime_wraps,
                    q_irr_frac=q,kgrid=grid,mesh_xy=mesh,component_action=action)
                if action=='polar':
                    kwargs.update(source_component=1,component_basis=None)
                # First creation under an outer trace, then reuse with other
                # wrap values, must not retain a tracer in the geometry cache.
                result=(jax.jit(lambda z:maps.unfold_isdf_one_leg(z,**kwargs))(data)
                        if scale==1 else maps.unfold_isdf_one_leg(data,**kwargs))
                expected=[]
                for parent,row in zip(idx,rows):
                    g=np.array([1,0,0]) if preselected else np.zeros(3)
                    spatial=row%2
                    tau=np.exp(-1j*(sym.sym_matrices[spatial]@(q[parent]+g))@sym.translations[spatial])
                    phase=tau*np.exp(-2j*np.pi*(runtime_wraps[row]@q[parent]))
                    value=host[parent,perm[row]] if preselected else base[parent,perm[row],1]
                    value=phase*value
                    if row>=2:
                        value=value.conj()
                    expected.append(value)
                expected=np.asarray(expected)
                if action=='polar':
                    expected=np.einsum('qi,qm->iqm',sym.sym_mats_k[rows,:,1],expected)
                for shard in result.addressable_shards:
                    actual=np.asarray(shard.data)
                    wanted=expected[shard.index]
                    np.testing.assert_allclose(actual,wanted,rtol=2e-13,atol=2e-13)
                    worst=max(worst,float(np.max(np.abs(actual-wanted))))
                    assert np.isfinite(actual).all()
    assert any(traced_metadata), 'nested kernel geometry was closed over instead of passed as traced arguments'
    return dict(single_q=single,local_map=local,n_mu=nmu,n_mu_logical=physical,
                largest_absolute_error=worst,hlo=captures)


def check_endpoint_rows(mesh):
    """The shared ring extraction preserves the existing spinor factor action."""
    from symmetry_maps import maps
    from common.collectives import device_put_process_local
    sym,rows,q,perm,wraps,_grid,physical=_tables(mesh,False,False)
    rng=np.random.default_rng(581)
    nmu=perm.shape[1]
    host=rng.normal(size=(len(q),nmu,2,int(mesh.shape['y']))).astype(complex)
    host+=1j*rng.normal(size=host.shape)
    host[:,physical:]=np.nan
    spec=NamedSharding(mesh,P(None,'x',None,'y'))
    data=device_put_process_local(host,spec)
    active=np.arange(nmu)<physical
    spin=np.tile(np.eye(2,dtype=complex),(len(rows),1,1))
    spin[rows>=2]=np.array([[0,1],[-1,0]])
    actual,_=maps.unfold_endpoint_panel(data,irr_idx=sym.irr_idx_q,sym_idx=rows,
        q_irr_frac=q,source_perm=perm,L_table=wraps,spin_action_full=spin,
        n_sym_spatial=2,active_mask=active,mesh=mesh,mesh_axis='x',max_live_bytes=1<<30)
    expected=[]
    for j,(parent,row) in enumerate(zip(sym.irr_idx_q,rows)):
        value=host[parent,perm[row]].copy()
        value*=np.exp(2j*np.pi*(wraps[row]@q[parent]))[:,None,None]
        if row>=2:
            value=value.conj()
        value=np.einsum('ac,mck->mak',spin[j],value)
        value[physical:]=0
        expected.append(value)
    expected=np.asarray(expected)
    for shard in actual.addressable_shards:
        np.testing.assert_allclose(np.asarray(shard.data),expected[shard.index],rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('side,single,local',[(2,False,False),(3,False,False),(2,True,False),(3,True,True)])
def test_one_leg_keeps_centroid_blocks_through_symmetry_and_antiunitary_phases(monkeypatch,side,single,local):
    if jax.device_count()<side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh=Mesh(np.asarray(jax.devices()[:side*side]).reshape(side,side),('x','y'))
    check_one_leg_owned_rows(monkeypatch,mesh,single=single,local=local)
    if not single:
        check_endpoint_rows(mesh)


def test_one_leg_p64_partitioned_hlo_and_single_q_identity():
    # A private single-process virtual-CPU structural gate, not qualification
    # of the unsupported stable CPU64 MPI runtime.
    code=r'''
import importlib.util,json,numpy as np,pytest,jax
from jax.sharding import Mesh
s=importlib.util.spec_from_file_location('one_leg_identity','tests/test_isdf_one_leg_owned_rows_cpu.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
m=Mesh(np.asarray(jax.devices()).reshape(8,8),('x','y'))
for single in (False,True):
 with pytest.MonkeyPatch.context() as p:
  print(json.dumps(t.check_one_leg_owned_rows(p,m,single=single)),flush=True)
'''
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=64')
    run=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,timeout=180)
    assert run.returncode==0,run.stdout+run.stderr
