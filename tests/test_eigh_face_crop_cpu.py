"""Padded eigensolver faces crop by exact copy through the shared row owner."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_FLAGS','--xla_force_host_platform_device_count=144')
import math,re
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from lxkit import permute_rows_local
from distrib_la._cyclic import face_crop
from distrib_la._cusolvermp import drop_sentinels


def mesh_for(side):
    if len(jax.devices())<side*side:pytest.skip('requires virtual square mesh')
    return Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))


def test_row_transport_changed_extent_duplicates_empty_and_legacy_permutations():
    jax.config.update('jax_enable_x64',True)
    for side in (1,2,3,12):
        mesh=mesh_for(side);source=side*7
        for dtype in (np.float64,np.complex128):
            host=np.arange(2*source*3).reshape(2,source,3).astype(dtype)
            if dtype==np.complex128:host=host+.125j*host[::-1]
            for target in (0,side*5,source):
                order=np.stack(((np.arange(target)*5)%source,(np.arange(target)*3+2)%source)).astype(np.int32)
                if target==source:order=np.stack((np.arange(source)[::-1],np.roll(np.arange(source),3))).astype(np.int32)
                local=jax.shard_map(lambda a,m:permute_rows_local(a,m,mesh_axis='y',parts=side,is_local=side==1),mesh=mesh,
                    in_specs=(P(None,'y',None),P()),out_specs=P(None,'y',None),check_vma=False)
                got=jax.jit(local)(host,order)
                want=np.take_along_axis(host,order[...,None],axis=1)
                np.testing.assert_array_equal(np.asarray(got),want)
                # JAX 0.9 canonicalizes a zero-element result to P(); no
                # field exists then. Every nonempty result keeps row ownership.
                if target:
                    assert got.sharding.is_equivalent_to(NamedSharding(mesh,P(None,'y',None)),3)


def test_true_column_and_raw_row_eigenvector_crops_match_literal_slices():
    jax.config.update('jax_enable_x64',True)
    for side in (1,2,3,12):
        mesh=mesh_for(side);source=side*11;n=side*7
        for dtype in (np.float64,np.complex128):
            for batch in ((),(2,)):
                sh=NamedSharding(mesh,P(*((None,)*len(batch)),'x','y'))
                host=np.arange(math.prod(batch or(1,))*source*source).reshape(batch+(source,source)).astype(dtype)
                if dtype==np.complex128:host=host+1j*(.125*host+3)
                for offset in ((0,source-n),(source-n,0),(side+1,2*side-1)):
                    fn=jax.jit(lambda q:face_crop(q,n,offsets=offset,mesh=mesh),in_shardings=sh,out_shardings=sh)
                    got=fn(host);np.testing.assert_array_equal(np.asarray(got),host[...,offset[0]:offset[0]+n,offset[1]:offset[1]+n])
                    assert got.sharding.is_equivalent_to(sh,host.ndim)
                empty=jax.jit(lambda q:face_crop(q,0,offsets=(source,source),mesh=mesh),in_shardings=sh,out_shardings=sh)(host)
                assert empty.shape==batch+(0,0) and empty.dtype==dtype
                same=jax.jit(lambda q:face_crop(q,source,offsets=(0,0),mesh=mesh),in_shardings=sh,out_shardings=sh)(host)
                np.testing.assert_array_equal(np.asarray(same),host)
                np.testing.assert_array_equal(np.asarray(face_crop(jnp.asarray(host),n,offsets=(1,2),mesh=None)),host[...,1:n+1,2:n+2])
            raw=np.arange(source*source).reshape(source,source).astype(dtype);w=np.arange(source,dtype=np.float64)
            fn=jax.jit(lambda w,q:drop_sentinels(w,q,n,mesh=mesh),in_shardings=(NamedSharding(mesh,P()),NamedSharding(mesh,P('x','y'))))
            gotw,gotq=fn(w,raw);np.testing.assert_array_equal(np.asarray(gotw),w[source-n:]);np.testing.assert_array_equal(np.asarray(gotq),raw[source-n:,:n])
        with pytest.raises(ValueError,match='outside'):face_crop(jnp.ones((source,source)),n,offsets=(source,0),mesh=mesh)
        if side>1:
            with pytest.raises(ValueError,match='divide'):face_crop(jnp.ones((source,source)),n+1,offsets=(0,0),mesh=mesh)


def test_large_crop_hlo_uses_only_dependent_full_rank_rings_and_owned_faces():
    jax.config.update('jax_enable_x64',True)
    for side in (2,3,12):
        mesh=mesh_for(side);source=49920 if side==12 else side*131;n=49164 if side==12 else side*127
        for dtype in (jnp.float64,jnp.complex128):
            sh=NamedSharding(mesh,P(None,'x','y'));arg=jax.ShapeDtypeStruct((2,source,source),dtype,sharding=sh)
            for offset in ((0,source-n),(source-n,0)):
                exe=jax.jit(lambda q:face_crop(q,n,offsets=offset,mesh=mesh),in_shardings=sh,out_shardings=sh).lower(arg).compile()
                text=exe.as_text();bound=2*(source//side)**2;perms=[]
                for line in text.splitlines():
                    m=re.match(r'\s*(?:ROOT )?%\S+ = (?:f64|c128)\[([0-9,]*)\]',line)
                    if m:assert(math.prod(map(int,m[1].split(',')))if m[1]else 1)<=bound,line
                    assert not re.search(r' (?:all-gather|all-to-all|all-reduce)(?:-start)?\(',line),line
                    if re.search(r' collective-permute(?:-start)?\(',line):
                        pairs=re.search(r'source_target_pairs=(\{\{.*?\}\})',line)
                        assert pairs,line
                        pairs=[(int(a),int(b))for a,b in re.findall(r'\{(\d+),(\d+)\}',pairs[1])]
                        assert {a for a,b in pairs}=={b for a,b in pairs}==set(range(side*side)),line
                        perms.append(line)
                assert len(perms)==2
                # The second scan consumes the first scan's selected rows.
                entry=text.rsplit('\nENTRY ',1)[1];defs={}
                for line in entry.splitlines():
                    m=re.match(r'\s*(?:ROOT )?%(\S+) = (.*)',line)
                    if m:defs[m[1]]=m[2]
                loops=[key for key,rhs in defs.items()if ' while(' in rhs]
                assert len(loops)==2
                def ancestors(name):
                    seen=set();pending=[name]
                    while pending:
                        key=pending.pop()
                        if key not in seen and key in defs:seen.add(key);pending.extend(re.findall(r'%([\w.\-]+)',defs[key]))
                    return seen
                assert loops[0]in ancestors(loops[1])
