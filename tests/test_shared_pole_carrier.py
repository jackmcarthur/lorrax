"""Tiny CPU layout/algebra checks; real GPU/HLO/peak gates remain separate."""
import hashlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh():
    from lxkit.testing import require_devices
    require_devices(4, 'cpu')
    return Mesh(np.asarray(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))


@pytest.mark.parametrize('columns', [None, (1, 4), (4, 5)])
def test_reader_keeps_both_axes_and_masks_ragged_poles(monkeypatch, columns):
    from file_io import shared_pole_store as store
    mesh = _mesh()
    indices = np.arange(12, dtype=np.int32).reshape(4, 3)
    basis = SimpleNamespace(mesh_xy=mesh, n_logical=4, n_canonical=4,
                            n_packed=4, canonical_indices=indices,
                            is_identity=True, active_mask=np.ones(4, bool),
                            pack_axis=lambda b, *_a, **_kw: b)
    header = dict(schema=store.SCHEMA, finalized=True, n_q_irr=2,
                  n_mu_logical=4, nspinor=1, Kmax=5, K=[5, 2],
                  centroid_digest=hashlib.sha256(indices.astype('<i4').tobytes()).hexdigest())
    meta = SimpleNamespace(mu_basis=basis, nspinor=1)
    monkeypatch.setattr(store, '_capacity', lambda _meta: None)
    monkeypatch.setattr(store, '_check_io_capacity', lambda *_a: None)
    monkeypatch.setattr(store, '_admit', lambda *_a, **_kw: None)
    factor = (np.arange(40).reshape(2, 4, 1, 5) + 1j).astype(np.complex128)
    poles = np.arange(10, dtype=np.float64).reshape(2, 5) + 1
    factor[1, :, :, 2:] = 0
    poles[1, 2:] = 1
    calls = []

    def read(name, *, shape, offset, valid_shape, partition_spec):
        calls.append((name, partition_spec))
        source = factor if name == 'factor' else poles
        result = np.zeros(shape, dtype=source.dtype)
        dest = tuple(slice(0, n) for n in valid_shape)
        origin = tuple(slice(o, o+n) for o, n in zip(offset, valid_shape))
        result[dest] = source[origin]
        return jax.device_put(result, NamedSharding(mesh, partition_spec))

    faces = store.read_shared_pole_faces(SimpleNamespace(mesh=mesh, read_slab=read),
                                         (0, 2), meta=meta, header=header,
                                         column_span=columns)
    lo, hi = columns or (0, 5)
    width = ((hi-lo+1)//2)*2
    expected = np.pad(factor[..., lo:hi], ((0,0),(0,0),(0,0),(0,width-(hi-lo))))
    for value, spec in zip(faces[:2], (P(None,'x',None,'y'), P(None,'y',None,'x'))):
        assert value.sharding.is_equivalent_to(NamedSharding(mesh, spec), 4)
        np.testing.assert_array_equal(value, expected)
        assert sum(s.data.nbytes for s in value.addressable_shards) == value.nbytes
    assert calls[-1] == ('poles2_ry2', P())
    np.testing.assert_array_equal(faces[2], np.pad(poles[:,lo:hi],
                                  ((0,0),(0,width-(hi-lo))), constant_values=1))
    np.testing.assert_array_equal(faces[3], np.clip(np.array([5,2])-lo, 0, hi-lo))


@pytest.mark.parametrize('tau', [.7+.2j, -.3j])
def test_g_face_contraction_preserves_causal_transpose(tau):
    from gw.mpa.sigma import synthesize_shared_pole_parents
    mesh = _mesh()
    rng = np.random.default_rng(361)
    b = rng.normal(size=(3,8,1,6)) + 1j*rng.normal(size=(3,8,1,6))
    omega = rng.uniform(.1, 2, size=(3,6))
    bounds = np.array([[0,5],[1,4],[0,0]], np.int32)
    put = lambda x,spec: jax.device_put(x, NamedSharding(mesh,spec))
    # CPU oracle backend only: production passes the warmed native G plan.
    gemm = jax.jit(lambda x,y: x@y, out_shardings=NamedSharding(mesh,P(None,'x','y')))
    fn = jax.jit(lambda x,y,p,r: synthesize_shared_pole_parents(
        x,y,p,r,.3,tau,mesh_xy=mesh,gemm=gemm))
    got = fn(put(b,P(None,'x',None,'y')),put(b,P(None,'y',None,'x')),
             put(omega**2,P()),put(bounds,P()))
    mask = ((np.arange(6) >= bounds[:,:1]) & (np.arange(6) < bounds[:,1:]))
    d = np.where(mask,np.exp(-1j*(omega-.3)*tau)/(2*omega),0)
    f = b[:,:,0,:]
    expected = ((f*d[:,None,:])@f.conj().swapaxes(-1,-2),
                (f.conj()*d[:,None,:])@f.swapaxes(-1,-2))
    for value, oracle in zip(got, expected):
        np.testing.assert_allclose(value, oracle, atol=2e-12)
        assert value.sharding.is_equivalent_to(NamedSharding(mesh,P(None,'x','y')),3)


def _synthesis_fixture(monkeypatch):
    """The carrier fixture: a real header, CPU stubs and a face reader.

    The header is the SHIPPED minimum, not a smaller one: every shared-pole
    consumer resolves its fixed-q policy through
    ``qgrid_trs_policy_from_shared_pole_store`` (``operations``), binds the
    physical realization through ``shared_pole_operator_realizer``
    (``recipe``) and checks the packed basis geometry (``n_mu_logical``).
    A header missing them does not exercise the synthesis; it raises
    ``KeyError`` before reaching it.
    """
    import distrib_la
    from file_io import shared_pole_store
    from common.grouped_layout import identity_square_grouped_shard_layout
    from gw.shared_pole_recipe import shared_real_pole_v1_r3b
    import runtime.aot_memory
    mesh = _mesh()
    tile = NamedSharding(mesh,P(None,'x','y'))
    layout = identity_square_grouped_shard_layout(8,8,(2,2))
    meta = SimpleNamespace(nk_tot=8,nspinor=1,n_rmu=8,mu_basis=SimpleNamespace(
        n_packed=8,layout=layout,active_mask=np.ones(8,bool)))
    parents = np.arange(8,dtype=np.int32)%2
    header = dict(n_q_irr=2,n_q_full=8,n_mu_logical=8,Kmax=5,nspinor=1,
        representation='scalar-trs-even-s',grid=(2,2,2),q_irr_full_idx=np.arange(2),
        operations=dict(authorized_rows=[0]),
        recipe=dict(operator_realization=shared_real_pole_v1_r3b["operator_realization"]),
        qirr=dict(irr_idx_q=parents,sym_idx_q=np.zeros(8,np.int32),
                  sym_perm=np.arange(8,dtype=np.int32)[None,:],
                  L_table=np.zeros((1,8,3),np.int32),q_irr_frac=np.zeros((2,3)),
                  n_sym_spatial=1))
    # Exercise orchestration on CPU without pretending to test a native plan.
    monkeypatch.setattr(distrib_la,'plan',lambda *_a,**_kw: None)
    monkeypatch.setattr(distrib_la,'workspace_bytes_per_rank',lambda *_a: 0)
    monkeypatch.setattr(distrib_la,'gemm_plan',lambda *_a,**_kw:
        jax.jit(lambda x,y:x@y,out_shardings=tile))
    monkeypatch.setattr(runtime.aot_memory,'aot_kernel_peak_bytes',lambda compiled:
        SimpleNamespace(total=compiled.memory_analysis().temp_size_in_bytes,cufft_measured=True))
    rng = np.random.default_rng(362)
    factor = rng.normal(size=(2,8,1,5))+1j*rng.normal(size=(2,8,1,5))
    omega = np.broadcast_to(np.linspace(.2,2,5),(2,5)).copy()
    reads = []
    put = lambda value,spec: jax.device_put(value,NamedSharding(mesh,spec))

    def read(_io, span, *, meta, header, column_span=None):
        reads.append((span,column_span))
        lo,hi=span
        first,last=column_span or (0,5)
        width=((last-first+1)//2)*2
        b=np.pad(factor[lo:hi,...,first:last],
                 ((0,0),(0,0),(0,0),(0,width-last+first)))
        p=np.pad(omega[lo:hi,first:last]**2,
                 ((0,0),(0,width-last+first)),constant_values=1)
        return (put(b,P(None,'x',None,'y')),put(b,P(None,'y',None,'x')),
                put(p,P()),put(np.full(hi-lo,last-first),P()))

    monkeypatch.setattr(shared_pole_store,'read_shared_pole_faces',read)
    return SimpleNamespace(mesh=mesh,meta=meta,header=header,parents=parents,
                           factor=factor,omega=omega,reads=reads,put=put)


@pytest.mark.parametrize('parent_capacity,column_capacity', [(2,5),(1,3)])
def test_synthesis_uses_same_carrier_for_resident_and_panels(
        monkeypatch, parent_capacity, column_capacity):
    from gw.mpa.sigma import _shared_pole_w_synthesis
    fx = _synthesis_fixture(monkeypatch)
    mesh, meta, header = fx.mesh, fx.meta, fx.header
    parents, factor, omega, reads, put = (
        fx.parents, fx.factor, fx.omega, fx.reads, fx.put)
    schedule=dict(status='PASS',parent_capacity=parent_capacity,
                  column_capacity=column_capacity,endpoint_budgets={})
    build=_shared_pole_w_synthesis(None,meta,header,omega,schedule,mesh_xy=mesh)
    bounds=np.tile([0,np.inf,-np.inf,-np.inf,np.inf,np.inf],(2,1))
    args=(None,None,put(np.arange(2),P()),put(bounds,P()),None,.3,.7+.2j)
    got=build(*args)
    f=factor[:,:,0,:]
    d=np.exp(-1j*(omega-.3)*(.7+.2j))/(2*omega)
    plus=(f*d[:,None,:])@f.conj().swapaxes(-1,-2)
    trans=(f.conj()*d[:,None,:])@f.swapaxes(-1,-2)
    # Every q of the 2x2x2 fixture is self-negative, with identity star maps.
    expected=(.5*(plus+trans))[parents]
    np.testing.assert_allclose(got,expected,atol=3e-12)
    before=len(reads)
    jax.block_until_ready(build(*args))
    assert len(reads)==before*(1 if parent_capacity==2 else 2)
