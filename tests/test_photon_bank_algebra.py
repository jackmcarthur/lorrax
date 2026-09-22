"""Known-answer FD body contact and signed photon Dyson/moment plants."""
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def check_free_electron_current_contact(mesh):
    from gw.static_gauge_response import fermi_dirac_current_drude
    from common.collectives import gather_to_host

    # Six equally weighted Fermi points +/-x,+/-y,+/-z, one band. At mu,
    # -f'=1/(4 kT). A nonuniform centroid profile makes a uniform embedding
    # fail pointwise, including off-diagonal centroid pairs.
    velocity = np.diag([1., 2., 3.])
    v = np.stack([velocity[0], -velocity[0], velocity[1], -velocity[1],
                  velocity[2], -velocity[2]])
    profile = np.array([1., 2., 0.5, -1.])
    current = (profile[None, :, None] * v[:, None, :]).reshape(6, 12)
    def put(value):
        return jax.make_array_from_callback(value.shape,
            NamedSharding(mesh, P(None, "x", "y")), lambda idx: value[idx])
    # Pad the band axis to the mesh divisor without admitting padded states.
    nb = np.lcm(mesh.shape['x'], mesh.shape['y'])
    left = np.zeros((6, 12, nb), complex)
    right = np.zeros((6, nb, 12), complex)
    left[:, :, 0], right[:, 0, :] = current, current
    f = np.zeros((6, nb)); f[:, 0] = 0.5
    state = SimpleNamespace(f_kn=f, smearing_family="fd", smearing_width_ry=0.5)
    d = fermi_dirac_current_drude((put(left), put(right)), state,
        state_capacity=1, cell_volume=2, kweights=np.full(6, 1/6), mesh_xy=mesh)
    expected = np.einsum("m,n,ab->manb", profile, profile,
                         np.diag([1., 4., 9.])/12).reshape(12, 12)
    np.testing.assert_allclose(gather_to_host(d), expected, rtol=1e-14, atol=1e-14)
    uniform_twin = np.tile(np.diag([1., 4., 9.])/12, (4, 4))
    assert np.linalg.norm(expected-uniform_twin) > 1


def test_free_electron_current_contact():
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    check_free_electron_current_contact(mesh)


def test_signed_photon_dyson_derivative_and_moments(monkeypatch):
    import distrib_la
    from gw.response_bank import response_algebra

    # This plant tests Dyson algebra, explicitly using native tiny CPU solves.
    monkeypatch.setattr(distrib_la, "matmul", lambda a, b, **kw: a @ b)
    monkeypatch.setattr(distrib_la, "plan", lambda *a, **kw: SimpleNamespace(
        batched=jnp.linalg.solve, describe=lambda: "tiny CPU test solve"))
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    put = lambda a: jax.device_put(np.asarray(a, complex)[None],
                                   NamedSharding(mesh, P(None, "x", "y")))
    meta = SimpleNamespace(nk_tot=1, nspin=1, nspinor_wfnfile=2, cell_volume=2.)
    value, slope, moments, receipt = response_algebra(
        meta, {"linalg": "local"}, mesh_xy=mesh, n=4, photon=True)
    def samples(v, chi, dchi, contact):
        wc = value(v, chi, contact)
        return wc, slope(v, wc, dchi)
    v = np.diag([1.2, -0.3, -0.2, -0.4]).astype(complex)
    contact = np.diag([0., 0.1, 0.07, 0.05])
    b = np.array([0.5, 0.1j, 0.2+0.1j, -0.3j])
    residue = np.outer(b, b.conj())
    pole = 1.7
    def chi(z):
        return residue/(z-pole) - residue.conj()/(z+pole)
    def w(z):
        return np.linalg.solve(np.eye(4)-v @ (chi(z)-contact), v)
    z = 0.3+0.8j
    derivative = (-residue/(z-pole)**2 + residue.conj()/(z+pole)**2)/(2*z)
    wc, dw = samples(put(v), put(chi(z)), put(derivative), put(contact/meta.cell_volume))
    np.testing.assert_allclose(wc[0], w(z)-v, rtol=2e-13, atol=2e-14)
    conjugate, _ = samples(put(v), put(chi(z.conjugate())), put(derivative), put(contact/meta.cell_volume))
    mirror, _ = samples(put(v), put(chi(-z.conjugate())), put(derivative), put(contact/meta.cell_volume))
    np.testing.assert_allclose(conjugate[0], np.asarray(wc[0]).conj().T, atol=2e-14)
    np.testing.assert_allclose(mirror[0], np.asarray(wc[0]).conj(), atol=2e-14)
    step = 1e-5
    numeric = (w(np.sqrt(z*z+step))-w(np.sqrt(z*z-step)))/(2*step)
    np.testing.assert_allclose(dw[0], numeric, rtol=1e-8, atol=2e-11)
    coefficients = [residue*pole**i - residue.conj()*(-pole)**i for i in range(4)]
    constant, *got = moments(put(v), put(coefficients[1]), put(coefficients[3]),
                              put(coefficients[0]), put(coefficients[2]), put(contact/meta.cell_volume))
    winf = np.linalg.solve(np.eye(4)+v @ contact, v)
    np.testing.assert_allclose(constant[0], winf-v, rtol=1e-13, atol=1e-14)
    # Independent coefficient extraction by a Cauchy contour in t=1/z.
    theta = 2*np.pi*np.arange(64)/64
    radius = 0.1
    values = np.asarray([w(1/(radius*np.exp(1j*t)))-winf for t in theta])
    for order, moment in enumerate(got, 1):
        coefficient = np.einsum("t,tij->ij", np.exp(-1j*order*theta), values)
        coefficient /= len(theta)*radius**order
        np.testing.assert_allclose(2*np.asarray(moment[0]), coefficient,
                                   rtol=1e-9, atol=1e-10)
    assert np.linalg.norm(constant) > 1e-3


def check_photon_face_packing(mesh):
    """Check both face orientations, mesh interleaving and internal padding."""
    from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC
    from gw.photon_layout import PhotonBasisLayout, pack_photon_faces

    layout = PhotonBasisLayout.from_centroid_extents(3, 5, mesh)
    for orientation, axis, spec in (("mun", 2, PSI_MUN_SPEC), ("nmu", 3, PSI_NMU_SPEC)):
        host = []
        for channel, extent in enumerate(layout.carrier_extents):
            shape = (2, 4, extent, 4) if orientation == "mun" else (2, 4, 4, extent)
            array = (np.arange(np.prod(shape)).reshape(shape)+10000*channel).astype(complex)
            host.append(array)
        def put(array):
            return jax.make_array_from_callback(array.shape, NamedSharding(mesh, spec),
                                                lambda index: array[index])
        packed = pack_photon_faces(tuple(map(put, host)), layout, mesh, orientation=orientation)
        for shard in packed.addressable_shards:
            owner = (shard.index[axis].start or 0) // (layout.packed_extent // layout.mesh_side)
            pieces = []
            for channel, source in enumerate(host):
                width = layout.carrier_extent(channel) // layout.mesh_side
                index = list(shard.index)
                index[axis] = slice(owner*width, (owner+1)*width)
                expected = source[tuple(index)].copy()
                valid = owner*width + np.arange(width) < layout.logical_extent(channel)
                shape = [1]*4
                shape[axis] = width
                pieces.append(np.where(valid.reshape(shape), expected, 0))
            np.testing.assert_array_equal(np.asarray(shard.data), np.concatenate(pieces, axis=axis))


def test_photon_face_packing():
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    check_photon_face_packing(mesh)


def check_photon_bank_store(mesh, path):
    """Full packed sample/derivative/moment/constant roundtrip on P4."""
    from test_shared_pole_bank import _bank_fixture
    from test_shared_pole_store import _fixture
    from gw.photon_layout import PhotonBasisLayout
    from gw.shared_pole_recipe import CapacityLedger
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO

    _, _, _, recipe, _ = _bank_fixture()
    meta, tables, _, identity = _fixture(mesh)
    meta.nspinor = 4
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh)
    meta.shared_pole_capacity.reserve('fixture', resident_bytes_per_rank=65536, workspace_bytes_per_rank=0)
    meta.shared_pole_capacity.live_stages = ('fixture',)
    basis = meta.mu_basis
    layout = PhotonBasisLayout.from_centroid_extents(basis.n_logical, basis.n_logical, mesh)
    header = store.initialize_shared_pole_bank(path, meta=meta, tables=tables,
        recipe=recipe, identity=identity, mesh_xy=mesh, photon_layout=layout, mu_bases=(basis,basis))
    nq, ns, n = header['bank_shape']['nq'], header['bank_shape']['nsample'], layout.packed_extent
    def put(a):
        spec = P(None,None,'x','y') if a.ndim == 4 else P(None,'x','y')
        return jax.make_array_from_callback(a.shape,NamedSharding(mesh,spec),lambda ix:a[ix])
    raw = np.arange(nq*ns*n*n).reshape(nq,ns,n,n).astype(complex)*(1+.3j)
    w, moment = put(raw), put(raw[:,0])
    header = store.write_shared_pole_bank(path,q_span=(0,nq),sample_span=(0,ns),
        Wc=w,dWc_ds=2*w,M0=moment,M1=2*moment,M2=3*moment,M3=4*moment,constant=-moment,
        meta=meta,expected_identity=identity,mesh_xy=mesh)
    assert header['complete']
    store.validate_shared_pole_bank(path,expected_identity=identity,mesh_xy=mesh,require_complete=True)
    with SlabIO(path,mode='r',mesh=mesh) as io:
        got=store.read_shared_pole_bank(io,(0,nq),meta=meta,header=header,
            sample_span=(0,ns),fields=('Wc','dWc_ds','M0','M1','M2','M3','constant'))
    for name, expected in dict(Wc=w,dWc_ds=2*w,M0=moment,M1=2*moment,M2=3*moment,M3=4*moment,constant=-moment).items():
        assert bool(jnp.all(got[name] == expected)),name
    # A parent-local constructor reads complete packed matrices on its owner.
    with SlabIO(path,mode='r',mesh=mesh) as io:
        got=store.read_shared_pole_bank(io,q_ids=[0,1,2,2],meta=meta,header=header,
            sample_span=(0,ns),fields=('Wc','constant'),partition_spec=P(('x','y'),None,None,None))
    from common.collectives import gather_to_host
    np.testing.assert_array_equal(gather_to_host(got['Wc']),raw[[0,1,2,2]])
    # Bounded native hyperslabs keep only the selected endpoint families.
    def indices(family):
        c,t=layout.carrier_extents[:2]
        width=layout.packed_extent//layout.mesh_side
        chunks=[]
        for rank in range(layout.mesh_side):
            lo=rank*width+(0 if family=='C' else c//layout.mesh_side)
            count=(c if family=='C' else 3*t)//layout.mesh_side
            chunks.extend(range(lo,lo+count))
        return np.asarray(chunks)
    for sector in (('C','C'),('C','T'),('T','C'),('T','T')):
        with SlabIO(path,mode='r',mesh=mesh) as io:
            got=store.read_shared_pole_bank(io,q_ids=[0,1,2,2],meta=meta,header=header,
                sample_span=(0,ns),fields=('Wc',),partition_spec=P(('x','y')),sector=sector)
        expected=raw[[0,1,2,2]][...,indices(sector[0]),:][...,indices(sector[1])]
        np.testing.assert_array_equal(gather_to_host(got['Wc']),expected)
