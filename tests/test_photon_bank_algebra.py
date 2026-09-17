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
    samples, moments, receipt = response_algebra(
        meta, {"linalg": "local"}, mesh_xy=mesh, n=4, photon=True)
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
