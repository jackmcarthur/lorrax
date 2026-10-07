"""Literal Γ Fourier, nonsymmetric endpoint and ordered spectral controls.

The normal pytest mesh is P1: these are algebra fixtures, not a P4 memory
claim.  The same callable tests can run after native P4 runtime bootstrap.
"""
import jax
import os
import subprocess
import sys
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import gather_to_host, transpose_xy
from gw import contour_reference as cd
from gw.gamma_charge_basis import (gamma_charge_basis,
                                  gamma_charge_basis_metadata, _map_program)
from gw.mixed_basis_pair_convolution import SphereSet


@pytest.fixture
def mesh():
    if jax.process_count() == 4:
        return Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def sphere(order=None, carrier=8):
    g = np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, 0],
                    [1, 0, 0], [0, -1, 0]], np.int64)
    if order is not None:
        g = g[order]
    full = np.full((1, carrier, 3), 91, np.int64)
    full[0, :len(g)] = g
    return SphereSet(full, np.asarray([len(g)]), np.zeros((1, 3)))


def face(value, mesh):
    return jax.device_put(np.asarray(value, np.complex128),
                          NamedSharding(mesh, P(None, "x", "y")))


def host(value):
    """Only tiny synthetic outputs may be gathered for independent literals."""
    assert value.nbytes <= 16 * 1024
    return np.asarray(gather_to_host(value))


@pytest.mark.parametrize("order", [None, [4, 0, 3, 2, 1]])
def test_literal_real_fourier_functions_and_physical_projector(mesh, order):
    s = sphere(order)
    u, adjoint, receipt = gamma_charge_basis(s, mesh=mesh)
    a, b = host(u)[0], host(adjoint)[0]
    g = s.gvecs[0, :5]
    points = np.asarray([[.17, .23, -.31], [.39, -.41, .07], [0., 0., 0.]])
    waves = np.zeros((8, len(points)), complex)
    waves[:5] = np.exp(2j * np.pi * g @ points.T)
    expected = np.zeros_like(waves)
    for i, vector in enumerate(g):
        if not vector.any():
            expected[i] = 1.
        elif tuple(vector) > tuple(-vector):
            expected[i] = np.sqrt(2.) * np.cos(2 * np.pi * vector @ points.T)
        else:
            expected[i] = np.sqrt(2.) * np.sin(-2 * np.pi * vector @ points.T)
    np.testing.assert_allclose(a @ waves, expected, atol=8e-16, rtol=8e-16)
    physical = np.diag(np.arange(8) < 5)
    np.testing.assert_allclose(a.conj().T @ a, physical, atol=3e-16)
    np.testing.assert_allclose(a @ a.conj().T, physical, atol=3e-16)
    np.testing.assert_array_equal(b, a.conj().T)
    np.testing.assert_array_equal(a.conj(), a[:, receipt["g_negation"]])
    assert np.count_nonzero(a[5:]) == np.count_nonzero(a[:, 5:]) == 0
    zero = int(np.flatnonzero(np.all(g == 0, axis=1))[0])
    np.testing.assert_array_equal(a[zero], np.eye(8)[zero])
    assert u.sharding == adjoint.sharding == NamedSharding(mesh, P(None, "x", "y"))
    assert receipt["replicated_metadata_bytes"] == 8 * 3 * 4


@pytest.mark.parametrize("bad", ["finiteq", "nozero", "missingpartner", "duplicate",
                                  "rowbool", "rowfloat", "rownegative", "overflow"])
def test_metadata_refuses_invalid_gamma_geometry(bad):
    s = sphere()
    row = 0
    if bad == "finiteq":
        s = SphereSet(s.gvecs, s.ngk, np.asarray([[0., .125, 0.]]))
    elif bad == "nozero":
        s = SphereSet(np.asarray([[[1, 0, 0], [-1, 0, 0]]]), np.asarray([2]), np.zeros((1, 3)))
    elif bad == "missingpartner":
        s.gvecs[0, 0] = [-2, 0, 0]
    elif bad == "duplicate":
        s.gvecs[0, 0] = s.gvecs[0, 1]
    elif bad == "rowbool":
        row = True
    elif bad == "rowfloat":
        row = 0.
    elif bad == "rownegative":
        row = -1
    elif bad == "overflow":
        s.gvecs[0, 0] = [np.iinfo(np.int64).min, 0, 0]
    with pytest.raises(ValueError):
        gamma_charge_basis_metadata(s, row=row)


def test_no_raw_array_geometry_api():
    with pytest.raises(TypeError, match="SphereSet"):
        gamma_charge_basis_metadata(np.zeros((5, 3), np.int64))


def test_square_subset_of_four_host_devices_is_refused_before_map():
    """A separate CPU bootstrap prevents the parent fixture hiding coverage."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    code = '''
import runtime
runtime.bootstrap(platform="cpu")
import jax,numpy as np
from jax.sharding import Mesh
from gw.mixed_basis_pair_convolution import SphereSet
from gw.gamma_charge_basis import gamma_charge_basis,_map_program
assert len(jax.devices())==4
s=SphereSet(np.asarray([[[0,0,0],[1,0,0],[-1,0,0],[99,99,99]]]),
            np.asarray([3]),np.zeros((1,3)))
mesh=Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),("x","y"))
before=_map_program.cache_info()
import common.collectives as cc
def forbidden(*args,**kwargs):raise AssertionError("metadata agreement ran before coverage guard")
cc.all_gather_processes=forbidden
try:gamma_charge_basis(s,mesh=mesh)
except ValueError as e:assert "every global JAX device" in str(e)
else:raise AssertionError("strict-subset map admitted")
assert _map_program.cache_info()==before
print("four-host-device subset REFUSED before map construction")
'''
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "subset REFUSED before map construction" in result.stdout


def test_cached_program_keeps_reordered_geometry_dynamic(mesh):
    first, _, receipt = gamma_charge_basis(sphere(), mesh=mesh)
    rebuilt = Mesh(np.asarray(mesh.devices), mesh.axis_names)
    assert _map_program(mesh, 8) is _map_program(rebuilt, 8)
    second, _, other = gamma_charge_basis(sphere([4, 0, 3, 2, 1]), mesh=rebuilt)
    assert receipt["signature"] != other["signature"]
    assert np.linalg.norm(host(first) - host(second)) > 1.


def test_complex_both_density_roles_nonsymmetric_operator_and_wrong_twist(mesh):
    u, adjoint, receipt = gamma_charge_basis(sphere(), mesh=mesh)
    rng = np.random.default_rng(683225)
    rho = rng.normal(size=(1, 3, 8)) + 1j * rng.normal(size=(1, 3, 8))
    operator = rng.normal(size=(1, 8, 8)) + 1j * rng.normal(size=(1, 8, 8))
    rho[..., 5:] = 0.; operator[:, 5:] = 0.; operator[..., 5:] = 0.
    reverse = rho[..., receipt["g_negation"]].conj()
    direct_r = cd.project_density_endpoints(face(rho, mesh), adjoint, mesh=mesh)
    reverse_r = cd.project_density_endpoints(face(reverse, mesh), adjoint, mesh=mesh)
    np.testing.assert_allclose(host(reverse_r), host(direct_r).conj(), atol=5e-16)
    assert np.max(abs(host(direct_r).imag)) > .1
    field_r = cd.lift_interaction_endpoints(face(operator, mesh), u, mesh=mesh, prefactor=1.)
    restored = cd.lift_interaction_endpoints(field_r, adjoint, mesh=mesh, prefactor=1.)
    np.testing.assert_allclose(host(restored), operator, rtol=3e-14, atol=3e-14)
    for original, mapped in [(rho, direct_r), (reverse, reverse_r)]:
        _, left = cd.project_interaction_diagonal(face(operator, mesh), face(original, mesh),
            mesh=mesh, prefactor=1/137., scalar_replication_bound_bytes=128)
        _, right = cd.project_interaction_diagonal(field_r, mapped,
            mesh=mesh, prefactor=1/137., scalar_replication_bound_bytes=128)
        np.testing.assert_allclose(host(left), host(right), rtol=2e-14, atol=2e-15)
    partner_r = transpose_xy(field_r, mesh)
    partner_g = cd.lift_interaction_endpoints(partner_r, adjoint, mesh=mesh, prefactor=1.)
    j = receipt["g_negation"]
    expected = operator.swapaxes(-1, -2)[:, j][:, :, j]
    np.testing.assert_allclose(host(partner_g), expected, rtol=3e-14, atol=3e-14)
    assert np.linalg.norm(expected - operator.swapaxes(-1, -2)) > 1.
    assert np.linalg.norm(expected - operator.conj()) > 1.


def test_ordered_complex_positive_residues_and_slope_round_trip(mesh):
    u, adjoint, receipt = gamma_charge_basis(sphere(), mesh=mesh)
    rng = np.random.default_rng(716236)
    v = rng.normal(size=(2, 8)) + 1j * rng.normal(size=(2, 8)); v[:, 5:] = 0.
    residues = np.einsum("pi,pj->pij", v, v.conj())
    j = receipt["g_negation"]
    reversed_residues = residues.swapaxes(-1, -2)[:, j][:, :, j]
    poles = np.asarray([.7, 1.3]); sites = np.asarray([.4+.23j, .17j, 1.7+.31j])
    values = np.zeros((len(sites), 8, 8), complex); slopes = np.zeros_like(values)
    for a, b, omega in zip(residues, reversed_residues, poles):
        values += a[None]/(sites[:, None, None]-omega) - b[None]/(sites[:, None, None]+omega)
        slopes += -a[None]/(sites[:, None, None]-omega)**2 + b[None]/(sites[:, None, None]+omega)**2
    mapped_residues = cd.lift_interaction_endpoints(face(residues, mesh), u, mesh=mesh, prefactor=1.)
    for a in host(mapped_residues):
        np.testing.assert_allclose(a, a.conj().T, rtol=1e-15, atol=1e-15)
        assert np.linalg.eigvalsh(a).min() > -1e-13
    # Real charge functions do not imply symmetric residues or an even field.
    assert np.linalg.norm(host(mapped_residues)-host(mapped_residues).swapaxes(-1, -2)) > 1.
    for original in [values, slopes, slopes/(2*sites[:, None, None])]:
        transformed = cd.lift_interaction_endpoints(face(original, mesh), u, mesh=mesh, prefactor=1.)
        restored = cd.lift_interaction_endpoints(transformed, adjoint, mesh=mesh, prefactor=1.)
        np.testing.assert_allclose(host(restored), original, rtol=3e-15, atol=3e-14)
