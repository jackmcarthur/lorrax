"""P4 spectral witnesses for stable, paired native eigenvalue ordering."""
import numpy as np
import pytest

from psp.run_dense_h import RUNTIME, order_dense_eigenpairs
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import gather_to_host, single_device_mesh


def test_ulp_inversion_preserves_complex_spectral_operator_on_all_p_face():
    mesh = RUNTIME.mesh
    n = 8
    x = np.arange(n)
    coefficients = np.exp(2j * np.pi * x[:, None] * x[None] / n) / np.sqrt(n)
    energies = np.array([0., 1., 2., np.nextafter(2., -np.inf), 3., 4., 5., 6.])
    assert np.min(np.diff(energies)) < 0
    spec = P('x', 'y')
    vectors = jax.device_put(coefficients, NamedSharding(mesh, spec))
    ordered, rotated = order_dense_eigenpairs(energies, vectors, mesh=mesh, spec=spec)
    assert isinstance(rotated.sharding, NamedSharding)
    assert rotated.sharding.mesh is mesh and rotated.sharding.spec == spec
    assert len(rotated.sharding.device_set) == int(mesh.size)
    actual = np.asarray(gather_to_host(rotated))  # explicitly bounded8x8 fixture
    hamiltonian = (coefficients * energies) @ coefficients.conj().T
    assert np.min(np.diff(ordered)) >= 0
    assert np.max(np.abs(hamiltonian @ actual - actual * ordered)) < 1e-12
    assert np.max(np.abs(actual.conj().T @ actual - np.eye(n))) < 1e-12
    assert np.max(np.abs((actual * ordered) @ actual.conj().T - hamiltonian)) < 1e-12
    assert not np.array_equal(actual, coefficients)


def test_exact_ties_preserve_column_order_on_existing_k_owner_mesh():
    mesh = single_device_mesh()
    energies = np.array([0., 3., 1., 1., 2., 3., 4., 5.])
    coefficients = np.diag(np.exp(1j * np.arange(8)))
    vectors = jax.device_put(coefficients, NamedSharding(mesh, P(None, None)))
    ordered, rotated = order_dense_eigenpairs(energies, vectors, mesh=mesh, spec=P(None, None))
    actual = np.asarray(rotated)
    assert np.array_equal(ordered, np.array([0., 1., 1., 2., 3., 3., 4., 5.]))
    assert np.array_equal(actual[:, 1:3], coefficients[:, 2:4])
    assert np.array_equal(actual[:, 4], coefficients[:, 1])
    assert np.array_equal(actual[:, 5], coefficients[:, 5])
    hamiltonian = (coefficients * energies) @ coefficients.conj().T
    assert np.max(np.abs(hamiltonian @ actual - actual * ordered)) < 1e-12


@pytest.mark.parametrize('energies', [np.arange(7.), np.array([0., 1., 2., 3., 4., 5., 6., np.nan])])
def test_incomplete_or_nonfinite_energy_metadata_refused(energies):
    mesh = single_device_mesh()
    vectors = jax.device_put(np.eye(8, dtype=np.complex128), NamedSharding(mesh, P(None, None)))
    with pytest.raises(ValueError, match='finite matching'):
        order_dense_eigenpairs(energies, vectors, mesh=mesh, spec=P(None, None))
