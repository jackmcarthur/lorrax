"""The planned eigh does not donate or overwrite its operand (DONATES['eigh'] == ()).

The CUDA provider once handed the XLA input tile to cuSOLVERMp Syevd, which
overwrites its operand, and shared-pole construction then reused M1 after its
eigenvectors (KNOWN_LORRAX_ISSUES 2026-09-15 TRMOM). This host-mesh test pins
the contract on the provider-free route and on every host provider that
resolves; the CUDA provider has its own P4/P16 detector
(runs/frequency_integration_sandbox/424_trmom_20260915/harness/m3_inplace_test.py).
"""
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("needs 4 host devices")
    return Mesh(np.array(devices[:4]).reshape(2, 2), ("x", "y"))


def _hermitian_stack(n, batch=2, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(batch, n, n)) + 1j * rng.normal(size=(batch, n, n))
    return a + a.conj().swapaxes(-1, -2)


def _plan_or_skip(distrib_la, mesh, n, backend):
    # "off" is the provider-free native route and always resolves; a provider arm
    # runs only where the host build resolves it for this mesh.
    if backend != "off":
        status = str(distrib_la.list_backends("eigh", mesh).get(backend, "absent"))
        if not status.startswith("available"):
            pytest.skip(f"eigh backend {backend} on this host mesh: {status}")
    return distrib_la.plan("eigh", mesh, n=n, backend=backend)


@pytest.mark.parametrize("backend", ["off", "scalapack", "slate"])
def test_planned_eigh_and_leading_eigenvectors_keep_the_operand(backend):
    import distrib_la

    mesh = _mesh()
    n = 8
    plan = _plan_or_skip(distrib_la, mesh, n, backend)
    assert distrib_la.DONATES["eigh"] == ()
    host = _hermitian_stack(n)
    operand = jax.device_put(host, NamedSharding(mesh, P(None, "x", "y")))
    before = np.asarray(operand).copy()

    values, vectors = plan.batched(operand)
    jax.block_until_ready((values, vectors))
    np.testing.assert_array_equal(np.asarray(operand), before)

    directions, kept = distrib_la.leading_eigenvectors(
        operand, 3, eigh_plan=plan, column_extent=lambda width: width + width % 2)
    jax.block_until_ready((directions, kept))
    np.testing.assert_array_equal(np.asarray(operand), before)

    # Negative control: an operand that IS changed must fail the same comparison.
    changed = jax.device_put(host.copy(), NamedSharding(mesh, P(None, "x", "y")))
    changed = changed.at[0, 0, 0].add(1.0)
    assert not np.array_equal(np.asarray(changed), before)
