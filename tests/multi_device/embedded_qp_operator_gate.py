"""Real-CUDA P4 gate for the full-G embedded QSGW operator algebra.

Every wavefunction-sized carrier is sharded over the product ``('x','y')``
mesh axis.  The retained band matrix is deliberately small and replicated.
"""

from __future__ import annotations

from runtime import initialize_communicator_stack

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from gw.embedded_qp_operator import apply_embedded_qp_hamiltonian  # noqa: E402


def _put(array, mesh, spec):
    return jax.device_put(np.asarray(array), NamedSharding(mesh, spec))


def _gather(array):
    if jax.process_count() == 1:
        return np.asarray(array)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(array, tiled=True))


def main():
    mesh = RUNTIME.mesh
    rank0 = print if jax.process_index() == 0 else (lambda *args, **kwargs: None)
    if tuple(int(n) for n in mesh.devices.shape) != (2, 2):
        raise ValueError(f"gate requires a 2x2 mesh, got {mesh.devices.shape}")

    rng = np.random.default_rng(202608274)
    nband, nspinor, ng, nvec = 4, 2, 32, 3
    dim = nspinor * ng
    raw_basis = (
        rng.standard_normal((dim, nband))
        + 1.0j * rng.standard_normal((dim, nband))
    )
    q_basis, _ = np.linalg.qr(raw_basis)
    basis = q_basis.T.reshape(nband, nspinor, ng)
    raw_h = (
        rng.standard_normal((nband, nband))
        + 1.0j * rng.standard_normal((nband, nband))
    )
    h_w = 0.5 * (raw_h + raw_h.conj().T)
    tail_diagonal = rng.standard_normal((nspinor, ng))
    x = (
        rng.standard_normal((nvec, nspinor, ng))
        + 1.0j * rng.standard_normal((nvec, nspinor, ng))
    )

    carrier_spec = P(None, None, ("x", "y"))
    basis_spec = P(None, None, ("x", "y"))
    tail_spec = P(None, ("x", "y"))
    x_dev = _put(x, mesh, carrier_spec)
    basis_dev = _put(basis, mesh, basis_spec)
    h_w_dev = _put(h_w, mesh, P())
    tail_dev = _put(tail_diagonal, mesh, tail_spec)

    kernel = jax.jit(
        lambda ket, stored, retained, diagonal:
            apply_embedded_qp_hamiltonian(
                ket, stored, retained, lambda q: q * diagonal[None]),
        out_shardings=NamedSharding(mesh, carrier_spec),
    )
    actual_dev = kernel(x_dev, basis_dev, h_w_dev, tail_dev)
    actual_dev.block_until_ready()
    actual = _gather(actual_dev)

    rows = basis.reshape(nband, dim)
    projector = rows.T @ rows.conj()
    complement = np.eye(dim) - projector
    h_tail = np.diag(tail_diagonal.reshape(-1))
    dense = rows.T @ h_w @ rows.conj() + complement @ h_tail @ complement
    expected = np.stack([dense @ row.reshape(-1) for row in x]).reshape(x.shape)
    rel = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    actual_spec = getattr(actual_dev.sharding, "spec", None)
    if actual_spec != carrier_spec:
        raise AssertionError(
            f"full-G result lost all-P carrier {carrier_spec}: {actual_spec}")
    if rel > 3.0e-12:
        raise AssertionError(f"embedded-QP P4 relative error {rel:.3e}")
    rank0(
        "EMBEDDED_QP_P4_PASS "
        f"mesh=2x2 carrier_spec={actual_spec} relative_error={rel:.3e}"
    )


if __name__ == "__main__":
    main()
