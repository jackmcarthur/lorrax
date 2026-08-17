"""P=4 gate for the WP3-A distributed resolvent path."""
from __future__ import annotations

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import process_count, process_rank, resolve_mesh  # noqa: E402
from gw.mpa import intraband_block as IB  # noqa: E402


def _put(mesh, value, spec):
    value = np.asarray(value)
    sharding = NamedSharding(mesh, spec)
    return jax.make_array_from_callback(
        value.shape, sharding, lambda index: value[index])


def main():
    if process_count() != 4:
        raise RuntimeError(
            f"intraband contour gate requires four processes, got "
            f"{process_count()}")
    mesh = resolve_mesh()
    if tuple(int(value) for value in mesh.devices.shape) != (2, 2):
        raise RuntimeError("intraband contour gate requires a 2x2 mesh")

    rng = np.random.default_rng(317)
    n_pair, n_mu = 18, 4
    raw = (rng.normal(size=(n_mu, n_mu))
           + 1j * rng.normal(size=(n_mu, n_mu)))
    W0 = 0.01 * (raw + raw.conj().T) + 0.18 * np.eye(n_mu)
    vertices = (rng.normal(size=(n_pair, n_mu))
                + 1j * rng.normal(size=(n_pair, n_mu))) / 4.0
    u = np.linspace(0.04, 0.19, n_pair)
    w = np.linspace(2.0e-4, 1.1e-3, n_pair)
    block = (
        _put(mesh, u, P(None)),
        _put(mesh, w, P(None)),
        (_put(mesh, vertices, P(None, "x")),
         _put(mesh, vertices, P(None, "y"))),
    )
    W0j = _put(mesh, W0, P("x", "y"))
    zetas = (0.0j, 0.03 + 0.02j, 0.11 - 0.04j)
    got_j = IB._resolvents_at_zeta(block, W0j, zetas)
    if got_j.sharding.spec != P(None, "x", "y"):
        raise AssertionError(
            f"resolvent stack has sharding {got_j.sharding.spec}")
    if any(tuple(shard.data.shape) != (len(zetas), 2, 2)
           for shard in got_j.addressable_shards):
        raise AssertionError("a rank materialized a full N_mu-square result")
    got = np.asarray(multihost_utils.process_allgather(got_j, tiled=True))

    want = []
    for zeta in zetas:
        d = -2.0 * w / (u * u - zeta)
        chi = vertices.T @ (d[:, None] * vertices.conj())
        want.append(np.linalg.solve(
            np.eye(n_mu) - W0 @ chi, W0 @ chi @ W0))
    want = np.asarray(want)
    relative = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    if relative > 1.0e-11:
        raise AssertionError(f"distributed resolvent relative error {relative}")

    M_j, V_j = IB._exact_moment_totals(block, W0j)
    M = np.asarray(multihost_utils.process_allgather(M_j, tiled=True))
    V = np.asarray(multihost_utils.process_allgather(V_j, tiled=True))
    C1 = vertices.T @ ((2.0 * w)[:, None] * vertices.conj())
    m_relative = float(
        np.linalg.norm(M - W0 @ C1 @ W0) / np.linalg.norm(W0 @ C1 @ W0))
    v_relative = float(np.linalg.norm(V - want[0]) / np.linalg.norm(want[0]))
    if m_relative > 1.0e-11 or v_relative > 1.0e-11:
        raise AssertionError(
            f"direct totals mismatch M={m_relative} V={v_relative}")
    if process_rank() == 0:
        print(
            "[intraband-contour-p4] PASS world=4 mesh=2x2 "
            f"resolvent_rel={relative:.3e} M_rel={m_relative:.3e} "
            f"V_rel={v_relative:.3e} local_shape=(3,2,2)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    finalize_process(main())
