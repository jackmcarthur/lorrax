"""The write price follows ``write_slab``'s route, including file-order pieces (CPU only).

A ζ tile ``(Q, μ, G_tile)`` at ``P(None, ('x','y'), None)`` is written as-is only
when each rank's file run, ``ceil(μ/P)·G·16`` bytes, reaches ``_FILE_ORDER_MIN_RUN``
(1 MiB).  A smaller tile leaves as Q-row pieces.  ``write_device_scratch`` used to
refuse that route, which stopped every P4 ``x_only`` producer between about 1 S
and 4 S (Si SOC, N_μ 120-332).  Run with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import numpy as np
import pytest

import runtime  # noqa: F401  (runtime before jax: the repo's bootstrap)
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

ZETA_SPEC = P(None, ("x", "y"), None)


def _cpu_mesh():
    cpu = jax.devices("cpu")
    if len(cpu) < 4:
        pytest.skip("needs 4 host CPU devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(cpu[:4]).reshape(2, 2), ("x", "y"))


def _tile(mesh, shape):
    return jax.ShapeDtypeStruct(shape, jnp.complex128,
                                sharding=NamedSharding(mesh, ZETA_SPEC))


def test_small_zeta_tile_takes_the_piece_route_and_a_large_one_goes_as_is():
    from file_io._slab_io_ffi import _write_route
    mesh = _cpu_mesh()
    # Q = 8, μ = 144, one G tile of 320: 36·320·16 B = 184 kB per rank < 1 MiB.
    assert _write_route((8, 144, 320), (8, 144, 320), ZETA_SPEC, 16, mesh,
                        (8, 144, 320)) == (0, 8, 1)
    # μ = 512, G = 640: 128·640·16 B = 1.31 MB per rank.
    assert _write_route((8, 512, 640), (8, 512, 640), ZETA_SPEC, 16, mesh,
                        (8, 512, 640)) == "as-is"


def test_piece_route_is_priced_from_the_exchange_it_runs():
    from file_io._slab_io_ffi import _file_order_piece_bytes, _write_route
    mesh = _cpu_mesh()
    shape = (8, 144, 320)
    value = _tile(mesh, shape)
    plan = _write_route(shape, shape, ZETA_SPEC, 16, mesh, shape)
    seen = []
    peak = _file_order_piece_bytes(value, shape, plan, mesh,
                                   lambda piece: seen.append(piece) or 0)
    local = 16 * shape[0] * (shape[1] // 4) * shape[2]
    piece_local = 16 * (8 // 4) * shape[1] * shape[2]
    assert [tuple(s.shape) for s in seen] == [(8, 144, 320)]
    assert seen[0].sharding.spec == P(("x", "y"), None, None)
    assert piece_local <= peak <= 4 * max(local, piece_local)
