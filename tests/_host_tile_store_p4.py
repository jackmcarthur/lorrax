"""P4 child: HostTileStore exact round trip on uneven tiles, plus red twins.

Run one process per rank (``tests/test_host_tile_store_p4.py`` launches it):

    python tests/_host_tile_store_p4.py --dir <shared dir>

The logical array is deliberately non-divisible (TASTE 11): μ = 37 over a
2x2 mesh (carrier 40 from ``runtime.padding``) and G = 23 cut into tiles of
8 (last tile 7 wide).  Every tile is padded to one carrier, stored in pinned
host memory, fetched back, spilled through slab_io and reloaded; the
logical array is reassembled on host and compared bit for bit.

Red twins, each of which must fire for the run to count:

* the exactness check reports a swapped tile as a mismatch;
* a tile carrier whose sharded axis is not mesh-divisible is refused;
* a put whose sharding is not the store's is refused (no silent reshard);
* spilling a partially written store is refused.
"""
from __future__ import annotations

import argparse
import os
import sys

from runtime import initialize_communicator_stack

RT = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import device_put_process_local, gather_to_host  # noqa: E402
from file_io.host_tile_store import HostTileStore  # noqa: E402
from runtime.padding import PaddedAxis, padded_axis  # noqa: E402

XY = ("x", "y")
SPEC = P(None, XY, None)
Q, MU, NG, GT = 5, 37, 23, 8


def _logical():
    q, m, g = np.meshgrid(np.arange(Q), np.arange(MU), np.arange(NG),
                          indexing="ij")
    return (q * 1000.0 + m + g * 1e-3) + 1j * (q - m * 1e-2 + g)


def _exact(a, b) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args(argv)
    mesh = RT.mesh
    assert jax.process_count() == 4 and dict(mesh.shape) == {"x": 2, "y": 2}, (
        f"P4 contract needs a 2x2 mesh over 4 processes; got "
        f"{jax.process_count()} processes, mesh {dict(mesh.shape)}")
    sh = NamedSharding(mesh, SPEC)
    mu = padded_axis(MU, mesh, name="mu", spec=SPEC, axis=1)       # 37 -> 40
    n_t = -(-NG // GT)
    g_tags = [PaddedAxis(name=f"G tile {t}", logical=min(GT, NG - t * GT),
                         carrier=GT, divisor=1) for t in range(n_t)]
    Z = _logical()

    def carrier(t):
        c = np.zeros((Q, mu.carrier, GT), np.complex128)
        w = g_tags[t].logical
        c[:, :mu.logical, :w] = Z[:, :, t * GT:t * GT + w]
        return c

    cells, failures = 0, []

    def cell(name, ok):
        nonlocal cells
        cells += 1
        if not ok:
            failures.append(name)
        if jax.process_index() == 0:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}", flush=True)

    store = HostTileStore(mesh=mesh, grid=(n_t,), tile_shape=(Q, mu.carrier, GT),
                          spec=SPEC, dtype=np.complex128)
    for t in range(n_t):
        store.put_tile_async(t, device_put_process_local(carrier(t), sh))
    store.wait()
    cell("tiles live in pinned_host memory",
         all(store.host_tile(t).sharding.memory_kind == "pinned_host"
             for t in range(n_t)))
    got = [gather_to_host(store.get_tile_async(t)) for t in range(n_t)]
    cell("every carrier tile round-trips exactly",
         all(_exact(got[t], carrier(t)) for t in range(n_t)))
    cell("pad region comes back exactly zero",
         all(not np.any(got[t][:, mu.logical:, :])
             and not np.any(got[t][:, :, g_tags[t].logical:])
             for t in range(n_t)))
    back = np.concatenate([got[t][:, :mu.logical, :g_tags[t].logical]
                           for t in range(n_t)], axis=2)
    cell("logical array reassembles exactly", _exact(back, Z))
    cell("red twin: swapped tile is reported as a mismatch",
         not _exact(got[1], carrier(0)))

    path = os.path.join(args.dir, "host_tile_store_p4.h5")
    store.to_disk(path)
    again = HostTileStore.from_disk(path, mesh=mesh, spec=SPEC)
    cell("slab_io spill + reload is exact",
         again.grid == store.grid and again.tile_shape == store.tile_shape
         and all(_exact(gather_to_host(again.get_tile_async(t)), carrier(t))
                 for t in range(n_t)))
    if jax.process_index() == 0:
        os.unlink(path)

    try:
        HostTileStore(mesh=mesh, grid=(1,), tile_shape=(Q, MU, GT), spec=SPEC,
                      dtype=np.complex128)
        refused = False
    except ValueError:
        refused = True
    cell("red twin: non-divisible tile carrier is refused", refused)
    try:
        other = NamedSharding(mesh, P(None, None, XY))
        store.put_tile_async(0, device_put_process_local(carrier(0), other))
        refused = False
    except ValueError:
        refused = True
    cell("red twin: foreign sharding is refused, not resharded", refused)
    partial = HostTileStore(mesh=mesh, grid=(2,), tile_shape=(Q, mu.carrier, GT),
                            spec=SPEC, dtype=np.complex128)
    partial.put_tile_async(0, device_put_process_local(carrier(0), sh))
    try:
        partial.to_disk(os.path.join(args.dir, "never_written.h5"))
        refused = False
    except ValueError:
        refused = True
    cell("red twin: partial store spill is refused", refused)
    store.close()
    again.close()
    partial.close()

    if jax.process_index() == 0:
        print(f"host_tile_store P4: {cells} cells ran, {len(failures)} failures"
              + (f" {failures}" if failures else ""), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
