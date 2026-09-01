"""P4 parity gate for the orbit-local panel selector."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from runtime import initialize_communicator_stack, run_main_and_finalize  # noqa: E402

RUNTIME = initialize_communicator_stack()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils as mh  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402


def _fail(message: str) -> None:
    print(f"[centroid-group-panel-p4] FAIL: {message}", flush=True)
    raise SystemExit(1)


def main() -> None:
    from common.collectives import device_put_process_local
    from common.pivoted_cholesky import (
        group_block_pivoted_cholesky_select,
        make_sharded_group_panel_pivoted_cholesky_select,
    )
    from runtime.grouped_layout import build_grouped_shard_layout

    if jax.process_count() != 4 or jax.device_count() != 4:
        _fail(
            "needs exactly four processes/four global devices; got "
            f"{jax.process_count()}/{jax.device_count()}")
    mesh = RUNTIME.mesh
    if tuple(mesh.axis_names) != ("x", "y") or mesh.devices.shape != (2, 2):
        _fail(f"expected a 2x2 ('x','y') mesh, got {mesh}")

    sizes = np.asarray([7, 6, 5, 4, 3, 2, 5, 4, 3, 2], dtype=np.int32)
    labels = np.repeat(np.arange(sizes.size, dtype=np.int32), sizes)
    logical = int(labels.size)
    budget = 28
    layout = build_grouped_shard_layout(labels, 4)
    if layout.n_padded % 4:
        _fail(f"packed extent {layout.n_padded} is not divisible by P4")
    for group in range(layout.n_groups):
        start, size = int(layout.group_start[group]), int(layout.group_size[group])
        if start // layout.shard_size != (start + size - 1) // layout.shard_size:
            _fail(f"group {group} crosses a packed row shard")

    rng = np.random.default_rng(260901)
    A = (rng.standard_normal((logical, 2 * logical))
         + 1j * rng.standard_normal((logical, 2 * logical)))
    gram = A @ A.conj().T
    gram = 0.5 * (gram + gram.conj().T)
    reference = group_block_pivoted_cholesky_select(
        jnp.asarray(gram), budget, jnp.asarray(labels),
        n_groups=int(sizes.size), tol_rel=1e-13)

    packed = layout.pack(layout.pack(gram, axis=0), axis=1)
    row = NamedSharding(mesh, P(("x", "y"), None))
    row1 = NamedSharding(mesh, P(("x", "y")))
    gram_dev = device_put_process_local(packed, row)
    group_dev = device_put_process_local(layout.packed_group_id, row1)
    canonical_dev = device_put_process_local(
        layout.packed_to_canonical.astype(np.int32), row1)
    active_dev = device_put_process_local(layout.active_mask, row1)
    selector = make_sharded_group_panel_pivoted_cholesky_select(
        mesh, layout.n_padded, budget,
        layout.group_start, layout.group_size,
        mesh_axis=("x", "y"), tol_rel=1e-13)
    got = selector(gram_dev, group_dev, canonical_dev, active_dev)
    jax.block_until_ready(got)

    piv_ref, piv_got = np.asarray(reference[0]), np.asarray(got[0])
    if not np.array_equal(piv_ref, piv_got):
        _fail(f"canonical pivot order differs: ref={piv_ref}, got={piv_got}")
    if int(reference[2]) != int(got[2]):
        _fail(f"rank differs: ref={int(reference[2])}, got={int(got[2])}")
    # Test-only bounded gathers for a 46-row synthetic fixture.  Production
    # factors/residuals remain all-P sharded and cannot be coerced to NumPy.
    factor_host = np.asarray(mh.process_allgather(got[1], tiled=True))
    residual_host = np.asarray(mh.process_allgather(got[3], tiled=True))
    np.testing.assert_allclose(
        layout.unpack(factor_host, axis=0), np.asarray(reference[1]),
        rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(
        layout.unpack(residual_host), np.asarray(reference[3]),
        rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(
        np.asarray(got[4]), np.asarray(reference[4]),
        rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(
        np.asarray(got[5]), np.asarray(reference[5]),
        rtol=3e-11, atol=3e-11)
    if tuple(got[1].sharding.spec) != (("x", "y"), None):
        _fail(f"factor layout drifted: {got[1].sharding.spec}")
    if tuple(got[3].sharding.spec) != (("x", "y"),):
        _fail(f"residual layout drifted: {got[3].sharding.spec}")

    used = piv_got[piv_got >= 0]
    counts = np.bincount(labels[used], minlength=sizes.size)
    picked = np.flatnonzero(counts)
    if not np.array_equal(counts[picked], sizes[picked]):
        _fail("selector emitted a partial canonical group")
    if jax.process_index() == 0:
        print(
            f"[centroid-group-panel-p4] logical/padded={logical}/"
            f"{layout.n_padded} shard={layout.shard_size} pad={layout.n_pad} "
            f"delivered={used.size}/{budget} groups={picked.size} "
            f"rank={int(got[2])}", flush=True)
        print("[centroid-group-panel-p4] PASS", flush=True)


if __name__ == "__main__":
    run_main_and_finalize(main)
