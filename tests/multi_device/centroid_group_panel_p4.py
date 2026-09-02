"""P4 gate for square orbit layout, local actions, and panel parity."""
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
    from common.shard_map import shard_map
    from common.grouped_layout import build_square_grouped_shard_layout
    from symmetry_maps import (
        centroid_source_map_and_wrap,
        permutation_orbit_labels,
    )

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
    square = build_square_grouped_shard_layout(labels, (2, 2))
    layout = square.fine
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

    packed = layout.pack_host(layout.pack_host(gram, axis=0), axis=1)
    row = NamedSharding(mesh, P(("x", "y"), None))
    gram_dev = device_put_process_local(packed, row)
    selector = make_sharded_group_panel_pivoted_cholesky_select(
        mesh, budget, layout,
        mesh_axis=("x", "y"), tol_rel=1e-13)
    got = selector(gram_dev)
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
        layout.unpack_host(factor_host, axis=0), np.asarray(reference[1]),
        rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(
        layout.unpack_host(residual_host), np.asarray(reference[3]),
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

    # Exact rank-floor fixture: every element of the seven-tuple is stable,
    # including the PSD receipt.  This is padded and exercises the canonical
    # post-floor group order rather than sub-floor roundoff.
    floor_sizes = np.asarray([4, 3, 5, 2, 4], dtype=np.int32)
    floor_labels = np.repeat(
        np.arange(floor_sizes.size, dtype=np.int32), floor_sizes)
    floor_logical, floor_budget = int(floor_labels.size), 14
    floor_diag = np.zeros((floor_logical,), dtype=np.float64)
    floor_diag[:7] = np.arange(18., 11., -1.)
    floor_gram = np.diag(floor_diag).astype(np.complex128)
    floor_ref = group_block_pivoted_cholesky_select(
        jnp.asarray(floor_gram), floor_budget, jnp.asarray(floor_labels),
        n_groups=floor_sizes.size, tol_rel=1e-10)
    floor_square = build_square_grouped_shard_layout(floor_labels, (2, 2))
    floor_layout = floor_square.fine
    if floor_layout.n_pad <= 0:
        _fail("rank-floor fixture must exercise internal fine-shard pads")
    floor_packed = floor_layout.pack_host(
        floor_layout.pack_host(floor_gram, axis=0), axis=1)
    floor_dev = device_put_process_local(floor_packed, row)
    floor_got = make_sharded_group_panel_pivoted_cholesky_select(
        mesh, floor_budget, floor_layout, mesh_axis=("x", "y"),
        tol_rel=1e-10)(floor_dev)
    jax.block_until_ready(floor_got)
    np.testing.assert_array_equal(
        np.asarray(floor_got[0]), np.asarray(floor_ref[0]))
    if int(floor_got[2]) != 7 or int(floor_ref[2]) != 7:
        _fail("rank-floor fixture rank drifted")
    floor_factor = np.asarray(mh.process_allgather(floor_got[1], tiled=True))
    floor_residual = np.asarray(mh.process_allgather(floor_got[3], tiled=True))
    np.testing.assert_allclose(
        floor_layout.unpack_host(floor_factor, axis=0), np.asarray(floor_ref[1]),
        rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        floor_layout.unpack_host(floor_residual), np.asarray(floor_ref[3]),
        rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        np.asarray(floor_got[4]), np.asarray(floor_ref[4]),
        rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        np.asarray(floor_got[5]), np.asarray(floor_ref[5]),
        rtol=2e-13, atol=2e-13)
    for got_receipt, ref_receipt in zip(floor_got[6], floor_ref[6]):
        np.testing.assert_array_equal(
            np.asarray(got_receipt), np.asarray(ref_receipt))

    # More shards than groups: two ranks own only pads.  They must still
    # participate in collectives without contributing a pivot or diagnostic.
    sparse_labels = np.repeat(
        np.arange(2, dtype=np.int32), np.asarray([5, 3], dtype=np.int32))
    sparse_gram = np.diag(np.arange(8., 0., -1.)).astype(np.complex128)
    sparse_ref = group_block_pivoted_cholesky_select(
        jnp.asarray(sparse_gram), 8, jnp.asarray(sparse_labels),
        n_groups=2, tol_rel=1e-10)
    sparse_square = build_square_grouped_shard_layout(sparse_labels, (2, 2))
    sparse_layout = sparse_square.fine
    if np.count_nonzero(sparse_layout.shard_load == 0) != 2:
        _fail("all-pad-shard fixture did not create two empty owners")
    sparse_packed = sparse_layout.pack_host(
        sparse_layout.pack_host(sparse_gram, axis=0), axis=1)
    sparse_got = make_sharded_group_panel_pivoted_cholesky_select(
        mesh, 8, sparse_layout, mesh_axis=("x", "y"), tol_rel=1e-10)(
            device_put_process_local(sparse_packed, row))
    jax.block_until_ready(sparse_got)
    np.testing.assert_array_equal(
        np.asarray(sparse_got[0]), np.asarray(sparse_ref[0]))
    np.testing.assert_array_equal(
        np.asarray(sparse_got[4]), np.asarray(sparse_ref[4]))
    if int(sparse_got[2]) != 8:
        _fail(f"all-pad-shard fixture rank={int(sparse_got[2])}, expected 8")

    # Real symmetry-service tables: a glide reflection on a 10x2 grid gives
    # ten two-point orbits, nonzero lattice wraps, and four internal pads
    # under P4.  TR rows duplicate the coordinate action.
    fft = np.asarray([10, 2, 1], dtype=np.int32)
    coords = np.indices(tuple(fft), dtype=np.int32).reshape(3, -1).T
    sym_matrices = np.stack([
        np.eye(3, dtype=np.int32),
        np.diag([1, -1, 1]).astype(np.int32),
    ])
    translations = np.zeros((2, 3), dtype=np.float64)
    translations[1, 0] = np.pi
    permutations, wraps = centroid_source_map_and_wrap(
        coords, sym_matrices, translations, fft,
        validate=True, extend_trs=True)
    np.testing.assert_array_equal(permutations[2:], permutations[:2])
    np.testing.assert_array_equal(wraps[2:], wraps[:2])
    if not np.any(wraps[1]):
        _fail("nonsymmorphic fixture did not produce a lattice wrap")
    action_labels = permutation_orbit_labels(permutations)
    action_square = build_square_grouped_shard_layout(action_labels, (2, 2))
    action_layout = action_square.fine
    if action_layout.n_pad != 4:
        _fail(f"expected four internal pads, got {action_layout.n_pad}")
    for group in range(action_layout.n_groups):
        rows = np.flatnonzero(action_layout.packed_group_id == group)
        if np.unique(rows // action_layout.shard_size).size != 1:
            _fail(f"action orbit {group} crosses a fine selector shard")
        if np.unique(rows // action_square.axis_shard_size).size != 1:
            _fail(f"action orbit {group} crosses the shared X/Y view")

    n_action = int(coords.shape[0])
    canonical_values = (
        np.arange(n_action, dtype=np.float64)
        + 1j * np.arange(n_action, dtype=np.float64)[::-1])
    packed_values = action_layout.pack_host(canonical_values, fill_value=0.0)
    flat_sharding = NamedSharding(mesh, P(("x", "y")))
    x_sharding = NamedSharding(mesh, P("x"))
    y_sharding = NamedSharding(mesh, P("y"))
    flat_values = device_put_process_local(packed_values, flat_sharding)
    x_values = device_put_process_local(packed_values, x_sharding)
    y_values = device_put_process_local(packed_values, y_sharding)

    # Authenticate JAX's actual mesh-order split rather than assuming it.
    for array, expected_local in (
        (flat_values, action_layout.shard_size),
        (x_values, action_square.axis_shard_size),
        (y_values, action_square.axis_shard_size),
    ):
        shard = array.addressable_shards[0]
        slc = shard.index[0]
        if int(slc.stop - slc.start) != expected_local:
            _fail(f"unexpected local view extent {slc} for {array.sharding}")
        np.testing.assert_array_equal(
            np.asarray(shard.data), packed_values[slc])

    local_indices = action_layout.pack_fine_local_permutations_host(
        permutations[1:2])[0]
    index_dev = device_put_process_local(local_indices, flat_sharding)
    packed_wrap = action_layout.pack_host(wraps[1], axis=0, fill_value=0)
    q_frac = np.asarray([0.125, 0.0, 0.0])
    packed_phase = np.exp(2j * np.pi * (packed_wrap @ q_frac))
    phase_dev = device_put_process_local(packed_phase, flat_sharding)

    def make_local_action(spec):
        @jax.jit
        def local_action(values, indices, phase):
            def body(value_slab, index_slab, phase_slab):
                return value_slab[index_slab] * phase_slab
            return shard_map(
                body, mesh=mesh,
                in_specs=(spec,) * 3, out_specs=spec, check_vma=False,
            )(values, indices, phase)
        return local_action

    flat_action = make_local_action(P(("x", "y")))
    acted = flat_action(flat_values, index_dev, phase_dev)
    jax.block_until_ready(acted)
    acted_host = np.asarray(mh.process_allgather(acted, tiled=True))
    expected_action = (
        canonical_values[permutations[1]]
        * np.exp(2j * np.pi * (wraps[1] @ q_frac)))
    np.testing.assert_allclose(
        action_layout.unpack_host(acted_host), expected_action,
        rtol=2e-14, atol=2e-14)
    forbidden = (
        "all-reduce(", "all-gather(", "all-to-all(",
        "collective-permute(", "reduce-scatter(",
    )
    def check_hlo(action, values, indices, phase, view):
        hlo = action.lower(values, indices, phase).compiler_ir(
            dialect="hlo").as_hlo_text().lower()
        present = [name for name in forbidden if name in hlo]
        if present:
            _fail(f"packed {view} action introduced collectives: {present}")

    check_hlo(flat_action, flat_values, index_dev, phase_dev, "flat")

    axis_indices = action_square.pack_axis_local_permutations_host(
        permutations[1:2])[0]
    expected_packed_action = action_layout.pack_host(
        expected_action, fill_value=0.0)
    for view, spec, values in (
        ("x", P("x"), x_values),
        ("y", P("y"), y_values),
    ):
        sharding = NamedSharding(mesh, spec)
        indices = device_put_process_local(axis_indices, sharding)
        phase = device_put_process_local(packed_phase, sharding)
        action = make_local_action(spec)
        result = action(values, indices, phase)
        jax.block_until_ready(result)
        shard = result.addressable_shards[0]
        np.testing.assert_allclose(
            np.asarray(shard.data), expected_packed_action[shard.index[0]],
            rtol=2e-14, atol=2e-14)
        check_hlo(action, values, indices, phase, view)

    if jax.process_index() == 0:
        print(
            f"[centroid-group-panel-p4] logical/padded={logical}/"
            f"{layout.n_padded} shard={layout.shard_size} pad={layout.n_pad} "
            f"delivered={used.size}/{budget} groups={picked.size} "
            f"rank={int(got[2])}", flush=True)
        print(
            "[centroid-group-panel-p4] square views/local nonsymmorphic "
            f"action PASS: logical/padded={n_action}/"
            f"{action_layout.n_padded}, "
            "HLO collectives=0",
            flush=True)
        print("[centroid-group-panel-p4] PASS", flush=True)


if __name__ == "__main__":
    run_main_and_finalize(main)
