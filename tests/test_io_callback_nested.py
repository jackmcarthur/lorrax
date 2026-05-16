"""G0 smoke test for Round 6 — io_callback × lax.scan × shard_map × lax.all_gather.

The Round 5 unified plan flagged the four-primitive composition as "novel
territory" with no in-tree precedent.  Each pair (io_callback inside scan,
io_callback inside shard_map, scan inside shard_map, all_gather inside
shard_map) has prior art; the four-way nesting does not.  This file is the
prerequisite that gates the Round 6 kernel rewrite — if any of these tests
fail, the Path B design is wrong and we fall back to Path A.

Tests run on JAX_PLATFORMS=cpu with a 1×1 mesh (the smoke-level question is
"does the composition lower without errors and produce correct values"; mesh
size doesn't change that).  A 4-rank GPU verification of (c) ordering is
deferred to Round 6 GPU validation under SLURM allocation.

Reference: round5_unified_plan.md §3.6 (Agent 2's G0 sketch).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import io_callback
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


MESH_1x1 = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                axis_names=('x', 'y'))


# ---------------------------------------------------------------------------
# G0a — basic four-primitive composition: io_callback inside lax.scan inside
# shard_map; per-iter all_gather across the shard_map axis.  Output equals a
# numpy reference at exact precision (the only floating-point op is sum).
# ---------------------------------------------------------------------------

def test_g0a_io_callback_scan_shard_map_all_gather():
    """All four primitives nested.  Synth: small bc table, scan accumulates
    sum of all bcs across all ranks; output must equal the host array's total
    sum (every rank sees every band after all_gather)."""
    n_bc = 3
    bpd_max = 4         # bands per rank per bc (1×1 mesh ⇒ full bands per rank)
    nk = 5
    ns = 2
    ngkmax = 7

    rng = np.random.default_rng(0)
    host_tile = (rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax))
                 + 1j * rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax))).astype(
        np.complex128)

    def host_slice(x_idx, y_idx, bc_idx):
        # Resolve traced ints to Python ints inside the host fn.
        return host_tile[int(bc_idx)]   # (nk, bpd_max, ns, ngkmax)

    out_sds = jax.ShapeDtypeStruct((nk, bpd_max, ns, ngkmax), jnp.complex128)

    @partial(shard_map, mesh=MESH_1x1,
             in_specs=(), out_specs=P(),
             check_rep=False)
    def _local():
        x_idx = jax.lax.axis_index('x')
        y_idx = jax.lax.axis_index('y')

        def body(carry, bc_idx):
            slab = io_callback(host_slice, out_sds, x_idx, y_idx, bc_idx,
                                ordered=False)
            slab_full = lax.all_gather(slab, axis_name=('x', 'y'),
                                        axis=1, tiled=True)
            return carry + jnp.sum(slab_full), None

        carry, _ = lax.scan(body, jnp.complex128(0),
                             jnp.arange(n_bc, dtype=jnp.int32))
        return carry

    out = np.asarray(_local())
    expected = host_tile.sum()
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# G0b — verify each io_callback receives the correct traced bc_idx and
# returns the right slab (no off-by-one or stale-callback issue inside the
# scan body).  Construct host_tile so that bc_idx i contributes a known
# value distinguishable from the others; check the per-iter sum.
# ---------------------------------------------------------------------------

def test_g0b_traced_bc_idx_routes_to_correct_slab():
    """Each scan iter's io_callback must see the right bc_idx.  Use distinct
    per-bc magnitudes so a bc-misroute produces a wrong sum."""
    n_bc = 4
    bpd_max = 2
    nk = 3
    ns = 1
    ngkmax = 5

    # Per-bc all-ones tensor scaled by a distinct prime — sum check
    # immediately catches any iter-bc mismatch.
    primes = np.array([2.0, 3.0, 5.0, 7.0])[:n_bc]
    host_tile = np.broadcast_to(
        primes.reshape(n_bc, 1, 1, 1, 1),
        (n_bc, nk, bpd_max, ns, ngkmax)).astype(np.complex128).copy()

    def host_slice(x_idx, y_idx, bc_idx):
        return host_tile[int(bc_idx)]

    out_sds = jax.ShapeDtypeStruct((nk, bpd_max, ns, ngkmax), jnp.complex128)

    @partial(shard_map, mesh=MESH_1x1,
             in_specs=(), out_specs=P(),
             check_rep=False)
    def _local():
        x_idx = jax.lax.axis_index('x')
        y_idx = jax.lax.axis_index('y')

        def body(carry, bc_idx):
            slab = io_callback(host_slice, out_sds, x_idx, y_idx, bc_idx,
                                ordered=False)
            slab_full = lax.all_gather(slab, axis_name=('x', 'y'),
                                        axis=1, tiled=True)
            return carry + jnp.sum(slab_full), None

        carry, _ = lax.scan(body, jnp.complex128(0),
                             jnp.arange(n_bc, dtype=jnp.int32))
        return carry

    out = np.asarray(_local())
    n_per_bc = nk * bpd_max * ns * ngkmax    # entries per bc
    expected = n_per_bc * primes.sum()
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# G0c — `ordered=True` and `ordered=False` give the same numerical result.
# Round 5 §3.4 recommends `ordered=False` (per-rank ordering only is enforced
# by `lax.scan(unroll=1)` regardless); confirm the recommendation doesn't
# affect correctness.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ordered", [False, True])
def test_g0c_ordered_invariance(ordered):
    """ordered=False vs ordered=True must give the same value (correctness
    invariant — `ordered` is a perf knob, not a correctness one)."""
    n_bc = 3
    bpd_max = 2
    nk = 2
    ns = 1
    ngkmax = 4

    rng = np.random.default_rng(7)
    host_tile = (rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax))
                 + 1j * rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax))).astype(
        np.complex128)

    def host_slice(x_idx, y_idx, bc_idx):
        return host_tile[int(bc_idx)]

    out_sds = jax.ShapeDtypeStruct((nk, bpd_max, ns, ngkmax), jnp.complex128)

    @partial(shard_map, mesh=MESH_1x1,
             in_specs=(), out_specs=P(),
             check_rep=False)
    def _local():
        x_idx = jax.lax.axis_index('x')
        y_idx = jax.lax.axis_index('y')

        def body(carry, bc_idx):
            slab = io_callback(host_slice, out_sds, x_idx, y_idx, bc_idx,
                                ordered=ordered)
            slab_full = lax.all_gather(slab, axis_name=('x', 'y'),
                                        axis=1, tiled=True)
            return carry + jnp.sum(slab_full), None

        carry, _ = lax.scan(body, jnp.complex128(0),
                             jnp.arange(n_bc, dtype=jnp.int32))
        return carry

    out = np.asarray(_local())
    expected = host_tile.sum()
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# G0d — non-trivial per-iter compute (mirrors the kernel body's contract):
# scan body slices the gathered tensor and accumulates a rank-5
# accumulator.  Validates that scan carry (rank-5 c128) survives across
# iters and the all_gather output participates correctly in einsum.
# ---------------------------------------------------------------------------

def test_g0d_rank5_accumulator_with_einsum():
    """Mirror the kernel body: scan body pulls a (nk, bpd_max, ns, ngkmax)
    slab via io_callback, all_gathers along the band axis, and einsums into
    a rank-5 accumulator.  Output equals the all-bcs einsum reference."""
    n_bc = 3
    bpd_max = 2
    nk = 2
    ns = 1
    ngkmax = 3
    n_rmu_loc = 4
    r_loc = 5

    rng = np.random.default_rng(11)
    psi_G_host = (rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax))
                  + 1j * rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax))).astype(
        np.complex128)
    # psi_X is band-replicated; pre-build per-bc, full-bands.
    psi_X_full = (rng.standard_normal((n_bc, nk, n_rmu_loc, bpd_max, ns))
                  + 1j * rng.standard_normal((n_bc, nk, n_rmu_loc, bpd_max, ns))).astype(
        np.complex128)
    # psi_Y: arbitrary deterministic (replace the pre-IFFT step with passthrough
    # for this composition test — we just need a (nk, bpd_max, ns, r_loc) slab
    # per iter to feed the einsum).  For simplicity, define psi_Y_host as
    # a separate tensor of the right shape per bc.
    psi_Y_host = (rng.standard_normal((n_bc, nk, bpd_max, ns, r_loc))
                  + 1j * rng.standard_normal((n_bc, nk, bpd_max, ns, r_loc))).astype(
        np.complex128)

    def host_psi_Y(x_idx, y_idx, bc_idx):
        return psi_Y_host[int(bc_idx)]

    out_sds = jax.ShapeDtypeStruct((nk, bpd_max, ns, r_loc), jnp.complex128)
    psi_X_const = jnp.asarray(psi_X_full)            # static closure

    @partial(shard_map, mesh=MESH_1x1,
             in_specs=(), out_specs=P(),
             check_rep=False)
    def _local():
        x_idx = jax.lax.axis_index('x')
        y_idx = jax.lax.axis_index('y')

        # Rank-5 carry.  On 1×1 mesh, full extent.
        P_acc = jnp.zeros((nk, ns, r_loc, n_rmu_loc, ns), dtype=jnp.complex128)

        def body(carry, bc_idx):
            slab = io_callback(host_psi_Y, out_sds, x_idx, y_idx, bc_idx,
                                ordered=False)
            slab_full = lax.all_gather(slab, axis_name=('x', 'y'),
                                        axis=1, tiled=True)   # (nk, bpd_max, ns, r_loc)
            # psi_X for this bc (closure-captured table; on production this is
            # a dynamic_slice on the consumer-side replicated psi_l_X).
            psi_X_bc = lax.dynamic_slice_in_dim(
                psi_X_const, bc_idx, 1, axis=0).reshape(
                nk, n_rmu_loc, bpd_max, ns)
            delta = jnp.einsum('kmna,knbr->karmb', psi_X_bc, slab_full,
                                optimize=True)
            return carry + delta, None

        P_final, _ = lax.scan(body, P_acc,
                               jnp.arange(n_bc, dtype=jnp.int32))
        return P_final

    out = np.asarray(_local())

    # Reference: same einsum across all bcs, summed.
    ref = np.zeros((nk, ns, r_loc, n_rmu_loc, ns), dtype=np.complex128)
    for bc in range(n_bc):
        ref += np.einsum('kmna,knbr->karmb', psi_X_full[bc], psi_Y_host[bc],
                          optimize=True)

    # Allow ULP-class drift from the einsum order; should be tight.
    np.testing.assert_allclose(out, ref, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# G0e — short-final-bc handling: pad rows must be EXACTLY zero (np.zeros,
# not np.empty) so the einsum-after-mask is math-neutral.  This validates
# the §3.2 pad-rows-must-be-zero contract from the unified plan.
# ---------------------------------------------------------------------------

def test_g0e_zero_padded_short_bc():
    """Last bc has fewer-than-bpd_max bands; pad rows must be zero so the
    masked einsum produces the same value as if those rows were absent."""
    n_bc = 3
    bpd_max = 4
    nk = 2
    ns = 1
    ngkmax = 3
    # bc 0 + 1 are full (bpd=4); bc 2 is short (bpd=1, padded with 3 zeros).
    bpd_per_bc = (4, 4, 1)

    rng = np.random.default_rng(23)
    host_tile = np.zeros((n_bc, nk, bpd_max, ns, ngkmax), dtype=np.complex128)
    for bc, bpd in enumerate(bpd_per_bc):
        host_tile[bc, :, :bpd, :, :] = (
            rng.standard_normal((nk, bpd, ns, ngkmax))
            + 1j * rng.standard_normal((nk, bpd, ns, ngkmax)))

    def host_slice(x_idx, y_idx, bc_idx):
        return host_tile[int(bc_idx)]
    out_sds = jax.ShapeDtypeStruct((nk, bpd_max, ns, ngkmax), jnp.complex128)

    # Mask table: per-bc, valid rows are [0, bpd_per_bc[bc]).
    bpd_table = jnp.asarray(bpd_per_bc, dtype=jnp.int32)

    @partial(shard_map, mesh=MESH_1x1,
             in_specs=(), out_specs=P(),
             check_rep=False)
    def _local():
        x_idx = jax.lax.axis_index('x')
        y_idx = jax.lax.axis_index('y')

        def body(carry, bc_idx):
            slab = io_callback(host_slice, out_sds, x_idx, y_idx, bc_idx,
                                ordered=False)
            slab_full = lax.all_gather(slab, axis_name=('x', 'y'),
                                        axis=1, tiled=True)
            band_idx = jnp.arange(bpd_max, dtype=jnp.int32)
            mask = band_idx < bpd_table[bc_idx]
            slab_masked = jnp.where(mask[None, :, None, None], slab_full, 0)
            return carry + jnp.sum(slab_masked), None

        carry, _ = lax.scan(body, jnp.complex128(0),
                             jnp.arange(n_bc, dtype=jnp.int32))
        return carry

    out = np.asarray(_local())
    # Reference: only valid rows contribute (pad rows in host_tile already
    # zero, so masking is a no-op for this test — but the test confirms the
    # masking idiom doesn't break the scan).
    expected = host_tile.sum()
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0)
