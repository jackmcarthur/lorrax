"""Pins for the 2026-08-07 mini-BZ geometry consolidation.

The 3D body head and the q→0 head were independent implementations of
draw → wrap → affine → kernel that diverged nine ways; the transposed-draw
bug (fixed 358bb0b) lived in exactly the axis they duplicated.  The
consolidation single-sourced the GEOMETRY (``minibz_frac_to_cart``,
``minibz_cell_affine``, the shared bare kernel) and deliberately kept the
estimator/generator/nmax/volume divergences (registered owner questions —
each would move the frozen si_bse_debug reference beyond the 358bb0b fix).

These cells pin three things: the primitive owns the row convention (with
the anti-tautology twin showing the transposed spelling actually differs
on the cell used), the production body head is bit-stable across the
consolidation (golden values captured at the pre-consolidation head,
b21b8679), and the body head IS the composition of the shared pieces
(so the wiring cannot silently fork again).
"""
import numpy as np
import pytest

from vcoul import (
    build_v_head_miniBZ_avg_3d,
    minibz_cell_affine,
    minibz_frac_to_cart,
    wrap_points_to_voronoi,
)
from vcoul.minibz import _minibz_kernel_bare

# The hexagonal MoS2-class 3D cell from tests/test_vcoul_minibz_head_draw.py
# — non-cubic on purpose: a cubic cell would make the anti-tautology arm
# vacuous (bvec.T = P.bvec there).
_A = 3.16 / 0.529177
_C = 12.3 / 0.529177
HEX_BVEC = 2.0 * np.pi * np.array([
    [1.0 / _A, 1.0 / (np.sqrt(3.0) * _A), 0.0],
    [0.0, 2.0 / (np.sqrt(3.0) * _A), 0.0],
    [0.0, 0.0, 1.0 / _C]])
HEX_KGRID = (4, 4, 2)
HEX_CELVOL = abs(8.0 * np.pi ** 3 / np.linalg.det(HEX_BVEC))


def test_frac_to_cart_is_the_row_convention_and_not_the_transpose():
    rng = np.random.RandomState(3)
    U = rng.uniform(0, 1, (256, 3))
    got = minibz_frac_to_cart(U, HEX_BVEC)
    # The convention, exactly.
    assert np.array_equal(got, U @ HEX_BVEC)
    # Anti-tautology: on this (non-cubic) cell the transposed spelling is a
    # DIFFERENT map, everywhere — if these ever agree, the cell went
    # symmetric and this file's discrimination power silently died.
    assert np.all(np.abs(got - U @ HEX_BVEC.T).max(axis=1) > 1e-3)


def test_cell_affine_matches_its_defining_expression():
    kg = np.asarray(HEX_KGRID, dtype=np.float64)
    expect = HEX_BVEC.T @ (np.diag(1.0 / kg) @ np.linalg.inv(HEX_BVEC.T))
    assert np.array_equal(minibz_cell_affine(HEX_BVEC, HEX_KGRID), expect)


# Golden values captured at b21b8679 (pre-consolidation production code),
# hexagonal cell, nmc=2048, seed=42 — full float64 precision.  The
# consolidation's claim is BIT-IDENTITY of the body head; the refreeze
# candidate generated at 358bb0b stays valid because of exactly this.
_GOLDEN = {
    (0, 0, 1): 1.3274212932013936,
    (1, 2, 0): 0.05511143967277558,
    (2, 0, 1): 0.093430231081166,
    (3, 3, 1): 0.12303079417392924,
}


def test_body_head_bit_stable_across_the_consolidation():
    h = build_v_head_miniBZ_avg_3d(
        HEX_KGRID, HEX_BVEC, HEX_CELVOL, nmc=2048, seed=42)
    for ix, val in _GOLDEN.items():
        assert float(h[ix]) == val, (
            f"body head moved at {ix}: {float(h[ix])!r} != {val!r} — the "
            f"consolidation (or a later edit) changed production bits; the "
            f"358bb0b refreeze candidate would no longer be valid.")


def test_body_head_is_the_composition_of_the_shared_pieces():
    """Rebuild the head table from the exported pieces only; must be
    bit-equal to production.  This is the pin that keeps the body head
    from quietly re-forking its own copy of the geometry."""
    import jax.numpy as jnp
    nmc, seed = 1024, 42
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (nmc, 3))
    randcart = minibz_frac_to_cart(randvals, HEX_BVEC)
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), jnp.asarray(HEX_BVEC, dtype=jnp.float64),
        nmax=1))
    dq = (minibz_cell_affine(HEX_BVEC, HEX_KGRID) @ wrapped.T).T
    kg = np.asarray(HEX_KGRID, dtype=np.float64)
    expect = np.zeros(HEX_KGRID, dtype=np.float64)
    for qx in range(HEX_KGRID[0]):
        for qy in range(HEX_KGRID[1]):
            for qz in range(HEX_KGRID[2]):
                qw = np.array([qx, qy, qz], dtype=np.float64)
                qw = np.where(qw > kg / 2, qw - kg, qw)
                q_cart = (qw / kg) @ HEX_BVEC
                if np.dot(q_cart, q_cart) < 1e-12:
                    continue
                v, _ = _minibz_kernel_bare(q_cart, dq, kind="bulk_3d")
                expect[qx, qy, qz] = np.mean(v)
    expect *= 1.0 / HEX_CELVOL
    got = build_v_head_miniBZ_avg_3d(
        HEX_KGRID, HEX_BVEC, HEX_CELVOL, nmc=nmc, seed=seed)
    assert np.array_equal(got, expect)


def test_composition_pin_can_fail():
    """Red twin for the composition pin: a transposed draw fed through the
    same composition must NOT reproduce production."""
    import jax.numpy as jnp
    nmc, seed = 1024, 42
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (nmc, 3))
    randcart = randvals @ HEX_BVEC.T          # the dead bug, on purpose
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), jnp.asarray(HEX_BVEC, dtype=jnp.float64),
        nmax=1))
    dq = (minibz_cell_affine(HEX_BVEC, HEX_KGRID) @ wrapped.T).T
    q_cart = (np.array([0.0, 0.0, 1.0]) / np.asarray(HEX_KGRID)) @ HEX_BVEC
    v, _ = _minibz_kernel_bare(q_cart, dq, kind="bulk_3d")
    wrong = float(np.mean(v)) / HEX_CELVOL
    got = float(build_v_head_miniBZ_avg_3d(
        HEX_KGRID, HEX_BVEC, HEX_CELVOL, nmc=nmc, seed=seed)[0, 0, 1])
    assert abs(wrong - got) / got > 1e-3
