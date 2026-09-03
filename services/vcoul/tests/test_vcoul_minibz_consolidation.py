"""Pins for the mini-BZ geometry consolidation and the body head's draw.

The 3D body head and the q→0 head were independent implementations of
draw → wrap → affine → kernel that diverged nine ways; the transposed-draw
bug (fixed 358bb0b) lived in exactly the axis they duplicated.  The
2026-08-07 consolidation single-sourced the GEOMETRY
(``minibz_frac_to_cart``, ``minibz_cell_affine``, the shared bare kernel)
and deliberately kept the estimator/generator/nmax/volume divergences
(registered owner questions).

The 2026-08-08 head-slot fix then changed the body head's SHAPE — it is a
function of the Cartesian ``K = q+G`` now, not a per-q table — and closed
its δq sample set under negation.  Both were forced by the injection rule
(``v_qG_table``'s HEAD SLOT note): the head is injected at every slot
attaining ``argmin |q+G|²``, each valued from its own K, so a per-q scalar
cannot carry the answer; and reciprocity ``V_q = conj(V_{−q})`` needs
``f(K) = f(−K)``, which a finite MC sample delivers exactly only when
``{δq} = {−δq}``.

These cells pin four things: the primitive owns the row convention (with
the anti-tautology twin showing the transposed spelling actually differs
on the cell used), the draw is centrosymmetric and the head it feeds is
therefore an EXACTLY even function of K, the production head is
bit-stable at its captured values, and the head IS the composition of the
shared pieces (so the wiring cannot silently fork again).
"""
import numpy as np
import pytest

from vcoul import (
    COULOMB_GAUGE_TT_SIGN,
    build_miniBZ_dq_cart,
    build_v_head_miniBZ_fn_3d,
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
SLAB_KGRID = (4, 4, 1)
ANISOTROPIC_SLAB_KGRID = (3, 5, 1)
HEX_CELVOL = abs(8.0 * np.pi ** 3 / np.linalg.det(HEX_BVEC))


def test_vcoul_owns_the_immutable_finite_interval_rule():
    import vcoul

    nodes, weights = vcoul.gauss_legendre_interval(4, 0.0, 1.0)
    for degree in range(8):
        np.testing.assert_allclose(
            np.sum(weights * nodes**degree), 1.0 / (degree + 1),
            rtol=2.0e-14, atol=2.0e-14)
    assert not nodes.flags.writeable and not weights.flags.writeable
    assert "leggauss" in vcoul.GAUSS_LEGENDRE_INTERVAL_PROVENANCE


def _q_cart(ix, kgrid=HEX_KGRID, bvec=HEX_BVEC):
    """BGW-wrapped grid index → Cartesian q, the convention the retired
    per-q table was indexed in.  Kept so the golden rows below name the
    same points the pre-2026-08-08 goldens did."""
    kg = np.asarray(kgrid, dtype=np.float64)
    qw = np.asarray(ix, dtype=np.float64)
    qw = np.where(qw > kg / 2, qw - kg, qw)
    return (qw / kg) @ bvec


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


# ---------------------------------------------------------------------------
# The centrosymmetry — the property the whole q → −q reciprocity rests on.
# ---------------------------------------------------------------------------

def test_the_minibz_draw_is_closed_under_negation():
    dq = build_miniBZ_dq_cart(HEX_KGRID, HEX_BVEC, nmc=512, seed=42)
    assert dq.shape == (1024, 3)
    # {δq} = {−δq} as a SET, exactly (not approximately, not in the limit).
    assert np.array_equal(np.sort(dq, axis=0), np.sort(-dq, axis=0))
    # And it is a real draw, not a degenerate one that would make the
    # closure trivially true.
    assert np.abs(dq).max() > 0.0
    assert len(np.unique(dq[:, 0])) > 512 // 2


def test_streamed_photon_samples_apply_the_spatial_metric_once():
    """The packed q0 provider returns physical ``D_TT=-v P_T``.

    This is the independent mini-BZ route that does not read the G-flat
    bispinor V file.  Its CC trace fixes ``v``, while the TT trace and null
    direction distinguish the metric sign from a positive projector sample.
    """
    import vcoul

    assert COULOMB_GAUGE_TT_SIGN == -1.0
    item = next(vcoul.iter_minibz_photon_samples(
        vcoul.get_kernel(2),
        vcoul.CoulombGeometry(
            bvec=HEX_BVEC, cell_volume=HEX_CELVOL),
        HEX_KGRID,
        nsamples=8, qmc_reps=1, chunk_size=8))
    _, _, _, q_cart, D_raw, valid_count, _, _ = item
    q = np.asarray(q_cart[:valid_count])
    D = np.asarray(D_raw[:valid_count])
    v = D[:, 0, 0]
    assert np.all(v > 0.0)
    np.testing.assert_allclose(
        np.trace(D[:, 1:, 1:], axis1=1, axis2=2), -2.0 * v,
        rtol=2e-15, atol=1e-14)
    np.testing.assert_allclose(
        np.einsum("sij,sj->si", D[:, 1:, 1:], q), 0.0,
        rtol=0.0, atol=2e-13)


def test_polygon_photon_receipt_binds_fixed_ladder_weights_and_counts():
    import vcoul

    receipt = vcoul.slab_minibz_photon_cubature(
        vcoul.get_kernel(2),
        vcoul.CoulombGeometry(bvec=HEX_BVEC, cell_volume=HEX_CELVOL),
        SLAB_KGRID)
    assert receipt.method == "true_ws_polygon_duffy_gauss_legendre_v1"
    assert receipt.orders == (16, 24, 32)
    assert receipt.kgrid == SLAB_KGRID
    assert receipt.cell_volume == HEX_CELVOL
    np.testing.assert_array_equal(
        np.asarray(receipt.reciprocal_lattice_rows), HEX_BVEC)
    assert receipt.dimension == 2
    assert vcoul.validate_minibz_photon_receipt(receipt) is receipt
    polygon = np.asarray(receipt.polytope_vertices)
    assert polygon.shape == (6, 2)
    mini = np.asarray(receipt.mini_lattice_rows)
    np.testing.assert_allclose(
        receipt.minibz_measure, abs(np.linalg.det(mini)), rtol=2e-12)
    assert receipt.physical_counts == tuple(
        6 * order * order for order in receipt.orders)
    assert receipt.padded_counts == (6 * 32 * 32,) * 3
    assert len(receipt.chunks) == 3
    measures = []
    for chunk in receipt.chunks:
        q, D = chunk.q_cart, chunk.D_raw
        valid, weight = chunk.physical_count, chunk.sample_weight
        assert q.shape == (6 * 32 * 32, 3)
        assert D.shape == (6 * 32 * 32, 4, 4)
        assert np.all(q[:valid, 2] == 0.0)
        assert np.all(weight[:valid] > 0.0)
        assert np.all(weight[valid:] == 0.0)
        assert np.all(D[valid:] == 0.0)
        measures.append(float(np.sum(weight[:valid])))
        np.testing.assert_allclose(
            np.sum(weight[:valid, None] * q[:valid], axis=0), 0.0,
            rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(measures, 1.0, rtol=5e-15)

    anisotropic = vcoul.slab_minibz_photon_cubature(
        vcoul.get_kernel(2),
        vcoul.CoulombGeometry(bvec=HEX_BVEC, cell_volume=HEX_CELVOL),
        ANISOTROPIC_SLAB_KGRID)
    np.testing.assert_allclose(
        anisotropic.minibz_measure,
        abs(np.linalg.det(np.asarray(anisotropic.mini_lattice_rows))),
        rtol=2e-12)

    with pytest.raises(ValueError, match="unsupported"):
        next(vcoul.iter_minibz_photon_samples(
            vcoul.get_kernel(2),
            vcoul.CoulombGeometry(
                bvec=HEX_BVEC, cell_volume=HEX_CELVOL),
            SLAB_KGRID, method="polygon_gl"))


@pytest.mark.parametrize(
    ("bvec", "kgrid", "expect_isotropic"),
    ((np.eye(3), (4, 4, 4), True), (HEX_BVEC, HEX_KGRID, False)),
)
def test_bulk_photon_receipt_is_true_ws_and_obeys_trace_identity(
        bvec, kgrid, expect_isotropic):
    import vcoul

    receipt = vcoul.minibz_photon_cubature(
        vcoul.get_kernel(3),
        vcoul.CoulombGeometry(bvec=bvec, cell_volume=HEX_CELVOL),
        kgrid)
    assert receipt.dimension == 3
    assert receipt.method == "true_ws_polyhedron_duffy_gauss_legendre_v1"
    assert receipt.orders == (10, 16, 22)
    assert receipt.slab_zc is None
    assert vcoul.validate_minibz_photon_receipt(receipt) is receipt
    mini = np.asarray(receipt.mini_lattice_rows)
    np.testing.assert_allclose(
        receipt.minibz_measure, abs(np.linalg.det(mini)), rtol=5e-12)
    assert all(len(face) >= 3 for face in receipt.polytope_faces)

    chunk = receipt.chunks[-1]
    physical = int(chunk.physical_count)
    weight = chunk.sample_weight[:physical]
    D = chunk.D_raw[:physical]
    assert np.all(np.linalg.norm(chunk.q_cart[:physical], axis=1) > 0.0)
    np.testing.assert_allclose(np.sum(weight), 1.0, rtol=8e-15)
    np.testing.assert_allclose(
        np.sum(weight[:, None] * chunk.q_cart[:physical], axis=0), 0.0,
        rtol=0.0, atol=5e-15)
    # Accumulate the pointwise identity so independent large reductions do
    # not inject cancellation noise into this algebra oracle.
    trace_identity = np.sum(weight * (
        np.trace(D[:, 1:, 1:], axis1=1, axis2=2) + 2.0 * D[:, 0, 0]))
    assert abs(trace_identity) / abs(np.sum(weight * D[:, 0, 0])) <= 1.0e-14

    D_mean = np.einsum("s,sij->ij", weight, D)
    if expect_isotropic:
        np.testing.assert_allclose(
            D_mean[1:, 1:], -(2.0 / 3.0) * D_mean[0, 0] * np.eye(3),
            rtol=1.0e-12, atol=1.0e-12)
    else:
        assert not np.isclose(
            D_mean[1, 1], D_mean[3, 3], rtol=1.0e-4, atol=0.0)


def test_photon_cubature_dimension_dispatch_refuses_mismatched_kernels():
    import vcoul

    geometry = vcoul.CoulombGeometry(
        bvec=HEX_BVEC, cell_volume=HEX_CELVOL)
    with pytest.raises(TypeError, match="Bulk3D"):
        vcoul.bulk_minibz_photon_cubature(
            vcoul.get_kernel(2), geometry, HEX_KGRID)
    with pytest.raises(TypeError, match="Slab2D"):
        vcoul.slab_minibz_photon_cubature(
            vcoul.get_kernel(3), geometry, SLAB_KGRID)


def test_polygon_photon_receipt_uses_one_area_reduction_on_cri3_geometry():
    """The real CrI3 cell distinguishes Python and NumPy sum rounding."""
    import vcoul

    bvec = np.asarray([
        [0.4820307462398321, -0.2783005810992434, 0.0],
        [0.0, 0.5566011621984867, 0.0],
        [0.0, 0.0, 0.14601087526378126],
    ])
    receipt = vcoul.slab_minibz_photon_cubature(
        vcoul.get_kernel(2),
        vcoul.CoulombGeometry(
            bvec=bvec, cell_volume=6331.9219276452886),
        (6, 6, 1))
    assert vcoul.validate_minibz_photon_receipt(receipt) is receipt

    polygon = np.asarray(receipt.polytope_vertices)
    crosses = [
        float(left[0] * right[1] - left[1] * right[0])
        for left, right in zip(polygon, np.roll(polygon, -1, axis=0))
    ]
    issued_order = 0.5 * float(sum(crosses))
    old_validator_order = 0.5 * float(np.sum(np.asarray(crosses)))
    assert issued_order != old_validator_order
    assert receipt.minibz_measure == issued_order


def _slab_photon_receipt():
    import vcoul

    return vcoul.slab_minibz_photon_cubature(
        vcoul.get_kernel(2),
        vcoul.CoulombGeometry(bvec=HEX_BVEC, cell_volume=HEX_CELVOL),
        SLAB_KGRID)


def test_polygon_photon_receipt_validation_is_exact_type_and_rechecks_token():
    from dataclasses import replace
    import vcoul

    receipt = _slab_photon_receipt()
    with pytest.raises(TypeError, match="issued only"):
        replace(receipt, orders=(8, 16, 24))

    class ForgedReceipt(vcoul.MinibzPhotonReceipt):
        def require_integrity(self):  # A virtual check would accept this.
            return self

    forged = object.__new__(ForgedReceipt)
    for name, value in vars(receipt).items():
        object.__setattr__(forged, name, value)
    with pytest.raises(TypeError, match="exact provider receipt type"):
        vcoul.validate_minibz_photon_receipt(forged)

    object.__setattr__(receipt, "_provider_token", object())
    with pytest.raises(TypeError, match="not issued"):
        vcoul.validate_minibz_photon_receipt(receipt)


@pytest.mark.parametrize("payload_name", ("q_cart", "D_raw", "sample_weight"))
def test_polygon_photon_receipt_detects_mutation_despite_reversible_write_flag(
        payload_name):
    import vcoul

    receipt = _slab_photon_receipt()
    payload = getattr(receipt.chunks[0], payload_name)
    assert not payload.flags.writeable
    # ``write=False`` is intentionally not the integrity argument: these
    # provider-owned ndarrays can be made writeable again by their holder.
    payload.setflags(write=True)
    payload.flat[0] = np.nextafter(payload.flat[0], np.inf)
    with pytest.raises(ValueError, match="payload or metadata changed"):
        vcoul.validate_minibz_photon_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    (("cell_volume", HEX_CELVOL * 1.01), ("kgrid", (5, 4, 1))),
)
def test_polygon_photon_receipt_detects_bound_geometry_mutation(field, value):
    import vcoul

    receipt = _slab_photon_receipt()
    object.__setattr__(receipt, field, value)
    with pytest.raises(ValueError):
        vcoul.validate_minibz_photon_receipt(receipt)


def test_slab_polygon_refuses_nonplanar_or_three_dimensional_cells():
    import vcoul

    with pytest.raises(ValueError, match="Nkz=1"):
        vcoul.slab_minibz_photon_cubature(
            vcoul.get_kernel(2),
            vcoul.CoulombGeometry(
                bvec=HEX_BVEC, cell_volume=HEX_CELVOL), HEX_KGRID)
    tilted = HEX_BVEC.copy()
    tilted[0, 2] = 1.0e-3
    with pytest.raises(ValueError, match="Cartesian xy plane"):
        vcoul.slab_minibz_photon_cubature(
            vcoul.get_kernel(2),
            vcoul.CoulombGeometry(bvec=tilted, cell_volume=HEX_CELVOL),
            SLAB_KGRID)
    tilted_b3 = HEX_BVEC.copy()
    tilted_b3[2, 0] = 1.0e-3
    with pytest.raises(ValueError, match="nonzero b3 directed"):
        vcoul.slab_minibz_photon_cubature(
            vcoul.get_kernel(2),
            vcoul.CoulombGeometry(bvec=tilted_b3, cell_volume=HEX_CELVOL),
            SLAB_KGRID)
    zero_b3 = HEX_BVEC.copy()
    zero_b3[2] = 0.0
    with pytest.raises(ValueError, match=r"b3 directed along Cartesian \+z"):
        vcoul.slab_minibz_photon_cubature(
            vcoul.get_kernel(2),
            vcoul.CoulombGeometry(bvec=zero_b3, cell_volume=HEX_CELVOL),
            SLAB_KGRID)
    reversed_b3 = HEX_BVEC.copy()
    reversed_b3[2, 2] *= -1.0
    with pytest.raises(ValueError, match="reversed or tilted"):
        vcoul.slab_minibz_photon_cubature(
            vcoul.get_kernel(2),
            vcoul.CoulombGeometry(
                bvec=reversed_b3, cell_volume=HEX_CELVOL),
            SLAB_KGRID)


def test_the_head_is_an_exactly_even_function_of_K():
    """``f(K) = f(−K)`` to the BIT, which is what the argmin injection
    needs: the tied set at −q is the negation of the tied set at +q, so
    anything less than exact equality reappears as MC-noise-sized
    reciprocity error in V_q."""
    fn = build_v_head_miniBZ_fn_3d(HEX_KGRID, HEX_BVEC, HEX_CELVOL,
                                   nmc=1024, seed=42)
    K = np.array([_q_cart(ix) for ix in ((0, 0, 1), (1, 2, 0), (2, 0, 1),
                                         (3, 3, 1), (1, 1, 1))])
    np.testing.assert_array_equal(fn(K), fn(-K))


def test_the_evenness_pin_can_fail():
    """RED TWIN: a one-sided draw — the pre-2026-08-08 sample set — fed
    through the same average is NOT even in K, so the cell above is
    measuring something and not asserting an identity of arithmetic."""
    dq = build_miniBZ_dq_cart(HEX_KGRID, HEX_BVEC, nmc=1024, seed=42)
    one_sided = dq[:1024]                     # the historical half-draw
    K = _q_cart((1, 2, 0))
    v_pos, _ = _minibz_kernel_bare(K, one_sided, kind="bulk_3d")
    v_neg, _ = _minibz_kernel_bare(-K, one_sided, kind="bulk_3d")
    rel = abs(np.mean(v_pos) - np.mean(v_neg)) / abs(np.mean(v_pos))
    assert rel > 1e-6, (
        f"the one-sided draw came out even in K anyway (rel {rel:.3e}) — "
        f"the centrosymmetrisation would then be untestable here, so find "
        f"out why before trusting the cell above")


# ---------------------------------------------------------------------------
# Golden bits.
# ---------------------------------------------------------------------------

#: Captured at the 2026-08-08 head-slot fix, hexagonal cell, nmc=2048,
#: seed=42 — full float64 precision.  These MOVED relative to the
#: pre-2026-08-08 goldens (which pinned the retired per-q table at
#: b21b8679: 1.3274212932013936 / 0.05511143967277558 /
#: 0.093430231081166 / 0.12303079417392924) and the move is the
#: centrosymmetrisation of the δq set, ~3e-3 relative at this sample
#: count — MC-noise scale, as it must be, since averaging v over {δq}
#: and over {−δq} estimates the same integral.
_GOLDEN = {
    (0, 0, 1): 1.3320892793844843,
    (1, 2, 0): 0.055226806084805596,
    (2, 0, 1): 0.09295617168034515,
    (3, 3, 1): 0.12304679980119634,
}


def test_body_head_bits_are_pinned():
    fn = build_v_head_miniBZ_fn_3d(HEX_KGRID, HEX_BVEC, HEX_CELVOL,
                                   nmc=2048, seed=42)
    for ix, val in _GOLDEN.items():
        got = float(fn(_q_cart(ix).reshape(1, 3))[0])
        assert got == val, (
            f"body head moved at {ix}: {got!r} != {val!r} — production "
            f"bits changed; if that was deliberate, the si_bse_debug "
            f"frozen reference moves with them and needs its own commit.")


def test_body_head_is_the_composition_of_the_shared_pieces():
    """Rebuild the head from the exported pieces only; must be bit-equal
    to production.  This is the pin that keeps the body head from quietly
    re-forking its own copy of the geometry."""
    import jax.numpy as jnp
    nmc, seed = 1024, 42
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (nmc, 3))
    randcart = minibz_frac_to_cart(randvals, HEX_BVEC)
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), jnp.asarray(HEX_BVEC, dtype=jnp.float64),
        nmax=1))
    half = (minibz_cell_affine(HEX_BVEC, HEX_KGRID) @ wrapped.T).T
    dq = np.concatenate([half, -half], axis=0)
    assert np.array_equal(
        dq, build_miniBZ_dq_cart(HEX_KGRID, HEX_BVEC, nmc=nmc, seed=seed))

    fn = build_v_head_miniBZ_fn_3d(HEX_KGRID, HEX_BVEC, HEX_CELVOL,
                                   nmc=nmc, seed=seed)
    for ix in _GOLDEN:
        K = _q_cart(ix)
        v, _ = _minibz_kernel_bare(K, dq, kind="bulk_3d")
        # Production's arithmetic ORDER, not just its value: the mean is
        # taken bare and the volume factor applied as a multiply by
        # 1/celvol afterwards.  ``mean(v) / celvol`` differs in the last
        # ulp and this cell claims bit-equality.
        expect = np.mean(v) * (1.0 / HEX_CELVOL)
        assert float(fn(K.reshape(1, 3))[0]) == expect


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
    half = (minibz_cell_affine(HEX_BVEC, HEX_KGRID) @ wrapped.T).T
    dq = np.concatenate([half, -half], axis=0)
    K = _q_cart((0, 0, 1))
    v, _ = _minibz_kernel_bare(K, dq, kind="bulk_3d")
    wrong = float(np.mean(v)) / HEX_CELVOL
    fn = build_v_head_miniBZ_fn_3d(HEX_KGRID, HEX_BVEC, HEX_CELVOL,
                                   nmc=nmc, seed=seed)
    got = float(fn(K.reshape(1, 3))[0])
    assert abs(wrong - got) / got > 1e-3


def test_the_retired_per_q_table_builder_is_gone():
    """The ``(nkx, nky, nkz)`` table cannot express the argmin rule — a
    per-q scalar belongs to ``q_frac @ bvec``, which is not the argmin
    ``q+G`` on 12 of Si's 64 q.  It was removed rather than left on the
    door for someone to reach for; ``v_qG_table`` refuses one by name if
    an out-of-tree caller still builds it."""
    import vcoul
    assert not hasattr(vcoul, "build_v_head_miniBZ_avg_3d")
    with pytest.raises(ImportError):
        from vcoul.minibz import build_v_head_miniBZ_avg_3d  # noqa: F401
