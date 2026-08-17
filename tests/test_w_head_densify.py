"""Gates for C1 — head-excluded W densification (``gw.head_densify``).

WHAT IS BEING GATED.  The coarse→fine W densifier trigonometrically
interpolates ``W_q``.  Before C1 its operand carried the Γ head as a Kronecker
delta with a ~10³ meV geometric prefactor, and a band-limited interpolant
cannot represent a delta: it produces a Dirichlet kernel that rings in sign
across the whole fine zone and supplies none of the 1/q² rise the fine q inside
the coarse Γ cell should carry.  C1 splits the head off before the densifier
and re-attaches it analytically per fine q.

The cells below are ordered by what they establish, and every one of them has
either an EXACT statement or a red twin:

  1. **On-grid identity** — with fine == coarse the re-attached head array is
     ``[whead at Γ, 0 elsewhere]`` BITWISE, and the loader's bundle is
     byte-identical to the no-densification path.  Free, and it is what proves
     split-and-reattach is algebraically the identity.
  2. **The partition** — the number of fine q in the coarse Γ cell is
     ``[Λ_f : Λ_c] = ∏(nf/nc)`` exactly, on every lattice, with no tolerance.
     Red twin: a geometric predicate with a tie rule, which over-counts.
  3. **The head sum rule** — the head channel's zone average is exact at
     ``m = 1``, converges under refinement, and the design's red twin
     (``gamma_cell='coarse'``: re-attach at the coarse mini-BZ scale) is
     invisible at ``m = 1`` and fails everywhere else.
  4. **The delta rings** — measured, against the SHIPPED densifier: the
     interpolated head changes sign and leaks out of the Γ cell; C1's does
     neither.  This is the defect, reproduced rather than described.
  5. **Hermiticity** — machine zero by construction, with the deliberately
     complex scalar as the red twin.
  6. **Refusals** — bulk-3D only, no pointwise value at q = 0, no complex head.

Fixture-free and CPU-only: every number here comes from synthetic cells and
the real production functions, so this runs on any box.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                        # noqa: E402

from gw.head_densify import (                                  # noqa: E402
    EIGHT_PI, attach_head_channel, build_fine_head_scalars,
    coarse_gamma_cell_weights, fine_q_cart, gamma_cell_head_scalar,
    head_channel_zone_average, head_scalar_pointwise)
from vcoul import CoulombGeometry                              # noqa: E402

pytestmark = pytest.mark.census


# ---------------------------------------------------------------------------
# Cells.  Three lattices, because every partition bug found so far was a
# SKEW-cell bug: simple cubic hides them (its fftfreq box IS its Voronoi cell).
# ---------------------------------------------------------------------------
def _geom(kind):
    if kind == "fcc":                      # Si, a = 10.26 bohr
        a = 10.26
        avec = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    elif kind == "sc":
        avec = 10.26 * np.eye(3)
    elif kind == "hex":                    # 60° in-plane, long c
        avec = np.array([[3.0, 0., 0.], [-1.5, 2.598076211, 0.], [0., 0., 12.]])
    else:                                  # deliberately ugly triclinic
        avec = np.array([[5.1, 0.3, -0.2], [1.7, 6.4, 0.5], [-0.9, 1.1, 7.3]])
    bvec = 2.0 * np.pi * np.linalg.inv(avec).T
    return CoulombGeometry(bvec=bvec, cell_volume=abs(np.linalg.det(avec)))


#: An isotropic S with ε∞ = 12 (Si-like): ``1 − 8π q̂ᵀSq̂ = 1/ε∞``.
_S_ISO = ((1.0 - 1.0 / 12.0) / EIGHT_PI) * np.eye(3)
#: An anisotropic one, so no cell can pass by accident on isotropy.
_S_ANISO = np.diag([(1 - 1 / 14.) / EIGHT_PI, (1 - 1 / 11.) / EIGHT_PI,
                    (1 - 1 / 8.) / EIGHT_PI])
_WHEAD = 150.3956          # the Si anchor deck's pinned whead_0freq, Ry·bohr³


def _build(geom, cg, fg, S=_S_ISO, *, gamma_cell="fine", whead=_WHEAD,
           ref_grid=None):
    ref_grid = tuple(ref_grid or cg)
    gamma_ref = gamma_cell_head_scalar(geom, ref_grid, S)
    # ``sys_dim=3`` EXPLICITLY.  Every cell in this file is about the bulk
    # pole, and since 2026-08-17 omitting the argument is a refusal rather
    # than a silent bulk-3D assumption — see
    # ``test_an_unstamped_sys_dim_is_refused_not_defaulted_to_bulk``.
    return build_fine_head_scalars(
        geom, cg, fg, S, head_ref=whead, gamma_ref=gamma_ref,
        ref_grid=ref_grid, sys_dim=3, gamma_cell=gamma_cell), gamma_ref


# ===========================================================================
# 1.  ON-GRID IDENTITY
# ===========================================================================
@pytest.mark.parametrize("kind", ["fcc", "sc", "hex", "tri"])
@pytest.mark.parametrize("cg", [(4, 4, 4), (3, 3, 1), (2, 2, 2)])
def test_on_grid_the_reattached_head_is_bitwise_the_injected_delta(kind, cg):
    """fine == coarse → exactly ``[whead at Γ, 0 elsewhere]``.  BITWISE.

    This is the design's "free" gate.  It is bitwise and not merely close
    because the anchor is applied as ``whead · (S/gamma_ref)`` and at
    fine == coarse the Γ entry IS ``gamma_ref``, so the ratio is a float
    divided by itself.  The other spelling, ``(whead/gamma_ref)·S``, would
    make this a 1e-16 claim; that it is bitwise is what proves
    split-and-reattach is the algebraic identity rather than a good
    approximation to it.
    """
    geom = _geom(kind)
    S_fine, _ = _build(geom, cg, cg)
    want = np.zeros(cg, dtype=np.float64)
    want[0, 0, 0] = _WHEAD
    assert np.array_equal(S_fine, want), (
        f"on-grid head array is not the injected delta: "
        f"Γ={S_fine[0, 0, 0]!r} vs {_WHEAD!r}, "
        f"{int(np.count_nonzero(S_fine))} nonzeros")


@pytest.mark.parametrize("kind", ["fcc", "hex"])
def test_on_grid_the_red_twin_is_invisible(kind):
    """The design says the red twin is invisible when the grids are equal.

    It must be — otherwise the twin would be caught by the on-grid gate and
    would say nothing about the densified case, which is the case it exists
    to police.  Checked so that a twin which stops being invisible (i.e. one
    that started differing for some OTHER reason) is not mistaken for a
    working false arm.
    """
    geom = _geom(kind)
    cg = (4, 4, 2)
    true_arm, _ = _build(geom, cg, cg, gamma_cell="fine")
    twin, _ = _build(geom, cg, cg, gamma_cell="coarse")
    assert np.array_equal(true_arm, twin)


def test_the_loader_does_not_defer_when_the_grids_are_equal(tmp_path):
    """``bse_k_grid == coarse`` returns the bundle the no-flag path returns.

    THE SEAM THIS GUARDS.  C1 works by having the loader DEFER the rank-1
    whead injection when a densification is pending, so the densifier's
    operand is the body.  That deferral is decided before the injection, from
    a grid comparison — and if the comparison were wrong in the "equal" case,
    every on-grid run would silently lose W's head.  So the equal case is
    exercised through the real loader, not reasoned about.
    """
    from bse.bse_io import load_bse_data_from_restart_sharded
    from bse.bse_w_exact import _create_mesh_xy

    restart = str(tmp_path / "isdf_tensors_test.h5")
    _write_synthetic_restart(restart)
    mesh_xy = _create_mesh_xy(1, 1)
    kw = dict(n_val=1, n_cond=1, mesh_xy=mesh_xy, inject_head=True,
              cell_volume=270.0, n_occ=2)
    d0 = load_bse_data_from_restart_sharded(restart, bse_k_grid=None, **kw)
    cg = (int(d0["nkx"]), int(d0["nky"]), int(d0["nkz"]))
    d1 = load_bse_data_from_restart_sharded(restart, bse_k_grid=cg, **kw)
    for key in ("W_q", "V_q0", "psi_c_X", "psi_v_X", "eps_c", "eps_v", "M_X"):
        a = np.asarray(jax.device_get(d0[key]))
        b = np.asarray(jax.device_get(d1[key]))
        assert np.array_equal(a, b), f"{key} moved on the equal-grid path"
    # …and the head really is on the W tile (i.e. the test above is not
    # comparing two head-LESS bundles and calling them identical).
    W = np.asarray(jax.device_get(d0["W_q"]))
    g0 = np.asarray(jax.device_get(d0["g0_X"]))
    rank1 = np.conj(g0)[:, None] * g0[None, :] * (_SYN_WHEAD / 270.0)
    assert np.max(np.abs(W[:, :, 0, 0, 0] - (_SYN_W0[0] + rank1))) < 1e-12


def test_defer_whead_is_not_the_w0_ready_gate():
    """The two skips are different, and say different things in the log.

    ``w0_ready=False`` means "this W is bare V, a screened head does not
    belong on it".  ``defer_whead=True`` means "the head belongs, but the
    densifier will attach it per fine q".  Spelling the second as the first
    would put a wrong reason in the log on a perfectly screened tile — the
    exact confusion ``tests/test_sharded_whead_gate.py``'s row exists to
    prevent — so the two are checked to be distinguishable from outside.
    """
    from bse.bse_io import _inject_q0_head

    n = 4
    rng = np.random.default_rng(0)
    g0 = jnp.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    V = jnp.asarray(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    W = jnp.asarray(rng.standard_normal((n, n, 1, 1, 1))
                    + 1j * rng.standard_normal((n, n, 1, 1, 1)))
    whead = jnp.asarray([1.25], dtype=jnp.complex128)

    _, W_plain, s_plain = _inject_q0_head(V, W, g0, g0, 3.5, whead, 270.0,
                                          w0_ready=True)
    _, W_defer, s_defer = _inject_q0_head(V, W, g0, g0, 3.5, whead, 270.0,
                                          w0_ready=True, defer_whead=True)
    _, W_bare, s_bare = _inject_q0_head(V, W, g0, g0, 3.5, whead, 270.0,
                                        w0_ready=False)
    # Both skips leave the tile alone…
    assert np.array_equal(np.asarray(W_defer), np.asarray(W))
    assert np.array_equal(np.asarray(W_bare), np.asarray(W))
    assert not np.array_equal(np.asarray(W_plain), np.asarray(W))
    # …and they are NOT the same event.
    assert "DEFERRED" in s_defer and "1.250000" in s_defer
    assert "whead=skipped" in s_bare and "DEFERRED" not in s_bare
    assert "vhead=3.500000" in s_bare, "vhead belongs on V either way"


def test_the_loader_resolves_the_fine_grid_before_it_injects():
    """AST gate: the deferral decision must precede the injection.

    Behavioural cells can only see the two orders agreeing on the cases they
    run.  This one reads the source: in
    ``load_bse_data_from_restart_sharded`` the ``_resolve_bse_k_grid`` call
    has to come BEFORE the ``_inject_q0_head`` call, because the whole of C1
    is that the injection knows whether a densification is coming.  Written
    fixture-free so it holds on a box with no deck at all.
    """
    import ast
    import inspect
    from bse import bse_io

    fn = ast.parse(inspect.getsource(bse_io.load_bse_data_from_restart_sharded))
    seen = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("_resolve_bse_k_grid", "_inject_q0_head"):
                seen.append((node.lineno, node.func.id))
    seen.sort()
    names = [n for _, n in seen]
    assert "_resolve_bse_k_grid" in names and "_inject_q0_head" in names
    assert names.index("_resolve_bse_k_grid") < names.index("_inject_q0_head"), (
        "the head is injected before the fine grid is known, so the C1 "
        "deferral cannot fire and the densifier will interpolate a delta")


# ===========================================================================
# 2.  THE PARTITION
# ===========================================================================
@pytest.mark.parametrize("kind", ["fcc", "sc", "hex", "tri"])
@pytest.mark.parametrize("cg,m", [((4, 4, 4), 2), ((4, 4, 4), 3), ((3, 3, 3), 2),
                                  ((3, 3, 3), 4), ((6, 6, 2), 2), ((2, 2, 2), 5),
                                  ((5, 5, 1), 3), ((4, 4, 4), 1)])
def test_the_coarse_gamma_cell_holds_exactly_the_index_many_fine_q(kind, cg, m):
    """``|Γ_c ∩ fine grid| = [Λ_f : Λ_c] = m³``, exactly, on every lattice.

    An exact integer identity with no tolerance and no lattice dependence,
    and it is the statement that C1's re-attachment domain is a genuine
    fundamental domain — no q counted twice, none dropped.  It is also the
    gate that caught the first implementation: deciding membership with a
    geometric ``|q| ≤ |q−K|`` predicate needs a tie rule, and for EVEN m the
    ties are generic (every face centre of the coarse cell is a fine q).  The
    naive rule kept 9 points where 8 were due on fcc and 18 where 8 were due
    on hex — an over-count that no norm-based check would have seen, because
    the extra points carry small heads.
    """
    geom = _geom(kind)
    fg = tuple(m * c for c in cg)
    q = fine_q_cart(geom.bvec, fg)
    w = coarse_gamma_cell_weights(q, cg, fg)
    assert np.isclose(w.sum(), m ** 3, rtol=1e-12), (
        f"{kind} {cg}→{fg}: Γ-cell weight {w.sum()!r}, want {m**3}")
    assert w[0, 0, 0] == 1.0, "Γ must carry full weight"
    assert np.all((w >= 0.0) & (w <= 1.0))


@pytest.mark.parametrize("kind", ["fcc", "hex"])
def test_the_kept_q_are_the_ones_nearest_gamma(kind):
    """Each coset's representative is its member of smallest |q|.

    "A fundamental domain" alone would be satisfied by any one-per-coset
    choice, including a perverse one that keeps the FARTHEST member — which
    would put the 1/q² head on the wrong q while still passing the count
    identity above.  So the count is not enough, and this pins the choice.
    """
    geom = _geom(kind)
    cg, m = (3, 3, 3), 3
    fg = tuple(m * c for c in cg)
    q = fine_q_cart(geom.bvec, fg)
    w = coarse_gamma_cell_weights(q, cg, fg)
    q2 = np.einsum("xyza,xyza->xyz", q, q)
    idx = np.indices(fg)
    coset = ((idx[0] % m) * m + (idx[1] % m)) * m + (idx[2] % m)
    for c in range(m ** 3):
        members = coset == c
        kept = (w > 0) & members
        assert np.isclose(w[members].sum(), 1.0), (
            f"coset {c} carries weight {w[members].sum()!r}, want 1.0")
        assert np.allclose(q2[kept], q2[members].min()), (
            f"coset {c}: weight sits at |q|²={q2[kept]} but the nearest "
            f"member is at {q2[members].min():.6e}")


def test_a_broken_partition_is_refused_not_absorbed():
    """RED TWIN: monkeypatch the mask to over-count → the builder REFUSES.

    The invariant is asserted inside ``build_fine_head_scalars``, not merely
    tested here, because an over-counted domain deposits extra heads at q the
    coarse tiles already screen and the resulting error is small per point.
    """
    import gw.head_densify as hd
    geom = _geom("fcc")
    cg, fg = (2, 2, 2), (4, 4, 4)
    good = hd.coarse_gamma_cell_weights

    def over_counting(q_cart, coarse_grid, fine_grid):
        w = good(q_cart, coarse_grid, fine_grid).copy()
        # Give one q that carries no weight a full share, exactly as the
        # tie-break bug did — an over-count of one in ∏(nf/nc) = 8.
        zero = np.argwhere(w == 0.0)[0]
        w[tuple(zero)] = 1.0
        return w

    gamma_ref = gamma_cell_head_scalar(geom, cg, _S_ISO)
    hd.coarse_gamma_cell_weights = over_counting
    try:
        with pytest.raises(AssertionError, match="fundamental domain"):
            hd.build_fine_head_scalars(
                geom, cg, fg, _S_ISO, head_ref=_WHEAD, gamma_ref=gamma_ref,
                ref_grid=cg, sys_dim=3)
    finally:
        hd.coarse_gamma_cell_weights = good


# ===========================================================================
# 3.  THE HEAD SUM RULE
# ===========================================================================
def test_the_head_channel_zone_average_is_the_grid_independent_number():
    """THE SUM RULE, with its exact arm, its refinement arm and its false arm.

    ``(1/N_q) Σ_q S(q)`` is a property of the material and the cell, not of
    the grid.  C1's Γ cell is integrated EXACTLY (it is a cell average) and
    the rest of the coarse Γ cell is a midpoint quadrature of the same
    integral, so:

      * at ``m = 1`` the zone average is the coarse one BITWISE;
      * for ``m > 1`` it converges back to it monotonically.

    The red twin — the design's "re-attach at the coarse mini-BZ scale
    instead of the fine one" — is invisible at ``m = 1`` and thereafter puts
    a head ~m² too small at fine Γ, so its zone average is badly short.

    A word on what this does and does not discriminate, since the naive
    reading is misleading.  The trigonometric interpolant ALSO conserves the
    zone average — ``ifft → zero-pad → fft`` is linear with a fixed R = 0
    component — but it conserves it by FREEZING the coarse answer and
    smearing it as a Dirichlet kernel, so refining the grid does not refine
    the quadrature.  The discriminating statement is therefore not "is the
    sum conserved" but "is the sum a REFINEMENT of the same integral", which
    is what the monotone arm measures and what cell 4 makes visible directly.
    """
    geom = _geom("fcc")
    cg = (4, 4, 4)
    n_c = cg[0] * cg[1] * cg[2]
    target = _WHEAD / n_c

    S1, _ = _build(geom, cg, cg)
    assert head_channel_zone_average(S1) == target, "the m=1 arm is not exact"

    err_true, err_twin = [], []
    for m in (2, 3, 4):
        fg = tuple(m * c for c in cg)
        St, _ = _build(geom, cg, fg, gamma_cell="fine")
        Sr, _ = _build(geom, cg, fg, gamma_cell="coarse")
        err_true.append(abs(head_channel_zone_average(St) - target) / target)
        err_twin.append(abs(head_channel_zone_average(Sr) - target) / target)

    assert err_true == sorted(err_true, reverse=True), (
        f"the C1 zone average does not converge monotonically: {err_true}")
    for m, (et, er) in zip((2, 3, 4), zip(err_true, err_twin)):
        assert er > 2.0 * et, (
            f"m={m}: the red twin's zone-average error {er:.3e} is not "
            f"clearly worse than C1's {et:.3e} — the false arm is not false")
    assert err_twin[-1] > err_true[0], (
        f"the red twin at m=4 ({err_twin[-1]:.3e}) is better than C1 at m=2 "
        f"({err_true[0]:.3e}); the twin is converging as fast as the truth")


def test_the_head_channel_is_even_in_q():
    """``S(q) = S(−q)`` exactly, which is what carries reciprocity through.

    ``W_q = conj(W_{−q})`` survives the re-attachment only because the
    coefficient depends on q through ``v(q)`` and ``qᵀSq``, both even.  Held
    to BITWISE equality — the two are the same float expression on mirrored
    inputs, and any drift would mean the q table is not centrosymmetric.
    """
    geom = _geom("hex")
    cg, fg = (3, 3, 2), (9, 9, 4)
    S_fine, _ = _build(geom, cg, fg, S=_S_ANISO)
    flipped = np.roll(np.flip(S_fine, axis=(0, 1, 2)), shift=(1, 1, 1),
                      axis=(0, 1, 2))
    assert np.array_equal(S_fine, flipped), "the head channel is not even in q"


# ===========================================================================
# 4.  THE DELTA RINGS — the defect, reproduced
# ===========================================================================
def test_the_interpolated_head_rings_and_leaks_where_c1_does_neither():
    """Run the SHIPPED densifier on a delta and on C1's channel; compare.

    Three exact statements about the interpolated head that C1's does not
    share, measured through ``bse.bse_io.pad_W_R_to_grid`` — the real thing,
    not a model of it:

      1. it CHANGES SIGN, though ``S(q) = v/(1 − 8π q̂ᵀSq̂)`` is strictly
         positive (``1 − 8π q̂ᵀSq̂ = 1/ε∞ > 0``), so a head channel with a
         negative value anywhere is not a screened Coulomb head;
      2. it LEAKS: most of its weight lands outside the coarse Γ cell, on q
         whose own coarse tiles already carry their heads;
      3. it is far too SMALL at fine Γ, where the physics says the head grows
         like the inverse square of the cell.
    """
    from bse.bse_io import pad_W_R_to_grid

    geom = _geom("fcc")
    cg, m = (4, 4, 4), 2
    fg = tuple(m * c for c in cg)

    # What today's densifier does to the head channel alone (it is linear, so
    # the head's fate is independent of the body it rides with).
    delta = np.zeros((1, 1, *cg), dtype=np.complex128)
    delta[0, 0, 0, 0, 0] = _WHEAD
    R = jnp.fft.ifftn(jnp.asarray(delta), axes=(-3, -2, -1), norm="ortho")
    interp = np.real(np.asarray(jnp.fft.fftn(
        pad_W_R_to_grid(R, fg), axes=(-3, -2, -1), norm="ortho")))[0, 0]

    S_c1, _ = _build(geom, cg, fg)
    inside = coarse_gamma_cell_weights(fine_q_cart(geom.bvec, fg), cg, fg) > 0

    # 1. sign
    assert interp.min() < -1e-8 * _WHEAD, (
        "the interpolated head did not go negative — the Dirichlet ringing "
        "this cell exists to demonstrate is absent, so the comparison below "
        "is not measuring the documented defect")
    assert S_c1.min() >= 0.0, "C1's head channel went negative"

    # 2. leakage
    leak = np.abs(interp[~inside]).sum() / np.abs(interp).sum()
    assert leak > 0.5, f"interpolated head leakage only {leak:.2f}"
    assert np.all(S_c1[~inside] == 0.0), "C1 deposited head outside the Γ cell"

    # 3. magnitude at fine Γ.  The cell average scales like 1/L², so halving
    # the cell should raise it by ~m²; the interpolant barely moves.
    grow_c1 = S_c1[0, 0, 0] / _WHEAD
    grow_interp = interp[0, 0, 0] / _WHEAD
    assert grow_c1 > 2.0, (
        f"C1's fine-Γ head grew by only ×{grow_c1:.2f} for a {m}× finer cell")
    assert grow_interp < 1.05, (
        f"the interpolated fine-Γ head grew by ×{grow_interp:.2f}; it is "
        f"supposed to be stuck at the coarse value")


# ===========================================================================
# 5.  HERMITICITY
# ===========================================================================
def test_the_reattached_kernel_is_hermitian_to_machine_zero():
    """Hermiticity by congruence, and the deliberately complex red twin.

    ``S(q)·conj(g₀)⊗g₀`` is Hermitian iff ``S(q)`` is REAL.  The true arm is
    real by construction (``head_scalar_pointwise`` returns float64), so the
    assembled tile's Hermiticity error is machine zero.  The twin makes the
    scalar complex by hand and must trip — by ``2|Im S|·|g₀|²``, which is the
    exact size of the asymmetry a complex coefficient introduces.
    """
    rng = np.random.default_rng(11)
    n_mu, fg = 6, (4, 4, 1)
    A = rng.standard_normal((n_mu, n_mu, *fg)) + 1j * rng.standard_normal((n_mu, n_mu, *fg))
    body = jnp.asarray(A + np.conj(np.swapaxes(A, 0, 1)))       # Hermitian
    g0 = jnp.asarray(rng.standard_normal(n_mu) + 1j * rng.standard_normal(n_mu))
    geom = _geom("fcc")
    S_fine, _ = _build(geom, (2, 2, 1), fg)

    out = np.asarray(attach_head_channel(body, g0, g0, S_fine, 270.0))
    herm = np.max(np.abs(out - np.conj(np.swapaxes(out, 0, 1))))
    scale = np.max(np.abs(out))
    assert herm <= 1e-13 * scale, f"Hermiticity error {herm:.3e} (scale {scale:.3e})"

    # RED TWIN: a deliberately complex scalar.
    g0g0 = np.conj(np.asarray(g0))[:, None] * np.asarray(g0)[None, :]
    s_cplx = (S_fine[0, 0, 0] / 270.0) * (1.0 + 0.25j)
    twin = np.asarray(body).copy()
    twin[:, :, 0, 0, 0] += s_cplx * g0g0
    twin_herm = np.max(np.abs(twin - np.conj(np.swapaxes(twin, 0, 1))))
    assert twin_herm > 1e-6 * np.max(np.abs(twin)), (
        "a complex head scalar did not break Hermiticity — the check above "
        "is not sensitive to the thing it claims to gate")
    expect = 2.0 * abs(np.imag(s_cplx)) * np.max(np.abs(g0g0))
    assert np.isclose(twin_herm, expect, rtol=1e-10)


def test_a_complex_head_scalar_is_refused_before_it_can_be_attached():
    """The twin above cannot be reached through the shipped path.

    ``head_scalar_pointwise`` refuses rather than casting, so a complex
    ``S_cart`` (a finite η, or a finite ω) cannot silently produce a
    non-Hermitian kernel.  Dropping the imaginary part instead would make
    the red twin above unreachable AND unnoticeable, which is worse than
    either alternative.
    """
    geom = _geom("fcc")
    q = fine_q_cart(geom.bvec, (4, 4, 4))[1, 0, 0]
    S_bad = _S_ISO * (1.0 + 0.3j)
    with pytest.raises(ValueError, match="Hermiticity"):
        head_scalar_pointwise(q[None, :], S_bad)


# ===========================================================================
# 6.  REFUSALS
# ===========================================================================
def test_the_slab_is_refused_by_name_not_run_with_the_bulk_pole():
    """sys_dim 2 → NotImplementedError naming the slab stage.

    In 2D the head is a ``|q|`` cusp, not a ``1/q²`` pole, and the estimator
    is ``slab_2d.q0_average``.  Running the 3D expression on a slab is a
    wrong number with no shape error, so the refusal is the feature.
    """
    geom = _geom("hex")
    cg, fg = (3, 3, 1), (6, 6, 1)
    gamma_ref = gamma_cell_head_scalar(geom, cg, _S_ISO)
    with pytest.raises(NotImplementedError, match="SLAB_2D"):
        build_fine_head_scalars(geom, cg, fg, _S_ISO, head_ref=_WHEAD,
                                gamma_ref=gamma_ref, ref_grid=cg, sys_dim=2)


def test_an_unstamped_sys_dim_is_refused_not_defaulted_to_bulk():
    """``sys_dim=None`` REFUSES.  This is the cell that keeps the escape shut.

    The refusal above could not fire on either shipping path until
    2026-08-17, and the reason was one line here: ``_refuse_non_bulk``
    returned on ``None``, and ``Meta`` has no ``sys_dim`` field for
    ``bse.bse_densify.build_w_head_channel``'s ``getattr(meta, "sys_dim",
    None)`` to find.  So a ``sys_dim = 2`` deck with ``bse_k_grid``
    densification and the default ``w_head_densify = c1`` re-attached
    ``8π/|q|²`` — with no exception, no warning, and a confident provenance
    ratio in the log — and the error GREW as the fine grid densified.

    Two arms, because the escape has two ends: the guard itself, and the
    caller that feeds it.
    """
    geom = _geom("fcc")
    cg, fg = (2, 2, 2), (4, 4, 4)
    gamma_ref = gamma_cell_head_scalar(geom, cg, _S_ISO)

    # (a) THE GUARD.  Omitting the keyword is an error, not a request for 3D.
    with pytest.raises(ValueError, match="sys_dim was not supplied"):
        build_fine_head_scalars(geom, cg, fg, _S_ISO, head_ref=_WHEAD,
                                gamma_ref=gamma_ref, ref_grid=cg,
                                sys_dim=None)

    # (b) THE CALLER.  A Meta with no stamp is refused BEFORE the S tensor is
    # resolved and before the 2.6M-sample Γ-cell integral, so the message is
    # about dimensionality rather than about a missing dipole.h5.  ``wfn`` /
    # ``sym`` are None on purpose: reaching them at all would mean the check
    # ran too late.
    import types
    from bse.bse_densify import build_w_head_channel
    with pytest.raises(ValueError, match="carries no ``sys_dim``"):
        build_w_head_channel(
            None, None, types.SimpleNamespace(), {},
            coarse_grid=cg, fine_grid=fg, whead=_WHEAD, ref_grid=cg,
            input_file=None, restart_file=None, log_fn=lambda *a, **k: None)


def test_the_bse_meta_is_stamped_with_the_decks_sys_dim():
    """RED TWIN for the OTHER end: ``htransform.initialize_wfns`` must stamp it.

    That function builds the only ``Meta`` on the bandstructure/BSE lane —
    the one ``bse.bse_densify``'s ``bse_k_grid`` leg, ``bse.exciton_bands``'
    ``--w-coarse-grid`` leg and ``bse.vq_interp.refit_prepare`` all hold — so
    if the stamp goes away the refusal above starts firing on WORKING 3D
    decks and someone will reach for a default.  Read as TEXT: importing
    ``bandstructure.htransform`` pulls the FFI gate in, and this cell has no
    business needing a built ``.so`` to read an assignment.
    """
    import ast
    import os as _os
    from bse import bse_densify

    src_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(bse_densify.__file__))),
        "bandstructure", "htransform.py")
    fn = next(n for n in ast.walk(ast.parse(open(src_path).read()))
              if isinstance(n, ast.FunctionDef) and n.name == "initialize_wfns")

    stamps = [n for n in ast.walk(fn)
              if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Attribute) and t.attr == "sys_dim"
                      for t in n.targets)]
    assert stamps, (
        "initialize_wfns no longer stamps ``sys_dim`` on the Meta it builds; "
        "gw.head_densify's bulk-3D refusal then sees None on every BSE "
        "densification and cannot decide anything")

    # and it comes from the DECK, not from a literal
    src = ast.unparse(stamps[0].value)
    assert "params" in src and "sys_dim" in src, (
        f"the sys_dim stamp is {src!r} — it must read the deck's key, not "
        f"invent a dimensionality")


def test_the_pointwise_head_refuses_q_equals_zero():
    """There is no pointwise value at Γ — the object there is an integral."""
    with pytest.raises(ValueError, match="cell average|CELL AVERAGE"):
        head_scalar_pointwise(np.zeros((1, 3)), _S_ISO)


def test_w_head_densify_takes_only_the_two_modes():
    from bse.bse_io import resolve_w_head_densify
    assert resolve_w_head_densify(None) == "c1"
    assert resolve_w_head_densify(None, {}) == "c1"
    assert resolve_w_head_densify(None, {"w_head_densify": "legacy"}) == "legacy"
    assert resolve_w_head_densify("c1", {"w_head_densify": "legacy"}) == "c1"
    with pytest.raises(ValueError, match="w_head_densify"):
        resolve_w_head_densify("interpolate")


def test_the_grids_must_nest():
    geom = _geom("fcc")
    gamma_ref = gamma_cell_head_scalar(geom, (3, 3, 3), _S_ISO)
    with pytest.raises(ValueError, match="positive multiple"):
        build_fine_head_scalars(geom, (3, 3, 3), (4, 4, 4), _S_ISO,
                                head_ref=_WHEAD, gamma_ref=gamma_ref,
                                ref_grid=(3, 3, 3), sys_dim=3)


# ---------------------------------------------------------------------------
# The synthetic restart for the loader cell.  Deliberately tiny; the 8-D
# legacy V/W layout so the loader resolves the k-grid from the shape and
# needs no WFN.
# ---------------------------------------------------------------------------
_SYN_MU, _SYN_K = 4, (2, 2, 1)
_SYN_WHEAD, _SYN_VHEAD = 1.25, 3.5
_SYN_W0 = None


def _write_synthetic_restart(path):
    global _SYN_W0
    import h5py
    rng = np.random.default_rng(20260810)

    def _c(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    nkx, nky, nkz = _SYN_K
    nk = nkx * nky * nkz
    V = _c((1, 1, 1, nkx, nky, nkz, _SYN_MU, _SYN_MU))
    W0 = _c((1, 1, 1, nkx, nky, nkz, _SYN_MU, _SYN_MU))
    psi = _c((nk, 4, 1, _SYN_MU))
    enk = np.array([[-2.0, -1.0, 1.0, 2.0], [-2.2, -1.2, 1.2, 2.2]])
    g0 = _c((_SYN_MU,))
    with h5py.File(path, "w") as f:
        f.create_dataset("V_qmunu", data=V).attrs["V_ready"] = True
        f.create_dataset("W0_qmunu", data=W0).attrs["W0_ready"] = True
        f.create_dataset("psi_full_y", data=psi)
        f.create_dataset("enk_full", data=enk)
        f.create_dataset("G0_mu_nu", data=g0)
        f.create_dataset("vhead", data=np.complex128(_SYN_VHEAD))
        f.create_dataset("whead", data=np.array([_SYN_WHEAD], dtype=np.complex128))
        f.create_dataset("kgrid", data=np.array(_SYN_K))
    # The q=0 W tile as written, for the "the head really is there" check.
    _SYN_W0 = (W0[0, 0, 0, 0, 0, 0],)
