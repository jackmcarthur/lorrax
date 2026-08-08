"""Layer L-b: the sharded q-axis unfolds on an EMULATED four-device mesh.

``unfold_isdf_operator`` is a ``shard_map`` over an ``('x','y')`` mesh
with a 1x single-tile memory contract: at no point may a rank hold a
full μ or ν axis, so the μ-permute and the ν-permute each go through
``lax.all_to_all`` to redistribute the OTHER spatial axis.  That machinery
is invisible at 1x1 — both ``all_to_all`` branches are skipped — so every
claim about it has to come from a mesh with Px > 1 and Py > 1.

WHAT THE SURVEY RANKED HIGH AND NOTHING COVERED (§7.2):

* **G10** — "No hostile geometry anywhere for ``unfold_isdf_operator``'s
  ``all_to_all`` path.  Its own guard requires ``n_rmu_padded % (Px·Py) ==
  0`` and raises otherwise — but nothing constructs the failing case, and
  nothing exercises the ``Px>1, Py>1`` redistribution at a non-trivial
  extent."  HIGH, and the CHARTER's mandatory band-pad class for a
  mesh-touching service.  Both halves are here: the refusal AND the red
  twin showing the divisible case going through the same code path.
* **G6** — ``mix_channels_by_proper_rotation`` has no direct test anywhere.

HOW THE FOUR DEVICES GET HERE.  ``conftest.py`` sets
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` before the first jax
import, which only takes when this suite is the reason the process started
(``pytest services/symmetry_maps/tests``).  In the FULL monorepo run
``testpaths = ["tests", "services"]`` collects tests/ first, something
there imports jax during collection, and the flag is inert — the cells
below then SKIP through ``lxkit.testing.require_devices``, which skips and
never asserts.  That is the correct outcome, not a degradation: forcing
four host devices on the whole monorepo run would change the device count
every other lorrax test sees.

THE GEOMETRY IS SYNTHETIC AND ORBIT-CLOSED BY CONSTRUCTION.  Extending
``tests/test_symmetry_unfold.py``'s ``_build_geometry_vq`` pattern to
twelve centroids, because 12 is divisible by Px·Py = 4 and 6 (the original)
is not.  No production ``centroids_frac_*.txt`` is read, and none is ever
regenerated (survey §8.2, register-don't-touch).
"""

from __future__ import annotations

import numpy as np
import pytest

from lxkit.testing import require_devices
from symmetry_maps import (KStarMap, centroid_source_map_and_wrap,
                           mix_channels_by_proper_rotation, star_broadcast,
                           star_select, star_spread,
                           unfold_isdf_operator)

#: {I, σ_y} on a 12x12x1 FFT grid — the smallest thing with a non-trivial
#: permutation, a real umklapp wrap, and a TRS half.
_FFT = np.array([12, 12, 1], dtype=np.int64)
_SYMS = np.stack([np.eye(3, dtype=np.int64),
                  np.diag([1, -1, 1]).astype(np.int64)])
_NTRAN = 2


# Seeds, and the two properties the set has to have.
#
# 1. SIZE.  On the 12x12x1 grid σ_y sends (x, y) to (x, −y mod 12), so a
#    seed with y in {0, 6} is its own image (one point) and any other seed
#    contributes two.  The counts below are therefore 2 self + 5 pairs = 12
#    and 2 self + 4 pairs = 10.
#
# 2. A LATTICE WRAP THAT VARIES ACROSS μ.  The image of (x, y) with y > 0
#    folds back with L_y = −1; a seed on y = 0 folds with L_y = 0.  The
#    umklapp phase in ``unfold_isdf_operator`` is ``exp(2πi q·(L_μ − L_ν))``, a
#    DIFFERENCE — so a set whose centroids all share one L makes the phase
#    identically 1 no matter how large L is, and the whole term goes
#    untested while looking exercised.  MEASURED writing this file: the
#    first draft used six y>0 seeds, every L_y was −1, and dropping the
#    phase from the reference changed the answer by 1.7e-16.  The self-image
#    seeds are what make the difference non-zero, and
#    :func:`test_the_hand_reference_is_not_a_reimplementation` is the cell
#    that caught it.
_SEEDS_12 = ((0, 0), (3, 6), (1, 1), (2, 1), (3, 2), (4, 3), (5, 1))
_SEEDS_10 = _SEEDS_12[:6]


def _geometry(seeds):
    """(sym_perm, L_table, n_rmu) for an orbit-closed centroid set.

    Closed BY CONSTRUCTION — the set is the union of the seeds' orbits —
    so ``validate=True`` passes and the closure is a property of this
    function rather than of a file.  No production
    ``centroids_frac_*.txt`` is read and none is ever regenerated (survey
    §8.2, register-don't-touch).
    """
    rinv = np.rint(np.linalg.inv(_SYMS)).astype(np.int64)
    pts = np.array([[x, y, 0] for x, y in seeds], dtype=np.int64)
    imgs = {tuple(((rinv[s] @ r) % _FFT).tolist())
            for r in pts for s in range(_SYMS.shape[0])}
    cent = np.array(sorted(imgs), dtype=np.int32)
    perm, L = centroid_source_map_and_wrap(
        cent, _SYMS, np.zeros((2, 3)), _FFT, validate=True, extend_trs=True)
    return perm, L, int(cent.shape[0])


def _divisible_geometry():
    """n_rmu = 12: divisible by Px*Py = 4 AND by 4x1 and 1x1."""
    perm, L, n = _geometry(_SEEDS_12)
    assert n == 12 and n % 4 == 0, f"expected 12 centroids, got {n}"
    return perm, L, n


def _hostile_geometry():
    """n_rmu = 10: NOT divisible by 4.  G10's failing case, constructed."""
    perm, L, n = _geometry(_SEEDS_10)
    assert n == 10 and n % 4 != 0, f"expected 10 centroids, got {n}"
    return perm, L, n


#: A full-BZ q map that uses BOTH halves of the sym table: two spatial rows
#: and two TRS rows, so the conj-wrap is live and so is the permutation.
_IRR = np.array([0, 1, 2, 0, 1], dtype=np.int32)
_SYM = np.array([0, 1, 0, _NTRAN + 0, _NTRAN + 1], dtype=np.int32)
#: The IBZ q's, in fractional reciprocal coordinates.  Parent 1 is the one
#: the non-identity sym rows fold to, and its q MUST have a non-zero
#: y-component: the only non-zero lattice wrap this geometry produces is on
#: the y axis (σ_y sends (x, y) to (x, −y), which folds back with L_y =
#: −1), so a q with q_y = 0 makes ``exp(2πi q·L)`` identically 1 and the
#: umklapp phase silently untested.  Caught by
#: :func:`test_the_hand_reference_is_not_a_reimplementation`, which is
#: exactly what that cell is for.
_Q_IRR = np.array([[0.0, 0.0, 0.0],
                   [1.0 / 12, 1.0 / 3, 0.0],
                   [1.0 / 4, 1.0 / 6, 0.0]])


def _hermitian_ibz(n_rmu, n_q=3, seed=17):
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((n_q, n_rmu, n_rmu))
         + 1j * rng.standard_normal((n_q, n_rmu, n_rmu)))
    return 0.5 * (a + np.swapaxes(a.conj(), -1, -2))


def _hand_unfold(V_ibz, *, perm, L, n_rmu):
    """The per-element reference, in plain numpy.

    Same expression as ``tests/test_symmetry_unfold.py``'s
    ``_hand_unfold_v_q`` — the ONE formula, written out where a reader can
    check it against ``unfold_isdf_operator``'s docstring::

        V_full[q, μ, ν] = exp(2πi q_irr·(L_{s,μ} − L_{s,ν}))
                          · V_ibz[i(q), α_s(μ), α_s(ν)]

    conjugated on TRS rows.  It is a reference and not a reimplementation
    of the kernel: no mesh, no all_to_all, no jit.
    """
    out = np.zeros((len(_IRR), n_rmu, n_rmu), dtype=V_ibz.dtype)
    for iq in range(len(_IRR)):
        parent, s = int(_IRR[iq]), int(_SYM[iq])
        p = np.asarray(perm[s])
        qL = np.asarray(L[s], dtype=float) @ _Q_IRR[parent]
        ph = np.exp(2j * np.pi * qL)
        blk = V_ibz[parent][np.ix_(p, p)]
        blk = ph[:, None] * blk * np.conj(ph)[None, :]
        out[iq] = np.conj(blk) if s >= _NTRAN else blk
    return out


def _mesh(px, py):
    import jax
    from jax.sharding import Mesh
    require_devices(px * py, platform="cpu")
    devs = np.asarray(jax.devices("cpu")[:px * py]).reshape(px, py)
    return Mesh(devs, ("x", "y"))


# ---------------------------------------------------------------------------
# Correctness across mesh shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("px,py", [(1, 1), (2, 2), (4, 1)])
def test_the_unfold_matches_the_hand_reference_on_every_mesh(px, py):
    """The SAME numbers from 1x1, 2x2 and 4x1.

    1x1 takes neither ``all_to_all`` branch, 4x1 takes the μ one only, 2x2
    takes both.  Three different HLOs, one answer — which is what the
    memory contract is worth: a redistribution that lost or duplicated a
    tile would show up as a permutation error here and nowhere else.

    The bar is 1e-13 relative and not bit-equality: the sharded kernel
    computes the umklapp phase per rank from a dynamic slice, which is the
    same arithmetic in a different order.
    """
    import jax.numpy as jnp
    mesh = _mesh(px, py)
    perm, L, n_rmu = _divisible_geometry()
    V = _hermitian_ibz(n_rmu)
    ref = _hand_unfold(V, perm=perm, L=L, n_rmu=n_rmu)
    got = np.asarray(unfold_isdf_operator(
        jnp.asarray(V), irr_idx=_IRR, sym_idx=_SYM, sym_perm=perm, L_table=L,
        q_irr_frac=_Q_IRR, mesh_xy=mesh, n_sym_spatial=_NTRAN))
    assert got.shape == (len(_IRR), n_rmu, n_rmu)
    rel = float(np.abs(got - ref).max() / np.abs(ref).max())
    assert rel < 1e-13, f"{px}x{py}: unfold differs from the reference: {rel:.3e}"


def test_the_hand_reference_is_not_a_reimplementation():
    """ANTI-TAUTOLOGY for the cell above.

    Drop the umklapp phase from the reference and it must DISAGREE with the
    kernel.  If it did not, the comparison would be measuring the
    permutation only and the phase — which is "essential whenever ``S r_μ +
    τ`` exits the unit cell", i.e. on every non-trivial full-BZ q — would
    be untested.
    """
    import jax.numpy as jnp
    mesh = _mesh(1, 1)
    perm, L, n_rmu = _divisible_geometry()
    V = _hermitian_ibz(n_rmu)
    got = np.asarray(unfold_isdf_operator(
        jnp.asarray(V), irr_idx=_IRR, sym_idx=_SYM, sym_perm=perm, L_table=L,
        q_irr_frac=_Q_IRR, mesh_xy=mesh, n_sym_spatial=_NTRAN))
    nophase = _hand_unfold(V, perm=perm, L=np.zeros_like(L), n_rmu=n_rmu)
    rel = float(np.abs(got - nophase).max() / np.abs(got).max())
    assert rel > 1e-3, (
        f"dropping the umklapp phase changed nothing ({rel:.3e}); this "
        f"geometry has no lattice wrap and the reference is measuring only "
        f"the permutation")
    assert np.abs(np.asarray(L)).max() > 0, "L_table is identically zero"


def test_the_conj_wrap_on_trs_rows_is_live():
    """ANTI-TAUTOLOGY two: the TRS half is doing something.

    Compare the kernel's TRS rows against the reference computed WITHOUT
    the conjugation.  A large disagreement is what says
    ``V_full[TRS-q] = conj(V_ibz[parent])`` is actually applied — the rule
    whose absence is the whole ``unfold_isdf_operator`` half of the TRS bug
    class.
    """
    import jax.numpy as jnp
    mesh = _mesh(1, 1)
    perm, L, n_rmu = _divisible_geometry()
    V = _hermitian_ibz(n_rmu)
    got = np.asarray(unfold_isdf_operator(
        jnp.asarray(V), irr_idx=_IRR, sym_idx=_SYM, sym_perm=perm, L_table=L,
        q_irr_frac=_Q_IRR, mesh_xy=mesh, n_sym_spatial=_NTRAN))
    ref = _hand_unfold(V, perm=perm, L=L, n_rmu=n_rmu)
    trs_rows = np.where(_SYM >= _NTRAN)[0]
    assert trs_rows.size >= 2, "PRECONDITION: this map must use TRS rows"
    unconj = ref.copy()
    unconj[trs_rows] = np.conj(unconj[trs_rows])
    rel = float(np.abs(got[trs_rows] - unconj[trs_rows]).max()
                / np.abs(ref).max())
    assert rel > 1e-3, (
        f"removing the conjugation on the TRS rows changed nothing "
        f"({rel:.3e}); the antiunitary branch is not live")


def test_the_trivial_map_short_circuits_to_its_input():
    """``irr_idx`` identity + ``sym_idx`` all-zero returns the operand.

    Not an optimization detail: the ``take_along_axis`` path was observed
    to trip an XLA HLO verifier dtype mismatch (s64 broadcast vs s32
    operand) on a 2x2, so the bypass is load-bearing on exactly the mesh
    this tier runs.  Asserted as IDENTITY, which is what "bypass entirely"
    means.
    """
    import jax.numpy as jnp
    mesh = _mesh(2, 2)
    perm, L, n_rmu = _divisible_geometry()
    V = jnp.asarray(_hermitian_ibz(n_rmu, n_q=5))
    out = unfold_isdf_operator(
        V, irr_idx=np.arange(5, dtype=np.int32),
        sym_idx=np.zeros(5, dtype=np.int32), sym_perm=perm, L_table=L,
        q_irr_frac=np.zeros((5, 3)), mesh_xy=mesh, n_sym_spatial=_NTRAN)
    assert out is V


# ---------------------------------------------------------------------------
# G10 — hostile geometry
# ---------------------------------------------------------------------------

def test_a_non_dividing_centroid_extent_refuses_on_a_2x2():
    """G10.  ``n_rmu_padded % (Px·Py) != 0`` RAISES, naming both numbers.

    The ``all_to_all`` redistribution splits ν/Py into ν/(Py·Px), so the
    extent has to divide the whole mesh — not just one axis.  Without the
    guard the split would be silently wrong, and the gathers downstream use
    ``mode='promise_in_bounds'``, which CLIPS out-of-range indices with no
    error: the TRS-bug failure shape.

    MEASURED: n_rmu = 10 against Px*Py = 4.  The refusal must say which
    extent and which product, because "not divisible" without the numbers
    sends the reader to go and print them.
    """
    import jax.numpy as jnp
    mesh = _mesh(2, 2)
    perm, L, n_rmu = _hostile_geometry()
    assert n_rmu % 4 != 0, (
        f"PRECONDITION: n_rmu={n_rmu} must NOT divide the 2x2 mesh, or "
        f"this cell is asserting a refusal that cannot happen")
    V = jnp.asarray(_hermitian_ibz(n_rmu))
    with pytest.raises(ValueError) as ei:
        unfold_isdf_operator(
            V, irr_idx=_IRR, sym_idx=_SYM, sym_perm=perm, L_table=L,
            q_irr_frac=_Q_IRR, mesh_xy=mesh, n_sym_spatial=_NTRAN)
    msg = str(ei.value)
    assert f"n_rmu_padded={n_rmu}" in msg and "Px*Py=4" in msg, msg
    assert "all_to_all" in msg and "n_rmu_padded" in msg


def test_the_same_hostile_extent_goes_through_at_1x1():
    """RED TWIN for the refusal: it is about the MESH, not the geometry.

    The exact operand and tables the 2x2 refuses run to completion on a
    1x1 and match the hand reference.  Without this the refusal could be
    hiding a broken geometry and the cell above would read as a pass.
    """
    import jax.numpy as jnp
    mesh = _mesh(1, 1)
    perm, L, n_rmu = _hostile_geometry()
    V = _hermitian_ibz(n_rmu)
    got = np.asarray(unfold_isdf_operator(
        jnp.asarray(V), irr_idx=_IRR, sym_idx=_SYM, sym_perm=perm, L_table=L,
        q_irr_frac=_Q_IRR, mesh_xy=mesh, n_sym_spatial=_NTRAN))
    ref = _hand_unfold(V, perm=perm, L=L, n_rmu=n_rmu)
    assert float(np.abs(got - ref).max() / np.abs(ref).max()) < 1e-13


def test_a_divisible_extent_passes_the_same_2x2_code_path():
    """The other half of the twin: 12 divides 4, so the 2x2 runs.

    Same mesh, same kernel, same tables shape — only the extent differs.
    That is what makes the refusal above evidence about divisibility rather
    than about the 2x2 path being broken in general.
    """
    import jax.numpy as jnp
    mesh = _mesh(2, 2)
    perm, L, n_rmu = _divisible_geometry()
    assert n_rmu % 4 == 0
    got = np.asarray(unfold_isdf_operator(
        jnp.asarray(_hermitian_ibz(n_rmu)), irr_idx=_IRR, sym_idx=_SYM,
        sym_perm=perm, L_table=L, q_irr_frac=_Q_IRR, mesh_xy=mesh,
        n_sym_spatial=_NTRAN))
    assert got.shape == (len(_IRR), n_rmu, n_rmu)


def test_a_sym_table_that_does_not_cover_the_trs_half_refuses():
    """S7.  ``sym_perm`` must reach every row ``sym_idx`` names.

    Two distinct refusals with two distinct fixes, both checked: a table
    too short for ``max(sym_idx)`` at all, and a table long enough by
    accident but not ``2·n_sym_spatial`` when TRS rows are in play.  Each
    exists because a ``promise_in_bounds`` gather CLIPS rather than raises.
    """
    import jax.numpy as jnp
    mesh = _mesh(1, 1)
    perm, L, n_rmu = _divisible_geometry()
    V = jnp.asarray(_hermitian_ibz(n_rmu))
    with pytest.raises(ValueError, match="only 2 rows"):
        unfold_isdf_operator(V, irr_idx=_IRR, sym_idx=_SYM, sym_perm=perm[:2],
                             L_table=L[:2], q_irr_frac=_Q_IRR, mesh_xy=mesh,
                             n_sym_spatial=_NTRAN)
    with pytest.raises(ValueError, match="2·n_sym_spatial"):
        unfold_isdf_operator(V, irr_idx=_IRR, sym_idx=_SYM, sym_perm=perm,
                             L_table=L, q_irr_frac=_Q_IRR, mesh_xy=mesh,
                             n_sym_spatial=3)


def test_a_mu_extent_mismatch_refuses_in_both_tables():
    """S7 continued: V, ``sym_perm`` and ``L_table`` share ONE padded extent.

    "Tables carry the μ pad from construction" — a caller that passed the
    logical V against padded tables would feed the gather out-of-range
    indices, which clip silently.  Both the ``sym_perm`` check and the
    ``L_table`` check are separate branches with separate messages.
    """
    import jax.numpy as jnp
    mesh = _mesh(1, 1)
    perm, L, n_rmu = _divisible_geometry()
    short = jnp.asarray(_hermitian_ibz(n_rmu)[:, :-2, :-2])
    with pytest.raises(ValueError, match="sym_perm extent"):
        unfold_isdf_operator(short, irr_idx=_IRR, sym_idx=_SYM, sym_perm=perm,
                             L_table=L, q_irr_frac=_Q_IRR, mesh_xy=mesh,
                             n_sym_spatial=_NTRAN)
    V = jnp.asarray(_hermitian_ibz(n_rmu))
    with pytest.raises(ValueError, match="L_table μ-extent"):
        unfold_isdf_operator(
            V, irr_idx=_IRR, sym_idx=_SYM, sym_perm=perm,
            L_table=np.asarray(L)[:, :-2, :], q_irr_frac=_Q_IRR, mesh_xy=mesh,
            n_sym_spatial=_NTRAN)


# ---------------------------------------------------------------------------
# G6 — mix_channels_by_proper_rotation
# ---------------------------------------------------------------------------

def _r_proper(rng, n_rows):
    """A ``(2·ntran, 3, 3)`` R_proper-shaped table: proper rotations, TRS
    half duplicating the spatial half (``SymMaps.R_proper``'s own shape)."""
    spatial = []
    for _ in range(n_rows):
        a = rng.standard_normal((3, 3))
        q, _r = np.linalg.qr(a)
        if np.linalg.det(q) < 0:
            q = -q
        spatial.append(q)
    sp = np.stack(spatial)
    return np.concatenate([sp, sp], axis=0)


@pytest.mark.parametrize("px,py", [(1, 1), (2, 2)])
def test_the_lorentz_mixing_matches_a_dense_numpy_reference(px, py):
    """G6.  The §A5 3-vector mixing, against an explicit double sum.

    ``mix_channels_by_proper_rotation`` applies, per full-BZ q::

        V^{i,j}_mixed = Σ_{α,β} R^{α,i}(q) · R^{β,j}(q) · V^{α,β}

    THE COMPENSATING TRANSPOSE, which is the trap this cell exists for.
    The derivation text writes ``R_deriv^{i,α} · R_deriv^{j,β}``, but the
    LIVE ``R_proper`` is built from ``A·mtrx·A⁻¹`` while the derivation's
    is ``A·mtrx⁻¹·A⁻¹`` — for orthogonal ``mtrx`` the two are transposes of
    each other, so the live contraction carries the indices the other way
    round.  ``SymMaps.R_proper``'s docstring records the same fact about
    the offline CrI3 fixture: it "differs from the OFFLINE fixture
    ``cri3_R_proper.npz`` by a transpose on every row".  The reference
    below is written in the LIVE convention, and
    :func:`test_the_lorentz_reference_notices_a_transposed_R` is the twin
    that proves the difference is visible.
    """
    import jax.numpy as jnp
    mesh = _mesh(px, py)
    rng = np.random.default_rng(23)
    n_q, n_mu = 5, 8
    R = _r_proper(rng, _NTRAN)
    tiles = {}
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            a = (rng.standard_normal((n_q, n_mu, n_mu))
                 + 1j * rng.standard_normal((n_q, n_mu, n_mu)))
            tiles[(i, j)] = a
    out = mix_channels_by_proper_rotation(
        {k: jnp.asarray(v) for k, v in tiles.items()},
        sym_idx=_SYM, R_proper_table=R, mesh_xy=mesh)

    Rq = R[np.asarray(_SYM)]                                  # (n_q, 3, 3)
    for i in (1, 2, 3):
        for j in (1, 2, 3):
            ref = np.zeros((n_q, n_mu, n_mu), dtype=np.complex128)
            for a in (1, 2, 3):
                for b in (1, 2, 3):
                    coeff = Rq[:, a - 1, i - 1] * Rq[:, b - 1, j - 1]
                    ref += coeff[:, None, None] * tiles[(a, b)]
            got = np.asarray(out[(i, j)])
            rel = float(np.abs(got - ref).max()
                        / max(np.abs(ref).max(), 1e-300))
            assert rel < 1e-13, f"{px}x{py} tile ({i},{j}): {rel:.3e}"


def test_the_lorentz_reference_notices_a_transposed_R():
    """ANTI-TAUTOLOGY / the compensating transpose, made VISIBLE.

    Contract with ``R.T`` per op instead of ``R`` and the answer must
    change.  That is the entire content of the §3.1 trap: the wrong
    convention leaves the tiles the right shape, keeps every norm and every
    trace, and is wrong.  A cell that only compared the function against
    itself would never see it.
    """
    rng = np.random.default_rng(29)
    n_q, n_mu = 5, 4
    R = _r_proper(rng, _NTRAN)
    Rq = R[np.asarray(_SYM)]
    tiles = {(i, j): (rng.standard_normal((n_q, n_mu, n_mu))
                      + 1j * rng.standard_normal((n_q, n_mu, n_mu)))
             for i in (1, 2, 3) for j in (1, 2, 3)}

    def mix(Rt):
        out = {}
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                acc = np.zeros((n_q, n_mu, n_mu), dtype=np.complex128)
                for a in (1, 2, 3):
                    for b in (1, 2, 3):
                        acc += (Rt[:, a - 1, i - 1] * Rt[:, b - 1, j - 1]
                                )[:, None, None] * tiles[(a, b)]
                out[(i, j)] = acc
        return out

    live = mix(Rq)
    flipped = mix(np.swapaxes(Rq, 1, 2))
    worst = max(float(np.abs(live[k] - flipped[k]).max()) for k in live)
    scale = max(float(np.abs(live[k]).max()) for k in live)
    assert worst / scale > 0.1, (
        f"transposing R changed the mixing by only {worst / scale:.3e}; "
        f"this draw of R is too close to symmetric to be a discriminator")


def test_the_lorentz_mix_refuses_a_missing_tile_and_a_bad_table():
    """Both refusals: all nine (i,j) required, and R is ``(n, 3, 3)``.

    Nine, not six: the caller may synthesise the Hermitian-redundant
    entries with ``conj(swapaxes(...))``, but the function will not guess.
    """
    import jax.numpy as jnp
    mesh = _mesh(1, 1)
    rng = np.random.default_rng(31)
    R = _r_proper(rng, _NTRAN)
    tiles = {(i, j): jnp.asarray(rng.standard_normal((5, 4, 4))
                                 + 0j)
             for i in (1, 2, 3) for j in (1, 2, 3)}
    partial = {k: v for k, v in tiles.items() if k != (2, 3)}
    with pytest.raises(ValueError, match=r"missing TT tile"):
        mix_channels_by_proper_rotation(partial, sym_idx=_SYM,
                                        R_proper_table=R, mesh_xy=mesh)
    with pytest.raises(ValueError, match=r"\(2·n_sym_spatial, 3, 3\)"):
        mix_channels_by_proper_rotation(
            tiles, sym_idx=_SYM, R_proper_table=R[:, :2, :], mesh_xy=mesh)


# ---------------------------------------------------------------------------
# KStarMap on a sharded operand (S3 / S11 check 4, the mesh half)
# ---------------------------------------------------------------------------

def test_the_star_helpers_keep_a_sharded_operand_sharded():
    """S11 check 4 on a REAL 2x2: bit-identical, and the spec survives.

    ``star_select`` and ``star_broadcast`` gather along axis 0 and pin the
    operand's own sharding when axis 0 is replicated — which every SC
    operand is (``P(None,'x','y')`` for U and Σ).  If axis 0 were
    mesh-sharded the new k extent need not divide that mesh axis, so the
    helpers deliberately hand the layout back to GSPMD instead; the case
    exercised here is the one the SC loop actually passes.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = _mesh(2, 2)
    irr = np.array([0, 2, 2, 6, 8, 7, 6, 7, 8], dtype=np.int32)
    sidx = np.array([0, 2, 0, 2, 2, 2, 0, 0, 0], dtype=np.int32)
    rng = np.random.default_rng(37)
    base = {int(v): None for v in irr}
    for v in base:
        m = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        base[v] = 0.5 * (m + m.conj().T)
    from symmetry_maps.maps import _star_conj_flags
    _, xor = _star_conj_flags(irr, sidx, 2)
    A = np.stack([np.conj(base[int(irr[k])]) if xor[k] else base[int(irr[k])]
                  for k in range(len(irr))])

    sh = NamedSharding(mesh, P(None, "x", "y"))
    dev = jax.device_put(jnp.asarray(A), sh)
    sel_d, labels = star_select(dev, irr)
    back_d = star_broadcast(sel_d, irr, sidx, 2, irr_labels=labels)
    assert isinstance(sel_d, jax.Array) and isinstance(back_d, jax.Array)
    assert sel_d.sharding.spec == sh.spec
    assert back_d.sharding.spec == sh.spec

    sel_h, _ = star_select(A, irr)
    back_h = star_broadcast(sel_h, irr, sidx, 2, irr_labels=labels)
    assert float(np.abs(np.asarray(sel_d) - sel_h).max()) == 0.0
    assert float(np.abs(np.asarray(back_d) - back_h).max()) == 0.0
    assert star_spread(dev, irr, sidx, 2) == star_spread(A, irr, sidx, 2)

    km = KStarMap(irr, sidx, 2)
    assert km.spread_rel(dev) == km.spread_rel(A)
    assert isinstance(km.spread_rel(dev), float)


def test_the_device_bit_identity_can_fail():
    """RED TWIN.  Perturb the device operand and the comparison must move.

    Without it, "0.0 difference" could be two readbacks of the same array
    and the cell would prove nothing about the device path at all.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = _mesh(2, 2)
    irr = np.array([0, 0, 1, 1], dtype=np.int32)
    sidx = np.zeros(4, dtype=np.int32)
    A = np.ones((4, 4, 4), dtype=np.complex128)
    sh = NamedSharding(mesh, P(None, "x", "y"))
    bad = np.array(A)
    bad[1, 0, 0] += 1.0
    sel_d, _ = star_select(jax.device_put(jnp.asarray(bad), sh), irr)
    sel_h, _ = star_select(A, irr)
    assert float(np.abs(np.asarray(sel_d) - sel_h).max()) == 0.0, (
        "row 1 is not a kept row, so this perturbation must NOT show")
    bad2 = np.array(A)
    bad2[0, 0, 0] += 1.0
    sel_d2, _ = star_select(jax.device_put(jnp.asarray(bad2), sh), irr)
    assert float(np.abs(np.asarray(sel_d2) - sel_h).max()) == 1.0
