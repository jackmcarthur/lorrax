"""The r-grid maps share their algebra; they do NOT share their identity.

The register counted **twelve** live expressions of ``Rinv·r + τ`` / ``S·(r−τ)``
across three packages.  Seven were restatements and are gone; five survive
because they answer different questions.  These cells pin both halves of that,
because a later "unification" pass would erase the second half first.

WHAT IS SHARED — :func:`snap_to_grid_and_split_wrap`, and it is the whole
point of the consolidation.  The snap-before-floor rule used to be written out
in the SOURCE map and *not* in the forward one, so it had to be upgraded twice
and only one copy was ever correct.  Its evidence: naive ``floor`` flips an
``L`` component 0 → −1 on tiny negative noise and produces a spurious
``exp(±iπ/2)`` in ``unfold_isdf_operator`` — measured at **14 of 64 q, rel err
~0.8**, on Si Fd-3m.

WHAT IS NOT SHARED — the two DIRECTIONS.  The module docstring cites a silent
4 eV gap on hex systems from confusing them, and only the source map may
consume the ``L`` the kernel returns.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))


def _sym():
    try:
        from ffi import _services
        _services.ensure_on_path()
        from symmetry_maps import orbit_syms
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"symmetry service unavailable ({type(exc).__name__})")
    return orbit_syms


# ---------------------------------------------------------------------------
# The shared kernel
# ---------------------------------------------------------------------------

def test_the_snap_beats_a_naive_floor_on_the_exact_noise_that_broke_it():
    """The 14/64 q defect, reproduced in four numbers.

    A point whose true image has integer part 0, perturbed by one ulp of
    NEGATIVE noise.  ``floor`` sends ``L`` to −1; the snap keeps it at 0.
    Without this cell the kernel could be "simplified" back to
    ``images - np.floor(images)`` and every test would still pass.
    """
    os_ = _sym()
    grid = np.array([24, 24, 24])
    exact = np.array([[0.0, 5.0 / 24.0, 0.0]])
    noisy = exact - 1e-16                       # the discontinuity, from below

    idx, L = os_.snap_to_grid_and_split_wrap(noisy, grid)
    assert np.array_equal(L, np.zeros_like(L)), (
        f"the snap let a negative-noise point wrap: L={L}")
    assert np.array_equal(idx, np.array([[0, 5, 0]]))

    # The construction it replaced, on the same input, gets it wrong.
    naive_L = np.floor(noisy).astype(np.int64)
    assert naive_L[0, 0] == -1, (
        "the naive floor no longer misbehaves on this input, so this cell "
        "is no longer demonstrating the defect it was written for")


def test_the_kernel_splits_a_real_wrap_correctly():
    """RED TWIN: it must not simply return L=0 always."""
    os_ = _sym()
    grid = np.array([24, 24, 24])
    # +1 cell in x, −1 cell in z, plus an interior offset.
    pts = np.array([[1.0 + 5.0 / 24.0, 3.0 / 24.0, -1.0 + 7.0 / 24.0]])
    idx, L = os_.snap_to_grid_and_split_wrap(pts, grid)
    assert np.array_equal(L, np.array([[1, 0, -1]])), L
    assert np.array_equal(idx, np.array([[5, 3, 7]])), idx


def test_the_kernel_index_is_always_inside_the_box():
    os_ = _sym()
    rng = np.random.default_rng(3)
    grid = np.array([12, 15, 20])
    pts = rng.uniform(-3.0, 3.0, size=(500, 3))
    pts = np.rint(pts * grid) / grid              # commensurate, as real ones are
    idx, L = os_.snap_to_grid_and_split_wrap(pts, grid)
    assert (idx >= 0).all() and (idx < grid).all()
    # And the split is exact: idx + L*N reconstructs the snapped integer.
    assert np.array_equal(idx + L * grid, np.rint(pts * grid).astype(np.int64))


# ---------------------------------------------------------------------------
# The directions stay apart
# ---------------------------------------------------------------------------

def test_the_two_named_maps_are_still_two_functions():
    """S1 and S2 must not have been merged into one entry point.

    They share the kernel; they are not the same map.  If a later pass
    collapses them into one function with a direction flag, this fails —
    which is the intent, because that flag is the 4 eV hex-system bug.
    """
    os_ = _sym()
    assert callable(getattr(os_, "centroid_source_map_and_wrap", None))
    assert callable(getattr(os_, "fft_grid_pullback_perm", None))
    assert os_.centroid_source_map_and_wrap is not os_.fft_grid_pullback_perm


def test_the_pullback_discards_L_and_the_source_map_keeps_it():
    """The exemption, asserted from the source rather than trusted.

    ``fft_grid_pullback_perm`` may use the cheap answer for ``L`` because it
    never reads ``L``; ``centroid_source_map_and_wrap`` may not, because its
    ``L`` drives an umklapp phase.  A pullback that started consuming ``L``
    would silently acquire the source map's correctness requirement.
    """
    import ast
    import inspect
    os_ = _sym()
    tree = ast.parse(inspect.getsource(os_.fft_grid_pullback_perm))
    call = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and "snap_to_grid_and_split_wrap" in ast.unparse(n.func)]
    assert len(call) == 1, "the pullback no longer routes through the kernel"
    # Its second return value must be bound to a throwaway, never used.
    assign = [n for n in ast.walk(tree)
              if isinstance(n, ast.Assign)
              and "snap_to_grid_and_split_wrap" in ast.unparse(n.value)]
    assert len(assign) == 1
    names = [t.id for t in ast.walk(assign[0].targets[0])
             if isinstance(t, ast.Name)]
    body = ast.unparse(tree)
    used = [n for n in names if n.startswith("_")
            and body.count(n) > 1]
    assert not used, (
        f"the pullback now READS the lattice wrap it used to discard ({used}); "
        f"if that is intended it inherits the source map's snap requirement "
        f"and this exemption must be re-argued")


# ---------------------------------------------------------------------------
# The forward action's wrap flag
# ---------------------------------------------------------------------------

def test_r_action_forward_requires_the_wrap_decision():
    """No default: both answers are live in this tree and both look right."""
    os_ = _sym()
    rng = np.random.default_rng(5)
    pts, Rinv, tau = (rng.random((7, 3)),
                      np.eye(3)[None].repeat(2, 0), np.zeros((2, 3)))
    with pytest.raises(TypeError):
        os_.r_action_forward(pts, Rinv, tau)
    with pytest.raises(TypeError):
        os_.r_action_forward_one(pts, Rinv[0], tau[0])


def test_the_wrap_flag_actually_changes_the_answer():
    """Otherwise the cell above would pin a parameter that does nothing."""
    os_ = _sym()
    pts = np.array([[1.25, -0.5, 0.75]])
    Rinv, tau = np.eye(3)[None], np.zeros((1, 3))
    wrapped = np.asarray(os_.r_action_forward(pts, Rinv, tau, wrap=True))
    plain = np.asarray(os_.r_action_forward(pts, Rinv, tau, wrap=False))
    assert np.allclose(wrapped, [[[0.25, 0.5, 0.75]]])
    assert np.allclose(plain, [[[1.25, -0.5, 0.75]]])
    assert not np.allclose(wrapped, plain)


def test_the_stack_and_single_op_forms_agree_to_roundoff_not_bitwise():
    """Same map — but NOT bit-identical, and the difference is instructive.

    ``r_action_forward`` contracts with ``np.einsum`` and
    ``r_action_forward_one`` with ``@``.  MEASURED on this input: they differ
    on **2 of 276** entries by **2.220e-16 absolute / 2.147e-16 relative** —
    one ulp, from BLAS choosing a different summation order for the
    three-term dot product.

    THIS IS WHY EACH CALL SITE KEPT THE FORM IT ALREADY HAD.  The two numpy
    orbit sites were on ``einsum`` and went to the stack form; the four
    ``kmeans_isdf`` sites were on ``@`` and went to the single-op form.  Had
    the consolidation crossed them over, it would have been a one-ulp change
    to centroid selection dressed as a refactor — and a pivoted-Cholesky
    pivot order is exactly the kind of thing one ulp can flip.

    So the bar here is roundoff, and the bar at the CALL SITES is bit
    equality (``consol_parity.py``, all seven restatements).
    """
    os_ = _sym()
    rng = np.random.default_rng(11)
    pts = rng.random((23, 3))
    Rinv = rng.integers(-1, 2, size=(4, 3, 3)).astype(np.float64)
    tau = rng.random((4, 3))
    for wrap in (True, False):
        stack = np.asarray(os_.r_action_forward(pts, Rinv, tau, wrap=wrap))
        for s in range(4):
            one = np.asarray(
                os_.r_action_forward_one(pts, Rinv[s], tau[s], wrap=wrap))
            d = float(np.abs(stack[s] - one).max())
            assert d < 1e-14, (
                f"stack and single-op forms disagree at s={s}, wrap={wrap} "
                f"by {d:.3e} — that is far above the one-ulp contraction-order "
                f"difference this cell allows, so it is a real divergence")


# ---------------------------------------------------------------------------
# S4 — the integer-grid map
# ---------------------------------------------------------------------------

def test_grid_point_image_perm_is_a_permutation():
    """A 4-fold rotation about z needs Nx == Ny to map the box to itself.

    Stated because the first version of this cell used ``[4, 6, 5]`` and
    failed: ``(x, y) -> (-y, x)`` sends a 4x6 footprint to a 6x4 one, so it
    is not a permutation of that grid at all.  The operation and the grid
    have to be compatible; that is the caller's job (``M`` must come from
    ``recover_symmorphic_density_point_group``, which only returns ops that
    ARE grid symmetries) and not something this function can check.
    """
    os_ = _sym()
    grid = np.array([6, 6, 5])
    M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    perm = os_.grid_point_image_perm(grid, M)
    assert perm.shape == (int(grid.prod()),)
    assert np.array_equal(np.sort(perm), np.arange(grid.prod()))


def test_grid_point_image_perm_matches_the_open_coding_it_replaced():
    """Bit equality against the expression deleted from two packages."""
    os_ = _sym()
    grid = np.array([4, 6, 5])
    ix, iy, iz = np.meshgrid(*(np.arange(int(n)) for n in grid), indexing="ij")
    n_idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)
    rng = np.random.default_rng(7)
    for _ in range(6):
        M = rng.permutation(np.eye(3, dtype=np.int64)) * rng.choice([-1, 1])
        img = (n_idx @ M.T) % grid[None, :]
        old = img[:, 0] * (grid[1] * grid[2]) + img[:, 1] * grid[2] + img[:, 2]
        assert np.array_equal(old, os_.grid_point_image_perm(grid, M))


def test_the_identity_is_the_identity_permutation():
    os_ = _sym()
    grid = np.array([3, 4, 5])
    perm = os_.grid_point_image_perm(grid, np.eye(3, dtype=np.int64))
    assert np.array_equal(perm, np.arange(grid.prod()))
