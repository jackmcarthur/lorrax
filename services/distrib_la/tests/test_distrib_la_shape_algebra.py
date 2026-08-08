"""Layer L-a: shape algebra, on a laptop, in milliseconds.

Hostile geometry is mandatory for every mesh-touching service (charter),
and it has three layers.  This is the first: pure arithmetic over the tile
decomposition, no mesh, no devices, no ``.so``.  Layers L-b (four emulated
devices, real arrays) and L-c (four real processes under ``srun``) are the
service test suite's own step; what is here is the part that must never
need a machine.

The one algebraic contract ``native2d`` has is its tile decomposition, and
it has THREE constraints at once — ``n % b == 0`` and ``J % Px == 0`` and
``J % Py == 0`` with ``J = n / b``.  Two of the three hold by accident for
most inputs, which is exactly why the third is worth a test.
"""

from __future__ import annotations

import math

import pytest
from lxkit.testing import hostile_extents

from distrib_la._native2d import block_size_for


def _mesh_shapes():
    """The mesh shapes this service is actually run on, plus the awkward
    ones.  4x4 is the shape the hostile-geometry table was measured at
    (Perlmutter job 56389339); 2x4 and 3x2 are non-square and coprime-ish,
    which is where lcm(Px, Py) stops being max(Px, Py)."""
    return [(1, 1), (2, 2), (4, 1), (1, 4), (2, 4), (4, 4), (3, 2)]


@pytest.mark.parametrize("mesh_shape", _mesh_shapes())
def test_the_block_size_satisfies_all_three_constraints(mesh_shape):
    """Whatever it returns must actually tile: this is the contract, and it
    is checked by RE-DERIVING it, not by re-running the search."""
    Px, Py = mesh_shape
    n = 12 * math.lcm(Px, Py)
    b, J = block_size_for(n, Px, Py)
    assert b > 0 and J > 0
    assert n % b == 0, (n, b)
    assert J == n // b
    assert J % Px == 0 and J % Py == 0, (J, Px, Py)


@pytest.mark.parametrize("mesh_shape", _mesh_shapes())
def test_hostile_extents_are_refused_or_tiled_honestly(mesh_shape):
    """Non-dividing extents either tile or REFUSE — never silently round.

    The five families come from :func:`lxkit.testing.hostile_extents`,
    which generalizes the table measured end-to-end on a 4x4 mesh (job
    56389339): prime extents on both axes, a tighter prime pair,
    non-divisible on one axis only, fewer slices than ranks, and empty
    tiles on one axis.  A prime n on a 2x2 mesh has NO valid decomposition,
    and the honest answer is the raise — silently padding would change the
    matrix.
    """
    Px, Py = mesh_shape
    for case in hostile_extents(mesh_shape):
        for n in (case.logical[0], case.padded[0]):
            try:
                b, J = block_size_for(n, Px, Py)
            except ValueError as exc:
                assert str(math.lcm(Px, Py)) in str(exc), (
                    f"a refusal must name the divisor it wanted: {exc}")
                continue
            assert n % b == 0 and (n // b) % Px == 0 and (n // b) % Py == 0, (
                f"block_size_for({n}, {Px}, {Py}) = {(b, J)} does not tile")


def test_a_prime_extent_on_a_real_mesh_refuses():
    """The anti-tautology cell: something in here has to actually raise, or
    the test above is a loop over cases that all succeed."""
    with pytest.raises(ValueError, match=r"no valid block size"):
        block_size_for(17, 2, 2)


def test_the_refusal_names_the_divisor_and_the_mesh():
    exc = pytest.raises(ValueError, block_size_for, 17, 2, 2).value
    assert "17" in str(exc) and "2x2" in str(exc) and "lcm(2,2)=2" in str(exc)


def test_dense_to_tiles_round_trips_on_the_lower_triangle():
    """L-a's other half: the layout conversion the facade hides.

    ``dense_to_tiles`` zeroes the strictly-upper TILES (``i < j`` in TILE
    indices), not the strictly-upper ELEMENTS — inside a DIAGONAL tile the
    upper triangle survives, because that is what ``jnp.linalg.cholesky``
    is handed and it reads the lower triangle itself.  So the round trip is
    the identity on ``tril`` and NOT on the full matrix, and the zero half
    is BLOCK-triangular, not triangular.

    Both halves are asserted, and asserting the wrong one is not
    hypothetical: the first draft of this cell claimed elementwise
    ``triu(back, 1) == 0`` and failed at ``b = 2`` on the diagonal tiles.
    A round-trip check that only compares ``tril`` cannot see the
    difference, which is how such a check becomes a no-op.
    """
    np = pytest.importorskip("numpy")
    jnp = pytest.importorskip("jax.numpy")
    from distrib_la._native2d import dense_to_tiles, tiles_to_dense

    rng = np.random.default_rng(4)
    A = jnp.asarray(rng.standard_normal((3, 12, 12)))
    idx = np.arange(12)
    for b in (1, 2, 3, 4, 6, 12):
        tiles = dense_to_tiles(A, b)
        assert tiles.shape == (3, 12 // b, 12 // b, b, b)
        back = tiles_to_dense(tiles, b)
        assert back.shape == A.shape
        assert bool(jnp.array_equal(jnp.tril(back), jnp.tril(A))), b
        above = jnp.asarray((idx[:, None] // b) < (idx[None, :] // b))
        assert bool(jnp.all(jnp.where(above, back, 0) == 0)), b
        # ...and the diagonal tiles are UNTOUCHED, which is the half a
        # tril-only comparison cannot see.
        on_diag = jnp.asarray((idx[:, None] // b) == (idx[None, :] // b))
        assert bool(jnp.array_equal(jnp.where(on_diag, back, 0),
                                    jnp.where(on_diag, A, 0))), b

    with pytest.raises(ValueError, match="divisible"):
        dense_to_tiles(A, 5)


# ---------------------------------------------------------------------------
# The other shape algebra: what the resolver refuses before anything runs
# ---------------------------------------------------------------------------
# Guard 6 (divisibility) and the SLATE/ScaLAPACK tile rules are the parts of
# the layout contract that are pure arithmetic, so they belong at this tier
# — a laptop can falsify them, and they must fire at RESOLVE time.  Bug L-1
# is the reason that "at resolve time" matters: a rule enforced only at call
# time turns a returned backend name into a broken promise, and the promise
# is the whole API.
#
# The mesh is a stand-in with the 1.5 attributes resolve_backend reads.  Not
# a convenience: a real jax Mesh at 4x4 needs sixteen devices, and the point
# of this tier is that the shape algebra needs NO machine.


class _FakeMesh:
    """``mesh.shape['x'/'y']``, ``mesh.devices.flat[0].platform``,
    ``mesh.devices.size`` — everything the resolver touches."""

    def __init__(self, px, py, platform="cpu"):
        from types import SimpleNamespace
        self.shape = {"x": px, "y": py}
        self.devices = SimpleNamespace(
            flat=[SimpleNamespace(platform=platform)], size=px * py)


@pytest.mark.parametrize("mesh_shape", [(2, 2), (4, 4)])
def test_native2d_refuses_every_hostile_logical_extent_and_takes_the_pad(
        mesh_shape):
    """The pad IS the fix, and both halves are asserted.

    For each hostile family the LOGICAL extent is refused (or tiles, when
    the family happens to be divisible on this mesh) and the PADDED extent
    tiles.  Asserting only the second would pass on a build that quietly
    rounded; asserting only the first would pass on one that refused
    everything.
    """
    from distrib_la.resolve import resolve_backend
    Px, Py = mesh_shape
    mesh = _FakeMesh(Px, Py)
    refused = 0
    for case in hostile_extents(mesh_shape):
        n_log, n_pad = case.logical[0], case.padded[0]
        try:
            resolve_backend("cholesky", "native2d", mesh, n=n_log)
        except ValueError as exc:
            refused += 1
            assert "no valid block size" in str(exc), (case.name, exc)
        # The padded extent always tiles — that is what padding is FOR.
        assert resolve_backend(
            "cholesky", "native2d", mesh, n=n_pad) == "native2d", case.name
    assert refused >= 1, (
        f"no hostile family was refused on a {Px}x{Py} mesh, so the loop "
        f"above proved nothing about refusal")


def test_both_axes_can_have_a_remainder_at_once():
    """The case a one-axis test cannot see.  ``n % Px`` and ``n % Py`` are
    two conditions, and a check that only ever violates one of them passes
    on an implementation that ANDs when it should OR."""
    from distrib_la._native2d import block_size_for
    # 4x6: lcm = 12.  n = 14 leaves a remainder against BOTH axes
    # (14 % 4 = 2, 14 % 6 = 2) and against their lcm.
    assert 14 % 4 and 14 % 6
    with pytest.raises(ValueError, match="lcm"):
        block_size_for(14, 4, 6)
    # ...and 24, a multiple of the lcm, is accepted.
    b, J = block_size_for(24, 4, 6)
    assert 24 % b == 0 and J % 4 == 0 and J % 6 == 0


def test_more_ranks_than_slices_is_refused_not_rounded():
    """``n`` smaller than the mesh: some rank owns nothing.  The kernel has
    no meaningful tiling and the honest answer is the raise — a silent
    round UP changes the matrix, a silent round DOWN drops rows."""
    from distrib_la._native2d import block_size_for
    for n, (Px, Py) in ((1, (2, 2)), (3, (4, 4)), (2, (4, 4))):
        with pytest.raises(ValueError, match="no valid block size"):
            block_size_for(n, Px, Py)


def test_an_empty_tile_row_is_representable_and_says_so():
    """``n == lcm(Px,Py)``: one tile per rank row, block size 1.  The
    smallest legal decomposition, and the boundary the refusals above sit
    just below — without it, "refuses everything small" would pass."""
    from distrib_la._native2d import block_size_for
    assert block_size_for(4, 2, 2) == (2, 2)
    assert block_size_for(2, 2, 2) == (1, 2)
    assert block_size_for(4, 4, 4) == (1, 4)


@pytest.mark.parametrize("mesh_shape", [(2, 2), (4, 4)])
def test_the_scalapack_divisibility_guard_fires_at_resolve_time(mesh_shape):
    """Guard 6 for an FFI backend, asserted WITHOUT a library.

    The capability probe runs before divisibility, so on a machine with no
    ``.so`` the refusal is the probe's.  Both are refusals at RESOLVE time
    with a reason, which is the contract; the cell asserts the ladder
    refuses rather than which rung it refused on, because the rung depends
    on the machine and the promise does not.
    """
    from distrib_la.resolve import resolve_backend
    Px, Py = mesh_shape
    mesh = _FakeMesh(Px, Py)
    for case in hostile_extents(mesh_shape):
        n_log = case.logical[0]
        if n_log % Px == 0 and n_log % Py == 0:
            continue
        with pytest.raises((ValueError, RuntimeError)) as ei:
            resolve_backend("solve_lu", "scalapack", mesh, n=n_log)
        assert str(ei.value).strip(), "a refusal with no reason"


def test_a_1d_mesh_refuses_cusolvermp_by_geometry_alone():
    """Guard 5, pure: cuSOLVERMp needs a true-2D grid and an EXPLICIT
    request refuses rather than demoting.  No GPU, no ``.so``: the
    geometry rung is reached before the capability rung on a CUDA mesh
    only when the handler is there, so this cell asserts the refusal and
    its 1x4 spelling, not which rung produced it."""
    from distrib_la.resolve import resolve_backend
    with pytest.raises((ValueError, RuntimeError)) as ei:
        resolve_backend("cholesky", "cusolvermp", _FakeMesh(1, 4, "gpu"), n=64)
    assert "cusolvermp" in str(ei.value)


def test_the_hostile_table_reproduces_the_measured_4x4_row():
    """``hostile_extents`` generalizes a table that was measured
    end-to-end on Perlmutter job 56389339 (4 nodes / 16 ranks, 4x4 mesh).
    Pinning the 4x4 instance is what makes the generalization checkable
    rather than decorative — and it is the only tie this tier has to a
    real run."""
    rows = {g.name: g.logical for g in hostile_extents((4, 4))}
    assert rows["prime-both-axes"] == (17, 23)
    assert rows["prime-both-axes-tighter"] == (13, 17)
    assert rows["nondivisible-axis0-only"] == (17, 16)
    assert rows["fewer-slices-than-ranks"] == (1, 1)
    assert rows["empty-tiles-on-axis0"] == (2, 16)
    # ...and every padded extent really is a multiple of the mesh axis,
    # which is the property the L-b and L-c tiers rely on.
    for g in hostile_extents((4, 4)):
        assert g.padded[0] % 4 == 0 and g.padded[1] % 4 == 0
        assert g.padded[0] >= g.logical[0] and g.padded[1] >= g.logical[1]


# ---------------------------------------------------------------------------
# The batched route toggle — vocabulary and the declaration it rests on
# ---------------------------------------------------------------------------
#
# ``Plan.batched`` is a ``lax.scan`` over the single-matrix op unless the
# backend owns a stacked FFI entry, and ONE thing decides which:
# ``Plan.batched_route``.  Executing either route needs a mesh and belongs
# to L-b; what belongs here is the part that must never need a machine —
# the route vocabulary, and the one declaration the scan route cannot
# discover for itself.
#
# That declaration is ``_IMPL[...]['one_handle']``.  The loop-and-stack path
# this replaced found out whether a single-matrix result was stackable by
# CALLING it and looking; a scan cannot, because a library handle is not a
# pytree and the failure arrives from inside ``lax.scan`` naming neither the
# op nor the backend.  So the table declares it — and a declaration that
# nothing checks is a comment, which is what the two cells below are for.

def _return_annotation(module_name: str, func_name: str) -> str:
    """The source-level return annotation of ``func_name``, by AST.

    Reads the wrapper's own ``.py`` and never imports it: this tier does
    not get to depend on jax being installed, let alone on a ``.so``.
    """
    import ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "distrib_la" / f"{module_name}.py")
    tree = ast.parse(src.read_text(), filename=str(src))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            assert node.returns is not None, (
                f"{module_name}.{func_name} has no return annotation, so "
                f"there is nothing here to cross-check the _IMPL "
                f"declaration against.  Annotate it.")
            return ast.unparse(node.returns)
    raise AssertionError(
        f"_IMPL names {func_name!r} in {module_name}.py and that module "
        f"has no such top-level function")


def _declares_arrays(annotation: str) -> bool:
    """Does this return annotation promise jax arrays rather than a handle?"""
    return "jax.Array" in annotation


#: (op, backend) -> (module, single-matrix entry) for every row of ``_IMPL``
#: that HAS a single-matrix entry.  Derived from the table, so a new row is
#: covered the day it is added rather than the day somebody remembers.
def _rows_with_a_single_matrix_entry():
    from distrib_la.plan import _IMPL
    module_of = {"scalapack": "_scalapack", "slate": "_slate",
                 "cusolvermp": "_cusolvermp", "native2d": "_native2d"}
    out = []
    for (op, backend), spec in _IMPL.items():
        if spec["one"] is not None:
            out.append((op, backend, module_of[backend], spec["one"],
                        bool(spec.get("one_handle"))))
    return out


@pytest.mark.parametrize(
    "op,backend,module,entry,declared_handle",
    _rows_with_a_single_matrix_entry(),
    ids=lambda v: str(v))
def test_one_handle_matches_the_wrappers_own_return_annotation(
        op, backend, module, entry, declared_handle):
    """The ``one_handle`` declaration must agree with the wrapper's code.

    ``one_handle=True`` says "this single-matrix entry returns a library
    HANDLE, so there is no array stack for a scan to build".  The wrapper
    itself already says the same thing in its return annotation
    (``-> SlateLowerL`` versus ``-> Tuple[jax.Array, jax.Array]``).  Two
    statements of one fact drift; this cell is what stops them.

    It fails if somebody sets the flag on an array-returning entry, clears
    it on a handle-returning one, or changes what a wrapper returns without
    revisiting the table — which is the case that would otherwise surface
    as a type error thrown from inside ``lax.scan``.
    """
    ann = _return_annotation(module, entry)
    assert _declares_arrays(ann) is not declared_handle, (
        f"_IMPL[({op!r}, {backend!r})] declares one_handle="
        f"{declared_handle}, but {module}.{entry} is annotated "
        f"-> {ann}")


def test_the_annotation_cross_check_can_fail():
    """RED TWIN for the cell above: it must reject both mismatches.

    Without this, ``_declares_arrays`` returning a constant would pass
    every row of the table and the cross-check would assert nothing.
    """
    assert _declares_arrays("Tuple[jax.Array, jax.Array]") is True
    assert _declares_arrays("jax.Array") is True
    assert _declares_arrays("SlateLowerL") is False
    assert _declares_arrays("CusolverMpBatchedLowerL") is False
    # ...and the reader really reads the file, so a name that is not there
    # is an error rather than a quiet default.
    with pytest.raises(AssertionError):
        _return_annotation("_slate", "no_such_entry_point")


def test_the_route_vocabulary_is_importable_and_closed():
    """The three routes are a vocabulary, like the backend names.

    Importable with no jax initialised and no ``.so`` anywhere, because a
    test (or a future deck key) must be able to name a route without
    building a mesh first.  Closed, because ``Plan.batched`` refuses an
    unknown route by listing these — a vocabulary with a hole in it would
    turn a typo into a silent different execution.
    """
    from distrib_la import (BATCHED_ROUTES, ROUTE_BACKEND_BATCHED,
                            ROUTE_BATCH_RESHARD, ROUTE_SCAN)
    assert BATCHED_ROUTES == (ROUTE_SCAN, ROUTE_BACKEND_BATCHED,
                              ROUTE_BATCH_RESHARD)
    assert len(set(BATCHED_ROUTES)) == 3
    assert all(isinstance(r, str) and r for r in BATCHED_ROUTES)


def test_the_scan_unroll_is_one_and_is_a_named_constant():
    """``unroll=1`` is the DEFAULT, deliberately, and deliberately named.

    One copy of the op in the compiled module and ``nb`` loop trips is what
    makes the batched surface compile once instead of per matrix.  Raising
    it trades module size for loop trips and nobody has measured a shape
    where that wins; the constant exists so that day is an edit in one
    place with a number behind it.
    """
    from distrib_la import BATCHED_SCAN_UNROLL
    assert BATCHED_SCAN_UNROLL == 1
    assert isinstance(BATCHED_SCAN_UNROLL, int)
