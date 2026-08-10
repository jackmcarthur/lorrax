"""The deterministic candidate pool — pure host algebra, no device needed.

These are the properties the ``--candidate-pool full_grid`` selector rests
on: the pool is a pure function of the grid, the orbit labels are a pure
function of the grid and the symmetry table, and a symmetry table that does
not map the grid to itself is REFUSED rather than rounded onto it.  The
device-side gate (two full runs producing byte-identical centroid files)
lives with the run; this is the part that can be checked in a second.
"""
import ast
import pathlib

import numpy as np
import pytest

from centroid import grid_pool


GRID = (4, 4, 6)
IDENT = np.eye(3, dtype=np.int64)
INVERT = -np.eye(3, dtype=np.int64)


def test_pool_is_the_whole_grid_in_c_order():
    cand = grid_pool.full_grid_candidates(GRID)
    assert cand.shape == (int(np.prod(GRID)), 3)
    # Row m is the point whose flat C-order index is m — that is the whole
    # ordering contract, and everything downstream inherits it.
    flat = (cand[:, 0] * GRID[1] + cand[:, 1]) * GRID[2] + cand[:, 2]
    assert np.array_equal(flat, np.arange(flat.size))


def test_pool_and_labels_are_bit_reproducible():
    Rinv = np.stack([IDENT, INVERT])
    tau = np.zeros((2, 3))
    a = grid_pool.full_grid_candidates(GRID)
    b = grid_pool.full_grid_candidates(GRID)
    assert np.array_equal(a, b)
    ida, _ = grid_pool.grid_orbit_ids(a, Rinv, tau, GRID)
    idb, _ = grid_pool.grid_orbit_ids(b, Rinv, tau, GRID)
    assert np.array_equal(ida, idb)


def test_orbits_partition_the_grid_and_are_closed():
    Rinv = np.stack([IDENT, INVERT])
    tau = np.zeros((2, 3))
    cand = grid_pool.full_grid_candidates(GRID)
    orbit_id, sizes = grid_pool.grid_orbit_ids(cand, Rinv, tau, GRID)
    assert int(sizes.sum()) == cand.shape[0]
    assert np.array_equal(np.bincount(orbit_id), sizes)
    # Under inversion on an even grid the fixed points are the eight
    # points whose every coordinate is 0 or n/2; everything else pairs up.
    assert set(np.unique(sizes).tolist()) == {1, 2}
    assert int((sizes == 1).sum()) == 8
    # Closure: the image of a pool point carries the same label.
    images = grid_pool.grid_images(cand, Rinv, tau, GRID)
    flat_img = ((images[..., 0] * GRID[1] + images[..., 1]) * GRID[2]
                + images[..., 2])
    for s in range(images.shape[0]):
        assert np.array_equal(orbit_id[flat_img[s]], orbit_id)


def test_labels_are_assigned_in_canonical_index_order():
    """Label k belongs to the orbit whose lex-smallest member comes k-th.

    This is what makes the labelling independent of how the pool was
    enumerated, and therefore quotable as "the same set every run".
    """
    Rinv = np.stack([IDENT, INVERT])
    tau = np.zeros((2, 3))
    cand = grid_pool.full_grid_candidates(GRID)
    orbit_id, _ = grid_pool.grid_orbit_ids(cand, Rinv, tau, GRID)
    first_seen = [int(np.flatnonzero(orbit_id == k)[0])
                  for k in range(orbit_id.max() + 1)]
    assert first_seen == sorted(first_seen)


def test_a_grid_the_group_does_not_map_to_itself_is_refused():
    """An incommensurate fractional translation must not be rounded away.

    Rounding it produces a point set that is not orbit-closed while still
    having the right shape and count — the failure mode that a downstream
    shape check cannot see.
    """
    Rinv = np.stack([IDENT, INVERT])
    tau = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    cand = grid_pool.full_grid_candidates(GRID)
    with pytest.raises(ValueError, match="do not land on the FFT grid"):
        grid_pool.grid_orbit_ids(cand, Rinv, tau, GRID)


def test_a_commensurate_glide_is_accepted():
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.5]])  # x4, x4, x6 integral
    Rinv = np.stack([IDENT, INVERT])
    cand = grid_pool.full_grid_candidates(GRID)
    orbit_id, sizes = grid_pool.grid_orbit_ids(cand, Rinv, tau, GRID)
    assert int(sizes.sum()) == cand.shape[0]


def test_orbit_block_pivots_deliver_a_closed_full_rank_set():
    """``orbit_block`` consumes an orbit entirely before opening the next.

    That is the property that makes the DELIVERED (orbit-closed) set
    full-rank.  One-pivot-per-orbit certifies one direction per orbit and
    then delivers all its members, so most delivered points are symmetry
    images whose independence was never checked — measured on Si 4x4x4 at
    24^3 as 1676 points spanning 801 directions.  Here, on a random PSD
    Gram with six orbits of four, the two modes are asked for the same
    twelve pivots and only one of them can certify twelve.
    """
    import jax.numpy as jnp
    from centroid.pivoted_cholesky import pivoted_cholesky_select

    rng = np.random.default_rng(0)
    m, n_orb, orb_size = 24, 6, 4
    a = rng.normal(size=(40, m)) + 1j * rng.normal(size=(40, m))
    gram = jnp.asarray(a.conj().T @ a)
    gram = 0.5 * (gram + gram.conj().T)
    orbit_id = jnp.asarray(np.repeat(np.arange(n_orb), orb_size)
                           .astype(np.int32))

    piv, _, rank, *_ = pivoted_cholesky_select(
        gram, 12, orbit_id, orbit_block=True)
    piv = np.asarray(piv)
    assert int(rank) == 12                       # every pivot a direction
    assert len(set(piv.tolist())) == piv.size    # no pivot taken twice
    counts = np.bincount(np.asarray(orbit_id)[piv], minlength=n_orb)
    # Orbits are consumed whole: every touched orbit is complete.
    assert set(counts[counts > 0].tolist()) == {orb_size}
    # ...and contiguously, so at most the LAST orbit could be partial.
    seq = np.asarray(orbit_id)[piv]
    assert (np.diff(np.flatnonzero(np.r_[True, seq[1:] != seq[:-1]]))
            == orb_size).all()

    # One pivot per orbit cannot do better than one direction per orbit.
    _, _, rank_orbit, *_ = pivoted_cholesky_select(gram, 12, orbit_id)
    assert int(rank_orbit) == n_orb < 12


# ─────────────────────────────────────────────────────────────────────────
# The selector option, end to end on the host: flag -> stamp -> deck key
# ─────────────────────────────────────────────────────────────────────────

def _stamped(tmp_path, stamp):
    """A centroid file with the header `kmeans_cli` writes, minus the table."""
    p = tmp_path / "centroids_frac_4.txt"
    p.write_text(
        "# x y z (snapped to FFT grid (4, 4, 4), 4 unique)\n"
        f"# centroid_source: {stamp}\n"
        "# determinism: no RNG on this path\n"
        "0.000000 0.000000 0.000000\n0.250000 0.250000 0.250000\n")
    return str(p)


def _cli_flag(name):
    """The `add_argument` call for one flag, read by AST.

    The CLI module cannot be IMPORTED here: it opens with
    `initialize_communicator_stack()`, which needs the built FFI, so a plain
    import turns this host-only test into a cluster-only one.  The parser is
    a literal in the source, so read it as one — the same no-import AST
    approach `tools/gen_input_reference.py` uses on the deck defaults, and
    for the same reason.
    """
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "src" / "centroid" / "kmeans_cli.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == name):
            return {k.arg: k.value for k in node.keywords}
    raise AssertionError(f"{name} is not in the parser")


def test_selector_flag_round_trips_from_cli_to_deck_assertion(tmp_path):
    """`--centroid-selector X` -> `centroid_source:` -> `centroid_selector = X`.

    The three names have to agree across two modules and a text file, and
    nothing else checks that: a typo in the writer's stamp or the reader's
    table would make the deck assertion quietly unsatisfiable — or worse,
    quietly satisfiable by the wrong selector, which is a wrong-basis run
    that no other gate would catch.
    """
    from file_io.centroids import (
        CENTROID_SOURCE_STAMPS, assert_centroid_selector, read_centroid_source,
    )

    choices = _cli_flag("--centroid-selector")["choices"]
    assert {e.value for e in choices.elts} == set(CENTROID_SOURCE_STAMPS)

    for selector, stamp in CENTROID_SOURCE_STAMPS.items():
        f = _stamped(tmp_path, stamp)
        assert read_centroid_source(f) == stamp
        assert_centroid_selector(f, selector, print_fn=lambda *_: None)
        # ...and the OTHER selector's deck key must not accept this file.
        other = next(k for k in CENTROID_SOURCE_STAMPS if k != selector)
        with pytest.raises(ValueError, match="was written by"):
            assert_centroid_selector(f, other, print_fn=lambda *_: None)


def test_the_writer_takes_its_stamp_from_the_shared_table():
    """No stamp literal outside `file_io.centroids`.

    The whole point of the shared table is that writer and reader cannot
    drift; a hand-written stamp string in the generator would restore
    exactly the drift the table removes, while still passing every test
    above on the day it was written.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    from file_io.centroids import CENTROID_SOURCE_STAMPS

    owner = root / "src" / "file_io" / "centroids.py"
    for stamp in CENTROID_SOURCE_STAMPS.values():
        holders = {py for py in (root / "src").rglob("*.py")
                   if f'"{stamp}"' in py.read_text()
                   or f"'{stamp}'" in py.read_text()}
        assert holders <= {owner}, (
            f"{stamp!r} is written out in {sorted(str(h) for h in holders)}; "
            f"it belongs only in {owner}")


def test_the_default_selector_is_the_historical_one():
    """Defaulting to current behaviour is the whole contract for existing decks."""
    assert _cli_flag("--centroid-selector")["default"].value == "kmeans"
    # Granularity is resolved FROM the selector in main(), so the parser
    # default must stay None — a literal here would silently pin the
    # whole-grid selector to the granularity measured WORSE on it.
    assert _cli_flag("--pivot-granularity")["default"].value is None


def test_an_empty_deck_key_asserts_nothing_and_an_unstamped_file_refuses(tmp_path):
    from file_io.centroids import assert_centroid_selector

    bare = tmp_path / "legacy.txt"
    bare.write_text("0.000000 0.000000 0.000000\n")
    # Every deck that predates the stamp: no key, no assertion, no refusal.
    assert_centroid_selector(str(bare), "", print_fn=lambda *_: None)
    # But a deck that DOES claim a selector may not be satisfied by silence.
    with pytest.raises(ValueError, match="no `centroid_source:` stamp"):
        assert_centroid_selector(str(bare), "kmeans", print_fn=lambda *_: None)
    with pytest.raises(ValueError, match="is not a selector"):
        assert_centroid_selector(_stamped(tmp_path, "kmeans_pivoted_cholesky"),
                                 "typo", print_fn=lambda *_: None)
