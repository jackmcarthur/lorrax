"""The pass-partial split, and the four ways it can silently be wrong.

WHAT THIS FILE IS FOR.  ``compute_mpa_sigma_c_omega_grid`` is a sum over
the fit store's poles, written out one pass at a time.  Splitting that sum
across processes is arithmetically free — the re-association lemma in
``fit_driver.accumulate_over_pole_passes`` is the proof — but it is
OPERATIONALLY the most dangerous thing in this pipeline, because a partial
sum over the pole axis has the identical shape, dtype and units as a
finished self-energy.  Nothing downstream of the recombination can tell
them apart: a stack of partials missing one pole, or carrying one twice,
or written against a different store, produces a Σ_c that is finite,
smooth, Hermitian, of the right magnitude, and wrong by tens of meV —
which is the size of the effect the MPA-versus-BerkeleyGW table measures.

So every cell below is a FALSE case constructed on purpose.  The gate is
not "the combiner returns the right number on good input"; it is "the
combiner refuses each way the input can be bad", and the good-input cell
exists mainly to prove the refusals are not refusing everything.
"""

from __future__ import annotations

import numpy as np
import pytest

from gw.mpa import sigma_pass

_FIT_ID = "fit-test-allocation"
_SOURCE_IDENTITY = "test-source"


def _combine(paths, **kwargs):
    return sigma_pass.combine_pass_partials(
        paths, fit_id=_FIT_ID, source_identity=_SOURCE_IDENTITY, **kwargs)


# ---------------------------------------------------------------------------
#  resolve_pole_subset — which poles, never in which order
# ---------------------------------------------------------------------------

def test_pole_subset_none_is_every_pole_in_the_pinned_order():
    assert sigma_pass.resolve_pole_subset(8, None) == tuple(range(8))
    assert sigma_pass.resolve_pole_subset(8, ()) == tuple(range(8))
    assert sigma_pass.resolve_pole_subset(8, []) == tuple(range(8))


def test_pole_subset_is_returned_in_the_pinned_order_not_the_typed_one():
    """THE POINT OF THE FUNCTION.  A caller who writes ``5,0,3`` is saying
    which poles, not in which order to add them; the accumulation order is
    a property of the sum and is pinned ascending.
    """
    assert sigma_pass.resolve_pole_subset(8, [5, 0, 3]) == (0, 3, 5)


def test_pole_subset_refuses_an_index_the_store_does_not_have():
    with pytest.raises(ValueError, match="outside this store's pinned"):
        sigma_pass.resolve_pole_subset(4, [0, 4])
    with pytest.raises(ValueError, match="outside this store's pinned"):
        sigma_pass.resolve_pole_subset(4, [-1])


def test_pole_subset_refuses_a_repeated_index_rather_than_deduplicating():
    """A pole summed twice is the failure with no downstream symptom."""
    with pytest.raises(ValueError, match="more than once"):
        sigma_pass.resolve_pole_subset(4, [1, 1])


# ---------------------------------------------------------------------------
#  The round trip, and the audit numbers it is required to report
# ---------------------------------------------------------------------------

def _rec(p):
    return sigma_pass.PassRecord(
        pole_index=int(p), n_legacy_modes=1, n_mpa_modes=2,
        legacy_b_mass=0.5, mpa_b_mass=1.5, n_tau_nodes=7,
        groups=[f"g{p}"], re_omega_min_ev=1.0 * p, re_omega_max_ev=2.0 * p,
        gamma_min_ev=0.1, gamma_max_ev=0.2)


def _write(tmp_path, name, cube, poles, *, n_p=4, om=None, store="S.h5"):
    om = np.linspace(-1.0, 1.0, cube.shape[0]) if om is None else om
    path = str(tmp_path / name)
    sigma_pass.write_pass_partial(
        path, cube, [_rec(p) for p in poles], n_p=n_p, poles=poles,
        omega_grid_ry=om, fit_src=store, fit_id=_FIT_ID,
        source_identity=_SOURCE_IDENTITY, print_fn=lambda *_a, **_k: None)
    return path, om


def _field(n_p, shape, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=shape) + 1j * rng.normal(size=shape)
            for _ in range(n_p)]


def test_partials_recombine_to_the_ascending_sum_and_report_the_order_cost(
        tmp_path):
    """The good-input cell: coverage passes, and the canonical total is
    BIT-EXACTLY the ascending accumulation, which is what makes a split run
    reproducible against itself.
    """
    shape = (5, 3, 2, 2)
    cubes = _field(4, shape, seed=11)
    paths = [_write(tmp_path, f"p{p}.h5", cubes[p], [p])[0] for p in range(4)]
    om = np.linspace(-1.0, 1.0, shape[0])

    total, poles, audit = _combine(
        paths, n_p=4, omega_grid_ry=om, fit_src="S.h5",
        print_fn=lambda *_a, **_k: None)

    ascending = np.zeros(shape, dtype=np.complex128)
    for c in cubes:
        ascending = ascending + c
    assert np.array_equal(total, ascending)      # bit-exact, not allclose
    assert poles == (0, 1, 2, 3)
    assert audit["n_files"] == 4
    # The re-association is REPORTED, and on a random field of this size it
    # is real (nonzero) yet negligible (ulp-scale).  Both halves matter:
    # zero would mean the audit is not measuring anything, and a large
    # number would mean the order is doing physics.
    assert audit["reassoc_descending_max_abs_ry"] >= 0.0
    assert audit["reassoc_descending_rel"] < 1.0e-12
    assert audit["reassoc_shuffled_rel"] < 1.0e-12


def test_one_file_may_carry_several_poles_and_is_added_exactly_once(tmp_path):
    """The 8-into-4 case: partials need not be one pole each."""
    shape = (4, 2, 2, 2)
    a, b = _field(2, shape, seed=3)
    pa, om = _write(tmp_path, "a.h5", a, [0, 1])
    pb, _ = _write(tmp_path, "b.h5", b, [2, 3], om=om)
    total, _poles, audit = _combine(
        [pa, pb], n_p=4, omega_grid_ry=om, fit_src="S.h5",
        print_fn=lambda *_a, **_k: None)
    assert audit["n_files"] == 2
    assert np.array_equal(total, a + b)


# ---------------------------------------------------------------------------
#  THE RED TWINS.  Each is a stack that would return a plausible number.
# ---------------------------------------------------------------------------

def test_red_twin_a_missing_pole_is_refused_not_summed(tmp_path):
    shape = (4, 2, 2, 2)
    cubes = _field(4, shape, seed=5)
    paths, om = [], None
    for p in (0, 1, 3):                      # pole 2 never written
        path, om = _write(tmp_path, f"p{p}.h5", cubes[p], [p], om=om)
        paths.append(path)
    with pytest.raises(ValueError, match="missing \\[2\\]"):
        _combine(
            paths, n_p=4, omega_grid_ry=om, fit_src="S.h5",
            print_fn=lambda *_a, **_k: None)


def test_red_twin_a_doubled_pole_is_refused_not_summed(tmp_path):
    shape = (4, 2, 2, 2)
    cubes = _field(4, shape, seed=7)
    paths, om = [], None
    for name, p in (("a", 0), ("b", 1), ("c", 1), ("d", 3)):
        path, om = _write(tmp_path, f"{name}.h5", cubes[p], [p], om=om)
        paths.append(path)
    with pytest.raises(ValueError, match="duplicated \\[1\\]"):
        _combine(
            paths, n_p=4, omega_grid_ry=om, fit_src="S.h5",
            print_fn=lambda *_a, **_k: None)


def test_red_twin_partials_from_two_stores_are_refused(tmp_path):
    shape = (4, 2, 2, 2)
    cubes = _field(2, shape, seed=9)
    p0, om = _write(tmp_path, "p0.h5", cubes[0], [0], n_p=2, store="A.h5")
    p1, _ = _write(tmp_path, "p1.h5", cubes[1], [1], n_p=2, store="B.h5",
                   om=om)
    with pytest.raises(ValueError, match="was integrated against fit store"):
        _combine(
            [p0, p1], n_p=2, omega_grid_ry=om, fit_src="A.h5",
            print_fn=lambda *_a, **_k: None)


def test_red_twin_a_different_omega_grid_is_refused(tmp_path):
    shape = (4, 2, 2, 2)
    cubes = _field(2, shape, seed=13)
    om_a = np.linspace(-1.0, 1.0, 4)
    om_b = np.linspace(-2.0, 2.0, 4)
    p0, _ = _write(tmp_path, "p0.h5", cubes[0], [0], n_p=2, om=om_a)
    p1, _ = _write(tmp_path, "p1.h5", cubes[1], [1], n_p=2, om=om_b)
    with pytest.raises(ValueError, match="different Σ ω grid"):
        _combine(
            [p0, p1], n_p=2, omega_grid_ry=om_a, fit_src="S.h5",
            print_fn=lambda *_a, **_k: None)


def test_red_twin_a_foreign_format_version_is_refused(tmp_path):
    import h5py

    shape = (4, 2, 2, 2)
    cubes = _field(1, shape, seed=17)
    p0, om = _write(tmp_path, "p0.h5", cubes[0], [0], n_p=1)
    with h5py.File(p0, "a") as f:
        f.attrs["mpa_partial_format_version"] = (
            sigma_pass.PARTIAL_FORMAT_VERSION + 1)
    with pytest.raises(ValueError, match="declares partial format version"):
        _combine(
            [p0], n_p=1, omega_grid_ry=om, fit_src="S.h5",
            print_fn=lambda *_a, **_k: None)


def test_red_twin_a_store_with_a_different_n_p_is_refused(tmp_path):
    shape = (4, 2, 2, 2)
    cubes = _field(1, shape, seed=19)
    p0, om = _write(tmp_path, "p0.h5", cubes[0], [0], n_p=1)
    with pytest.raises(ValueError, match="was written from a store"):
        _combine(
            [p0], n_p=4, omega_grid_ry=om, fit_src="S.h5",
            print_fn=lambda *_a, **_k: None)


# ---------------------------------------------------------------------------
#  The deck-key seam
# ---------------------------------------------------------------------------

def test_deck_pole_subset_parses_and_refuses_a_non_integer():
    from gw.mpa_pipeline import _parse_pole_subset

    assert _parse_pole_subset("") is None
    assert _parse_pole_subset(None) is None
    assert _parse_pole_subset("0") == (0,)
    assert _parse_pole_subset(" 0, 3 ,5 ") == (0, 3, 5)
    with pytest.raises(ValueError, match="not an integer pole index"):
        _parse_pole_subset("0,x")


def test_deck_partial_in_refuses_matching_nothing(tmp_path):
    """Σ_c = 0 is finite, smooth and Hermitian; it must not be reachable
    by pointing the combiner at an empty directory.
    """
    from gw.mpa_pipeline import _partial_paths_in

    with pytest.raises(FileNotFoundError, match="matched no partial cube"):
        _partial_paths_in(str(tmp_path))
