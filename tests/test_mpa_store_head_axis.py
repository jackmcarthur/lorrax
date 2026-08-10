"""The fit store's q -> 0 head axis, and the pole-sliced read beside it.

The head axis is format version 2's only addition, and the reason it is a
version rather than an optional dataset is the refusal: a Sigma built from a
headless store is finite, smooth and missing the q -> 0 term at every
frequency.  Every cell here is about making that state legible.
"""

import numpy as np
import pytest

from file_io import mpa_store


def _store(tmp_path, n_p=3, n_q=2, n_mu=4, seed=0):
    path = str(tmp_path / "fit.h5")
    mpa_store.allocate_fit_store(path, n_q=n_q, n_mu=n_mu, n_p=n_p,
                                 energy_unit="Ry", screening_content="W_c")
    rng = np.random.default_rng(seed)
    for q in range(n_q):
        cols = list(range(n_mu))
        # Fourth-quadrant poles, ascending in Re Omega along the leading axis,
        # exactly as pade_fit returns them.
        a = np.sort(rng.uniform(0.2, 3.0, size=(n_p, n_mu, len(cols))), axis=0)
        g = rng.uniform(0.01, 0.5, size=a.shape)
        Om = a - 1j * g
        Bp = rng.normal(size=Om.shape) + 1j * rng.normal(size=Om.shape)
        mpa_store.write_fit_block(
            path, q, cols, Om, Bp,
            {"condition": np.ones((n_mu, len(cols))),
             "backward_error": np.full((n_mu, len(cols)), 1e-12)})
    mpa_store.finalize_fit_store(path)
    return path


# ---------------------------------------------------------------------------
#  The pole-sliced read, now in the store
# ---------------------------------------------------------------------------

def test_read_pole_slice_is_the_leading_axis_slab(tmp_path):
    path = _store(tmp_path)
    Om_all, B_all, _, ledger = mpa_store.read_fit_tensors(path)
    for p in range(ledger["n_p"]):
        Om, Bp = mpa_store.read_pole_slice(path, p)
        assert Om.shape == Om_all.shape[1:]
        assert np.array_equal(Om, Om_all[p])
        assert np.array_equal(Bp, B_all[p])


def test_read_pole_slice_refuses_an_unfinalized_store(tmp_path):
    """A store with no fit data in it is refused, not served as zeros.

    TWO GUARDS NOW STAND HERE, AND THE CELL PINS THE PROPERTY RATHER THAN
    WHICH ONE WINS.  This cell used to match "NOT FINALIZED", the
    finalize-flag refusal.  The wedge lane
    (feat/mpa-wedge-pole-unfold-2026-08-10) added a second, stricter one:
    the q_done completeness check, which names WHICH q have no data
    instead of reporting a flag.  An empty store trips both, and which
    fires first depends on the ledger's ``complete`` reading -- observed
    both ways on this fixture between a bare interpreter and a pytest
    session, which is a difference worth its own row and not worth a cell
    pinning one answer.

    What must never change, and is what is asserted: a freshly allocated
    store is REFUSED, not served as a field of zero poles.  A zero pole is
    not an absent pole -- it looks exactly like a screening channel that
    is genuinely dark -- so serving one silently is the failure this cell
    exists to forbid, and either message forbids it.

    THE ESCAPE HATCH MOVED, and that is the second half of this cell.
    ``allow_partial=True`` used to serve the holes; the wedge lane split
    the two questions deliberately, because they are not the same
    question.  ``allow_partial`` says "this file is mid-fit and I know
    it"; it does NOT say "serve me q that contain no data", because a
    resumed farm fit is mid-file by design and must still never integrate
    a hole.  ``raw=True`` is now the deliberate-holes escape.  Asserted
    here so the split is pinned rather than inferred.
    """
    path = str(tmp_path / "partial.h5")
    mpa_store.allocate_fit_store(path, n_q=1, n_mu=2, n_p=2,
                                 energy_unit="Ry", screening_content="W_c")
    with pytest.raises(ValueError, match="NOT FINALIZED|NO fit data"):
        mpa_store.read_pole_slice(path, 0)
    # allow_partial alone is NOT enough once a q is empty -- it answers the
    # finalize question, not the missing-data one.
    with pytest.raises(ValueError, match="NO fit data"):
        mpa_store.read_pole_slice(path, 0, allow_partial=True)
    Om, Bp = mpa_store.read_pole_slice(path, 0, allow_partial=True, raw=True)
    assert Om.shape == (1, 2, 2)


def test_read_pole_slice_names_the_pass_order_on_a_bad_index(tmp_path):
    path = _store(tmp_path)
    with pytest.raises(IndexError, match="pole_pass_order"):
        mpa_store.read_pole_slice(path, 99)


def test_fit_driver_re_exports_the_moved_reader(tmp_path):
    from gw.mpa import fit_driver
    path = _store(tmp_path)
    a = fit_driver.read_pole_slice(path, 1)
    b = mpa_store.read_pole_slice(path, 1)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ---------------------------------------------------------------------------
#  THE HEAD-AXIS RED TWIN
# ---------------------------------------------------------------------------

def test_a_store_without_the_head_axis_refuses_by_name(tmp_path):
    """The whole point of the version.  The store is FINE -- and refused."""
    path = _store(tmp_path)
    # It reads back perfectly for everything that does not need the head.
    assert mpa_store.read_pole_slice(path, 0)[0].shape == (2, 4, 4)
    with pytest.raises(ValueError) as exc:
        mpa_store.read_head_poles(path)
    msg = str(exc.value)
    assert "q -> 0 head" in msg
    assert "REFUSES it by name" in msg
    assert "finite, smooth and wrong" in msg
    # and it points at both ways forward, not just at the failure
    assert "allocate_head_axis" in msg and "gn_ppm" in msg


def test_an_allocated_but_unwritten_head_refuses_as_zeros_not_as_absence(
        tmp_path):
    path = _store(tmp_path, n_p=3)
    mpa_store.allocate_head_axis(path, n_p=3)
    with pytest.raises(ValueError, match="not stamped ready"):
        mpa_store.read_head_poles(path)


def test_the_head_axis_is_two_n_p_samples_and_n_p_poles(tmp_path):
    path = _store(tmp_path, n_p=3)
    mpa_store.allocate_head_axis(path, n_p=3)
    z = np.array([1 + 1j, 2 + 1j, 3 + 1j, 1 + 4j, 2 + 4j, 3 + 4j])
    w = np.arange(6) + 0.5j
    Om = np.array([1.0 - 0.1j, 2.0 - 0.2j, 3.0 - 0.3j])
    B = np.array([0.5 + 0j, 0.25 + 0j, 0.125 + 0j])
    mpa_store.write_head_axis(path, z, w, Om, B, vhead=-3.25 + 0j)
    got = mpa_store.read_head_poles(path)
    assert got["n_p"] == 3
    assert got["z"].shape == (6,) and got["w"].shape == (6,)
    assert got["Omega_p"].shape == (3,) and got["B_p"].shape == (3,)
    assert np.array_equal(got["z"], z) and np.array_equal(got["Omega_p"], Om)
    assert complex(got["vhead"]) == -3.25 + 0j


def test_the_head_sample_axis_must_be_two_n_p(tmp_path):
    """A head built on a different grid from the body is the silent failure."""
    path = _store(tmp_path, n_p=3)
    mpa_store.allocate_head_axis(path, n_p=3)
    with pytest.raises(ValueError, match="sample axis is 2\\*n_p"):
        mpa_store.write_head_axis(
            path, np.zeros(4), np.zeros(6),
            np.ones(3) - 0.1j, np.ones(3))


def test_a_head_pole_in_the_wrong_half_plane_refuses(tmp_path):
    """``Im Omega > 0`` enters the tau stage as growth; the head owes the
    body's own quadrant guard."""
    path = _store(tmp_path, n_p=2)
    mpa_store.allocate_head_axis(path, n_p=2)
    with pytest.raises(ValueError, match="Im Omega > 0"):
        mpa_store.write_head_axis(
            path, np.zeros(4), np.zeros(4),
            np.array([1.0 - 0.1j, 2.0 + 0.3j]), np.ones(2))


def test_the_head_axis_refuses_a_second_stamp(tmp_path):
    path = _store(tmp_path, n_p=2)
    mpa_store.allocate_head_axis(path, n_p=2)
    args = (np.zeros(4), np.zeros(4), np.array([1.0 - 0.1j, 2.0 - 0.2j]),
            np.ones(2))
    mpa_store.write_head_axis(path, *args)
    with pytest.raises(ValueError, match="already stamped ready"):
        mpa_store.write_head_axis(path, *args)


def test_writing_the_head_without_allocating_refuses(tmp_path):
    path = _store(tmp_path, n_p=2)
    with pytest.raises(ValueError, match="allocate_head_axis"):
        mpa_store.write_head_axis(
            path, np.zeros(4), np.zeros(4),
            np.array([1.0 - 0.1j, 2.0 - 0.2j]), np.ones(2))


# ---------------------------------------------------------------------------
#  Format-version discipline
# ---------------------------------------------------------------------------

def test_version_two_is_written_and_version_one_still_reads(tmp_path):
    """v1 stores keep working -- the first-light field is one of them."""
    path = _store(tmp_path)
    assert mpa_store.MPA_FIT_FORMAT_VERSION == 2
    assert 1 in mpa_store.MPA_FIT_READABLE_VERSIONS
    led = mpa_store.fit_completion_ledger(path)
    assert led["format_version"] == 2

    import h5py
    with h5py.File(path, "a") as f:
        f.attrs["mpa_fit_format_version"] = np.int64(1)
    # A v1 store reads its poles and refuses only the head.
    assert mpa_store.read_pole_slice(path, 0)[0].shape == (2, 4, 4)
    with pytest.raises(ValueError, match="format version 1"):
        mpa_store.read_head_poles(path)


def test_an_unknown_format_version_still_refuses(tmp_path):
    path = _store(tmp_path)
    import h5py
    with h5py.File(path, "a") as f:
        f.attrs["mpa_fit_format_version"] = np.int64(99)
    with pytest.raises(ValueError, match="format version 99"):
        mpa_store.fit_completion_ledger(path)


# ---------------------------------------------------------------------------
#  LABELLED head sets -- the two sign conventions living side by side
# ---------------------------------------------------------------------------

def _head_values(n_p, scale=1.0):
    """A well-formed head set: 2*n_p samples and n_p fourth-quadrant poles."""
    z = np.concatenate([np.linspace(0.0, 2.0, n_p),
                        1.0j + np.linspace(0.0, 2.0, n_p)])
    w = scale * (np.arange(2 * n_p) + 1.0) * (1.0 + 0.5j)
    Om = scale * (np.arange(1, n_p + 1) - 0.1j)
    B = scale * (np.arange(1, n_p + 1) + 0.25j)
    return z, w, Om, B


def test_the_default_label_is_the_group_this_format_already_had(tmp_path):
    """Labels must not migrate a store that has one head.

    The FALSE case is a format that renamed the group when labels arrived:
    every v2 store written before today would then read as headless, and
    ``read_head_poles`` would refuse a file whose head is right there.
    """
    assert mpa_store.head_group_name(None) == mpa_store.MPA_HEAD_SUFFIX
    assert (mpa_store.head_group_name(mpa_store.MPA_HEAD_DEFAULT_LABEL)
            == mpa_store.MPA_HEAD_SUFFIX)
    assert (mpa_store.head_group_name("commutator_flipped")
            == mpa_store.MPA_HEAD_SUFFIX + "__commutator_flipped")

    path = _store(tmp_path, n_p=3)
    z, w, Om, B = _head_values(3)
    mpa_store.allocate_head_axis(path, n_p=3)
    mpa_store.write_head_axis(path, z, w, Om, B, vhead=3303.7)
    # Written with no label, read back with no label AND with the default
    # label spelled out: the two must be the same object.
    a = mpa_store.read_head_poles(path)
    b = mpa_store.read_head_poles(path, label="as_shipped")
    assert a["group"] == b["group"] == mpa_store.MPA_HEAD_SUFFIX
    assert np.array_equal(a["Omega_p"], b["Omega_p"])


def test_two_sign_conventions_coexist_and_are_told_apart(tmp_path):
    """Both head sets in one store, each answering under its own name.

    This is the whole point of the label: the owner's decision on the
    nonlocal velocity commutator's sign is pending, so the table carries
    both columns and neither convention is chosen by the file format.
    """
    n_p = 3
    path = _store(tmp_path, n_p=n_p)
    z, w, Om, B = _head_values(n_p, scale=1.0)
    z2, w2, Om2, B2 = _head_values(n_p, scale=1.377)

    mpa_store.allocate_head_axis(path, n_p=n_p)
    mpa_store.write_head_axis(path, z, w, Om, B, vhead=3303.7,
                              provenance={"velocity_sign": "as shipped"})
    mpa_store.allocate_head_axis(path, n_p=n_p, label="commutator_flipped")
    mpa_store.write_head_axis(path, z2, w2, Om2, B2, vhead=3303.7,
                              label="commutator_flipped",
                              provenance={"velocity_sign": "flipped"})

    shipped = mpa_store.read_head_poles(path)
    flipped = mpa_store.read_head_poles(path, label="commutator_flipped")

    assert shipped["label"] == "as_shipped"
    assert flipped["label"] == "commutator_flipped"
    # The FALSE case this cell exists for: a label that is parsed, stored
    # and never read would return the SAME numbers for both, which is
    # exactly the "a key can be present and never read" failure the project
    # has paid for twice.  So the two must actually differ.
    assert not np.allclose(shipped["Omega_p"], flipped["Omega_p"])
    assert np.allclose(flipped["Omega_p"], 1.377 * shipped["Omega_p"])
    assert not np.allclose(shipped["w"], flipped["w"])


def test_a_missing_label_refuses_and_names_what_the_store_does_have(tmp_path):
    """Asking for a head set that was never written must not fall back.

    Falling back to the default would hand a Sigma the OTHER sign
    convention under the name of the one that was asked for -- finite,
    smooth, and attributed to the wrong physics.
    """
    path = _store(tmp_path, n_p=3)
    z, w, Om, B = _head_values(3)
    mpa_store.allocate_head_axis(path, n_p=3)
    mpa_store.write_head_axis(path, z, w, Om, B)
    with pytest.raises(ValueError) as exc:
        mpa_store.read_head_poles(path, label="commutator_flipped")
    msg = str(exc.value)
    assert "__mpahead__commutator_flipped" in msg
    # and it says what IS there, so the operator can pick a real one
    assert "__mpahead" in msg


def test_an_unusable_label_is_refused_before_it_becomes_a_group(tmp_path):
    for bad in ("", " leading_space", "with/slash"):
        with pytest.raises(ValueError):
            mpa_store.head_group_name(bad)
