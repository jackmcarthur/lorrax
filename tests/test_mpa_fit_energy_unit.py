"""The fit store's pole-axis unit: declared once, converted once, refused loud.

THE DEFECT THIS RETIRES, so the cells below read as its negations.  The
first end-to-end MPA Sigma dispatch fed Hartree poles into a Rydberg
window planner: the fit was solved against the W store's abscissae
(stamped ``mpa_omega_units = "Ha"``) while ``sigma_pass`` read
``Re Omega_p`` straight into ``plan_branch_groups(a_ry=...)`` beside Ry
band energies.  Every pole entered Sigma at half its energy, the width
split and the Laplace buckets were mis-sized from the same numbers, and
NO internal gate could see it -- the model is invariant under rescaling
z, Omega and B together, so only an external oracle can: the n_p = 1
head pole reads 18.118 eV as Ha against BerkeleyGW's own 18.009 eV, and
9.06 eV as Ry against a 16.7 eV measured plasmon.

THE SHAPE OF THE FIX: the WRITER declares (``allocate_fit_store``
requires ``energy_unit``; the fit driver inherits it from the W store's
own stamp), the READERS convert to Ry at one seam
(``read_pole_slice`` / ``read_fit_block`` / ``read_fit_tensors``), an
UNDECLARED store is refused by name with both fixes stated, and the one
legacy escape (``declare_fit_energy_unit``) stamps exactly once.

THE TWIN GATE.  The table-closer verified the conversion on the real
store by building a Ry-rescaled twin at an exact elementwise ratio of
2.000000; the first cell below is that gate in miniature -- a Ha store
and its ``x2`` Ry twin must read back IDENTICALLY, and ``x2`` is exact
in binary so the assertion is ``array_equal``, not ``allclose``.
"""

from __future__ import annotations

import numpy as np
import pytest

from file_io import mpa_store


def _filled_store(path, *, energy_unit, scale=1.0, n_p=2, n_q=3, n_mu=4,
                  declare=True, seed=11):
    """A finalized synthetic store whose pole field is ``scale x`` the base.

    ``declare=False`` builds the LEGACY shape -- allocated through the
    current writer (which refuses undeclared), then the declaration attr
    is DELETED with h5py, because that is exactly what a store written
    before the attr existed looks like on disk.
    """
    rng = np.random.default_rng(seed)
    mpa_store.allocate_fit_store(path, n_q=n_q, n_mu=n_mu, n_p=n_p,
                                 energy_unit=energy_unit)
    for q in range(n_q):
        cols = list(range(n_mu))
        a = np.sort(rng.uniform(0.2, 3.0, size=(n_p, n_mu, n_mu)), axis=0)
        g = rng.uniform(0.01, 0.15, size=a.shape)
        Om = (a - 1j * g) * scale
        Bp = (rng.normal(size=a.shape)
              + 1j * rng.normal(size=a.shape)) * scale
        mpa_store.write_fit_block(
            path, q, cols, Om, Bp,
            {"condition": np.ones((n_mu, n_mu)),
             "backward_error": np.full((n_mu, n_mu), 1e-12)})
    mpa_store.finalize_fit_store(path)
    if not declare:
        import h5py
        with h5py.File(path, "a") as f:
            del f.attrs[mpa_store.FIT_ENERGY_UNIT_ATTR]
    return path


# ---------------------------------------------------------------------------
# 1. The twin gate, and the red twin beside it
# ---------------------------------------------------------------------------

def test_a_ha_store_and_its_x2_ry_twin_read_back_bit_identically(tmp_path):
    """THE CLOSER'S TWIN GATE, in miniature and in-tree.

    Same pole field written twice: once as Hartree numbers declared 'Ha',
    once as the exactly-doubled Rydberg numbers declared 'Ry'.  The
    converting readers must hand back the SAME arrays -- and because
    multiplying by 2 is exact in binary floating point, "the same" is
    ``np.array_equal`` and not a tolerance.  All three readers are held
    to it, because a conversion living in one reader and not another is
    the two-authorities state this fix exists to end.
    """
    ha = _filled_store(str(tmp_path / "ha.h5"), energy_unit="Ha", scale=1.0)
    ry = _filled_store(str(tmp_path / "ry.h5"), energy_unit="Ry", scale=2.0)

    for p in range(2):
        om_ha, bp_ha = mpa_store.read_pole_slice(ha, p)
        om_ry, bp_ry = mpa_store.read_pole_slice(ry, p)
        assert np.array_equal(om_ha, om_ry)
        assert np.array_equal(bp_ha, bp_ry)

    om_ha, bp_ha, _, led_ha = mpa_store.read_fit_tensors(ha)
    om_ry, bp_ry, _, led_ry = mpa_store.read_fit_tensors(ry)
    assert np.array_equal(om_ha, om_ry) and np.array_equal(bp_ha, bp_ry)
    assert led_ha["energy_unit"] == "Ha" and led_ry["energy_unit"] == "Ry"

    ob_ha = mpa_store.read_fit_block(ha, 1, [0, 1])
    ob_ry = mpa_store.read_fit_block(ry, 1, [0, 1])
    assert np.array_equal(ob_ha[0], ob_ry[0])
    assert np.array_equal(ob_ha[1], ob_ry[1])


def test_the_red_twin_the_same_bytes_under_the_two_declarations_differ_by_2(
        tmp_path):
    """The FALSE case, at the closer's own measured ratio.

    One pole field, two declarations: read as Ha it must come back
    exactly DOUBLE what it reads as Ry -- elementwise ratio 2.000000,
    which is the number the closer's rescale stage verified on the real
    store.  A conversion that silently did nothing (the defect) fails
    the first assertion; one that converted twice fails the second.
    """
    ha = _filled_store(str(tmp_path / "ha.h5"), energy_unit="Ha")
    ry = _filled_store(str(tmp_path / "ry.h5"), energy_unit="Ry")

    om_ha, bp_ha = mpa_store.read_pole_slice(ha, 0)
    om_ry, bp_ry = mpa_store.read_pole_slice(ry, 0)
    assert np.array_equal(om_ha, 2.0 * om_ry)
    assert np.array_equal(bp_ha, 2.0 * bp_ry)
    assert not np.array_equal(om_ha, om_ry)


def test_raw_reads_hand_back_the_stored_bytes_for_tooling(tmp_path):
    """``raw=True`` is the migration escape: no refusal, no conversion.

    Its FALSE case is that a raw read of the Ha store equals the
    CONVERTED read -- it must not; raw is the bytes, converted is Ry.
    """
    ha = _filled_store(str(tmp_path / "ha.h5"), energy_unit="Ha")
    om_raw, bp_raw = mpa_store.read_pole_slice(ha, 0, raw=True)
    om_ry, bp_ry = mpa_store.read_pole_slice(ha, 0)
    assert np.array_equal(2.0 * om_raw, om_ry)
    assert np.array_equal(2.0 * bp_raw, bp_ry)


# ---------------------------------------------------------------------------
# 2. The refusals
# ---------------------------------------------------------------------------

def test_an_undeclared_store_is_refused_by_name_with_both_fixes(tmp_path):
    """The legacy store -- the first-light field's shape -- cannot be read
    by a Sigma consumer at all.  The refusal must name the attr, the
    migration call and the oracle, because 'refused' without 'and here is
    how to proceed' is half a refusal."""
    legacy = _filled_store(str(tmp_path / "old.h5"), energy_unit="Ry",
                           declare=False)
    for reader in (
            lambda: mpa_store.read_pole_slice(legacy, 0),
            lambda: mpa_store.read_fit_tensors(legacy),
            lambda: mpa_store.read_fit_block(legacy, 0, [0, 1])):
        with pytest.raises(ValueError) as exc:
            reader()
        msg = str(exc.value)
        assert mpa_store.FIT_ENERGY_UNIT_ATTR in msg
        assert "declare_fit_energy_unit" in msg
        assert "18.118" in msg and "18.009" in msg      # the oracle
    # ...and raw=True still reads it, which is how a migration looks at
    # what it is migrating.
    om, bp = mpa_store.read_pole_slice(legacy, 0, raw=True)
    assert om.shape == (3, 4, 4)


def test_allocate_without_a_unit_refuses_and_names_the_fit_driver(tmp_path):
    with pytest.raises(ValueError) as exc:
        mpa_store.allocate_fit_store(str(tmp_path / "x.h5"),
                                     n_q=1, n_mu=2, n_p=1)
    msg = str(exc.value)
    assert "energy_unit" in msg and "no default" in msg
    assert "mpa_omega_units" in msg


def test_an_unknown_unit_spelling_refuses_everywhere(tmp_path):
    with pytest.raises(ValueError, match="not one of"):
        mpa_store.allocate_fit_store(str(tmp_path / "x.h5"),
                                     n_q=1, n_mu=2, n_p=1, energy_unit="eV")
    with pytest.raises(ValueError, match="not one of"):
        mpa_store.canonical_energy_unit("hartrees", where="test")
    # Case-insensitive on input, canonical out -- a deck's 'ha' is 'Ha'.
    assert mpa_store.canonical_energy_unit("ha", where="test") == "Ha"
    assert mpa_store.canonical_energy_unit("RY", where="test") == "Ry"


# ---------------------------------------------------------------------------
# 3. The migration stamp
# ---------------------------------------------------------------------------

def test_declare_fit_energy_unit_stamps_once_and_only_once(tmp_path):
    legacy = _filled_store(str(tmp_path / "old.h5"), energy_unit="Ry",
                           declare=False)
    with pytest.raises(ValueError):
        mpa_store.read_pole_slice(legacy, 0)          # undeclared: refused
    assert mpa_store.declare_fit_energy_unit(legacy, "Ha") == "Ha"
    om, _ = mpa_store.read_pole_slice(legacy, 0)      # now converts
    om_raw, _ = mpa_store.read_pole_slice(legacy, 0, raw=True)
    assert np.array_equal(om, 2.0 * om_raw)
    # Idempotent at the same value; a DIFFERENT value is the two-claims
    # state and refuses.
    assert mpa_store.declare_fit_energy_unit(legacy, "Ha") == "Ha"
    with pytest.raises(ValueError, match="already declares"):
        mpa_store.declare_fit_energy_unit(legacy, "Ry")


def test_declare_reaches_the_legacy_head_sets_too(tmp_path):
    """``include_heads`` stamps undeclared ``__mpahead*`` groups with the
    same unit, because the first-light heads were fitted against the same
    abscissae as the body -- and a declared head comes back CONVERTED."""
    legacy = _filled_store(str(tmp_path / "old.h5"), energy_unit="Ry",
                           declare=False, n_p=2)
    mpa_store.allocate_head_axis(legacy, n_p=2)
    z = np.array([0.0, 0.5, 1.0, 1.5]) + 0.05j
    z[0] = 0.0
    Om = np.array([0.66 - 0.02j, 1.44 - 0.05j])
    Bp = np.array([-0.30 + 0.01j, -0.10 - 0.02j])
    w = np.sum(Bp / (z[:, None] - Om) - Bp / (z[:, None] + Om), axis=1)
    mpa_store.write_head_axis(legacy, z, w, Om, Bp)   # legacy: no unit

    before = mpa_store.read_head_poles(legacy)
    assert before["energy_unit"] is None
    assert np.array_equal(before["Omega_p"], Om)

    mpa_store.declare_fit_energy_unit(legacy, "Ha")
    after = mpa_store.read_head_poles(legacy)
    assert after["energy_unit"] == "Ry"
    assert np.array_equal(after["Omega_p"], 2.0 * Om)
    assert np.array_equal(after["z"], 2.0 * z)
    assert np.array_equal(after["B_p"], 2.0 * Bp)
    # w and vhead are energy VALUES of W in the producers' Ry -- the
    # ordinate, not the axis -- and must NOT be scaled.
    assert np.array_equal(after["w"], w)


def test_a_new_head_write_can_declare_its_own_unit(tmp_path):
    store = _filled_store(str(tmp_path / "new.h5"), energy_unit="Ry", n_p=1)
    mpa_store.allocate_head_axis(store, n_p=1, label="declared")
    z = np.array([0.0, 1.0 + 0.1j])
    Om = np.array([0.7 - 0.02j])
    Bp = np.array([-0.2 + 0.01j])
    w = (Bp / (z[:, None] - Om) - Bp / (z[:, None] + Om)).sum(axis=1)
    mpa_store.write_head_axis(store, z, w, Om, Bp, label="declared",
                              energy_unit="Ha")
    got = mpa_store.read_head_poles(store, label="declared")
    assert got["energy_unit"] == "Ry"
    assert np.array_equal(got["Omega_p"], 2.0 * Om)
    assert np.array_equal(got["w"], w)


# ---------------------------------------------------------------------------
# 4. The producer inherits, the consumer converts -- end to end
# ---------------------------------------------------------------------------

def test_the_fit_driver_inherits_the_unit_from_the_w_stores_own_stamp():
    """Read from source: the allocate call must take the unit from the W
    header (``omega_units``), never from a literal.  A run of the full
    driver needs a W(omega) file with unfold tables and a device; the
    inheritance is one call site, and pinning WHERE the value comes from
    is the load-bearing fact -- a human-typed unit here would be the
    defect reborn with a declaration's clothes on."""
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "gw" / "mpa" / "fit_driver.py").read_text()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "allocate_fit_store"]
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kw.get("energy_unit") == "header['omega_units']"


def test_sigma_pass_consumes_the_converted_read_with_no_unit_logic_of_its_own():
    """The Sigma loop's read is ``read_pole_slice``; the conversion lives
    there and NOWHERE in ``sigma_pass`` -- one seam, checked by absence.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "gw" / "mpa" / "sigma_pass.py").read_text()
    assert "read_pole_slice" in src
    assert "FIT_ENERGY_UNITS" not in src
    assert "energy_unit" not in src, (
        "sigma_pass grew unit logic of its own; the conversion seam is "
        "mpa_store's readers, once")
