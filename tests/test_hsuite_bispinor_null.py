"""The bispinor null and linearity comparators of the suite, on synthetic files.

CPU, pytest; no driver runs.  The suite's null test (``bisp_bt_null`` against
``bt_twin``, ``tests/hsuite/chain.twin_problems``) and the two-process
linearity leg (``tests/hsuite/bispinor_linearity.py``) are only as good as
their comparators and decks: the twin deck must differ from the null deck by
the representation alone, a carrier-sized difference in any compared matrix
must fail, an exact zero must pass, and the alpha^2 law must reject a term
that scales by s instead of s^2.  The end-to-end runs are P = 4 legs.
"""

import numpy as np
import pytest


def _deck_keys(text):
    keys = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            keys[key] = value
    return keys


def test_null_decks_differ_by_the_representation_only():
    from tests.hsuite import chain

    twin = _deck_keys(chain.RESTART_DECKS["bt_twin.in"])
    null = _deck_keys(chain.RESTART_DECKS["bisp_bt_null.in"])
    ref = _deck_keys(chain.RESTART_DECKS["bisp_bt.in"])
    assert null["bispinor_gw"] == "bare_transverse" and null["bispinor"] == "true"
    assert null["bispinor_transverse_hartree"] == "false"
    assert null["bispinor_transverse_exchange"] == "false"
    assert "bispinor_transverse_hartree" not in ref
    diff = {k for k in set(twin) | set(null) if twin.get(k) != null.get(k)}
    per_file = {"sigma_diag_file", "eqp0_file", "eqp1_file", "report_file",
                "sigma_omega_h5_file"}
    assert diff - per_file == {
        "bispinor", "bispinor_gw", "centroids_file_current", "kin_ion_file",
        "bispinor_transverse_hartree", "bispinor_transverse_exchange"}
    assert twin["centroids_file"] == null["centroids_file"]
    names = [name for name, _module, _argv in chain.STAGES]
    # bt_twin reads dipole_bisp's file (the window the head authenticates).
    assert (names.index("kin_ion") < names.index("dipole_bisp") < names.index("bt_twin")
            < names.index("bisp_bt") < names.index("bisp_bt_null") < names.index("bisp_sc"))
    # Each is a fresh tangential one-shot after another one (tmp/mpa cleared).
    assert all(chain._CLEAR_BEFORE[n] == ("tmp/mpa",)
               for n in ("bt_twin", "bisp_bt", "bisp_bt_null"))
    for deck in ("bt_twin.in", "bisp_bt.in", "bisp_bt_null.in"):
        keys = _deck_keys(chain.RESTART_DECKS[deck])
        dip = _deck_keys(chain.RESTART_DECKS["bisp_sc.in"])     # dipole_bisp's input file
        assert all(keys[k] == dip[k] for k in ("nval", "ncond", "number_bands",
                                               "wfn_file", "sys_dim")), deck


def test_twin_comparator_fails_on_a_carrier_sized_difference(tmp_path, capsys):
    import h5py
    from tests.hsuite import chain

    rng = np.random.default_rng(1)
    m = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    pair = rng.normal(size=(2, 2, 3, 3)) + 0j

    def write(name, **members):
        with h5py.File(tmp_path / name, "w") as h5:
            for key, value in members.items():
                h5[key] = value

    write("twin.h5", hartree_kij_ev=m, sigma_sx_kij_ev=m, sigma_c_kij_ev=pair)
    rule = chain.CHECKS["bisp_bt_null"]["twin"]
    rule = ("mine.h5", "twin.h5") + rule[2:]
    write("mine.h5", hartree_scalar_kij_ev=m + 1e-6, sigma_sx_kij_ev=m,
          sigma_c_kij_ev=pair, hartree_transverse_kij_ev=np.zeros_like(m))
    assert chain.twin_problems(tmp_path, "bisp_bt_null", rule) == []
    assert "hartree_scalar_kij_ev max|diff|" in capsys.readouterr().out
    # The old carrier moved V_H by meV; 0.1 meV must already fail.
    write("mine.h5", hartree_scalar_kij_ev=m, sigma_sx_kij_ev=m + 1e-4,
          sigma_c_kij_ev=pair, hartree_transverse_kij_ev=1e-12 + np.zeros_like(m))
    problems = chain.twin_problems(tmp_path, "bisp_bt_null", rule)
    assert any("sigma_sx_kij_ev differs" in p for p in problems)
    assert any("hartree_transverse_kij_ev is not exactly zero" in p for p in problems)


def test_linearity_law_is_s_squared():
    from tests.hsuite.bispinor_linearity import linearity_problems

    rng = np.random.default_rng(2)
    h, b, vh, c = (rng.normal(size=(2, 3, 3)) for _ in range(4))
    x = rng.normal(size=(2, 3, 3))
    s = 2.0 ** -0.5
    full = dict(hartree_transverse_kij_ev=h, sigma_sx_kij_ev=x + b,
                hartree_scalar_kij_ev=vh, sigma_c_kij_ev=c)
    charge = dict(full, sigma_sx_kij_ev=x)
    good = dict(full, hartree_transverse_kij_ev=s**2 * h, sigma_sx_kij_ev=x + s**2 * b)
    assert linearity_problems(full, good, charge, scale=s, rtol=1e-10) == []
    linear = dict(good, sigma_sx_kij_ev=x + s * b)
    assert any("Sigma^B" in p for p in linearity_problems(full, linear, charge,
                                                          scale=s, rtol=1e-6))
    moved = dict(good, sigma_c_kij_ev=c + 1e-3)
    assert any("sigma_c" in p for p in linearity_problems(full, moved, charge,
                                                          scale=s, rtol=1e-6))
    with pytest.raises(KeyError):
        linearity_problems({}, good, charge, scale=s, rtol=1e-6)


def test_missing_reference_is_reported_not_passed():
    from tests.hsuite import chain
    written = chain.reference_missing_line("bisp_bt", written=True)
    assert "reference missing" in written and "numeric review" in written
    failed = chain.reference_missing_line("bisp_bt", written=False)
    assert failed.startswith("bisp_bt: reference missing")


def test_unwritten_h5_is_a_comparison_failure_not_a_crash(tmp_path):
    from tests.hsuite import chain
    got = chain.captured(tmp_path, "bisp_bt", {"h5": ["sigma_lorentz_oneshot.h5"]})
    assert list(got) == ["sigma_lorentz_oneshot.h5:FILE_NOT_WRITTEN"]
    problems = chain.compare("bisp_bt", got, {"sigma_lorentz_oneshot.h5:sigma_lorentz_skij_ev": got[
        "sigma_lorentz_oneshot.h5:FILE_NOT_WRITTEN"]})
    assert any("missing" in p for p in problems) and any("not in the reference" in p for p in problems)
    assert chain.twin_problems(tmp_path, "bisp_bt_null", ("a.h5", "b.h5", (), ())) == [
        "bisp_bt_null: a.h5 was not written", "bisp_bt_null: b.h5 was not written"]
