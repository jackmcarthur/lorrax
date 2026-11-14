"""A ``dipole.h5`` without ``prov_bispinor`` is a mismatch on a bispinor run.

CPU, pytest.  ``file_io.restart_bundle.dipole_operator_mismatches`` is the
always-on operator check of the q->0 head (``gw.head_correction``).  Since the
Pauli carrier (2026-11-11) a bispinor run's charge head is the two-spinor
dipole (``prov_bispinor = False``); a file that predates the stamp was the
old carrier's four-spinor dipole with the same shape, so its absence must
refuse on a bispinor run and stay accepted on a two-component one.
"""

import h5py


def _dipole(path, **attrs):
    from file_io.dipole import DIPOLE_FORMAT_ATTR, DIPOLE_FORMAT_VERSION
    with h5py.File(path, "w") as h5:
        h5.attrs[DIPOLE_FORMAT_ATTR] = DIPOLE_FORMAT_VERSION
        h5.attrs["prov_skip_vnl"] = False
        h5.attrs["prov_vnl_mode"] = "analytic"
        for key, value in attrs.items():
            h5.attrs[key] = value
    return path


def test_absent_bispinor_stamp_refuses_only_on_a_bispinor_run(tmp_path):
    from file_io.restart_bundle import dipole_operator_mismatches
    op = dict(skip_vnl=False, vnl_mode="analytic")
    legacy = _dipole(tmp_path / "legacy.h5")
    two = _dipole(tmp_path / "two.h5", prov_bispinor=False)
    four = _dipole(tmp_path / "four.h5", prov_bispinor=True)
    # Two-component run: a legacy file still reads (no stamp to compare).
    assert dipole_operator_mismatches(legacy, bispinor=False, **op) == []
    # Bispinor run (charge head = two-spinor dipole).
    run = dict(bispinor=False, bispinor_run=True, **op)
    assert dipole_operator_mismatches(two, **run) == []
    assert dipole_operator_mismatches(legacy, **run) == [
        ("prov_bispinor", "<absent>", False)]
    assert [k for k, _g, _w in dipole_operator_mismatches(four, **run)] == [
        "prov_bispinor"]
