"""Compare typed Cartesian actions with the retained CrI3 derivation."""

from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.extra

_FIXTURE_PATH = Path(
    "/pscratch/sd/j/jackm/lorrax_sandbox/reports/bispinor_ibz_2026-05-16/"
    "cri3_R_proper.npz"
)
_WFN_CANDIDATES = (
    Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/"
         "M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5"),
    Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/"
         "07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/WFN.h5"),
)


def _wfn_path():
    return next((path for path in _WFN_CANDIDATES if path.exists()), None)


@pytest.mark.skipif(
    not _FIXTURE_PATH.exists(),
    reason=f"Cartesian-action fixture not available at {_FIXTURE_PATH}",
)
def test_typed_cartesian_actions_match_cri3_fixture():
    wfn_path = _wfn_path()
    if wfn_path is None:
        pytest.skip(f"No retained CrI3 WFN; tried {_WFN_CANDIDATES}.")

    from symmetry_maps import SymMaps
    from wfn_loader import WfnLoader

    fixture = np.load(_FIXTURE_PATH, allow_pickle=True)
    axial_fixture = np.asarray(fixture["R_proper"], dtype=np.float64)
    polar_fixture = np.asarray(fixture["R_spatial"], dtype=np.float64)
    ntran = int(np.asarray(fixture["ntran"]))

    sym = SymMaps(WfnLoader(str(wfn_path)))
    rows = np.arange(2 * ntran, dtype=np.int32)
    axial = np.asarray(
        sym.cartesian_action(rows, axial=True, time_odd=False))
    polar = np.asarray(
        sym.cartesian_action(rows, axial=False, time_odd=False))

    assert axial.shape == polar.shape == (2 * ntran, 3, 3)
    np.testing.assert_allclose(
        axial[:ntran], axial_fixture[:ntran], atol=1.0e-9)
    np.testing.assert_allclose(
        polar[:ntran], polar_fixture[:ntran], atol=1.0e-6)
    np.testing.assert_allclose(axial[ntran:], axial[:ntran], atol=1.0e-9)

    for row, action in enumerate(axial):
        np.testing.assert_allclose(
            action @ action.T, np.eye(3), atol=1.0e-6,
            err_msg=f"axial action {row} is not orthogonal")
        assert abs(float(np.linalg.det(action)) - 1.0) < 1.0e-6
