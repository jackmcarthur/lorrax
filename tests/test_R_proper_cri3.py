"""Verify ``SymMaps.R_proper`` matches the offline-derived fixture (mod
transpose) and is real-orthogonal on every row.

The fixture lives at
``/pscratch/sd/j/jackm/lorrax_sandbox/reports/bispinor_ibz_2026-05-16/cri3_R_proper.npz``
and pairs with the math derivation
(``reports/bispinor_ibz_2026-05-16/derivation.md`` §A2).

Two convention notes (both documented inside SymMaps.__init__):

  1. The fixture uses ``O = A · U · A⁻¹`` with ``U = mtrx⁻¹`` (forward
     direct-coord rotation), i.e. ``A.T · inv(mtrx) · inv(A.T)``.  The
     LIVE code reuses ``R_cart`` (= ``A.T · mtrx · inv(A.T)``) because
     that's the rotation matrix that LORRAX's actual ``U_spinor``
     satisfies the σ-sandwich identity with.  For orthogonal mtrx the
     two are inverses (= transposes) of each other; the §A5 mixing
     formula then absorbs the transpose into its index ordering.  This
     test compares the live spatial half to ``fixture.T``.

  2. The fixture stores ``R_proper_TRS = -R_proper_spatial`` (formally
     correct from ``T σ^i T⁻¹ = -σ^i`` on i=1,2,3); the LIVE code
     stores ``R_proper_TRS = +R_proper_spatial`` because the σ-flip on
     (μ_L, ν_L) ∈ {1,2,3}² UNIQUE_TILES factorises as (−1)·(−1) = +1
     and is absorbed by the existing ``unfold_isdf_operator`` conj-wrap
     (derivation §A4).  This test compares the spatial halves only.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Depends on out-of-repo CrI3 fixtures (runs/ WFN + reports npz) and
# skips where absent — behind the `extra` marker (run with `-m extra`).
pytestmark = pytest.mark.extra


# Skip cleanly when the fixture isn't reachable (CI / minimal sandbox).
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


def _resolve_wfn_path() -> Path | None:
    for p in _WFN_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.mark.skipif(not _FIXTURE_PATH.exists(),
                    reason=f"R_proper fixture not available at {_FIXTURE_PATH}")
def test_R_proper_cri3_matches_fixture():
    """SymMaps.R_proper spatial half == offline fixture R_proper_spatial."""
    wfn_path = _resolve_wfn_path()
    if wfn_path is None:
        pytest.skip(
            f"No CrI3 6x6 30Ry SOC WFN.h5 reachable; tried: "
            f"{[str(p) for p in _WFN_CANDIDATES]}.")

    # JAX x64 is set by tests/conftest.py via JAX_ENABLE_X64=1.
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    from symmetry_maps import SymMaps

    npz = np.load(_FIXTURE_PATH, allow_pickle=True)
    R_proper_fixture = np.asarray(npz['R_proper'], dtype=np.float64)
    R_spatial_fixture = np.asarray(npz['R_spatial'], dtype=np.float64)
    ntran_fixture = int(np.asarray(npz['ntran']))

    # Live SymMaps
    wfn = WfnLoader(str(wfn_path))
    sym = SymMaps(wfn)
    R_proper_live = np.asarray(sym.R_proper, dtype=np.float64)

    # Shape contract: (2·ntran, 3, 3)
    assert R_proper_live.shape == (2 * ntran_fixture, 3, 3), (
        f"R_proper shape mismatch: live={R_proper_live.shape}, "
        f"expected=({2 * ntran_fixture}, 3, 3)"
    )
    assert R_proper_fixture.shape == R_proper_live.shape, (
        f"Fixture shape mismatch: {R_proper_fixture.shape} vs "
        f"{R_proper_live.shape}"
    )

    # Compare SPATIAL halves (rows [0, ntran)) modulo the convention
    # transpose described in the module docstring: live R_proper uses
    # ``A.T · mtrx · inv(A.T)``, fixture uses ``A.T · inv(mtrx) · inv(A.T)``
    # = transpose of the live form on every row.
    fixture_spatial_transposed = np.transpose(
        R_proper_fixture[:ntran_fixture], axes=(0, 2, 1))
    np.testing.assert_allclose(
        R_proper_live[:ntran_fixture],
        fixture_spatial_transposed,
        atol=1e-9,
        err_msg="R_proper spatial half differs from fixture (after the "
                "documented convention transpose).",
    )

    # TRS half in the LIVE code reuses the SPATIAL R_proper.
    np.testing.assert_allclose(
        R_proper_live[ntran_fixture:],
        R_proper_live[:ntran_fixture],
        atol=1e-9,
        err_msg="R_proper TRS half should equal the spatial half "
                "(derivation §A4).",
    )

    # Every row of R_proper is real-orthogonal (det=+1, R Rᵀ = I) to
    # within the ``np.around(decimals=10)`` snapping that
    # ``syms_crystal_to_cartesian`` applies to its output.
    for s in range(R_proper_live.shape[0]):
        R = R_proper_live[s]
        np.testing.assert_allclose(
            R @ R.T, np.eye(3),
            atol=1e-6,
            err_msg=f"R_proper[{s}] not orthogonal.",
        )
        det = float(np.linalg.det(R))
        assert abs(det - 1.0) < 1e-6, (
            f"R_proper[{s}] det={det}, expected +1 (proper rotation)."
        )

    # Sanity: R_cart matches the fixture spatial Cartesian rotations
    # modulo the same convention transpose described above.  The fixture
    # R_spatial is shape (2*ntran, 3, 3) and the TRS half is -R_spatial
    # (its own convention); we compare only the spatial halves here.
    R_cart_live_spatial = np.asarray(sym.R_cart[:ntran_fixture],
                                       dtype=np.float64)
    fixture_R_spatial_spatial_transposed = np.transpose(
        R_spatial_fixture[:ntran_fixture], axes=(0, 2, 1))
    np.testing.assert_allclose(
        R_cart_live_spatial, fixture_R_spatial_spatial_transposed,
        atol=1e-6,
        err_msg="R_cart spatial half differs from fixture R_spatial "
                "(after the documented convention transpose).",
    )
