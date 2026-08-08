"""non-TDA dense build: the trial width is derived from the T footprint.

RED TWIN recorded in FIX_driver_blockers.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

# ===========================================================================
# 2. the non-TDA dense build sizes its trial width from the T footprint
# ===========================================================================
class _FakeMesh:
    def __init__(self, px, py):
        self.devices = np.empty((px, py), dtype=object)


def _args_for(mu, nu, nk3, nspinor=2):
    """The two operands ``dense_col_chunk`` reads: psi_c_X (arg 0), W_R (arg 6)."""
    psi_c_X = np.zeros((int(np.prod(nk3)), 1, nspinor, mu), dtype=np.complex128)
    W_R = np.zeros((mu, nu) + tuple(nk3), dtype=np.complex128)
    return (psi_c_X, None, None, None, None, None, W_R)


def test_col_chunk_is_derived_not_eight():
    """The Si 4x4x4 / 480-centroid case must NOT come back at the old 8.

    RED TWIN: restore ``col_chunk=8`` as a hard-coded default and this fails —
    which is the whole defect, since 8 columns of a 900 MiB T tensor is the
    56.47 GiB peak that OOMs a 40 GB A100.
    """
    from bse.bse_nontda import dense_col_chunk, _DENSE_T_PEAK_FACTOR

    args = _args_for(480, 480, (4, 4, 4))
    chunk = dense_col_chunk(args, _FakeMesh(1, 1), 1024)
    assert 1 <= chunk <= 1024
    # one column is 900 MiB here; its compiled peak is ~4.7 GiB, so no
    # plausible per-device budget admits eight of them
    per_col = 480 * 480 * 2 * 2 * 64 * 16
    assert per_col == 900 * 2 ** 20
    assert chunk * per_col * _DENSE_T_PEAK_FACTOR < 200e9


def test_col_chunk_shrinks_as_the_T_tensor_grows():
    """Width is monotone-decreasing in the footprint — the property that makes
    it a bound rather than a constant.

    RED TWIN: any hard-coded width makes this sequence constant.
    """
    from bse.bse_nontda import dense_col_chunk

    mesh = _FakeMesh(1, 1)
    widths = [dense_col_chunk(_args_for(mu, mu, (4, 4, 4)), mesh, 4096)
              for mu in (60, 120, 240, 480, 960)]
    assert widths == sorted(widths, reverse=True), widths
    assert widths[0] > widths[-1], widths
    assert min(widths) >= 1


def test_col_chunk_floors_at_one_and_never_refuses():
    """An impossible footprint clamps to 1 rather than raising: 1 is the
    narrowest this build has, so refusing would only pre-empt a run that the
    budget estimate might have mispriced."""
    from bse.bse_nontda import dense_col_chunk

    args = _args_for(20000, 20000, (8, 8, 8))
    assert dense_col_chunk(args, _FakeMesh(1, 1), 4096) == 1


def test_col_chunk_accounts_for_the_mesh():
    """Sharding mu over x and nu over y shrinks the LOCAL tensor, so a wider
    mesh must admit a wider trial block.

    RED TWIN: read the global W_R shape instead of dividing by (px, py) and
    the two widths come back equal.
    """
    from bse.bse_nontda import dense_col_chunk

    args = _args_for(480, 480, (4, 4, 4))
    one = dense_col_chunk(args, _FakeMesh(1, 1), 4096)
    four = dense_col_chunk(args, _FakeMesh(2, 2), 4096)
    assert four > one, (one, four)


def test_col_chunk_budget_is_total_memory_not_free():
    """Determinism: the width must not move with ambient GPU occupancy.

    RED TWIN: source the budget from ``get_device_memory_gb`` (which falls back
    to *free* memory) and two calls straddling an allocation can disagree —
    observed live as col_chunk 2 vs 5 on the same deck twenty minutes apart.
    """
    import bse.bse_nontda as bn

    src = Path(bn.__file__).read_text(encoding="utf-8")
    assert "get_device_memory_info" in src
    assert not re.search(r"get_device_memory_gb\s*\(", src), \
        "free-memory budget reintroduced; the width stops being reproducible"

    args = _args_for(480, 480, (4, 4, 4))
    mesh = _FakeMesh(1, 1)
    a = bn.dense_col_chunk(args, mesh, 1024)
    blob = np.zeros((256, 1 << 20), dtype=np.uint8)   # perturb ambient memory
    b = bn.dense_col_chunk(args, mesh, 1024)
    del blob
    assert a == b


def test_materialize_A_B_default_is_auto():
    """``col_chunk`` defaults to None (derive), not to a number."""
    import inspect

    from bse.bse_nontda import _materialize_A_B

    assert inspect.signature(
        _materialize_A_B).parameters["col_chunk"].default is None


