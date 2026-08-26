"""Pure policy twins for whole-state Galerkin QRCP rank delivery."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) in sys.path:
    sys.path.remove(str(_SRC))
sys.path.insert(0, str(_SRC))


def _owner():
    pytest.importorskip("jax")
    from isdf import galerkin
    assert Path(galerkin.__file__).resolve() == _SRC / "isdf" / "galerkin.py"
    return galerkin


def test_raw_qrcp_rank_above_legacy_2500_cap_is_not_clipped():
    galerkin = _owner()

    rank = galerkin._resolve_qrcp_physical_rank(
        rank_qr=2501, max_search=5000,
        state_dim=7000, state_count=6000)

    assert rank == 2501


def test_raw_qrcp_rank_owns_saturation_before_any_delivery_policy():
    galerkin = _owner()

    # The removed min(raw, 2500) cap reduced this ratio from 90.02% to 50%
    # and therefore hid a saturated search.
    with pytest.raises(ValueError, match=(
            r"raw QRCP rank 4501.*90%.*max_search=5000")):
        galerkin._resolve_qrcp_physical_rank(
            rank_qr=4501, max_search=5000,
            state_dim=7000, state_count=6000)


def test_unfit_raw_rank_refuses_before_selected_rank_allocation():
    galerkin = _owner()
    ledger = {
        "selected_gram_stream": 840.0,
        "selected_gram_fold": 850.0,
        "physical_projection": 851.0,
        "WFN_CUFFT_WORKSPACE": 100.0,
    }

    # Bispinor policy reserves 15%, leaving an 850-byte target from this
    # synthetic 1000-byte live pool.  The one-byte red twin must refuse and
    # name both exact byte counts plus the scientifically honest remedies.
    with pytest.raises(MemoryError) as exc:
        galerkin._refuse_unfit_selected_rank(
            ledger, rank_physical=2501, rank_carrier=2504,
            max_search=5000, qr_eps=1.0e-3, nspinor=2,
            device_pool_limit=1000.0, log_fn=lambda *_args: None)

    message = str(exc.value)
    assert "requires 851 bytes/device" in message
    assert "available target 850 bytes/device" in message
    assert "before the selected Gram/factor/coefficient allocation" in message
    assert "increase qr_eps only if" in message
    assert "Do not lower htransform_rank_multiplier" in message
