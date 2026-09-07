"""Restart transverse parents are checked before carrier construction."""
from types import SimpleNamespace
import inspect
import numpy as np
import pytest
from common import sanity
from gw.gw_init import _restart_current_carrier

@pytest.mark.parametrize("field,stage", [
    ("psi_nmu_parent_transverse", "psi_parent_y_transverse"),
    ("psi_mun_parent_transverse", "psi_parent_y_transverse_mun"),
])
def test_nan_transverse_parent_refuses_before_construction(monkeypatch, field, stage):
    """A corrupt stored face refuses with its dataset named under strict sanity."""
    monkeypatch.setenv("LORRAX_SANITY", "strict")
    rs = SimpleNamespace(psi_nmu_parent_transverse=np.ones((1, 1, 4, 1)),
                         psi_mun_parent_transverse=np.ones((1, 4, 1, 1)))
    getattr(rs, field).flat[0] = np.nan
    args = dict.fromkeys(inspect.signature(_restart_current_carrier).parameters)
    args.update(cfg=SimpleNamespace(bispinor=True), rs=rs, print0=lambda *a: None)
    with pytest.raises(sanity.SanityError, match=stage):
        _restart_current_carrier(**args)
