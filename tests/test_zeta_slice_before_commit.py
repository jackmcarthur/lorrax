"""The ζ fit exposes only selected q rows at its host commit boundaries."""
from __future__ import annotations

import inspect


def test_c_selects_before_outer_block_until_ready():
    """Pin the host seam whose old ordering retained K-Q rows."""
    from gw import isdf_fitting

    fit_source = inspect.getsource(isdf_fitting.fit_zeta_to_h5)
    c_slice = fit_source.index("C_q_flat = slice_q_full_to_ibz(")
    c_wait = fit_source.index("C_q_flat.block_until_ready()", c_slice)
    assert c_slice < c_wait
    assert "C_q.block_until_ready()" not in fit_source
