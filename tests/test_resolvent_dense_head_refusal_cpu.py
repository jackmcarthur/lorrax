"""A dense DFT head must not be silently ignored by a density resolvent."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gw.gw_config import (ComputeMode, HeadCorrection, LorraxConfig,
                          ScreeningDiagrams, refuse_unsupported_head_dense)


def _config(tmp_path, *, mode="gn_ppm", diagrams="w_rpa", dense=True):
    path = tmp_path / "gw.in"
    path.write_text(f"[cohsex]\nsys_dim = 3\ncompute_mode = {mode}\nscreening_diagrams = {diagrams}\n"
                    "head_correction = full\nnumber_bands = 30\n"
                    "nval = 4\nncond = 4\n"
                    + ("head_dense_file = dense_dipole.h5\n" if dense else ""))
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                       runtime_platform="cpu", print_fn=lambda *_a: None)


def test_dense_head_refuses_actual_resolvent_decks_and_preserves_rpa(tmp_path):
    for mode in ("cohsex", "gn_ppm", "tangential"):
        rpa = _config(tmp_path, mode=mode)
        assert rpa.head.dense_file.endswith("dense_dipole.h5")
        for diagrams in ("w_bse", "w_rpa_resolvent"):
            if mode == "tangential" and diagrams == "w_rpa_resolvent":
                continue  # This combination already refuses independently.
            with pytest.raises(ValueError, match="GATE head_dense_resolvent"):
                _config(tmp_path, mode=mode, diagrams=diagrams)
            assert not _config(tmp_path, mode=mode, diagrams=diagrams, dense=False).head.dense_file

    # The dense-source rule is diagram-owned, independent of the frequency
    # ansatz. Other parse gates may reject some combinations even earlier.
    rpa = _config(tmp_path)
    for mode in ComputeMode:
        for diagram in (ScreeningDiagrams.W_BSE, ScreeningDiagrams.W_RPA_RESOLVENT):
            candidate = SimpleNamespace(head=rpa.head, compute_mode=mode,
                                        screening=replace(rpa.screening, diagrams=diagram),
                                        bispinor=False, qp_solver=rpa.qp_solver, sc=rpa.sc)
            with pytest.raises(ValueError, match="GATE head_dense_resolvent"):
                refuse_unsupported_head_dense(candidate)
    with pytest.raises(ValueError, match="GATE head_dense_operator"):
        refuse_unsupported_head_dense(replace(rpa, head=replace(rpa.head, correction=HeadCorrection.OFF)))
    with pytest.raises(ValueError, match="GATE head_dense_operator"):
        refuse_unsupported_head_dense(replace(rpa, bispinor=True))
