"""The run report and the output headers say which W Sigma used and which W is stored."""
from gw.gw_config import LorraxConfig
from gw.production_report import method_rows, stored_w0_row


def _config(tmp_path, diagrams, *, mode="cohsex", extra=""):
    path = tmp_path / f"{diagrams}_{mode}.in"
    path.write_text(f"[cohsex]\nsys_dim = 3\ncompute_mode = {mode}\n"
                    f"screening_diagrams = {diagrams}\nhead_correction = off\n"
                    "number_bands = 30\nnval = 4\nncond = 4\n" + extra)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                       runtime_platform="cpu", print_fn=lambda *_a: None)


def test_stored_w0_row_names_the_rpa_rung(tmp_path):
    rows = dict(method_rows(_config(tmp_path, "w_bse")))
    assert "electron-hole ladder" in rows["screening"]
    assert "RPA W(0) the ladder used as its direct rung" in rows["stored W(0)"]
    assert "not stored" in rows["stored W(0)"]
    resolvent = dict(method_rows(_config(tmp_path, "w_rpa_resolvent")))
    assert "resolvent route" in resolvent["screening"]
    assert "Dyson RPA W(0)" in resolvent["stored W(0)"]
    # The default route stores Sigma's own W(0): no extra row.
    assert "stored W(0)" not in dict(method_rows(_config(tmp_path, "w_rpa")))
    assert stored_w0_row(_config(tmp_path, "w_bse", extra="write_restart_tensors = false\n")) is None
