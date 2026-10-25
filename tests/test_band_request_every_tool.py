"""Every tool reads the same band request from one input file.

An input file in the ``number_bands_protected`` form resolves to the same
nval/ncond in gw, kin-ion and dipole (``LorraxConfig.with_band_request``), in
the htransform consumers (``fh_interp.initialize_wfns``: htransform, BSE
kgrid_fine, exciton-bands) and in the centroid selection (``kmeans_cli``).
No wavefunctions are read: the WFN is its occupied count.
"""
from types import SimpleNamespace

import pytest

DECK = "[cohsex]\nsys_dim = 3\nnumber_bands = 304\ncentroids_file = c.txt\n"


class _Reached(Exception):
    pass


@pytest.mark.parametrize("request_lines, expected", [
    ("number_bands_protected = 40\n", (18, 22)),
    ("nval = 5\nncond = 7\n", (5, 7)),
])
def test_band_request_every_tool(tmp_path, monkeypatch, request_lines, expected):
    from gw.gw_config import LorraxConfig, read_lorrax_input, resolve_band_request
    deck = tmp_path / "gw.in"
    deck.write_text(DECK + request_lines)
    wfn = SimpleNamespace(nelec=18, nbands=304, fft_grid=(4, 4, 4))
    quiet = lambda *a, **k: None
    seen = {}

    cfg = LorraxConfig.from_input_file(
        str(deck), print_fn=quiet, resolve_hardware=False).with_band_request(wfn, quiet)
    seen["gw / kin-ion / dipole"] = (cfg.nval, cfg.ncond)

    params = resolve_band_request(read_lorrax_input(str(deck)), wfn, quiet)
    seen["read_lorrax_input"] = (params["nval"], params["ncond"])

    import file_io.qp_wfn
    from bandstructure import fh_interp
    import file_io.centroids
    monkeypatch.setattr(file_io.qp_wfn, "refuse_conflicting_qp_state_sources",
                        lambda **k: None)

    def _stop(*a, **k):
        raise _Reached
    monkeypatch.setattr(file_io.centroids, "load_centroid_basis", _stop)
    params = read_lorrax_input(str(deck))
    with pytest.raises(_Reached):
        fh_interp.initialize_wfns(str(deck), params, quiet, mesh_xy=object(),
                                  wfn_sym=(wfn, None))
    seen["htransform / bse / exciton-bands"] = (params["nval"], params["ncond"])

    # kmeans_cli._resolve_weight: the input file's parameters, then the
    # left pair window [0, nelec + ncond) (kmeans_cli itself starts the runtime).
    from centroid.production_output import prune_band_ranges, resolve_sigma_ncond
    args = SimpleNamespace(fit_window=None, sigma_ncond=None,
                           sigma_deck=read_lorrax_input(str(deck)))
    resolve_sigma_ncond(args, wfn, quiet)
    left, _, _ = prune_band_ranges(args, wfn.nelec, wfn.nbands - wfn.nelec)
    seen["centroids (left window)"] = left

    from gw.gw_config import _DECK_NAMED_KEYS, missing_deck_keys
    named = read_lorrax_input(str(deck))[_DECK_NAMED_KEYS]
    assert missing_deck_keys(named, "number_bands", "band_request",
                             "ncond_request", "centroids_file") == []
    assert missing_deck_keys({"nval"}, "band_request", "number_bands") == [
        "ncond is required (or number_bands_protected)", "number_bands is required"]

    windows = {k: v for k, v in seen.items() if k != "centroids (left window)"}
    assert set(windows.values()) == {expected}, seen
    assert left == (0, 18 + expected[1]), seen
