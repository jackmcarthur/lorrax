"""Every tool reads the same band request from one input file.

An input file in the ``number_bands_protected`` form resolves to the same
nval/ncond in gw, kin-ion and dipole (``LorraxConfig.with_band_request``), in
the htransform consumers (``fh_interp.initialize_wfns``: htransform, BSE
kgrid_fine, exciton-bands), in the BSE parent restart reader
(``restart_bundle.unfold_parent_faces``, reached by the ladder W reload) and
in the centroid selection (``kmeans_cli``).
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

    # The BSE parent restart reader (restart_bundle.unfold_parent_faces, which
    # the ladder W reload reaches) counts its band window from the same
    # request; it stops at Meta.from_system, where the window is built.
    import h5py
    import numpy as np
    import common
    import file_io.wfn_basis
    import wfn_loader
    from common import centroid_basis
    from file_io import restart_bundle
    from gw import gw_init
    restart = tmp_path / "restart.h5"
    with h5py.File(restart, "w") as f:
        f["psi_parent_k_rows"] = np.arange(1)
        f.attrs["centroids_charge_md5"] = "md5"
    sym = SimpleNamespace(nk_red=1, nk_tot=1)
    wfn_r = SimpleNamespace(nelec=18, nbands=304, fft_grid=(4, 4, 4),
                            symmetry=lambda: sym)
    centroids = SimpleNamespace(centroid_indices=np.zeros((1, 3), dtype=int),
                                orbit_closed=True, n_rmu=1)
    monkeypatch.setattr(restart_bundle, "_require_current", lambda f: None)
    monkeypatch.setattr(wfn_loader, "WfnLoader", lambda path, **k: wfn_r)
    monkeypatch.setattr(file_io.qp_wfn, "authenticate_restart_qp_state_source_for_wfn",
                        lambda **k: None)
    monkeypatch.setattr(file_io.centroids, "load_centroid_basis",
                        lambda *a, **k: centroids)
    monkeypatch.setattr(file_io.wfn_basis, "centroid_table_md5", lambda idx: "md5")
    monkeypatch.setattr(centroid_basis.PackedCentroidBasis, "build",
                        lambda *a, **k: SimpleNamespace(describe=lambda: ""))
    monkeypatch.setattr(gw_init, "coarse_class_for_deck", lambda *a: None)

    def _meta(wfn_, sym_, nval, ncond, *a, **k):
        seen["BSE parent restart"] = (nval, ncond)
        raise _Reached
    monkeypatch.setattr(common.Meta, "from_system", _meta)
    raw = LorraxConfig.from_input_file(str(deck), print_fn=quiet, resolve_hardware=False)
    if raw.number_bands_protected is not None:
        # An unresolved request is a named refusal, not a TypeError.
        with pytest.raises(ValueError, match="GATE band_request_unresolved"):
            gw_init.prepare_band_metadata(None, raw, None, 1, quiet, sym, wfn_r)
    with pytest.raises(_Reached):
        restart_bundle.unfold_parent_faces(
            (np.zeros(1),), str(restart), str(deck), mesh_xy=None)

    windows = {k: v for k, v in seen.items() if k != "centroids (left window)"}
    assert set(windows.values()) == {expected}, seen
    assert left == (0, 18 + expected[1]), seen
