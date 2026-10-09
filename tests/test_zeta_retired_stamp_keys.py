"""A ζ stamp an older build wrote with ``distributed_lu`` is the current stamp (CPU only, seconds).

``gw_init._zeta_fit_provenance`` no longer writes ``distributed_lu``; it named a
transverse LU backend, and ``transverse_solver_kind`` records the factor the fit
ran.  ``_RETIRED_STAMP_KEYS`` drops it from an on-disk stamp in the two places a
stamp is judged:

1. ``_zeta_reuse_ok``: the old stamp's bytes differ from this run's, so the
   comparison parses both and drops the retired key; the ζ is reused.
2. ``charge_zeta_identity``: the MPA and restart receipt digest is the same for
   the old and the new stamp.

A key that is not retired still refits and still moves the digest, so neither
check passes by ignoring the stamp.
"""
import json
from types import SimpleNamespace

import numpy as np

CENTROIDS = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int32)


def _current_stamp():
    from gw.gw_init import _zeta_fit_provenance
    wfn = SimpleNamespace(_filename="", ecutwfc=20.0, ecutrho=80.0)
    meta = SimpleNamespace(n_rmu=2, nspinor_wfnfile=2, fft_grid=(4, 4, 4))
    cfg = SimpleNamespace(bispinor=False, backend=SimpleNamespace(
        zeta_ridge=0.0, zeta_rcond=1e-8, charge_zeta_solve="pinv",
        gamma_contract_mode="auto"))
    return _zeta_fit_provenance(
        wfn=wfn, meta=meta, cfg=cfg, band_range_left=(0, 4), band_range_right=(0, 4),
        logical_band_stop=4, zeta_cutoff=10.0, zeta_vcoul_cutoff=10.0,
        write_ibz_only=True, band_norms=None)


def _stamp_with(stamp, **extra):
    return json.dumps({**json.loads(stamp), **extra}, sort_keys=True)


def _reuse(tmp_path, monkeypatch, on_disk, now):
    import file_io.restart_bundle as rb
    from gw.gw_init import _zeta_reuse_ok
    path = tmp_path / "zeta_q.h5"
    path.touch()
    header = SimpleNamespace(zeta_is_done=True, fit_provenance=on_disk,
                             r_mu_fft_idx=CENTROIDS)
    monkeypatch.setattr(rb, "read_isdf_header", lambda p: header)
    lines = []
    ok = _zeta_reuse_ok(str(path), now, CENTROIDS, print_fn=lines.append)
    return ok, lines


def test_old_stamp_with_distributed_lu_is_reused(tmp_path, monkeypatch):
    from gw.gw_init import _RETIRED_STAMP_KEYS
    now = _current_stamp()
    assert "distributed_lu" in _RETIRED_STAMP_KEYS
    assert "distributed_lu" not in json.loads(now)
    old = _stamp_with(now, distributed_lu="cusolvermp")
    assert old != now
    ok, lines = _reuse(tmp_path, monkeypatch, old, now)
    assert ok, lines
    ok, lines = _reuse(tmp_path, monkeypatch, _stamp_with(now, zeta_cutoff_ry=11.0), now)
    assert not ok and "DIFFERENT inputs" in lines[-1], lines


def test_old_stamp_with_distributed_lu_has_the_current_identity():
    from gw.gw_init import charge_zeta_identity
    wfn = SimpleNamespace(energies=np.zeros((1, 4)), kpoints=np.zeros((1, 3)),
                          nelec=2, nspinor=2, nbands=4)
    now = _current_stamp()
    ident = charge_zeta_identity(now, wfn=wfn)
    old = _stamp_with(now, distributed_lu="scalapack")
    assert charge_zeta_identity(old, wfn=wfn) == ident
    assert charge_zeta_identity(_stamp_with(now, n_rmu=3), wfn=wfn) != ident
