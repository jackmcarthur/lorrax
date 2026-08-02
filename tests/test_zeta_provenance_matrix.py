"""Reader matrix of the ζ-fit provenance stamp (gw_init).

The stamp records solve-family keys whose change must force a refit;
keys ADDED after old stamps were written must not (legacy exception).
Two generations of keys ride the same machinery:

* ``distributed_zeta_solve`` (2026-08-01, commit 8b02c72) — legacy
  implied value ``'replicated'``;
* ``transverse_zeta_solve`` / ``transverse_zeta_rcond`` (this change
  set) — legacy implied values ``'ridge'`` / ``None``.

Matrix pinned here (pure logic — header reads are stubbed; the on-disk
h5 path is a real temp file so the existence check is genuine):

1. legacy stamp (missing ALL three keys) + a run requesting the legacy
   values → REUSE;
2. legacy stamp + a bispinor run requesting the rank_truncate family →
   REFIT (family mismatch against the implied legacy value);
3. stamp missing ONLY ``distributed_zeta_solve`` → replicated rerun
   reuses, distributed rerun refits (regression of the 8b02c72
   exception, now expressed through the keyed table);
4. stamp WITH the transverse keys but a different tau → REFIT (generic
   inputs-changed path);
5. ``_zeta_fit_provenance`` collapse: non-bispinor decks produce
   IDENTICAL provenance for any transverse family/tau (the keys are
   inert there — no spurious refit), and bispinor+ridge collapses tau
   to None (tau is not read by the ridge family).
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gw import gw_init  # noqa: E402


CENTS = np.arange(12, dtype=np.int32).reshape(4, 3)


def _cfg(bispinor=True, family="ridge", tau=1e-10, tier="auto"):
    backend = SimpleNamespace(
        zeta_ridge=0.0, zeta_rcond=1e-8,
        charge_zeta_solve="rank_truncate",
        distributed_zeta_solve=tier,
        transverse_zeta_solve=family,
        transverse_zeta_rcond=tau,
        gamma_contract_mode="take",
    )
    return SimpleNamespace(bispinor=bispinor, gspace_mode="host_cache",
                           backend=backend)


def _prov(cfg):
    wfn = SimpleNamespace(_filename="", ecutwfc=30.0, ecutrho=120.0)
    meta = SimpleNamespace(n_rmu=CENTS.shape[0],
                           fft_grid=np.array([9, 9, 45]))
    return gw_init._zeta_fit_provenance(
        wfn=wfn, meta=meta, cfg=cfg,
        band_range_left=(0, 26), band_range_right=(1, 128),
        zeta_cutoff=30.0, zeta_vcoul_cutoff=30.0,
        write_ibz_only=True, band_norms=None)


class _Hdr(SimpleNamespace):
    pass


@pytest.fixture()
def stub_reader(monkeypatch, tmp_path):
    """Point _zeta_reuse_ok at a real (empty) file + a stubbed header."""
    import file_io.isdf_header as ih

    path = str(tmp_path / "zeta_q.h5")
    open(path, "w").close()
    holder = {}

    def _read(_p):
        return holder["hdr"]

    monkeypatch.setattr(ih, "read_isdf_header", _read)
    monkeypatch.delenv("LORRAX_FORCE_REFIT", raising=False)

    def set_stamp(prov_json):
        holder["hdr"] = _Hdr(zeta_is_done=True, fit_provenance=prov_json,
                             r_mu_fft_idx=CENTS.copy())
        return path

    return set_stamp


def _strip(prov_json, keys):
    d = json.loads(prov_json)
    for k in keys:
        d.pop(k, None)
    return json.dumps(d, sort_keys=True)


_ALL3 = ("distributed_zeta_solve", "transverse_zeta_solve",
         "transverse_zeta_rcond")


def test_legacy_stamp_reused_by_legacy_value_run(stub_reader):
    new = _prov(_cfg(bispinor=True, family="ridge"))
    path = stub_reader(_strip(new, _ALL3))
    assert gw_init._zeta_reuse_ok(path, new, CENTS, print_fn=lambda *a: None)


def test_legacy_stamp_refits_for_rank_truncate_family(stub_reader):
    new = _prov(_cfg(bispinor=True, family="rank_truncate", tau=1e-10))
    path = stub_reader(_strip(new, _ALL3))
    msgs = []
    ok = gw_init._zeta_reuse_ok(path, new, CENTS, print_fn=msgs.append)
    assert not ok
    assert any("transverse_zeta_solve" in m for m in msgs)


def test_8b02c72_tier_exception_regression(stub_reader):
    # replicated rerun over a stamp missing only the tier key: reuse.
    new = _prov(_cfg(bispinor=False))
    path = stub_reader(_strip(new, ("distributed_zeta_solve",)))
    assert gw_init._zeta_reuse_ok(path, new, CENTS, print_fn=lambda *a: None)
    # distributed rerun over the same legacy stamp: refit.
    new_d = _prov(_cfg(bispinor=False, tier="distributed"))
    path = stub_reader(_strip(new_d, ("distributed_zeta_solve",)))
    msgs = []
    assert not gw_init._zeta_reuse_ok(path, new_d, CENTS,
                                      print_fn=msgs.append)
    assert any("distributed_zeta_solve" in m for m in msgs)


def test_tau_change_refits_within_family(stub_reader):
    old = _prov(_cfg(bispinor=True, family="rank_truncate", tau=1e-10))
    new = _prov(_cfg(bispinor=True, family="rank_truncate", tau=1e-8))
    path = stub_reader(old)
    assert not gw_init._zeta_reuse_ok(path, new, CENTS,
                                      print_fn=lambda *a: None)


def test_matching_stamp_reused(stub_reader):
    new = _prov(_cfg(bispinor=True, family="rank_truncate", tau=1e-10))
    path = stub_reader(new)
    assert gw_init._zeta_reuse_ok(path, new, CENTS, print_fn=lambda *a: None)


def test_provenance_collapse_nonbispinor_and_ridge_tau():
    # Non-bispinor: transverse keys are inert — provenance identical.
    a = _prov(_cfg(bispinor=False, family="ridge", tau=1e-10))
    b = _prov(_cfg(bispinor=False, family="rank_truncate", tau=1e-6))
    assert a == b
    # Bispinor + ridge: tau collapses to None (not read by ridge).
    c = json.loads(_prov(_cfg(bispinor=True, family="ridge", tau=1e-6)))
    assert c["transverse_zeta_solve"] == "ridge"
    assert c["transverse_zeta_rcond"] is None
    # Bispinor + rank_truncate records the tau.
    d = json.loads(_prov(_cfg(bispinor=True, family="rank_truncate",
                              tau=1e-6)))
    assert d["transverse_zeta_rcond"] == 1e-6


if __name__ == "__main__":
    # In-container convenience runner (pytest owns the fixtures).
    sys.exit(pytest.main([__file__, "-q"]))
