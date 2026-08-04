"""Reader matrix of the ζ-fit provenance stamp (gw_init).

The stamp records solve-family keys whose change must force a refit;
keys ADDED after old stamps were written must not (legacy exception).
Three generations of keys ride the same machinery:

* ``distributed_zeta_solve`` (2026-08-01, commit 8b02c72) — legacy
  implied value ``'replicated'``;
* ``transverse_zeta_solve`` / ``transverse_zeta_rcond`` (2026-08-01) —
  legacy implied values ``'ridge'`` / ``None``;
* ``n_rmu_transverse`` / ``centroids_transverse_md5`` /
  ``distributed_lu`` / ``transverse_solver_kind`` (2026-08-04, when ζ
  reuse was extended to bispinor runs) — legacy implied value ``None``
  for all four.

Matrix pinned here (pure logic — header reads are stubbed; the on-disk
h5 path is a real temp file so the existence check is genuine):

1. legacy stamp (missing ALL three of the first two generations) + a
   run requesting the legacy values → REUSE;
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
   to None (tau is not read by the ridge family);
6. TRANSVERSE CHANNEL identity (2026-08-04): every one of the four new
   keys forces a refit when it changes; a bispinor stamp cannot be
   built without them at all; a charge-only run over a stamp written
   before they existed still REUSES; a bispinor run over that same
   stamp REFITS; and the per-vertex stamps of one bispinor
   configuration differ in exactly ``vertex_mu_L`` and ``n_rmu``.
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
CENTS_T = np.arange(100, 118, dtype=np.int32).reshape(6, 3)


def _cfg(bispinor=True, family="ridge", tau=1e-10, tier="auto",
         distributed_lu="off"):
    backend = SimpleNamespace(
        zeta_ridge=0.0, zeta_rcond=1e-8,
        charge_zeta_solve="rank_truncate",
        distributed_zeta_solve=tier,
        distributed_lu=distributed_lu,
        transverse_zeta_solve=family,
        transverse_zeta_rcond=tau,
        gamma_contract_mode="take",
    )
    return SimpleNamespace(bispinor=bispinor, gspace_mode="host_cache",
                           backend=backend)


def _tid(cfg, cents_T=CENTS_T, solver_kind="lu"):
    """The transverse-identity dict fit_zeta's pre-flight builds."""
    if not cfg.bispinor:
        return None
    return {
        'n_rmu':          int(cents_T.shape[0]),
        'centroids_md5':  gw_init._centroid_table_md5(cents_T),
        'distributed_lu': str(cfg.backend.distributed_lu),
        'solver_kind':    str(solver_kind),
    }


def _prov(cfg, *, vertex_mu_L=0, n_rmu=None, transverse_identity=None,
          cents_T=CENTS_T, solver_kind="lu"):
    wfn = SimpleNamespace(_filename="", ecutwfc=30.0, ecutrho=120.0)
    meta = SimpleNamespace(
        n_rmu=(CENTS.shape[0] if n_rmu is None else int(n_rmu)),
        fft_grid=np.array([9, 9, 45]))
    if transverse_identity is None:
        transverse_identity = _tid(cfg, cents_T=cents_T,
                                   solver_kind=solver_kind)
    return gw_init._zeta_fit_provenance(
        wfn=wfn, meta=meta, cfg=cfg,
        band_range_left=(0, 26), band_range_right=(1, 128),
        zeta_cutoff=30.0, zeta_vcoul_cutoff=30.0,
        write_ibz_only=True, band_norms=None,
        vertex_mu_L=vertex_mu_L,
        transverse_identity=transverse_identity)


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


# ---------------------------------------------------------------------------
# Transverse-channel identity (2026-08-04, bispinor ζ reuse)
# ---------------------------------------------------------------------------

_T_KEYS = ("n_rmu_transverse", "centroids_transverse_md5",
           "distributed_lu", "transverse_solver_kind")


def test_bispinor_stamp_requires_transverse_identity():
    """A bispinor stamp that does not name its transverse basis is the
    exact hole this change set closes — refuse to build one."""
    cfg = _cfg(bispinor=True)
    with pytest.raises(ValueError) as exc:
        _prov(cfg, transverse_identity={})   # {} is falsy → treated as absent
    assert "transverse_identity" in str(exc.value)


def test_transverse_keys_present_for_bispinor_and_none_otherwise():
    b = json.loads(_prov(_cfg(bispinor=True)))
    assert b["n_rmu_transverse"] == CENTS_T.shape[0]
    assert b["centroids_transverse_md5"] == \
        gw_init._centroid_table_md5(CENTS_T)
    assert b["distributed_lu"] == "off"
    assert b["transverse_solver_kind"] == "lu"
    s = json.loads(_prov(_cfg(bispinor=False)))
    assert all(s[k] is None for k in _T_KEYS)


def test_transverse_centroid_table_change_refits(stub_reader):
    """Same COUNT, different POINTS: the count-only check passes and the
    md5 does not (pattern #10, transverse side)."""
    other = CENTS_T.copy()
    other[0, 0] += 1
    old = _prov(_cfg(bispinor=True), cents_T=CENTS_T)
    new = _prov(_cfg(bispinor=True), cents_T=other)
    assert json.loads(old)["n_rmu_transverse"] == \
        json.loads(new)["n_rmu_transverse"]
    path = stub_reader(old)
    msgs = []
    assert not gw_init._zeta_reuse_ok(path, new, CENTS, print_fn=msgs.append)
    assert any("centroids_transverse_md5" in m for m in msgs)


def test_transverse_count_change_refits(stub_reader):
    old = _prov(_cfg(bispinor=True), cents_T=CENTS_T)
    new = _prov(_cfg(bispinor=True), cents_T=CENTS_T[:4])
    path = stub_reader(old)
    msgs = []
    assert not gw_init._zeta_reuse_ok(path, new, CENTS, print_fn=msgs.append)
    assert any("n_rmu_transverse" in m for m in msgs)


def test_distributed_lu_change_refits(stub_reader):
    """A ζ_T fit under ScaLAPACK's block-cyclic LU is a different gauge
    from the in-tree per-q solve — KNOWN_LORRAX_ISSUES gw row."""
    old = _prov(_cfg(bispinor=True, distributed_lu="off"))
    new = _prov(_cfg(bispinor=True, distributed_lu="scalapack"),
                solver_kind="scalapack_lu")
    path = stub_reader(old)
    msgs = []
    assert not gw_init._zeta_reuse_ok(path, new, CENTS, print_fn=msgs.append)
    assert any("distributed_lu" in m for m in msgs)


def test_transverse_solver_kind_change_refits(stub_reader):
    """Same deck strings, different RESOLVED kind (the GPU `auto`
    mesh-shape case) still refits."""
    old = _prov(_cfg(bispinor=True, distributed_lu="auto"),
                solver_kind="lu")
    new = _prov(_cfg(bispinor=True, distributed_lu="auto"),
                solver_kind="cusolvermp_lu")
    path = stub_reader(old)
    msgs = []
    assert not gw_init._zeta_reuse_ok(path, new, CENTS, print_fn=msgs.append)
    assert any("transverse_solver_kind" in m for m in msgs)


def test_pre_20260804_stamp_reused_by_charge_run_refit_by_bispinor(
        stub_reader):
    """The legacy exception for the four new keys, both directions."""
    charge = _prov(_cfg(bispinor=False))
    path = stub_reader(_strip(charge, _T_KEYS))
    msgs = []
    assert gw_init._zeta_reuse_ok(path, charge, CENTS, print_fn=msgs.append)
    assert any("legacy stamp" in m for m in msgs)
    # Same on-disk stamp, but this run is bispinor: the stamp carries no
    # evidence about the ζ_T beside it, so it must refit.
    bisp = _prov(_cfg(bispinor=True))
    path = stub_reader(_strip(bisp, _T_KEYS))
    msgs = []
    assert not gw_init._zeta_reuse_ok(path, bisp, CENTS, print_fn=msgs.append)
    assert any("n_rmu_transverse" in m for m in msgs)


def test_added_key_equal_to_its_legacy_default_still_reuses(stub_reader):
    """REGRESSION (found by this suite on 2026-08-04, job 7888522).

    The legacy exception used to be expressed as a value-difference list
    over the key UNION.  A key ADDED with a value equal to its legacy
    default lands in NEITHER stamp's difference set — ``old.get(k)`` is
    ``None`` and ``new[k]`` is ``None`` — so the list came out empty
    while the JSON strings still differed, and an empty list fell
    through to the generic 'fit under DIFFERENT inputs' refit with an
    EMPTY 'Changed:' detail.  Every pre-2026-08-04 charge ζ would have
    been refit once, for nothing.  The exception is now keyed on the
    ADDED/DROPPED key sets, not on differing values.
    """
    charge = _prov(_cfg(bispinor=False))
    # All four added keys are None here, exactly equal to their legacy
    # defaults, so nothing about the fit changed.
    assert all(json.loads(charge)[k] is None for k in _T_KEYS)
    path = stub_reader(_strip(charge, _T_KEYS))
    msgs = []
    assert gw_init._zeta_reuse_ok(path, charge, CENTS, print_fn=msgs.append)
    assert any("legacy stamp" in m for m in msgs)
    assert not any("DIFFERENT inputs" in m for m in msgs)


def test_stamp_from_a_newer_lorrax_gets_no_exception(stub_reader):
    """A stamp carrying a key this build does not write came from a
    NEWER LORRAX; its meaning is unknown here, so no exception."""
    charge = _prov(_cfg(bispinor=False))
    d = json.loads(charge)
    d["some_future_key"] = "whatever"
    path = stub_reader(json.dumps(d, sort_keys=True))
    msgs = []
    assert not gw_init._zeta_reuse_ok(path, charge, CENTS,
                                      print_fn=msgs.append)
    assert any("some_future_key" in m for m in msgs)


def test_real_input_change_alongside_a_legacy_gap_still_refits(stub_reader):
    """A shared key that genuinely differs defeats the exception even
    when the only OTHER difference is a legacy gap."""
    old_cfg = _cfg(bispinor=False)
    old = json.loads(_prov(old_cfg))
    for k in _T_KEYS:
        old.pop(k, None)
    old["zeta_cutoff_ry"] = 40.0          # a real input change
    path = stub_reader(json.dumps(old, sort_keys=True))
    msgs = []
    assert not gw_init._zeta_reuse_ok(path, _prov(old_cfg), CENTS,
                                      print_fn=msgs.append)
    assert any("zeta_cutoff_ry" in m for m in msgs)


def test_per_vertex_stamps_differ_only_in_vertex_and_mu():
    """One bispinor configuration produces four stamps; they differ in
    exactly the two entries that make the four ζ files different."""
    cfg = _cfg(bispinor=True)
    charge = json.loads(_prov(cfg, vertex_mu_L=0))
    stamps = {L: json.loads(_prov(cfg, vertex_mu_L=L,
                                  n_rmu=CENTS_T.shape[0]))
              for L in (1, 2, 3)}
    for L, s in stamps.items():
        assert s["vertex_mu_L"] == L
        assert s["n_rmu"] == CENTS_T.shape[0]
        assert sorted(k for k in set(charge) | set(s)
                      if charge.get(k) != s.get(k)) == ["n_rmu",
                                                        "vertex_mu_L"]
    # ζ_T for μ_L=1 is not interchangeable with μ_L=2.
    assert stamps[1] != stamps[2]


def test_dataset_extent_probe_declines_reuse(stub_reader, tmp_path):
    """A good header over a ζ of the wrong μ extent must not be reused
    (the header and the dataset are written by separate calls, so one
    can be right while the other is stale).  Declines, never raises —
    that is the difference from ``_check_zeta_h5_matches_basis``."""
    import h5py
    new = _prov(_cfg(bispinor=False))
    stub_reader(new)          # installs the matching header stub
    quiet = lambda *a: None   # noqa: E731

    # Valid HDF5, correct header, no ζ dataset at all → refit.
    p_empty = str(tmp_path / "no_dataset.h5")
    with h5py.File(p_empty, "w") as f:
        f.create_group("isdf_header")
    msgs = []
    assert not gw_init._zeta_reuse_ok(p_empty, new, CENTS,
                                      print_fn=msgs.append,
                                      n_rmu_expected=CENTS.shape[0])
    assert any("NO ζ dataset" in m for m in msgs)

    # Wrong μ extent → refit; right extent → reuse.  The stub header is
    # the same in both cases, so only the dataset probe can tell them
    # apart.
    for extent, expect in ((CENTS.shape[0] + 1, False),
                           (CENTS.shape[0], True)):
        p2 = str(tmp_path / f"z{extent}.h5")
        with h5py.File(p2, "w") as f:
            f.create_dataset("zeta_q_G", shape=(2, extent, 5), dtype="f8")
        assert gw_init._zeta_reuse_ok(
            p2, new, CENTS, print_fn=quiet,
            n_rmu_expected=CENTS.shape[0]) is expect
        # Without the probe the wrong-extent file would be REUSED — that
        # is what the new argument buys.
        assert gw_init._zeta_reuse_ok(p2, new, CENTS, print_fn=quiet) is True


if __name__ == "__main__":
    # In-container convenience runner (pytest owns the fixtures).
    sys.exit(pytest.main([__file__, "-q"]))
