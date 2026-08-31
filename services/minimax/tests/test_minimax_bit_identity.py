"""Bit-identity: the bytes the door hands out are the bytes on disk.

The extraction's central promise is that it MOVED code without moving a
number, and "bit-identical by construction" is only a construction if
something measures it.  Two claims here, and they are different:

1. **The payload is not touched.**  Every array the door serves is
   ``np.load``'s output, cast exactly as the pre-extraction loader cast
   it, with no rescaling, rounding or re-solving in between.  Asserted on
   the raw bytes rather than with a tolerance — a tolerance would pass a
   loader that had quietly started rounding.
2. **The selection rule chooses the same table.**  The WP1 census measured
   54 distinct requests across the frozen decks, the campaign decks and
   the suite, of which 51 were served, touching a small pinned subset of the
   shipped tables.  Those six, and the requests that reach them, are
   pinned here by name.  If the selection rule drifts, this file names the
   request that moved and the table it moved to.

The six-table fact is worth pinning for its own sake: it is the measured
answer to "how much of this bundle is load-bearing", and it is what makes
the certification campaign (WP6) affordable.
"""

from __future__ import annotations

import numpy as np
import pytest

import minimax as M
from minimax import _catalog as C

#: WP1 census §2.6 — the only six shipped tables the entire measured
#: surface ever loads, with the load counts it measured.  The counts are
#: not asserted (they are a property of the decks, not of this service);
#: the SET is.
_CENSUS_TABLES = {
    "noncrossing/noncrossing_R_10p000000_eps_1p0em06.npz": 47,
    "noncrossing/noncrossing_R_46p415888_eps_1p0em06.npz": 10,
    "noncrossing/noncrossing_R_21p544347_eps_1p0em06.npz": 8,
    "crossing/crossing_hgl_A_24p000000_eps_1p8em08_epsq_1p0em03.npz": 6,
}

_CAUSAL_PAYLOAD_SHA256 = {
    "crossing_causal/crossing_causal_A_10p000000_G_100p000000_eps_1p0em04.npz":
        "61483406d5cd05b55e4d4bf179d1dcc524894c950d578c993280ed7941b07966",
    "crossing_causal/crossing_causal_A_10p000000_G_100p000000_eps_1p0em05.npz":
        "6b22b610c1cdd0a3e15d44697f856bdaa470c9f121b70883eee91b83c5087a71",
    "crossing_causal/crossing_causal_A_20p000000_G_100p000000_eps_1p0em04.npz":
        "6c3b24ae7ac8d7426ec0ac946818ed96ab8b529d6bea76b5860c7500ce4f10eb",
    "crossing_causal/crossing_causal_A_20p000000_G_100p000000_eps_1p0em05.npz":
        "1b9b72e32392513240880247b8572b1aa2983e7fe5266fc9185bd4b3b40feef2",
    "crossing_causal/crossing_causal_A_40p000000_G_100p000000_eps_1p0em04.npz":
        "3f65843794eda55312b04d58f01278d69a1c555e22fb38222eaa711f3bed85cf",
    "crossing_causal/crossing_causal_A_40p000000_G_100p000000_eps_1p0em05.npz":
        "6fafaac3e9a2589b856b3f9a7c2ec8972f3bbeb533507138feca48f31ce8b8b8",
}

#: Requests taken verbatim from the census's deck and suite tables, with
#: the table each one resolved to BEFORE the extraction.
#:
#: ``(family, target, range_value, error_bound, n_max, eps_q) -> file``
_CENSUS_REQUESTS = [
    # cohsex_debug / cohsex_test.in and its selfcheck twin: R = 37.1173
    (("noncrossing", "inverse", 37.1173, 1.0e-6, 64, None),
     "noncrossing/noncrossing_R_46p415888_eps_1p0em06.npz"),
    # the sub-10 suite requests, which round UP to the R = 10 table
    (("noncrossing", "inverse", 1.0045, 1.0e-6, 64, None),
     "noncrossing/noncrossing_R_10p000000_eps_1p0em06.npz"),
    (("noncrossing", "inverse", 10.0, 1.0e-6, 64, None),
     "noncrossing/noncrossing_R_10p000000_eps_1p0em06.npz"),
    # the middle bucket
    (("noncrossing", "inverse", 15.0, 1.0e-6, 64, None),
     "noncrossing/noncrossing_R_21p544347_eps_1p0em06.npz"),
    # the production crossing request, which pins at A_core = 24.0 exactly
    # for as long as the conditioning floor stays engaged (census §2.3)
    (("crossing", "hgl", 24.0, 1.0e-6, 500, 1.0e-3),
     "crossing/crossing_hgl_A_24p000000_eps_1p8em08_epsq_1p0em03.npz"),
]


def _load_raw(rel: str):
    """The payload, read the way the pre-extraction loader read it."""
    path = C._asset_root().joinpath(rel)
    with path.open("rb") as fh:
        with np.load(fh, allow_pickle=False) as data:
            alpha = np.asarray(data["alpha"])
            if not np.iscomplexobj(alpha):
                alpha = np.asarray(alpha, dtype=np.float64)
            return (np.asarray(data["tau"], dtype=np.float64), alpha,
                    float(data["max_error"][()]))


@pytest.mark.parametrize("request_, expected_file", _CENSUS_REQUESTS,
                         ids=[r[1].split("/")[-1][:38]
                              for r in _CENSUS_REQUESTS])
def test_the_census_requests_resolve_to_the_same_tables_as_before(
        request_, expected_file):
    family, target, range_value, error_bound, n_max, eps_q = request_
    kw = {"eps_q": eps_q} if eps_q is not None else {}
    q = M.lookup(family=family, target=target, range_value=range_value,
                 error_bound=error_bound, n_max=n_max, **kw)
    assert q.provenance.catalog_entry == expected_file


@pytest.mark.parametrize("request_, expected_file", _CENSUS_REQUESTS,
                         ids=[r[1].split("/")[-1][:38]
                              for r in _CENSUS_REQUESTS])
def test_the_served_arrays_are_the_payload_bytes(request_, expected_file):
    """BYTE identity, not ``allclose``.

    ``tobytes()`` on both sides is the assertion, because the failure this
    guards against is a loader that starts normalising, sorting or
    round-tripping through a different dtype — every one of which passes a
    tolerance and changes a frozen reference.
    """
    family, target, range_value, error_bound, n_max, eps_q = request_
    kw = {"eps_q": eps_q} if eps_q is not None else {}
    q = M.lookup(family=family, target=target, range_value=range_value,
                 error_bound=error_bound, n_max=n_max, **kw)
    tau, alpha, err = _load_raw(expected_file)
    assert q.nodes.tobytes() == tau.tobytes()
    assert q.weights.tobytes() == alpha.tobytes()
    assert q.max_error == err
    assert q.nodes.dtype == np.float64 and q.weights.dtype == np.float64


def test_the_served_error_is_the_payloads_and_not_the_catalogs_claim():
    """The catalog CLAIMS a ``max_error`` and the payload CARRIES one.

    Serving the claim instead of the payload would be invisible today —
    they agree — and would be exactly the substitution this whole service
    exists to prevent: an advertised number standing in for a measured
    one.  So the door reads the payload, and this cell is what keeps that
    from being quietly reversed.
    """
    view = M.catalog()
    entry = next(e for e in view.entries
                 if e.file.endswith("noncrossing_R_10p000000_eps_1p0em06.npz"))
    _tau, _alpha, err, _k, _h = C.load_table(entry)
    assert entry.claimed_max_error is not None
    assert err == entry.claimed_max_error       # they agree TODAY
    q = M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
    _t2, _a2, raw_err = _load_raw(entry.file)
    assert q.max_error == raw_err               # and the payload is the source


def test_the_measured_surface_stays_inside_its_pinned_tables():
    """The census's most useful single number, pinned.

    Everything the frozen decks, the campaign decks and the suite ever
    load is six of the 31 shipped tables — which is what makes certifying
    the load-bearing part of this bundle a small job rather than a
    campaign, and it is the fact the WP6 tier is sized against.
    """
    served = set()
    for (family, target, range_value, error_bound, n_max, eps_q), _f in \
            _CENSUS_REQUESTS:
        kw = {"eps_q": eps_q} if eps_q is not None else {}
        q = M.lookup(family=family, target=target, range_value=range_value,
                     error_bound=error_bound, n_max=n_max, **kw)
        served.add(q.provenance.catalog_entry)
    assert served <= set(_CENSUS_TABLES), served - set(_CENSUS_TABLES)
    assert len(M.catalog()) == 40


@pytest.mark.parametrize("file, expected_sha", _CAUSAL_PAYLOAD_SHA256.items(),
                         ids=lambda value: value.split("/")[-1][:42])
def test_every_causal_crossing_table_is_bit_pinned(file, expected_sha):
    """Every new complex payload is selected and served byte-for-byte."""
    entry = next(e for e in M.catalog().for_family("crossing_causal")
                 if e.file == file)
    q = M.lookup(
        family="crossing_causal", target="causal_reciprocal",
        range_value=entry.range_max, error_bound=entry.error_bound,
        n_max=entry.node_count)
    tau, alpha, max_error = _load_raw(file)
    assert q.provenance.catalog_entry == file
    assert q.nodes.tobytes() == tau.tobytes()
    assert q.weights.tobytes() == alpha.tobytes()
    assert q.nodes.dtype == np.float64
    assert q.weights.dtype == np.complex128
    assert q.max_error == max_error
    assert C.payload_sha256(q.nodes, q.weights) == expected_sha
    assert entry.raw["payload_sha256"] == expected_sha
    assert q.provenance.certified is True


def test_the_table_hash_is_stable_across_reads():
    """The cross-machine promise is about the ARTIFACT, so the hash has to
    be a property of the bytes and not of the process that read them."""
    q1 = M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                  error_bound=1.0e-6, n_max=64)
    M.clear_caches()
    q2 = M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                  error_bound=1.0e-6, n_max=64)
    assert q1.provenance.table_hash == q2.provenance.table_hash


def test_a_corrupted_payload_changes_the_hash(tmp_path, monkeypatch):
    """RED TWIN for the hash.

    A digest that never changes is indistinguishable from a constant, and
    a constant would make every cross-machine identity claim in this
    service vacuous.  One flipped byte must move it.
    """
    root = tmp_path / "minimax_assets"
    (root / "noncrossing").mkdir(parents=True)
    src = C._asset_root().joinpath(
        "noncrossing/noncrossing_R_10p000000_eps_1p0em06.npz")
    with src.open("rb") as fh:
        blob = bytearray(fh.read())
    (root / "noncrossing" / "good.npz").write_bytes(bytes(blob))
    blob[-1] ^= 0x01
    (root / "noncrossing" / "flipped.npz").write_bytes(bytes(blob))
    monkeypatch.setattr(C, "_asset_root", lambda: root)
    C.clear_caches()
    a = C._sha256_of(root / "noncrossing" / "good.npz")
    b = C._sha256_of(root / "noncrossing" / "flipped.npz")
    assert a != b
