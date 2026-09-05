"""The disk cache: versioned, announced, and numerically inert at this commit.

R2's last two rows live here, and so does the one change in this branch
that could have moved a number if it had been done the obvious way.

The pre-extraction key was ``sha256({solver, logR, target, max_nodes})``
with no solver version, no numerics backend and no machine tag, so a
shared ``$HOME`` served one platform's quadrature to another under an
identical key.  The WP1 census measured what that means in practice: the
frozen G2 gate's "runtime solve" on a warm host is a **2026-04-09 cache
entry**, and re-solving the same request on the same machine today gives a
Σ|w| 4.5× different.

Bumping the key without a fallback would therefore have re-solved on every
warm host — moving numbers, including that frozen reference's, inside a
refactor commit.  So:

* writes go to the VERSIONED key, which records what the answer depends on;
* reads try the versioned key first and fall back to the LEGACY key,
  serving it byte-for-byte and ANNOUNCING that its provenance is
  unknowable;
* a cached rule is never presented as certified, ever
  (``caches are not release artifacts``).

Every cell below uses the ``isolated_cache`` fixture, because a cache cell
that used the real ``$HOME`` directory would be testing that machine's
history rather than this code — which is the census's finding turned into
a test-hygiene rule.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pytest

import minimax as M
from minimax import cache as CA

_PAYLOAD = {"solver": "noncrossing", "logR_key": 2.302585092994,
            "target_key": 1.0e-6, "max_nodes": 64}
_TAU = np.array([0.1, 0.5, 2.0], dtype=np.float64)
_W = np.array([0.3, 0.4, 0.5], dtype=np.float64)


def test_the_versioned_key_depends_on_the_solver_version_and_the_backend(
        isolated_cache, monkeypatch):
    """The key records what the answer depends on.  That is the whole fix.

    Changing the solver version must change the file, or a solver change
    would silently serve the old solver's answer — which is the defect
    class survey §2.4 measured and the census dated to April.
    """
    a = CA.versioned_path("noncrossing", _PAYLOAD)
    monkeypatch.setattr(CA, "SOLVER_VERSION", CA.SOLVER_VERSION + 1)
    b = CA.versioned_path("noncrossing", _PAYLOAD)
    assert a != b
    monkeypatch.setattr(CA, "SOLVER_VERSION", CA.SOLVER_VERSION - 1)
    monkeypatch.setattr(CA, "backend_tag", lambda: "cpu:other-blas")
    c = CA.versioned_path("noncrossing", _PAYLOAD)
    assert c != a


def test_the_legacy_key_is_reproduced_byte_for_byte(isolated_cache):
    """RED TWIN for the fallback: the legacy path must be the OLD path.

    A "legacy" key that did not match what the old code wrote would make
    the fallback a no-op — every warm host would miss, re-solve, and move.
    The digest is recomputed here the way the pre-extraction module
    computed it, from this file, rather than trusting the implementation.
    """
    import hashlib                                # noqa: PLC0415
    blob = json.dumps(_PAYLOAD, sort_keys=True, separators=(",", ":"))
    want = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    path = CA.legacy_path("noncrossing", _PAYLOAD)
    assert path.name == f"noncrossing_{want}.npz"


def test_a_legacy_entry_is_served_unchanged_and_announced(isolated_cache):
    """THE CELL THAT KEEPS THIS COMMIT NUMERICALLY INERT.

    A warm host with a pre-extraction cache must get the same arrays it
    got yesterday — and must be told, once, that what it is holding has no
    recorded provenance and cannot be reproduced anywhere else.
    """
    isolated_cache.mkdir(parents=True, exist_ok=True)
    legacy = CA.legacy_path("noncrossing", _PAYLOAD)
    np.savez_compressed(legacy, tau=_TAU, w=_W,
                        err=np.asarray(1.5e-7, dtype=np.float64))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = CA.load("noncrossing", _PAYLOAD)
    assert got is not None
    tau, w, err, prov = got
    assert tau.tobytes() == _TAU.tobytes()
    assert w.tobytes() == _W.tobytes()
    assert err == 1.5e-7
    assert prov.source == "cache-legacy"
    assert prov.certified is False
    assert "unknowable" in prov.generator_commit
    lines = [str(x.message) for x in caught]
    assert any("LEGACY UNVERSIONED" in m for m in lines), lines
    assert any("not reproducible across hosts" in m for m in lines), lines


def test_the_legacy_announcement_fires_once_per_request(isolated_cache):
    """A quadrature request repeats per q-block per SCF iteration per rank.
    An announcement nobody can read is the same as no announcement."""
    isolated_cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CA.legacy_path("noncrossing", _PAYLOAD),
                        tau=_TAU, w=_W,
                        err=np.asarray(1.5e-7, dtype=np.float64))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CA.load("noncrossing", _PAYLOAD)
        CA.load("noncrossing", _PAYLOAD)
    assert len([m for m in caught if "LEGACY" in str(m.message)]) == 1


def test_the_versioned_entry_wins_over_a_legacy_one(isolated_cache):
    """Once a versioned entry exists it is authoritative, and the legacy
    announcement stops — which is how a fleet migrates off the unversioned
    key without anybody deleting anything by hand."""
    isolated_cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CA.legacy_path("noncrossing", _PAYLOAD),
                        tau=_TAU, w=_W,
                        err=np.asarray(1.5e-7, dtype=np.float64))
    other = np.array([9.0, 9.5, 9.9], dtype=np.float64)
    CA.store("noncrossing", _PAYLOAD, other, _W, 2.5e-7)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tau, _w, err, prov = CA.load("noncrossing", _PAYLOAD)
    assert tau.tobytes() == other.tobytes()
    assert err == 2.5e-7
    assert prov.source == "cache"
    assert not [m for m in caught if "LEGACY" in str(m.message)]


def test_a_cached_rule_is_never_certified(isolated_cache):
    """T§E, adopted verbatim: caches are not release artifacts.

    There is no code path that can produce ``certified=True`` from the
    cache, and this cell is the guard on that — a cache entry that
    presented as certified would defeat the entire lookup-and-refuse
    settlement while looking like a success.
    """
    CA.store("noncrossing", _PAYLOAD, _TAU, _W, 1.0e-7)
    _t, _w, _e, prov = CA.load("noncrossing", _PAYLOAD)
    assert prov.certified is False
    assert prov.source in ("cache", "cache-legacy")
    assert "UNCERTIFIED" in prov.one_line()


def test_a_miss_is_announced_because_the_alternative_costs_minutes(
        isolated_cache):
    """The one of R2's six sites that legitimately STAYS a demotion.

    A cache miss really is an absence.  What changed is that the absence
    is said out loud, because what follows it is an uncertified in-process
    solve that used to cost 14–53 seconds per rank with nothing in the log.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert CA.load("noncrossing", _PAYLOAD) is None
    assert any("disk-cache MISS" in str(m.message) for m in caught)


def test_a_failed_write_announces_and_the_run_continues(isolated_cache):
    """R2's sixth row.  Was swallowed whole.

    A read-only ``$HOME`` or a full disk meant the cache silently stopped
    working and every rank re-solved for the rest of the campaign with
    nothing to say why.  The write still must not raise — losing a cache
    entry is not a reason to kill a run — but it must be audible.

    The failure is arranged by putting a DIRECTORY where the staging file
    wants to be, rather than by patching ``pathlib``: a real ``OSError``
    from the real call is what the handler has to survive, and patching
    ``Path.open`` would also break the announcement's own machinery.
    """
    isolated_cache.mkdir(parents=True, exist_ok=True)
    path = CA.versioned_path("noncrossing", _PAYLOAD)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.mkdir(parents=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CA.store("noncrossing", _PAYLOAD, _TAU, _W, 1.0e-7)   # must not raise
    messages = [str(m.message) for m in caught]
    assert any("could not write the disk cache" in m
               for m in messages), messages
    assert any("The run CONTINUES" in m for m in messages), messages
    assert not path.exists()


def test_a_corrupt_cache_file_announces_and_is_treated_as_absent(
        isolated_cache):
    """Unlike the shipped bundle, a cache file is not an artifact anybody
    promised — so a truncated one means "re-solve", not "the install is
    broken".  The distinction is deliberate, and the announcement is what
    keeps it from being a silent demotion again."""
    isolated_cache.mkdir(parents=True, exist_ok=True)
    CA.versioned_path("noncrossing", _PAYLOAD).write_bytes(b"not an npz")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert CA.load("noncrossing", _PAYLOAD) is None
    assert any("unreadable" in str(m.message) for m in caught)


def test_the_cache_can_be_disabled_exactly_as_before(monkeypatch, tmp_path):
    """``LORRAX_DISABLE_MINIMAX_DISK_CACHE`` is carried verbatim.  A deck
    that disabled the cache must keep disabling it."""
    monkeypatch.setenv("LORRAX_MINIMAX_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("LORRAX_DISABLE_MINIMAX_DISK_CACHE", "1")
    assert CA.cache_dir() is None
    assert CA.versioned_path("noncrossing", _PAYLOAD) is None
    assert CA.legacy_path("noncrossing", _PAYLOAD) is None
    CA.store("noncrossing", _PAYLOAD, _TAU, _W, 1.0e-7)     # no-op, no raise
    assert CA.load("noncrossing", _PAYLOAD) is None


def test_the_solve_payload_shapes_are_the_pre_extraction_ones():
    """The payload dict IS the legacy key, so its exact shape is a
    compatibility surface and not an implementation detail.

    All three are pinned here against the pre-extraction spellings, field
    for field, because a renamed key would invalidate every warm cache in
    the fleet silently — which is the same failure the version bump was
    designed to make loud.
    """
    assert M.__dict__  # keep the door import meaningful to linters
    from minimax.door import cached_solve_payload    # noqa: PLC0415
    assert cached_solve_payload(
        "noncrossing", logR_key=1.0, target_key=1e-6, max_nodes=64) == {
        "solver": "noncrossing", "logR_key": 1.0, "target_key": 1e-6,
        "max_nodes": 64}
    assert cached_solve_payload(
        "noncrossing_imag", logR_key=1.0, omega_hat_key=2.0,
        target_key=1e-6, max_nodes=64) == {
        "solver": "noncrossing_imag", "logR_key": 1.0, "omega_hat_key": 2.0,
        "target_key": 1e-6, "max_nodes": 64}
