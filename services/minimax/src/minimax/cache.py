"""The disk cache — versioned, self-describing, and never certified.

Design §3.4: *it rides, versioned, or it dies.*  Before the extraction the
key was ``sha256({solver, logR_key, target_key, max_nodes})`` with **no
solver version, no scipy/BLAS version and no machine tag**, so a shared
``$HOME`` served one platform's quadrature to another under an identical
key — and survey §2.4 proved those are different mathematical objects.

The WP1 census turned that from a latent hazard into a measured one: the
frozen G2 gate's "runtime solve" on a warm host is not a solve at all, it
is a **2026-04-09 cache entry**, and re-solving the same request on the
same machine today gives Σ|w| 4.22e4 against the cached 1.90e5.  The
cross-platform disagreement is four answers, not two, and one of them is
four months old.

THREE RULES, and the third is the one that keeps this branch numerically
inert:

1. **Writes are versioned.**  The key gains the solver version and a
   backend tag (the centroid-stamp idiom), and the payload gains the same
   provenance block every table carries, so a cached entry is
   self-describing.
2. **A cached rule is never presented as certified** (T§E, adopted
   verbatim: *caches are not release artifacts*).  Its provenance says
   ``source='cache'`` and ``certified=False``, always.
3. **Reads fall back to the legacy key, ANNOUNCED.**  Bumping the key
   without a fallback would silently re-solve on every warm host, which
   would move numbers — including the frozen G2 reference's — inside a
   refactor commit, which the phase forbids.  So a legacy hit is served
   exactly as before and says out loud that it is unversioned archaeology
   whose provenance is unknowable.  That is R2's shape applied to the one
   of the six sites that legitimately stays a demotion: a cache miss IS an
   absence, and it announces.

Rule 3 also means the escape hatch's numbers do not move at this commit
and the pathology becomes visible in every log instead of in a design
document.  Retiring the legacy read is a ratchet for the replumb, not a
refactor's side effect.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from minimax.records import Provenance

#: Bumped when a SOLVER BODY changes.  This is the field the pre-extraction
#: key was missing, and the reason survey §2.4's three-hosts-three-answers
#: measurement could hide inside one key.
SOLVER_VERSION = 1

#: Bumped when the cache FILE FORMAT changes.  Separate from the solver
#: version because they invalidate for different reasons.
CACHE_FORMAT_VERSION = 2

_LEGACY_ANNOUNCED: set[str] = set()
_MISS_ANNOUNCED: set[str] = set()


def backend_tag() -> str:
    """The measured numerics backend, as it goes into the key.

    numpy and scipy versions plus the machine, because those are what
    survey 2.4 measured moving the answer.  scipy is imported lazily
    and tolerated absent: a lookup-only process never has to have it,
    and a process about to solve will import it a moment later anyway.
    """
    try:
        import scipy                                   # noqa: PLC0415
        scipy_v = scipy.__version__
    except Exception:                                  # pragma: no cover
        # Genuinely broad, and it is not a demotion: this value is a CACHE
        # KEY COMPONENT, so "scipy could not be interrogated" must produce
        # a distinct, stable tag rather than an exception that would take
        # down a lookup that does not need scipy at all.
        scipy_v = "absent"
    return (f"cpu:numpy-{np.__version__}/scipy-{scipy_v}/"
            f"{os.uname().machine if hasattr(os, 'uname') else 'unknown'}")


def cache_dir() -> Path | None:
    """The persistent cache directory, or ``None`` when disabled.

    The two env keys are carried verbatim from the pre-extraction module —
    a deck that disabled the cache must keep disabling it.
    """
    off = os.environ.get("LORRAX_DISABLE_MINIMAX_DISK_CACHE", "")
    if off.strip().lower() in {"1", "true", "yes"}:
        return None
    d = os.environ.get("LORRAX_MINIMAX_CACHE_DIR")
    if not d:
        d = os.path.join(Path.home(), ".cache", "lorrax",
                         "minimax_quadratures")
    path = Path(d).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def legacy_path(namespace: str, payload: dict[str, Any]) -> Path | None:
    """The pre-extraction key, byte-for-byte.  Read-only from here on."""
    d = cache_dir()
    if d is None:
        return None
    return d / f"{namespace}_{_digest(payload)}.npz"


def versioned_path(namespace: str, payload: dict[str, Any]) -> Path | None:
    """The key that names what it depends on."""
    d = cache_dir()
    if d is None:
        return None
    keyed = dict(payload)
    keyed["_solver_version"] = SOLVER_VERSION
    keyed["_backend"] = backend_tag()
    return d / (f"{namespace}_v{CACHE_FORMAT_VERSION}_"
                f"{_digest(keyed)}.npz")


def _announce_once(bucket: set[str], key: str, message: str) -> None:
    """Announce per distinct request, not per call.

    A quadrature request repeats once per q-block per SCF iteration per
    rank; a per-call warning is a demotion nobody reads.
    """
    if key in bucket:
        return
    bucket.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def load(namespace: str, payload: dict[str, Any]
         ) -> tuple[np.ndarray, np.ndarray, float, Provenance] | None:
    """A cached rule with its provenance, or ``None`` — announced either way.

    R2's fifth row.  This is the only one of the six sites that stays a
    demotion, because a cache miss genuinely IS an absence; what changes is
    that the absence is said out loud.
    """
    key = f"{namespace}:{_digest(payload)}"
    vpath = versioned_path(namespace, payload)
    if vpath is not None and vpath.exists():
        got = _read(vpath)
        if got is not None:
            tau, w, err = got
            return tau, w, err, Provenance(
                source="cache",
                catalog_entry=str(vpath),
                table_hash=_payload_hash(tau, w),
                generator_commit="n/a (solved in-process)",
                generation_backend=backend_tag(),
                certified=False)

    lpath = legacy_path(namespace, payload)
    if lpath is not None and lpath.exists():
        got = _read(lpath)
        if got is not None:
            tau, w, err = got
            _announce_once(
                _LEGACY_ANNOUNCED, key,
                f"minimax: served a LEGACY UNVERSIONED disk-cache entry for "
                f"{namespace} {payload!r} from {str(lpath)!r}.  That key "
                f"records no solver version, no numerics backend and no "
                f"machine, so WHEN and WHERE this rule was computed is "
                f"unknowable from the artifact — the WP1 census found a "
                f"four-month-old entry serving a frozen gate this way.  It "
                f"is uncertified and not reproducible across hosts.  Delete "
                f"{str(lpath)!r} to force a versioned re-solve.")
            return tau, w, err, Provenance(
                source="cache-legacy",
                catalog_entry=str(lpath),
                table_hash=_payload_hash(tau, w),
                generator_commit="unknowable (unversioned cache key)",
                generation_backend="unknowable (unversioned cache key)",
                certified=False)

    _announce_once(
        _MISS_ANNOUNCED, key,
        f"minimax: disk-cache MISS for {namespace} {payload!r} "
        f"(looked for {str(vpath)!r}).  A miss is a legitimate absence; it "
        f"is announced because the alternative — an uncertified in-process "
        f"solve — costs minutes and used to happen silently.")
    return None


def _read(path: Path) -> tuple[np.ndarray, np.ndarray, float] | None:
    """One cache file.  A corrupt file announces and is treated as absent.

    Deliberately NOT a refusal: unlike the shipped bundle, a cache file is
    not an artifact anybody promised, so a truncated one means "re-solve",
    not "the install is broken".  What changed is that it says so.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            tau = np.asarray(data["tau"], dtype=np.float64)
            w = np.asarray(data["w"], dtype=np.float64)
            err = float(data["err"][()])
        return tau, w, err
    except (OSError, ValueError, KeyError, EOFError) as exc:
        warnings.warn(
            f"minimax: disk-cache entry {str(path)!r} is unreadable "
            f"({type(exc).__name__}: {exc}); treating it as absent and "
            f"re-solving.  Delete it if this repeats.",
            RuntimeWarning, stacklevel=3)
        return None


def _payload_hash(tau: np.ndarray, w: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tau, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(w).tobytes())
    return f"sha256:{h.hexdigest()[:16]}"


def store(namespace: str, payload: dict[str, Any],
          tau: np.ndarray, w: np.ndarray, err: float) -> None:
    """Write the versioned entry.  A write failure ANNOUNCES; it never hides.

    R2's sixth row.  Before, a failing write was swallowed whole, so a
    read-only ``$HOME`` or a full disk meant the cache silently stopped
    working and every rank re-solved for the rest of the campaign with
    nothing in the log to say why.
    """
    path = versioned_path(namespace, payload)
    if path is None:
        return
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                tau=np.asarray(tau, dtype=np.float64),
                w=np.asarray(w, dtype=np.float64),
                err=np.asarray(float(err), dtype=np.float64),
                solver_version=np.asarray(SOLVER_VERSION, dtype=np.int64),
                backend=np.asarray(backend_tag()),
                certified=np.asarray(False))
        os.replace(tmp, path)
    except OSError as exc:
        warnings.warn(
            f"minimax: could not write the disk cache entry {str(path)!r}: "
            f"{type(exc).__name__}: {exc}.  The run CONTINUES and the rule "
            f"just computed is used — but the cache is not working, so every "
            f"subsequent request for it will pay the solve again, on every "
            f"rank.  This used to be swallowed.",
            RuntimeWarning, stacklevel=3)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            # The cleanup of a failed write is the one place a swallow is
            # right: we are already announcing the real failure above, and
            # a second exception here would replace that message with a
            # less informative one.
            pass


def reset_announcements() -> None:
    """Test hook: forget what has already been announced."""
    _LEGACY_ANNOUNCED.clear()
    _MISS_ANNOUNCED.clear()
