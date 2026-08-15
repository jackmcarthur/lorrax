"""One HDF5 library instance per open file — the registry that enforces it.

WHY THIS MODULE EXISTS.  In the deployed Perlmutter run environment this
process maps **two independent HDF5 library instances**:

* the h5py wheel's bundled ``h5py.libs/libhdf5-<hash>.so.320.0.0``
  (serial, HDF5 2.0.0), and
* the FFI transport's ``/opt/cray/pe/lib64/libhdf5_parallel_gnu_123.so.200``
  (cray parallel HDF5), pulled in by ``liblorrax_ffi.so``.

Measured 2026-08-15 by reading the deployed artifacts' ``NEEDED`` entries;
the two carry different sonames, so both load, and each keeps its **own**
metadata cache, its own open-file table and its own free-space manager.
Neither can see the other's state.  Two such instances alternately
writing and reading ONE file is undefined behaviour, and this tree has
already paid for it twice:

* ``_slab_io_ffi._dataset_geom`` carries the measured note that an
  unconditional serial-h5py introspect on a file open for collective
  MPI-IO writing reads a superblock that is not durable yet and fails
  with "file signature not found" (job 7888644);
* the metallic MPA-QSGW driver segfaults (exit 139, native, rank 0 — the
  only serial-h5py writer — first) in iteration 3, immediately after a
  fit-store ``SlabIO.close``.  Static audit A1
  (``reports/metal_mpa_plan_2026-08-15/METALLIC_INVARIANT_AUDIT.md``)
  names the per-iteration store lifecycle as the standing explanation;
  the sandbox recorded the hazard class as claims/0110 ("two mapped
  HDF5 objects... resolution succeeds and tells you nothing").

Three call sites in this package already avoid the hazard **by ordering**
and say so in prose (``_slab_io_ffi.close``'s deferred rank-0 reopen,
``tagged_arrays``' "SlabIO has released the file and no other writer may
open it between these two statements", ``mpa_store``'s barrier-separated
allocate).  That is a convention every new call site has to rediscover.
This module turns it into an invariant enforced at the door.

WHAT IS ENFORCED, AND WHY EXACTLY THAT
--------------------------------------
**Refused (hard, always):** an open of a path through one stack while the
other stack holds a live handle on the same path AND either side can
write.  That is the undefined case — one instance's dirty metadata cache
against another instance's stale one — and it is the shape both measured
failures above have.

**Allowed, counted:** cross-stack *read-only* concurrency.  Two libraries
reading a file nobody is mutating cannot diverge; ``WFN.h5`` is held open
read-only by both ``wfn_loader``'s h5py handle and its SlabIO handle for
the whole run, and that is fine.  Refusing it would refuse the pipeline
for no defect.

**Measured, escalatable:** *sequential* cross-stack alternation where a
write happened on either side — open/close/open/close through different
libraries.  Every close flushes, so this mostly works, which is why the
pipeline runs today; it is still two caches taking turns on one file and
it is what audit A1 wants gone.  It is counted per path and reported, and
``LORRAX_HDF5_ONE_OWNER=strict`` promotes it to the same refusal so the
remaining sites can be enumerated in one run.  The permanent fix is one
owner per file — see the sibling-split spec in the audit.

The registry is per PROCESS and in-memory: it says nothing about other
ranks, and it is not a lock.  It answers exactly one question — "has this
path already been opened through the other HDF5 library in THIS process,
and is that handle still live" — which is the question neither library
can answer for itself.
"""
from __future__ import annotations

import contextlib
import os
import threading

__all__ = [
    "STACK_FFI",
    "STACK_H5PY",
    "alternation_rows",
    "is_write_mode",
    "mapped_hdf5_libraries",
    "note_close",
    "note_open",
    "open_scope",
    "policy",
    "probe",
    "reset_for_test",
    "shared_paths",
]

#: The bundled-libhdf5 h5py wheel.
STACK_H5PY = "h5py"
#: cray parallel HDF5, reached through ``liblorrax_ffi.so``/``SlabIO``.
STACK_FFI = "ffi"

_STACK_LIB = {
    STACK_H5PY: "h5py's bundled libhdf5",
    STACK_FFI: "the FFI's cray libhdf5_parallel",
}

# HDF5 open modes that can mutate the file.  "r" is the only one that
# cannot; anything unrecognised is treated as a writer, because guessing
# "read-only" wrong is the direction that loses data.
_READ_ONLY_MODES = frozenset({"r"})

#: ``measure`` (default) counts sequential cross-stack alternation and
#: reports it; ``strict`` refuses it.  The live-overlap-with-a-writer
#: refusal is unconditional under both.
POLICY_ENV = "LORRAX_HDF5_ONE_OWNER"
_POLICIES = ("measure", "strict")

_LOCK = threading.RLock()
# abspath -> list of live handle dicts
_LIVE: dict[str, list[dict]] = {}
# abspath -> counters
_HISTORY: dict[str, dict] = {}
_TOKEN = [0]

# HDF5_USE_FILE_LOCKING IS NOT SET HERE.  It is a variable LORRAX ships,
# and ``runtime.set_default_env`` is the one place that decides those —
# a module that sets one just by being imported is what
# ``tests/test_layering.py::test_only_entry_points_write_the_environment_at_import``
# refuses, and it is right to: an env default applied at import time
# depends on import ORDER, which is exactly the property an HDF5 knob
# read at file-open time must not depend on.  This module only REPORTS
# the setting, in :func:`probe`.


def policy() -> str:
    """The alternation policy in force, from :data:`POLICY_ENV`."""
    got = str(os.environ.get(POLICY_ENV, "measure")).strip().lower()
    if got not in _POLICIES:
        raise ValueError(
            f"{POLICY_ENV}={got!r} is not one of {_POLICIES}.  "
            f"'measure' counts cross-stack alternation on one file and "
            f"reports it; 'strict' refuses it.  The live-overlap refusal "
            f"is unconditional either way.")
    return got


def is_write_mode(mode) -> bool:
    """True unless ``mode`` is HDF5's one read-only mode."""
    return str(mode).strip() not in _READ_ONLY_MODES


def _norm(path) -> str:
    return os.path.abspath(os.fspath(path))


def _record(path: str) -> dict:
    rec = _HISTORY.get(path)
    if rec is None:
        rec = {
            "opens": {STACK_H5PY: 0, STACK_FFI: 0},
            "writes": {STACK_H5PY: 0, STACK_FFI: 0},
            "last_stack": None,
            "alternations": 0,
            "alternations_with_write": 0,
        }
        _HISTORY[path] = rec
    return rec


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def _overlap_refusal(path, stack, mode, where, live) -> str:
    other = [h for h in live if h["stack"] != stack]
    held = "; ".join(
        f"mode={h['mode']!r} from {h['where']}" for h in other) or "?"
    writers = [h for h in other if is_write_mode(h["mode"])]
    who = ("this open" if is_write_mode(mode)
           else "the live " + other[0]["stack"] + " handle")
    return (
        "LORRAX HDF5 one-owner-per-file refusal (audit A1; sandbox "
        "claims/0110)\n"
        f"  file  : {path}\n"
        f"  got   : {stack} open(mode={mode!r}) from {where}, while "
        f"{other[0]['stack']} holds {len(other)} live handle(s) on the "
        f"same path [{held}] — and {who} can write.\n"
        f"  want  : at most ONE HDF5 library instance holding a live "
        f"handle on a file whenever either side can write.  This process "
        f"maps {_STACK_LIB[STACK_H5PY]} and {_STACK_LIB[STACK_FFI]} as "
        f"separate objects; each has its own metadata cache and open-file "
        f"table, so the writer's flush lands on top of a cache that never "
        f"saw the other's changes.  Measured consequences on this stack: "
        f"'file signature not found' (job 7888644) and the iteration-3 "
        f"rank-0 segfault A1 names.\n"
        f"  fix   : close the {other[0]['stack']} handle before opening "
        f"this one — the ordering the callers around it already use "
        f"(_slab_io_ffi.close defers its rank-0 h5py reopen until after "
        f"H5Fclose; tagged_arrays stamps only once SlabIO has released "
        f"the file).  If the two really must be concurrent, they must not "
        f"be the same file: give the metadata a sibling touched by one "
        f"library only."
        + ("" if writers else
           "\n  note  : the incoming open is the writer; the live handles "
           "are read-only.")
    )


def _alternation_refusal(path, stack, mode, where, rec) -> str:
    return (
        "LORRAX HDF5 one-owner-per-file refusal, strict policy "
        "(audit A1; sandbox claims/0110)\n"
        f"  file  : {path}\n"
        f"  got   : {stack} open(mode={mode!r}) from {where}; this file was "
        f"last opened through {rec['last_stack']}, and has been written "
        f"through {'h5py' if rec['writes'][STACK_H5PY] else 'the FFI'} "
        f"already (h5py: {rec['opens'][STACK_H5PY]} opens / "
        f"{rec['writes'][STACK_H5PY]} writes; FFI: "
        f"{rec['opens'][STACK_FFI]} opens / {rec['writes'][STACK_FFI]} "
        f"writes; {rec['alternations']} stack alternations so far).\n"
        f"  want   : one HDF5 library instance per file for the life of "
        f"the process.  Sequential alternation flushes at each close and "
        f"so usually survives, which is why it is DEFAULT-ALLOWED and "
        f"merely counted; you asked for {POLICY_ENV}=strict, which is the "
        f"target state audit A1 names.\n"
        f"  fix   : move this file's small metadata to a sibling file "
        f"touched only by h5py, leaving the payload touched only by the "
        f"FFI — or set {POLICY_ENV}=measure to go back to counting."
    )


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

def note_open(path, stack, mode, *, where: str) -> int:
    """Declare an HDF5 open.  Returns a token for :func:`note_close`.

    Refuses a cross-stack overlap that includes a writer, always; refuses
    sequential cross-stack alternation under ``strict``.
    """
    if stack not in _STACK_LIB:
        raise ValueError(f"unknown HDF5 stack {stack!r}")
    key = _norm(path)
    with _LOCK:
        live = _LIVE.get(key, [])
        foreign = [h for h in live if h["stack"] != stack]
        if foreign and (is_write_mode(mode)
                        or any(is_write_mode(h["mode"]) for h in foreign)):
            raise RuntimeError(
                _overlap_refusal(key, stack, mode, where, live))
        rec = _record(key)
        if rec["last_stack"] not in (None, stack):
            rec["alternations"] += 1
            wrote = (rec["writes"][STACK_H5PY] or rec["writes"][STACK_FFI]
                     or is_write_mode(mode))
            if wrote:
                rec["alternations_with_write"] += 1
                if policy() == "strict":
                    raise RuntimeError(
                        _alternation_refusal(key, stack, mode, where, rec))
        rec["opens"][stack] += 1
        if is_write_mode(mode):
            rec["writes"][stack] += 1
        rec["last_stack"] = stack
        _TOKEN[0] += 1
        token = _TOKEN[0]
        _LIVE.setdefault(key, []).append(
            {"stack": stack, "mode": str(mode), "where": str(where),
             "token": token})
        return token


def note_close(path, token: int) -> None:
    """Release a handle declared by :func:`note_open`.  Never raises."""
    key = _norm(path)
    with _LOCK:
        live = _LIVE.get(key)
        if not live:
            return
        _LIVE[key] = [h for h in live if h["token"] != token]
        if not _LIVE[key]:
            del _LIVE[key]


@contextlib.contextmanager
def open_scope(path, stack, mode, *, where: str):
    """Declare an open for the duration of a ``with`` block.

    The release is in a ``finally``: a handle whose open raised, or whose
    body raised, must not leave the registry believing it is still live —
    that would refuse every later legitimate open on the path.
    """
    token = note_open(path, stack, mode, where=where)
    try:
        yield
    finally:
        note_close(path, token)


# ---------------------------------------------------------------------------
# The measurement (GATE-7 style)
# ---------------------------------------------------------------------------

#: HDF5 ships companion libraries beside the core one, and they are NOT
#: separate library instances: they are thin wrappers that call into the
#: core object they were built against.  Counting them would make a
#: perfectly safe single-stack process (h5py's wheel alone maps both
#: ``libhdf5-<hash>.so`` and ``libhdf5_hl-<hash>.so``) report TWO, and the
#: escalation in :func:`probe` keys on that count — so the distinction is
#: load-bearing, not cosmetic.  Anything else beginning ``libhdf5`` is a
#: core build, including cray's ``libhdf5_parallel_gnu_123.so``.
_HDF5_COMPANION_PREFIXES = (
    "libhdf5_hl", "libhdf5_cpp", "libhdf5_hl_cpp", "libhdf5hl_fortran",
    "libhdf5_fortran", "libhdf5_java", "libhdf5_tools",
)


def _mapped_hdf5_objects():
    """``(core, companion)`` libhdf5 basenames mapped into this process."""
    try:
        with open("/proc/self/maps", "r") as fh:
            lines = fh.readlines()
    except OSError:
        return [], []
    core, companion = set(), set()
    for line in lines:
        cut = line.rfind(" /")
        if cut < 0:
            continue
        base = os.path.basename(line[cut + 1:].strip())
        low = base.lower()
        if not (low.startswith("libhdf5") and ".so" in low):
            continue
        if low.startswith(_HDF5_COMPANION_PREFIXES):
            companion.add(base)
        else:
            core.add(base)
    return sorted(core), sorted(companion)


def mapped_hdf5_libraries() -> list[str]:
    """Distinct CORE libhdf5 objects mapped into THIS process, sorted.

    Reads ``/proc/self/maps``.  Companions (``libhdf5_hl`` and friends)
    are excluded — see :data:`_HDF5_COMPANION_PREFIXES`; they are
    reported separately by :func:`probe`.  Returns ``[]`` where
    ``/proc/self/maps`` does not exist (non-Linux); an empty list means
    "not measurable here", not "safe", and the printed line says so.
    """
    return _mapped_hdf5_objects()[0]


def shared_paths() -> list[str]:
    """Paths this process has opened through BOTH stacks, worst first."""
    with _LOCK:
        rows = [
            (p, r) for p, r in _HISTORY.items()
            if r["opens"][STACK_H5PY] and r["opens"][STACK_FFI]
        ]
    rows.sort(key=lambda pr: -pr[1]["alternations"])
    return [p for p, _ in rows]


def alternation_rows() -> list[tuple[str, int, int, int, int, int]]:
    """``(path, h5py_opens, ffi_opens, alternations, alt_with_write, writes)``."""
    with _LOCK:
        rows = [
            (p,
             r["opens"][STACK_H5PY], r["opens"][STACK_FFI],
             r["alternations"], r["alternations_with_write"],
             r["writes"][STACK_H5PY] + r["writes"][STACK_FFI])
            for p, r in _HISTORY.items()
        ]
    rows.sort(key=lambda row: -row[3])
    return rows


def probe(where: str, *, print_fn=print, escalate: bool = True) -> dict:
    """Print the mapped-libhdf5 count and the registry's shared paths.

    MEASURED, NOT ASSERTED, by default: this prints what is true and
    returns it.  ``escalate`` only decides whether the *strict* policy is
    allowed to turn the measured unsafe condition — two mapped libhdf5
    objects AND a path this process has written through both — into the
    same refusal :func:`note_open` raises.  Under the default ``measure``
    policy nothing is raised; the line is the deliverable.
    """
    libs, companions = _mapped_hdf5_objects()
    rows = alternation_rows()
    both = [r for r in rows if r[1] and r[2]]
    unsafe = [r for r in both if r[4]]
    locking = os.environ.get("HDF5_USE_FILE_LOCKING", "<unset>")

    if not libs:
        shown = "unmeasurable (no /proc/self/maps)"
    else:
        shown = f"{len(libs)} core mapped: " + ", ".join(libs)
        if companions:
            shown += (f" (+{len(companions)} companion: "
                      + ", ".join(companions) + ")")
    print_fn(f"  [hdf5-probe {where}] libhdf5 {shown}; "
             f"HDF5_USE_FILE_LOCKING={locking}; policy={policy()}")
    if both:
        worst = both[0]
        print_fn(
            f"  [hdf5-probe {where}] {len(both)} path(s) opened through "
            f"BOTH stacks, {len(unsafe)} of them with a write; "
            f"worst: {os.path.basename(worst[0])} "
            f"(h5py {worst[1]} opens, FFI {worst[2]}, "
            f"{worst[3]} alternations)")
    else:
        print_fn(f"  [hdf5-probe {where}] no path opened through both "
                 f"stacks so far")
    if len(libs) >= 2 and unsafe:
        # The condition audit A1 describes, now measured rather than
        # inferred.  Loud on every iteration it holds, because it is the
        # standing explanation for the iteration-3 segfault.
        print_fn(
            f"  [hdf5-probe {where}] UNSAFE-BY-A1: two HDF5 library "
            f"instances are mapped AND {len(unsafe)} file(s) are written "
            f"through both — "
            + ", ".join(os.path.basename(r[0]) for r in unsafe[:4])
            + f"{' ...' if len(unsafe) > 4 else ''}.  Sequential only "
            f"(no live overlap was allowed), which is why the run "
            f"proceeds; the permanent fix is one owner per file.")
        if escalate and policy() == "strict":
            raise RuntimeError(
                "LORRAX HDF5 probe refusal, strict policy (audit A1)\n"
                f"  got  : {len(libs)} libhdf5 objects mapped ("
                + ", ".join(libs) + f") and {len(unsafe)} file(s) written "
                "through both stacks in this process.\n"
                "  want : one HDF5 library instance per written file.\n"
                f"  fix  : split the metadata onto a sibling file, or set "
                f"{POLICY_ENV}=measure to record the condition instead of "
                f"refusing on it.")
    return {"libraries": libs, "companions": companions,
            "shared": [r[0] for r in both],
            "unsafe": [r[0] for r in unsafe], "rows": rows}


def reset_for_test() -> None:
    """Drop all registry state.  Tests only."""
    with _LOCK:
        _LIVE.clear()
        _HISTORY.clear()
