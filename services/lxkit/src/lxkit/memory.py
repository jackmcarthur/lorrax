"""Device memory, one model: a stage's :class:`Price` is booked to the timing section that runs
it (:func:`book`), routed by the one policy (:func:`admit`) against :func:`room`, and receipted
against the allocator's high-water over that section (``memory <stage>: priced P | measured M |
γ``); :func:`sizing` gives the minimum P.  Platform differences are ``lxkit.LEGS`` data.
Production reads the allocator at section boundaries only; ``LORRAX_MEMORY_RECEIPTS=calibrate``
(implied by STRICT) waits on priced stages, reads every section, and on a BFC arena
takes the first raise of its mark (exact) or a polled reading (``>=``).  ``LORRAX_MEMORY_TRACE=
<path>``: JSON Lines per closed section (``lxkit.memory_residuals``) in either mode: the boundary
readings and the live arrays, with no wait of its own.  ``LORRAX_MEMORY_STRICT=1``:
the census (GATE memory_unpriced, :func:`findings`).  ``LORRAX_NMU2_BACKUP=report``: GATE
nmu2_no_backup is a finding."""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

#: Receipts and route decisions are diagnostics: DEBUG lines with their receipt fields
#: (``memory.receipts``, ``routes``).  ``lxkit`` is a service and imports no ``common.log``, so it
#: logs on the ``lorrax.`` namespace directly, with the ``common.log.record`` extras by hand.
_LOG = logging.getLogger("lorrax.memory")

TERMS = ("operands", "xla", "vendor", "outside")
STRICT = os.environ.get("LORRAX_MEMORY_STRICT", "") == "1"
TRACE = os.environ.get("LORRAX_MEMORY_TRACE", "")
CALIBRATE = STRICT or os.environ.get("LORRAX_MEMORY_RECEIPTS", "") == "calibrate"
NMU2_REPORT = os.environ.get("LORRAX_NMU2_BACKUP", "") == "report"
POLL_SECONDS = 0.01                     # the BFC bytes_in_use poll (calibrate only)


@dataclass(frozen=True)
class Term:
    """Per-rank bytes ``fixed + spread / P (+ grid(P))`` and their provenance."""
    fixed: float = 0.0
    spread: float = 0.0
    source: str = ""
    grid: Optional[Callable[[int], float]] = field(default=None, compare=False)

    def at(self, ranks: int) -> float:
        """Bytes per rank on a mesh of ``ranks`` devices (``ranks < 1`` is taken as 1)."""
        p = max(1, int(ranks))
        return self.fixed + self.spread / p + (float(self.grid(p)) if self.grid else 0.0)

    def __add__(self, b: "Term") -> "Term":
        grids = [g for g in (self.grid, b.grid) if g]
        return Term(self.fixed + b.fixed, self.spread + b.spread, _join(self.source, b.source),
                    (lambda p: sum(g(p) for g in grids)) if grids else None)


@dataclass(frozen=True)
class Price:
    """A stage's per-rank price at ``ranks``.  ``increment``: the allocator's bytes in use when it
    is metered join its operands; ``whole``: the largest matrix side a rank holds whole (0: none)."""
    operands: Term = Term()
    xla: Term = Term()
    vendor: Term = Term()
    outside: Term = Term()
    ranks: int = 1
    label: str = ""
    increment: bool = False
    whole: int = 0

    def terms(self, ranks: Optional[int] = None) -> dict:
        """``{term: bytes per rank}`` for each of :data:`TERMS` at ``ranks`` (default ``self.ranks``)."""
        return {n: getattr(self, n).at(self.ranks if ranks is None else ranks) for n in TERMS}

    def bytes(self, ranks: Optional[int] = None) -> float:
        """The total per-rank bytes at ``ranks`` (default ``self.ranks``): the sum of :meth:`terms`."""
        return float(sum(self.terms(ranks).values()))

    def unsourced(self) -> list:
        """Names of the nonzero terms that carry no provenance (a finding under STRICT)."""
        return [n for n, v in self.terms().items() if v and not getattr(self, n).source]

    def describe(self, ranks: Optional[int] = None) -> str:
        """One line, ``"operands 1.20 GB, xla 0.40 GB, ..."``, for receipts and banners."""
        return ", ".join(f"{n} {_gb(v)}" for n, v in self.terms(ranks).items())

    def __add__(self, o: "Price") -> "Price":
        return Price(*(getattr(self, n) + getattr(o, n) for n in TERMS), ranks=max(self.ranks, o.ranks),
                     label=_join(self.label, o.label), increment=self.increment or o.increment,
                     whole=max(self.whole, o.whole))


def priced(nbytes: float, source: str, *, term: str = "xla", spread: float = 0.0,
           ranks: int = 1, increment: bool = False) -> Price:
    """A one-term Price: ``nbytes`` per rank plus ``spread`` / P."""
    return replace(Price(ranks=int(ranks), label=source, increment=bool(increment)),
                   **{term: Term(float(nbytes), float(spread), source)})


# Process state, plain tables; reset() clears the run's records, never the instrument (_I, _T).
_RUN: dict = {}          # limit, budget, reserve, linalg, per_node, platform, nmu
_OPEN: list = []         # the main thread's open sections (dicts), outermost first
_PENDING: dict = {}      # stage -> {label: Price} booked before it opens
_RECEIPTS: dict = {}     # stage -> receipt row (max over calls)
_PLANNED: dict = {}      # stage -> {"routes": ((name, Price), ...), "limit": bytes, "parts": {..}}
_ROUTES: dict = {}       # (site, route, reason) -> row
_DOOMED: dict = {}       # stage -> its over-the-limit banner, repeated when it opens
_WARNED: dict = {}       # stage kind -> the largest need warned
_FINDINGS: list = []
_HOOK: dict = {"sink": None, "agree": None, "available": None}
_I = dict(source=None, lib=None, handle=None, device=None, mark=-1, poll=-1, live=-1, reads=0, s=0.0)
_T: dict = {"on": False, "wrote": False, "exe": {}, "last": None, "born": {}}
_LOCK = threading.Lock()


def _gb(value: float) -> str:
    return f"{float(value) / 1e9:.2f} GB"


def _join(*parts) -> str:
    return " + ".join(s for s in parts if s)


def _rank0() -> bool:
    from lxkit.gate import rank0
    return rank0()


def _leg():
    import jax
    from lxkit.gate import device_vendor
    from lxkit.legs import leg_for_vendor
    return leg_for_vendor(device_vendor(jax.local_devices()[0]))


def _main() -> bool:
    return threading.current_thread() is threading.main_thread()


def _backend_up() -> bool:
    import sys
    return bool(getattr(sys.modules.get("jax._src.xla_bridge"), "_backends", None))


def _held(a) -> tuple:
    """(bytes on each device holding ``a``, those devices), from its sharding alone."""
    return int(math.prod(a.sharding.shard_shape(a.shape)) * a.dtype.itemsize), a.sharding.addressable_devices


def _cpu_in_use_by_device() -> dict:
    """Unique live CPU buffer bytes, including overlapping local shard views.

    A global array and its addressable local array can hold the same storage.
    Counting both as independent operands understates room. Reading a local
    array's cached ``addressable_shards`` also creates a self-reference in JAX;
    use its pointer directly. A global array with one local buffer uses
    ``addressable_data(0)``: JAX 0.12's replicated wrapper can retain the global
    sharding in ``addressable_shards``, which then rejects it as non-local.
    Multiple local buffers use their individual shards.
    """
    import jax
    spans = {}
    for a in jax.live_arrays():
        if not a.nbytes or a.is_deleted():
            continue
        devices = a.devices()
        if not any(d.platform == "cpu" for d in devices):
            continue
        if len(devices) == 1:
            parts = (a,)
        elif len(a.sharding.addressable_devices) == 1:
            parts = (a.addressable_data(0),)
        else:
            parts = (s.data for s in a.addressable_shards)
        for part in parts:
            device = next(iter(part.devices()))
            if device.platform != "cpu":
                continue
            start = int(part.unsafe_buffer_pointer())
            spans.setdefault(device.id, []).append((start, start + int(part.nbytes)))
    held = {}
    for device, intervals in spans.items():
        end = total = 0
        for start, stop in sorted(intervals):
            total += max(0, stop - max(start, end))
            end = max(end, stop)
        held[device] = total
    return held


def _in_use(per_device: bool = True) -> float:
    """Allocator bytes in use, max over devices; CPU: live array bytes (``per_device``: / devices)."""
    import jax
    devices = jax.local_devices()
    if devices[0].platform == "cpu":
        return sum(_cpu_in_use_by_device().values()) / (len(devices) if per_device else 1)
    return float(max(int((d.memory_stats() or {}).get("bytes_in_use", 0)) for d in devices))


def _sync() -> None:
    if CALIBRATE:
        try:
            import jax
            jax.block_until_ready(jax.live_arrays())
        except Exception:                                      # noqa: BLE001
            pass


# --- the instrument: the allocator high-water per interval between section boundaries ------------
def _setup() -> Optional[str]:
    """The leg's ``peak_reader``: "pool" (CUDA mempool, read and reset), "bfc" (a mark that never
    resets; polled in calibrate), else "live" (CPU live arrays)."""
    if not _backend_up():
        return None
    import ctypes
    import jax
    device, leg = jax.local_devices()[0], _leg()
    reader, _I["device"] = getattr(leg, "peak_reader", "live") if leg is not None else "live", device
    if reader == "pool":
        arrays = [a for a in jax.live_arrays() if a.nbytes and not a.is_deleted()
                  and getattr(a.sharding, "memory_kind", None) in (None, "device")]
        if not arrays:
            return None                      # no live buffer yet: ask again next time
        lib, handle = ctypes.CDLL("libcuda.so.1"), ctypes.c_void_p()
        pointer = arrays[0].addressable_shards[0].data.unsafe_buffer_pointer()
        if not lib.cuPointerGetAttribute(ctypes.byref(handle), 17, ctypes.c_uint64(pointer)) and handle.value:
            _I.update(lib=lib, handle=handle, source="pool")
            # Drop the high-water since process start (the pool's preallocation)
            # so the first stage read is that stage's use, not the reservation.
            lib.cuMemPoolSetAttribute(handle, 8, ctypes.byref(ctypes.c_uint64(0)))
            return "pool"
        reader = "bfc"                        # no CUDA pool behind this allocator
    stats = device.memory_stats() or {}
    if reader != "bfc" or "peak_bytes_in_use" not in stats:
        _I["source"] = "live" if device.platform == "cpu" else "none"
        return _I["source"]
    _I.update(source="bfc", mark=int(stats["peak_bytes_in_use"]), poll=int(stats.get("bytes_in_use", 0)))

    def sample():
        while True:
            time.sleep(POLL_SECONDS)
            try:
                use = int((device.memory_stats() or {}).get("bytes_in_use", 0))
            except Exception:                                  # noqa: BLE001
                return
            with _LOCK:
                _I["poll"] = max(_I["poll"], use)
    if CALIBRATE:
        threading.Thread(target=sample, name="lorrax-memory-sampler", daemon=True).start()
    return "bfc"


def instrument() -> dict:
    """The reader (``source``), and the boundary ``reads`` and ``seconds`` it has cost."""
    return dict(source=_I["source"], reads=int(_I["reads"]), seconds=float(_I["s"]))


def _reads_every_section() -> bool:
    # A trace reads every boundary (no wait: only calibrate syncs) so each record has its own.
    return CALIBRATE or bool(TRACE) or (_I["source"] or _setup()) in ("pool", "bfc")


def boundary() -> Optional[int]:
    """Read the high-water since the last boundary: it goes to every open section (``whole``),
    the innermost (``own``) and the innermost priced one (``metered``: it owns its interval)."""
    if not _main():
        return None
    t0, high, exact = time.perf_counter(), None, False
    try:
        src = _I["source"] or _setup()
        if src == "pool":
            import ctypes
            value = ctypes.c_uint64(0)        # CU_MEMPOOL_ATTR_USED_MEM_HIGH, read and reset
            if not _I["lib"].cuMemPoolGetAttribute(_I["handle"], 8, ctypes.byref(value)):
                high, exact, value.value = int(value.value), True, 0
                _I["lib"].cuMemPoolSetAttribute(_I["handle"], 8, ctypes.byref(value))
        elif src == "bfc":
            stats = _I["device"].memory_stats() or {}
            peak, use = int(stats.get("peak_bytes_in_use", 0)), int(stats.get("bytes_in_use", 0))
            with _LOCK:
                polled, _I["poll"] = max(_I["poll"], use), use
            exact, _I["mark"] = peak > _I["mark"], max(peak, _I["mark"])
            high = peak if exact else polled
        elif src == "live":
            live = int(_in_use(per_device=False))
            high, _I["live"] = max(_I["live"], live), live
    except Exception:                                          # noqa: BLE001
        high, exact = None, False
    _I["reads"], _I["s"] = _I["reads"] + 1, _I["s"] + time.perf_counter() - t0
    if high is not None and _OPEN:
        for t in _OPEN:
            t["whole"] = max(t["whole"], high)
        owner = next((t for t in reversed(_OPEN) if t["mark0"] is not None), None)
        if owner is not None:
            owner["metered"], owner["exact"] = max(owner["metered"], high), owner["exact"] and exact
        _OPEN[-1]["own"] = max(_OPEN[-1]["own"], high)
    if TRACE and _OPEN:
        _trace(_OPEN[-1], "boundary")
    return high


# --- sections and receipts (common.timing opens one per timing section) --------------------------
def _meter(t: dict) -> None:
    """Meter ``t`` from its first part on (after a wait in calibrate); settle its increments."""
    if not _backend_up():
        return
    if t["mark0"] is None:
        _sync()
        boundary()
        t.update(metered=-1, exact=CALIBRATE, mark0=_I["mark"])
    # An increment's base, the bytes in use, is its own part of the stage's price, read once.
    if any(p.increment for p in t["parts"].values()) and ENTRY not in t["parts"]:
        t["parts"][ENTRY] = Price(operands=Term(float(_in_use()), source=ENTRY), label=ENTRY)
    t["parts"] = {k: replace(p, increment=False) if p.increment else p for k, p in t["parts"].items()}


ENTRY = "resident at stage entry (allocator bytes in use)"


def open_tracker(name: str) -> Optional[dict]:
    """Open section ``name`` with the parts booked to it; repeat its banner if it has one."""
    if not _main():
        return None
    t = dict(name=str(name), parts=_PENDING.pop(str(name), {}), whole=-1, own=-1, metered=-1,
             exact=True, mark0=None, rec=None)
    if t["name"] in _DOOMED:
        _say(_DOOMED[t["name"]])
    if t["parts"]:
        _meter(t)
    elif _backend_up() and _reads_every_section():
        boundary()
    _OPEN.append(t)
    if TRACE and _backend_up():
        _trace(t, "open")
    return t


def close_tracker(t: Optional[dict]) -> tuple:
    """Close a section: (whole, own, exact); a priced one records and prints its receipt."""
    if t is None:
        return -1, -1, False
    metered = t["mark0"] is not None
    if metered:
        _sync()
    if metered or _reads_every_section():
        boundary()
    _OPEN[:] = [o for o in _OPEN if o is not t]
    if metered:
        if _I["source"] == "bfc" and CALIBRATE:      # it lifted the mark: its peak, exactly
            t["exact"] = _I["mark"] > t["mark0"]
            t["metered"] = _I["mark"] if t["exact"] else t["metered"]
        parent = next((o for o in reversed(_OPEN) if o["mark0"] is not None), None)
        if parent is not None:                       # the parent's first raise is past this one's
            parent["mark0"] = max(parent["mark0"], _I["mark"])
    if TRACE and t["rec"] is not None:
        _trace(t, "close")
    if t["parts"]:
        price, src = sum(t["parts"].values(), Price()), _I["source"] or "none"
        row = _RECEIPTS.setdefault(t["name"], dict(stage=t["name"], count=0, priced=0.0,
                                                   measured=-1.0, exact=True, price=price, source=src))
        row["count"] += 1
        say = row["count"] == 1 or t["metered"] > row["measured"] or price.bytes() > row["priced"]
        if price.bytes() >= row["priced"]:
            row["priced"], row["price"] = price.bytes(), price
        if t["metered"] > row["measured"]:
            row["measured"], row["exact"] = float(t["metered"]), t["exact"]
        if say:
            _debug(receipt_line(t["name"], price, t["metered"], t["exact"], src), "memory.receipts",
                   lambda: _receipt_fields(t["name"], price, t["metered"], t["exact"], src))
    return t["whole"], t["own"], t["exact"]


def book(price: Price, stage: Optional[str] = None) -> None:
    """Book ``price`` as a part of ``stage``: its innermost open section, else the next to open
    under that name (``None``: the innermost open section).  A stage's price is the sum of its
    parts; a part booked again under its label keeps the larger."""
    if not isinstance(price, Price):
        raise TypeError(f"memory.book takes a memory.Price, got {type(price).__name__}")
    if STRICT and price.unsourced():
        _FINDINGS.append(f"no provenance: {stage or price.label}: {', '.join(price.unsourced())}")
    target = None if not _main() else (_OPEN[-1] if _OPEN else None) if stage is None else next(
        (t for t in reversed(_OPEN) if t["name"] == stage), None)
    key = target["name"] if target is not None else str(stage if stage is not None else price.label)
    parts = target["parts"] if target is not None else _PENDING.setdefault(key, {})
    if price.label not in parts or price.bytes() > parts[price.label].bytes():
        parts[price.label] = price
    _PLANNED.setdefault(key, {})["parts"] = dict(parts)
    if target is not None:
        _meter(target)
    elif stage is None:
        _PENDING.pop(key, None)              # no section open (a planning replay): sizing only


def receipt_line(name: str, price: Price, measured: float, exact: bool, src: str) -> str:
    """``memory <stage>: priced P (terms) | measured M | γ g``."""
    total, head = price.bytes(), f"memory {name}: priced {_gb(price.bytes())} ({price.describe()})"
    if measured is None or measured < 0:
        return head + " | measured not measured | γ –"
    mark = "" if exact else ({"bfc": ">=", "live": "live "}.get(src, "~") if CALIBRATE else "~")
    return head + f" | measured {mark}{_gb(measured)} | γ " + (f"{measured / total:.2f}" if total else "–")


def _receipt_fields(name: str, price: Price, measured: float, exact: bool, src: str) -> dict:
    """The numbers of a receipt line, in bytes: the DEBUG receipt of ``memory.receipts``."""
    total, known = price.bytes(), measured is not None and measured >= 0
    return dict(stage=str(name), priced_bytes=total, **{f"{n}_bytes": v for n, v in price.terms().items()},
                measured_bytes=float(measured) if known else None, measured_exact=bool(exact),
                source=str(src), gamma=float(measured) / total if known and total else None)


def receipts() -> list:
    """The receipt rows (max over calls; ``common.timing``'s gather adds ``measured_ranks``)."""
    return list(_RECEIPTS.values())


def findings() -> list:
    """The census's findings, and parts booked to stages that never opened."""
    return list(_FINDINGS) + [f"never opened: {name}" for name in _PENDING]


# --- the trace (LORRAX_MEMORY_TRACE): what is live, what the allocator holds, what runs ----------
def _snapshot() -> dict:
    """Live arrays grouped by (shape, dtype, sharding, creating section), bytes per device."""
    import jax
    groups, per_device, born, alive = {}, {}, _T["born"], set()
    for a in jax.live_arrays():
        alive.add(id(a))
        born.setdefault(id(a), _OPEN[-1]["name"] if _OPEN else "(top level)")
        nbytes, held = _held(a)
        for device in held:
            per_device[device.id] = per_device.get(device.id, 0) + nbytes
        spec = str(getattr(a.sharding, "spec", type(a.sharding).__name__))
        groups.setdefault((tuple(a.shape), str(a.dtype), spec, born[id(a)]), [0, nbytes])[0] += 1
    for gone in set(born) - alive:
        del born[gone]
    if jax.local_devices()[0].platform == "cpu":
        per_device = _cpu_in_use_by_device()
    stats = [d.memory_stats() or {} for d in jax.local_devices()]
    return dict(live=max(per_device.values(), default=0),
                in_use=max(int(st.get("bytes_in_use", 0)) for st in stats),
                peak=max(int(st.get("peak_bytes_in_use", 0)) for st in stats),
                arrays=[dict(shape=list(k[0]), dtype=k[1], spec=k[2], born=k[3], count=v[0], bytes_per_device=v[1])
                        for k, v in sorted(groups.items(), key=lambda kv: -kv[1][0] * kv[1][1])])


def _append(record: Optional[dict]) -> None:
    """Append ``record`` to this process's trace (``.p<i>`` on process i > 0); None: truncate."""
    import json

    import jax
    path = TRACE if jax.process_index() == 0 else f"{TRACE}.p{jax.process_index()}"
    with open(path, "a" if record else "w") as out:
        out.write(json.dumps(record, default=str) + "\n" if record else "")


def _start_trace() -> None:
    """Credit each executable's compiled figures to the sections open when it runs; at exit,
    append the run's figures and receipts (sections are appended as they close)."""
    import atexit
    from jax._src.interpreters import pxla
    execute, names = pxla.ExecuteReplicated.__call__, ("temp", "argument", "output", "alias", "generated_code")

    def run(self, *args):
        try:
            exe = self.xla_executable
            if id(exe) not in _T["exe"]:
                m = exe.get_compiled_memory_stats()
                _T["exe"][id(exe)] = (exe, dict(program=str(self.name), **{
                    n.replace("generated_", ""): int(getattr(m, f"{n}_size_in_bytes")) for n in names}))
            row = _T["exe"][id(exe)][1]
            for t in _OPEN:
                if t["rec"] is not None and not any(r is row for r in t["rec"]["runs"]):
                    t["rec"]["runs"].append(row)
        except Exception:                                      # noqa: BLE001
            pass
        return execute(self, *args)
    pxla.ExecuteReplicated.__call__ = run
    _append(None)
    atexit.register(lambda: _T["wrote"] and _append(dict(run=dict(_RUN), receipts=[
        {k: v for k, v in r.items() if k != "price"} for r in receipts()])))
    _T["on"] = True


def _trace(t: dict, event: str) -> None:
    try:
        if not _T["on"]:
            _start_trace()
        last = _T["last"]
        snap = last[1] if last and last[0] == _I["reads"] else _snapshot()
        _T["last"] = (_I["reads"], snap)
        if event == "open":
            t["rec"] = dict(name=t["name"], depth=len(_OPEN) - 1, open=snap, peak=snap, runs=[], t_open=time.time())
        for o in ([t] if event == "close" else _OPEN):
            if o["rec"] is not None and snap["live"] > o["rec"]["peak"]["live"]:
                o["rec"]["peak"] = snap
        if event == "close" and t["rec"] is not None:
            price = sum(t["parts"].values(), Price()) if t["parts"] else None
            t["rec"].update(close=snap, t_close=time.time(), measured=t["whole"], exact=t["exact"],
                            source=_I["source"], priced=price and price.terms(), run=dict(_RUN))
            _append(t["rec"])
            _T["wrote"] = True
    except Exception:                                          # noqa: BLE001
        pass


# --- lines and warnings ----------------------------------------------------------------------------
def set_sink(sink=None) -> None:
    """Route the banner and warned-route lines through the driver's report (receipts and routes are
    DEBUG logs and receipts, not sink lines)."""
    _HOOK["sink"] = sink


def _say(text: str, *, local: bool = False) -> None:
    try:
        (local or _rank0()) and (_HOOK["sink"] or (lambda line: print(line, flush=True)))(text)
    except Exception:                                          # noqa: BLE001
        pass


def _debug(text: str, key: str, fields: Callable[[], dict], *, local: bool = False) -> None:
    """A diagnostic line: DEBUG, plus the record ``fields()`` under ``key`` (printed with verbose)."""
    try:
        (local or _rank0()) and _LOG.debug(text, extra={"lorrax": {"record_key": key, "record": fields()}})
    except Exception:                                          # noqa: BLE001
        pass


def _warn(stage: str, need: float, message: str, *, local: bool) -> None:
    """A RuntimeWarning once per stage kind (``.N``/``:N`` dropped), again only if it grows."""
    import sys
    import warnings
    kind = re.sub(r"([.:]\d+)+$", "", str(stage))
    if need > _WARNED.get(kind, -1.0):
        _WARNED[kind] = need
        if _rank0():
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        elif local:
            print("WARNING: " + message, file=sys.stderr, flush=True)


def vendor(leg, lib: str, op: str, n: int, ranks: int, itemsize: int, unit: Optional[float] = None) -> Term:
    """A vendor workspace with no size query: ``leg``'s MEASURED row within its stated n and P
    range; outside it, or with no row, :func:`unpriced` (extrapolated, or priced 0)."""
    row = dict(leg.vendor_rows).get((lib, op))
    where = (f"no measured row for {leg.key}/{lib}/{op}" if row is None else None
             if row.n[0] <= n <= row.n[1] and row.p[0] <= ranks <= row.p[1] else
             f"{leg.key}/{lib}/{op} at n {int(n)}, P {int(ranks)} outside the measured "
             f"n {row.n[0]}-{row.n[1]}, P {row.p[0]}-{row.p[1]}")
    if where:
        unpriced(where, "priced 0" if row is None else "extrapolated")
    if row is None:
        return Term(0.0, source=f"UNPRICED {leg.key}/{lib}/{op}")
    nbytes = row.ratio * (row.unit(int(n), int(ranks), int(itemsize)) if unit is None else unit)
    return Term(nbytes, source=("EXTRAPOLATED " if where else "") + f"{row.ratio:g} x {row.unit_name} ({row.source})")


def unpriced(what: str, then: str = "priced 0") -> None:
    """A figure no compile, query or measurement gives: GATE memory_unpriced under STRICT, else
    one warning per kind."""
    if STRICT:
        raise RuntimeError(f"GATE memory_unpriced: got: {what}; want: a compiled figure, a "
                           "library query or a measurement there")
    _warn(f"unpriced {what.split(' ')[0]}", 0.0, f"memory: {what}; {then}", local=False)


# --- limit, budget, room ---------------------------------------------------------------------------
def _outside(leg, linalg: str = "local") -> float:
    """Bytes outside the allocator: the leg's row, plus its providers' contexts if distributed."""
    return float(leg.outside_allowance) + (float(leg.distributed_context[0]) if linalg == "distributed" else 0.0)


def pool_reservation(total: float, leg) -> float:
    """The pool reserved before the deck: the card less the distributed allowance, >= half."""
    return max(float(total) - _outside(leg, "distributed"), 0.5 * float(total))


def contexts() -> Term:
    """The providers' contexts a route opens where the budget did not withhold them (local)."""
    leg = _leg()
    size, src = getattr(leg, "distributed_context", (0.0, "")) if leg.accelerator else (0.0, "")
    return Term(size if _RUN.get("linalg", "local") == "local" else 0.0, source=src)


def _agree(value: float) -> float:
    """The minimum over processes, through :func:`resolve`'s ``agree`` collective."""
    import jax
    if _HOOK["agree"] is None and jax.process_count() > 1:
        raise RuntimeError("memory: several processes and no agreement collective; resolve the "
                           "budget through common.gpu_utils.resolve_device_budget_gb")
    return float(_HOOK["agree"](float(value)) if _HOOK["agree"] else value)


def enforced(stats) -> bool:
    """The allocator refuses past ``bytes_limit`` (a BFC arena; not cuda_async's threshold)."""
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default").lower()
    return allocator in ("default", "bfc") and bool((stats or {}).get("bytes_limit"))


def resolve(deck_bytes: float = 0.0, linalg: str = "local", *, total: Optional[float] = None,
            outside: Optional[float] = None, agree: Optional[Callable] = None,
            available: Optional[Callable[[], float]] = None) -> float:
    """THE run budget per device (every process enters): limit = an enforced ``bytes_limit``,
    else ``total`` less what is ``outside`` (>= the leg's allowance), CPU the host share; budget
    = (limit - pending contexts) x (1 - reserve) - T_ord, the minimum over processes (``agree``),
    never above a positive deck value. On CPU, ``available`` reads current host
    bytes per local device, including non-JAX consumers, without a collective."""
    import jax
    from lxkit import ordered_transient_bytes
    _HOOK["agree"] = agree
    leg, device = _leg(), jax.local_devices()[0]
    stats = device.memory_stats() or {}
    if leg is None or not leg.accelerator:
        lim, pending = float(total or 0.0) / max(1, jax.local_device_count()), 0.0
    elif enforced(stats) or not total:
        lim, pending = float(stats.get("bytes_limit", 0)), 0.0
    else:
        lim = float(total) - max(float(outside or 0.0), _outside(leg))
        pending = _outside(leg, linalg) - _outside(leg)
    rule = (lim - pending) * (1.0 - float(getattr(leg, "memory_reserve", 0.0)))
    lim, rule, deck = _agree(lim), _agree(max(0.0, rule - float(ordered_transient_bytes()))), float(deck_bytes or 0)
    if deck > rule and _rank0():
        import warnings
        warnings.warn(f"memory_per_device_gb = {deck / 1e9:g} exceeds the memory model's {rule / 1e9:.2f} GB on "
                      f"this card (linalg = {linalg}); running at {rule / 1e9:.2f} GB", RuntimeWarning, stacklevel=2)
    reset()
    set_budget(min(deck, rule) if deck > 0 else rule, limit_bytes=lim, linalg=linalg,
               per_node=int(getattr(leg, "node_devices", 1)), platform=str(getattr(device, "device_kind", "") or "cpu"))
    # A smaller user deck limits JAX allocations, not physical host capacity.
    # Withhold only the platform reserve from the live host reader.
    host_reserve = max(0.0, lim - rule)
    _HOOK["available"] = ((lambda: float(available()) - host_reserve)
                          if available is not None and (leg is None or not leg.accelerator) else None)
    return float(_RUN["budget"])


def set_budget(budget: float, *, limit_bytes: Optional[float] = None, per_node: int = 1,
               platform: str = "test", linalg: str = "local") -> None:
    """Record the run's figures (:func:`resolve`, or a replay that fakes the card)."""
    lim = float(budget if limit_bytes is None else limit_bytes)
    _RUN.clear()
    _RUN.update(limit=lim, budget=float(budget), reserve=max(0.0, lim - float(budget)),
                linalg=str(linalg), per_node=int(per_node), platform=str(platform))
    _HOOK["available"] = None


def run() -> dict:
    """The run's figures (empty before :func:`resolve`): limit, budget (the planning budget per
    device), reserve, linalg, per_node, platform, nmu."""
    return dict(_RUN)


def room(*, bucketed: bool = True) -> int:
    """limit - reserve - bytes in use, the minimum over processes (all enter), :func:`bucket`-ed."""
    left = float(_RUN["budget"]) - _in_use()
    if _HOOK["available"] is not None:
        left = min(left, float(_HOOK["available"]()))
    left = int(_agree(max(0.0, left)))
    return bucket(left) if bucketed else left


def bucket(value) -> int:
    """A whole GiB at or above 1 GiB, exact below, 0 when negative."""
    value = int(value)
    return (value >> 30) << 30 if value >= 1 << 30 else max(value, 0)


# --- admission: the one policy point, route completeness, sizing ----------------
def record_route(site, route, reason, *, warning=None) -> None:
    """Record one route decision; rank 0 logs it the first time (DEBUG and the ``routes`` record; a
    warned route is also printed)."""
    key = (str(site), str(route), str(reason))
    if key in _ROUTES:
        _ROUTES[key]["count"] += 1
        return
    line = f"route {site}: {route} ({reason})" + (f"; WARNING {warning}" if warning else "")
    _ROUTES[key] = dict(site=key[0], route=key[1], reason=key[2], warning=warning, line=line, count=1)
    text = ("  WARNING: " if warning else "  ") + line
    if warning:                                               # a warning stays in the main log
        _say(text)
    _debug(text, "routes", lambda: dict(site=key[0], route=key[1], reason=key[2], warning=warning))


def routes() -> list:
    """Every route decision recorded this run (:func:`record_route`, :func:`admit`), as dicts
    ``site, route, reason, warning, line, count``, in first-decision order."""
    return [dict(row) for row in _ROUTES.values()]


def admit(stage: str, routes, room_bytes: float, *, probe=None, runs_in: Optional[str] = None,
          local: bool = False, device: bool = True, detail: bool = False):
    """THE admission policy.  ``routes``: ``(name, Price | callable[, room])`` fastest first, each
    against its room (default ``room_bytes``, the same on every rank).  1. The fastest route that
    fits runs.  2. None fits and the cheapest is over the physical limit (room + reserve): a
    banner with the shortfall and the minimum P, now and when its stage (``runs_in``, else the
    open one) opens; it runs.  3. Otherwise one warning; it runs.  ``device=False`` (a host
    share, a tile cap): no banner, not in :func:`sizing`.  ``probe()`` raising drops the later
    routes; ``local``: warned on its rank.  Returns the route, or with ``detail`` an unrecorded
    dict(route, reason, warning) the caller records."""
    routes = (("chosen", routes),) if isinstance(routes, Price) else tuple(routes)
    key, reserve = str(runs_in or (_OPEN[-1]["name"] if _OPEN and _main() else stage)), _RUN.get("reserve", 0.0)
    known = [p for _, p, *_ in routes if isinstance(p, Price)]
    if device and any(p.whole for p in known):
        require_backup(stage, [p.whole for p in known], any(not p.whole for p in known) or len(known) < len(routes))
    done, warning = [], None                     # (name, price, room) tried, in order
    for i, (name, price, *own) in enumerate(routes):
        if i == 1 and probe is not None:
            try:
                probe()
            except Exception as exc:                           # noqa: BLE001
                warning = f"the {name} route is unavailable here ({type(exc).__name__}: {exc})"
                break
        done.append((name, price() if callable(price) else price, float(own[0] if own else room_bytes)))
        if STRICT and done[-1][1].unsourced():
            _FINDINGS.append(f"no provenance: {stage} ({name}): {', '.join(done[-1][1].unsourced())}")
        if device:
            _PLANNED.setdefault(key, {}).update(routes=tuple(d[:2] for d in done), limit=done[-1][2] + reserve)
        if done[-1][1].bytes() <= done[-1][2]:
            break
    fits = done[-1][1].bytes() <= done[-1][2]
    name, price, have = done[-1] if fits or warning else min(done, key=lambda d: d[1].bytes() - d[2])
    need, limit = price.bytes(), have + reserve if device else float("inf")
    reason = "; ".join(f"{_gb(p.bytes())}/rank of room {_gb(r)}" if p.bytes() <= r else
                       f"{n} {_gb(p.bytes())}/rank over room {_gb(r)}" for n, p, r in done)
    if not (fits or warning) and need <= limit:
        warning = "no route fits the planning room"
        _warn(stage, need, f"memory over budget at {stage}: {name} needs {_gb(need)}/rank, room {_gb(have)}/rank"
              + (f", within the allocator limit {_gb(limit)}" if device else "") + "; continuing (an OOM is "
              "possible)", local=local)
    elif not (fits or warning):
        warning, line, bar = "priced over the allocator limit", sizing()["line"], "!" * 78
        _DOOMED[key] = "\n".join([bar, f"!! MEMORY: {stage} is PRICED OVER the allocator limit. It runs anyway.",
                                  f"!!   {name}: priced {_gb(need)}/rank ({price.describe()})", "!!   allocator "
                                  f"limit beside what is live {_gb(limit)}; short by {_gb(need - limit)}",
                                  "!!   " + line, bar])
        _say(_DOOMED[key], local=local)
        _warn(stage, need, f"memory: {stage} priced over the limit ({name} {_gb(need)}/rank over {_gb(limit)}); "
              f"{line}", local=local)
    if detail:
        return dict(route=name, reason=reason, warning=warning)
    if len(routes) > 1:
        record_route(stage, name, reason, warning=warning)
    return name


def set_nmu(n: int) -> None:
    """The run's N_mu: a whole matrix with a side >= N_mu / 2 needs a distributed backup."""
    _RUN["nmu"] = int(n)


def require_backup(stage: str, wholes, distributed: bool) -> None:
    """Route completeness (owner): a stage whose candidates hold an N_mu-scale matrix whole on a
    rank (``wholes``: each one's largest whole side) needs a distributed candidate; with none,
    GATE nmu2_no_backup (``NMU2_REPORT``: a finding)."""
    import jax
    nmu = int(_RUN.get("nmu", 0))
    big = max([int(w) for w in wholes if nmu and int(w) * 2 >= nmu], default=0)
    if not big or distributed or jax.device_count() < 2:
        return
    what = (f"GATE nmu2_no_backup: got: {stage} holds a whole {big} x {big} matrix on a rank (N_mu {nmu}) "
            "with no distributed candidate; want: a mesh route beside it")
    if not NMU2_REPORT:
        raise RuntimeError(what)
    _FINDINGS.extend([what] if what not in _FINDINGS else [])


def sizing(stages: Optional[dict] = None, *, limit_bytes: Optional[float] = None,
           per_node: Optional[int] = None, platform: Optional[str] = None) -> dict:
    """The estimated smallest square P from the current memory quotes, not a capacity proof.

    Every device stage fits its modeled price (its
    best admitted route, at least its booked parts, against its admission's limit, else the
    run's); ``binding``: the least headroom there; nodes = ceil(P / devices per node).
    ``stages``: ``{stage: ((route, Price), ...)}`` instead of this run's plans."""
    plans = _PLANNED if stages is None else {k: dict(routes=v) for k, v in stages.items()}
    run_lim = float(_RUN.get("limit", 0.0) if limit_bytes is None else limit_bytes)
    limit = {k: float(v.get("limit", run_lim)) for k, v in plans.items()}

    def price(k, p):             # the booked parts beside the cheapest admitted route at P
        v = plans[k]
        return sum(v.get("parts", {}).values(), Price()) + min(
            (x for _, x in v.get("routes", ())), default=Price(), key=lambda x: x.bytes(p))

    def need(k, p):              # the larger of the cheapest route and the parts (parts often hold it)
        v = plans[k]
        return max(min((x.bytes(p) for _, x in v.get("routes", ())), default=0.0),
                   sum(x.bytes(p) for x in v.get("parts", {}).values()))
    short = lambda p: [k for k in plans if need(k, p) > limit[k]] or None
    p = next((p * p for p in range(1, 65) if not short(p * p)), None)
    if not plans or p is None:
        k = (short(4096) or [None])[0]
        n = k and max(TERMS, key=lambda n: getattr(price(k, 4096), n).fixed)
        return dict(p=None, nodes=None, binding=k, line="estimated minimum from this memory quote: no priced stage" if not plans else
                    f"estimated minimum from this memory quote: no square P up to 4096 fits: {k} quotes {_gb(getattr(price(k, 4096), n).fixed)}/rank "
                    f"fixed in {n} ({getattr(price(k, 4096), n).source}) against the "
                    f"{_gb(limit[k])} limit")
    nodes = -(-p // max(1, int(_RUN.get("per_node", 1) if per_node is None else per_node)))
    binding = max(plans, key=lambda k: need(k, p) / limit[k])
    return dict(p=p, nodes=nodes, binding=binding, line=f"estimated minimum from this memory quote: P = {p} ({nodes} node"
                f"{'s' * (nodes != 1)} of {_RUN.get('platform', 'this platform') if platform is None else platform})"
                f"; binding stage {binding}")


def reset(*, run: bool = False) -> None:
    """Forget the run's records (a new run in this process); ``run``: its figures too."""
    for table in (_RECEIPTS, _PENDING, _ROUTES, _PLANNED, _DOOMED, _WARNED, _FINDINGS) + ((_RUN,) if run else ()):
        table.clear()
    if run:
        _HOOK["available"] = None
