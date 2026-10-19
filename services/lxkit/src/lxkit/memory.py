"""Device memory: one price per stage, one room, one admission policy, one receipt
(MEMORY_MODEL_AUDIT §3).  Platform differences are data in ``lxkit.LEGS``.

:class:`Price` (four sourced terms, ``fixed + spread / P``) -> :func:`book` (a stage's price
is the sum of its parts; its receipt ``memory <stage>: priced P | measured M | γ``) ->
:func:`resolve` / :func:`room` -> :func:`admit` (the one policy) -> :func:`sizing` (minimum P).
Stdlib-only at import; jax is imported lazily.  An instrument never raises."""

from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

TERMS = ("operands", "xla", "vendor", "outside")
#: The census: a vendor row outside its range raises GATE memory_unpriced; :func:`findings`
#: lists sections over CENSUS_BYTES outside a priced stage and terms without provenance.
STRICT = os.environ.get("LORRAX_MEMORY_STRICT", "") == "1"
CENSUS_BYTES = 64 << 20
#: Calibrate (and STRICT): priced stages wait on their work and a BFC leg polls, so receipts
#: are exact.  Production: no wait, no poll, receipts ``~``; the CPU leg reads priced stages only.
#: ``LORRAX_MEMORY_TRACE=<path>`` (implies calibrate): at every section's open and close and at
#: its in-section peak, the live arrays (shape, dtype, sharding, bytes per device, the stage
#: that created them), the allocator's bytes in use and peak, and every executable compiled in
#: it (``memory_analysis``); one JSON per process at exit (``tools/memory_residuals.py``).
TRACE = os.environ.get("LORRAX_MEMORY_TRACE", "")
CALIBRATE = STRICT or bool(TRACE) or os.environ.get("LORRAX_MEMORY_RECEIPTS", "") == "calibrate"


@dataclass(frozen=True)
class Term:
    """Per-rank bytes ``fixed + spread / P (+ grid(P))`` and their provenance."""
    fixed: float = 0.0
    spread: float = 0.0
    source: str = ""
    grid: Optional[Callable[[int], float]] = field(default=None, compare=False)

    def at(self, ranks: int) -> float:
        p = max(1, int(ranks))
        return self.fixed + self.spread / p + (float(self.grid(p)) if self.grid else 0.0)


@dataclass(frozen=True)
class Price:
    """One stage's per-rank price at ``ranks`` processes.  ``increment``: the stage's
    resident set is the allocator's bytes in use when it is metered (:meth:`settled`)."""
    operands: Term = Term()
    xla: Term = Term()
    vendor: Term = Term()
    outside: Term = Term()
    ranks: int = 1
    label: str = ""
    increment: bool = False

    def settled(self) -> "Price":
        if not self.increment:
            return self
        return replace(self, increment=False, operands=Term(
            self.operands.at(self.ranks) + in_use(), source=" + ".join(s for s in (
                self.operands.source, "resident at stage entry (allocator bytes in use)") if s)))

    def terms(self, ranks: Optional[int] = None) -> dict:
        return {n: getattr(self, n).at(self.ranks if ranks is None else ranks) for n in TERMS}

    def bytes(self, ranks: Optional[int] = None) -> float:
        return float(sum(self.terms(ranks).values()))

    def unsourced(self) -> list:
        return [n for n in TERMS if getattr(self, n).at(self.ranks) and not getattr(self, n).source]

    def describe(self, ranks: Optional[int] = None) -> str:
        return ", ".join(f"{n} {_gb(v)}" for n, v in self.terms(ranks).items())

    def __add__(self, other: "Price") -> "Price":
        def join(a, b):
            grids = [g for g in (a.grid, b.grid) if g]
            return Term(a.fixed + b.fixed, a.spread + b.spread,
                        " + ".join(s for s in (a.source, b.source) if s),
                        (lambda p: sum(g(p) for g in grids)) if grids else None)
        return Price(*(join(getattr(self, n), getattr(other, n)) for n in TERMS),
                     ranks=max(self.ranks, other.ranks), increment=self.increment or other.increment,
                     label=" + ".join(s for s in (self.label, other.label) if s))


def priced(nbytes: float, source: str, *, term: str = "xla", spread: float = 0.0,
           ranks: int = 1, increment: bool = False) -> Price:
    """A one-term Price: ``nbytes`` per rank plus ``spread`` / P."""
    return replace(Price(ranks=int(ranks), label=source, increment=bool(increment)),
                   **{term: Term(float(nbytes), float(spread), source)})


def _gb(value: float) -> str:
    return f"{float(value) / 1e9:.2f} GB"


def _rank0() -> bool:
    from lxkit.gate import rank0
    return rank0()


def _leg():
    import jax
    from lxkit.gate import device_vendor
    from lxkit.legs import leg_for_vendor
    return leg_for_vendor(device_vendor(jax.local_devices()[0]))


# ---------------------------------------------------------------------------
# Instrument: the allocator high-water per interval between boundaries
# ---------------------------------------------------------------------------
POLL_SECONDS = 0.01              # the BFC bytes_in_use poll (calibrate only)
_I = {"source": None, "lib": None, "handle": None, "device": None, "mark": -1, "poll": -1,
      "live": -1, "reserved": None}
_LOCK = threading.Lock()
_COST = [0, 0.0]


def _backend_up() -> bool:
    import sys
    bridge = sys.modules.get("jax._src.xla_bridge")
    return bool(bridge and getattr(bridge, "_backends", None))


def _held(a) -> tuple:
    """(bytes on each device holding ``a``, those devices), from its sharding: reading
    ``addressable_shards[i].data`` would create the single-device arrays it counts."""
    shard = math.prod(a.sharding.shard_shape(a.shape)) * a.dtype.itemsize
    return int(shard), a.sharding.addressable_devices


def _live_bytes() -> int:
    import jax
    return sum(n * len(d) for n, d in map(_held, jax.live_arrays()))


def _setup() -> Optional[str]:
    """The leg's reader: "pool" (CUDA mempool, resettable), "bfc" (never resets), "live" (CPU)."""
    if not _backend_up():
        return None
    import ctypes
    import jax
    device = _I["device"] = jax.local_devices()[0]
    leg = _leg()
    reader = getattr(leg, "peak_reader", "live") if leg is not None else "live"
    if reader == "pool":
        arrays = [a for a in jax.live_arrays() if a.nbytes and not a.is_deleted()
                  and getattr(a.sharding, "memory_kind", None) in (None, "device")]
        if not arrays:
            return None                      # no live buffer yet: ask again next time
        lib, handle = ctypes.CDLL("libcuda.so.1"), ctypes.c_void_p()
        pointer = arrays[0].addressable_shards[0].data.unsafe_buffer_pointer()
        if not lib.cuPointerGetAttribute(ctypes.byref(handle), 17, ctypes.c_uint64(pointer)) \
                and handle.value:
            _I.update(lib=lib, handle=handle, source="pool")
            return "pool"
        reader = "bfc"                        # no CUDA pool behind this allocator
    if reader == "bfc" and "peak_bytes_in_use" in (device.memory_stats() or {}):
        stats = device.memory_stats()
        _I.update(source="bfc", mark=int(stats["peak_bytes_in_use"]),
                  poll=int(stats.get("bytes_in_use", 0)))
        if CALIBRATE:
            threading.Thread(target=_sample, name="lorrax-memory-sampler", daemon=True).start()
        return "bfc"
    _I["source"] = "live" if device.platform == "cpu" else "none"
    return _I["source"]


def _sample() -> None:
    """Poll ``bytes_in_use`` into the interval maximum (a BFC arena's lower bound)."""
    while True:
        time.sleep(POLL_SECONDS)
        try:
            use = int((_I["device"].memory_stats() or {}).get("bytes_in_use", 0))
        except Exception:                                      # noqa: BLE001
            return
        with _LOCK:
            _I["poll"] = max(_I["poll"], use)


def _interval() -> tuple:
    """(high-water since the last boundary, exact); sets ``_I["reserved"]`` likewise."""
    src = _I["source"] or _setup()
    _I["reserved"] = None
    if src == "pool":
        import ctypes
        out = []
        for attr in (8, 6):           # CU_MEMPOOL_ATTR_USED_MEM_HIGH, RESERVED_MEM_HIGH
            value = ctypes.c_uint64(0)
            if _I["lib"].cuMemPoolGetAttribute(_I["handle"], attr, ctypes.byref(value)):
                return None, False
            out.append(int(value.value))
            value.value = 0
            _I["lib"].cuMemPoolSetAttribute(_I["handle"], attr, ctypes.byref(value))
        _I["reserved"] = out[1]
        return out[0], True
    if src == "bfc":
        stats = _I["device"].memory_stats() or {}
        reserved = stats.get("peak_pool_bytes", stats.get("pool_bytes"))
        _I["reserved"] = None if reserved is None else int(reserved)
        peak, use = int(stats.get("peak_bytes_in_use", 0)), int(stats.get("bytes_in_use", 0))
        with _LOCK:
            polled, _I["poll"] = max(_I["poll"], use), use
        raised, _I["mark"] = peak > _I["mark"], max(peak, _I["mark"])
        return (peak, True) if raised else (polled, False)
    if src == "live":
        live = _live_bytes()
        high, _I["live"] = max(_I["live"], live), live
        return high, False
    return None, False


def source() -> Optional[str]:
    return _I["source"]


def instrument_cost() -> tuple:
    """(boundary reads, seconds) the instrument has cost this process."""
    return int(_COST[0]), float(_COST[1])


class Tracker:
    """An open section's high-waters: ``whole``, ``own`` (outside children), ``metered``
    (since its first part was booked: the receipt's figure)."""
    __slots__ = ("name", "parts", "whole", "own", "metered", "exact", "mark0", "base", "reserved",
                 "rec")

    def __init__(self, name, parts):
        self.name, self.parts = str(name), dict(parts or {})
        self.whole = self.own = self.metered = self.reserved = -1
        self.exact, self.mark0, self.base, self.rec = True, None, -1, None

    @property
    def price(self) -> Optional[Price]:
        return sum(self.parts.values(), Price()) if self.parts else None

    def meter(self) -> None:
        if self.mark0 is None and _backend_up():
            _sync()
            boundary()
            self.metered, self.exact, self.mark0, self.reserved = -1, CALIBRATE, _I["mark"], -1
        if _backend_up():
            self.parts = {k: p.settled() for k, p in self.parts.items()}


_OPEN: list = []                 # the main thread's open trackers, outermost first
_PENDING: dict = {}              # stage -> {label: Price} booked before it opens
_RECEIPTS: dict = {}             # stage -> receipt row (max over calls)
_FINDINGS: list = []             # the census's findings (STRICT)
_SINK = [None]


# ---------------------------------------------------------------------------
# Trace (LORRAX_MEMORY_TRACE): what is live, what the allocator holds, what compiled
# ---------------------------------------------------------------------------
_T: dict = {"sections": [], "born": {}, "hooked": False}


def _snapshot() -> dict:
    """The live arrays grouped by (shape, dtype, sharding, creating stage), bytes per device."""
    import jax
    devices, groups, per_device = jax.local_devices(), {}, {}
    stage = _OPEN[-1].name if _OPEN else "(top level)"
    born, alive = _T["born"], set()
    for a in jax.live_arrays():
        alive.add(id(a))
        if id(a) not in born:
            born[id(a)] = stage
        nbytes, held = _held(a)
        for device in held:
            per_device[device.id] = per_device.get(device.id, 0) + nbytes
        key = (tuple(a.shape), str(a.dtype), str(getattr(a.sharding, "spec", type(a.sharding).__name__)),
               born[id(a)])
        groups.setdefault(key, [0, nbytes])[0] += 1
    for gone in set(born) - alive:
        del born[gone]
    stats = [d.memory_stats() or {} for d in devices]
    return dict(live=max(per_device.values(), default=0),
                in_use=max(int(st.get("bytes_in_use", 0)) for st in stats),
                peak=max(int(st.get("peak_bytes_in_use", 0)) for st in stats),
                arrays=[dict(shape=list(k[0]), dtype=k[1], spec=k[2], born=k[3], count=v[0],
                             bytes_per_device=v[1]) for k, v in
                        sorted(groups.items(), key=lambda kv: -kv[1][0] * kv[1][1])])


def _hook_compiles() -> None:
    """Record every executable XLA compiles (cache hits included) under the open section."""
    from jax._src import compiler
    original = compiler.compile_or_get_cached

    def traced(backend, computation, *args, **kwargs):
        executable = original(backend, computation, *args, **kwargs)
        try:
            m = executable.get_compiled_memory_stats()
            row = dict(program=str(computation.operation.attributes["sym_name"]).strip('"'),
                       temp=int(m.temp_size_in_bytes), argument=int(m.argument_size_in_bytes),
                       output=int(m.output_size_in_bytes), alias=int(m.alias_size_in_bytes),
                       code=int(m.generated_code_size_in_bytes))
            for t in _OPEN:
                if t.rec is not None:
                    t.rec["compiles"].append(row)
        except Exception:                                      # noqa: BLE001
            pass
        return executable
    compiler.compile_or_get_cached, _T["hooked"] = traced, True
    import atexit
    atexit.register(trace_dump)


def _trace(tracker: Tracker, event: str) -> None:
    try:
        if not _T["hooked"]:
            _hook_compiles()
        last = _T.get("last")
        snap = last[1] if last and last[0] == _COST[0] else _snapshot()
        _T["last"] = (_COST[0], snap)
        if event == "open":
            tracker.rec = dict(name=tracker.name, depth=len(_OPEN) - 1, open=snap, peak=snap,
                               compiles=[], t_open=time.time())
        for t in ([tracker] if event == "close" else _OPEN):
            if t.rec is not None and snap["live"] > t.rec["peak"]["live"]:
                t.rec["peak"] = snap
        if event == "close" and tracker.rec is not None:
            price = tracker.price
            tracker.rec.update(close=snap, t_close=time.time(), measured=tracker.whole,
                               exact=tracker.exact, source=_I["source"],
                               priced=None if price is None else price.terms())
            _T["sections"].append(tracker.rec)
    except Exception:                                          # noqa: BLE001
        pass


def trace_dump(path: str = "") -> Optional[str]:
    """Write the trace (``path``, else LORRAX_MEMORY_TRACE; ``.p<i>`` on process i > 0)."""
    import json
    path = path or TRACE
    if not path or not _T["sections"]:
        return None
    try:
        import jax
        index = jax.process_index()
    except Exception:                                          # noqa: BLE001
        index = 0
    path = path if index == 0 else f"{path}.p{index}"
    with open(path, "w") as out:
        json.dump(dict(run=dict(_RUN), process=index, sections=_T["sections"],
                       receipts=[{k: v for k, v in r.items() if k != "price"} for r in receipts()]),
                  out, default=str)
    return path


def set_sink(sink=None) -> None:
    """Route receipt, route and banner lines through the driver's report."""
    _SINK[0] = sink


def _say(text: str, *, local: bool = False) -> None:
    if local or _rank0():
        try:
            (_SINK[0] or (lambda line: print(line, flush=True)))(text)
        except Exception:                                      # noqa: BLE001
            pass


def _main() -> bool:
    return threading.current_thread() is threading.main_thread()


def _sync() -> None:
    if CALIBRATE:
        try:
            import jax
            jax.block_until_ready(jax.live_arrays())
        except Exception:                                      # noqa: BLE001
            pass


def _reads_every_section() -> bool:
    return CALIBRATE or (_I["source"] or _setup()) in ("pool", "bfc")


def boundary() -> Optional[int]:
    """Close the current interval; its high-water goes to every open tracker."""
    if not _main():
        return None
    t0 = time.perf_counter()
    try:
        high, exact = _interval()
    except Exception:                                          # noqa: BLE001
        high, exact = None, False
    _COST[0] += 1
    _COST[1] += time.perf_counter() - t0
    if high is not None:
        reserved = -1 if _I["reserved"] is None else _I["reserved"]
        for t in _OPEN:
            t.whole = max(t.whole, high)
            if t.mark0 is not None:
                t.metered, t.exact = max(t.metered, high), t.exact and exact
                t.reserved = max(t.reserved, reserved)
        if _OPEN:
            _OPEN[-1].own = max(_OPEN[-1].own, high)
    if TRACE and _OPEN:
        _trace(_OPEN[-1], "boundary")
    return high


def open_tracker(name: str, price: Optional[Price] = None) -> Optional[Tracker]:
    """Open a tracker with the parts booked to ``name``; repeat its banner if it has one."""
    if not _main():
        return None
    parts = _PENDING.pop(str(name), {})
    if price is not None:
        parts[price.label] = price
    tracker = Tracker(name, parts)
    if str(name) in _DOOMED:
        _say(_DOOMED[str(name)])
    if tracker.parts:
        tracker.meter()
    elif _backend_up() and _reads_every_section():
        boundary()
    tracker.base = _I["live"]
    _OPEN.append(tracker)
    if TRACE and _backend_up():
        _trace(tracker, "open")
    return tracker


def close_tracker(tracker: Optional[Tracker]) -> tuple:
    """Close ``tracker``: (whole, own, exact); a priced one records its receipt."""
    if tracker is None:
        return -1, -1, False
    if tracker.mark0 is not None:
        _sync()
    if tracker.mark0 is not None or _reads_every_section():
        boundary()
    if tracker in _OPEN:
        _OPEN.remove(tracker)
    if tracker.mark0 is not None and _I["source"] == "bfc" and CALIBRATE:
        # First raise: a stage that lifted the never-resetting mark peaked at it exactly.
        tracker.exact = _I["mark"] > tracker.mark0
        tracker.metered = _I["mark"] if tracker.exact else tracker.metered
    if TRACE and tracker.rec is not None:
        _trace(tracker, "close")
    if tracker.parts:
        _receipt(tracker)
    elif (STRICT and _I["source"] == "live" and tracker.whole - tracker.base > CENSUS_BYTES
          and not any(t.parts for t in _OPEN)):
        _FINDINGS.append(f"unpriced: {tracker.name} allocated "
                         f"{_gb(tracker.whole - tracker.base)} outside a priced stage")
    return tracker.whole, tracker.own, tracker.exact


def book(price: Price, stage: Optional[str] = None) -> None:
    """Book ``price`` as one part of ``stage`` (its open tracker, innermost; else the next to
    open; ``None``: the innermost open tracker).  A stage's price is the sum of its parts;
    a part booked again under the same label keeps the larger."""
    if not isinstance(price, Price):
        raise TypeError(f"memory.book takes a memory.Price, got {type(price).__name__}")
    if STRICT and price.unsourced():
        _FINDINGS.append(f"no provenance: {stage or price.label}: {', '.join(price.unsourced())}")
    target = None
    if _main():
        target = (_OPEN[-1] if _OPEN else None) if stage is None else next(
            (t for t in reversed(_OPEN) if t.name == stage), None)
    parts = target.parts if target is not None else _PENDING.setdefault(
        str(stage if stage is not None else price.label), {})
    old = parts.get(price.label)
    if old is None or price.bytes() > old.bytes():
        parts[price.label] = price
    key = target.name if target is not None else str(stage or price.label)
    _PLANNED.setdefault(key, {})["parts"] = dict(parts)
    if target is not None:
        target.meter()
    elif stage is None:
        _PENDING.pop(price.label, None)      # no section open (a planning replay): sizing only


def receipt_line(name: str, price: Price, measured: float, exact: bool, src: str,
                 reserved: float = -1) -> str:
    """``memory <stage>: priced P (terms) | measured M | γ g [| reserved R (slack R - M)]``."""
    total = price.bytes()
    if measured is None or measured < 0:
        meas, gamma = "not measured", "–"
    else:
        mark = "" if exact else ({"bfc": ">=", "live": "live "}.get(src, "~") if CALIBRATE
                                 else "~")
        meas, gamma = f"{mark}{_gb(measured)}", (f"{measured / total:.2f}" if total else "–")
    tail = (f" | reserved {_gb(reserved)} (slack {_gb(reserved - measured)})"
            if reserved is not None and reserved >= 0 and measured is not None and measured >= 0
            else "")
    return (f"memory {name}: priced {_gb(total)} ({price.describe()}) | measured {meas} | "
            f"γ {gamma}{tail}")


def _receipt(t: Tracker) -> None:
    src, price = _I["source"] or "none", t.price
    row = _RECEIPTS.setdefault(t.name, dict(stage=t.name, count=0, priced=0.0, measured=-1.0,
                                            exact=True, price=price, source=src, reserved=-1.0))
    row["reserved"] = max(row["reserved"], float(t.reserved))
    first, row["count"] = row["count"] == 0, row["count"] + 1
    rise = t.metered > row["measured"] or price.bytes() > row["priced"]
    if price.bytes() >= row["priced"]:
        row["priced"], row["price"] = price.bytes(), price
    if t.metered > row["measured"]:
        row["measured"], row["exact"] = float(t.metered), t.exact
    if first or rise:
        _say(receipt_line(t.name, price, t.metered, t.exact, src, t.reserved))


def receipts() -> list:
    """The receipt rows (max over calls; the report's gather adds ``measured_ranks``)."""
    return list(_RECEIPTS.values())


def findings() -> list:
    """The census's findings, and parts booked to stages that never opened."""
    return list(_FINDINGS) + [f"never opened: {name}" for name in _PENDING]


# ---------------------------------------------------------------------------
# Vendor workspace: the leg's MEASURED row in its range (no query at trace time)
# ---------------------------------------------------------------------------
def vendor(leg, lib: str, op: str, n: int, ranks: int, itemsize: int,
           unit: Optional[float] = None) -> Term:
    """The vendor workspace per rank: ``leg``'s measured row (GATE outside it under STRICT)."""
    row = dict(leg.vendor_rows).get((lib, op))
    where = (f"no measured row for {leg.key}/{lib}/{op}" if row is None else
             f"{leg.key}/{lib}/{op} at n {int(n)}, P {int(ranks)} outside the measured "
             f"n {row.n[0]}-{row.n[1]}, P {row.p[0]}-{row.p[1]}")
    if row is not None and row.n[0] <= n <= row.n[1] and row.p[0] <= ranks <= row.p[1]:
        where = None
    if where and STRICT:
        raise RuntimeError(f"GATE memory_unpriced: got: {where}; want: a measurement there")
    if where:
        _warn(f"unpriced {leg.key}/{lib}/{op}", 0.0, f"memory: {where}; "
              + ("priced 0" if row is None else "extrapolated"), local=False)
    if row is None:
        return Term(0.0, source=f"UNPRICED {leg.key}/{lib}/{op}")
    nbytes = row.ratio * (row.unit(int(n), int(ranks), int(itemsize)) if unit is None else unit)
    return Term(nbytes, source=("EXTRAPOLATED " if where else "")
                + f"{row.ratio:g} x {row.unit_name} ({row.source})")


# ---------------------------------------------------------------------------
# Limit, budget, room
# ---------------------------------------------------------------------------
_RUN: dict = {}
_AGREE: list = [None]


def outside_allowance(leg, linalg: str = "local") -> float:
    """Bytes outside the allocator: the leg's row, plus its providers' contexts if distributed."""
    return float(leg.outside_allowance) + (float(leg.distributed_context[0])
                                           if linalg == "distributed" else 0.0)


def pool_reservation(total: float, leg) -> float:
    """The pool reserved before the deck: the card less the distributed allowance, >= half."""
    return max(float(total) - outside_allowance(leg, "distributed"), 0.5 * float(total))


def contexts() -> Term:
    """The providers' contexts a route opens, where the budget did not withhold them (local)."""
    leg = _leg()
    size, src = getattr(leg, "distributed_context", (0.0, ""))
    return Term(size if _RUN.get("linalg", "local") == "local" else 0.0, source=src)


def _agree(value: float) -> float:
    """The minimum over processes, through :func:`resolve`'s ``agree`` collective."""
    if _AGREE[0] is not None:
        return float(_AGREE[0](float(value)))
    import jax
    if jax.process_count() > 1:
        raise RuntimeError("memory: several processes and no agreement collective; resolve the "
                           "budget through common.gpu_utils.resolve_device_budget_gb")
    return float(value)


def enforced(stats) -> bool:
    """The allocator refuses past ``bytes_limit`` (a BFC arena; not cuda_async's threshold)."""
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default").lower()
    return allocator in ("default", "bfc") and bool((stats or {}).get("bytes_limit"))


def resolve(deck_bytes: float = 0.0, linalg: str = "local", *, total: Optional[float] = None,
            outside: Optional[float] = None, agree: Optional[Callable] = None) -> float:
    """THE run budget per device (every process enters): limit = an enforced ``bytes_limit``,
    else ``total`` less what is ``outside`` (>= the leg's allowance), CPU the host share;
    budget = (limit - pending contexts) x (1 - reserve) - T_ord, minimum over processes
    (``agree``), never above a positive deck value."""
    import jax
    from lxkit import ordered_transient_bytes
    _AGREE[0] = agree
    leg = _leg()
    stats = jax.local_devices()[0].memory_stats() or {}
    if leg is None or not leg.accelerator:
        lim, pending_out = float(total or 0.0) / max(1, jax.local_device_count()), 0.0
    elif enforced(stats) or not total:
        lim, pending_out = float(stats.get("bytes_limit", 0)), 0.0
    else:
        lim = float(total) - max(float(outside or 0.0), outside_allowance(leg))
        pending_out = outside_allowance(leg, linalg) - outside_allowance(leg)
    rule = (lim - pending_out) * (1.0 - float(getattr(leg, "memory_reserve", 0.0)))
    lim, rule = _agree(lim), _agree(max(0.0, rule - float(ordered_transient_bytes())))
    deck = float(deck_bytes or 0.0)
    if deck > rule and _rank0():
        import warnings
        warnings.warn(f"memory_per_device_gb = {deck / 1e9:g} exceeds the memory model's "
                      f"{rule / 1e9:.2f} GB on this card (linalg = {linalg}); running at "
                      f"{rule / 1e9:.2f} GB", RuntimeWarning, stacklevel=2)
    reset()
    set_budget(min(deck, rule) if deck > 0 else rule, limit_bytes=lim, linalg=linalg,
               per_node=int(getattr(leg, "node_devices", 1)),
               platform=str(getattr(jax.local_devices()[0], "device_kind", "") or "cpu"))
    return float(_RUN["budget"])


def set_budget(budget: float, *, limit_bytes: Optional[float] = None, per_node: int = 1,
               platform: str = "test", linalg: str = "local") -> None:
    """Record the run's figures (:func:`resolve`, or a replay that fakes the card)."""
    lim = float(budget if limit_bytes is None else limit_bytes)
    _RUN.clear()
    _RUN.update(limit=lim, budget=float(budget), reserve=max(0.0, lim - float(budget)),
                linalg=str(linalg), per_node=int(per_node), platform=str(platform))


def budget() -> float:
    """The run's per-device planning budget in bytes (``common.gpu_utils`` resolves it)."""
    return float(_RUN["budget"])


def run() -> dict:
    """The run's figures (empty before :func:`resolve`)."""
    return dict(_RUN)


def in_use() -> float:
    """Allocator bytes in use, max over this process's devices (CPU: live arrays per device)."""
    import jax
    devices = jax.local_devices()
    if devices[0].platform == "cpu":
        return _live_bytes() / len(devices)
    return float(max(int((d.memory_stats() or {}).get("bytes_in_use", 0)) for d in devices))


def room(stage: str = "", *, bucketed: bool = True) -> int:
    """budget - bytes in use, the minimum over processes, bucketed (every process enters)."""
    left = int(_agree(max(0.0, budget() - in_use())))
    return bucket(left) if bucketed else left


def bucket(value) -> int:
    """A whole GiB at or above 1 GiB, exact below, 0 when negative."""
    value = int(value)
    return (value >> 30) << 30 if value >= 1 << 30 else max(value, 0)


# ---------------------------------------------------------------------------
# Admission: the one policy point (owner 2026-10-19), and the sizing estimate
# ---------------------------------------------------------------------------
_ROUTES: dict = {}
_PLANNED: dict = {}      # stage -> {"routes": ((name, Price), ...), "limit": bytes, "parts": {...}}
_DOOMED: dict = {}       # stage -> banner, repeated when it opens
_WARNED: dict = {}


def record_route(site, route, reason, *, warning=None) -> None:
    """Record one route decision; rank 0 prints it the first time."""
    key = (str(site), str(route), str(reason))
    if key in _ROUTES:
        _ROUTES[key]["count"] += 1
        return
    line = f"route {site}: {route} ({reason})" + (f"; WARNING {warning}" if warning else "")
    _ROUTES[key] = dict(site=str(site), route=str(route), reason=str(reason),
                        warning=warning, line=line, count=1)
    _say(("  WARNING: " if warning else "  ") + line)


def routes() -> list:
    return [dict(row) for row in _ROUTES.values()]


def admit(stage: str, routes, room_bytes: float, *, probe=None, forced=None, quiet: bool = False,
          runs_in: Optional[str] = None, local: bool = False, device: bool = True,
          detail: bool = False):
    """THE admission policy (owner 2026-10-19).  ``routes``: ``(name, Price | callable[, room])``
    fastest first, each against its room (default the rank-invariant ``room_bytes``).
    1. The fastest route that fits.  2. None fits and the cheapest is over the physical limit
    (room + reserve): a banner now and when its stage (``runs_in``, else the open one) opens,
    with the minimum process count; it runs, never raises.  3. Otherwise one warning; it runs.
    ``device=False`` (a host share, a tile cap): never a banner, not in :func:`sizing`.
    ``probe()`` raising drops the later routes; ``forced``: a deck-named route; ``quiet``: the
    caller records it; ``local``: warned on its rank.  Returns the name (``detail``: a dict)."""
    routes = (("chosen", routes),) if isinstance(routes, Price) else tuple(routes)
    key = str(runs_in or (_OPEN[-1].name if _OPEN and _main() else stage))

    def record(route, reason, warning=None):
        if not quiet and len(routes) > 1:
            record_route(stage, route, reason, warning=warning)
        return dict(route=route, reason=reason, warning=warning) if detail else route
    if forced is not None:
        return record(forced, "named")
    done = []                    # (name, price, room)
    for i, (name, price, *own) in enumerate(routes):
        room = float(own[0] if own else room_bytes)
        if i == 1 and probe is not None:
            try:
                probe()
            except Exception as exc:                           # noqa: BLE001
                return record(routes[0][0], f"{_gb(done[0][1].bytes())}/rank over room "
                              f"{_gb(done[0][2])}", warning=f"the {name} route is unavailable "
                              f"here ({type(exc).__name__}: {exc})")
        done.append((name, price() if callable(price) else price, room))
        if STRICT and done[-1][1].unsourced():
            _FINDINGS.append(f"no provenance: {stage} ({name}): {', '.join(done[-1][1].unsourced())}")
        if device:
            _PLANNED.setdefault(key, {}).update(routes=tuple(d[:2] for d in done),
                                                limit=room + _RUN.get("reserve", 0.0))
        if done[-1][1].bytes() <= room:
            return record(name, "".join(f"{n} {_gb(p.bytes())}/rank over room {_gb(r)}; "
                                        for n, p, r in done[:-1])
                          + f"{_gb(done[-1][1].bytes())}/rank of room {_gb(room)}")
    name, price, room = min(done, key=lambda item: item[1].bytes() - item[2])
    physical = room + _RUN.get("reserve", 0.0) if device else float("inf")
    reason = "; ".join(f"{n} {_gb(p.bytes())}/rank over room {_gb(r)}" for n, p, r in done)
    if price.bytes() <= physical:
        _warn(stage, price.bytes(), f"memory over budget at {stage}: {name} needs "
              f"{_gb(price.bytes())}/rank, room {_gb(room)}/rank"
              + (f", within the allocator limit {_gb(physical)}" if device else "")
              + "; continuing (an OOM is possible)", local=local)
        return record(name, reason, warning="no route fits the planning room")
    line, rule = sizing()["line"], "!" * 78
    _DOOMED[key] = "\n".join([
        rule, f"!! MEMORY: {stage} is PRICED OVER the allocator limit. It runs anyway.",
        f"!!   {name}: priced {_gb(price.bytes())}/rank ({price.describe()})",
        f"!!   allocator limit beside what is live {_gb(physical)}; short by "
        f"{_gb(price.bytes() - physical)}", "!!   " + line, rule])
    _say(_DOOMED[key], local=local)
    _warn(stage, price.bytes(), f"memory: {stage} priced over the limit ({name} "
          f"{_gb(price.bytes())}/rank over {_gb(physical)}); {line}", local=local)
    return record(name, reason, warning="priced over the allocator limit")


def _warn(stage: str, need: float, message: str, *, local: bool) -> None:
    """A RuntimeWarning once per stage kind (``.N``/``:N`` dropped), again only if it grows."""
    import sys
    import warnings
    kind = re.sub(r"([.:]\d+)+$", "", str(stage))
    if need <= _WARNED.get(kind, -1.0):
        return
    _WARNED[kind] = need
    if _rank0():
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    elif local:
        print("WARNING: " + message, file=sys.stderr, flush=True)


def legal_ranks(largest: int = 4096) -> list:
    """Every count the mesh builder takes: p x p (``collectives.resolve_mesh``), any nodes."""
    return [p * p for p in range(1, math.isqrt(largest) + 1)]


def sizing(stages: Optional[dict] = None, *, limit_bytes: Optional[float] = None,
           per_node: Optional[int] = None, platform: Optional[str] = None) -> dict:
    """The smallest legal P at which every device stage fits: its price the larger of its best
    admitted route and its booked parts, its limit the admission's (else the run's);
    ``binding`` the least headroom there.  ``stages``: ``{stage: ((route, Price), ...)}``."""
    run_lim = float(_RUN.get("limit", 0.0) if limit_bytes is None else limit_bytes)
    plans = _PLANNED if stages is None else {k: dict(routes=v) for k, v in stages.items()}
    per = max(1, int(_RUN.get("per_node", 1) if per_node is None else per_node))
    plat = _RUN.get("platform", "this platform") if platform is None else platform
    limit = {k: float(v.get("limit", run_lim)) for k, v in plans.items()}

    def need(k, p):
        plan = plans[k]
        best = min((pr.bytes(p) for _, pr in plan.get("routes", ())), default=0.0)
        return max(best, sum(pr.bytes(p) for pr in plan.get("parts", {}).values()))
    grid = legal_ranks()
    if not plans:
        return dict(p=None, nodes=None, binding=None, line="minimum to complete: no priced stage")
    for k in plans:
        if need(k, grid[-1]) > limit[k]:
            price = sum(plans[k].get("parts", {}).values(), Price()) + min(
                (pr for _, pr in plans[k].get("routes", ())), default=Price(),
                key=lambda pr: pr.bytes(grid[-1]))
            term = max(TERMS, key=lambda n: getattr(price, n).fixed)
            return dict(p=None, nodes=None, binding=k, line=(
                f"minimum to complete: no P suffices: {k} holds {_gb(getattr(price, term).fixed)}"
                f"/rank of {term} that no mesh divides ({getattr(price, term).source}) against "
                f"the {_gb(limit[k])} limit"))
    for p in grid:
        if all(need(k, p) <= limit[k] for k in plans):
            nodes = -(-p // per)
            binding = max(plans, key=lambda k: need(k, p) / limit[k], default=None)
            return dict(p=p, nodes=nodes, binding=binding, line=(
                f"minimum to complete: P = {p} ({nodes} node{'s' * (nodes != 1)} of {plat})"
                + (f"; binding stage {binding}" if binding else "")))


def reset(*, run: bool = False) -> None:
    """Forget receipts, bookings, routes and plans (a new run in the same process); ``run``:
    the run's figures too."""
    _T["sections"].clear()
    for table in (_RECEIPTS, _PENDING, _ROUTES, _PLANNED, _DOOMED, _WARNED, _FINDINGS) + (
            (_RUN,) if run else ()):
        table.clear()
