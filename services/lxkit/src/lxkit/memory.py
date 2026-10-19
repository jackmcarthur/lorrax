"""Device memory: one price per stage, one room, one admission policy, one receipt.

The single implementation of LORRAX's device-memory model (MEMORY_MODEL_AUDIT §3).
Physics code builds shapes and programs and calls this module; platform
differences are data in ``lxkit.LEGS`` (peak reader, outside allowance, reserve,
measured vendor rows, devices per node).

* :class:`Price`: per-rank bytes in four terms (``operands``, ``xla``, ``vendor``,
  ``outside``), each a :class:`Term` ``fixed + spread / P (+ grid(P))`` with provenance,
  so one Price holds at any process count.
* :func:`stage` / :func:`book`: every ``common.timing`` section is a tracker; a priced
  one waits on the work before it, meters the allocator high-water, waits on its own
  work and prints ``memory <stage>: priced P (operands a, xla b, vendor c, outside d) |
  measured M | γ g`` (γ = measured / priced; ``>=M`` a polled lower bound, ``live M``
  the CPU live-array reading).
* :func:`resolve` / :func:`room`: budget = (limit - pending outside) x (1 - reserve) - T_ord;
  room = budget - bytes in use, the minimum over processes.
* :func:`admit`: the one admission policy (owner 2026-10-19); :func:`sizing`: the
  smallest legal process count at which every planned stage fits, from the same Prices.

Stdlib-only at import; jax is imported lazily.  An instrument never raises.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

TERMS = ("operands", "xla", "vendor", "outside")
#: The census (``LORRAX_MEMORY_STRICT=1``): a MEASURED row outside its range raises ``GATE
#: memory_unpriced``, and :func:`findings` lists every section that allocated more than
#: :data:`CENSUS_BYTES` outside a priced stage and every price term without provenance.
#: Off, an out-of-range row is extrapolated, warned once, its provenance saying so.
STRICT = os.environ.get("LORRAX_MEMORY_STRICT", "") == "1"
CENSUS_BYTES = 64 << 20
#: ``LORRAX_MEMORY_RECEIPTS=calibrate`` (and STRICT): a priced stage waits on its work at
#: entry and exit, every timing section reads the allocator, and a BFC leg polls
#: ``bytes_in_use``, so receipts are exact.  Default (production): no wait and no poll; the
#: allocator is read only where a priced stage opens or closes, and its receipt is ``~``
#: (work still in flight at the boundary is charged to the next one).
CALIBRATE = STRICT or os.environ.get("LORRAX_MEMORY_RECEIPTS", "") == "calibrate"


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
    """One stage's per-rank price at ``ranks`` processes."""
    operands: Term = Term()
    xla: Term = Term()
    vendor: Term = Term()
    outside: Term = Term()
    ranks: int = 1
    label: str = ""
    #: An increment: the stage's resident set is the allocator's bytes in use when it is
    #: metered, added as ``operands`` then (:meth:`settled`).
    increment: bool = False

    def settled(self) -> "Price":
        if not self.increment:
            return self
        live = Term(in_use(), source="resident at stage entry (allocator bytes in use)")
        return replace(self, operands=Term(self.operands.at(self.ranks) + live.fixed,
                                           source=" + ".join(s for s in (self.operands.source,
                                                                          live.source) if s)),
                       increment=False)

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
                     ranks=max(self.ranks, other.ranks),
                     label=" + ".join(s for s in (self.label, other.label) if s))


def priced(nbytes: float, source: str, *, term: str = "xla", spread: float = 0.0,
           ranks: int = 1, increment: bool = False) -> Price:
    """A Price of one term: ``nbytes`` per rank (``fixed``) plus ``spread`` / P;
    ``increment``: new bytes beside the resident set, which the stage adds when it meters."""
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
#: Sampler period of the BFC ``bytes_in_use`` poll, seconds.
POLL_SECONDS = 0.01
_I = {"source": None, "lib": None, "handle": None, "device": None, "mark": -1, "poll": -1,
      "live": -1, "reserved": None}
_LOCK = threading.Lock()
_COST = [0, 0.0]


def _backend_up() -> bool:
    import sys
    bridge = sys.modules.get("jax._src.xla_bridge")
    return bool(bridge and getattr(bridge, "_backends", None))


def _live_bytes() -> int:
    import jax
    return sum(int(s.data.nbytes) for a in jax.live_arrays() for s in a.addressable_shards)


def _setup() -> Optional[str]:
    """The leg's peak reader once the backend is up: "pool" (CUDA mempool USED_MEM_HIGH,
    resettable), "bfc" (peak_bytes_in_use, never resets) or "live" (CPU)."""
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
    """(high-water since the last boundary, exact); starts the next interval.  Sets
    ``_I["reserved"]``: the allocator's reserved high-water over it (None where unread)."""
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


# ---------------------------------------------------------------------------
# Trackers, stages, receipts
# ---------------------------------------------------------------------------

class Tracker:
    """An open section: ``whole``/``own`` high-water (own: outside open children);
    ``metered``: since its price was attached, after a wait (the receipt's figure)."""
    __slots__ = ("name", "price", "whole", "own", "metered", "exact", "mark0", "base", "reserved")

    def __init__(self, name, price):
        self.name, self.price = str(name), price
        self.whole = self.own = self.metered = self.reserved = -1
        self.exact, self.mark0, self.base = True, None, -1

    def meter(self) -> None:
        if self.mark0 is None and _backend_up():
            _sync()
            boundary()
            self.metered, self.exact, self.mark0, self.reserved = -1, CALIBRATE, _I["mark"], -1
        if self.price is not None and self.price.increment and _backend_up():
            self.price = self.price.settled()


_OPEN: list = []                 # the main thread's open trackers, outermost first
_PENDING: dict = {}              # stage -> booked Price
_RECEIPTS: dict = {}             # stage -> receipt row (max over calls)
_FINDINGS: list = []             # the census's findings (STRICT)
_SINK = [None]


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
    if not CALIBRATE:
        return
    try:
        import jax
        jax.block_until_ready(jax.live_arrays())
    except Exception:                                          # noqa: BLE001
        pass


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
    return high


def open_tracker(name: str, price: Optional[Price] = None) -> Optional[Tracker]:
    """Open a tracker (``common.timing`` sections call this); it takes a price booked to
    ``name``, and a doomed stage repeats its banner."""
    if not _main():
        return None
    booked = _PENDING.pop(str(name), None)
    if booked is not None and (price is None or booked.bytes() > price.bytes()):
        price = booked
    tracker = Tracker(name, price)
    if str(name) in _DOOMED:
        _say(_DOOMED[str(name)])
    if price is not None:
        _ROOM[0] = None
        tracker.meter()
    elif CALIBRATE:
        boundary()
    tracker.base = _I["live"]
    _OPEN.append(tracker)
    return tracker


def close_tracker(tracker: Optional[Tracker]) -> tuple:
    """Close ``tracker``: (whole, own, exact); a priced one waits and records its receipt."""
    if tracker is None:
        return -1, -1, False
    if tracker.mark0 is not None:
        _sync()
        _ROOM[0] = None
    if tracker.mark0 is not None or CALIBRATE:
        boundary()
    if tracker in _OPEN:
        _OPEN.remove(tracker)
    if tracker.mark0 is not None and _I["source"] == "bfc" and CALIBRATE:
        # First raise: a stage that lifted the never-resetting mark peaked at it exactly.
        tracker.exact = _I["mark"] > tracker.mark0
        tracker.metered = _I["mark"] if tracker.exact else tracker.metered
    if tracker.price is not None:
        _receipt(tracker)
    elif (STRICT and _I["source"] == "live" and tracker.whole - tracker.base > CENSUS_BYTES
          and not any(t.price is not None for t in _OPEN)):
        _FINDINGS.append(f"unpriced: {tracker.name} allocated "
                         f"{_gb(tracker.whole - tracker.base)} outside a priced stage")
    return tracker.whole, tracker.own, tracker.exact


def book(price: Price, stage: Optional[str] = None, *, planned: Optional[Price] = None) -> None:
    """Book ``price`` to ``stage``: its open tracker (innermost), else the next to open;
    ``None``: the innermost open tracker.  The largest price booked to a stage stands.
    ``planned``: the price :func:`sizing` sees instead (a ledger's planning aggregate beside a
    receipt priced as an increment over what is resident)."""
    if not isinstance(price, Price):
        raise TypeError(f"memory.book takes a memory.Price, got {type(price).__name__}")
    if STRICT and price.unsourced():
        _FINDINGS.append(f"no provenance: {stage or price.label}: {', '.join(price.unsourced())}")
    target = None
    if _main():
        target = (_OPEN[-1] if _OPEN else None) if stage is None else next(
            (t for t in reversed(_OPEN) if t.name == stage), None)
    if target is not None:
        if target.price is None or price.bytes() > target.price.bytes():
            target.price = price
        target.meter()
        _plan_booked(target.name, planned or target.price)
        return
    key = str(stage if stage is not None else price.label)
    _plan_booked(key, planned or price)
    if stage is None:
        return                   # no section open (a planning replay): sizing only
    if key not in _PENDING or price.bytes() > _PENDING[key].bytes():
        _PENDING[key] = price


def _plan_booked(key: str, price: Price) -> None:
    """Sizing sees the largest price booked to ``key``; an admission's routes supersede it."""
    old = _PLANNED.get(key)
    if old is None or (old[0][0][0] == "booked" and price.bytes() > old[0][0][1].bytes()):
        _PLANNED[key] = ((("booked", price),), None)


@contextmanager
def stage(name: str, price: Optional[Price] = None):
    """A priced stage outside ``common.timing`` (a service's own)."""
    tracker = open_tracker(name, price)
    try:
        yield tracker
    finally:
        close_tracker(tracker)


def receipt_line(name: str, price: Price, measured: float, exact: bool, src: str,
                 reserved: float = -1) -> str:
    """``memory <stage>: priced P (terms) | measured M | γ g [| reserved R (slack S)]``: R the
    allocator's reserved high-water, S = R - M (fragmentation and cached free blocks)."""
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
    src = _I["source"] or "none"
    row = _RECEIPTS.setdefault(t.name, dict(stage=t.name, count=0, priced=0.0, measured=-1.0,
                                            exact=True, price=t.price, source=src, reserved=-1.0))
    row["reserved"] = max(row["reserved"], float(t.reserved))
    first, row["count"] = row["count"] == 0, row["count"] + 1
    rise = t.metered > row["measured"] or t.price.bytes() > row["priced"]
    if t.price.bytes() >= row["priced"]:
        row["priced"], row["price"] = t.price.bytes(), t.price
    if t.metered > row["measured"]:
        row["measured"], row["exact"] = float(t.metered), t.exact
    row["path"] = tuple(o.name for o in _OPEN) + (t.name,)
    if first or rise:
        _say(receipt_line(t.name, t.price, t.metered, t.exact, src, t.reserved))


def receipts() -> list:
    """Every priced stage this run, max over its calls on this process."""
    return [dict(row) for row in _RECEIPTS.values()]


def pending() -> dict:
    """Prices booked to stages that never opened (a census finding)."""
    return dict(_PENDING)


def findings() -> list:
    """The census's findings this run (:data:`STRICT`), and prices never taken."""
    return list(_FINDINGS) + [f"never opened: {name}" for name in _PENDING]


# ---------------------------------------------------------------------------
# Vendor workspace: a cached plan-time query, else the leg's MEASURED row in range
# ---------------------------------------------------------------------------
_VENDOR: dict = {}


def remember(lib: str, op: str, n: int, ranks: int, itemsize: int, nbytes: float,
             source: str) -> None:
    """Cache a workspace query issued at plan construction (never at trace time)."""
    _VENDOR[(lib, op, int(n), int(ranks), int(itemsize))] = (float(nbytes), str(source))


def vendor(leg, lib: str, op: str, n: int, ranks: int, itemsize: int,
           unit: Optional[float] = None) -> Term:
    """The vendor workspace term per rank on ``leg``: a remembered query, else the leg's
    measured row (``PlatformLeg.vendor_rows``), GATE memory_unpriced outside its range
    under :data:`STRICT`."""
    hit = _VENDOR.get((lib, op, int(n), int(ranks), int(itemsize)))
    if hit is not None:
        return Term(hit[0], source=hit[1])
    row = dict(leg.vendor_rows).get((lib, op))
    if row is None:
        where = f"no query and no measured row for {leg.key}/{lib}/{op}"
        if STRICT:
            raise RuntimeError(f"GATE memory_unpriced: got: {where}; want: memory.remember "
                               f"at plan construction or a measured row")
        _warn(f"unpriced {leg.key}/{lib}/{op}", 0.0, f"memory: {where}; priced 0", local=False)
        return Term(0.0, source=f"UNPRICED {leg.key}/{lib}/{op}")
    nbytes = row.ratio * (row.unit(int(n), int(ranks), int(itemsize)) if unit is None else unit)
    src = f"{row.ratio:g} x {row.unit_name} ({row.source})"
    if not (row.n[0] <= n <= row.n[1] and row.p[0] <= ranks <= row.p[1]):
        where = (f"{leg.key}/{lib}/{op} at n {int(n)}, P {int(ranks)} outside the measured "
                 f"n {row.n[0]}-{row.n[1]}, P {row.p[0]}-{row.p[1]}")
        if STRICT:
            raise RuntimeError(f"GATE memory_unpriced: got: {where}; want: a measurement there")
        _warn(f"unpriced {leg.key}/{lib}/{op}", 0.0, f"memory: {where}; extrapolated", local=False)
        src = "EXTRAPOLATED " + src
    return Term(nbytes, source=src)


# ---------------------------------------------------------------------------
# Limit, budget, room
# ---------------------------------------------------------------------------
_RUN: dict = {}


def stage_outside(stage: str) -> Term:
    """The bytes ``stage`` adds outside the allocator on this leg (``LEGS.stage_outside``)."""
    leg = _leg()
    row = next((r for r in getattr(leg, "stage_outside", ()) if r[0] == stage), None)
    return Term() if row is None else Term(float(row[1]), source=row[2])


def outside_allowance(leg, linalg: str = "local") -> float:
    """Bytes a process may hold outside the allocator (the leg's row; plus the distributed
    providers' contexts under ``linalg = distributed``)."""
    from lxkit.legs import DISTRIBUTED_CONTEXT_BYTES
    return float(leg.outside_allowance) + (DISTRIBUTED_CONTEXT_BYTES[0]
                                           if linalg == "distributed" else 0.0)


def pool_reservation(total: float, leg) -> float:
    """The pool a process reserves before the deck is read: the card less the larger outside
    allowance, at least half the card."""
    return max(float(total) - outside_allowance(leg, "distributed"), 0.5 * float(total))


_AGREE: list = [None]


def _agree(value: float) -> float:
    """The minimum over processes, through the caller's collective (:func:`resolve`'s
    ``agree``: the source tree's all-gather; lxkit owns none)."""
    if _AGREE[0] is not None:
        return float(_AGREE[0](float(value)))
    import jax
    if jax.process_count() > 1:
        raise RuntimeError("memory: several processes and no agreement collective; resolve the "
                           "budget through common.gpu_utils.resolve_device_budget_gb")
    return float(value)


def _enforced(stats) -> bool:
    """The allocator refuses past ``bytes_limit`` (a BFC arena; not cuda_async's threshold)."""
    import os
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default").lower()
    return allocator in ("default", "bfc") and bool((stats or {}).get("bytes_limit"))


def resolve(deck_bytes: float = 0.0, linalg: str = "local", *, total: Optional[float] = None,
            outside: Optional[float] = None, agree: Optional[Callable] = None) -> float:
    """THE run budget, bytes per device, recorded once (every process enters).

    limit = ``bytes_limit`` where the allocator enforces it, else ``total`` less the bytes
    ``outside`` it now (at least the leg's allowance); CPU: ``total`` (the host share) over
    the local devices.  budget = (limit - pending outside) x (1 - reserve) - T_ord, the
    minimum over processes; ``pending``: the distributed providers' contexts, which open
    later outside an allocator that does not enforce its limit.  A positive deck value can
    only lower it.  ``total``/``outside`` come from the caller's card readers, ``agree``
    (value -> minimum over processes) from its collectives; :func:`room` reuses it."""
    import jax
    from lxkit import ordered_transient_bytes
    _AGREE[0] = agree
    leg = _leg()
    stats = jax.local_devices()[0].memory_stats() or {}
    if leg is None or not leg.accelerator:
        lim, pending_out = float(total or 0.0) / max(1, jax.local_device_count()), 0.0
    elif _enforced(stats) or not total:
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


_ROOM: list = [None]


def stage_room() -> Optional[int]:
    """:func:`room` (exact) read once per stage boundary: the first call after a tracker opens
    or closes reads it (every process enters), later calls reuse it; None before a budget."""
    if "budget" not in _RUN:
        return None
    if _ROOM[0] is None:
        _ROOM[0] = room(bucketed=False)
    return _ROOM[0]


def bucket(value) -> int:
    """A whole GiB at or above 1 GiB, exact below, 0 when negative."""
    value = int(value)
    return (value >> 30) << 30 if value >= 1 << 30 else max(value, 0)


# ---------------------------------------------------------------------------
# Admission: the one policy point (owner 2026-10-19), and the sizing estimate
# ---------------------------------------------------------------------------
_ROUTES: dict = {}
_PLANNED: dict = {}              # stage -> (((route, Price), ...), physical limit or None)
_DOOMED: dict = {}               # stage -> banner, repeated when it opens
_LAST: dict = {}
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


def last_decision() -> dict:
    return dict(_LAST)


def admit(stage: str, routes, room_bytes: float, *, probe=None, forced=None,
          quiet: bool = False, runs_in: Optional[str] = None, local: bool = False,
          limit_bytes: Optional[float] = None) -> str:
    """THE admission policy.  ``routes``: ``(name, Price | callable -> Price[, room])``,
    fastest first (or one Price), each against its room (default ``room_bytes``), the
    planning room (rank-invariant).

    1. A route fits its room: the fastest that fits.
    2. None fits and the cheapest exceeds the physical limit (``limit_bytes``, default its
       room plus the run's reserve): a loud banner now and when ``runs_in`` opens (terms,
       limit, shortfall, the minimum process count); the cheapest runs.  Never raises.
    3. None fits, the cheapest is within the limit: one warning; it runs.

    ``probe()`` raising: the routes after the first cannot run here, the first stays.
    ``forced``: a deck-named route.  ``quiet``: the caller records the route.  ``local``: a
    rank-local price, warned on its rank.  Returns the route's name."""
    routes = (("chosen", routes),) if isinstance(routes, Price) else tuple(routes)
    key = str(runs_in or stage)

    def record(route, reason, warning=None):
        _LAST.clear()
        _LAST.update(route=route, reason=reason, warning=warning)
        if not quiet and len(routes) > 1:
            record_route(stage, route, reason, warning=warning)
        return route
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
        _PLANNED[key] = (tuple(d[:2] for d in done), room + _RUN.get("reserve", 0.0))
        if done[-1][1].bytes() <= room:
            return record(name, "".join(f"{n} {_gb(p.bytes())}/rank over room {_gb(r)}; "
                                        for n, p, r in done[:-1])
                          + f"{_gb(done[-1][1].bytes())}/rank of room {_gb(room)}")
    name, price, room = min(done, key=lambda item: item[1].bytes() - item[2])
    physical = room + _RUN.get("reserve", 0.0) if limit_bytes is None else float(limit_bytes)
    _PLANNED[key] = (_PLANNED[key][0], physical)
    reason = "; ".join(f"{n} {_gb(p.bytes())}/rank over room {_gb(r)}" for n, p, r in done)
    if price.bytes() <= physical:
        _warn(stage, price.bytes(), f"memory over budget at {stage}: {name} needs "
              f"{_gb(price.bytes())}/rank, room {_gb(room)}/rank, within the allocator "
              f"limit {_gb(physical)}; continuing (an OOM is possible)", local=local)
        return record(name, reason, warning="no route fits the planning room")
    line = sizing()["line"]
    rule = "!" * 78
    _DOOMED[key] = "\n".join([
        rule, f"!! MEMORY: {stage} WILL NOT FIT. It runs anyway; the allocator will fail.",
        f"!!   {name}: priced {_gb(price.bytes())}/rank ({price.describe()})",
        f"!!   allocator limit beside what is live {_gb(physical)}; short by "
        f"{_gb(price.bytes() - physical)}", "!!   " + line, rule])
    _say(_DOOMED[key], local=local)
    _warn(stage, price.bytes(), f"memory: {stage} will not fit ({name} {_gb(price.bytes())}"
          f"/rank over the {_gb(physical)} limit); {line}", local=local)
    return record(name, reason, warning="over the allocator limit: this run will run out of memory")


def contexts() -> Term:
    """The distributed providers' contexts a route opens outside the allocator, charged where
    the run's budget did not withhold them (``linalg = local``)."""
    from lxkit.legs import DISTRIBUTED_CONTEXT_BYTES as C
    return Term(C[0] if _RUN.get("linalg", "local") == "local" else 0.0, source=C[1])


def _warn(stage: str, need: float, message: str, *, local: bool) -> None:
    """Once per stage kind (trailing ``.N``/``:N`` dropped) and process, again only when a
    later call needs more.  Rank 0: a RuntimeWarning (kept by production reports)."""
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


def legal_ranks(per_node: int, largest: int = 4096) -> list:
    """Square meshes (the runtime mesh builder and ``distrib_la.place`` take p x p only)
    in whole nodes of ``per_node`` devices."""
    return [p * p for p in range(1, math.isqrt(largest) + 1) if (p * p) % max(1, per_node) == 0]


def sizing(stages: Optional[dict] = None, *, limit_bytes: Optional[float] = None,
           per_node: Optional[int] = None, platform: Optional[str] = None) -> dict:
    """The smallest legal P at which every planned stage's best route fits its limit (the
    physical limit beside what was live at its admission; a booked price, which holds its
    resident set, the run's limit), from the admitted and booked Prices: ``p``, ``nodes``,
    ``binding``, ``line``.  ``stages``: ``{stage: ((route, Price), ...)}``."""
    run_lim = float(_RUN.get("limit", 0.0) if limit_bytes is None else limit_bytes)
    planned = _PLANNED if stages is None else {k: (v, None) for k, v in stages.items()}
    limits = {k: run_lim if lim is None else float(lim) for k, (_, lim) in planned.items()}
    stages = {k: routes for k, (routes, _) in planned.items()}
    per = int(_RUN.get("per_node", 1) if per_node is None else per_node)
    plat = _RUN.get("platform", "this platform") if platform is None else platform
    grid = legal_ranks(per)
    best = lambda routes, p: min(price.bytes(p) for _, price in routes)
    for stage_name, routes in stages.items():
        lim = limits[stage_name]
        if best(routes, grid[-1]) > lim:
            route, price = min(routes, key=lambda item: item[1].bytes(grid[-1]))
            term = max(TERMS, key=lambda n: getattr(price, n).fixed)
            change = {"operands": "its replicated operands sharded over the mesh",
                      "outside": "a route that opens no provider context"}.get(
                term, "a distributed route for its whole-matrix local work")
            return dict(p=None, nodes=None, binding=stage_name, line=(
                f"minimum to complete: no P suffices: {stage_name} ({route}) holds "
                f"{_gb(getattr(price, term).fixed)}/rank of {term} that no mesh divides "
                f"({getattr(price, term).source}) against the {_gb(lim)} limit; needs {change}"))
    binding = None
    for p in grid:
        over = [s for s, routes in stages.items() if best(routes, p) > limits[s]]
        if not over:
            nodes = p // per
            return dict(p=p, nodes=nodes, binding=binding, line=(
                f"minimum to complete: P = {p} ({nodes} node{'s' * (nodes != 1)} of {plat})"
                + (f"; binding stage {binding}" if binding else "")))
        binding = max(over, key=lambda s: best(stages[s], p))
    return dict(p=None, nodes=None, binding=None, line="minimum to complete: no priced stage")


def reset() -> None:
    """Forget receipts, bookings, routes and plans (a new run in the same process)."""
    for table in (_RECEIPTS, _PENDING, _ROUTES, _PLANNED, _DOOMED, _LAST, _WARNED, _FINDINGS):
        table.clear()
