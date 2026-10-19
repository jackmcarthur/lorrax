"""Device memory: one price per stage and one measured receipt against it.

The single implementation of LORRAX's device-memory accounting
(MEMORY_MODEL_AUDIT §3).  Physics code builds shapes and programs and calls
this module; it holds no card sizes and no platform fractions.

Price
    Per-rank bytes of one stage in four terms, each with its provenance:
    ``operands`` (the stage's inputs and outputs, from their shardings),
    ``xla`` (the compiled program's temporaries), ``vendor`` (library
    workspace XLA cannot see) and ``outside`` (bytes the stage opens outside
    the allocator: a new provider context or communicator).  A term is
    ``fixed + spread / P`` (``spread``: global bytes split evenly over the P
    ranks), so one Price is valid at any process count.

Receipt
    ``with memory.stage(name, price):`` blocks on the device work before it,
    reads (and on CUDA resets) the allocator high-water mark, runs the stage,
    blocks on its work, reads the mark again, and prints one rank-0 line::

        memory <stage>: priced P (operands a, xla b, vendor c, outside d) | measured M | γ g

    with γ = measured / priced.  A planner that prices a stage before the
    stage runs books the price to the stage by name (:func:`book`); the stage
    takes it when it opens.  Every :class:`common.timing` section is a tracker
    of this module, so a section opened with a booked price is a stage.

Instrument
    CUDA (``cuda_async``): the pool's ``USED_MEM_HIGH``, reset at every
    boundary (exact).  ROCm and other BFC arenas: ``peak_bytes_in_use`` never
    resets, so a stage that raised the run's mark reads the new mark (exact,
    "first raise"); a stage below an earlier mark reads the largest
    ``bytes_in_use`` a sampler thread polled while it ran, a lower bound
    (marked ``>=``).  CPU: the bytes of the live arrays at the boundaries
    (marked ``live``).  Main thread only; an instrument never raises.

Stdlib-only at import, like the rest of lxkit; jax is imported lazily.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

TERMS = ("operands", "xla", "vendor", "outside")


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Term:
    """Per-rank bytes ``fixed + spread / P (+ grid(P))``, and where they come from."""
    fixed: float = 0.0
    spread: float = 0.0
    source: str = ""
    #: bytes as a function of the process count (a distributed library's
    #: workspace on its grid); None = no such part.
    grid: Optional[Callable[[int], float]] = field(default=None, compare=False)

    def at(self, ranks: int) -> float:
        p = max(1, int(ranks))
        return float(self.fixed) + float(self.spread) / p + (
            float(self.grid(p)) if self.grid is not None else 0.0)


_ZERO = Term()


@dataclass(frozen=True)
class Price:
    """One stage's per-rank price at ``ranks`` processes, in four terms."""
    operands: Term = _ZERO
    xla: Term = _ZERO
    vendor: Term = _ZERO
    outside: Term = _ZERO
    ranks: int = 1
    label: str = ""

    def terms(self, ranks: Optional[int] = None) -> dict:
        p = self.ranks if ranks is None else int(ranks)
        return {name: getattr(self, name).at(p) for name in TERMS}

    def bytes(self, ranks: Optional[int] = None) -> float:
        return float(sum(self.terms(ranks).values()))

    def unsourced(self) -> list:
        """Names of the terms that carry bytes but no provenance."""
        return [n for n in TERMS if getattr(self, n).at(self.ranks) and not getattr(self, n).source]

    def describe(self, ranks: Optional[int] = None) -> str:
        t = self.terms(ranks)
        return ", ".join(f"{n} {_gb(t[n])}" for n in TERMS)

    def __add__(self, other: "Price") -> "Price":
        def join(a: Term, b: Term) -> Term:
            if not b.at(1) and b.grid is None:
                return a
            if not a.at(1) and a.grid is None:
                return b
            ga, gb = a.grid, b.grid
            grid = None if ga is None and gb is None else (
                lambda p: (ga(p) if ga else 0.0) + (gb(p) if gb else 0.0))
            return Term(a.fixed + b.fixed, a.spread + b.spread,
                        " + ".join(s for s in (a.source, b.source) if s), grid)
        return Price(*(join(getattr(self, n), getattr(other, n)) for n in TERMS),
                     ranks=max(self.ranks, other.ranks),
                     label=" + ".join(s for s in (self.label, other.label) if s))


def priced(nbytes: float, source: str, *, term: str = "xla", ranks: int = 1,
           label: str = "") -> Price:
    """A Price with one per-rank term of ``nbytes`` (a closed form or a compiled figure)."""
    return replace(Price(ranks=int(ranks), label=label or source),
                   **{term: Term(fixed=float(nbytes), source=source)})


def _gb(value: float) -> str:
    return f"{float(value) / 1e9:.2f} GB"


# ---------------------------------------------------------------------------
# Instrument: the allocator's high-water mark per interval between boundaries
# ---------------------------------------------------------------------------

_CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
_CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8
#: Sampler period of the BFC ``bytes_in_use`` poll, seconds.
POLL_SECONDS = 0.01

_INST = {"source": None, "lib": None, "handle": None, "device": None,
         "mark": -1, "poll": -1, "live": -1, "thread": None}
_INST_LOCK = threading.Lock()
_COST = [0, 0.0]


def _backend_up() -> bool:
    import sys
    bridge = sys.modules.get("jax._src.xla_bridge")
    return bool(bridge and getattr(bridge, "_backends", None))


def _setup() -> Optional[str]:
    """Choose the reader once the backend is up: "pool", "bfc", "live" or "none"."""
    if not _backend_up():
        return None
    import jax
    from lxkit.gate import device_vendor
    device = jax.local_devices()[0]
    _INST["device"] = device
    vendor = device_vendor(device)
    if vendor == "cpu":
        _INST["source"] = "live"
        return "live"
    if vendor == "cuda":
        pointer = None
        for array in jax.live_arrays():       # one device buffer; nothing is allocated
            try:
                if (array.nbytes and not array.is_deleted()
                        and getattr(array.sharding, "memory_kind", None) in (None, "device")):
                    pointer = array.addressable_shards[0].data.unsafe_buffer_pointer()
                    break
            except Exception:                                  # noqa: BLE001
                continue
        if pointer is None:
            return None                       # no live buffer yet: ask again next time
        import ctypes
        lib = ctypes.CDLL("libcuda.so.1")
        handle = ctypes.c_void_p()
        rc = lib.cuPointerGetAttribute(ctypes.byref(handle), _CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                                       ctypes.c_uint64(pointer))
        if rc == 0 and handle.value:
            _INST.update(lib=lib, handle=handle, source="pool")
            return "pool"
    stats = device.memory_stats() or {}
    if "peak_bytes_in_use" in stats:
        _INST.update(source="bfc", mark=int(stats["peak_bytes_in_use"]),
                     poll=int(stats.get("bytes_in_use", 0)))
        _start_sampler()
        return "bfc"
    _INST["source"] = "none"
    return "none"


def _start_sampler() -> None:
    """Poll ``bytes_in_use`` into the interval maximum (BFC arenas only)."""
    def loop():
        device = _INST["device"]
        while True:
            time.sleep(POLL_SECONDS)
            try:
                use = int((device.memory_stats() or {}).get("bytes_in_use", 0))
            except Exception:                                  # noqa: BLE001
                return
            with _INST_LOCK:
                if use > _INST["poll"]:
                    _INST["poll"] = use
    thread = threading.Thread(target=loop, name="lorrax-memory-sampler", daemon=True)
    _INST["thread"] = thread
    thread.start()


def _live_bytes() -> int:
    import jax
    total = 0
    for array in jax.live_arrays():
        try:
            total += sum(int(s.data.nbytes) for s in array.addressable_shards)
        except Exception:                                      # noqa: BLE001
            continue
    return total


def _interval() -> tuple:
    """(high-water bytes since the last boundary, exact) and start a new interval."""
    source = _INST["source"] or _setup()
    if source == "pool":
        import ctypes
        value = ctypes.c_uint64(0)
        lib, handle = _INST["lib"], _INST["handle"]
        if lib.cuMemPoolGetAttribute(handle, _CU_MEMPOOL_ATTR_USED_MEM_HIGH, ctypes.byref(value)):
            return None, False
        high = int(value.value)
        value.value = 0
        lib.cuMemPoolSetAttribute(handle, _CU_MEMPOOL_ATTR_USED_MEM_HIGH, ctypes.byref(value))
        return high, True
    if source == "bfc":
        stats = _INST["device"].memory_stats() or {}
        peak, use = int(stats.get("peak_bytes_in_use", 0)), int(stats.get("bytes_in_use", 0))
        with _INST_LOCK:
            polled, _INST["poll"] = max(_INST["poll"], use), use
        raised = peak > _INST["mark"]
        _INST["mark"] = max(peak, _INST["mark"])
        return (peak, True) if raised else (polled, False)
    if source == "live":
        live = _live_bytes()
        high, _INST["live"] = max(_INST["live"], live), live
        return high, False
    return None, False


def source() -> Optional[str]:
    """What the instrument reads: "pool", "bfc", "live", "none", or None before the backend."""
    return _INST["source"]


def instrument_cost() -> tuple:
    """(boundary reads, seconds) the instrument has cost this process."""
    return int(_COST[0]), float(_COST[1])


def high_water() -> Optional[int]:
    """This process's high-water bytes since the last boundary, without starting a new
    interval on BFC and CPU (a progress line's reading).  None before the backend."""
    try:
        src = _INST["source"] or _setup()
        if src == "pool":
            import ctypes
            value = ctypes.c_uint64(0)
            if _INST["lib"].cuMemPoolGetAttribute(_INST["handle"], _CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                                                  ctypes.byref(value)):
                return None
            return int(value.value)
        if src == "bfc":
            return int((_INST["device"].memory_stats() or {}).get("peak_bytes_in_use", 0)) or None
        if src == "live":
            return _live_bytes()
    except Exception:                                          # noqa: BLE001
        return None
    return None


# ---------------------------------------------------------------------------
# Trackers, stages and receipts
# ---------------------------------------------------------------------------

class Tracker:
    """One open interval set: a timing section, or a priced stage.

    ``whole``/``own``: the high-water over the tracker's life and over its own
    intervals (outside open children).  ``metered``: the high-water since its price
    was attached, after a wait on the work dispatched before it (the receipt's figure).
    """
    __slots__ = ("name", "price", "whole", "own", "metered", "exact", "mark0")

    def __init__(self, name: str, price: Optional[Price]):
        self.name, self.price = str(name), price
        self.whole = self.own = self.metered = -1
        self.exact = True
        self.mark0 = None            # BFC mark when metering began; None = not metering

    def _meter(self) -> None:
        """Start metering: wait for the work before the stage, close that interval."""
        if self.mark0 is None and _backend_up():
            _sync()
            boundary()
            self.metered, self.exact, self.mark0 = -1, True, _INST["mark"]


_OPEN: list = []                 # main thread's open trackers, outermost first
_PENDING: dict = {}              # stage name -> booked Price
_RECEIPTS: dict = {}             # stage name -> receipt row (max over calls)
_SINK = [None]


def set_sink(sink=None) -> None:
    """Route receipt lines through the active driver's report (``common.timing.set_trace_sink``)."""
    _SINK[0] = sink


def _main() -> bool:
    return threading.current_thread() is threading.main_thread()


def boundary() -> Optional[int]:
    """Close the current interval: its high-water goes to every open tracker and to the
    innermost one's own figure.  Returns it (None: nothing to read)."""
    if not _main():
        return None
    t0 = time.perf_counter()
    try:
        high, exact = _interval()
    except Exception:                                          # noqa: BLE001
        high, exact = None, False
    finally:
        _COST[0] += 1
        _COST[1] += time.perf_counter() - t0
    if high is None:
        return None
    for tracker in _OPEN:
        tracker.whole = max(tracker.whole, high)
        if tracker.mark0 is not None:
            tracker.metered = max(tracker.metered, high)
            tracker.exact = tracker.exact and exact
    if _OPEN:
        _OPEN[-1].own = max(_OPEN[-1].own, high)
    return high


def _sync() -> None:
    """Block on every dispatched device computation of this process."""
    try:
        import jax
        jax.block_until_ready(jax.live_arrays())
    except Exception:                                          # noqa: BLE001
        pass


def open_tracker(name: str, price: Optional[Price] = None) -> Optional[Tracker]:
    """Open a tracker (``common.timing`` sections call this); it takes a price booked
    for ``name``.  A priced tracker first blocks on the work dispatched before it."""
    if not _main():
        return None
    booked = _PENDING.pop(str(name), None)
    if booked is not None:
        price = booked if price is None or booked.bytes() > price.bytes() else price
    tracker = Tracker(name, price)
    if price is not None:
        tracker._meter()
    else:
        boundary()
    _OPEN.append(tracker)
    return tracker


def close_tracker(tracker: Optional[Tracker]) -> tuple:
    """Close ``tracker``: (whole high-water, own high-water, exact).  A priced tracker
    blocks on its work first and records its receipt."""
    if tracker is None:
        return -1, -1, False
    metering = tracker.mark0 is not None
    if metering:
        _sync()
    boundary()
    if tracker in _OPEN:
        _OPEN.remove(tracker)
    if metering and _INST["source"] == "bfc":
        # First raise: a stage that lifted the never-resetting mark peaked at it exactly.
        tracker.exact = _INST["mark"] > tracker.mark0
        if tracker.exact:
            tracker.metered = _INST["mark"]
    if tracker.price is not None:
        _receipt(tracker)
    return tracker.whole, tracker.own, tracker.exact


def book(price: Price, stage: Optional[str] = None) -> None:
    """Book ``price`` to ``stage``: the open tracker of that name (innermost), else the
    next one to open; ``stage=None`` books to the innermost open tracker.  Of several
    prices booked to one stage the largest stands."""
    if not isinstance(price, Price):
        raise TypeError(f"memory.book takes a memory.Price, got {type(price).__name__}")
    target = None
    if _main():
        if stage is None:
            target = _OPEN[-1] if _OPEN else None
        else:
            target = next((t for t in reversed(_OPEN) if t.name == stage), None)
    if target is not None:
        if target.price is None or price.bytes() > target.price.bytes():
            target.price = price
        target._meter()
        return
    key = str(stage) if stage is not None else "(unscoped)"
    old = _PENDING.get(key)
    if old is None or price.bytes() > old.bytes():
        _PENDING[key] = price


@contextmanager
def stage(name: str, price: Optional[Price] = None):
    """Run a priced stage: sync, reset, run, sync, read, receipt (see the module doc)."""
    tracker = open_tracker(name, price)
    try:
        yield tracker
    finally:
        close_tracker(tracker)


def _rank0() -> bool:
    from lxkit.gate import rank0
    return rank0()


def receipt_line(name: str, price: Price, measured: float, exact: bool, src: str) -> str:
    total = price.bytes()
    if measured is None or measured < 0:
        meas, gamma = "not measured", "–"
    else:
        mark = "" if exact else (">=" if src == "bfc" else ("live " if src == "live" else "~"))
        meas = f"{mark}{_gb(measured)}"
        gamma = f"{measured / total:.2f}" if total > 0 else "–"
    return (f"memory {name}: priced {_gb(total)} ({price.describe()}) | measured {meas} "
            f"| γ {gamma}")


def _receipt(tracker: Tracker) -> None:
    price, src = tracker.price, _INST["source"] or "none"
    row = _RECEIPTS.get(tracker.name)
    first = row is None
    if first:
        row = _RECEIPTS[tracker.name] = dict(stage=tracker.name, count=0, priced=0.0,
                                             measured=-1.0, exact=True, price=price, source=src)
    row["count"] += 1
    rise = tracker.metered > row["measured"] or price.bytes() > row["priced"]
    if price.bytes() >= row["priced"]:
        row["priced"], row["price"] = price.bytes(), price
    row["path"] = tuple(t.name for t in _OPEN) + (tracker.name,)
    if tracker.metered > row["measured"]:
        row["measured"], row["exact"] = float(tracker.metered), tracker.exact
    if (first or rise) and _rank0():
        line = receipt_line(tracker.name, price, tracker.metered, tracker.exact, src)
        try:
            (_SINK[0] or (lambda text: print(text, flush=True)))(line)
        except Exception:                                      # noqa: BLE001
            pass


def receipts() -> list:
    """Every priced stage this run, max over its calls on this process."""
    return [dict(row) for row in _RECEIPTS.values()]


def gather(all_gather) -> None:
    """Every rank's measured figure per receipt, by ``all_gather(np.ndarray)`` (every
    process enters; the stage sets may differ).  Rows gain ``measured_ranks``."""
    import hashlib
    import numpy as np
    key = lambda name: int.from_bytes(hashlib.blake2b(name.encode(), digest_size=7).digest(),
                                      "little")
    rows = list(_RECEIPTS.values())
    local = np.array([(key(r["stage"]), int(r["measured"])) for r in rows],
                     dtype=np.int64).reshape(-1, 2)
    counts = np.asarray(all_gather(np.asarray(len(local), dtype=np.int64))).reshape(-1)
    padded = np.full((max(1, int(counts.max(initial=0))), 2), -1, dtype=np.int64)
    padded[:len(local)] = local
    table = np.asarray(all_gather(padded))
    ranks = [dict(table[r, :int(counts[r])].tolist()) for r in range(len(counts))]
    for row in rows:
        k = key(row["stage"])
        row["measured_ranks"] = [float(rank[k]) if rank.get(k, -1) >= 0 else float("nan")
                                 for rank in ranks]


def pending() -> dict:
    """Prices booked to stages that never opened (a census finding)."""
    return dict(_PENDING)


def reset() -> None:
    """Forget receipts and bookings (a new run in the same process)."""
    _RECEIPTS.clear()
    _PENDING.clear()


__all__ = ["Price", "Term", "TERMS", "book", "boundary", "close_tracker", "high_water",
           "gather", "instrument_cost", "open_tracker", "priced", "receipt_line", "receipts", "pending",
           "reset", "set_sink", "source", "stage"]
