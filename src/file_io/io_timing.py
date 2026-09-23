"""Optional SlabIO wall trace; one switch, one report per rank.

API wall times include setup and waits.  Native H5D wall totals are
collected by phdf5 at the actual calls, including collective peer waits.
"""
from __future__ import annotations

import os
import sys
import time
from functools import wraps
from contextlib import contextmanager, nullcontext
from pathlib import Path

import jax

from runtime.env_flags import env_bool


def enabled() -> bool:
    return env_bool("LORRAX_SLAB_IO_TIMING", False)


class SlabIOTiming:
    def __init__(self, path: str, mode: str):
        self.path = path
        self.mode = mode
        self.rank = jax.process_index()
        self.events: list[tuple[str, str, float, str]] = []

    @contextmanager
    def measure(self, op: str, name: str = "-"):
        start = time.perf_counter_ns()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            self.events.append((op, name,
                                (time.perf_counter_ns() - start) / 1e6,
                                status))

    def finish(self, native=None):
        # One buffered append per file handle, never a write in the hot path.
        path = Path(f"slab_io_timing.rank{self.rank:03d}.log")
        try:
            with path.open("a", encoding="utf-8") as out:
                out.write("# API wall includes setup and waits; write_enqueue "
                          "and read_union_dispatch do not mean I/O completion. "
                          "H5D totals include collective peer waits.\n")
                for op, name, ms, status in self.events:
                    out.write(f"path={self.path} mode={self.mode} op={op} "
                              f"ds={name} api_wall_ms={ms:.3f} "
                              f"status={status}\n")
                if native is not None:
                    nr, br, tr, nw, bw, tw = native
                    out.write(f"path={self.path} native_H5Dread_calls={nr} "
                              f"native_H5Dread_bytes={br} "
                              f"native_H5Dread_wall_ms={tr / 1e6:.3f} "
                              f"native_H5Dwrite_calls={nw} "
                              f"native_H5Dwrite_bytes={bw} "
                              f"native_H5Dwrite_wall_ms={tw / 1e6:.3f}\n")
        except OSError as exc:
            print(f"[SlabIO timing rank={self.rank}] cannot write {path}: "
                  f"{exc}", file=sys.stderr, flush=True)
        if self.rank == 0:
            totals: dict[str, tuple[int, float]] = {}
            for op, _, ms, _ in self.events:
                n, total = totals.get(op, (0, 0.0))
                totals[op] = (n + 1, total + ms)
            api = " ".join(f"{op}={n}/{ms:.1f}ms"
                           for op, (n, ms) in totals.items())
            h5d = ("" if native is None else
                   f" H5Dread={native[0]}/{native[2] / 1e6:.1f}ms"
                   f" H5Dwrite={native[3]}/{native[5] / 1e6:.1f}ms")
            print(f"[SlabIO timing r0] {os.path.basename(self.path)} "
                  f"{api}{h5d} per_rank=slab_io_timing.rank*.log",
                  file=sys.stderr, flush=True)


class _DisabledTiming:
    def measure(self, op: str, name: str = "-"):
        return nullcontext()

    def finish(self, native=None):
        return None


DISABLED = _DisabledTiming()


def timed(op: str):
    """Measure the full public method, including setup and implicit drains."""
    def decorate(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            if self._timing is DISABLED:
                return method(self, *args, **kwargs)
            name = args[0] if args else "-"
            label = ("write_slab_sync" if op == "write_enqueue"
                     and self._stack == "h5py" else
                     "read_union_sync" if op == "read_union_dispatch"
                     and self._stack == "h5py" else op)
            with self._timing.measure(label, str(name)):
                return method(self, *args, **kwargs)
        return call
    return decorate
