"""Lightweight progress reporting for long-running loops.

  LoopProgress   — context manager for Python for-loops (no JAX overhead)

It emits BGW-style output:

    Started frequency integration at 17:50:20.
    [ 17:50:21 | ██████░░░░ |  60% ] tau node  6 / 10 · ETA 2 s
    Finished frequency integration at 17:50:23.  Elapsed: 3 s.

Milestone steps are precomputed as a static boolean mask, and
``enabled=False`` (non-rank-0 processes) prints nothing.
"""

from __future__ import annotations

import time
from typing import Callable

import jax
import numpy as np


PrintFn = Callable[[str], None]


def _fmt_time(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def _milestone_mask(num_steps: int, max_updates: int) -> np.ndarray:
    """Boolean mask of length num_steps+1 (1-indexed); True at milestone steps."""
    n_emit = min(num_steps, max(1, max_updates))
    milestones = np.rint(np.linspace(1, num_steps, n_emit)).astype(np.int32)
    mask = np.zeros(num_steps + 1, dtype=np.bool_)
    mask[milestones] = True
    return mask


def _format_progress(
    step: int, num_steps: int, elapsed: float,
    title: str, item_name: str, bar_width: int,
) -> str:
    pct = int(round(100.0 * step / num_steps))
    filled = min(max(int(round(bar_width * step / num_steps)), 0), bar_width)
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
    if step >= num_steps:
        eta = 0
    else:
        eta = max(0, int(round(elapsed / max(step, 1) * (num_steps - step))))
    digits = len(str(num_steps))
    return (
        f"[ {_fmt_time(time.time())} | {bar} | {pct:3d}% ] "
        f"{item_name} {step:{digits}d} / {num_steps} \u00b7 ETA {eta} s"
    )


# ---------------------------------------------------------------------------
#  Python for-loop progress (no JAX overhead)
# ---------------------------------------------------------------------------

class LoopProgress:
    """Progress tracker for Python for-loops.

    Usage::

        total_nodes = sum(w.n_tau for w in windows)
        progress = LoopProgress(total_nodes, print_fn, title="sigma convolution")
        for win in windows:
            for t_node in win.nodes.t:
                do_work(t_node)
                progress.step()
        progress.finish()

    Or as a context manager (calls finish() automatically)::

        with LoopProgress(n, print_fn) as p:
            for i in range(n):
                do_work(i)
                p.step()
    """

    def __init__(
        self, num_steps: int, print_fn: PrintFn, *,
        title: str = "loop", item_name: str = "step",
        max_updates: int = 10, bar_width: int = 10,
        enabled: bool | None = None,
    ):
        self.num_steps = num_steps
        self.print_fn = print_fn
        self.title = title
        self.item_name = item_name
        self.bar_width = bar_width
        self.enabled = (jax.process_index() == 0) if enabled is None else enabled
        self._mask = _milestone_mask(num_steps, max_updates)
        self._current = 0
        self._start: float | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()

    def start(self):
        """Begin timing (and print the banner) BEFORE the first iteration.

        ``step()`` starts the clock lazily, which is the right default for a
        loop whose first iteration is cheap: the banner then carries a
        meaningful timestamp.  It is the WRONG default when the point of the
        cadence is that the stage must not be silent *while* it runs — a lazy
        banner appears only once the first (possibly hour-long) iteration has
        already finished, i.e. exactly when it is no longer needed.  Callers
        in that situation call ``start()`` first.

        Idempotent, and a no-op for every existing caller (``step()``'s lazy
        branch is unchanged and simply finds the clock already running).
        """
        if self._start is None:
            self._start = time.time()
            if self.enabled:
                self.print_fn(f"Started {self.title} at {_fmt_time(self._start)}.")
        return self

    def step(self, wait=None):
        """Call after each iteration completes.

        ``wait`` is an optional JAX value; at a milestone step it is
        blocked on before the line is timed, so a loop that only DISPATCHES
        asynchronous work (the Sigma tau sweep) reports execution progress
        rather than how far ahead the Python loop has run.  Blocking on a
        JAX value can still be globally sharded, so every process performs
        the wait (INVARIANTS row 21).  Non-milestone steps never block.
        """
        if self._start is None:
            self.start()
        self._current += 1
        milestone = (
            self._current <= self.num_steps and self._mask[self._current])
        # ``enabled`` defaults to process_index() == 0.  A wait on a JAX
        # value must nevertheless be reached by every process: a caller can
        # pass a globally sharded value even though today's Sigma caller uses
        # a local shard.  Only formatting and printing are rank-conditional
        # (INVARIANTS row 21).
        if milestone and wait is not None:
            jax.block_until_ready(wait)
        if self.enabled and milestone:
            elapsed = time.time() - self._start
            self.print_fn(_format_progress(
                self._current, self.num_steps, elapsed,
                self.title, self.item_name, self.bar_width))

    def finish(self):
        """Print the final summary line."""
        if self._start is not None and self.enabled:
            now = time.time()
            elapsed = int(round(now - self._start))
            self.print_fn(f"Finished {self.title} at {_fmt_time(now)}.  Elapsed: {elapsed} s.")
        self._start = None
