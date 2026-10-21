"""Lightweight progress reporting for long-running loops.

  LoopProgress   — context manager for Python for-loops (no JAX overhead)

It emits BGW-style output:

    Started frequency integration at 17:50:20.
    [ 17:50:21 | ██████░░░░ |  60% ] tau node  6 / 10 · ETA 2 s
    Finished frequency integration at 17:50:23.  Elapsed: 3 s.

Each line is an INFO record of the ``lorrax.progress`` logger carrying the
``progress`` block, so a driver session shows it in the main log
(``common.log``): ``Started``, one bar line per milestone (``max_updates`` of
them, first and last included) and ``Finished``.  Every bar line gives the
ETA from step 1.  Within one manual stage (``common.timing``), a later loop
with the title of an earlier one is a repeat: the main log shows only its
``Started``, its final bar and its ``Finished``; its other milestones are
untagged INFO, which only the rank stream prints.  Milestone steps are
precomputed as a static boolean mask, and ``enabled=False`` (non-rank-0
processes) logs nothing.

A loop that only dispatches asynchronous work (the Sigma tau sweep) measures
dispatch unless a milestone waits on its ``wait`` value
(``LORRAX_STAGE_SYNC=1``); its ETA is the dispatch clock's.
"""

from __future__ import annotations

import time
from typing import Callable

import jax
import numpy as np

from common.log import block, get_logger


PrintFn = Callable[[str], None]

#: The progress logger; its lines carry the ``progress`` block.
_LOG = get_logger("progress")
_BLOCK = block("progress")


def _say(text: str) -> None:
    _LOG.info(text, extra=_BLOCK)


def _fmt_time(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t))


def _milestone_mask(num_steps: int, max_updates: int) -> np.ndarray:
    """Boolean mask of length num_steps+1 (1-indexed); True at milestone steps."""
    n_emit = min(num_steps, max(1, max_updates))
    milestones = np.rint(np.linspace(1, num_steps, n_emit)).astype(np.int32)
    mask = np.zeros(num_steps + 1, dtype=np.bool_)
    mask[milestones] = True
    return mask


def _repeated_loop(title: str) -> bool:
    """Whether the open manual stage already ran a loop titled ``title``
    (``timing.ManualStage.first_loop``); never outside a manual stage."""
    try:
        from common.timing import current_manual_stage
        stage = current_manual_stage()
        return stage is not None and not stage.first_loop(title)
    except Exception:          # noqa: BLE001 - output never fails a run
        return False


def _format_progress(
    step: int, num_steps: int, elapsed: float,
    title: str, item_name: str, bar_width: int,
) -> str:
    pct = int(round(100.0 * step / num_steps))
    filled = min(max(int(round(bar_width * step / num_steps)), 0), bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    if step >= num_steps:
        eta = 0
    else:
        eta = max(0, int(round(elapsed / max(step, 1) * (num_steps - step))))
    tail = f"ETA {eta} s"
    digits = len(str(num_steps))
    return (
        f"[ {_fmt_time(time.time())} | {bar} | {pct:3d}% ] "
        f"{item_name} {step:{digits}d} / {num_steps} · {tail}"
    )


# ---------------------------------------------------------------------------
#  Python for-loop progress (no JAX overhead)
# ---------------------------------------------------------------------------

class LoopProgress:
    """Progress tracker for Python for-loops.

    Usage::

        total_nodes = sum(w.n_tau for w in windows)
        progress = LoopProgress(total_nodes, title="sigma convolution")
        for win in windows:
            for t_node in win.nodes.t:
                do_work(t_node)
                progress.step()
        progress.finish()

    Or as a context manager (calls finish() automatically)::

        with LoopProgress(n) as p:
            for i in range(n):
                do_work(i)
                p.step()

    ``max_updates`` is the number of milestones, each a bar line in the main
    log.  ``print_fn`` is accepted and unused: the lines go to
    ``lorrax.progress``.
    """

    def __init__(
        self, num_steps: int, print_fn: PrintFn | None = None, *,
        title: str = "loop", item_name: str = "step",
        max_updates: int = 10, bar_width: int = 10,
        enabled: bool | None = None,
    ):
        self.num_steps = num_steps
        self.title = title
        self.item_name = item_name
        self.bar_width = bar_width
        self.enabled = (jax.process_index() == 0) if enabled is None else enabled
        self._mask = _milestone_mask(num_steps, max_updates)
        self._current = 0
        self._start: float | None = None
        self._repeat = False

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
            self._repeat = _repeated_loop(self.title)
            if self.enabled:
                _say(f"Started {self.title} at {_fmt_time(self._start)}.")
        return self

    def step(self, wait=None):
        """Call after each iteration completes.

        ``wait`` is an optional JAX value; under ``LORRAX_STAGE_SYNC=1``
        (``common.timing.stage_sync``) it is blocked on at a milestone step
        before the line is timed, so a loop that only DISPATCHES asynchronous
        work (the Sigma tau sweep) reports execution progress; by default no
        timing-only wait runs and the line reports dispatch progress.  Blocking on a
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
            # A timing-only wait: under LORRAX_STAGE_SYNC=1 alone (common.timing.stage_sync).
            # Without it the line reports how far the loop has dispatched.
            from common.timing import stage_wait
            stage_wait(wait)
        if self.enabled and milestone:
            elapsed = time.time() - self._start
            line = _format_progress(
                self._current, self.num_steps, elapsed,
                self.title, self.item_name, self.bar_width)
            if not self._repeat or self._current == self.num_steps:
                _say(line)
            else:
                _LOG.info(line)          # untagged: the rank stream only

    def finish(self):
        """Print the final summary line."""
        if self._start is not None and self.enabled:
            now = time.time()
            elapsed = int(round(now - self._start))
            _say(f"Finished {self.title} at {_fmt_time(now)}.  Elapsed: {elapsed} s.")
        self._start = None
