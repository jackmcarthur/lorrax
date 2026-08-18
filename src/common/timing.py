from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable

# ---------------------------------------------------------------------------
# STAGE CADENCE (scorecard AI; the observability gap AC.2 / AC.3c / AF.4c
# each hit independently).
#
# ``timing.section`` accumulates into a tree that is printed ONCE, at the
# end of the run.  Every stage in this codebase is therefore silent while
# it runs, and three separate 30 min-to-3 h stages (the ``pzheevd`` eigh,
# the restart-tensor write, the screening solve) have now been
# indistinguishable from a hang while alive — a job that prints nothing
# cannot be triaged, only waited out or killed.
#
# ``LORRAX_TIMING_TRACE=1`` makes every section announce its entry and
# exit with a wall-clock timestamp on rank 0.  That is a milestone
# cadence for the WHOLE code, including stages that are one monolithic
# jit call and so cannot carry a ``LoopProgress``.  Print-only; the
# accumulated tree and the final report are byte-identical either way.
# ``LORRAX_TIMING_TRACE_DEPTH`` caps nesting depth (default 3).
#
# Sections that must ALWAYS announce (env-independent stage cadence —
# e.g. the screening phases, whose silence was paid for three times:
# AC.2's 30-minute silent pzheevd, AF.4c's 2 h 55 m silent restart
# write, the 2.5 h silent c2406 screening stage) pass
# ``announce=True`` (optionally with a human ``label``) to
# ``timing.section``.  Same formatting path, same rank-0 gate, no
# depth cap — ONE cadence mechanism, not two.
# ---------------------------------------------------------------------------
# Both knobs are read at USE time, not import time: common.timing is
# imported by essentially every LORRAX CLI, and an import-time ``int()``
# of a malformed LORRAX_TIMING_TRACE_DEPTH killed every rank before
# main() even ran, even with tracing disabled — an observability knob
# must never take down the run it observes (QUALITY_PATTERNS #8; audit
# 2026-07-28).  Use-time reads also match the rest of LORRAX's env
# flags (cf. ``file_io._slab_io_ffi._env_flag``): tests/drivers that set
# the env after import see the new value.
_TRACE_TRUE = ("1", "true", "yes", "on")
_TRACE_RANK0: bool | None = None


def _trace_flag() -> bool:
    return (os.environ.get("LORRAX_TIMING_TRACE", "0").strip().lower()
            in _TRACE_TRUE)


def _trace_depth() -> int:
    """``LORRAX_TIMING_TRACE_DEPTH`` (nesting-depth cap, default 3).

    Unset/empty → default (the module falsy-parse convention, matching
    :func:`_trace_flag`); anything else must be a plain int.  Malformed
    input refuses with the variable name (doctrine 3) — and only once
    tracing is actually on, since :func:`_trace_enabled` short-circuits
    on the flag first.
    """
    raw = os.environ.get("LORRAX_TIMING_TRACE_DEPTH", "").strip()
    if not raw:
        return 3
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"LORRAX_TIMING_TRACE_DEPTH={raw!r} is not a valid trace "
            f"depth: expected a plain integer nesting cap (default 3)."
        ) from None


def _rank0() -> bool:
    """Rank-0 gate for cadence prints.

    Memoizes ONLY once ``jax.distributed`` is initialised.  Pre-init,
    ``jax.process_index()`` reports 0 on EVERY process — the previous
    first-call-wins cache could latch True on all ranks for the whole
    run (P duplicate cadence lines) if any announced section ran before
    ``jax.distributed.initialize`` — and calling it would also force
    XLA backend init as a side effect of a print path (the
    QUALITY_PATTERNS #8 init-order hazard).  We therefore consult only
    ``jax._src.distributed.global_state`` (pure state, no backend
    init): pre-init and no-jax callers get an UNCACHED True, and the
    first post-init call resolves and caches the real rank (audit
    2026-07-28).
    """
    global _TRACE_RANK0
    if _TRACE_RANK0 is not None:
        return _TRACE_RANK0
    try:
        from jax._src import distributed as _jax_distributed
    except ImportError:
        # No jax in this interpreter (standalone tooling): single
        # process, rank 0 by definition.
        _TRACE_RANK0 = True
        return True
    state = getattr(_jax_distributed, "global_state", None)
    if state is None:
        # Compat fallback: jax internals moved ``global_state`` (not the
        # case for the pinned jax>=0.9).  Fall back to the process
        # index, cached — the pre-fix behaviour, better than printing on
        # every rank forever.
        try:
            import jax
            _TRACE_RANK0 = jax.process_index() == 0
        except (ImportError, RuntimeError):
            _TRACE_RANK0 = True
        return _TRACE_RANK0
    pid = getattr(state, "process_id", None)
    if getattr(state, "client", None) is None or pid is None:
        # Distributed runtime not (yet) initialised: either a
        # single-process run (never initialises; True is correct on
        # every call) or a multi-process run before
        # ``jax.distributed.initialize`` (identity unknowable yet — do
        # NOT cache, so post-init calls resolve correctly).
        return True
    _TRACE_RANK0 = int(pid) == 0
    return _TRACE_RANK0


def _trace_enabled(depth: int) -> bool:
    if not _trace_flag() or depth > _trace_depth():
        return False
    return _rank0()


def _trace(msg: str) -> None:
    print(f"[stage {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TimingNode:
	__slots__ = ("name", "count", "inclusive", "exclusive", "children")

	def __init__(self, name: str):
		self.name = name
		self.count = 0
		self.inclusive = 0.0
		self.exclusive = 0.0
		self.children: OrderedDict[str, TimingNode] = OrderedDict()

	def child(self, name: str) -> "TimingNode":
		node = self.children.get(name)
		if node is None:
			node = TimingNode(name)
			self.children[name] = node
		return node

	def record(self, inclusive: float, exclusive: float) -> None:
		self.count += 1
		self.inclusive += inclusive
		self.exclusive += exclusive


class TimingSection:
	__slots__ = ("collector", "node", "stack", "start", "child_elapsed",
	             "_watchers", "announce", "label")

	def __init__(self, collector: "TimingCollector", node: TimingNode,
	             stack: list["TimingSection"], *, announce: bool = False,
	             label: str | None = None):
		self.collector = collector
		self.node = node
		self.stack = stack
		self.start = 0.0
		self.child_elapsed = 0.0
		self._watchers: list[Callable[[], Any]] = []
		# ``announce=True``: enter/exit lines are printed regardless of
		# LORRAX_TIMING_TRACE (rank-0 only), through the SAME _trace
		# formatter.  ``label`` replaces the node name in those lines
		# (the tree/report always keeps the node name).
		self.announce = announce
		self.label = label

	def _display_name(self) -> str:
		return self.label if self.label else self.node.name

	def _cadence_on(self) -> bool:
		return (_trace_enabled(len(self.stack))
		        or (self.announce and _rank0()))

	def __enter__(self) -> "TimingSection":
		self.stack.append(self)
		if self._cadence_on():
			_trace("  " * (len(self.stack) - 1) + f"-> {self._display_name()}")
		self.start = time.perf_counter()
		self.child_elapsed = 0.0
		self._watchers = []
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		if exc_type is None and self._watchers:
			for watcher in self._watchers:
				watcher()
		end = time.perf_counter()
		inclusive = end - self.start
		exclusive = inclusive - self.child_elapsed
		with self.collector._lock:
			self.node.record(inclusive, max(0.0, exclusive))
		if self._cadence_on():
			_trace("  " * (len(self.stack) - 1)
			       + f"<- {self._display_name()}  {inclusive:.1f} s"
			       + ("  [EXC]" if exc_type is not None else ""))
		self.stack.pop()
		if self.stack:
			self.stack[-1].child_elapsed += inclusive
		# Do not suppress exceptions
		return False

	def watch(self, *values: Any) -> None:
		for value in values:
			self._collect_watchers(value)

	def _collect_watchers(self, value: Any) -> None:
		if value is None:
			return
		blocker = getattr(value, "block_until_ready", None)
		if callable(blocker):
			self._watchers.append(blocker)
			return
		if callable(value):
			self._watchers.append(value)
			return
		if isinstance(value, dict):
			for item in value.values():
				self._collect_watchers(item)
			return
		if isinstance(value, (list, tuple, set)):
			for item in value:
				self._collect_watchers(item)


class TimingCollector:
	def __init__(self) -> None:
		self._root = TimingNode("root")
		self._lock = threading.RLock()
		self._local = threading.local()

	def reset(self) -> None:
		with self._lock:
			self._root = TimingNode("root")
		stack = getattr(self._local, "stack", None)
		if stack is not None:
			stack.clear()

	def _stack(self) -> list[TimingSection]:
		stack = getattr(self._local, "stack", None)
		if stack is None:
			stack = []
			setattr(self._local, "stack", stack)
		return stack

	def section(self, name: str, *, announce: bool = False,
	            label: str | None = None) -> TimingSection:
		stack = self._stack()
		parent = stack[-1].node if stack else self._root
		node = parent.child(name)
		return TimingSection(self, node, stack, announce=announce, label=label)

	def timed(self, name: str | None = None, *, watch: bool = False) -> Callable:
		def decorator(func: Callable) -> Callable:
			label = name or func.__qualname__

			@wraps(func)
			def wrapper(*args, **kwargs):
				with self.section(label) as sec:
					result = func(*args, **kwargs)
					if watch:
						sec.watch(result)
					return result

			return wrapper

		return decorator

	def _rows(self, min_percent: float | None, max_depth: int | None) -> tuple[float, list[str]]:
		children = list(self._root.children.values())
		total = sum(child.inclusive for child in children)
		if total <= 0.0:
			return 0.0, []

		rows: list[str] = []
		header = f"{'Section':<48} {'Count':>7} {'Total[s]':>11} {'Self[s]':>11} {'%':>6}"
		rows.append(header)

		def emit(node: TimingNode, depth: int) -> None:
			if max_depth is not None and depth > max_depth:
				return
			line = f"{'  ' * depth}{node.name}"
			percent = (node.inclusive / total * 100.0) if total > 0.0 else 0.0
			if min_percent is not None and percent < min_percent:
				pass
			else:
				rows.append(
					f"{line:<48} {node.count:>7d} {node.inclusive:>11.3f} {node.exclusive:>11.3f} {percent:>6.1f}"
				)
			for child in node.children.values():
				emit(child, depth + 1)

		for child in children:
			emit(child, 0)
		return total, rows

	def record(self, name: str, seconds: float, *, count: int = 1) -> None:
		"""Add ``seconds`` to a TOP-LEVEL row without wrapping any code.

		The escape hatch for a phase that cannot take a ``with`` block —
		in practice a driver's PROLOGUE, which runs before the first
		timed stage and whose statements are owned by someone else.
		Wrapping it would mean re-indenting the block; this needs one
		line at the end of it:

		    _t0 = time.perf_counter()
		    ...prologue, unchanged...
		    timing.record("gw_jax.startup", time.perf_counter() - _t0)

		Recorded as inclusive == exclusive: a ``record``ed row has no
		children by construction, so it cannot double-count a nested
		``section``.  Repeated calls with the same name ACCUMULATE, like
		re-entering a section.
		"""
		with self._lock:
			self._root.child(name).record(float(seconds), float(seconds))
			if count != 1:
				self._root.child(name).count += count - 1

	def report(
		self,
		*,
		print_fn: Callable[[str], Any] = print,
		title: str = "--- Timing (seconds) ---",
		min_percent: float | None = None,
		max_depth: int | None = None,
		wall: float | None = None,
	) -> None:
		"""Print the accumulated tree.

		``wall`` — the driver's own end-to-end wall clock, in seconds.
		When given, two extra rows close the table:

		    (untimed)   wall − Σ(top-level rows)
		    TOTAL       wall

		WHY THIS IS NOT COSMETIC.  Without them the table reports only
		what someone remembered to wrap, and a reader has no way to tell
		a complete accounting from a 43%-complete one.  That is exactly
		how a 4633 s exciton-bandstructure run read as "866 s htransform
		+ 1767 s solve and ~2000 s of mystery" when the ~2000 s was in
		fact a fully-executed second pass the table simply did not name
		(job 7882533).  A table that always sums to the wall makes the
		gap a number instead of a question.
		"""
		total, rows = self._rows(min_percent, max_depth)
		if not rows:
			print_fn(f"{title} (no data)")
			if wall is not None:
				print_fn(f"{'(untimed)':<48} {'':>7} {wall:>11.3f} "
				         f"{wall:>11.3f} {100.0:>6.1f}")
				print_fn(f"{'TOTAL (wall)':<48} {'':>7} {wall:>11.3f} "
				         f"{'':>11} {100.0:>6.1f}")
			return
		print_fn(title)
		if total > 0.0:
			print_fn(f"Total recorded: {total:.3f} s")
		for line in rows:
			print_fn(line)
		if wall is not None:
			untimed = wall - total
			pct = (untimed / wall * 100.0) if wall > 0.0 else 0.0
			print_fn(f"{'(untimed)':<48} {'':>7} {untimed:>11.3f} "
			         f"{untimed:>11.3f} {pct:>6.1f}")
			print_fn(f"{'TOTAL (wall)':<48} {'':>7} {wall:>11.3f} "
			         f"{'':>11} {100.0:>6.1f}")

	def format(self, **kwargs) -> list[str]:
		_, rows = self._rows(kwargs.get("min_percent"), kwargs.get("max_depth"))
		return rows


_GLOBAL_COLLECTOR = TimingCollector()


def get_collector() -> TimingCollector:
	return _GLOBAL_COLLECTOR


def reset() -> None:
	_GLOBAL_COLLECTOR.reset()


def section(name: str, *, announce: bool = False,
            label: str | None = None) -> TimingSection:
	return _GLOBAL_COLLECTOR.section(name, announce=announce, label=label)


def timed(name: str | None = None, *, watch: bool = False) -> Callable:
	return _GLOBAL_COLLECTOR.timed(name, watch=watch)


def record(name: str, seconds: float, *, count: int = 1) -> None:
	_GLOBAL_COLLECTOR.record(name, seconds, count=count)


def process_elapsed_s() -> float | None:
	"""Seconds since THIS PROCESS started, or ``None`` if unavailable.

	Every LORRAX driver does real work before ``main()`` runs: the module
	body brings up the whole runtime stack (env, ``jax.distributed``,
	backend init, mesh + clique warm-up), and imports alone take 75.0 s on
	a cold Frontera node against 2.1 s warm (job 7881949).  A table whose
	clock starts at ``main()`` therefore reports a wall that can be a
	minute short of the job's, which is the difference between "this
	accounting is complete" and "there is a minute somewhere else".

	Read from ``/proc`` (uptime minus the process's own start tick) rather
	than from a module-level ``perf_counter``, because a module-level
	timestamp cannot precede the import that defines it — and the startup
	call sits ABOVE every import in these drivers by design.  Returns
	``None`` off Linux or if the layout ever changes; callers must treat
	that as "no pre-main row", never as zero.
	"""
	try:
		with open("/proc/uptime", encoding="ascii") as fh:
			uptime = float(fh.readline().split()[0])
		with open("/proc/self/stat", encoding="ascii") as fh:
			# Field 22 (1-based) is starttime, in clock ticks since boot.
			# Split after the ")" that closes comm so a process name
			# containing spaces or parentheses cannot shift the index.
			fields = fh.read().rsplit(") ", 1)[1].split()
		start_ticks = float(fields[19])
		return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
	except Exception:          # noqa: BLE001 — an observability helper must
		return None            # never take down the run it observes


def report(
	*,
	print_fn: Callable[[str], Any] = print,
	title: str = "--- Timing (seconds) ---",
	min_percent: float | None = None,
	max_depth: int | None = None,
	wall: float | None = None,
) -> None:
	_GLOBAL_COLLECTOR.report(print_fn=print_fn, title=title, min_percent=min_percent, max_depth=max_depth, wall=wall)
