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
_TRACE = os.environ.get("LORRAX_TIMING_TRACE", "0").strip().lower() in (
    "1", "true", "yes", "on")
_TRACE_DEPTH = int(os.environ.get("LORRAX_TIMING_TRACE_DEPTH", "3"))
_TRACE_RANK0: bool | None = None


def _rank0() -> bool:
    global _TRACE_RANK0
    if _TRACE_RANK0 is None:
        try:
            import jax
            _TRACE_RANK0 = jax.process_index() == 0
        except Exception:                                    # noqa: BLE001
            _TRACE_RANK0 = True
    return _TRACE_RANK0


def _trace_enabled(depth: int) -> bool:
    if not _TRACE or depth > _TRACE_DEPTH:
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

	def report(
		self,
		*,
		print_fn: Callable[[str], Any] = print,
		title: str = "--- Timing (seconds) ---",
		min_percent: float | None = None,
		max_depth: int | None = None,
	) -> None:
		total, rows = self._rows(min_percent, max_depth)
		if not rows:
			print_fn(f"{title} (no data)")
			return
		print_fn(title)
		if total > 0.0:
			print_fn(f"Total recorded: {total:.3f} s")
		for line in rows:
			print_fn(line)

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


def report(
	*,
	print_fn: Callable[[str], Any] = print,
	title: str = "--- Timing (seconds) ---",
	min_percent: float | None = None,
	max_depth: int | None = None,
) -> None:
	_GLOBAL_COLLECTOR.report(print_fn=print_fn, title=title, min_percent=min_percent, max_depth=max_depth)
