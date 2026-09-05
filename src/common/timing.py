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
# ``LORRAX_DEBUG_PRINT=1`` makes every section announce its entry and
# exit with a wall-clock timestamp on rank 0.  That is a milestone
# cadence for the WHOLE code, including stages that are one monolithic
# jit call and so cannot carry a ``LoopProgress``.  Print-only; the
# accumulated tree and the final report are byte-identical either way.
# Debug tracing is capped at nesting depth 3.  There is deliberately no
# second depth knob: one driver has one debug-print control.
#
# Sections that must ALWAYS announce (env-independent stage cadence —
# e.g. the screening phases, whose silence was paid for three times:
# AC.2's 30-minute silent pzheevd, AF.4c's 2 h 55 m silent restart
# write, the 2.5 h silent c2406 screening stage) pass
# ``announce=True`` (optionally with a human ``label``) to
# ``timing.section``.  A collector-owned daemon emits elapsed-wall heartbeats
# every 60 seconds until the section and its registered async watchers finish.
# It proves that the Python scope is alive, not that a GPU counter advanced.
# Same formatting path, same rank-0 gate, no depth cap — ONE cadence mechanism,
# not two.  Driver-owned announced sections begin after runtime/bootstrap; raw
# process forks after runtime startup are unsupported throughout LORRAX.
# ---------------------------------------------------------------------------
# The knob is read at USE time, not import time: common.timing is
# imported by essentially every LORRAX CLI.  Use-time reads also let tests
# and driver wrappers set the shared runtime switch after import.
_TRACE_RANK0: bool | None = None
_TRACE_DEPTH = 3
_ANNOUNCED_HEARTBEAT_SECONDS = 60.0


def _trace_flag() -> bool:
    from runtime import debug_print_enabled
    return debug_print_enabled()


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
    if not _trace_flag() or depth > _TRACE_DEPTH:
        return False
    return _rank0()


def _trace(msg: str) -> None:
    print(f"[stage {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _safe_trace(msg: str) -> None:
	"""Best-effort cadence output that can never replace numerical errors."""
	try:
		_trace(msg)
	except Exception:
		# Closed stdout/captures and diagnostic formatting bugs are not a
		# reason to abort a calculation or mask a body/watcher exception.
		pass



def _exc_text(error: BaseException | None) -> str:
	"""One line naming the exception that closed a stage, so the cause is
	on the raising rank's log even when the teardown that follows hangs."""
	if error is None:
		return ""
	first = (str(error).strip().splitlines() or [""])[0]
	return f"{type(error).__name__}: {first[:200]}" if first else type(error).__name__

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
		# driver debug (rank-0 only), through the SAME _trace
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
		return self.collector._enter(self)

	def __exit__(self, exc_type, exc, tb) -> None:
		# Keep cadence alive while a watcher synchronizes asynchronous JAX work.
		# Cleanup belongs in ``finally``: a failed ``block_until_ready`` is the
		# most important time not to leave a false heartbeat behind.
		watcher_failed = False
		try:
			if exc_type is None and self._watchers:
				for watcher in self._watchers:
					watcher()
		except BaseException:
			watcher_failed = True
			raise
		finally:
			end = time.perf_counter()
			self.collector._leave(
				self, end=end,
				failed=exc_type is not None or watcher_failed,
				error=exc)
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
		self._heartbeat_condition = threading.Condition(self._lock)
		self._heartbeat_sections: dict[TimingSection, float] = {}
		self._heartbeat_thread: threading.Thread | None = None
		self._active_sections: set[TimingSection] = set()
		self._local = threading.local()

	def _heartbeat_loop(self) -> None:
		"""Emit one cadence line per active thread, with atomic ownership.

		The collector lock covers section-stack mutation, deepest-owner choice,
		and the print itself.  Thus an exit cannot overtake a heartbeat already
		chosen, and a parent cannot print from a snapshot taken before a nested
		announced child entered.  The daemon performs no JAX or collective call.
		"""
		while True:
			with self._heartbeat_condition:
				if not self._heartbeat_sections:
					self._heartbeat_thread = None
					return
				now = time.perf_counter()
				delay = min(self._heartbeat_sections.values()) - now
				if delay > 0.0:
					self._heartbeat_condition.wait(timeout=delay)
					continue

				due = {section for section, deadline
				       in self._heartbeat_sections.items() if deadline <= now}
				for section in due:
					self._heartbeat_sections[section] = (
						now + _ANNOUNCED_HEARTBEAT_SECONDS)
				speakers: list[tuple[int, TimingSection]] = []
				for section in due:
					try:
						index = section.stack.index(section)
					except ValueError:
						continue
					if any(item in self._heartbeat_sections
					       for item in section.stack[index + 1:]):
						continue
					speakers.append((index, section))
				for index, section in sorted(
						speakers, key=lambda item: (id(item[1].stack), item[0])):
					_safe_trace(
						"  " * index
						+ f".. {section._display_name()} still running; "
						+ f"elapsed={now - section.start:.1f} s")

	def _ensure_heartbeat_thread_locked(self) -> None:
		thread = self._heartbeat_thread
		if thread is not None and thread.is_alive():
			return
		thread = threading.Thread(
			target=self._heartbeat_loop, name="lorrax-stage-heartbeat",
			daemon=True)
		self._heartbeat_thread = thread
		try:
			thread.start()
		except Exception:
			self._heartbeat_thread = None
			raise

	def _enter(self, section: TimingSection) -> TimingSection:
		with self._heartbeat_condition:
			section.stack.append(section)
			self._active_sections.add(section)
			try:
				if section._cadence_on():
					_safe_trace("  " * (len(section.stack) - 1)
					            + f"-> {section._display_name()}")
				section.start = time.perf_counter()
				section.child_elapsed = 0.0
				section._watchers = []
				if section.announce and _rank0():
					self._heartbeat_sections[section] = (
						section.start + _ANNOUNCED_HEARTBEAT_SECONDS)
					self._ensure_heartbeat_thread_locked()
					self._heartbeat_condition.notify_all()
			except BaseException:
				self._heartbeat_sections.pop(section, None)
				self._active_sections.discard(section)
				section.stack.pop()
				self._heartbeat_condition.notify_all()
				raise
		return section

	def _leave(self, section: TimingSection, *, end: float,
	           failed: bool, error: BaseException | None = None) -> None:
		inclusive = end - section.start
		exclusive = inclusive - section.child_elapsed
		with self._heartbeat_condition:
			if not section.stack or section.stack[-1] is not section:
				raise RuntimeError(
					"timing sections must exit in last-entered, first-exited order")
			cadence = section._cadence_on()
			depth = len(section.stack) - 1
			self._heartbeat_sections.pop(section, None)
			self._active_sections.discard(section)
			section.node.record(inclusive, max(0.0, exclusive))
			section.stack.pop()
			if section.stack:
				section.stack[-1].child_elapsed += inclusive
			self._heartbeat_condition.notify_all()
			# State is already clean if the ordinary foreground print sink fails.
			# Holding the same lock as the scheduler preserves exit ordering.
			if cadence:
				_safe_trace("  " * depth
				            + f"<- {section._display_name()}  {inclusive:.1f} s"
				            + (f"  [EXC] {_exc_text(error)}" if failed else ""))

	def reset(self) -> None:
		stack = getattr(self._local, "stack", None)
		with self._lock:
			if self._active_sections:
				raise RuntimeError(
					"cannot reset timing while a section is active")
			if stack:
				raise RuntimeError(
					"cannot reset timing while a section is active")
			self._root = TimingNode("root")
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
		with self._lock:
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

	def records(self) -> list[dict[str, Any]]:
		"""A stable, structured snapshot of the accumulated timing tree.

		Human-facing drivers should not parse :meth:`format`'s aligned text in
		order to build their own stage summary.  Each record keeps the node's
		local name, full path, depth, count and inclusive/exclusive wall time.
		The snapshot is read-only and preserves insertion order.
		"""
		out: list[dict[str, Any]] = []

		def visit(node: TimingNode, parents: tuple[str, ...]) -> None:
			path = parents + (node.name,)
			out.append({
				"name": node.name,
				"path": path,
				"depth": len(parents),
				"count": int(node.count),
				"inclusive": float(node.inclusive),
				"exclusive": float(node.exclusive),
			})
			for child in node.children.values():
				visit(child, path)

		with self._lock:
			for child in self._root.children.values():
				visit(child, ())
		return out


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


def records() -> list[dict[str, Any]]:
	"""Structured snapshot of the global timing collector."""
	return _GLOBAL_COLLECTOR.records()


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
