from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import jax


_PROFILE_DIR_ENV = "ISDF_JAX_PROFILE_DIR"
_PROFILE_SECTIONS_ENV = "ISDF_JAX_PROFILE_SECTIONS"
_warned_trace_failure = False


def _section_selected(section: str) -> bool:
	"""True unless an allowlist is set and this section is not on it.

	``ISDF_JAX_PROFILE_SECTIONS`` is a comma-separated list of substrings;
	only sections containing one of them open a profiler session.  Unset
	(the default) traces every section, which is what every existing
	caller already gets.

	The knob exists because a profiler session is not side-effect-free at
	every call site.  On a multi-node mesh a session that is live across
	the phdf5 collective write inside ``zeta_fit`` segfaults rank 0
	(reproduced twice at a 3x3 mesh over three nodes, 2026-08-09; a 2x2
	single-node mesh is unaffected).  Tracing the sigma tau kernel above
	P=4 is therefore impossible without being able to say which sections
	to open, and this is the smallest thing that makes it obtainable.
	"""
	allow = os.environ.get(_PROFILE_SECTIONS_ENV, "").strip()
	if not allow:
		return True
	wanted = [w.strip() for w in allow.split(",") if w.strip()]
	return any(w in section for w in wanted)


def _trace_path(section: str) -> Path | None:
	base = os.environ.get(_PROFILE_DIR_ENV)
	if not base:
		return None
	if not _section_selected(section):
		return None
	base_path = Path(base).expanduser()
	timestamp = time.strftime("%Y%m%d-%H%M%S")
	suffix = f"{section}-{timestamp}-p{jax.process_index()}"
	return base_path / suffix


def _log_once(msg: str) -> None:
	global _warned_trace_failure
	if _warned_trace_failure:
		return
	if jax.process_index() == 0:
		print(f"[jax_profile] {msg}")
	_warned_trace_failure = True


@contextmanager
def trace_section(section: str) -> Iterator[None]:
	"""Start a JAX profiler trace when ISDF_JAX_PROFILE_DIR is set."""
	profiler = getattr(jax, "profiler", None)
	if profiler is None:
		yield
		return
	trace_path = _trace_path(section)
	if trace_path is None:
		yield
		return
	trace_path.mkdir(parents=True, exist_ok=True)
	try:
		ctx = profiler.trace(str(trace_path))
	except Exception as exc:  # pragma: no cover - extremely rare
		_log_once(f"Profiler trace disabled ({exc}). Continuing without trace output.")
		ctx = nullcontext()
	with ctx:
		yield


@contextmanager
def step_annotation(name: str, *, step_num: int | None = None, detail: str | None = None) -> Iterator[None]:
	"""Annotate host-side regions so they show up inside a profiler trace."""
	profiler = getattr(jax, "profiler", None)
	step_cls = getattr(profiler, "StepTraceAnnotation", None) if profiler else None
	label = name if detail is None else f"{name}[{detail}]"
	if step_cls is None:
		yield
		return
	kwargs = {}
	if step_num is not None:
		kwargs["step_num"] = int(step_num)
	with step_cls(label, **kwargs):
		yield


@contextmanager
def annotation(name: str) -> Iterator[None]:
	"""Light-weight annotation that does not bump the step counter."""
	profiler = getattr(jax, "profiler", None)
	trace_cls = getattr(profiler, "TraceAnnotation", None) if profiler else None
	if trace_cls is None:
		yield
		return
	with trace_cls(name):
		yield


# ``profile_section`` -- the pf-hook context manager -- was DELETED here
# 2026-08-09 (completeness audit P21).  It did not crash and it did not
# no-op with a warning: it printed a plausible ``[pf] <name> 1.484s`` line
# on every run and produced NO TRACE AT ALL, because ``pf.region()`` is
# only a ``TraceAnnotation`` and the session starter ``pf.trace_profile()``
# was never called; its two ``snapshot_memory`` calls failed silently as
# well.  To reach the helper it inserted
# ``/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling`` at
# ``sys.path[0]`` of every LORRAX process -- a path outside the repo that
# has not existed for months -- so any module there would have shadowed
# the repo copy of a colliding name.  Its one production call site was the
# Sigma pipeline, where ``timing.section("sigma.exec")`` already carries
# the wall line.
#
# The WORKING profiler entry points are above and are untouched:
# ``trace_section`` (driven by ISDF_JAX_PROFILE_DIR), ``annotation`` and
# ``step_annotation``.  Guarded by
# tests/test_no_sandbox_path_injection.py.
