from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import jax


_PROFILE_DIR_ENV = "ISDF_JAX_PROFILE_DIR"
_warned_trace_failure = False


def _trace_path(section: str) -> Path | None:
	base = os.environ.get(_PROFILE_DIR_ENV)
	if not base:
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
