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


# ---------------------------------------------------------------------------
# pf-hook profiling context manager (moved from gw/gw_driver_helpers.py
# 2026-07-09 — generic profiling infra, not GW-specific).
# ---------------------------------------------------------------------------

class profile_section:
    """Context manager wrapping the optional ``pf`` profiling hooks.

    The sandbox keeps a small ``pf`` helper at
    ``scripts/profiling/pf.py`` for memory-snapshot + xprof-region
    annotation.  When importable, this CM:

    1. Adds ``scripts/profiling`` to ``sys.path`` and imports ``pf``.
    2. Takes a memory snapshot at ``{artifacts_dir}/memprof/{name}_pre.prof``.
    3. Enters ``pf.region(name)``.
    4. On exit takes ``{name}_post.prof``.

    Failures in any of (1)–(4) degrade gracefully — the caller's body
    still runs; the only visible effect is one ``print_fn`` line.
    Useful so call sites read as one ``with`` statement instead of
    six lines of ``try/except`` noise.

    Example::

        with profile_section("sigma_ppm", artifacts_dir, print_fn=print0):
            sigma_omega = compute_sigma_c_ppm_omega_grid(...)
    """

    __slots__ = ("_name", "_artifacts_dir", "_print_fn", "_pf", "_region_cm")

    def __init__(self, name: str, artifacts_dir: str | None = None,
                 *, print_fn=print):
        self._name = name
        self._artifacts_dir = (
            artifacts_dir if artifacts_dir is not None
            else os.environ.get("PF_ARTIFACTS_DIR", "profile")
        )
        self._print_fn = print_fn
        self._pf = None
        self._region_cm = None

    def _try_import_pf(self):
        try:
            import sys
            sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
            import pf  # type: ignore
            return pf
        except Exception as exc:
            self._print_fn(f"  [pf] profiling hooks unavailable: {exc}")
            return None

    def _snapshot(self, suffix: str):
        if self._pf is None:
            return
        try:
            path = f"{self._artifacts_dir}/memprof/{self._name}_{suffix}.prof"
            self._pf.snapshot_memory(path, label=f"{self._name}_{suffix}")
        except Exception:
            pass

    def __enter__(self):
        self._pf = self._try_import_pf()
        self._snapshot("pre")
        if self._pf is not None:
            try:
                self._region_cm = self._pf.region(self._name)
                self._region_cm.__enter__()
            except Exception:
                self._region_cm = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._region_cm is not None:
            try:
                self._region_cm.__exit__(exc_type, exc, tb)
            except Exception:
                pass
            self._region_cm = None
        self._snapshot("post")
        return False
