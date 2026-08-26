"""One production stdout stream for a driver invocation.

LORRAX still contains low-level components that call :func:`print` directly
instead of accepting the driver's ``print_fn``.  Refactoring every physics
service merely to control presentation would create dozens of new logging
seams.  This small runtime boundary gives a production driver one authoritative
stdout instead: ordinary component chatter is discarded, while the driver's
scientific reporter writes through :meth:`emit` to the original stream.

Exceptions, tracebacks, rank-local refusals, and C/C++ diagnostics use stderr
and are deliberately untouched.  The distributed fail-fast hook also retains
this stream's original launcher stdout as a redundant failure sink: stderr
written immediately before ``os._exit`` is known to disappear under some
``srun`` captures, while production's ordinary ``sys.stdout`` is ``/dev/null``.
``LORRAX_DEBUG_PRINT=1`` leaves stdout entirely unchanged, exposing the
historical forensic stream.
"""

from __future__ import annotations

import os
import sys
import warnings


_PRODUCTION_ONLY_WARNING_NOISE = (
    # JAX buffer donation is an optimization hint, not a numerical or physical
    # warning.  It remains visible in the driver's forensic debug stream.
    "Some donated buffers were not usable",
)


_failure_stdout = None


def failure_output_streams():
    """Return distinct streams on which a fatal diagnostic should survive.

    Production mode replaces :data:`sys.stdout` with ``/dev/null``.  The
    fail-fast hook historically wrote to both ``sys.stdout`` and stderr, but
    that made its promised redundant stdout copy a no-op exactly when the GW
    production report was active.  The active :class:`ProductionStdout`
    registers the launcher's original stream here, at the runtime boundary
    that owns the redirection.  Non-production callers retain the historical
    ``(sys.stdout, sys.stderr)`` behavior.
    """
    primary = _failure_stdout if _failure_stdout is not None else sys.stdout
    streams = []
    for stream in (primary, sys.stderr):
        if all(stream is not seen for seen in streams):
            streams.append(stream)
    return tuple(streams)


class ProductionStdout:
    """Route incidental stdout away from one driver's scientific output."""

    def __init__(self, *, debug: bool, rank: int, warning_fn=None) -> None:
        self.debug = bool(debug)
        self.rank = int(rank)
        self.warning_fn = warning_fn
        self._console = sys.stdout
        self._sink = None
        self._showwarning = None
        self._warning_handler = None
        self._previous_failure_stdout = None

    def install(self) -> None:
        """Start routing incidental Python stdout in production mode."""
        global _failure_stdout
        if self.debug or self._sink is not None:
            return
        self._sink = open(os.devnull, "w", encoding="utf-8")
        self._previous_failure_stdout = _failure_stdout
        _failure_stdout = self._console
        sys.stdout = self._sink
        # Python warnings otherwise bypass stdout and repeat once per process.
        # A driver reporter can retain them once in its final warning block;
        # workers without a reporter stay silent.  Exceptions and native
        # diagnostics still use stderr and are never intercepted here.
        if self.warning_fn is not None or self.rank != 0:
            self._showwarning = warnings.showwarning

            def _route_warning(message, category, filename, lineno,
                               file=None, line=None):
                if any(fragment in str(message)
                       for fragment in _PRODUCTION_ONLY_WARNING_NOISE):
                    return
                if self.rank == 0 and self.warning_fn is not None:
                    self.warning_fn(f"{category.__name__}: {message}")

            self._warning_handler = _route_warning
            warnings.showwarning = self._warning_handler

    def emit(self, text: str = "", *, end: str = "\n") -> None:
        """Write scientific output once, from process zero."""
        if self.rank == 0:
            print(str(text), end=end, file=self._console, flush=True)

    def close(self) -> None:
        """Restore stdout without closing the launcher's original stream."""
        global _failure_stdout
        if self._sink is None:
            return
        if sys.stdout is self._sink:
            sys.stdout = self._console
        self._sink.close()
        self._sink = None
        if (self._warning_handler is not None
                and warnings.showwarning is self._warning_handler):
            warnings.showwarning = self._showwarning
        self._showwarning = None
        self._warning_handler = None
        if _failure_stdout is self._console:
            _failure_stdout = self._previous_failure_stdout
        self._previous_failure_stdout = None


__all__ = ["ProductionStdout", "failure_output_streams"]
