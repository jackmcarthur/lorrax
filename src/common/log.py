"""LORRAX output on Python's standard :mod:`logging`.

Every module logs through ``log = get_logger(__name__)``, a logger under
``lorrax.``; that prefix keeps JAX, absl and h5py records out of the handlers
below.  The level says where a record appears:

=========  =================================================================
DEBUG      only when verbose: rank 0's stdout, and the main log under the
           current stage (indent 6, one row per line of the message)
INFO       rank 0's stdout; the main log only when the record carries a block
WARNING    printed where it happens in the main log and counted in its
           summary; on a rank above 0, that rank's stderr
ERROR      the main log's ERROR block; on a rank above 0, that rank's stderr
=========  =================================================================

A record carries structured fields as ``extra={"lorrax": {...}}``, built by
:func:`message` (a warning's id, found / needs / fix), :func:`block` (an INFO
line for the main log: a ``label value`` detail row under the current stage)
and :func:`record` (a receipt: a DEBUG record's fields, under a named key such
as ``sigma.quadrature``).  No handler guesses from words.  No file holds the
receipts: a run writes no JSON sidecar.  When verbose, a receipt's fields print
as one ``<key>: {...}`` line under its message.

**Verbose** is the one switch of ``runtime.debug_print_enabled``: the
command-line flag ``--verbose``, the deck key ``verbose = true``, or the
environment alias ``LORRAX_DEBUG_PRINT=1``.  The handlers read it on every
record, not at install, because a deck may be read after the session opened.

Two blocks are the main log's own cadence and print verbatim at column 0:
``progress`` (``LoopProgress``, logger ``lorrax.progress``) and ``stage``
(``common.timing``'s ``[i/N]`` banners, ``=>`` lines and ``.. still running``
heartbeats, logger ``lorrax.stage``).  A plain section's ``[stage hh:mm:ss]``
trace is DEBUG, so only the manual stages' banners reach the main log; a
heartbeat and a line naming an exception (``[EXC]``) stay INFO at any depth.
A ``memory`` block is held by the report and printed once, in its MEMORY
MODEL block.

The two handlers are installed once per driver run by
``runtime.run_session`` (:func:`install_session`), and only there:

* **main log, rank 0** (:class:`MainLogHandler`): WARNING and above go to the
  report's ``log_message`` (the pass-2 ``Message`` blocks); INFO records
  tagged with a block go to its ``log_detail``; when verbose, DEBUG records
  go to its ``log_debug``;
* **rank stream, every rank** (:class:`RankStreamHandler`).  On rank 0:
  ``%(message)s`` on the stream the process held when the session opened
  (the stream the harness captures as the rank-0 log), level INFO, or DEBUG
  when verbose, leaving out what the main log already printed there, since
  the main log is mirrored on that stream.  On a rank above 0: nothing by
  default; WARNING and above on that rank's stderr, rendered as the main log
  renders them; INFO and DEBUG on its stdout only when verbose.

Python ``warnings.warn`` becomes a ``py.warnings`` record
(``logging.captureWarnings``) whose text is ``<Category>: <message>``, which
the report maps to an id through its one id table.

Outside a session (a test, a tool, the GW driver before its report opens) a
fallback handler writes INFO records to the current ``sys.stdout``, as the
``print`` they replace did: INFO and DEBUG from process zero only (the
``rank0_print`` rule, read from the launcher variables), WARNING and above
from every process, a process above zero on its stderr.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from typing import Any, Callable, Mapping

__all__ = [
    "ROOT",
    "MainLogHandler",
    "RankStreamHandler",
    "block",
    "fields_of",
    "get_logger",
    "install_session",
    "message",
    "record",
    "record_text",
]

#: The logger every LORRAX logger lives under.
ROOT = "lorrax"
#: The logger :func:`logging.captureWarnings` sends Python warnings to.
WARNINGS_LOGGER = "py.warnings"

#: Python warnings the main log never shows: optimization hints, not results.
#: When verbose they are DEBUG records (rank log and main log).
WARNING_NOISE = (
    "Some donated buffers were not usable",
)


def get_logger(name: str) -> logging.Logger:
    """The logger ``lorrax.<name>``; a module passes its ``__name__``."""
    return logging.getLogger(f"{ROOT}.{name}")


# ── structured extras ────────────────────────────────────────────────────────
def message(id: str, *, found: tuple[str, str] | None = None,
            needs: str | None = None, fix: str | None = None,
            lines: tuple[str, ...] = (), kind: str | None = None,
            quiet: bool = False) -> dict:
    """``extra=`` for a WARNING or ERROR that is one pass-2 ``Message``.

    ``found`` is ``(what, value)``, ``needs`` what this run needs, ``fix`` the
    command or setting that resolves it, ``lines`` free lines.  ``kind``
    (``"NOTE"``) overrides the kind the level gives; ``quiet`` counts the
    message in the summary without printing it (another block prints it).
    """
    fields: dict[str, Any] = {"id": str(id)}
    if found is not None:
        fields["found"] = (str(found[0]), str(found[1]))
    if needs is not None:
        fields["needs"] = str(needs)
    if fix is not None:
        fields["fix"] = str(fix)
    if lines:
        fields["lines"] = tuple(str(line) for line in lines)
    if kind is not None:
        fields["kind"] = str(kind)
    if quiet:
        fields["quiet"] = True
    return {"lorrax": fields}


def block(name: str, *, label: str | None = None) -> dict:
    """``extra=`` for an INFO line the main log shows.

    ``progress`` and ``stage`` print verbatim; ``memory`` is a MEMORY MODEL
    row (printed once, after TIMING); any other block is a detail row under
    the current stage, ``label`` padded to 15 at indent 6 when given
    (``report_blocks.stage_detail``).
    """
    fields = {"block": str(name)}
    if label is not None:
        fields["label"] = str(label)
    return {"lorrax": fields}


def record(key: str | None = None, /, **fields: Any) -> dict:
    """``extra=`` for a receipt: a DEBUG record's fields, under ``key``
    (``"sigma.quadrature"``).  They print with the message when verbose
    (:func:`record_text`); no file holds them."""
    fields_ = {"record": dict(fields)}
    if key is not None:
        fields_["record_key"] = str(key)
    return {"lorrax": fields_}


def fields_of(log_record: logging.LogRecord) -> Mapping[str, Any]:
    """The ``lorrax`` extras of a record (empty when it has none)."""
    fields = getattr(log_record, "lorrax", None)
    return fields if isinstance(fields, Mapping) else {}


def _json_value(value: Any) -> Any:
    """A receipt value JSON can hold: NumPy scalars and arrays, paths, else text."""
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", None) == 0:
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


def record_text(fields: Mapping[str, Any]) -> str | None:
    """A receipt's fields as one ``<key>: {...}`` line (compact JSON), or
    ``None`` when the record carries none."""
    receipt = fields.get("record")
    if receipt is None:
        return None
    try:
        body = json.dumps(receipt, default=_json_value, ensure_ascii=False,
                          separators=(", ", ": "))
    except Exception:          # noqa: BLE001 - output never fails a run
        body = str(receipt)
    return f"{fields.get('record_key') or 'record'}: {body}"


# ── handlers ─────────────────────────────────────────────────────────────────
class _QuietHandler(logging.Handler):
    """A handler whose failure prints nothing.

    ``logging``'s default reports a failing handler with a traceback on
    stderr, and ``Traceback (most recent call last)`` in a rank log is a
    failure signature of the test harness.  Output must never fail a run.
    """

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        pass


#: ``on_message(text, fields, levelno)``: print and count one message.
MessageSink = Callable[[str, Mapping[str, Any], int], None]
#: ``on_detail(text, fields)``: one block-tagged line (``fields["block"]``).
DetailSink = Callable[[str, Mapping[str, Any]], None]
#: ``render(text, fields, levelno)``: the lines the main log prints for a message.
Renderer = Callable[[str, Mapping[str, Any], int], list]
#: ``on_debug(text, fields)``: one DEBUG record, when verbose.
DebugSink = Callable[[str, Mapping[str, Any]], None]
#: The verbose switch, read per record (``runtime.debug_print_enabled``).
Switch = Callable[[], bool]


def _switch(value: "bool | Switch | None") -> Switch:
    """A constant or a callable switch, as a callable."""
    if callable(value):
        return value
    on = bool(value)
    return lambda: on


class MainLogHandler(_QuietHandler):
    """Rank 0: messages and block-tagged INFO lines into the driver's main log;
    DEBUG records too when ``verbose()`` (to ``on_debug``, under the stage)."""

    def __init__(self, on_message: MessageSink,
                 on_detail: DetailSink | None = None, *,
                 on_debug: DebugSink | None = None,
                 verbose: "bool | Switch" = False) -> None:
        super().__init__(level=logging.DEBUG)
        self.on_message = on_message
        self.on_detail = on_detail
        self.on_debug = on_debug
        self.verbose = _switch(verbose)

    def accepts(self, log_record: logging.LogRecord) -> bool:
        """Whether this handler prints ``log_record`` (pure: no side effect)."""
        if log_record.levelno >= logging.WARNING:
            return True
        if log_record.levelno >= logging.INFO:
            return (self.on_detail is not None
                    and bool(fields_of(log_record).get("block")))
        return self.on_debug is not None and bool(self.verbose())

    def emit(self, log_record: logging.LogRecord) -> None:
        if not self.accepts(log_record):
            return
        text = log_record.getMessage()
        if log_record.levelno >= logging.WARNING:
            self.on_message(text, fields_of(log_record), log_record.levelno)
        elif log_record.levelno >= logging.INFO:
            self.on_detail(text, fields_of(log_record))
        else:
            self.on_debug(text, fields_of(log_record))


class _RankFormatter(logging.Formatter):
    """``%(message)s``; a message rendered as the main log renders it, when a
    renderer is given, so every rank shows the same text for it; a DEBUG
    receipt's fields on a line of their own."""

    def __init__(self, render: Renderer | None = None) -> None:
        super().__init__("%(message)s")
        self.render = render

    def formatMessage(self, log_record: logging.LogRecord) -> str:  # noqa: N802
        if self.render is not None and log_record.levelno >= logging.WARNING:
            return "\n".join(self.render(log_record.message, fields_of(log_record),
                                         log_record.levelno))
        text = super().formatMessage(log_record)
        if log_record.levelno < logging.INFO:
            extra = record_text(fields_of(log_record))
            if extra:
                text = f"{text}\n{extra}"
        return text


class RankStreamHandler(logging.StreamHandler, _QuietHandler):
    """Every rank's own lines (module docstring).

    Rank 0 (``rank == 0``): INFO and above on ``stream``, DEBUG too while the
    verbose switch is on; ``exclude`` drops what the main log already prints
    there.  A rank above 0: WARNING and above on the current ``sys.stderr``;
    INFO and DEBUG on ``stream`` only while verbose.
    """

    def __init__(self, stream, *, debug: "bool | Switch",
                 exclude: Callable[[logging.LogRecord], bool] | None = None,
                 render: Renderer | None = None, rank: int = 0) -> None:
        super().__init__(stream)
        self.setLevel(logging.DEBUG)
        self.setFormatter(_RankFormatter(render))
        self.verbose = _switch(debug)
        self.exclude = exclude
        self.rank = int(rank)
        self._stdout = stream

    def filter(self, log_record: logging.LogRecord):
        if self.rank != 0 and log_record.levelno < logging.WARNING:
            if not self.verbose():
                return False
        elif log_record.levelno < logging.INFO and not self.verbose():
            return False
        if self.exclude is not None and self.exclude(log_record):
            return False
        return super().filter(log_record)

    def emit(self, log_record: logging.LogRecord) -> None:
        if self.rank != 0:
            # A warning or error of a rank above 0 goes to its stderr (the
            # stream of the moment: a harness may have replaced it).
            self.stream = (sys.stderr if log_record.levelno >= logging.WARNING
                           else self._stdout)
        super().emit(log_record)


class _PrintHandler(_QuietHandler):
    """Outside a session: INFO lines on the current ``sys.stdout``, as ``print``."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)

    def emit(self, log_record: logging.LogRecord) -> None:
        from runtime import _resolve_proc_id
        rank = _resolve_proc_id()
        if log_record.levelno < logging.WARNING and rank != 0:
            return                      # as rank0_print: one rank's status
        if log_record.levelno < logging.INFO:
            from runtime import debug_print_enabled
            if not debug_print_enabled():
                return
        stream = sys.stdout if rank == 0 else sys.stderr
        stream.write(self.format(log_record) + "\n")
        stream.flush()


class _WarningNoise(logging.Filter):
    """Drop :data:`WARNING_NOISE`, or keep it at DEBUG when verbose."""

    def __init__(self, debug: "bool | Switch") -> None:
        super().__init__()
        self.verbose = _switch(debug)

    def filter(self, log_record: logging.LogRecord) -> bool:
        text = log_record.getMessage()
        if not any(fragment in text for fragment in WARNING_NOISE):
            return True
        if not self.verbose():
            return False
        log_record.levelno, log_record.levelname = logging.DEBUG, "DEBUG"
        return True


def _warning_text(message, category, filename, lineno, line=None) -> str:
    """``<Category>: <message>``: the text a captured warning's record carries."""
    return f"{category.__name__}: {message}"


_root = logging.getLogger(ROOT)
_root.setLevel(logging.DEBUG)       # the handlers choose; the logger passes all
# One fallback per process, even when this module is executed twice (a test
# that loads a package under stubs imports a second copy of it).
for _h in list(_root.handlers):
    if type(_h).__name__ == "_PrintHandler":
        _root.removeHandler(_h)
_FALLBACK = _PrintHandler()
_root.addHandler(_FALLBACK)

#: The uninstall of the installed session, if one is installed.
_SESSION: list[Callable[[], None]] = []


def install_session(*, rank: int, debug: "bool | Switch", stream,
                    on_message: MessageSink, on_detail: DetailSink | None = None,
                    on_debug: DebugSink | None = None,
                    render: Renderer | None = None) -> Callable[[], None]:
    """Install one driver run's handlers; return the call that removes them.

    ``debug`` is the verbose switch, a bool or a callable the handlers read
    on every record (``runtime.debug_print_enabled``).  ``stream`` is the
    process's stdout when the session opened (the rank log).
    ``on_message`` / ``on_detail`` / ``on_debug`` are the report's message,
    stage-detail and verbose-DEBUG entries (rank 0 only); ``render`` gives
    the lines a message prints as, on the other ranks' stderr.  A session
    left installed by a run that did not close is removed first.
    """
    for uninstall in list(_SESSION):
        uninstall()
    loggers = (logging.getLogger(ROOT), logging.getLogger(WARNINGS_LOGGER))
    handlers: list[logging.Handler] = []
    main = None
    if int(rank) == 0:
        main = MainLogHandler(on_message, on_detail, on_debug=on_debug, verbose=debug)
        handlers.append(main)
    handlers.append(RankStreamHandler(
        stream, debug=debug, exclude=main.accepts if main is not None else None,
        render=None if main is not None else render, rank=int(rank)))
    noise = _WarningNoise(debug)
    saved = [(logger, logger.propagate, logger.level) for logger in loggers]
    loggers[0].removeHandler(_FALLBACK)
    for logger in loggers:
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        for handler in handlers:
            logger.addHandler(handler)
    loggers[1].addFilter(noise)
    saved_format = warnings.formatwarning
    warnings.formatwarning = _warning_text
    logging.captureWarnings(True)

    def uninstall() -> None:
        if uninstall not in _SESSION:
            return
        _SESSION.remove(uninstall)
        logging.captureWarnings(False)
        if warnings.formatwarning is _warning_text:
            warnings.formatwarning = saved_format
        loggers[1].removeFilter(noise)
        for logger, propagate, level in saved:
            for handler in handlers:
                logger.removeHandler(handler)
            logger.propagate = propagate
            logger.setLevel(level)
        for handler in handlers:
            try:
                handler.flush()
                handler.close()     # a StreamHandler leaves the launcher's stream open
            except Exception:       # noqa: BLE001 - closing output never fails a run
                pass
        if _FALLBACK not in loggers[0].handlers:
            loggers[0].addHandler(_FALLBACK)

    _SESSION.append(uninstall)
    return uninstall
