"""LORRAX output on Python's standard :mod:`logging`.

Every module logs through ``log = get_logger(__name__)``, a logger under
``lorrax.``; that prefix keeps JAX, absl and h5py records out of the handlers
below.  The level says where a record appears:

=========  =================================================================
DEBUG      only when verbose: the rank log, and the main log under the
           current stage (indent 6, one row per line of the message)
INFO       the rank log; the main log only when the record carries a block
WARNING    printed where it happens in the main log and counted in its
           summary, plus the rank log
ERROR      the main log's ERROR block, plus the rank log
=========  =================================================================

A record carries structured fields as ``extra={"lorrax": {...}}``, built by
:func:`message` (a warning's id, found / needs / fix), :func:`block` (an INFO
line for the main log: a ``label value`` detail row under the current stage)
and :func:`record` (a receipt for the JSON sidecar, under a named key such as
``sigma.quadrature``).  No handler guesses from words.

**Verbose** is the one switch of ``runtime.debug_print_enabled``: the
command-line flag ``--verbose``, the deck key ``verbose = true``, or the
environment alias ``LORRAX_DEBUG_PRINT=1``.  The handlers read it on every
record, not at install, because a deck may be read after the session opened.
A record's ``record`` payload is never printed (it is in the sidecar); its
message is.

Two blocks are the main log's own cadence and print verbatim at column 0:
``progress`` (``LoopProgress``, logger ``lorrax.progress``) and ``stage``
(``common.timing``'s ``[i/N]`` banners, ``=>`` lines and ``.. still running``
heartbeats, logger ``lorrax.stage``).  A plain section's ``[stage hh:mm:ss]``
trace is DEBUG, so only the manual stages' banners reach the main log; a
heartbeat and a line naming an exception (``[EXC]``) stay INFO at any depth.

The three handlers are installed once per driver run by
``runtime.run_session`` (:func:`install_session`), and only there:

* **main log, rank 0** (:class:`MainLogHandler`): WARNING and above go to the
  report's ``log_message`` (the pass-2 ``Message`` blocks); INFO records
  tagged with a block go to its ``log_detail``; when verbose, DEBUG records
  go to its ``log_debug``;
* **rank stream, every rank** (:class:`RankStreamHandler`): ``%(message)s``
  on the stream the process held when the session opened, which is the
  stream the harness captures as the rank log; level INFO, or DEBUG when
  verbose.  On rank 0 it leaves out what the main log already printed there,
  since the main log is mirrored on that stream;
* **JSON sidecar, rank 0** (:class:`JsonSidecarHandler`): ``<report
  stem>.json``, one JSON document holding each ``record`` extra under its
  ``record_key`` (dots nest: ``memory.receipts`` is ``{"memory":
  {"receipts": ...}}``).  The file is created by the first such record,
  verbose or not, so a run without receipts writes none
  (:func:`written_sidecar` says whether this session wrote one).

Python ``warnings.warn`` becomes a ``py.warnings`` record
(``logging.captureWarnings``) whose text is ``<Category>: <message>``, which
the report maps to an id through its one id table.

Outside a session (a test, a tool, the GW driver before its report opens) a
fallback handler writes INFO records to the current ``sys.stdout``, as the
``print`` they replace did: INFO and DEBUG from process zero only (the
``rank0_print`` rule, read from the launcher variables), WARNING and above
from every process.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Callable, Mapping

__all__ = [
    "ROOT",
    "JsonSidecarHandler",
    "MainLogHandler",
    "RankStreamHandler",
    "block",
    "fields_of",
    "get_logger",
    "install_session",
    "message",
    "record",
    "session_sidecar",
    "written_sidecar",
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

    ``progress`` and ``stage`` print verbatim; any other block is a detail row
    under the current stage, ``label`` padded to 15 at indent 6 when given
    (``report_blocks.stage_detail``).
    """
    fields = {"block": str(name)}
    if label is not None:
        fields["label"] = str(label)
    return {"lorrax": fields}


def record(key: str | None = None, /, **fields: Any) -> dict:
    """``extra=`` for a receipt that tools parse, kept in the JSON sidecar.

    ``key`` (``"sigma.quadrature"``) names where the receipt goes; a key logged
    twice holds the list of its receipts.  Without a key it is appended to the
    sidecar's ``records`` list with its logger and message.
    """
    fields_ = {"record": dict(fields)}
    if key is not None:
        fields_["record_key"] = str(key)
    return {"lorrax": fields_}


def fields_of(log_record: logging.LogRecord) -> Mapping[str, Any]:
    """The ``lorrax`` extras of a record (empty when it has none)."""
    fields = getattr(log_record, "lorrax", None)
    return fields if isinstance(fields, Mapping) else {}


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
    renderer is given, so every rank's log shows the same text for it."""

    def __init__(self, render: Renderer | None = None) -> None:
        super().__init__("%(message)s")
        self.render = render

    def formatMessage(self, log_record: logging.LogRecord) -> str:  # noqa: N802
        if self.render is not None and log_record.levelno >= logging.WARNING:
            return "\n".join(self.render(log_record.message, fields_of(log_record),
                                         log_record.levelno))
        return super().formatMessage(log_record)


class RankStreamHandler(logging.StreamHandler, _QuietHandler):
    """Every rank: ``%(message)s`` lines on the stream its rank log is made of.

    INFO and above; DEBUG too while ``debug`` (the verbose switch, a bool or
    a callable read per record) is on.  ``exclude`` drops what another
    handler already prints on the same stream (rank 0's main log is mirrored
    there).
    """

    def __init__(self, stream, *, debug: "bool | Switch",
                 exclude: Callable[[logging.LogRecord], bool] | None = None,
                 render: Renderer | None = None) -> None:
        super().__init__(stream)
        self.setLevel(logging.DEBUG)
        self.setFormatter(_RankFormatter(render))
        self.verbose = _switch(debug)
        self.exclude = exclude

    def filter(self, log_record: logging.LogRecord):
        if log_record.levelno < logging.INFO and not self.verbose():
            return False
        if self.exclude is not None and self.exclude(log_record):
            return False
        return super().filter(log_record)


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


class JsonSidecarHandler(_QuietHandler):
    """Rank 0: the run's ``record`` extras as one JSON document in ``path``.

    A record with a ``record_key`` is stored under that key, dots nesting
    (``sigma.quadrature`` is ``doc["sigma"]["quadrature"]``); a key logged
    again holds the list of its receipts in order.  A record without a key is
    appended to ``doc["records"]`` with its time, logger and message.  The
    document opens with ``lorrax_sidecar`` (the format), ``report`` and
    ``started``.

    The first record creates the file (replacing a previous run's) and every
    record rewrites it whole through a temporary sibling, so the file on disk
    is always a complete document, also when the process dies.  A run without
    receipts writes no file.
    """

    def __init__(self, path: str, *, report: str | None = None) -> None:
        super().__init__(level=logging.DEBUG)
        self.path = str(path)
        self.report = report
        self._doc: dict[str, Any] | None = None
        self._repeated: set[str] = set()

    @property
    def written(self) -> bool:
        """Whether a record has created the file in this session."""
        return self._doc is not None

    def emit(self, log_record: logging.LogRecord) -> None:
        fields = fields_of(log_record)
        receipt = fields.get("record")
        if receipt is None:
            return
        if self._doc is None:
            self._doc = {"lorrax_sidecar": 1, "report": self.report,
                         "started": datetime.now().astimezone().isoformat(
                             timespec="seconds")}
        key = fields.get("record_key")
        if key:
            self._put(str(key), receipt)
        else:
            self._doc.setdefault("records", []).append({
                "time": datetime.fromtimestamp(log_record.created).astimezone()
                .isoformat(timespec="milliseconds"),
                "logger": log_record.name, "level": log_record.levelname,
                "message": log_record.getMessage(), "record": receipt})
        self._write()

    def _put(self, key: str, value: Any) -> None:
        *parents, leaf = key.split(".")
        node = self._doc
        for part in parents:
            child = node.get(part)
            if not isinstance(child, dict):
                child = node[part] = {} if child is None else {"value": child}
            node = child
        if key in self._repeated:
            node[leaf].append(value)
        elif leaf in node:
            node[leaf] = [node[leaf], value]
            self._repeated.add(key)
        else:
            node[leaf] = value

    def _write(self) -> None:
        text = json.dumps(self._doc, default=_json_value, ensure_ascii=False, indent=1)
        partial = self.path + ".partial"
        with open(partial, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
        os.replace(partial, self.path)


class _PrintHandler(_QuietHandler):
    """Outside a session: INFO lines on the current ``sys.stdout``, as ``print``."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)

    def emit(self, log_record: logging.LogRecord) -> None:
        if log_record.levelno < logging.WARNING:
            from runtime import _resolve_proc_id
            if _resolve_proc_id() != 0:     # as rank0_print: one rank's status
                return
        if log_record.levelno < logging.INFO:
            from runtime import debug_print_enabled
            if not debug_print_enabled():
                return
        stream = sys.stdout
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
#: The installed session's JSON sidecar handler, if it has one.
_SIDECAR: list[JsonSidecarHandler] = []


def session_sidecar() -> str | None:
    """The installed session's JSON sidecar path (written or not), or ``None``."""
    return _SIDECAR[0].path if _SIDECAR else None


def written_sidecar() -> str | None:
    """The installed session's JSON sidecar path, once a record has written it."""
    for handler in _SIDECAR:
        if handler.written:
            return handler.path
    return None


def install_session(*, rank: int, debug: "bool | Switch", stream,
                    on_message: MessageSink, on_detail: DetailSink | None = None,
                    on_debug: DebugSink | None = None,
                    render: Renderer | None = None,
                    sidecar: str | None = None,
                    report_name: str | None = None) -> Callable[[], None]:
    """Install one driver run's handlers; return the call that removes them.

    ``debug`` is the verbose switch, a bool or a callable the handlers read
    on every record (``runtime.debug_print_enabled``).  ``stream`` is the
    process's stdout when the session opened (the rank log).
    ``on_message`` / ``on_detail`` / ``on_debug`` are the report's message,
    stage-detail and verbose-DEBUG entries (rank 0 only); ``render`` gives
    the lines a message prints as, for the other ranks' logs; ``sidecar`` is
    the JSON sidecar's path (rank 0).  A session left installed by a run that
    did not close is removed first.
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
        render=None if main is not None else render))
    sidecar_handler = None
    if int(rank) == 0 and sidecar:
        sidecar_handler = JsonSidecarHandler(sidecar, report=report_name)
        handlers.append(sidecar_handler)
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
        if sidecar_handler in _SIDECAR:
            _SIDECAR.remove(sidecar_handler)
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
    if sidecar_handler is not None:
        _SIDECAR.append(sidecar_handler)
    return uninstall
