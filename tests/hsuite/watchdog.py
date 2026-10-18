"""A bounded first collective for the driver chain: a node fault becomes an error, not a hang.

The chain's first stage brings the runtime up (``jax.distributed.initialize``
and the device warm-up collective).  A rank on a faulty node or GPU (e.g. bad
MI250X GCDs) never finishes that, and its peers then wait in the collective
with nothing in any log.  :class:`FirstCollectiveWatchdog` is a host thread
that waits on an event for at most ``timeout`` seconds; if start-up has not
completed by then it writes one line naming this rank, its node, its visible
GPU, the stage and the jax.distributed coordinator to every sink, and ends
the process with status 1, so every rank of the step ends and the job's log
says which ranks were waiting where.  No device call, no synchronisation on
any hot path: the only cost is one sleeping thread during start-up.

Standard library only, so the harness can be tested without jax.
"""
from __future__ import annotations

import os
import socket
import threading

#: Bound on the first stage's start-up (seconds).  Start-up is 15-45 s on one
#: Frontier node and up to ~75 s on a cold node; the bound only has to separate
#: "slow" from "never".  Test-only override, never read by src.
DEFAULT_TIMEOUT_S = 600.0
TIMEOUT_ENV = "HSUITE_FIRST_COLLECTIVE_TIMEOUT_S"

_GPU_ENVS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def timeout_s(environ=None) -> float:
    raw = (environ if environ is not None else os.environ).get(TIMEOUT_ENV, "").strip()
    return float(raw) if raw else DEFAULT_TIMEOUT_S


def placement(environ=None) -> dict:
    """This rank's node and visible GPU, for the rank map and the watchdog line."""
    env = environ if environ is not None else os.environ
    gpu = next((f"{k}={env[k]}" for k in _GPU_ENVS if env.get(k, "").strip()), "no GPU variable")
    return {"host": socket.gethostname(), "gpu": gpu}


def rank_map_line(count, coordinator, placements) -> str:
    """Rank 0's start-up line: the coordinator and where every rank runs."""
    ranks = "; ".join(f"{r}: {p['host']} {p['gpu']}" for r, p in enumerate(placements))
    return f"hsuite: P={count}, jax.distributed coordinator {coordinator}; ranks {ranks}"


def terminal_writer(config):
    """A writer that reaches the terminal past pytest's capture (``config``: pytest's).

    The watchdog fires from its own thread while the main thread is blocked;
    suspending the global capture restores the launcher's fds 1/2 (it does not
    resume: the process ends next).  Returns None outside pytest.
    """
    capman = config.pluginmanager.getplugin("capturemanager") if config is not None else None
    if capman is None:
        return None

    def write(text: str) -> None:
        try:
            capman.suspend_global_capture(in_=True)
        except Exception:                                   # noqa: BLE001
            pass
        os.write(2, text.encode(errors="replace"))
    return write


class FirstCollectiveWatchdog:
    """Fire once if :meth:`done` is not called within ``timeout`` seconds."""

    def __init__(self, *, stage, rank, count, coordinator, timeout, fds=(), writers=(),
                 exit=os._exit, environ=None):
        self.stage, self.rank, self.count = stage, rank, count
        self.coordinator, self.timeout = coordinator, float(timeout)
        self.fds, self.writers, self._exit = tuple(fds), tuple(w for w in writers if w), exit
        self._where = placement(environ)
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hsuite-first-collective-watchdog",
                                        daemon=True)
        self.fired = False

    def message(self) -> str:
        return (f"hsuite WATCHDOG: rank {self.rank}/{self.count} on node {self._where['host']} "
                f"({self._where['gpu']}) did not finish the first collective (stage "
                f"{self.stage} start-up: jax.distributed.initialize with coordinator "
                f"{self.coordinator}, then the device warm-up) within {self.timeout:g} s. "
                f"A rank that never gets there usually means a faulty node or GPU; the ranks "
                f"that print this line were waiting.  Ending this rank with status 1.\n")

    def start(self) -> "FirstCollectiveWatchdog":
        self._thread.start()
        return self

    def done(self) -> None:
        self._event.set()

    def _run(self) -> None:
        if self._event.wait(self.timeout):
            return
        self.fired = True
        text = self.message()
        for fd in self.fds:
            try:
                os.write(fd, text.encode(errors="replace"))
            except OSError:
                pass
        for write in self.writers:
            try:
                write(text)
            except Exception:                               # noqa: BLE001
                pass
        self._exit(1)
