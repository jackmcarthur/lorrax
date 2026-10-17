"""The ordered process end (``runtime.ordered_exit``, ``runtime.install_ordered_exit``).

Each behavioural case runs a small script as ``__main__`` in a child
interpreter (the hook is process-lifetime and fires at interpreter shutdown,
so it cannot run inside the pytest process).  The finalizer is a stub that
prints ``FINALIZE <rc>`` and ends with ``os._exit``, standing in for
``runtime.finalize_process``.  No jax, no GPU, no MPI.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
_LAUNCHER_VARS = ("JAX_PROCESS_COUNT", "JAX_NUM_PROCESSES", "OMPI_COMM_WORLD_SIZE",
                  "MV2_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS")

pytestmark = pytest.mark.skipif(not hasattr(sys, "monitoring"),
                                reason="runtime.ordered_exit needs sys.monitoring (Python >= 3.12)")

STUB = """
import os, sys
def stub_finalize(rc=0):
    print(f"FINALIZE {rc}", flush=True)
    os._exit(rc)
"""


def run(tmp_path, body, *, env=None):
    script = tmp_path / "entry.py"
    script.write_text(STUB + textwrap.dedent(body))
    child_env = {k: v for k, v in os.environ.items() if k not in _LAUNCHER_VARS}
    child_env["PYTHONPATH"] = str(SRC) + os.pathsep + child_env.get("PYTHONPATH", "")
    child_env.update(env or {})
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       env=child_env, timeout=120)
    return p.returncode, p.stdout, p.stderr


INSTALL = """
from runtime import ordered_exit
state = ordered_exit.install(stub_finalize)
assert state["installed"], state
"""


@pytest.mark.parametrize("ending, rc, finalized", [
    ("pass", 0, True),                                     # fell off the end
    ("sys.exit(0)", 0, True),
    ("sys.exit()", 0, True),
    ("raise SystemExit(None)", 0, True),
    ("sys.exit(3)", 3, False),                             # nonzero: no collective teardown
    ("def main():\n    raise SystemExit(4)\nmain()", 4, False),
    ("try:\n    sys.exit(5)\nexcept SystemExit:\n    pass", 0, True),   # caught: the script went on
    ("raise ValueError('boom')", 1, False),
])
def test_exit_status_decides_finalize_or_immediate_exit(tmp_path, ending, rc, finalized):
    code, out, err = run(tmp_path, INSTALL + ending + "\n")
    assert code == rc, (out, err)
    assert out.count("FINALIZE") == (1 if finalized else 0), (out, err)
    if finalized:
        assert "FINALIZE 0" in out


def test_a_message_exit_is_status_one_printed_once(tmp_path):
    code, out, err = run(tmp_path, INSTALL + "raise SystemExit('bad input deck')\n")
    assert code == 1 and "FINALIZE" not in out
    assert err.count("bad input deck") == 1


def test_a_live_daemon_thread_does_not_reach_interpreter_teardown(tmp_path):
    body = INSTALL + textwrap.dedent("""
        import threading, time
        threading.Thread(target=lambda: [time.sleep(0.001) for _ in iter(int, 1)], daemon=True).start()
        time.sleep(0.05)
        sys.exit(0)
    """)
    code, out, err = run(tmp_path, body)
    assert code == 0 and out.count("FINALIZE 0") == 1, (out, err)


def test_install_is_idempotent_and_takes_one_tool_id(tmp_path):
    body = INSTALL + textwrap.dedent("""
        again = ordered_exit.install(stub_finalize)
        assert again == state and sys.monitoring.get_tool(state["tool"]) == ordered_exit.TOOL_NAME
    """)
    code, out, err = run(tmp_path, body)
    assert code == 0 and out.count("FINALIZE 0") == 1, (out, err)


def test_atexit_duties_still_run_once_under_the_finalizer(tmp_path):
    # The real finalize_process runs atexit._run_exitfuncs before os._exit; the
    # hook fires before CPython's own atexit pass, so each duty runs once.
    body = textwrap.dedent("""
        import atexit
        from runtime import ordered_exit
        atexit.register(lambda: print("DUTY", flush=True))
        def finalize(rc=0):
            atexit._run_exitfuncs()
            print(f"FINALIZE {rc}", flush=True)
            os._exit(rc)
        assert ordered_exit.install(finalize)["installed"]
    """)
    code, out, err = run(tmp_path, body)
    assert code == 0 and out.count("DUTY") == 1 and out.count("FINALIZE 0") == 1, (out, err)


def test_runtime_arms_it_only_at_p_greater_than_one(tmp_path):
    body = textwrap.dedent("""
        import runtime
        from runtime import ordered_exit
        runtime.finalize_process = stub_finalize
        runtime.install_ordered_exit()
        print("ARMED", ordered_exit.status()["installed"], flush=True)
        sys.exit(0)
    """)
    code, out, err = run(tmp_path, body, env={"SLURM_NTASKS": "2"})
    assert code == 0 and "ARMED True" in out and out.count("FINALIZE 0") == 1, (out, err)
    code, out, err = run(tmp_path, body, env={"SLURM_NTASKS": "1"})
    assert code == 0 and "ARMED False" in out and "FINALIZE" not in out, (out, err)


def test_explicit_run_main_and_finalize_ends_first_and_finalizes_once(tmp_path):
    body = textwrap.dedent("""
        import runtime
        runtime.finalize_process = stub_finalize
        runtime.install_ordered_exit()
        runtime.run_main_and_finalize(lambda: 0)
    """)
    code, out, err = run(tmp_path, body, env={"SLURM_NTASKS": "2"})
    assert code == 0 and out.count("FINALIZE 0") == 1, (out, err)


def test_finalize_process_is_reentrant_safe(monkeypatch):
    import runtime
    monkeypatch.setattr(runtime, "_FINALIZE_STARTED", True)
    assert runtime.finalize_process(0) is None             # the first call owns the end


@pytest.mark.parametrize("exc, want", [
    (None, (0, None)), (SystemExit(), (0, None)), (SystemExit(0), (0, None)),
    (SystemExit(7), (7, None)), (SystemExit(True), (1, None)),
    (SystemExit("msg"), (1, "msg")), (KeyboardInterrupt(), (1, None)), (ValueError(), (1, None)),
])
def test_exit_status_mirrors_cpython(exc, want):
    from runtime.ordered_exit import exit_status
    assert exit_status(exc) == want
