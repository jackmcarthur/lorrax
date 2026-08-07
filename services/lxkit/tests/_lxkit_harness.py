"""Stdlib-only scaffolding, so the pure cells run on a bare interpreter too.

``test_lxkit_gate.py`` and ``test_lxkit_probe.py`` import NOTHING but the
standard library and ``lxkit`` itself, and each ends with a ``__main__``
block calling :func:`run_module`.  They therefore run two ways:

    <venv>/python -m pytest services/lxkit/tests     # collected as usual
    python3 services/lxkit/tests/test_lxkit_gate.py  # no pytest, no jax

The second is not a curiosity: lxkit's whole point is that a service's
policy layer works on a machine with nothing installed, and a suite that
needed pytest to demonstrate it would be asserting the opposite of the
property under test.  Precedent in this tree: ``tests/test_layering.py``
carries the same bare-interpreter runner.
"""

from __future__ import annotations

import contextlib
import os
import sys
import traceback

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@contextlib.contextmanager
def raises(exc, match=""):
    """``pytest.raises`` for an interpreter that has no pytest."""
    try:
        yield
    except exc as e:                                      # noqa: BLE001
        if match and match not in str(e):
            raise AssertionError(
                f"{exc.__name__} raised, but {match!r} not in {str(e)!r}")
        return
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


class _Dev:
    def __init__(self, platform):
        self.platform = platform


class _Flat(list):
    pass


class _Devices:
    def __init__(self, platform):
        self.flat = _Flat([_Dev(platform)])


class FakeMesh:
    """The 1.5 attributes lxkit.gate reads off a mesh: ``devices.flat[0]
    .platform``.  A real ``jax.sharding.Mesh`` would need jax, which is the
    dependency these cells exist to prove is unnecessary."""

    def __init__(self, platform: str):
        self.devices = _Devices(platform)


@contextlib.contextmanager
def env(**kw):
    """Set/unset env vars for the duration; ``None`` means unset."""
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_module(namespace) -> int:
    """Run every ``test_*`` in ``namespace``; print a census; return an rc.

    Prints the counts because a bare interpreter has no reporter, and a
    silent zero-exit run is indistinguishable from a run that collected
    nothing (the 38-byte-junitxml lesson).
    """
    from lxkit.gate import reset_gate_state
    cells = sorted((n, f) for n, f in namespace.items()
                   if n.startswith("test_") and callable(f))
    if not cells:
        print("COLLECTED 0 CELLS — that is a failure, not a pass")
        return 1
    failed = []
    for name, fn in cells:
        reset_gate_state()          # what lxkit.testing.gate_state does
        try:
            fn()
        except Exception:                                 # noqa: BLE001
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(cells) - len(failed)} passed, {len(failed)} failed "
          f"({len(cells)} collected) in {namespace.get('__file__')}")
    return 1 if failed else 0
