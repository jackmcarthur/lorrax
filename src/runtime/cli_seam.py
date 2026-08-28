"""Answer ``--help`` and refuse bad argv BEFORE the process starts a runtime.

THE PROBLEM.  Every LORRAX driver runs ``initialize_communicator_stack``
at MODULE scope, above its own ``import jax`` -- it has to, because the
env defaults only bind before jax reads them.  Its argument parser,
however, is built inside ``main()``, i.e. long after.  So ``--help``
paid the whole bring-up: env defaults, ``jax.distributed``, backend
init, the mesh and every MPI/NCCL clique in it, the FFI gate, the
compile cache.  On a box whose FFI library is not built, ``--help`` did
not print usage at all -- all eight drivers exited 1 at
``_enforce_required_ffi`` with a library-path traceback (measured
2026-08-27, WSL box, 0.83-2.90 s each).

THE SEAM.  A driver whose parser needs nothing but ``argparse`` can
define that parser above the startup call and hand it here.  This runs
``parse_args`` for its EXIT behaviour and throws the result away:
argparse prints usage and raises ``SystemExit(0)`` on ``-h``/``--help``,
prints usage and raises ``SystemExit(2)`` on an unknown flag or a
missing required one, and returns otherwise -- at which point the module
body carries on into the normal startup.  ``main()`` re-parses from the
SAME factory, so there is no second source of truth and nothing to keep
in sync; re-parsing argv is free next to a device mesh.

WHAT THIS IS NOT.  It is not a place to decide anything.  It holds no
policy about which flags exist, prints nothing itself, and has no notion
of "help" beyond the one argparse already has.  Anything richer belongs
in the parser.

WHERE IT DOES NOT APPLY.  A parser whose ``choices=`` or defaults come
from a module that imports jax cannot move above the startup call --
hoisting it would drag jax above the env defaults, which is the ordering
the startup call exists to protect.  ``tests/test_cli_help_seam.py``
carries that list as debt with a count that must ratchet DOWN.
"""
from __future__ import annotations

import sys

__all__ = ["refuse_bad_argv_before_startup"]


def refuse_bad_argv_before_startup(parser, argv=None) -> None:
    """Parse ``argv`` for its exit behaviour only; discard the result.

    Parameters
    ----------
    parser
        An ``argparse.ArgumentParser``.  Pass the driver's own factory
        output so this cannot disagree with what ``main()`` will parse.
    argv
        Argument list WITHOUT the program name; default ``sys.argv[1:]``.

    Returns ``None`` when ``argv`` is acceptable.  Otherwise argparse
    raises ``SystemExit`` and the process ends here, before any runtime.

    Call it under an ``if __name__ == "__main__":`` guard at module
    scope.  Importing a driver as a LIBRARY must never consult argv --
    ``bse.exciton_bands`` imports ``bandstructure.htransform``, and
    ``gw.sigma_dispatch`` imports ``gw.kin_ion_io``, with argv that
    belongs to a different program.
    """
    parser.parse_args(sys.argv[1:] if argv is None else argv)
