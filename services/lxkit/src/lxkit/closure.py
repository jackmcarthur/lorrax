"""``python -m lxkit.closure verify|identity <leg.so> [platform]``.

Checks a leg's recorded dependency closure (``<leg>.closure.json``, or its
sealed bundle's manifest) against the leg's bytes and the recorded files,
without loading anything, and prints the identity the record promises.  Exit 0
when it verifies, 3 with the ``GATE native-closure`` refusal when it does not.
Build-reuse tooling (a site's FFI source-key resolver) requires exit 0 before
it reuses a build.  See :mod:`lxkit.native_provider`.
"""
from lxkit.native_provider import _main

if __name__ == "__main__":
    raise SystemExit(_main())
