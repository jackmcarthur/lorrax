"""Make this monorepo's ``services/*/src`` importable.  Transitional.

``services/<name>/src/<name>`` is a standalone, independently installable
package (charter, "Standalone mechanism").  Nothing in the launch chain
knows that yet: ``lx`` rewrites the container's ``PYTHONPATH`` to exactly
``<checkout>/src`` (``~/bin/lx``, ``retarget_pythonpath``) and the Shifter
image pip-installs nothing, so on the machine this code runs on there is
no install step in which a service could register itself.

So the tree says where its own services are, once, here.  Stdlib only,
idempotent, and it appends rather than prepends: an environment that HAS
been taught about ``services/`` (an editable install, a PYTHONPATH entry)
keeps winning, and this is the fallback that keeps a bare launch working.

THIS FILE IS PLUMBING WITH AN OWNER DECISION BEHIND IT.  The durable fix
is one of: ``pip install -e services/*`` in the image build, a
``services/*/src`` entry in the base modulefile's ``PYTHONPATH``, or a
``[tool.uv.workspace]`` member list — all three touch SHARED resources
(the modulefile is owner-scoped; ~9 worktrees share the image), so the
choice is not this task's to make.  When it is made, delete this module
and the three imports of it.
"""

from __future__ import annotations

import os
import sys

__all__ = ["ensure_on_path", "service_roots"]

#: ``<repo>/services`` — resolved from this file, not from cwd, so it is
#: right for a worktree, a bare checkout and an installed copy alike.
_SERVICES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services")


def service_roots() -> list[str]:
    """Every ``services/*/src`` directory that exists, sorted."""
    if not os.path.isdir(_SERVICES):
        return []
    out = []
    for name in sorted(os.listdir(_SERVICES)):
        src = os.path.join(_SERVICES, name, "src")
        if os.path.isdir(src):
            out.append(src)
    return out


def ensure_on_path() -> None:
    """Append the service source roots to ``sys.path``.  Idempotent."""
    have = {os.path.abspath(p) for p in sys.path}
    for src in service_roots():
        if src not in have:
            sys.path.append(src)
