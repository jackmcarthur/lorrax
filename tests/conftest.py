"""Seal the first-party source closure before any test imports a service.

A test body that imports ``gw.*`` first would resolve ``distrib_la`` (and its
peers) from whatever the interpreter installs, the sealed release under
``lx test``; a later ``file_io`` import in the same pytest worker then refuses
the mix (``SourceClosureError``).  The drivers seal at start-up; this does the
same once per test process (``runtime.source_closure``, idempotent).
"""
from runtime.source_closure import ensure_source_closure

ensure_source_closure()
