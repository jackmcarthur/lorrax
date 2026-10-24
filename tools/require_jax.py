#!/usr/bin/env python3
"""Refuse any LORRAX launch unless JAX and JAXLIB are in the supported series.

Package metadata is inspected without importing JAX, so this preflight cannot
initialize a CPU client, start distributed coordination, or consume the one
MPI initialization owned by XLA.

The window is ``SUPPORTED_MIN`` / ``SUPPORTED_MAX_EXCLUSIVE`` in
``src/runtime/jax_support.py``, loaded from that file by path: it imports
only the standard library at module scope, and loading it by path keeps the
``runtime`` package (and its jaxlib import) out of this process.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from importlib import metadata
from pathlib import Path


_FALSE_VALUES = {"", "0", "false", "no", "off"}
_JAX_SUPPORT = Path(__file__).resolve().parents[1] / "src" / "runtime" / "jax_support.py"


def _window() -> tuple[tuple[int, ...], tuple[int, ...]]:
    spec = importlib.util.spec_from_file_location("_lorrax_jax_support", _JAX_SUPPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.SUPPORTED_MIN), tuple(module.SUPPORTED_MAX_EXCLUSIVE)


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        raise RuntimeError(f"{distribution} is not installed") from None


def _release(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if match is None:
        raise RuntimeError(f"cannot parse version {version!r}")
    return tuple(int(v) for v in match.groups(default="0"))


def main() -> int:
    override = os.environ.get("LORRAX_JAX_UNSUPPORTED_OK", "")
    if override.strip().lower() not in _FALSE_VALUES:
        print(
            "JAX_ENV_REFUSED: LORRAX_JAX_UNSUPPORTED_OK is enabled; "
            "unsupported-JAX overrides are forbidden for LORRAX drivers",
            file=sys.stderr,
        )
        return 86

    lo, hi = _window()
    series = ".".join(str(v) for v in lo[:2])
    try:
        versions = {name: _version(name) for name in ("jax", "jaxlib")}
        wrong = {
            name: version
            for name, version in versions.items()
            if not (lo <= _release(version) < hi)
        }
    except RuntimeError as exc:
        print(f"JAX_ENV_REFUSED: {exc}", file=sys.stderr)
        return 86

    if wrong:
        rendered = ", ".join(f"{name}={version}" for name, version in wrong.items())
        print(
            f"JAX_ENV_REFUSED: required jax/jaxlib {series}.x; got " + rendered,
            file=sys.stderr,
        )
        print(
            f"Fix: activate a JAX/JAXLIB {series} environment built from this "
            "checkout's uv.lock before launch.",
            file=sys.stderr,
        )
        return 86

    print(
        "JAX_ENV_OK "
        f"jax={versions['jax']} jaxlib={versions['jaxlib']} "
        f"python={sys.executable} "
        f"LX_BASE_MODULE={os.environ.get('LX_BASE_MODULE', '<direct-launch>')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
