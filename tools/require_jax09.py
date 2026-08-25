#!/usr/bin/env python3
"""Refuse any LORRAX launch unless JAX and JAXLIB are 0.9.x.

Package metadata is inspected without importing JAX, so this preflight cannot
initialize a CPU client, start distributed coordination, or consume the one
MPI initialization owned by XLA.
"""
from __future__ import annotations

import os
import re
import sys
from importlib import metadata


_FALSE_VALUES = {"", "0", "false", "no", "off"}


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        raise RuntimeError(f"{distribution} is not installed") from None


def _series(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"cannot parse version {version!r}")
    return int(match.group(1)), int(match.group(2))


def main() -> int:
    override = os.environ.get("LORRAX_JAX_UNSUPPORTED_OK", "")
    if override.strip().lower() not in _FALSE_VALUES:
        print(
            "JAX09_ENV_REFUSED: LORRAX_JAX_UNSUPPORTED_OK is enabled; "
            "unsupported-JAX overrides are forbidden for LORRAX drivers",
            file=sys.stderr,
        )
        return 86

    try:
        versions = {name: _version(name) for name in ("jax", "jaxlib")}
        wrong = {
            name: version
            for name, version in versions.items()
            if _series(version) != (0, 9)
        }
    except RuntimeError as exc:
        print(f"JAX09_ENV_REFUSED: {exc}", file=sys.stderr)
        return 86

    if wrong:
        rendered = ", ".join(f"{name}={version}" for name, version in wrong.items())
        print(
            "JAX09_ENV_REFUSED: required jax/jaxlib 0.9.x; got " + rendered,
            file=sys.stderr,
        )
        print(
            "Fix: on Perlmutter select LX_BASE_MODULE=lorrax_A; elsewhere "
            "activate the documented JAX/JAXLIB 0.9 environment before launch.",
            file=sys.stderr,
        )
        return 86

    print(
        "JAX09_ENV_OK "
        f"jax={versions['jax']} jaxlib={versions['jaxlib']} "
        f"python={sys.executable} "
        f"LX_BASE_MODULE={os.environ.get('LX_BASE_MODULE', '<direct-launch>')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
