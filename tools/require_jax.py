#!/usr/bin/env python3
"""Refuse any LORRAX launch unless JAX and JAXLIB are in the platform's window.

Package metadata is inspected without importing JAX, so this preflight cannot
initialize a CPU client, start distributed coordination, or consume the one
MPI initialization owned by XLA.

The windows are ``SUPPORTED_WINDOWS`` in ``src/runtime/jax_support.py``, keyed
by the leg ``lxkit.legs.startup_gpu_leg`` names (``JAX_PLATFORMS``'s first
entry, else the installed jaxlib GPU plugin's leg, else ``cpu``).  Both files
are loaded by path: each imports only the standard library at module scope,
and loading them by path keeps the ``runtime`` package (and its jaxlib import)
out of this process.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from importlib import metadata
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_JAX_SUPPORT = _ROOT / "src" / "runtime" / "jax_support.py"
_LEGS = _ROOT / "services" / "lxkit" / "src" / "lxkit" / "legs.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


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
    support = _load("_lorrax_jax_support", _JAX_SUPPORT)
    legs = _load("_lorrax_legs", _LEGS)
    plugin_names = legs.installed_plugin_names()
    leg = legs.startup_gpu_leg(plugin_names=plugin_names)
    key = leg.key if leg is not None else "cpu"
    window = support.describe_window(key)
    try:
        versions = {name: _version(name) for name in ("jax", "jaxlib")}
        releases = {name: _release(version) for name, version in versions.items()}
    except RuntimeError as exc:
        print(f"JAX_ENV_REFUSED: {exc}", file=sys.stderr)
        return 86
    wrong = {name: versions[name] for name, vi in releases.items()
             if not support.in_window(vi, key)}
    if not wrong and releases["jax"][:2] != releases["jaxlib"][:2]:
        wrong = dict(versions)
    gpu_problems = support.check_gpu_packages(
        versions["jaxlib"], leg.jax_plugin_prefix if leg is not None else "", plugin_names)

    if wrong or gpu_problems:
        rendered = ", ".join(f"{name}={version}" for name, version in wrong.items())
        rendered = "; ".join(filter(None, (rendered, *gpu_problems)))
        print(
            f"JAX_ENV_REFUSED: required jax/jaxlib of one series, {window}, on the "
            f"{key} leg; got " + rendered,
            file=sys.stderr,
        )
        print(
            "Fix: activate a JAX/JAXLIB environment built from this checkout's "
            "uv.lock (Frontier: config/frontier/constraints.txt) before launch.",
            file=sys.stderr,
        )
        return 86

    print(
        "JAX_ENV_OK "
        f"jax={versions['jax']} jaxlib={versions['jaxlib']} leg={key} "
        f"python={sys.executable} "
        f"LX_BASE_MODULE={os.environ.get('LX_BASE_MODULE', '<direct-launch>')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
