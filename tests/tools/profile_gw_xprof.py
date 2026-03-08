#!/usr/bin/env python3
"""Capture a full GW/COHSEX run trace for XProf memory analysis.

Usage example:
  uv run python tools/profile_gw_xprof.py \
      -i cohsex_prod.in \
      --workdir /home/jackm/projects/tests_isdf/cohsex_prod \
      --logdir ./profiles/xprof \
      --name cohsex_prod
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import jax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="GW input file passed to gw_jax (-i).")
    p.add_argument(
        "--workdir",
        default=".",
        help="Working directory for gw_jax execution (defaults to cwd).",
    )
    p.add_argument(
        "--logdir",
        default="profiles/xprof",
        help="Root directory where profiler traces are written.",
    )
    p.add_argument(
        "--name",
        default="gw_jax",
        help="Run name prefix for the output trace directory.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    if not workdir.exists():
        raise FileNotFoundError(f"Workdir does not exist: {workdir}")

    trace_root = Path(args.logdir).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_dir = trace_root / f"{args.name}-{stamp}"
    trace_dir.mkdir(parents=True, exist_ok=True)

    print(f"[xprof] workdir: {workdir}")
    print(f"[xprof] input:   {args.input}")
    print(f"[xprof] trace:   {trace_dir}")

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    try:
        os.chdir(workdir)
        # Mirror CLI invocation: python -m gw_isdf.gw_jax -i <input>
        sys.argv = ["gw_jax", "-i", args.input]
        from gw_isdf import gw_jax

        # Collect a full trace for XProf (TensorBoard memory_viewer).
        with jax.profiler.trace(str(trace_dir)):
            rc = gw_jax.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv

    print(f"[xprof] done: return_code={int(rc)}")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

