#!/usr/bin/env python3
"""Summarize XProf memory_viewer peaks for selected HLO modules.

Usage:
  uv run python tools/analyze_xprof_memory.py \
      profiles/xprof/<run>/plugins/profile/<stamp>/GPUtop.xplane.pb \
      --module-regex "compute_ZCT|solve_all_q|compute_CCT|extract_centroids"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from xprof.convert.raw_to_tool_data import xspace_to_tool_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xplane", help="Path to GPUtop.xplane.pb")
    p.add_argument(
        "--module-regex",
        default="compute_ZCT|solve_all_q|compute_CCT|extract_centroids",
        help="Regex for module filenames (without .hlo_proto.pb).",
    )
    p.add_argument("--top", type=int, default=6, help="Number of top peak buffers to print.")
    return p.parse_args()


def module_names_from_dir(profile_dir: Path, pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for f in sorted(profile_dir.glob("*.hlo_proto.pb")):
        stem = f.name.removesuffix(".hlo_proto.pb")
        if not pattern.search(stem):
            continue
        m = re.search(r"\((\d+)\)$", stem)
        if not m:
            continue
        out.append((stem, m.group(1)))
    return out


def main() -> int:
    args = parse_args()
    xplane = Path(args.xplane).expanduser().resolve()
    if not xplane.exists():
        raise FileNotFoundError(xplane)

    profile_dir = xplane.parent
    pattern = re.compile(args.module_regex)
    modules = module_names_from_dir(profile_dir, pattern)
    if not modules:
        print("No matching .hlo_proto.pb modules found.")
        return 1

    for mod, pid in modules:
        raw, _ = xspace_to_tool_data(
            [str(xplane)],
            "memory_viewer",
            {"module_name": mod, "program_id": pid},
        )
        if raw is None:
            print(f"\n== {mod} ==\n  memory_viewer: no data")
            continue
        j = json.loads(raw)
        print(f"\n== {mod} ==")
        print(
            f"  peakHeapMib={float(j.get('peakHeapMib', 0.0)):.3f}  "
            f"entryParamsMib={float(j.get('entryComputationParametersMib', 0.0)):.3f}  "
            f"totalAllocMib={float(j.get('totalBufferAllocationMib', 0.0)):.3f}"
        )
        for b in j.get("maxHeapBySize", [])[: max(1, args.top)]:
            src = b.get("sourceInfo") or {}
            where = ""
            if src.get("fileName"):
                where = f" {src.get('fileName')}:{src.get('lineNumber', '')}"
            print(
                "  "
                f"{float(b.get('logicalBufferSizeMib', 0.0)):8.3f} MiB  "
                f"{str(b.get('groupName', '')):10s} "
                f"{str(b.get('opCode', '')):14s} "
                f"{str(b.get('instructionName', '')):26s} "
                f"tf={str(b.get('tfOpName', ''))}{where}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

