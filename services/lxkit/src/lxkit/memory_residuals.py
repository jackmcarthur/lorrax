"""The residual table of a memory trace (``LORRAX_MEMORY_TRACE``, ``lxkit.memory``).

Per section: residual = measured peak - (inherited live + compiled + vendor), bytes per device:
measured is the allocator's high-water (GPU pool / BFC) or, on CPU, the largest live set seen;
inherited live is what was live when the section opened; compiled is the largest
``temp + output - alias`` of the executables compiled in it; vendor is the vendor term its
price booked (the libraries' own size queries).  A residual over max(2% of the limit, 0.25 GB)
is flagged and named by the arrays new at the peak (created inside the section).

    python -m lxkit.memory_residuals trace.json [more.json ...] [--all]
"""
from __future__ import annotations

import json
import sys

GB = 1e9


def rows(trace: dict) -> list:
    """One row per section name: the call with the largest |residual|."""
    limit = float(trace.get("run", {}).get("limit") or 0.0)
    tol = max(0.02 * limit, 0.25 * GB)
    best = {}
    for s in trace["sections"]:
        live = float(s["open"]["live"])
        allocator = s.get("source") in ("pool", "bfc")
        measured = float(s["measured"]) if allocator and s["measured"] >= 0 else float(s["peak"]["live"])
        compiled = max((c["temp"] + c["output"] - c["alias"] for c in s["compiles"]), default=0)
        vendor = float((s.get("priced") or {}).get("vendor", 0.0))
        residual = measured - (live + compiled + vendor)
        opened = {(tuple(a["shape"]), a["dtype"], a["spec"], a["born"]) for a in s["open"]["arrays"]}
        new = [a for a in s["peak"]["arrays"]
               if (tuple(a["shape"]), a["dtype"], a["spec"], a["born"]) not in opened]
        row = dict(section=s["name"], depth=s["depth"], measured=measured, live=live,
                   compiled=compiled, vendor=vendor, residual=residual, flagged=abs(residual) > tol,
                   priced=sum((s.get("priced") or {}).values()) if s.get("priced") else None,
                   peak_live=float(s["peak"]["live"]), new=new[:4], tol=tol)
        if s["name"] not in best or abs(residual) > abs(best[s["name"]]["residual"]):
            best[s["name"]] = row
    return sorted(best.values(), key=lambda r: -abs(r["residual"]))


def _array(a: dict) -> str:
    return (f"{a['count']}x {a['dtype']}{a['shape']} {a['spec']} "
            f"{a['bytes_per_device'] / GB:.2f} GB/dev from {a['born']}")


def table(trace: dict, everything: bool = False) -> str:
    out = ["section | measured | inherited live | compiled | vendor | residual | priced | "
           "new at peak (untracked candidates)", "---|---|---|---|---|---|---|---"]
    for r in rows(trace):
        if not (everything or r["flagged"]):
            continue
        priced = "-" if r["priced"] is None else f"{r['priced'] / GB:.2f}"
        out.append(f"{'  ' * r['depth']}{r['section']}{' **' if r['flagged'] else ''} | "
                   f"{r['measured'] / GB:.2f} | {r['live'] / GB:.2f} | {r['compiled'] / GB:.2f} | "
                   f"{r['vendor'] / GB:.2f} | {r['residual'] / GB:+.2f} | {priced} | "
                   + "; ".join(map(_array, r["new"])))
    flagged = sum(r["flagged"] for r in rows(trace))
    out.append(f"\n{flagged} section(s) over the tolerance "
               f"{rows(trace)[0]['tol'] / GB if trace['sections'] else 0:.2f} GB "
               "(max of 2% of the limit and 0.25 GB); GB per device.")
    return "\n".join(out)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    everything = "--all" in args
    for path in (a for a in args if a != "--all"):
        with open(path) as handle:
            print(f"## {path}\n" + table(json.load(handle), everything))
    return 0


if __name__ == "__main__":
    sys.exit(main())
