"""O(1) collective cost model for the planners.

One collective moving ``V`` bytes per rank among ``n_peers`` other ranks costs

    T = alpha0 + alpha_peer * n_peers + V / beta.

``V`` is the per-rank buffer: the all_to_all input (= output), the
all_gather output.  The model is generic: no topology, no intra- versus
inter-node term (owner ruling 2026-09-23: LORRAX stays transferable across
HPC architectures).  Planners use only three things from it: the minimum
efficient payload, the number of calls, and the overlap estimate (max versus
sum).  Rules and the P100/P1000 extrapolation are in
docs/architecture/memory-model.md, "Communication cost model".  Pure Python.
"""
from __future__ import annotations

import math

#: Per-machine constants, one row per calibrated machine, each measured by
#: ``tools/comm_model_bench.py`` (run it, then ``--fit`` its log) and stored
#: with the machine, date and pool.  ``t_dispatch_s``: one synced jitted call,
#: host to host; ``t_scan_step_s``: one lax.scan step with no collective.
MACHINES = {
    "perlmutter-a100-ofi": dict(
        alpha0_s=31.6e-6, alpha_peer_s=2.74e-6, beta_Bps=19.7e9,
        t_dispatch_s=72e-6, t_scan_step_s=5.4e-6,
        measured="2026-09-23, pool 58814236, P16 (4 nodes x 4 A100-80GB), "
                 "NCCL over OFI/cxi, FI_CXI_RDZV_THRESHOLD=0; all_to_all "
                 "1 KB-1 GB, model/measured 0.70-1.16 (sandbox "
                 "runs/runtime/comm_model_20260923)"),
}
# ponytail: every machine prices with the one calibrated row until a second
# machine is benchmarked; add its row and a site lookup then.
_M = MACHINES["perlmutter-a100-ofi"]
# ponytail: one constant set for every collective kind, fitted on
# all_to_all.  all_gather measured 2.6-3.7x faster at >= 16 MB, so it is
# over-priced; all_to_all itself runs 25-30 % slower than this at 16-64 MB.
# Split per kind only when a plan decision flips on it.
ALPHA0_S = _M["alpha0_s"]
ALPHA_PEER_S = _M["alpha_peer_s"]
BETA_BPS = _M["beta_Bps"]


def comm_time(bytes_per_rank: float, n_peers: int) -> float:
    """Seconds for one collective of ``bytes_per_rank`` among ``n_peers`` peers."""
    return ALPHA0_S + ALPHA_PEER_S * int(n_peers) + float(bytes_per_rank) / BETA_BPS


def min_efficient_payload(n_peers: int) -> float:
    """Bytes per rank at which latency is 20 % of a collective: 4·beta·alpha."""
    return 4.0 * BETA_BPS * (ALPHA0_S + ALPHA_PEER_S * int(n_peers))


def split_calls(total_bytes: float, buf_bytes: float,
                n_peers: int) -> tuple[int, bool]:
    """``(n_calls, efficient)`` for moving ``total_bytes`` per rank through a
    ``buf_bytes`` comm buffer: the fewest calls the buffer allows, and whether
    each call is at or above :func:`min_efficient_payload`."""
    total, buf = float(total_bytes), float(buf_bytes)
    if total <= 0:
        return 0, True
    if buf <= 0:
        raise ValueError(f"comm buffer must be positive, got {buf_bytes}")
    n = math.ceil(total / min(buf, total))
    return n, total / n >= min_efficient_payload(n_peers)


def overlapped_time(t_compute: float, t_comm: float, n_steps: int) -> float:
    """Double-buffered wall for ``n_steps`` equal steps: the longer stream plus
    one step of the shorter (pipeline fill), against ``t_compute + t_comm``
    without overlap."""
    n = max(1, int(n_steps))
    return max(t_compute, t_comm) + min(t_compute, t_comm) / n
