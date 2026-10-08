"""O(1) collective cost model, for receipts only.

One collective moving ``V`` bytes per rank among ``n_peers`` other ranks costs

    T = alpha0 + alpha_peer * n_peers + V / beta.

``V`` is the per-rank buffer: the all_to_all input (= output), the
all_gather output.  The model is generic: no topology, no intra- versus
inter-node term (owner ruling 2026-09-23).  No planner chooses a size or a
route from it (owner 2026-10-08: the constants are one machine's); the
route-G receipt prints its collectives against :func:`min_efficient_payload`
and the all-to-all floor at ``BETA_BPS``.  Pure Python.
"""
from __future__ import annotations

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
# ponytail: every receipt prices with the one calibrated row; one constant set
# for every collective kind, fitted on all_to_all (all_gather measured
# 2.6-3.7x faster at >= 16 MB).
_M = MACHINES["perlmutter-a100-ofi"]
ALPHA0_S = _M["alpha0_s"]
ALPHA_PEER_S = _M["alpha_peer_s"]
BETA_BPS = _M["beta_Bps"]


def min_efficient_payload(n_peers: int) -> float:
    """Bytes per rank at which latency is 20 % of a collective: 4·beta·alpha."""
    return 4.0 * BETA_BPS * (ALPHA0_S + ALPHA_PEER_S * int(n_peers))
