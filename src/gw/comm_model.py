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

import math

from runtime.network_env import site_profile

#: The site profile's calibrated row (``config/<site>/site.toml``
#: ``[comm_model]``, fitted by ``tools/comm_model_bench.py``).  One constant
#: set prices every collective kind, fitted on all_to_all (all_gather measured
#: 2.6-3.7x faster at >= 16 MB).  A site without a row prices as NaN.
_M = (site_profile() or {}).get("comm_model", {})
ALPHA0_S = _M.get("alpha0_s", math.nan)
ALPHA_PEER_S = _M.get("alpha_peer_s", math.nan)
BETA_BPS = _M.get("beta_Bps", math.nan)


def min_efficient_payload(n_peers: int) -> float:
    """Bytes per rank at which latency is 20 % of a collective: 4·beta·alpha."""
    return 4.0 * BETA_BPS * (ALPHA0_S + ALPHA_PEER_S * int(n_peers))
