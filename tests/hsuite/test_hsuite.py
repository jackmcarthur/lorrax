"""The driver suite as one pytest cell: the whole chain of ``chain.py``.

P4 (the landing shape, four pytest ranks, each running its own driver
process):

    lx run -N 1 -G 4 -n 4 -- python3 -m pytest tests/hsuite -q -p no:cacheprovider

``lx test tests/hsuite`` runs the same cell at P1.  The compile cache lives
in ``.hsuite_jax_cache`` in the checkout (warm after the first run);
``HSUITE_CACHE_DIR`` points it elsewhere, for example at an empty directory
for a cold measurement.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

from tests.hsuite import chain
from tests.hsuite import rank_session


def test_driver_chain_matches_references():
    stamp = rank_session.exchange(time.strftime("%Y%m%d-%H%M%S"))[0]
    job = os.environ.get("SLURM_JOB_ID", "local")
    worker = os.environ.get("PYTEST_XDIST_WORKER", "p")
    out = chain.REPO / ".hsuite_runs" / f"{job}-{worker}-{stamp}"
    cache = Path(os.environ.get("HSUITE_CACHE_DIR")
                 or chain.REPO / ".hsuite_jax_cache")
    walls, problems = chain.run_chain(out, cache_dir=cache)
    print("hsuite walls (s):", {k: round(v, 1) for k, v in walls.items()})
    assert not problems, "\n".join(problems)
