"""End a rank that failed alone after pytest has printed its report.

``chain.run_stage`` sets ``chain.LONE_FAILURE`` when a failed rank's peers did
not join it (they are blocked in a collective).  A normal interpreter exit
would then wait on them in the jax.distributed shutdown, so the process ends
with ``os._exit`` here, after the terminal summary, and srun ends the step.
"""
import os
import sys


def pytest_unconfigure(config):
    chain = sys.modules.get("tests.hsuite.chain")
    if chain is not None and chain.LONE_FAILURE:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
