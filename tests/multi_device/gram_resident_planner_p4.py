"""P4 gate for rank-invariant automatic Gram tile planning.

Run as four processes with one device each.  The planted process-local
resident floors straddle the 512-wide rung: planning from them independently
must diverge, while the production worst-process policy must select width 256
on every rank.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack()

import jax  # noqa: E402


def _fail(message: str) -> None:
    print(f"[gram-resident-planner-p4] FAIL: {message}", flush=True)
    raise SystemExit(1)


def main() -> None:
    from centroid.pivoted_cholesky import (
        _auto_gram_width_from_compiled_peaks,
    )
    from common.collectives import all_gather_processes
    from common.gpu_utils import worst_process_resident_bytes

    if jax.process_count() != 4 or jax.device_count() != 4:
        _fail(
            "needs exactly four processes and four global devices; got "
            f"{jax.process_count()} process(es), {jax.device_count()} device(s)"
        )

    rank = jax.process_index()
    local_resident = (100, 200, 300, 700)[rank]
    budget = 1000

    def choose(resident: int) -> int:
        width, _ = _auto_gram_width_from_compiled_peaks(
            256,
            max_width=1024,
            divisor=4,
            budget_bytes=budget,
            peak_for_width=lambda w: {"peak": int(resident) + int(w)},
        )
        return width

    naive_width = choose(local_resident)
    naive_widths = np.asarray(
        all_gather_processes(np.asarray(naive_width, dtype=np.int32))
    ).reshape(-1)
    if np.array_equal(naive_widths, np.full(4, naive_widths[0])):
        _fail(f"negative control did not diverge: {naive_widths.tolist()}")

    shared_resident = worst_process_resident_bytes(local_resident)
    shared_missing = worst_process_resident_bytes(
        None if rank == 3 else local_resident)
    fixed_width = choose(shared_resident)
    fixed_widths = np.asarray(
        all_gather_processes(np.asarray(fixed_width, dtype=np.int32))
    ).reshape(-1)
    if shared_resident != 700:
        _fail(f"shared resident is {shared_resident}, expected worst rank 700")
    if shared_missing is not None:
        _fail(
            "one rank's missing allocator sample did not propagate to all "
            f"ranks: {shared_missing}")
    if not np.array_equal(fixed_widths, np.full(4, 256, dtype=np.int32)):
        _fail(f"fixed widths differ or select the wrong rung: {fixed_widths.tolist()}")

    if rank == 0:
        print(
            "[gram-resident-planner-p4] negative-control widths="
            f"{naive_widths.tolist()} shared-resident={shared_resident} "
            f"shared-missing={shared_missing} "
            f"fixed-widths={fixed_widths.tolist()}",
            flush=True,
        )
        print("[gram-resident-planner-p4] PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_process()
