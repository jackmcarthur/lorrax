"""Pure distributed-descriptor shape algebra with no runtime side effects."""

from __future__ import annotations


def ipiv_local_len(n: int, process_rows: int, row_block: int) -> int:
    """Return the local ``LOCr(M_A) + MB_A`` LU pivot extent.

    The service's LU wrappers admit only matrix extents divisible by the
    process-row count and use one block row per process row.
    """
    return n // process_rows + row_block
