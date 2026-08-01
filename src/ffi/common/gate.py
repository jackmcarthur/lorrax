"""Re-export shim — the gate grammar moved to ``ffi.gate`` (one level up).

Retained one wave so existing ``ffi.common.gate`` imports keep working
(docs/architecture/ffi_layout.md §4); new code imports ``ffi.gate``.
"""
from ffi.gate import (  # noqa: F401
    MODE_HELP,
    MODE_SPELLINGS,
    Gate,
    announce_once,
    mesh_ffi_platform,
    rank0,
    rank_id,
    reset_gate_state,
)

__all__ = [
    "Gate", "MODE_SPELLINGS", "MODE_HELP",
    "rank_id", "rank0", "announce_once", "mesh_ffi_platform",
    "reset_gate_state",
]
