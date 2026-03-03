"""Backward compatibility shim for gw_file_io.

All functionality has been moved to isdf.io module.
This file re-exports for backward compatibility.
"""
from isdf.io import (
    write_sigma_to_file,
    write_eqp_table,
    write_restart_state_to_h5,
    read_restart_state_from_h5,
    load_restart_state_from_h5,
    save_restart_state_per_proc,
)

__all__ = [
    "write_sigma_to_file",
    "write_eqp_table", 
    "write_restart_state_to_h5",
    "read_restart_state_from_h5",
    "load_restart_state_from_h5",
    "save_restart_state_per_proc",
]
