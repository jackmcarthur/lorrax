"""
Post-processing tools for COHSEX/GW calculations.

This module contains utilities for:
- Rotating DFT wavefunctions to QP basis
- Creating WFN_qp.h5 files with updated energies and rotated coefficients

Usage:
    python -m isdf.postprocess.rotate_wfn_to_qp WFN.h5 qp_wfn_rotations.h5
"""

__all__ = [
    'rotate_wfn_coefficients',
    'find_kpoint_mapping',
    'add_kpoint_mapping_to_rotation_file',
]


def __getattr__(name):
    """Lazy import to avoid issues when running submodules as __main__."""
    if name in __all__:
        from . import rotate_wfn_to_qp
        return getattr(rotate_wfn_to_qp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

