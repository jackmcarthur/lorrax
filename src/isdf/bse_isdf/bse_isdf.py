"""Deprecated wrapper for the older BSE prototype.

Use isdf.bse_isdf.bse_jax for the current JAX implementation.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "isdf.bse_isdf.bse_isdf is deprecated; use isdf.bse_isdf.bse_jax instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .deprecated.bse_isdf import *  # noqa: F401,F403
