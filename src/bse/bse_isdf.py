"""Deprecated wrapper for the older BSE prototype.

Use bse.bse_jax for the current JAX implementation.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "bse.bse_isdf is deprecated; use bse.bse_jax instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .deprecated.bse_isdf import *  # noqa: F401,F403
