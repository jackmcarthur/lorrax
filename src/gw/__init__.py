"""GW/COHSEX driver package."""

import os

# Ensure double precision is enabled before any module in this package imports JAX.
os.environ.setdefault("JAX_ENABLE_X64", "1")

from .gw_init import get_bandranges
from .gw_config import read_lorrax_input, read_cohsex_input

__all__ = ["get_bandranges", "read_lorrax_input", "read_cohsex_input"]
