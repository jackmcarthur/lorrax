#!/usr/bin/env python3
"""Run the frozen insulating MPA fit through this worktree."""

import os
from pathlib import Path


os.environ.setdefault("LORRAX_DEBUG_PRINT", "1")
from gw import gw_jax

fit_path = Path(os.environ["LORRAX_CERTIFIED_MPA_FIT"]).resolve()


def reuse_certified_fit(mode, *args, print_fn=print, **kwargs):
    if mode.value != "mpa" or kwargs.get("occupation_state") is not None:
        raise RuntimeError("fit-reuse harness requires insulating MPA")
    print_fn(f"  MPA screening reuse: certified_fit={fit_path}")
    return {"mpa_fit": str(fit_path)}


def skip_unused_static_rule(*args, print_fn=print, **kwargs):
    print_fn("  MPA screening reuse: static minimax solve skipped")
    return None, None


gw_jax.compute_screening_model = reuse_certified_fit
gw_jax.build_static_quadrature = skip_unused_static_rule
raise SystemExit(gw_jax.main(["-i", "mpa_sigma.in"]) or 0)
