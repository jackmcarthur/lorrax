#!/usr/bin/env python3
"""Run the production driver against one completed insulating MPA fit."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    os.environ.setdefault("LORRAX_DEBUG_PRINT", "1")
    from gw import gw_jax

    fit_path = Path(os.environ["LORRAX_CERTIFIED_MPA_FIT"]).resolve()
    if not fit_path.is_file() or fit_path.stat().st_size == 0:
        raise RuntimeError(f"certified MPA fit is missing or empty: {fit_path}")

    def reuse_certified_fit(mode, *args, print_fn=print, **kwargs):
        if mode.value != "mpa":
            raise RuntimeError("fit-reuse harness requires compute_mode=mpa")
        if kwargs.get("occupation_state") is not None:
            raise RuntimeError("this lane is restricted to insulating MPA")
        print_fn(f"  MPA screening reuse: certified_fit={fit_path}")
        return {"mpa_fit": str(fit_path)}

    def skip_unused_static_rule(*args, print_fn=print, **kwargs):
        print_fn("  MPA screening reuse: static minimax solve skipped")
        return None, None

    gw_jax.compute_screening_model = reuse_certified_fit
    gw_jax.build_static_quadrature = skip_unused_static_rule
    result = gw_jax.main(["-i", "mpa_sigma.in"])
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
