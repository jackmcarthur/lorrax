"""ISDF mini-library: ψ + centroids -> ζ interpolation vectors; GW and BSE are consumers."""
from isdf.core import (
    pair_density,        # centroid-selection Gram building block
    pair_density_aot_peak_bytes,  # compiler peak for the same kernel
    gram_q0_from_pair,   # q=0 Gram (centroid selection)
    gram_q0_aot_peak_bytes,       # compiler peak for the same fold
    c_q_from_psi_sm,     # centroid ψ -> C_q metric
    z_q_from_psi_sm,     # ψ(G) -> Z_q rhs (exported for tests)
    factor_c_q,          # C_q -> L_q (chol factor / indefinite passthrough)
    solve_zeta,          # (L_q, Z_q) -> ζ chunk
    fit_one_rchunk,      # fused per-r-chunk ζ workhorse; consumers loop this
    solve_zeta_charge_dense,  # (C, Z) -> ζ on ONE whole tile, producer's solve
)

__all__ = [
    "pair_density", "pair_density_aot_peak_bytes",
    "gram_q0_from_pair", "gram_q0_aot_peak_bytes",
    "c_q_from_psi_sm", "z_q_from_psi_sm",
    "factor_c_q", "solve_zeta", "fit_one_rchunk",
    "solve_zeta_charge_dense",
]
