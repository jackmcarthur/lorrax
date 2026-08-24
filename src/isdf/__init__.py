"""ISDF mini-library: ζ fitting and streamed Galerkin projection."""
from isdf.core import (
    pair_density,        # centroid-selection Gram building block
    gram_q0_from_pair,   # q=0 Gram (centroid selection)
    c_q_from_psi_sm,     # centroid ψ -> C_q metric
    z_q_from_psi_sm,     # ψ(G) -> Z_q rhs (exported for tests)
    factor_c_q,          # C_q -> L_q (chol factor / indefinite passthrough)
    solve_zeta,          # (L_q, Z_q) -> ζ chunk
    fit_one_rchunk,      # fused per-r-chunk ζ workhorse; consumers loop this
    solve_zeta_charge_dense,  # (C, Z) -> ζ on ONE whole tile, producer's solve
)

__all__ = [
    "pair_density", "gram_q0_from_pair",
    "c_q_from_psi_sm", "z_q_from_psi_sm",
    "factor_c_q", "solve_zeta", "fit_one_rchunk",
    "solve_zeta_charge_dense",
]
