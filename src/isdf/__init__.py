"""ISDF mini-library: ζ fitting and streamed Galerkin projection."""
from isdf.core import (
    pair_density,        # centroid-selection Gram building block
    pair_density_aot_peak_bytes,  # compiler peak for the same kernel
    gram_q0_from_pair,   # q=0 Gram (centroid selection)
    transverse_gram_q0_from_pair,  # stacked-current PSD candidate Gram
    gram_q0_aot_peak_bytes,       # compiler peak for the same fold
    c_q_from_psi_sm,     # centroid ψ -> C_q metric
    z_q_from_psi_sm,     # ψ(G) -> Z_q rhs (exported for tests)
    complete_ordered_pair_normal_equations,  # LR -> conjugation-closed LR+RL
    factor_c_q,          # C_q -> L_q (chol factor / indefinite passthrough)
    solve_zeta,          # (L_q, Z_q) -> ζ chunk
    fit_one_rchunk,      # fused per-r-chunk ζ workhorse; consumers loop this
    solve_zeta_charge_dense,  # (C, Z) -> ζ on ONE whole tile, producer's solve
)
from isdf.galerkin import (GalerkinBasis, fit_galerkin_basis,
                           iter_galerkin_rchunks)

__all__ = [
    "pair_density", "pair_density_aot_peak_bytes",
    "gram_q0_from_pair", "transverse_gram_q0_from_pair",
    "gram_q0_aot_peak_bytes",
    "c_q_from_psi_sm", "z_q_from_psi_sm",
    "complete_ordered_pair_normal_equations",
    "factor_c_q", "solve_zeta", "fit_one_rchunk",
    "solve_zeta_charge_dense", "GalerkinBasis", "fit_galerkin_basis",
    "iter_galerkin_rchunks",
]
