"""ISDF mini-library: ζ fitting and streamed Galerkin projection."""
from isdf.core import (
    c_q_from_psi_sm,     # centroid ψ -> C_q metric
    complete_ordered_pair_normal_equations,  # LR -> conjugation-closed LR+RL
    factor_c_q,          # C_q -> L_q (chol factor / indefinite passthrough)
    solve_zeta_charge_dense,  # (C, Z) -> (ζ, cut figures) on ONE whole tile, producer's solve
)
from isdf.galerkin import (GalerkinBasis, fit_galerkin_basis,
                           iter_galerkin_rchunks)

__all__ = [
    "c_q_from_psi_sm",
    "complete_ordered_pair_normal_equations",
    "factor_c_q", "solve_zeta_charge_dense", "GalerkinBasis", "fit_galerkin_basis",
    "iter_galerkin_rchunks",
]
