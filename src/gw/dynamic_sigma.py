"""Ansatz-neutral post-processing for frequency-dependent Sigma_c(omega).

The pole model owns production of a body cube and its q->0 head.  Everything
after that point -- head injection, interpolation at DFT energies and the
canonical ``sigma_mnk.h5`` write -- is common to every dynamic ansatz and
lives here exactly once.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from common.units import RYD_TO_EV
from common.wfn_transforms import get_enk_bandrange


def add_head_sigma_diag(
    sigma_c_body_omega: jax.Array,
    head_sigma_diag_w_kn_ry: np.ndarray | None,
) -> jax.Array:
    """Add a band-diagonal dynamic q->0 head to a Sigma_c body cube.

    Parameters
    ----------
    sigma_c_body_omega
        Body contribution with shape ``(n_omega, nk, nb, nb)`` in Ry.
    head_sigma_diag_w_kn_ry
        Head-only diagonal with shape ``(n_omega, nk, nb)`` in Ry, or
        ``None`` when the ansatz has no separate head contribution.
    """
    if head_sigma_diag_w_kn_ry is None:
        return sigma_c_body_omega

    head = np.asarray(head_sigma_diag_w_kn_ry)
    if head.shape != sigma_c_body_omega.shape[:3]:
        raise ValueError(
            "dynamic Sigma head shape must match the body diagonal: "
            f"head={head.shape}, body={sigma_c_body_omega.shape}")

    from .qsgw_utils import add_band_diag_sharded, is_band_sharded_sigma_omega
    if is_band_sharded_sigma_omega(sigma_c_body_omega):
        return add_band_diag_sharded(sigma_c_body_omega, head)

    n_w, nk, nb = head.shape
    dense = np.zeros((n_w, nk, nb, nb), dtype=np.complex128)
    idx = np.arange(nb)
    dense[:, :, idx, idx] = head
    return sigma_c_body_omega + jnp.asarray(
        dense, dtype=sigma_c_body_omega.dtype)


def eval_sigma_c_at_dft_energies(
    sigma_c_omega: jax.Array, *,
    config,
    band_slices, wfn, sym, meta, mesh_xy,
    print_fn,
):
    """Interpolate diag(Sigma_c)(omega) at every DFT band energy."""
    enk_dft, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range,
        band_slices.sigma_range, nspinor=meta.nspinor)
    enk_dft_ev = np.asarray(enk_dft) * RYD_TO_EV
    vbm_ev = float(wfn.vbm) * RYD_TO_EV
    cbm_ev = float(wfn.cbm) * RYD_TO_EV
    efermi_dft_ev = float(wfn.efermi) * RYD_TO_EV
    omega_dft_rel_ev = enk_dft_ev - efermi_dft_ev
    print_fn(
        f"  E_F(midgap) = {efermi_dft_ev:.6f} eV  "
        f"(VBM={vbm_ev:.6f}, CBM={cbm_ev:.6f})"
    )

    from .qsgw_utils import extract_sigma_diag_replicated, interp_along_omega
    sig_c_diag = (
        np.asarray(extract_sigma_diag_replicated(sigma_c_omega, mesh_xy))
        * RYD_TO_EV)
    sigma_c_at_dft_ev = interp_along_omega(
        sig_c_diag, np.asarray(config.omega_grid_ev, dtype=np.float64),
        omega_dft_rel_ev)

    return (
        sigma_c_at_dft_ev,
        omega_dft_rel_ev,
        efermi_dft_ev,
    )


def sigma_omega_output_path(config, input_dir: str) -> str:
    """Resolve the canonical dynamic-Sigma output path."""
    out_path = config.paths.sigma_omega_h5_file
    if not os.path.isabs(out_path):
        out_path = os.path.join(input_dir, out_path)
    return out_path


def write_sigma_omega(
    sigma_c_omega: jax.Array, *,
    sig_x: jax.Array,
    sig_h: jax.Array,
    config,
    input_dir: str,
    meta,
    mesh_xy,
    sym=None,
    band_extrapolation=None,
    print_fn=None,
) -> str:
    """Write canonical ``sigma_mnk.h5`` for any dynamic Sigma ansatz.

    ``band_extrapolation`` is the optional
    ``{"arrays": {...}, "attrs": {...}}`` the Σ_c band-convergence fit
    produces (``gw.band_extrapolation.extrapolation_h5_payload``).  None —
    the default and every non-extrapolating run — writes exactly the file
    that was written before the feature existed.
    """
    from file_io import write_sigma_omega_h5

    out_path = sigma_omega_output_path(config, input_dir)

    # One derivation of the star tables keeps the producer and consumer of
    # the antiunitary conjugation convention together.
    star = None
    if sym is not None:
        from .kin_ion_io import star_tables
        star = star_tables(sym)

    from .qsgw_utils import is_band_sharded_sigma_omega
    if is_band_sharded_sigma_omega(sigma_c_omega):
        shd = sigma_c_omega.sharding

        def _ev_tensors(c_ry, x_ry, h_ry):
            from file_io.sigma_output import derive_sigma_total
            c_ev = RYD_TO_EV * c_ry
            total = derive_sigma_total(
                c_ev, RYD_TO_EV * x_ry, RYD_TO_EV * h_ry)
            return total, c_ev

        total_ev, sigma_c_ev = jax.jit(
            _ev_tensors, out_shardings=(shd, shd))(
                sigma_c_omega, sig_x, sig_h)
        write_sigma_omega_h5(
            out_path, config.omega_grid_ev, total_ev,
            sigma_c_kij_ev=sigma_c_ev,
            sigma_sx_kij_ev=RYD_TO_EV * sig_x,
            hartree_kij_ev=RYD_TO_EV * sig_h,
            mesh=mesh_xy, star=star,
            band_extrapolation=band_extrapolation, print_fn=print_fn,
        )
        return out_path

    write_sigma_omega_h5(
        out_path, config.omega_grid_ev, None,
        sigma_c_kij_ev=RYD_TO_EV * sigma_c_omega,
        sigma_sx_kij_ev=RYD_TO_EV * sig_x,
        hartree_kij_ev=RYD_TO_EV * sig_h,
        mesh=mesh_xy, star=star,
        band_extrapolation=band_extrapolation, print_fn=print_fn,
    )
    return out_path


__all__ = [
    "add_head_sigma_diag",
    "eval_sigma_c_at_dft_energies",
    "sigma_omega_output_path",
    "write_sigma_omega",
]
