"""Ansatz-neutral post-processing for frequency-dependent Sigma_c(omega).

The pole model owns production of a body cube and its q->0 head.  Everything
after that point -- head injection, interpolation at DFT energies and the
canonical ``sigma_mnk.h5`` write -- is common to every dynamic ansatz and
lives here exactly once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from common.units import RYD_TO_EV
from common.wfn_transforms import get_enk_bandrange


@dataclass(frozen=True)
class OmegaCoverage:
    """WHICH (k, n) the Sigma(omega) grid actually sampled, and what was done.

    Returned beside the at-DFT Sigma_c so that every artifact written from
    it can say which of its cells are measurements and which are endpoint
    values.  Before this existed the two were indistinguishable on disk: the
    exact-origin Na run wrote finite semicore EQP values whose stored
    ``sigC`` is bit-for-shown-digits the ``omega = -5 eV`` endpoint of
    ``sigma_mnk.h5`` rather than Sigma at the state's DFT energy, and the
    only trace was a ``QSGW: 10142 clipped (41.3%)`` line in the log.

    ``mask_kn`` is True where the state WAS sampled.  ``policy`` is the
    resolved :data:`gw.qsgw_utils.OUT_OF_RANGE_POLICIES` value, so a reader
    knows whether the uncovered cells hold an endpoint value (``clamp``) or
    a non-finite marker (``mask``); ``refuse`` never reaches a consumer.
    """

    mask_kn: np.ndarray
    n_uncovered: int
    fraction_uncovered: float
    omega_min_ev: float
    omega_max_ev: float
    policy: str

    def summary(self) -> str:
        """One line, suitable for an artifact comment or a log."""
        return (f"omega_coverage: {self.n_uncovered} of "
                f"{int(np.asarray(self.mask_kn).size)} "
                f"({100.0 * self.fraction_uncovered:.2f}%) evaluation "
                f"energies outside the sampled grid "
                f"[{self.omega_min_ev:.4f}, {self.omega_max_ev:.4f}] eV; "
                f"out_of_range_policy={self.policy}")


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
    efermi_ry=None,
    efermi_provenance=None,
):
    """Interpolate diag(Sigma_c)(omega) at every DFT band energy.

    ``efermi_ry`` is the reference the Sigma(omega) GRID was built with.
    None keeps the loader's midgap/VBM ``wfn.efermi`` (the insulating PPM
    convention). A metallic caller MUST pass the same fixed-N mu its grid
    used: mixing references samples Sigma at energies shifted by
    (mu - efermi_midgap) — measured +2.79 eV on the sodium deck, showing
    up as a spurious ~+2.4 eV near-E_F QP correction.

    ``efermi_provenance`` is what ``gw.efermi.resolve_sigma_efermi_ry``
    said the reference WAS.  Given, it is stamped verbatim.  Omitted, the
    provenance falls back to the ``efermi_ry is None`` proxy below, which
    is right for every caller that has not been routed through the
    resolver yet and WRONG for one that has: the proxy labels any explicit
    reference "fixed-N mu", so an MPA run at ``fermi_reference = midgap``
    would be stamped as a metal's chemical potential.  Pass it.

    Returns ``(sigma_c_at_dft_ev, omega_dft_rel_ev, efermi_dft_ev,
    provenance, coverage)``.  THE PROVENANCE IS RETURNED, not re-derived by
    the caller: this function owns the ``efermi_ry is None`` decision, and
    the writer that stamps the answer into ``sigma_mnk.h5`` must record the
    same one it interpolated with, not a second opinion about it.  The
    :class:`OmegaCoverage` rides beside it for exactly the same reason — the
    at-DFT array cannot say by itself which of its cells were sampled and
    which are grid endpoints.
    """
    enk_dft, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range,
        band_slices.sigma_range, nspinor=meta.nspinor)
    enk_dft_ev = np.asarray(enk_dft) * RYD_TO_EV
    vbm_ev = float(wfn.vbm) * RYD_TO_EV
    cbm_ev = float(wfn.cbm) * RYD_TO_EV
    ref_ry = float(wfn.efermi) if efermi_ry is None else float(efermi_ry)
    efermi_dft_ev = ref_ry * RYD_TO_EV
    omega_dft_rel_ev = enk_dft_ev - efermi_dft_ev
    from file_io.sigma_output import (OMEGA_REFERENCE_FIXED_N_MU,
                                      OMEGA_REFERENCE_MIDGAP)
    if efermi_provenance is not None:
        provenance = str(efermi_provenance)
    else:
        provenance = (OMEGA_REFERENCE_MIDGAP if efermi_ry is None
                      else OMEGA_REFERENCE_FIXED_N_MU)
    print_fn(
        f"  omega reference = {efermi_dft_ev:.6f} eV "
        f"({provenance}; VBM={vbm_ev:.6f}, CBM={cbm_ev:.6f})"
    )

    from .qsgw_utils import (extract_sigma_diag_replicated,
                             interp_along_omega, omega_coverage,
                             resolve_out_of_range_policy)
    sig_c_diag = (
        np.asarray(extract_sigma_diag_replicated(sigma_c_omega, mesh_xy))
        * RYD_TO_EV)
    grid_ev = np.asarray(config.omega_grid_ev, dtype=np.float64)
    # THE OUTPUT PATH, so the policy is named and the count is REPORTED.
    # Until 2026-08-22 this call clamped every uncovered state to the grid
    # endpoint and said nothing, and the eqp0/eqp1 writer downstream wrote
    # those endpoint values as if they were Sigma at the state's own energy
    # (measured ~4 eV on the Na semicore deck; 41.3% of cells).  The SC
    # Hamiltonian path has always counted and rerouted these cells; this is
    # the output path catching up.
    policy = resolve_out_of_range_policy()
    covered, n_uncovered, frac_uncovered = omega_coverage(
        grid_ev, omega_dft_rel_ev)
    sigma_c_at_dft_ev = interp_along_omega(
        sig_c_diag, grid_ev, omega_dft_rel_ev,
        out_of_range=policy, context="Sigma_c at E_DFT (eqp0/eqp1)",
        print_fn=print_fn)

    return (
        sigma_c_at_dft_ev,
        omega_dft_rel_ev,
        efermi_dft_ev,
        provenance,
        OmegaCoverage(mask_kn=covered, n_uncovered=n_uncovered,
                      fraction_uncovered=frac_uncovered,
                      omega_min_ev=float(grid_ev[0]),
                      omega_max_ev=float(grid_ev[-1]),
                      policy=policy),
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
    omega_reference_ev,
    omega_reference_provenance,
    eval_energies_rel_ev,
    eval_energies_provenance,
    omega_coverage=None,
    sym=None,
    band_extrapolation=None,
    print_fn=None,
) -> str:
    """Write canonical ``sigma_mnk.h5`` for any dynamic Sigma ansatz.

    ``omega_reference_ev`` / ``omega_reference_provenance`` are REQUIRED
    keyword arguments, not defaulted ones: the ω axis this function
    writes is relative, and a caller that does not say what it is
    relative to produces the unstamped file audit A2 is about.  Both come
    straight from :func:`eval_sigma_c_at_dft_energies`, which owns the
    choice.

    ``eval_energies_rel_ev`` / ``eval_energies_provenance`` are REQUIRED for
    the same reason, one question further along.  The ω reference says what
    the AXIS is measured from; these say WHERE THIS Σ WAS EVALUATED, which
    is a different fact and was the one the file did not carry.  A from-disk
    reassembly (``eqp_bgw.make_eqp_bgw``) therefore could not distinguish a
    one-shot cube (evaluated at E_DFT, so centring eqp1's linearization
    there is correct) from a self-consistent one (evaluated at the converged
    QP spectrum, where centring at E_DFT is a different and wrong
    calculation).  Its docstring says so plainly — "Nothing in the files
    distinguishes the two cases, so this function will not guess" — and the
    guess it then made silently was E_DFT.  ``eval_energies_provenance`` is
    ``"at_e_dft"`` or ``"self_consistent_qp"``, MEASURED by the caller as an
    array comparison rather than inferred from a config key.

    ``omega_coverage`` is the optional :class:`OmegaCoverage` for the at-DFT
    interpolation; given it, the file states how many of its evaluation
    energies were never sampled and what was done about them.

    ``band_extrapolation`` is the optional
    ``{"arrays": {...}, "attrs": {...}}`` the Σ_c band-convergence fit
    produces (``gw.band_extrapolation.extrapolation_h5_payload``).  None —
    the default and every non-extrapolating run — writes exactly the file
    that was written before the feature existed.
    """
    from file_io import write_sigma_omega_h5
    from .ppm_windows import sigma_regularization_for_config

    out_path = sigma_omega_output_path(config, input_dir)
    # RE-DERIVED, not threaded.  ``resolve_sigma_regularization`` is a pure
    # function of the config the driver already ran against, so calling it
    # here stamps exactly the xi the kernel used -- and a threaded value
    # would be a second thing to keep in step, which is the whole class of
    # defect this seam was consolidated to remove.
    sigma_regularization = sigma_regularization_for_config(config)

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
            omega_reference_ev=omega_reference_ev,
            omega_reference_provenance=omega_reference_provenance,
            sigma_regularization=sigma_regularization,
            eval_energies_rel_ev=eval_energies_rel_ev,
            eval_energies_provenance=eval_energies_provenance,
            omega_coverage=omega_coverage,
            band_extrapolation=band_extrapolation,
            print_fn=print_fn,
        )
        return out_path

    write_sigma_omega_h5(
        out_path, config.omega_grid_ev, None,
        sigma_c_kij_ev=RYD_TO_EV * sigma_c_omega,
        sigma_sx_kij_ev=RYD_TO_EV * sig_x,
        hartree_kij_ev=RYD_TO_EV * sig_h,
        mesh=mesh_xy, star=star,
        omega_reference_ev=omega_reference_ev,
        omega_reference_provenance=omega_reference_provenance,
        sigma_regularization=sigma_regularization,
        eval_energies_rel_ev=eval_energies_rel_ev,
        eval_energies_provenance=eval_energies_provenance,
        omega_coverage=omega_coverage,
        band_extrapolation=band_extrapolation,
        print_fn=print_fn,
    )
    return out_path


__all__ = [
    "OmegaCoverage",
    "add_head_sigma_diag",
    "eval_sigma_c_at_dft_energies",
    "sigma_omega_output_path",
    "write_sigma_omega",
]
