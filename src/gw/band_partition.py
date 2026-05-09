"""Three-way band partition for QSGW: protected / non-protected-in-range / out-of-range.

The QSGW iteration map carries ``H_qp_dft`` over the **active subspace**
(``band_slices.sigma`` of the wfn bundle).  Within that subspace, each
band falls into one of three categories per the user-configured
:class:`BandPartition`:

================================  ===========================  =======================================
Category                          ``protected_mask`` element   Diagonal of ``H_qp_dft`` per iteration
================================  ===========================  =======================================
Protected                         ``True``                     Full Σ at QP energy (off-diag mixed in)
Non-protected, in ω-range         ``False``, ``in_range=True`` Diagonal Σ at actual band energy
Non-protected, out of ω-range     ``False``, ``in_range=False`` Scissor extrapolation α·E_DFT + β
================================  ===========================  =======================================

Off-diagonals of ``H_qp_dft`` are kept **only** for protected×protected
pairs.  All other off-diagonals are zeroed each iteration so the
non-protected / out-of-range bands never mix into the protected
subspace's eigenproblem.

Why pre-compute the masks
-------------------------
Both masks are static across iterations (band identity doesn't change)
and small (``nb_active`` booleans each), so they live in :class:`BandPartition`
and are passed as JAX arrays to a single jit'd primitive
:func:`apply_band_partition` that the iteration map calls per step.

Validation
----------
:func:`BandPartition.warn_if_protected_outside_grid` prints a strong
all-caps warning if any protected band is outside the ω-grid — that
would mean the off-diagonal mixing pulls in Σ values that were
extrapolated by edge-clamping rather than properly evaluated, which is
exactly what the partition was designed to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Partition descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandPartition:
    """How each band in the active subspace is treated by the QSGW H build.

    Both masks are 1-D over the active band axis ``nb_active`` and are
    constant across SC iterations.

    Attributes
    ----------
    protected_mask : (nb_active,) bool
        True for bands that get full off-diagonal Σ corrections in
        ``H_qp_dft`` and participate in the basis rotation.
    in_range_mask : (nb_active,) bool
        True for bands whose ``E_DFT`` lies inside ``[ω_min, ω_max]`` at
        every k.  Used to decide between Σ_diag (in-range) and scissor
        (out-of-range) for the *non-protected* bands' diagonal.
    """

    protected_mask: jax.Array
    in_range_mask: jax.Array

    @classmethod
    def all_protected(cls, nb_active: int) -> "BandPartition":
        """Default: every active band is protected, every band in range.
        ``apply_band_partition`` reduces to identity in this case (no
        change to current behaviour).  Used as the default in
        :class:`sc_iteration.SCInputs` so existing code paths are
        unaffected until the partition is configured deliberately."""
        ones = jnp.ones(nb_active, dtype=bool)
        return cls(protected_mask=ones, in_range_mask=ones)

    def warn_if_protected_outside_grid(self, *, print_fn=print) -> None:
        """Loud all-caps warning if any protected band is outside the
        ω-grid — that breaks the partition's whole purpose."""
        leak = bool(jnp.any(self.protected_mask & ~self.in_range_mask))
        if leak:
            n_leak = int(jnp.sum(self.protected_mask & ~self.in_range_mask))
            print_fn("=" * 72)
            print_fn(
                f"!!! WARNING: {n_leak} PROTECTED BANDS LIE OUTSIDE THE "
                f"Σ_c(ω) GRID — !!!"
            )
            print_fn(
                "!!! THE QSGW H WILL HAVE OFF-DIAGONAL MIXING WITH BANDS "
                "WHOSE Σ IS UNRELIABLY CLAMPED AT THE GRID EDGE.       !!!"
            )
            print_fn(
                "!!! INCREASE sigma_omega_min/max_ev OR REDUCE THE "
                "PROTECTED BAND RANGE.                                  !!!"
            )
            print_fn("=" * 72)


# ---------------------------------------------------------------------------
# Per-iteration mask primitive
# ---------------------------------------------------------------------------

@jax.jit
def apply_band_partition(
    H_full: jax.Array,
    *,
    protected_mask: jax.Array,
    in_range_mask: jax.Array,
    scissor_E_qp_kn: jax.Array,
) -> jax.Array:
    """Apply the three-way band partition to a full QSGW Hamiltonian.

    Parameters
    ----------
    H_full : (nk, nb_active, nb_active) complex
        The full QSGW H = ``kin_ion + V_H + Σ_xc`` in the DFT basis,
        as if every band were protected.
    protected_mask, in_range_mask : (nb_active,) bool
        See :class:`BandPartition`.
    scissor_E_qp_kn : (nk, nb_active) real
        Scissor-extrapolated QP energies E_QP = α·E_DFT + β computed by
        the caller.  Used as the diagonal of ``H_partitioned`` for
        out-of-range bands; ignored for in-range bands.  Pass zeros if
        no scissor is in play (in-range ≡ all bands).

    Returns
    -------
    H_partitioned : (nk, nb_active, nb_active) complex
        ``H_full`` with:
          - off-diagonals zeroed for any (m, n) where m or n is not protected;
          - diagonals replaced by ``scissor_E_qp_kn`` for non-protected
            out-of-range bands; otherwise kept from ``H_full``.

    Identity case
    -------------
    When all bands are protected and in range
    (``BandPartition.all_protected``), this returns ``H_full`` unchanged.
    """
    nk, nb, _ = H_full.shape
    eye = jnp.eye(nb, dtype=H_full.dtype)

    # Off-diagonal keep-mask: 1 where both m and n are protected, else 0.
    p = protected_mask.astype(H_full.dtype)
    offdiag_keep = p[:, None] * p[None, :]
    # Zero the off-diagonal portion outside protected×protected.
    offdiag_part = H_full * (1.0 - eye) * offdiag_keep[None, :, :]

    # Diagonal: in-range bands keep H_full's diag; out-of-range get scissor.
    diag_full = jnp.diagonal(H_full, axis1=1, axis2=2)            # (nk, nb)
    diag_kept = jnp.where(
        in_range_mask[None, :], diag_full,
        scissor_E_qp_kn.astype(H_full.dtype),
    )                                                              # (nk, nb)

    # Reassemble: off-diag matrix + diag(diag_kept).
    return offdiag_part + diag_kept[:, :, None] * eye[None, :, :]


__all__ = ["BandPartition", "apply_band_partition"]
