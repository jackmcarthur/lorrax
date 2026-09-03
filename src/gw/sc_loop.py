"""Fixed-index self-consistent QSGW update law.

This module owns the only energy-to-operator decision in both the live QSGW
map and the fixed-table ``eqp2`` loop.  Screening and self-energy production
remain elsewhere; the objects here say only how one completed Sigma table is
read and how its protected/outer blocks become a Hermitian Hamiltonian.

The protected/outer split is structural and immutable:

``P = [b0, b3)`` and ``U = [b3, b4_user)``.

No mutable band partition, hysteresis, or state matching enters this module.
The only state-identification convention is the owner-set exact-degeneracy
symmetrisation (at most 0.1 meV), which makes the update independent of the
arbitrary eigenvector gauge inside an unresolved QP block.  See
``docs/dev/notes/DESIGN_self_consistent_loop.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
import numpy as np

from common.units import RYD_TO_EV


class EvaluationPolicy(str, Enum):
    """How a protected pair selects its two Sigma frequencies."""

    FERMI = "fermi"
    CLAMP = "clamp"

    @classmethod
    def coerce(cls, value: "EvaluationPolicy | str") -> "EvaluationPolicy":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"SC evaluation policy must be one of {allowed}; got "
                f"{value!r}.") from exc


@dataclass(frozen=True)
class BandClasses:
    """Fixed band-index classes for one QSGW calculation.

    Parameters
    ----------
    band_start
        Global index of the first represented band (``b0``).
    protected_stop
        Global, half-open upper edge of P (``b3``).
    outer_stop
        Global, half-open logical upper edge of U (``b4_user``).  Mesh
        padding above this edge is not represented in the Hamiltonian.
    occupied_stop
        Global, half-open occupied-band edge.  It is used only to identify
        the two highest protected conduction bands for the rigid outer
        scissor.
    """

    band_start: int
    protected_stop: int
    outer_stop: int
    occupied_stop: int

    def __post_init__(self) -> None:
        edges = tuple(int(v) for v in (
            self.band_start, self.occupied_stop,
            self.protected_stop, self.outer_stop))
        if not edges[0] <= edges[1] <= edges[2] <= edges[3]:
            raise ValueError(
                "BandClasses requires b0 <= occupied_stop <= b3 <= "
                f"b4_user; got {edges}.")
        if self.n_protected_conduction < 2 and self.n_outer:
            raise ValueError(
                "BandClasses needs at least two protected conduction bands "
                "to define the outer-block scissor; got "
                f"{self.n_protected_conduction}.")

    @classmethod
    def from_run(cls, band_slices, meta) -> "BandClasses":
        """Build the immutable P/U split from the canonical run edges."""
        return cls(
            band_start=int(band_slices.b0),
            protected_stop=int(band_slices.b3),
            outer_stop=int(meta.b_id_4_user),
            occupied_stop=int(meta.nelec),
        )

    @property
    def n_protected(self) -> int:
        return int(self.protected_stop - self.band_start)

    @property
    def n_outer(self) -> int:
        return int(self.outer_stop - self.protected_stop)

    @property
    def n_total(self) -> int:
        return int(self.outer_stop - self.band_start)

    @property
    def occupied_stop_local(self) -> int:
        return int(self.occupied_stop - self.band_start)

    @property
    def n_protected_conduction(self) -> int:
        return int(self.protected_stop - self.occupied_stop)

    @property
    def scissor_bands_local(self) -> slice:
        """The two highest protected conduction indices, half open."""
        return slice(self.n_protected - 2, self.n_protected)


def _validate_uniform_grid(omega_ev: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega_ev, dtype=np.float64)
    if omega.ndim != 1 or omega.size < 2:
        raise ValueError("SigmaTable omega_ev must contain at least two points.")
    steps = np.diff(omega)
    if not np.all(steps > 0.0):
        raise ValueError("SigmaTable omega_ev must be strictly increasing.")
    if not np.allclose(steps, steps[0], rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("SigmaTable omega_ev must be uniform.")
    if not omega[0] <= 0.0 <= omega[-1]:
        raise ValueError(
            "SigmaTable omega_ev must contain the Fermi-relative zero.")
    return omega


@dataclass(frozen=True)
class SigmaTable:
    """One iteration's self-energy table in the current QP basis.

    Array shapes are ``W=(n_omega,nk,P,P)``, ``PP=(nk,P,P)``,
    ``PU=(nk,P,U)``, and ``NN=(nk,P+U,P+U)``.  Energies are eV for the
    interpolation axis and Ry for operators/eigenvalues.

    ``sigma_c_pp_wkij_ry`` already includes the dynamic head.  The head is
    band diagonal, so the P x U block has no separate head term.
    ``sigma_xc_pu_fermi_kij_ry`` is the complete exchange plus correlation
    block evaluated at the Fermi level.
    """

    omega_ev: np.ndarray
    sigma_c_pp_wkij_ry: object
    sigma_x_pp_kij_ry: object
    v_h_pp_kij_ry: object
    sigma_xc_pu_fermi_kij_ry: object
    v_h_pu_kij_ry: object
    kin_ion_qp_kij_ry: object
    dft_h_qp_kij_ry: object
    e_dft_kn_ry: object

    def __post_init__(self) -> None:
        omega = _validate_uniform_grid(self.omega_ev)
        object.__setattr__(self, "omega_ev", omega)
        cube_shape = tuple(np.shape(self.sigma_c_pp_wkij_ry))
        if len(cube_shape) != 4:
            raise ValueError(
                "SigmaTable correlation table must have shape "
                f"(n_omega,nk,P,P); got {cube_shape}.")
        n_omega, nk, n_p, n_p2 = cube_shape
        if n_omega != omega.size or n_p != n_p2:
            raise ValueError(
                "SigmaTable correlation/grid shape mismatch: "
                f"omega={omega.shape}, cube={cube_shape}.")
        pp = (nk, n_p, n_p)
        for name in ("sigma_x_pp_kij_ry", "v_h_pp_kij_ry"):
            got = tuple(np.shape(getattr(self, name)))
            if got != pp:
                raise ValueError(f"SigmaTable {name} has {got}, expected {pp}.")
        full = tuple(np.shape(self.kin_ion_qp_kij_ry))
        if len(full) != 3 or full[0] != nk or full[1] != full[2]:
            raise ValueError(
                "SigmaTable kin_ion_qp_kij_ry must have shape (nk,N,N); "
                f"got {full}.")
        if tuple(np.shape(self.dft_h_qp_kij_ry)) != full:
            raise ValueError("SigmaTable DFT and kin+ion Hamiltonians disagree.")
        n_u = full[1] - n_p
        pu = (nk, n_p, n_u)
        for name in ("sigma_xc_pu_fermi_kij_ry", "v_h_pu_kij_ry"):
            got = tuple(np.shape(getattr(self, name)))
            if got != pu:
                raise ValueError(f"SigmaTable {name} has {got}, expected {pu}.")
        if tuple(np.shape(self.e_dft_kn_ry)) != (nk, full[1]):
            raise ValueError(
                "SigmaTable e_dft_kn_ry must have shape "
                f"{(nk, full[1])}; got {np.shape(self.e_dft_kn_ry)}.")

    def at(self, energies_rel_ev) -> object:
        """Interpolate Sigma_c at one row energy per ``(k,m)``.

        Parameters
        ----------
        energies_rel_ev : array_like, (nk, P)
            Fermi-relative evaluation energies in eV.  Every value must be
            inside the table.  Policy decisions (including clipping) belong
            to :func:`effective_sigma`, not this primitive.

        Returns
        -------
        array, (nk, P, P)
            ``out[k,m,n] = Sigma_c(energies[k,m]; k,m,n)`` in Ry.
        """
        energy = np.asarray(energies_rel_ev, dtype=np.float64)
        _, nk, n_p, _ = np.shape(self.sigma_c_pp_wkij_ry)
        if energy.shape != (nk, n_p):
            raise ValueError(
                f"SigmaTable.at needs energies shape {(nk, n_p)}; "
                f"got {energy.shape}.")
        lo, hi = float(self.omega_ev[0]), float(self.omega_ev[-1])
        outside = (energy < lo) | (energy > hi)
        if np.any(outside):
            k, band = np.argwhere(outside)[0]
            raise ValueError(
                "SigmaTable.at received an out-of-domain request: "
                f"E[{int(k)},{int(band)}]={energy[k, band]:.9g} eV, "
                f"grid=[{lo:.9g},{hi:.9g}] eV. Apply an EvaluationPolicy "
                "before interpolation.")
        idx_hi = np.clip(
            np.searchsorted(self.omega_ev, energy, side="left"),
            1, self.omega_ev.size - 1)
        idx_lo = idx_hi - 1
        omega_lo = self.omega_ev[idx_lo]
        omega_hi = self.omega_ev[idx_hi]
        weight_hi = (energy - omega_lo) / (omega_hi - omega_lo)
        weight_lo = 1.0 - weight_hi

        cube = jnp.asarray(self.sigma_c_pp_wkij_ry)
        one = (1,) + tuple(cube.shape[1:])
        ilo = jnp.broadcast_to(jnp.asarray(idx_lo)[None, :, :, None], one)
        ihi = jnp.broadcast_to(jnp.asarray(idx_hi)[None, :, :, None], one)
        value_lo = jnp.take_along_axis(cube, ilo, axis=0)[0]
        value_hi = jnp.take_along_axis(cube, ihi, axis=0)[0]
        return (jnp.asarray(weight_lo)[:, :, None] * value_lo
                + jnp.asarray(weight_hi)[:, :, None] * value_hi)

    def at_columns(self, energies_rel_ev) -> object:
        """Interpolate Sigma_c at one column energy per ``(k,n)``.

        Returns ``out[k,m,n] = Sigma_c(energies[k,n]; k,m,n)``.  This is
        intentionally not an adjoint of :meth:`at`: a frequency-resolved
        self-energy need not be Hermitian before the QSGW construction's
        explicit final Hermitisation.
        """
        energy = np.asarray(energies_rel_ev, dtype=np.float64)
        _, nk, n_p, _ = np.shape(self.sigma_c_pp_wkij_ry)
        if energy.shape != (nk, n_p):
            raise ValueError(
                f"SigmaTable.at_columns needs energies shape {(nk, n_p)}; "
                f"got {energy.shape}.")
        lo, hi = float(self.omega_ev[0]), float(self.omega_ev[-1])
        outside = (energy < lo) | (energy > hi)
        if np.any(outside):
            k, band = np.argwhere(outside)[0]
            raise ValueError(
                "SigmaTable.at_columns received an out-of-domain request: "
                f"E[{int(k)},{int(band)}]={energy[k, band]:.9g} eV, "
                f"grid=[{lo:.9g},{hi:.9g}] eV.")
        idx_hi = np.clip(
            np.searchsorted(self.omega_ev, energy, side="left"),
            1, self.omega_ev.size - 1)
        idx_lo = idx_hi - 1
        omega_lo = self.omega_ev[idx_lo]
        omega_hi = self.omega_ev[idx_hi]
        weight_hi = (energy - omega_lo) / (omega_hi - omega_lo)
        weight_lo = 1.0 - weight_hi

        cube = jnp.asarray(self.sigma_c_pp_wkij_ry)
        one = (1,) + tuple(cube.shape[1:])
        ilo = jnp.broadcast_to(jnp.asarray(idx_lo)[None, :, None, :], one)
        ihi = jnp.broadcast_to(jnp.asarray(idx_hi)[None, :, None, :], one)
        value_lo = jnp.take_along_axis(cube, ilo, axis=0)[0]
        value_hi = jnp.take_along_axis(cube, ihi, axis=0)[0]
        return (jnp.asarray(weight_lo)[:, None, :] * value_lo
                + jnp.asarray(weight_hi)[:, None, :] * value_hi)


def _adjoint(matrix):
    return jnp.conj(jnp.swapaxes(matrix, -1, -2))


def _symmetrize_exact_qp_blocks(
    sigma_pp,
    energies_kp_ev: np.ndarray,
    tol_ev: float,
):
    """Make unresolved QP blocks scalar in their degenerate subspaces.

    For a block ``B`` whose adjacent QP energies differ by less than
    ``tol_ev``, this applies the unitary-group average

    ``Sigma_B -> trace(Sigma_B) / dim(B) * I_B``.

    That is the unique within-block operator invariant under an arbitrary
    unitary choice of eigenvectors.  It is deliberately stronger than merely
    averaging the diagonal: retaining an off-diagonal component would select
    a preferred basis and allow an exact multiplet to alternate gauges from
    one map to the next.  Couplings to resolved states are left untouched.

    Parameters
    ----------
    sigma_pp : array, shape (nk, P, P)
        Hermitian effective self-energy on the protected block, in Ry.
    energies_kp_ev : np.ndarray, shape (nk, P)
        Sorted current QP energies in eV.
    tol_ev : float
        Maximum adjacent splitting treated as unresolved, in eV.  Zero
        disables the convention; callers cap enabled values at 0.1 meV.

    Returns
    -------
    array, shape (nk, P, P)
        Symmetrised effective self-energy.
    tuple[int, int]
        Number of nontrivial blocks and largest block size.
    """
    tol = float(tol_ev)
    if not (0.0 <= tol <= 1.0e-4):
        raise ValueError(
            "exact QP degeneracy tolerance must be in [0, 1e-4] eV; "
            f"got {tol_ev!r}.")
    energies = np.asarray(energies_kp_ev, dtype=np.float64)
    if energies.ndim != 2 or tuple(np.shape(sigma_pp)) != (
            energies.shape[0], energies.shape[1], energies.shape[1]):
        raise ValueError(
            "exact QP block symmetrisation needs Sigma (nk,P,P) and "
            f"energies (nk,P); got {np.shape(sigma_pp)} and "
            f"{energies.shape}.")
    if tol == 0.0:
        return sigma_pp, (0, 1)

    out = sigma_pp
    n_blocks = 0
    largest = 1
    nk, n_p = energies.shape
    for k in range(nk):
        start = 0
        for stop in range(1, n_p + 1):
            at_edge = stop == n_p
            split = (not at_edge
                     and abs(energies[k, stop] - energies[k, stop - 1])
                     >= tol)
            if not (at_edge or split):
                continue
            size = stop - start
            if size > 1:
                block = out[k, start:stop, start:stop]
                scalar = jnp.trace(block) / size
                out = out.at[k, start:stop, start:stop].set(
                    scalar * jnp.eye(size, dtype=out.dtype))
                n_blocks += 1
                largest = max(largest, size)
            start = stop
    return out, (n_blocks, largest)


def effective_sigma(
    table: SigmaTable,
    classes: BandClasses,
    policy: EvaluationPolicy | str,
    energies_kn_ev,
    efermi_ev: float,
    *,
    exact_degeneracy_tol_ev: float,
):
    """Build the sole fixed-index effective self-energy update.

    This computes the operator added to ``kin_ion``.  The protected block is
    the QSGW half-average, P x U is evaluated once at E_F, and U x U is
    chosen so ``kin_ion + effective_sigma`` has zero off-diagonal entries
    and diagonal ``E_DFT + Delta``.  The closing Hermitisation is part of the
    defining equation, not a numerical repair.

    Equation
    --------
    ``Sigma^QSGW_mn = 1/2 [Sigma_mn(E_m) + Sigma_mn(E_n)]`` on P x P,
    followed by ``(M + M^dagger)/2``.  Within a current-QP block unresolved
    at ``exact_degeneracy_tol_ev``, the protected operator is replaced by
    its unitary-group average ``trace(block)/dim(block) * I``.
    """
    selected = EvaluationPolicy.coerce(policy)
    energy = np.asarray(energies_kn_ev, dtype=np.float64)
    nk = int(np.shape(table.sigma_c_pp_wkij_ry)[1])
    n_p = classes.n_protected
    n_total = classes.n_total
    if energy.shape != (nk, n_total):
        raise ValueError(
            f"effective_sigma needs energies shape {(nk, n_total)}; "
            f"got {energy.shape}.")
    if tuple(np.shape(table.kin_ion_qp_kij_ry)) != (nk, n_total, n_total):
        raise ValueError(
            "BandClasses and SigmaTable represent different band extents.")

    omega_min = float(table.omega_ev[0])
    omega_max = float(table.omega_ev[-1])
    e_rel = energy[:, :n_p] - float(efermi_ev)
    e_clip = np.clip(e_rel, omega_min, omega_max)
    row = table.at(e_clip)
    column = table.at_columns(e_clip)
    correlation = 0.5 * (row + column)
    inside = None
    if selected is EvaluationPolicy.FERMI:
        # Membership is one boolean per BAND over the whole k mesh.  A band
        # flips coherently; a single out-of-domain k sends every pair that
        # touches it to E_F at every k.
        inside = np.all(
            (e_rel >= omega_min) & (e_rel <= omega_max), axis=0)
        at_fermi = table.at(np.zeros_like(e_rel))
        pair_inside = inside[:, None] & inside[None, :]
        correlation = jnp.where(
            jnp.asarray(pair_inside)[None, :, :], correlation, at_fermi)

    sigma_pp = (correlation
                + jnp.asarray(table.sigma_x_pp_kij_ry)
                + jnp.asarray(table.v_h_pp_kij_ry))
    sigma_pp = 0.5 * (sigma_pp + _adjoint(sigma_pp))
    sigma_pp, (n_degenerate_blocks, largest_degenerate_block) = (
        _symmetrize_exact_qp_blocks(
            sigma_pp, energy[:, :n_p], exact_degeneracy_tol_ev))

    # Delta is the mean QP-minus-DFT correction of the two highest
    # protected conduction indices.  ``dft_h_qp`` is the immutable DFT
    # Hamiltonian expressed in the same current QP basis, so this remains a
    # correction even after the orbitals rotate.
    h_pp = (jnp.asarray(table.kin_ion_qp_kij_ry)[:, :n_p, :n_p]
            + sigma_pp)
    dft_pp = jnp.asarray(table.dft_h_qp_kij_ry)[:, :n_p, :n_p]
    correction_diag = jnp.real(jnp.diagonal(
        h_pp - dft_pp, axis1=-2, axis2=-1))
    delta_ry = jnp.mean(correction_diag[:, classes.scissor_bands_local])

    dtype = jnp.result_type(sigma_pp, table.kin_ion_qp_kij_ry)
    sigma_eff = jnp.zeros((nk, n_total, n_total), dtype=dtype)
    sigma_eff = sigma_eff.at[:, :n_p, :n_p].set(sigma_pp)
    if classes.n_outer:
        sigma_pu = (jnp.asarray(table.sigma_xc_pu_fermi_kij_ry)
                    + jnp.asarray(table.v_h_pu_kij_ry))
        sigma_eff = sigma_eff.at[:, :n_p, n_p:].set(sigma_pu)
        sigma_eff = sigma_eff.at[:, n_p:, :n_p].set(_adjoint(sigma_pu))

        # Cancel the current-basis kin+ion U x U block, then install the
        # prescribed diagonal Hamiltonian.  Off-diagonal U x U is therefore
        # exactly zero after QpHamiltonian adds kin+ion.
        kin_uu = jnp.asarray(table.kin_ion_qp_kij_ry)[:, n_p:, n_p:]
        sigma_uu = -kin_uu
        target_diag = (jnp.asarray(table.e_dft_kn_ry)[:, n_p:]
                       + delta_ry)
        sigma_uu = sigma_uu.at[
            :, jnp.arange(classes.n_outer), jnp.arange(classes.n_outer)
        ].add(target_diag)
        sigma_eff = sigma_eff.at[:, n_p:, n_p:].set(sigma_uu)

    sigma_eff = 0.5 * (sigma_eff + _adjoint(sigma_eff))
    diagnostics = {
        "delta_ry": delta_ry,
        "delta_ev": delta_ry * RYD_TO_EV,
        "boundary_mismatch_ev": (
            jnp.max(jnp.abs(correction_diag[:, -1] - delta_ry))
            * RYD_TO_EV),
        "n_outside": int(np.count_nonzero(
            (e_rel < omega_min) | (e_rel > omega_max))),
        "inside_bands": (None if inside is None else tuple(
            bool(value) for value in inside)),
        # Debug consumers can select a small physical probe without
        # gathering the full Sigma cube.  This is the raw row-frequency
        # correlation diagonal before exact-block symmetrisation, which is
        # the quantity that distinguishes a noisy store from the update law.
        "sigma_c_on_shell_diag_ev": (
            jnp.diagonal(row, axis1=-2, axis2=-1) * RYD_TO_EV),
        "protected_energies_ev": energy[:, :n_p],
        "n_degenerate_blocks": n_degenerate_blocks,
        "largest_degenerate_block": largest_degenerate_block,
        "policy": selected.value,
    }
    return sigma_eff, diagnostics


@dataclass(frozen=True)
class QpHamiltonian:
    """Hamiltonian assembly in the current QP basis."""

    kin_ion_qp_kij_ry: object

    def build(self, sigma_effective_kij_ry):
        """Return ``H = kin_ion + effective_sigma``, exactly Hermitian."""
        hamiltonian = (jnp.asarray(self.kin_ion_qp_kij_ry)
                       + jnp.asarray(sigma_effective_kij_ry))
        return 0.5 * (hamiltonian + _adjoint(hamiltonian))


__all__ = [
    "BandClasses",
    "EvaluationPolicy",
    "QpHamiltonian",
    "SigmaTable",
    "effective_sigma",
]
