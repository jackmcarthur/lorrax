"""The q -> 0 head of the MPA self-energy: n_p Godby-Needs heads, summed.

STAGING LOCATION, inherited from the package docstring.

THE IDENTIFICATION, AND WHY IT MAKES THE REUSE EXACT
----------------------------------------------------
The multipole model this tree fits (``gw.mpa.pade_fit``, and the head's
own fit in ``gw.mpa.head_dipole.build_head_axis``) is

    W_c(z) = sum_p [ B_p/(z - Omega_p) - B_p/(z + Omega_p) ]
           = sum_p 2 Omega_p B_p / (z**2 - Omega_p**2)

and the scalar Godby-Needs head the two-point path fits
(``gw.head_correction.fit_head_ppm``) is

    W_c^head(z) = B_h/(z**2 - omega_h**2)
                = R_h [ 1/(z - omega_h) - 1/(z + omega_h) ],
      R_h = B_h/(2 omega_h).

THESE ARE ONE FUNCTIONAL FORM.  An MPA head with n_p poles is n_p
Godby-Needs heads added together, under ``R_h -> B_p`` and
``omega_h -> Omega_p`` -- with the single difference that ``Omega_p`` is
COMPLEX and sits in the closed fourth quadrant, ``Omega_p = a_p - i
Gamma_p``, while ``omega_h`` is real.

That difference costs nothing, because the two-point path's head kernel
already carries an imaginary part in exactly the place the width belongs.
``head_correction.compute_ppm_head_sigma_diag`` evaluates

    occ_term = f       / (delta + omega_h - i eta)
    emp_term = (1 - f) / (delta - omega_h + i eta)
    Sigma    = R_h/(V_cell N_k) * (occ_term + emp_term),
    delta    = omega - (eps - E_F),

and substituting ``omega_h = a_p``, ``eta = Gamma_p`` turns the two
denominators into ``delta + Omega_p`` and ``delta - Omega_p`` EXACTLY --
not to leading order in the width, and not up to a regularisation.  The
retarded regulator of the real-pole case and the physical width of the
complex-pole case are the same number in the same slot, which is why this
module has no kernel of its own: it is a loop over poles around a shipped
function, and the loop is a re-association of a sum that is linear in the
poles.

VERIFIED, NOT ASSERTED.  ``tests/test_mpa_sigma_head.py`` builds the
complex-pole self-energy from the written-out expression above and
compares it against ``compute_ppm_head_sigma_diag`` fed the mapping; the
two agree BIT-IDENTICALLY (``np.array_equal``), and the same file exhibits
the ``Gamma -> 0``, ``n_p = 1`` limit as bit-identical to the Godby-Needs
head the two-point path builds.  A red twin that flips the sign of the
width fails both.

UNITS, AND THE GATE THAT CAN AND CANNOT SEE THEM
------------------------------------------------
``compute_ppm_head_sigma_diag`` takes ``omega_grid_ry`` and ``enk_ry`` in
RYDBERG, so the poles it is handed must be in Rydberg too.  The store does
not say what its head poles are in: ``mpa_store.read_head_poles`` returns
``z``/``w``/``Omega_p``/``B_p`` with no unit record, and the
``mpa_omega_units = "Ha"`` stamp lives on the BODY ``W(omega)`` tensor,
which is a different object written by a different producer.  So the unit
is an ARGUMENT here (:data:`POLE_ENERGY_UNITS`), converted through one
named factor, and never a 2 hard-coded into an expression.

The conversion applies to BOTH pole arrays and by the SAME factor.  From
``W_c(z) = sum_p B_p/(z - Omega_p) - ...`` with ``W_c`` an energy, the
residue carries one power of energy in the numerator -- ``[B_p] = [W_c] *
[z]`` -- so rescaling the frequency axis by ``c`` sends ``Omega_p -> c
Omega_p`` AND ``B_p -> c B_p``, leaving ``W_c`` itself invariant.  Hartree
to Rydberg is ``c = 2``.

WHICH IS WHY THE z = 0 GATE DOES NOT PIN THE UNIT, and saying so is the
point of this paragraph.  ``head_model_at(head, 0)`` is
``-2 sum_p B_p/Omega_p``, a ratio, and the common factor ``c`` cancels
identically -- MEASURED, not argued: the model at z = 0 is bit-identical
under ``c = 1`` and ``c = 2``.  What that gate DOES pin is everything
else, and it is worth running for that: that the stored residues and poles
reproduce the stored samples at all (so the file's two halves belong
together), that ``head_w`` is the CORRELATION part ``W - v`` and not
``W``, and that this module's spelling of the model has the signs the
producer used.  :func:`head_sample_residual` is that check over the whole
sample axis, not only at z = 0.

The gate that IS sensitive to the common factor is the f-sum rule --
``sum_p 2 Omega_p B_p = v_head * omega_p**2`` scales as ``c**2``, so a
Hartree-vs-Rydberg mix-up shows up as a factor of four.  It already exists
(``gw.mpa.head_dipole.fsum_residual``) and the pipeline announces it
rather than this module re-deriving it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "POLE_ENERGY_UNITS",
    "head_model_at",
    "head_poles_in_ry",
    "head_sample_residual",
    "mpa_head_sigma_diag",
    "pole_energy_scale_to_ry",
]


#: The energy units a head pole axis can be stated in, and the factor that
#: takes each to Rydberg.  ONE table, so the conversion is a lookup rather
#: than a literal 2 sitting in an expression where a reader cannot tell
#: which direction it goes.
POLE_ENERGY_UNITS = {"Ry": 1.0, "Ha": 2.0}

#: Below this the residue is not a pole, it is an absent one -- the fit's
#: ``prune_coincident`` / ``prune_out_of_range`` guards zero a residue
#: rather than deleting the slot, so a store legitimately carries B_p = 0
#: entries and they must cost nothing.  ``compute_ppm_head_sigma_diag``
#: makes the same cut at the same value on its own operand.
_RESIDUE_FLOOR = 1.0e-30


def pole_energy_scale_to_ry(unit):
    """The multiplicative factor taking ``unit`` to Rydberg, or refuse.

    Refuses an unknown spelling rather than defaulting, because a default
    here is a silent factor of two on the whole q -> 0 head -- roughly
    200 meV on silicon, finite, smooth and with no other symptom.
    """
    try:
        return float(POLE_ENERGY_UNITS[str(unit)])
    except KeyError:
        raise ValueError(
            f"mpa sigma head: pole_energy_unit={unit!r} is not one of "
            f"{sorted(POLE_ENERGY_UNITS)}.  The store's __mpahead group "
            f"carries no unit stamp, so the caller owns this and there is "
            f"no default to fall back to: guessing wrong scales every pole "
            f"and every residue by two and moves the whole q -> 0 head by "
            f"a factor that no shape check and no finiteness check can "
            f"see.") from None


def _poles(head):
    """``(Omega_p, B_p)`` as complex128 1-D arrays, validated.

    The two structural facts the tau stage and the kernel below both rely
    on are checked HERE, once, rather than at the two places they would
    otherwise bite: ``Im Omega_p <= 0`` (a leaked upper-half pole enters
    the width slot as a NEGATIVE eta, which flips the retarded kernel onto
    the advanced branch) and ``Re Omega_p > 0`` for every pole carrying
    weight (the occupied/empty split below puts the pole at
    ``delta = -a_p`` for occupied states and ``+a_p`` for empty ones, so a
    negative ``a_p`` silently swaps which side of the Fermi level each
    term belongs to).  ``mpa_store.write_head_axis`` refuses the first at
    write time; nothing refuses the second, and nothing re-checks either
    on the way back out.
    """
    Om = np.asarray(head["Omega_p"], dtype=np.complex128).reshape(-1)
    Bp = np.asarray(head["B_p"], dtype=np.complex128).reshape(-1)
    if Om.shape != Bp.shape:
        raise ValueError(
            f"mpa sigma head: the store carries {Om.size} head poles and "
            f"{Bp.size} residues.  One residue per pole -- a mismatch means "
            f"the two arrays were written by different fits.")
    if Om.size == 0:
        raise ValueError(
            "mpa sigma head: the store's head axis carries no poles.  An "
            "empty pole set integrates to exactly zero, which is "
            "indistinguishable from a head that was never injected -- the "
            "failure read_head_poles exists to refuse.")
    if np.any(np.imag(Om) > 0.0):
        bad = int(np.count_nonzero(np.imag(Om) > 0.0))
        raise ValueError(
            f"mpa sigma head: {bad} of {Om.size} head poles have "
            f"Im Omega_p > 0.  This module hands -Im Omega_p to the shipped "
            f"head kernel as its retarded regulator eta, so a positive "
            f"imaginary part becomes a NEGATIVE eta and evaluates the "
            f"advanced self-energy under the retarded one's name.  The "
            f"body fit's guards and mpa_store.write_head_axis both put "
            f"poles in the closed fourth quadrant; this store did not "
            f"come through them.")
    live = np.abs(Bp) > _RESIDUE_FLOOR
    if np.any(live & (np.real(Om) <= 0.0)):
        bad = int(np.count_nonzero(live & (np.real(Om) <= 0.0)))
        raise ValueError(
            f"mpa sigma head: {bad} head poles carry weight at "
            f"Re Omega_p <= 0.  The occupied term of the head kernel puts "
            f"its pole at delta = -Re Omega_p and the empty term at "
            f"+Re Omega_p, so a non-positive real part swaps which side of "
            f"the Fermi level each term describes and returns a finite, "
            f"smooth, wrong Sigma.  A pruned pole (B_p = 0) is exempt "
            f"because it contributes nothing.")
    return Om, Bp


def head_model_at(head, z, *, pole_energy_unit="Ry", out_unit=None):
    """``sum_p [ B_p/(z - Omega_p) - B_p/(z + Omega_p) ]`` at ``z``.

    The model evaluated in ONE place so the z = 0 gate is a call and not
    an inline re-derivation of an expression whose sign is the thing being
    checked.  Broadcasts over an array ``z``; returns a scalar for a
    scalar.

    ``pole_energy_unit`` names the unit ``Omega_p`` and ``B_p`` are in and
    ``out_unit`` the unit ``z`` is in; ``out_unit=None`` (the default)
    means "z is in the poles' own unit", which is what the store's own
    ``head_z`` axis is and therefore what the gate wants.  ``W_c`` itself
    is invariant under the conversion (the residue's power of energy
    cancels the frequency's), so the RETURN is in whatever energy unit the
    producer's ``head_w`` samples were -- Rydberg, for
    ``gw.mpa.head_dipole``.
    """
    Om, Bp = _poles(head)
    if out_unit is not None:
        c = pole_energy_scale_to_ry(out_unit) / pole_energy_scale_to_ry(
            pole_energy_unit)
        Om, Bp = Om * c, Bp * c
    zz = np.asarray(z, dtype=np.complex128)
    val = np.sum(Bp / (zz[..., None] - Om) - Bp / (zz[..., None] + Om),
                 axis=-1)
    return complex(val) if np.ndim(z) == 0 else val


def head_sample_residual(head, *, pole_energy_unit="Ry"):
    """Does the stored pole model reproduce the stored samples?

    THE CHECK THAT CAN FAIL, and the one the driver seam runs before it
    builds anything.  ``mpa_store``'s head axis keeps the ``2*n_p``
    samples ``(z_i, W_c(z_i))`` BESIDE the ``n_p`` poles fitted to them,
    for exactly this: the file carries its own evidence, so a consumer can
    ask whether the poles it is about to sum are the poles that fit those
    samples instead of trusting that they are.

    WHAT IT PINS: that the two halves of the head axis belong to one fit;
    that ``head_w`` is the CORRELATION part ``W - v`` and not ``W`` (a
    stored ``W`` misses by ``v_head``, ~3300 Ry on silicon, so this fails
    by three orders of magnitude rather than subtly); and that this
    module's spelling of the model -- both signs, and the ``2 Omega B``
    versus ``B/(z-Omega) - B/(z+Omega)`` factor of two -- matches the
    producer's.

    WHAT IT DOES NOT PIN: the energy unit.  At ``z = 0`` the model is
    ``-2 sum_p B_p/Omega_p``, in which a common rescaling of the pole axis
    cancels identically, and at every other sample the stored ``z`` is
    rescaled along with the poles, so the whole residual is blind to the
    factor.  See the module docstring for the f-sum rule, which is not.

    Returns a dict: ``z0_residual`` (the ``z = 0`` sample, the one the
    insulator protocol places exactly at the origin), ``max_residual`` and
    ``max_rel_residual`` over the whole axis, ``sample_scale`` (the
    ``max |W_c|`` the relative numbers are taken against), ``n_samples``,
    and ``z0_index`` (``None`` when the grid carries no exact zero, which
    is the metals protocol's ``z = i*1e-5`` origin shift and not an
    error).
    """
    z = np.asarray(head["z"], dtype=np.complex128).reshape(-1)
    w = np.asarray(head["w"], dtype=np.complex128).reshape(-1)
    if z.shape != w.shape:
        raise ValueError(
            f"mpa sigma head: the head axis carries {z.size} sample points "
            f"and {w.size} sample values.  They are one array of pairs; a "
            f"mismatch means the store's evidence cannot be matched to its "
            f"abscissae.")
    model = np.asarray(head_model_at(head, z,
                                     pole_energy_unit=pole_energy_unit))
    resid = np.abs(model - w)
    # NORMALISED BY THE AXIS SCALE, NOT PER SAMPLE.  ``W_c`` legitimately
    # passes near zero on the far line, so dividing each residual by its
    # own sample would turn a rounding error at a small sample into a
    # large relative number and drown the one that matters.  The fit is
    # against the whole axis at once, so the whole axis sets the scale.
    scale = max(float(np.max(np.abs(w))), np.finfo(np.float64).tiny)
    zeros = np.flatnonzero(z == 0.0)
    i0 = int(zeros[0]) if zeros.size else None
    return {
        "n_samples": int(z.size),
        "z0_index": i0,
        "z0_residual": (float(resid[i0]) if i0 is not None else None),
        "z0_rel_residual": (float(resid[i0] / scale)
                            if i0 is not None else None),
        "z0_sample": (complex(w[i0]) if i0 is not None else None),
        "z0_model": (complex(model[i0]) if i0 is not None else None),
        "max_residual": float(np.max(resid)),
        "max_rel_residual": float(np.max(resid) / scale),
        "sample_scale": scale,
    }


def head_poles_in_ry(head, *, pole_energy_unit="Ry"):
    """``(Omega_p, B_p)`` in Rydberg -- one conversion, one place.

    Both arrays take the SAME factor; see the module docstring for why the
    residue converts at all.
    """
    Om, Bp = _poles(head)
    c = pole_energy_scale_to_ry(pole_energy_unit)
    return Om * c, Bp * c


def mpa_head_sigma_diag(
    head,
    *,
    omega_grid_ry,
    enk_ry,
    efermi_ry,
    n_occ,
    cell_volume,
    nk_tot,
    pole_energy_unit="Ry",
    eta_floor=0.0,
):
    """Band-diagonal q -> 0, G = G' = 0 head of the MPA ``Sigma_c(omega)``.

    ``(n_omega, nk, nb)`` complex128 in Rydberg -- the same representation
    and the same units the two-point path's
    :func:`gw.head_correction.compute_ppm_head_sigma_diag` returns, because
    it IS that function, called once per head pole and summed.  See the
    module docstring for the identification and for why the reuse is exact
    rather than approximate.

    Parameters
    ----------
    head
        The dict :func:`file_io.mpa_store.read_head_poles` returns.
    omega_grid_ry
        ``(n_omega,)`` Sigma frequency grid relative to E_F, in Ry.
    enk_ry
        ``(nk, nb)`` absolute band energies for the Sigma window, in Ry.
    efermi_ry, n_occ, cell_volume, nk_tot
        As in :func:`gw.head_correction.compute_ppm_head_sigma_diag`.
    pole_energy_unit
        The unit the store's ``Omega_p``/``B_p`` are stated in; see
        :data:`POLE_ENERGY_UNITS`.  The store does not record it.
    eta_floor
        A LOWER BOUND on the retarded regulator, in Ry, default ``0.0`` --
        i.e. off, so ``eta = Gamma_p`` exactly and the reuse stays
        bit-exact.  A head pole with a genuinely zero width would put a
        pole on the real axis of the integrand; that is a property of the
        fit worth seeing rather than smoothing, which is why the floor is
        not on by default.
    """
    from gw.head_correction import HeadGNParams, compute_ppm_head_sigma_diag

    omega = np.asarray(omega_grid_ry, dtype=np.float64).reshape(-1)
    enk = np.asarray(enk_ry, dtype=np.float64)
    if enk.ndim != 2:
        raise ValueError("enk_ry must be 2D (nk, nb)")
    Om, Bp = head_poles_in_ry(head, pole_energy_unit=pole_energy_unit)

    out = np.zeros((int(omega.size), int(enk.shape[0]), int(enk.shape[1])),
                   dtype=np.complex128)
    for p in range(int(Om.size)):
        a_p = float(np.real(Om[p]))
        gamma_p = float(-np.imag(Om[p]))
        # ``HeadGNParams`` IS THE CARRIER, not a coincidence of field
        # names.  ``R_h`` is the residue of the ``1/(z - pole)`` form and
        # ``omega_h`` its position -- exactly what ``B_p`` and ``a_p``
        # are.  It holds a complex ``R_h`` here, which the two-point fit
        # never produces and the dataclass does not forbid; the kernel
        # multiplies by it and nothing else, so the arithmetic is
        # unchanged.  The other fields are diagnostics of the two-point
        # FIT and have no meaning for a pole read off a store, so they are
        # filled with the values that make them read as absent rather than
        # with plausible-looking numbers.
        pole = HeadGNParams(
            omega_h_sq=a_p * a_p,
            omega_h=a_p,
            B_h=complex(2.0 * a_p * Bp[p]),
            R_h=complex(Bp[p]),
            wc_head_0=0.0,
            wc_head_iwp=0.0,
            vc0=0.0,
            omega_p=a_p,
        )
        out += compute_ppm_head_sigma_diag(
            pole,
            omega_grid_ry=omega,
            enk_ry=enk,
            efermi_ry=efermi_ry,
            n_occ=n_occ,
            cell_volume=cell_volume,
            nk_tot=nk_tot,
            eta=max(gamma_p, float(eta_floor)),
        )
    return out
