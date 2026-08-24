"""Helpers for the q=0, G=G'=0 Coulomb head.

The modern GWJAX paths keep the head separate from the ISDF body tensors:

- Dynamic GN-PPM uses scalar head samples ``(v_h, W_h(0), W_h(iω_p))``.
- Static COHSEX uses exact band-diagonal head shifts for ``Σ^X``, ``Σ^SX``,
  ``Σ^(SX-X)``, and ``Σ^COH``.
- Downstream BSE / Σ-builders that already consume ``V_qmunu``/``W_qmunu``
  can absorb the head as a rank-1 update at ``q=0`` via
  ``apply_q0_head_rank1`` (see below).

This module centralizes:

- head source resolution (`override`, `epshead`, `s_tensor`)
- scalar GN-PPM head fitting
- exact static COHSEX head terms
- bounded small-field/body Schur folding on the 2-D mesh
- rank-1 (μ,ν)-basis head injection at q=0
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
import enum
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map
from common.units import RYD_TO_EV


def _analytic_q0_sphere(params) -> bool:
    """One compatibility-preserving resolver for the q=0 v estimator."""
    return (
        bool(params.get("head_minibz_average", False))
        or str(params.get("bgw_metal_q0_treatment", "exact")).strip().lower()
        == "bgw_q0shift"
    )


class HeadResponseKind(str, enum.Enum):
    """Reduction state of the response that produced a scalar head.

    The distinction is operational, not documentation: ``DIRECT_IRREDUCIBLE``
    still needs the head/body Schur complement for a physical macroscopic W,
    whereas ``MICRO_REDUCIBLE`` already contains that local-field resummation
    and must never be folded again.
    """

    DIRECT_IRREDUCIBLE = "direct_irreducible"
    FULL_LOCAL_FIELDS = "full_local_fields"
    MICRO_REDUCIBLE = "micro_reducible"
    OVERRIDE = "override"
    OFF = "off"


@dataclass(frozen=True)
class HeadSample:
    """Resolved q=0 Coulomb head sample at one frequency."""

    vc0: complex
    wcoul0: complex
    source: str
    omega: complex
    #: The Cartesian q²-coefficient tensor ``S(ω)`` that PRODUCED ``wcoul0``,
    #: when there was one — ``(3, 3)`` complex, the convention of
    #: ``docs/theory/s-tensor-convention.md``.  ``None`` on the ``epshead``
    #: branch, which has no tensor (it fits an isotropic γ instead).
    #:
    #: Carried because ``wcoul0`` alone is the cell average on ONE grid, and
    #: the coarse→fine W densifier (``gw.head_densify``) needs the INTEGRAND
    #: to re-attach the head per fine q.  Persisted to the restart by
    #: ``file_io.write_head_scalars_to_h5`` so the BSE need not rebuild it.
    S_cart: np.ndarray | None = None
    response_kind: HeadResponseKind = HeadResponseKind.DIRECT_IRREDUCIBLE


@dataclass(frozen=True)
class HeadGNParams:
    """Fitted GN parameters for the scalar Coulomb head."""

    omega_h_sq: float
    omega_h: float
    B_h: float
    R_h: float
    wc_head_0: float
    wc_head_iwp: float
    vc0: float
    omega_p: float


@dataclass(frozen=True)
class StaticHeadTerms:
    """Exact static q=0 head terms for bare X / SX / COHSEX.

    All values are diagonal-in-band shifts in Rydberg atomic units.
    The head contributes equally at every k-point, with the Brillouin-zone
    average carried by the explicit ``1 / N_k`` factor.
    """

    sigma_x_diag: jnp.ndarray
    sigma_sx_diag: jnp.ndarray
    sigma_sx_minus_x_diag: jnp.ndarray
    sigma_coh_diag: jnp.ndarray
    vc0: complex
    wcoul0: complex
    wc_head_0: complex
    source: str


@dataclass(frozen=True)
class BGWQ0Channel:
    """One finite grid-q head channel used as BGW's epsilon q0 sample."""

    requested_full_index: int
    representative_full_index: int
    wedge_row: int
    q0_reduced: tuple[float, float, float]
    g_head: object                 # (1, n_mu), sharded on the centroid axis
    v_bare: float


def resolve_bgw_q0_channel(
    config, sym, q_wedge_full_indices, head_channel, *, kgrid,
):
    """Bind the deck's reduced q0 vector to one stored W-wedge row.

    The shifted point must be exactly on the WFN grid.  Its irreducible
    representative may point in another symmetry-equivalent direction; the
    scalar epsilon-inverse head is invariant under that operation, and the
    head-channel vector is therefore taken from the representative row that
    the Dyson solve actually stores.
    """
    if not bool(config.head.uses_bgw_metal_q0shift):
        return None
    if head_channel is None:
        raise ValueError(
            "bgw_metal_q0_treatment=bgw_q0shift requires the finite-q "
            "Coulomb head channel, but GW initialization did not build it.")
    kgrid = np.asarray(kgrid, dtype=np.int64)
    if kgrid.shape != (3,) or np.any(kgrid <= 0):
        raise ValueError(
            "BGW q0 resolution requires a positive 3-D kgrid; "
            f"got {kgrid}.")
    q0 = np.asarray(config.head.bgw_metal_q0_vector, dtype=np.float64)
    steps_float = q0 * kgrid
    steps = np.rint(steps_float).astype(np.int64)
    if not np.allclose(steps_float, steps, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            "bgw_metal_q0_vector must lie on the deck's reciprocal grid: "
            f"q0={tuple(q0)}, kgrid={tuple(int(n) for n in kgrid)}, "
            f"q0*kgrid={tuple(float(x) for x in steps_float)}.")
    full_steps = np.asarray(sym.kvecs_asints, dtype=np.int64)
    target_mod = np.mod(steps, kgrid)
    matches = np.flatnonzero(np.all(
        np.mod(full_steps, kgrid[None, :]) == target_mod[None, :], axis=1))
    if matches.size != 1:
        raise ValueError(
            "bgw_metal_q0_vector did not identify exactly one full-grid q: "
            f"q0={tuple(q0)}, matches={matches.tolist()}.")
    requested = int(matches[0])
    wedge_row = int(np.asarray(sym.irr_idx_q, dtype=np.int64)[requested])
    q_idx = np.asarray(q_wedge_full_indices, dtype=np.int64)
    if not 0 <= wedge_row < q_idx.size:
        raise ValueError(
            "BGW q0 irreducible row is outside the stored W wedge: "
            f"row={wedge_row}, wedge size={q_idx.size}.")
    representative = int(q_idx[wedge_row])
    multiplicity = int(np.asarray(head_channel.mult)[representative])
    if multiplicity != 1:
        raise ValueError(
            "bgw_metal_q0_vector must select a unique G=0-like head slot; "
            f"its irreducible representative has multiplicity {multiplicity}.")
    v_bare = float(np.asarray(head_channel.v_bare)[representative])
    if not np.isfinite(v_bare) or v_bare <= 0.0:
        raise ValueError(
            "bgw_metal_q0_vector selected no finite bare Coulomb head "
            f"(representative full-q row {representative}, v={v_bare}).")
    return BGWQ0Channel(
        requested_full_index=requested,
        representative_full_index=representative,
        wedge_row=wedge_row,
        q0_reduced=tuple(float(x) for x in q0),
        g_head=head_channel.g_head[representative, 0:1, :],
        v_bare=v_bare,
    )


@functools.partial(jax.jit, static_argnames=("mesh_xy",))
def finite_q0_epsinv_head(
    chi_q0,
    W_q0,
    g_head_q0,
    v_bare,
    chi_prefactor,
    *,
    mesh_xy: Mesh,
):
    r"""Return the full finite-q ``epsilon^{-1}_{00}``, including wings.

    In the centroid representation the selected plane-wave channel is
    ``V_0 = v_0 |conj(g)><g|``.  With the already solved
    ``W=(1-V chi)^{-1}V``, the exact bordered-Dyson scalar is

    ``epsinv_00 = 1 + v_0 <g| chi (1 + W chi) |conj(g)>``.

    Thus the regular finite-q W tile supplies the head, both wings, and the
    body Schur fold without forming a plane-wave epsilon matrix.  Every
    ``(mu,nu)`` object remains two-dimensionally sharded; only two vectors
    and the final scalar are resharded/reduced.
    """
    chi = jnp.asarray(chi_q0) * jnp.asarray(
        chi_prefactor, dtype=jnp.asarray(chi_q0).dtype)
    W = jnp.asarray(W_q0)
    g = jnp.asarray(g_head_q0)
    sh_x = NamedSharding(mesh_xy, P(None, "x"))
    sh_y = NamedSharding(mesh_xy, P(None, "y"))
    sh_q = NamedSharding(mesh_xy, P(None))
    u_x = jax.lax.with_sharding_constraint(jnp.conj(g), sh_x)
    u_y = jax.lax.with_sharding_constraint(jnp.conj(g), sh_y)
    r_x = jax.lax.with_sharding_constraint(g, sh_x)
    chi_u_x = jnp.einsum("qmn,qn->qm", chi, u_y, optimize=True)
    chi_u_y = jax.lax.with_sharding_constraint(chi_u_x, sh_y)
    W_chi_u_x = jnp.einsum("qmn,qn->qm", W, chi_u_y, optimize=True)
    l_x = u_x + W_chi_u_x
    l_y = jax.lax.with_sharding_constraint(l_x, sh_y)
    chi_l_x = jnp.einsum("qmn,qn->qm", chi, l_y, optimize=True)
    response = jnp.einsum("qm,qm->q", r_x, chi_l_x, optimize=True)
    epsinv = 1.0 + jnp.asarray(v_bare, dtype=response.dtype) * response
    return jax.lax.with_sharding_constraint(epsinv, sh_q)


def bgw_q0shift_vhead(wfn, meta):
    """BGW analytic-sphere mini-BZ bare head, in LORRAX's raw units."""
    from gw.vcoul import compute_q0_averages

    vc0, _ = compute_q0_averages(
        wfn,
        jnp.asarray(1.0, dtype=jnp.float64),
        meta,
        S_cart=None,
        analytic_sphere=True,
    )
    return complex(vc0)


def bgw_q0shift_head_sample(vc0, epsinv, omega) -> HeadSample:
    """Compose BGW's finite-q epsilon head with its q=0 v-cell average."""
    eps = complex(epsinv)
    v = complex(vc0)
    return HeadSample(
        vc0=v,
        wcoul0=eps * v,
        source="bgw_q0shift(analytic-sphere v; finite-q0 epsinv head+wings)",
        omega=complex(omega),
        S_cart=None,
        response_kind=HeadResponseKind.FULL_LOCAL_FIELDS,
    )


def _representative_entry(diag: jnp.ndarray) -> complex:
    """Return a representative diagonal value for diagnostics."""

    arr = np.asarray(diag).reshape(-1)
    if arr.size == 0:
        return 0.0 + 0.0j
    nz = np.flatnonzero(np.abs(arr) > 0.0)
    idx = int(nz[0]) if nz.size else 0
    return complex(arr[idx])


def resolve_head_override(params, omega) -> HeadSample | None:
    """Return explicit head overrides when both v and W are provided."""

    omega_val = complex(omega)
    vhead_override = params.get("vhead")
    w_key = "whead_0freq" if abs(omega_val) <= 1.0e-14 else "whead_imfreq"
    whead_override = params.get(w_key)
    if vhead_override is None or whead_override is None:
        return None
    source = "override" if abs(omega_val) <= 1.0e-14 else f"override(omega={omega_val} Ry)"
    return HeadSample(
        vc0=complex(vhead_override),
        wcoul0=complex(whead_override),
        source=source,
        omega=omega_val,
        response_kind=HeadResponseKind.OVERRIDE,
    )


def _check_dipole_coverage(
    dipole_path, *, nb_file, nk_file, nk_run, nb_run, nelec, print_fn,
):
    """Loud coverage check on ``dipole.h5`` at the point of use.

    ``dipole.h5`` is generated once by ``psp.get_dipole_mtxels`` at
    whatever ``nbands`` the generating run happened to use, and it is
    *not* namespaced by that count.  The head ``S(ω)`` built from it sums
    over ``arange(nelec, nb_file)`` conduction states — so a file written
    at 120 bands feeding a run whose Σ window spans 160 silently
    truncates the transition space in ``wcoul0``, and therefore in every
    q→0 Σ_SX / Σ_COH correction.  That exact mismatch shipped in the
    2026-07 production runs and was found by hand, not by the code.

    The file stamps ``nbands`` / ``nk`` as HDF5 attrs; nothing read them.
    This warns rather than raises: a short dipole file is a *convergence*
    defect, not a corrupt one, and refusing would break every existing
    run directory.  It is loud enough to see.
    """
    from common import sanity

    if not sanity.sanity_enabled():
        return
    attrs_nb, attrs_nk = None, None
    try:
        import h5py
        with h5py.File(dipole_path, "r") as h5:
            if "nbands" in h5.attrs:
                attrs_nb = int(np.asarray(h5.attrs["nbands"]))
            if "nk" in h5.attrs:
                attrs_nk = int(np.asarray(h5.attrs["nk"]))
    except (OSError, KeyError, ValueError) as exc:
        print_fn(f"  [dipole guard] could not read attrs from {dipole_path} "
                 f"({type(exc).__name__}: {exc})")
    if attrs_nk is not None and attrs_nk != int(nk_run):
        sanity.warn(
            f"{dipole_path} was generated on nk={attrs_nk} but this run has "
            f"nk_tot={int(nk_run)}.  The head S(ω) would be assembled from a "
            f"different k-sampling than Σ — refusing to trust it is the only "
            f"safe reading of this file.",
            print_fn=print_fn)
    n_cond_file = max(0, int(nb_file) - int(nelec))
    print_fn(
        f"  dipole.h5 coverage: {int(nb_file)} bands on disk "
        f"({int(nelec)} occ + {n_cond_file} cond)"
        + (f", run Σ window = {int(nb_run)} bands" if nb_run else ""))
    if nb_run and int(nb_file) < int(nb_run):
        sanity.warn(
            f"{dipole_path} carries only {int(nb_file)} bands but this run's "
            f"Σ window spans {int(nb_run)}.  The q→0 head S(ω) sums "
            f"conduction states arange({int(nelec)}, {int(nb_file)}) — "
            f"{n_cond_file} of them — so wcoul0, and every Σ_SX/Σ_COH head "
            f"correction built from it, is converged to a SMALLER transition "
            f"space than Σ itself.  This does not crash and does not change "
            f"the exit code; it makes the head systematically wrong.  "
            f"Regenerate dipole.h5 with nbands >= {int(nb_run)} "
            f"(psp.get_dipole_mtxels) to close it.",
            print_fn=print_fn)


def _dipole_window_from_params(params, wfn) -> tuple[int, int, int]:
    """``(nval, ncond, nband)`` — the RUN's resolved band window, or a refusal.

    THE THREE NUMBERS ARE READ, NEVER INVENTED.  This helper used to default
    ``nval``/``ncond`` to 5 and ``nband`` to ``max(wfn.nbands, nelec+ncond)``,
    on the stated grounds that it "mirrors the writer".  The writer resolves
    those defaults against the DECK; this side only ever saw
    ``config.head`` — a six-key dict with no band window in it — so the
    defaults were not a mirror, they were the only thing the comparison ever
    used.  Measured on the MoS2 production deck (JID 57269074): a dipole.h5
    generated from the very same WFN and deck reported
    ``file=26/26/600`` against an invented ``run=5/5/610``, and under
    ``LORRAX_SANITY=strict`` that false warning is an unconditional refusal
    of a correct file.

    So an ABSENT field is a refusal, not a guess.  A provenance check whose
    reference is fabricated cannot fail for the reason it claims and cannot
    pass for one either — it is the class of check
    ``TASTE.md``/"a check that cannot fail is not evidence" is about.  The
    one supported caller (:class:`HeadResolver`) carries the resolved
    ``config.nval``/``config.ncond``/``config.nband``; a direct caller must
    do the same.

    ``wfn`` is retained for the refusal message only — it is what makes the
    "which numbers were missing, and what would they have been" line
    actionable — and is deliberately NOT consulted for a value.
    """
    missing = [k for k in ("nval", "ncond", "nband")
               if params.get(k) is None]
    if missing:
        raise ValueError(
            f"dipole provenance: the run's band window is missing "
            f"{', '.join(missing)} — the checker was handed a params dict "
            f"that carries no band window (got keys "
            f"{sorted(str(k) for k in params)}).  These three numbers are "
            f"the reference the file's prov_nval/prov_ncond/prov_nband "
            f"stamps are compared against; defaulting them to 5/5/"
            f"max(wfn.nbands={int(getattr(wfn, 'nbands', 0) or 0)}, ...) "
            f"made the comparison invent its own reference and accuse every "
            f"correctly-stamped file whose deck is not 5/5.  Pass the "
            f"resolved config window (gw.head_correction.HeadResolver does "
            f"this from config.nval/config.ncond/config.nband).")
    return (int(params["nval"]), int(params["ncond"]), int(params["nband"]))


def _check_dipole_provenance(dipole_path, *, params, wfn, print_fn) -> None:
    """Was ``dipole.h5`` built from THIS DFT solution and THIS band window?

    The coverage check above answers "is the file big enough"; this answers
    "is it the right file at all".  They are different failures and neither
    implies the other: a dipole.h5 regenerated from a *different* WFN has
    exactly the right shape, so every shape-based check passes and nothing
    downstream notices that the q→0 head S(ω) — and therefore every
    Σ_SX/Σ_COH head correction — is built from stale velocity matrix
    elements.

    ``psp.get_dipole_mtxels`` has stamped ``prov_*`` attrs (WFN sha256 plus
    the band window) since the guard landed, and shipped
    ``check_dipole_provenance`` to read them back.  Nothing called it; the
    writer and the checker both existed and the consumer did neither.

    Reports through ``common.sanity`` — loud by default, a refusal under
    ``LORRAX_SANITY=strict`` — and is gated on ``sanity_enabled()`` like
    its sibling.  An UNSTAMPED file (written before the guard) reports as
    unverifiable and does not fail the run.

    A caller that supplies no band window is refused outright by
    :func:`_dipole_window_from_params` (a code defect, not a deck error):
    an invented reference makes this check accuse correct files and vouch
    for nothing.
    """
    from common import sanity

    if not sanity.sanity_enabled():
        return
    nval, ncond, nband = _dipole_window_from_params(params, wfn)
    try:
        from psp.get_dipole_mtxels import check_dipole_provenance
    except Exception as exc:            # psp stack unavailable (h5py-less env)
        print_fn(f"  [dipole provenance] check unavailable "
                 f"({type(exc).__name__}: {exc})")
        return
    check_dipole_provenance(dipole_path, wfn=wfn, nval=nval, ncond=ncond,
                            nband=nband, print_fn=print_fn)


def resolve_head_sample(params, input_dir, wfn, sym, meta, print_fn, omega) -> HeadSample:
    """Resolve a q=0 head sample using overrides and configured source order."""

    override = resolve_head_override(params, omega)
    if override is not None:
        return override

    # MIXED-HEAD ANNOUNCEMENT.  ``resolve_head_override`` keys on ONE deck
    # scalar per frequency: ``whead_0freq`` at ω=0 and ``whead_imfreq`` at
    # the PPM probe (the only non-zero ω the driver ever asks for).  A deck
    # that sets ``vhead``/``whead_0freq`` and omits ``whead_imfreq``
    # therefore overrides one end of the plasmon-pole fit and computes the
    # other, which is a legitimate thing to want — it pins the static head
    # to a reference code and lets the dispersion come from here — but it
    # is not what "head override" reads like, and the per-sample
    # diagnostics say "Head source: override" at ω=0 without ever naming
    # the other end.  Printed once per run: the resolver memoizes per
    # frequency.
    if (abs(complex(omega)) > 1.0e-14
            and params.get("vhead") is not None
            and params.get("whead_0freq") is not None
            and params.get("whead_imfreq") is None):
        print_fn(
            f"  [head] MIXED HEAD: 'vhead'/'whead_0freq' override W(q→0) at "
            f"omega=0, but 'whead_imfreq' is unset, so W(q→0) at the PPM "
            f"probe omega={complex(omega)} Ry is COMPUTED from "
            f"wcoul0_source="
            f"{str(params.get('wcoul0_source', 's_tensor')).strip().lower()}. "
            f"The two-point pole fit therefore mixes an overridden head "
            f"with a computed one; set 'whead_imfreq' to override both."
        )

    want_source = str(params.get("wcoul0_source", "s_tensor")).strip().lower()
    if want_source not in ("epshead", "s_tensor"):
        raise ValueError(
            f"wcoul0_source={want_source!r} is invalid; expected 'epshead' "
            "or 's_tensor'. The head source never falls back silently.")

    omega_val = complex(omega)
    eta = float(params.get("wcoul0_eta", 0.0) or 0.0)
    eps0_path = os.path.join(input_dir, "eps0mat.h5")
    dipole_path = os.path.join(input_dir, "dipole.h5")

    def from_epshead() -> HeadSample:
        if not os.path.exists(eps0_path):
            raise FileNotFoundError(
                f"wcoul0_source=epshead requested {eps0_path}, but it does "
                "not exist; refusing to substitute s_tensor.")
        if abs(omega_val) > 1.0e-14:
            print_fn(
                f"wcoul0_source=epshead is static-only; using epshead(0) for omega={omega_val} Ry"
            )
        from file_io.epsreader import EPSReader
        from gw.vcoul import compute_q0_averages

        # ``with``: EPSReader's matrix attributes are h5py dataset HANDLES
        # since 2026-08-22 (it used to slurp the whole ``mats/matrix`` into
        # host memory on every rank, for the six numbers this branch wants),
        # so the file's lifetime is now the caller's to state.  ``epshead``
        # is read in the constructor, so nothing here outlives the block.
        with EPSReader(eps0_path) as eps0:
            head = jnp.asarray(eps0.epshead, dtype=jnp.complex128)
        vc0_mean, wcoul0 = compute_q0_averages(
            wfn,
            head,
            meta,
            S_cart=None,
            analytic_sphere=_analytic_q0_sphere(params),
        )
        source = "epshead(0)" if abs(omega_val) > 1.0e-14 else "epshead"
        return HeadSample(
            vc0=complex(vc0_mean),
            wcoul0=complex(wcoul0),
            source=source,
            omega=omega_val,
        )

    def from_s_tensor() -> HeadSample:
        if not os.path.exists(dipole_path):
            raise FileNotFoundError(
                f"wcoul0_source=s_tensor requested {dipole_path}, but it "
                "does not exist; refusing to substitute epshead.")
        from gw.vcoul import compute_q0_averages

        from common import timing as _tmg
        S_cart_omega = build_S_cart_omega(
            wfn, sym, meta, params, dipole_path, omega_val, eta=eta,
            print_fn=print_fn)
        with _tmg.section("head.q0_avg"):
            vc0_mean, wcoul0 = compute_q0_averages(
                wfn,
                jnp.asarray(0.0, dtype=jnp.float64),
                meta,
                S_cart=S_cart_omega,
                analytic_sphere=_analytic_q0_sphere(params),
            )
        source = "s_tensor" if abs(omega_val) <= 1.0e-14 else f"s_tensor(omega={omega_val} Ry)"
        return HeadSample(
            vc0=complex(vc0_mean),
            wcoul0=complex(wcoul0),
            source=source,
            omega=omega_val,
            S_cart=np.asarray(S_cart_omega, dtype=np.complex128),
        )

    return from_epshead() if want_source == "epshead" else from_s_tensor()


def build_S_cart_omega(wfn, sym, meta, params, dipole_path, omega,
                       *, eta: float = 0.0, print_fn=print) -> np.ndarray:
    """``S(ω)``, the Cartesian q²-coefficient tensor, from ``dipole.h5``.

    THE ONE SPELLING of the dipole → ``S(ω)`` build.  It has two consumers and
    they must not drift: ``resolve_head_sample``'s ``s_tensor`` branch (the GW
    run, which then averages it into ``wcoul0``) and
    :func:`resolve_head_S_cart` (the BSE, which needs the integrand itself to
    re-attach W's head per fine q under ``gw.head_densify``).  A second copy
    would be a tensor that agrees with the run's head everywhere except where
    it matters.

    Units and convention are ``docs/theory/s-tensor-convention.md``: Cartesian,
    the canonical form, ``1/(Ry·bohr²)`` such that ``v(q)·qᵀSq`` is
    dimensionless.

    Parameters
    ----------
    wfn, sym, meta
        The run's loader / symmetry table / system parameters.
    params : dict
        Deck keys; read for the dipole provenance check only.
    dipole_path : str
        Absolute path to ``dipole.h5``.
    omega : complex
        Frequency in Ry.  0 for the static head this stage consumes.
    eta : float
        Broadening in Ry (deck ``wcoul0_eta``).  Non-zero makes ``S`` complex.

    Returns
    -------
    numpy.ndarray, shape (3, 3), complex128
    """
    from common.chi_from_dipole import read_dipole_h5, compute_S_omega
    from common import timing as _tmg

    with _tmg.section("head.read_dipole"):
        dipole_cart, deltaE = read_dipole_h5(dipole_path)
    nk_tot = int(sym.nk_tot)
    nb = int(dipole_cart.shape[2])
    nelec = int(wfn.nelec)
    with _tmg.section("head.checks"):
        _check_dipole_coverage(
            dipole_path, nb_file=nb, nk_file=int(dipole_cart.shape[1]),
            nk_run=nk_tot, nb_run=int(getattr(meta, "nb_sigma", 0) or 0),
            nelec=nelec, print_fn=print_fn)
        _check_dipole_provenance(dipole_path, params=params or {}, wfn=wfn,
                                 print_fn=print_fn)
    # TODO(metal-head): this legacy one-shot dipole path retains the ifmax
    # step occupation; metallic QSGW uses the explicit weighted PT head.
    occ = np.zeros((nk_tot, nb), dtype=float)
    occ[:, :max(0, min(nelec, nb))] = 1.0
    f_nk = jnp.asarray(occ, dtype=jnp.float64)
    omega_grid = jnp.asarray([complex(omega)], dtype=jnp.complex128)
    with _tmg.section("head.S_omega"):
        S_cart_omega = compute_S_omega(
            dipole_cart, deltaE, f_nk, float(wfn.cell_volume), int(sym.nk_tot),
            int(wfn.nspin), int(wfn.nspinor), omega_grid, eta=float(eta),
        )[0]
    return np.asarray(S_cart_omega, dtype=np.complex128)


@functools.partial(jax.jit, static_argnames=("mesh_xy",))
def fold_small_head_wings_sharded(
    R_direct: jax.Array,
    Y_x: jax.Array,
    W_body_xy: jax.Array,
    Z_y: jax.Array,
    Vcell: float,
    *,
    mesh_xy: Mesh,
) -> jax.Array:
    r"""Fold a bounded small-field response through the screened body.

    This is the single production owner of the small head/body Schur fold.
    It accepts independently sized left and right field bases; both field
    extents are replicated and therefore must remain bounded.  The body is
    never gathered: every rank contracts its local ``(I_x, J_y)`` tile and
    only the small output is reduced across the two-dimensional mesh.

    .. math::

        R_{AB}^{\mathrm{eff}}(z) = R_{AB}^{0}(z)
          + \frac{1}{V_{\mathrm{cell}}}
            \sum_{IJ}Y_{AI}(z)W_{IJ}(z)Z_{JB}(z).

    Any replicated batch/frequency axes may precede the displayed axes and
    must match exactly (no broadcasting).  Body axes remain tiled exactly
    like screening: ``Y`` on ``x``, ``W`` on ``(x,y)``, and ``Z`` on ``y``.
    The caller supplies those shardings; this kernel deliberately does not
    defensively reshard large inputs.

    Parameters
    ----------
    R_direct
        Direct response, ``(..., F_left, F_right)``, replicated.  Its units
        are set by the caller's field basis.
    Y_x
        Left wing, ``(..., F_left, n_I)``, body axis sharded on ``x``.
    W_body_xy
        Screened body, ``(..., n_I, n_J)``, sharded on ``(x,y)``.
    Z_y
        Right wing, ``(..., n_J, F_right)``, body axis sharded on ``y``.
        Wing/body units must make ``Y W Z / Vcell`` match ``R_direct``.
    Vcell
        Primitive-cell volume in bohr³; it appears exactly once.
    mesh_xy
        Production two-dimensional device mesh.

    Returns
    -------
    jax.Array
        ``R_eff`` with shape ``(..., F_left, F_right)``, the same units as
        ``R_direct``, replicated on ``mesh_xy``.
    """
    n_lead = W_body_xy.ndim - 2
    arrays = (R_direct, Y_x, W_body_xy, Z_y)
    if n_lead < 0 or any(a.ndim != n_lead + 2 for a in arrays):
        raise ValueError(
            "R_direct, Y, W_body, and Z must have two non-leading axes")
    leading = tuple(W_body_xy.shape[:-2])
    if any(tuple(a.shape[:-2]) != leading for a in arrays):
        raise ValueError(
            "R_direct, Y, W_body, and Z must share their leading axes")
    if (
        int(R_direct.shape[-2]) != int(Y_x.shape[-2])
        or int(Y_x.shape[-1]) != int(W_body_xy.shape[-2])
        or int(W_body_xy.shape[-1]) != int(Z_y.shape[-2])
        or int(Z_y.shape[-1]) != int(R_direct.shape[-1])
    ):
        raise ValueError(
            "small-field/body extents do not compose: "
            f"R={R_direct.shape}, Y={Y_x.shape}, "
            f"W={W_body_xy.shape}, Z={Z_y.shape}")
    lead = [None] * n_lead

    def _local_fold(s_direct, y_local, w_local, z_local, volume):
        # Each rank owns exactly one (mu_X, nu_Y) body tile.  Contract that
        # tile before communicating, then reduce only the small-field result.
        # Leaving this as a global einsum lets XLA select a full-matrix
        # temporary at large body extent even though the public field axes
        # are bounded.
        local = jnp.einsum(
            "...am,...mn,...nb->...ab",
            y_local, w_local, z_local, optimize=True)
        correction = jax.lax.psum(local, ("x", "y"))
        return s_direct + correction / volume

    folded = shard_map(
        _local_fold,
        mesh=mesh_xy,
        in_specs=(
            P(*lead, None, None),
            P(*lead, None, "x"),
            P(*lead, "x", "y"),
            P(*lead, "y", None),
            P(),
        ),
        out_specs=P(*lead, None, None),
    )(
        R_direct,
        Y_x,
        W_body_xy,
        Z_y,
        jnp.asarray(Vcell),
    )
    return folded


def fold_cartesian_head_wings_sharded(
    S_direct: jax.Array,
    Y_x: jax.Array,
    W_body_xy: jax.Array,
    Z_y: jax.Array,
    cell_volume: float,
    *,
    mesh_xy: Mesh,
) -> jax.Array:
    """Charge-head adapter to :func:`fold_small_head_wings_sharded`.

    ``S_direct`` and the result have shape ``(..., 3, 3)`` and units
    ``1/(Ry·bohr²)``; the centroid/body axes retain their existing
    ``x``/``(x,y)``/``y`` shardings.
    """
    return fold_small_head_wings_sharded(
        S_direct, Y_x, W_body_xy, Z_y, cell_volume, mesh_xy=mesh_xy)


@functools.partial(jax.jit, static_argnames=("mesh_xy",))
def _small_head_wing_halves_sharded(Y_x, W_body_xy, Z_y, *, mesh_xy):
    """Compiled body of :func:`small_head_wing_halves_sharded`."""
    def _local(y_local, w_local, z_local):
        yw = jax.lax.psum(
            jnp.einsum("aAi,ij->aAj", y_local, w_local, optimize=True),
            "x")
        wz = jax.lax.psum(
            jnp.einsum("ij,bjB->biB", w_local, z_local, optimize=True),
            "y")
        return yw, wz

    return shard_map(
        _local,
        mesh=mesh_xy,
        in_specs=(P(None, None, "x"), P("x", "y"),
                  P(None, "y", None)),
        out_specs=(P(None, None, "y"), P(None, "x", None)),
        check_vma=False,
    )(Y_x, W_body_xy, Z_y)


def small_head_wing_halves_sharded(
    Y_x: jax.Array,
    W_body_xy: jax.Array,
    Z_y: jax.Array,
    *,
    mesh_xy: Mesh,
) -> tuple[jax.Array, jax.Array]:
    r"""Contract each small photon wing through one resident body ``W``.

    For the two in-plane directions and four Lorentz fields this returns

    ``YW[a,A,J] = sum_I Y[a,A,I] W[I,J]`` and
    ``WZ[b,I,B] = sum_J W[I,J] Z[b,J,B]``.

    Only the contracted centroid axis is reduced.  The outputs remain
    respectively y- and x-sharded one-index objects; the body is neither
    gathered nor transposed.  No conjugation, cell-volume factor, or head
    model is implicit.
    """
    if Y_x.ndim != 3 or W_body_xy.ndim != 2 or Z_y.ndim != 3:
        raise ValueError(
            "small photon-head halves require Y=(2,4,N), W=(N,N), "
            f"Z=(2,N,4); got {Y_x.shape}, {W_body_xy.shape}, {Z_y.shape}")
    n_body = int(W_body_xy.shape[0])
    expected = ((2, 4, n_body), (n_body, n_body), (2, n_body, 4))
    if (tuple(Y_x.shape), tuple(W_body_xy.shape), tuple(Z_y.shape)) != expected:
        raise ValueError(
            "small photon-head half-contraction extents do not compose: "
            f"Y={Y_x.shape}, W={W_body_xy.shape}, Z={Z_y.shape}")

    required = (
        (Y_x, "Y_x", P(None, None, "x")),
        (W_body_xy, "W_body_xy", P("x", "y")),
        (Z_y, "Z_y", P(None, "y", None)),
    )
    for array, name, spec in required:
        wanted = NamedSharding(mesh_xy, spec)
        sharding = getattr(array, "sharding", None)
        if (sharding is None
                or not sharding.is_equivalent_to(wanted, array.ndim)):
            raise ValueError(
                f"{name} must already have sharding {spec}; got {sharding}. "
                "Refusing an implicit photon-body reshard.")
    return _small_head_wing_halves_sharded(
        Y_x, W_body_xy, Z_y, mesh_xy=mesh_xy)


@jax.jit
def _static_slab_photon_head_moment_chunk(
    q_cart: jax.Array,
    D_raw: jax.Array,
    R_linear: jax.Array,
    R_quadratic: jax.Array,
    valid_count: jax.Array,
):
    r"""Accumulate one fixed-size chunk of the coupled small-head solve.

    ``R(q) = q_a R_linear[a] + q_a q_b R_quadratic[a,b]`` uses the two
    periodic in-plane Cartesian coordinates of a slab.  For every valid
    mini-BZ sample this evaluates the *coupled* four-field Dyson equation

    ``W_h(q) = [I - D(q) R(q)]^-1 D(q)``

    before averaging.  The returned ``(1,qx,qy)`` moments are sufficient to
    rebuild the head, both single wings, and the double-wing body update as
    repeated rank-four outer products; no sample-by-centroid array exists.

    This is the sole sample-sized graph.  The vcoul provider zero-pads its
    final chunk to the same fixed size and passes ``valid_count``, preventing
    a tail-shape recompile and keeping the invalid q=0 rows outside every
    accumulated quantity.
    """
    q = jnp.asarray(q_cart, dtype=jnp.float64)
    D = jnp.asarray(D_raw, dtype=jnp.complex128)
    H = jnp.asarray(R_linear, dtype=jnp.complex128)
    Q = jnp.asarray(R_quadratic, dtype=jnp.complex128)
    qxy = q[:, :2]
    R = (
        jnp.einsum("sa,aij->sij", qxy, H, optimize=True)
        + jnp.einsum("sa,sb,abij->sij", qxy, qxy, Q, optimize=True)
    )
    identity = jnp.eye(4, dtype=jnp.complex128)[None, :, :]
    lhs = identity - jnp.einsum("sik,skj->sij", D, R, optimize=True)
    W_head = jnp.linalg.solve(lhs, D)

    valid = jnp.arange(q.shape[0], dtype=jnp.int32) < valid_count
    weight = valid.astype(jnp.float64)
    basis = jnp.concatenate(
        (jnp.ones((q.shape[0], 1), dtype=jnp.float64), qxy), axis=1)
    moments = jnp.einsum(
        "s,su,sij,sv->uvij", weight, basis, W_head, basis,
        optimize=True)
    D_sum = jnp.einsum("s,sij->ij", weight, D, optimize=True)

    residual = jnp.einsum(
        "sik,skj->sij", lhs, W_head, optimize=True) - D
    residual_norm = jnp.linalg.norm(residual, axis=(-2, -1))
    D_norm = jnp.linalg.norm(D, axis=(-2, -1))
    relative = residual_norm / jnp.maximum(
        D_norm, jnp.asarray(1.0e-300, dtype=jnp.float64))
    max_relative = jnp.max(jnp.where(valid, relative, 0.0))
    return moments, D_sum, valid_count, max_relative


def static_slab_photon_head_moment_chunk(
    q_cart,
    D_raw,
    R_linear,
    R_quadratic,
    valid_count,
):
    """Validated entry to the fixed-size static slab photon-head graph.

    Parameters follow :func:`_static_slab_photon_head_moment_chunk`:
    ``q_cart`` is ``(chunk,3)``, ``D_raw`` is ``(chunk,4,4)`` in raw vcoul
    units (no cell-volume factor), and the response coefficients are
    ``(2,4,4)`` and ``(2,2,4,4)``.  The caller averages these sums per
    independent mini-BZ repetition, then across repetitions; it applies the
    one and only ``1/Vcell`` while rebuilding the packed q=Gamma row.

    The function is intentionally slab/static-only.  A bulk analytic-sphere
    correction cannot be added after this nonlinear coupled solve, and must
    have its own derived integrator before that policy is admitted.
    """
    q_shape = tuple(np.shape(q_cart))
    d_shape = tuple(np.shape(D_raw))
    h_shape = tuple(np.shape(R_linear))
    Q_shape = tuple(np.shape(R_quadratic))
    if len(q_shape) != 2 or q_shape[1] != 3:
        raise ValueError(f"q_cart must be (chunk,3), got {q_shape}")
    expected_D = (q_shape[0], 4, 4)
    if d_shape != expected_D:
        raise ValueError(f"D_raw must be {expected_D}, got {d_shape}")
    if h_shape != (2, 4, 4) or Q_shape != (2, 2, 4, 4):
        raise ValueError(
            "static slab response coefficients must be R_linear=(2,4,4) "
            f"and R_quadratic=(2,2,4,4); got {h_shape}/{Q_shape}")
    n_valid = int(valid_count)
    if not 0 <= n_valid <= q_shape[0]:
        raise ValueError(
            f"valid_count must lie in [0,{q_shape[0]}], got {n_valid}")
    return _static_slab_photon_head_moment_chunk(
        q_cart, D_raw, R_linear, R_quadratic,
        jnp.asarray(n_valid, dtype=jnp.int32))


@dataclass(frozen=True)
class StaticSlabPhotonHeadCompletion:
    """Small evidence record for one coupled slab q=0 reconstruction."""

    bare_D_mean: np.ndarray
    screened_moments: np.ndarray
    samples_per_replicate: tuple[int, ...]
    max_dyson_relative_residual: float
    estimator: str = "vcoul_minibz_equal_replicate_mean_v1"


_STATIC_PHOTON_DYSON_RESIDUAL_MAX = 1.0e-8


def complete_static_slab_photon_q0(
    V_packed: jax.Array,
    W_packed: jax.Array,
    coefficients,
    g0_X: jax.Array,
    g0_Y: jax.Array,
    photon_sample_chunks,
    *,
    cell_volume: float,
    mesh_xy: Mesh,
) -> tuple[jax.Array, jax.Array, StaticSlabPhotonHeadCompletion]:
    r"""Complete bare and screened packed photon operators in the Γ cell.

    ``photon_sample_chunks`` is the sole vcoul provider's streamed slab
    output.  Each sample first solves the coupled four-field head Dyson
    equation; only its ``(1,qx,qy)`` moments survive.  The packed body is
    then updated by one bare and nine screened rank-four outer products.
    No sample-by-centroid tensor or second photon packing convention exists.
    """
    from .photon_layout import (
        MAX_Q0_UPDATE_RANK, add_photon_q0_low_rank)

    layout = coefficients.layout
    layout.assert_mesh(mesh_xy)
    packed_shape = (int(V_packed.shape[0]), layout.packed_extent,
                    layout.packed_extent)
    if (tuple(V_packed.shape) != packed_shape
            or tuple(W_packed.shape) != packed_shape):
        raise ValueError(
            "coupled photon head requires equal packed V/W bodies; got "
            f"V={V_packed.shape}, W={W_packed.shape}, expected={packed_shape}")
    if tuple(coefficients.H_direct.shape) != (2, 4, 4):
        raise ValueError(
            f"photon H_direct must be (2,4,4); got "
            f"{coefficients.H_direct.shape}")
    if tuple(coefficients.Q_direct.shape) != (2, 2, 4, 4):
        raise ValueError(
            f"photon Q_direct must be (2,2,4,4); got "
            f"{coefficients.Q_direct.shape}")
    factor_shape = (MAX_Q0_UPDATE_RANK, layout.packed_extent)
    if tuple(g0_X.shape) != factor_shape or tuple(g0_Y.shape) != factor_shape:
        raise ValueError(
            f"packed Γ vectors must both be {factor_shape}; got "
            f"{g0_X.shape}/{g0_Y.shape}")
    if not np.isfinite(float(cell_volume)) or float(cell_volume) <= 0.0:
        raise ValueError(f"cell_volume must be positive; got {cell_volume}")

    # The headless Gamma body remains resident and 2-D sharded.  Four calls
    # reuse the sole bounded Schur-fold graph; broadcasting W over the two
    # coordinate axes would create four body views in one executable.
    W_gamma = W_packed[0]
    Q_effective = coefficients.Q_direct
    for a in range(2):
        for b in range(2):
            folded = fold_small_head_wings_sharded(
                coefficients.Q_direct[a, b],
                coefficients.Y_x[a], W_gamma, coefficients.Z_y[b],
                float(cell_volume), mesh_xy=mesh_xy)
            Q_effective = Q_effective.at[a, b].set(folded)
    YW_y, WZ_x = small_head_wing_halves_sharded(
        coefficients.Y_x, W_gamma, coefficients.Z_y, mesh_xy=mesh_xy)
    jax.block_until_ready((Q_effective, YW_y, WZ_x))

    # Finish one Sobol replicate before starting the next.  This preserves
    # the provider's equal-replicate estimator rather than weighting a short
    # or padded tail as another draw.
    per_rep_moments = []
    per_rep_D = []
    samples_per_rep = []
    current_rep = None
    expected_start = 0
    moment_sum = D_sum = residual_max = None
    count_sum = 0

    def _finish_replicate():
        if current_rep is None:
            return
        if count_sum <= 0:
            raise ValueError(
                f"photon mini-BZ replicate {current_rep} has no samples")
        moment_host, D_host, residual_host = jax.device_get(
            (moment_sum, D_sum, residual_max))
        per_rep_moments.append(
            np.asarray(moment_host, dtype=np.complex128) / float(count_sum))
        per_rep_D.append(
            np.asarray(D_host, dtype=np.complex128) / float(count_sum))
        samples_per_rep.append(int(count_sum))
        return float(np.asarray(residual_host))

    residual_per_rep = []
    for item in photon_sample_chunks:
        if len(item) != 8:
            raise ValueError(
                "vcoul photon sample chunks must have eight fields")
        rep, start, stop, q_cart, D_raw, valid_count, mc_weight, analytic_D = item
        rep, start, stop = int(rep), int(start), int(stop)
        n_valid = int(valid_count)
        if current_rep is None or rep != current_rep:
            if current_rep is not None:
                value = _finish_replicate()
                residual_per_rep.append(value)
                if rep != current_rep + 1:
                    raise ValueError(
                        "vcoul photon replicate indices must be contiguous; "
                        f"got {current_rep} then {rep}")
            current_rep = rep
            expected_start = 0
            moment_sum = D_sum = residual_max = None
            count_sum = 0
        if start != expected_start or stop - start != n_valid:
            raise ValueError(
                "vcoul photon chunks must be contiguous and valid_count must "
                f"equal stop-start; rep={rep}, start/expected={start}/"
                f"{expected_start}, stop={stop}, valid={n_valid}")
        expected_start = stop

        q_host = np.asarray(q_cart, dtype=np.float64)
        weight = np.asarray(mc_weight, dtype=np.float64)
        analytic = np.asarray(analytic_D)
        if (np.any(q_host[:n_valid, 2] != 0.0)
                or np.any(weight[:n_valid] != 1.0)
                or np.any(weight[n_valid:] != 0.0)
                or np.any(analytic != 0.0)):
            raise ValueError(
                "coupled photon q0 completion currently accepts only the "
                "slab estimator (qz=0, unit valid weights, no 3-D analytic "
                "sphere addend)")
        moments, bare_sum, returned_count, residual = (
            static_slab_photon_head_moment_chunk(
                q_host, D_raw, coefficients.H_direct, Q_effective, n_valid))
        if int(np.asarray(returned_count)) != n_valid:
            raise RuntimeError(
                "static photon head moment kernel changed valid_count")
        moment_sum = moments if moment_sum is None else moment_sum + moments
        D_sum = bare_sum if D_sum is None else D_sum + bare_sum
        residual_max = (residual if residual_max is None
                        else jnp.maximum(residual_max, residual))
        count_sum += n_valid

    if current_rep is None:
        raise ValueError("vcoul photon sample provider yielded no chunks")
    residual_per_rep.append(_finish_replicate())
    D_mean = np.mean(np.stack(per_rep_D, axis=0), axis=0)
    moments_mean = np.mean(np.stack(per_rep_moments, axis=0), axis=0)
    max_residual = float(max(residual_per_rep))
    if (not np.all(np.isfinite(D_mean))
            or not np.all(np.isfinite(moments_mean))
            or not np.isfinite(max_residual)):
        raise ValueError("coupled photon mini-BZ average is non-finite")
    if max_residual > _STATIC_PHOTON_DYSON_RESIDUAL_MAX:
        raise ValueError(
            "coupled photon mini-BZ Dyson solve failed its relative-residual "
            f"gate: {max_residual:.3e} > "
            f"{_STATIC_PHOTON_DYSON_RESIDUAL_MAX:.1e}")

    dtype = V_packed.dtype
    sh_x = NamedSharding(mesh_xy, P(None, "x"))
    sh_y = NamedSharding(mesh_xy, P(None, "y"))
    volume = jnp.asarray(float(cell_volume), dtype=jnp.float64)
    with mesh_xy:
        left_bare = jax.lax.with_sharding_constraint(
            jnp.conj(g0_X).astype(dtype), sh_x)
        right_bare = jax.lax.with_sharding_constraint(
            jnp.einsum(
                "AB,Bj->Aj", jnp.asarray(D_mean, dtype=dtype), g0_Y,
                optimize=True) / volume,
            sh_y)
    V_packed = add_photon_q0_low_rank(
        V_packed, layout, mesh_xy,
        left_rows_X=left_bare, right_rows_Y=right_bare)

    left_basis = (
        left_bare,
        jax.lax.with_sharding_constraint(jnp.swapaxes(WZ_x[0], 0, 1), sh_x),
        jax.lax.with_sharding_constraint(jnp.swapaxes(WZ_x[1], 0, 1), sh_x),
    )
    right_basis = (g0_Y, YW_y[0], YW_y[1])
    for u in range(3):
        for v in range(3):
            with mesh_xy:
                right_rows = jax.lax.with_sharding_constraint(
                    jnp.einsum(
                        "AB,Bj->Aj",
                        jnp.asarray(moments_mean[u, v], dtype=dtype),
                        right_basis[v], optimize=True) / volume,
                    sh_y)
            W_packed = add_photon_q0_low_rank(
                W_packed, layout, mesh_xy,
                left_rows_X=left_basis[u], right_rows_Y=right_rows)

    evidence = StaticSlabPhotonHeadCompletion(
        bare_D_mean=D_mean,
        screened_moments=moments_mean,
        samples_per_replicate=tuple(samples_per_rep),
        max_dyson_relative_residual=max_residual,
    )
    return V_packed, W_packed, evidence


def resolve_head_S_cart(restart_file=None, *, input_file=None, wfn=None,
                        sym=None, meta=None, params=None, print_fn=print):
    """The ``S`` tensor behind the restart's ``whead`` — read it, or rebuild it.

    ``whead`` alone is the head CELL AVERAGE on one grid.  A coarse→fine
    densification needs the INTEGRAND that average was taken of, so it can be
    re-evaluated on a different cell and pointwise inside the old one — that
    integrand is ``v/(1 − v qᵀS q)`` and this returns its ``S``.

    Two routes, in order, because the first is exact and free and the second
    exists for restarts written before the first one did:

    1. **The restart's own ``S_cart_head``** — written beside ``vhead`` /
       ``whead`` by :func:`file_io.write_head_scalars_to_h5` since this change.
       This is the tensor that PRODUCED that ``whead``, so the provenance ratio
       is 1 by construction and nothing has to be recomputed.
    2. **Rebuilt from ``dipole.h5``** through :func:`build_S_cart_omega`, the
       same call the GW run made.  Needs ``wfn``/``sym``/``meta`` (the BSE
       coarse→fine paths already load all three for the htransform leg) and a
       ``dipole.h5`` beside the deck.  The rebuild is deterministic, so the
       provenance ratio it produces is a real check on whether the head in the
       restart and this tensor describe the same screening.

    Returns
    -------
    tuple[numpy.ndarray | None, str]
        ``(S_cart, provenance)``.  ``S_cart`` is ``(3, 3)`` complex128 or
        ``None`` when neither route is available; ``provenance`` names which
        route ran, or why none did, and is meant to be logged verbatim.
    """
    if restart_file is not None:
        try:
            import h5py
            with h5py.File(restart_file, "r") as f:
                if "S_cart_head" in f:
                    S = np.asarray(f["S_cart_head"][:], dtype=np.complex128)
                    if S.shape == (3, 3):
                        return S, f"restart S_cart_head ({os.path.basename(restart_file)})"
                    print_fn(f"BSE head: restart S_cart_head has shape "
                             f"{S.shape}, expected (3,3); ignoring it")
        except Exception as exc:                    # never crash a load on this
            print_fn(f"BSE head: could not read S_cart_head ({exc})")

    if wfn is None or sym is None or meta is None or input_file is None:
        return None, ("no S_cart: the restart carries none and the caller "
                      "passed no wfn/sym/meta to rebuild one from dipole.h5")
    dipole_path = os.path.join(os.path.dirname(os.path.abspath(input_file)),
                               "dipole.h5")
    if not os.path.exists(dipole_path):
        return None, f"no S_cart: restart carries none and {dipole_path} is absent"
    try:
        S = build_S_cart_omega(wfn, sym, meta, params or {}, dipole_path, 0.0,
                               eta=float((params or {}).get("wcoul0_eta", 0.0) or 0.0),
                               print_fn=print_fn)
    except Exception as exc:
        return None, f"no S_cart: rebuild from dipole.h5 failed ({exc})"
    return S, "rebuilt from dipole.h5 (S(ω=0), the GW run's own route)"


class HeadResolver:
    """Memoized q=0 head-sample resolver for a single GW run.

    The driver needs the head sample at up to two frequencies (ω=0 always,
    and a second probe ω for the dynamic PPM path).  Building it requires
    reading ``eps0mat.h5`` or ``dipole.h5`` and crunching a Voronoi-cell
    integral, which is non-trivial; without memoization the same work was
    being done three times per run.

    Construct once at the top of ``main()``::

        head = HeadResolver(config, input_dir, wfn, sym, meta, print_fn)
        head_static = head.at(0.0 + 0.0j)
        head_probe  = head.at(probe_omega)
    """

    __slots__ = ("_params", "_input_dir", "_wfn", "_sym", "_meta",
                 "_print_fn", "_cache", "_direct_cache", "_policy",
                 "_screened")

    def __init__(self, config, input_dir, wfn, sym, meta, print_fn):
        head = config.head
        self._params = {
            # These are GW-run controls, not head sub-config fields.  The
            # dipole provenance reader must compare against the consumer's
            # requested window rather than reconstructing the writer defaults.
            "nval": config.nval,
            "ncond": config.ncond,
            "nband": config.nband,
            "wcoul0_source": head.wcoul0_source,
            "wcoul0_eta": head.wcoul0_eta,
            "vhead": head.vhead,
            "whead_0freq": head.whead_0freq,
            "whead_imfreq": head.whead_imfreq,
            "head_minibz_average": head.head_minibz_average,
            "bgw_metal_q0_treatment": head.bgw_metal_q0_treatment,
        }
        from gw.gw_config import coerce_head_correction
        self._policy = coerce_head_correction(
            getattr(head, "correction", "full"))
        self._screened = bool(getattr(config, "do_screened", True))
        self._input_dir = input_dir
        self._wfn = wfn
        self._sym = sym
        self._meta = meta
        self._print_fn = print_fn
        self._cache: dict[tuple[float, float], HeadSample] = {}
        self._direct_cache: dict[tuple[float, float], HeadSample] = {}

    def _cache_key(self, omega) -> tuple[float, float]:
        z = complex(omega)
        return (round(z.real, 12), round(z.imag, 12))

    @property
    def wfn(self):
        """The WFN whose transition manifold this provider validates."""
        return self._wfn

    def direct_at(self, omega) -> HeadSample:
        """Return the configured no-local-field/direct diagnostic sample."""
        key = self._cache_key(omega)
        cached = self._direct_cache.get(key)
        if cached is not None:
            return cached
        sample = resolve_head_sample(
            self._params, self._input_dir, self._wfn, self._sym,
            self._meta, self._print_fn, omega=omega,
        )
        self._direct_cache[key] = sample
        return sample

    def install_samples(self, samples) -> None:
        """Install finalized samples, rejecting incomplete or double folds."""
        from gw.gw_config import HeadCorrection

        for sample in tuple(samples):
            if not isinstance(sample, HeadSample):
                raise TypeError("head provider accepts HeadSample objects only")
            if self._policy is HeadCorrection.OFF:
                raise ValueError(
                    "head_correction=off cannot install a screened head")
            if (self._policy is HeadCorrection.FULL
                    and sample.response_kind
                    not in (HeadResponseKind.FULL_LOCAL_FIELDS,
                            HeadResponseKind.MICRO_REDUCIBLE,
                            HeadResponseKind.OVERRIDE)):
                raise ValueError(
                    "head_correction=full requires a once-folded "
                    "full_local_fields response or an already "
                    "micro_reducible response; got "
                    f"{sample.response_kind.value}. Refusing the unfolded "
                    "epsilon head.")
            self._cache[self._cache_key(sample.omega)] = sample

    def at(self, omega) -> HeadSample:
        """Resolve the policy-selected finalized head at ``omega`` in Ry."""
        from gw.gw_config import HeadCorrection

        key = self._cache_key(omega)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self._policy is HeadCorrection.OFF:
            sample = HeadSample(
                vc0=0.0j, wcoul0=0.0j, source="head_correction=off",
                omega=complex(omega), S_cart=None,
                response_kind=HeadResponseKind.OFF)
            self._cache[key] = sample
            return sample
        direct = self.direct_at(omega)
        if direct.response_kind is HeadResponseKind.OVERRIDE:
            self._cache[key] = direct
            return direct
        if (self._policy is HeadCorrection.NO_LOCAL_FIELDS
                or not self._screened):
            self._cache[key] = direct
            return direct
        raise RuntimeError(
            "head_correction=full was requested, but no finalized head was "
            f"installed at omega={complex(omega)} Ry. Build compatible "
            "wings/body (direct RPA) or a micro-reducible BSE resolvent; "
            "the direct epsilon head is not a silent fallback.")


def format_head_sample_diagnostics(head: HeadSample, *, include_screened: bool = True) -> str:
    """Return a compact diagnostic summary for one resolved head sample."""

    lines = [
        "",
        "-" * 72,
        "  FINITE-SIZE CORRECTIONS",
        "-" * 72,
        f"  Head source: {head.source}",
        f"  v(q→0)  = {head.vc0.real:12.3f} a.u.  (bare Coulomb head)",
    ]
    if include_screened:
        if abs(head.omega) > 1.0e-14:
            lines.append(f"  Head frequency ω = {head.omega} Ry")
        lines.append(f"  W(q→0)  = {head.wcoul0.real:12.3f} a.u.  (screened Coulomb head)")
        lines.append(
            f"  ΔW      = {(head.wcoul0.real - head.vc0.real):12.3f} a.u.  (screening correction)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dynamic GN-PPM scalar head
# ---------------------------------------------------------------------------

def fit_head_ppm(
    vc0: float,
    wcoul0_static: float,
    wcoul0_probe: float,
    probe_omega: complex,
) -> HeadGNParams:
    """Fit a scalar PPM pole from two W^c head samples.

    Model-agnostic two-point fit: the same algebra serves both the
    Godby-Needs PPM (purely imaginary probe ``probe_omega = i·ωp``) and
    the Hybertsen-Louie PPM (real probe ``probe_omega = Ω`` above all
    transitions).  The signed quantity ``z² = (probe_omega)²`` carries
    the model choice — negative for GN, positive for HL.
    """

    z = complex(probe_omega)
    omega_2_sq = float((z * z).real)
    # Real-axis log-magnitude for diagnostics ("ω_p" was historically the
    # imaginary-axis magnitude; for HL it's the real frequency itself).
    probe_mag = float(abs(z))

    w1 = wcoul0_static - vc0
    w2 = wcoul0_probe - vc0

    denom = w1 - w2
    if abs(denom) < 1.0e-30:
        return HeadGNParams(
            omega_h_sq=1.0,
            omega_h=1.0,
            B_h=0.0,
            R_h=0.0,
            wc_head_0=w1,
            wc_head_iwp=w2,
            vc0=vc0,
            omega_p=probe_mag,
        )

    omega_h_sq = -w2 * omega_2_sq / denom
    B_h = -w1 * omega_h_sq

    if omega_h_sq <= 0.0:
        omega_h = abs(omega_h_sq) ** 0.5 if omega_h_sq != 0.0 else 1.0
        # Bug fix (2026-07-04): B_h at L318 used the SIGNED (negative) omega_h_sq
        # while omega_h = abs(...)^0.5 is positive, so R_h = B_h/(2 omega_h) came
        # out sign-flipped vs the positive branch — the whole q->0 head Sigma_c
        # (hundreds of meV) had the wrong sign whenever the GN head fit went
        # imaginary.  Use the magnitude so |R_h| is continuous across Omega^2=0.
        B_h = -w1 * abs(omega_h_sq)
        R_h = B_h / (2.0 * omega_h) if omega_h > 1.0e-30 else 0.0
        return HeadGNParams(
            omega_h_sq=omega_h_sq,
            omega_h=omega_h,
            B_h=B_h,
            R_h=R_h,
            wc_head_0=w1,
            wc_head_iwp=w2,
            vc0=vc0,
            omega_p=probe_mag,
        )

    omega_h = omega_h_sq ** 0.5
    R_h = B_h / (2.0 * omega_h)
    return HeadGNParams(
        omega_h_sq=omega_h_sq,
        omega_h=omega_h,
        B_h=B_h,
        R_h=R_h,
        wc_head_0=w1,
        wc_head_iwp=w2,
        vc0=vc0,
        omega_p=probe_mag,
    )


def fit_head_ppm_from_samples(
    head_static: HeadSample,
    head_probe: HeadSample,
    *,
    probe_omega: complex,
) -> HeadGNParams:
    """Fit the scalar PPM head from resolved static and probe-frequency samples."""
    return fit_head_ppm(
        vc0=float(head_static.vc0.real),
        wcoul0_static=float(head_static.wcoul0.real),
        wcoul0_probe=float(head_probe.wcoul0.real),
        probe_omega=probe_omega,
    )


def fit_head_hl_analytic(
    vc0: float,
    wcoul0_static: float,
    omega_p_sq_ry: float,
) -> HeadGNParams:
    """Set the HL-PPM head pole analytically from the bulk plasmon, BGW-style.

    The 2-point HL fit at finite probe Ω asymptotes to the f-sum-rule
    value as Ω → ∞, but at finite Ω the static-vs-probe head W^c samples
    can be sensitive to numerical convention (mini-BZ averaging, head
    truncation), giving an Ω_h that drifts ~10–20 % from the exact
    bulk-plasmon limit.  BGW sidesteps this by taking the head pole
    directly from the analytic f-sum-rule: ``Ω̃²(0,0) = ω_p²`` (set in
    ``Sigma/wpeff.f90`` as the q=g=g'=0 special case), and the kernel
    pole ``wtilde² = Ω² / I_ε(0,0) = ω_p² / (1 − ε⁻¹(0,0))``.

    This mirrors that: ``Ω_h² = ω_p² / I_ε_head`` where
    ``I_ε_head = (v_head − W(0)) / v_head`` is computed from the same
    mini-BZ-averaged static head ``W(0)`` LORRAX already resolves.
    The static W^c(0) head is still used (for B_h and R_h via the GN/HL
    pole ansatz), so the magnitude of the head correction stays
    consistent with the COHSEX block.
    """
    w1 = wcoul0_static - vc0  # W^c(0) head, in a.u.
    if abs(w1) < 1.0e-30 or abs(vc0) < 1.0e-30:
        return HeadGNParams(
            omega_h_sq=1.0, omega_h=1.0, B_h=0.0, R_h=0.0,
            wc_head_0=w1, wc_head_iwp=0.0, vc0=vc0,
            omega_p=float(omega_p_sq_ry) ** 0.5 if omega_p_sq_ry > 0 else 1.0,
        )

    # I_ε_head = 1 − ε⁻¹(0,0) = 1 − W(0)/v(0) = (v − W)/v = −W^c/v.
    I_eps_head = -w1 / float(vc0)
    if I_eps_head <= 0.0:
        I_eps_head = 1.0  # graceful fallback; prevents sqrt of negative

    omega_h_sq = float(omega_p_sq_ry) / I_eps_head
    omega_h = omega_h_sq ** 0.5
    B_h = -w1 * omega_h_sq
    R_h = B_h / (2.0 * omega_h)
    return HeadGNParams(
        omega_h_sq=omega_h_sq,
        omega_h=omega_h,
        B_h=B_h,
        R_h=R_h,
        wc_head_0=w1,
        wc_head_iwp=0.0,  # not used in this analytic path
        vc0=vc0,
        omega_p=float(omega_p_sq_ry) ** 0.5,
    )


def fit_head_hl_analytic_from_sample(
    head_static: HeadSample,
    *,
    omega_p_sq_ry: float,
) -> HeadGNParams:
    """Wrapper for :func:`fit_head_hl_analytic` using a resolved head sample."""
    return fit_head_hl_analytic(
        vc0=float(head_static.vc0.real),
        wcoul0_static=float(head_static.wcoul0.real),
        omega_p_sq_ry=omega_p_sq_ry,
    )


def fit_head_with_fixed_omega(
    vc0: float,
    wcoul0_static: float,
    omega_h_ry: float,
) -> HeadGNParams:
    """Build head params with a user-supplied pole frequency Ω_h.

    Useful for cross-validation against BGW: take BGW's analytic head
    pole ``Ω_h(BGW) = √(ω_p²/(1 − ε_head⁻¹))`` (with ε_head⁻¹ from BGW's
    ``epshead(q→0)``), set this option to that value, and isolate any
    LORRAX-vs-BGW residual that's *not* due to the head pole frequency.

    The static W^c(0) head is still LORRAX's, so B_h and R_h scale with
    the LORRAX mini-BZ-averaged static head — same logic as
    :func:`fit_head_hl_analytic`.
    """
    w1 = wcoul0_static - vc0
    omega_h = float(omega_h_ry)
    omega_h_sq = omega_h ** 2
    B_h = -w1 * omega_h_sq
    R_h = B_h / (2.0 * omega_h) if abs(omega_h) > 1.0e-30 else 0.0
    return HeadGNParams(
        omega_h_sq=omega_h_sq,
        omega_h=omega_h,
        B_h=B_h,
        R_h=R_h,
        wc_head_0=w1,
        wc_head_iwp=0.0,
        vc0=vc0,
        omega_p=omega_h,
    )


def fit_head_with_fixed_omega_from_sample(
    head_static: HeadSample,
    *,
    omega_h_ry: float,
) -> HeadGNParams:
    """Wrapper for :func:`fit_head_with_fixed_omega` using a HeadSample."""
    return fit_head_with_fixed_omega(
        vc0=float(head_static.vc0.real),
        wcoul0_static=float(head_static.wcoul0.real),
        omega_h_ry=omega_h_ry,
    )


# ---------------------------------------------------------------------------
# Exact static COHSEX head
# ---------------------------------------------------------------------------

def compute_static_head_terms(
    *,
    vc0: complex,
    wcoul0_static: complex,
    occ: np.ndarray | jnp.ndarray,
    cell_volume: float,
    nk_tot: int,
    source: str = "unknown",
) -> StaticHeadTerms:
    """Build exact static COHSEX head terms (Σ^X, Σ^SX, Σ^{SX-X}, Σ^COH) in band space.

    ``vc0`` / ``wcoul0_static`` are the bare and static-screened Coulomb heads
    in a.u.; ``occ`` is the (nb,) {0,1} occupation mask for the active window.
    Returns diagonal-in-band shifts in Rydberg, with the Brillouin-zone
    average carried by an explicit ``1 / (V_cell · N_k)`` prefactor.
    """

    occ_arr = jnp.asarray(occ, dtype=jnp.complex128)
    ones = jnp.ones_like(occ_arr, dtype=jnp.complex128)
    pref = jnp.asarray(1.0 / (float(cell_volume) * float(nk_tot)), dtype=jnp.complex128)

    v_h = jnp.asarray(vc0, dtype=jnp.complex128)
    w_h = jnp.asarray(wcoul0_static, dtype=jnp.complex128)
    wc_h = w_h - v_h

    sigma_x_diag = -(v_h * pref) * occ_arr
    sigma_sx_diag = -(w_h * pref) * occ_arr
    sigma_sx_minus_x_diag = -(wc_h * pref) * occ_arr
    sigma_coh_diag = 0.5 * (wc_h * pref) * ones

    return StaticHeadTerms(
        sigma_x_diag=sigma_x_diag,
        sigma_sx_diag=sigma_sx_diag,
        sigma_sx_minus_x_diag=sigma_sx_minus_x_diag,
        sigma_coh_diag=sigma_coh_diag,
        vc0=complex(vc0),
        wcoul0=complex(wcoul0_static),
        wc_head_0=complex(wcoul0_static) - complex(vc0),
        source=source,
    )


def compute_static_head_terms_from_sample(head: HeadSample, *,
                                          occ, cell_volume: float,
                                          nk_tot: int) -> StaticHeadTerms:
    """Build exact static COHSEX head terms from a resolved head sample."""
    return compute_static_head_terms(vc0=head.vc0, wcoul0_static=head.wcoul0,
                                     occ=occ, cell_volume=cell_volume,
                                     nk_tot=nk_tot, source=head.source)


def format_static_head_diagnostics(head: StaticHeadTerms) -> str:
    """Return a concise summary of the exact static COHSEX head terms."""

    x_occ = _representative_entry(head.sigma_x_diag)
    sx_occ = _representative_entry(head.sigma_sx_diag)
    sxmx_occ = _representative_entry(head.sigma_sx_minus_x_diag)
    coh_all = _representative_entry(head.sigma_coh_diag)
    lines = [
        "",
        "-" * 72,
        "  STATIC HEAD TERMS (exact COHSEX / BGW-style)",
        "-" * 72,
        f"  Head source: {head.source}",
        f"  v_h(q→0)           = {head.vc0.real:12.6f} a.u.",
        f"  W_h(q→0, ω=0)      = {head.wcoul0.real:12.6f} a.u.",
        f"  W_h^c              = {head.wc_head_0.real:12.6f} a.u.",
        f"  Σ^X head (occ)     = {x_occ.real:12.6e} Ry",
        f"  Σ^SX head (occ)    = {sx_occ.real:12.6e} Ry",
        f"  Σ^(SX-X) head(occ) = {sxmx_occ.real:12.6e} Ry",
        f"  Σ^COH head (all)   = {coh_all.real:12.6e} Ry",
    ]
    return "\n".join(lines)


@functools.partial(jax.jit, static_argnames=('nk_tot', 'nb'))
def _expand_band_diagonal_to_kij_jit(diag, *, nk_tot: int, nb: int):
    """JIT'd body of :func:`expand_band_diagonal_to_kij`."""
    eye = jnp.eye(nb, dtype=jnp.complex128)
    if diag.ndim == 2:
        return eye[None, :, :] * diag[:, :, None]
    one_k = eye[None, :, :] * diag[None, :, None]
    return jnp.broadcast_to(one_k, (nk_tot, nb, nb))


def expand_band_diagonal_to_kij(diag: jnp.ndarray, nk_tot: int) -> jnp.ndarray:
    """Broadcast a band-diagonal shift to a dense ``(nk, nb, nb)`` matrix.

    Thin Python wrapper that pulls ``nb`` from ``diag.shape`` and
    forwards to ``_expand_band_diagonal_to_kij_jit`` — collapses
    ~6 eager-pjit cache misses per call into one cached XLA module.
    """
    diag_arr = jnp.asarray(diag, dtype=jnp.complex128)
    if diag_arr.ndim == 1:
        nb = int(diag_arr.shape[0])
    elif diag_arr.ndim == 2:
        if int(diag_arr.shape[0]) != int(nk_tot):
            raise ValueError(
                f"k-dependent diagonal has {diag_arr.shape[0]} rows, "
                f"expected nk_tot={nk_tot}")
        nb = int(diag_arr.shape[1])
    else:
        raise ValueError("band diagonal must be (nb,) or (nk,nb)")
    return _expand_band_diagonal_to_kij_jit(diag_arr, nk_tot=int(nk_tot), nb=nb)


def static_head_terms_to_kij(
    head: StaticHeadTerms,
    *,
    nk_tot: int,
    do_screened: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Expand exact static head shifts to dense ``(k, i, j)`` matrices.

    Parameters
    ----------
    head
        Exact static head terms from :func:`compute_static_head_terms`.
    nk_tot
        Total number of k-points in the full-zone average.
    do_screened
        If ``True``, return the screened-exchange head ``Sigma^SX``.
        If ``False``, return the bare-exchange head ``Sigma^X``.

    Returns
    -------
    sigma_sx_kij, sigma_coh_kij
        Dense diagonal matrices shaped ``(nk_tot, nb, nb)`` suitable for adding
        directly to the static COHSEX matrices in GWJAX.
    """

    sx_diag = head.sigma_sx_diag if do_screened else head.sigma_x_diag
    return (
        expand_band_diagonal_to_kij(sx_diag, nk_tot),
        expand_band_diagonal_to_kij(head.sigma_coh_diag, nk_tot),
    )


def compute_ppm_head_sigma_kij(
    head: HeadGNParams,
    *,
    omega_grid_ry: np.ndarray,
    enk_ry: np.ndarray,
    efermi_ry: float,
    n_occ: int,
    cell_volume: float,
    nk_tot: int,
    eta: float = 1.0e-6,
) -> np.ndarray:
    """q→0, G=G'=0 head contribution to PPM ``Σ^c_kij(ω)``.

    At q=0, ``M_{nm}(k, q→0, G=0) = δ_{nm}``, so the head only enters the
    band-diagonal ``(i, i)`` of the PPM ``Σ^c`` matrix.  With the GN pole
    extracted in :func:`fit_head_ppm` (``R_h = B_h / (2 Ω_h)``,
    ``B_h = -W^c(0) · Ω_h²``):

        Σ^c_n^head(ω - E_F) =
            +R_h / (V_cell · N_k) · [
                  f_n     / (ω - ε_n + Ω_h - iη)
                + (1-f_n) / (ω - ε_n - Ω_h + iη)
            ]

    where ω, ε_n are taken in the same E_F-relative convention (the difference
    ω - ε_n is invariant under that shift).  In the static limit ω → ε_n
    this reduces to ``-W^c(0) / (2 V_cell N_k)`` for occupied bands and
    ``+W^c(0) / (2 V_cell N_k)`` for empty bands, matching the COHSEX
    static-head pieces (``Σ^{SX-X} + Σ^COH``) built by
    :func:`compute_static_head_terms`.

    Parameters
    ----------
    head
        Fitted GN head pole.
    omega_grid_ry
        Σ^c frequency grid (relative to E_F), shape ``(n_omega,)`` in Ry.
    enk_ry
        Absolute band energies for the σ window, shape ``(nk, nb)`` in Ry.
    efermi_ry
        Fermi level in Ry (subtracted from ``enk_ry`` to get ``ε - E_F``).
    n_occ
        Number of occupied bands at the bottom of the σ window
        (``f_n = 1`` for ``n < n_occ``, else ``0``).
    cell_volume, nk_tot
        Unit-cell volume and full-zone k-point count.
    eta
        Imaginary regularization for the retarded poles.

    Returns
    -------
    sigma_kij : np.ndarray, shape ``(n_omega, nk, nb, nb)``, dtype complex128
        Diagonal-in-band head contribution; off-diagonals are zero.
    """

    omega = np.asarray(omega_grid_ry, dtype=np.float64).reshape(-1)
    enk = np.asarray(enk_ry, dtype=np.float64)
    if enk.ndim != 2:
        raise ValueError("enk_ry must be 2D (nk, nb)")
    n_omega = int(omega.size)
    nk, nb = enk.shape
    sigma_diag = compute_ppm_head_sigma_diag(
        head, omega_grid_ry=omega, enk_ry=enk, efermi_ry=efermi_ry,
        n_occ=n_occ, cell_volume=cell_volume, nk_tot=nk_tot, eta=eta)
    out = np.zeros((n_omega, nk, nb, nb), dtype=np.complex128)
    idx = np.arange(nb)
    out[:, :, idx, idx] = sigma_diag
    return out


def compute_ppm_head_sigma_diag(
    head: HeadGNParams,
    *,
    omega_grid_ry: np.ndarray,
    enk_ry: np.ndarray,
    efermi_ry: float,
    n_occ: int,
    occupations: np.ndarray | None = None,
    cell_volume: float,
    nk_tot: int,
    eta: float = 1.0e-6,
) -> np.ndarray:
    """Band-DIAGONAL of :func:`compute_ppm_head_sigma_kij` — ``(nω, nk, nb)``.

    The q→0 head enters only the band diagonal (``M_{nm}(k, q→0, G=0) =
    δ_{nm}``), so this is the complete information content of the dense
    ``(nω, nk, nb, nb)`` tensor at nb× less memory — the representation the
    sharded-Σ layout (``sigma_omega_layout=sharded``) injects rank-locally
    instead of materializing the dense cube on every rank.  The dense
    builder above embeds exactly this array, so the two representations are
    bit-identical by construction (single source of truth).
    """
    omega = np.asarray(omega_grid_ry, dtype=np.float64).reshape(-1)
    enk = np.asarray(enk_ry, dtype=np.float64)
    if enk.ndim != 2:
        raise ValueError("enk_ry must be 2D (nk, nb)")
    n_omega = int(omega.size)
    nk, nb = enk.shape
    if abs(head.R_h) < 1.0e-30 or abs(head.omega_h) < 1.0e-30:
        return np.zeros((n_omega, nk, nb), dtype=np.complex128)

    if occupations is None:
        f = np.zeros((nb,), dtype=np.float64)
        f[: max(0, min(int(n_occ), nb))] = 1.0
    else:
        f = np.asarray(occupations, dtype=np.float64)
    return compute_complex_pole_head_sigma_diag(
        omega_grid_ry=omega,
        enk_ry=enk,
        efermi_ry=efermi_ry,
        occupations=f,
        poles_ry=np.asarray([head.omega_h - 1j * eta]),
        residues_ry=np.asarray([head.R_h]),
        cell_volume=cell_volume,
        nk_tot=nk_tot,
    )


def on_shell_occupied_head_sigma_ry(
    head: HeadGNParams,
    *,
    cell_volume: float,
    nk_tot: int,
    eta: float = 1.0e-6,
) -> float:
    """Re(Σ^head) for an OCCUPIED band evaluated ON SHELL (ω = ε_nk − E_F).

    THE ONE PLACE the concise-log scalar comes from.  It is *derived from*
    :func:`compute_ppm_head_sigma_diag` — the same kernel that builds the
    tensor the ansatz-neutral finalizer injects — by evaluating it at a
    synthetic single occupied state whose ω sits exactly on shell
    (``δ = ω − (ε − E_F) = 0``).  Nothing here restates the closed form.

    WHY IT EXISTS.  ``gw/ppm_pipeline.py`` used to print this number from a
    hand-written ``-R_h/(Ω_h·V·N_k)``, while the kernel and the named
    ``sig_c_head(Edft).Re`` output column evaluate ``+R_h/(Ω_h·V·N_k)``.
    Measured on the Si 6×6×6 two-update controls (JID 57243214): the log
    said ``-0.8071 eV`` where ``sigma_freq_debug.dat`` carried
    ``+0.807048 eV`` for the same occupied state.  The physics array was
    always right; the duplicated formula in the log had drifted in sign.
    A second spelling of a formula is a second thing to keep in step, so
    there is now only one.

    Returns Ry.  ``0.0`` for a degenerate head (``R_h`` or ``Ω_h`` ≈ 0),
    which is what the kernel returns there too.
    """
    val = compute_ppm_head_sigma_diag(
        head,
        omega_grid_ry=np.zeros(1, dtype=np.float64),
        enk_ry=np.zeros((1, 1), dtype=np.float64),
        efermi_ry=0.0,
        n_occ=1,
        cell_volume=cell_volume,
        nk_tot=nk_tot,
        eta=eta,
    )
    return float(np.real(val[0, 0, 0]))


def compute_complex_pole_head_sigma_diag(
    *,
    omega_grid_ry: np.ndarray,
    enk_ry: np.ndarray,
    efermi_ry: float,
    occupations: np.ndarray,
    poles_ry: np.ndarray,
    residues_ry: np.ndarray,
    cell_volume: float,
    nk_tot: int,
) -> np.ndarray:
    r"""Band-diagonal head self-energy for generic retarded complex poles.

    For poles ``Omega_p = a_p - i Gamma_p`` and head residues ``R_p``,

    .. math::

        \Sigma_n^{\mathrm{head}}(\omega) =
        \frac{1}{V_{\mathrm{cell}}N_k}\sum_p R_p
        \left[\frac{f_{nk}}{\delta_{nk}+\Omega_p}
        + \frac{1-f_{nk}}{\delta_{nk}-\Omega_p}\right],

    where ``delta_nk = omega - (epsilon_nk - E_F)``.  Occupations are
    accepted per band or per ``(k,band)``; this keeps the denominator valid
    when an energy window straddles the Fermi level without deciding how the
    occupations themselves are produced.

    All energy-like inputs and residues are in Ry.  The result is complex Ry
    with shape ``(n_omega, n_k, n_band)``.
    """
    omega = np.asarray(omega_grid_ry, dtype=np.float64).reshape(-1)
    enk = np.asarray(enk_ry, dtype=np.float64)
    if enk.ndim != 2:
        raise ValueError("enk_ry must be 2D (nk, nb)")
    nk, nb = enk.shape
    occ = np.asarray(occupations, dtype=np.float64)
    if occ.shape == (nb,):
        occ = np.broadcast_to(occ[None, :], (nk, nb))
    elif occ.shape != (nk, nb):
        raise ValueError(
            f"occupations must have shape {(nb,)} or {(nk, nb)}, got "
            f"{occ.shape}")
    poles = np.asarray(poles_ry, dtype=np.complex128).reshape(-1)
    # Preserve a real residue dtype.  Besides avoiding a needless complex
    # multiply for real fits, this keeps the one-real-pole PPM wrapper
    # bit-identical to its historical arithmetic.
    residues = np.asarray(residues_ry).reshape(-1)
    if residues.shape != poles.shape:
        raise ValueError("poles_ry and residues_ry must have the same length")
    if poles.size == 0:
        return np.zeros((omega.size, nk, nb), dtype=np.complex128)

    eps_rel = enk - float(efermi_ry)
    delta = omega[:, None, None] - eps_rel[None, :, :]              # (nω, nk, nb)
    pole = poles[:, None, None, None]
    f = occ[None, None, :, :]
    occ_term = f / (delta[None, :, :, :] + pole)
    emp_term = (1.0 - f) / (delta[None, :, :, :] - pole)
    pref = residues / (float(cell_volume) * float(nk_tot))
    sigma_diag = np.sum(
        pref[:, None, None, None] * (occ_term + emp_term), axis=0)
    return np.asarray(sigma_diag, dtype=np.complex128)


def format_head_diagnostics(head: HeadGNParams, cell_volume: float) -> str:
    """Return a short multiline diagnostic summary for the scalar head fit."""

    lines = [
        "",
        "-" * 72,
        "  HEAD CORRECTION (scalar GN, separate from ISDF body)",
        "-" * 72,
        f"  v(q→0)             = {head.vc0:12.3f} a.u.",
        f"  W^c(q→0, ω=0)      = {head.wc_head_0:12.3f} a.u.",
        f"  W^c(q→0, ω=iωp)    = {head.wc_head_iwp:12.3f} a.u.  [ωp={head.omega_p:.4f} Ry]",
        f"  Ω_h²               = {head.omega_h_sq:12.6f} Ry²",
        f"  Ω_h                = {head.omega_h:12.6f} Ry  ({head.omega_h * RYD_TO_EV:.6f} eV)",
        f"  B_h                = {head.B_h:12.6f} Ry² · a.u.",
        f"  R_h                = {head.R_h:12.6f} Ry · a.u.",
    ]
    if abs(head.omega_h) > 1.0e-30:
        lines.append(
            f"  R_h / (Ω_h · vol)  = {head.R_h / (head.omega_h * cell_volume):12.6e} (Ry)"
        )
    else:
        lines.append("  R_h / (Ω_h · vol)  = 0.0 (degenerate)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Rank-1 head injection in the (μ, ν) ISDF basis at q=0
# ---------------------------------------------------------------------------
#
# ``compute_vcoul`` zeroes the ``G=G'=0`` element of ``v(q+G)`` at q=0 to
# avoid the divergence; the BGW-equivalent mini-BZ-averaged value is the
# scalar ``vhead = v_h``.  In the centroid basis the missing piece factors as
#
#     ΔV_{q=0,μν} = (v_h / V_cell) · ζ̄(0, μ, G=0) · ζ(0, ν, G=0)
#                 = (v_h / V_cell) · conj(G0[μ]) · G0[ν]            (rank 1)
#
# The ``1/V_cell`` factor matches the LORRAX storage convention for
# ``V_qmunu`` / ``W_qmunu`` (see ``scripts/checks/sigma_direct_check.py`` for the canonical
# reference).  Conjugation lands on ``μ`` because
# ``V_{qμν} = Σ_GG' ζ*(q,μ,G) v(G,G') ζ(q,ν,G')``.


def _head_rank1_scalars(vhead, whead, cell_volume, omega_index, dtype):
    """Resolve (v_scalar, w_scalar) = (head / V_cell) for a rank-1 update."""
    inv_V = 1.0 / float(cell_volume)
    v_scalar = (jnp.asarray(complex(vhead) * inv_V, dtype=dtype)
                if vhead is not None else None)
    if whead is None:
        return v_scalar, None
    whead_arr = jnp.asarray(whead, dtype=jnp.complex128)
    w_val = whead_arr if whead_arr.ndim == 0 else whead_arr[omega_index]
    return v_scalar, jnp.asarray(w_val * inv_V, dtype=dtype)


def apply_q0_head_rank1(
    V_qmunu: jnp.ndarray,
    W_qmunu: jnp.ndarray | None,
    G0_mu_nu: jnp.ndarray,
    vhead: complex | float | None,
    whead: jnp.ndarray | complex | float | None,
    cell_volume: float,
    *,
    omega_index: int = 0,
):
    """Inject the q=0 Coulomb head as a rank-1 update in the centroid basis.

    Args:
        V_qmunu:   (..., nkx, nky, nkz, n_μ, n_ν) bare-Coulomb body.
        W_qmunu:   same shape (single ω) or ``None`` to skip W.
        G0_mu_nu:  (n_μ,) — ``ζ(q=0, μ, G=0)``.
        vhead, whead: scalar or ``(n_omega,)`` in Ry, or ``None`` to skip.
        cell_volume: V_cell in Bohr³.
        omega_index: slot of ``whead`` to apply (default 0).

    Returns:
        (V_qmunu, W_qmunu) with the q=0 slice updated.
    """
    g0g0 = jnp.einsum('m,n->mn', jnp.conj(G0_mu_nu), G0_mu_nu)
    v_scalar, w_scalar = _head_rank1_scalars(
        vhead, whead, cell_volume, omega_index,
        dtype=(W_qmunu.dtype if W_qmunu is not None else V_qmunu.dtype))

    if v_scalar is not None:
        # V layout: (..., nkx, nky, nkz, n_μ, n_ν); q=0 is index 0 on each k axis.
        V_qmunu = V_qmunu.at[..., 0, 0, 0, :, :].add(v_scalar * g0g0)
    if W_qmunu is not None and w_scalar is not None:
        W_qmunu = W_qmunu.at[..., 0, 0, 0, :, :].add(w_scalar * g0g0)
    return V_qmunu, W_qmunu


def apply_q0_head_rank1_sharded(
    V_q0: jnp.ndarray,
    W_q: jnp.ndarray | None,
    g0_X: jnp.ndarray,
    g0_Y: jnp.ndarray,
    vhead: complex | float | None,
    whead: jnp.ndarray | complex | float | None,
    cell_volume: float,
    *,
    omega_index: int = 0,
):
    """Sharded q=0 head injection — local on every proc.

    Variant of :func:`apply_q0_head_rank1` for BSE-side sharded
    (``P("x", "y")``-on-(μ,ν)) tensors.  ``g0_X`` and ``g0_Y`` are the
    same ``ζ(0,μ,G=0)`` vector duplicated under ``P("x")`` and ``P("y")``
    so the rank-1 ``conj(g0_X)[:, None] * g0_Y[None, :]`` is local.

    Args:
        V_q0:  ``(n_μ, n_ν)``                       sharded ``P("x", "y")``.
        W_q:   ``(n_μ, n_ν, nkx, nky, nkz)`` or ``None``.
        g0_X:  ``(n_μ,)`` sharded ``P("x")`` — μ-axis copy of ζ(0,μ,G=0).
        g0_Y:  ``(n_ν,)`` sharded ``P("y")`` — ν-axis copy of ζ(0,ν,G=0).
        vhead, whead, cell_volume, omega_index: as in
            :func:`apply_q0_head_rank1`.
    """
    g0g0 = jnp.conj(g0_X)[:, None] * g0_Y[None, :]
    v_scalar, w_scalar = _head_rank1_scalars(
        vhead, whead, cell_volume, omega_index,
        dtype=(W_q.dtype if W_q is not None else V_q0.dtype))

    if v_scalar is not None:
        V_q0 = V_q0 + v_scalar * g0g0
    if W_q is not None and w_scalar is not None:
        # W_q layout: (n_μ, n_ν, nkx, nky, nkz). Add to the q=0 slice only.
        W_q = W_q.at[:, :, 0, 0, 0].add(w_scalar * g0g0)
    return V_q0, W_q
