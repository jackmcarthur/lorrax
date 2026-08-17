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
- rank-1 (μ,ν)-basis head injection at q=0
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV


def _analytic_q0_sphere(params) -> bool:
    """One compatibility-preserving resolver for the q=0 v estimator."""
    return (
        bool(params.get("head_minibz_average", False))
        or str(params.get("bgw_metal_q0_treatment", "exact")).strip().lower()
        == "bgw_q0shift"
    )


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
    """``(nval, ncond, nband)`` the way ``psp.get_dipole_mtxels`` derives it.

    Mirrors the writer's own derivation (``get_dipole_mtxels.main``:
    ``nval``/``ncond`` default 5, ``nband`` defaults to
    ``max(wfn.nbands, nelec + ncond)`` and honours an explicit ``nband``),
    because the provenance stamp records exactly those three numbers and
    the check is only meaningful if the reader reconstructs them the same
    way.  Kept as a named helper so the mirroring is visible and greppable
    rather than inlined into the head path.
    """
    nval = int(params.get("nval", 5))
    ncond = int(params.get("ncond", 5))
    try:
        nband_param = params.get("nband", None)
        if nband_param is None:
            nband = max(int(wfn.nbands), int(wfn.nelec) + ncond)
        else:
            nband = int(nband_param)
    except Exception:
        nband = max(int(wfn.nbands), int(wfn.nelec) + ncond)
    return nval, ncond, nband


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
        print_fn(f"Unknown wcoul0_source={want_source}; defaulting to 's_tensor'")
        want_source = "s_tensor"

    omega_val = complex(omega)
    eta = float(params.get("wcoul0_eta", 0.0) or 0.0)
    eps0_path = os.path.join(input_dir, "eps0mat.h5")
    dipole_path = os.path.join(input_dir, "dipole.h5")

    def from_epshead() -> HeadSample | None:
        if not os.path.exists(eps0_path):
            return None
        try:
            if abs(omega_val) > 1.0e-14:
                print_fn(
                    f"wcoul0_source=epshead is static-only; using epshead(0) for omega={omega_val} Ry"
                )
            from file_io.epsreader import EPSReader
            from gw.vcoul import compute_q0_averages

            eps0 = EPSReader(eps0_path)
            vc0_mean, wcoul0 = compute_q0_averages(
                wfn,
                jnp.asarray(eps0.epshead, dtype=jnp.complex128),
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
        except Exception as exc:  # pragma: no cover - diagnostic path
            print_fn(f"epshead wcoul0 failed: {exc}")
            return None

    def from_s_tensor() -> HeadSample | None:
        if not os.path.exists(dipole_path):
            print_fn(f"dipole.h5 not found at {dipole_path}; cannot build S(omega) wcoul0")
            return None
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

    source_order = [want_source] + [s for s in ("epshead", "s_tensor") if s != want_source]
    for source in source_order:
        result = from_epshead() if source == "epshead" else from_s_tensor()
        if result is not None:
            return result

    raise RuntimeError(
        "Failed to resolve q=0 Coulomb head: neither explicit overrides nor supported sources are available."
    )


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
def fold_cartesian_head_wings_sharded(
    S_direct: jax.Array,
    Y_x: jax.Array,
    W_body_xy: jax.Array,
    Z_y: jax.Array,
    cell_volume: float,
    *,
    mesh_xy: Mesh,
) -> jax.Array:
    r"""Fold dynamic body screening into the Cartesian q² head tensor.

    This is the Cartesian-leading-axis generalization of the production-tested
    sharded ``Y W Z`` reduction in ``gw.experimental.head_wing_schur``:

    .. math::

        S_{ab}^{\mathrm{eff}}(z) = S_{ab}^{0}(z)
          + \frac{1}{V_{\mathrm{cell}}}
            \sum_{\mu\nu}Y_{a\mu}(z)W_{\mu\nu}(z)Z_{\nu b}(z).

    Any replicated batch/frequency axes may precede the displayed axes.  The
    centroid axes remain tiled exactly like the screening body: ``Y`` on
    ``x``, ``W`` on ``(x,y)``, and ``Z`` on ``y``.  Therefore the only
    communication is the reduction of the tiny Cartesian output.  The caller
    must supply those shardings; this kernel deliberately does not defensively
    reshard the large inputs.

    Parameters
    ----------
    S_direct
        Direct dipole tensor, ``(..., 3, 3)``, replicated.
    Y_x
        Left wing, ``(..., 3, n_mu)``, centroid axis sharded on ``x``.
    W_body_xy
        Already screened body, ``(..., n_mu, n_mu)``, sharded on ``(x,y)``.
    Z_y
        Right wing, ``(..., n_mu, 3)``, centroid axis sharded on ``y``.
    cell_volume
        Primitive-cell volume in bohr³.  It appears exactly once.
    mesh_xy
        Production two-dimensional device mesh.

    Returns
    -------
    jax.Array
        Effective tensor ``(..., 3, 3)``, replicated on ``mesh_xy``.
    """
    n_lead = W_body_xy.ndim - 2
    if n_lead < 0 or Y_x.ndim != n_lead + 2 or Z_y.ndim != n_lead + 2:
        raise ValueError("Y, W_body, and Z must share their leading axes")
    correction = jnp.einsum(
        "...am,...mn,...nb->...ab", Y_x, W_body_xy, Z_y, optimize=True)
    S_effective = S_direct + correction / jnp.asarray(cell_volume)
    return jax.lax.with_sharding_constraint(
        S_effective,
        NamedSharding(mesh_xy, P(*([None] * n_lead), None, None)),
    )


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
                 "_print_fn", "_cache", "_bare_vc0_override",
                 "_bare_vc0_source")

    def __init__(self, config, input_dir, wfn, sym, meta, print_fn, *,
                 bare_vc0_override=None, bare_vc0_source=None):
        head = config.head
        self._params = {
            "wcoul0_source": head.wcoul0_source,
            "wcoul0_eta": head.wcoul0_eta,
            "vhead": head.vhead,
            "whead_0freq": head.whead_0freq,
            "whead_imfreq": head.whead_imfreq,
            "head_minibz_average": head.head_minibz_average,
            "bgw_metal_q0_treatment": head.bgw_metal_q0_treatment,
        }
        self._input_dir = input_dir
        self._wfn = wfn
        self._sym = sym
        self._meta = meta
        self._print_fn = print_fn
        self._bare_vc0_override = (
            None if bare_vc0_override is None else complex(bare_vc0_override))
        self._bare_vc0_source = bare_vc0_source
        self._cache: dict[tuple[float, float], HeadSample] = {}

    @property
    def bare_vc0_override(self) -> complex | None:
        """Raw q0,G0 value supplied by ``bgw_metal_vcoul_file``."""
        return self._bare_vc0_override

    def _cache_key(self, omega) -> tuple[float, float]:
        z = complex(omega)
        return (round(z.real, 12), round(z.imag, 12))

    def at(self, omega) -> HeadSample:
        """Resolve (and memoize) the head sample at ``omega`` in Ry."""
        key = self._cache_key(omega)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        sample = resolve_head_sample(
            self._params, self._input_dir, self._wfn, self._sym,
            self._meta, self._print_fn, omega=omega,
        )
        if self._bare_vc0_override is not None:
            if abs(sample.vc0) == 0.0:
                raise ValueError(
                    "bgw_metal_vcoul_file cannot replace a zero computed "
                    "q0 bare head while preserving the selected screened-W "
                    "head convention.")
            eps_eff = sample.wcoul0 / sample.vc0
            sample = HeadSample(
                vc0=self._bare_vc0_override,
                wcoul0=self._bare_vc0_override * eps_eff,
                source=(
                    f"{self._bare_vc0_source or 'bgw_metal_vcoul_file'} "
                    f"bare v; {sample.source} screened ratio"),
                omega=sample.omega,
                S_cart=sample.S_cart,
            )
        self._cache[key] = sample
        return sample


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
