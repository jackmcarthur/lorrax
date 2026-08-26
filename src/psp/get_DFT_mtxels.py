#!/usr/bin/env python3
"""
DFT Hamiltonian matrix elements calculation.

This module computes all terms of the DFT Hamiltonian:
- Kinetic energy: <mk|T|nk> 
- Ionic potential: <mk|V_ion|nk>
- Hartree potential: <mk|V_H[n_v]|nk> 
- Nonlocal pseudopotential: <mk|V_NL|nk>

Also computes valence (n_v) and core (n_c) charge densities.
"""

import os
import argparse
import configparser
import re
import glob
import warnings
from pathlib import Path
from common.fft_helpers import local_fftn3, local_ifftn3

# Set JAX configs BEFORE importing JAX (prefer GPU if available)
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Respect user/project overrides; otherwise prefer GPU-capable platforms
if "JAX_PLATFORMS" not in os.environ and "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
# Canonical value, single-sourced in runtime.set_default_env() — kept here
# only because this module is also a standalone CLI that does not call
# bootstrap().  See that function for the measurement (jobs 7882442/7882447).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# REMOVED, both measured on 8 GPUs (job 7882442) rather than argued:
#   XLA_PYTHON_CLIENT_ALLOCATOR=platform — `platform` is plain cudaMalloc, NOT
#     cudaMallocAsync as the old comment here and three docs claimed.  Under it
#     memory_stats() reports bytes_limit=0 and peak_bytes_in_use=0, so it
#     silently zeroes gw_init's GPU high-water report and gw_output's XLA-pool
#     banner.  Leaving it unset selects BFC, which keeps those readings.
#   TF_GPU_ALLOCATOR=cuda_malloc_async — a TensorFlow variable, inert for JAX.
#     A cell setting only this was identical to the unset cell on every metric,
#     including an 11.805 GB BFC pool that cuda_async never has.
# NOTE these two also only ever took effect when this module was imported
# BEFORE the first jax.devices(); under gw/kin_ion_io.py they ran after it and
# changed nothing but the strings in os.environ.

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


# Support both `python -m psp.get_DFT_mtxels` and direct script execution
try:
    from .normalize import normalize_dataclass
    from .load_upf import load_upf
    from ..io import WFNReader
    from ..common.wfn_transforms import read_Gvecs_to_devices
    from ..common import Meta
except ImportError:
    # Fallback for direct script execution: add project `src` to sys.path and use absolute imports
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.append(str(_Path(__file__).resolve().parents[2]))  # .../src
    from psp.upf.normalize import normalize_dataclass
    from psp.upf.load_upf import load_upf
    from ffi import _services      # noqa: F401  (path bootstrap; dies
                                     # with the owner's workspace fix)
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    from common.wfn_transforms import read_Gvecs_to_devices
    from common import Meta
from psp.radial.build_projectors_qe import (
    build_local_ionic_potential_on_G_total,
)
from psp.dft_operators import vnl_matrix_from_kdata
from common.collectives import prepare_mesh, shard_over_k
from dataclasses import dataclass
import h5py
import psp.vnl_ops as vnl_ops

import common.timing as timing
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

# ONE spelling.  This module used to import ``symmetry_maps`` TWICE — once
# as ``from ..common import`` in the try arm above, once as ``from common
# import`` in its fallback — a relative/absolute pair for "``python -m``
# or direct script".  The relative arm never ran: ``from ..io import
# WFNReader`` on the line above it names a package that does not exist, so
# the try arm raises on every interpreter and the fallback is the only
# live code.  The service door is reached by ONE absolute import, and a
# dual spelling of it would be two module objects waiting to happen.
import symmetry_maps                                            # noqa: E402


def report_devices(print_fn=print) -> None:
    """Lightweight device banner — call it, do not trigger it on import.

    This used to run at module IMPORT time.  ``jax.devices()`` brings up
    the XLA backend, and ``jax.distributed.initialize()`` refuses to run
    after that ("must be called before any JAX calls that might
    initialise the XLA backend"), so merely importing this module made
    every downstream CLI single-process **whatever its own header did**.
    That is what pinned ``gw.kin_ion_io`` to one rank; a diagnostic
    print is not worth a backend init, so it moved into a function.
    """
    try:
        devs = jax.devices()
        plat = devs[0].platform if devs else 'none'
        print_fn(f"JAX: {len(devs)} {plat} devices")
    except Exception:
        pass

# Import ISDF modules
 


# NOTE: the cohsex.in parser lived here as a duplicate of
# gw.gw_config.read_lorrax_input.  It was consolidated 2026-07-02 — all
# consumers now import ``read_lorrax_input as read_cohsex_input`` from
# gw.gw_config (single source of truth for the [cohsex] flag surface).


def get_bandranges(nv, nc, nband, nelec):
    """Return ranges of bands necessary for nonlocal potential calculation"""
    nvrange = [int(nelec - nv), int(nelec)]
    ncrange = [int(nelec), int(nelec + nc)]
    nsigmarange = [int(nelec - nv), int(nelec + nc)]
    n_fullrange = [0, int(nband)]
    n_valrange = [0, int(nelec)]
    return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange


# ── Re-exports from psp.pseudos (canonical location) ──
from psp.pseudos import (                              # noqa: F401
    load_pseudopotentials,
    symbol_to_Z as _symbol_to_Z,
    AtomPP,
    build_atom_pp_assignments,
    print_atomic_structure,
)


def spin_degeneracy_factor(wfn) -> float:
    """Electrons per occupied band for this WFN's spin treatment.

    ``WfnLoader.nelec`` is ``max(ifmax)`` — the number of occupied
    *bands*, not electrons.  A spin-restricted scalar calculation
    (``nspin == 1``, ``nspinor == 1``) puts **two** electrons in each of
    those bands; a non-collinear/spinor calculation (``nspinor == 2``,
    the LORRAX default for the MoS₂ decks) puts one, and so does each
    channel of a collinear spin-polarised run (``nspin == 2``).

    Getting this wrong scales ρ — and therefore ⟨V_H⟩, a ~500 eV
    quantity — by a factor of two, so it is derived here once and
    checked against ``∫ρ d³r`` by :func:`build_hartree_potential`.
    """
    return float(wfn.occupation_state_capacity)


@partial(jax.jit, static_argnames=("nocc", "include_dirac_current"))
def _valence_density_kernel(
    psi_k_box: jnp.ndarray,
    weight: jnp.ndarray,
    cell_volume: jnp.ndarray,
    spin_degeneracy: jnp.ndarray,
    band_occupations: jnp.ndarray | None,
    *,
    nocc: int | None,
    include_dirac_current: bool,
) -> jnp.ndarray:
    """Jitted body of :func:`valence_density_from_kpoint`.

    ``nocc`` is static (it fixes the slice shape); the three scalars are
    traced operands so the per-k sweep with varying ``weight`` reuses ONE
    compiled module per box shape instead of dispatching ~17 eager 1-op
    modules per call (scorecard §BE).
    """
    nx, ny, nz = psi_k_box.shape[-3:]
    ngrid = int(nx) * int(ny) * int(nz)
    scale = jnp.sqrt(jnp.asarray(float(ngrid), dtype=jnp.float64) / cell_volume)
    psi_occ = psi_k_box if nocc is None else psi_k_box[: int(nocc)]
    psi_r = local_ifftn3(psi_occ, axes=(-3, -2, -1), norm='ortho') * scale
    prefactor = weight * spin_degeneracy
    if band_occupations is None:
        # Keep the exact-unit insulating path's incumbent arithmetic: no
        # multiply-by-one is inserted into its band/spinor reduction.
        rho = prefactor * jnp.sum(
            jnp.real(jnp.conj(psi_r) * psi_r), axis=(0, 1))
    else:
        occ = band_occupations[:, None, None, None, None]
        rho = prefactor * jnp.sum(
            occ * jnp.real(jnp.conj(psi_r) * psi_r), axis=(0, 1))
    if not include_dirac_current:
        return rho

    # The same occupied bispinors, weights, IFFT and normalisation as rho.
    # These are the signed normalized currents J_i = j_i/c = Psi^dagger
    # alpha_i Psi.  In particular there is NO 1/alpha_fs rescaling: that
    # belongs only to centroid selection's squared-current weight.
    from common.gamma_matrices import gamma_apply, gamma_perm_phase
    psi_dag = jnp.conj(psi_r)
    currents = []
    for mu in (1, 2, 3):
        perm, phase = gamma_perm_phase(mu)
        alpha_psi = gamma_apply(psi_r, perm, phase, axis=1)
        if band_occupations is None:
            currents.append(prefactor * jnp.sum(
                jnp.real(psi_dag * alpha_psi), axis=(0, 1)))
        else:
            currents.append(prefactor * jnp.sum(
                occ * jnp.real(psi_dag * alpha_psi), axis=(0, 1)))
    return jnp.stack((rho, *currents))


def valence_density_from_kpoint(
    psi_k_box: jnp.ndarray,
    *,
    nocc: int | None,
    weight: float,
    cell_volume: float,
    spin_degeneracy: float = 1.0,
    band_occupations: jnp.ndarray | np.ndarray | None = None,
    include_dirac_current: bool = False,
) -> jnp.ndarray:
    """One k-point's contribution to ρ_v(r), on the ψ FFT box grid.

    ``psi_k_box`` is ``(nb, nspinor, nx, ny, nz)`` G-space coefficients
    in the FFT box (the ``load_kpoint_fftbox`` / ``to_box`` layout).
    Returns ``w_k · f_spin · Σ_{n<nocc, s} f_nk |ψ_{nks}(r)|²`` with the
    same ``√(N_grid/Ω)`` normalisation :func:`compute_local_V_k` assumes,
    so ``ΔV · Σ_r ρ = f_spin · w_k · Σ_n f_nk``.  Here ``f_nk=1`` when
    ``band_occupations`` is omitted.

    ``nocc=None`` means "every band in ``psi_k_box`` contributes" — the
    contract the **band-chunked** distributed sweep needs, where a rank has
    been handed the band sub-window ``[b_lo, b_hi)`` of the occupation
    support and there is no band-0-based cut to apply.  ``nocc=n`` keeps the
    legacy
    "first n rows of a full-window box" behaviour.  Both spellings
    produce identical arithmetic for the same set of bands; this is a
    slicing convention, not a second quadrature.

    ``band_occupations`` carries one canonical WFN occupation per band in
    the selected contribution.  ``None`` means exact unit occupation and
    retains the incumbent insulating reduction without a multiply-by-one.
    When supplied, the SAME weights multiply rho and every signed Dirac
    current component inside this one IFFT transaction.

    Single source of truth for the per-k density quadrature: the same-grid
    arm of all-k-resident :func:`compute_valence_density`, the chunked per-k
    CLI and the k/band-partitioned distributed sweep
    (``gw.kin_ion_io.build_valence_density_distributed``) all go
    through this one function.  The arithmetic runs in ONE jitted module
    (:func:`_valence_density_kernel`; ``nocc`` static, scalars traced).

    ``include_dirac_current=True`` requires four-component bispinors and
    returns ``(rho,Jx,Jy,Jz)`` from that same transform, where
    ``J_i = Psi^dagger alpha_i Psi = j_i/c``.  The default scalar branch is
    unchanged and does not trace any gamma operation.
    """
    include_current = bool(include_dirac_current)
    if include_current and int(psi_k_box.shape[1]) != 4:
        raise ValueError(
            "Dirac current requires four-component kinetic-balance "
            f"bispinors; got nspinor={int(psi_k_box.shape[1])}")
    occupations = None
    if band_occupations is not None:
        occupations = jnp.asarray(band_occupations, dtype=jnp.float64)
        n_contributing = (int(psi_k_box.shape[0]) if nocc is None else
                          min(int(nocc), int(psi_k_box.shape[0])))
        if occupations.ndim != 1 or int(occupations.shape[0]) != n_contributing:
            raise ValueError(
                "band_occupations must have one entry per contributing "
                f"band; got shape={occupations.shape}, expected "
                f"({n_contributing},)")
    return _valence_density_kernel(
        psi_k_box,
        jnp.asarray(float(weight), dtype=jnp.float64),
        jnp.asarray(float(cell_volume), dtype=jnp.float64),
        jnp.asarray(float(spin_degeneracy), dtype=jnp.float64),
        occupations,
        nocc=None if nocc is None else int(nocc),
        include_dirac_current=include_current)


# ===========================================================================
# ρ(r) → THE POINT-GROUP-SYMMETRIC ρ(r)
# ===========================================================================
# ρ of a crystal obeys ρ({S|τ}r) = ρ(r) for every operation of its space
# group.  Nothing in the quadrature above enforces that:
#
#   * a sum over the FULL BZ is symmetric only to the accuracy of the ψ(Sk)
#     unfold that produced its summands, so it inherits that unfold's
#     residual;
#   * a WEIGHTED sum over the IBZ is not symmetric at all — it is the
#     density of the representatives, and only the star average of it is
#     the density of the crystal.
#
# Both are cured by the same projection, ρ → (1/N_op) Σ_op ρ({S|τ}r), which
# is the projector onto the invariant subspace and is therefore idempotent:
# applying it to an already-symmetric ρ returns ρ.  Two properties make it
# safe to apply unconditionally, and both matter:
#
#   * ∫ρ d³r is EXACTLY preserved — each term is a permutation of the grid
#     points, so every term has the same grid sum as ρ itself;
#   * the change it makes is bounded by the asymmetry it removes, so on a
#     ρ that is already symmetric to 1e-9 it cannot alter the physics by
#     more than 1e-9.  It is a projection, not a smoothing.
#
# Why V_H is where this is load-bearing.  T, V_loc and V_NL are built from
# the atomic positions and commute with the space group exactly; V_H is
# built from ρ and inherits ρ's residual instead.  With ρ merely NEARLY
# symmetric, ⟨m,Sk|V_H|n,Sk⟩ is merely nearly equal to ⟨m,k|V_H|n,k⟩ across
# a star, so a consumer that computes on the irreducible k-set and
# broadcasts over the star disagrees with a full-BZ sweep at the level of
# that residual, while every other term agrees to round-off.
#
# The permutation table and the star average are BOTH taken from the
# existing implementations (``symmetry_maps.fft_grid_pullback_perm``,
# ``gw.qsgw_density.symmetrise_density``, already used by the QSGW density
# loop and by ζ's full-BZ unfold) so there is one grid-rotation convention
# in the code base, not two.

#: (fft_grid, sym block) → ``(n_op, nx*ny*nz)`` int32 gather table.  The
#: build is O(n_op·N_r) on the host; a QSGW loop rebuilds ρ every iteration
#: and must not pay for it more than once.  Small: 2.6 MB for 48 ops on a
#: 24³ grid.
_RHO_SYM_PERM_CACHE: dict[tuple, "np.ndarray | None"] = {}

#: ‖ρ_sym − ρ‖∞/‖ρ‖∞ above which the projection is announced as a warning.
#: A full-BZ ρ that moves this much is not a nearly-symmetric density being
#: cleaned up — either the k-set was reduced (legitimate, and the
#: projection is then mandatory rather than cosmetic) or the file's
#: symmetry block does not describe these wavefunctions (not legitimate,
#: and ``symmetry_maps.check_density_symmetries`` will have said so at
#: load time).
RHO_SYM_WARN = 1.0e-3


def density_symmetrisation_enabled() -> bool:
    """``LORRAX_RHO_SYMMETRISE``: ``0``/``off``/``no`` turns it off.

    An escape hatch for A/B measurement and for a deck whose symmetry
    block is under suspicion; on by default, because an unsymmetrised ρ is
    wrong for every reduced k-set and imprecise for every other one.
    """
    raw = os.environ.get("LORRAX_RHO_SYMMETRISE", "1").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _rho_sym_perm(wfn, fft_grid: tuple[int, int, int]):
    """The space group's FFT-grid permutation table, or None.

    None means "do not symmetrise": either the handle carries no symmetry
    block (a stub, a ``nosym`` file), or ``ntran <= 1`` and the projection
    is the identity, or some ``τ`` is not commensurate with this grid so
    the operation is not representable as a grid permutation at all.  The
    last case is reported, never rounded away: averaging over a subset of
    the group is not a projector, so a partial table is refused.

    NOT CONSOLIDATED ONTO THE CLOSURE VERDICT, deliberately (fan-out audit
    2026-08-08).  ``gw.qgrid_symmetry`` resolves whether the CENTROID SET
    is closed under the group, and hands back the ``(α, L)`` tables that
    permute the μ index of an ISDF operator.  This site asks a different
    question of a different table: ``fft_grid_pullback_perm`` permutes the
    FULL FFT grid, which is closed under the group by construction, so the
    only way it fails is a ``τ`` that is not commensurate with the grid.
    ``CentroidClosureVerdict`` has nothing to say about that — routing this
    through the q-grid resolution would answer a question about centroids
    to justify a decision about ρ, and would announce "solving on the full
    BZ" for a fallback that is about the Hartree potential and not about q
    at all.  The two share a convention, not a decision.  The warning below
    IS this site's announcement; it names the same consequence shape.
    """
    ntran = int(getattr(wfn, "ntran", 0) or 0)
    mats = getattr(wfn, "sym_matrices", None)
    taus = getattr(wfn, "translations", None)
    if ntran <= 1 or mats is None or taus is None:
        return None
    mats = np.asarray(mats)[:ntran].astype(np.int64)
    taus = np.asarray(taus, dtype=np.float64)[:ntran]
    key = (tuple(int(s) for s in fft_grid), mats.tobytes(), taus.tobytes())
    if key in _RHO_SYM_PERM_CACHE:
        return _RHO_SYM_PERM_CACHE[key]

    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import fft_grid_pullback_perm
    try:
        perm = fft_grid_pullback_perm(mats, taus, fft_grid, validate=True)
    except Exception as exc:
        warnings.warn(
            f"rho symmetrisation is UNAVAILABLE on the {tuple(fft_grid)} "
            f"grid: {exc!r}.  The density is left as accumulated, so V_H "
            f"keeps whatever symmetry residual it carries and is only as "
            f"star-invariant as the density that generated it.",
            RuntimeWarning)
        perm = None
    _RHO_SYM_PERM_CACHE[key] = perm
    return perm


def symmetrize_valence_density(rho_r, wfn, *, print_fn=None):
    """ρ(r) → (1/N_op) Σ_op ρ({S|τ}r) over the WFN file's space group.

    The projector onto the point-group-invariant subspace; see the block
    comment above for why it is applied and why it is safe.  Idempotent, so
    a caller that has already symmetrised pays one gather and gets its own
    array back to round-off.  ∫ρ d³r is preserved exactly.

    ``wfn`` is any handle carrying ``ntran`` / ``sym_matrices`` /
    ``translations`` (a :class:`wfn_loader.WfnLoader` everywhere in
    production); one that does not is left alone.
    """
    if not density_symmetrisation_enabled():
        if print_fn is not None:
            print_fn("    rho symmetrisation: OFF "
                     "(LORRAX_RHO_SYMMETRISE); rho used as accumulated")
        return rho_r
    grid = tuple(int(s) for s in np.shape(rho_r))
    perm = _rho_sym_perm(wfn, grid)
    if perm is None:
        return rho_r

    from gw.qsgw_density import symmetrise_density
    rho_j = jnp.asarray(rho_r, dtype=jnp.float64)
    rho_sym = symmetrise_density(rho_j, perm)
    # How far ρ moved is the one number that says whether this was a
    # round-off cleanup or a change of physics, so it is always computed —
    # printed when there is somewhere to print it, escalated when it is
    # large enough to mean something other than accumulation error.
    scale = float(jnp.max(jnp.abs(rho_j))) or 1.0
    moved = float(jnp.max(jnp.abs(rho_sym - rho_j))) / scale
    if print_fn is not None:
        print_fn(f"    rho symmetrisation: {int(perm.shape[0])} space-group "
                 f"operations, max|Δρ|/max|ρ| = {moved:.3e}")
    if moved > RHO_SYM_WARN:
        warnings.warn(
            f"the density moved by {moved:.3e} (relative) under the "
            f"space-group average, far more than an accumulation residual. "
            f"Expected when rho was built from a REDUCED k-set (the "
            f"projection is then what makes it the crystal's density at "
            f"all); otherwise the file's symmetry block does not describe "
            f"these wavefunctions, and "
            f"symmetry_maps.check_density_symmetries will have said so at "
            f"load time.",
            RuntimeWarning)
    return rho_sym


def build_hartree_potential(
    rho_r: jnp.ndarray,
    wfn,
    *,
    truncation_2d: bool,
    expected_electrons: float | None = None,
    charge_tol: float = 1.0e-3,
    print_fn=print,
) -> jnp.ndarray:
    """ρ(r) → V_H(r) (Ry) with a hard charge-normalisation check.

    ``truncation_2d`` MUST match the Coulomb convention the rest of the
    run uses (``sys_dim == 2`` ⇒ Ismail-Beigi slab cutoff, which is also
    what QE's ``assume_isolated='2D'`` applies to its Hartree, local
    pseudopotential and Ewald terms).  Mixing conventions between
    ``kin_ion``'s V_loc and V_H puts a large *systematic* error straight
    into H₀, where it cannot be told apart from a basis-convergence
    problem.

    ρ IS SYMMETRISED FIRST.  This is the funnel every exact-V_H consumer
    passes through, and V_H is the one term of H₀ that does not commute
    with the space group by construction — it inherits whatever symmetry
    residual its ρ carries (see the block comment above
    :func:`symmetrize_valence_density`).  Doing the projection here rather
    than at each accumulator means there is one place where ρ becomes the
    crystal's density, and the ``∫ρ`` check below then guards the
    projected array rather than its precursor.  The projection is
    idempotent, so a caller that already symmetrised loses nothing.

    The ``∫ρ d³r`` check is the cheap guard against a silent factor-2
    (spin degeneracy) or grid-normalisation slip: both would rescale a
    ~500 eV term.  Because it now runs on the projected ρ, it is also the
    guard on the projection: a star average cannot change ∫ρ, so anything
    that did would surface here rather than downstream in V_H.
    """
    rho_r = symmetrize_valence_density(rho_r, wfn, print_fn=print_fn)
    volume = float(wfn.cell_volume)
    ngrid = int(np.prod(rho_r.shape))
    charge = float(jnp.sum(rho_r)) * volume / ngrid
    if expected_electrons is not None:
        rel = abs(charge - expected_electrons) / max(1.0, abs(expected_electrons))
        print_fn(
            f"    rho normalisation: ∫ρ d³r = {charge:.6f} e "
            f"(expected {expected_electrons:.6f}, rel err {rel:.2e})"
        )
        if rel > charge_tol:
            raise ValueError(
                f"Valence density normalisation is off: ∫ρ d³r = {charge:.6f} "
                f"but {expected_electrons:.6f} electrons were expected "
                f"(rel err {rel:.2e} > {charge_tol:.1e}).  V_H is a ~500 eV "
                "term in H0 — refusing to fold in a mis-normalised density."
            )
    else:
        print_fn(f"    rho normalisation: ∫ρ d³r = {charge:.6f} e")
    print_fn(
        f"    Hartree Coulomb: {'2D slab-truncated (Ismail-Beigi)' if truncation_2d else '3D periodic'}"
    )
    V_H_r = compute_hartree_potential_real(
        rho_r,
        jnp.asarray(wfn.bdot, dtype=jnp.float64),
        bvec=jnp.asarray(wfn.bvec, dtype=jnp.float64),
        blat=float(wfn.blat),
        truncation_2d=bool(truncation_2d),
    )
    hartree_energy = 0.5 * float(jnp.sum(rho_r * V_H_r)) * volume / ngrid
    print_fn(f"    Hartree energy (½∫ρV_H) = {hartree_energy:.6f} Ry")
    return V_H_r


def compute_valence_density(wfn_k, sym, wfn, *, k_source: str):
    """
    Compute valence charge density rho_v(r) from occupied valence wavefunctions.

    The accumulated sum is projected onto the crystal's point group before
    it is returned (:func:`symmetrize_valence_density`).  That is required,
    not cosmetic, for the irreducible-mesh branch below — a k-weighted sum
    over the IBZ is the density of the representatives, not of the crystal
    — and it removes the unfold's residual from the full-mesh branch.

    This resident compatibility path is exact-integer only.  Fractional
    Hartree density is owned by the band-streamed distributed sweep;
    fractional centroid selection uses the canonical QE density.

    Returns:
        Valence charge density rho_v(r) on an ecutrho-based FFT grid if available
    """
    # Compute on configured rho grid if present, else fall back to 2x
    nk_local, nb_all, nspinor, nx, ny, nz = wfn_k.shape
    if k_source == "file":
        kweights = np.asarray(wfn.kweights, dtype=np.float64)
    elif k_source == "full_bz":
        kweights = np.ones(nk_local, dtype=np.float64) / float(sym.nk_tot)
    else:
        raise ValueError(
            "compute_valence_density: k_source must be 'file' or 'full_bz', "
            f"got {k_source!r}.")
    try:
        nx_pad, ny_pad, nz_pad = int(wfn.grid_rho[0]), int(wfn.grid_rho[1]), int(wfn.grid_rho[2])
    except Exception:
        nx_pad, ny_pad, nz_pad = nx, ny, nz

    same_grid = (nx_pad == nx) and (ny_pad == ny) and (nz_pad == nz)

    rho_val_local = jnp.zeros((nx_pad, ny_pad, nz_pad), dtype=jnp.float64)
    volume = jnp.asarray(wfn.cell_volume, dtype=jnp.float64)
    ngrid_pad = nx_pad * ny_pad * nz_pad
    scale_pad = jnp.sqrt(ngrid_pad / volume)
    # Electrons per occupied band (2 only for spin-restricted scalar runs;
    # 1 for the nspinor=2 decks LORRAX actually runs).  ``wfn.nelec`` is a
    # BAND count (max(ifmax)), so this factor is what turns it into charge.
    f_spin = spin_degeneracy_factor(wfn)

    if not wfn.occupations_are_exact_integer:
        raise ValueError(
            "compute_valence_density is the legacy resident all-k FFT-box "
            "path and may not materialize every band of a fractional WFN. "
            "Use gw.kin_ion_io.build_valence_density_distributed for exact "
            "band-streamed Hartree density, or the canonical QE density for "
            "centroid selection.")
    nocc_all = min(int(wfn.nelec), int(nb_all))

    gvecs_by_k = ngk_by_k = None
    if not same_grid:
        gvecs_by_k = wfn.gvecs(k=k_source)
        ngk_by_k = wfn.ngk_valid(k=k_source)

    for ik in range(nk_local):
        nocc = nocc_all
        wk = float(kweights[ik])  # k-point weight

        if same_grid:
            # Single-sourced with the chunked per-k CLI path.
            rho_val_local += valence_density_from_kpoint(
                wfn_k[ik], nocc=nocc, weight=wk,
                cell_volume=float(wfn.cell_volume),
                spin_degeneracy=f_spin,
            )
        else:
            gvecs_k = np.asarray(
                gvecs_by_k[ik, :int(ngk_by_k[ik])])
            Gx = jnp.asarray(gvecs_k[:, 0], dtype=jnp.int32)
            Gy = jnp.asarray(gvecs_k[:, 1], dtype=jnp.int32)
            Gz = jnp.asarray(gvecs_k[:, 2], dtype=jnp.int32)

            for ispin in range(nspinor):
                C_src = wfn_k[ik, :nocc, ispin, :, :, :]

                def gather_one(arr3d):
                    return arr3d[Gx, Gy, Gz]

                C_occ = jax.vmap(gather_one, in_axes=0, out_axes=0)(C_src)

                def scatter_one(row):
                    buf = jnp.zeros((nx_pad, ny_pad, nz_pad), dtype=jnp.complex128)
                    return buf.at[Gx, Gy, Gz].set(row)

                psi_G_padded_batch = jax.vmap(scatter_one, in_axes=0, out_axes=0)(C_occ)
                psi_r_batch = local_ifftn3(psi_G_padded_batch, axes=(-3, -2, -1), norm='ortho') * scale_pad
                rho_val_local += (wk * f_spin) * jnp.sum(
                    jnp.real(psi_r_batch.conj() * psi_r_batch), axis=0)
    
    # With proper k-point weights included above, no division needed
    # (weights sum to 1 for irreducible mesh, or 1/nk_tot each for full mesh)
    rho_v = symmetrize_valence_density(rho_val_local, wfn)

    # Caller reports integrated charge if needed
    return rho_v

def compute_core_density(atom_positions, atom_types, pseudos, meta):
    """
    Compute core charge density rho_c(r) from atomic core states in pseudopotentials.
    
    Args:
        atom_positions: Atomic positions in crystal coordinates, shape (nat, 3)
        atom_types: Atom type indices, shape (nat,)
        pseudos: Dictionary mapping element names to pseudopotential objects
        meta: System metadata object
        
    Returns:
        Core charge density rho_c(r), shape (nx, ny, nz)
    """
    print("  Computing core charge density rho_c(r)...")
    
    # TODO: Implement the following steps:
    # 1. For each atom:
    #    a. Get core charge density from pseudopotential file (usually rho_core(r))
    #    b. Place at atomic position with proper structure factor
    #    c. Transform to real space grid
    # 2. Sum contributions from all atoms
    # 3. Apply proper normalization
    
    # Note: For norm-conserving pseudopotentials, core density is often
    # represented as a smooth function that reproduces the correct
    # integrated charge within some cutoff radius
    
    # Placeholder implementation
    nx, ny, nz = meta.fft_grid
    rho_core = jnp.zeros((nx, ny, nz), dtype=jnp.float64)
    
    return rho_core


def compute_hartree_potential_real(
    rho_valence_padded: jnp.ndarray,
    bdot: jnp.ndarray,
    bvec: jnp.ndarray | None = None,
    blat: float | None = None,
    truncation_2d: bool = False,
) -> jnp.ndarray:
    """Compute Hartree potential via the shared Poisson solver.
    
    Args:
        rho_valence_padded: Valence density on FFT grid
        bdot: Reciprocal lattice metric tensor (3x3)
        bvec: Reciprocal lattice vectors (3x3), needed if truncation_2d=True
        blat: Lattice constant (bohr), needed if truncation_2d=True  
        truncation_2d: If True, apply 2D slab truncation for Coulomb
    """
    rho_G = local_fftn3(
        rho_valence_padded, axes=(-3, -2, -1), norm="ortho")
    V_H_r = poisson_potential_from_rhoG(rho_G, bdot, bvec, blat, truncation_2d)

    return V_H_r

# ── Re-exports from dft_operators (canonical location) ──
from psp.dft_operators import poisson_potential_from_rhoG  # noqa: F401
from psp.dft_operators import padded_gvectors              # noqa: F401
from psp.dft_operators import generate_gvectors_k          # noqa: F401  (D10 reference)


class LocalPotentialScalars(NamedTuple):
    """The four constants of a ``⟨m|V(r)|n⟩`` FFT round trip.

    ``scale`` (ψ_G → ψ_r), ``deltaV`` and ``fft_norm`` (φ_r → φ_G) and
    ``post`` (applied to the finished block).
    """
    scale: jax.Array
    deltaV: jax.Array
    fft_norm: jax.Array
    post: jax.Array


def local_potential_scalars(volume, ngrid) -> LocalPotentialScalars:
    """THE normalisation chain of every local-potential matrix element.

    One definition, two sharding plans.  ``_compute_local_V_k_jit`` (the
    local plan: whole FFT box on one rank) and
    ``common.mtxel_sweep.local_potential_operator`` (the 2-D band-sharded
    plan) both call this, so the two agree to round-off BY CONSTRUCTION
    and the only difference between them is the reassociation the
    sharding forces on the G sum.  Before this existed the chain was
    written out three times and the agreement was a coincidence
    maintained by hand.

    Works on traced or concrete inputs.  The sweep calls it once at
    factory-build time and takes ``float()`` of each field, so the
    constants are baked into the jaxpr as literals rather than riding
    through the scan as operands.
    """
    ngrid = jnp.asarray(ngrid, dtype=jnp.float64)
    volume = jnp.asarray(volume, dtype=jnp.float64)
    return LocalPotentialScalars(
        scale=jnp.sqrt(ngrid / volume),
        deltaV=volume / ngrid,
        fft_norm=jnp.sqrt(ngrid),
        post=jnp.sqrt(1.0 / volume),
    )


def kinetic_diagonal(G_float, kpoint_crys, bdot, g_mask=None):
    """``T_G = |k+G|²`` in Ry — THE kinetic diagonal, one definition.

    Shared by ``_compute_kinetic_k_jit`` (local plan) and
    ``common.mtxel_sweep.kinetic_operator`` (2-D plan).

    ``g_mask`` is applied to the DIAGONAL rather than to ψ.  That is
    sufficient — a masked ``T_G`` makes the pad column contribute exactly
    zero to the contraction however ψ is laid out — and it is what the
    local plan has always done, so masking here keeps the two routes
    term-for-term identical.

    NOT the same job as the ``|k+G|²`` in ``psp.gvec_utils`` /
    ``psp.dft_operators`` / ``file_io.wfn_writer``: those are host-side
    and SELECT or SORT a G-sphere by kinetic energy.  This one applies an
    operator to ψ.  Same formula, different question; they are not
    unified on purpose.
    """
    K_crys = jnp.asarray(G_float, dtype=jnp.float64) + \
        jnp.asarray(kpoint_crys, dtype=jnp.float64)[None, :]
    T_G = jnp.einsum('gi,ij,gj->g', K_crys,
                     jnp.asarray(bdot, dtype=jnp.float64), K_crys,
                     optimize=True)
    if g_mask is not None:
        T_G = T_G * jnp.asarray(g_mask, dtype=jnp.float64)
    return T_G


def compute_kinetic_k(wfn_k, Gk_crys, kpoint_crys, bdot, g_mask: jax.Array | None = None):
    """
    Compute kinetic energy matrix elements <mk|T|nk> for a single k-point.
    
    T = -∇² = |k+G|² in reciprocal space (Ry units)
    
    Args:
        wfn_k: Wavefunction coefficients for single k-point, shape (nb, nspinor, nx, ny, nz)
        Gk_crys: G-vectors in crystal coordinates, shape (nG, 3)
        kpoint_crys: k-point in crystal coordinates, shape (3,)
        bdot: reciprocal metric matrix, shape (3, 3)
        
    Returns:
        Kinetic energy matrix elements, shape (nb, nb)
    """
    G_int = jnp.asarray(Gk_crys, dtype=jnp.int32)
    G_float = jnp.asarray(Gk_crys, dtype=jnp.float64)
    k_crys = jnp.asarray(kpoint_crys, dtype=jnp.float64)
    bdot = jnp.asarray(bdot, dtype=jnp.float64)
    g_mask_j = None if g_mask is None else jnp.asarray(g_mask, dtype=jnp.float64)
    return _compute_kinetic_k_jit(wfn_k, G_int, G_float, k_crys, bdot, g_mask_j)


@jax.jit
def _compute_kinetic_k_jit(
    wfn_k: jax.Array,
    G_int: jax.Array,
    G_float: jax.Array,
    k_crys: jax.Array,
    bdot: jax.Array,
    g_mask: jax.Array | None,
) -> jax.Array:
    T_G = kinetic_diagonal(G_float, k_crys, bdot, g_mask=g_mask)
    Gx = G_int[:, 0]
    Gy = G_int[:, 1]
    Gz = G_int[:, 2]
    psi_G = wfn_k[:, :, Gx, Gy, Gz]
    T_psi = T_G[None, None, :] * psi_G
    return jnp.einsum('msg,nsg->mn', jnp.conj(psi_G), T_psi, optimize=True)

def compute_local_V_k(wfn_k, Gk_crys, V_r, cell_volume, g_mask: jax.Array | None = None):
    """
    Compute elements of a local potential (V_ion or V_H) <mk|V|nk> for a single k-point.
    
    Args:
        wfn_k: Wavefunction coefficients for single k-point, shape (nb, nspinor, nx, ny, nz)
        Gk_crys: G-vectors in crystal coordinates, shape (nG, 3)
        V_H_r: Real-space Hartree potential on the 2x FFT grid, shape (2*nx, 2*ny, 2*nz)
        
    Returns:
        Hartree potential matrix elements, shape (nb, nb)
    """
    V_r = jnp.asarray(V_r, dtype=jnp.complex128)
    Gx = jnp.asarray(Gk_crys[:, 0], dtype=jnp.int32)
    Gy = jnp.asarray(Gk_crys[:, 1], dtype=jnp.int32)
    Gz = jnp.asarray(Gk_crys[:, 2], dtype=jnp.int32)
    volume = jnp.asarray(cell_volume, dtype=jnp.float64)
    g_mask_j = None if g_mask is None else jnp.asarray(g_mask, dtype=jnp.float64)
    return _compute_local_V_k_jit(wfn_k, Gx, Gy, Gz, V_r, volume, g_mask_j)


@jax.jit
def _compute_local_V_k_jit(
    wfn_k: jax.Array,
    Gx: jax.Array,
    Gy: jax.Array,
    Gz: jax.Array,
    V_r: jax.Array,
    volume: jax.Array,
    g_mask: jax.Array | None,
) -> jax.Array:
    psi_G = jnp.asarray(wfn_k, dtype=jnp.complex128)
    nb = psi_G.shape[0]
    nspinor = psi_G.shape[1]
    nx, ny, nz = psi_G.shape[-3:]

    ngrid = nx * ny * nz
    sc = local_potential_scalars(volume, ngrid)

    psi_r = local_ifftn3(psi_G, axes=(-3, -2, -1), norm='ortho') * sc.scale
    phi_r = psi_r * V_r
    phi_G = local_fftn3(phi_r, axes=(-3, -2, -1), norm='ortho') * (
        sc.deltaV * sc.fft_norm)

    psi_coeffs = psi_G[:, :, Gx, Gy, Gz]
    vpsi = phi_G[:, :, Gx, Gy, Gz]
    if g_mask is not None:
        psi_coeffs = psi_coeffs * g_mask[None, None, :]
        vpsi = vpsi * g_mask[None, None, :]
    V_loc = jnp.einsum('bsg,nsg->bn', jnp.conj(psi_coeffs), vpsi, optimize=True)
    return V_loc * sc.post


@timing.timed("psp.get_DFT_mtxels.get_H_matrix_elements", watch=True)
def get_H_matrix_elements(wfn, sym, pseudos, global_psi_G, meta, mesh_xy,
                          exact_hartree_full, sys_dim: int = 3):
    """
    Compute nonlocal pseudopotential matrix elements <mk|V_NL|nk> for all k-points.

    This implementation distributes k-points across the XY processor grid and
    computes V_NL elements for each k-point independently.

    Args:
        wfn: WfnLoader object
        sym: SymMaps object
        pseudos: Dictionary of loaded pseudopotentials
        global_psi_G: Global sharded wavefunction coefficients in G-space
        meta: System metadata
        mesh_xy: JAX device mesh for sharding
        exact_hartree_full: exact FFT-grid Hartree matrix from the shared
            distributed owner, on the full BZ and the same band window as
            ``global_psi_G``.
        sys_dim: system dimensionality (0/2/3) from the DECK.  Both the
            Hartree and the local ionic potential take their Coulomb
            convention from it — ``truncation_2d = (sys_dim == 2)``, the
            Ismail-Beigi slab cutoff, which is also what QE's
            ``assume_isolated='2D'`` applies.  This used to be hardwired
            ``True`` at both call sites, so a 3D bulk deck silently got a
            slab-truncated Hartree AND a slab-truncated V_loc.

    Returns:
        Array of nonlocal potential matrix elements, shape (nk, nb, nb)
    """
    from psp.operator_checks import validate_operator_inputs
    ctx = validate_operator_inputs(
        pseudos=pseudos, wfn=wfn, sys_dim=sys_dim,
        caller="get_H_matrix_elements",
    )
    print("\nComputing DFT Hamiltonian (Ry units)...")
    print(f"  Coulomb truncation: "
          f"{'2D slab' if ctx.truncation_2d else '3D bulk'} (sys_dim={sys_dim})")

    # 1. Reshard wavefunctions to distribute k-points over XY grid
    print("  Resharding wavefunctions to device mesh")
    wfn_k_sharded = shard_over_k(global_psi_G, mesh_xy)
    
    # 2. Prepare atomic structure data (replicated on all devices)
    atom_positions = jnp.asarray(wfn.atom_crys, dtype=jnp.float64)  # Crystal coordinates
    atom_types = jnp.asarray(wfn.atom_types, dtype=jnp.int32)
    bvec = jnp.asarray(wfn.bvec, dtype=jnp.float64)
    print(f"\nSystem: {len(atom_positions)} atoms, {len(pseudos)} pseudopotential types")
    assignments = build_atom_pp_assignments(atom_positions, atom_types, pseudos)
    for ap in assignments:
        ppname = os.path.basename(getattr(ap.pseudo, '_source_path', 'N/A')) if ap.pseudo else 'None'
        print(f"  atom {ap.index}: Z={ap.atomic_number} elem={ap.element} pp={ppname}")

    species_payload: list[tuple[object, np.ndarray]] = []
    species_tmp: dict[int, dict[str, object]] = {}
    for ap in assignments:
        pseudo = ap.pseudo
        if pseudo is None:
            continue
        key = id(pseudo)
        entry = species_tmp.setdefault(key, {"pseudo": pseudo, "positions": []})
        entry["positions"].append(np.asarray(ap.position, dtype=float))
    for entry in species_tmp.values():
        positions = np.asarray(entry["positions"], dtype=float) if entry["positions"] else np.zeros((0, 3), dtype=float)
        species_payload.append((entry["pseudo"], positions))

    assignment_payload = [
        {
            "pseudo": ap.pseudo,
            "position": np.asarray(ap.position, dtype=float),
        }
        for ap in assignments
    ]

    # G scaffolding: the loader's OWN fixed-shape (nk, ngkmax, 3) table
    # plus its pad mask.  This replaces a per-k `generate_gvectors_k`
    # sweep that sliced the same table back to each k's ngk and then
    # re-padded it by hand to the max — three passes to arrive where the
    # loader already was (owner decision D10).
    gtab = padded_gvectors(wfn, k="full_bz")
    kpoints = jnp.asarray(gtab.kvecs, dtype=jnp.float64)
    vnl_setup = vnl_ops.build_vnl_setup(
        wfn,
        sym,
        meta,
        pseudos,
        nspinor=int(wfn.nspinor),
    )

    # 3. The Hartree term comes from gw.kin_ion_io's distributed exact owner.
    # This CLI's resident FFT-box carrier covers only its requested output
    # window (nelec+ncond), which is not the physical-density support of a
    # smeared WFN with signed occupation tails.  Rebuilding rho here would
    # therefore either truncate those tails or force every full-BZ band into
    # one resident box.  The shared owner streams the complete canonical
    # support in bounded band chunks, then projects the same potential into
    # this output window.
    rho_valence = None
    bdot_j = jnp.asarray(wfn.bdot, dtype=jnp.float64)
    hartree_shape = tuple(int(s) for s in np.shape(exact_hartree_full))
    expected_hartree_shape = (
        int(sym.nk_tot), int(global_psi_G.shape[1]),
        int(global_psi_G.shape[1]))
    if hartree_shape != expected_hartree_shape:
        raise ValueError(
            "get_H_matrix_elements: shared exact-Hartree matrix has shape "
            f"{hartree_shape}, expected {expected_hartree_shape} for the "
            "resident output carrier.")
    print("\n  Using shared distributed exact FFT-grid Hartree matrix")
    # Compute core density from pseudopotentials (disabled for performance)  
    # rho_core = compute_core_density(atom_positions, atom_types, pseudos, meta)
    # Build local ionic potential on rho grid via G-space and FFT (return total only)
    print("  Computing local ionic potential V_loc(r) on rho grid...")
    try:
        rho_grid = tuple(int(x) for x in getattr(wfn, 'grid_rho'))
    except Exception:
        rho_grid = (int(2*meta.fft_grid[0]), int(2*meta.fft_grid[1]), int(2*meta.fft_grid[2]))
    V_loc_r = build_local_ionic_potential_on_G_total(
        assignments=assignment_payload,
        species_groups=species_payload,
        fft_grid=rho_grid,
        bdot=np.asarray(wfn.bdot, dtype=float),
        cell_volume=float(wfn.cell_volume),
        bvec=np.asarray(wfn.bvec, dtype=float),
        blat=float(wfn.blat),
        truncation_2d=ctx.truncation_2d,   # deck sys_dim, NOT hardwired
    )
    V_loc_r = jnp.asarray(V_loc_r, dtype=jnp.float64)

    # 4. Execute DFT Hamiltonian calculation over k-points using precomputed G-vectors
    print("\n  Building H(k) on first k-point for debug...")
    H_list = []
    first_k_components = None
    for i in range(1):
        wfn_k = wfn_k_sharded[i]  # (nb, nspinor, nx, ny, nz)
        kpoint = kpoints[i]
        Gk_crys, g_mask = gtab.at(i)

        # Every term takes the SAME (padded) G-list and the SAME mask.
        # Before D10 the local terms were padded and V_NL was not, so the
        # four contributions to H(k=0) were summed over two different
        # G-layouts inside a ~500 eV cancellation.
        T_k = compute_kinetic_k(wfn_k, Gk_crys, kpoint, bdot_j, g_mask)
        V_ion_k = compute_local_V_k(wfn_k, Gk_crys, V_loc_r, wfn.cell_volume, g_mask)
        V_H_k = jnp.asarray(exact_hartree_full[i], dtype=jnp.complex128)
        kdata = vnl_ops.build_vnl_kdata_from_kvec(
            np.asarray(kpoint, dtype=float),
            np.asarray(Gk_crys, dtype=int),
            vnl_setup,
        )
        V_NL_k = vnl_matrix_from_kdata(wfn_k, Gk_crys, kdata, mask=g_mask)

        # Temporary debug prints: first 4x4 blocks (2 decimals, scientific)
        # (per-matrix debug prints removed)

        H_k = T_k + V_ion_k + V_H_k + V_NL_k
        H_list.append(H_k)

        # Save components for k=0 (first iter only)
        if i == 0:
            first_k_components = {
                'T': T_k,
                'V_ion': V_ion_k,
                'V_H': V_H_k,
                'V_NL': V_NL_k,
                'H_no_NL': T_k + V_ion_k + V_H_k,
            }

    H_sharded = jnp.stack(H_list, axis=0)
    
    print(f"  Completed: DFT Hamiltonian matrix shape {H_sharded.shape}")
    
    return H_sharded, rho_valence, first_k_components
    
# ---------------------------------------------------------------------------
# REMOVED 2026-07-30 (owner decision D10 workstream): ``get_kin_ion`` and
# ``write_kin_ion_h5``.
#
# They computed T + V_loc + V_NL (+ optional V_H) for all k and wrote
# ``kin_ion.h5`` — the same physics, the same dataset name and the same
# output PATH as ``gw.kin_ion_io``, which is the route the GW pipeline
# actually reads and which is pinned to pw2bgw's ``kih.dat`` at
# rms 6.19e-5 eV / max 2.39e-4 eV (job 7882058).  Three reasons the
# duplicate had to go rather than be kept as a reference:
#
# 1. WRONG COULOMB CONVENTION FROM THIS MODULE'S OWN CLI.  ``get_kin_ion``
#    defaulted to ``sys_dim=3``; ``main()`` called it without one.  So on
#    any slab deck it built V_loc 3D-periodic while ``get_H_matrix_elements``
#    100 lines earlier hardwired ``truncation_2d=True`` — and then wrote the
#    result to ``<input_dir>/kin_ion.h5``, silently clobbering the correct
#    file with a wrong-convention, V_H-free one.  Inside H0's ~500 eV
#    cancellation that is not a small error.
# 2. IT DID NOT DISTRIBUTE.  Every rank held a replicated
#    ``np.zeros((nk, nb, nb))`` (1.0 GiB at nb=2048) and pulled each k out
#    of a k-SHARDED global array one at a time — a gather per k.
#    ``gw.kin_ion_io`` partitions the k loop and gathers once.
# 3. Its one genuinely useful idea — building ``Gk_crys_pad`` + ``G_mask``
#    and passing the mask so every k presents one shape — is now the
#    ``psp.dft_operators.PaddedGVectors`` route, used by ``gw.kin_ion_io``
#    and gated at 1e-12 by tests/test_kin_ion_padded_gvectors.py.
#
# ``get_H_matrix_elements`` below is KEPT: its k=0 per-term decomposition
# (K / V_ion / V_H / V_NL written to k0_diag.txt) has no replacement.
# ---------------------------------------------------------------------------



def main(argv=None):
    """Main function for nonlocal pseudopotential calculation."""
    print("="*60)
    print("Nonlocal Pseudopotential Matrix Elements Calculator")
    print("="*60)

    argp = argparse.ArgumentParser(description="Nonlocal pseudopotential V_NL calculator")
    argp.add_argument(
        "-i",
        "--input", 
        default="tests/cohsex_debug/cohsex_test.in",
        help="Input file",
    )
    args = argp.parse_args(argv)
    
    timing.reset()

    # Read input parameters (canonical parser — single source of truth)
    from gw.gw_config import read_lorrax_input as read_cohsex_input
    print(f"\nReading input from: {args.input}")
    params = read_cohsex_input(args.input)
    
    # Resolve relative paths against the input file's directory
    input_dir = os.path.dirname(os.path.abspath(args.input))
    def _resolve_path(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(input_dir, path)
    params["wfn_file"] = _resolve_path(params["wfn_file"])
    
    # Extract parameters
    nval = params["nval"]
    ncond = params["ncond"] 
    nband = params["nband"]
    bispinor = params["bispinor"]
    # The deck owns the Coulomb convention.  Absent sys_dim, 3 (bulk) is
    # the same default gw.kin_ion_io uses, so the two CLIs cannot disagree
    # by accident on the same deck.
    sys_dim = int(params.get("sys_dim") if params.get("sys_dim") is not None else 3)

    print(f"Parameters: nval={nval}, ncond={ncond}, nband={nband}, "
          f"bispinor={bispinor}, sys_dim={sys_dim}")
    
    # Load wavefunction file
    print(f"\nLoading wavefunction file: {os.path.basename(params['wfn_file'])}")
    with timing.section("psp.get_DFT_mtxels.load_wfn"):
        try:
            wfn = WfnLoader(params["wfn_file"])
            print(f"  Success: {wfn.nkpts} k-points, {wfn.nbands} bands, {wfn.nelec} electrons")
        except Exception as e:
            print(f"  Error loading WFN file: {e}")
            return 1
    
    # Initialize symmetry mappings
    print("\nInitializing symmetry mappings...")
    with timing.section("psp.get_DFT_mtxels.symmetry"):
        sym = symmetry_maps.SymMaps(wfn)
    print(f"  Success: {sym.nk_tot} total k-points, {sym.nk_red} irreducible k-points")
    
    # Get band ranges
    nvrange, ncrange, nsigmarange, n_fullrange, n_valrange = get_bandranges(
        nval, ncond, nband, wfn.nelec
    )
    print(f"Band ranges: valence={nvrange}, conduction={ncrange}, sigma={nsigmarange}")
    
    # Create system metadata
    print("\nCreating system metadata...")
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, 0, bispinor)  # n_rmu=0 for now
    print(f"  FFT grid: {meta.fft_grid}")
    print(f"  Spinor components: {meta.nspinor}")

    # Density-grid cutoff (Ry): explicit ``ecutrho`` from the input, else default
    # to the WFN's own ``ecutwfc`` so the rho grid is always attached.
    _ecutrho_in = params.get("ecutrho")
    ecutrho_ry = float(_ecutrho_in) if _ecutrho_in is not None else float(wfn.ecutwfc)
    setattr(wfn, 'grid_rho', tuple(2 * n for n in meta.fft_grid))  # 2× wavefunction grid
    print(f"  Using ecutrho = {ecutrho_ry:.3f} Ry "
          f"({'input' if _ecutrho_in is not None else 'default=WFN ecutwfc'}); "
          f"rho grid = {wfn.grid_rho}")
    
    mesh_xy = prepare_mesh(print_fn=print)
    print(f"JAX device mesh: {'x'.join(str(int(n)) for n in mesh_xy.devices.shape)}"
          f" = {int(mesh_xy.devices.size)} devices")
    
    # Load G-vectors and wavefunction coefficients
    print("\nLoading wavefunction coefficients to devices...")
    brange = (0, nsigmarange[1])  # Load all bands for now
    with timing.section("psp.get_DFT_mtxels.read_Gvecs") as timer_read:
        global_psi_G, nb_actual = read_Gvecs_to_devices(wfn, sym, brange, meta, bispinor, mesh_xy)
        timer_read.watch(global_psi_G)
    print(f"  Loaded {nb_actual} bands in G-space, shape: {global_psi_G.shape}")
    
    # Load pseudopotentials from working directory
    print("\nScanning for pseudopotential files...")
    with timing.section("psp.get_DFT_mtxels.load_pseudos"):
        pseudos = load_pseudopotentials(input_dir)
    
    # Print atomic structure information
    print_atomic_structure(wfn, pseudos)
    
    # Compute the exact Hartree matrix through the same band-streamed owner
    # used by stored/gspace GW.  Density support is loader-owned and therefore
    # independent of this diagnostic CLI's nelec+ncond output window.
    from gw.kin_ion_io import compute_hartree_matrix
    with timing.section("psp.get_DFT_mtxels.build_V_H"):
        exact_hartree_full = compute_hartree_matrix(
            wfn, sym, meta, truncation_2d=(sys_dim == 2),
            nb=int(nb_actual), mesh=mesh_xy, include_transverse=False,
            print_fn=print, k_set="full")

    # Compute DFT Hamiltonian matrix elements
    print(f"\nComputing DFT Hamiltonian matrix elements...")
    H_DFT, rho_valence, k0 = get_H_matrix_elements(
        wfn, sym, pseudos, global_psi_G, meta, mesh_xy,
        exact_hartree_full,
        sys_dim=sys_dim)
    print(f"  Hamiltonian matrix elements shape: {H_DFT.shape}")
    
    print(f"  Total electrons in system: {wfn.nelec}")
    # Report both grid-sum and ΔV-weighted integral of the valence density (rho grid)
    if rho_valence is not None:
        nx2, ny2, nz2 = rho_valence.shape
        Ntot = float(nx2 * ny2 * nz2)
        raw_sum = float(jnp.sum(rho_valence))
        deltaV = float(wfn.cell_volume) / Ntot
        charge_int = raw_sum * deltaV
        print(f"  Valence density (rho grid): Σ rho = {raw_sum:.6f}, ΔV = {deltaV:.6e}, ∫ rho d³r = {charge_int:.6f}")

    # Write requested k=0 diagonal elements: (band id, K, V_ion, V_H, K+I+H)
    try:
        if k0 is not None:
            T0 = np.asarray(k0['T'])
            VI0 = np.asarray(k0['V_ion'])
            VI_SR0 = np.asarray(k0.get('V_ion_SR', VI0*0))
            VI_LR0 = np.asarray(k0.get('V_ion_LR', VI0))
            VH0 = np.asarray(k0['V_H'])
            VNL0 = np.asarray(k0['V_NL'])
            HnoNL0 = np.asarray(k0['H_no_NL'])
            nb0 = T0.shape[0]
            m = min(26, nb0)
            out = np.zeros((m, 6), dtype=float)
            for b in range(m):
                out[b, 0] = b + 1  # 1-based band id
                out[b, 1] = np.real(T0[b, b])
                out[b, 2] = np.real(VI0[b, b])
                out[b, 3] = np.real(VH0[b, b])
                out[b, 4] = np.real(VNL0[b, b])
                out[b, 5] = np.real(HnoNL0[b, b] + VNL0[b, b])
            out_path = os.path.join(input_dir, 'k0_diag.txt')
            np.savetxt(
                out_path,
                out,
                fmt=['%4d','% .5f','% .5f','% .5f','% .5f','% .5f'],
                header='band_id  K(Ry)  V_ion(Ry)  V_H(Ry)  V_NL(Ry)  K+I+H+NL(Ry)'
            )
            print(f"\nWrote k=0 diagonals to {out_path}")

            # If a reference exists (k0_diag_check), compare K+I+H diagonals
            # Accept either k0_diag_check or k0_diag_check.txt
            ref_path = os.path.join(input_dir, 'k0_diag_check')
            if not os.path.exists(ref_path):
                alt = ref_path + '.txt'
                ref_path = alt if os.path.exists(alt) else ref_path
            if os.path.exists(ref_path):
                try:
                    ref = np.loadtxt(ref_path)
                    if ref.ndim == 1:
                        ref = ref.reshape(-1,)
                    # Accept either 1-col or 5-col (assume last col is K+I+H)
                    if ref.ndim == 2 and ref.shape[1] >= 1:
                        ref_vec = ref[:, -1]
                    else:
                        ref_vec = ref
                    mref = min(m, ref_vec.shape[0])
                    ours = out[:mref, 5]
                    diff = ours - ref_vec[:mref]
                    print(f"  k0_diag_check present; comparing first {mref} K+I+H+NL diagonals...")
                    print(f"    max|Δ|={np.max(np.abs(diff)):.5e}, rms|Δ|={np.sqrt(np.mean(diff**2)):.5e}")

                    # Constrained fit: fix a=1 for K and e=1 for V_NL; solve for b,d in ref - K - V_NL ≈ b*V_ion + d*V_H
                    Kcol   = out[:mref, 1]
                    VIcol  = out[:mref, 2]
                    VHcol  = out[:mref, 3]
                    VNLcol = out[:mref, 4]
                    y = ref_vec[:mref]
                    y_red = y - Kcol - VNLcol
                    X_red = np.stack([VIcol, VHcol], axis=1)
                    coeffs_red, residuals, rank, s = np.linalg.lstsq(X_red, y_red, rcond=None)
                    b, d = coeffs_red.tolist()
                    # Replace constrained fit with full LS: ref ≈ A*K + B*Vion + C*VH + D*VNL + E
                    ones = np.ones_like(Kcol)
                    X = np.stack([Kcol, VIcol, VHcol, VNLcol, ones], axis=1)
                    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                    A, B, C, D, E = coeffs.tolist()
                    y_fit = X @ coeffs
                    err_vec = y_fit - y
                    l2 = float(np.linalg.norm(err_vec))
                    rmse = float(np.sqrt(np.mean(err_vec**2)))
                    print("  Unconstrained fit: ref ≈ A*K + B*Vion + C*VH + D*VNL + E:")
                    print(f"    A={A:+.6f}  B={B:+.6f}  C={C:+.6f}  D={D:+.6f}  E={E:+.6f}")
                    print(f"    total_L2={l2:.6e}  rmse={rmse:.6e}")
                except Exception as e:
                    print(f"  Warning: failed to compare with k0_diag_check ({e})")

            # Vloc-only fit removed per user request
            
    except Exception as e:
        print(f"Warning: failed to write k=0 diagonals ({e})")

    print("\nDFT Hamiltonian calculation completed successfully!")
    print("="*60)

    # NO kin_ion.h5 IS WRITTEN HERE any more.  This CLI is the per-term
    # H(k=0) DIAGNOSTIC; the generator is ``python -m gw.kin_ion_io``, which
    # inherits the deck's sys_dim, distributes the k loop and stores V_H as
    # its own dataset.  Writing both from here is what let a 3D-truncated
    # file overwrite a 2D-truncated one — see the removal note above.
    print("kin_ion.h5 is NOT written by this diagnostic; "
          "use `python -m gw.kin_ion_io -i <deck> -o kin_ion.h5`.")

    timing.report(title="--- Timing (seconds) ---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
