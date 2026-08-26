"""
psp/dft_operators.py — Plane-wave DFT Hamiltonian: build, apply, differentiate.

Canonical module for all DFT Hamiltonian operations.  Other modules
(kin_ion_io, get_dipole_mtxels, davidson) should call these
functions rather than reimplementing operator construction.

Public API (Hamiltonian construction + application):
  HamiltonianK         — per-k data: T_diag, V_scf, G-indices, VNL Z/E, h_diag, mask
  setup_H_k            — build HamiltonianK from SymMaps path (GW pipeline)
  setup_H_k_from_kvec  — build HamiltonianK from k-vector (standalone/Davidson)
  apply_H_k            — fused JIT H|ψ⟩, psi_box donated; 2 ms on A100 for Si
  build_matrix_k       — full ⟨m|H|n⟩ matrix

Public API (G-vector layout):
  padded_gvectors       — the loader's FIXED-shape (nk, ngkmax, 3) table
                          + pad mask, as a PaddedGVectors.  THE ROUTE
                          every operator in this package takes.
  generate_gvectors_k_padded — per-k twin of the above
  generate_gvectors_k   — one k's RAGGED (ngk[ik], 3) G-list.  Retained
                          as the D10 comparison reference (and for
                          misc/ scripts); no production consumer left.

Public API (per-component builders):
  build_T_diag          — |k+G|² + G-indices from SymMaps
  build_T_diag_from_kvec — same from explicit k-vector + ecutwfc (no SymMaps)
  build_V_scf           — combine V_loc + V_H + V_xc into one array
  compute_V_H_and_V_xc  — @jax.jit: V_H (Poisson) + V_xc (PBE GGA) in 1.2 ms cached
  build_h_diag          — preconditioner diagonal: T + V_loc(G=0) + V_NL_diag
  build_vnl_kdata       — dense VNL projectors (Z, E) from vnl_ops

Public API (uniform kinetic gauge actions):
  apply_kinetic_velocity_to_ket — dT/dK applied to a ket block
  apply_kinetic_contact_to_ket  — d2T/dK2 applied to a ket block

V_scf = V_loc + V_H + V_xc is a single (nx,ny,nz) real-space potential.
The caller builds it from charge_density.py (V_xc, V_H) and
build_projectors_qe.py (V_loc), then passes it to setup_H_k*.

Normalization: ⟨m|O|n⟩ = Σ_{s,G} conj(ψ_m[s,G]) · (O ψ)_n[s,G].
No volume prefactors.  Ortho-FFT convention for local potentials:
(V ψ)_G = FFT_ortho(V(r) · IFFT_ortho(ψ_box))_G.

ngkmax padding: setup_H_k_from_kvec accepts ngkmax to pad all arrays
to uniform size.  Combined with warmup_jit() from davidson.py, this
ensures one JIT compilation serves all k-points.  Mask field on
HamiltonianK zeros padding in apply_H_k and build_matrix_k.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from common.fft_helpers import local_fftn3, local_ifftn3


# ═══════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HamiltonianK:
    """Everything needed to apply H at one k-point.

    Built by ``setup_H_k``, consumed by ``apply_H_k`` and ``build_matrix_k``.
    """
    # Kinetic: T|ψ⟩_G = T_diag[G] · ψ_G
    T_diag: jax.Array               # (nG,) float64 — |k+G|² in Ry

    # Local self-consistent potential: V_scf = V_loc + V_H + V_xc
    V_scf: jax.Array                # (nx, ny, nz) float64 — Ry

    # G-vector map: FFT-box indices for the valid G-vectors at this k
    Gx: jax.Array                   # (nG,) int32
    Gy: jax.Array                   # (nG,) int32
    Gz: jax.Array                   # (nG,) int32

    # Nonlocal: Kleinman–Bylander projectors (dense, all channels concatenated)
    vnl_Z: jax.Array                # (total_R, nG) complex128
    vnl_E: jax.Array                # (nspinor, nspinor, total_R, total_R) complex128

    # Preconditioner diagonal (for Davidson): h_diag = T + v_of_0 + V_NL_diag
    h_diag: jax.Array               # (nG_padded,) float64 — QE's g_psi convention

    # G-vector mask: True for valid, False for padding
    mask: jax.Array                  # (nG_padded,) bool

    nG: int                          # actual (unpadded) count
    fft_grid: tuple[int, int, int]


# ═══════════════════════════════════════════════════════════════════════
#  Poisson solver
# ═══════════════════════════════════════════════════════════════════════

def _poisson_reciprocal_geometry(
    fft_grid: tuple[int, int, int],
    bdot: jnp.ndarray,
    bvec: jnp.ndarray | None,
    blat: float | None,
    truncation_2d: bool,
):
    """One reciprocal-grid/G2 source for scalar and transverse Poisson."""
    nx2, ny2, nz2 = (int(s) for s in fft_grid)
    fx = jnp.fft.fftfreq(nx2) * nx2
    fy = jnp.fft.fftfreq(ny2) * ny2
    fz = jnp.fft.fftfreq(nz2) * nz2
    ix, iy, iz = fx[:, None, None], fy[None, :, None], fz[None, None, :]
    M = jnp.asarray(bdot, dtype=jnp.float64)
    G2 = (M[0, 0] * ix * ix + M[1, 1] * iy * iy + M[2, 2] * iz * iz
          + 2 * M[0, 1] * ix * iy + 2 * M[0, 2] * ix * iz
          + 2 * M[1, 2] * iy * iz)
    zero_mask = ((jnp.arange(nx2)[:, None, None] == 0)
                 & (jnp.arange(ny2)[None, :, None] == 0)
                 & (jnp.arange(nz2)[None, None, :] == 0))
    G2_safe = jnp.where(zero_mask, 1.0, G2)

    G_cart = None
    f2d = None
    if bvec is not None and blat is not None:
        B = jnp.asarray(bvec, dtype=jnp.float64) * float(blat)
        G_cart = jnp.stack(jnp.broadcast_arrays(
            ix * B[0, 0] + iy * B[1, 0] + iz * B[2, 0],
            ix * B[0, 1] + iy * B[1, 1] + iz * B[2, 1],
            ix * B[0, 2] + iy * B[1, 2] + iz * B[2, 2],
        ))
        if truncation_2d:
            zc = jnp.pi / B[2, 2]
            kxy = jnp.sqrt(G_cart[0] ** 2 + G_cart[1] ** 2)
            f2d = 1.0 - jnp.exp(-zc * kxy) * jnp.cos(G_cart[2] * zc)
    return G2_safe, zero_mask, G_cart, f2d


def poisson_potential_from_rhoG(
    rho_G: jnp.ndarray,
    bdot: jnp.ndarray,
    bvec: jnp.ndarray | None = None,
    blat: float | None = None,
    truncation_2d: bool = True,
    *,
    _geometry=None,
) -> jnp.ndarray:
    """Solve Poisson equation: V(G) = 8π ρ(G) / |G|² (G≠0), V(G=0) = 0.

    If truncation_2d=True, applies Ismail-Beigi 2D slab truncation:
        v_2D(G) = v_3D(G) × (1 − e^{−|G_xy|·L_z/2} · cos(G_z·L_z/2))
    """
    rho_G = jnp.asarray(rho_G)
    if rho_G.ndim < 3:
        raise ValueError(
            "rho_G must have at least three FFT axes; "
            f"got shape {tuple(int(s) for s in rho_G.shape)}")
    fft_grid = tuple(int(s) for s in rho_G.shape[-3:])
    geometry = (_poisson_reciprocal_geometry(
        fft_grid, bdot, bvec, blat, truncation_2d)
        if _geometry is None else _geometry)
    G2_safe, zero_mask, _G_cart, f2d = geometry

    V_G = 8.0 * jnp.pi * rho_G / G2_safe

    if f2d is not None:
        V_G = V_G * jnp.where(zero_mask, 0.0, f2d)

    V_G = V_G.at[..., 0, 0, 0].set(0.0)
    return jnp.real(local_ifftn3(V_G, axes=(-3, -2, -1), norm='ortho'))


def transverse_potential_from_current(
    current_r: jnp.ndarray,
    bdot: jnp.ndarray,
    bvec: jnp.ndarray,
    blat: float,
    truncation_2d: bool,
    *,
    tt_metric_sign: float,
) -> jnp.ndarray:
    r"""Periodic Coulomb-gauge direct field from ``J = j/c``.

    ``current_r`` has shape ``(3, nx, ny, nz)`` and contains the signed
    occupied Dirac-current vertices ``Psi^dagger alpha_i Psi``.  The result
    is the three real-space fields

    ``A_i(G) = s_TT v(G) (delta_ij - G_i G_j/G^2) J_j(G)``.

    The scalar :func:`poisson_potential_from_rhoG` remains the sole owner of
    ``v(G)``, including the Ry ``8*pi/G^2`` prefactor, the 2-D slab factor,
    FFT normalisation, and the periodic ``G=0`` zero.  The geometric
    projector comes from the public ``vcoul.transverse_projector`` SSOT used
    by finite-q photon tiles.  ``tt_metric_sign`` is required rather than
    respelled here; the caller passes ``vcoul.COULOMB_GAUGE_TT_SIGN``.
    Consequently an exchange mini-BZ/head value can never enter this direct
    periodic solve.
    """
    J_r = jnp.asarray(current_r, dtype=jnp.float64)
    if J_r.ndim != 4 or int(J_r.shape[0]) != 3:
        raise ValueError(
            "current_r must have shape (3,nx,ny,nz); "
            f"got {tuple(int(s) for s in J_r.shape)}")
    if not np.isfinite(float(tt_metric_sign)):
        raise ValueError(
            f"tt_metric_sign must be finite; got {tt_metric_sign!r}")

    geometry = _poisson_reciprocal_geometry(
        tuple(int(s) for s in J_r.shape[-3:]),
        bdot, bvec, blat, truncation_2d)
    G2_safe, zero_mask, G_cart, _f2d = geometry
    if G_cart is None:
        raise ValueError(
            "transverse current requires bvec and blat for its Cartesian "
            "projector")

    from ffi import _services
    _services.ensure_on_path()
    from vcoul import transverse_projector
    J_G = local_fftn3(J_r, axes=(-3, -2, -1), norm='ortho')
    projector = transverse_projector(
        jnp.moveaxis(G_cart, 0, -1),
        jnp.where(zero_mask, 0.0, G2_safe))
    # Spell this as an elementwise reduction, not an einsum: a rank-2×rank-1
    # contraction with a three-wide inner axis is not a GEMM and must not
    # acquire a cuBLAS autotune/compile family on production FFT grids.
    J_transverse_G = jnp.moveaxis(jnp.sum(
        projector * jnp.moveaxis(J_G, 0, -1)[..., None, :], axis=-1),
        -1, 0)
    J_transverse_G = jnp.where(
        (~zero_mask)[None, ...], J_transverse_G,
        jnp.zeros_like(J_transverse_G))

    sign = jnp.asarray(float(tt_metric_sign), dtype=jnp.float64)
    return poisson_potential_from_rhoG(
        sign * J_transverse_G, bdot, bvec, blat, truncation_2d,
        _geometry=geometry)


def _as_loader(wfn):
    """The ``WfnLoader`` behind ``wfn`` — never a second file handle if
    one can be avoided.

    Re-opening the file costs a fresh ``(ngktot, 3)`` G-table on the
    host (hundreds of MB at CrI3-class ``ngktot``), so the legacy
    ``WFNReader`` fallback exists only for tests that still hold the old
    reader object.
    """
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader as _WFNLoader
    if isinstance(wfn, _WFNLoader):
        return wfn
    return _WFNLoader(wfn._filename)


def generate_gvectors_k(kpoint_idx, sym, wfn, meta):
    """G-vectors for one k-point via SymMaps (GW path) — **ragged**.

    LEGACY / REFERENCE.  Every operator in ``psp`` and ``gw.kin_ion_io``
    now takes :func:`padded_gvectors`; this is kept because owner
    decision D10 gates the padded route *against* it, so the comparison
    needs both routes to stay callable from one process.

    Returns (Gk_crys, kpoint_crys): (ngk[ik], 3) int and (3,) float.

    Post-P5 migration: ``sym.get_gvecs_kfull`` moved into ``WfnLoader``;
    re-fetch the full-BZ G-table from the loader and slice the single
    requested k.  ``WfnLoader`` caches the table so this is cheap on
    repeated calls.

    The slice back to ``ngk[ik]`` is what makes every consumer's per-k
    kernel recompile once per DISTINCT ``ngk`` and blocks ``lax.scan``
    over k.  Consumers that can carry a mask should use
    :func:`padded_gvectors` instead — the loader's table is *already*
    the padded one, so the fixed-shape route is strictly less work.
    """
    loader = _as_loader(wfn)
    gvecs_full = loader.gvecs(k="full_bz")           # (n_full, ngkmax, 3) int32
    ngk_full = loader.ngk_valid(k="full_bz")         # (n_full,) int32
    kvecs_full = loader.kvecs(k="full_bz")           # paired reciprocal representative
    nk = int(kpoint_idx)
    kpoint_crys = jnp.asarray(kvecs_full[nk], dtype=jnp.float64)
    Gk_crys = jnp.asarray(gvecs_full[nk, : int(ngk_full[nk])], dtype=jnp.int32)
    return Gk_crys, kpoint_crys


@dataclass(frozen=True)
class PaddedGVectors:
    """Every k's paired representative/G-list at one fixed G shape.

    This is the *native* layout of ``WfnLoader.gvecs()``: the loader
    stores the G table padded to the file's ``ngkmax`` and reports the
    logical extent separately through ``ngk_valid()``.  ``kvecs`` is read
    from the same loader transaction and is the coordinate half explicitly
    paired with those G labels; physical ``k+G`` consumers must not replace
    it with an independently fetched symmetry representative.
    :func:`generate_gvectors_k` throws that away by slicing back to
    ``ngk[ik]``; this class hands it over intact, plus the 1/0 mask that
    makes the pad columns inert.

    Why the fixed shape is worth carrying a mask for
    ------------------------------------------------
    ``ngk`` differs between k, so a ragged G-list gives every per-k
    kernel a different operand shape.  That costs one JIT lowering per
    DISTINCT ``ngk`` (bounded by the IBZ k-count), it forces one device
    dispatch — and therefore one blocking readback — per k, and it makes
    ``lax.scan`` over the k loop illegal outright, since scan requires
    the carried shapes to be uniform.

    Pad-column contract
    -------------------
    Pad rows hold the FFT-box **pad sentinel**
    (:func:`common.gvec_fft_box.fft_box_pad_sentinel` — the Nyquist
    corner cell ``(nx//2, ny//2, nz//2)``, whose Miller index is
    ``(-nx/2, -ny/2, -nz/2)`` on the even grids BGW writes), which is
    still a *valid* FFT-box
    index: a consumer that forgets the mask does not crash, it silently
    adds ``ngkmax − ngk`` extra copies of ψ at that corner cell.  Every
    consumer must therefore multiply one factor of each contraction over
    G by :attr:`mask`.  The kernels that take a ``g_mask`` argument
    (``psp.get_DFT_mtxels.compute_kinetic_k`` /
    ``compute_local_V_k``, ``dft_operators.gather_psi_G``) already do
    exactly that.

    What the sentinel buys over the pre-2026-08 zero pad: zero rows are
    Miller ``(0,0,0)``, a component every physical G-sphere contains, so
    a forgotten mask was *indistinguishable* from a ragged list.  The
    sentinel cell is one no physical G occupies — enforced, not assumed,
    by :func:`common.gvec_fft_box.pad_gvecs_to_sentinel` — so
    ``gw.kin_ion_io`` can refuse the unmasked call outright.  It also
    makes the error LOUDER if it ever slips through: the corner carries
    the largest ``|k+G|²`` in the box, so a leaked pad row perturbs T by
    the maximum the grid allows instead of by ``|k|²``.

    On the arithmetic: appending exact zeros does not change a sum
    (``x + 0.0 == x`` in IEEE-754).  What a shape change does move is
    XLA's choice of reduction BLOCKING, so the ragged route already had a
    per-``ngk``-dependent association and this one makes the association
    UNIFORM across k.  Neither is "the" reference; agreement between them
    is what is checked (owner decision D10, gate 1e-12).
    """

    gvecs: np.ndarray        # (n_k, ngkmax, 3) int32, sentinel-padded
    mask: np.ndarray         # (n_k, ngkmax) float64, 1 valid / 0 pad
    ngk: np.ndarray          # (n_k,) int32 — logical extent per k
    kvecs: np.ndarray        # (n_k, 3) float64 — loader-paired representative

    @property
    def ngkmax(self) -> int:
        return int(self.gvecs.shape[1])

    @property
    def n_k(self) -> int:
        return int(self.gvecs.shape[0])

    def at(self, ik: int) -> tuple[np.ndarray, np.ndarray]:
        """``(G_pad, mask)`` for one k — host arrays, no device traffic."""
        j = int(ik)
        return self.gvecs[j], self.mask[j]


def padded_gvectors(wfn, *, k="full_bz") -> PaddedGVectors:
    """The loader's fixed-shape ``(n_k, ngkmax, 3)`` G table + pad mask.

    Costs nothing beyond what the ψ loader already builds: ``gvecs()``
    and ``ngk_valid()`` are memoised on the loader, and
    ``WfnLoader.box_index`` (which every FFT-box load calls) has already
    materialised the same table.  The only new array is the ``(n_k,
    ngkmax)`` f64 mask — 0.25 MB at MoS₂ 4×4, 23 MB at a 144-k /
    20000-G deck.

    ``k`` is any ``WfnLoader`` k-spec (``"full_bz"``, ``"ibz"``, or an
    explicit index list), so a rank can build the table for exactly the
    k it owns.  The returned carrier includes ``loader.kvecs(k=k)`` from
    the same row selection; no second wrapping or symmetry lookup occurs.
    """
    from common.gvec_fft_box import pad_mask as _pad_mask
    loader = _as_loader(wfn)
    gvecs = np.asarray(loader.gvecs(k=k), dtype=np.int32)
    ngk = np.asarray(loader.ngk_valid(k=k), dtype=np.int32)
    kvecs = np.asarray(loader.kvecs(k=k), dtype=np.float64)
    if gvecs.ndim != 3 or int(gvecs.shape[-1]) != 3:
        raise ValueError(
            "padded_gvectors: WfnLoader.gvecs must have shape "
            f"(n_k, ngkmax, 3); got {gvecs.shape}.")
    if ngk.shape != (int(gvecs.shape[0]),):
        raise ValueError(
            "padded_gvectors: WfnLoader.ngk_valid must have one logical "
            f"extent per G row; got {ngk.shape} for gvecs {gvecs.shape}.")
    if np.any(ngk < 0) or np.any(ngk > int(gvecs.shape[1])):
        raise ValueError(
            "padded_gvectors: WfnLoader.ngk_valid rows must lie in "
            f"[0,{int(gvecs.shape[1])}]; got {ngk.tolist()}.")
    if kvecs.shape != (int(gvecs.shape[0]), 3):
        raise ValueError(
            "padded_gvectors: WfnLoader.kvecs must carry the coordinate "
            f"half of the G table with shape ({int(gvecs.shape[0])}, 3); "
            f"got {kvecs.shape} for gvecs {gvecs.shape}.")
    # ``pad_mask`` is the same "which slots are real" expression the
    # loader's box_index and the ζ writer use — one definition.
    mask = _pad_mask(ngk, int(gvecs.shape[1])).astype(np.float64)
    return PaddedGVectors(gvecs=gvecs, mask=mask, ngk=ngk, kvecs=kvecs)


def generate_gvectors_k_padded(kpoint_idx, sym, wfn, meta):
    """Per-k twin of :func:`generate_gvectors_k`, fixed-shape.

    Returns ``(Gk_pad, g_mask, kpoint_crys)`` with ``Gk_pad`` at
    ``(ngkmax, 3)`` and ``g_mask`` at ``(ngkmax,)``.  Prefer
    :func:`padded_gvectors` when sweeping more than one k — it builds the
    mask once for the whole table instead of once per call.
    """
    tab = padded_gvectors(wfn, k="full_bz")
    G_pad, g_mask = tab.at(kpoint_idx)
    kpoint_crys = jnp.asarray(tab.kvecs[int(kpoint_idx)], dtype=jnp.float64)
    return jnp.asarray(G_pad, dtype=jnp.int32), g_mask, kpoint_crys


# ═══════════════════════════════════════════════════════════════════════
#  G-vector utilities
# ═══════════════════════════════════════════════════════════════════════

def build_G_cart(nx: int, ny: int, nz: int, B: np.ndarray) -> jax.Array:
    """Cartesian G-vector grid from FFT frequencies and lattice matrix B.

    B = blat * bvec (rows = reciprocal lattice vectors in bohr⁻¹).
    Returns (nx, ny, nz, 3) float64.
    """
    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing='ij')
    G_crys = np.stack([Gx, Gy, Gz], axis=-1).astype(float)
    return jnp.asarray(np.einsum('...i,ij->...j', G_crys, B), dtype=jnp.float64)


# ═══════════════════════════════════════════════════════════════════════
#  Per-component builders
# ═══════════════════════════════════════════════════════════════════════
#
# Each function constructs one piece of the Hamiltonian.
# They are called by setup_H_k and also available individually.

def compute_ngkmax(kpoints, bdot, ecutwfc, fft_grid):
    """Maximum number of G-vectors across all k-points.

    Pure integer counting — no large arrays stored.  Use this once at
    startup so every ``setup_H_k_from_kvec`` call can pad to the same
    ``ngkmax``, giving one JIT compilation for all k-points.

    Parameters
    ----------
    kpoints : (nk, 3) — k-points in crystal coordinates
    bdot : (3,3) — reciprocal metric in bohr⁻²
    ecutwfc : float — PW cutoff (Ry)
    fft_grid : (nx, ny, nz)
    """
    kpoints = np.asarray(kpoints, dtype=np.float64)
    bdot = np.asarray(bdot, dtype=np.float64)
    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])

    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1).astype(np.float64)

    ngk_max = 0
    for ik in range(len(kpoints)):
        KG = G_all + kpoints[ik][None, :]
        KG_sq = np.einsum("gi,ij,gj->g", KG, bdot, KG)
        ngk_max = max(ngk_max, int(np.sum(KG_sq <= ecutwfc)))
    return ngk_max


def build_T_diag(
    k_idx: int,
    wfn,
    sym,
    meta,
    *,
    gvectors: PaddedGVectors | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, np.ndarray]:
    """Kinetic diagonal |k+G|² and G-vector FFT-box indices (SymMaps path).

    Returns ``(T_diag, Gx, Gy, Gz, g_mask)``, all at the loader's FIXED
    ``ngkmax``: ``T_diag`` is ``(ngkmax,)`` float64 in Ry, ``Gx/Gy/Gz``
    are ``(ngkmax,)`` int32 FFT-box indices and ``g_mask`` is
    ``(ngkmax,)`` float64, 1 on the ``ngk[k]`` physical rows and 0 on the
    pad.

    Pad rows carry ``G = (0,0,0)``, so ``T_diag`` there is ``|k|²`` — a
    finite, physically-meaningless value that :func:`setup_H_k` replaces
    with its ``1e10`` preconditioner sentinel.  The mask is what makes
    the pad inert; it is returned rather than left implicit because
    ``(0,0,0)`` is a VALID box index that aliases Γ, so a consumer that
    drops it double-counts ψ(G=0) instead of crashing.

    Pass ``gvectors`` to reuse one :class:`PaddedGVectors` table across a
    k sweep instead of rebuilding it per k.
    """
    tab = padded_gvectors(wfn, k="full_bz") if gvectors is None else gvectors
    G_pad, g_mask = tab.at(k_idx)
    kvec = np.asarray(tab.kvecs[int(k_idx)], dtype=float)
    T_diag, Gx, Gy, Gz = _T_diag_from_G(
        np.asarray(G_pad, dtype=int), kvec, wfn.bdot)
    return T_diag, Gx, Gy, Gz, g_mask


def build_T_diag_from_kvec(
    kvec: np.ndarray,
    crystal,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Kinetic diagonal |k+G|² and G-vector indices (standalone, no SymMaps).

    Generates the PW sphere {G : |k+G|² ≤ ecutwfc} from the cutoff and
    reciprocal metric.  Use this for self-consistent DFT (Davidson) where
    SymMaps / WFN.h5 are not available.

    Parameters
    ----------
    kvec : (3,) — k-point in crystal coordinates
    crystal : CrystalData or WFNReader — needs bdot, ecutwfc, fft_grid
    """
    kvec = np.asarray(kvec, dtype=float)
    bdot = np.asarray(crystal.bdot, dtype=float)
    nx, ny, nz = crystal.fft_grid

    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1)

    KG = G_all.astype(float) + kvec[None, :]
    KG_sq = np.einsum("gi,ij,gj->g", KG, bdot, KG)
    mask = KG_sq <= crystal.ecutwfc
    Gk = G_all[mask]

    # Sort by |k+G|² (matches QE's convention at Gamma; convenient for
    # any k-point even though QE's ordering isn't strictly sorted for k≠0)
    order = np.argsort(KG_sq[mask])
    Gk = Gk[order]

    return _T_diag_from_G(Gk, kvec, bdot)


def _T_diag_from_G(
    Gk_int: np.ndarray,
    kvec: np.ndarray,
    bdot: np.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Common core: G-vectors + k-vector + metric → (T_diag, Gx, Gy, Gz)."""
    Gx = jnp.asarray(Gk_int[:, 0], dtype=jnp.int32)
    Gy = jnp.asarray(Gk_int[:, 1], dtype=jnp.int32)
    Gz = jnp.asarray(Gk_int[:, 2], dtype=jnp.int32)

    bdot = np.asarray(bdot, dtype=float)
    K_crys = np.asarray(Gk_int, dtype=float) + np.asarray(kvec, dtype=float)[None, :]
    T_diag = jnp.asarray(
        np.einsum("gi,ij,gj->g", K_crys, bdot, K_crys),
        dtype=jnp.float64,
    )
    return T_diag, Gx, Gy, Gz


@functools.partial(jax.jit, static_argnames=("truncation_2d", "blat"))
def compute_V_H_and_V_xc(
    rho_val: jax.Array,
    rho_core: jax.Array,
    rhog_core: jax.Array,
    G_cart: jax.Array,
    bdot: jax.Array,
    bvec: jax.Array,
    blat: float,
    *,
    truncation_2d: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Compute V_H and V_xc.  Returns (V_H_r, V_xc_r) both (nx, ny, nz) in Ry.

    Uses PBE by default.  For a different functional, call compute_V_xc
    from psp.xc directly.
    """
    from psp.xc import compute_V_xc, pbe_functional

    # ── V_H via Poisson ──
    rho_G_ortho = jnp.fft.fftn(rho_val, norm='ortho')
    V_H_r = jnp.real(poisson_potential_from_rhoG(
        rho_G_ortho, bdot, bvec, blat, truncation_2d=truncation_2d))

    # ── V_xc ──
    rho_total = rho_val + rho_core
    # Precise G-space total density (analytic core + FFT valence)
    rho_core_gridded = jnp.real(jnp.fft.ifftn(rhog_core))
    rho_G_total = jnp.fft.fftn(rho_total - rho_core_gridded) + rhog_core

    xc_fn, level = pbe_functional()
    V_xc_r = compute_V_xc(rho_total, rho_G_total, G_cart, xc_fn, level)

    return V_H_r, V_xc_r


def build_V_scf(
    V_loc_r: jax.Array,
    V_H_r: jax.Array | None = None,
    V_xc_r: jax.Array | None = None,
) -> jax.Array:
    """Combine the local potentials into V_scf = V_loc + V_H + V_xc.

    Each argument is (nx, ny, nz) in Ry on the FFT grid, or None to omit.

    Components
    ----------
    V_loc : ionic pseudopotential (local part), from build_local_ionic_potential_on_G_total
    V_H   : Hartree potential from valence density, from poisson_potential_from_rhoG
    V_xc  : exchange-correlation from ρ_val + ρ_core, from charge_density.build_V_xc
    """
    V_scf = jnp.asarray(V_loc_r, dtype=jnp.float64)
    if V_H_r is not None:
        V_scf = V_scf + jnp.asarray(V_H_r, dtype=jnp.float64)
    if V_xc_r is not None:
        V_scf = V_scf + jnp.asarray(V_xc_r, dtype=jnp.float64)
    return V_scf


def build_vnl_kdata(
    k_idx: int,
    vnl_setup,
    wfn,
    sym,
    meta,
    nspinor: int | None = None,
    gvectors: PaddedGVectors | None = None,
):
    """Dense VNL projectors (Z, E) for one k-point via vnl_ops.

    Returns (vnl_Z, vnl_E) where:
      vnl_Z : (total_R, ngkmax) — all channels × atoms × betas concatenated,
              already zero on the pad columns (see
              ``vnl_ops.build_vnl_kdata``), so no caller-side mask is needed
      vnl_E : (nspinor, nspinor, total_R, total_R) — block-diagonal D matrix
    """
    import psp.vnl_ops as vnl_ops

    if nspinor is None:
        nspinor = int(meta.nspinor)

    kdata = vnl_ops.build_vnl_kdata(k_idx, vnl_setup, wfn, sym, meta,
                                     gvectors=gvectors)
    return kdata.Z, kdata.E_super


@jax.jit
def gather_psi_G(psi_box, Gx, Gy, Gz, mask=None):
    """Gather sparse plane-wave coefficients from the FFT box.

    Parameters
    ----------
    psi_box : (nb, nspinor, nx, ny, nz) complex128
    Gx, Gy, Gz : (nG,) int32 FFT-box indices
    mask : (nG,) bool or float, optional

    Returns
    -------
    psi_G : (nb, nspinor, nG) complex128
    """
    psi_G = psi_box[:, :, Gx, Gy, Gz]
    if mask is None:
        return psi_G
    mask_f = jnp.asarray(mask, dtype=psi_G.dtype)[None, None, :]
    return psi_G * mask_f


def gather_psi_G_from_crys(psi_box, Gk_crys, mask=None):
    """Convenience wrapper around ``gather_psi_G`` for an integer G-list."""
    G_int = jnp.asarray(Gk_crys, dtype=jnp.int32)
    return gather_psi_G(psi_box, G_int[:, 0], G_int[:, 1], G_int[:, 2], mask)


def vnl_matrix_from_kdata(psi_box, Gk_crys, kdata, mask=None):
    """Convenience: <m|V_NL|n> from FFT-box states and prebuilt VNL k-data."""
    import psp.vnl_ops as vnl_ops

    psi_G = gather_psi_G_from_crys(psi_box, Gk_crys, mask)
    # Slice to physical spinor components if bispinor wavefunctions have
    # more spinor components than the E_super block-diagonal (e.g. 4 vs 2).
    nspinor_E = kdata.E_super.shape[0]
    if psi_G.shape[1] > nspinor_E:
        psi_G = psi_G[:, :nspinor_E, :]
    return vnl_ops.vnl_matrix(psi_G, kdata.Z, kdata.E_super)


def vnl_velocity_from_kdata(psi_box, Gk_crys, kdata, mask=None):
    """Convenience: dV_NL/dK_cart from FFT-box states and prebuilt VNL k-data."""
    import psp.vnl_ops as vnl_ops

    if kdata.dZ is None:
        raise ValueError("kdata.dZ is required for vnl_velocity_from_kdata")
    psi_G = gather_psi_G_from_crys(psi_box, Gk_crys, mask)
    # Slice to physical spinor components if bispinor wavefunctions have
    # more spinor components than the E_super block-diagonal (e.g. 4 vs 2).
    nspinor_E = kdata.E_super.shape[0]
    if psi_G.shape[1] > nspinor_E:
        psi_G = psi_G[:, :nspinor_E, :]
    return vnl_ops.vnl_velocity_matrix(psi_G, kdata.Z, kdata.dZ, kdata.E_super)


def build_h_diag(
    T_diag: jax.Array,
    V_loc_r: jax.Array,
    vnl_Z: jax.Array,
    vnl_E: jax.Array,
) -> jax.Array:
    """Hamiltonian diagonal for the Davidson preconditioner (QE convention).

    h_diag(G) = |k+G|² + V_loc(G=0) + V_NL_diag(G)

    - V_loc(G=0) = mean(V_loc_r): the G=0 Fourier component of the local
      ionic potential.  This is the `v_of_0` in QE's `setlocal.f90`.
      V_H and V_xc are NOT included (V_H(G=0)=0 by convention;
      V_xc(G=0) is small and not included in QE's h_diag).

    - V_NL_diag(G) = Σ_R |Z(R,G)|² E(R,R): the diagonal of the KB
      nonlocal potential in the plane-wave basis.

    Parameters
    ----------
    T_diag : (nG,) — |k+G|²
    V_loc_r : (nx, ny, nz) — ionic local potential (NOT V_scf)
    vnl_Z : (total_R, nG) — KB projectors
    vnl_E : (nspinor, nspinor, total_R, total_R) — D-matrix
    """
    v_of_0 = jnp.mean(V_loc_r)

    # V_NL diagonal: sum over spinor channels and projectors
    # ⟨G,s|V_NL|G,s⟩ = Σ_{R,Q} conj(Z[R,G]) E[s,s,R,Q] Z[Q,G]
    # For the diagonal, sum over s:
    nspinor = vnl_E.shape[0]
    vnl_diag = jnp.zeros(T_diag.shape[0], dtype=jnp.float64)
    for s in range(nspinor):
        # E_ss[R,Q] = vnl_E[s,s,R,Q]
        E_ss = vnl_E[s, s]  # (total_R, total_R)
        # Σ_R,Q conj(Z[R,G]) E_ss[R,Q] Z[Q,G]
        EZ = E_ss @ vnl_Z  # (total_R, nG)
        vnl_diag = vnl_diag + jnp.real(
            jnp.sum(jnp.conj(vnl_Z) * EZ, axis=0)
        )

    return T_diag + v_of_0 + vnl_diag


# ═══════════════════════════════════════════════════════════════════════
#  Setup: build HamiltonianK for one k-point
# ═══════════════════════════════════════════════════════════════════════

def setup_H_k(
    k_idx: int,
    V_scf: jax.Array,
    vnl_setup,
    wfn,
    sym,
    meta,
    V_loc_r: jax.Array | None = None,
    ngkmax: int | None = None,
    gvectors: PaddedGVectors | None = None,
) -> HamiltonianK:
    """Assemble all per-k Hamiltonian data (SymMaps path).

    Parameters
    ----------
    k_idx : index into the loader's full-BZ row order
    V_scf : (nx, ny, nz) — V_loc + V_H + V_xc, from build_V_scf
    vnl_setup : from vnl_ops.build_vnl_setup (k-independent, built once)
    wfn, sym, meta : standard LORRAX objects
    V_loc_r : (nx, ny, nz) — ionic local potential alone, for h_diag.
        If None, h_diag falls back to T_diag only.
    ngkmax : int, optional — pad beyond the loader's own ``ngkmax``.
        The G table is ALREADY fixed-shape at the file's ``ngkmax``, so
        this is only needed when a caller wants a still larger uniform
        size; a smaller value is refused rather than silently truncating
        a k's physical G-sphere.
    gvectors : PaddedGVectors, optional — reuse one table across a sweep.
    """
    tab = padded_gvectors(wfn, k="full_bz") if gvectors is None else gvectors
    T_diag, Gx, Gy, Gz, g_mask = build_T_diag(
        k_idx, wfn, sym, meta, gvectors=tab)
    nG_actual = int(np.count_nonzero(g_mask))
    nG_pad = int(g_mask.shape[0])

    if ngkmax is not None and int(ngkmax) < nG_pad:
        raise ValueError(
            f"setup_H_k: ngkmax={int(ngkmax)} is smaller than the loader's "
            f"own padded width {nG_pad}; truncating would drop physical "
            f"G-vectors at the k with the largest ngk.")
    if ngkmax is not None and int(ngkmax) > nG_pad:
        pad = int(ngkmax) - nG_pad
        T_diag = jnp.pad(T_diag, (0, pad), constant_values=0.0)
        Gx = jnp.pad(Gx, (0, pad), constant_values=0)
        Gy = jnp.pad(Gy, (0, pad), constant_values=0)
        Gz = jnp.pad(Gz, (0, pad), constant_values=0)
        g_mask = np.concatenate([g_mask, np.zeros(pad, dtype=g_mask.dtype)])

    mask = jnp.asarray(g_mask, dtype=jnp.bool_)
    # Pad rows hold G=(0,0,0), i.e. T = |k|² there.  Restore the 1e10
    # preconditioner sentinel the padded path has always published, so
    # anything reading ``HamiltonianK.T_diag`` sees the same thing.
    T_diag = jnp.where(mask, T_diag, jnp.asarray(1e10, dtype=T_diag.dtype))

    # Build VNL at padded size — one JIT trace for all k-points
    Gk_int = np.stack([np.asarray(Gx), np.asarray(Gy), np.asarray(Gz)], axis=-1)
    kvec = np.asarray(tab.kvecs[int(k_idx)], dtype=float)
    import psp.vnl_ops as vnl_ops
    kdata = vnl_ops.build_vnl_kdata_from_kvec(kvec, Gk_int, vnl_setup)

    # Tail-G mask: padded Z entries are non-zero (computed at K=kvec) —
    # zero them so apply_vnl / build_h_diag never see spurious overlap.
    vnl_Z = jnp.where(mask[None, :], kdata.Z, jnp.zeros((), dtype=kdata.Z.dtype))

    h_diag = (build_h_diag(T_diag, V_loc_r, vnl_Z, kdata.E_super)
              if V_loc_r is not None else T_diag)
    h_diag = jnp.where(mask, h_diag, jnp.asarray(1e10, dtype=h_diag.dtype))

    return HamiltonianK(
        T_diag=T_diag,
        V_scf=V_scf,
        Gx=Gx, Gy=Gy, Gz=Gz,
        vnl_Z=vnl_Z,
        vnl_E=kdata.E_super,
        h_diag=h_diag,
        mask=mask,
        nG=nG_actual,
        fft_grid=tuple(int(x) for x in meta.fft_grid),
    )


def setup_H_k_from_kvec(
    kvec: np.ndarray,
    V_scf: jax.Array,
    vnl_setup,
    crystal,
    meta,
    V_loc_r: jax.Array | None = None,
    ngkmax: int | None = None,
) -> HamiltonianK:
    """Assemble per-k Hamiltonian data (standalone, no SymMaps / WFN.h5).

    Parameters
    ----------
    kvec : (3,) — k-point in crystal coordinates
    V_scf : (nx, ny, nz) — combined local potential
    vnl_setup : from vnl_ops.build_vnl_setup
    crystal : CrystalData or any object with bdot, ecutwfc, fft_grid
    meta : Meta object (for nspinor)
    V_loc_r : (nx, ny, nz) — ionic local potential alone, for h_diag.
    ngkmax : int, optional — pad all arrays to this size for uniform JIT.
        If None, no padding (arrays have natural nG length).
    """
    import psp.vnl_ops as vnl_ops

    T_diag, Gx, Gy, Gz = build_T_diag_from_kvec(kvec, crystal)
    nG_actual = int(Gx.shape[0])

    # Pre-pad Gk_int + per-G arrays to ngkmax BEFORE the vnl call so
    # build_vnl_kdata_from_kvec sees a shape-stable G-sphere across all
    # k-points — its assembly (the _assemble_Z_jit kernel) then compiles
    # once and is reused for every k.  Tail-G entries hold (0,0,0); Z is
    # computed there at K=kvec (finite, in-table values) and zeroed
    # post-call via ``mask``.
    if ngkmax is not None and ngkmax > nG_actual:
        pad = ngkmax - nG_actual
        T_diag = jnp.pad(T_diag, (0, pad), constant_values=1e10)
        Gx = jnp.pad(Gx, (0, pad), constant_values=0)
        Gy = jnp.pad(Gy, (0, pad), constant_values=0)
        Gz = jnp.pad(Gz, (0, pad), constant_values=0)
        mask = jnp.concatenate([jnp.ones(nG_actual, dtype=jnp.bool_),
                                jnp.zeros(pad, dtype=jnp.bool_)])
    else:
        mask = jnp.ones(nG_actual, dtype=jnp.bool_)

    Gk_int = np.stack([np.asarray(Gx), np.asarray(Gy), np.asarray(Gz)], axis=-1)
    kdata = vnl_ops.build_vnl_kdata_from_kvec(kvec, Gk_int, vnl_setup)

    # Tail-G mask: padded Z entries are non-zero (computed at K=kvec) —
    # mask them so apply_vnl / Q-projections never see spurious overlap.
    vnl_Z = jnp.where(mask[None, :], kdata.Z, jnp.zeros((), dtype=kdata.Z.dtype))
    vnl_E = kdata.E_super

    h_diag = (build_h_diag(T_diag, V_loc_r, vnl_Z, vnl_E)
              if V_loc_r is not None else T_diag)
    # Force h_diag = 1e10 at padded entries (T_diag is already 1e10
    # there; build_h_diag adds the constant v_of_0, so this snap-back
    # restores the old sentinel exactly for any preconditioner that
    # tests against it).
    h_diag = jnp.where(mask, h_diag, jnp.asarray(1e10, dtype=h_diag.dtype))

    return HamiltonianK(
        T_diag=T_diag,
        V_scf=V_scf,
        Gx=Gx, Gy=Gy, Gz=Gz,
        vnl_Z=vnl_Z,
        vnl_E=vnl_E,
        h_diag=h_diag,
        mask=mask,
        nG=nG_actual,
        fft_grid=tuple(int(x) for x in crystal.fft_grid),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Core fused JIT kernels
# ═══════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, donate_argnums=(0,))
def apply_H_k(psi_box, T_diag, V_scf, Gx, Gy, Gz, vnl_Z, vnl_E, mask):
    """H|ψ⟩ = (T + V_scf + V_NL)|ψ⟩.  Single fused JIT dispatch.

    The input psi_box is **donated** (its buffer is reused by XLA).
    After this call the caller must not read psi_box.

    Padding G-vectors (where mask=False) are zeroed in the output.

    Parameters
    ----------
    psi_box  : (nvec, nspinor, nx, ny, nz) complex128
    T_diag   : (nG_padded,) float64
    V_scf    : (nx, ny, nz) float64
    Gx,Gy,Gz : (nG_padded,) int32
    vnl_Z    : (total_R, nG_padded) complex128
    vnl_E    : (nspinor, nspinor, total_R, total_R) complex128
    mask     : (nG_padded,) bool — True for valid G-vectors

    Returns
    -------
    H_psi : (nvec, nspinor, nG_padded) complex128 — sparse-G (padding zeroed)
    """
    mask_f = mask[None, None, :].astype(psi_box.dtype)
    psi_G = psi_box[:, :, Gx, Gy, Gz] * mask_f

    # ── T: kinetic energy (diagonal in G-space) ──────────────────────
    H_G = T_diag[None, None, :] * psi_G

    # ── V_scf: self-consistent local potential (real-space multiply) ─
    psi_r = local_ifftn3(psi_box, axes=(-3, -2, -1), norm='ortho')
    H_G = H_G + local_fftn3(
       psi_r * V_scf, axes=(-3, -2, -1), norm='ortho'
   )[:, :, Gx, Gy, Gz] * mask_f

    # ── V_NL: Kleinman–Bylander (project → D → unproject) ───────────
    P = jnp.einsum('RG,vsG->Rsv', jnp.conj(vnl_Z), psi_G, optimize=True)
    D = jnp.einsum('stRQ,Qtv->Rsv', vnl_E, P, optimize=True)
    H_G = H_G + jnp.einsum('RG,Rsv->vsG', vnl_Z, D, optimize=True) * mask_f

    return H_G


def apply(psi_box: jax.Array, H_k: HamiltonianK) -> jax.Array:
    """Convenience: H|ψ⟩ from a HamiltonianK dataclass."""
    return apply_H_k(
        psi_box, H_k.T_diag, H_k.V_scf,
        H_k.Gx, H_k.Gy, H_k.Gz,
        H_k.vnl_Z, H_k.vnl_E, H_k.mask,
    )


@jax.jit
def apply_H_k_from_G(psi_G, T_diag, V_scf, Gx, Gy, Gz, vnl_Z, vnl_E, mask):
    """H|ψ⟩ where the input is the **G-sphere** representation (not the FFT
    box).  Skips the redundant ``scatter→gather`` round trip used by
    ``_apply_A_inline``: callers that already have ψ in G-form (e.g. CG
    iterates) can use this directly, paying for ONE scatter (only for the
    FFT path) and ONE gather (only after the FFT path), instead of one
    scatter (caller-side) plus two gathers (apply_H_k internal).

    Saves one O(nG) operation per Sternheimer matvec — ~5–10 % of the inner
    CG body cost on A100 with the MoS2 3×3 batch shapes.

    Parameters
    ----------
    psi_G : (nvec, nspinor, nG_padded) complex128
    All other args as in ``apply_H_k``.
    """
    nx, ny, nz = V_scf.shape
    mask_f = mask[None, None, :].astype(psi_G.dtype)
    psi_G_m = psi_G * mask_f

    # T·ψ on the G-sphere directly — no scatter/gather needed.
    H_G = T_diag[None, None, :] * psi_G_m

    # V_scf path: scatter ψ_G → box, FFT, multiply by V_scf, FFT back, gather.
    # NOTE: must use ``.add()``, not ``.set()`` — under ngkmax padding, all
    # padded G-indices map to (0,0,0) which collides with the real G=0
    # entry.  ``.set()`` overwrites in undefined order and zeros out the
    # physical G=0 coefficient; ``.add()`` accumulates (padded values are
    # already masked to 0 above) so the real entry survives.
    psi_box = jnp.zeros((*psi_G.shape[:2], nx, ny, nz), dtype=psi_G.dtype)
    psi_box = psi_box.at[:, :, Gx, Gy, Gz].add(psi_G_m)
    psi_r = local_ifftn3(psi_box, axes=(-3, -2, -1), norm='ortho')
    Vpsi_box = local_fftn3(psi_r * V_scf, axes=(-3, -2, -1), norm='ortho')
    H_G = H_G + Vpsi_box[:, :, Gx, Gy, Gz] * mask_f

    # V_NL on the G-sphere directly.
    P = jnp.einsum('RG,vsG->Rsv', jnp.conj(vnl_Z), psi_G_m, optimize=True)
    D = jnp.einsum('stRQ,Qtv->Rsv', vnl_E, P, optimize=True)
    H_G = H_G + jnp.einsum('RG,Rsv->vsG', vnl_Z, D, optimize=True) * mask_f

    return H_G


@jax.jit
def build_matrix_k(psi_box, T_diag, V_scf, Gx, Gy, Gz, vnl_Z, vnl_E, mask):
    """Full ⟨m|H|n⟩ matrix at one k-point.  Returns (nb, nb) complex128."""
    import psp.vnl_ops as vnl_ops

    mask_f = mask[None, None, :].astype(psi_box.dtype)
    psi_G = psi_box[:, :, Gx, Gy, Gz] * mask_f

    # T
    H_mn = jnp.einsum(
        'msG,nsG->mn', jnp.conj(psi_G),
        T_diag[None, None, :] * psi_G, optimize=True,
    )

    # V_scf
    psi_r = local_ifftn3(psi_box, axes=(-3, -2, -1), norm='ortho')
    Vpsi_G = local_fftn3(
       psi_r * V_scf, axes=(-3, -2, -1), norm='ortho'
   )[:, :, Gx, Gy, Gz] * mask_f
    H_mn = H_mn + jnp.einsum(
        'msG,nsG->mn', jnp.conj(psi_G), Vpsi_G, optimize=True,
    )

    # V_NL
    H_mn = H_mn + vnl_ops.vnl_matrix(psi_G, vnl_Z, vnl_E)

    return H_mn


def matrix(psi_box: jax.Array, H_k: HamiltonianK) -> jax.Array:
    """Convenience: ⟨m|H|n⟩ from a HamiltonianK dataclass."""
    return build_matrix_k(
        psi_box, H_k.T_diag, H_k.V_scf,
        H_k.Gx, H_k.Gy, H_k.Gz,
        H_k.vnl_Z, H_k.vnl_E, H_k.mask,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Velocity / dipole matrix elements
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def apply_kinetic_velocity_to_ket(psi_G, G_int, k_crys, B):
    """v_kin^α |n,k⟩ on the same G-sphere as ``psi_G``.  Returns
    ``(3, nb, nspinor, nG)`` complex.

    The kinetic velocity in Rydberg atomic units is
    ``v_kin = 2(k + G)_cart`` (i.e. ``-i∇`` with the factor-of-2
    Rydberg-momentum convention used by the rest of LORRAX); see
    :func:`momentum_matrix_k` for the q=0 matrix-element wrapper.
    Pulling the apply step out lets the q=0 dipole path AND the
    finite-q SOS pipeline share a single source of truth for the
    velocity operator.
    """
    K_cart = (G_int.astype(jnp.float64) + k_crys[None, :]) @ B            # (nG, 3)
    return 2.0 * jnp.einsum('Gi,nsG->insG', K_cart, psi_G, optimize=True)


def apply_kinetic_contact_to_ket(psi_G):
    r"""``d2T/dK_a dK_b |psi> = 2 delta_ab |psi>`` in Ry units."""
    eye = 2.0 * jnp.eye(3, dtype=psi_G.real.dtype)
    return jnp.einsum('ab,nsG->abnsG', eye, psi_G, optimize=True)


@jax.jit
def momentum_matrix_k(psi_G, G_int, k_crys, B):
    """Kinetic part of velocity: p_i = 2(k+G)_i.  Returns (3, nb, nb).

    Thin bra-contraction wrapper around :func:`apply_kinetic_velocity_to_ket`.
    """
    v_ket = apply_kinetic_velocity_to_ket(psi_G, G_int, k_crys, B)         # (3, nb, ns, nG)
    return jnp.einsum('msG,insG->imn', jnp.conj(psi_G), v_ket,
                      optimize=True)
