"""Canonical wavefunction storage for ISDF-basis GW calculations.

Four device-distributed copies of ψ_nk(r_μ), one for each combination of
{device axis} × {memory layout}:

  psi_xn : (nk, s, μ_X, n)  bands fast, μ on X  →  G/χ₀ LHS (conj)
  psi_xr : (nk, n, s, μ_X)  centroids fast, μ on X  →  Σ projection LHS (conj)
  psi_yr : (nk, n, s, μ_Y)  centroids fast, μ on Y  →  G/χ₀ RHS
  psi_yn : (nk, s, μ_Y, n)  bands fast, μ on Y  →  Σ projection RHS

In all four layouts the spinor index s sits adjacent to the centroid
index μ, so contractions that sum over (s, μ) pairs sweep contiguous memory.
All four copies store the *un-conjugated* ψ; consumers that need ψ*
apply :func:`jnp.conj` themselves.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ---------------------------------------------------------------------------
# Band-edge bookkeeping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandSlices:
    """Precomputed local slices from the five canonical band edges.

    Band edges (global indices):
        b0  lowest band in the calculation
        b1  start of mixed valence/conduction sigma region
        b2  LUMO (first unoccupied)
        b3  end of the sigma/QP evaluation window
        b4  highest band (end of full computational window)

    All slices are LOCAL (relative to b0).
    """
    b0: int
    b1: int
    b2: int
    b3: int
    b4: int
    val:   slice   # [0, b2-b0)       valence
    cond:  slice   # [b2-b0, b4-b0)   conduction
    sigma: slice   # [0, b3-b0)       QP evaluation window
    full:  slice   # [0, b4-b0)       all bands
    occ:   slice   # [0, b2-b0)       occupied

    @classmethod
    def from_band_edges(cls, b0: int, b1: int, b2: int, b3: int, b4: int) -> BandSlices:
        if not (b0 <= b1 <= b2 <= b3 <= b4):
            raise ValueError(f"Invalid band edges: {(b0, b1, b2, b3, b4)}")
        nb_full = b4 - b0
        return cls(
            b0=b0, b1=b1, b2=b2, b3=b3, b4=b4,
            val=slice(0, b2 - b0),
            cond=slice(b2 - b0, nb_full),
            sigma=slice(0, b3 - b0),
            full=slice(0, nb_full),
            occ=slice(0, b2 - b0),
        )

    @property
    def nb_full(self) -> int:
        return self.b4 - self.b0

    @property
    def nb_sigma(self) -> int:
        return self.b3 - self.b0

    @property
    def sigma_range(self) -> tuple[int, int]:
        """Global (start, end) for sigma band window: (b0, b3)."""
        return (self.b0, self.b3)

    @property
    def full_range(self) -> tuple[int, int]:
        """Global (start, end) for full band window: (b0, b4)."""
        return (self.b0, self.b4)


# ---------------------------------------------------------------------------
# Sharding specs for the four ψ copies (2-D mesh with axes 'x', 'y').
# ---------------------------------------------------------------------------
PSI_XN_SPEC = P(None, None, 'x', None)   # (nk, s, μ_X, n)
PSI_XR_SPEC = P(None, None, None, 'x')   # (nk, n, s, μ_X)
PSI_YR_SPEC = P(None, None, None, 'y')   # (nk, n, s, μ_Y)
PSI_YN_SPEC = P(None, None, 'y', None)   # (nk, s, μ_Y, n)

# ---------------------------------------------------------------------------
# Sharding specs for the intermediate tensors that flow between chi0 / W /
# sigma / cohsex kernels.  Kept here (not in each consumer) so every
# module sees the same canonical layout and a reshard-mismatch is caught
# at import time rather than at HLO compile.
# ---------------------------------------------------------------------------
# G(k) in 7-D FFT-box form, used by chi0's build_G and sigma's Gij
# construction.  (nkx, nky, nkz, s, μ_X, spinor, μ_Y) with μ on the
# (x, y) mesh.  The s-axis and spinor-axis are replicated because they
# sum contiguously in downstream einsums.
G_FFT7D_SPEC = P(None, None, None, None, 'x', None, 'y')

# V_q / W_q in 5-D k-space form: (nkx, nky, nkz, μ_X, μ_Y) both μ-axes
# sharded.  Used by compute_vcoul, w_isdf's V_q factory, and the W_q
# input to sigma.
V_FFT5D_SPEC = P(None, None, None, 'x', 'y')

# chi(q) / σ^τ(k, μ, ν) in 5-D flat-q or k-sharded form: both μ axes on
# the (x, y) mesh, q/k replicated.  Used by the inner tau kernel and
# the chi0 → W solve.
CHI_Q_SPEC = P(None, None, None, 'x', 'y')

# G(k) in 5-D flat-k form — the output of make_flat_k_fftn when the
# upstream op operates on a 3-D k-grid view.  (nk_flat, s, μ_X, spinor,
# μ_Y).
G_FLATK_SPEC = P(None, None, 'x', None, 'y')

# chi / W in R-space, flat-k form, used by the chi0-to-W pipeline post
# iFFT: (nk_flat, μ_X, μ_Y).
CHI_R_SPEC = P(None, 'x', 'y')


# ---------------------------------------------------------------------------
# Wavefunction storage
# ---------------------------------------------------------------------------

@dataclass
class Wavefunctions:
    """Four device-distributed copies of ψ_nk(r_μ) spanning [b0, b4)."""

    psi_xn: jax.Array   # (nk, s, μ_X, n)
    psi_xr: jax.Array   # (nk, n, s, μ_X)
    psi_yr: jax.Array   # (nk, n, s, μ_Y)
    psi_yn: jax.Array   # (nk, s, μ_Y, n)
    enk: jax.Array       # (nk, nb_full) replicated
    occ: jax.Array       # (nk, nb_full) replicated
    slices: BandSlices

    # Slice accessors — bands is a Python ``slice`` (hashable in 3.12+) so
    # jit can take it as static_argname.  Without these jits each accessor
    # call (used heavily by chi/W/Σ) emits a fresh eager-pjit ``gather``,
    # producing a tail of cache misses (~17/run on Si 4×4×4).
    @functools.partial(jax.jit, static_argnames=('bands',))
    def xn(self, bands: slice) -> jax.Array:
        return self.psi_xn[:, :, :, bands]

    @functools.partial(jax.jit, static_argnames=('bands',))
    def xr(self, bands: slice) -> jax.Array:
        return self.psi_xr[:, bands, :, :]

    @functools.partial(jax.jit, static_argnames=('bands',))
    def yr(self, bands: slice) -> jax.Array:
        return self.psi_yr[:, bands, :, :]

    @functools.partial(jax.jit, static_argnames=('bands',))
    def yn(self, bands: slice) -> jax.Array:
        return self.psi_yn[:, :, :, bands]


# Register as JAX pytree so Wavefunctions can be passed to @jax.jit functions.
# Array fields are traced; slices (static metadata) are compile-time constants.
jax.tree_util.register_dataclass(
    Wavefunctions,
    data_fields=['psi_xn', 'psi_xr', 'psi_yr', 'psi_yn', 'enk', 'occ'],
    meta_fields=['slices'],
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def _build_occ(enk_full, slices, efermi):
    """Build occupation array (nk, nb_full), float64 in {0.0, 1.0}.

    ─ NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS INTENTIONAL ─
    ``enk_full`` is tiny (nk × nb, usually < 10K doubles).  The original
    all-``jnp`` version did ``jnp.zeros_like(sharded_input) .at[].set``,
    which not only emitted 8 standalone pjits at trace time but also
    had a **non-trivial runtime cost** tied to the cross-device
    scatter — the ``wavefunction_setup`` timing section dropped
    1.79 s → 0.18 s when this function was converted (commit 7781b80,
    2026-04-18).  The D2H on ``enk_full`` is sub-ms, numpy arithmetic
    is immediate, and the ``jnp.asarray`` at the end is a cheap H2D of
    a small replicated array.  DO NOT "fix" back to ``jnp``.
    """
    enk_host = np.asarray(enk_full)
    if efermi is None:
        occ = np.zeros_like(enk_host, dtype=np.float64)
        occ[:, slices.occ] = 1.0
    else:
        occ = (enk_host <= float(efermi)).astype(np.float64)
    return jnp.asarray(occ)


def build_wavefunctions(
    psi_rmu_Y, psi_rmuT_X, *, enk_full, slices, mesh_xy, efermi=None,
) -> Wavefunctions:
    """Assemble the four-copy ``Wavefunctions`` bundle from the two
    centroid-sampled arrays produced by ``load_centroids_band_chunked``.

    Both inputs cover the full band range [b0, b4).

    ``psi_rmu_Y``   (nk, nb, ns, n_rmu)   P(None, None, None, 'y')
                    un-conjugated ψ.
    ``psi_rmuT_X``  (nk, n_rmu, nb, ns)   P(None, 'x', None, None)
                    conjugated ψ* (layout picked by the ISDF pair-density
                    kernel); a single :func:`jnp.conj` here undoes that
                    convention to match the bundle-wide un-conjugated
                    storage.

    No cross-device reshards are emitted: the y-sharded copies are
    derived from ``psi_rmu_Y`` and the x-sharded copies from
    ``psi_rmuT_X``.  Each transpose preserves the μ axis's sharding.
    """
    with mesh_xy:
        # y-sharded copies — both from psi_rmu_Y.
        psi_yr = jax.lax.with_sharding_constraint(
            psi_rmu_Y, NamedSharding(mesh_xy, PSI_YR_SPEC))
        psi_yn = jax.lax.with_sharding_constraint(
            psi_rmu_Y.transpose(0, 2, 3, 1),
            NamedSharding(mesh_xy, PSI_YN_SPEC))

        # x-sharded copies — conj to undo the pair-density kernel's ψ*.
        psi_X = jnp.conj(psi_rmuT_X)
        psi_xn = jax.lax.with_sharding_constraint(
            psi_X.transpose(0, 3, 1, 2),    # (nk, ns, μ_X, nb)
            NamedSharding(mesh_xy, PSI_XN_SPEC))
        psi_xr = jax.lax.with_sharding_constraint(
            psi_X.transpose(0, 2, 3, 1),    # (nk, nb, ns, μ_X)
            NamedSharding(mesh_xy, PSI_XR_SPEC))

        occ_full = _build_occ(enk_full, slices, efermi)
        rep2 = NamedSharding(mesh_xy, P(None, None))
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(occ_full, rep2)

    return Wavefunctions(
        psi_xn=psi_xn, psi_xr=psi_xr, psi_yr=psi_yr, psi_yn=psi_yn,
        enk=enk_full, occ=occ_full, slices=slices,
    )


# ---------------------------------------------------------------------------
# Self-consistent QSGW: rotate the bundle into a new band basis
# ---------------------------------------------------------------------------

@jax.jit
def _rotate_psi_bandlast(psi: jax.Array, U: jax.Array) -> jax.Array:
    # band-last layout (k,s,μ,n): ψ'[k,s,μ,n] = Σ_m ψ[k,s,μ,m]·U[k,m,n]  (xn, yn)
    return jnp.einsum('ksum,kmn->ksun', psi, U, optimize=True)


@jax.jit
def _rotate_psi_bandfirst(psi: jax.Array, U: jax.Array) -> jax.Array:
    # band-first layout (k,m,s,μ): ψ'[k,n,s,μ] = Σ_m U[k,m,n]·ψ[k,m,s,μ]  (xr, yr)
    return jnp.einsum('kmn,kmsu->knsu', U, psi, optimize=True)


def rotate_wavefunctions(
    wfns_dft: Wavefunctions,
    U_dft_to_qp_active: jax.Array,
    *,
    enk_active_new: jax.Array,
    efermi: float | None,
    mesh_xy: Mesh,
    active_slice: slice | None = None,
) -> Wavefunctions:
    """Return a new ``Wavefunctions`` bundle with the **active subspace**
    rotated by ``U_dft_to_qp_active[k, m, n] = ⟨DFT_m | QP_n⟩``.

    Active / inactive partition
    ---------------------------
    The ``active_slice`` (default: ``wfns_dft.slices.sigma`` — the QP
    evaluation window) selects a contiguous band block ``[start, stop)``
    where the QP Hamiltonian has full off-diagonal Σ and we apply the
    band-mixing unitary.  Bands **outside** that window keep their DFT
    wavefunctions and DFT energies untouched; their QP corrections come
    from the scissor extrapolation downstream.  This avoids rotating ψ
    for bands the QP calculation never touched (which is both wasteful
    and physically wrong since we have no Σ for them).

    Validation
    ----------
    Errors if ``active_slice`` extends past ``wfns_dft.slices.sigma`` —
    that would imply we're rotating bands the calculation can't have
    produced QP-basis information for.

    Parameters
    ----------
    wfns_dft
        The original DFT bundle (preserved unchanged across iterations).
    U_dft_to_qp_active
        Per-k unitary on the active block, shape ``(nk, nb_active, nb_active)``.
    enk_active_new
        New eigenvalues on the active block, shape ``(nk, nb_active)``.
    efermi
        Fermi level; used to rebuild ``occ``.
    mesh_xy
        2-D device mesh; sharding of the four ψ copies is preserved.
    active_slice
        Contiguous active band block.  Defaults to ``wfns_dft.slices.sigma``.
    """
    sigma_slice = wfns_dft.slices.sigma
    if active_slice is None:
        active_slice = sigma_slice
    a_lo = int(active_slice.start or 0)
    a_hi = int(active_slice.stop)
    s_lo = int(sigma_slice.start or 0)
    s_hi = int(sigma_slice.stop)
    if a_lo < s_lo or a_hi > s_hi:
        raise ValueError(
            f"rotate_wavefunctions: active_slice [{a_lo}, {a_hi}) leaks "
            f"outside the σ-window [{s_lo}, {s_hi}); we have no QP basis "
            f"information for bands beyond the protected window.")
    nb_active = a_hi - a_lo
    if U_dft_to_qp_active.shape[-2:] != (nb_active, nb_active):
        raise ValueError(
            f"rotate_wavefunctions: U shape {U_dft_to_qp_active.shape} "
            f"inconsistent with active block size {nb_active}.")

    # Pull the active sub-blocks via the bundle's jit'd accessors (cached),
    # rotate, then dynamic-update-slice back into a copy of the full ψ.
    with mesh_xy:
        psi_xn_act = _rotate_psi_bandlast(wfns_dft.xn(active_slice), U_dft_to_qp_active)
        psi_xr_act = _rotate_psi_bandfirst(wfns_dft.xr(active_slice), U_dft_to_qp_active)
        psi_yr_act = _rotate_psi_bandfirst(wfns_dft.yr(active_slice), U_dft_to_qp_active)
        psi_yn_act = _rotate_psi_bandlast(wfns_dft.yn(active_slice), U_dft_to_qp_active)

        # Reassemble — bands outside the active block stay DFT.
        psi_xn = jax.lax.dynamic_update_slice_in_dim(
            wfns_dft.psi_xn, psi_xn_act, a_lo, axis=-1)
        psi_xr = jax.lax.dynamic_update_slice_in_dim(
            wfns_dft.psi_xr, psi_xr_act, a_lo, axis=1)
        psi_yr = jax.lax.dynamic_update_slice_in_dim(
            wfns_dft.psi_yr, psi_yr_act, a_lo, axis=1)
        psi_yn = jax.lax.dynamic_update_slice_in_dim(
            wfns_dft.psi_yn, psi_yn_act, a_lo, axis=-1)

        # enk: copy DFT, replace the active block with the new eigenvalues.
        enk_full = wfns_dft.enk.at[:, active_slice].set(
            jnp.asarray(enk_active_new, dtype=wfns_dft.enk.dtype))
        rep2 = NamedSharding(mesh_xy, P(None, None))
        enk_full = jax.lax.with_sharding_constraint(enk_full, rep2)
        occ_full = jax.lax.with_sharding_constraint(
            _build_occ(enk_full, wfns_dft.slices, efermi), rep2)

    return Wavefunctions(
        psi_xn=psi_xn, psi_xr=psi_xr, psi_yr=psi_yr, psi_yn=psi_yn,
        enk=enk_full, occ=occ_full, slices=wfns_dft.slices,
    )


# ---------------------------------------------------------------------------
# Band-basis projection — Σ_mn(k) = Σ_{s,μ,s',μ'} ψ*_m(s,μ) Σ(s,μ,s',μ') ψ_n(s',μ')
#
# Lives here because the only state these contractions need is the (xr, yn)
# pair of sharded ψ copies that the bundle owns; consumers (cohsex_sigma,
# the AOT memory model) operate at the bundle's seam.
# ---------------------------------------------------------------------------

def project(psi_xr, psi_yn, sigma_k):
    """Σ(nk, s, μ, s, μ) → Σ(nk, m, n) in band basis."""
    left = jnp.einsum('kmsx,ksxty->kmty',
                      jnp.conj(psi_xr), sigma_k, optimize=True)
    return jnp.einsum('kmty,ktyn->kmn', left, psi_yn, optimize=True)


def project_ri(psi_xr, psi_yn, sigma_k):
    """Σ(nk, s, μ, s, μ) → (2, nk, m, n) with [Re, Im] channels.

    Used by the windowed PPM Σ^c(ω) τ-loop, where the crossing window
    keeps only ``Im[coeff·σ^τ]`` so σ^τ has to carry both channels.
    A sharded reduce-scatter variant lives in ``ppm_sigma`` for the
    multi-device path.
    """
    sigma_ri = jnp.stack((jnp.real(sigma_k), jnp.imag(sigma_k)), axis=0)
    left = jnp.einsum('kmsx,cksxty->ckmty',
                      jnp.conj(psi_xr), sigma_ri, optimize=True)
    return jnp.einsum('ckmty,ktyn->ckmn',
                      left, psi_yn, optimize=True).astype(jnp.complex128)
