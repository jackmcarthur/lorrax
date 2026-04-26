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
