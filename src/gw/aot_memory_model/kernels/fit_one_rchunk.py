"""fit_one_rchunk: driver-level AOT of the per-r-chunk zeta fit body.

Mirrors ``common.isdf_fitting.fit_one_rchunk`` — the jit that covers the
FFT+reshard per band-chunk, pair-density streaming (spin-traced), ZCT,
Z-col reshard, and Cholesky solve, all in one HLO.  AOT-lowering this
captures the driver-level memory high-water mark including *coexisting*
buffers (G-space cache + centroid copies + L_q + live P_l/P_r +
per-bc FFT outputs), which per-stage kernels cannot model.

Input shapes (all complex128 unless noted):
    psi_G_bc_i : (n_k, bc_size, n_s, n_r) on P(None,('x','y'),None,None)
                 one per band-chunk — passed as a tuple of tensors
    psi_l_rmuT_X_fit : (n_k, n_rmu, n_b_l, n_s) on P(None, None, 'x', None)
    psi_r_rmuT_X_fit : (n_k, n_rmu, n_b_r, n_s) on P(None, None, 'x', None)
    L_q              : (n_q, n_rmu, n_rmu) on P(None, 'x', 'y')
    norms_l          : (n_b_l,) float64 replicated
    norms_r          : (n_b_r,) float64 replicated
    r_start_dyn      : () int32 scalar (dynamic)

Output:
    zeta_chunk : (n_q, n_rmu, chunk_r) on P(None, None, ('x','y'))

Key primitives dominating driver-level peak (observed empirically):
  * ``PrBc``   = 16 · B_r · (4·n_k + n_q) · n_rmu / (p_x · p_y)
                 (ZCT stage) — 4 concurrent pair-density-sized temps.
  * ``Pacc``   = 16 · n_k · n_rmu · B_r / (p_x · p_y)
                 (P_l + P_r accumulators, 2 copies, persistent across bc loop).
  * ``psiBc``  = 16 · n_k · bc_size · n_s · n_r / (p_x · p_y)
                 (per-bc FFT output, superseded by next bc but alive during
                  accumulate) — this and ``psi_r_y`` are the bc-step peaks.
  * ``rchunk_y`` = 16 · n_k · bc_size · n_s · B_r / p_y
                 (reshard stage).
  * ``psi_cent`` = 16 · n_k · n_rmu · (n_b_l + n_b_r) · n_s / p_x
                 (centroid copies: always alive, cheap-ish).
  * ``L_q_shard`` = 16 · n_q · n_rmu^2 / (p_x · p_y)
                 (L factor: always alive).

Knobs:
    * ``chunk_r``  (r-slab width; primary DoE axis for driver peak)
    * ``band_chunk`` (bc_size for the streaming pair-density)

Note: ``psi_bc_G_tuple`` is a *pytree* of tensors passed positionally.
We use the conservative path of duplicating the same shape across all
band chunks for AOT specs — production rounds up to the full band count
so most chunks are uniform; the remainder chunk is folded via a separate
compile (the actual jit cache key distinguishes them).  A single AOT
compile with ``n_bc`` uniform-width chunks is representative of the
steady-state cost.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel

_B = 16.0


# ---------------------------------------------------------------------------
# Effective knob access
# ---------------------------------------------------------------------------

def _Br(sys, knobs):
    """r-chunk width (slab)."""
    cr = knobs.get("chunk_r", sys.n_r)
    return sys.n_r if (cr is None or cr <= 0) else int(cr)


def _bc(sys, knobs):
    """band_chunk size — number of bands per streaming FFT call."""
    bc = knobs.get("band_chunk", 16)
    return int(bc) if bc and bc > 0 else 16


def _nbc(sys, knobs):
    """Number of band chunks over the full n_b range."""
    nb = sys.n_b
    bc = _bc(sys, knobs)
    return max(1, (nb + bc - 1) // bc)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _T_Pacc(sys, knobs, mesh):
    """Two pair-density accumulators P_l, P_r: (n_k, n_rmu, B_r)/P."""
    return (2.0 * _B * sys.n_k * sys.n_rmu * _Br(sys, knobs)
            / (mesh.p_x * mesh.p_y))


def _T_PrBc(sys, knobs, mesh):
    """ZCT-stage buffer family: (4·n_k + n_q) · n_rmu · B_r / P.
    Matches the zct_lr kernel primitive — 4 concurrent pair-sized temps
    + 1 output.  Kept as a separate primitive for easy diagnosis."""
    nq = sys.n_k  # for Γ-centered k = q grids; safe conservative proxy
    return (_B * (4.0 * sys.n_k + nq) * sys.n_rmu * _Br(sys, knobs)
            / (mesh.p_x * mesh.p_y))


def _T_psiBc(sys, knobs, mesh):
    """Per-bc FFT output: (n_k, bc_size, n_s, n_r) / P.  Peak inside
    the streaming loop, superseded bc-by-bc but alive during accumulate."""
    n_r = sys.n_r or (sys.fft_shape[0] * sys.fft_shape[1] * sys.fft_shape[2])
    return (_B * sys.n_k * _bc(sys, knobs) * sys.n_s * n_r
            / (mesh.p_x * mesh.p_y))


def _T_psiBcY(sys, knobs, mesh):
    """Per-bc reshard output: (n_k, bc_size, n_s, B_r) / p_y."""
    return (_B * sys.n_k * _bc(sys, knobs) * sys.n_s * _Br(sys, knobs)
            / mesh.p_y)


def _T_psi_centroid(sys, knobs, mesh):
    """Centroid copies psi_l_rmuT_X_fit + psi_r_rmuT_X_fit:
    (n_k, n_rmu, n_b, n_s) / p_x — always alive through the bc loop."""
    return (_B * sys.n_k * sys.n_rmu * sys.n_b * sys.n_s
            / mesh.p_x)


def _T_L_q(sys, knobs, mesh):
    """L_q Cholesky factor: (n_q, n_rmu, n_rmu) / (p_x·p_y).  Always alive."""
    nq = sys.n_k
    return _B * nq * sys.n_rmu * sys.n_rmu / (mesh.p_x * mesh.p_y)


def _T_psiG_total(sys, knobs, mesh):
    """Total psi_G cache across all band chunks:
    (n_k · n_b · n_s · n_r) / (p_x · p_y).  Natively 6D in production
    but same byte volume.  Alive throughout the fit (Phase 1b will move
    this host-resident → exclude from peak)."""
    n_r = sys.n_r or (sys.fft_shape[0] * sys.fft_shape[1] * sys.fft_shape[2])
    return (_B * sys.n_k * sys.n_b * sys.n_s * n_r
            / (mesh.p_x * mesh.p_y))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@register_kernel
class FitOneRChunkKernel(AotKernel):
    """Composite r-chunk-body jit covering all coexisting buffers."""
    name = "fit_one_rchunk"
    SYSTEM_DIMS = ("n_k", "n_rmu", "n_s", "n_b", "n_r", "kgrid")
    KNOBS = ("chunk_r", "band_chunk")
    PRIMITIVES = {
        "Pacc":       _T_Pacc,
        "PrBc":       _T_PrBc,
        "psiBc":      _T_psiBc,
        "psiBcY":     _T_psiBcY,
        "psi_cent":   _T_psi_centroid,
        "L_q":        _T_L_q,
        "psiG_total": _T_psiG_total,
    }

    # ---------- specs ----------
    def build_specs(self, sys: SysDims, knobs: Knobs, mesh: Mesh):
        nk = sys.n_k
        mu = sys.n_rmu
        ns = sys.n_s
        nb = sys.n_b
        Br = _Br(sys, knobs)
        bc = _bc(sys, knobs)
        nbc = _nbc(sys, knobs)
        fft_shape = sys.fft_shape
        n_r_total = sys.n_r or (fft_shape[0] * fft_shape[1] * fft_shape[2])

        psiG_shard = NamedSharding(
            mesh, P(None, ("x", "y"), None, None, None, None))
        cent_shard = NamedSharding(mesh, P(None, None, "x", None))
        L_shard    = NamedSharding(mesh, P(None, "x", "y"))
        rep        = NamedSharding(mesh, P())

        nx, ny, nz = fft_shape
        specs = []
        # psi_G tuple — one entry per band chunk, each natively 6D:
        # (n_k, bc_size, n_s, nx, ny, nz) matching read_Gvecs_to_devices.
        for _ in range(nbc):
            specs.append(jax.ShapeDtypeStruct(
                (nk, bc, ns, nx, ny, nz), jnp.complex128,
                sharding=psiG_shard))
        # psi_l_rmuT_X_fit, psi_r_rmuT_X_fit
        specs.append(jax.ShapeDtypeStruct(
            (nk, mu, nb, ns), jnp.complex128, sharding=cent_shard))
        specs.append(jax.ShapeDtypeStruct(
            (nk, mu, nb, ns), jnp.complex128, sharding=cent_shard))
        # L_q
        specs.append(jax.ShapeDtypeStruct(
            (nk, mu, mu), jnp.complex128, sharding=L_shard))
        # norms
        specs.append(jax.ShapeDtypeStruct(
            (nb,), jnp.float64, sharding=rep))
        specs.append(jax.ShapeDtypeStruct(
            (nb,), jnp.float64, sharding=rep))
        # r_start scalar — dynamic int32 passed inside the jit
        specs.append(jax.ShapeDtypeStruct((), jnp.int32))
        return tuple(specs)

    # ---------- callable ----------
    def build_callable(self, sys: SysDims, knobs: Knobs, mesh: Mesh):
        """Assembles the r-chunk body via the production factory so this
        kernel tracks ``common.isdf_fitting._make_fit_one_rchunk_kernel``
        bit-for-bit.  We pass synthetic ``meta`` and ``band_chunk_ranges``
        derived from the DoE point.

        Returns a jit that accepts ``(psi_G_tuple, psi_l_X, psi_r_X, L_q,
        norms_l, norms_r, r_start_dyn)`` exactly as the AOT specs above."""
        from common.isdf_fitting import _make_fit_one_rchunk_kernel
        from common.load_wfns import Meta

        nb = sys.n_b
        bc = _bc(sys, knobs)
        Br = _Br(sys, knobs)
        nbc = _nbc(sys, knobs)

        band_chunk_ranges = tuple(
            (i * bc, min((i + 1) * bc, nb)) for i in range(nbc)
        )
        band_range_left  = (0, nb)
        band_range_right = (0, nb)
        band_range_full  = (0, nb)

        fft_shape = sys.fft_shape
        n_r_total = sys.n_r or (fft_shape[0] * fft_shape[1] * fft_shape[2])
        kx, ky, kz = sys.kgrid
        # The factory and get_sharded_wfns_rchunk_slice only read nk_tot,
        # n_rmu, nspinor, fft_grid, kgrid, n_rtot — populate those and
        # leave everything else at zero / safe defaults.
        meta = Meta(
            rank=0, n_proc=mesh.size,
            b_id_0=0, b_id_1=0, b_id_2=0, b_id_3=nb, b_id_4=nb,
            fft_grid=fft_shape,
            cell_volume=1.0,
            n_rtot=n_r_total,
            n_rmu=sys.n_rmu,
            npol=1, nfreq=1,
            nspin=sys.n_s, nspinor=sys.n_s, nspinor_wfnfile=sys.n_s,
            nkx=kx, nky=ky, nkz=kz, nk_tot=sys.n_k,
            nbnd_jax=nb, n_rtot_jax=n_r_total, n_rmu_jax=sys.n_rmu,
        )

        # kvecs_frac synthetic but stable — AOT specs only use shape;
        # we pick the canonical MP grid.
        kx, ky, kz = sys.kgrid
        grid = np.array(np.meshgrid(
            np.arange(kx) / kx, np.arange(ky) / ky, np.arange(kz) / kz,
            indexing="ij",
        )).reshape(3, -1).T.astype(np.float64)
        kvecs_frac = np.ascontiguousarray(grid)

        q_chunk_size = 1
        kernel = _make_fit_one_rchunk_kernel(
            mesh, meta, band_chunk_ranges,
            band_range_left, band_range_right, band_range_full,
            Br, q_chunk_size, kvecs_frac,
        )

        # The factory returns the bare jit.  Wrap so the AOT specs'
        # positional order matches: tuple_of_psiG + X + Y + L_q + norms + r_start.
        def _apply(*args):
            psi_tuple = args[:nbc]
            psi_l_X, psi_r_X, L_q, norms_l, norms_r, r_start = args[nbc:]
            return kernel(psi_tuple, psi_l_X, psi_r_X, L_q,
                          norms_l, norms_r, r_start)

        return jax.jit(_apply)
