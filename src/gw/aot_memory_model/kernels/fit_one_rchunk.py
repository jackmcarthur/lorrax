"""fit_one_rchunk: driver-level AOT of the per-r-chunk zeta fit body.

Mirrors ``common.isdf_fitting.fit_one_rchunk`` — the jit that covers the
FFT+reshard per band-chunk, pair-density streaming (spin-traced), ZCT,
Z-col reshard, and Cholesky solve, all in one HLO.  AOT-lowering this
captures the driver-level memory high-water mark including *coexisting*
buffers (G-space cache + centroid copies + L_q + live P_l/P_r +
per-bc FFT outputs), which per-stage kernels cannot model.

Input shapes (all complex128 unless noted):
    [ψ(G) is NO LONGER a jit arg.  Each bc's ψ(G) is fetched from the
     host via io_callback inside the jit body — see
     common.psi_G_store.PsiGStore.fetch_psi_G.]
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

Note: the production kernel fetches ψ(G) per band-chunk from a
host-side :class:`common.psi_G_store.PsiGStore` via io_callback inside
its bc-loop.  For AOT memory analysis we plug in an
``_AotStubPsiGStore`` whose ``fetch_psi_G`` returns zeros of the
correct per-device shape, so the compiled HLO has the same io_callback
-> FFT -> accumulate structure as production.  The io_callback does not
itself allocate persistent device memory — the peak it contributes is
captured in the single-bc ψ(G) live during FFT (``psiG_bc`` primitive).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel


class _AotStubPsiGStore:
    """AOT-only :class:`common.psi_G_store.PsiGStore` stand-in.

    ``fetch_psi_G`` returns a sharded jax.Array of zeros in the same
    shape/sharding the production store produces, so the HLO emitted
    by ``_make_fit_one_rchunk_kernel`` is structurally identical to
    production — same io_callback + FFT + pair-density sequence, same
    ``memory_analysis`` — without needing a real WFN.h5 on hand.
    Strictly for AOT peak measurement; never actually executed.
    """

    def __init__(self, mesh: Mesh, meta, band_chunk_ranges):
        self.mesh = mesh
        self.band_chunk_ranges = band_chunk_ranges
        p = int(mesh.shape['x']) * int(mesh.shape['y'])
        bc_uniform = band_chunk_ranges[0][1] - band_chunk_ranges[0][0]
        assert bc_uniform % p == 0, \
            f"band_chunk={bc_uniform} must divide total devices {p}"
        nx, ny, nz = meta.fft_grid
        self._per_device_shape = (
            int(meta.nk_tot), bc_uniform // p, int(meta.nspinor), nx, ny, nz)

    def begin_rchunk(self, *_): pass
    def end_rchunk(self): pass
    def close(self): pass

    def fetch_psi_G(self, band_range, k_range=None):
        del band_range, k_range
        per_dev = self._per_device_shape
        out_sds = jax.ShapeDtypeStruct(per_dev, jnp.complex128)
        spec = P(None, ('x', 'y'), None, None, None, None)
        zeros = np.zeros(per_dev, dtype=np.complex128)

        def _cb(_x, _y):
            return zeros

        @partial(shard_map, mesh=self.mesh,
                 in_specs=(), out_specs=spec, check_rep=False)
        def _sm():
            x = jax.lax.axis_index('x')
            y = jax.lax.axis_index('y')
            return io_callback(_cb, out_sds, x, y, ordered=True)
        return _sm()

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
# Primitives — each one is the bytes of a SINGLE physical shard at a
# distinct HLO allocation shape.  β_i from the NNLS fit then counts how
# many concurrent instances of that shard XLA holds alive at peak (for
# the small kernels like zct_lr this comes out as an integer like 4).
#
# Orthogonality requirements met by the DoE (`mos2_ortho` preset):
#   * n_rmu axis at fixed n_r → separates T_centroid (∝ μ) from T_psiG (∝ r)
#   * n_r axis via fft_grid sweep → separates T_psiG from T_centroid
#     more directly (at fixed n_rmu, n_b, n_s, only T_psiG varies)
#   * n_s axis (1 vs 2) → anything scaling in n_s pops out
#   * kgrid axis → varies n_k (and n_q, collinear at Γ-centered q)
#   * mesh asymmetry (2×2, 1×4, 4×1) → separates p_x-sharded from
#     p_y-sharded from (p_x·p_y)-sharded
#   * bc × cr cross terms → identify T_psiY ∝ bc·cr separately from
#     T_pair ∝ cr (linear in bc at fixed cr is the break).
#
# Primitives known to be COLLINEAR at fixed total_devices (single node):
#   * T_Lq_sharded and T_Lq_rep differ only by the 1/(p_x·p_y) factor.
#     Any mesh with the same product keeps their ratio fixed, so NNLS
#     can't split them on a 1-node DoE.  I keep both primitive names
#     for physical clarity but expect β_Lq_sharded to absorb everything
#     at β ≈ 1 + P (5 at P=4 if both are alive at the solve stage).
# ---------------------------------------------------------------------------

def _T_pair(sys, knobs, mesh):
    """One pair-density-shape shard: (n_k, n_rmu, chunk_r) on
    P(None, 'x', 'y').  β counts concurrent instances at peak
    (accumulators P_l, P_r; ZCT-stage temps; Z_q; zeta output).
    Expected β ≈ 5–7."""
    return (_B * sys.n_k * sys.n_rmu * _Br(sys, knobs)
            / (mesh.p_x * mesh.p_y))


def _T_psiG_cache(sys, knobs, mesh):
    """Full G-space cache (all nbc band chunks alive): (n_k, n_b,
    n_s, n_r) / (p_x·p_y).  β counts how many full caches coexist —
    typically 1 since the tuple is passed as jit input.  Held constant
    throughout the fit until Phase 1b's phdf5-on-demand wins."""
    n_r = sys.n_r or (sys.fft_shape[0] * sys.fft_shape[1] * sys.fft_shape[2])
    return (_B * sys.n_k * sys.n_b * sys.n_s * n_r
            / (mesh.p_x * mesh.p_y))


def _T_psiG_bc(sys, knobs, mesh):
    """Per-bc G-space slice: (n_k, bc, n_s, n_r) / (p_x·p_y) — one
    instance alive at a time during the streaming bc-loop.  β captures
    the transient FFT workspace (input + output + cuFFT scratch — may
    show β ≈ 2–3)."""
    n_r = sys.n_r or (sys.fft_shape[0] * sys.fft_shape[1] * sys.fft_shape[2])
    return (_B * sys.n_k * _bc(sys, knobs) * sys.n_s * n_r
            / (mesh.p_x * mesh.p_y))


def _T_psiY_bc(sys, knobs, mesh):
    """Per-bc r-chunk slab after reshard: (n_k, bc, n_s, chunk_r) /
    p_y (Y-sharded only).  β counts instances alive during the
    streaming bc-loop — expected 1 if streaming works.  Scales with
    chunk_r·bc so it's the most restrictive knob interaction."""
    return (_B * sys.n_k * _bc(sys, knobs) * sys.n_s * _Br(sys, knobs)
            / mesh.p_y)


def _T_centroid(sys, knobs, mesh):
    """Combined X-sharded centroid bytes for L+R:
    (n_k, n_rmu, n_b_sum, n_s) / p_x, where n_b_sum = n_b_L + n_b_R.

    Symmetric case (L == R == full) has n_b_sum = 2·n_b — captures
    BOTH centroid copies in a single primitive so β counts "1 unit of
    L+R centroid bytes" (expected β = 1 since both copies are alive
    as jit arguments).  Asymmetric case uses sys.n_b_sum explicitly.
    """
    return (_B * sys.n_k * sys.n_rmu * sys.nb_sum * sys.n_s
            / mesh.p_x)


def _T_Lq_sharded(sys, knobs, mesh):
    """XY-sharded Cholesky factor: (n_q, n_rmu, n_rmu) / (p_x·p_y).
    β counts persistent instances (typically 1 = just L_q as input).
    NOTE: at fixed total_devices this is collinear with T_Lq_rep —
    NNLS will lump them together (expected β ≈ 5 at P=4 if both
    sharded + replicated L are alive during solve)."""
    return _B * sys.n_k * sys.n_rmu * sys.n_rmu / (mesh.p_x * mesh.p_y)


def _T_Lq_rep(sys, knobs, mesh):
    """Fully replicated L: (n_q, n_rmu, n_rmu) — no mesh division.
    β counts instances during the replicated solve stage (expected 1).
    Collinear with T_Lq_sharded at fixed total_devices; see above."""
    return _B * sys.n_k * sys.n_rmu * sys.n_rmu


# Dropped: T_zcol (identical bytes to T_pair — NNLS can't separate
# them on any DoE with fixed total_devices).  Z_col instances fold
# into T_pair's β count.


# ---------------------------------------------------------------------------
# FLOPs primitives — per-call cost of one fit_one_rchunk jit invocation.
#
# Derived from the algebra of the kernel body:
#
#   1. Stream band-chunks (nbc iterations of size bc = band_chunk):
#        - IFFT3D on (nk, bc, n_s, nx, ny, nz)   ~ bc·nk·n_s·n_r·log(n_r)
#        - pair-density einsum 'kmnm,knms->knmm-like'   ~ bc·nk·n_rmu·B_r
#          (L contribution and R contribution both; nbc×2 times total)
#   2. ZCT stage (ONCE per rchunk call, independent of bc):
#        - IFFT over kgrid, elementwise mul, FFT back   ~ n_k·n_rmu·B_r·log(n_k)
#   3. Cholesky solve (ONCE per rchunk call):
#        - per-q triangular solve n_rmu²·B_r   ~ n_q·n_rmu²·B_r
#
# We fold (log factors, constants) into the NNLS fit coefficients.
# Primitives are CHOSEN so that each has a distinct scaling — the fit
# can split the per-call cost into its components.
# ---------------------------------------------------------------------------

def _F_fixed(sys, knobs, mesh):
    """Per-call intercept — compile overhead + small ops with no
    explicit scaling.  Represented as a constant 1.0 feature via the
    fit's intercept column; kept here for naming only."""
    return 1.0


def _F_pair_density(sys, knobs, mesh):
    """Pair-density accumulate across all band-chunks, summed over
    both L and R contributions:
        nbc_L × bc · nk · n_rmu · B_r  +  nbc_R × bc · nk · n_rmu · B_r
      = (n_b_L + n_b_R) · nk · n_rmu · B_r
      = n_b_sum · nk · n_rmu · B_r

    Symmetric case has n_b_sum = 2·n_b (factor of 2 for L+R).  The
    formula is mesh-invariant for total FLOPs."""
    Br = _Br(sys, knobs)
    return sys.nb_sum * sys.n_k * sys.n_rmu * Br


def _F_bc_fft(sys, knobs, mesh):
    """Per-bc IFFT+phase across all band-chunks:
        nbc × bc × nk × n_s × n_r × log(n_r)
      = n_b · nk · n_s · n_r · log2(n_r)"""
    n_r = sys.n_r or (sys.fft_shape[0] * sys.fft_shape[1] * sys.fft_shape[2])
    return sys.n_b * sys.n_k * sys.n_s * n_r * np.log2(max(2, n_r))


def _F_zct(sys, knobs, mesh):
    """ZCT FFT over k-grid (ONE call per r-chunk):
        n_k · n_rmu · B_r · log(n_k)"""
    Br = _Br(sys, knobs)
    return sys.n_k * sys.n_rmu * Br * np.log2(max(2, sys.n_k))


def _F_solve(sys, knobs, mesh):
    """Per-q triangular solve (ONE call per r-chunk):
        n_q · n_rmu^2 · B_r    (q = k for Γ-centered grids)"""
    Br = _Br(sys, knobs)
    return sys.n_k * (sys.n_rmu ** 2) * Br


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
        "pair":       _T_pair,
        "psiG_cache": _T_psiG_cache,
        "psiG_bc":    _T_psiG_bc,
        "psiY_bc":    _T_psiY_bc,
        "centroid":   _T_centroid,
        "Lq_sharded": _T_Lq_sharded,
        "Lq_rep":     _T_Lq_rep,
    }
    # Scaling class of each primitive in (chunk_r, bc) — the analytic
    # chooser groups β_i·T_i into (α₀, α_cr, α_bc, α_crbc) and inverts
    # the feasibility bound `peak ≤ M` in closed form.  Classes:
    #   "const"   : T_i depends on neither chunk_r nor bc.
    #   "cr"      : T_i is linear in chunk_r only.
    #   "bc"      : T_i is linear in bc only.
    #   "crbc"    : T_i scales like chunk_r·bc.
    PRIMITIVE_CLASSES = {
        "pair":       "cr",
        "psiG_cache": "const",
        "psiG_bc":    "bc",
        "psiY_bc":    "crbc",
        "centroid":   "const",
        "Lq_sharded": "const",
        "Lq_rep":     "const",
    }
    # Cost-side primitives — distinct from memory primitives because
    # FLOPs count != bytes-per-device.  See _F_* helpers above.
    FLOPS_PRIMITIVES = {
        "pair_density": _F_pair_density,
        "bc_fft":       _F_bc_fft,
        "zct":          _F_zct,
        "solve":        _F_solve,
    }

    # ---------- specs ----------
    def build_specs(self, sys: SysDims, knobs: Knobs, mesh: Mesh):
        nk = sys.n_k
        mu = sys.n_rmu
        ns = sys.n_s
        nb = sys.n_b

        cent_shard = NamedSharding(mesh, P(None, None, "x", None))
        L_shard    = NamedSharding(mesh, P(None, "x", "y"))
        rep        = NamedSharding(mesh, P())

        # psi_bc_G is no longer a jit argument — the provider pulls each
        # bc from host via io_callback inside the jit body.  Specs cover
        # only: psi_l_X, psi_r_X, L_q, norms_l, norms_r, r_start.
        return (
            jax.ShapeDtypeStruct(
                (nk, mu, nb, ns), jnp.complex128, sharding=cent_shard),
            jax.ShapeDtypeStruct(
                (nk, mu, nb, ns), jnp.complex128, sharding=cent_shard),
            jax.ShapeDtypeStruct(
                (nk, mu, mu), jnp.complex128, sharding=L_shard),
            jax.ShapeDtypeStruct((nb,), jnp.float64, sharding=rep),
            jax.ShapeDtypeStruct((nb,), jnp.float64, sharding=rep),
            jax.ShapeDtypeStruct((), jnp.int32),
        )

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

        # ψ(G) is no longer a jit arg — provide a stub PsiGStore whose
        # ``fetch_psi_G`` returns zeros of the right per-device shape.
        # This gives the AOT compile the same graph structure the
        # production kernel emits (one io_callback per bc, one FFT per
        # bc, one pair-density accumulate per bc), without needing an
        # actual WFN.h5 file open on the DoE driver.
        stub_store = _AotStubPsiGStore(mesh, meta, band_chunk_ranges)
        kernel = _make_fit_one_rchunk_kernel(
            mesh, meta, band_chunk_ranges,
            band_range_left, band_range_right, band_range_full,
            Br, q_chunk_size, kvecs_frac,
            stub_store,
        )

        # The factory returns the bare jit.  Wrap so the AOT specs'
        # positional order matches: X + Y + L_q + norms + r_start.
        def _apply(psi_l_X, psi_r_X, L_q, norms_l, norms_r, r_start):
            return kernel(psi_l_X, psi_r_X, L_q,
                          norms_l, norms_r, r_start)

        return jax.jit(_apply)
