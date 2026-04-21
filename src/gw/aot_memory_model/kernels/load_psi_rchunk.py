"""Load-psi-rchunk jits — the G-space → r-space FFT + r-slice + reshard.

Production splits this into two jits inside
``common.load_wfns.get_sharded_wfns_rchunk_slice`` so XLA's SPMD
partitioner doesn't rematerialize the FFT to satisfy the reshard
output sharding.  We model each jit separately because they have
distinct memory peaks.

Kernel 1 — ``_fft_and_rslice``
    IFFT3D + k·r phase multiply + contiguous r-slice, leaving the
    result band-sharded ``{-, XY, -, -}``.

    Input :  psi_G  (nk, nb_pad, n_s, n_gmax)   P(None,('x','y'),None,None)
    Output:  psi_rchunk  (nk, nb_pad, n_s, n_rchunk)  P(None,('x','y'),None,None)

Kernel 2 — ``_reshard_rchunk``
    Reshards the r-chunk  ``{-, XY, -, -}``  →  ``{-, Y, -, -}``
    (all-gather along x) →  ``{-, -, -, Y}``  (all-to-all along y).
    The stage_Y intermediate is the binding peak.

    Input :  psi_rchunk_xy  (nk, nb_pad, n_s, n_rchunk)  P(None,('x','y'),None,None)
    Output:  psi_rchunk_y   (nk, nb_pad, n_s, n_rchunk)  P(None,None,None,'y')

Together these cover ``MEMORY_MODEL.md`` stages "Band-chunked FFT"
and "R-chunk reshard" — the second of which is the existing model's
`M_reshard = 16·n_k·ceil(nb_pad/p_y)·n_s·B_r` formula.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..core import AotKernel, Knobs, MeshSpec, SysDims, register_kernel

_B = 16.0


def _k(sys, knobs):
    """Effective k-count per jit invocation.  When the caller sets
    ``k_chunk``, the iter_psi_rchunk_bandwise loop processes that many
    k-points per FFT/reshard call (not the full ``sys.n_k``).  The
    clamp to ``sys.n_k`` keeps callers safe when k_chunk=0 / unset
    (meaning "all k-points at once")."""
    k = knobs.get("k_chunk", 0)
    return sys.n_k if (k is None or k <= 0 or k >= sys.n_k) else int(k)


def _b(sys, knobs):
    """Effective band count per jit invocation.  ``band_chunk`` = the
    number of bands handed to a single FFT call (padded up to ``nb_pad``
    divisible by P in production; we use nb_pad if provided, else
    band_chunk itself, else sys.n_b)."""
    nb_pad = knobs.get("nb_pad", None)
    if nb_pad is not None and nb_pad > 0:
        return int(nb_pad)
    b = knobs.get("band_chunk", 0)
    return sys.n_b if (b is None or b <= 0 or b >= sys.n_b) else int(b)


# ---------------------------------------------------------------------------
# Primitives  —  all scale with effective (k_chunk × band_chunk),
# NOT full sys.n_k × sys.n_b, so the fit tracks production correctly.
# ---------------------------------------------------------------------------

def _T_fft_full(sys, knobs, mesh):
    """FFT workspace shard: (k_chunk, band_chunk, n_s, n_rtot) / (p_x·p_y)."""
    return (_B * _k(sys, knobs) * _b(sys, knobs)
            * sys.n_s * sys.n_r / (mesh.p_x * mesh.p_y))

def _T_psi_G(sys, knobs, mesh):
    """Input G-space shard: (k_chunk, band_chunk, n_s, n_gmax).  n_gmax ≈
    n_r / 4–8 (ecut-bounded) but we pass n_r as upper bound; fit
    coefficient absorbs the ratio.  Collinear with T_fft_full at this
    primitive level — NNLS will put all weight on one of them."""
    return (_B * _k(sys, knobs) * _b(sys, knobs)
            * sys.n_s * sys.n_r / (mesh.p_x * mesh.p_y))

def _T_rchunk_xy(sys, knobs, mesh):
    """r-chunk band-sharded output: (k_chunk, band_chunk, n_s, chunk_r)
    / (p_x·p_y)."""
    return (_B * _k(sys, knobs) * _b(sys, knobs)
            * sys.n_s * knobs.get("chunk_r", sys.n_r)
            / (mesh.p_x * mesh.p_y))

def _T_rchunk_y(sys, knobs, mesh):
    """r-chunk y-only intermediate: (k_chunk, band_chunk, n_s, chunk_r) /
    p_y.  Stage buffer in the reshard (all_to_all-y output + final Y
    output — same per-device size, different sharding)."""
    return (_B * _k(sys, knobs) * _b(sys, knobs)
            * sys.n_s * knobs.get("chunk_r", sys.n_r) / mesh.p_y)


# ---------------------------------------------------------------------------
# Kernel 1 — FFT + r-slice
# ---------------------------------------------------------------------------

@register_kernel
class LoadPsiRchunkFftKernel(AotKernel):
    """_fft_and_rslice: IFFT + phase + r-slice, band-sharded output.

    Knob contract matching production's per-jit invocation:
      * ``k_chunk``: bands a single jit call processes (default = sys.n_k).
        When the outer ``iter_psi_rchunk_bandwise`` loop k-chunks, each
        call to _fft_and_rslice sees only ``k_chunk`` k-points.
      * ``nb_pad`` or ``band_chunk``: bands per jit call, padded up to P
        for clean band-sharding.  ``nb_pad`` overrides ``band_chunk``.
      * ``chunk_r``: real-space slice width output (number of r-indices).
    """
    name = "load_psi_rchunk_fft"
    SYSTEM_DIMS = ("n_k", "n_b", "n_s", "kgrid", "n_r")
    KNOBS = ("k_chunk", "band_chunk", "nb_pad", "chunk_r")
    PRIMITIVES = {
        "psi_G":     _T_psi_G,
        "fft_full":  _T_fft_full,
        "rchunk_xy": _T_rchunk_xy,
    }

    def build_specs(self, sys, knobs, mesh):
        nk = _k(sys, knobs)
        nb = _b(sys, knobs)
        ns = sys.n_s
        nx, ny, nz = sys.kgrid
        n_r = nx * ny * nz
        sh = NamedSharding(mesh, P(None, ("x", "y"), None, None))
        # Using n_r as the G-space dim (upper bound of n_gmax).
        return (
            jax.ShapeDtypeStruct((nk, nb, ns, n_r), jnp.complex128, sharding=sh),
        )

    def build_callable(self, sys, knobs, mesh):
        from common.fft_helpers import make_jittable_local_ifftn_3d
        nk = _k(sys, knobs)
        nb = _b(sys, knobs)
        nx, ny, nz = sys.kgrid
        n_rtot = nx * ny * nz
        chunk_r = knobs.get("chunk_r", n_rtot)
        mesh_xy = mesh

        local_ifftn = make_jittable_local_ifftn_3d(
            mesh_xy,
            P(None, ("x", "y"), None, None, None, None),
            P(None, ("x", "y"), None, None, None, None),
        )

        band_shard = P(None, ("x", "y"), None, None)
        kvecs_cached = jnp.zeros((nk, 3), dtype=jnp.float64)
        fx = jnp.arange(nx, dtype=jnp.float64)[None, :, None, None] / nx
        fy = jnp.arange(ny, dtype=jnp.float64)[None, None, :, None] / ny
        fz = jnp.arange(nz, dtype=jnp.float64)[None, None, None, :] / nz

        @jax.jit
        def _fft_and_rslice(psi_G):
            psi_G_6d = psi_G.reshape(nk, psi_G.shape[1], sys.n_s, nx, ny, nz)
            psi_r = local_ifftn(psi_G_6d)
            phase = jnp.exp(
                2j * jnp.pi * (
                    kvecs_cached[:, 0:1, None, None] * fx
                    + kvecs_cached[:, 1:2, None, None] * fy
                    + kvecs_cached[:, 2:3, None, None] * fz
                )
            )
            psi_r = psi_r * phase[:, None, None, :, :, :]
            psi_r = psi_r * jnp.sqrt(float(n_rtot))
            nb_padded = psi_r.shape[1]
            psi_flat = psi_r.reshape(nk, nb_padded, sys.n_s, n_rtot)

            def _local_rslice(psi_local, r_start_arr):
                return jax.lax.dynamic_slice_in_dim(
                    psi_local, r_start_arr[0], chunk_r, axis=3)
            psi_rchunk = shard_map(
                _local_rslice, mesh=mesh_xy,
                in_specs=(band_shard, P()), out_specs=band_shard,
            )(psi_flat, jnp.array([0]))
            return psi_rchunk
        return _fft_and_rslice


# ---------------------------------------------------------------------------
# Kernel 2 — reshard only
# ---------------------------------------------------------------------------

@register_kernel
class LoadPsiRchunkReshardKernel(AotKernel):
    """_reshard_rchunk: {-, XY, -, -} → {-, X, -, Y} → {-, -, -, Y} (y-first).

    Matches the production form in ``common.load_wfns`` after 2026-04-21
    reshard audit — y-first ordering + out_shardings on the jit forces
    the final all_gather-x to run instead of being dropped as a
    suggestion by XLA's SPMD partitioner.
    """
    name = "load_psi_rchunk_reshard"
    SYSTEM_DIMS = ("n_k", "n_b", "n_s")
    KNOBS = ("k_chunk", "band_chunk", "nb_pad", "chunk_r")
    PRIMITIVES = {
        "rchunk_xy": _T_rchunk_xy,
        "rchunk_y":  _T_rchunk_y,
    }

    def build_specs(self, sys, knobs, mesh):
        nk = _k(sys, knobs)
        nb = _b(sys, knobs)
        ns = sys.n_s
        Br = knobs.get("chunk_r", sys.n_r)
        sh_xy = NamedSharding(mesh, P(None, ("x", "y"), None, None))
        return (
            jax.ShapeDtypeStruct((nk, nb, ns, Br), jnp.complex128, sharding=sh_xy),
        )

    def build_callable(self, sys, knobs, mesh):
        _final_Y = NamedSharding(mesh, P(None, None, None, "y"))
        _stage_X_rchunk_Y = NamedSharding(mesh, P(None, "x", None, "y"))

        @partial(jax.jit, out_shardings=_final_Y)
        def _reshard_rchunk(psi_rchunk):
            return jax.lax.with_sharding_constraint(psi_rchunk, _stage_X_rchunk_Y)
        return _reshard_rchunk
