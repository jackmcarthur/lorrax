"""The typed spin-centroid unfold as load tables, for a consumer that fuses it into its own load.

:func:`maps.unfold_spin_centroid_operator` (axis-local, pair-transpose arm)
transports a raw-parent operator ``G`` (and, on antiunitary rows, its
transposed partner ``Gt``) to the full zone and then rotates the spin.  On
rank (x, y), with merged local endpoints ``i = mu*ns + s`` (X shard) and
``j = nu*ns + s'`` (Y shard), that transport is exactly::

    V_k[i, j] = (mph[k, i] * S_k[row[k], lsrc[k, i], rsrc[k, j]]) * nph[k, j]
                S_k = Gt if trs[k] else G;  a source of -1 is an exact zero
    O_k[mu, a, nu, b] = sum_d (sum_c U_k[a, c] V_k[mu c, nu d]) conj(U_k[b, d])

:func:`unfold_load_tables` returns those tables, so a kernel that reads the
parents can do the unfold on its load (the nvidia-mathdx k-leading
convolution, ``ffi.fft.make_kconv_klead_unfold``) instead of a gather pass, a
spin pass and a transpose.  The phases come out of the same expressions as the
unfold's own (constants folded the same way) and
``tests/test_kconv_klead_unfold.py`` holds the two equal bit for bit, so there
is still one definition of the typed action: ``maps.py``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from symmetry_maps.maps import certify_endpoint_locality

__all__ = ["UnfoldLoadTables", "unfold_load_tables", "local_unfold_load_tables",
           "apply_unfold_load_tables_local"]


class UnfoldLoadTables(NamedTuple):
    """Host tables of the typed unfold (module docstring), global over both endpoints.

    Host arrays on purpose: a jitted consumer bakes them as constants and
    slices this rank's part inside its ``shard_map``
    (:func:`local_unfold_load_tables`), the way the unfold kernel itself
    carries its tables; a closed-over multi-process device array is refused
    by JAX.
    """
    row: np.ndarray     # (nk,) int32: the parent row of each full k
    trs: np.ndarray     # (nk,) int32: 1 on an antiunitary row (read Gt)
    lsrc: np.ndarray    # (nk, n_left*ns) int32: X-shard-local merged source, -1 = zero
    rsrc: np.ndarray    # (nk, n_right*ns) int32: Y-shard-local merged source
    mph: np.ndarray     # (nk, n_left*ns) c128: left umklapp phase, TRS rule applied
    nph: np.ndarray     # (nk, n_right*ns) c128: right umklapp phase, TRS rule applied
    spin: np.ndarray    # (nk, ns, ns) c128: U_k


def _merged(perm, wraps, ns):
    """(mu) source maps and wraps to the merged (mu*ns + s) endpoint, as the unfold merges them."""
    slot = np.arange(ns, dtype=np.int32)
    perm_ms = (perm[:, :, None] * ns + slot[None, None, :]).reshape(perm.shape[0], -1)
    perm_ms[np.all(perm == -1, axis=1)] = -1
    return perm_ms, np.repeat(wraps, ns, axis=1)


def unfold_load_tables(*, irr_idx, sym_idx, sym_perm, L_table, k_irr_frac, spin_action_full,
                       n_sym_spatial, mesh_xy, logical_centroid_extent=None,
                       right_sym_perm=None, right_L_table=None) -> UnfoldLoadTables:
    """The load tables of :func:`maps.unfold_spin_centroid_operator` with ``axis_local=True``.

    Same arguments as that function, minus the operators.  Refuses a source
    map that crosses an X (left) or Y (right) shard: the fused load reads
    only this rank's parent tile, as the axis-local unfold does.
    """
    spin = np.asarray(spin_action_full, dtype=np.complex128)
    nk, ns = int(spin.shape[0]), int(spin.shape[-1])
    irr = np.asarray(irr_idx, dtype=np.int32)
    sym = np.asarray(sym_idx, dtype=np.int32)
    trs = sym >= int(n_sym_spatial)
    perm_l, wraps_l = _merged(np.asarray(sym_perm, np.int32), np.asarray(L_table), ns)
    perm_r, wraps_r = ((perm_l, wraps_l) if right_sym_perm is None else
                       _merged(np.asarray(right_sym_perm, np.int32), np.asarray(right_L_table), ns))
    n_left, n_right = int(perm_l.shape[1]), int(perm_r.shape[1])
    logical_l = n_left if logical_centroid_extent is None else int(logical_centroid_extent) * ns

    def local(perm, axis, logical):
        cert = certify_endpoint_locality(perm[sym], mesh=mesh_xy, mesh_axis=axis)
        if not cert["is_local"]:
            raise ValueError(
                f"unfold_load_tables: the {'left' if axis == 'x' else 'right'} source map "
                f"crosses a {axis} shard ({cert['crossing_count']} rows); the fused load "
                "reads only this rank's parent tile, like the axis-local unfold")
        src = cert["local_perm"].astype(np.int32)
        valid = np.arange(perm.shape[1]) < logical
        return np.where(valid[None, :], src, -1).astype(np.int32)

    lsrc = local(perm_l, "x", logical_l)
    rsrc = local(perm_r, "y", n_right)

    # The umklapp phases, spelled as the unfold kernel spells them
    # (maps._get_unfold_isdf_operator_jit._kernel and
    # _apply_unfold_phase_and_trs_local), from the same host constants.
    q_per = np.asarray(k_irr_frac, dtype=np.float64)[irr]
    L_l = np.asarray(wraps_l, dtype=np.float64)[sym]
    L_r = np.asarray(wraps_r, dtype=np.float64)[sym]

    @jax.jit
    def phases():
        pl = jnp.exp(2j * jnp.pi * jnp.einsum('qi,qmi->qm', q_per, L_l).astype(jnp.complex128))
        pr = jnp.exp(2j * jnp.pi * jnp.einsum('qi,qmi->qm', q_per, L_r).astype(jnp.complex128))
        return (jnp.where(trs[:, None], jnp.conj(pl), pl),
                jnp.where(trs[:, None], pr, jnp.conj(pr)))
    mph, nph = (np.asarray(jax.device_get(a)) for a in phases())
    return UnfoldLoadTables(row=irr, trs=trs.astype(np.int32), lsrc=lsrc, rsrc=rsrc,
                            mph=mph, nph=nph, spin=spin)


def local_unfold_load_tables(t: UnfoldLoadTables) -> UnfoldLoadTables:
    """This rank's slice of the host tables, inside a ``shard_map`` over ('x', 'y').

    Left tables are cut at this rank's X shard, right tables at its Y shard;
    the per-k tables stay whole.
    """
    ml = int(t.lsrc.shape[1]) // jax.lax.axis_size("x")
    nl = int(t.rsrc.shape[1]) // jax.lax.axis_size("y")
    x0, y0 = jax.lax.axis_index("x") * ml, jax.lax.axis_index("y") * nl
    cut = lambda a, start, width: jax.lax.dynamic_slice_in_dim(jnp.asarray(a), start, width, axis=1)
    return UnfoldLoadTables(
        row=jnp.asarray(t.row), trs=jnp.asarray(t.trs),
        lsrc=cut(t.lsrc, x0, ml), rsrc=cut(t.rsrc, y0, nl),
        mph=cut(t.mph, x0, ml), nph=cut(t.nph, y0, nl), spin=jnp.asarray(t.spin))


def apply_unfold_load_tables_local(G, Gt, t: UnfoldLoadTables, spin_host):
    """The tables applied in XLA on one rank's tiles: ``O`` ``(nk, mx, ns, my, ns)`` centroid-major.

    ``G``/``Gt`` ``(n_parent, mx*ns, my*ns)`` are this rank's parent tiles and
    ``t`` holds this rank's slices (inside a ``shard_map``); ``spin_host`` is
    ``t.spin`` on the host (the rotation skips its structural zeros).  The
    reference composition of the fused kernels, and their cpu leg.
    """
    from symmetry_maps.maps import _rotate_open_spin_centroid_operator
    n_par, ml, nl = (int(v) for v in G.shape)
    ns = int(t.spin.shape[-1])
    src = jnp.concatenate((G, Gt), axis=0)[t.row + n_par * t.trs]
    flat = jnp.take_along_axis(
        src.reshape(src.shape[0], -1),
        (jnp.maximum(t.lsrc, 0)[:, :, None] * nl + jnp.maximum(t.rsrc, 0)[:, None, :]).reshape(
            src.shape[0], -1), axis=1).reshape(src.shape)
    V = t.mph[:, :, None] * flat * t.nph[:, None, :]
    V = jnp.where((t.lsrc >= 0)[:, :, None] & (t.rsrc >= 0)[:, None, :], V, 0)
    spatial = V.reshape(V.shape[0], ml // ns, ns, nl // ns, ns)
    return _rotate_open_spin_centroid_operator(spatial, np.asarray(spin_host))
