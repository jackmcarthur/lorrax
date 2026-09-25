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
spin pass and a transpose.

An interaction (W, V, a pole field) on the q wedge uses the same tables
(``trs_rule="conj"``, :func:`maps.unfold_isdf_operator`'s Hermitian arm):
``V_k = conj(mph * S[...] * nph)`` on an antiunitary row, with no partner
tile, and the endpoint actions are 1 (scalar) or the Lorentz rotation of a
current block, left and right widths apart (``right_spin_action_full``).
``ffi.fft.make_kfft_klead_unfold`` (mathdx mode 9) reads them.  The phases come out of the same expressions as the
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
    n_parent: int       # parent rows the G tiles must carry (row < n_parent)
    mesh_shape: tuple   # (Px, Py) the local sources were cut for
    conj_trs: int = 0   # 1: an antiunitary row conjugates the product (no partner tile)
    spin_r: np.ndarray | None = None   # (nk, nr, nr) c128 right action; None = spin


def _merged(perm, wraps, ns):
    """(mu) source maps and wraps to the merged (mu*ns + s) endpoint, as the unfold merges them."""
    slot = np.arange(ns, dtype=np.int32)
    perm_ms = (perm[:, :, None] * ns + slot[None, None, :]).reshape(perm.shape[0], -1)
    perm_ms[np.all(perm == -1, axis=1)] = -1
    return perm_ms, np.repeat(wraps, ns, axis=1)


def unfold_load_tables(*, irr_idx, sym_idx, sym_perm, L_table, k_irr_frac, spin_action_full,
                       n_sym_spatial, mesh_xy, logical_centroid_extent=None,
                       right_sym_perm=None, right_L_table=None, trs_rule="pair_transpose",
                       right_spin_action_full=None,
                       right_logical_centroid_extent=None) -> UnfoldLoadTables:
    """The load tables of :func:`maps.unfold_spin_centroid_operator` with ``axis_local=True``.

    Same arguments as that function, minus the operators.  Refuses a source
    map that crosses an X (left) or Y (right) shard: the fused load reads
    only this rank's parent tile, as the axis-local unfold does.
    ``trs_rule="conj"`` gives :func:`maps.unfold_isdf_operator`'s Hermitian
    arm (an interaction on the q wedge; ``k_irr_frac`` is then the wedge's
    q), and ``right_spin_action_full`` a right endpoint action of its own
    width (a Lorentz block's).
    """
    if trs_rule not in ("pair_transpose", "conj"):
        raise ValueError(f"unfold_load_tables: trs_rule must be pair_transpose|conj, got {trs_rule!r}")
    spin = np.asarray(spin_action_full, dtype=np.complex128)
    spin_r = (None if right_spin_action_full is None
              else np.asarray(right_spin_action_full, dtype=np.complex128))
    nk, ns = int(spin.shape[0]), int(spin.shape[-1])
    ns_r = ns if spin_r is None else int(spin_r.shape[-1])
    if spin_r is not None and spin_r.shape != (nk, ns_r, ns_r):
        raise ValueError(f"unfold_load_tables: right_spin_action_full must be ({nk}, n, n); "
                         f"got {spin_r.shape}")
    irr = np.asarray(irr_idx, dtype=np.int32)
    sym = np.asarray(sym_idx, dtype=np.int32)
    n_parent = int(np.asarray(k_irr_frac).shape[0])
    if irr.shape != (nk,) or sym.shape != (nk,) or np.any(irr < 0) or np.any(irr >= n_parent):
        raise ValueError(
            f"unfold_load_tables: irr_idx/sym_idx must be ({nk},) with parent rows in "
            f"[0, {n_parent}); got {irr.shape}/{sym.shape}, rows {irr.min(initial=0)}.."
            f"{irr.max(initial=0)}")
    trs = sym >= int(n_sym_spatial)
    perm_l, wraps_l = _merged(np.asarray(sym_perm, np.int32), np.asarray(L_table), ns)
    if right_sym_perm is None:
        perm_r, wraps_r = ((perm_l, wraps_l) if ns_r == ns else
                           _merged(np.asarray(sym_perm, np.int32), np.asarray(L_table), ns_r))
    else:
        perm_r, wraps_r = _merged(np.asarray(right_sym_perm, np.int32), np.asarray(right_L_table), ns_r)
    n_left, n_right = int(perm_l.shape[1]), int(perm_r.shape[1])
    logical_l = n_left if logical_centroid_extent is None else int(logical_centroid_extent) * ns
    logical_r = (n_right if right_logical_centroid_extent is None
                 else int(right_logical_centroid_extent) * ns_r)

    def local(perm, axis, logical):
        used = perm[sym]
        if np.any(used[:, :logical] >= logical) or np.any(used[:, logical:] < logical):
            raise ValueError(
                f"unfold_load_tables: the {'left' if axis == 'x' else 'right'} source maps do "
                f"not preserve the logical/padded split at {logical}/{perm.shape[1]} "
                "(unfold_isdf_operator refuses the same tables)")
        cert = certify_endpoint_locality(used, mesh=mesh_xy, mesh_axis=axis)
        if not cert["is_local"]:
            raise ValueError(
                f"unfold_load_tables: the {'left' if axis == 'x' else 'right'} source map "
                f"crosses a {axis} shard ({cert['crossing_count']} rows); the fused load "
                "reads only this rank's parent tile, like the axis-local unfold")
        src = cert["local_perm"].astype(np.int32)
        valid = np.arange(perm.shape[1]) < logical
        return np.where(valid[None, :], src, -1).astype(np.int32)

    lsrc = local(perm_l, "x", logical_l)
    rsrc = local(perm_r, "y", logical_r)

    # The umklapp phases, spelled as the unfold kernel spells them
    # (maps._get_unfold_isdf_operator_jit._kernel and
    # _apply_unfold_phase_and_trs_local), from the same host constants.
    q_per = np.asarray(k_irr_frac, dtype=np.float64)[irr]
    L_l = np.asarray(wraps_l, dtype=np.float64)[sym]
    L_r = np.asarray(wraps_r, dtype=np.float64)[sym]

    conj_arm = trs_rule == "conj"

    @jax.jit
    def phases():
        pl = jnp.exp(2j * jnp.pi * jnp.einsum('qi,qmi->qm', q_per, L_l).astype(jnp.complex128))
        pr = jnp.exp(2j * jnp.pi * jnp.einsum('qi,qmi->qm', q_per, L_r).astype(jnp.complex128))
        if conj_arm:   # the whole product is conjugated on an antiunitary row
            return pl, jnp.conj(pr)
        return (jnp.where(trs[:, None], jnp.conj(pl), pl),
                jnp.where(trs[:, None], pr, jnp.conj(pr)))
    mph, nph = (np.asarray(jax.device_get(a)) for a in phases())
    return UnfoldLoadTables(row=irr, trs=trs.astype(np.int32), lsrc=lsrc, rsrc=rsrc,
                            mph=mph, nph=nph, spin=spin, n_parent=n_parent,
                            mesh_shape=(int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])),
                            conj_trs=int(conj_arm), spin_r=spin_r)


def local_unfold_load_tables(t: UnfoldLoadTables) -> UnfoldLoadTables:
    """This rank's slice of the host tables, inside a ``shard_map`` over ('x', 'y').

    Left tables are cut at this rank's X shard, right tables at its Y shard;
    the per-k tables stay whole.
    """
    ml = int(t.lsrc.shape[1]) // jax.lax.axis_size("x")
    nl = int(t.rsrc.shape[1]) // jax.lax.axis_size("y")
    x0, y0 = jax.lax.axis_index("x") * ml, jax.lax.axis_index("y") * nl
    cut = lambda a, start, width: jax.lax.dynamic_slice_in_dim(jnp.asarray(a), start, width, axis=1)
    return t._replace(
        row=jnp.asarray(t.row), trs=jnp.asarray(t.trs),
        lsrc=cut(t.lsrc, x0, ml), rsrc=cut(t.rsrc, y0, nl),
        mph=cut(t.mph, x0, ml), nph=cut(t.nph, y0, nl), spin=jnp.asarray(t.spin),
        spin_r=jnp.asarray(t.spin if t.spin_r is None else t.spin_r))


def apply_unfold_load_tables_local(G, Gt, t: UnfoldLoadTables, spin_host, spin_r_host=None):
    """The tables applied in XLA on one rank's tiles: ``O`` ``(nk, mx, ns, my, nr)`` centroid-major.

    ``G``/``Gt`` ``(n_parent, mx*ns, my*nr)`` are this rank's parent tiles
    (``Gt`` unread on the conj arm) and ``t`` holds this rank's slices
    (inside a ``shard_map``); ``spin_host`` / ``spin_r_host`` are the left /
    right actions on the host (the rotation skips their structural zeros;
    ``None`` right = the left).  The reference composition of the fused
    kernels, and their cpu leg.
    """
    from symmetry_maps.maps import _rotate_open_spin_centroid_operator
    n_par, ml, nl = (int(v) for v in G.shape)
    ns = int(np.asarray(spin_host).shape[-1])
    nr = ns if spin_r_host is None else int(np.asarray(spin_r_host).shape[-1])
    if t.conj_trs:
        src = G[t.row]
    else:
        src = jnp.concatenate((G, Gt), axis=0)[t.row + n_par * t.trs]
    flat = jnp.take_along_axis(
        src.reshape(src.shape[0], -1),
        (jnp.maximum(t.lsrc, 0)[:, :, None] * nl + jnp.maximum(t.rsrc, 0)[:, None, :]).reshape(
            src.shape[0], -1), axis=1).reshape(src.shape)
    V = t.mph[:, :, None] * flat * t.nph[:, None, :]
    if t.conj_trs:   # maps._apply_unfold_phase_and_trs_local's conj arm
        V = jnp.where((t.trs != 0)[:, None, None], jnp.conj(V), V)
    V = jnp.where((t.lsrc >= 0)[:, :, None] & (t.rsrc >= 0)[:, None, :], V, 0)
    spatial = V.reshape(V.shape[0], ml // ns, ns, nl // nr, nr)
    if spin_r_host is None:
        return _rotate_open_spin_centroid_operator(spatial, np.asarray(spin_host))
    return _rotate_endpoints(spatial, np.asarray(spin_host), np.asarray(spin_r_host))


def _rotate_endpoints(spatial, spin_l, spin_r):
    """``L O R^dagger`` on centroid-major ``(k, mu, a, nu, b)``: the two-width form of
    :func:`maps._rotate_open_spin_centroid_operator` (same sum order, same zero skips)."""
    L, R = jnp.asarray(spin_l), jnp.asarray(spin_r)
    nl_s, nr_s = int(spatial.shape[2]), int(spatial.shape[4])
    left = jnp.stack([sum(L[:, a, c, None, None, None] * spatial[:, :, c]
                         for c in range(nl_s) if np.any(spin_l[:, a, c] != 0))
                      for a in range(nl_s)], axis=2)
    return jnp.stack([sum(left[..., d] * jnp.conj(R[:, b, d])[:, None, None, None]
                         for d in range(nr_s) if np.any(spin_r[:, b, d] != 0))
                      for b in range(nr_s)], axis=4)
