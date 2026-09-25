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

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from symmetry_maps.maps import certify_endpoint_locality

__all__ = ["QirrOperator", "DeviceLoadTables", "DEVICE_LOAD_SPECS", "device_load_tables",
           "UnfoldLoadTables", "unfold_load_tables", "local_unfold_load_tables", "umklapp_phase",
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
    pl, pr = umklapp_phase(q_per, L_l), umklapp_phase(q_per, L_r)
    if conj_arm:   # the whole product is conjugated on an antiunitary row
        mph, nph = pl, np.conj(pr)
    else:
        mph = np.where(trs[:, None], np.conj(pl), pl)
        nph = np.where(trs[:, None], pr, np.conj(pr))
    return UnfoldLoadTables(row=irr, trs=trs.astype(np.int32), lsrc=lsrc, rsrc=rsrc,
                            mph=mph, nph=nph, spin=spin, n_parent=n_parent,
                            mesh_shape=(int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])),
                            conj_trs=int(conj_arm), spin_r=spin_r)


def umklapp_phase(q_per, L):
    """``exp(2 pi i q . L)`` per ``(q, endpoint)`` on the host, bit for bit the value the
    unfold kernel's constant-folded ``jnp.exp(2j*pi*einsum('qi,qmi->qm', q, L))`` gets.

    Built on the host because XLA folds that no-argument program on the CPU at
    compile time (~0.3 s per table on a GPU node, seconds on a login node).
    The three-term dot is summed in XLA's order (``tests/test_kfft_klead_unfold``
    pins the equality)."""
    q = np.asarray(q_per, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    s = q[:, None, 0] * L[..., 0]
    s = s + q[:, None, 1] * L[..., 1]
    s = s + q[:, None, 2] * L[..., 2]
    return np.exp(2j * np.pi * s.astype(np.complex128))


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


class DeviceLoadTables(NamedTuple):
    """The load tables on the devices, each already cut to the shards that read it.

    A consumer passes them to its jit as ARGUMENTS (a ``QirrOperator`` carries
    them as pytree leaves), so its program holds no table constants: baked
    host tables are megabytes of HLO literal per door and cost compile time."""
    row: object
    trs: object
    lsrc: object
    rsrc: object
    mph: object
    nph: object
    spin: object
    spin_r: object


#: The ``shard_map`` in-specs of :class:`DeviceLoadTables`, in field order.
from jax.sharding import NamedSharding as _NS, PartitionSpec as _P  # noqa: E402
DEVICE_LOAD_SPECS = (_P(), _P(), _P(None, "x"), _P(None, "y"), _P(None, "x"), _P(None, "y"),
                     _P(), _P())


def device_load_tables(t: UnfoldLoadTables, mesh_xy) -> DeviceLoadTables:
    """``t`` placed once: per-k tables replicated, left tables on X, right tables on Y."""
    from lxkit import device_put_process_local
    spin_r = t.spin if t.spin_r is None else t.spin_r
    host = (t.row, t.trs, t.lsrc, t.rsrc, t.mph, t.nph, t.spin, spin_r)
    return DeviceLoadTables(*(device_put_process_local(np.asarray(a), _NS(mesh_xy, spec))
                              for a, spec in zip(host, DEVICE_LOAD_SPECS)))


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


@dataclasses.dataclass(frozen=True)
class QirrOperator:
    """An interaction held on its irreducible q wedge, with the unfold that defines the full zone.

    ``values`` ``(n_wedge, n_left, n_right)`` at ``P(None,'x','y')``; the
    tables are exactly the arguments :func:`maps.unfold_isdf_operator` takes
    for it, and ``full_rows[i]`` is the full-zone row of wedge row ``i``,
    where the unfold is the identity.  A consumer that reads the interaction
    through a k-convolution takes :meth:`load_tables` (the unfold on the
    transform's load, ``ffi.fft.make_kfft_klead_unfold``); :meth:`unfold`
    materializes the full zone for a consumer that needs it whole.
    """
    values: object
    irr_idx: np.ndarray
    sym_idx: np.ndarray
    sym_perm: np.ndarray
    L_table: np.ndarray
    q_irr_frac: np.ndarray
    n_sym_spatial: int
    full_rows: np.ndarray
    trs_rule: str = "conj"
    #: The load tables on the devices (:meth:`with_load`), or ``None``.
    load: object = None

    @classmethod
    def whole_zone(cls, values, *, trs_rule="conj") -> "QirrOperator":
        """The trivial wedge: every q its own row (a deck without a reducing q group)."""
        nq, n_l = int(values.shape[0]), int(values.shape[1])
        if int(values.shape[2]) != n_l:
            raise ValueError("QirrOperator.whole_zone: a square operator is required")
        return cls(values=values, irr_idx=np.arange(nq, dtype=np.int32),
                   sym_idx=np.zeros(nq, np.int32),
                   sym_perm=np.arange(n_l, dtype=np.int32)[None, :],
                   L_table=np.zeros((1, n_l, 3)), q_irr_frac=np.zeros((nq, 3)),
                   n_sym_spatial=1, full_rows=np.arange(nq, dtype=np.int32), trs_rule=trs_rule)

    @classmethod
    def of(cls, interaction) -> "QirrOperator":
        """``interaction`` as an operator: a full-zone array is its own trivial wedge."""
        return interaction if isinstance(interaction, cls) else cls.whole_zone(interaction)

    def is_whole_zone(self) -> bool:
        """Whether every q is its own wedge row (the unfold is the identity)."""
        return (self.n_wedge == self.n_full
                and np.array_equal(np.asarray(self.irr_idx), np.arange(self.n_full))
                and not np.any(np.asarray(self.sym_idx)))

    def restrict(self, wedge: "QirrOperator") -> "QirrOperator":
        """This interaction on ``wedge``'s rows and tables.  Exact: a full-zone row
        that is a wedge representative is the identity unfold of that wedge row."""
        if self.same_wedge(wedge):
            return self
        if not self.is_whole_zone():
            raise ValueError("QirrOperator.restrict: only a whole-zone operator restricts "
                             "onto another wedge")
        return dataclasses.replace(wedge, values=_rows_of(self.values, wedge.full_rows))

    def at_rows(self, q_full_rows):
        """The values at full-zone rows ``q_full_rows`` (the wedge's representatives, in
        order, or any rows of a whole-zone operator): no unfold involved."""
        rows = np.asarray(q_full_rows).reshape(-1)
        if np.array_equal(rows, np.asarray(self.full_rows)):
            return self.values
        if not self.is_whole_zone():
            raise ValueError("QirrOperator.at_rows: rows other than the wedge's "
                             "representatives need the unfold")
        return _rows_of(self.values, rows)

    @property
    def n_full(self) -> int:
        return int(np.asarray(self.irr_idx).shape[0])

    @property
    def n_wedge(self) -> int:
        return int(self.values.shape[0])

    def with_values(self, values) -> "QirrOperator":
        """The same wedge and tables around other values (an elementwise function of these)."""
        if tuple(values.shape) != tuple(self.values.shape):
            raise ValueError(f"QirrOperator.with_values: shape {values.shape} != {self.values.shape}")
        return dataclasses.replace(self, values=values)

    def same_wedge(self, other: "QirrOperator") -> bool:
        """Whether ``other`` unfolds by the same tables (so the two combine on the wedge)."""
        pairs = ((self.irr_idx, other.irr_idx), (self.sym_idx, other.sym_idx),
                 (self.sym_perm, other.sym_perm), (self.L_table, other.L_table),
                 (self.q_irr_frac, other.q_irr_frac), (self.full_rows, other.full_rows))
        return (self.trs_rule == other.trs_rule and self.n_sym_spatial == other.n_sym_spatial
                and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in pairs))

    def representative_row(self, q_full: int):
        """The full-zone row ``q_full`` of a wedge representative (q = 0 always is one)."""
        hit = np.flatnonzero(np.asarray(self.full_rows) == int(q_full))
        if hit.size != 1:
            raise ValueError(f"QirrOperator: q row {q_full} is not a wedge representative")
        return self.values[int(hit[0])]

    def unfold(self, mesh_xy):
        """The full-zone interaction ``(n_full, n_left, n_right)``: :func:`maps.unfold_isdf_operator`."""
        from symmetry_maps.maps import unfold_isdf_operator
        if self.is_whole_zone():
            return self.values
        return unfold_isdf_operator(
            self.values, irr_idx=self.irr_idx, sym_idx=self.sym_idx, sym_perm=self.sym_perm,
            L_table=self.L_table, q_irr_frac=self.q_irr_frac, mesh_xy=mesh_xy,
            n_sym_spatial=self.n_sym_spatial, trs_rule=self.trs_rule)

    def load_tables(self, mesh_xy, *, in_trace: bool = False) -> UnfoldLoadTables:
        """:meth:`unfold` as load tables (scalar endpoints), for ``make_kfft_klead_unfold``.

        Host tables.  Build them outside any jit (one small phase program);
        ``in_trace=True`` builds them while a consumer's jit traces, eagerly
        and op by op, which is correct but compiles each primitive apart."""
        def build():
            return unfold_load_tables(
                irr_idx=self.irr_idx, sym_idx=self.sym_idx, sym_perm=self.sym_perm,
                L_table=self.L_table, k_irr_frac=self.q_irr_frac,
                spin_action_full=np.ones((self.n_full, 1, 1), np.complex128),
                n_sym_spatial=self.n_sym_spatial, mesh_xy=mesh_xy, trs_rule=self.trs_rule)
        if not in_trace:
            return build()
        with jax.ensure_compile_time_eval():
            return build()

    def wedge_key(self) -> "_WedgeKey":
        """A hashable key of the tables (not the values): equal keys unfold alike."""
        return _WedgeKey(self)

    def with_load(self, mesh_xy) -> "QirrOperator":
        """This operator carrying its load tables on the devices of ``mesh_xy``
        (built once per mesh and tables, outside any jit)."""
        if self.load is not None:
            return self
        key = (tuple(d.id for d in np.asarray(mesh_xy.devices).flat), self.wedge_key())
        dev = _device_load_cache.get(key)
        if dev is None:
            dev = _device_load_cache[key] = device_load_tables(self.load_tables(mesh_xy), mesh_xy)
        return dataclasses.replace(self, load=dev)


def _rows_of(values, rows):
    """Rows of a ``P(None,'x','y')`` q stack, kept on that sharding (the service's slice);
    inside a trace the consumer's program owns the placement."""
    from symmetry_maps.maps import slice_q_full_to_ibz
    mesh = getattr(getattr(values, "sharding", None), "mesh", None)
    if isinstance(values, jax.core.Tracer) or mesh is None:
        return jnp.take(values, jnp.asarray(np.asarray(rows)), axis=0)
    return slice_q_full_to_ibz(
        values, np.asarray(rows), out_sharding=_NS(mesh, _P(None, "x", "y")))


_device_load_cache: dict = {}


class _WedgeKey:
    """The tables of a :class:`QirrOperator`, hashable: the pytree aux data and a cache key."""
    __slots__ = ("fields", "_hash")
    _NAMES = ("irr_idx", "sym_idx", "sym_perm", "L_table", "q_irr_frac", "full_rows")

    def __init__(self, op):
        arrays = tuple(np.ascontiguousarray(getattr(op, n)) for n in self._NAMES)
        self.fields = arrays + (int(op.n_sym_spatial), str(op.trs_rule))
        self._hash = hash(tuple(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()
                                for a in arrays) + self.fields[len(arrays):])

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, _WedgeKey) or self._hash != other._hash:
            return False
        n = len(self._NAMES)
        return (self.fields[n:] == other.fields[n:]
                and all(a.dtype == b.dtype and np.array_equal(a, b)
                        for a, b in zip(self.fields[:n], other.fields[:n])))

    def rebuild(self, values, load=None):
        n = len(self._NAMES)
        kw = dict(zip(self._NAMES, self.fields[:n]))
        return QirrOperator(values=values, n_sym_spatial=self.fields[n],
                            trs_rule=self.fields[n + 1], load=load, **kw)


# A pytree whose leaves are the values (and the device load tables, when
# attached): a jitted consumer takes the operator as an argument, its host
# tables as static structure and its device tables as arguments.
def _flatten(op):
    if op.load is None:
        return (op.values,), (_WedgeKey(op), False)
    return (op.values, *op.load), (_WedgeKey(op), True)


def _unflatten(aux, leaves):
    key, has_load = aux
    return key.rebuild(leaves[0], DeviceLoadTables(*leaves[1:]) if has_load else None)


jax.tree_util.register_pytree_node(QirrOperator, _flatten, _unflatten)

