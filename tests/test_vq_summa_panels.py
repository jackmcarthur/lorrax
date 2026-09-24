"""The V_q q-tile kernel: SUMMA G panels, bounded ζ reads, sub-sphere one-leg.

``gw.v_q_g_flat`` gathers one ``(μ, g_chunk)`` panel per operand per G step
instead of the whole ``(μ, n_G)`` face, reads ζ in budget-sized q-tiles, and
feeds the IBZ one-leg unfold only the parent columns it reads.  Each cell
below is a parity claim against an independent evaluation on a real 2x2
mesh, with the red case constructed where the claim could be vacuous:

* the panel kernel against a frozen copy of the whole-face kernel it
  replaced (same GEMMs, so bitwise on the host backend) and against NumPy;
* a G tail the chunk does not divide, against NumPy — the clamped last
  chunk double-counts without its mask, and the cell shows that it would;
* a forced multi-tile budget, including a remainder tile, against the
  single-tile run of the same function (bitwise);
* the one-leg sub-sphere against the whole sphere under an inversion whose
  star needs a NONZERO parent G, plus the control that dropping one of the
  named columns makes the service refuse;
* route G's kept columns on the Si 4x4x4 star table (host only): the
  one-leg names no column it does not read, so Γ's padded row stays inside
  the kept columns; the pre-fix sphere fillers are the red twin.

Scope: single-process meshes (``mesh(4)``: four emulated host devices, or
four GPUs in the mesh child).  The multi-process P16 timing, memory and
parity against a production ``V_qmunu`` live in the sandbox run
``runs/runtime/vq_summa_20260923``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from gw.v_q_g_flat import (  # noqa: E402
    _compute_V_q_g_flat_one_tile, _head_shell, _make_q_tile_kernel,
    _one_leg_columns, _plan_vq_tiles, vq_tile_bytes)


def _mesh22():
    devs = jax.devices()
    if len(devs) < 4:
        pytest.skip(f"needs 4 devices for a 2x2 mesh; have {len(devs)}")
    return Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))


def _zeta(rng, n_q, n_mu, n_g):
    return rng.standard_normal((n_q, n_mu, n_g)) + 1j * rng.standard_normal(
        (n_q, n_mu, n_g))


def _put(host, mesh, spec):
    return jax.device_put(jnp.asarray(host), NamedSharding(mesh, spec))


def _reference(zL, zR, v):
    return np.einsum("qmg,qg,qng->qmn", np.conj(zL), v, zR, optimize=True)


def _whole_face_kernel(mesh, n_l, n_r, ngkmax, g_chunk):
    """The pre-2026-09-23 per-q contraction, frozen: gather both whole faces
    (L onto 'x', R onto 'y'), then scan the G chunks locally."""
    wsc = jax.lax.with_sharding_constraint
    blk_x = NamedSharding(mesh, P("x", None))
    blk_y = NamedSharding(mesh, P("y", None))
    blk = NamedSharding(mesh, P("x", "y"))

    @jax.jit
    def fn(face_l, face_r, v_q):
        L = wsc(face_l, blk_x)
        R = wsc(face_r, blk_y)
        V = wsc(jnp.zeros((n_l, n_r), dtype=L.dtype), blk)

        def body(Vc, i):
            s = i * g_chunk
            Lc = jax.lax.dynamic_slice_in_dim(L, s, g_chunk, axis=-1)
            Rc = jax.lax.dynamic_slice_in_dim(R, s, g_chunk, axis=-1)
            vc = jax.lax.dynamic_slice_in_dim(v_q, s, g_chunk, axis=0)
            return Vc + (jnp.conj(Lc) * vc[None, :]) @ Rc.T, None

        V, _ = jax.lax.scan(body, V, jnp.arange(ngkmax // g_chunk), unroll=1)
        return wsc(V, blk), L[:, 0]

    return fn


def _run_tile_kernel(mesh, zL, zR, v, g_chunk, *, same_zeta):
    n_q, n_l, ngk = zL.shape
    n_r = zR.shape[1]
    kern = _make_q_tile_kernel(mesh, n_l, n_r, ngk, g_chunk, n_q,
                               write_g0=True, same_zeta=same_zeta)
    V_acc = _put(np.zeros((n_q, n_l, n_r), complex), mesh, P(None, "x", "y"))
    g0 = _put(np.zeros((n_q, n_l), complex), mesh, P(None, "x"))
    zl = _put(zL, mesh, P(None, ("x", "y"), None))
    zr = zl if same_zeta else _put(zR, mesh, P(None, ("x", "y"), None))
    V_acc, g0 = kern(V_acc, g0, zl, zr, _put(v, mesh, P(None, None)),
                     jnp.int32(0))
    return np.asarray(V_acc), np.asarray(g0)


@pytest.mark.mesh(4)
@pytest.mark.parametrize("same_zeta", [True, False])
def test_panel_kernel_matches_whole_face_kernel(same_zeta):
    mesh = _mesh22()
    rng = np.random.default_rng(20260923)
    n_q, n_l, n_r, ngk, g_chunk = 3, 8, (8 if same_zeta else 12), 48, 12
    zL = _zeta(rng, n_q, n_l, ngk)
    zR = zL if same_zeta else _zeta(rng, n_q, n_r, ngk)
    v = rng.standard_normal((n_q, ngk)) + 0.1j * rng.standard_normal((n_q, ngk))

    V_new, g0_new = _run_tile_kernel(mesh, zL, zR, v, g_chunk,
                                     same_zeta=same_zeta)

    old = _whole_face_kernel(mesh, n_l, n_r, ngk, g_chunk)
    face_sh = P(("x", "y"), None)
    V_old = np.empty_like(V_new)
    for q in range(n_q):
        Vq, g0q = old(_put(zL[q], mesh, face_sh), _put(zR[q], mesh, face_sh),
                      _put(v[q], mesh, P(None)))
        V_old[q] = np.asarray(Vq)
        np.testing.assert_array_equal(g0_new[q], np.asarray(g0q))

    ref = _reference(zL, zR, v)
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(V_new - ref)) <= 1e-13 * scale
    assert np.max(np.abs(V_new - V_old)) <= 1e-13 * scale
    if jax.devices()[0].platform == "cpu":
        # Same local GEMMs on the same operands: the restructure moved the
        # gathers, not the arithmetic.
        np.testing.assert_array_equal(V_new, V_old)


@pytest.mark.mesh(4)
def test_panel_kernel_masks_an_indivisible_g_tail():
    mesh = _mesh22()
    rng = np.random.default_rng(7)
    n_q, n_mu, ngk, g_chunk = 2, 8, 50, 16          # 50 = 3*16 + 2
    z = _zeta(rng, n_q, n_mu, ngk)
    v = rng.standard_normal((n_q, ngk)) + 0j
    ref = _reference(z, z, v)
    V, g0 = _run_tile_kernel(mesh, z, z, v, g_chunk, same_zeta=True)
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(V - ref)) <= 1e-13 * scale
    np.testing.assert_array_equal(g0, z[:, :, 0])
    # Red twin: the clamped last chunk re-covers columns 34..47, so without
    # the mask those columns count twice — the reference must see that.
    doubled = ref + _reference(z[:, :, 34:48], z[:, :, 34:48], v[:, 34:48])
    assert np.max(np.abs(V - doubled)) > 1e-6 * scale


class _ArrayZetaLoader:
    """The ``ZetaLoader`` members the V tile uses, over an in-memory ζ."""

    zeta_layout = "G_flat"

    def __init__(self, zeta, gvec, mesh):
        self._zeta, self.gvec_components, self._mesh = zeta, gvec, mesh
        self.n_rmu = zeta.shape[1]
        self.reads = []

    def read_zeta_G_slab(self, *, q_offset, q_count, mu_offset, mu_count,
                         mesh=None):
        self.reads.append((q_offset, q_count))
        block = np.zeros((q_count, mu_count, self._zeta.shape[2]), complex)
        src = self._zeta[q_offset:q_offset + q_count, mu_offset:]
        block[:, :src.shape[1]] = src[:, :mu_count]
        return _put(block, mesh or self._mesh, P(None, ("x", "y"), None))


@pytest.mark.mesh(4)
def test_budget_tiles_the_zeta_read_and_leaves_v_unchanged():
    mesh = _mesh22()
    rng = np.random.default_rng(11)
    kgrid, n_mu, ngk = (5, 1, 1), 6, 40               # 6 pads to 8 on 2x2
    n_q = 5
    zeta = _zeta(rng, n_q, n_mu, ngk)
    gvec = np.zeros((n_q, 3, ngk), dtype=np.int32)
    v_table = rng.standard_normal((n_q, ngk)) + 0j

    def run(budget):
        loader = _ArrayZetaLoader(zeta, gvec, mesh)
        V, g0 = _compute_V_q_g_flat_one_tile(
            loader, None, v_per_G_builder=lambda q, g: v_table,
            kgrid=kgrid, fft_grid=(4, 4, 4), mesh_xy=mesh, g_chunk=8,
            sym=None, centroid_indices=None, is_charge_cc=True,
            write_g0=True, one_leg_action="scalar", timing_label="test",
            verbose=True, budget_bytes=budget)
        return np.asarray(V), np.asarray(g0), loader.reads

    V1, g01, reads1 = run(1e12)
    assert reads1 == [(0, 5)], reads1                 # every q fits: one read
    # vq_tile_bytes: V_acc+g0 = 16*(5*64/4 + 5*8/2) = 1600 B; one q =
    # 16*(8*40/4 + 40) = 1920 B; work = faces 16*16*40/4 + carry 16*2*64/4
    # + panels 16*8*(2*8/2 + 8/2) = 4608 B.  An 11 kB budget leaves room for
    # two q per read: tiles (0,2),(2,2),(4,1).
    V2, g02, reads2 = run(11_000)
    assert reads2 == [(0, 2), (2, 2), (4, 1)], reads2
    np.testing.assert_array_equal(V2, V1)
    np.testing.assert_array_equal(g02, g01)
    ref = _reference(zeta, zeta, v_table)
    assert np.max(np.abs(V1[:, :n_mu, :n_mu] - ref)) <= 1e-13 * np.max(
        np.abs(ref))
    assert not np.any(V1[:, n_mu:, :]) and not np.any(V1[:, :, n_mu:])
    with pytest.raises(ValueError, match="GATE vq_tile_budget"):
        run(8_000)                                    # < 1600 + 4608 + 1920


@pytest.mark.mesh(4)
def test_host_read_staging_bounds_the_q_tile():
    """The phdf5 read keeps each open file's largest tile staged on the host,
    so the q-tile also fits the host budget, whatever the device allows."""
    mesh = _mesh22()
    shape = dict(n_q=10, n_rmu_L=8, n_rmu_R=8, ngkmax=40, same_zeta=True,
                 n_sub=0)
    host_q = vq_tile_bytes(**shape, p_x=2, p_y=2, g_chunk=8)["host_per_q"]
    assert host_q == 16 * 8 * 40 / 4
    plan = dict(shape, mesh_xy=mesh, g_chunk=8, budget_bytes=1e12)
    assert _plan_vq_tiles(**plan)[0] == 10                 # device-only: all q
    q_tile, _, priced = _plan_vq_tiles(**plan, host_budget_bytes=3.5 * host_q)
    assert q_tile == 3 and priced["n_tiles"] == 4          # 3,3,3,1 -> 3
    assert priced["host_staged"] <= 3.5 * host_q
    with pytest.raises(ValueError, match="GATE vq_tile_budget"):
        _plan_vq_tiles(**plan, host_budget_bytes=0.5 * host_q)


def _inversion_star(kgrid, n_mu, rng):
    """A duck-typed ``SymMaps`` for {E, I} on a 2D grid, no time reversal.

    Only the attributes the one-leg service reads.  Under I the parent of
    q_full is -q_full wrapped, so ``I (q_p + G_p) = q_full`` needs
    ``G_p = -(q_p + q_full)``, which is nonzero on the zone boundary.
    """
    from symmetry_maps import bgw_integer_q_to_fractional
    nx, ny, _ = kgrid
    ints = np.array([(i, j, 0) for i in range(nx) for j in range(ny)])
    lin = {tuple(k): n for n, k in enumerate(ints.tolist())}
    parents, irr_idx, rows = [], np.empty(len(ints), np.int32), np.empty(
        len(ints), np.int32)
    for n, k in enumerate(ints):
        partner = lin[tuple((-k) % np.array([nx, ny, 1]))]
        rep = min(n, partner)
        if rep not in parents:
            parents.append(rep)
        irr_idx[n] = parents.index(rep)
        rows[n] = 0 if n == rep else 1
    eye = np.eye(3, dtype=np.int64)
    sym_mats_k = np.stack([eye, -eye, -eye, eye])      # spatial | TRS halves
    perm = np.stack([np.arange(n_mu), rng.permutation(n_mu)] * 2)
    wraps = np.zeros((4, n_mu, 3))
    wraps[1] = wraps[3] = rng.integers(-1, 2, size=(n_mu, 3))
    q_int = ints[parents]
    sym = SimpleNamespace(
        kvecs_asints=ints, q_irr_kgrid_int=q_int, irr_idx_q=irr_idx,
        sym_mats_k=sym_mats_k, sym_matrices=sym_mats_k[:2],
        translations=np.zeros((2, 3)))
    return sym, rows, perm, wraps, bgw_integer_q_to_fractional(q_int, kgrid)


@pytest.mark.mesh(4)
def test_one_leg_sub_sphere_matches_the_whole_sphere():
    from symmetry_maps import isdf_one_leg_source_slots, unfold_isdf_one_leg
    mesh = _mesh22()
    rng = np.random.default_rng(3)
    kgrid, n_mu = (4, 4, 1), 8
    sym, rows, perm, wraps, q_frac = _inversion_star(kgrid, n_mu, rng)
    n_q = q_frac.shape[0]
    # Every parent sphere holds the same 27 Miller vectors, shuffled per q.
    ball = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
                     for c in (-1, 0, 1)], dtype=np.int32)
    gvec = np.stack([ball[rng.permutation(len(ball))].T for _ in range(n_q)])
    zeta = _zeta(rng, n_q, n_mu, gvec.shape[2])
    args = dict(sym=sym, sym_idx=rows, sym_perm=perm, L_table=wraps,
                q_irr_frac=q_frac, kgrid=kgrid, mesh_xy=mesh,
                component_action="scalar")
    spec = P(None, ("x", "y"), None)

    whole = np.asarray(unfold_isdf_one_leg(
        _put(zeta, mesh, spec), gvec_components=gvec, **args))

    cols, g_sub = _one_leg_columns(gvec, sym=sym, sym_idx=rows,
                                   q_irr_frac=q_frac, kgrid=kgrid,
                                   fft_grid=(8, 8, 8))
    z_sub = np.take_along_axis(zeta, cols[:, None, :], axis=2)
    slots = isdf_one_leg_source_slots(
        gvec, sym=sym, sym_idx=rows, q_irr_frac=q_frac, kgrid=kgrid)
    parent = sym.irr_idx_q
    nonzero = [iq for iq in range(len(slots))
               if np.any(gvec[parent[iq], :, slots[iq]])]
    assert nonzero                                    # a nonzero parent G
    assert cols.shape == (n_q, 2)                     # {G=0, G_p} at most
    sub = np.asarray(unfold_isdf_one_leg(
        _put(z_sub, mesh, spec), gvec_components=g_sub, **args))
    np.testing.assert_array_equal(sub, whole)

    # Control: the named columns are load-bearing.  Replace the nonzero-G
    # column one full q needs with an unneeded one and the service refuses.
    p = int(parent[nonzero[0]])
    j = int(np.flatnonzero(cols[p] == slots[nonzero[0]])[0])
    spare = int(np.setdiff1d(np.arange(gvec.shape[2]), cols[p])[0])
    broken = cols.copy()
    broken[p, j] = spare
    with pytest.raises(ValueError, match="GATE isdf_one_leg_parent_g"):
        unfold_isdf_one_leg(
            _put(np.take_along_axis(zeta, broken[:, None, :], axis=2), mesh,
                 spec),
            gvec_components=np.take_along_axis(
                gvec, broken[:, None, :], axis=2), **args)


# Si 4x4x4 (fcc, 48 ops + TRS, orbit-closed 480 centroids): per IBZ parent,
# its star size and the distinct parent slots its full q's literal G=0 come
# from (``isdf_one_leg_source_slots``), measured on the kconv-stage-2 Si BSE
# deck: runs/runtime/route_g_head_shell_20260924/d00_diag_filereuse/diag.txt.
_SI444_STARS = [(1, [0]), (8, [0]), (4, [0, 9]), (6, [0]), (24, [0, 9, 73]),
                (12, [0, 1, 67]), (3, [0]), (6, [0, 1, 65, 551])]


def test_route_g_shell_keeps_what_the_one_leg_reads_si444(monkeypatch):
    import symmetry_maps
    from common.gvec_fft_box import fft_box_pad_sentinel
    from isdf.zeta_mubatch import shell_positions
    parent = np.concatenate([np.full(n, p, np.int32)
                             for p, (n, _) in enumerate(_SI444_STARS)])
    slots = np.concatenate([np.resize(np.asarray(need, np.int32), n)
                            for n, need in _SI444_STARS])
    assert parent.size == 64
    # 600 distinct Miller vectors per parent (slot 0 is G = 0).
    box = np.array([(a, b, c) for a in range(-4, 5) for b in range(-4, 5)
                    for c in range(-4, 5)], dtype=np.int32)
    box = box[np.argsort(np.sum(box * box, axis=1), kind="stable")][:600]
    gvec = np.stack([box.T] * len(_SI444_STARS))
    fft_grid = (20, 20, 20)
    monkeypatch.setattr(symmetry_maps, "isdf_one_leg_source_slots",
                        lambda *a, **k: slots)

    cols, g_sub = _one_leg_columns(
        gvec, sym=SimpleNamespace(irr_idx_q=parent), sym_idx=None,
        q_irr_frac=None, kgrid=(4, 4, 4), fft_grid=fft_grid)
    sentinel = fft_box_pad_sentinel(fft_grid)[0]
    assert cols.shape == (8, 4)
    for p, (_, need) in enumerate(_SI444_STARS):
        assert set(cols[p].tolist()) == set(need)        # no unread column
        np.testing.assert_array_equal(cols[p, :len(need)], need)
        np.testing.assert_array_equal(g_sub[p][:, :len(need)], gvec[p][:, need])
        assert np.all(g_sub[p][:, len(need):] == sentinel[:, None])
    np.testing.assert_array_equal(cols[0], [0, 0, 0, 0])  # Γ: slot 0 only

    # The kept columns are the union the consumers name; head slots join it.
    head_sel = np.zeros((8, 2), np.int32)
    head_sel[3] = (5, 6)
    keep = _head_shell(8, cols, head_sel)
    np.testing.assert_array_equal(keep[0], np.zeros(keep.shape[1]))
    np.testing.assert_array_equal(keep[7], [0, 1, 65, 551])
    np.testing.assert_array_equal(np.unique(keep[3]), [0, 5, 6])
    pos = shell_positions(keep, cols)
    np.testing.assert_array_equal(np.take_along_axis(keep, pos, axis=1), cols)
    shell_positions(keep, head_sel)

    # Red twin: the pre-fix pads (the lowest sphere slots a star does not
    # use) name slot 1 at Γ, which nothing reads and the pass never formed.
    old = np.stack([np.r_[need, np.setdiff1d(np.arange(600), need)[:4 - len(need)]]
                    for _, need in _SI444_STARS]).astype(np.int32)
    with pytest.raises(ValueError, match=r"GATE zeta-mubatch-shell: got head "
                                         r"slot 1 at stored q 0"):
        shell_positions(keep, old)
