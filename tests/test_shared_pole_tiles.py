"""The face pencil's tile-interleaved block order (``shared_pole_pencil``) on the CPU 2x2 mesh.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
On a p x p face a pencil axis made of logical blocks holds, on rank row (column) b,
piece b of every block. Joining, splitting and the adjoint must equal the
whole-matrix operations up to that permutation of rows and columns, and the CT
pencil (``ordered_cross_pencil``, ``joint_sector_pencil``) on the face must equal its
whole-matrix assembly once each side's columns are taken in the interleaved order.
"""
import numpy as np
import pytest
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _order(sizes, p):
    """Position k of the tile-interleaved order holds whole-order index order[k]."""
    from gw.shared_pole_pencil import join_vectors, split_vectors
    return join_vectors(split_vectors(np.arange(sum(sizes))[None], sizes, 1), p)[0]


def _random(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_tile_algebra_is_the_permuted_whole_matrix_algebra():
    from gw.shared_pole_pencil import (join_vectors, split_vectors, tile_adjoint, tile_block, tile_join,
                                       tile_split, interleave_tables)
    mesh = _mesh()
    face = NamedSharding(mesh, P(None, "x", "y"))
    rng = np.random.default_rng(1)
    rows, cols = (4, 6), (2, 4, 6)
    blocks = [[_random(rng, 2, r, c) for c in cols] for r in rows]
    whole = np.block(blocks)
    pr, pc = _order(rows, 2), _order(cols, 2)
    # Each block of a face matrix is itself an ordinary face matrix; joined, it is the
    # whole block matrix with its rows and columns in the interleaved order.
    got = tile_block([[jax.device_put(b, face) for b in row] for row in blocks], face)
    assert got.sharding.spec == P(None, "x", "y")
    assert np.array_equal(np.asarray(got), whole[:, pr][:, :, pc])
    parts = tile_split(got, cols, -1, face)
    for part, want in zip(parts, (np.concatenate([row[i] for row in blocks], axis=-2) for i in range(len(cols)))):
        assert np.array_equal(np.asarray(part), want[:, pr])
    # The adjoint of an interleaved matrix is the adjoint with the orders exchanged.
    assert np.array_equal(np.asarray(tile_adjoint(got, face)), np.conj(np.swapaxes(whole, -1, -2))[:, pc][:, :, pr])
    # Whole matrices (p = 1): the plain operations.
    assert np.array_equal(np.asarray(tile_join([blocks[0][0], blocks[0][1]], -1, None)),
                          np.concatenate([blocks[0][0], blocks[0][1]], axis=-1))
    v = rng.normal(size=(3, 12))
    assert np.array_equal(join_vectors(split_vectors(v, (4, 8), 2), 2), v)
    assert np.array_equal(join_vectors(split_vectors(v, (4, 8), 1), 2), v[:, _order((4, 8), 2)])
    tables = dict(order=np.arange(16).reshape(2, 8), points=np.arange(16.).reshape(2, 8) * 1j,
                  active=np.arange(24).reshape(2, 12) % 3 == 0, own=np.array([3, 4]))
    t = interleave_tables(tables, 2)
    assert np.array_equal(t["order"], tables["order"][:, _order((4, 4), 2)])
    assert np.array_equal(t["active"], tables["active"][:, _order((4, 4, 2, 2), 2)])
    assert t["own"] is tables["own"]


def test_face_cross_pencil_is_the_whole_matrix_cross_pencil():
    """``ordered_cross_pencil`` and ``joint_sector_pencil`` on the face, each sector's columns in
    the interleaved order, against whole matrices (local products, p = 1)."""
    from gw.shared_pole_execution import face_matmul
    from gw.shared_pole_local import _mm
    from gw.shared_pole_sectors import joint_sector_pencil, ordered_cross_pencil
    mesh = _mesh()
    face = NamedSharding(mesh, P(None, "x", "y"))
    put = lambda a: jax.device_put(a, face)
    rng = np.random.default_rng(4)
    b, nc, nt, fc, ft, ic_w, it_w, kc, kt = 2, 6, 8, 8, 4, 2, 2, 4, 6
    zc, zt = _random(rng, b, fc), _random(rng, b, ft)
    qc, qt, ic, it = _random(rng, b, nc, fc), _random(rng, b, nt, ft), _random(rng, b, nc, ic_w), _random(rng, b, nt, it_w)
    tc, ct, dct = _random(rng, b, nt, fc), _random(rng, b, nc, ft), _random(rng, b, nc, ft)
    moments = tuple(_random(rng, b, nc, nt) for _ in range(4))
    whole = ordered_cross_pencil((zc, qc, ic), (zt, qt, it), (tc, ct, dct), moments, matmul=_mm)
    # The face takes each sector's finite columns in its interleaved order (the tables'),
    # and returns its pencil rows and columns in that sector's interleaved order.
    oc_, ot_ = _order((fc // 2, fc // 2), 2), _order((ft // 2, ft // 2), 2)
    rc, rt = _order((fc // 2, fc // 2, ic_w, ic_w), 2), _order((ft // 2, ft // 2, it_w, it_w), 2)
    got = ordered_cross_pencil((zc[:, oc_], put(qc[:, :, oc_]), put(ic)), (zt[:, ot_], put(qt[:, :, ot_]), put(it)),
                               (put(tc[:, :, oc_]), put(ct[:, :, ot_]), put(dct[:, :, ot_])),
                               tuple(put(m) for m in moments), matmul=face_matmul(mesh), matrix_sharding=face)
    for g, w, (r, c) in zip(got, whole, ((rc, rt), (rc, rt), (None, rc), (None, rt))):
        w = np.asarray(w)
        w = w[:, :, c] if r is None else w[:, r][:, :, c]
        assert np.allclose(np.asarray(g), w, rtol=1e-12, atol=1e-12)
    # The joint pencil: spans Y [b, side, K] with rows in each sector's pencil order.
    sc, st = fc + 2 * ic_w, ft + 2 * it_w
    yc, yt = _random(rng, b, sc, kc), _random(rng, b, st, kt)
    vc, vt = rng.normal(size=(b, kc)), rng.normal(size=(b, kt))
    own_c, own_t = _random(rng, b, nc, sc), _random(rng, b, nt, st)
    cross = (whole[1], whole[0])
    want = joint_sector_pencil((yc, vc, own_c, whole[2]), (yt, vt, own_t, whole[3]), cross, matmul=_mm)
    rows = lambda a, r: a[:, r]
    got = joint_sector_pencil((put(rows(yc, rc)), vc, put(own_c[:, :, rc]), got[2]),
                              (put(rows(yt, rt)), vt, put(own_t[:, :, rt]), got[3]),
                              (got[1], got[0]), matmul=face_matmul(mesh), matrix_sharding=face)
    joint = _order((kc, kt), 2)
    for g, w, square in zip(got, want, (True, True, False, False)):
        w = np.asarray(w)
        w = w[:, joint][:, :, joint] if square else w[:, :, joint]
        assert np.allclose(np.asarray(g), w, rtol=1e-12, atol=1e-12)
