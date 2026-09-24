"""P=4 bitwise gate: the per-k sphere index replaces the dense box table.

Loader tables 2026-09-23.  ``build_sphere_box_index`` (``(nk, ngk)`` flat box
cell per G slot, ``n_rtot + g`` on a pad) is the one ψ(G)↔box table.  This
gate holds it to the RETIRED dense ``(nk, nx, ny, nz)`` gather table, which
lives only here, as the reference:

* ψ(G) → box → ψ(r) (``_box_kernel`` + the local IFFT, and
  ``to_rchunk_inner``) and ψ(r) → ψ(G) (FFT + ``_sphere_gather``) are
  ``np.array_equal`` to the dense-table route on every local shard;
* ``psi_cylinder_tables`` built from the sphere list equals the dense-table
  derivation exactly (the plane route consumes nothing else);

on odd and mixed-parity grids, ragged spheres padded to a mesh multiple
through ``runtime.padding``, bands band-sharded over a 2×2 mesh with a
non-divisible logical count.  Red twin: two swapped slots of one k must
make both directions differ.  Parity class: bit-exact (every box value is a
copy; the FFT is the same executable on identical input).

Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/loader_tables_p4.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from common.shard_map import shard_map  # noqa: E402
from common.fft_helpers import local_fftn3, local_ifftn3  # noqa: E402
from common.gvec_fft_box import build_sphere_box_index  # noqa: E402
from common.wfn_transforms import (  # noqa: E402
    _box_kernel, _sphere_gather, psi_cylinder_tables, to_rchunk_inner,
    _plane_geometry)
from runtime.padding import padded_axis  # noqa: E402

XY = ("x", "y")
TAG = "[loader-tables-p4]"


def _fail(msg):
    print(f"{TAG} FAIL: {msg}", flush=True)
    raise RuntimeError(msg)


def _spheres(grid, nk, rng):
    """Ragged per-k G lists (Miller indices) inside a shifted ellipsoid."""
    n = np.asarray(grid)
    ax = [np.arange(-(v // 2), v - v // 2) for v in grid]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    rows = []
    for k in range(nk):
        kk = rng.uniform(-0.5, 0.5, 3)
        q = ((G + kk) / (0.42 * n)) ** 2
        sel = G[np.sum(q, axis=1) < 1.0]
        rows.append(sel[rng.permutation(sel.shape[0])].astype(np.int32))
    return rows


def _dense_reference(rows, grid, ngk):
    """The retired table: g_index[k, cell] = g (``ngk`` on an empty cell)."""
    nx, ny, nz = grid
    out = np.full((len(rows), nx, ny, nz), ngk, dtype=np.int32)
    for k, r in enumerate(rows):
        w = r % np.asarray(grid)
        out[k, w[:, 0], w[:, 1], w[:, 2]] = np.arange(r.shape[0])
    return out


def _dense_box(psi, dense):
    """The retired gather: one ``take`` from ψ padded with a zero slot."""
    nk, nb, ns, ngk = psi.shape
    pad = jnp.concatenate([psi, jnp.zeros((nk, nb, ns, 1), psi.dtype)], -1)
    return jax.vmap(lambda p, g: jnp.take(p, g, axis=-1, mode="clip"))(
        pad, dense)


def _dense_cylinder(dense, grid, axis, ngk):
    """The retired ``psi_cylinder_tables`` body, on the dense table."""
    n_a, (n_b, n_c), _ = _plane_geometry(grid, axis)
    g = np.asarray(dense)
    occ = g < ngk
    a = 1 + axis
    cols = np.flatnonzero(np.any(occ, axis=(0, a)).reshape(-1)).astype(np.int32)
    s = np.flatnonzero(np.any(occ, axis=tuple(i for i in range(4) if i != a))
                       ).astype(np.int32)
    g_t = np.moveaxis(g, a, -1).reshape(g.shape[0], n_b * n_c, n_a)
    cyl = g_t[:, cols][:, :, s]
    pfc = np.full((n_b * n_c,), cols.size, np.int32)
    pfc[cols] = np.arange(cols.size, dtype=np.int32)
    return cyl, s, pfc


def _equal_shards(a, b, label):
    for sa, sb in zip(a.addressable_shards, b.addressable_shards):
        if sa.index != sb.index:
            _fail(f"{label}: shard index mismatch")
        x, y = np.asarray(sa.data), np.asarray(sb.data)
        if not np.array_equal(x, y):
            _fail(f"{label}: max |Δ| = {np.max(np.abs(x - y)):.3e} (not bitwise)")


def _differs(a, b):
    """Global verdict: a pad-band rank sees zeros either way, so OR over ranks."""
    from jax.experimental import multihost_utils
    local = any(not np.array_equal(np.asarray(sa.data), np.asarray(sb.data))
                for sa, sb in zip(a.addressable_shards, b.addressable_shards))
    return bool(np.any(multihost_utils.process_allgather(np.int32(local))))


def _routes(mesh, grid, n_rtot):
    band = P(None, XY, None, None)
    box6 = P(None, XY, None, None, None, None)

    def up_new(psi, sidx):
        return local_ifftn3(_box_kernel(psi, sidx, fft_grid=grid),
                            axes=(-3, -2, -1), norm="ortho")

    def up_ref(psi, dense):
        return local_ifftn3(_dense_box(psi, dense), axes=(-3, -2, -1),
                            norm="ortho")

    def down_new(r, sidx):
        return _sphere_gather(local_fftn3(r, axes=(-3, -2, -1), norm="ortho"),
                              sidx)

    def down_ref(r, dense_cells, mask):
        box = local_fftn3(r, axes=(-3, -2, -1), norm="ortho")
        flat = box.reshape(*box.shape[:3], n_rtot)
        out = jax.vmap(lambda b, c: jnp.take(b, c, axis=-1))(flat, dense_cells)
        return jnp.where(mask[:, None, None, :], out, 0)

    def rchunk(psi, sidx):
        return to_rchunk_inner(psi, sidx, grid, jnp.int32(0), n_rtot,
                               norm="ortho")

    def rchunk_ref(psi, dense):
        return up_ref(psi, dense).reshape(*psi.shape[:3], n_rtot)

    sm = lambda f, ins, out: jax.jit(shard_map(
        f, mesh=mesh, in_specs=ins, out_specs=out, check_vma=False))
    return dict(
        up_new=sm(up_new, (band, P()), box6),
        up_ref=sm(up_ref, (band, P()), box6),
        down_new=sm(down_new, (box6, P()), band),
        down_ref=sm(down_ref, (box6, P(), P()), band),
        rchunk=sm(rchunk, (band, P()), P(None, XY, None, None)),
        rchunk_ref=sm(rchunk_ref, (band, P()), P(None, XY, None, None)),
    )


def main():
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), XY)
    P_ = int(mesh.size)
    rng = np.random.default_rng(20260923)
    rep = NamedSharding(mesh, P())
    band = NamedSharding(mesh, P(None, XY, None, None))
    receipts = []
    for grid in ((9, 7, 5), (15, 12, 11)):
        n_rtot = int(np.prod(grid))
        nk, ns, nb_logical = 3, 2, 6
        rows = _spheres(grid, nk, rng)
        ngk = np.asarray([r.shape[0] for r in rows])
        ngk_c = padded_axis(int(ngk.max()), P_, name="test G carrier").carrier
        nb_c = padded_axis(nb_logical, P_, name="test band carrier").carrier
        if nb_logical % P_ == 0 or ngk_c == int(ngk.max()) and len(set(ngk)) == 1:
            _fail("anti-tautology: the geometry must pad bands and be ragged")
        sidx_np = build_sphere_box_index(rows, grid, ngk_c)
        dense_np = _dense_reference(rows, grid, ngk_c)
        # The table is the dense one's inverse on the sphere, pads out of box.
        for k in range(nk):
            live = sidx_np[k, :ngk[k]]
            if not np.array_equal(dense_np[k].reshape(-1)[live],
                                  np.arange(ngk[k])):
                _fail(f"grid {grid} k={k}: sphere index is not the inverse")
            if not np.array_equal(sidx_np[k, ngk[k]:],
                                  n_rtot + np.arange(ngk[k], ngk_c)):
                _fail(f"grid {grid} k={k}: pad slots are not n_rtot+g")
        psi_np = (rng.standard_normal((nk, nb_c, ns, ngk_c))
                  + 1j * rng.standard_normal((nk, nb_c, ns, ngk_c)))
        psi_np[:, nb_logical:] = 0.0
        for k in range(nk):
            psi_np[k, :, :, ngk[k]:] = 0.0
        mask_np = np.arange(ngk_c)[None, :] < ngk[:, None]
        cells_np = np.where(mask_np, sidx_np, 0).astype(np.int32)

        psi = jax.make_array_from_callback(psi_np.shape, band, lambda i: psi_np[i])
        put = lambda a: jax.make_array_from_callback(a.shape, rep, lambda i: a[i])
        sidx, dense = put(sidx_np), put(dense_np)
        cells, mask = put(cells_np), put(mask_np)
        f = _routes(mesh, grid, n_rtot)

        r_new, r_ref = f["up_new"](psi, sidx), f["up_ref"](psi, dense)
        _equal_shards(r_new, r_ref, f"{grid} ψ(G)→ψ(r)")
        g_new, g_ref = f["down_new"](r_new, sidx), f["down_ref"](r_ref, cells, mask)
        _equal_shards(g_new, g_ref, f"{grid} ψ(r)→ψ(G)")
        _equal_shards(f["rchunk"](psi, sidx), f["rchunk_ref"](psi, dense),
                      f"{grid} to_rchunk_inner")
        err = max(float(np.max(np.abs(np.asarray(s.data) - psi_np[s.index])))
                  for s in g_new.addressable_shards)
        if err > 1e-12:
            _fail(f"{grid}: round trip |Δ| {err:.3e}")
        for axis in range(3):
            cyl = psi_cylinder_tables(sidx, grid, axis, ngkmax=ngk_c)
            ref = _dense_cylinder(dense_np, grid, axis, ngk_c)
            for got, want, name in zip(cyl, ref, ("cyl_index", "cyl_axis",
                                                  "plane_from_col")):
                if not np.array_equal(np.asarray(got), want):
                    _fail(f"{grid} axis {axis}: {name} differs")

        # RED TWIN: swap two live slots of k=0 in the sphere list only.
        bad = sidx_np.copy()
        bad[0, [0, 1]] = bad[0, [1, 0]]
        bad_d = put(bad)
        if not _differs(f["up_new"](psi, bad_d), r_ref):
            _fail(f"{grid}: red twin (swapped slots) passed ψ(G)→ψ(r)")
        if not _differs(f["down_new"](r_ref, bad_d), g_ref):
            _fail(f"{grid}: red twin (swapped slots) passed ψ(r)→ψ(G)")
        receipts.append(
            f"grid={grid} nk={nk} ngk={ngk.tolist()}→{ngk_c} "
            f"nb={nb_logical}→{nb_c} table {sidx_np.nbytes}B vs dense "
            f"{dense_np.nbytes}B")
    if jax.process_index() == 0:
        for r in receipts:
            print(f"{TAG} {r}", flush=True)
        print(f"{TAG} PASS world={jax.process_count()} bitwise up/down/rchunk/"
              "cylinder on 2 grids × 3 axes; red twin fired on both grids",
              flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        finalize_process()
