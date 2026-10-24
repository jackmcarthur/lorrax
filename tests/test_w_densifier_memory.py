"""Independent W interpolation values and bounded centroid-face scratch.

Run under the normal rank launcher on a compute node, for example P4 CPU or
CUDA: ``python tests/test_w_densifier_memory.py``.  The reference enumerates
lattice images by physical distance and uses NumPy FFTs.  The memory check
compiles large abstract operands; the numerical cases allocate small arrays.
"""


def _reference(w, pos, lattice, coarse, fine, output):
    """Wigner-Seitz interpolation by brute-force physical image distances."""
    import itertools
    import numpy as np
    from common.ws_images import WS_TIE_TOL_BOHR2
    wr = np.fft.ifftn(w, axes=(2, 3, 4), norm="ortho")
    out = np.zeros(w.shape[:2] + fine, np.complex128)
    translations = np.asarray(list(itertools.product(
        *[range(-3, 4) if n > 1 else [0] for n in coarse])))
    delta = pos[:, None, :] - pos[None, :, :]
    for index in np.ndindex(coarse):
        rep = np.asarray(index)
        rep = np.where(rep >= (np.asarray(coarse) + 1) // 2, rep - coarse, rep)
        images = rep + translations * np.asarray(coarse)
        cart = (delta[..., None, :] + images) @ lattice
        dist2 = np.sum(cart * cart, axis=-1)
        keep = dist2 <= dist2.min(axis=-1, keepdims=True) + WS_TIE_TOL_BOHR2
        weights = keep / keep.sum(axis=-1, keepdims=True)
        for image, weight in zip(images, np.moveaxis(weights, -1, 0)):
            target = tuple(image % fine)
            out[(slice(None), slice(None)) + target] += wr[(slice(None), slice(None)) + index] * weight
    out *= np.sqrt(np.prod(fine) / np.prod(coarse))
    return out if output == "R" else np.fft.fftn(out, axes=(2, 3, 4), norm="ortho")


def main():
    import runtime
    runtime.initialize_communicator_stack()
    import gc
    import json
    import weakref
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.ws_images import pair_ws_images, pair_ws_embed
    from bse.bse_densify import make_w_densifier, densify_w
    from bse.bse_ring_comm import create_mesh_2d
    from runtime import tiles
    mesh = create_mesh_2d()
    spec = P("x", "y", None, None, None)
    sh = NamedSharding(mesh, spec)
    divisor = int(np.lcm(*mesh.devices.shape))
    n = ((18 + divisor - 1) // divisor) * divisor
    rng = np.random.default_rng(309)
    cases = [((3, 2, 1), (5, 4, 1), np.array([[6., 0, 0], [-3, 5.196, 0], [0, 0, 20]]), 2),
             ((2, 2, 2), (3, 3, 3), np.array([[0., 5, 5], [5, 0, 5], [5, 5, 0]]), 3)]
    errors = {}
    old_tile = tiles.TILE_BYTES
    for cg, fg, lat, dim in cases:
        pos = rng.random((n, 3))
        pos[:4] = [[0, 0, 0], [.5, 0, 0], [0, .5, 0], [0, 0, 0]]
        w = rng.standard_normal((n, n) + cg) + 1j * rng.standard_normal((n, n) + cg)
        emb = pair_ws_images(cg, lat, pos, dim)
        for output in ("k", "R"):
            ref = _reference(w, pos, lat, cg, fg, output)
            for tile in (old_tile, 20000, 2000):
                tiles.TILE_BYTES = tile
                coarse = device_put_process_local(w, sh)
                coarse_ref = weakref.ref(coarse)
                local_refs = [weakref.ref(coarse.addressable_data(i))
                              for i in range(len(coarse.sharding.addressable_devices))]
                value = densify_w(coarse, mesh_xy=mesh, densifier=make_w_densifier(
                    mesh, spec, fg, embedding=emb, output=output))
                del coarse
                jax.block_until_ready(value)
                gc.collect()
                assert coarse_ref() is None, "interpolation retained its consumed coarse input"
                assert all(ref() is None for ref in local_refs), "coarse local storage is still held"
                got = np.asarray(gather_to_host(value))
                rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
                key = f"{dim}D/{output}/tile{tile}"
                errors[key] = rel
                assert rel < 1e-12, (key, rel)
            tiles.TILE_BYTES = old_tile
        # Rectangular row/column sets use the full table's candidates, and
        # reproduce their independent full-matrix reference including ties.
        xr = np.fft.ifftn(w, axes=(2, 3, 4), norm="ortho")
        rows, cols = slice(1, 7), slice(5, 14)
        rect = pair_ws_embed(xr[rows, cols], dict(emb, pos=pos[rows]), fg, pos_b=pos[cols])
        want = _reference(w, pos, lat, cg, fg, "R")[rows, cols]
        rect = np.asarray(rect) * np.sqrt(np.prod(fg) / np.prod(cg))
        errors[f"{dim}D/rectangular"] = float(np.max(np.abs(rect - want)) / np.max(np.abs(want)))
        assert errors[f"{dim}D/rectangular"] < 1e-12
        # The no-embedding identity remains usable by real-space consumers.
        for output in ("k", "R"):
            got = np.asarray(gather_to_host(make_w_densifier(mesh, spec, cg, output=output)(
                device_put_process_local(w, sh))))
            want = w if output == "k" else xr
            assert np.max(np.abs(got - want)) < 1e-12
    tiles.TILE_BYTES = old_tile
    nmu = ((10000 + divisor - 1) // divisor) * divisor
    pos = rng.random((nmu, 3)); pos[:, 2] = .5
    emb = pair_ws_images((3, 3, 1), cases[0][2], pos, 2)
    shape = (nmu, nmu, 3, 3, 1)
    abstract = jax.ShapeDtypeStruct(shape, np.complex128, sharding=sh)
    try:
        make_w_densifier(mesh, spec, (6, 6, 1), embedding=dict(emb, pos=pos[:-1])).lower(abstract)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched centroid positions must refuse before slicing")
    mem = make_w_densifier(mesh, spec, (6, 6, 1), embedding=emb).lower(abstract).compile().memory_analysis()
    # A complete fine-W-sized scratch carrier is unnecessary: all pairwise
    # work is independent.  This catches a layout copy after the tile scan.
    assert mem.temp_size_in_bytes < .5 * mem.output_size_in_bytes, mem
    if jax.process_index() == 0:
        print("W_DENSIFIER_MEMORY " + json.dumps(dict(errors=errors, temp_bytes=mem.temp_size_in_bytes,
              output_bytes=mem.output_size_in_bytes, max_relative=max(errors.values()))), flush=True)
        print("W_DENSIFIER_MEMORY PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
