"""P4 portable restart seam for distinct four-spinor parent families."""
from pathlib import Path
import argparse
import json


def main():
    """Canonical files restore both packed parent faces without changing a byte."""
    from runtime import initialize_communicator_stack
    runtime = initialize_communicator_stack(platform="gpu")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.centroid_basis import PackedCentroidBasis
    from common.meta import Meta
    from common.wfn_transforms import load_centroids_band_chunked
    from file_io.centroids import load_centroids
    from file_io.tagged_arrays import write_restart_state_to_h5, read_restart_state_from_h5
    from gw.wavefunction_bundle import parent_faces
    from wfn_loader import WfnLoader

    parser = argparse.ArgumentParser()
    for name in ("wfn", "charge", "current"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--input")
    parser.add_argument("--bands", type=int, default=8)
    args = parser.parse_args()
    mesh = runtime.mesh
    assert mesh.size in (4, 16)
    wfn = WfnLoader(args.wfn, mesh=mesh)
    sym = wfn.symmetry()
    bases, packed, disk = [], [], []
    incoming = (None if args.input is None else
                read_restart_state_from_h5(args.input, mesh, low_mem_bands=True))
    for family, path in enumerate((args.charge, args.current)):
        _, points, count = load_centroids(path, wfn.fft_grid)
        basis = PackedCentroidBasis.build(points, sym, wfn.fft_grid, mesh)
        if incoming is None:
            meta = Meta.from_system(wfn, sym, 8, 0, args.bands, count, True,
                                    mesh_xy=mesh, mu_basis=basis)
            faces = parent_faces(*load_centroids_band_chunked(
                wfn, sym, meta, points, True, mesh, band_range=(0, args.bands),
                band_chunk_size=args.bands, k_domain="ibz", bispinor_lift="raw"), mesh_xy=mesh)
        else:
            source = incoming[13:15] if family == 0 else incoming[16:18]
            faces = tuple(basis.pack_axis(value, axis) for value, axis in zip(source, (3, 2)))
        bases.append(basis)
        packed.append(faces)
        disk.append(tuple(basis.unpack_axis(face, axis) for face, axis in zip(faces, (3, 2))))
    zero = jax.jit(lambda: jnp.zeros((1, 480, 480), jnp.complex128),
                   out_shardings=NamedSharding(mesh, P(None, "x", "y")))()
    write_restart_state_to_h5("parent_restart.h5", n_rmu_logical=480,
        n_rmu_transverse_logical=168, V_qmunu=zero,
        psi_parent_y=disk[0][0], psi_parent_y_mun=disk[0][1],
        psi_parent_y_transverse=disk[1][0], psi_parent_y_transverse_mun=disk[1][1],
        parent_k_rows=sym.kirr_fullids, mesh=mesh, mode="w")
    read = read_restart_state_from_h5("parent_restart.h5", mesh, low_mem_bands=True)
    errors = []
    for basis, faces, restored in zip(bases, packed, (read[13:15], read[16:18])):
        for face, value, axis in zip(faces, restored, (3, 2)):
            error = float(jnp.max(jnp.abs(face - basis.pack_axis(value, axis))))
            assert error == 0.0, error
            errors.append(error)
    if jax.process_index() == 0:
        Path("restart_parity.json").write_text(json.dumps({"absolute_errors": errors}) + "\n")
        print("BISPINOR_PARENT_RESTART_PASS", errors, flush=True)


if __name__ == "__main__":
    main()
