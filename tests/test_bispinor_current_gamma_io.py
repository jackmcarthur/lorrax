"""Actual Gamma current IO through public slab/Qirr/centroid owners.

The P4 entry is ``exercise_current_gamma_io``.  It uses shared fixture paths
and all-rank collective order; the CPU pytest wrapper uses a private temp
directory.  This is an IO/packing test, not a material-current calculation.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _fixture(directory, mesh, kind, case):
    """Adapt the public reader-coordinate fixture to three q rows and P4."""
    import h5py
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import barrier, device_put_process_local
    from common.centroid_basis import PackedCentroidBasis
    from file_io.slab_io import SlabIO
    from gw.v_q_bispinor import (UNIQUE_TILES, tile_dataset_name,
                                _publish_unique_tile_inventory)
    from symmetry_maps import (QirrTables, bgw_integer_q_to_fractional,
        stamp_qirr_tensor, verify_centroid_orbit_closure)

    directory = Path(directory)
    tag = "current_gamma_io." + kind + "." + case
    if jax.process_index() == 0:
        directory.mkdir(parents=True, exist_ok=True)
    barrier(tag + ".directory")
    grid = np.asarray([8, 10, 12], np.int32)
    points = (np.asarray([[1, 2, 3], [5, 6, 7], [3, 8, 9]], np.int32)
        if kind == "fft_indices" else np.asarray([
            [.123456789, .234567891, .345678912],
            [.623456789, .734567891, .845678912],
            [.373456789, .834567891, .945678912]], np.float64))
    first_components = {"valid": [1, 0, -1],
                        "missing": [1, 2, -1],
                        "duplicate": [1, 0, 0]}[case]
    q_integer = np.zeros((3, 3), np.int32)
    q_integer[:, 0] = first_components
    # Gamma is stored at row one, not at the first row.  Full q labels
    # map to the corresponding permuted wedge; TR is physically disabled.
    sym = SimpleNamespace(sym_matrices=np.eye(3, dtype=np.int32)[None],
        translations=np.zeros((1, 3)), irr_idx_q=np.asarray([1, 0, 2], np.int32),
        sym_idx_q=np.zeros(3, np.int32), q_irr_full_idx=np.asarray([1, 0, 2], np.int32),
        q_irr_kgrid_int=q_integer, trs_allowed=False,
        active_symmetry_rows=np.asarray([0], np.int32))
    basis = PackedCentroidBasis.build(points, sym, grid, mesh, coordinate_kind=kind)
    physical = points / grid if kind == "fft_indices" else points
    closure = verify_centroid_orbit_closure(physical, sym.sym_matrices,
                                           tnp=sym.translations)
    perm = np.tile(np.arange(3, dtype=np.int32), (2, 1))
    wraps = np.zeros((2, 3, 3), np.int64)
    tables = QirrTables(irr_idx_q=sym.irr_idx_q, sym_idx_q=sym.sym_idx_q,
        q_irr_frac=bgw_integer_q_to_fractional(q_integer, (3, 1, 1)),
        sym_perm=perm, L_table=wraps, n_sym_spatial=1)
    packed_perm, packed_wraps = basis.pack_tables(perm, wraps)
    plan = SimpleNamespace(sym=sym, n_sym_spatial=1, fft_grid=grid,
        spatial_ops=sym.sym_matrices, translations=sym.translations,
        sym_perm=packed_perm, L_table=packed_wraps)
    path = directory / (kind + "_" + case + ".h5")
    expected = {}
    entries = np.arange(9, dtype=float).reshape(3, 3)
    with SlabIO(path, mode="w", mesh=mesh) as io:
        for number, pair in enumerate(UNIQUE_TILES):
            raw = np.stack([100 * number + 10 * q + entries
                + 1j * (.17 * entries.T - .13 * entries) for q in range(3)])
            if pair[0] == pair[1]:
                raw = (raw + raw.conj().transpose(0, 2, 1)) / 2
            expected[pair] = np.asarray(raw, np.complex128)
            # Public SlabIO clips the canonical suffix pad at the physical
            # dataset boundary; poison verifies those slots cannot persist.
            stage = np.full((3, basis.n_canonical, basis.n_canonical),
                            1e8 + 7e8j, np.complex128)
            stage[:, :3, :3] = raw
            name = tile_dataset_name(*pair)
            io.create_dataset(name, shape=raw.shape, dtype=np.complex128)
            array = device_put_process_local(stage, NamedSharding(mesh, P(None, "x", "y")))
            io.write_slab(name, array)
            io.sync_writes()
    barrier(tag + ".payload_closed")
    if jax.process_index() == 0:
        for pair in UNIQUE_TILES:
            stamp_qirr_tensor(path, tile_dataset_name(*pair), tables=tables,
                              closure_verdict=closure, n_rmu_logical=3)
        with h5py.File(path, "a") as f:
            for name, value in {"kgrid": np.asarray([3, 1, 1], np.int32),
                    "n_rmu_C": 3, "n_rmu_T": 3, "n_q_total": 3}.items():
                f.create_dataset(name, data=value)
            _publish_unique_tile_inventory(f, filename=path,
                n_q_total=3, n_rmu_C=3, n_rmu_T=3)
    barrier(tag + ".metadata_ready")
    return path, basis, plan, expected


def _expect_refusal(call, message):
    try:
        call()
    except ValueError as exc:
        if message not in str(exc):
            raise AssertionError("Wrong refusal: " + str(exc)) from exc
    else:
        raise AssertionError("Missing current-Gamma refusal: " + message)


def exercise_current_gamma_io(mesh, directory):
    """Run identical collective fixture/read order on every participating rank."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host
    from file_io.restart_bundle import BispinorVqReader
    from gw.v_q_bispinor import UNIQUE_TILES

    results = []
    for kind in ("fft_indices", "fractional"):
        path, basis, plan, expected = _fixture(directory, mesh, kind, "valid")
        host = lambda value: np.asarray(gather_to_host(value))
        with BispinorVqReader(path, mesh, mu_bases=(basis, basis),
                              family_plans=(plan, plan)) as reader:
            for pair in UNIQUE_TILES:
                got = host(reader.get_tile(*pair))
                literal = basis.pack_host(basis.pack_host(expected[pair], axis=-1), axis=-2)
                np.testing.assert_array_equal(got, literal)
                if pair != (0, 0):
                    gamma_array = reader.get_current_gamma_tile(*pair)
                    assert gamma_array.sharding.is_equivalent_to(
                        NamedSharding(mesh, P("x", "y")), 2)
                    gamma = host(gamma_array)
                    np.testing.assert_array_equal(gamma, literal[1])
                    np.testing.assert_array_equal(gamma, got[1])
                    assert not np.array_equal(gamma, got[0])
                    assert not gamma[~basis.active_mask].any()
                    assert not gamma[:, ~basis.active_mask].any()
            gamma_lower = host(reader.get_current_gamma_tile(2, 1))
            upper = basis.pack_host(basis.pack_host(expected[(1, 2)][1], axis=-1), axis=-2)
            np.testing.assert_array_equal(gamma_lower, upper.conj().T)
            np.testing.assert_array_equal(gamma_lower, host(reader.get_tile(2, 1))[1])
            _expect_refusal(lambda: reader.get_current_gamma_tile(0, 1),
                            "spatial blocks")
        with BispinorVqReader(path, mesh, mu_bases=(basis, basis),
                              family_plans=(plan, None)) as reader:
            _expect_refusal(lambda: reader.get_current_gamma_tile(1, 1),
                            "authenticated current family")
        for case in ("missing", "duplicate"):
            bad_path, bad_basis, bad_plan, _ = _fixture(directory, mesh, kind, case)
            with BispinorVqReader(bad_path, mesh, mu_bases=(bad_basis, bad_basis),
                                  family_plans=(bad_plan, bad_plan)) as reader:
                _expect_refusal(lambda: reader.get_current_gamma_tile(1, 1),
                                "exactly one physical Gamma row")
        results.append(dict(coordinate_kind=kind, stored_q_count=3,
            gamma_row=1, physical_mu=3, packed_mu=basis.n_packed,
            positive_unique_tiles=7, current_gamma_tiles=6,
            lower_adjoint=True, old_get_tile_parity=True,
            ghost_zero=True, negatives=["missing Gamma", "duplicate Gamma",
                                        "current family", "charge/current index"]))
    return dict(status="PASS_CURRENT_GAMMA_PUBLIC_IO_AND_PACKING_ONLY",
        cases=results, mesh_shape={name: int(size) for name, size in mesh.shape.items()},
        scope="Actual collective SlabIO/public Qirr metadata/current Gamma reads. Identity spatial group; no material current, source/frame pairing, arbitrary orbit permutation or GW admission.")


def test_current_gamma_public_reader_cpu(tmp_path):
    import jax
    from jax.sharding import Mesh
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    exercise_current_gamma_io(mesh, tmp_path)
