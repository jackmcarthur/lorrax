"""Operator I/O refuses logical extents before distributed layout conversion.

The P4 entry point verifies the valid optimized conversion on a literal
nonidentity six-point layout with two inert suffix/packed slots.
"""
from types import SimpleNamespace

import numpy as np
import pytest


def _basis(mesh, *, identity=False):
    from common.centroid_basis import PackedCentroidBasis

    first = np.asarray([[.11, .17, .23], [.19, .31, .43], [.27, .41, .59]])
    points = np.concatenate((first, 1. - first))
    sym = SimpleNamespace(
        sym_matrices=np.asarray([np.eye(3), -np.eye(3)], np.int32),
        translations=np.zeros((2, 3)))
    basis = PackedCentroidBasis.build(
        points, sym, (4, 4, 6), mesh, identity=identity,
        coordinate_kind="fractional")
    assert basis.n_logical == 6 and basis.n_canonical == basis.n_packed == 8
    if not identity:
        literal = np.asarray([0, 3, 2, 5, 1, 4, -1, -1])
        np.testing.assert_array_equal(basis.layout.axis.packed_to_canonical, literal)
        assert not basis.is_identity
    else:
        assert basis.is_identity
    return basis


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("method", ["pack_operator", "unpack_operator"])
@pytest.mark.parametrize("shape", [(1, 6, 6), (1, 6, 8), (1, 8, 6), (8,)])
def test_wrong_operator_carrier_refuses_before_layout_dispatch(monkeypatch, identity, method, shape):
    from common.centroid_basis import PackedCentroidBasis

    mesh = SimpleNamespace(shape={"x": 2, "y": 2})
    basis = _basis(mesh, identity=identity)
    def forbidden(*args, **kwargs):
        raise AssertionError("malformed operator reached a layout executable")
    monkeypatch.setattr(PackedCentroidBasis, "_operator_kernel", forbidden)
    carrier = "canonical" if method == "pack_operator" else "packed"
    with pytest.raises(ValueError, match=f"expects the {carrier} carrier 8 on both trailing axes"):
        getattr(basis, method)(np.zeros(shape), spec=(None, "x", "y"))


def check_p4(runtime):
    """Compare valid P4 permutation with literal host rows on both axes."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host

    if runtime.process_count != 4 or dict(runtime.mesh.shape) != {"x": 2, "y": 2}:
        raise ValueError("operator-carrier proof requires the actual P4 mesh")
    basis = _basis(runtime.mesh)
    spec = P(None, "x", "y")
    labels = np.arange(2 * 6 * 6, dtype=np.float64).reshape(2, 6, 6)
    physical = (labels + 1.) / 37. + 1j * (labels[::-1] + 2.) / 53.
    canonical = np.zeros((2, 8, 8), np.complex128)
    canonical[:, :6, :6] = physical
    literal = np.asarray([0, 3, 2, 5, 1, 4, -1, -1])
    valid = literal >= 0
    expected = canonical[:, np.maximum(literal, 0)][:, :, np.maximum(literal, 0)]
    expected *= valid[None, :, None] * valid[None, None, :]
    put = lambda value: device_put_process_local(value, NamedSharding(runtime.mesh, spec))
    packed = basis.pack_operator(put(canonical), spec=spec)
    actual = np.asarray(gather_to_host(packed))
    np.testing.assert_array_equal(actual, expected)
    restored = np.asarray(gather_to_host(basis.unpack_operator(packed, spec=spec)))
    np.testing.assert_array_equal(restored, canonical)
    # Six is divisible by each face axis, so JAX admits the original malformed
    # distributed shape. The basis must refuse it before its fused kernel.
    for method in (basis.pack_operator, basis.unpack_operator):
        for shape in ((2, 6, 6), (2, 6, 8), (2, 8, 6)):
            with pytest.raises(ValueError, match="on both trailing axes"):
                method(put(np.zeros(shape, np.complex128)), spec=spec)
    assert np.max(np.abs(actual - canonical)) > .1
    return dict(status="PASS", logical_mu=6, canonical_mu=8, packed_mu=8,
                literal_packed_to_canonical=literal.tolist(),
                packed_max_error=float(np.max(abs(actual - expected))),
                canonical_roundtrip_max_error=float(np.max(abs(restored - canonical))),
                ghost_maximum=float(np.max(abs(actual[:, ~valid]))),
                malformed_distributed_shapes_refused=6)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from runtime import initialize_communicator_stack, run_main_and_finalize

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runtime = initialize_communicator_stack()
    def main():
        import jax
        result = check_p4(runtime)
        if jax.process_index() == 0:
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        return 0
    run_main_and_finalize(main)
