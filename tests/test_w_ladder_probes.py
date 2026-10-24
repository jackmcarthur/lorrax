"""Exact host probe construction and bounded chunk allocation, without MPI."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _host_probe_builder(numpy=np):
    # Importing w_ladder starts the native/JAX runtime. Extract its pure NumPy
    # producer so this host-memory regression also runs before an allocation.
    source = Path(__file__).parents[1] / "src/bse/w_ladder.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_identity_probe_blocks")
    namespace = {"np": numpy}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


def test_identity_probe_chunks_match_dense_columns_and_zero_tail():
    build = _host_probe_builder()
    # Includes P4/P9-compatible padding, a one-row tail, and full-width probes.
    for logical, padded, chunk in ((7, 8, 2), (17, 18, 3), (9, 12, 4),
                                   (7, 8, 8), (8, 8, 2)):
        dense = np.eye(padded, dtype=np.float64)
        for c0, n_real, probes in build(logical, padded, chunk):
            expected = np.zeros((chunk, padded), dtype=np.float64)
            expected[:n_real] = dense[c0:c0 + n_real]
            np.testing.assert_array_equal(probes, expected)
            assert probes.dtype == np.float64
        # The q/z sweep needs a fresh iterable for each repeated visit.
        assert sum(n for _, n, _ in build(logical, padded, chunk)) == logical


def test_large_probe_stream_never_allocates_dense_identity_or_all_chunks():
    allocations = []

    def zeros(shape, *, dtype):
        assert shape == (64, 10048)
        allocations.append(shape)
        return np.zeros(shape, dtype=dtype)

    def eye(*_args, **_kwargs):
        raise AssertionError("A dense host identity defeats probe chunking")

    build = _host_probe_builder(SimpleNamespace(zeros=zeros, arange=np.arange,
                                               float64=np.float64, eye=eye))
    blocks = build(10000, 10048, 64)
    assert not allocations
    for block_number in range(2):
        c0, n_real, probes = next(blocks)
        assert len(allocations) == block_number + 1
        assert (c0, n_real) == (block_number * 64, 64)
        assert probes.nbytes == 64 * 10048 * 8
        assert np.count_nonzero(probes) == n_real
