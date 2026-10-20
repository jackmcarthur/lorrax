"""Host-only canonical G0 handoff; no JAX execution/material BSE claim."""
import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


ROOT = Path(__file__).parent
BSE = module("source_fit_bse", ROOT/"cri3_full_microscopic/native_bse_probe.py")
PROBE = ROOT/"cri3_qgrid_750_campaign/native_frozen_map_probe.py"


def basis(logical=7, canonical=8, side=2):
    from common.centroid_basis import PackedCentroidBasis
    from common.grouped_layout import build_square_grouped_shard_layout
    # Nontrivial packing and interleaved pads, not a guessed prefix map.
    groups = np.array([0, 1, 0, 2, 3, 2, 4]) if logical == 7 else np.arange(logical)
    layout = build_square_grouped_shard_layout(groups, (side, side))
    return PackedCentroidBasis(SimpleNamespace(shape={"x": side, "y": side}),
        layout, np.arange(logical*3, dtype=np.int32).reshape(logical, 3), canonical)


def capture(vector, current_basis):
    tree = ast.parse(PROBE.read_text())
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
              and node.name == "capture_vq")
    calls = []
    meta = SimpleNamespace(mu_basis=current_basis, n_rmu=current_basis.n_logical)
    result = (object(), vector, object(), None)
    def original(*a, **kw):
        calls.append((a, kw))
        return result
    scope = dict(np=np, hashlib=hashlib, original_vq=original, G0=[], G0_receipts=[])
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(PROBE), "exec"), scope)
    args = (object(), object(), object(), meta, None, print)
    assert scope["capture_vq"](*args) is result
    assert calls == [(args, {})]
    assert scope["G0"][0] is not vector  # Only the small host vector is copied.
    return dict(meta=meta, G0_canonical=scope["G0"][0], G0_receipt=scope["G0_receipts"][0])


@pytest.mark.parametrize("padded", [False, True])
def test_original_fit_g0_capture_and_owner_pack_roundtrip(padded):
    b = basis()
    logical = np.arange(7)+1j*np.arange(7)[::-1]
    source = capture(np.pad(logical, (0, 1)) if padded else logical, b)
    actual = BSE.source_fit_g0(source)
    np.testing.assert_array_equal(actual, b.pack_host(logical))
    np.testing.assert_array_equal(b.unpack_host(actual), logical)
    assert np.all(actual[~b.active_mask] == 0)


def test_actual_material_extent3000_to3008_accepts_exact_inert_suffix():
    b = basis(3000, 3008, 4)
    logical = np.linspace(-1., 1., 3000).astype(complex)
    source = capture(np.pad(logical, (0, 8)), b)
    np.testing.assert_array_equal(BSE.source_fit_g0(source), b.pack_host(logical))
    assert source["G0_receipt"]["shape"] == [3008]


@pytest.mark.parametrize("value", [None, np.ones((1, 8)), np.ones(6),
                                   np.r_[np.ones(7), 1.e-200], np.r_[np.ones(7), np.nan]])
def test_missing_wrong_shape_or_nonzero_canonical_pad_refuses(value):
    b = basis()
    with pytest.raises(ValueError, match="exact source-fit G0"):
        BSE.source_fit_g0(dict(meta=SimpleNamespace(mu_basis=b, n_rmu=7), G0_canonical=value))


@pytest.mark.parametrize("field", ["canonical_value_sha256", "canonical_centroids_sha256",
                                   "producer", "q_fractional", "order"])
def test_source_value_basis_and_q0_association_refuse_tampering(field):
    source = capture(np.r_[np.arange(7), 0.].astype(complex), basis())
    source["G0_receipt"][field] = "incorrect"
    with pytest.raises(ValueError, match="producer/basis/value receipt"):
        BSE.source_fit_g0(source)
