"""Host-only discovery pins: filesystem history cannot choose an ISDF basis.

Exercise each driver's actual import and resolver call, isolated from startup
and eigensolves. This is entry-seam coverage, not a full numerical driver run.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from bse import bse_loading


@pytest.fixture(params=["bse_jax", "bse_feast", "absorption_haydock", "exciton_bands"])
def resolve(request):
    """Execute the source's import binding and discovery expression verbatim."""
    path = Path(bse_loading.__file__).with_name(request.param + ".py")
    tree = ast.parse(path.read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and any(alias.name == "_find_restart_file" for alias in node.names)]
    assert len(imports) == 1
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "_find_restart_file"]
    assert len(calls) == 1, "each entry must resolve its restart once"
    namespace = {"__package__": "bse"}
    exec(compile(ast.Module(body=imports, type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["_find_restart_file"] is bse_loading._find_restart_file
    expression = compile(ast.Expression(body=calls[0]), str(path), "eval")

    def invoke(input_file):
        return eval(expression, namespace,
                    {"input_file": input_file, "args": SimpleNamespace(input=input_file)})

    return invoke


def _bundle(path, nmu):
    """A small ready canonical tensor bundle with a distinguishable μ basis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, factor, ready in [("V_qmunu", 2., "V_ready"),
                                    ("W0_qmunu", 1., "W0_ready")]:
            d = f.create_dataset(name, data=(factor * np.eye(nmu, dtype=complex)).reshape(
                1, 1, 1, 1, 1, 1, nmu, nmu))
            d.attrs[ready] = True
        f["psi_full_y"] = np.arange(1, 2*nmu+1, dtype=complex).reshape(1, 2, 1, nmu)
        f["enk_full"] = np.array([[-1., 1.]])
        f["G0_mu_nu"] = np.ones(nmu, dtype=complex)
        f["kgrid"] = np.ones(3, dtype=int)
        f["band_window"] = np.array([0, 0, 1, 2, 2])
        f["vhead"] = 0.
        f["whead"] = np.array([0.], dtype=complex)
    return path


@pytest.mark.parametrize("locations", [("tmp", "tmp"), (".", "."), ("tmp", ".")])
def test_ambiguity_refuses_independent_of_mtime(resolve, tmp_path, locations):
    paths = [_bundle(tmp_path / directory / f"isdf_tensors_{nmu}.h5", nmu)
             for directory, nmu in zip(locations, (2, 3))]
    assert paths[0].read_bytes() != paths[1].read_bytes()
    messages = []
    for times in [(1000, 2000), (2000, 1000)]:
        for path, stamp in zip(paths, times):
            os.utime(path, (stamp, stamp))
        with pytest.raises(ValueError) as exc:
            resolve(str(tmp_path / "cohsex.in"))
        messages.append(str(exc.value))
    assert messages[0] == messages[1]
    for path in paths:
        assert str(path.resolve()) in messages[0]
    assert "GATE bse_restart_ambiguous" in messages[0]
    assert "leave exactly one canonical bundle" in messages[0]
    assert "Σ/eqp" in messages[0] and "tmp/" in messages[0]


@pytest.mark.parametrize("directory", [".", "tmp"])
def test_single_candidate_is_unchanged(resolve, tmp_path, monkeypatch, directory):
    path = _bundle(tmp_path / directory / "isdf_tensors_2.h5", 2)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    assert resolve("cohsex.in") == str(path.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_missing_candidate_still_refuses(resolve, tmp_path):
    with pytest.raises(FileNotFoundError, match="canonical restart file"):
        resolve(str(tmp_path / "cohsex.in"))


@pytest.mark.parametrize("nmu", [2, 3])
def test_single_candidate_loads_the_same_tensor_values(tmp_path, nmu):
    """Both distinguishable fixtures are loadable; discovery changes no data."""
    import jax

    path = _bundle(tmp_path / f"isdf_tensors_{nmu}.h5", nmu)
    resolved = bse_loading._find_restart_file(str(tmp_path / "cohsex.in"))
    # Host cell: the fixture validates file/discovery semantics, not sharding.
    with jax.default_device(jax.devices("cpu")[0]):
        expected = bse_loading._load_ring_subset(str(path), 1, 1, 1, 1, n_occ=1)
        actual = bse_loading._load_ring_subset(resolved, 1, 1, 1, 1, n_occ=1)
    assert actual["n_rmu_pad"] == nmu
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    np.testing.assert_array_equal(actual["V_q0"], 2 * np.eye(nmu))
