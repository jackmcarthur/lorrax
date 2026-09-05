"""Tiny A-system htransform, TDA BSE, and finite-Q reference checks."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

import harness


RYD_TO_EV = 13.6056980659


def _run_module(run_dir, module, argv, *, timeout=120):
    env = os.environ.copy()
    env.setdefault("JAX_ENABLE_X64", "1")
    env.setdefault("JAX_PLATFORMS", "cuda,cpu")
    env.setdefault("JAX_PLATFORM_NAME", "gpu")
    env.setdefault("PYTHONUNBUFFERED", "1")
    cache = run_dir / ".jax-cache"
    cache.mkdir(exist_ok=True)
    env["ISDF_JAX_CACHE_DIR"] = str(cache)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    result = subprocess.run(
        [sys.executable, "-u", "-m", module, *argv], cwd=run_dir,
        env=env, capture_output=True, text=True, timeout=timeout, check=False,
    )
    assert result.returncode == 0, (
        f"{module} failed\n--- stdout ---\n{result.stdout[-6000:]}\n"
        f"--- stderr ---\n{result.stderr[-6000:]}"
    )
    return result


def _stage(source, target):
    shutil.copytree(source, target)
    harness.make_writable(target)
    return target


def _rows(path, *, skip_columns):
    return np.asarray([
        [float(value) for value in line.split()[skip_columns:]]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ])


def test_fixture_a_htransform_reference_matches_coarse_grid(core_fixtures):
    source = core_fixtures / "A"
    ref = json.loads((source / "excited_state_ref.json").read_text())
    tol = ref["tolerances_ev"]
    # All three requested points lie on the coarse mesh.  Independently read
    # their source eigenvalues and apply the htransform writer's path-VBM
    # convention.  This makes the stored three-point reference independently
    # checkable without paying for a redundant driver launch in default core.
    with h5py.File(source / "WFN.h5", "r") as wfn:
        coarse = np.asarray(wfn["mf_header/kpoints/el"])[0, [0, 2, 3], :2]
    coarse_ev = coarse * RYD_TO_EV
    coarse_ev -= coarse_ev[:, 1].max()
    np.testing.assert_allclose(
        ref["htransform_path_ev"], coarse_ev, rtol=0.0,
        atol=tol["htransform"],
    )


@pytest.mark.gpu
def test_fixture_a_exciton_band_reference(core_fixtures, tmp_path):
    """Exercise htransform plus a tiny TDA solve at Gamma and one finite Q."""
    if not harness.gpu_available():
        pytest.skip("tiny exciton reference is the GPU core cell")
    source = core_fixtures / "A"
    run = _stage(source, tmp_path / "A-exciton")
    ref = json.loads((source / "excited_state_ref.json").read_text())
    _run_module(run, "bse.exciton_bands", [
        "-i", "exciton.in", "--n-val", "1", "--n-cond", "1",
        "--n-eig", "2", "--block-size", "1", "--max-iter", "9",
        "--vq-mode", "ongrid", "--q-per-segment", "1",
        "--band-degeneracy", "off", "--px", "1", "--py", "1",
        "--out-prefix", "core_exciton", "--report-file", "core_exciton.out",
    ])
    exciton = _rows(run / "core_exciton.dat", skip_columns=6)
    np.testing.assert_allclose(
        exciton, ref["exciton_bands_ev"], rtol=0.0,
        atol=ref["tolerances_ev"]["exciton_bands"],
    )


@pytest.mark.core_extended
@pytest.mark.gpu
def test_fixture_a_standalone_htransform_and_direct_tda_reference(
    core_fixtures, tmp_path,
):
    """Run the standalone views already covered by the fused exciton cell."""
    if not harness.gpu_available():
        pytest.skip("standalone excited-state drivers require a GPU")
    source = core_fixtures / "A"
    run = _stage(source, tmp_path / "A-excited-standalone")
    ref = json.loads((source / "excited_state_ref.json").read_text())
    tol = ref["tolerances_ev"]

    _run_module(run, "bandstructure.htransform", [
        "-i", "htransform.in", "--guard-bands", "1",
        "-o", "htransform.dat", "--report-file", "htransform.out",
        "--eigh-backend", "off",
    ])
    transformed = _rows(run / "htransform.dat", skip_columns=6).reshape(3, 2)
    np.testing.assert_allclose(
        transformed, ref["htransform_path_ev"], rtol=0.0,
        atol=tol["htransform"],
    )

    bse = _run_module(run, "bse.bse_jax", [
        "-i", "cohsex.in", "--bse", "--lanczos", "--tda",
        "--n-val", "1", "--n-cond", "1", "--n-occ", "2",
        "--band-degeneracy", "off", "--max-lanczos-iter", "9",
        "--n-eig", "2", "--block-size", "1", "--px", "1", "--py", "1",
        "--report-file", "core_bse.out",
    ])
    bse_ev = np.asarray([
        float(value) for value in re.findall(
            r"^\s*S\d+\s+([0-9.+-]+)\s*$", bse.stdout, re.MULTILINE
        )
    ])
    np.testing.assert_allclose(
        bse_ev, ref["bse_tda_eigenvalues_ev"], rtol=0.0,
        atol=tol["bse_tda"],
    )
