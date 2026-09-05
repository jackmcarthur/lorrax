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
from core import rank_session


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
    return rank_session.run_child(lambda: subprocess.run(
        [sys.executable, "-u", "-m", module, *argv], cwd=run_dir,
        env=env, capture_output=True, text=True, timeout=timeout, check=False,
    ), run_dir / module)


def _stage(source, target):
    shutil.copytree(source, target)
    harness.make_writable(target)
    # _find_restart_file deliberately chooses the newest candidate when a
    # sweep directory holds several bases. Core keeps both the 21-centroid
    # GW bundle and the 31-centroid excited-state bundle, so remove the
    # inapplicable copy from this private staged directory.
    (target / "tmp" / "isdf_tensors_21.h5").unlink()
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
def test_fixture_a_exciton_band_reference(core_fixtures):
    """Exercise htransform plus a tiny TDA solve at Gamma and one finite Q."""
    if not harness.gpu_available():
        pytest.skip("tiny exciton reference is the GPU core cell")
    source = core_fixtures / "A"
    run = rank_session.stage(source, _stage)
    ref = json.loads((source / "excited_state_ref.json").read_text())
    # The 2x2 mesh pads the nine physical transitions to 36 states. Nine
    # Krylov steps do not converge the padded spectrum to the reference.
    _run_module(run, "bse.exciton_bands", [
        "-i", "exciton.in", "--n-val", "1", "--n-cond", "1",
        "--n-eig", "2", "--block-size", "1", "--max-iter", "36",
        "--vq-mode", "ongrid", "--q-per-segment", "1",
        "--band-degeneracy", "off", "--px", "2" if rank_session._resolve_proc_count() == 4 else "1",
        "--py", "2" if rank_session._resolve_proc_count() == 4 else "1",
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
    core_fixtures,
):
    """Run the standalone views already covered by the fused exciton cell."""
    if not harness.gpu_available():
        pytest.skip("standalone excited-state drivers require a GPU")
    source = core_fixtures / "A"
    run = rank_session.stage(source, _stage)
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

    # Davidson seeds physical transitions and converges individual states;
    # scalar Lanczos at the full padded dimension suffers exact breakdown.
    bse = _run_module(run, "bse.bse_jax", [
        "-i", "cohsex.in", "--bse", "--lanczos", "--tda",
        "--solver", "davidson",
        "--n-val", "1", "--n-cond", "1", "--n-occ", "2",
        "--band-degeneracy", "off", "--max-lanczos-iter", "36",
        "--n-eig", "2", "--block-size", "1",
        "--px", "2" if rank_session._resolve_proc_count() == 4 else "1",
        "--py", "2" if rank_session._resolve_proc_count() == 4 else "1",
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
