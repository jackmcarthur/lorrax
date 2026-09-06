"""Run production entry points in one real P4 runtime, without pytest collection."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


STARTED = time.perf_counter()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True,
                        help="new shared run directory; existing paths refuse")
    parser.add_argument("--budget-seconds", type=float, default=120.0,
                        help="whole-suite budget, including runtime startup")
    args = parser.parse_args()
    if args.budget_seconds <= 0:
        parser.error("--budget-seconds must be positive")

    from runtime import initialize_communicator_stack
    runtime = initialize_communicator_stack(platform="gpu")
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from runtime import run_main_and_finalize
    from tests.mini.checks import require_p4, check_results, EQP_ATOL_EV

    def suite():
        require_p4(runtime)
        output = args.output.resolve()
        # Every rank checks before the single writer creates the directory.
        assert not output.exists(), f"mini refuses to overwrite {output}"
        mh.sync_global_devices("mini-output-checked")
        if runtime.process_index == 0:
            output.mkdir(parents=True)
        mh.sync_global_devices("mini-output-created")
        receipt = {"schema": "lorrax-mini-v1", "status": "running",
                   "processes": 4, "mesh": [2, 2], "budget_seconds": args.budget_seconds,
                   "source_commit": subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                   "mini_sources_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                            for name in ("__main__.py", "routes.py", "checks.py")},
                   "persistent_compile_cache": os.environ.get("ISDF_JAX_CACHE_DIR"),
                   "jobid": os.environ.get("SLURM_JOB_ID"),
                   "stepid": os.environ.get("SLURM_STEP_ID"), "checks": []}

        def save():
            receipt["elapsed_seconds"] = time.perf_counter() - STARTED
            if runtime.process_index == 0:
                (output / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")

        def check(name, fn):
            mh.sync_global_devices("mini-start-" + name)
            started = time.perf_counter()
            if runtime.process_index == 0:
                print(f"MINI RUN {name}", flush=True)
            with (output / f"{name}.rank{runtime.process_index}.log").open("w") as log:
                try:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        result = fn()
                        jax.effects_barrier()
                except BaseException:
                    traceback.print_exc(file=log)
                    log.flush()
                    receipt["status"] = "failed"
                    receipt["failed_check"] = name
                    save()
                    raise
            mh.sync_global_devices("mini-end-" + name)
            elapsed = float(np.max(mh.process_allgather(
                np.asarray(time.perf_counter() - started))))
            receipt["checks"].append({"name": name, "seconds": elapsed, "result": result})
            save()
            if runtime.process_index == 0:
                print(f"MINI PASS {name} {elapsed:.3f}s", flush=True)
            return result

        def stage(source, name):
            target = output / name
            if runtime.process_index == 0:
                shutil.copytree(source, target)
                from tests.harness import make_writable
                make_writable(target)
            mh.sync_global_devices("mini-stage-" + name)
            return target

        def invoke(fn, argv, directory):
            previous = sys.argv
            try:
                sys.argv = [fn.__module__, *argv]
                with contextlib.chdir(directory):
                    result = fn()
                assert result in (None, 0), result
                # Workers may return while rank zero is still publishing its
                # small text outputs. Join before any rank reads those files.
                mh.sync_global_devices("mini-published-" + directory.name + fn.__module__)
            finally:
                sys.argv = previous

        source = HERE / "fixtures" / "H2-screw"
        h2 = stage(source, "hydrogen")

        def authenticate():
            from wfn_loader import WfnLoader
            provenance = json.loads((source / "PROVENANCE.json").read_text())
            for name, digest in provenance["files"].items():
                assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
            with WfnLoader(h2 / "WFN.h5", backend="eager",
                           qe_schema=h2 / "data-file-schema.xml") as wfn:
                sym = wfn.symmetry()
                assert tuple(wfn.kgrid) == (5, 5, 1)
                assert wfn.nbands == 9 and wfn.nspinor == 2
                assert sym.trs_allowed is False
                translations = np.asarray(wfn.translations)[:wfn.ntran] / (2 * np.pi)
                rotations = np.asarray(wfn.sym_matrices)[:wfn.ntran]
                assert any(np.allclose(rotation, np.diag([-1, 1, -1]))
                           and np.allclose(translation, [0, 0.5, 0])
                           for rotation, translation in zip(rotations, translations)), (
                               "fixture lost its two-fold screw along y")
                from common.band_degeneracy import boundary_min_gaps, DEGENERACY_TOL_RY
                gaps = boundary_min_gaps(np.asarray(wfn.energies)[0], is_full_spectrum=True)
                for edge in (1, 3, 7):
                    assert np.isfinite(gaps[edge]) and gaps[edge] > DEGENERACY_TOL_RY
                facts = {"stored_k": int(wfn.nkpts), "full_k": 25,
                         "stored_bands": 9, "spinor": 2,
                         "operations": int(wfn.ntran), "trs": False,
                         "translations": translations.tolist()}
            scf = json.loads((source / "scf_receipt.json").read_text())
            assert scf["indirect_gap_ev"] > 0.01
            assert np.linalg.norm(scf["total_magnetization_bohr_magneton"]) > 0.9
            return facts

        check("fixture", authenticate)

        def select(density, orbit, count):
            from centroid.kmeans_cli import main as kmeans
            suffix = "_current" if density == "current" else ("" if orbit else "_literal")
            oversample = "1.5" if orbit else "1.0"
            invoke(kmeans, [str(count), "--seed", "42", "--force-shard",
                           "--orbit" if orbit else "--no-orbit", "--oversample", oversample,
                           "--fit-window", "0:1,0:7", "--density-mode", density,
                           "--out-suffix", suffix], h2)
            paths = list(h2.glob(f"centroids_frac_*{suffix}.txt"))
            if not suffix:
                paths = [p for p in paths if "current" not in p.name]
            assert len(paths) == 1, paths
            points = np.loadtxt(paths[0])
            assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
            assert len(np.unique(points, axis=0)) == len(points)
            assert len(points) % 4 != 0, "fixture stopped exercising centroid padding"
            if orbit:
                from symmetry_maps import verify_centroid_orbit_closure
                from wfn_loader import WfnLoader
                with WfnLoader(h2 / "WFN.h5", backend="eager") as wfn:
                    verdict = verify_centroid_orbit_closure(
                        points, wfn.sym_matrices[:wfn.ntran],
                        tnp=wfn.translations[:wfn.ntran], fft_grid=wfn.fft_grid, tol=1.1e-5)
                assert verdict.closed
            else:
                assert len(points) == count
            return {"centroids": len(points), "file": paths[0].name,
                    "density": density, "orbit": orbit, "oversample": float(oversample)}

        charge = check("charge_centroids", lambda: select("scalar", True, 6))["file"]
        check("current_centroids", lambda: select("current", True, 6))
        check("literal_centroids", lambda: select("scalar", False, 19))
        base_deck = """[cohsex]
restart = false
centroids_file = {centroids}
nval = 1
ncond = 2
number_bands = 7
sys_dim = 3
compute_mode = cohsex
qp_solver = one_shot_dft
linalg = {linalg}
low_mem_bands = {low_mem}
bispinor = false
head_correction = off
mc_average_vcoul_body = false
write_restart_tensors = true
fermi_reference = midgap
"""
        if runtime.process_index == 0:
            (h2 / "cohsex.in").write_text(base_deck.format(
                centroids=charge, linalg="local", low_mem="false"))
        mh.sync_global_devices("mini-deck-written")

        def preprocess():
            from gw.kin_ion_io import main as kin_ion
            invoke(kin_ion, ["-i", "cohsex.in"], h2)
            assert (h2 / "kin_ion.h5").is_file()
            return {"entry": "gw.kin_ion_io.main"}

        check("kin_ion", preprocess)
        arms = []
        for layout, low in (("local", False), ("distributed", True),
                            ("local", True), ("distributed", False)):
            name = f"gw_{layout}_{'low' if low else 'high'}"
            target = stage(h2, name)
            if runtime.process_index == 0:
                (target / "cohsex.in").write_text(base_deck.format(
                    centroids=charge, linalg=layout, low_mem=str(low).lower()))
            mh.sync_global_devices("mini-deck-" + name)

            def gw():
                from gw.gw_jax import main as gwjax
                from tests.harness import eqp_column, parse_eqp_rows
                from tests.mini.routes import observe_routes
                with observe_routes(low_mem_bands=low) as routes:
                    invoke(gwjax, ["-i", "cohsex.in"], target)
                assert len(routes["zeta_fits"]) == 1, routes
                assert routes["dyson"] and {e["route"] for e in routes["dyson"]} == {layout}, routes
                if layout == "distributed":
                    assert routes["zeta_fits"][0]["distributed_zeta_solve"] == "distributed"
                    assert any(e["backend"] == "cusolvermp" for e in routes["algebra"]), routes
                eqp = eqp_column(target / "eqp1.dat")
                sigma = parse_eqp_rows(target / "sigma_diag.dat")
                check_results(eqp, sigma, reference=(
                    eqp_column(source / "reference_eqp1.dat"),
                    parse_eqp_rows(source / "reference_sigma_diag.dat")))
                np.testing.assert_allclose(eqp_column(target / "eqp0.dat"),
                                           eqp_column(source / "reference_eqp0.dat"),
                                           rtol=0, atol=EQP_ATOL_EV)
                check_results(eqp, sigma, reference=arms[0] if arms else None)
                arms.append((eqp, sigma))
                report = (target / "gwjax.out").read_text()
                assert f"linalg = {layout} (deck)" in report
                assert f"low_mem_bands = {str(low).lower()} (deck)" in report
                return {"linalg": layout, "low_mem_bands": low, "restart": False,
                        "max_eqp_delta_ev": float(np.max(np.abs(eqp - arms[0][0]))),
                        "executed_routes": routes}

            check(name, gw)

        def mpa():
            import h5py
            from unittest.mock import patch
            from tests.core.fixtures.stamp_references import check as check_stamp
            from tests.harness import copy_fixture, eqp_column
            from gw.gw_jax import main as gwjax
            source_b = REPO / "tests" / "core" / "fixtures" / "B"
            check_stamp("B")
            target_b = output / "mpa"
            if runtime.process_index == 0:
                copy_fixture(source_b, target_b)
                # The shared copier excludes ordinary GW output names, but
                # this fixture uses prefixed names. Remove those references
                # so missing publication cannot pass on stale output bytes.
                for name in ("mpa_eqp0.dat", "mpa_eqp1.dat", "mpa_sigma.dat",
                             "mpa_sigma.h5", "mpa.out"):
                    (target_b / name).unlink(missing_ok=True)
                shutil.copytree(source_b / "tmp" / "sigma_quadrature_rules",
                                target_b / "tmp" / "sigma_quadrature_rules")
            mh.sync_global_devices("mini-mpa-staged")
            # This is the same announced development permission as the core
            # MPA cell; frozen quadrature rules are reused, never W or zeta.
            with patch.dict(os.environ, {"LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE": "1"}):
                invoke(gwjax, ["-i", "mpa.in"], target_b)
            for name in ("mpa_eqp0.dat", "mpa_eqp1.dat"):
                got = eqp_column(target_b / name)
                assert got.shape == (3,) and np.isfinite(got).all()
                # The core MPA reference's 0.2-meV approximation budget.
                np.testing.assert_allclose(got, eqp_column(source_b / name),
                                           rtol=0, atol=2e-4)
            with h5py.File(target_b / "mpa_sigma.h5") as h5:
                sigma = np.asarray(h5["sigma_total_kij_ev"])
                assert sigma.shape == (25, 1, 3, 3) and np.isfinite(sigma).all()
            return {"compute_mode": "mpa", "poles": 2, "centroids": 13,
                    "bands": 7, "sigma_shape": list(sigma.shape), "restart": False,
                    "quadrature_rules": "committed core B rules"}

        check("mpa", mpa)

        # These are the service's existing numerical bodies, not another LA oracle.
        def algebra():
            from services.distrib_la.tests import test_distrib_la_multiproc as la
            from tests.core.distrib_la_p4 import CORE_CELLS
            cells = [row for row in la._CLI_CELLS if row[0] in CORE_CELLS]
            assert {row[0] for row in cells} == CORE_CELLS
            results = {}
            for name, _, fn in cells:
                results[name] = str(fn(runtime.mesh, "complex128"))
            return results

        check("dense_algebra", algebra)
        receipt["status"] = "passed"
        save()
        if receipt["elapsed_seconds"] > args.budget_seconds:
            receipt["status"] = "budget_exceeded"
            save()
            raise AssertionError(f"mini completed but exceeded {args.budget_seconds}s: "
                                 f"{receipt['elapsed_seconds']:.3f}s; see {output}/summary.json")
        if runtime.process_index == 0:
            print(f"MINI PASSED {len(receipt['checks'])} checks in "
                  f"{receipt['elapsed_seconds']:.3f}s; {output}/summary.json", flush=True)
        return 0

    return run_main_and_finalize(suite)


if __name__ == "__main__":
    main()
