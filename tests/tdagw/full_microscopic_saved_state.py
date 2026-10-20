"""One saved-state full-microscopic GW refresh; no physical-time propagation.

Run on a four-rank compute step. The saved small active U/E/rho tables are
in bundle k order (the campaign snapshot observer's convention). This checks
the native response/complex Sigma route at a fixed supplied Hamiltonian; it
does not assert that the old saved Hamiltonian solves the new GW fixed point.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="full shared-pole GW source deck")
    parser.add_argument("--snapshot", required=True, help="saved U/E_ry/rho_in_basis NPZ")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--b-lo", type=int, default=128, help="zero-based active band start")
    parser.add_argument("--out", required=True, help="new diagnostic directory")
    parser.add_argument("--gamma-ev", type=float, default=0.05)
    parser.add_argument("--fit-tolerance", type=float, default=0.02)
    parser.add_argument("--support-sites-ev", default="", help="native GW recipe override")
    parser.add_argument("--refined-support-sites-ev", default="",
                        help="optional second native recipe for complex-Sigma comparison")
    args = parser.parse_args()
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from file_io.commit_state import agree_io_refusal
    from tdagw import dynsig, snapshot_runtime as sr

    root = Path(args.out).resolve()
    rank0 = jax.process_index() == 0
    say = (lambda message: print(message, flush=True)) if rank0 else (lambda _: None)
    if jax.process_count() != 4:
        raise ValueError("GATE full_microscopic_saved_state: this verification requires P=4")
    directory_error = None
    if rank0:
        try:
            if root.exists():
                raise FileExistsError(f"diagnostic output exists: {root}; choose a new directory")
            root.mkdir(parents=True)
        except OSError as error:
            directory_error = error
    agree_io_refusal(directory_error, path=root, stage="full_microscopic_saved_state/output")
    # All ranks read the identical small metadata; no microscopic face is staged.
    with np.load(args.snapshot) as saved:
        basis = {key: np.asarray(saved[name][args.trial]) for key, name in
                 (("U", "U"), ("e", "E_ry"), ("rho", "rho_in_basis"))}
        snapshot_time = float(saved["time_fs"])
    nk, na = basis["e"].shape
    if (basis["U"].shape != (nk, na, na) or basis["rho"].shape != (nk, na, na)
            or not all(np.isfinite(a).all() for a in basis.values())):
        raise ValueError("GATE full_microscopic_saved_state: invalid active snapshot tables")
    report = dict(status="running", physical_time_propagated_fs=0,
        source=str(Path(args.source).resolve()), snapshot=str(Path(args.snapshot).resolve()),
        saved_time_fs=snapshot_time, source_tree=os.environ.get("LEG_SOURCE_TREE"),
        archive_sha256=os.environ.get("LEG_SOURCE_ARCHIVE_SHA256"),
        ranks=jax.process_count(), devices=[str(d) for d in jax.devices()],
        active_bands=[args.b_lo, args.b_lo + na], walls_s={}, evaluations=[])
    start = time.perf_counter()

    def save():
        if rank0:
            (root / "receipt.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    def timed(name, function):
        report["phase"] = name
        save()
        before = time.perf_counter()
        try:
            return function()
        finally:
            report["walls_s"][name] = time.perf_counter() - before
            say(f"[full microscopic saved state] {name}: {report['walls_s'][name]:.3f} s")
            save()

    rc = 0
    try:
        source = timed("source", lambda: dynsig.gw_window_source(args.source, stack.mesh,
                       workdir=str(root / "driver"), screening=True, say=say))
        prepared = timed("prepare", lambda: sr.prepare_native_source(source, stack.mesh))
        if prepared["nk"] != nk or prepared["nb"] < args.b_lo + na:
            raise ValueError("GATE full_microscopic_saved_state: source/snapshot grid or bands differ")
        report["microscopic_space"] = dict(nk=nk, nmu=prepared["nmu_logical"],
                                          virtual_bands=prepared["nb"], spatial_projection=False)
        rho = basis["U"] @ basis["rho"] @ basis["U"].conj().swapaxes(-1, -2)
        results = []
        sites = [args.support_sites_ev]
        if args.refined_support_sites_ev:
            sites.append(args.refined_support_sites_ev)
        for index, support_sites in enumerate(sites):
            options = dict(b_lo=args.b_lo, n_active=na, screening="joint",
                statistics="retarded_reference", spm_scratch=str(root / f"models_{index}"),
                spm_support_sites=support_sites, adiabatic_ry=args.gamma_ev / RYD_TO_EV,
                eta_ry=float(source["config"].sigma.regularization_ev) / RYD_TO_EV,
                fit_tolerance=args.fit_tolerance, say=say)
            evaluate, _ = sr.make_native_snapshot_evaluator(source, prepared, options)
            result = timed(f"evaluate_{index}", lambda: evaluate(basis, fit_bases={}))
            results.append(result)
            gamma, injection = (np.asarray(result[key]) for key in ("gamma", "injection"))
            collision = injection - 0.5 * (gamma @ rho + rho @ gamma)
            outgoing = -np.trace(gamma @ rho, axis1=-2, axis2=-1).real.mean()
            incoming = np.trace(injection, axis1=-2, axis2=-1).real.mean()
            flux = 0.5 * (abs(outgoing) + abs(incoming))
            values = np.linalg.eigvalsh(basis["rho"])
            row = dict(index=index, support_sites_ev=support_sites or "source GW recipe",
                native_receipt=result.get("native_receipt"),
                screening_receipts=result.get("screening_receipts"),
                input_density_min=float(values.min()), input_density_max=float(values.max()),
                trace_collision_ry=float(np.trace(collision, axis1=-2, axis2=-1).real.mean()),
                trace_residual_over_flux=float(abs(incoming + outgoing) / flux) if flux else 0.,
                flux_ry=float(flux),
                offdiagonal_m_h_max_mev=float(np.max(abs(np.asarray(result["m_h"])
                    * (1. - np.eye(na))))) * RYD_TO_EV * 1e3,
                min_injection_mev=float(np.linalg.eigvalsh(injection).min()) * RYD_TO_EV * 1e3,
                min_blocking_mev=float(np.linalg.eigvalsh(gamma-injection).min()) * RYD_TO_EV * 1e3)
            report["evaluations"].append(row)
            small_host = lambda value: np.asarray(gather_to_host(value)) if isinstance(
                value, jax.Array) else np.asarray(value)
            arrays = {key: small_host(value) for key, value in result.items()
                      if isinstance(value, (np.ndarray, jax.Array))}
            raw = result.get("raw", {})
            for name in ("retarded_pair", "branch_pair", "active_branch_pair"):
                values = raw.get(name)
                if isinstance(values, dict):
                    arrays.update({f"{name}_{branch}": small_host(value)
                                   for branch, value in values.items()})
                elif values is not None:
                    arrays[name] = small_host(values)
            if rank0:
                np.savez_compressed(root / f"operators_{index}.npz", **arrays,
                                    rho_fixed=rho, C_raw_ry=collision)
            save()
        if len(results) == 2:
            report["refinement"] = {key: dict(
                max_abs_mev=float(np.max(abs(np.asarray(results[1][key]) - results[0][key])))
                            * RYD_TO_EV * 1e3,
                relative_frobenius=float(np.linalg.norm(np.asarray(results[1][key]) - results[0][key])
                    / max(np.linalg.norm(results[1][key]), np.finfo(float).tiny)))
                for key in ("m_h", "gamma", "injection")}
        report["status"] = "complete_fixed_snapshot_not_dynamics"
    except Exception:
        report["status"] = "refused"
        report["error"] = traceback.format_exc()
        say(report["error"])
        rc = 1
    finally:
        report["total_wall_s"] = time.perf_counter() - start
        report["active_gpu_hours"] = report["total_wall_s"] * 4 / 3600
        save()
    runtime.finalize_process(rc)


if __name__ == "__main__":
    main()
