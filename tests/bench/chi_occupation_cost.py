"""Matched P4 timing of existing binary and occupation-weighted chi APIs.

Synthetic, deterministic centroid amplitudes are generated per local shard.
This measures chi only, not QE/ISDF/Dyson/Sigma or a production material.
All physics and quadrature evaluations use existing source implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

CASES = {
    "tiny_spinor": ((5, 5, 1), 7, 1, 2, 11),
    "medium_spinor": ((3, 3, 3), 79, 19, 2, 319),
    "large_spinor": ((3, 3, 3), 159, 39, 2, 799),
    "medium_scalar_trs": ((3, 3, 3), 79, 19, 1, 319),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--layout", choices=("legacy", "face"), required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--probe-band-window", action="store_true")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three timing repetitions are required")

    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack(platform="gpu")

    def run():
        import jax
        import jax.numpy as jnp
        from jax.experimental import multihost_utils as mh
        from jax.sharding import NamedSharding, PartitionSpec as P
        from common.collectives import resolve_mesh
        from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC
        from gw.wavefunction_bundle import (
            BandSlices, Wavefunctions, PSI_XN_SPEC, PSI_XR_SPEC,
            PSI_YR_SPEC, PSI_YN_SPEC,
        )
        from gw import w_isdf
        from gw.minimax_config import MinimaxConfig
        from gw.minimax_screening import build_static_quadrature, build_imag_quadrature
        from gw.mpa.evaluator import damped_line_rule
        from symmetry_maps import q_negation_index

        mesh = resolve_mesh()
        if (runtime.process_count != 4 or runtime.n_local_devices != 1
                or tuple(mesh.devices.shape) != (2, 2)):
            raise RuntimeError("requires four processes, one GPU each, mesh 2x2")
        out = args.output.resolve()
        if out.exists():
            raise FileExistsError(out)
        mh.sync_global_devices("chi-cost-output")
        if runtime.process_index == 0:
            out.mkdir(parents=True)
        mh.sync_global_devices("chi-cost-created")
        started = time.perf_counter()
        root = Path(__file__).resolve().parents[2]
        receipt = {
            "schema": "chi-occupation-cost-v1", "status": "running",
            "case": args.case, "layout": args.layout, "processes": 4,
            "jobid": os.environ.get("SLURM_JOB_ID"),
            "stepid": os.environ.get("SLURM_STEP_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "jax": jax.__version__, "devices": [str(x) for x in jax.devices()],
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("src/gw/w_isdf.py",
                                           "src/gw/greens_function_kernel.py",
                                           "src/gw/mpa/evaluator.py")},
            "persistent_cache": os.environ.get("ISDF_JAX_CACHE_DIR"),
            "scope": "synthetic chi API; no parent symmetry acceleration; no driver timing",
            "timings": {}, "comparisons": {},
        }

        def save():
            receipt["elapsed_seconds"] = time.perf_counter() - started
            if runtime.process_index == 0:
                (out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")

        def note(message):
            if runtime.process_index == 0:
                print("CHI COST " + message, flush=True)

        def max_rank(dt):
            return float(np.max(mh.process_allgather(np.asarray(dt, dtype=np.float64))))

        def call_timed(fn):
            mh.sync_global_devices("chi-cost-before-call")
            tick = time.perf_counter()
            value = fn()
            jax.block_until_ready(value)
            elapsed = time.perf_counter() - tick
            return value, max_rank(elapsed)

        grid, nb, nv, ns, nmu = CASES[args.case]
        nk = int(np.prod(grid))
        nb_pad, mu_pad = 4 * ((nb + 3) // 4), 4 * ((nmu + 3) // 4)
        trs = args.case.endswith("_trs")
        kcoords = np.asarray(np.unravel_index(np.arange(nk), grid)).T
        ksignal = np.cos(2 * np.pi * kcoords / np.asarray(grid)).sum(axis=1)
        energies = np.tile(np.r_[np.linspace(-1.3, -0.15, nv),
                                        np.linspace(0.15, 3.3, nb - nv),
                                        np.full(nb_pad - nb, 3.3)], (nk, 1))
        energies += 0.01 * ksignal[:, None]
        occ = np.zeros((nk, nb_pad))
        occ[:, :nv] = 1.0
        partial_occ = occ.copy()
        partial_occ[:, nv - 1] = 0.8
        partial_occ[:, nv] = 0.2
        canonical_shape = (nk, nb_pad, ns, mu_pad)

        def put_small(value):
            return jax.make_array_from_callback(
                value.shape, NamedSharding(mesh, P(None, None)),
                lambda index: value[index])

        def amplitudes(order, spec):
            shape = tuple(canonical_shape[i] for i in order)

            def local(index):
                coords = [None] * 4
                for axis, role in enumerate(order):
                    coords[role] = np.arange(shape[axis])[index[axis]]
                k, n, s, mu = np.ix_(*coords)
                # The scalar control is real and k-negation even; the spinor
                # cells are complex and deliberately have no assumed TRS.
                phase = (0.173 * (n + 1) * (mu + 1)
                         + 0.319 * (s + 1) * (mu + 2)
                         + 0.119 * ksignal[k] * (n + 1))
                value = np.cos(phase).astype(np.complex128)
                if not trs:
                    value += 1j * np.sin(
                        0.137 * (n + 2) * (mu + 3)
                        + 0.217 * (k + 1) * (s + 1))
                value *= ((n < nb) & (mu < nmu)) / np.sqrt(nmu * ns)
                return np.ascontiguousarray(value.transpose(order))

            return jax.make_array_from_callback(shape, NamedSharding(mesh, spec), local)

        slices = BandSlices.from_band_edges(
            0, 0, nv, nb_pad, nb_pad, b4_logical=nb)
        fields = {}
        if args.layout == "legacy":
            fields = {
                "psi_xn": amplitudes((0, 2, 3, 1), PSI_XN_SPEC),
                "psi_xr": amplitudes((0, 1, 2, 3), PSI_XR_SPEC),
                "psi_yr": amplitudes((0, 1, 2, 3), PSI_YR_SPEC),
                "psi_yn": amplitudes((0, 2, 3, 1), PSI_YN_SPEC),
            }
        else:
            fields = {
                "psi_mun": amplitudes((0, 2, 3, 1), PSI_MUN_SPEC),
                "psi_nmu": amplitudes((0, 1, 2, 3), PSI_NMU_SPEC),
            }
        wfns = Wavefunctions(enk=put_small(energies), occ=put_small(occ),
                             slices=slices, layout=args.layout, **fields)
        partial = put_small(partial_occ)
        jax.block_until_ready((wfns, partial))
        meta = SimpleNamespace(nkx=grid[0], nky=grid[1], nkz=grid[2],
                               nk_tot=nk, n_rmu=mu_pad)
        qneg = q_negation_index(grid)
        config = MinimaxConfig(target_error=1e-6)
        tick = time.perf_counter()
        quad, eref = build_static_quadrature(wfns, config, print_fn=note)
        quad_imag = build_imag_quadrature(
            quad, 0.5, config, print_fn=note, with_odd_kernel=not trs)
        planning = max_rank(time.perf_counter() - tick)
        z = np.asarray([0.4 + 0.5j, 0.8 + 0.5j])
        bandwidth = float(np.max(energies) - np.min(energies))
        rule = damped_line_rule(0.5, bandwidth + 0.8, rel_tol=1e-6)
        ti, hi = rule["t"], rule["h"]
        # Both contour arms see exactly these nodes, weights and z values.
        def integer_dynamic(bundle=wfns):
            if not trs:
                return w_isdf.compute_chi0_contour_ordered(
                    bundle, ti, hi, z, meta, mesh, q_neg_index=qneg,
                    energy_reference=eref)
            tau = np.r_[1j * ti, -1j * ti]
            signs = np.r_[np.ones(ti.size), -np.ones(ti.size)]
            weights = np.broadcast_to(np.r_[1j * hi, -1j * hi], (z.size, 2 * ti.size))
            return w_isdf.compute_chi0_contour(
                bundle, tau, weights, signs, z, meta, mesh, energy_reference=eref)

        def fractional_dynamic(occupations, bundle=wfns):
            return w_isdf.compute_chi0_contour_fractional(
                bundle, ti, hi, z, meta, mesh, occupations=occupations,
                energy_reference=eref)

        imag_rule = damped_line_rule(0.5, bandwidth, rel_tol=1e-6)
        if runtime.process_index == 0:
            np.savez(out / "quadratures.npz", energies_ry=energies,
                     occupations=occ, partial_occupations=partial_occ,
                     static_tau=quad.tau, static_alpha=quad.alpha,
                     imag_tau=quad_imag.tau, imag_alpha=quad_imag.alpha,
                     imag_alpha_odd=(np.empty(0) if quad_imag.alpha_odd is None
                                     else quad_imag.alpha_odd),
                     dynamic_time=ti, dynamic_weights=hi, dynamic_z=z,
                     imaginary_time=imag_rule["t"], imaginary_weights=imag_rule["h"])
        def integer_imaginary():
            if trs:
                return w_isdf.compute_chi0(wfns, quad_imag, meta, mesh,
                                           energy_reference=eref)
            return w_isdf.compute_chi0_imag_ordered(
                wfns, quad_imag, meta, mesh, q_neg_index=qneg,
                energy_reference=eref)

        methods = {
            "integer_dynamic": integer_dynamic,
            "weighted_binary_dynamic": lambda: fractional_dynamic(wfns.occ),
            "weighted_partial_dynamic": lambda: fractional_dynamic(partial),
            "integer_imaginary": integer_imaginary,
            "weighted_binary_imaginary": lambda: w_isdf.compute_chi0_contour_fractional(
                wfns, imag_rule["t"], imag_rule["h"], np.asarray([0.5j]),
                meta, mesh, occupations=wfns.occ, energy_reference=eref),
            "integer_static": lambda: w_isdf.compute_chi0(
                wfns, quad, meta, mesh, energy_reference=eref),
        }
        if args.case == "tiny_spinor":
            methods["weighted_binary_static_gamma"] = lambda: (
                w_isdf.compute_chi0_static_fractional_gamma(
                    wfns, wfns.enk, wfns.occ, jnp.zeros_like(wfns.occ),
                    meta, mesh, nb_logical=nb))
            oracle_state = SimpleNamespace(
                f_kn=wfns.occ, mu_ry=0.0, smearing_family="mp1",
                smearing_width_ry=1e-6)
            methods["direct_dynamic_gamma"] = lambda: (
                w_isdf.compute_chi0_direct_fractional(
                    wfns, z, meta, mesh, occupation_state=oracle_state,
                    kminq_rows=np.arange(nk, dtype=np.int32)[None, :],
                    nb_logical=nb))
        receipt.update(
            grid=list(grid), bands_logical=nb, bands_padded=nb_pad, nval=nv,
            centroids_logical=nmu, centroids_padded=mu_pad, nspinor=ns,
            assumed_trs_control=trs, quadrature_planning_seconds=planning,
            nodes={"dynamic_weighted": int(ti.size),
                   "dynamic_integer": int(ti.size * (2 if trs else 1)),
                   "imaginary_weighted": int(imag_rule["t"].size),
                   "imaginary_integer": int(quad_imag.node_count),
                   "static_integer": int(quad.node_count)},
            energy_window_ry=[float(quad.x_min), float(quad.x_max)],
            relative_quadrature_target=1e-6,
        )
        values = {}
        for name, fn in methods.items():
            note("first call " + name)
            values[name], cold = call_timed(fn)
            receipt["timings"][name] = {"first_call_seconds": cold, "seconds": []}
            save()

        def compare(label, a, b, limit):
            pairs = zip(a, b) if isinstance(a, tuple) else [(a, b)]
            error, scale = 0.0, 0.0
            for left, right in pairs:
                error = max(error, float(jnp.max(jnp.abs(left - right))))
                scale = max(scale, float(jnp.max(jnp.abs(right))))
            relative = error / max(scale, 1e-300)
            receipt["comparisons"][label] = {
                "max_abs": error, "max_rel": relative, "limit": limit,
                "passed": bool(np.isfinite(relative) and relative < limit)}
            save()
            note(f"{label} relative error {relative:.3e}")

        compare("dynamic_binary_parity", values["integer_dynamic"],
                values["weighted_binary_dynamic"], 1e-10)
        compare("imaginary_binary_parity", values["integer_imaginary"],
                values["weighted_binary_imaginary"], 2e-5)
        compare("dynamic_qneg_transpose_diagnostic", values["integer_dynamic"],
                tuple(jnp.swapaxes(jnp.take(v, qneg, axis=0), -1, -2)
                      for v in values["weighted_binary_dynamic"]), 1e-10)
        if "weighted_binary_static_gamma" in values:
            compare("static_gamma_binary_parity",
                    values["integer_static"][:1],
                    values["weighted_binary_static_gamma"], 2e-5)
            reference = tuple(values["direct_dynamic_gamma"][i] for i in range(z.size))
            for arm in ("integer_dynamic", "weighted_binary_dynamic"):
                compare(arm + "_vs_direct_gamma",
                        tuple(v[:1] for v in values[arm]), reference, 2e-5)

        if args.probe_band_window:
            from dataclasses import replace
            chi_stop = nv + max(1, (nb - nv) // 2)
            narrowed = replace(wfns, slices=BandSlices.from_band_edges(
                0, 0, nv, nb_pad, nb_pad, b4_chi=chi_stop,
                b4_sigma=nb_pad, b4_logical=nb))
            narrow_integer = jax.block_until_ready(integer_dynamic(narrowed))
            narrow_weighted = jax.block_until_ready(
                fractional_dynamic(wfns.occ, narrowed))
            compare("integer_changed_chi_window", narrow_integer,
                    values["integer_dynamic"], 1e-12)
            compare("weighted_changed_chi_window", narrow_weighted,
                    values["weighted_binary_dynamic"], 1e-12)
            receipt["band_window_probe"] = {
                "full_chi_top": nb_pad, "narrow_chi_top": chi_stop,
                "sigma_top_unchanged": nb_pad,
                "note": "comparison fields measure change; zero means no response to the window"}

        for warmup in range(2):
            for fn in methods.values():
                jax.block_until_ready(fn())
        names = list(methods)
        for repeat in range(args.repeats):
            # Alternate order; report the slowest rank for each synchronized call.
            for name in (names if repeat % 2 == 0 else names[::-1]):
                _, seconds = call_timed(methods[name])
                receipt["timings"][name]["seconds"].append(seconds)
            save()
        for name, row in receipt["timings"].items():
            sample = np.asarray(row["seconds"])
            row.update(median_seconds=float(np.median(sample)),
                       min_seconds=float(np.min(sample)),
                       max_seconds=float(np.max(sample)))
            note(f"{name}: median {row['median_seconds']:.6f} s")
        receipt["binary_parity_passed"] = all(
            row["passed"] for name, row in receipt["comparisons"].items()
            if name in ("dynamic_binary_parity", "imaginary_binary_parity",
                        "static_gamma_binary_parity"))
        receipt["status"] = ("complete" if receipt["binary_parity_passed"]
                             else "complete_with_mismatch")
        save()
        return 0

    run_main_and_finalize(run)


if __name__ == "__main__":
    main()
