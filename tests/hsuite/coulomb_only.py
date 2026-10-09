"""P4 Coulomb-only driver evidence on the magnetic H2 fixture.

Run with ``python -m tests.hsuite.coulomb_only --out NEW_DIRECTORY`` through
``lx run --pool POOL -N 1 -G 4 -n 4``.  The existing chain owns fixture setup,
launches and failure-signature scanning.  This focused leg checks the full
small Sigma_X/Hartree matrices, charge-only artifacts, restart, and scalar
screening with a four-component charge carrier.  It is a software regression
fixture, not a converged physical benchmark.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tests.hsuite import chain, rank_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    run = out / "run"
    lead = rank_session._resolve_proc_id() == 0
    if lead:
        chain.stage_fixture(run)
    rank_session.exchange("coulomb fixture")
    env = chain._env(out / "jax-cache")
    walls, problems, captures = {}, [], {}

    def stage(name, module, argv):
        records, walls[name] = chain.run_stage(run, name, module, argv, env, 600)
        problems.extend(chain.stage_verdict(name, records))
        if problems:
            raise RuntimeError("\n".join(problems))

    _, km_module, km_args = chain.STAGES[0]
    stage("kmeans", km_module, km_args)
    if lead:
        centroids = chain._centroid_file(run)
        common = (chain._DECK_COMMON
                  .replace("bispinor = false", "bispinor = true")
                  .replace("kin_ion.h5", "kin_ion_bisp.h5")
                  .format(centroids=centroids))
        common += ("bispinor_gw = coulomb_only\nlinalg = local\n"
                   "no_degen_averaging = true\n"
                   "centroids_file_current = deliberately_missing_current.txt\n")
        for name, extra in (
            ("fresh", "restart = false\ncompute_mode = x_only\n"),
            ("restart", "restart = true\ncompute_mode = x_only\n"),
            ("screen", "restart = true\ncompute_mode = cohsex\n"),
            ("sc", "restart = true\ncompute_mode = x_only\n"
             "qp_solver = self_consistent\nsc_max_iter = 2\nsc_tol_ev = 3.0\n"),
        ):
            deck = common + extra + (
                f"sigma_diag_file = {name}_sigma.dat\neqp0_file = {name}_eqp0.dat\n"
                f"eqp1_file = {name}_eqp1.dat\nreport_file = {name}.out\n")
            (run / f"{name}.in").write_text(deck)
    rank_session.exchange("coulomb decks")
    stage("kin_ion", "gw.kin_ion_io", ["-i", "fresh.in", "-o", "kin_ion_bisp.h5"])
    stage("dipole", "psp.get_dipole_mtxels", ["-i", "fresh.in"])

    import gw.gw_jax as driver
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    original = driver.compute_sigma_xc
    import gw.sc_iteration as sc_driver
    original_hartree = sc_driver.rebuild_hartree_dft_basis
    sc_hartree_calls = [0]

    def sc_hartree(*args, **kwargs):
        result = original_hartree(*args, **kwargs)
        assert result.transverse_dft is None
        sc_hartree_calls[0] += 1
        return result

    sc_driver.rebuild_hartree_dft_basis = sc_hartree
    active = [None]

    def capture(*args, **kwargs):
        assert kwargs.get("wfns_transverse") is None
        assert kwargs.get("bispinor_v_q_path") is None
        result = original(*args, **kwargs)
        assert result.h_transverse_kij_ry is None
        fields = {}
        for name, array in (("sigma_x", result.sigma_x_kij_ry),
                            ("hartree", result.v_h_kij_ry),
                            ("charge_hartree", result.v_h_scalar_kij_ry)):
            # Diagnostic gathers are bounded to this tiny fixture explicitly.
            assert np.prod(array.shape) < 100_000
            fields[name] = np.asarray(gather_to_host(array))
        captures[active[0]] = fields
        if lead:
            np.savez(out / f"{active[0]}_matrices_ry.npz", **fields)
        return result

    driver.compute_sigma_xc = capture
    try:
        for name in ("fresh", "restart", "screen", "sc"):
            active[0] = name
            stage(name, "gw.gw_jax", ["-i", f"{name}.in"])
    finally:
        driver.compute_sigma_xc = original
        sc_driver.rebuild_hartree_dft_basis = original_hartree
    assert sc_hartree_calls[0] >= 1

    differences_mev = {}
    for name in ("sigma_x", "hartree", "charge_hartree"):
        error = float(np.max(np.abs(captures["fresh"][name]
                                     - captures["restart"][name]))) * RYD_TO_EV * 1000
        differences_mev[name] = error
        assert error < 1e-6, (name, error)
    for fields in captures.values():
        assert np.all(np.isfinite(fields["sigma_x"]))
        assert np.max(np.abs(fields["hartree"] - fields["charge_hartree"])) == 0
        assert np.max(np.abs(fields["sigma_x"]
                             - fields["sigma_x"].swapaxes(-2, -1).conj())) < 1e-10

    refusal_gates = {}
    for variant, policy in (("missing_policy", None), ("wrong_policy", "bare_transverse")):
        bad_run = out / variant
        if lead:
            shutil.copytree(run, bad_run)
            import h5py
            bundle = next((bad_run / "tmp").glob("isdf_tensors_*.h5"))
            with h5py.File(bundle, "a") as stream:
                if policy is None:
                    del stream.attrs["bispinor_gw"]
                else:
                    stream.attrs["bispinor_gw"] = policy
        rank_session.exchange(variant)
        records, walls[variant] = chain.run_stage(
            bad_run, variant, "gw.gw_jax", ["-i", "restart.in"], env, 600)
        assert all(rec["rc"] != 0 and "restart_bispinor_interaction" in rec["gates"]
                   for rec in records), records
        refusal_gates[variant] = [rec["gates"] for rec in records]

    if lead:
        from file_io.restart_bundle import read_metadata
        bundles = list((run / "tmp").glob("isdf_tensors_*.h5"))
        assert len(bundles) == 1
        metadata = read_metadata(bundles[0])
        assert metadata["bispinor_gw"] == "coulomb_only"
        assert metadata["family_shapes"]["charge"][2] == 4
        assert "current" not in metadata["family_shapes"]
        for name in ("zeta_q_mu1.h5", "zeta_q_mu2.h5", "zeta_q_mu3.h5", "v_q_bispinor.h5"):
            assert not (run / "tmp" / name).exists(), name
        (out / "summary.json").write_text(json.dumps(dict(
            ranks=rank_session._resolve_proc_count(), walls_s=walls,
            fresh_restart_max_difference_mev=differences_mev,
            charge_representation=metadata["charge_representation"],
            bispinor_gw=metadata["bispinor_gw"],
            charge_face_shape=metadata["family_shapes"]["charge"],
            current_artifacts=False, sc_charge_hartree_rebuilds=sc_hartree_calls[0],
            expected_refusal_gates=refusal_gates, problems=problems), indent=2) + "\n")
    rank_session.exchange("coulomb evidence complete")
    if lead:
        print("Coulomb-only P4 evidence PASS", differences_mev, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
