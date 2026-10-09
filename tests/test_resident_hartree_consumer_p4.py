"""Physical-source and FILE-band serving oracle for the resident Hartree consumer.

The standalone P4 proof uses real AgI symmetry/header/source provenance and a
deliberately synthetic complex native band matrix. It tests the public serving
seam, not the numerical accuracy of that synthetic matrix as a Hartree field.
"""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    from runtime import initialize_communicator_stack
    RUNTIME = initialize_communicator_stack(
        platform="cpu" if "--cpu" in sys.argv else "gpu")

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _forbidden(*args, **kwargs):
    raise AssertionError("consumer reread a WFN coefficient or recomputed its fingerprint")


def _config(*, augmented=True):
    from gw.gw_config import BispinorGWMode, QPSolver
    return SimpleNamespace(bispinor=True,
        bispinor_gw=BispinorGWMode.COULOMB_ONLY,
        qp_solver=QPSolver.ONE_SHOT_DFT,
        paths=SimpleNamespace(atomic_reconstruction_dir="bound-source" if augmented else None))


def test_omit_and_default_hartree_paths_keep_their_existing_contract(monkeypatch):
    """Omission performs no source reads; the ordinary live route is unchanged."""
    from gw import augmentation_hartree as receiving
    from gw import sigma_dispatch as dispatch
    from gw.gw_config import ComputeMode

    sig_x = jnp.zeros((1, 2, 2), jnp.complex128)
    slices = SimpleNamespace(b0=0, b3=2)
    monkeypatch.setattr(receiving, "serve_resident_hartree", _forbidden)
    monkeypatch.setattr(dispatch, "_compute_live_hartree", _forbidden)
    omitted = dispatch._sigma_hartree_fields(slices, _config(), None, None,
        None, True, lambda *a: None, sig_x, None, None,
        resident_hartree=object())
    assert omitted[0].shape == () and omitted[0].item() == 0
    assert omitted[1].shape == () and omitted[1].item() == 0
    assert omitted[2] is None
    result = dispatch._static_sigma_result(None, ComputeMode.X_ONLY, True,
        None, None, sig_x, omitted[0], sig_x, sig_x, None, omitted[1])
    assert result.hartree_omitted and result.h_transverse_kij_ry is None

    live = jnp.asarray([[[2., .4+.7j], [.4-.7j, 3.]]], jnp.complex128)
    calls = []
    def original_live(config, meta, band_slices, mesh, **kwargs):
        calls.append((config, meta, band_slices, mesh, kwargs))
        return live, None
    monkeypatch.setattr(dispatch, "_compute_live_hartree", original_live)
    ordinary = dispatch._sigma_hartree_fields(slices, _config(augmented=False),
        None, None, None, False, lambda *a: None, sig_x, None, None)
    assert len(calls) == 1
    np.testing.assert_array_equal(ordinary[0], live)
    np.testing.assert_array_equal(ordinary[1], live)
    assert ordinary[2] is None
    with pytest.raises(ValueError, match="resident_hartree_required"):
        dispatch._sigma_hartree_fields(slices, _config(), None, None,
            None, False, lambda *a: None, sig_x, None, None)


def run_actual_consumer_oracle(runtime, *, fixture_root):
    from common.collectives import device_put_process_local
    from common.mtxel_sweep import blocks_to_host
    from common import parallel_transport
    from file_io.tagged_arrays import normalize_resident_hartree_provenance
    from gw import augmentation_hartree as receiving
    from gw import isdf_augmentation as stage
    from gw import sigma_dispatch as dispatch
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.gw_config import ComputeMode, QPSolver
    from gw.ppm_sigma import sigma_band_axis
    from symmetry_maps import star_tables_of
    from wfn_loader import WfnLoader

    if runtime.process_count not in (1, 4):
        raise ValueError("This bounded consumer oracle requires G0 or P4")
    root = Path(fixture_root)
    dev = root / "runs/DEV/780_augmented_isdf_20261006"
    canon = dev / "canonical_ae_large_target_v1"
    source_path = dev / "resident_exact_hartree_receiving_v2/resident_receipt.json"
    if _sha(source_path) != "5260bfba170fbf4b2ed92c0fe805652f6ed08e8d3dd87090c4c329f6cd2ef0a6":
        raise ValueError("The sealed real-source metadata fixture changed")
    source_receipt = json.loads(source_path.read_text())
    artifact = stage.read_augmentation_manifest(canon / "manifest_ae_large_ready")
    wfn_path = root / "runs/AgI/01_zb_fr_I4d_pbe80_k6_20261005/qe/nscf/WFN.h5"
    rng = np.random.default_rng(740138)
    errors, negatives = {}, []
    mesh = runtime.mesh
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    # Explicit tiny diagnostics boundary; never infer that a distributed
    # global array is addressable on the current process.
    def host(value):
        return blocks_to_host(value, nb=int(value.shape[-1]))
    lo, hi = 30, 33
    axis = sigma_band_axis(hi-lo, mesh, ansatz="static")

    with WfnLoader(str(wfn_path), mesh=mesh, backend="eager") as wfn:
        sym = wfn.symmetry()
        raw_coords = np.asarray(wfn.kvecs(k=sym.parent_k_domain))
        plan = build_centroid_k_unfold_plan(sym, np.asarray([[0, 0, 0]], np.int32),
            tuple(map(int, wfn.fft_grid)), mesh, nspinor=4,
            parent_k_frac=raw_coords, coordinate_kind="fft_indices")
        assert plan.n_parent == 16 and plan.n_full == 216
        # This is the normal one-time orchestration binding. All subsequent
        # serving/rotation calls below must use it without coefficient IO.
        fingerprint = parallel_transport.bind_wfn_fingerprint(wfn)
        assert parallel_transport.fingerprint_from_binding(fingerprint, wfn) == (
            source_receipt["source_binding"]["prepared_raw_binding"]["wfn_fingerprint"])
        source = copy.deepcopy(source_receipt["source_binding"])
        operator = receiving._operator_binding(wfn=wfn, sym=sym, artifact=artifact,
            source_identity=source_receipt["source_identity"], band_range=(lo, hi))
        provenance = normalize_resident_hartree_provenance(dict(
            schema="lorrax.resident_charge_hartree.v1",
            source_identity=source_receipt["source_identity"], source_binding=source,
            operator_identity=receiving._identity(operator), operator_binding=operator,
            band_range=[lo, hi], parent_full_rows=operator["parent_full_rows"],
            parent_k_frac=operator["parent_k_frac"], units="Ry",
            k_domain="file_wedge", trs_rule="conj"))
        z = rng.normal(size=(16, 3, 3)) + 1j*rng.normal(size=(16, 3, 3))
        native = np.asarray(z + z.conj().swapaxes(-2, -1), np.complex128)
        clean = np.zeros((16, axis.carrier, axis.carrier), np.complex128)
        clean[:, :3, :3] = native
        poisoned = np.full_like(clean, np.nan+1j*np.nan)
        poisoned[:, :3, :3] = native
        def bind(data):
            payload = device_put_process_local(data, sharding)
            return receiving.bind_resident_hartree(
                dict(parent_kij_ry=payload, provenance=provenance), wfn=wfn,
                sym=sym, plan=plan, artifact=artifact,
                wfn_fingerprint_binding=fingerprint, band_range=(lo, hi))
        record, poison_record = bind(clean), bind(poisoned)
        context = record["source_context"]
        assert set(context) == {"artifact", "plan", "wfn_fingerprint_binding"}
        assert set(context["artifact"]) == {"identity", "radial", "angular", "raw_parent_binding"}
        assert not any(key in context for key in ("state", "parent_psi", "caches", "inverse_sqrt"))
        irr, sym_idx, nspatial = star_tables_of(sym)
        expected = native[np.asarray(irr)]
        tr_rows = np.asarray(sym_idx) >= int(nspatial)
        assert np.any(tr_rows) and np.any(~tr_rows)
        expected = np.where(tr_rows[:, None, None], expected.conj(), expected)
        # A real complex offdiagonal makes omitted time reversal a meaningful red.
        wrong_no_tr = float(np.max(abs(native[np.asarray(irr)]-expected)))
        assert wrong_no_tr > .1

        with pytest.MonkeyPatch.context() as patch:
            for name in ("load", "load_process_local", "bands"):
                patch.setattr(wfn, name, _forbidden)
            patch.setattr(parallel_transport, "wfn_fingerprint", _forbidden)
            patch.setattr(parallel_transport, "_fingerprint_wfn_file", _forbidden)
            config = _config()
            def serve(value, receiving_range=(lo, hi), **kwargs):
                return receiving.serve_resident_hartree(value, config=config,
                    wfn=wfn, sym=sym, mesh=mesh, band_range=receiving_range, **kwargs)
            served = serve(record)
            poisoned_served = serve(poison_record)
            actual = host(served)
            np.testing.assert_array_equal(actual, host(poisoned_served))
            np.testing.assert_allclose(actual[:, :3, :3], expected, rtol=0, atol=2e-14)
            assert served.sharding == sharding and served.shape == (216, axis.carrier, axis.carrier)
            assert np.all(actual[:, 3:, :] == 0) and np.all(actual[:, :, 3:] == 0)
            errors["FILE_TR_max_Ry"] = float(abs(actual[:, :3, :3]-expected).max())
            sub = host(serve(record, (lo+1, hi)))
            np.testing.assert_allclose(sub[:, :2, :2], expected[:, 1:, 1:], rtol=0, atol=2e-14)

            # Complex non-diagonal U exercises both the conjugate bra and the
            # sole existing QP rotation. Its physical extent is deliberately odd.
            u0, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
            U = np.broadcast_to(u0, (216, 3, 3)).copy()
            rotated_expected = np.einsum("kai,kab,kbj->kij", U.conj(), expected, U)
            twice = np.einsum("kai,kab,kbj->kij", U.conj(), rotated_expected, U)
            assert np.max(abs(twice-rotated_expected)) > .1
            sig_x = device_put_process_local(np.zeros_like(actual), sharding)
            slices = SimpleNamespace(b0=lo, b3=hi)
            def fields(rotation):
                return dispatch._sigma_hartree_fields(slices, config, rotation, mesh,
                    None, False, lambda *a: None, sig_x, sym, wfn,
                    resident_hartree=poison_record)
            for name, rotation in (("host_U3", U),):
                result_fields = fields(rotation)
                result_matrix = host(result_fields[0])
                np.testing.assert_allclose(result_matrix[:, :3, :3], rotated_expected, rtol=0, atol=3e-13)
                assert np.all(result_matrix[:, 3:, :] == 0) and np.all(result_matrix[:, :, 3:] == 0)
                errors[name+"_max_Ry"] = float(abs(result_matrix[:, :3, :3]-rotated_expected).max())
            Ucarrier = np.full((216, axis.carrier, axis.carrier), 1e300+1e300j, np.complex128)
            Ucarrier[:, :3, :3] = U
            result_fields = fields(device_put_process_local(Ucarrier, sharding))
            device_rotated = host(result_fields[0])
            np.testing.assert_allclose(device_rotated[:, :3, :3], rotated_expected, rtol=0, atol=3e-13)
            errors["device_U_poison_max_Ry"] = float(np.max(abs(
                device_rotated[:, :3, :3]-rotated_expected)))
            result = dispatch._static_sigma_result(None, ComputeMode.X_ONLY, False,
                None, None, sig_x, result_fields[0], sig_x, sig_x, None, result_fields[1])
            assert result.h_transverse_kij_ry is None and not result.hartree_omitted
            np.testing.assert_array_equal(host(result.v_h_kij_ry), host(result.v_h_scalar_kij_ry))
            np.testing.assert_array_equal(host(result.sigma_x_kij_ry), host(sig_x))

            physical = stage._hartree_source_request(
                dict(occupations=None, full_kweights=None, spin_degeneracy=1.),
                wfn=wfn, plan=plan, public_range=source["public_band_range"])
            occ = physical["occupations"][plan.irr_idx]
            # Wider zero occupation ghosts are legal; populated extras are not.
            np.testing.assert_array_equal(host(serve(record, occupation_state=SimpleNamespace(
                f_kn=np.pad(occ, ((0, 0), (0, 2)))))), actual)
            bad_occ = occ.copy(); bad_occ[0, 0] = .5
            with pytest.raises(ValueError, match="occupations changed"):
                serve(record, occupation_state=SimpleNamespace(f_kn=bad_occ))
            negatives.append("changed_one_shot_occupations")
            old_solver = config.qp_solver
            config.qp_solver = QPSolver.SELF_CONSISTENT
            with pytest.raises(ValueError, match="fixed_source"):
                serve(record)
            config.qp_solver = old_solver
            negatives.append("density_self_consistency")
            stop = int(wfn.physical_density_band_stop)
            changed_file_occupations = physical["occupations"][:, :stop].copy()
            changed_file_occupations[0, 0] = .5
            def changed_density(*, k, unit_as_none=True):
                return (changed_file_occupations[np.asarray(plan.irr_idx)]
                    if k == "full_bz" else changed_file_occupations)
            with pytest.MonkeyPatch.context() as density_patch:
                density_patch.setattr(wfn, "physical_density_occupations", changed_density)
                with pytest.raises(ValueError, match="physical occupations"):
                    serve(record)
            negatives.append("changed_physical_WFN_density")
            changed_artifact = dict(context["artifact"])
            changed_artifact["radial"] = dict(changed_artifact["radial"])
            changed_artifact["radial"]["radius"] = np.asarray(
                changed_artifact["radial"]["radius"]).copy()
            changed_artifact["radial"]["radius"][1] += 1e-9
            changed_geometry_record = dict(record, source_context=dict(
                context, artifact=changed_artifact))
            with pytest.raises(ValueError, match="resident_hartree_operator"):
                serve(changed_geometry_record)
            negatives.append("changed_radial_operator")
            for name, mutate in (
                ("physical_occupations", lambda s: s.__setitem__("occupations_sha256", "0"*64)),
                ("physical_quadrature", lambda s: s.__setitem__("full_kweights_sha256", "0"*64)),
                ("wrong_frame_policy", lambda s: s.__setitem__("source_frame_policy", "raw_WFN")),
                ("wrong_frame_extent", lambda s: s.__setitem__("physical_frame_shape", [16, 120, 120])),
                ("malformed_factor_digest", lambda s: s.__setitem__("full150_frame_sha256", "not-a-factor")),
                ("wrong_factor_convention", lambda s: s.__setitem__("physical_frame_convention", "inverse_on_rows")),
                ("different_raw_binding", lambda s: s["prepared_raw_binding"].__setitem__("band_range", [0, 149])),
            ):
                bad = copy.deepcopy(provenance)
                mutate(bad["source_binding"])
                # Recompute JSON digests: this red must exercise the physical
                # comparison, not merely the elementary checksum validator.
                bad["source_identity"] = receiving._identity(bad["source_binding"])
                bad["operator_binding"]["source_identity"] = bad["source_identity"]
                bad["operator_identity"] = receiving._identity(bad["operator_binding"])
                with pytest.raises(ValueError, match="resident_hartree"):
                    receiving.bind_resident_hartree(dict(parent_kij_ry=record["parent_kij_ry"],
                        provenance=bad), wfn=wfn, sym=sym, plan=plan, artifact=artifact,
                        wfn_fingerprint_binding=fingerprint, band_range=(lo, hi))
                negatives.append(name)
            bad = copy.deepcopy(provenance)
            bad["parent_k_frac"][3][2] += 1.
            with pytest.raises(ValueError, match="resident_hartree_source"):
                receiving.bind_resident_hartree(dict(parent_kij_ry=record["parent_kij_ry"],
                    provenance=bad), wfn=wfn, sym=sym, plan=plan, artifact=artifact,
                    wfn_fingerprint_binding=fingerprint, band_range=(lo, hi))
            negatives.append("raw_parent_integer_wrap_is_not_FILE_row_identity")
            bad_native = clean.copy(); bad_native[0, 1, 2] = np.nan
            with pytest.raises(ValueError, match="nonfinite physical"):
                serve(bind(bad_native))
            negatives.append("nonfinite_active_matrix")

        # Portable omission/default checks share no actual source or IO path.
        with pytest.MonkeyPatch.context() as patch:
            test_omit_and_default_hartree_paths_keep_their_existing_contract(patch)
        return dict(schema="lorrax.test.resident_hartree_consumer.v1", status="PASS",
            P=runtime.process_count, mesh=dict(mesh.shape), actual_native_rows=16,
            actual_full_rows=216, actual_TR_rows=int(np.count_nonzero(tr_rows)),
            band_range=[lo, hi], physical_bands=3, carrier_bands=axis.carrier,
            errors_Ry=errors, wrong_omitted_TR_error_Ry=wrong_no_tr,
            poisoned_band_and_rotation_suffixes_inert=True,
            complex_QP_rotation_exactly_once=True, no_coefficient_reread_after_binding=True,
            defaults_unchanged=True, refusal_controls=negatives,
            fixture_source_receipt_sha256=_sha(source_path),
            actual_augmentation_identity=artifact["identity"],
            actual_source_identity=source_receipt["source_identity"],
            producer_sha256=_sha(__file__),
            owners_sha256={name: _sha(module.__file__) for name, module in (
                ("gw.augmentation_hartree", receiving), ("gw.sigma_dispatch", dispatch))},
            scope="Synthetic complex band operator in actual AgI FILE/source metadata; public serving, padding and rotation only. Not physical Hartree accuracy or SC admission.")


if __name__ == "__main__":
    import argparse
    from runtime import run_main_and_finalize
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    def main():
        if args.receipt.exists():
            raise FileExistsError("Preserve the completed consumer proof")
        receipt = run_actual_consumer_oracle(RUNTIME, fixture_root=args.fixture_root)
        if RUNTIME.process_index == 0:
            args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
            print(json.dumps(receipt), flush=True)
        return 0
    run_main_and_finalize(main)
