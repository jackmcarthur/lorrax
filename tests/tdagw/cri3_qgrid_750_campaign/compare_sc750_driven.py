"""Plot completed SC750 A/B/C data; never infer damping from an overlay.

Read-only postprocessing. Requires actual completed legs and identical native
time/field samples. No interpolation, dynamics subtraction or Pauli repair.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

FS_PER_AU = .024188843265857
LABELS = ("A: static TD-aGW, no collision", "B: MGKBA, frozen W",
          "C: MGKBA, updated W")
ARMS = ("A_STATIC_TDAGW_NO_COLLISION", "B_FROZEN_W_MGKBA", "C_UPDATED_W_MGKBA")


def waveform_difference(reference, candidate):
    """Raw and scale-separated waveform errors, not a relaxation-rate fit."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite matching waveform arrays required")
    norm2 = float(np.vdot(a, a).real)
    if norm2 <= 0:
        raise ValueError("zero reference waveform; relative errors undefined")
    scale = float(np.vdot(a, b).real/norm2)
    return dict(relative_l2=float(np.linalg.norm(b-a)/np.sqrt(norm2)),
                fitted_real_scale=scale,
                shape_l2_after_scale=float(np.linalg.norm(b-scale*a)/np.sqrt(norm2)))


def exact_common(items):
    """Require exact producer sample axes and drive, not rounded matches."""
    if not items:
        raise ValueError("no actual trajectories")
    for item in items[1:]:
        for name in ("time_fs", "field", "weights", "kpts"):
            if not np.array_equal(items[0][name], item[name]):
                raise ValueError("exact matched-arm "+name+" mismatch; no interpolation")


def material_gaps(directory):
    """Optional actual refresh gaps; never invent full750 or static-arm values."""
    from common.units import RYD_TO_EV
    rows = []
    for path in sorted(Path(directory).glob("snapshot_*.json")):
        record = json.loads(path.read_text())
        gap = record.get("gap_and_mixing")
        if gap is None:
            continue
        rows.append(dict(time_fs=float(record["history"]["t_fs"]),
            fundamental_gap_ev=float(gap["fundamental_gap_ry"])*RYD_TO_EV,
            minimum_direct_gap_ev=float(gap["minimum_direct_gap_ry"])*RYD_TO_EV,
            scope=gap["scope"]))
    return rows


def read_run(directory, arm, amplitude, window, duration_fs, measure_as):
    """Read only checkpoint-committed small observables from a complete leg."""
    directory = Path(directory)
    done = json.loads((directory/"complete.json").read_text())
    if (done.get("completed") is not True or done.get("arm") != arm
            or done.get("amplitude_v_m") != amplitude):
        raise ValueError("missing/mismatched completed-arm receipt: "+str(directory))
    with h5py.File(directory/"tdgw.h5", "r") as f:
        cp = f["tdgw_data/checkpoint"]
        n, end = int(cp["it_next"][()]), float(cp["tchkp"][()])*FS_PER_AU
        if n <= 1 or not np.isclose(end, duration_fs, rtol=0., atol=1.e-6):
            raise ValueError("incomplete actual checkpoint: "+str(directory))
        header, td = f["tdgw_header/wfn_header"], f["tdgw_data/time_domain"]
        bands = tuple(int(header["bands"][key][()]) for key in ("start", "stop", "step"))
        if bands != (window[0]+1, window[1]+1, 1):
            raise ValueError("active-window header mismatch")
        result = dict(weights=np.asarray(header["w_k"]), kpts=np.asarray(header["kpts"]))
        times = None
        for name, key in (("polarization", "polarization"), ("Efield", "field"),
                          ("carrier_density", "carrier")):
            group = td[name]
            if n > len(group["times"]) or n > len(group["data"]):
                raise ValueError("checkpoint sample count exceeds actual datasets")
            current = np.asarray(group["times"][:n])*FS_PER_AU
            if times is not None and not np.array_equal(times, current):
                raise ValueError("observable time axes disagree")
            times = current
            value = np.asarray(group["data"][:n])
            if np.max(np.abs(value.imag)) > 1.e-9:
                raise ValueError("observable has resolved imaginary component")
            result[key] = value.real.reshape(n, -1)
        result["time_fs"] = times
    step_fs = measure_as/1000.
    expected_count = round(duration_fs/step_fs)
    if (n != expected_count or not np.allclose(times, np.arange(n)*step_fs, rtol=0., atol=1.e-7)
            or result["weights"].shape != (64,) or not np.isclose(result["weights"].sum(), 1.)
            or any(not np.isfinite(x).all() for x in result.values())
            or np.any(np.diff(times) <= 0) or times[-1] > end+1.e-7):
        raise ValueError("invalid actual full64 trajectory")
    if result["polarization"].shape[1] != 3 or result["field"].shape[1] != 3:
        raise ValueError("one three-component trial required")
    result["path"] = str(directory)
    return result


def calibration_inputs(root, tags=None):
    """Validate completed 10fs A legs, even while other amplitudes still run.

    Every committed calibration row must match its completion receipt, native
    checkpoint and plot NPZ. No missing campaign completion or selected target
    is manufactured. Different amplitudes share exact time/k/weight axes;
    proportional drives are checked with an explicit FP64 roundoff tolerance.
    """
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    plan = manifest["plan"]
    if plan["duration_fs"] != 10.:
        raise ValueError("calibration-only requires completed full10fs A legs")
    terminal = root/"complete.json"
    status = "CAMPAIGN_NOT_COMPLETED_COMPLETED_A_LEGS_ONLY"
    if terminal.exists():
        status = json.loads(terminal.read_text()).get("status")
        if status not in ("THREE_ARM_EXPLORATORY_COMPLETE",
                          "A_CALIBRATION_ONLY_COMPLETE_BC_NOT_ATTEMPTED"):
            raise ValueError("unrecognized campaign completion; no completion inferred")
    records = sorted(root.glob("calibration_[0-9][0-9].json"))
    if not records:
        raise ValueError("no committed completed calibration amplitudes")
    committed_tags = ["A_"+path.stem for path in records]
    if tags is not None:
        if (not tags or len(tags) != len(set(tags))
                or any(tag not in committed_tags for tag in tags)):
            raise ValueError("explicit calibration tags must be distinct committed actual legs")
        records = [path for path in records if "A_"+path.stem in tags]
    rows, items = [], []
    for path in records:
        row = json.loads(path.read_text())
        tag = "A_"+path.stem
        directory = root/tag
        complete = json.loads((directory/"complete.json").read_text())
        if (row != complete or row.get("tag") != tag or row.get("zero_gate") is not False
                or row.get("actual_active") != manifest["actual_active"]
                or row.get("response_policy") != manifest["response_policy"]
                or row.get("duration_fs") != 10.):
            raise ValueError("calibration receipt/window/response-policy join mismatch")
        amplitude = float(row["amplitude_v_m"])
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("actual positive calibration amplitude required")
        item = read_run(directory, ARMS[0], amplitude, manifest["actual_active"],
                        plan["duration_fs"], plan["measure_as"])
        with np.load(directory/"plot_data.npz", allow_pickle=False) as payload:
            mapping = dict(time_fs=item["time_fs"], polarization_au=item["polarization"],
                field_au=item["field"], conduction_electrons_per_cell=item["carrier"][:, 0],
                valence_holes_per_cell=item["carrier"][:, 1])
            for key, expected in mapping.items():
                value = np.asarray(payload[key])
                if (not np.isfinite(value).all() or np.max(abs(value.imag), initial=0.) > 1.e-9
                        or not np.array_equal(value.real, expected)):
                    raise ValueError("native H5/plot NPZ mismatch: "+key)
        if (row.get("completed_samples") != len(item["time_fs"])
                or not np.isclose(row.get("checkpoint_fs", np.nan), 10., rtol=0., atol=1.e-8)
                or row.get("carrier_peak") != float(item["carrier"][:, 0].max())
                or row.get("hole_peak") != float(item["carrier"][:, 1].max())):
            raise ValueError("calibration completion/native measurements mismatch")
        rows.append(row)
        items.append(item)
    amplitudes = [row["amplitude_v_m"] for row in rows]
    if amplitudes != sorted(set(amplitudes)):
        raise ValueError("distinct increasing calibration amplitudes required")
    comparisons = []
    reference = items[0]
    reference_amplitude = rows[0]["amplitude_v_m"]
    field_peak = float(abs(reference["field"]).max())
    if field_peak <= 0:
        raise ValueError("finite-amplitude calibration has zero applied field")
    for row, item in zip(rows, items):
        for key in ("time_fs", "weights", "kpts"):
            if not np.array_equal(reference[key], item[key]):
                raise ValueError("cross-amplitude exact "+key+" mismatch; no interpolation")
        scaled_field = item["field"]*(reference_amplitude/row["amplitude_v_m"])
        roundoff = 64*np.finfo(np.float64).eps
        if not np.allclose(reference["field"], scaled_field,
                           rtol=roundoff, atol=roundoff*field_peak):
            raise ValueError("cross-amplitude normalized applied field mismatch")
        comparison = dict(amplitude_v_m=row["amplitude_v_m"],
            normalized_drive_relative_max=float(abs(scaled_field-reference["field"]).max()/field_peak),
            normalized_drive_fp64_tolerance=roundoff,
            reference_amplitude_v_m=reference_amplitude)
        if np.linalg.norm(reference["polarization"]) > 0:
            comparison["P_per_amplitude_versus_reference"] = waveform_difference(
                reference["polarization"]/reference_amplitude,
                item["polarization"]/row["amplitude_v_m"])
        else:
            comparison["P_per_amplitude_versus_reference"] = "UNDEFINED_ZERO_REFERENCE_POLARIZATION"
        comparisons.append(comparison)
    selected_path = root/"calibration_selected.json"
    selected = None
    selection_status = "NOT_AVAILABLE_NOT_INFERRED"
    if selected_path.exists():
        selection = json.loads(selected_path.read_text())
        selected = selection["selected"]
        if selection.get("arm") != ARMS[0] or not isinstance(selected, list) or len(selected) != 2:
            raise ValueError("invalid actual calibration selection")
        by_tag = {row["tag"]: row for row in rows}
        if tags is not None and any(pick["calibration_tag"] not in by_tag for pick in selected):
            selected = None
            selection_status = "NOT_EVALUATED_EXPLICIT_SUBSET_EXCLUDES_SELECTED_LEGS"
        else:
            for pick in selected:
                row = by_tag.get(pick["calibration_tag"])
                if (row is None or pick["amplitude_v_m"] != row["amplitude_v_m"]
                        or pick["actual_A_carrier_peak"] != row["carrier_peak"]
                        or pick["target_carriers"] not in plan["carrier_targets"]):
                    raise ValueError("selected target does not join actual completed A leg")
            selection_status = "JOINED_ACTUAL_COMPLETED_LEGS"
    return manifest, rows, items, dict(
        scope="STATIC_A_CALIBRATION_ONLY_NO_BC_NO_DAMPING_INFERENCE",
        campaign_status=status, completed_calibration_amplitudes=len(rows),
        requested_calibration_tags=tags,
        unplotted_committed_tags=[tag for tag in committed_tags if tag not in {row["tag"] for row in rows}],
        selected=selected, selection_status=selection_status, amplitude_normalized_nonlinearity=comparisons,
        interpolation=False, zero_field_subtraction=False,
        qualification="Only complete native10fs static-A legs. No B/C result or damping conclusion.")


def plot_calibration(root, out, tags=None):
    """Plot actual static-A calibration observables without an MGKBA claim."""
    manifest, rows, items, report = calibration_inputs(root, tags=tags)
    plan = manifest["plan"]
    projection = np.asarray(plan["pulse"]["polarization"], float)
    if projection.shape != (3,) or not np.isfinite(projection).all() or not np.isclose(projection@projection, 1.):
        raise ValueError("actual literal unit pulse polarization required")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    reference = items[0]["polarization"]/rows[0]["amplitude_v_m"]
    report["amplitudes"] = []
    for row, item in zip(rows, items):
        t, amplitude = item["time_fs"], row["amplitude_v_m"]
        label = f"{amplitude:g} V/m; peak {row['carrier_peak']:.5g} e/cell"
        p, e = item["polarization"]@projection, item["field"]@projection
        axes[0, 0].plot(t, p, label=label)
        axes[0, 1].plot(t, item["carrier"][:, 0], label=label)
        axes[0, 1].plot(t, item["carrier"][:, 1], linestyle="--", alpha=.6)
        axes[0, 2].plot(t, e, label=label)
        axes[1, 0].plot(t, p/amplitude, label=label)
        axes[1, 1].plot(t, e/amplitude, label=label)
        axes[1, 2].plot(t, (item["polarization"]/amplitude-reference)@projection, label=label)
        report["amplitudes"].append(dict(amplitude_v_m=amplitude,
            actual_carrier_peak=row["carrier_peak"], actual_hole_peak=row["hole_peak"],
            actual_coverage_fs=[float(t[0]), float(t[-1])], checkpoint_fs=10.,
            path=item["path"]))
    labels = ("raw P along pulse (native au)", "electrons/cell; holes dashed",
              "applied E along pulse (native au)", "P / amplitude (au / (V/m))",
              "E / amplitude (au / (V/m))", "P/amplitude minus lowest plotted amplitude")
    for axis, label in zip(axes.flat, labels):
        axis.set(xlabel="time (fs)", ylabel=label)
        axis.axvline(plan["pulse"]["end_fs"], color=".7", linestyle=":")
        axis.legend(fontsize=7)
    fig.suptitle("Static-A calibration only — no B/C comparison or damping inference")
    fig.tight_layout()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    fig.savefig(out/"calibration_only.png", dpi=150)
    plt.close(fig)
    report.update(execution_revision=manifest["execution_revision"],
                  actual_active=manifest["actual_active"], response_policy=manifest["response_policy"])
    (out/"calibration_only.json").write_text(json.dumps(report, indent=2)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", help="completed sc750_driven_callback output")
    parser.add_argument("--out", required=True, help="new plot/report directory")
    parser.add_argument("--calibration-only", action="store_true",
                        help="only complete10fs static-A calibration legs; no B/C claim")
    parser.add_argument("--calibration-tags", nargs="+",
                        help="explicit subset of committed A_calibration_NN legs (calibration-only)")
    args = parser.parse_args(argv)
    root, out = Path(args.campaign).resolve(), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError("choose a NEW plot output directory")
    if args.calibration_only:
        return plot_calibration(root, out, tags=args.calibration_tags)
    if args.calibration_tags is not None:
        raise ValueError("calibration-tags requires explicit calibration-only mode")
    manifest = json.loads((root/"manifest.json").read_text())
    done = json.loads((root/"complete.json").read_text())
    plan = manifest["plan"]
    if done.get("status") != "THREE_ARM_EXPLORATORY_COMPLETE":
        raise ValueError("actual completed three-arm campaign required; calibration-only is not A/B/C")
    if out.exists():
        raise FileExistsError("choose a NEW plot output directory")
    selected = json.loads((root/"calibration_selected.json").read_text())["selected"]
    if len(selected) != 2:
        raise ValueError("two actually calibrated intensities required")
    data, comparisons = [], []
    for level, pick in enumerate(selected):
        amp = pick["amplitude_v_m"]
        tags = (pick["calibration_tag"], f"B_level_{level}", f"C_level_{level}")
        items = [read_run(root/tag, arm, amp, manifest["actual_active"], plan["duration_fs"], plan["measure_as"])
                 for tag, arm in zip(tags, ARMS)]
        exact_common(items)
        data.append(items)
        p = [item["polarization"] for item in items]
        comparisons.append(dict(amplitude_v_m=amp, target_carriers=pick["target_carriers"],
            actual_A_carrier_peak=pick["actual_A_carrier_peak"],
            C_versus_B=waveform_difference(p[1], p[2]), B_versus_A=waveform_difference(p[0], p[1]),
            C_minus_B_scope="W update changes real/coherent AND collision terms; not isolated damping",
            paths=[item["path"] for item in items]))
        mask = items[0]["time_fs"] >= plan["pulse"]["end_fs"]
        if not mask.any():
            raise ValueError("actual post-pulse samples required")
        comparisons[-1]["postpulse_rms_polarization_au"] = [
            float(np.sqrt(np.mean(np.sum(abs(item["polarization"][mask])**2, axis=1))))
            for item in items]
    for low, high in zip(*data):
        for name in ("time_fs", "weights", "kpts"):
            if not np.array_equal(low[name], high[name]):
                raise ValueError("cross-intensity "+name+" mismatch")
    nonlinear = [dict(arm=label, **waveform_difference(
        data[0][i]["polarization"]/selected[0]["amplitude_v_m"],
        data[1][i]["polarization"]/selected[1]["amplitude_v_m"]))
        for i, label in enumerate(LABELS)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    projection = np.asarray(plan["pulse"]["polarization"])
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), squeeze=False)
    for level, items in enumerate(data):
        t = items[0]["time_fs"]
        for label, item in zip(LABELS, items):
            axes[level, 0].plot(t, item["polarization"]@projection, label=label)
            axes[level, 1].plot(t, item["carrier"][:, 0], label=label)
        b, c = items[1]["polarization"], items[2]["polarization"]
        peak = np.linalg.norm(b, axis=1).max()
        if peak <= 0:
            raise ValueError("zero frozen-W polarization")
        scale = comparisons[level]["C_versus_B"]["fitted_real_scale"]
        axes[level, 2].plot(t, ((c-b)@projection)/peak, label="(C-B)/peak|B|")
        axes[level, 2].plot(t, ((c-scale*b)@projection)/peak, label="(C-scale×B)/peak|B|")
        gaps = [material_gaps(item["path"]) for item in items]
        comparisons[level]["actual_material_gap_records"] = gaps
        for label, rows in zip(LABELS, gaps):
            if rows:
                axes[level, 3].plot([row["time_fs"] for row in rows],
                    [row["fundamental_gap_ev"] for row in rows], ".-", label=label)
        if not any(gaps):
            axes[level, 3].text(.5, .5, "No material-H snapshots", transform=axes[level, 3].transAxes,
                                ha="center")
        for axis, ylabel in zip(axes[level], ("raw P along pulse (native au)", "carriers (e/cell)",
                                               "relative waveform", "active-window gap (eV; NOT full750)")):
            axis.set(xlabel="time (fs)", ylabel=ylabel)
            axis.axvline(plan["pulse"]["end_fs"], color=".7", linestyle=":")
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=7)
        axes[level, 0].set_title(f"E={selected[level]['amplitude_v_m']:g} V/m; target {selected[level]['target_carriers']:g} e/cell")
    fig.suptitle("SC750 exploratory dynamics — raw observables; no damping-rate claim")
    fig.tight_layout()
    # Output only after every input/comparison has been validated.
    out.mkdir(parents=True, exist_ok=False)
    fig.savefig(out/"polarization_carriers.png", dpi=150)
    plt.close(fig)
    report = dict(execution_revision=manifest["execution_revision"], active_window=manifest["actual_active"],
        comparisons=comparisons, amplitude_normalized_nonlinearity=nonlinear,
        interpolation=False, zero_field_subtraction=False,
        qualification="Exploratory discretization only. Waveform changes alone do not establish irreversible damping.")
    (out/"comparison.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
