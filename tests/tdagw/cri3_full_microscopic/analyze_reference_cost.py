"""Read-only reference-cost report from literal production SC log receipts.

Standard library only: no runtime, JAX, device initialization or remote IO.
Stage rows are host wall-time evidence, not synchronized device-kernel costs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


STAGES = ("W response", "Sigma rule refit", "Sigma rule plan",
          "Sigma tau sweep", "Sigma exchange", "Sigma Hartree", "other", "wall")
MAP_ROW = re.compile(r"SC map (\d+) stages \(s\):\s*(.*)$")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def file_record(path):
    """Return local file identity; content hashes do not prove physical association."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def parse_deck(text):
    """Read literal INI-style key/value declarations, not production defaults."""
    values = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("["):
            continue
        match = re.fullmatch(r"([^=:]+)[=:](.*)", line)
        if match:
            key, value = match.group(1).strip().lower(), match.group(2).strip()
            if key in values:
                raise ValueError(f"ambiguous duplicate deck key: {key}")
            values[key] = value
    return values


def _finite(value, name, *, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _json_literal(text):
    def invalid(value):
        raise ValueError(f"non-JSON numeric constant {value}")
    return json.loads(text, parse_constant=invalid)


def _tau_summary(value, line, raw):
    if not isinstance(value, dict):
        raise ValueError(f"Sigma receipt at line {line} is not a JSON object")
    names = ("window_tau_pairs", "distinct_tau_count", "n_windows", "rules_built",
             "plan_seconds", "critical_fit_wall_seconds", "planning_process_count",
             "sc_fixed_iteration", "sc_plan_event", "sc_rule_mode",
             "sc_fixed_rebuilds_this_iteration")
    row = {key: value[key] for key in names if key in value}
    row.update(line=line, literal_json_sha256=hashlib.sha256(raw.encode()).hexdigest(),
               association="preceding receipt in this map-stage log interval")
    row["windows"] = [{key: window[key] for key in
        ("name", "node_count", "fit_seconds", "rule_source", "kind") if key in window}
        for branch in value.get("branches", []) for window in branch.get("windows", [])]
    # Preserve literal counts. Do not infer distinct nodes by summing windows.
    for key in ("window_tau_pairs", "distinct_tau_count", "n_windows", "rules_built"):
        if key in row and (isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0):
            raise ValueError(f"invalid literal {key} at line {line}")
    return row


def parse_log(text):
    """Parse individual map rows, ordered JSON receipts and literal status lines."""
    maps, pending, verdicts, sync, declarations = [], [], [], [], {}
    seen = set()
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line.startswith("Sigma quadrature receipt:"):
            literal = line.split(":", 1)[1].strip()
            pending.append(_tau_summary(_json_literal(literal), line_number, literal))
        elif line.startswith("Stage synchronization receipt:"):
            value = _json_literal(line.split(":", 1)[1].strip())
            if not isinstance(value, dict):
                raise ValueError("stage synchronization receipt must be a JSON object")
            sync.append(dict(line=line_number, receipt=value))
        elif line.startswith("{"):
            # Optional literal JSON diagnostics; prose is never treated as fencing proof.
            try:
                value = _json_literal(line)
            except (ValueError, json.JSONDecodeError):
                value = None
            if isinstance(value, dict) and "stage_sync" in value:
                sync.append(dict(line=line_number, receipt=value["stage_sync"]))
        if "SC verdict:" in line:
            verdicts.append(dict(line=line_number, text=line.split("SC verdict:", 1)[1].strip()))
        for key, pattern in (
            ("method", r"Method\s*:\s*(.*)"),
            ("mpi_ranks", r"MPI ranks\s*:\s*(\d+)"),
            ("accelerators", r"Accelerators\s*:\s*(\d+) GPU devices.*"),
            ("head", r"Long wavelength\s*:\s*(.*)"),
            ("dense_linalg", r"Dense linalg\s*:\s*(.*)"),
            ("full_k_points", r"Full BZ grid\s*:\s*(\d+) k points.*"),
            ("stored_k_points", r"Stored IBZ\s*:\s*(\d+) k points.*")):
            match = re.fullmatch(pattern, line)
            if match:
                declarations[key] = int(match.group(1)) if key in {
                    "mpi_ranks", "accelerators", "full_k_points", "stored_k_points"} else match.group(1)
        for key, label in (("chi_band_range", "chi0/W sum"),
                           ("sigma_band_range", "Sigma sum"), ("qp_band_range", "QP matrix")):
            match = re.fullmatch(re.escape(label)+r"\s*:\s*(\d+)-(\d+)", line)
            if match:
                declarations[key] = [int(match.group(1)), int(match.group(2))]
        match = MAP_ROW.search(line)
        if not match:
            continue
        index = int(match.group(1))
        if index in seen:
            raise ValueError(f"duplicate map {index}; split concatenated sessions before reporting")
        seen.add(index)
        stages = {}
        for part in match.group(2).split("|"):
            item = re.fullmatch(r"\s*(.+?)\s+(" + NUMBER + r")\s*", part)
            if not item or item.group(1) in stages:
                raise ValueError(f"malformed/duplicate stage at line {line_number}: {part}")
            stages[item.group(1)] = _finite(item.group(2), "stage seconds")
        if set(stages) != set(STAGES):
            raise ValueError(f"unexpected/incomplete SC stage labels at line {line_number}")
        maps.append(dict(map=index, line=line_number, stages_seconds=stages,
            phase="map0_cold_candidate" if index == 0 else "later_individual_changed_state_map",
            w_response_wall_seconds=stages["W response"],
            sigma_rule_setup_wall_seconds=stages["Sigma rule refit"]+stages["Sigma rule plan"],
            sigma_tau_stage_wall_seconds=stages["Sigma tau sweep"],
            other_wall_seconds=stages["Sigma exchange"]+stages["Sigma Hartree"]+stages["other"],
            stage_sum_minus_wall_seconds=sum(v for k, v in stages.items() if k != "wall")-stages["wall"],
            executable_only_seconds=None, device_kernel_seconds=None,
            tau_receipts=pending))
        pending = []
    if not maps:
        raise ValueError("no literal production 'SC map N stages (s)' rows found")
    if [m["map"] for m in maps] != sorted(m["map"] for m in maps):
        raise ValueError("map rows are not in chronological index order")
    status = "UNKNOWN"
    if verdicts:
        last = verdicts[-1]["text"]
        status = ("ONE_MAP_DIAGNOSTIC" if "ONE-MAP DIAGNOSTIC" in last else
                  "NOT_CONVERGED" if "NOT CONVERGED" in last or "STALLED" in last else
                  "CONVERGED_LOG_DECLARATION" if last.startswith("CONVERGED") else "UNKNOWN")
    return dict(maps=maps, unmatched_tau_receipts=pending, sc_status=status,
        sc_verdicts=verdicts, log_declarations=declarations,
        stage_sync=dict(status="LITERAL_RECEIPTS_PRESENT_NOT_INDEPENDENTLY_VERIFIED" if sync else "UNPROVEN",
                        receipts=sync, device_kernel_attribution="UNPROVEN"))


def analyze(log, *, deck=None, source_code_prefix=None, source_files=(),
            expected_log_sha256=None, gpu_count=None, active_step_seconds=None,
            reservation_seconds=None):
    """Return a provenance-bound report, without authenticating reference-state physics."""
    identity = file_record(log)
    if expected_log_sha256 is not None and identity["sha256"] != expected_log_sha256:
        raise ValueError("log SHA256 differs from supplied expected hash")
    result = parse_log(Path(log).read_text())
    values = parse_deck(Path(deck).read_text()) if deck is not None else {}
    count = lambda key: int(values[key]) if key in values else None
    total = count("number_bands")
    nchi = count("number_bands_chi") or total
    nsigma = count("number_bands_sigma") or total
    for declared, key in ((nchi, "chi_band_range"), (nsigma, "sigma_band_range")):
        band_range = result["log_declarations"].get(key)
        if declared is not None and band_range and band_range[0] == 1 and declared != band_range[1]:
            raise ValueError(f"deck band count conflicts with literal log {key}")
    ranges = result["log_declarations"]
    nchi = nchi if nchi is not None else (ranges["chi_band_range"][1] if
        ranges.get("chi_band_range", [None])[0] == 1 else None)
    nsigma = nsigma if nsigma is not None else (ranges["sigma_band_range"][1] if
        ranges.get("sigma_band_range", [None])[0] == 1 else None)
    result.update(schema="reference-cost-readonly-v1", status="COMPLETE_REPORT_NOT_MODEL_ADMISSION",
        provenance=dict(log=identity, deck=None if deck is None else file_record(deck),
            source_code_prefix=source_code_prefix,
            source_code_prefix_authenticated=False,
            additional_source_files=[file_record(path) for path in source_files]),
        deck_declarations=values,
        physical_scope=dict(number_bands_chi=nchi, number_bands_sigma=nsigma,
            number_bands_protected=count("number_bands_protected"),
            compute_mode=values.get("compute_mode"), sigma_w_model=values.get("sigma_w_model"),
            qp_solver=values.get("qp_solver"), head_correction=values.get("head_correction"),
            linalg=values.get("linalg"), use_band_extrapolation=values.get("use_band_extrapolation"),
            full_frequency_shared_pole_declared=(values.get("compute_mode") == "mpa" and
                values.get("sigma_w_model") == "shared_pole") if values else None,
            both_band_sums_at_least_800=(nchi >= 800 and nsigma >= 800) if nchi and nsigma else None,
            converged_reference_log_declaration=result["sc_status"] == "CONVERGED_LOG_DECLARATION",
            reference_state_authenticated=False, orbital_energy_occupation_gauge_join="NOT_CHECKED",
            spatial_rank_convergence="NOT_CHECKED", td_tail_adapter=False, td_dynamics_computed=False))
    result["totals_seconds"] = {key: sum(row["stages_seconds"][key] for row in result["maps"])
                                for key in STAGES}
    accounting = dict(gpu_count=gpu_count, gpu_count_source="explicit CLI/API assumption" if gpu_count else None,
        logged_accelerators=result["log_declarations"].get("accelerators"),
        active_step_seconds=active_step_seconds, reservation_seconds=reservation_seconds,
        active_step_gpu_hours=None, reservation_gpu_hours=None, sum_map_wall_gpu_hours=None,
        assumptions="Map walls are not whole-job wall or reservation; costs below are not additive.")
    if gpu_count is not None:
        if isinstance(gpu_count, bool) or int(gpu_count) != gpu_count or gpu_count < 1:
            raise ValueError("gpu_count must be a positive integer")
        logged = accounting["logged_accelerators"]
        if logged is not None and logged != gpu_count:
            raise ValueError("GPU-count assumption conflicts with the literal log architecture")
        accounting["sum_map_wall_gpu_hours"] = gpu_count*result["totals_seconds"]["wall"]/3600
    for key in ("active_step", "reservation"):
        seconds = accounting[f"{key}_seconds"]
        if seconds is not None:
            seconds = _finite(seconds, f"{key} seconds")
            accounting[f"{key}_seconds"] = seconds
            if gpu_count is None:
                raise ValueError("GPU accounting seconds require explicit gpu_count")
            accounting[f"{key}_gpu_hours"] = gpu_count*seconds/3600
    if active_step_seconds is not None and reservation_seconds is not None and reservation_seconds < active_step_seconds:
        raise ValueError("reservation duration cannot be shorter than the supplied active step")
    result["gpu_accounting"] = accounting
    result["interpretation"] = [
        "Each map is reported individually; later maps change physical state and are NOT fixed-input warm repeats.",
        "Stage wall includes dispatch/setup/compile/synchronization effects; fencing and executable-only time are unproven unless separately measured.",
        "Sigma tau stage includes W(tau) synthesis, Green/W FFTs, projection and frequency folding, not one isolated FFT or node.",
        "Only named stage rows are summed; rounded rows may not sum exactly. Initial setup/compile probes and whole-job overhead are not inferred.",
        "A convergence log plus file hashes does not authenticate matched reference orbitals, energies, occupations, head and restart."]
    return result


def write_exclusive_atomic(path, report):
    """Publish a complete JSON via atomic exclusive hard-link; never replace a report."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--deck")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-code-prefix", help="declared source revision label, not inferred from old logs")
    parser.add_argument("--source-file", action="append", default=[])
    parser.add_argument("--expected-log-sha256")
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--active-step-seconds", type=float)
    parser.add_argument("--reservation-seconds", type=float)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    report = analyze(args.log, deck=args.deck, source_code_prefix=args.source_code_prefix,
        source_files=args.source_file, expected_log_sha256=args.expected_log_sha256,
        gpu_count=args.gpu_count, active_step_seconds=args.active_step_seconds,
        reservation_seconds=args.reservation_seconds)
    write_exclusive_atomic(args.output, report)
    print(f"{args.output}: {len(report['maps'])} individual maps; stage_sync={report['stage_sync']['status']}")


if __name__ == "__main__":
    main()
