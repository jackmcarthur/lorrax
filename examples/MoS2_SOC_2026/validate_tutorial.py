#!/usr/bin/env python3
"""Read-only checks of the real BGW/LORRAX SOC tutorial outputs.

Run with NumPy/h5py, optionally via SSH stdin. No source imports, writes or MPI.
JSON goes to stdout. Missing outputs are PENDING, format errors are FAIL, and
numerical differences between implementations are descriptive, never equality
gates. A file being written must not be treated as a completed receipt.
"""
import argparse
import itertools
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(path, columns):
    a = np.loadtxt(path, comments="#", ndmin=2)
    require(a.shape[0] > 0 and a.shape[1] == columns, f"expected {columns} columns")
    require(np.isfinite(a).all(), "nonfinite table entry")
    return a


def integral_ids(a):
    require(np.array_equal(a, a.astype(np.int64)), "noninteger spin/band id")
    require(np.all(a > 0), "nonpositive spin/band id")


def eqp(path):
    lines = [line.split() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    rows, blocks, cursor = [], [], 0
    while cursor < len(lines):
        header = np.asarray(lines[cursor], dtype=float)
        require(header.shape == (4,), "malformed k-block header")
        count = int(header[3])
        require(count > 0 and count == header[3], "invalid block count")
        body = np.asarray(lines[cursor + 1:cursor + 1 + count], dtype=float)
        require(body.shape == (count, 4), "incomplete/malformed eqp block")
        require(np.isfinite(header).all() and np.isfinite(body).all(), "nonfinite eqp")
        integral_ids(body[:, :2])
        ids = [(int(s), int(b)) for s, b in body[:, :2]]
        require(len(set(ids)) == count, "duplicate spin/band in k block")
        require(ids == sorted(ids), "spin/band order is not increasing")
        blocks.append({"k": header[:3].tolist(), "count": count})
        rows.extend((int(s), int(b), *header[:3], emf, qp) for s, b, emf, qp in body)
        cursor += count + 1
    a = np.asarray(rows)
    result = {"blocks": blocks, "k_count": len(blocks), "rows": len(rows),
              "bands": sorted(set(a[:, 1].astype(int).tolist()))}
    for col, label in ((5, "dft"), (6, "qp")):
        v, c = a[a[:, 1] == 26, col], a[a[:, 1] == 27, col]
        if len(v) and len(c):
            result[label + "_indirect_gap_ev"] = float(c.min() - v.max())
            result[label + "_minimum_direct_gap_ev"] = float((c - v).min())
    return result, a


def eqp_reader_layout(path):
    """BGW numeric reads may not encounter comments until all k blocks end."""
    details, _ = eqp(path)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    cursor = 0
    for block in details["blocks"]:
        require(not lines[cursor].lstrip().startswith("#"),
                "BGW numeric reader encounters a comment before completing k blocks")
        tokens = lines[cursor].split()
        require(len(tokens) == 4, "non-numeric BGW k header")
        np.asarray(tokens[:3], dtype=float)
        require(int(tokens[3]) == block["count"], "BGW block count mismatch")
        for line in lines[cursor+1:cursor+1+block["count"]]:
            require(not line.lstrip().startswith("#"), "comment within BGW band block")
        cursor += 1 + block["count"]
    require(all(line.lstrip().startswith("#") for line in lines[cursor:]),
            "non-metadata records after BGW k blocks")
    return {"numeric_k_blocks": len(details["blocks"]), "numeric_band_rows": details["rows"],
            "footer_lines": len(lines)-cursor, "leading_comments": False}, None


def bands(path):
    a = table(path, 8)
    integral_ids(a[:, :2])
    residual = float(np.max(np.abs(a[:, 7] - (a[:, 6] - a[:, 5]))))
    require(residual <= 3e-8, "Delta E differs from E(QP)-E(MF)")
    groups = {}
    for s, b in np.unique(a[:, :2].astype(int), axis=0):
        groups[(s, b)] = a[(a[:, 0] == s) & (a[:, 1] == b), 2:5]
    grid = next(iter(groups.values()))
    require(all(g.shape == grid.shape and np.array_equal(g, grid)
                for g in groups.values()), "bands do not share k-path order")
    band_major = [(int(s), int(b)) for s, b in a[:, :2]]
    result = {"shape": list(a.shape), "path_count": len(grid),
              "band_count": len(groups), "band_major_order": band_major == sorted(band_major),
              "delta_energy_residual_ev": residual}
    for col, label in ((5, "dft"), (6, "qp")):
        v, c = a[a[:, 1] == 26, col], a[a[:, 1] == 27, col]
        if len(v) and len(c):
            result[label + "_indirect_gap_ev"] = float(c.min() - v.max())
            result[label + "_minimum_direct_gap_ev"] = float((c - v).min())
    return result, a


def native_path(path):
    a = table(path, 7)
    require(np.array_equal(a[:, :2], a[:, :2].astype(int)), "noninteger k/band index")
    require(np.all(a[:, :2] >= 0), "negative k/band index")
    k_ids, band_ids = np.unique(a[:, 0]), np.unique(a[:, 1])
    require(np.array_equal(k_ids, np.arange(len(k_ids))), "noncontiguous path indices")
    require(np.array_equal(band_ids, np.arange(len(band_ids))), "noncontiguous band indices")
    require(len(a) == len(k_ids) * len(band_ids), "missing path/band pair")
    for k in k_ids:
        rows = a[a[:, 0] == k]
        require(np.array_equal(rows[:, 1], band_ids), "band order differs by k")
        require(np.all(rows[:, 2:6] == rows[0, 2:6]), "k/distance differs within one k")
    ref = re.search(r"^# energy_reference_eV=([^\n]+)", path.read_text(), re.M)
    return {"shape": list(a.shape), "path_count": len(k_ids), "band_count": len(band_ids),
            "all_finite": True, "numeric_energy_reference": "VBM relative",
            "absolute_energy_offset_ev": float(ref.group(1)) if ref else None}, a


def adapter_semantics(path):
    dft_path = path.with_name("bandstructure_dft.lorrax.dat")
    qp_path = path.with_name("bandstructure_qp.lorrax.dat")
    dd, dft = native_path(dft_path)
    qd, qp = native_path(qp_path)
    offsets = [dd["absolute_energy_offset_ev"], qd["absolute_energy_offset_ev"]]
    require(all(v is not None and np.isfinite(v) for v in offsets),
            "missing exact absolute-energy reference header")
    require(dft.shape == qp.shape and np.array_equal(dft[:, :6], qp[:, :6]),
            "DFT/QP raw path identities differ")
    match = re.search(r"absolute_band_window=\[(\d+),", dft_path.read_text())
    require(match is not None, "missing absolute band-window receipt")
    start = int(match[1])
    with h5py.File(path.with_name("WFN.h5"), "r") as h:
        reciprocal = h["mf_header/crystal/bvec"][()] * float(h["mf_header/crystal/blat"][()])
    drows = {(int(r[0]), int(r[1])): r for r in dft}
    qrows = {(int(r[0]), int(r[1])): r for r in qp}
    expected = []
    for b in sorted(set(dft[:, 1].astype(int))):
        for k in sorted(set(dft[:, 0].astype(int))):
            d, q = drows[k, b], qrows[k, b]
            edft, eqp_ = d[6] + offsets[0], q[6] + offsets[1]
            expected.append([1, b + start + 1, *(d[2:5] @ reciprocal), edft, eqp_, eqp_ - edft])
    expected = np.asarray(expected)
    actual = table(path, 8)
    require(actual.shape == expected.shape and np.array_equal(actual[:, :2], expected[:, :2]),
            "BGW spin/band/path order does not match native identities")
    k_error = float(np.max(abs(actual[:, 2:5] - expected[:, 2:5])))
    energy_error = float(np.max(abs(actual[:, 5:] - expected[:, 5:])))
    require(k_error <= 5.1e-6, "Cartesian k values disagree with kfrac@bvec*blat")
    require(energy_error <= 1.1e-8, "adapter does not restore exact DFT/QP energy references")
    return {"raw_shapes": [list(dft.shape), list(qp.shape)], "output_shape": list(actual.shape),
            "dft_reference_ev": offsets[0], "qp_reference_ev": offsets[1],
            "cartesian_k_max_error_inv_bohr": k_error,
            "restored_energy_max_error_ev": energy_error}, None


def eigenvalues(path):
    a = table(path, 4)
    text = path.read_text()
    declared = re.search(r"neig\s*=\s*(\d+)", text)
    require(declared is not None and int(declared.group(1)) == len(a), "neig/count mismatch")
    volume = re.search(r"vol\s*=\s*([0-9.eE+-]+)", text)
    spins = re.search(r"nspin,\s*nspinor\s*=\s*(\d+)\s+(\d+)", text)
    require(volume is not None and float(volume[1]) > 0 and spins is not None,
            "missing/invalid eigenvalue volume or spin header")
    require(np.all(np.diff(a[:, 0]) >= -1e-7), "eigenvalues are not ascending")
    strength_error = float(np.max(np.abs(a[:, 1] - a[:, 2] ** 2 - a[:, 3] ** 2)))
    scale = max(1., float(np.max(np.abs(a[:, 1]))))
    require(strength_error / scale < 3e-7, "dipole square differs from re/im columns")
    return {"roots": len(a), "energy_min_ev": float(a[0, 0]),
            "volume_supercell_bohr3": float(volume[1]), "nspin": int(spins[1]), "nspinor": int(spins[2]),
            "energy_max_ev": float(a[-1, 0]), "dipole_square_relative_error": strength_error / scale}, a


def absorption(path):
    a = table(path, 4)
    require(np.all(np.diff(a[:, 0]) > 0), "omega is not strictly increasing")
    peak = int(np.argmax(a[:, 1]))
    return {"shape": list(a.shape), "omega_range_ev": a[[0, -1], 0].tolist(),
            "eps2_min": float(a[:, 1].min()), "eps2_peak_ev": float(a[peak, 0]),
            "eps2_peak": float(a[peak, 1]),
            "eps2_integral": float(np.trapezoid(a[:, 1], a[:, 0]))}, a


def finite_hdf(ds):
    if ds.ndim == 0:
        return bool(np.isfinite(ds[()]).all())
    per_row = max(1, int(np.prod(ds.shape[1:])) * ds.dtype.itemsize)
    step = max(1, (16 * 2**20) // per_row)
    return all(np.isfinite(ds[i:i + step]).all() for i in range(0, len(ds), step))


def eps(path):
    with h5py.File(path, "r") as h:
        q = np.asarray(h["eps_header/qpoints/qpts"])
        nq = int(h["eps_header/qpoints/nq"][()])
        nm = np.asarray(h["eps_header/gspace/nmtx"])
        maxm = int(h["eps_header/gspace/nmtx_max"][()])
        nf = int(h["eps_header/freqs/nfreq"][()])
        nmat = int(h["eps_header/params/nmatrix"][()])
        ds = h["mats/matrix"]
        require(q.shape == (nq, 3) and nm.shape == (nq,), "q/nmtx shape mismatch")
        require(np.all(nm > 0) and int(nm.max()) == maxm, "nmtx/max mismatch")
        require(ds.shape == (nq, nmat, nf, maxm, maxm, 2), "epsilon matrix layout mismatch")
        require(int(h["eps_header/params/matrix_type"][()]) == 0, "matrix_type is not inverse epsilon")
        require(np.all(h["eps_header/qpoints/qpt_done"][()] == 1), "unfinished q block")
        require(finite_hdf(ds), "nonfinite epsilon matrix")
        ng = int(h["mf_header/gspace/ng"][()])
        gmap = h["eps_header/gspace/gind_eps2rho"]
        for i, n in enumerate(nm):
            ids = gmap[i, :int(n)]
            require(np.all((ids >= 1) & (ids <= ng)) and len(np.unique(ids)) == n,
                    "invalid/duplicate 1-based G indices")
        return {"matrix_shape": list(ds.shape), "q_count": nq, "qpoints": q.tolist(),
                "nmtx": nm.tolist(), "ecut_ry": float(h["eps_header/params/ecuts"][()]),
                "nband_header": int(h["eps_header/params/nband"][()]),
                "schema_version": int(h["eps_header/versionnumber"][()])}, None


def bsemat(path):
    with h5py.File(path, "r") as h:
        nk = int(h["bse_header/kpoints/nk"][()])
        ns = int(h["bse_header/bands/ns"][()])
        n1 = int(h["bse_header/bands/n1b"][()])
        n2 = int(h["bse_header/bands/n2b"][()])
        shape = (nk * ns, nk * ns, n2, n2, n1, n1, 2)
        require(h["bse_header/kpoints/kpts"].shape == (nk, 3), "kernel k-grid shape")
        for name in ("head", "wing", "body", "exchange"):
            ds = h["mats/" + name]
            require(ds.shape == shape, name + " matrix layout")
            require(finite_hdf(ds), name + " has nonfinite entries")
        return {"matrix_shape": list(shape), "k_count": nk, "ns": ns,
                "n_val": n1, "n_cond": n2,
                "schema_version": int(h["bse_header/versionnumber"][()])}, None


def sigma_coverage(path):
    log_path = path.with_name("gwjax.out")
    require(log_path.is_file(), "missing native GW coverage log")
    log = log_path.read_text()
    verdict = re.findall(r"\bcoverage\s+(COMPLETE|INCOMPLETE):[^\n]*", log)
    require(verdict and verdict[-1] == "COMPLETE", "reported on-shell coverage is not COMPLETE")
    require("sigma_grid_incomplete" not in log, "native GW reports out-of-grid fallback")
    with h5py.File(path, "r") as h:
        marker = h["sigma_evaluation"][()]
        marker = marker.decode() if isinstance(marker, bytes) else str(marker)
        require(marker == "on_shell", "Sigma is not evaluated on shell")
        ds = h["sigma_eval_rel_ev"]
        attrs = ds.attrs
        count = int(attrs["omega_uncovered_count"])
        fraction = float(attrs["omega_uncovered_fraction"])
        require(count == 0 and fraction == 0.,
                f"persisted fallback receipt has {count} uncovered states ({fraction:.6g})")
        require(attrs["sigma_eval_provenance"] == "at_e_dft", "unexpected evaluation spectrum")
        target, actual = ds[()], h["onshell_pair_rel_ev"][()]
        require(target.shape == actual.shape and target.ndim == 2, "on-shell table shape mismatch")
        require(np.isfinite(target).all() and np.isfinite(actual).all(), "nonfinite evaluation table")
        difference = float(np.max(np.abs(target-actual)))
        require(difference <= 1e-12, "actual pair evaluation energies differ from requested E_DFT")
        nk_full = int(attrs["nk_full"])
        require(nk_full == 9, "tutorial Sigma full-BZ count differs from9")
        nb = int(h["sigma_total_kij_ev"].attrs["band_logical_extent"])
        require(nb == 48 and target.shape[1] == nb, "tutorial Sigma band count differs from48")
        eqp_details, _ = eqp(path.with_name("eqp1.dat"))
        require(eqp_details["rows"] == 192 and eqp_details["k_count"] == 4,
                "tutorial does not expose all192 raw-file QP states")
        return {"reported_coverage": "COMPLETE", "persisted_uncovered_states": count,
                "persisted_uncovered_fraction": fraction, "canonical_table_shape": list(target.shape),
                "full_bz_evaluation_states": nk_full * nb, "raw_file_eqp_states": eqp_details["rows"],
                "pair_vs_requested_eval_max_difference_ev": difference}, None


def eps_q_coverage(path):
    """BGW fullbz uses the file's spatial matrices, without implicit TR."""
    with h5py.File(path, "r") as h:
        grid = h["eps_header/qpoints/qgrid"][()].astype(int)
        ntran = int(h["mf_header/symmetry/ntran"][()])
        ops = h["mf_header/symmetry/mtrx"][:ntran]
        qpoints = np.vstack([np.zeros((1, 3)), h["eps_header/qpoints/qpts"][()]])
        covered = set()
        for q in qpoints:
            for op in ops:
                scaled = (q @ op) * grid
                require(np.max(np.abs(scaled-np.rint(scaled))) < 2e-6,
                        "declared spatial action does not close on q grid")
                covered.add(tuple(map(int, np.rint(scaled).astype(int) % grid)))
        required = set(itertools.product(*(range(int(n)) for n in grid)))
        missing = sorted(required-covered)
        require(not missing, f"spatial-only q coverage {len(covered)}/{len(required)}; missing integer-grid rows {missing}")
        return {"grid": grid.tolist(), "spatial_symmetry_count": ntran,
                "covered_q_count": len(covered), "required_q_count": len(required),
                "implicit_time_reversal_used": False}, None


def bse_convergence(path):
    text = path.read_text()
    match = re.search(r"Davidson:\s+Converged;\s+(\d+)/(\d+) roots[^\n]*"
                      r"max residual=([0-9.eE+-]+)", text, re.I)
    require(match is not None, "missing converged Davidson residual receipt")
    converged, requested = int(match[1]), int(match[2])
    residual_ry = float(match[3])
    require(converged == requested == 3072, "not all3072 roots converged")
    details, values = eigenvalues(path.with_name("eigenvalues.dat"))
    require(details["roots"] == requested, "solver/root-file count mismatch")
    with h5py.File(path.with_name("WFN.h5"), "r") as h:
        cell_volume = float(h["mf_header/crystal/celvol"][()])
    require(details["nspin"] == 1 and details["nspinor"] == 2,
            "SOC tutorial spin header mismatch")
    require(np.isclose(details["volume_supercell_bohr3"], 64 * cell_volume, rtol=2e-8),
            "eigenvalue normalization volume differs from64 fine k times cell volume")
    # Native stopping rule is absolute below1Ry, relative above1Ry.
    tol_mev = .01
    bound_ry = (tol_mev / 13605.6980659) * max(1., float(np.max(abs(values[:, 0]))) / 13.6056980659)
    require(np.isfinite(residual_ry) and residual_ry <= bound_ry,
            "reported residual exceeds the configured0.01meV stopping rule")
    require("BSE COMPLETED" in text or "bse COMPLETED" in text or "eigensolve" in text,
            "missing completed BSE report")
    return {"converged_roots": converged, "requested_roots": requested,
            "max_residual_ry": residual_ry, "max_residual_mev": residual_ry * 13605.6980659,
            "solver_tolerance_mev": tol_mev}, None


def velocity_guard(path):
    with h5py.File(path, "r") as h:
        e = h["band_energies"][()] * 13.6056980659
        ds = h["dipole_cart"]
        require(e.shape == (9, 200) and ds.shape == (3, 9, 200, 200),
                "tutorial polar velocity/energy shape differs from200 bands on9k")
        require(np.isfinite(e).all() and finite_hdf(ds), "nonfinite polar velocity/energy")
        top, cut = float(e[:, 47].max()), float(e[:, 48].min())
        require(cut > top, "48-band velocity fit is not globally isolated on the coarse grid")
        return {"fitted_bands": 48, "polar_operator_shape": list(ds.shape),
                "fit_top_max_ev": top, "first_unfitted_min_ev": cut,
                "global_coarse_isolation_ev": cut-top, "velocity_weight_route": "g_equals_one"}, None


def dft_boundary(path):
    xml_path = path.with_name("data-file-schema.xml")
    mode = "none"
    for _, element in ET.iterparse(xml_path, events=("end",)):
        if element.tag.split("}")[-1] == "assume_isolated" and element.text:
            mode = element.text.strip() or "none"
        element.clear()
    require(mode in ("none", "2D"), "unsupported QE boundary in tutorial XML")
    expected = 2 if mode == "2D" else 3
    with h5py.File(path, "r") as h:
        attrs = h["kin_ion"].attrs
        recorded, truncated = int(attrs["sys_dim"]), bool(attrs["truncation_2d"])
        require(recorded == expected and truncated == (expected == 2),
                "kinetic/ionic artifact boundary differs from associated QE XML")
        return {"qe_assume_isolated": mode, "dft_sys_dim": recorded,
                "truncation_2d": truncated, "kin_ion_shape": list(h["kin_ion"].shape)}, None


def compare_tables(kind, a, b):
    if kind in ("eqp", "bandstructure"):
        cols = (2, 3, 4)
        digits = 6 if kind == "eqp" else 5
        def key(row):
            k = row[list(cols)]
            if kind == "eqp":
                k = np.mod(k, 1.)
                k = np.where(np.isclose(k, 1., atol=1e-6), 0., k)
            return (int(row[0]), int(row[1]), *np.round(k, digits))
        d = {key(row): row for row in b}
        matches = [(row, d[key(row)]) for row in a if key(row) in d]
        if not matches:
            return {"matched_rows": 0}
        aa, bb = map(np.asarray, zip(*matches))
        out = {"matched_rows": len(aa), "native_rows": len(a), "bgw_rows": len(b)}
        for col, label in ((5, "dft"), (6, "qp")):
            delta = aa[:, col] - bb[:, col]
            out[label + "_mean_difference_ev"] = float(delta.mean())
            out[label + "_rms_difference_ev"] = float(np.sqrt(np.mean(delta**2)))
            out[label + "_max_abs_difference_ev"] = float(np.max(np.abs(delta)))
            frontier = (aa[:, 1] >= 24) & (aa[:, 1] <= 30)
            out[label + "_frontier_mean_difference_ev"] = float(delta[frontier].mean())
            out[label + "_frontier_max_abs_difference_ev"] = float(np.max(np.abs(delta[frontier])))
        return out
    if kind == "eigenvalues":
        n = min(len(a), len(b))
        d = a[:n, 0] - b[:n, 0]
        return {"roots_compared_in_energy_order": n, "lowest_energy_difference_ev": float(d[0]),
                "energy_rms_difference_ev": float(np.sqrt(np.mean(d**2))),
                "energy_max_abs_difference_ev": float(np.max(np.abs(d)))}
    keep = (a[:, 0] >= b[0, 0]) & (a[:, 0] <= b[-1, 0])
    require(keep.any(), "no overlapping spectral frequency range")
    ref = np.interp(a[keep, 0], b[:, 0], b[:, 1])
    diff = a[keep, 1] - ref
    return {"omega_samples_compared": int(keep.sum()),
            "eps2_relative_l2_difference": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-300)),
            "eps2_max_abs_difference": float(np.max(np.abs(diff)))}


def compare_eps(native, bgw):
    out = []
    with h5py.File(native, "r") as a, h5py.File(bgw, "r") as b:
        qa, qb = a["eps_header/qpoints/qpts"][()], b["eps_header/qpoints/qpts"][()]
        for i, q in enumerate(qa):
            matches = np.where(np.max(np.abs(qb - q), axis=1) < 2e-7)[0]
            if len(matches) != 1:
                out.append({"q": q.tolist(), "status": "UNMATCHED"})
                continue
            j = int(matches[0])
            na = int(a["eps_header/gspace/nmtx"][i])
            nb = int(b["eps_header/gspace/nmtx"][j])
            ia = a["eps_header/gspace/gind_eps2rho"][i, :na] - 1
            ib = b["eps_header/gspace/gind_eps2rho"][j, :nb] - 1
            ga, gb = a["mf_header/gspace/components"][()][ia], b["mf_header/gspace/components"][()][ib]
            permuted = not np.array_equal(ga, gb)
            lookup = {tuple(g): k for k, g in enumerate(gb)}
            if len(ga) != len(gb) or set(map(tuple, ga)) != set(lookup):
                out.append({"q": q.tolist(), "status": "G_SETS_DIFFER"})
                continue
            order = np.asarray([lookup[tuple(g)] for g in ga])
            x = a["mats/matrix"][i, 0, 0, :na, :na]
            y = b["mats/matrix"][j, 0, 0, :nb, :nb]
            x, y = x[..., 0] + 1j*x[..., 1], y[..., 0] + 1j*y[..., 1]
            y = y[np.ix_(order, order)]
            out.append({"q": q.tolist(), "status": "COMPARED", "nmtx": na,
                        "exact_miller_permutation_applied": permuted,
                        "relative_frobenius_difference": float(np.linalg.norm(x-y)/np.linalg.norm(y)),
                        "max_abs_difference": float(np.max(np.abs(x-y))),
                        "native_00": [float(x[0, 0].real), float(x[0, 0].imag)],
                        "bgw_00": [float(y[0, 0].real), float(y[0, 0].imag)]})
    return out


def run(root):
    report = {"root": str(root), "format_checks": [], "comparisons": [],
              "comparison_policy": "Differences are descriptive; no cross-code equality gate.",
              "physics_parity_verdict": "NOT_ASSERTED",
              "verdict_scope": "Formats, complete on-shell evaluation, and BGW spatial q coverage.",
              "file_read_policy": "Read only; completed artifacts required for a final verdict."}
    arrays = {}
    parsers = {"eqp": eqp, "bandstructure": bands, "eigenvalues": eigenvalues,
               "absorption": absorption, "eps": eps, "bsemat": bsemat,
               "native_path": native_path, "sigma_coverage": sigma_coverage,
               "bse_convergence": bse_convergence, "adapter_semantics": adapter_semantics}
    parsers.update(velocity_guard=velocity_guard, dft_boundary=dft_boundary)
    parsers["eqp_reader_layout"] = eqp_reader_layout
    parsers["eps_q_coverage"] = eps_q_coverage
    jobs = []
    for impl, directory in (("bgw", "2-fully_rel"), ("native", "3-lorrax_fully_rel")):
        gw = root / "2-gw" / directory
        bs = root / "3-bse" / directory
        for name in ("eqp0.dat", "eqp1.dat"):
            jobs.append((impl, "eqp", name, gw / "2-sigma" / name))
            jobs.append((impl, "eqp_reader_layout", name+"_BGW_reader", gw / "2-sigma" / name))
        jobs.append((impl, "bandstructure", "bandstructure.dat", gw / "3-bs/bandstructure.dat"))
        if impl == "native":
            jobs.extend((impl, "native_path", p.name, p)
                        for p in sorted((gw / "3-bs").glob("bandstructure*.lorrax.dat")))
            jobs.append((impl, "sigma_coverage", "sigma_mnk.h5", gw / "2-sigma/sigma_mnk.h5"))
            jobs.append((impl, "adapter_semantics", "bandstructure_absolute_reference", gw / "3-bs/bandstructure.dat"))
            jobs.append((impl, "velocity_guard", "48_band_velocity_isolation", gw / "1-epsilon/dipole.h5"))
            jobs.append((impl, "dft_boundary", "DFT_boundary_provenance", gw / "1-epsilon/kin_ion.h5"))
        for name in ("eps0mat.h5", "epsmat.h5"):
            jobs.append((impl, "eps", name, gw / "1-epsilon" / name))
        jobs.append((impl, "eps_q_coverage", "epsmat_spatial_q_coverage", gw / "1-epsilon/epsmat.h5"))
        jobs.append((impl, "bsemat", "bsemat.h5", bs / "1-kernel/bsemat.h5"))
        ab = bs / "2-absorption"
        if impl == "native":
            jobs.append((impl, "bse_convergence", "BSE3072_residual", ab / "absorption.out"))
        eig_files = sorted(ab.glob("eigenvalues*.dat"))
        eig_files = [p for p in eig_files if "noeh" not in p.name]
        jobs.extend((impl, "eigenvalues", p.name, p) for p in eig_files or [ab / "eigenvalues.dat"])
        abs_files = sorted(ab.glob("absorption*.dat"))
        jobs.extend((impl, "absorption", p.name, p) for p in abs_files or
                    [ab / "absorption_eh.dat", ab / "absorption_noeh.dat"])
    for impl, kind, name, path in jobs:
        rec = {"implementation": impl, "kind": kind, "name": name,
               "category": "coverage" if kind in ("sigma_coverage", "eps_q_coverage", "bse_convergence", "adapter_semantics", "velocity_guard", "dft_boundary", "eqp_reader_layout") else "format",
               "path": str(path), "status": "PENDING"}
        try:
            if path.is_file():
                rec["bytes"] = path.stat().st_size
                details, a = parsers[kind](path)
                rec.update(status="PASS", details=details)
                if a is not None:
                    arrays[(impl, kind, name)] = a
        except Exception as exc:
            rec.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        report["format_checks"].append(rec)
    for (impl, kind, name), a in arrays.items():
        other = ("bgw", kind, name)
        if impl == "native" and other in arrays:
            if kind == "native_path":
                continue
            report["comparisons"].append({"kind": kind, "name": name,
                                           **compare_tables(kind, a, arrays[other])})
    for name in ("eps0mat.h5", "epsmat.h5"):
        a = root / "2-gw/3-lorrax_fully_rel/1-epsilon" / name
        b = root / "2-gw/2-fully_rel/1-epsilon" / name
        if a.is_file() and b.is_file():
            report["comparisons"].append({"kind": "eps", "name": name,
                                           "blocks": compare_eps(a, b)})
    failures = [r for r in report["format_checks"] if r["status"] == "FAIL"]
    pending = [r for r in report["format_checks"] if r["status"] == "PENDING"]
    report["status"] = "FAIL" if failures else "PENDING" if pending else "PASS"
    for category in ("format", "coverage"):
        rows = [r for r in report["format_checks"] if r["category"] == category]
        report[category + "_status"] = ("FAIL" if any(r["status"] == "FAIL" for r in rows)
                                         else "PENDING" if any(r["status"] == "PENDING" for r in rows)
                                         else "PASS")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutorial-root", "--root", dest="root", type=Path, default=Path("."),
                    help="6a-SOC tutorial directory; defaults to the current directory")
    args = ap.parse_args()
    print(json.dumps(run(args.root), indent=2, allow_nan=False))
