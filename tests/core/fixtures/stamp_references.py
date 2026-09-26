"""Write or verify the portable SHA-256 stamps for core fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REFERENCE_COMMIT = "298c89ac588300460f241191bff2e65d8a50b824"

FILES = {
    "A": (
        "H.upf", "Li.upf", "WFN.h5", "data-file-schema.xml",
        "MEAN_FIELD.json", "kmeans.out",
        "centroids_frac_21.txt", "centroids_frac_31_htransform.txt",
        "kmeans_htransform.out", "build/scf.in", "build/nscf.in",
        "build/pw2bgw.in", "kin_ion.h5", "dipole.h5", "cohsex.in",
        "cohsex_eqp0.dat", "cohsex_eqp1.dat", "cohsex_sigma.dat",
        "gnppm.in", "gnppm_eqp0.dat", "gnppm_eqp1.dat",
        "gnppm_sigma.dat", "gnppm_sigma.h5",
        "htransform.in", "exciton.in", "exciton_gw.in",
        "excited_state_ref.json", "tmp/isdf_tensors_21.h5",
        "tmp/isdf_tensors_31.h5", "tmp/zeta_q.h5",
    ),
    "A-prime": (
        "H.upf", "Li.upf", "WFN.h5", "data-file-schema.xml",
        "MEAN_FIELD.json", "centroids_frac_21_current.txt",
        "kmeans_current.out", "build/scf.in", "build/nscf.in",
        "build/pw2bgw.in",
    ),
    "A-cubic": (
        "H.upf", "WFN.h5", "data-file-schema.xml", "MEAN_FIELD.json",
        "centroids_frac_48.txt", "centroids_frac_23_literal.txt",
        "kmeans.out", "kmeans_literal.out",
        "build/scf.in", "build/nscf.in", "build/pw2bgw.in",
    ),
    "B": (
        "He.upf", "WFN.h5", "data-file-schema.xml", "MEAN_FIELD.json",
        "centroids_frac_13.txt", "kmeans.out",
        "build/scf.in", "build/nscf.in",
        "build/pw2bgw.in", "kin_ion.h5", "dipole.h5", "mpa.in",
        "mpa_eqp0.dat", "mpa_eqp1.dat", "mpa_sigma.dat", "mpa_sigma.h5",
        "mpa_sc1.in", "mpa_sc1.out", "mpa_sc1_eqp0.dat",
        "mpa_sc1_eqp1.dat", "mpa_sc1_sigma.dat", "mpa_sc1_sigma.h5",
        "eqp0_iter0000.dat", "eqp0_iter0001.dat", "eqp1_iter0000.dat",
        "eqp1_iter0001.dat", "WFN_qp.h5", "qp_wfn_rotations.h5",
        "tmp/isdf_tensors_13.h5", "tmp/zeta_q.h5",
    ),
}

MEASUREMENTS = {
    "A": {
        "mean_field_base_seconds": 10.644153629007633,
        "mean_field_plus_htransform_basis_seconds": 13.720153629007633,
        "htransform_centroid_selection_seconds": 3.076,
        "cohsex_seconds": 15.48,
        "gn_ppm_seconds_cold": 67.00,
        "htransform_seconds": 10.96,
        "bse_tda_seconds": 7.25,
        "exciton_bands_seconds": 15.29,
    },
    "A-prime": {"mean_field_seconds": 56.0},
    "A-cubic": {"mean_field_seconds": 30.52218297199579},
    "B": {
        "mean_field_seconds": 13.608722165998188,
        "mpa_one_shot_seconds_cold": 25.38,
        "mpa_one_update_seconds_cached": 15.49,
    },
}

SHAPES = {
    "A": {"atoms": 2, "kgrid": [3, 3, 1], "stored_k": 5,
          "full_k": 9, "active_bands": 7, "guard_bands": 1,
          "centroids": 21, "htransform_centroids": 31,
          "fft_grid": [15, 15, 15],
          "spinor": 1, "trs": True, "spatial_operations": 1},
    "A-prime": {"atoms": 2, "kgrid": [3, 3, 1], "stored_k": 9,
                "full_k": 9, "active_bands": 7, "guard_bands": 0,
                "centroids": 21, "fft_grid": [15, 15, 15],
                "spinor": 2, "trs": False, "spatial_operations": 1},
    "A-cubic": {"atoms": 2, "kgrid": [2, 2, 2], "stored_k": 3,
                "full_k": 8, "active_bands": 8, "guard_bands": 0,
                "centroids": 48, "non_orbit_centroids": 23,
                "fft_grid": [12, 12, 12], "spinor": 1, "trs": True,
                "spatial_operations": 48,
                "fractional_translation_operations": 24},
    "B": {"atoms": 1, "kgrid": [1, 1, 1], "stored_k": 1,
          "full_k": 1, "active_bands": 7, "guard_bands": 1,
          "centroids": 13, "fft_grid": [18, 18, 20],
          "spinor": 1, "trs": True, "spatial_operations": 2,
          "dft_gap_ev": 14.99870},
}

EXTRA_GLOBS = {
    "A": ("tmp/sigma_quadrature_rules/*.npz",),
    "B": ("tmp/sigma_quadrature_rules/*.npz", "tmp/mpa/*.h5"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected(label: str) -> dict:
    root = HERE / label
    names = list(FILES[label])
    for pattern in EXTRA_GLOBS.get(label, ()):
        names.extend(
            str(path.relative_to(root)) for path in sorted(root.glob(pattern))
        )
    if len(names) != len(set(names)):
        raise ValueError(f"{label}: duplicate provenance file names")
    result = {
        "schema": "lorrax-core-fixture-v1",
        "system": label,
        "reference_source_commit": REFERENCE_COMMIT,
        "shape": SHAPES[label],
        "cold_build": MEASUREMENTS[label],
        "files": {name: sha256(root / name) for name in names},
    }
    if label == "A":
        # 2026-09-26 re-freeze (owner approval 2026-09-26): gnppm.in runs at
        # the production Sigma default 3e-5, on the omega-split rule partition
        # (RC2).  GN-PPM refs + box rules (schema v5, the rules the cut run
        # wrote; the v4 set was no longer served) cut at 8aeb755fc, P4 one
        # node.  Old (1e-3, main's partition) -> new, max / median over 25
        # rows: eqp0 4.753 / 0.450 meV, eqp1 6.035 / 0.997 meV -- the old
        # reference's own 1e-3 quadrature error.  Main's partition at 3e-5
        # agrees with the new refs to 0.045 meV (eqp0) / 0.035 meV (eqp1).
        # Earlier cuts: tests/known_failures/2026-09-24-bispinor-and-core-
        # fixture-refreeze.md.
        # - excited_state_ref.json: scalar-singlet exchange weight
        #   (a0bae022, re-frozen by 01465925).
        result["additional_reference_source_commits"] = {
            "htransform_bse_exciton":
                "1fc5cb8f2b974a14ac1c5f97f5c9d7ee2be274b0",
            "bse_scalar_singlet_exchange":
                "01465925b7a12d63181a737f866a5c81f03a7b68",
            "gnppm_sigma_eps_3e5_box_rules_v5":
                "8aeb755fc70ff597263c266f6483b83eb457a5ad",
        }
    if label == "B":
        # MPA one-shot / one-update references regenerated under the
        # energy-proportional SC pad and per-k identity partition: band 3
        # (9.2 eV, outside the requested +-12 eV grid) is evaluated on the
        # padded support instead of clamped to the scissor (eqp0 -3.97 meV,
        # eqp1 -10.32 meV; bands 1-2 unchanged to 0.07 meV).
        # 2026-09-24 restamp: the mpa.in / mpa_sc1.in deck bytes changed
        # (89eaa9a3 retired the quadrature time-budget key; d4214ace and
        # a5a35701 renamed the SC accelerator).  No reference was regenerated;
        # the mpa_sc1_* family stays the strict xfail in tests/KNOWN_FAILURES.md.
        # 2026-09-26: mpa.in / mpa_sc1.in run at the test-deck floor 5e-4; the
        # shipped rule cache is the schema-v5 set the 8aeb755fc run wrote (the
        # 23 older-schema files were ignored).  No reference was regenerated.
        result["additional_reference_source_commits"] = {
            "mpa_sc_pad_identity":
                "0262d4833c470ef5768270ca048de1faaaf06a9b",
        }
    return result


def write(label: str) -> None:
    path = HERE / label / "PROVENANCE.json"
    path.write_text(
        json.dumps(expected(label), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}: {len(expected(label)['files'])} authenticated files")


def check(label: str) -> None:
    path = HERE / label / "PROVENANCE.json"
    got = json.loads(path.read_text(encoding="utf-8"))
    want = expected(label)
    if got != want:
        raise ValueError(
            f"{path}: provenance mismatch; run "
            "`python -m tests.core.fixtures.stamp_references` only after "
            "deliberately regenerating and reviewing the references"
        )
    print(f"verified {path}: {len(want['files'])} authenticated files")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("systems", nargs="*", choices=tuple(FILES))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    labels = args.systems or list(FILES)
    for label in labels:
        (check if args.check else write)(label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
