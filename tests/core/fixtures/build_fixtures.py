"""Build the tiny, hash-addressed core-suite mean-field fixture family.

This is maintenance tooling, never imported by production code.  Run it on a
compute node; the ordinary core suite consumes the committed files and never
invokes Quantum ESPRESSO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
DEFAULT_PSP_DIR = Path(
    "/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
    "assets/pseudopotentials/standard"
)
DEFAULT_QE_BIN = Path(
    "/global/common/software/nersc9/espresso/"
    "7.3.1-libxc-6.2.2-cpu/bin"
)
DEFAULT_BGW_BIN = Path(
    "/global/common/software/nersc9/berkeleygw/zen3/gcc-12/mpich/"
    "berkeleygw/BerkeleyGW-4.0/bin"
)

SYSTEMS = {
    "A": {
        "prefix": "core_lih_p1",
        "pseudos": {
            "H.upf": "8272fc45fbee2b45a490830c61ef9b2f2b894591b2c9a310fe4af152b76502a6",
            "Li.upf": "3654e8a51fb6c60c124c97ea3a67a54c87a0e477e49effd65aa73a45288f48f9",
        },
        "centroids": "centroids_frac_21.txt",
        "centroid_args": (
            "21", "--seed", "42", "--orbit", "--oversample", "1.0",
            "--fit-window", "0:2,0:7",
        ),
        "extra_centroids": {
            "centroids_frac_31_htransform.txt": (
                "31", "--seed", "42", "--orbit", "--oversample", "1.0",
                "--fit-window", "0:2,0:3", "--out-suffix", "_htransform",
            ),
        },
    },
    "A-prime": {
        "prefix": "core_lih_p1_trbroken",
        "pseudos": {
            "H.upf": "8272fc45fbee2b45a490830c61ef9b2f2b894591b2c9a310fe4af152b76502a6",
            "Li.upf": "3654e8a51fb6c60c124c97ea3a67a54c87a0e477e49effd65aa73a45288f48f9",
        },
        "centroids": "centroids_frac_21_current.txt",
        "centroid_report": "kmeans_current.out",
        # Measured as the 25 s mean-field step plus the required 31 s P=4
        # transverse-Gram selection step (lx steps .52 and .54).
        "staged_cold_build_seconds": 56.0,
        "centroid_args": (
            "21", "--seed", "42", "--orbit", "--oversample", "1.5",
            "--fit-window", "0:3,0:7", "--density-mode", "current",
        ),
    },
    "A-cubic": {
        "prefix": "core_hdiamond",
        "pseudos": {
            "H.upf": "8272fc45fbee2b45a490830c61ef9b2f2b894591b2c9a310fe4af152b76502a6",
        },
        "centroids": "centroids_frac_48.txt",
        "centroid_args": (
            "24", "--seed", "42", "--orbit", "--oversample", "1.0",
            "--fit-window", "0:2,0:8",
        ),
        "extra_centroids": {
            "centroids_frac_23_literal.txt": (
                "23", "--seed", "42", "--no-orbit", "--oversample", "1.0",
                "--fit-window", "0:2,0:8", "--out-suffix", "_literal",
            ),
        },
    },
    "B": {
        "prefix": "core_he",
        "pseudos": {
            "He.upf": "b0c5b4036abe1d2ccbc97fe1d183a511a39dcee85c7801cdf488e3701207d428",
        },
        "centroids": "centroids_frac_13.txt",
        "generated_centroids": "centroids_frac_14.txt",
        "odd_inversion_target": 13,
        "centroid_args": (
            "13", "--seed", "42", "--orbit", "--oversample", "1.0",
            "--fit-window", "0:1,0:7",
        ),
    },
}
INPUT_NAMES = ("scf.in", "nscf.in", "pw2bgw.in")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_digest(label: str, pseudos: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(SYSTEMS[label]["centroid_args"]).encode("utf-8"))
    if "odd_inversion_target" in SYSTEMS[label]:
        digest.update(json.dumps({
            "generated_centroids": SYSTEMS[label]["generated_centroids"],
            "odd_inversion_target": SYSTEMS[label]["odd_inversion_target"],
        }, sort_keys=True).encode("utf-8"))
    if SYSTEMS[label].get("extra_centroids"):
        digest.update(json.dumps(
            SYSTEMS[label]["extra_centroids"], sort_keys=True
        ).encode("utf-8"))
    digest.update(b"\0")
    paths = [HERE / label / "build" / name for name in INPUT_NAMES] + pseudos
    for path in paths:
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run(argv: list[str], *, cwd: Path, stdin: Path | None, log: Path) -> None:
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            argv,
            cwd=cwd,
            stdin=stdin.open("rb") if stdin is not None else None,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"fixture build command failed with rc={result.returncode}: "
            f"{argv!r}; inspect {log}"
        )


def _validate_tools(qe_bin: Path, bgw_bin: Path) -> dict[str, Path]:
    tools = {
        "pw.x": qe_bin / "pw.x",
        "pw2bgw.x": qe_bin / "pw2bgw.x",
        "wfn2hdf.x": bgw_bin / "wfn2hdf.x",
    }
    missing = [str(path) for path in tools.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "core fixture build requires these compute-node tools: "
            + ", ".join(missing)
        )
    return tools


def _write_odd_inversion_subset(cache: Path, spec: dict) -> None:
    """Reduce an even k-means inversion closure to an odd closed point set."""
    generated = cache / spec["generated_centroids"]
    target_path = cache / spec["centroids"]
    target = int(spec["odd_inversion_target"])
    lines = generated.read_text(encoding="utf-8").splitlines()
    header = [line for line in lines if line.lstrip().startswith("#")]
    rows = [
        tuple(round(float(value) % 1.0, 6) for value in line.split())
        for line in lines if line.strip() and not line.lstrip().startswith("#")
    ]

    def partner(row: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(round((-value) % 1.0, 6) for value in row)

    available = set(rows)
    orbits: list[tuple[tuple[float, float, float], ...]] = []
    while available:
        first = min(available)
        image = partner(first)
        if image not in available:
            raise ValueError(
                f"{generated}: inversion image {image} of {first} is absent"
            )
        orbit = tuple(sorted({first, image}))
        orbits.append(orbit)
        available.difference_update(orbit)

    fixed = (0.5, 0.5, 0.5)
    if all(fixed not in orbit for orbit in orbits):
        orbits.append((fixed,))
    selected: list[tuple[tuple[float, float, float], ...]] = []
    remaining = target
    for orbit in sorted(orbits, key=lambda group: (len(group), group)):
        if len(orbit) <= remaining:
            selected.append(orbit)
            remaining -= len(orbit)
        if remaining == 0:
            break
    if remaining:
        raise ValueError(
            f"cannot choose {target} inversion-closed sites from orbit sizes "
            f"{[len(orbit) for orbit in orbits]}"
        )
    output_rows = sorted(row for orbit in selected for row in orbit)
    output = header + [
        "# postprocess: deterministic inversion-closed odd subset; "
        f"target={target}; fixed_point=0.5,0.5,0.5",
        *[" ".join(f"{value:.6f}" for value in row) for row in output_rows],
    ]
    target_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _published_hit(label: str, input_sha256: str) -> bool:
    stamp = HERE / label / "MEAN_FIELD.json"
    wfn = HERE / label / "WFN.h5"
    pseudos = [HERE / label / name for name in SYSTEMS[label]["pseudos"]]
    schema = HERE / label / "data-file-schema.xml"
    centroids = HERE / label / SYSTEMS[label]["centroids"]
    extra_centroids = [
        HERE / label / name
        for name in SYSTEMS[label].get("extra_centroids", {})
    ]
    extra_reports = []
    for args in SYSTEMS[label].get("extra_centroids", {}).values():
        suffix = (args[args.index("--out-suffix") + 1]
                  if "--out-suffix" in args else "")
        extra_reports.append(HERE / label / f"kmeans{suffix}.out")
    if (not stamp.is_file() or not wfn.is_file()
            or any(not pseudo.is_file() for pseudo in pseudos)
            or not schema.is_file() or not centroids.is_file()
            or any(not path.is_file() for path in extra_centroids)
            or any(not path.is_file() for path in extra_reports)):
        return False
    record = json.loads(stamp.read_text(encoding="utf-8"))
    return (
        record.get("input_sha256") == input_sha256
        and record.get("outputs", {}).get("WFN.h5") == _sha256(wfn)
        and all(
            record.get("outputs", {}).get(pseudo.name) == _sha256(pseudo)
            for pseudo in pseudos
        )
        and record.get("outputs", {}).get(schema.name) == _sha256(schema)
        and record.get("outputs", {}).get(centroids.name) == _sha256(centroids)
        and all(
            record.get("outputs", {}).get(path.name) == _sha256(path)
            for path in extra_centroids
        )
        and all(
            record.get("outputs", {}).get(path.name) == _sha256(path)
            for path in extra_reports
        )
    )


def build_mean_field(
    label: str,
    *,
    pseudo_dir: Path,
    qe_bin: Path,
    bgw_bin: Path,
    rebuild: bool,
) -> dict:
    spec = SYSTEMS[label]
    pseudos = [pseudo_dir / name for name in spec["pseudos"]]
    for pseudo in pseudos:
        if not pseudo.is_file():
            raise FileNotFoundError(f"missing pseudopotential {pseudo}")
        got = _sha256(pseudo)
        expected = spec["pseudos"][pseudo.name]
        if got != expected:
            raise ValueError(
                f"{pseudo}: SHA-256 {got}, expected {expected}"
            )
    input_sha = _input_digest(label, pseudos)
    if not rebuild and _published_hit(label, input_sha):
        print(f"fixture {label}: HIT {input_sha}")
        return json.loads(
            (HERE / label / "MEAN_FIELD.json").read_text(encoding="utf-8")
        )

    tools = _validate_tools(qe_bin, bgw_bin)
    cache = HERE / ".build-cache" / f"{label}-{input_sha}"
    cache.mkdir(parents=True, exist_ok=True)
    for name in INPUT_NAMES:
        shutil.copy2(HERE / label / "build" / name, cache / name)
    for pseudo in pseudos:
        shutil.copy2(pseudo, cache / pseudo.name)

    source_schema = cache / f"{spec['prefix']}.save" / "data-file-schema.xml"
    source_centroids = cache / spec["centroids"]
    cached_products = [cache / "WFN.h5", source_schema, source_centroids]
    cached_products.extend(
        cache / name for name in spec.get("extra_centroids", {})
    )
    reuse_staged = (
        not rebuild
        and "staged_cold_build_seconds" in spec
        and all(path.is_file() for path in cached_products)
    )
    if reuse_staged:
        cold_wall = float(spec["staged_cold_build_seconds"])
    else:
        started = time.perf_counter()
        _run([str(tools["pw.x"]), "-in", "scf.in"], cwd=cache,
             stdin=None, log=cache / "scf.out")
        _run([str(tools["pw.x"]), "-in", "nscf.in"], cwd=cache,
             stdin=None, log=cache / "nscf.out")
        _run([str(tools["pw2bgw.x"]), "-in", "pw2bgw.in"], cwd=cache,
             stdin=None, log=cache / "pw2bgw.out")
        _run([str(tools["wfn2hdf.x"]), "BIN", "WFN", "WFN.h5"],
             cwd=cache, stdin=None, log=cache / "wfn2hdf.out")
        _run([sys.executable, "-m", "centroid.kmeans_cli",
              *spec["centroid_args"]], cwd=cache, stdin=None,
             log=cache / "kmeans.stdout")
        if "odd_inversion_target" in spec:
            _write_odd_inversion_subset(cache, spec)
        for name, centroid_args in spec.get("extra_centroids", {}).items():
            _run([sys.executable, "-m", "centroid.kmeans_cli", *centroid_args],
                 cwd=cache, stdin=None,
                 log=cache / f"{Path(name).stem}.stdout")
        cold_wall = time.perf_counter() - started

    wfn = cache / "WFN.h5"
    if not wfn.is_file() or wfn.stat().st_size < 4096:
        raise RuntimeError(f"fixture {label}: missing/truncated {wfn}")
    shutil.copy2(wfn, HERE / label / "WFN.h5")
    for pseudo in pseudos:
        shutil.copy2(pseudo, HERE / label / pseudo.name)
    if not source_schema.is_file():
        raise RuntimeError(f"fixture {label}: missing QE schema {source_schema}")
    shutil.copy2(source_schema, HERE / label / "data-file-schema.xml")
    if not source_centroids.is_file():
        raise RuntimeError(
            f"fixture {label}: centroid selector did not write "
            f"{source_centroids}"
        )
    shutil.copy2(source_centroids, HERE / label / source_centroids.name)
    centroid_report = spec.get("centroid_report", "kmeans.out")
    shutil.copy2(cache / centroid_report, HERE / label / centroid_report)
    for name in spec.get("extra_centroids", {}):
        source_extra = cache / name
        if not source_extra.is_file():
            raise RuntimeError(
                f"fixture {label}: centroid selector did not write {source_extra}"
            )
        shutil.copy2(source_extra, HERE / label / name)
        extra_args = tuple(spec["extra_centroids"][name])
        suffix = (extra_args[extra_args.index("--out-suffix") + 1]
                  if "--out-suffix" in extra_args else "")
        extra_report = cache / f"kmeans{suffix}.out"
        if extra_report.is_file():
            shutil.copy2(extra_report, HERE / label / extra_report.name)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=HERE, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    source_dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=HERE, check=False,
    ).returncode != 0
    record = {
        "schema": "lorrax-core-mean-field-v1",
        "system": label,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "input_sha256": input_sha,
        "pseudopotentials": {
            pseudo.name: _sha256(pseudo) for pseudo in pseudos
        },
        "tools": {name: str(path) for name, path in tools.items()},
        "cold_build_seconds": cold_wall,
        "staged_cache_reused": reuse_staged,
        "outputs": {
            "WFN.h5": _sha256(HERE / label / "WFN.h5"),
            **{
                pseudo.name: _sha256(HERE / label / pseudo.name)
                for pseudo in pseudos
            },
            "data-file-schema.xml": _sha256(
                HERE / label / "data-file-schema.xml"
            ),
            source_centroids.name: _sha256(
                HERE / label / source_centroids.name
            ),
            **{
                name: _sha256(HERE / label / name)
                for name in spec.get("extra_centroids", {})
            },
            **{
                f"kmeans{args[args.index('--out-suffix') + 1]}.out":
                    _sha256(
                        HERE / label /
                        f"kmeans{args[args.index('--out-suffix') + 1]}.out"
                    )
                for args in spec.get("extra_centroids", {}).values()
                if "--out-suffix" in args
            },
            centroid_report: _sha256(HERE / label / centroid_report),
        },
        "cache": str(cache),
    }
    (HERE / label / "MEAN_FIELD.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"fixture {label}: BUILT {input_sha} in {cold_wall:.3f} s")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("systems", nargs="*", choices=tuple(SYSTEMS))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--pseudo-dir",
        type=Path,
        default=Path(os.environ.get("LORRAX_CORE_PSP_DIR", DEFAULT_PSP_DIR)),
    )
    parser.add_argument("--qe-bin", type=Path, default=DEFAULT_QE_BIN)
    parser.add_argument("--bgw-bin", type=Path, default=DEFAULT_BGW_BIN)
    args = parser.parse_args(argv)
    labels = args.systems or list(SYSTEMS)
    for label in labels:
        build_mean_field(
            label,
            pseudo_dir=args.pseudo_dir,
            qe_bin=args.qe_bin,
            bgw_bin=args.bgw_bin,
            rebuild=args.rebuild,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
