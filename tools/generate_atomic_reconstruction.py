#!/usr/bin/env python3
"""Generate authenticated OCEAN-style atomic reconstruction from an FR UPF.

Run on a compute node with --oncv-source pointing to ONCVPSP3.3.1 source.
The source is copied into an isolated output directory. No source input,
pseudopotential, old run, scheduler allocation, or external installation is
modified. The source operator and atomic configuration must be reproduced.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess

from psp.atomic_reconstruction import (
    embedded_oncv_input, restored_oncv_input, generated_upf_text,
    compare_regenerated_upf, radial_operator_difference,
    read_atomic_scattering_bank, read_atomic_frozen_core,
    compress_atomic_scattering_bank, write_atomic_reconstruction,
    scattering_branch_diagnostics,
)


def copy_generator(source, destination):
    destination.joinpath("src").mkdir(parents=True)
    for path in source.joinpath("src").iterdir():
        if path.suffix in (".f", ".f90", ".F90") or path.name == "Makefile":
            shutil.copy2(path, destination / "src" / path.name)
    shutil.copy2(source / "COPYING", destination / "COPYING")
    destination.joinpath("make.inc").write_text(
        "F77=gfortran\nF90=gfortran\nCC=gcc\nFCCPP=cpp\nFLINKER=$(F90)\n"
        "FCCPPFLAGS=\nFFLAGS=-O2 -fallow-argument-mismatch\nCFLAGS=-O2\n"
        "LIBS=-l:liblapack.so.3 -l:libblas.so.3\nOBJS_LIBXC=exc_libxc_stub.o\n")
    writer = destination / "src/upfout_r.f90"
    writer.write_text(writer.read_text().replace("(t4a", "(t4,a"))


def build_and_run(source, workdir):
    with workdir.joinpath("build.log").open("w") as log:
        subprocess.run(["make", "-j", "8", "oncvpspr"], cwd=source / "src",
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    with workdir.joinpath("oncv.in").open() as deck, workdir.joinpath("generation.out").open("w") as log:
        subprocess.run([str(source / "src/oncvpspr.x")], cwd=workdir,
                       stdin=deck, stdout=log, stderr=subprocess.STDOUT, check=True)
    # ONCV's STOP can return zero after an atomic failure; artifacts decide.
    generated = generated_upf_text(workdir / "generation.out")
    workdir.joinpath("regenerated.upf").write_text(generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oncv-source", required=True, type=Path)
    parser.add_argument("--upf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--discarded-weight", type=float, default=1e-8)
    parser.add_argument("--energy-max", type=float, default=5.0,
                        help="largest pseudo scattering energy in Ha")
    parser.add_argument("--energy-samples", type=int, default=128)
    parser.add_argument("--heldout-energies", type=int, default=32)
    args = parser.parse_args()
    if (args.energy_max <= 0 or args.energy_samples < 8 or args.heldout_energies < 1
            or not 0 < args.discarded_weight < 1):
        parser.error("require energy-max>0, energy-samples>=8, heldout-energies>=1 and 0<discarded-weight<1")
    source, upf, output = args.oncv_source.resolve(), args.upf.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline, exported = output / "baseline", output / "export"
    baseline.mkdir()
    exported.mkdir()
    baseline_tree, export_tree = output / "baseline_source", output / "export_source"
    copy_generator(source, baseline_tree)
    baseline.joinpath("oncv.in").write_text(embedded_oncv_input(upf))
    build_and_run(baseline_tree, baseline)
    deck, restoration = restored_oncv_input(upf, baseline / "generation.out")
    exported.joinpath("oncv.in").write_text(deck)
    exported.joinpath("atomic_export.in").write_text(
        f"{args.energy_max:.16e} {args.energy_samples} {args.heldout_energies}\n")
    copy_generator(source, export_tree)
    exporter = Path(__file__).with_name("atomic_reconstruction_export.f90")
    shutil.copy2(exporter, export_tree / "src" / exporter.name)
    driver = export_tree / "src/oncvpsp_r.f90"
    text = driver.read_text()
    marker = "! accumulate charge and eigenvalues"
    if text.count(marker) != 1:
        raise ValueError("ONCV source does not match the audited export insertion point")
    driver.write_text(text.replace(marker,
        "call export_atomic_reconstruction(lmax,lloc,nproj,rr,vfull,vp,vkb,evkb, &\n"
        "&                                   ep,rc,zz,mmax,nc,na,la,ea,fa)\n\n" + marker))
    makefile = export_tree / "src/Makefile"
    makefile.write_text(makefile.read_text().replace("OBJS_R =", "OBJS_R = atomic_reconstruction_export.o ", 1))
    build_and_run(export_tree, exported)
    version_match = re.search(r"fully-relativistic version ([\d.]+)",
                              exported.joinpath("generation.out").read_text())
    if version_match is None or version_match.group(1) != "3.3.1":
        raise ValueError("atomic exporter requires the audited ONCVPSP3.3.1 generator")
    comparison = compare_regenerated_upf(upf, exported / "regenerated.upf",
                                         relative_tolerance=2e-7, absolute_tolerance=2e-7)
    drift = radial_operator_difference(upf, exported / "regenerated.upf")
    if drift["hamiltonian_bound_ha"] > 1e-6:
        raise ValueError(f"atomic regeneration Hamiltonian drift exceeds 1e-6 Ha: {drift}")
    comparison["radial_hamiltonian_drift"] = drift
    bank = read_atomic_scattering_bank(exported / "atomic_bank.dat")
    arrays, diagnostics = compress_atomic_scattering_bank(bank, discarded_weight=args.discarded_weight)
    core = read_atomic_frozen_core(exported / "atomic_core.dat", bank["r"])
    arrays.update(core)
    branches = scattering_branch_diagnostics(bank, core)
    if not branches["validated"]:
        raise ValueError(f"atomic scattering phase branch failed: {branches}")
    metadata = {"source_sha256": hashlib.sha256(upf.read_bytes()).hexdigest(),
                "generator_version": "ONCVPSP3.3.1; portable UPF writer + kappa-resolved export",
                "generator_input_restoration": restoration, "operator_comparison": comparison,
                "phase_branch_validated": True, "phase_branch_diagnostics": branches,
                "reconstruction_energy_max_ha": args.energy_max,
                "energy_training_count": args.energy_samples,
                "energy_heldout_count": args.heldout_energies,
                "radius_augmentation": bank["radius_augmentation"],
                "normalization": "ONCV AE large-component reference; shared PS PCA coefficients",
                "core_small_phase": "iQ Omega_minus_kappa; sigma.rhat Omega_kappa=-Omega_minus_kappa",
                "reconstruction_diagnostics": diagnostics,
                "atomic_exporter_sha256": hashlib.sha256(exporter.read_bytes()).hexdigest(),
                "generator_binary_sha256": hashlib.sha256((export_tree / "src/oncvpspr.x").read_bytes()).hexdigest(),
                "source_bank_sha256": hashlib.sha256(exported.joinpath("atomic_bank.dat").read_bytes()).hexdigest()}
    write_atomic_reconstruction(output / "atomic_reconstruction.npz", arrays, metadata)
    output.joinpath("generation_receipt.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"sidecar": str(output / "atomic_reconstruction.npz"),
                      "hamiltonian_drift_ha": drift["hamiltonian_bound_ha"],
                      "ranks": arrays["channel_nopf"].tolist(),
                      "heldout": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
