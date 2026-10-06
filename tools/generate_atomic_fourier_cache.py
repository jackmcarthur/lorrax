#!/usr/bin/env python3
"""Prepare immutable paired species Fourier tables from declared manifest controls."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="directory containing manifest.json")
    parser.add_argument("--output", required=True, type=Path, help="new immutable cache directory")
    args = parser.parse_args()
    from runtime import bootstrap
    bootstrap(platform="cpu")
    from psp.atomic_reconstruction import load_atomic_reconstruction
    from psp.atomic_fourier_cache import (build_atomic_fourier_caches,
        write_atomic_fourier_caches, load_atomic_fourier_caches)

    root, output = args.manifest.resolve(), args.output.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "lorrax.isdf_augmentation.v1"
            or manifest.get("carrier") != "normalized_rkb"):
        raise ValueError("Fourier cache generation requires a normalized-RKB augmentation manifest")
    controls = manifest["fourier_cache"]
    output.mkdir(parents=True, exist_ok=False)
    receipts, species_files = {}, {}
    for z, entry in sorted(manifest["species"].items(), key=lambda item: int(item[0])):
        data = load_atomic_reconstruction(root / entry["reconstruction"], root / entry["source_upf"])
        if (str(int(z)) != z or int(z) <= 0
                or float(data["metadata"]["operator_comparison"]["source"]["atomic_number"]) != int(z)):
            raise ValueError("Fourier cache species label differs from authenticated source atomic number")
        start = time.perf_counter()
        caches = build_atomic_fourier_caches(data, controls)
        build_seconds = time.perf_counter() - start
        path = output / f"species_{z}_pauli_fourier.npz"
        start = time.perf_counter()
        metadata = write_atomic_fourier_caches(path, caches, data, controls)
        write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        load_atomic_fourier_caches(path, data, controls)
        read_seconds = time.perf_counter() - start
        receipts[z] = dict(metadata, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           build_seconds=build_seconds, write_seconds=write_seconds, read_seconds=read_seconds)
        species_files[z] = os.path.relpath(path, root)
        print(json.dumps({"species": z, "path": str(path), "build_seconds": build_seconds,
                          "read_seconds": read_seconds, "interpolation": metadata["interpolation"]}), flush=True)
    (output / "generation_receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (output / "manifest_fourier_cache_patch.json").write_text(json.dumps(
        {"fourier_cache": {**controls, "species_files": species_files}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
