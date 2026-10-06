#!/usr/bin/env python3
"""Prepare immutable normalized-RKB species caches from an augmentation manifest."""
from pathlib import Path
import argparse
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
    from psp.augmentation_cache import (build_normalized_cache, write_normalized_cache,
        load_normalized_cache, normalized_cache_tail_diagnostics)

    root, output = args.manifest.resolve(), args.output.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "lorrax.isdf_augmentation.v1"
            or manifest.get("carrier") != "normalized_rkb"):
        raise ValueError("normalized cache generation requires an authenticated augmentation manifest")
    control = manifest["cache"]
    radial = manifest["radial"]
    support = float(radial["support_radius"] if "support_radius" in radial else radial["r_max"])
    output.mkdir(parents=True, exist_ok=False)
    receipts, species_files = {}, {}
    for z, entry in sorted(manifest["species"].items(), key=lambda item: int(item[0])):
        data = load_atomic_reconstruction(root / entry["reconstruction"], root / entry["source_upf"])
        start = time.perf_counter()
        cache = build_normalized_cache(data, control, support_radius=support)
        build_seconds = time.perf_counter() - start
        path = output / f"species_{z}_normalized_rkb.npz"
        start = time.perf_counter()
        metadata = write_normalized_cache(path, cache, data, control, support_radius=support)
        write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        restored = load_normalized_cache(path, data, control, support_radius=support)
        read_seconds = time.perf_counter() - start
        tails = normalized_cache_tail_diagnostics(restored, support_radius=support)
        receipts[z] = dict(metadata, build_seconds=build_seconds, write_seconds=write_seconds,
                           read_seconds=read_seconds, tail_diagnostics=tails)
        species_files[z] = os.path.relpath(path, root)
        print(json.dumps({"species": z, "path": str(path), "build_seconds": build_seconds,
                          "read_seconds": read_seconds, "tail_diagnostics": tails}), flush=True)
    (output / "generation_receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (output / "manifest_cache_patch.json").write_text(json.dumps(
        {"cache": {"species_files": species_files}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
