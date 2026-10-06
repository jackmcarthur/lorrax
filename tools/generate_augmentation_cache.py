#!/usr/bin/env python3
"""Prepare source-bound compact normalized-RKB caches from a manifest.

The cache control must give taper_start, retaining every native atomic sphere.
The unwindowed Hankel arrays are retained as evidence; the declared served
field derives its small component from the tapered large Hermite polynomial.
"""
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
    parser.add_argument("--served-moments", action="store_true",
        help="also prepare exact compact-field B and Fourier bra tables for subsequent raw-parent preparation")
    parser.add_argument("--served-maximum-wavevector", type=float, default=16.,
        help="prepared served Fourier interval in bohr^-1; the raw-parent producer refuses source momenta outside it")
    parser.add_argument("--served-fourier-points", type=int, default=4097,
        help="served Fourier nodes, subject to the unchanged direct-quadrature guard")
    args = parser.parse_args()
    from runtime import bootstrap
    bootstrap(platform="cpu")
    from psp.atomic_reconstruction import load_atomic_reconstruction
    from psp.augmentation_cache import (build_normalized_cache, write_normalized_cache,
        load_normalized_cache, normalized_cache_tail_diagnostics)
    from gw.isdf_augmentation import _normalized_cache_control

    root, output = args.manifest.resolve(), args.output.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema") != "lorrax.isdf_augmentation.v1"
            or manifest.get("carrier") != "normalized_rkb"):
        raise ValueError("normalized cache generation requires an authenticated augmentation manifest")
    control = _normalized_cache_control(manifest["cache"])
    radial = manifest["radial"]
    support = float(radial["support_radius"] if "support_radius" in radial else radial["r_max"])
    output.mkdir(parents=True, exist_ok=False)
    receipts, species_files, served_files, served_sha = {}, {}, {}, {}
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
        if args.served_moments:
            from isdf.atomic_moments import (build_served_overlap_cache, write_served_moment_cache,
                                             load_served_moment_cache)
            start = time.perf_counter()
            served = build_served_overlap_cache(restored, support_radius=support,
                momentum_max=args.served_maximum_wavevector, momentum_points=args.served_fourier_points)
            served_build_seconds = time.perf_counter()-start
            served_path = output/f"species_{z}_served_moment.npz"
            norm_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            served_metadata = write_served_moment_cache(served_path, served, normalized_cache_sha256=norm_sha)
            start = time.perf_counter()
            load_served_moment_cache(served_path, normalized_cache_sha256=norm_sha, support_radius=support)
            served_read_seconds = time.perf_counter()-start
            served_files[z] = os.path.relpath(served_path, root)
            served_sha[z] = hashlib.sha256(served_path.read_bytes()).hexdigest()
            receipts[z]['served_moments'] = dict(served_metadata, build_seconds=served_build_seconds,
                read_seconds=served_read_seconds, file_sha256=served_sha[z])
        print(json.dumps({"species": z, "path": str(path), "build_seconds": build_seconds,
                          "read_seconds": read_seconds, "tail_diagnostics": tails}), flush=True)
    (output / "generation_receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
    patch = {"cache": {"species_files": species_files}}
    if args.served_moments:
        # This is an offline preparation patch. The WFN-dependent producer
        # subsequently adds raw_parent_file/raw_parent_sha256; a fitting
        # manifest refuses an incomplete served_moments entry.
        patch['served_moments'] = dict(species_files=served_files, species_sha256=served_sha)
    (output / "manifest_cache_patch.json").write_text(json.dumps(patch, indent=2) + "\n")


if __name__ == "__main__":
    main()
