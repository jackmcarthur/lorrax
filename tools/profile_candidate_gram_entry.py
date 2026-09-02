#!/usr/bin/env python3
"""Run a pinned candidate-Gram probe and record its actual tile width.

The science probe owns candidate/WFN construction. This small wrapper pins
that immutable file by SHA-256 and instruments the production tiled-kernel
call itself, avoiding inference from human log text. It also requires the two
warmups and measurement to select one stable execution shape.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_tiled_recorder(owner, widths: list[int], extents: list[int]):
    """Patch the public ISDF owner and return its original callable."""
    original = owner.gram_q0_tiled_from_psi_sm

    def _recording_tiled_gram(*args, **kwargs):
        widths.append(int(kwargs["tile_width"]))
        extents.append(int(args[0].shape[0]))
        return original(*args, **kwargs)

    owner.gram_q0_tiled_from_psi_sm = _recording_tiled_gram
    return original


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-sha256", required=True)
    owned, remainder = parser.parse_known_args()
    probe = owned.probe.resolve()
    observed_sha = _sha256(probe)
    if observed_sha != owned.probe_sha256:
        raise RuntimeError(
            "PROFILE_REFUSAL: probe digest mismatch: "
            f"{observed_sha} != {owned.probe_sha256}")

    spec = importlib.util.spec_from_file_location(
        "_pinned_candidate_gram_probe", probe)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"PROFILE_REFUSAL: cannot import probe {probe}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import isdf

    selected_widths: list[int] = []
    destination_extents: list[int] = []
    original = _install_tiled_recorder(
        isdf, selected_widths, destination_extents)
    old_argv = sys.argv
    sys.argv = [str(probe), *remainder]
    try:
        probe_args = module._arguments()
        module.main()
    finally:
        sys.argv = old_argv
        isdf.gram_q0_tiled_from_psi_sm = original

    expected_calls = int(probe_args.warmups) + int(probe_args.measurements)
    if selected_widths and (
            len(selected_widths) != expected_calls
            or len(set(selected_widths)) != 1
            or len(set(destination_extents)) != 1):
        raise RuntimeError(
            "PROFILE_REFUSAL: tiled execution shape changed across warmups/"
            f"measurement: widths={selected_widths}, "
            f"extents={destination_extents}, expected calls={expected_calls}")

    import jax

    rank = int(jax.process_index())
    root = probe_args.artifact_root.resolve()
    timing_path = root / f"timing.rank{rank:04d}.json"
    payload = json.loads(timing_path.read_text(encoding="utf-8"))
    route_path = str(payload["route_path"])
    if route_path.endswith("q0_sum.fused") and not selected_widths:
        raise RuntimeError(
            "PROFILE_REFUSAL: fused route executed without a recorded width")
    if route_path.endswith("q0_sum.sequential") and selected_widths:
        raise RuntimeError(
            "PROFILE_REFUSAL: sequential route unexpectedly called tiled Gram")

    width = selected_widths[0] if selected_widths else None
    M = int(payload["candidate_pool"]["M"])
    executed_M = destination_extents[0] if destination_extents else M
    tiles_per_axis = (
        None if width is None else (executed_M + width - 1) // width)
    payload["probe"] = {"path": str(probe), "sha256": observed_sha}
    payload["selected_tile_widths"] = selected_widths
    payload["selected_tile_width"] = width
    payload["destination_extents"] = destination_extents
    payload["executed_candidate_extent"] = executed_M
    payload["tiles_per_axis"] = tiles_per_axis
    payload["expected_pair_gemms_per_rank"] = (
        None if tiles_per_axis is None else 2 * tiles_per_axis ** 2)
    timing_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if rank == 0:
        (root / "route_receipt.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "PROFILE_WIDTH_RECEIPT "
        f"rank={rank} width={width} tiles={tiles_per_axis} "
        f"calls={selected_widths} probe_sha256={observed_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
