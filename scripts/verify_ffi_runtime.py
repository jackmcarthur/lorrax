#!/usr/bin/env python3
"""Attest a ROCm leg's recorded closure in a fresh process, without JAX init.

Unlike ``ldd``, this uses the same exact-file preloads and live mapping/ABI
checks as the runtime.  The tab-separated SONAME/path receipt is written only
after acceptance; the shell verifier uses it for its platform-specific gates.
This is artifact acceptance, not a substitute for a compute-node GPU test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "services" / "lxkit" / "src"))
from lxkit import LEGS, native_provider  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--abi", type=int, required=True)
    parser.add_argument("--map-file", type=Path, required=True)
    args = parser.parse_args(argv)
    # A failed/retried check must not leave an earlier successful receipt.
    args.map_file.unlink(missing_ok=True)
    try:
        native_provider.open_and_attest(
            args.library, platform="ROCM", expected_abi=args.abi,
            abi_symbols={key: leg.abi_symbol for key, leg in LEGS.items()},
            build_hint="bash config/frontier/build_ffi_rocm.sh --fresh",
            strict_unstamped=True)
        mapped = native_provider._mapped_sonames()
        args.map_file.write_text("".join(
            f"{soname}\t{path}\n"
            for path, soname in sorted(mapped.items(), key=lambda item: str(item[0]))))
    except (OSError, ValueError) as exc:
        print(f"[verify_ffi_runtime] REFUSED: {exc}", file=sys.stderr)
        return 1
    print("[verify_ffi_runtime] recorded closure, live mappings and ABI accepted; "
          "JAX/GPU execution remains a separate compute-node gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
