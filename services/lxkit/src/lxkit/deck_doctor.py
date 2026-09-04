"""Login-safe preflight for one LORRAX GW input deck.

The deployed ``lx`` front door runs this module in a one-rank step.  Its
default step owns no GPU and forces the CPU JAX backend; ``--gpu`` instead
uses one GPU to add live device and provider evidence.  The requested
science geometry is inspected, never allocated by the doctor step.

Imports of LORRAX, JAX, NumPy and h5py are deliberately function-local.
Importing :mod:`lxkit.deck_doctor` therefore preserves lxkit's standard-
library-only package contract and keeps the pure formatting helpers unit
testable on a login node.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


class DeckDoctorError(RuntimeError):
    """One named preflight contract was not satisfied."""

    def __init__(self, rule: str, got: str, want: str, fix: str):
        super().__init__(f"{rule}: {got}; want {want}; fix: {fix}")
        self.rule = rule
        self.got = got
        self.want = want
        self.fix = fix


@dataclass(frozen=True)
class InputPath:
    """One file the resolved deck will consume."""

    role: str
    path: Path


def _fail(rule: str, got: str, want: str, fix: str) -> None:
    raise DeckDoctorError(rule, got, want, fix)


def _git_receipt(source_root: Path) -> tuple[str, bool]:
    """Return the selected checkout's full commit and tracked-dirty bit."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.STDOUT,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain",
             "--untracked-files=no"],
            text=True, stderr=subprocess.STDOUT,
        ).strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(
            "LX-DECK-SOURCE-GIT",
            f"cannot identify {source_root}: {exc}",
            "a readable LORRAX Git checkout",
            "run lx from the intended checkout or repair the installed site tree",
        )
    return commit, dirty


def _resolve_beside(deck: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (deck.parent / path).resolve()


def required_input_paths(config, deck: Path, *, n_rmu: int) -> tuple[InputPath, ...]:
    """Return every file path consumed by this resolved GW configuration.

    Output paths are intentionally absent.  Optional inputs appear only when
    the resolved mode selects their consumer; for example, a one-shot run
    does not need the default-named parallel-transport artifact.
    """
    rows = [
        InputPath("DFT wavefunctions", Path(config.paths.wfn_file).resolve()),
        InputPath("ISDF centroids", Path(config.paths.centroids_file).resolve()),
        InputPath("mean-field Hamiltonian",
                  Path(config.paths.kin_ion_file).resolve()),
    ]
    current = getattr(config.paths, "centroids_file_current", None)
    if current:
        rows.append(InputPath("current centroids", _resolve_beside(deck, current)))

    correction = getattr(getattr(config, "head", None), "correction", "off")
    correction = str(getattr(correction, "value", correction)).lower()
    if correction != "off":
        source = str(getattr(config.head, "wcoul0_source", "s_tensor")).lower()
        if source == "epshead":
            rows.append(InputPath("long-wave epsilon head",
                                  (deck.parent / "eps0mat.h5").resolve()))
        else:
            rows.append(InputPath("long-wave dipoles",
                                  (deck.parent / "dipole.h5").resolve()))

    solver = getattr(config, "qp_solver", "")
    solver = str(getattr(solver, "value", solver)).lower()
    head_update = str(getattr(getattr(config, "sc", None),
                              "head_update", "off")).lower()
    if solver == "self_consistent" and head_update != "off":
        rows.append(InputPath(
            "parallel transport",
            _resolve_beside(deck, config.paths.parallel_transport_file),
        ))

    hall = str(getattr(config.paths, "static_gauge_hall_file", "")).strip()
    if hall:
        rows.append(InputPath("static Hall response", _resolve_beside(deck, hall)))

    if bool(getattr(config.head, "use_bgw_vcoul", False)):
        bgw = getattr(config.head, "bgw_vcoul_file", None)
        if bgw:
            rows.append(InputPath("BerkeleyGW Coulomb matrix",
                                  _resolve_beside(deck, bgw)))
        sym_wfn = getattr(config.head, "bgw_vcoul_sym_wfn", None)
        if sym_wfn:
            rows.append(InputPath("BerkeleyGW Coulomb symmetry WFN",
                                  _resolve_beside(deck, sym_wfn)))

    fit_reuse = getattr(getattr(config, "mpa", None), "fit_reuse_file", None)
    if fit_reuse:
        rows.append(InputPath("authenticated MPA fit reuse",
                              _resolve_beside(deck, fit_reuse)))

    if bool(getattr(config, "restart", False)):
        rows.append(InputPath(
            "ISDF restart tensors",
            (deck.parent / "tmp" / f"isdf_tensors_{int(n_rmu)}.h5").resolve(),
        ))
    return tuple(rows)


def _print_refusal(exc: DeckDoctorError) -> None:
    print(f"DECK_DOCTOR_REFUSED {exc.rule}", file=sys.stderr)
    print(f"  got  : {exc.got}", file=sys.stderr)
    print(f"  want : {exc.want}", file=sys.stderr)
    print(f"  fix  : {exc.fix}", file=sys.stderr)


def _source_root_and_runtime(source_root: Path, *, gpu: bool):
    """Import and bootstrap exactly the runtime selected by the launcher."""
    expected = (source_root / "src" / "runtime" / "__init__.py").resolve()
    if not expected.is_file():
        _fail(
            "LX-DECK-SOURCE",
            f"{source_root} has no src/runtime/__init__.py",
            "the source tree selected and printed by lx",
            "run from the intended checkout or run `lx doctor --refresh`",
        )
    source = str(source_root / "src")
    if sys.path[0] != source:
        sys.path.insert(0, source)
    import runtime

    actual = Path(runtime.__file__).resolve()
    if actual != expected:
        _fail(
            "LX-DECK-SOURCE-SEAL",
            f"runtime imported from {actual}",
            f"runtime imported from {expected}",
            "remove an ambient PYTHONPATH override; lx must put the selected "
            "checkout first",
        )
    runtime.bootstrap(platform="gpu" if gpu else "cpu")
    return runtime, actual


def _device_lines(*, gpu: bool) -> list[str]:
    if not gpu:
        return [
            "DEVICE_EXPECTED platform=Perlmutter NVIDIA A100; HBM class is "
            "allocation-specific (40/80 GB); use --gpu to measure it",
            "BACKEND_PROBE skipped (zero-GPU doctor; add --gpu)",
        ]
    import jax

    devices = jax.local_devices()
    if not devices:
        _fail(
            "LX-DECK-NOGPU", "JAX returned no local devices",
            "one visible CUDA device", "retry `lx doctor --deck ... --gpu`",
        )
    kinds = sorted({str(getattr(device, "device_kind", "unknown"))
                    for device in devices})
    memory = []
    for device in devices:
        stats = device.memory_stats() if hasattr(device, "memory_stats") else None
        limit = (stats or {}).get("bytes_limit", 0)
        memory.append(f"{float(limit) / 1e9:.2f} GB" if limit else "unknown")
    return [
        f"DEVICE_ACTUAL backend={jax.default_backend()} kind={','.join(kinds)} "
        f"memory_limit={','.join(memory)}",
        "BACKEND_PROBE live GPU backend initialized; detailed providers are "
        "reported by runtime.bootstrap/source capability gates",
    ]


def inspect_deck(args) -> None:
    """Run the source, deck, artifact and geometry preflights."""
    deck = Path(args.deck).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    if not deck.is_file():
        _fail(
            "LX-DECK-MISSING", f"deck={deck} is not a file",
            "one readable cohsex.in", "correct the --deck path",
        )

    _, actual_runtime = _source_root_and_runtime(source_root, gpu=args.gpu)
    commit, dirty = _git_receipt(source_root)
    print(f"SOURCE_ROOT {source_root}")
    print(f"SOURCE_COMMIT {commit} dirty_tracked={str(dirty).lower()}")
    print(f"IMPORT_ORIGIN runtime {actual_runtime}")

    from gw.gw_config import (
        LorraxConfig,
        infer_material_class,
        validate_material_inputs,
    )

    # ``resolve_hardware=False`` still constructs the complete typed config
    # and runs all cross-key validators.  It leaves only an auto memory
    # budget unresolved and treats explicit CUDA providers as requests for
    # the expected GPU execution platform, which is exactly the question a
    # zero-GPU preflight can answer honestly.
    config = LorraxConfig.from_input_file(
        str(deck),
        strict_keys=True,
        runtime_platform=None if args.gpu else "gpu",
        resolve_hardware=args.gpu,
    )
    print(f"DECK_PARSE_OK strict_keys=true deck={deck}")

    from file_io.centroids import load_centroids
    from file_io.mf_header import read_mf_header

    header = read_mf_header(config.paths.wfn_file)
    material = infer_material_class(header.occs)
    validate_material_inputs(config, material)
    _, _, n_rmu = load_centroids(config.paths.centroids_file, header.fft_grid)
    print(f"MATERIAL_CLASS {material} source=WFN occupations")

    rows = required_input_paths(config, deck, n_rmu=n_rmu)
    missing = []
    for row in rows:
        exists = row.path.is_file()
        print(f"INPUT_FILE {'OK' if exists else 'MISSING'} "
              f"role={row.role!r} path={row.path}")
        if not exists:
            missing.append(row)
    if missing:
        _fail(
            "LX-DECK-INPUT-MISSING",
            ", ".join(f"{row.role}={row.path}" for row in missing),
            "every file selected by the resolved deck",
            "create or relink the named input artifacts beside the deck",
        )

    print(f"CONFIG_MODE compute_mode={config.compute_mode.value} "
          f"qp_solver={config.qp_solver.value} bispinor={config.bispinor}")
    memory = config.memory
    if float(memory.per_device_gb) > 0.0:
        memory_value = f"{float(memory.per_device_gb):.2f} GB/device"
    else:
        memory_value = "auto at GPU startup"
    print("MEMORY_DIALS "
          f"per_device={memory_value}; chunk_utilization="
          f"{memory.chunk_target_utilization:g}; "
          f"band={memory.band_chunk_size}; r={memory.r_chunk_override}; "
          f"gflat={memory.gflat_chunk_size}; vq_g={memory.vq_g_chunk_size}; "
          f"low_mem_bands={memory.low_mem_bands}")
    print(f"LINALG_DIALS {config.backend.summary()}")

    from lxkit.launcher_policy import square_mesh, validate_geometry
    validate_geometry(args.nodes, args.gpus_per_node, args.ranks,
                      site_gpus_per_node=args.site_gpus_per_node)
    px, py = square_mesh(args.ranks)
    sc_buffer = (int(config.sc.buffer_nbands)
                 if config.qp_solver.value == "self_consistent" else 0)
    nelec = (int(header.ifmax.max()) if header.ifmax.size
             else int((header.occs[0, 0] > 0.5).sum()))
    sigma_window = nelec + int(config.ncond) + sc_buffer
    divides = sigma_window % px == 0 and sigma_window % py == 0
    print(f"GEOMETRY nodes={args.nodes} gpus_per_node={args.gpus_per_node} "
          f"ranks={args.ranks} mesh={px}x{py}")
    print(f"SIGMA_WINDOW bands={sigma_window} nelec={nelec} "
          f"ncond={config.ncond} sc_buffer={sc_buffer} "
          f"divides_mesh={str(divides).lower()}")
    if config.compute_mode.is_dynamic and not divides:
        _fail(
            "LX-DECK-SIGMA-MESH",
            f"dynamic Sigma window {sigma_window} does not divide {px}x{py}",
            "the Sigma band window divisible by both mesh axes",
            "choose a fitting square rank count or adjust the physical band window",
        )

    for line in _device_lines(gpu=args.gpu):
        print(line)
    print("DECK_DOCTOR_OK")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lx doctor --deck")
    parser.add_argument("--deck", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--site-gpus-per-node", type=int, default=4)
    parser.add_argument("--gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inspect_deck(args)
    except DeckDoctorError as exc:
        _print_refusal(exc)
        return 2
    except Exception as exc:  # the source validator owns its detailed text
        _print_refusal(DeckDoctorError(
            "LX-DECK-VALIDATION",
            f"{type(exc).__name__}: {exc}",
            "a strict, internally consistent deck and authenticated inputs",
            "apply the source validator's named fix above, then rerun doctor",
        ))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
