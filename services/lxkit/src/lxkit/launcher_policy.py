"""Pure policy shared by the source-tested and deployed ``lx`` launcher.

The deployed front door runs before the LORRAX environment exists, so this
module is standard-library-only.  It owns decisions that otherwise drift
between ``lx``, its pool helper, and deck-doctor tests: allocation pin
spelling, newest-first ordering, per-node GPU geometry, persistent-cache
intent, and checkout selection.  Slurm probes and command execution remain in
the deployed launcher.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence, TypeVar


class LauncherPolicyError(ValueError):
    """A launcher request is ambiguous or cannot describe the target site."""

    def __init__(self, rule: str, got: str, want: str, fix: str):
        super().__init__(f"{rule}: {got}; want {want}; fix: {fix}")
        self.rule = rule
        self.got = got
        self.want = want
        self.fix = fix


@dataclass(frozen=True)
class AllocationPin:
    """One validated allocation choice, or the automatic-pool request."""

    jid: str | None
    source: str


@dataclass(frozen=True)
class LaunchGeometry:
    """Requested Slurm geometry, with ``gpus`` explicitly per node."""

    nodes: int
    gpus_per_node: int
    ranks: int
    site_gpus_per_node: int

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node


def _jid(value: str | int | None, spelling: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if not text.isdigit() or int(text) <= 0:
        raise LauncherPolicyError(
            "LX-BADJID",
            f"{spelling}={value!r} is not a positive numeric Slurm job id",
            "one positive allocation job id",
            f"remove {spelling}, or set it to the numeric JID shown by squeue",
        )
    return text


def resolve_allocation_pin(
    cli_jid: str | int | None,
    environ: Mapping[str, str] | None = None,
) -> AllocationPin:
    """Resolve ``--jid`` or one inherited Slurm spelling without guessing.

    ``SLURM_JOBID`` is the supported environment spelling.  The historical
    ``SLURM_JOB_ID`` spelling remains accepted for already-running wrappers,
    but two different values refuse.  A CLI pin must agree with an inherited
    pin so a stale shell cannot silently redirect an explicit request.
    """
    env = os.environ if environ is None else environ
    cli = _jid(cli_jid, "--jid")
    canonical = _jid(env.get("SLURM_JOBID"), "SLURM_JOBID")
    legacy = _jid(env.get("SLURM_JOB_ID"), "SLURM_JOB_ID")
    if canonical and legacy and canonical != legacy:
        raise LauncherPolicyError(
            "LX-JID-CONFLICT",
            f"SLURM_JOBID={canonical} but SLURM_JOB_ID={legacy}",
            "at most one allocation id",
            "unset the stale spelling and use `lx run --jid <n>` or "
            "SLURM_JOBID alone",
        )
    inherited = canonical or legacy
    if cli and inherited and cli != inherited:
        raise LauncherPolicyError(
            "LX-JID-CONFLICT",
            f"--jid={cli} but the inherited Slurm job id is {inherited}",
            "one allocation id",
            "unset the inherited Slurm variable, or make --jid agree",
        )
    if cli:
        return AllocationPin(cli, "--jid")
    if canonical:
        return AllocationPin(canonical, "SLURM_JOBID")
    if legacy:
        return AllocationPin(legacy, "SLURM_JOB_ID compatibility")
    return AllocationPin(None, "newest fitting lx-alloc pool")


def export_allocation_pin(environ: MutableMapping[str, str], jid: str) -> None:
    """Publish both internal spellings for old helpers and payload wrappers."""
    value = _jid(jid, "jid")
    assert value is not None
    environ["SLURM_JOBID"] = value
    environ["SLURM_JOB_ID"] = value


def apply_cache_policy(environ: MutableMapping[str, str]) -> str:
    """Default persistent reuse off; preserve every explicit user request."""
    if "ISDF_JAX_CACHE_DIR" not in environ:
        environ["ISDF_JAX_CACHE_DIR"] = ""
        return "default cold"
    if not environ["ISDF_JAX_CACHE_DIR"].strip():
        return "explicit cold"
    return "explicit warm"


def validate_geometry(
    nodes: int,
    gpus_per_node: int,
    ranks: int,
    *,
    site_gpus_per_node: int = 4,
) -> LaunchGeometry:
    """Validate launch geometry while naming that ``-G`` is per node."""
    if nodes < 1:
        raise LauncherPolicyError(
            "LX-BADGEOMETRY", f"-N {nodes}", "at least one node",
            "set -N to a positive node count")
    if site_gpus_per_node < 1:
        raise LauncherPolicyError(
            "LX-BADGEOMETRY", f"site GPU capacity is {site_gpus_per_node}",
            "a positive site GPU capacity", "run `lx doctor --refresh`")
    if gpus_per_node < 0:
        raise LauncherPolicyError(
            "LX-BADGEOMETRY", f"-G {gpus_per_node}",
            "a nonnegative GPU count per node", "use -G 0 through -G 4")
    if gpus_per_node > site_gpus_per_node:
        raise LauncherPolicyError(
            "LX-GPUS-PER-NODE",
            f"-N {nodes} -G {gpus_per_node} asks for {gpus_per_node} GPUs "
            f"on each node, but this site has {site_gpus_per_node}",
            f"at most -G {site_gpus_per_node}; -G is per node",
            f"for {nodes * site_gpus_per_node} GPUs use -N {nodes} "
            f"-G {site_gpus_per_node} -n {nodes * site_gpus_per_node}",
        )
    if ranks < 1:
        raise LauncherPolicyError(
            "LX-BADGEOMETRY", f"-n {ranks}", "at least one rank",
            "set -n to a positive rank count, or omit it for nodes*GPUs")
    return LaunchGeometry(nodes, gpus_per_node, ranks, site_gpus_per_node)


def square_mesh(ranks: int) -> tuple[int, int]:
    """Return LORRAX's square process mesh or refuse a non-square rank count."""
    side = math.isqrt(int(ranks))
    if side * side != int(ranks):
        raise LauncherPolicyError(
            "LX-NONSQUARE-MESH", f"-n {ranks} gives no square process mesh",
            "a perfect-square rank count",
            "use one of 1, 4, 9, 16, 25, 36, ... ranks",
        )
    return side, side


_T = TypeVar("_T")


def newest_first(allocations: Sequence[_T]) -> list[_T]:
    """Return allocation records by descending numeric ``.jobid``."""
    return sorted(allocations, key=lambda item: int(item.jobid), reverse=True)


def _checkout_containing(start: Path) -> Path | None:
    """Return the bounded parent checkout containing ``start``."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if ((candidate / "pyproject.toml").is_file()
                and (candidate / "src" / "runtime" / "__init__.py").is_file()
                and (candidate / "src" / "gw" / "__init__.py").is_file()):
            return candidate
    return None


def select_source_root(
    cwd: str | os.PathLike[str],
    module_root: str | os.PathLike[str],
    requested: str | os.PathLike[str] | None = None,
) -> tuple[Path, str]:
    """Select the cwd checkout, preserving ``LORRAX_CHECKOUT`` compatibility."""
    if requested is not None and str(requested).strip():
        root = Path(requested).expanduser().resolve()
        reason = "LORRAX_CHECKOUT compatibility"
    else:
        root = _checkout_containing(Path(cwd))
        if root is not None:
            reason = "cwd checkout"
        else:
            root = Path(module_root).expanduser().resolve()
            reason = "site runtime (cwd is not in a checkout)"
    runtime = root / "src" / "runtime" / "__init__.py"
    if not runtime.is_file():
        raise LauncherPolicyError(
            "LX-BADCHECKOUT", f"{root} has no src/runtime/__init__.py",
            "a LORRAX checkout containing the runtime package",
            "run lx from the intended checkout, or repair the installed "
            "site runtime with `lx doctor --refresh`",
        )
    return root, reason


def runtime_origin(source_root: str | os.PathLike[str]) -> Path:
    """The runtime module path the launcher's PYTHONPATH head selects."""
    return (Path(source_root).resolve() / "src" / "runtime" / "__init__.py")


__all__ = [
    "AllocationPin", "LaunchGeometry", "LauncherPolicyError",
    "apply_cache_policy", "export_allocation_pin", "newest_first",
    "resolve_allocation_pin", "runtime_origin", "select_source_root",
    "square_mesh", "validate_geometry",
]
