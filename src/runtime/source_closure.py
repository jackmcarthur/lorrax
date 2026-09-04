"""Seal one coherent LORRAX application/service source closure.

This module is deliberately standard-library-only.  A core driver calls it
from :func:`runtime.bootstrap` before importing JAX, physics modules, or a
first-party service.  Two deployment forms are supported:

``source``
    ``runtime`` was imported from ``<checkout>/src``.  The checkout's
    ``pyproject.toml`` is the authority: uv workspace members name the service
    projects, each project names its source root, and the ``lorrax.services``
    entry-point group maps distributions to public import doors.  Those source
    roots are placed before ambient/site packages and their actual origins are
    verified without importing them.

``installed``
    ``runtime`` came from an installed LORRAX distribution.  The same entry-
    point group is read from installed metadata and each declared service
    distribution/import door must exist.  No source path is synthesized.

``LORRAX_CHECKOUT`` is a launcher request, not Python import machinery.  If it
is present, the already-imported runtime must be the runtime in that exact
checkout; otherwise startup refuses before JAX.  This catches the observed
failure in which a launcher announced one checkout while Python imported a
base module's older LORRAX tree.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib import machinery, metadata
import json
import os
from pathlib import Path
import re
import sys
import tomllib
from typing import Mapping, MutableSequence
from urllib.parse import unquote, urlsplit


SERVICE_ENTRYPOINT_GROUP = "lorrax.services"


class SourceClosureError(RuntimeError):
    """The selected application and service packages are not one closure."""


@dataclass(frozen=True)
class ServiceSpec:
    """One declared service distribution and its public import door."""

    distribution: str
    module: str
    source_dir: Path | None = None


@dataclass(frozen=True)
class SourceClosureReceipt:
    """The actual package origins accepted for this process."""

    mode: str
    root: str
    runtime: str
    services: tuple[tuple[str, str], ...]

    def line(self) -> str:
        payload = {
            "mode": self.mode,
            "root": self.root,
            "runtime": self.runtime,
            "services": dict(self.services),
        }
        return "[lorrax source closure] " + json.dumps(
            payload, sort_keys=True, separators=(",", ":"))


_RECEIPT: SourceClosureReceipt | None = None
_DIST_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SourceClosureError(
            f"cannot read package metadata {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _project_name(data: Mapping, *, path: Path) -> str:
    value = data.get("project", {}).get("name")
    if not isinstance(value, str) or not value.strip():
        raise SourceClosureError(f"package metadata {path} has no project.name")
    return value.strip()


def _runtime_source_root(runtime_init: Path) -> Path | None:
    """Return the lexical checkout root for an in-tree/editable runtime."""
    runtime_init = runtime_init.absolute()
    runtime_real = runtime_init.resolve()
    # Bounded parent inspection, never a recursive filesystem traversal.
    for parent in tuple(runtime_init.parents)[:6]:
        pyproject = parent / "pyproject.toml"
        candidate_runtime = parent / "src" / "runtime" / "__init__.py"
        if not pyproject.is_file() or not candidate_runtime.is_file():
            continue
        data = _read_toml(pyproject)
        if _canonical_distribution(_project_name(data, path=pyproject)) \
                != "lorrax":
            continue
        if candidate_runtime.resolve() == runtime_real:
            return parent
    return None


def _require_checkout_matches(
    source_root: Path | None,
    runtime_init: Path,
    environ: Mapping[str, str],
) -> None:
    requested = environ.get("LORRAX_CHECKOUT", "").strip()
    if not requested:
        return
    checkout = Path(requested).expanduser().resolve()
    expected_runtime = checkout / "src" / "runtime" / "__init__.py"
    if not expected_runtime.is_file():
        raise SourceClosureError(
            "LORRAX_CHECKOUT does not name a LORRAX checkout: "
            f"{checkout} has no src/runtime/__init__.py")
    actual = runtime_init.resolve()
    if source_root is None or expected_runtime.resolve() != actual:
        raise SourceClosureError(
            "LORRAX_CHECKOUT disagrees with the runtime Python actually "
            "imported; refusing a mixed-source launch. "
            f"requested={checkout} expected_runtime={expected_runtime.resolve()} "
            f"actual_runtime={actual}. Fix the launcher/source overlay; merely "
            "exporting LORRAX_CHECKOUT does not change Python imports.")


def _requirement_names(data: Mapping) -> set[str]:
    names: set[str] = set()
    for requirement in data.get("project", {}).get("dependencies", ()):
        if not isinstance(requirement, str):
            raise SourceClosureError("project.dependencies entries must be strings")
        match = _DIST_NAME.match(requirement)
        if match is None:
            raise SourceClosureError(
                f"cannot read distribution name from requirement {requirement!r}")
        names.add(_canonical_distribution(match.group(1)))
    return names


def _service_entrypoints(data: Mapping, *, path: Path) -> dict[str, str]:
    raw = (data.get("project", {}).get("entry-points", {})
           .get(SERVICE_ENTRYPOINT_GROUP, {}))
    if not isinstance(raw, dict) or not raw:
        raise SourceClosureError(
            f"{path} has no nonempty project.entry-points."
            f"{SERVICE_ENTRYPOINT_GROUP}")
    out: dict[str, str] = {}
    for distribution, value in raw.items():
        if not isinstance(distribution, str) or not isinstance(value, str):
            raise SourceClosureError(
                f"{path} service entry points must map strings to strings")
        module = value.partition(":")[0].strip()
        if not module or any(not part.isidentifier() for part in module.split(".")):
            raise SourceClosureError(
                f"{path} service {distribution!r} has invalid import door "
                f"{value!r}")
        key = _canonical_distribution(distribution)
        if key in out:
            raise SourceClosureError(
                f"{path} declares service distribution {distribution!r} twice")
        out[key] = module
    return out


def _workspace_member_dirs(root: Path, data: Mapping) -> tuple[Path, ...]:
    workspace = data.get("tool", {}).get("uv", {}).get("workspace", {})
    members = workspace.get("members", ())
    if not isinstance(members, list) or not members:
        raise SourceClosureError(
            f"{root / 'pyproject.toml'} has no uv workspace members")
    out: list[Path] = []
    for pattern in members:
        if not isinstance(pattern, str) or not pattern.strip():
            raise SourceClosureError("uv workspace member patterns must be strings")
        rel = Path(pattern)
        if rel.is_absolute() or ".." in rel.parts or "**" in pattern:
            raise SourceClosureError(
                f"unsafe uv workspace member pattern {pattern!r}; expected a "
                "bounded path or one-level glob inside the checkout")
        matches = sorted(path for path in root.glob(pattern) if path.is_dir())
        if not matches:
            raise SourceClosureError(
                f"uv workspace member pattern {pattern!r} matches no directory")
        escaped = [path for path in matches
                   if not _is_beneath(path, root)]
        if escaped:
            raise SourceClosureError(
                f"uv workspace member pattern {pattern!r} escapes the "
                f"selected checkout through a symlink: {escaped}")
        out.extend(matches)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in out:
        real = path.resolve()
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return tuple(unique)


def _source_dirs(project_dir: Path, data: Mapping) -> tuple[Path, ...]:
    raw = (data.get("tool", {}).get("setuptools", {})
           .get("packages", {}).get("find", {}).get("where", ["."]))
    if not isinstance(raw, list) or not raw:
        raise SourceClosureError(
            f"{project_dir / 'pyproject.toml'} packages.find.where must be a "
            "nonempty list")
    out = []
    for item in raw:
        if not isinstance(item, str):
            raise SourceClosureError("packages.find.where entries must be strings")
        rel = Path(item)
        if rel.is_absolute() or ".." in rel.parts:
            raise SourceClosureError(
                f"service source root {item!r} must stay inside {project_dir}")
        path = (project_dir / rel).absolute()
        if not path.is_dir():
            raise SourceClosureError(f"declared service source root is missing: {path}")
        if not _is_beneath(path, project_dir):
            raise SourceClosureError(
                f"declared service source root escapes its project through "
                f"a symlink: {path}")
        out.append(path)
    return tuple(out)


def source_service_specs(root: Path) -> tuple[ServiceSpec, ...]:
    """Derive and cross-check bare-source services from package metadata."""
    root = root.absolute()
    root_metadata = root / "pyproject.toml"
    data = _read_toml(root_metadata)
    entrypoints = _service_entrypoints(data, path=root_metadata)
    requirements = _requirement_names(data)
    uv_sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    if not isinstance(uv_sources, dict):
        raise SourceClosureError("tool.uv.sources must be a table")
    workspace_sources = {
        _canonical_distribution(name)
        for name, value in uv_sources.items()
        if isinstance(value, dict) and value.get("workspace") is True
    }

    projects: dict[str, tuple[Path, dict]] = {}
    for member in _workspace_member_dirs(root, data):
        child_metadata = member / "pyproject.toml"
        if not child_metadata.is_file():
            raise SourceClosureError(
                f"workspace member {member} has no pyproject.toml")
        child_data = _read_toml(child_metadata)
        name = _canonical_distribution(
            _project_name(child_data, path=child_metadata))
        if name in projects:
            raise SourceClosureError(
                f"workspace declares distribution {name!r} more than once")
        projects[name] = (member, child_data)

    declared = set(entrypoints)
    workspace = set(projects)
    if declared != workspace:
        raise SourceClosureError(
            "lorrax.services entry points and uv workspace projects disagree: "
            f"entrypoints={sorted(declared)} workspace={sorted(workspace)}")
    missing_requirements = declared - requirements
    missing_sources = declared - workspace_sources
    if missing_requirements or missing_sources:
        raise SourceClosureError(
            "service workspace metadata is incomplete: "
            f"missing_root_dependencies={sorted(missing_requirements)} "
            f"missing_uv_workspace_sources={sorted(missing_sources)}")

    specs = []
    for distribution in sorted(declared):
        module = entrypoints[distribution]
        project_dir, child_data = projects[distribution]
        candidates = []
        module_rel = Path(*module.split("."))
        for source_dir in _source_dirs(project_dir, child_data):
            if (source_dir / module_rel / "__init__.py").is_file() \
                    or (source_dir / module_rel).with_suffix(".py").is_file():
                candidates.append(source_dir)
        if len(candidates) != 1:
            raise SourceClosureError(
                f"service {distribution!r} import door {module!r} must resolve "
                "under exactly one declared packages.find.where root; "
                f"found={list(map(str, candidates))}")
        specs.append(ServiceSpec(distribution, module, candidates[0]))
    return tuple(specs)


def _module_origins(module) -> tuple[Path, ...]:
    paths = []
    filename = getattr(module, "__file__", None)
    if filename:
        paths.append(Path(filename).resolve())
    package_path = getattr(module, "__path__", None)
    if package_path is not None:
        paths.extend(Path(item).resolve() for item in package_path)
    return tuple(paths)


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _refuse_loaded_stale_service(
    spec: ServiceSpec,
    loaded_modules: Mapping[str, object],
) -> None:
    assert spec.source_dir is not None
    stale = []
    for name, module in tuple(loaded_modules.items()):
        if name != spec.module and not name.startswith(spec.module + "."):
            continue
        origins = _module_origins(module)
        if not origins or any(
                not _is_beneath(origin, spec.source_dir) for origin in origins):
            stale.append((name, tuple(map(str, origins)) or ("<unknown>",)))
    if stale:
        raise SourceClosureError(
            f"service {spec.module!r} was imported before source closure from "
            f"outside {spec.source_dir}: {stale}. Refusing to combine loaded "
            "stale classes/functions with the selected checkout.")


def _find_origin(module: str, search_path: MutableSequence[str]) -> Path:
    importlib.invalidate_caches()
    found = machinery.PathFinder.find_spec(module, list(search_path))
    if found is None:
        raise SourceClosureError(
            f"declared service import door {module!r} cannot be found")
    if found.origin is None:
        locations = tuple(found.submodule_search_locations or ())
        if len(locations) != 1:
            raise SourceClosureError(
                f"service {module!r} resolved as an ambiguous namespace "
                f"package: {locations}")
        return Path(locations[0]).resolve()
    return Path(found.origin).resolve()


def _display_origin(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            pass
    return str(path.resolve())


def _seal_source(
    root: Path,
    runtime_init: Path,
    search_path: MutableSequence[str],
    loaded_modules: Mapping[str, object],
) -> SourceClosureReceipt:
    specs = source_service_specs(root)
    for spec in specs:
        _refuse_loaded_stale_service(spec, loaded_modules)

    selected_real = {spec.source_dir.resolve() for spec in specs
                     if spec.source_dir is not None}
    retained = []
    for item in search_path:
        try:
            real = Path(item or os.curdir).absolute().resolve()
        except OSError:
            retained.append(item)
            continue
        # Remove exact and symlink-equivalent duplicates of selected service
        # roots, then insert each selected lexical path once at the front.
        if real not in selected_real:
            retained.append(item)
    search_path[:] = [str(spec.source_dir) for spec in specs] + retained

    accepted = []
    for spec in specs:
        assert spec.source_dir is not None
        origin = _find_origin(spec.module, search_path)
        if not _is_beneath(origin, spec.source_dir):
            raise SourceClosureError(
                f"service {spec.module!r} resolved outside the selected "
                f"checkout: expected_under={spec.source_dir} actual={origin}")
        accepted.append((spec.distribution, _display_origin(origin, root)))
    return SourceClosureReceipt(
        mode="source",
        root=str(root.resolve()),
        runtime=_display_origin(runtime_init, root),
        services=tuple(accepted),
    )


def _installed_service_specs() -> tuple[ServiceSpec, ...]:
    try:
        lorrax_dist = metadata.distribution("lorrax")
    except metadata.PackageNotFoundError as exc:
        raise SourceClosureError(
            "runtime is not inside a source checkout and no installed lorrax "
            "distribution metadata exists") from exc
    requirements = set()
    for requirement in lorrax_dist.requires or ():
        match = _DIST_NAME.match(requirement)
        if match is not None:
            requirements.add(_canonical_distribution(match.group(1)))
    specs = []
    seen = set()
    for entrypoint in lorrax_dist.entry_points:
        if entrypoint.group != SERVICE_ENTRYPOINT_GROUP:
            continue
        module = entrypoint.value.partition(":")[0].strip()
        if not module or any(not part.isidentifier()
                             for part in module.split(".")):
            raise SourceClosureError(
                f"installed lorrax service {entrypoint.name!r} has invalid "
                f"import door {entrypoint.value!r}")
        distribution = _canonical_distribution(entrypoint.name)
        if distribution in seen:
            raise SourceClosureError(
                f"installed lorrax declares service distribution "
                f"{distribution!r} twice")
        seen.add(distribution)
        specs.append(ServiceSpec(distribution, module, None))
    if not specs:
        raise SourceClosureError(
            f"installed lorrax metadata has no {SERVICE_ENTRYPOINT_GROUP} "
            "service manifest; reinstall lorrax and its declared dependencies")
    missing_requirements = seen - requirements
    if missing_requirements:
        raise SourceClosureError(
            "installed lorrax service manifest contains undeclared runtime "
            f"dependencies: {sorted(missing_requirements)}")
    return tuple(sorted(specs, key=lambda item: item.distribution))


def _distribution_owns_origin(dist, origin: Path) -> bool:
    """Whether installed distribution metadata binds ``origin`` to ``dist``.

    A top-level-name/provider match is insufficient: two distributions can
    advertise the same import door and path precedence can select the stale
    one.  Wheel ``RECORD`` entries are exact ownership evidence.  PEP 610's
    ``direct_url.json`` supplies the equivalent root for an editable install.
    """
    wanted = origin.resolve()
    for item in dist.files or ():
        try:
            if Path(dist.locate_file(item)).resolve() == wanted:
                return True
        except OSError:
            continue
    raw_direct_url = dist.read_text("direct_url.json")
    if raw_direct_url:
        try:
            url = json.loads(raw_direct_url).get("url", "")
            parsed = urlsplit(url)
            if parsed.scheme == "file":
                project_root = Path(unquote(parsed.path)).resolve()
                return _is_beneath(wanted, project_root)
        except (AttributeError, json.JSONDecodeError, OSError, TypeError):
            pass
    return False


def _seal_installed(
    runtime_init: Path,
    search_path: MutableSequence[str],
) -> SourceClosureReceipt:
    specs = _installed_service_specs()
    providers = metadata.packages_distributions()
    accepted = []
    for spec in specs:
        try:
            service_dist = metadata.distribution(spec.distribution)
        except metadata.PackageNotFoundError as exc:
            raise SourceClosureError(
                f"installed lorrax requires service distribution "
                f"{spec.distribution!r}, but it is not installed") from exc
        origin = _find_origin(spec.module, search_path)
        top_level = spec.module.split(".", 1)[0]
        named_providers = {
            _canonical_distribution(item)
            for item in providers.get(top_level, ())
        }
        if named_providers and spec.distribution not in named_providers:
            raise SourceClosureError(
                f"installed service door {spec.module!r} is provided by "
                f"{sorted(named_providers)}, not declared distribution "
                f"{spec.distribution!r}; actual={origin}")
        if not _distribution_owns_origin(service_dist, origin):
            raise SourceClosureError(
                f"installed service door {spec.module!r} resolved to "
                f"{origin}, but distribution {spec.distribution!r} does not "
                "own that file according to RECORD/direct_url metadata; "
                "refusing a duplicate-provider or stale-path launch")
        accepted.append((spec.distribution, str(origin)))
    try:
        dist_root = str(metadata.distribution("lorrax").locate_file("").resolve())
    except (AttributeError, OSError):
        dist_root = "<installed>"
    return SourceClosureReceipt(
        mode="installed",
        root=dist_root,
        runtime=str(runtime_init.resolve()),
        services=tuple(accepted),
    )


def ensure_source_closure(
    *,
    print_fn=print,
    runtime_file: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    search_path: MutableSequence[str] | None = None,
    loaded_modules: Mapping[str, object] | None = None,
) -> SourceClosureReceipt:
    """Seal, report, and return this process's first-party source closure.

    Optional arguments are test seams.  The production call supplies only
    ``print_fn`` and therefore mutates the real ``sys.path`` exactly once.
    """
    global _RECEIPT
    production_call = (runtime_file is None and environ is None
                       and search_path is None and loaded_modules is None)
    if production_call and _RECEIPT is not None:
        return _RECEIPT

    runtime_init = Path(runtime_file) if runtime_file is not None else Path(
        __file__).with_name("__init__.py")
    env = os.environ if environ is None else environ
    paths = sys.path if search_path is None else search_path
    modules = sys.modules if loaded_modules is None else loaded_modules
    source_root = _runtime_source_root(runtime_init)
    _require_checkout_matches(source_root, runtime_init, env)
    receipt = (_seal_source(source_root, runtime_init, paths, modules)
               if source_root is not None
               else _seal_installed(runtime_init, paths))
    print_fn(receipt.line())
    if production_call:
        _RECEIPT = receipt
    return receipt


__all__ = [
    "SERVICE_ENTRYPOINT_GROUP",
    "ServiceSpec",
    "SourceClosureError",
    "SourceClosureReceipt",
    "ensure_source_closure",
    "source_service_specs",
]
