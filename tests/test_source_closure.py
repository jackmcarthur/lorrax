"""Source/package closure gates for bare and installed LORRAX launches.

These tests are intentionally login-safe: ``runtime.source_closure`` is
stdlib-only, and none of the service doors is imported while its origin is
checked.  The hostile arms reproduce the launcher class seen in the Bi run:
one checkout requested while a stale first-party package remains importable.
"""
from __future__ import annotations

import ast
from importlib import machinery
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import types

import pytest

from runtime.source_closure import (
    SourceClosureError,
    ensure_source_closure,
    source_service_specs,
)


_REPO = Path(__file__).resolve().parents[1]


def _make_source_tree(root: Path, service: str = "demo_service") -> Path:
    runtime = root / "src" / "runtime"
    package = root / "services" / service / "src" / service
    runtime.mkdir(parents=True)
    package.mkdir(parents=True)
    (runtime / "__init__.py").write_text("# runtime\n")
    (package / "__init__.py").write_text("ORIGIN = 'selected'\n")
    (root / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'lorrax'\n"
        f"dependencies = ['{service}==0.1.0']\n"
        "[project.entry-points.'lorrax.services']\n"
        f"{service} = '{service}'\n"
        "[tool.uv.sources]\n"
        f"{service} = {{ workspace = true }}\n"
        "[tool.uv.workspace]\n"
        "members = ['services/*']\n")
    service_root = root / "services" / service
    (service_root / "pyproject.toml").write_text(
        "[project]\n"
        f"name = '{service}'\n"
        "version = '0.1.0'\n"
        "[tool.setuptools.packages.find]\n"
        "where = ['src']\n")
    return runtime / "__init__.py"


def _make_stale_service(root: Path, service: str = "demo_service") -> Path:
    package = root / service
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("ORIGIN = 'stale'\n")
    return root


def _write_record(dist_info: Path, *files: str) -> None:
    (dist_info / "RECORD").write_text(
        "".join(f"{item},,\n" for item in files))


def test_root_metadata_declares_every_workspace_service_as_one_closure():
    """The service roster is derived from child package metadata, not copied."""
    specs = source_service_specs(_REPO)
    with (_REPO / "pyproject.toml").open("rb") as handle:
        root_data = tomllib.load(handle)
    expected = set()
    for child_metadata in sorted((_REPO / "services").glob("*/pyproject.toml")):
        with child_metadata.open("rb") as handle:
            expected.add(tomllib.load(handle)["project"]["name"].replace("_", "-"))

    assert len(expected) == 7
    assert {item.distribution for item in specs} == expected
    dependencies = " ".join(root_data["project"]["dependencies"])
    assert all(name.replace("-", "_") in dependencies for name in expected)
    assert all(item.source_dir.is_dir() for item in specs)


def test_wrong_checkout_refuses_before_any_path_change(tmp_path):
    selected = tmp_path / "selected"
    other = tmp_path / "other"
    runtime = _make_source_tree(selected)
    _make_source_tree(other)
    paths = ["unchanged"]

    with pytest.raises(SourceClosureError, match="disagrees with the runtime"):
        ensure_source_closure(
            runtime_file=runtime,
            environ={"LORRAX_CHECKOUT": str(other)},
            search_path=paths,
            loaded_modules={},
            print_fn=lambda _line: None,
        )

    assert paths == ["unchanged"]


def test_selected_service_precedes_a_stale_installed_copy(tmp_path):
    selected = tmp_path / "selected"
    runtime = _make_source_tree(selected)
    stale = _make_stale_service(tmp_path / "site-packages")
    paths = [str(stale)]
    lines = []

    receipt = ensure_source_closure(
        runtime_file=runtime,
        environ={"LORRAX_CHECKOUT": str(selected)},
        search_path=paths,
        loaded_modules={},
        print_fn=lines.append,
    )

    wanted = (selected / "services" / "demo_service" / "src").resolve()
    assert Path(paths[0]).resolve() == wanted
    found = machinery.PathFinder.find_spec("demo_service", paths)
    assert found is not None
    assert Path(found.origin).resolve().is_relative_to(wanted)
    assert receipt.mode == "source"
    assert '"demo-service":"services/demo_service/src/demo_service/__init__.py"' \
        in lines[0]


def test_an_already_loaded_stale_service_refuses(tmp_path):
    selected = tmp_path / "selected"
    runtime = _make_source_tree(selected)
    stale_root = _make_stale_service(tmp_path / "site-packages")
    stale = types.ModuleType("demo_service")
    stale.__file__ = str(stale_root / "demo_service" / "__init__.py")

    with pytest.raises(SourceClosureError, match="imported before source closure"):
        ensure_source_closure(
            runtime_file=runtime,
            environ={},
            search_path=[str(stale_root)],
            loaded_modules={"demo_service": stale},
            print_fn=lambda _line: None,
        )


def test_missing_workspace_service_source_refuses(tmp_path):
    selected = tmp_path / "selected"
    runtime = _make_source_tree(selected)
    package = selected / "services" / "demo_service" / "src" / "demo_service"
    (package / "__init__.py").unlink()

    with pytest.raises(SourceClosureError, match="exactly one"):
        ensure_source_closure(
            runtime_file=runtime,
            environ={},
            search_path=[],
            loaded_modules={},
            print_fn=lambda _line: None,
        )


def test_symlink_equivalent_service_paths_are_collapsed(tmp_path):
    """A path's spelling cannot preserve a stale duplicate ahead of the seal."""
    selected = tmp_path / "selected"
    runtime = _make_source_tree(selected)
    service_src = selected / "services" / "demo_service" / "src"
    alias = tmp_path / "service-alias"
    alias.symlink_to(service_src, target_is_directory=True)
    stale = _make_stale_service(tmp_path / "site-packages")
    paths = [str(alias), str(stale), str(service_src)]

    ensure_source_closure(
        runtime_file=runtime,
        environ={},
        search_path=paths,
        loaded_modules={},
        print_fn=lambda _line: None,
    )

    equivalent = [item for item in paths
                  if Path(item).resolve() == service_src.resolve()]
    assert equivalent == [str(service_src.absolute())]
    assert Path(paths[0]).resolve() == service_src.resolve()


def test_installed_distribution_uses_declared_service_packages(
    tmp_path, monkeypatch,
):
    """An installed wheel needs no checkout and synthesizes no source path."""
    site = tmp_path / "site-packages"
    runtime_dir = site / "runtime"
    service_dir = site / "demo_service"
    runtime_dir.mkdir(parents=True)
    service_dir.mkdir()
    runtime_init = runtime_dir / "__init__.py"
    runtime_init.write_text("# installed runtime\n")
    (service_dir / "__init__.py").write_text("# installed service\n")

    lorrax_info = site / "lorrax-0.1.0.dist-info"
    lorrax_info.mkdir()
    (lorrax_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lorrax\nVersion: 0.1.0\n"
        "Requires-Dist: demo_service==0.1.0\n")
    (lorrax_info / "entry_points.txt").write_text(
        "[lorrax.services]\ndemo_service = demo_service\n")
    (lorrax_info / "top_level.txt").write_text("runtime\n")
    service_info = site / "demo_service-0.1.0.dist-info"
    service_info.mkdir()
    (service_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo_service\nVersion: 0.1.0\n")
    (service_info / "top_level.txt").write_text("demo_service\n")
    _write_record(
        service_info,
        "demo_service/__init__.py",
        "demo_service-0.1.0.dist-info/METADATA",
    )

    paths = [str(site), *sys.path]
    original_paths = list(paths)
    monkeypatch.setattr(sys, "path", paths)
    receipt = ensure_source_closure(
        runtime_file=runtime_init,
        environ={},
        search_path=paths,
        loaded_modules={},
        print_fn=lambda _line: None,
    )

    assert receipt.mode == "installed"
    assert dict(receipt.services)["demo-service"] == str(
        (service_dir / "__init__.py").resolve())
    assert paths == original_paths


def test_installed_distribution_refuses_a_duplicate_provider_first_on_path(
    tmp_path, monkeypatch,
):
    """Provider-name overlap cannot bless an import owned by stale bytes."""
    selected = tmp_path / "selected-site"
    stale = tmp_path / "stale-site"
    runtime_dir = selected / "runtime"
    service_dir = selected / "demo_service"
    stale_service_dir = stale / "demo_service"
    runtime_dir.mkdir(parents=True)
    service_dir.mkdir()
    stale_service_dir.mkdir(parents=True)
    runtime_init = runtime_dir / "__init__.py"
    runtime_init.write_text("# installed runtime\n")
    (service_dir / "__init__.py").write_text("ORIGIN = 'selected'\n")
    (stale_service_dir / "__init__.py").write_text("ORIGIN = 'stale'\n")

    lorrax_info = selected / "lorrax-0.1.0.dist-info"
    lorrax_info.mkdir()
    (lorrax_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lorrax\nVersion: 0.1.0\n"
        "Requires-Dist: demo_service==0.1.0\n")
    (lorrax_info / "entry_points.txt").write_text(
        "[lorrax.services]\ndemo_service = demo_service\n")
    expected_info = selected / "demo_service-0.1.0.dist-info"
    expected_info.mkdir()
    (expected_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo_service\nVersion: 0.1.0\n")
    (expected_info / "top_level.txt").write_text("demo_service\n")
    _write_record(expected_info, "demo_service/__init__.py")
    stale_info = stale / "stale_provider-9.0.dist-info"
    stale_info.mkdir()
    (stale_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: stale_provider\nVersion: 9.0\n")
    (stale_info / "top_level.txt").write_text("demo_service\n")
    _write_record(stale_info, "demo_service/__init__.py")

    paths = [str(stale), str(selected), *sys.path]
    monkeypatch.setattr(sys, "path", paths)
    with pytest.raises(SourceClosureError, match="does not own that file"):
        ensure_source_closure(
            runtime_file=runtime_init,
            environ={},
            search_path=paths,
            loaded_modules={},
            print_fn=lambda _line: None,
        )


def test_bootstrap_seals_source_before_existing_runtime_steps():
    """The new step must not reorder autotune, coordination, or failfast."""
    tree = ast.parse((_REPO / "src" / "runtime" / "__init__.py").read_text())
    bootstrap = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "bootstrap")
    calls = [
        node.value.func.id
        for node in bootstrap.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    ]
    assert calls == [
        "_ensure_source_closure",
        "set_default_env",
        "announce_cpu_collectives",
        "skip_gpu_plugin_discovery",
        "init_jax_distributed",
        "fallback_to_cpu_if_no_gpu_backend",
        "install_failfast_excepthook",
        "pin_matmul_precision",
    ]


def test_production_source_seal_does_not_import_jax():
    """Importing runtime and sealing its services stays before JAX."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO / "src")
    env["LORRAX_CHECKOUT"] = str(_REPO)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from runtime.source_closure import ensure_source_closure; "
            "ensure_source_closure(); "
            "print('JAX_LOADED', 'jax' in sys.modules)",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[lorrax source closure]" in result.stdout
    assert "JAX_LOADED False" in result.stdout
