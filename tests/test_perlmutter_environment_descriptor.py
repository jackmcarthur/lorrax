"""Static contracts for the Perlmutter environment descriptor.

The module is consumed by the external ``lx`` launcher.  It describes site
capabilities; process policy belongs to :mod:`runtime`.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "config/modulefiles/lorrax/0.1.0.lua"
INSTALL = ROOT / "config/perlmutter/install.sh"
SHIFTER = ROOT / "src/ffi/cpp/run_shifter.sh"

_RUNTIME_POLICY = (
    "HDF5_USE_FILE_LOCKING",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "TF_GPU_ALLOCATOR",
)


def test_site_launchers_do_not_duplicate_runtime_policy():
    text = MODULE.read_text() + SHIFTER.read_text()
    for name in _RUNTIME_POLICY:
        assert name not in text, f"{name} is owned by src/runtime"


def test_module_is_a_descriptor_not_a_second_launcher():
    text = MODULE.read_text()
    assert 'setenv("LORRAX_ROOT"' in text
    assert 'setenv("LORRAX_SHIFTER"' in text
    assert "set_shell_function" not in text
    assert not (ROOT / "config/perlmutter/run_gw.slurm").exists()


def test_standalone_wrapper_has_one_selected_nvhpc_runtime():
    line = next(
        item for item in SHIFTER.read_text().splitlines()
        if item.startswith('LDLIB="')
    )
    assert line.count("/lorrax_nvhpc/") == 1
    assert "25.5_cuda12.9" not in line


def test_installer_substitutes_every_module_placeholder():
    module_keys = set(re.findall(r"@LORRAX_[A-Z0-9_]+@", MODULE.read_text()))
    install_keys = set(re.findall(r"@LORRAX_[A-Z0-9_]+@", INSTALL.read_text()))
    assert module_keys == install_keys
