"""Admission and restart identity gates for matched atomic Coulomb augmentation."""
from pathlib import Path

import h5py
import pytest

from file_io.restart_bundle import require_atomic_augmentation_match
from gw.gw_config import LorraxConfig


def _config(tmp_path, body):
    deck = tmp_path / "cohsex.in"
    deck.write_text("[cohsex]\nnval=1\nncond=2\nnumber_bands=7\n"
                    "qp_solver=one_shot_dft\nlinalg=local\ncompute_mode=x_only\n"
                    + ("sys_dim=3\n" if "sys_dim=" not in body else "") + body)
    return LorraxConfig.from_input_file(
        str(deck), resolve_hardware=False, print_fn=lambda *_: None)


def test_augmentation_path_is_explicit_and_resolved(tmp_path):
    cfg = _config(tmp_path, "bispinor=true\nbispinor_gw=coulomb_only\n"
                  "atomic_reconstruction_dir=atoms\n")
    assert Path(cfg.paths.atomic_reconstruction_dir) == tmp_path / "atoms"
    assert _config(tmp_path, "").paths.atomic_reconstruction_dir is None


@pytest.mark.parametrize("body", [
    "bispinor=false\n",
    "bispinor=true\nbispinor_gw=bare_transverse\n",
    "bispinor=true\nbispinor_gw=coulomb_only\nsys_dim=2\n",
])
def test_unsupported_compensation_kernel_refuses(tmp_path, body):
    with pytest.raises(ValueError, match="atomic_augmentation_domain"):
        _config(tmp_path, body + "atomic_reconstruction_dir=atoms\n")


def test_restart_reconstruction_must_match_in_both_directions(tmp_path):
    filename = tmp_path / "tensors.h5"
    with h5py.File(filename, "w"):
        pass
    require_atomic_augmentation_match(filename, None)
    with pytest.raises(ValueError, match="restart_atomic_augmentation"):
        require_atomic_augmentation_match(filename, "ae-ps-a")
    with h5py.File(filename, "a") as f:
        f.attrs["atomic_augmentation"] = "ae-ps-a"
    require_atomic_augmentation_match(filename, "ae-ps-a")
    for wanted in (None, "ae-ps-b"):
        with pytest.raises(ValueError, match="restart_atomic_augmentation"):
            require_atomic_augmentation_match(filename, wanted)
