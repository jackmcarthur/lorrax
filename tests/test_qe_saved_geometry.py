"""Saved output geometry and reciprocal normalization form one QE frame."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from file_io.qe_save_reader import CrystalData, _all, _saved_geometry, _text, _vec

FIXTURE = Path(__file__).resolve().parent / "hsuite/fixture_na/data-file-schema.xml"


def fixture(tmp_path):
    root = ET.parse(FIXTURE).getroot()
    output = _all(root, "output")[0]
    saved = _all(output, "atomic_structure")[0]
    initial = _all(_all(root, "input")[0], "atomic_structure")[0]
    return root, output, saved, initial


def load(tmp_path, root):
    ET.ElementTree(root).write(tmp_path / "data-file-schema.xml")
    return CrystalData.from_qe_save(str(tmp_path))


def test_output_alat_normalizes_saved_reciprocal_vectors(tmp_path):
    root, _, saved, initial = fixture(tmp_path)
    alat = float(saved.attrib["alat"])
    initial.set("alat", str(alat * np.sqrt(3.) / 2.))
    crystal = load(tmp_path, root)
    assert crystal.alat == alat
    assert crystal.blat == 2. * np.pi / alat
    np.testing.assert_allclose(
        (crystal.avec * crystal.alat) @ (crystal.bvec * crystal.blat).T,
        2. * np.pi * np.eye(3), atol=2e-14, rtol=0.)


def test_saved_cell_and_atoms_ignore_changed_input_geometry(tmp_path):
    root, _, saved, initial = fixture(tmp_path)
    want_cell = np.array([_vec(_text(saved, f"a{i}")) for i in (1, 2, 3)])
    want_atoms = np.array([_vec(e.text) for e in _all(saved, "atom")])
    for tag in ["a1", "a2", "a3"]:
        _all(initial, tag)[0].text = " ".join(map(str, 1.25 * _vec(_text(initial, tag))))
    _all(initial, "atom")[0].text = "1 2 3"
    crystal = load(tmp_path, root)
    np.testing.assert_array_equal(crystal.avec * crystal.alat, want_cell)
    np.testing.assert_allclose(crystal.atom_crys @ want_cell, want_atoms, atol=1e-15, rtol=0.)


def test_equal_input_output_frame_preserves_geometry(tmp_path):
    root, output, saved, initial = fixture(tmp_path)
    initial.set("alat", saved.attrib["alat"])
    crystal = load(tmp_path, root)
    assert crystal.alat == float(saved.attrib["alat"])
    np.testing.assert_array_equal(crystal.bvec, np.array([_vec(_text(output, f"b{i}")) for i in (1, 2, 3)]))


@pytest.mark.parametrize("poison", ["duplicate_output", "duplicate_structure", "missing_saved_structure"])
def test_ambiguous_or_missing_saved_frame_refuses(tmp_path, poison):
    root, output, saved, _ = fixture(tmp_path)
    if poison == "duplicate_output":
        root.append(copy.deepcopy(output))
    elif poison == "duplicate_structure":
        output.append(copy.deepcopy(saved))
    else:
        output.remove(saved)
    with pytest.raises(ValueError, match="ambiguous|missing"):
        load(tmp_path, root)


def test_legacy_sole_structure_without_output_wrapper():
    root = ET.fromstring('<root><atomic_structure alat="5.42" nat="1"/></root>')
    frame, structure = _saved_geometry(root)
    assert frame is root and structure.attrib["alat"] == "5.42"


def test_legacy_multiple_unframed_structures_refuse():
    root = ET.fromstring('<root><atomic_structure/><atomic_structure/></root>')
    with pytest.raises(ValueError, match="ambiguous"):
        _saved_geometry(root)
