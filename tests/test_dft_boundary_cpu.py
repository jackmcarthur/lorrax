"""DFT boundary provenance from authenticated QE metadata, without device work."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from file_io.qe_save_reader import CrystalData, dft_sys_dim
from file_io import validate_kin_ion_against_run
from symmetry_maps import resolve_qe_symmetry_binding


def _xml(folder, boundary):
    folder.mkdir()
    path = folder / 'data-file-schema.xml'
    isolated = '' if boundary is None else f'<assume_isolated>{boundary}</assume_isolated>'
    path.write_text(f'''<root>
      <input><control_variables><calculation>bands</calculation></control_variables>
        <starting_k_points><monkhorst_pack nk1="1" nk2="1" nk3="1"/></starting_k_points>
        {isolated}</input>
      <atomic_structure nat="1" alat="8"><cell>
        <a1>8 0 0</a1><a2>0 8 0</a2><a3>0 0 8</a3></cell>
        <atomic_positions><atom name="H">0 0 0</atom></atomic_positions>
      </atomic_structure>
      <reciprocal_lattice><b1>1 0 0</b1><b2>0 1 0</b2><b3>0 0 1</b3></reciprocal_lattice>
      <symmetries><nsym>1</nsym><symmetry>
        <info time_reversal="false">crystal_symmetry</info>
        <rotation>1 0 0 0 1 0 0 0 1</rotation>
        <fractional_translation>0 0 0</fractional_translation>
      </symmetry></symmetries>
      <band_structure><noncolin>false</noncolin><lsda>false</lsda>
        <spinorbit>false</spinorbit><nelec>1</nelec><nbnd>2</nbnd>
        <ks_energies><k_point>0 0 0</k_point></ks_energies></band_structure>
      <ecutwfc>5</ecutwfc><ecutrho>20</ecutrho><fft_grid nr1="4" nr2="4" nr3="4"/>
    </root>''')
    return path


def _loader(schema):
    wfn = SimpleNamespace(kgrid=np.ones(3, dtype=int), nspinor=1, nkpts=1,
                          kpoints=np.zeros((1, 3)), ntran=1,
                          sym_matrices=np.eye(3, dtype=int)[None],
                          translations=np.zeros((1, 3)), qe_symmetry_binding=None)
    def resolve():
        if wfn.qe_symmetry_binding is None:
            wfn.qe_symmetry_binding, _ = resolve_qe_symmetry_binding(
                wfn, wfn_path=Path(schema).parent / 'WFN.h5', schema=schema)
        return wfn.qe_symmetry_binding
    wfn.resolve_qe_symmetry = resolve
    return wfn


def test_bound_qe_boundary_overrides_screening_and_legacy_absence(tmp_path):
    for index, (value, expected) in enumerate(((None, 3), ('none', 3), ('2D', 2))):
        path = _xml(tmp_path / str(index), value)
        crystal = CrystalData.from_qe_save(path.parent)
        assert (2 if crystal.assume_isolated == '2D' else 3) == expected
        # Explicit custom filenames must use the authenticated file, not a
        # reconstructed *.save/data-file-schema.xml path.
        custom = path.with_name('source-nscf.xml')
        path.rename(custom)
        loader = _loader(custom)
        assert loader.qe_symmetry_binding is None
        assert dft_sys_dim(loader, fallback=5 - expected) == expected
        assert loader.qe_symmetry_binding.schema_path == str(custom)
    assert dft_sys_dim(SimpleNamespace(), fallback=2) == 2
    assert dft_sys_dim(SimpleNamespace(resolve_qe_symmetry=lambda: None), fallback=3) == 3

    unsupported = _xml(tmp_path / 'unsupported', 'mt')
    with pytest.raises(ValueError, match="Unsupported assume_isolated='mt'"):
        dft_sys_dim(_loader(unsupported), fallback=3)
    with pytest.raises(ValueError, match="Unsupported assume_isolated='mt'"):
        CrystalData.from_qe_save(unsupported.parent)
    mismatch = _loader(custom)
    mismatch.kgrid = np.array([2, 1, 1])
    with pytest.raises(ValueError, match='does not authenticate this WFN'):
        dft_sys_dim(mismatch, fallback=3)


def test_old_boundary_binding_recovers_fact_and_rejects_changed_source(tmp_path):
    path = _xml(tmp_path / 'old', '2D')
    loader = _loader(path)
    binding = replace(loader.resolve_qe_symmetry(), assume_isolated=None)
    loader.resolve_qe_symmetry = lambda: binding
    assert dft_sys_dim(loader, fallback=3) == 2
    path.write_text(path.read_text().replace('<assume_isolated>2D', '<assume_isolated>none'))
    with pytest.raises(ValueError, match='changed after WFN authentication'):
        dft_sys_dim(loader, fallback=3)


def test_wrong_ionic_boundary_is_refused_before_operator_read(tmp_path):
    path = _xml(tmp_path / 'periodic', None)
    loader = _loader(path)
    expected = dft_sys_dim(loader, fallback=2)
    operator = tmp_path / 'kin_ion.h5'
    with h5py.File(operator, 'w') as file:
        matrix = file.create_dataset('kin_ion', data=np.zeros((1, 2, 2), complex))
        matrix.attrs.update(bispinor=False, sys_dim=2, nspinor=1)
    options = dict(expected_bispinor=False, sys_dim=expected, nk=1,
                   band_stop=2, nspinor=1, print_fn=lambda *_: None)
    with pytest.raises(ValueError, match='has sys_dim=2.*uses sys_dim=3'):
        validate_kin_ion_against_run(operator, **options)
    with h5py.File(operator, 'r+') as file:
        file['kin_ion'].attrs['sys_dim'] = 3
    validate_kin_ion_against_run(operator, **options)
