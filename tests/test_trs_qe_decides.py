"""Two-component TRS verdict: QE's data decides, the WFN check guards (CPU, seconds).

TRS holds only if QE types no row t_rev = 1 and the SCF absolute
magnetization is below tolerance; no schema means TRS off. When QE says
nonmagnetic the occupied-density check runs as a guard and refuses on a
residual above tolerance. ``scf_absolute_magnetization`` reads |m| from the
SCF schema whose density file the NSCF .save holds byte for byte, because
an NSCF schema writes ``absolute`` = 0.
"""
import numpy as np
import pytest

from ffi import _services

_services.ensure_on_path()

from symmetry_maps import QESymmetryBinding, check_spinor_reference_trs  # noqa: E402
from symmetry_maps.qe_schema import scf_absolute_magnetization  # noqa: E402

_G = np.indices((3, 3, 3)).reshape(3, -1).T - 1      # G and -G both present
_IDX = {tuple(g): i for i, g in enumerate(_G)}
_K = [0.1, 0.2, 0.3]                                 # not a TRIM


def _theta(psi):
    """Theta = i sigma_y K on (band, 2, G) rows: the -k partner on the G list -G."""
    out = np.empty_like(psi)
    for i, g in enumerate(_G):
        j = _IDX[tuple(-g)]
        out[:, 0, j] = np.conj(psi[:, 1, i])
        out[:, 1, j] = -np.conj(psi[:, 0, i])
    return out


def _orthonormal(rng, nb):
    a = rng.standard_normal((2 * len(_G), nb)) + 1j * rng.standard_normal((2 * len(_G), nb))
    q, _ = np.linalg.qr(a)
    return q.T.reshape(nb, 2, len(_G))


def _binding(antiunitary, m):
    return QESymmetryBinding(
        schema_path="toy/data-file-schema.xml", schema_sha256="0" * 64,
        antiunitary=np.asarray(antiunitary, bool),
        qe_permitted_pure_time_reversal=False, scf_absolute_magnetization=m)


class _ToyLoader:
    """The loader surface ``check_spinor_reference_trs`` reads."""

    nspin, nspinor = 1, 2
    avec = np.eye(3)
    kgrid = np.array([4, 4, 4])
    shift = np.zeros(3)
    fft_grid = (6, 6, 6)
    path = "toy/WFN.h5"

    def __init__(self, kpoints, psi_by_k, sym_matrices, binding):
        self.kpoints = np.asarray(kpoints, float)
        self.nkpts = len(self.kpoints)
        self.kweights = np.full(self.nkpts, 1.0 / self.nkpts)
        self.nbands = psi_by_k[0].shape[0]
        self.physical_density_band_stop = self.nbands
        self.num_electrons = float(self.nbands)
        self.sym_matrices = np.asarray(sym_matrices, np.int64)
        self.ntran = len(self.sym_matrices)
        self.translations = np.zeros((self.ntran, 3))
        self.ngk = np.full(self.nkpts, len(_G))
        self.kpt_starts = np.arange(self.nkpts) * len(_G)
        self._gvecs_raw = np.concatenate([_G] * self.nkpts)
        coeffs = np.concatenate(psi_by_k, axis=-1)
        self._file = {"wfns/coeffs": np.stack([coeffs.real, coeffs.imag], -1)}
        self._binding = binding

    def physical_density_occupations(self, *, k, unit_as_none):
        return np.ones((self.nkpts, self.nbands))

    def resolve_qe_symmetry(self):
        return self._binding


def _pt_magnet(binding):
    """Occupied space at k closed under Theta*I (row -I); -k not stored."""
    psi1 = _orthonormal(np.random.default_rng(0), 1)
    psi2 = np.stack([np.conj(psi1[:, 1]), -np.conj(psi1[:, 0])], axis=1)
    return _ToyLoader([_K], [np.concatenate([psi1, psi2])],
                      [np.eye(3), -np.eye(3)], binding)


def test_no_schema_is_off():
    report = check_spinor_reference_trs(_pt_magnet(None))
    assert report.trs_basis == "qe-magnetic" and not report.trs_holds


def test_t_rev_row_is_off():
    """Theta*I typed t_rev = 1; the WFN would pass a test through -I as unitary."""
    report = check_spinor_reference_trs(_pt_magnet(_binding([False, True], 0.0)))
    assert report.trs_basis == "qe-magnetic" and not report.trs_holds


def test_magnetization_without_t_rev_rows_is_off():
    """A type-I magnetic group: every row unitary, m != 0."""
    report = check_spinor_reference_trs(_pt_magnet(_binding([False, False], 6.66)))
    assert report.trs_basis == "qe-magnetic" and not report.trs_holds
    report = check_spinor_reference_trs(_pt_magnet(_binding([False, False], None)))
    assert not report.trs_holds                       # no SCF magnetization found


def test_nonmagnetic_soc_without_inversion_is_on():
    """P1 with SOC, m = 0: no unitary k -> -k, -k not stored. The guard has
    only the Gamma Kramers pair, and QE's data turns TRS on."""
    rng = np.random.default_rng(1)
    gamma = _orthonormal(rng, 1)
    gamma = np.concatenate([gamma, _theta(gamma)])
    loader = _ToyLoader([[0, 0, 0], _K], [gamma, _orthonormal(rng, 2)],
                        [np.eye(3)], _binding([False], 0.0))
    report = check_spinor_reference_trs(loader)
    assert dict(report.evidence_counts) == {"raw-pair": 0, "spatial-pair": 0, "trim": 1}
    assert report.trs_holds and report.subspace_residual < report.tol_trs


def test_nonmagnetic_guard_refuses_inconsistent_wfn():
    """QE says nonmagnetic but psi(-k) is not Theta psi(k)."""
    rng = np.random.default_rng(2)
    loader = _ToyLoader([_K, [-v for v in _K]],
                        [_orthonormal(rng, 2), _orthonormal(rng, 2)],
                        [np.eye(3)], _binding([False], 0.0))
    with pytest.raises(RuntimeError, match="GATE trs_qe_nonmagnetic_wfn_consistent"):
        check_spinor_reference_trs(loader)


def _schema(directory, *, calculation, do_mag, absolute, density):
    directory.mkdir(parents=True)
    (directory / "data-file-schema.xml").write_text(f"""<qes:espresso xmlns:qes="x">
<input><control_variables><calculation>{calculation}</calculation></control_variables></input>
<output><symmetries><nsym>1</nsym><symmetry><info>crystal_symmetry</info>
<rotation>1 0 0 0 1 0 0 0 1</rotation></symmetry></symmetries>
<magnetization><noncolin>true</noncolin><absolute>{absolute}</absolute>
<do_magnetization>{do_mag}</do_magnetization></magnetization>
<basis_set><reciprocal_lattice><b1>1 0 0</b1><b2>0 1 0</b2><b3>0 0 1</b3></reciprocal_lattice></basis_set>
</output></qes:espresso>""")
    (directory / "charge-density.hdf5").write_bytes(density)
    return str(directory / "data-file-schema.xml")


def test_scf_magnetization_from_the_matching_density(tmp_path):
    from symmetry_maps import read_qe_symmetry_receipt
    nscf = read_qe_symmetry_receipt(_schema(
        tmp_path / "nscf/x.save", calculation="nscf", do_mag="true",
        absolute=0.0, density=b"rho-scf"))
    scf = _schema(tmp_path / "scf/x.save", calculation="scf", do_mag="true",
                  absolute=6.66, density=b"rho-scf")
    other = _schema(tmp_path / "old/x.save", calculation="scf", do_mag="true",
                    absolute=0.0, density=b"rho-old")
    assert scf_absolute_magnetization([nscf], [other, scf]) == 6.66
    assert scf_absolute_magnetization([nscf], [other]) is None
    plain = read_qe_symmetry_receipt(_schema(
        tmp_path / "nm/x.save", calculation="nscf", do_mag="false",
        absolute=0.0, density=b"rho-nm"))
    assert scf_absolute_magnetization([plain], []) == 0.0
