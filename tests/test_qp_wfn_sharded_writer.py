"""The collective QP WFN writer puts every G-slab where the readers look for it.

``file_io.qp_wfn.write_qp_wfn_h5`` streams ``wfns/coeffs`` window by window,
each rank reading, rotating and writing its own G-slab through SlabIO.  A
slab written at the wrong k offset, the wrong G column or across a k
boundary leaves a file with the right shape and wrong ψ, so the checks here
compare every coefficient against a NumPy rotation of the source, read two
ways: the raw BGW layout with h5py, and the WfnLoader reader that restart,
BSE and htransform use.

The fixture is ragged on purpose: four k-points with different ``ngk``, not
all divisible by the mesh size, so windows carry pad columns (TASTE 11), two spinor components, an active
window strictly inside the band range, and a second arm with G windows
narrower than one k so the last window of every k is clipped by
``valid_shape``.

Single process on an emulated 2x2 CPU mesh (SlabIO's serial tier).  The
same checks on the phdf5 transport at P1 and P4 are
``tests/multi_device/qp_wfn_sharded_writer_p4.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

NBANDS, NSPINOR = 7, 2
BAND_START, BAND_STOP = 2, 6
ALAT, ECUTWFC = 10.0, 2.0
KPOINTS = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                    [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]])

pytestmark = pytest.mark.filterwarnings(
    "ignore:WFN density symmetry check FAILED:RuntimeWarning")


def _crystal():
    blat = 2.0 * np.pi / ALAT
    mtrx = np.zeros((48, 3, 3), dtype=np.int32)
    mtrx[0] = np.eye(3, dtype=np.int32)
    return SimpleNamespace(
        nspin=1, nspinor=NSPINOR, nelec=4, nat=1,
        ecutwfc=ECUTWFC, ecutrho=4.0 * ECUTWFC, fft_grid=(8, 8, 8),
        alat=ALAT, blat=blat, cell_volume=ALAT ** 3,
        avec=np.eye(3), bvec=np.eye(3), bdot=(blat ** 2) * np.eye(3),
        atom_crys=np.zeros((1, 3)), atom_types=np.array([1], dtype=np.int32),
        ntran=1, sym_matrices=mtrx, translations=np.zeros((48, 3)),
        assume_isolated=None)


def write_source(path, seed=20260925):
    """A ragged 4-k WFN.h5 through the repo's own writer.

    Returns ``(coeffs_per_k, U_kmn, E_active_ry)``: the source ψ and the
    rotation inputs, identical on every rank for a given ``seed``.
    """
    from file_io import WFNWriter
    from psp.gvec_utils import build_master_gvec_list, select_gvecs_for_k

    crystal = _crystal()
    master, _ = build_master_gvec_list(crystal)
    gvecs = [select_gvecs_for_k(k, master, crystal.bdot, ECUTWFC)[0]
             for k in KPOINTS]
    rng = np.random.default_rng(seed)
    coeffs = [rng.standard_normal((NBANDS, NSPINOR, g.shape[0]))
              + 1j * rng.standard_normal((NBANDS, NSPINOR, g.shape[0]))
              for g in gvecs]
    energies = np.sort(rng.uniform(-1.0, 3.0, (len(KPOINTS), NBANDS)), axis=1)
    nb_a = BAND_STOP - BAND_START
    U = np.linalg.qr(rng.standard_normal((len(KPOINTS), nb_a, nb_a))
                     + 1j * rng.standard_normal((len(KPOINTS), nb_a, nb_a)))[0]
    E = energies[:, BAND_START:BAND_STOP] + rng.uniform(-0.1, 0.1, (len(KPOINTS), nb_a))
    if path is not None:
        writer = WFNWriter(str(path), crystal, KPOINTS,
                           np.full(len(KPOINTS), 0.25), (2, 2, 1), NBANDS,
                           gvecs, nosym=True)
        for ik, c in enumerate(coeffs):
            writer.write_k(ik, energies[ik], c)
        writer.close()
    return coeffs, U, E


def expected_qp_coeffs(coeffs, U):
    """``c_qp[n] = Σ_m U[m, n] c[m]`` on the active window, per k."""
    out = []
    for c, u in zip(coeffs, U):
        q = c.copy()
        q[BAND_START:BAND_STOP] = np.einsum(
            "mn,msg->nsg", u, c[BAND_START:BAND_STOP])
        out.append(q)
    return out


def write_qp(src_path, out_path, mesh, U, E):
    """Every rank: the production writer on the source file."""
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    from file_io.qp_wfn import write_qp_wfn_h5

    with WfnLoader(str(src_path)) as wfn:
        write_qp_wfn_h5(str(out_path), wfn=wfn, U_kmn=U, enk_active_qp_ry=E,
                        band_start=BAND_START, band_stop=BAND_STOP, mesh=mesh)
        return np.array(wfn.energies[0], dtype=np.float64)


def check_file(src_path, out_path, want, E, dft_energies):
    """Raw BGW layout (h5py) and the WfnLoader reader against ``want``."""
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    from file_io.qp_wfn import QP_WFN_ATTR, QP_WFN_SCHEME

    ngk = np.array([c.shape[-1] for c in want])
    starts = np.concatenate([[0], np.cumsum(ngk)[:-1]])
    with h5py.File(str(src_path), "r") as src, h5py.File(str(out_path), "r") as out:
        coeffs = out["wfns/coeffs"]
        assert coeffs.shape == (NBANDS, NSPINOR, int(ngk.sum()), 2)
        assert coeffs.dtype == np.float64
        got = coeffs[...]
        for ik, w in enumerate(want):
            blk = got[:, :, starts[ik]:starts[ik] + ngk[ik]]
            np.testing.assert_allclose(blk[..., 0] + 1j * blk[..., 1], w,
                                       rtol=0, atol=1e-12,
                                       err_msg=f"wfns/coeffs at k={ik}")
        for inactive in (slice(0, BAND_START), slice(BAND_STOP, NBANDS)):
            assert np.array_equal(got[inactive], src["wfns/coeffs"][inactive]), (
                "an inactive band is not a bitwise copy of the source")
        headers = []
        src.visititems(lambda n, o: headers.append(n)
                       if isinstance(o, h5py.Dataset) else None)
        for name in headers:
            if name in ("wfns/coeffs", "mf_header/kpoints/el"):
                continue
            a, b = src[name], out[name]
            assert (a.dtype, a.shape) == (b.dtype, b.shape), name
            assert np.array_equal(a[()], b[()]), name
        el = dft_energies.copy()
        el[:, BAND_START:BAND_STOP] = E
        assert np.array_equal(out["mf_header/kpoints/el"][0], el)
        assert out.attrs[QP_WFN_ATTR] == QP_WFN_SCHEME
        assert int(out.attrs["qp_wfn_band_start"]) == BAND_START
        assert int(out.attrs["qp_wfn_band_stop"]) == BAND_STOP

    with WfnLoader(str(out_path)) as reader:
        psi = np.asarray(reader.load(bands=(0, NBANDS), k="ibz"))
    for ik, w in enumerate(want):
        np.testing.assert_allclose(psi[ik, :NBANDS, :, :ngk[ik]], w,
                                   rtol=0, atol=1e-12,
                                   err_msg=f"WfnLoader ψ at k={ik}")
        assert not np.any(psi[ik, :, :, ngk[ik]:]), f"pad columns at k={ik}"


def _emulated_mesh():
    import jax
    from jax.sharding import Mesh

    devs = jax.devices("cpu")
    if len(devs) < 4:
        pytest.skip("needs 4 cpu devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))


@pytest.mark.mesh(4)
@pytest.mark.parametrize("windows_per_k", ["one", "several"])
def test_qp_wfn_round_trips_through_both_readers(tmp_path, monkeypatch,
                                                 windows_per_k):
    from file_io import qp_wfn

    mesh = _emulated_mesh()
    if windows_per_k == "several":
        # 12 columns per window: every k spans several windows and its last
        # one is clipped by valid_shape, the path a small device budget takes.
        monkeypatch.setattr(qp_wfn, "_coefficient_window",
                            lambda mesh, **_: 3 * int(mesh.devices.size))
    src, out = tmp_path / "WFN.h5", tmp_path / "WFN_qp.h5"
    coeffs, U, E = write_source(src)
    assert any(c.shape[-1] % 4 for c in coeffs), "fixture must carry pad columns"
    assert len({c.shape[-1] for c in coeffs}) > 1, "fixture must be ragged"
    dft = write_qp(src, out, mesh, U, E)
    check_file(src, out, expected_qp_coeffs(coeffs, U), E, dft)
