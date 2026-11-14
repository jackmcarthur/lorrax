"""A bispinor one-shot writes its sector matrices under ``sigma_lorentz_debug_output``.

CPU, pytest.  ``gw.gw_output.write_oneshot_sigma_lorentz`` writes the SC maps'
format (``sigma_lorentz_skij_ev``, CC / CT+TC / TT, eV) with an identity
rotation on the full BZ, so the TT sector of a ``bare_transverse`` one-shot
(Sigma^B) is read from one run instead of the difference of two.  Off the
key, or on a two-component run, nothing is written.
"""

from types import SimpleNamespace

import numpy as np


def test_oneshot_writes_sector_matrices_with_identity_rotation(tmp_path):
    import h5py
    from common.units import RYD_TO_EV
    from gw.gw_output import write_oneshot_sigma_lorentz
    from gw.sigma_dispatch import SIGMA_KSET_FULL_BZ

    nk, nb = 2, 3
    rng = np.random.default_rng(5)
    parts = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(size=(3, nk, nb, nb))
    result = SimpleNamespace(sigma_lorentz_skij_ry=parts, kset=SIGMA_KSET_FULL_BZ)
    sym = SimpleNamespace(unfolded_kpts=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]))
    slices = SimpleNamespace(b0=0)
    said = []
    cfg = lambda bisp, debug: SimpleNamespace(
        bispinor=bisp, debug=SimpleNamespace(sigma_lorentz_debug_output=debug))
    for bisp, debug in ((False, True), (True, False)):
        assert write_oneshot_sigma_lorentz(cfg(bisp, debug), str(tmp_path), result,
                                            sym, slices, said.append) is None
    assert not list(tmp_path.iterdir())
    from common.provenance import set_run_provenance
    stamp = "hartree=on; exchange=off; semicore=excluded_bands:8"
    set_run_provenance(four_current_carrier="pauli_charge_rkb_current_v1",
                       bispinor_transverse_terms=stamp)
    try:
        path = write_oneshot_sigma_lorentz(cfg(True, True), str(tmp_path), result, sym,
                                            slices, said.append)
    finally:
        set_run_provenance()
    with h5py.File(path, "r") as h5:
        assert h5.attrs["basis"] == "dft"
        assert h5.attrs["bispinor_transverse_terms"].decode() == stamp
        assert h5.attrs["four_current_carrier"].decode() == "pauli_charge_rkb_current_v1"
        assert np.allclose(h5["sigma_lorentz_skij_ev"][()], parts * RYD_TO_EV, rtol=0, atol=1e-12)
        assert np.array_equal(h5["U_dft_to_qp_kij"][()],
                              np.broadcast_to(np.eye(nb), (nk, nb, nb)))
        assert h5.attrs["role"] == "one_shot" and int(h5.attrs["call_index"]) == 0
        assert list(h5.attrs["sectors"]) == ["CC", "CT+TC", "TT"]
    assert "sigma_lorentz_oneshot.h5" in said[-1]


def test_debug_writer_runs_after_the_primary_outputs():
    """In ``gw_jax._write_gw_results`` the sector writer follows ``write_results``.

    Read from the source (the driver needs the native library to import): a
    raising debug writer must not cost the run its eqp and sigma files.
    """
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "gw" / "gw_jax.py"
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_write_gw_results")
    calls = [(n.lineno, n.func.id) for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("write_results", "write_oneshot_sigma_lorentz")]
    order = [name for _line, name in sorted(calls)]
    assert order == ["write_results", "write_oneshot_sigma_lorentz"], order
