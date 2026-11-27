"""The suite's full-BSE stages (``bse_full``, ``bse_full_dense``, ``exciton_bands_full``): the parts that need no GPU.

CPU, pytest; no driver runs.  The stages run on four GPU ranks (``tests/hsuite``);
what can go wrong without a solve is checked here: the input files parse under the
two drivers' refusals (bse_jax reads the full-BSE keys, and refuses solver keys the
dense solve does not read; exciton_bands refuses ``solver_max_iter`` under the dense
solve), the scissor eqp file the stages read, the roots comparator that pins the dense
solve to the Davidson one, and the stage order.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parent / "hsuite" / "fixture"


def _keys(text):
    keys = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            keys[key] = value
    return keys


@pytest.fixture()
def run(tmp_path):
    from tests.hsuite import chain
    shutil.copy(FIXTURE / "WFN.h5", tmp_path)
    (tmp_path / "centroids_frac_6.txt").write_text("0 0 0\n")
    chain.write_decks(tmp_path)
    return tmp_path


def test_the_full_bse_decks_differ_from_the_bse_deck_by_the_full_bse_keys(run):
    base = _keys((run / "gnppm.in").read_text())
    davidson = _keys((run / "bse_full.in").read_text())
    dense = _keys((run / "bse_full_dense.in").read_text())
    assert {k: v for k, v in davidson.items() if base.get(k) != v} == {
        "eqp_corrections": "full_bse_eqp.dat", "full_bse": "true",
        "full_bse_solver": "davidson"}
    # the dense solve reads no iteration cap (bse_jax refuses one): the only key it drops
    assert set(base) - set(dense) == {"solver_max_iter"}
    assert {k: v for k, v in dense.items() if base.get(k) != v} == {
        "eqp_corrections": "full_bse_eqp.dat", "full_bse": "true",
        "full_bse_solver": "dense"}
    exb, exb_full = (_keys((run / n).read_text()) for n in ("excited.in", "excited_full.in"))
    assert {k: v for k, v in exb_full.items() if exb.get(k) != v} == {
        "eqp_corrections": "full_bse_eqp.dat", "full_bse": "true",
        "full_bse_solver": "davidson"}
    assert set(exb) <= set(exb_full)


def test_bse_jax_parses_both_full_bse_decks(run):
    from bse import bse_jax
    _, dav = bse_jax.parse_args(["-i", str(run / "bse_full.in")])
    _, den = bse_jax.parse_args(["-i", str(run / "bse_full_dense.in")])
    for args, solver in ((dav, "davidson"), (den, "dense")):
        assert args.tda is False and args.full_bse_solver == solver
        assert args.full_bse_instability == "refuse"
        assert (args.n_val, args.n_cond, args.n_eig) == (1, 2, 2)
        assert not args.rpa and Path(args.eqp).name == "full_bse_eqp.dat"
    assert dav.max_lanczos_iter == 40 and den.max_lanczos_iter is None
    # the Tamm-Dancoff stage's own deck still parses as before
    _, tda = bse_jax.parse_args(["-i", str(run / "gnppm.in")])
    assert tda.tda is True and tda.eqp is None


def test_exciton_bands_parses_the_full_bse_deck(run):
    from bse import exciton_bands as exb
    parser = exb.build_parser()

    def parse(name):
        args = parser.parse_args(["-i", str(run / name)])
        exb.deck_settings(parser, args)
        return args
    full, tda = parse("excited_full.in"), parse("excited.in")
    assert full.full_bse and full.full_solver == "davidson" and full.full_instability == "refuse"
    assert full.solver == "davidson" and full.max_iter == 40 and not tda.full_bse
    assert (full.n_val, full.n_cond, full.n_eig) == (1, 1, 1) and Path(full.eqp).name == "full_bse_eqp.dat"


def test_scissor_eqp_file(tmp_path):
    from file_io.restart_bundle import read_bgw_eqp
    from gw.eqp_bgw import write_bgw_eqp
    from tests.hsuite import chain
    rng = np.random.default_rng(0)
    kpts = np.array([[0, 0, 0], [0.2, 0, 0], [0.4, 0.4, 0]])
    e_dft = np.sort(rng.normal(size=(3, 5)), axis=1) * 3
    write_bgw_eqp(str(tmp_path / "gnppm_eqp1.dat"), kpts, e_dft, e_dft + 0.37, band_offset=0,
                  comments=("footer",))
    chain.write_full_bse_eqp(tmp_path)
    k, dft, qp, offset = read_bgw_eqp(str(tmp_path / chain.FULL_BSE_EQP))
    assert offset == 0 and np.allclose(k, kpts) and dft.shape == (3, 5)
    assert np.allclose(dft, e_dft, atol=1e-8)
    assert np.allclose(qp[:, :3], e_dft[:, :3], atol=1e-8)
    assert np.allclose(qp[:, 3:], e_dft[:, 3:] + chain.FULL_BSE_SCISSOR_EV, atol=1e-8)


def test_roots_row_and_the_dense_against_davidson_comparator(tmp_path, capsys):
    import re
    from tests.hsuite import chain
    sample = "  state       energy (eV)\n  S001        4.12346\n  S002        4.50011\n"
    assert [float(v) for v in re.findall(chain.BSE_ROOT_ROW, sample, re.MULTILINE)] == [4.12346, 4.50011]

    def log(name, roots):
        (tmp_path / f"{name}.rank0.log").write_text(
            "  state       energy (eV)\n" + "".join(
                f"  S{i + 1:03d}  {r:15.5f}\n" for i, r in enumerate(roots)))
    log("bse_full", [4.12346, 4.50011])
    log("bse_full_dense", [4.12346, 4.50011])
    assert chain.stdout_twin_problems(tmp_path, "bse_full_dense", "bse_full") == []
    log("bse_full_dense", [4.12346, 4.50011 + 2e-4])           # beyond ATOL["bse_ev"]
    assert "differ" in chain.stdout_twin_problems(tmp_path, "bse_full_dense", "bse_full")[0]
    log("bse_full_dense", [4.12346 + 5e-5, 4.50011])           # within it
    assert chain.stdout_twin_problems(tmp_path, "bse_full_dense", "bse_full") == []
    log("bse_full_dense", [4.12346])                            # a root missing
    assert "roots" in chain.stdout_twin_problems(tmp_path, "bse_full_dense", "bse_full")[0]
    (tmp_path / "bse_full.rank0.log").unlink()
    assert "--only" in chain.stdout_twin_problems(tmp_path, "bse_full_dense", "bse_full")[0]
    assert "roots max|diff|" in capsys.readouterr().out


def test_stage_order_and_rules():
    from tests.hsuite import chain
    names = [name for name, _module, _argv in chain.STAGES]
    assert (names.index("gnppm") < names.index("bse") < names.index("bse_full")
            < names.index("bse_full_dense") < names.index("htransform")
            < names.index("exciton_bands") < names.index("exciton_bands_full")
            < names.index("cohsex"))
    assert len([n for n in names if n not in chain.OPT_IN]) == 21
    argv = {name: a for name, _module, a in chain.STAGES}
    assert argv["exciton_bands_full"][-4:] == ["--out-prefix", "exciton_full",
                                               "--report-file", "exciton_full.out"]
    assert "--vq-mode" in argv["exciton_bands_full"] and "ongrid" in argv["exciton_bands_full"]
    assert chain.CHECKS["bse_full_dense"]["stdout_twin"] == "bse_full"
    assert chain.CHECKS["exciton_bands_full"]["rows"] == [("exciton_full.dat", 6)]
    assert set(chain._PREPARE) == {"bse_full", "bse_full_dense", "exciton_bands_full"}
