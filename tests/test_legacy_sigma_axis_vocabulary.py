"""One canonical vocabulary for the self-energy axis; the other one warns.

Register row: "config carries two full vocabularies for the self-energy
axis: canonical ``compute_mode`` plus legacy ``do_screened`` /
``use_ppm_sigma`` / ``ppm.model`` / ``self_consistent`` /
``sigma_at_dft_energies`` ... Every deck writes the legacy spelling."

Retiring the legacy keys is a separate decision and has not been taken --
every deck and fixture in the tree writes them, so removing them now would
be a flag day.  This is the WARNING stage, the same migration shape the
tree already uses for ``nband`` -> ``number_bands`` and
``sigma_band_extrapolation`` -> ``use_band_extrapolation``: the key is
honored, and naming it prints what to write instead plus the axis values it
actually resolved to.

Why the note matters and silence does not: a legacy key honored in silence
beside a canonical twin leaves a run log that cannot say which vocabulary
the run went through, which is the same failure the ``ctsp`` ruling names
(decisions.md 2026-08-06) -- a field that parses, normalises, and selects
nothing.

Pure host: exercises the table and the announcer directly, plus a source
gate on the one runtime read that had regressed.
"""
import pathlib

import pytest

from gw.gw_config import (LEGACY_SIGMA_AXIS_KEYS,
                          announce_legacy_sigma_axis_keys)


def _lines(named, mode="gn_ppm", solver="one_shot_dft"):
    out = []
    hit = announce_legacy_sigma_axis_keys(
        named, mode, solver, print_fn=out.append)
    return hit, "\n".join(out)


def test_the_table_names_every_key_the_register_lists():
    assert set(LEGACY_SIGMA_AXIS_KEYS) == {
        "do_screened", "use_ppm_sigma", "ppm_model",
        "self_consistent", "sigma_at_dft_energies"}
    # Every entry points at a CANONICAL spelling, not at another legacy one.
    # The test is on the ASSIGNED KEY, not on any substring: `qp_solver =
    # self_consistent` legitimately contains a legacy key's spelling as the
    # canonical key's VALUE, and forbidding that would forbid the correct
    # advice.
    for key, advice in LEGACY_SIGMA_AXIS_KEYS.items():
        assigned = advice.split("=", 1)[0].strip()
        assert assigned in ("compute_mode", "qp_solver"), (key, advice)
        for other in LEGACY_SIGMA_AXIS_KEYS:
            assert f"{other} =" not in advice, (key, other)


def test_a_deck_that_names_no_legacy_key_prints_nothing():
    """The note must not fire on a canonical deck -- that is the whole point.

    A warning that fires on every run is a warning nobody reads, and it
    would make the canonical spelling look as deprecated as the one it
    replaces.
    """
    hit, text = _lines({"compute_mode", "qp_solver", "nval", "ncond"})
    assert hit == () and text == ""


def test_each_named_legacy_key_gets_its_canonical_replacement():
    hit, text = _lines({"use_ppm_sigma", "ppm_model", "nval"})
    assert set(hit) == {"use_ppm_sigma", "ppm_model"}
    assert "use_ppm_sigma -> write compute_mode = gn_ppm | hl_ppm" in text
    assert "ppm_model -> write compute_mode = gn_ppm | hl_ppm" in text
    assert "nval" not in text


def test_the_note_quotes_the_RESOLVED_axes_not_the_raw_keys():
    """So the log records which vocabulary the run went through.

    Naming the legacy key without saying what it resolved to would leave
    the reader exactly where they started.
    """
    hit, text = _lines({"self_consistent"},
                       mode="mpa", solver="self_consistent")
    assert hit == ("self_consistent",)
    assert "compute_mode = mpa" in text
    assert "qp_solver = self_consistent" in text
    assert "1 LEGACY" in text


def test_the_announcer_accepts_enums_as_well_as_strings():
    from gw.gw_config import ComputeMode, QPSolver

    _, text = _lines({"do_screened"},
                     mode=ComputeMode.COHSEX, solver=QPSolver.ONE_SHOT_DFT)
    assert "compute_mode = cohsex" in text
    assert "qp_solver = one_shot_dft" in text


def test_no_runtime_consumer_reads_the_legacy_ppm_model():
    """``compute_mode.ppm_model`` is the canonical read; ``ppm.model`` is not.

    ``ppm_model`` is consulted ONLY by ``compute_mode = auto``.  On a deck
    with an explicit ``compute_mode`` it selects nothing, so a driver that
    reads ``ppm_cfg.model`` at run time would be pivoting on a key the deck
    may not have meant -- and the Sigma broadening resolver and the sharded
    window precondition both need the ansatz name.  This cell pins that the
    Sigma path takes it from ``compute_mode``.
    """
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "gw"
    ppm_sigma = (src / "ppm_sigma.py").read_text()
    code = "\n".join(l.split("#", 1)[0] for l in ppm_sigma.splitlines())
    assert "ppm_cfg.model" not in code, (
        "the Sigma driver reads the LEGACY ppm_model; take the ansatz from "
        "compute_mode, which every other runtime consumer already uses")
    assert "ansatz: str," in code, "the driver must be TOLD its ansatz"
    pipeline = (src / "ppm_pipeline.py").read_text()
    assert "ansatz=config.compute_mode" in pipeline
