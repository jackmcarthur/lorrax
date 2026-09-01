"""Fail-closed CLI and early-exit contract for basis-only htransform."""
from __future__ import annotations

import inspect

import pytest


def test_basis_only_requires_an_output(capsys):
    from bandstructure import htransform

    with pytest.raises(SystemExit) as exc:
        htransform.main(["--basis-only"])
    assert exc.value.code == 2
    assert "--basis-only requires --basis-output" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--qp-rotations", "qp.h5"], "does not consume"),
        (["--eqp-file", "eqp.dat"], "does not consume"),
        (["--plot"], "does not produce a plot"),
    ],
)
def test_basis_only_refuses_unused_expensive_or_misleading_options(
        tmp_path, capsys, extra, message):
    from bandstructure import htransform

    argv = ["--basis-only", "--basis-output", str(tmp_path / "basis.h5")]
    with pytest.raises(SystemExit) as exc:
        htransform.main(argv + extra)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_basis_only_branch_precedes_every_post_fit_stage():
    from bandstructure import htransform

    source = inspect.getsource(htransform.main)
    branch = source.index("if args.basis_only:", source.index("initialize_wfns("))
    assert branch < source.index("ctilde, B_at_mu =", branch)
    assert branch < source.index("resolve_qp_hamiltonian_state(", branch)
    assert branch < source.index("h_transform(", branch)
    assert branch < source.index("write_bands_to_file(", branch)
    basis_block = source[branch:source.index("ctilde, B_at_mu =", branch)]
    assert "report.basis_only_result(" in basis_block
    assert "_close_wfn_collectively(wfn, log)" in basis_block
    assert 'report.finish(status="basis-only completed")' in basis_block
    assert "return 0" in basis_block


def test_basis_publication_receives_the_input_relative_resolved_path():
    """A caller's cwd must not redirect an input-relative basis output."""
    from bandstructure import htransform

    source = inspect.getsource(htransform.main)
    assert "basis_output_path = (" in source
    assert "basis_output=basis_output_path" in source
    assert "basis_output=args.basis_output" not in source


def test_basis_only_help_names_the_terminal_boundary(capsys):
    from bandstructure import htransform

    with pytest.raises(SystemExit) as exc:
        htransform.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--basis-only" in help_text
    assert "before QP rotation" in help_text
    assert "path eigensolves" in help_text
