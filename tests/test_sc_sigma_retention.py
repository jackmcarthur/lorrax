"""The SC loop must not hold two ω-cubes at once.

``SigmaResult.sigma_c_omega_kij_ry`` is ``(nω, nk, nb, nb)`` and, at the
default ``sigma_omega_layout = "replicated"``, it is replicated on every
rank: ``gw_config.py`` prices it at 2751 MB/rank at nb=512, and that
figure does not shrink with P.  Both accelerator drivers used to keep
iteration *i−1*'s ``SigmaResult`` alive for the whole of iteration *i* —
rCROP held it in a closure cell AND passed it into the input state,
linear mixing passed the loop carry itself — so the peak was two of
them, a P-independent doubling of the largest object on the surface.

Nothing in the loop reads it.  ``SCState``'s own docstring says so
("purely for the final output writers; it does not feed the next
iteration"); the only consumers are ``dump_sigma_omega_h5_final`` and
``run_sc_driver``'s finalize, and both want the LAST one, which still
survives.

That "nothing reads it" is the load-bearing claim, so it is the
assertion: ``gw_iteration_map`` may touch ``state.iteration`` and
``state.H_qp_dft`` and nothing else.  A later change that made the map
depend on the previous Σ would silently make the drop wrong, and would
fail here first.

AST plus source text, no jax, so it runs anywhere.
"""
import ast
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest


_PATH = (pathlib.Path(__file__).resolve().parents[1]
         / "src" / "gw" / "sc_iteration.py")
_SRC = _PATH.read_text()
_TREE = ast.parse(_SRC)


def _func(name):
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_PATH}")


def _block(name):
    """Source text of a top-level ``def``, up to the next top-level one."""
    start = _SRC.index(f"\ndef {name}(") + 1
    nxt = _SRC.find("\ndef ", start + 1)
    return _SRC[start: nxt if nxt != -1 else len(_SRC)]


def _state_attrs(fn, var="state"):
    return {n.attr for n in ast.walk(fn)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == var}


# The carried state one QSGW map call READS: the QP Hamiltonian, the
# counter, the previous protected-band decision, and the previous call's
# OccupationState — the last of these for the mu-drift diagnostic ONLY.
# Since the entry-solve rule (2026-08-15)
# the map solves its own occupations from the spectrum of the H it is
# handed, so neither correctness nor the head consumes the carry;
# head_surface_weight_kn is carried for continuity but never read here.
_CARRY_KEYS = {
    "iteration", "H_qp_dft", "partition", "occupation_state",
}

# What a bare INPUT SCState is constructed with in the drivers: the read
# set above plus head_surface_weight_kn, which rides along for carry
# continuity (the map re-derives it at entry and never reads the carry).
_STATE_INPUT_KEYS = _CARRY_KEYS | {"head_surface_weight_kn"}


def test_gw_iteration_map_reads_only_the_carry_and_the_counter():
    assert _state_attrs(_func("gw_iteration_map")) == _CARRY_KEYS


@pytest.mark.parametrize("driver", ["_run_rcrop", "_run_linear_mixing"])
def test_no_driver_feeds_a_stale_sigma_result_into_the_map(driver):
    """Every ``SCState(...)`` built as a map ARGUMENT carries H and i only.

    The finalize state at the end of ``_run_rcrop`` legitimately carries
    the last ``SCOutputs``, so the check is on the constructions whose
    keywords are exactly the two fields the map reads — there must be at
    least one — and on every other construction still naming ``outputs``.
    """
    fn = _func(driver)
    inputs, finals = [], []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "SCState":
            keys = {kw.arg for kw in node.keywords}
            (inputs if keys == _STATE_INPUT_KEYS else finals).append(keys)
    assert inputs, f"{driver} builds no bare input SCState"
    for keys in finals:
        assert "outputs" in keys, (
            f"{driver} builds an SCState with {sorted(keys)} — an argument "
            f"to gw_iteration_map must carry exactly the carry fields")


def test_rcrop_clears_the_capture_cells_before_the_map_call():
    """The cell must be emptied BEFORE ``gw_iteration_map``, not after.

    Assigning after the call is what the previous code did, and that is
    the whole defect: the reference stays live for the entire build of
    the next Σ.
    """
    body = _block("_run_rcrop")
    clear = body.index("_last_outputs[0] = None")
    call = body.index("gw_iteration_map(")
    assert clear < call


def test_linear_mixing_rebuilds_the_carry_before_the_map_call():
    body = _block("_run_linear_mixing")
    rebuild = body.index("state = SCState(")
    assert "H_qp_dft=state.H_qp_dft" in body[rebuild:rebuild + 400]
    call = body.index("gw_iteration_map(")
    assert rebuild < call


def test_the_omega_cube_is_in_the_residency_census():
    """The saving has to be measurable, or the claim is unverifiable."""
    body = _block("gw_iteration_map")
    census = body[body.index("_residency_census("):]
    assert '"sigma_c_omega_kij_ry"' in census


def test_sc_output_lifecycle_has_one_owner_per_large_artifact():
    body = _block("gw_iteration_map")
    assert "retain_iteration_artifacts(" in body
    assert body.index("_check_sigma_stage(") < body.index(
        "retain_iteration_artifacts(")

    dump = _block("dump_qp_wfn_artifacts")
    assert "enk_full_base_ry=enk_full_base_ry" in dump


def test_rcrop_preserves_output_metadata_from_the_last_map():
    body = _block("_run_rcrop")
    assert "_last_outputs[0] = None" in body
    assert "_last_outputs[0] = state_out.outputs" in body
    assert "outputs=_last_outputs[0]" in body


def test_linear_mixing_returns_the_last_evaluated_input_not_mixed_candidate(
    monkeypatch,
):
    """Distinct sentinels pin H/output ownership at budget exhaustion."""
    jnp = pytest.importorskip("jax.numpy")
    from gw import sc_iteration

    payload = object()

    def _map(state, _inputs):
        return sc_iteration.SCState(
            H_qp_dft=state.H_qp_dft + 2.0,
            iteration=state.iteration + 1,
            occupation_state="occupation-from-input-zero",
            head_surface_weight_kn="surface-from-input-zero",
            outputs=payload,
        )

    monkeypatch.setattr(sc_iteration, "gw_iteration_map", _map)
    monkeypatch.setattr(sc_iteration, "_write_sc_eqp_snapshot",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sc_iteration, "_maybe_dump_e_history",
                        lambda *_args, **_kwargs: None)
    inputs = SimpleNamespace(partition=SimpleNamespace(
        protected_mask=np.array([True]), in_range_mask=np.array([True])))
    initial = sc_iteration.SCState(
        H_qp_dft=jnp.zeros((1, 1, 1), dtype=jnp.complex128), iteration=0)
    final, _ = sc_iteration._run_linear_mixing(
        initial, inputs, max_iter=1, tol_ev=1.0e-12, mixing=0.25,
        eigvalsh_kshard=lambda H: jnp.real(H[..., 0]),
        print_fn=lambda *_args: None, dump_dir=None)

    np.testing.assert_array_equal(np.asarray(final.H_qp_dft), 0.0)
    assert final.outputs is payload
    assert final.occupation_state == "occupation-from-input-zero"
    assert final.head_surface_weight_kn == "surface-from-input-zero"


def test_rcrop_early_stop_binds_outputs_to_the_accepted_map_input():
    """AST red twin for the exception path hidden inside rcrop_nojit."""
    fn = _func("_run_rcrop")
    converged = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_Converged"]
    assert len(converged) == 1
    state_call = converged[0].args[0]
    values = {kw.arg: kw.value for kw in state_call.keywords}
    assert isinstance(values["H_qp_dft"], ast.Name)
    assert values["H_qp_dft"].id == "H"
    assert {"occupation_state", "head_surface_weight_kn", "outputs"} <= values.keys()


def test_per_map_files_are_output_diagnostics_not_final_eqp_math():
    body = _block("_write_sc_eqp_snapshot")
    assert "write_bgw_eqp(" in body
    assert "assemble_eqp" not in body
    assert "output_candidate_active_scissor" in body
    assert "input_tail_scissor" in body
    # The rows are the FILE wedge (wfn.kpoints), like every other .dat.
    # This used to assert ``"kirr_fullids" in body`` — the same fact
    # spelled as the index table the writer gathered by.  It now goes
    # through the service instead (2026-08-15): the table itself is the
    # service's business, and under ``sc_on_ibz`` the loop's own rows are
    # the STAR wedge, which is a different length (5 vs 9 on gnppm_debug).
    # See tests/test_sc_on_ibz_wedges.py.
    assert "reduce_full_bz_to_file_wedge" in body


def test_history_and_fixed_head_serial_writes_are_rank_gated():
    history = _block("_maybe_dump_e_history")
    assert "process_rank() == 0" in history
    assert history.index("process_rank() == 0") < history.index("np.save(")
    assert "barrier(" in history

    # The head fit is COLLECTIVE now (every rank enters SlabIO), so the
    # rank gate moved from the model to the writer: model.py must route
    # the head only through write_head_fit_collective, never through the
    # serial h5py test seam write_head_fit.
    model_src = (_PATH.parents[2] / "src" / "gw" / "mpa" / "model.py")
    text = model_src.read_text()
    assert "write_head_fit_collective(" in text
    assert "write_head_fit(" not in text.replace(
        "write_head_fit_collective(", "")
