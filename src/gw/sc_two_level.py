"""Two-level QSGW orchestration around the existing iteration map.

The outer map rebuilds chi0, W, and the pole model once.  Its inner solve
then calls :func:`gw.sc_iteration.gw_iteration_map` with that screening
transaction frozen, so only the current Green function/orbitals and Sigma are
rebuilt.  Each expensive Sigma rebuild is followed by the existing
fixed-Sigma-table evSC engine, which converges rotations and energies before
another convolution is allowed.  No screening or self-energy equation is
duplicated here.

``sc_max_iter`` deliberately bounds both the number of outer refits and each
inner accelerator solve.  Reusing the existing cap avoids a second convergence
dial; ``sc_max_iter = 1`` remains the historical one-map diagnostic exactly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from common.units import RYD_TO_EV


def _carry_only(state, sc_iteration):
    """Drop output-sized artifacts before the next map allocation."""
    return sc_iteration.SCState(
        H_qp_dft=state.H_qp_dft,
        iteration=int(state.iteration),
        partition=state.partition,
        occupation_state=state.occupation_state,
        head_surface_weight_kn=state.head_surface_weight_kn,
    )


def _eigenvalues_ev(H_qp_dft, eigvalsh_kshard):
    """QP eigenvalues ``(nk, nb)`` in eV for one DFT-basis Hamiltonian."""
    return np.asarray(eigvalsh_kshard(H_qp_dft), dtype=np.float64) * RYD_TO_EV


def _verdict(state_out, state_in, inputs, eigvalsh_kshard, tol_ev,
             sc_iteration):
    """Protected-state residual of one outer or first-inner map."""
    partition = sc_iteration._state_partition(state_out, inputs)
    verdict = sc_iteration.protected_band_convergence(
        _eigenvalues_ev(state_out.H_qp_dft, eigvalsh_kshard),
        _eigenvalues_ev(state_in.H_qp_dft, eigvalsh_kshard),
        np.asarray(partition.protected_mask, dtype=bool),
        np.asarray(partition.in_range_mask, dtype=bool),
        float(tol_ev),
    )
    return sc_iteration._include_fixed_table_verdict(
        state_out, verdict, inputs)


def _cost_receipt(inputs):
    """Print the measured expensive-evaluation count and walls once."""
    ledger = getattr(inputs, "two_level_cost", None)
    if ledger is None:
        return
    walls = [float(value) for value in ledger.get("sigma_walls_s", ())]
    total = float(sum(walls))
    inputs.print_fn(
        "[ SC two-level cost | "
        f"W_refits={int(ledger.get('w_refits', 0))}, "
        f"Sigma(omega)_evaluations={len(walls)}, "
        f"Sigma_wall_total={total:.6f} s, "
        f"Sigma_wall_mean={(total / len(walls) if walls else 0.0):.6f} s, "
        f"Sigma_wall_each={walls} ]")


def run_two_level_self_consistency(
    state_init,
    inputs,
    *,
    max_iter: int,
    tol_ev: float,
    accelerator: str,
    history_depth: int,
    mixing: float,
):
    """Converge QSGW with frozen-screening inner solves and outer W refits.

    Parameters
    ----------
    state_init : gw.sc_iteration.SCState
        Initial ``(nk, nb, nb)`` QP Hamiltonian in the DFT basis.
    inputs : gw.sc_iteration.SCInputs
        Immutable driver bundle.  It must not already request frozen
        screening; this routine owns that transition at the outer boundary.
    max_iter : int
        Both the outer-refit cap and the cap passed to every inner solve.
        A value of one takes the established one-map path without any
        two-level wrapping, preserving one-shot parity.
    tol_ev : float
        L-infinity protected-state cutoff for both inner and outer changes.
    accelerator, history_depth : str, int
        Existing inner-solve rCROP/linear controls.  A fresh invocation is
        made after every W refit, which resets rCROP history by construction;
        the same controls bound each innermost fixed-table cycle.
    mixing : float
        Existing ``sc_mixing`` coefficient.  It remains the Picard damping
        for a linear inner solve and also linearly mixes the outer Hamiltonian
        before the next W refit.  Convergence is tested on the unmixed outer
        candidate, so damping cannot manufacture convergence.

    Returns
    -------
    state_final : gw.sc_iteration.SCState
        Last evaluated state.  Its output transaction and Hamiltonian refer
        to the same inner-map input.
    residual_history_ev : list[float]
        RMS energy displacement for every evaluated map, flattened over the
        outer steps.  The L-infinity verdict printed beside each map remains
        the stopping authority.
    """
    # Local import avoids making sc_iteration import itself at module load;
    # the public driver imports this orchestrator only at its call site.
    from . import sc_iteration

    if int(max_iter) < 1:
        raise ValueError("two-level QSGW requires max_iter >= 1")
    if inputs.frozen_screening is not None:
        raise ValueError(
            "run_two_level_self_consistency owns the frozen-screening "
            "boundary; initial SCInputs.frozen_screening must be None")

    # Exact historical route, including its output ownership and map-1
    # half-sum update law.  This branch is the iteration-1 parity contract.
    if int(max_iter) == 1:
        return sc_iteration.run_self_consistency(
            state_init, inputs, max_iter=1, tol_ev=tol_ev,
            accelerator=accelerator, history_depth=history_depth,
            mixing=mixing)

    _, eigvalsh_kshard = sc_iteration._kshard_eigh_kernels(inputs.mesh_xy)
    outer_input = _carry_only(state_init, sc_iteration)
    residual_history: list[float] = []
    snapshot_offset = 0
    last_inner = None
    last_outer_verdict = None

    inputs.print_fn(
        "[ SC two-level | outer chi0/W/pole refit; inner frozen screening "
        f"| max_outer={max_iter}, max_inner={max_iter}, "
        f"tol={tol_ev:.3e} eV | outer alpha={mixing:.3f}, pole_refit="
        f"{inputs.config.sc.outer_refit_policy} ]")

    for outer_index in range(int(max_iter)):
        outer_number = outer_index + 1
        inputs.print_fn(
            f"[ SC outer {outer_number}/{max_iter} | rebuilding "
            "chi0 -> W -> pole model from the current outer H ]")

        # One ordinary map is both the outer screening refit and the first
        # evaluation F_W(H) of the new inner problem.  Reusing its result
        # avoids paying a duplicate Sigma build merely to obtain W.
        # One quadrature transaction per outer model.  The frozen pole census
        # cannot drift inside this solve, so it needs only a small state-edge
        # allowance and no pole padding.  The next outer refit gets a fresh
        # session and is therefore free to replan for its new pole set.
        outer_rule_session = {
            "state_edge_padding_ev": 0.25,
            "pole_extent_padding_fraction": 0.0,
        }
        live_inputs = replace(
            inputs,
            frozen_screening=None,
            fixed_quadrature_session=outer_rule_session,
        )
        first_map, _ = sc_iteration.run_self_consistency(
            outer_input, live_inputs,
            max_iter=1, tol_ev=tol_ev,
            accelerator=accelerator, history_depth=history_depth,
            mixing=mixing,
            reset_diagnostics=(outer_index == 0),
            snapshot_offset=snapshot_offset,
        )
        ledger = getattr(inputs, "two_level_cost", None)
        if ledger is not None:
            ledger["w_refits"] = int(ledger.get("w_refits", 0)) + 1
        snapshot_offset += 1
        if first_map.outputs is None:
            raise RuntimeError(
                "GATE sc_outer_screening_missing: the outer map returned "
                "no output transaction")
        screening = first_map.outputs.screening
        if screening.sigma_model is None:
            raise RuntimeError(
                "GATE sc_outer_screening_model_missing: the outer map did "
                "not retain the W/pole model consumed by Sigma")

        first_verdict = _verdict(
            first_map, outer_input, inputs, eigvalsh_kshard, tol_ev,
            sc_iteration)
        e_first = _eigenvalues_ev(first_map.H_qp_dft, eigvalsh_kshard)
        e_outer = _eigenvalues_ev(outer_input.H_qp_dft, eigvalsh_kshard)
        first_rms = float(np.sqrt(np.mean((e_first - e_outer) ** 2)))
        residual_history.append(first_rms)
        inputs.print_fn(
            f"  SC outer {outer_number} inner map 1 (screening producer): "
            f"RMS dE={first_rms:.6f} eV; {first_verdict.summary()}")

        frozen_inputs = None
        inner_seed = None
        if first_verdict.converged:
            # The outputs describe F_W evaluated AT outer_input.  Bind them
            # to that evaluated input, not to the unevaluated F_W(H), exactly
            # like the incumbent accelerated runners do on convergence.
            inner_final = replace(
                first_map,
                H_qp_dft=outer_input.H_qp_dft,
                convergence_verdict=first_verdict,
            )
            inner_calls = 1
        else:
            frozen_inputs = replace(
                inputs,
                frozen_screening=screening,
                fixed_quadrature_session=outer_rule_session,
            )
            inner_seed = _carry_only(first_map, sc_iteration)
            inner_final, inner_history = sc_iteration.run_self_consistency(
                inner_seed, frozen_inputs,
                max_iter=int(max_iter), tol_ev=tol_ev,
                accelerator=accelerator, history_depth=history_depth,
                mixing=mixing,
                reset_diagnostics=False,
                snapshot_offset=snapshot_offset,
            )
            snapshot_offset += len(inner_history)
            residual_history.extend(inner_history)
            inner_calls = 1 + len(inner_history)

        inner_verdict = inner_final.convergence_verdict
        if inner_verdict is None:
            raise RuntimeError(
                "GATE sc_inner_convergence_verdict_missing: frozen-W "
                f"inner solve {outer_number} returned no verdict")
        if not inner_verdict.converged:
            inputs.print_fn(
                f"[ SC outer {outer_number} | INNER NOT CONVERGED after "
                f"{inner_calls} map calls; outer W will not be refitted | "
                f"{inner_verdict.summary()} ]")
            _cost_receipt(inputs)
            return inner_final, residual_history

        # Outer residual is the unmixed fixed point at the newly fitted W
        # against the H that produced that W.  sc_mixing is applied only
        # after this test, so alpha < 1 cannot make the verdict look green.
        last_outer_verdict = _verdict(
            inner_final, outer_input, inputs, eigvalsh_kshard, tol_ev,
            sc_iteration)
        inputs.print_fn(
            f"[ SC outer {outer_number} | inner converged in {inner_calls} "
            f"map calls | outer {last_outer_verdict.summary()} ]")
        last_inner = replace(
            inner_final, convergence_verdict=last_outer_verdict)
        if last_outer_verdict.converged:
            inputs.print_fn(
                f"[ SC two-level CONVERGED after {outer_number} W refits "
                f"and {snapshot_offset} total map calls ]")
            _cost_receipt(inputs)
            return last_inner, residual_history

        if outer_number < int(max_iter):
            H_next = (
                float(mixing) * inner_final.H_qp_dft
                + (1.0 - float(mixing)) * outer_input.H_qp_dft)
            inputs.print_fn(
                f"  SC outer mixing: alpha={float(mixing):.3f}; next W "
                "refit uses (1-alpha) H_in + alpha H_inner*")
            outer_input = sc_iteration.SCState(
                H_qp_dft=H_next,
                iteration=int(inner_final.iteration),
                partition=inner_final.partition,
                occupation_state=inner_final.occupation_state,
                head_surface_weight_kn=inner_final.head_surface_weight_kn,
            )

        # Release the old W role table before the next outer producer builds
        # its replacement.  ``outer_input`` above retains only the bounded H
        # carry; MPA's previous disk store is deleted by its next live map
        # after the replacement has passed the Sigma gates.
        first_map = None
        screening = None
        if outer_number < int(max_iter):
            last_inner = None
            inner_final = None
            frozen_inputs = None
            inner_seed = None

    if last_inner is None or last_outer_verdict is None:
        raise RuntimeError("two-level QSGW completed no outer map")
    inputs.print_fn(
        f"[ SC two-level NOT CONVERGED after {max_iter} W refits | "
        f"{last_outer_verdict.summary()} ]")
    _cost_receipt(inputs)
    return last_inner, residual_history


__all__ = ["run_two_level_self_consistency"]
