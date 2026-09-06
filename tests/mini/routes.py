"""Observe the real fit and algebra calls without changing their arguments."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch


@contextmanager
def observe_routes(*, low_mem_bands):
    import distrib_la
    import file_io
    from gw import isdf_fitting, w_isdf

    events = {"zeta_fits": [], "restart_loads": [], "algebra": [], "dyson": []}
    fit = isdf_fitting.fit_zeta_to_h5
    load_restart = file_io.load_restart_state_from_h5
    batched = distrib_la.Plan.batched
    single = distrib_la.Plan.__call__

    def observed_fit(*args, **kwargs):
        assert kwargs["low_mem_bands"] is low_mem_bands
        events["zeta_fits"].append({
            "low_mem_bands": kwargs["low_mem_bands"],
            "distributed_zeta_solve": kwargs["distributed_zeta_solve"],
        })
        return fit(*args, **kwargs)

    def observed_restart(*args, **kwargs):
        assert kwargs["low_mem_bands"] is low_mem_bands
        events["restart_loads"].append({"file": str(args[0]),
                                       "low_mem_bands": kwargs["low_mem_bands"]})
        return load_restart(*args, **kwargs)

    def record(plan, operand, kind):
        event = {"op": plan.op, "backend": plan.backend,
                 "route": plan.batched_route, "call": kind,
                 "shape": list(operand.shape),
                 "sharding": str(getattr(getattr(operand, "sharding", None), "spec", None))}
        if event not in events["algebra"]:
            events["algebra"].append(event)

    def observed_batched(plan, operand, *args, **kwargs):
        record(plan, operand, "batched")
        return batched(plan, operand, *args, **kwargs)

    def observed_single(plan, operand, *args, **kwargs):
        record(plan, operand, "single")
        return single(plan, operand, *args, **kwargs)

    def observe_solver_builder(builder, route):
        def build(*args, **kwargs):
            solve = builder(*args, **kwargs)

            def observed(operand, *args, **kwargs):
                events["dyson"].append({"route": route, "shape": list(operand.shape),
                                        "sharding": str(operand.sharding.spec)})
                return solve(operand, *args, **kwargs)

            return observed
        return build

    with ExitStack() as patches:
        patches.enter_context(patch.object(isdf_fitting, "fit_zeta_to_h5", observed_fit))
        patches.enter_context(patch.object(file_io, "load_restart_state_from_h5", observed_restart))
        patches.enter_context(patch.object(distrib_la.Plan, "batched", observed_batched))
        patches.enter_context(patch.object(distrib_la.Plan, "__call__", observed_single))
        for route in ("local", "distributed"):
            name = "_get_w_solve_fn_" + route
            patches.enter_context(patch.object(w_isdf, name, observe_solver_builder(
                getattr(w_isdf, name), route)))
        yield events
