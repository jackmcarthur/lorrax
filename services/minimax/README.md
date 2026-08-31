# minimax service

`minimax` serves scalar quadrature artifacts and physics-free fitting tools.
LORRAX consumers import from the package door:

```python
import minimax
quad = minimax.lookup(
    family="noncrossing", target="inverse",
    range_value=100.0, error_bound=1.0e-6, n_max=64,
)
```

Do not import service submodules from LORRAX. `lookup` only selects certified
catalog entries and refuses an uncovered request. `serve` has the same lookup
path, followed on a miss by the announced uncertified runtime-solve policy
controlled by `LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE`. Cache location and disable
policy are `LORRAX_MINIMAX_CACHE_DIR` and
`LORRAX_DISABLE_MINIMAX_DISK_CACHE`. Cached solves are not certified assets.

## Measure-adapted ROQ

The delivered Sigma planner uses the lazy top-level ROQ API:

```python
windows = (
    minimax.RoqWindow(name, fit, validation, target, branch, sigma),
)
plan = minimax.plan_measure_adapted_roq(windows, eta)
```

`fit` and `validation` are `ReciprocalMeasureProblem` objects. `target` is the
apportioned delivered residual for that product window; `sigma` is its
half-plane sign (`+1` or `-1`). `eta` is the only contour-scale input.
`RoqPlan.rules` contains the nodes, weights, validation error, amplification,
and derived contour data. `RoqPlan.branches` contains the aggregate delivered
error/noise evidence and selected strategy.

The planner derives contour, horizon, rank, and grouping. It fits
decay-compatible product groups, tests a whole-branch consolidation only when
it can beat the split node count, and falls back to individual product windows.
Failure is a refusal; there is no explicit state--pole path. The mathematical
and acceptance contracts belong to
[`docs/theory/sigma-quadrature-problem.md`](../../docs/theory/sigma-quadrature-problem.md).

Run the service tests from the repository root:

```bash
uv run python -m pytest -q services/minimax/tests
```
