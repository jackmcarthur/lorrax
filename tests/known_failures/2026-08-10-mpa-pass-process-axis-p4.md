# AMENDMENT — THE MPA Σ PASS AT P>1, AND THE CELLS THAT NOW COVER IT (2026-08-10) — **CLOSED, with one open gap**

The defect itself is fixed and gated; what stays open is a **structural gap in
the suite**, not a red.  Both halves are recorded here because the second one
outlives the first.

## What was wrong

The MPA Σ pass died on every rank at any process count above one.  Measured on
`integration/mpa-table-2026-08-09` @ `5b8bdbea`, pole 6 of the Si 4×4×4
production deck, `-G=4 -n=4`, a 2×2 mesh over four processes, BFC@0.85:

```
TypeError: cannot reshape array of shape (4, 64, 100) (size 25600)
           into shape (64, 100) (size 6400)
  gw/greens_function_kernel.py   mask = jnp.reshape(mask, enk.shape)
  gw/mpa/sigma_pass.py           run_pass_branch
```

`(64, 100)` is `(n_k, n_band)` and the leading `4` is `jax.process_count()`.
The pass loop gathers its A-side operands to host at their source shape and
then handed them back down to `ppm_windows._build_windows_for_branch` as
`jnp.asarray(host_array)` — a process-local, fully addressable device array —
whose own `process_allgather(tiled=False)` re-invented the axis it had just
stripped.  At one process that axis has length 1 and the reshape in
`build_G_tau` absorbed it silently; at four it has length 4 and the reshape
raises.  The plain two-point Σ path never met it above one process because its
operands are mesh-global — not fully addressable, so the gather is forced to
`tiled=True` and returns the global shape.

Fixed on `fix/mpa-pass-p4-2026-08-10`: a host array is already at its source
shape and is taken as it stands; only a device operand is gathered.

## The cells, and which of them could have caught it

`tests/test_mpa_pass_p4.py`, four cells, all census-class:

| cell | what it needs | would it have caught C1 |
|---|---|---|
| `test_the_pass_loop_survives_a_four_process_gather` | nothing | **yes** |
| `test_the_installed_gather_really_does_prepend_the_process_axis` | nothing | it is the instrument's own control |
| `test_mpa_pass_branch_on_a_2x2_mesh_matches_one_device` | ≥4 devices (`mesh`) | no |
| `test_the_pass_sink_shards_the_band_axes_it_says_it_does` | ≥4 devices (`mesh`) | no |

**THE GAP, and it is the durable part of this row.**  The axis is a function of
`jax.process_count()`, and pytest is ONE process however many devices it can
see.  So a four-DEVICE gate is not a four-PROCESS gate: neither `mesh`-marked
cell above could have caught C1, and no in-suite cell can produce the axis
natively.  The cell that catches it installs the four-process
`process_allgather` semantics verbatim and drives the production planner and
the production consumer through them.  Anyone reading the ≥4-GPU rule as
"four devices under the suite is enough" should read this row first: the only
thing that exercises a multi-process gather natively is a multi-rank driver
leg.

## What is skipped, and until when

The two `mesh`-marked cells **skip under the suite today**.
`tests/conftest.py::pytest_configure` pins every non-controller test process to
one GPU by design, so `jax.device_count()` is 1 at collection.  They were run
by calling them directly on a four-GPU allocation, which is the same workaround
the sixteen-GPU baseline lane used for the two sharded cells in
`tests/test_ppm_crossing_completion.py`.  The `mesh` marker they carry is the
one `fix/conftest-mesh-cells-2026-08-10` is adding; when that lands these cells
run under the suite on a ≥4-device allocation and this paragraph can be struck.

The `mesh` marker is registered in `pyproject.toml` by this branch.  If the
conftest lane ships a different spelling, the rename is one line here and one
line there — the cells are the same cells either way.

Evidence: `/pscratch/sd/j/jackm/mpa_p4fix_0810/` (red twin, post-fix P=4,
post-fix P=1, pre-fix P=1, the direct-cell logs and the two cube comparisons).
