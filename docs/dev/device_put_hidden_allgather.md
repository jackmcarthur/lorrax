# `device_put` onto a multi-process sharding is an all-gather

Status: RECORD + SURVEY. Measured 2026-08-04. **No call site in §6 was
changed by this page** — it is a record of the failure and an inventory of
where the same shape exists today.

The rule, one line: on a multi-process mesh
`jax.device_put(host_array, NamedSharding(mesh, spec))` pays `O(P × array)`
in wall time and in resident memory *before any useful work starts*, because
JAX all-gathers the whole array across every process to assert that every
process passed the same value. Prefer `jax.make_array_from_callback`, or the
in-tree wrapper `common.collectives.device_put_process_local`
(`src/common/collectives.py:462`).

Not new in the tree: `src/common/collectives.py:421-460` derives the cost,
`docs/architecture/memory-model.md:810-822` prices it into the memory floor,
`docs/dev/QUALITY_PATTERNS.md` §5 files it under hidden framework cost. What
this page adds is the 2026-08-04 measurement — in which the antipattern was
misread as a regression in unrelated code — and the survey.

---

## 1. The path

jax 0.9.1 (`lorrax_env/.venv/lib/python3.12/site-packages/jax`).

| step | location |
|---|---|
| entry | `jax/_src/dispatch.py:578` → `:422 _device_put_sharding_impl` |
| operand is eligible | `dispatch.py:483` — `x` is numpy, a python scalar, **or an uncommitted `jax.Array`** |
| every process participates | `dispatch.py:493` — `process_count() == len(s._internal_device_list.process_indices)` |
| the assertion | `dispatch.py:494` — `multihost_utils.assert_equal(x, ...)` |
| the collective | `jax/experimental/multihost_utils.py:179` — `process_allgather(in_tree, tiled=True)` |
| a **second** `P×` buffer | `multihost_utils.py:177` — `np.concat([x] * jax.process_count())` builds the expected tree host-side |

Two `P × x.nbytes` buffers are materialised on every rank, one from the
gather and one from the expectation. The purpose is to verify that all
processes passed the same value. It is a debug assertion, it is silent, it
is P-linear, and it is charged to whichever stage happened to call
`device_put`.

The branch is **not** taken when:

| condition | why |
|---|---|
| `process_count() == 1` | `dispatch.py:493` false |
| target is a `Device`, or `device_put` is called with no second argument | not a `Sharding`; `:493` unreachable |
| target sharding is fully addressable | the multi-process branch at `dispatch.py:482` is not entered |
| operand is a **committed** `jax.Array` | `dispatch.py:483` false — this is a genuine reshard, and reshards do not assert |
| some processes hold no addressable device in the sharding | `dispatch.py:493` false (JAX cannot distinguish subset-meshes from divergent input) |

"Committed" is the load-bearing word. `jnp.zeros(...)`, `jnp.ones(...)` and
`jnp.asarray(numpy)` produce **uncommitted** arrays and take the branch. The
output of a `jax.jit` with explicit `out_shardings`, or of any operation on
committed operands, is committed and does not.

---

## 2. Measured

Harness `tests/multi_device/mtxel_sweep_bench.py`, one arm per launch.
`before` = the production k-partitioned route (`collectives.gather_k_blocks`
over `compute_local_V_k`); `after` = the new 2-D band-sharded k-scan
(`common/mtxel_sweep.py`). Only the `after` arm staged the resident sphere ψ
onto the mesh, and it staged it with `jax.device_put(psi, NamedSharding(...))`.
Peaks are per-rank `VmHWM` from `/proc/self/status`.

| job | P | nb | arm | peak | best |
|---|---|---|---|---|---|
| 7888820 | 16 | 64 | `before` | 1.500 GiB | — |
| 7888820 | 16 | 64 | `after`, ψ staged by `device_put` | **13.031 GiB** | — |
| 7888820 | 16 | 128 | `before` | — | 0.947 s |
| 7888820 | 16 | 128 | `after`, ψ staged by `device_put` | — | **>900 s, killed by slurm** |
| 7888821 | 64 | — | `after`, ψ staged by `device_put` | raised; traceback terminates in `multihost_utils.assert_equal` → `process_allgather` | — |
| 7888868 | 16 | 64 | `after`, ψ staged by `make_array_from_callback` | 1.216 GiB | — |
| 7888868 | 16 | 128 | " | 2.053 GiB | — |
| 7888868 | 16 | 256 | " | 3.732 GiB | — |
| 7888868 | 16 | 512 | " | 7.089 GiB | — |

At nb=64/P=16 that ψ is **0.336 GiB globally and 0.021 GiB/rank once
sharded**. The plan under test never needs more than 0.021 GiB/rank of it.

Closed form, to the extent it reconciles: the assertion's two buffers are
`2·P·|ψ| = 2 · 16 · 0.336 = 10.75 GiB`. The measured excess over the same
shape without the assertion (13.031 − 1.216) is 11.82 GiB, i.e. `35.2 × |ψ|`
against a predicted `2P = 32 × |ψ|`. The residual ≈3 × |ψ| is not separately
attributed and is recorded as unexplained rather than absorbed into
"overhead".

Note the `after` arm at nb=64 is **1.216 GiB, below the `before` arm's
1.500 GiB**, and scales to nb=512 in 7.089 GiB. The 13 GiB and the 900 s
timeout were entirely the staging call.

---

## 3. The cost that actually mattered: false attribution

The benchmark exists to compare two plans for the matrix-element sweep. The
`device_put` was harness plumbing — one line staging an input, in neither
plan's kernel. Its cost was nonetheless charged to the `after` arm's peak and
wall, and the honest reading of the first run was:

> the 2-D band-sharded k-scan uses 8.7× the memory of the k-partitioned
> route it replaces and does not finish at nb=128.

That reading is wrong in every particular, and it is wrong about a code path
the failing call does not touch. It was avoided only because the P=64 leg
(job 7888821) raised with a traceback that named `assert_equal` — i.e. by
luck of which tier failed loudly. At P=16 the same defect produced a large
number and a timeout, both of which look exactly like a real regression.

This is the reason to treat the call as an antipattern rather than merely as
a cost. A P-linear collective attached to an unrelated line does not report
itself as a P-linear collective; it reports itself as *whatever stage is
running*. Compare `memory-model.md:816-822`, where the same assertion is what
made the planner read 0.48× of the measured node peak.

Practical consequence for benchmarking: stage every input with the
collective-free idiom, or the measurement is of the staging.

---

## 4. The correct idiom

Before (`tests/multi_device/mtxel_sweep_bench.py`, the failing form):

```python
sharding = NamedSharding(mesh, P(None, ('x', 'y'), None, None))
psi_j = jax.device_put(psi, sharding)          # psi: host numpy
```

After (`tests/multi_device/mtxel_sweep_bench.py:160-162`):

```python
sharding = NamedSharding(mesh, P(None, ('x', 'y'), None, None))
psi_j = jax.make_array_from_callback(
    psi.shape, sharding, lambda idx: psi[idx])
```

`make_array_from_callback` calls back once per **addressable** shard, so each
process materialises only what its own devices own. No collective, no
transient.

Three variants, pick by what the process already holds:

| the process holds | use | precedent |
|---|---|---|
| the whole global array (identical on every rank) | `jax.make_array_from_callback(shape, sharding, lambda idx: host[idx])` | `tests/multi_device/mtxel_sweep_bench.py:161` |
| only its own local slice | `jax.make_array_from_process_local_data(sharding, local, global_shape)` | `src/bse/bse_io.py:489` |
| the whole global array, inside `src/` | `common.collectives.device_put_process_local(host, sharding)` | `src/common/collectives.py:462` |

Inside `src/`, prefer the wrapper. It resolves the single-process and
fully-addressable cases back to plain `device_put`, passes an already-sharded
operand straight through as a reshard, handles the process-owns-no-device
case, and keeps `LORRAX_CHECK_REPLICA=1` as a way to re-arm JAX's assertion
for a debugging run. Its correctness precondition is the one `device_put` was
spending `2P × |x|` to check: **the host array must be identical on every
process.**

---

## 5. Where `device_put` is still the right call

This is an antipattern to avoid where it applies, not a ban. `device_put` is
correct and idiomatic in all of:

| case | why it is fine |
|---|---|
| single-process runs | `dispatch.py:493` is false; no assertion |
| `device_put(x, jax.local_devices()[0])` and friends | a `Device` target never reaches the multi-process branch |
| `device_put(x)` with no target | same |
| resharding an already-sharded `jax.Array` | committed operand, `dispatch.py:483` false — this is the intended use |
| small replicated scalars, index tables, config arrays | `2P × nbytes` of a few KB is not worth an idiom change; `src/runtime/__init__.py:938` and `src/bse/bse_ring_comm.py:147` both take the branch deliberately and cost `O(P²)` *bytes* |

The threshold is size × P, not the call itself. The failure mode is that
nothing in the call site says which side of the threshold it is on.

---

## 6. Survey of `device_put` in `src/`

Every `device_put` call in `src/` as of `22fa0aa`: 35 sites, of which 20 pass
a `Sharding`. Verdicts are on the operand's commitment state and on the size
of the array, per §1.

**Takes the branch today (2 sites, both intentional and negligible):**

| file:line | operand / shape | verdict |
|---|---|---|
| `src/runtime/__init__.py:938` | `jnp.ones((px,py))` and `jnp.ones((n_ax,))` f64, uncommitted | Benign. Collective warm-up; the assertion gathers `P × ≤P × 8` bytes and is itself part of what is being warmed. |
| `src/bse/bse_ring_comm.py:147` | `jnp.zeros(n_devices)` f64, uncommitted | Benign. This function exists to warm `process_allgather`; the assertion runs one. |

**Does not take the branch — committed operand (genuine reshard):**

| file:line | array class | verdict |
|---|---|---|
| `src/isdf/core.py:1944` | `(nq, n_μ, n_μ)` — μ² | Benign. Parts are outputs of the `jax.jit(..., out_shardings=out_sh)` at `core.py:1858`, so the concat is committed. **μ²-class: re-check commitment before editing this line.** |
| `src/isdf/core.py:2313` | `(nq, n_μ, n_μ)` — μ² | Same, from `core.py:2294`'s `out_shardings=(out_sh, ...)`. Same caution. |
| `src/isdf/core.py:2315` | `(nq, n_log)` int32 perm | Benign; small and committed. |
| `src/bandstructure/htransform.py:312` | `(nk·nb)²`, N_μ-free | Benign. `V` is an `eigh_plan` output. |
| `src/bandstructure/htransform.py:689` | `(nk, nb, rank)` | Benign. From the `@jax.jit _finalize` at `htransform.py:681` on committed operands. |
| `src/bandstructure/htransform.py:728` | `(rank, nk·nb)` replicated, ≲ tens of MB | Benign. Eager `solve_triangular` on committed operands. |
| `src/bandstructure/htransform.py:1358` | `(rank, rank)` ≈7 MB | Benign. Slice of a committed sharded array. |
| `src/bandstructure/bse_setup.py:357` | `(rank, ns, n_μ)` replicated | Benign. See the comment at `bse_setup.py:350`: the `fH_R` version of this same line was **not** benign (11 GiB at nb=16 rising to 50 GiB) and was removed. |
| `src/bse/exciton_bands.py:814` | `(n_μpad, n_μpad)` — μ² | Benign. `data["V_q0"]` is committed on both producer paths (`bse_io.py:811`, `bse_io.py:981`). |
| `src/bse/exciton_bands.py:827` | μ² slice | Benign. Slice of `_read_wq_sharded`'s committed output (`bse_io.py:990`). |
| `src/bse/exciton_bands.py:866` | `(nQ, n_μpad, n_μpad)` — μ² | Benign. `V_rows` are all committed device arrays. |
| `src/bse/bse_io.py:811` | `(n_μpad, n_μpad)` — μ² | Benign. `eval_vq` is `jax.jit(..., out_shardings=grid_out)` (`vq_interp.py:1175`), so its output is committed. |

**Does not take the branch — guarded or structurally excluded:**

> Line numbers below are at the head this audit ran on. `wfn_loader.py`
> has since moved to `services/wfn_loader/src/wfn_loader/loader.py`
> (2026-08-07; `src/file_io/wfn_loader.py` no longer exists) — the guard and the
> verdict are unchanged, the line numbers are not.
> See [docs/services/wfn_loader.md](../services/wfn_loader.md).

| file:line | verdict |
|---|---|
| `src/file_io/wfn_loader.py:1259` | **Largest operand in the tree** — the whole numpy ψ window, `nk·nb·ns·ngkmax·16`. Unreachable at `P>1`: the guard at `wfn_loader.py:1243` routes multi-process to `_eager_build_process_local`. Benign **only because of that guard**; it is the one line where the guard is all that stands between the tree and a worst-case gather. |
| `src/common/collectives.py:508` | Inside the wrapper. Operand is an already-sharded non-fully-addressable `jax.Array` — reshard branch. |
| `src/common/collectives.py:516` | Inside the wrapper. Reached only at `P==1` or on a fully addressable target. |
| `src/common/collectives.py:529` | Inside the wrapper. The **opt-in** `LORRAX_CHECK_REPLICA=1` path — this one *is* the all-gather, deliberately. Off by default (`collectives.py:519-527`). |
| `src/common/collectives.py:537` | Inside the wrapper. This process owns no addressable device in the sharding, so JAX skips the assertion too. |

**Structurally out of reach — `Device` target or no target (15 sites):**
`collectives.py:539,620,693`; `_slab_io_allgather.py:385`;
`_slab_io_mpi_host.py:624`; `wfn_loader.py:781,1325`;
`ppm_accumulators.py:421,426`; `ppm_sigma.py:1078,1135`;
`bse_io.py:488,593,644`; `w_omega_chain.py:325`.

### The one hazard worth naming

`src/gw/qsgw_utils.py:617`:

```python
sig_x_rep = jax.device_put(jnp.asarray(sig_x),
    NamedSharding(mesh_xy, P(None, None, None)))
```

`jnp.asarray` is the identity on a committed `jax.Array`, so this is benign
**exactly as long as every producer of `SigmaResult.sigma_x_kij_ry` returns a
committed array**. Today they all do — `gw/cohsex_sigma.py:82` and `:198` and
`gw/sigma_dispatch.py:413` place it with `device_put_process_local`, and
`gw/sc_iteration.py:809` derives it from those. If any producer ever hands
back host numpy, `jnp.asarray` yields an uncommitted array, the branch fires,
and the target is a **replicated** `(nk, nb, nb)`: `P × nk·nb²·16` twice
over — 9.2 GB × P at nb=2000 — with no symptom but memory.

It is the only site in `src/` whose safety is inherited from a caller rather
than established locally. Not changed here (this page is a record), but it is
the first line to convert if the survey is ever acted on: swapping it for
`device_put_process_local` makes the safety local and costs nothing.

---

## 7. See also

| where | what it covers |
|---|---|
| `src/common/collectives.py:421-460` | the derivation, with the WFN loader's 17.4 GB/rank at P=144 |
| `src/common/collectives.py:462` | `device_put_process_local` — the wrapper and its precondition |
| `docs/architecture/memory-model.md:810-822` | why the two staged loader tables stay in the memory floor |
| `docs/dev/QUALITY_PATTERNS.md` §5 | the hidden-framework-cost class this belongs to |
| `docs/architecture/multihost.md:473-522` | JAX's own guidance and the error text for the non-addressable case |
| `docs/dev/matrix_element_sweep_handoff.md` | the sweep the 2026-08-04 benchmark was actually measuring |

Not in the mkdocs nav: `mkdocs.yml:26-28` excludes `docs/dev/` from the site
build, and `docs/dev/` has no index page.
