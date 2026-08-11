# `refit_vq` at P>1, and a second named certification grade (2026-08-11)

**Base `origin/main` `9c70b5a3`. Branch
`fix/refit-shard-and-cert-grade-2026-08-11`, pushed, UNMERGED. Workspace
`/pscratch/sd/j/jackm/refitshard_0811/`, sole writer.** Three jobs: fix
`SMALL_ISSUES` row 39 with a red twin, add a visualization certification
grade, and run the owner's dense exciton band structures on the un-downfolded
μ=960 parent.

## 1. The defect, and why parity decided it

`bse/vq_interp.py:2778` fetched the per-Q ζ'(G) rows to host with a bare
`jax.device_get`. ζ inherits its μ-axis sharding from `bse_setup.psi_rmu_Y`,
which is laid out `P(None, None, None, 'y')`, so at P>1 that array is spread
across processes and the fetch raises.

What made it survive months of P=4 legs is that `common.sharding_fit` silently
drops a mesh axis the extent cannot divide, and **`jax`'s `Array._value`
serves a fully REPLICATED array out of the local shard before it ever reaches
the addressability check**:

| n_μ | `sharding_fit` | layout at P=4 | `device_get` |
|---|---|---|---|
| 191 (downfolded child) | `191 % 2 != 0` → drops `'y'` | replicated | **works**, via the replicated fast path |
| 960 (parent) | divides → keeps `'y'` | genuinely sharded | **raises** |

Every P=4 leg the refit path had ever had was the odd-μ child. The coverage
was not thin; it was blind by construction, and no amount of it could have
found this.

**The fix is `_to_host` (= `common.collectives.gather_to_host`)**, whose three
arms are exactly the three layouts: `device_get` when fully addressable
(P=1), `addressable_data(0)` when replicated (the odd-μ arm — still no
collective, so the child arm gains nothing to pay for), and
`process_allgather(tiled=True)` when genuinely sharded. Placement only; the
tile contraction below it is untouched arithmetic. It is the sibling of the
`refit_prepare` fix the BSE-window lane landed and missed here.

One import moved with it: `compute_wfns_fi` is now imported inside the
htransform branch that uses it instead of at function scope. The `"stored"`
leg never calls it, and importing it drags in `bandstructure.htransform`,
whose module body runs `initialize_communicator_stack()` and the required-FFI
gate — which is what made a four-process twin unrunnable anywhere but a
compute node.

## 2. The red twin — even n_μ, four PROCESSES

`tests/test_refit_vq_shard_p4.py` drives `tests/_refit_shard_twin.py` through
`tests/mesh_launch.py`. The twin builds a synthetic `zx`/`rst` with
production's shapes, takes the μ spec from `sharding_fit` exactly as
`bse_setup` does, and runs the real `refit_vq` at n_μ = 8 (even, shards) and
n_μ = 7 (odd, replicates).

| arm | tree | shape | result |
|---|---|---|---|
| `twin_pre` | `9c70b5a3` + the twin files | 4 real A100s, 4 processes, 1 GPU each | **RED** — dies at `tree_base/src/bse/vq_interp.py:2778`, `RuntimeError: Fetching value for jax.Array that spans non-addressable (non process local) devices`, the production traceback verbatim |
| `twin_post` | `93f8b572` | same | **6 passed** in 16.6 s |

Six cells, and three of them exist so the green cannot be vacuous:

* **the even arm must really span processes** — the twin records the ζ'(G)
  box's own `is_fully_addressable` / `is_fully_replicated` and the cell
  refuses anything that came back addressable or replicated. Without it a
  green would mean only that nothing sharded, which is the exact failure mode
  of the coverage this replaces;
* **the odd arm must still be REPLICATED** — a "fix" that pushed it through
  `process_allgather` would be correct and would quietly add a collective to
  the arm the production child takes;
* **every rank's tile must be byte-identical** — a gather that returned this
  process's shard is Hermitian, right-shaped, and wrong.

Plus: sharded tile == replicated tile to < 1e-12 relative (placement only).

**Trap worth inheriting: the μ sharding survives to ζ on XLA:GPU and does NOT
on XLA:CPU.** An emulated four-device CPU mesh replicates at the Cholesky in
`_solve_zeta` and shows `P()` all the way to `ztG_box` — measured here across
five shape regimes including production's exact ratios. A CPU twin would have
gone green on the broken tree. This needed a real four-GPU leg.

## 3. The second grade

`CERT_TOL_VISUALIZATION_MEV = 1.0` beside `REFIT_CERT_TOL_MEV = 0.01`, both
module constants, selected by NAME through `CERT_TOL_BY_GRADE` with
`--cert-grade=reference|visualization` (default `reference`, unchanged).

The concession is the number and never the refusal: the dual-solve
certification runs identically and raises above whichever grade it was given.
`_certify_refit_against_stored` takes a grade key rather than a float, so
there is no parameter through which a caller can invent a third tolerance,
and there is no env var or deck key either.

Why it exists: this route's own floor is **0.858 meV on the un-downfolded
μ=960 parent** (`2026-08-11-qsign-recut-verdicts.md` §2) — the windowed-ζ'
representability error with no downfold in it at all, reproduced to five
decimals across two trees and two mesh shapes. Against the reference gate
that refuses by 86x, so at reference grade this route draws nothing. The
owner's deliverable is a band-structure *picture* whose features live at tens
of meV. The 2x-centroid GW re-run remains the only path to 0.01 meV off-grid.

On a pass the grade and the certified NUMBER travel together, as one string,
into three places: the run's rank-0 provenance line, the `.dat` header (plus
an explicit "these levels are NOT reference numbers" line), and the rendered
`.png`. The figure is the artefact that leaves the directory, so it is the one
that must not be able to arrive without its grade.

## 4. What the parent arms measured

See §5 of this file's companion rows in `EVIDENCE_MANIFEST.md`; the numbers
and the PNG paths are in the lane's strike row.
