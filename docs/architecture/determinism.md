# Determinism: bitwise-reproducible reductions

This page owns what "bitwise identical" means for a LORRAX run, why a
multi-rank run was not bitwise reproducible across GPU placements, and the
rank-order reductions that make it so. The collectives' API is in `lxkit`; their
use sites are listed below.

## 1 The gate {#gate}

Two runs are **bitwise identical** when every stored product compares equal byte
for byte after its header. The products are ζ (every dataset), the ISDF tensors,
the W model, `sigma_mnk.h5`, and `eqp0.dat` / `eqp1.dat` after their first line
(the timestamp). A change that claims "no numerical effect" passes this gate on
the **same node and the same rank→GPU placement** as its base:

```bash
cmp <(grep -v '^#' base/eqp0.dat) <(grep -v '^#' new/eqp0.dat)    # and eqp1.dat, eqp_g0w0.dat
```

A change that moves numbers states `max|ΔE_qp|` in eV over every state of
eqp0/eqp1 against a named reference run. On Fe V2 P=4 the gate is ≤ 1e-7 eV
(INT7 release notes, issue 3). A metal is quoted over all states and relative to
|Re Σ_c|.

## 2 Why runs differed: placement-dependent rings {#why}

RCCL builds its rings from the links between the GPUs it was given; NCCL has the
same topology dependence. An all-reduce or reduce-scatter therefore adds
the P terms of each element in an order that depends on the rank→GPU map. f64
addition is not associative, so two placements give two sets of last bits.
Measured on Frontier MI250X (lane det, Fe V2 P=4):
- Runs on the same rank→GCD map are bitwise identical across five nodes. Runs
  differ between maps: the half-node maps 4,2,3,1 and 4,5,2,3 (spanning
  packages) gave two classes, at 4.1e-7 and 7.5e-7 eV in eqp.
- A two-map micro-test gives two hash classes at every size from 1e3 to 2e7
  elements, f64 and c128.
- Cold and warm compile caches give identical bits. No autotune or atomics
  nondeterminism was seen.
- The first divergence is ψ(r_μ): 1 ulp in 33 % of its elements, from the
  `psum_scatter` of the ranks' partial G-slot DFTs
  (`common.psi_G_store._gslot_face_kernel`). The ζ fit amplifies it: ζ 1.3e-8
  relative, V_q 3.5e-9, then W, Σ_c and eqp.

## 3 The fix: rank-order reductions {#ordered}

`lxkit.ordered_psum_scatter(x, axis_name, *, scatter_dimension=0, tiled=False)`
has the same arguments and result as `jax.lax.psum_scatter`. It is one
`all_to_all` (row j is rank j's slice for this device) followed by a fixed
left-to-right sum over j. `lxkit.ordered_psum(x, axis_name)` is the ordered
reduce-scatter of the flattened, zero-padded operand, then an `all_gather`.
`all_to_all` and `all_gather` only move data, so the bits are the same on any
route. The ordered reduce-scatter moves the same bytes as a ring reduce-scatter. Extra device memory: one transient `all_to_all` output the size
of the operand, per call. No host sync is added.

Every four-rank reduction on the Fe V2 path uses them:

| reduction | site |
|---|---|
| ψ(r_μ) G-slot DFT | `common/psi_G_store.py` (`ordered_psum_scatter`) |
| route-G X | `isdf/zeta_mubatch.py` |
| `tr C` | `gw/isdf_fitting.py` |
| Σ / SX band block | `common/contract_bands.py:reduce_scatter_to_band_block` |
| `psum_replicate` | `common/collectives.py` |
| separable terms | `common/mtxel_sweep.py` |
| Hartree ρ | `gw/qsgw_density.py` |

A reduction over one mesh axis at P=4 has two terms, and a two-term sum is
exact under commutation, so those are left as `psum`. At P ≥ 9 (3×3 and larger
meshes) a single axis has three or more terms; those sites are listed in [§5](#open).

## 4 Result and cost {#result}

With default RCCL settings and no environment variables, three Fe V2 P=4 runs on
different nodes, GCD sets and maps gave identical bits:
- i4a: frontier00622, GCDs {4..7};
- i4b: frontier00620, {0..3}, map 0,2,1,3;
- i4c: frontier00620, {0..3}, map 3,1,2,0.

That covers ζ (48/48 datasets), the ISDF tensors (20/20), the W model (22/22),
`sigma_mnk.h5` (12/12) and eqp0/eqp1. The gates were the P=4 hsuite scalar chain
(9 stages) and `tests/test_ordered_collectives.py`: a bitwise rank-order fold,
plus agreement with `psum_scatter`/`psum`.

The wall cost was not measurable. Same node, alternating runs: totals
115.02 / 112.69 s base against 114.51 / 110.80 s fixed. The visible stage change
was `zeta_fit.mubatch.loop` 0.337 → 0.408 s.

**Rejected:** pinning the ring with `NCCL_RINGS`. It made Fe on GCDs {0..3}
bitwise across three maps, at +1.7 s Σ τ and +0.55 s sample_dyson. But RCCL
falls back to its own ring when the requested one does not fit the links: maps
on {4..7} split again, and the Hartree split between {0..3} and {4..7} stayed.

## 5 Not covered {#open}

- Four-rank reductions off the Fe V2 path are still library-ordered: the BSE
  `bse_ring_comm` reduce-scatters, `pivoted_cholesky`, `qsgw_head`,
  `photon_direct_head`, `head_correction`, `galerkin`, `centroid.distribution`,
  and `common.collectives.psum_scatter_checked`.
- Single-axis reductions (`distrib_la` GEMM `psum_scatter`, the `contract_bands`
  x/y stages) at P ≥ 9.
- Reductions XLA inserts by itself were not audited; none showed up on the Fe
  path.
- Not measured: multi-node, P ≥ 8, CUDA. The fix is leg-neutral JAX, and NCCL
  has the same topology dependence, so CUDA bits change too, deterministically.
