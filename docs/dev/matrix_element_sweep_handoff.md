# One k-scan for V_H, kin+ion and dipole — implementation handoff

Status: WIRED. All three call sites in §1 now run the sweep. Supersedes
`rho_vh_2d_design.md` §5.1 for the matrix-element sweep (that document's ρ,
wall and self-consistency analysis still stands).

Landed:

* `src/common/mtxel_sweep.py` — the skeleton (`SweepGeometry`,
  `sweep_matrix_elements`), the kinetic operator and the local-potential
  operator. `use_scan=False` runs the identical body in a Python loop and is
  the control the scan is gated against.
* The physics is CONSOLIDATED, not copied. `psp.get_DFT_mtxels` gained
  `local_potential_scalars` (the FFT normalisation chain) and
  `kinetic_diagonal` (`|k+G|²`); the local plan's `_compute_local_V_k_jit`
  and `_compute_kinetic_k_jit` now call them, and so does the sweep. The two
  plans therefore agree BY CONSTRUCTION and differ only in the reassociation
  the sharding forces. Before this the chain was written out three times.
* `compute_local_V_k_2d` DELETED (119 lines) — the dead seam this spec
  subsumes, as §4 required.
* Gate `tests/multi_device/mtxel_sweep_gate.py`.
* §6.3 — `vnl_operator` and `dipole_operator`. Both reuse the existing
  apply-to-ket kernels (`vnl_ops.apply_vnl`,
  `vnl_ops.apply_vnl_velocity_to_ket`,
  `dft_operators.apply_kinetic_velocity_to_ket`), so no velocity or
  projector physics is written twice; `vnl_ops.build_vnl_kdata_traced` is
  the only new entry point, and it exists solely because the eager one
  coerces its two per-k operands with `np.asarray`.
  V_NL DID fit the `Operator` protocol: its G sum runs over the replicated
  G axis with the band index free, so it needs no collective and forms no
  `(nb, nb)`.
* `Operator.ncomp` — an optional replicated component axis, so the dipole's
  three Cartesian directions ride ONE sweep (one hoisted m-side reshard, one
  einsum) instead of three.
* `sum_operators` — `⟨m|T+V_loc+V_NL|n⟩` summed ON THE KET, so the kin+ion
  site is one sweep rather than three.
* `blocks_to_host` — the explicit boundary; see §6.4 below.
* The three call sites: `gw.kin_ion_io.compute_hartree_matrix`,
  `gw.kin_ion_io.main`, `psp.get_dipole_mtxels.main`.
  `--vnl-mode=numeric` stays on the per-k route (its finite difference picks
  a step from that k's median |K| on the host and needs 4–8 extra projector
  builds per component per k; it is a cross-check of the analytic dZ).
* Gate `tests/multi_device/mtxel_callsite_gate.py` — all three sites vs the
  local plan on MoS₂ 4×4 (deck_b300, nk=16, nb=28) at P=4, per shard.
  Job 7889210: V_H 9.6e-16, kin+ion 1.4e-15, dipole 2.8e-15, the boundary
  and `compute_hartree_matrix` end to end at the same figures; ONE compile
  per sweep for all three.

Not yet: the SlabIO write of `H[m_X,n_Y,k]` and the direct-field basis-rotation
consumer audit (§6.4). The three CLI sinks are
serial h5py writes, so the sharding is undone at the boundary by name
(`blocks_to_host`) rather than pushed into them; that is the trade §6.4
exists to remove and it has not been removed.

Where the physics lives, and why it is split that way: `gw.kin_ion_io`,
`psp.get_DFT_mtxels` and `psp.get_dipole_mtxels` all carry a jax-plumbing
budget of **0** (`tests/test_layering.py`), so no operator factory naming
`Mesh`/`PartitionSpec` may live in them. The sweep therefore holds the
plumbing and imports the pure-physics scalars from `psp` — L1→L1, which
`test_layering` allows. `common/wfn_transforms.py` is the precedent: L1 by
default, names plumbing, not budgeted.

Read `docs/architecture/decisions.md` first — D10 (fixed-shape ngkmax) and the
2026-08-04 SlabIO padding entry are both load-bearing here.

---

## 1. What is being replaced

Three sweeps, three call sites, one shape:

| operator | file:line | block fn | output |
|---|---|---|---|
| ⟨mk\|V_H\|nk⟩ | `gw/kin_ion_io.py:453-461` | `_vh_block` | `(nk, nb, nb)` |
| ⟨mk\|T+V_loc+V_NL\|nk⟩ | `gw/kin_ion_io.py:728-741` | `_kin_ion_block` | `(nk, nb, nb)` |
| ⟨mk\|r\|nk⟩ | `psp/get_dipole_mtxels.py:750, 849` | `_dipole_block` | `(nk, 3, nb, nb)` |

(Line numbers are pre-conversion. `_vh_block` and the default-mode
`_dipole_block` are gone; `_dipole_block` survives for `--vnl-mode=numeric`
and the `LORRAX_DEBUG_PRINT` table, which needs p and p+v_NL separately.)

All three called `common/collectives.py:985 gather_k_blocks`, which is
**k-partitioned and returns an array identical on every rank**. Three
consequences, measured in `rho_vh_2d_design.md` §1: the per-k full-band FFT
box (1.77 GB at b600 bispinor, 37 GB and OOM at 12×12), the replicated
`(nk,nb,nb)` (829 MB at 12×12, 9.2 GB at nb=2000, ×3 for dipole), and
and the inability to use more than `nk` ranks at all — each rank takes whole
k, so the wall is one full-band k however large P is.

The replacement's parallelism, stated once: the scan does **one k at a time**
and shards **that k's bands over every process**. `nk` is the scan trip count
and does not enter parallel efficiency; efficiency is `nb_logical/nb_padded`,
and the only way a rank goes idle is `nb_logical < P`, where the band pad
leaves it holding only zero-bands. The sweep scales to `P = nb` where the
k-partitioned plan stopped at `P = nk`, and pays a per-k reshard collective
for it.

Fixing the scaling today means making the same change in three places.
This spec makes it one routine.

---

## 2. The structural fact that collapses the three

Only the **local-potential** terms need a real-space excursion. Everything
else is diagonal in G or a projector sum:

| operator | `O ∘ ψ` | FFT? |
|---|---|---|
| kinetic | `T_G · ψ` | no |
| dipole | `(k+G) · ψ` | no |
| V_NL | projector sum (`psp/vnl_ops.py`) | no |
| V_H, V_loc | `F[ V(r) · F⁻¹ψ ]` | yes |

So all four are `H[m,n] = Σ_{s,G} conj(ψ_m) · (O ∘ ψ)_n` and differ **only in
`O ∘ ψ`**. One skeleton, one pluggable operator.

**The m side is never transformed** — for V_H it is the raw stored sphere; for
the rest the operator is applied on the n side alone. This is what removes the
FFT box: the local plan boxes both sides because it takes both out of one box.

---

## 3. The design

```
                                  ψ(G) resident, sharded P(None, ('x','y'), None, None)
    psi_m_X  ←  reshard ONCE, outside the loop        # shared by ALL operators and ALL k
    scan over k:
        Opsi    ←  O ∘ psi_XY[k]                      # nb/P bands per rank
        Opsi_Y  ←  reshard along 'x'                  # the ONE per-k collective
        H[k]    ←  einsum('bsg,nsg->bn', conj(psi_m_X[k]), Opsi_Y)   # local, no reduction
```

Three claims, each with its reason:

**(a) ψ(G) for all k is resident-able.** The *box* is huge; the G-sphere is
not. `nk·nb·ns·nGmax·16` = 1.2 GB globally at b600, **≈19 MB/rank sharded at
P=64**. This is what makes a genuine `lax.scan` over k possible — the reason
`sweep_local_k` is a Python loop is that the ψ load is host I/O, and that
obstacle disappears if ψ is already on device. Check the number for your deck
before assuming it; at 12×12 with nb=2000 it is ~10× larger.

**(b) Hoist the m-side reshard.** ψ_m is never transformed, so build it once
and reuse across every k *and* all four operators: `nk+1` all-gathers, not
`2·nk`, and not `3×` that for three sweeps. Cost ≈151 MB/rank at b600/P=64.

**(c) Shard the OUTPUT, replicate the contraction axis.** `H[m_X, n_Y, k]`.
The alternative — shard over G and psum partials — requires every rank to
hold a full `(nb,nb)` to reduce into, which is exactly the wall being removed.
Not a preference; forced.

---

## 4. Existing code to use — do not reinvent

| need | use | notes |
|---|---|---|
| fixed-shape G table + mask | `psp/dft_operators.py:248 padded_gvectors` → `.at(ik)` | D10. Table is free — `WfnLoader.gvecs()` already stores `(n_k,ngkmax,3)` zero-padded and `ngk_valid()` is memoised. |
| sphere → box → r | `common/wfn_transforms.py:372 to_rbox` | goes through `_local_box_fft` (shard_map'd FFI FFT). Never call `jnp.fft` directly here. |
| r → box → sphere | `common/wfn_transforms.py from_rbox` | added 726bcf7; the return leg. Takes `gvecs` + `g_mask`. |
| the V_H kernel body | `psp/get_DFT_mtxels.py compute_local_V_k_2d` | **absorb and DELETE it.** It is a dead seam (no caller, no resolver) and this spec subsumes it. Its certification (job 7888534) transfers. |
| io_callback inside scan | `common/psi_G_store.py:299 _slice_local_tile_bc` | **the precedent.** Read its lifetime contract before copying. |
| pad the band axis for uniform scan shapes | `common/wfn_transforms.py:1789 band_pad_to`, or `psi_G_store._bpd_max` | see §5. |
| per-k pipelining today | `common/collectives.py:917 sweep_local_k` | what the scan replaces; read its docstring — it explains why the single readback is only legal under D10. |
| writing `H[m_X,n_Y,k]` | `SlabIO.write_slab(name, A)` | **no padding argument.** decisions.md 2026-08-04; 15 `valid_shape=` were deleted precisely so new code does not add more. |

`gather_k_blocks` is **not** usable — it materialises the `(nb,nb)` tile per k
on every rank, which is the wall. A new collector is part of this work.

---

## 5. JAX features, with the traps

**`lax.scan` + `io_callback`.** Both are viable and the codebase does the
harder one already.

* If ψ(G) is resident (§3a), scan over a device array — simplest, no callback.
* If it is not, `jax.experimental.io_callback` **works inside `lax.scan`** —
  `psi_G_store` does exactly this. Two hard requirements it documents:
  `io_callback` needs a **static `out_sds` at trace time**, and `lax.scan`
  needs the body output shape **uniform across iterations**. `psi_G_store`
  satisfies both with `_bpd_max = max(bpd_per_bc)` closed over at `__init__`
  and short chunks zero-padded to it. Its `np.zeros`-not-`np.empty` note is
  deliberate: garbage in pad rows can pollute IFFT precision even when a mask
  would zero it at the einsum.
* **Lifetime**: host buffers must stay valid for the whole enclosing jit —
  the callback fires *asynchronously* inside the scan. `isdf_fitting.py`
  blocks on the completed ψ(r) cache before closing `psi_G_store`. Reproduce
  that discipline or you get a use-after-free with no symptom at small scale.

**`with_sharding_constraint` as the collective.** Do not hand-roll the
all-gather. Carry the n side at `P(None, ('x','y'), ...)` for `O ∘ ψ` and
constrain to `P(None, 'y', ...)`; XLA inserts it. Certified at P=4 (job
7888481). Constraining *inside* a scan body is fine; the shapes are uniform.

**Do NOT wrap the body in an outer `jit`.** Measured (job 7888526):
`to_rbox`/`from_rbox` memoise a **device** G-index, so under an enclosing
trace that cache captures a tracer which escapes —
`UnexpectedTracerError`. If fusion across the FFT/multiply/gather boundary is
ever wanted, hoist the device G-index out of the transforms and pass it as an
operand; do not wrap.

**Compiles vs dispatches.** Under D10 every transform lowers **once** for the
whole sweep regardless, because shapes are fixed. Eager glue costs
*dispatches*, not compiles. Do not "fix" a compile storm that is not there —
measure with `LORRAX_DEBUG_PRINT=1`
(`common/jax_compile_cache.py:31`) before optimising.

**`dynamic_slice` clamps silently.** `lax.dynamic_slice` clamps
`start → axis_size - slice_size` rather than erroring. `isdf/core.py:588-621`
pads at **both ends** to make clamping unreachable, because a clamp there
produces physically wrong bands that a downstream mask cannot recover.
`wfn_transforms.py:1747 _slice_bands_gflat` raises instead. Pick one; do not
leave it implicit.

**Band padding is now routine.** `nb` must divide both `px` and `py`, so
wholly-padded band slices are the norm, not the corner case. That is only safe
because of d935ce7 + the implicit-padding merge.

**Masking.** Measured (job 7888534): with pad G rows at the **box corner**
`(nx//2, ny//2, nz//2)`, which cannot intersect the sphere, the mask is
unnecessary and the result is **bit-identical** to the masked run
(`0.000e+00`). The rule, and it is narrower than it looks: *a corner sentinel
removes the need for a mask **iff both operands' pad entries come from stored
sphere coefficients***. `phi_G` at the sentinel is NOT zero — multiplying by
V(r) spreads support over the box — so what kills the term is the m side being
exact zeros. If either operand is gathered from the **box**, `(0,0,0)`=Γ is a
real coefficient and the mask stays mandatory. `compute_local_V_k_2d` refuses
an unmasked `(0,0,0)`-padded table; keep that guard.

---

## 6. Order of work

1. The skeleton with `O ∘ ψ = T_G·ψ` (kinetic): no FFT, so it isolates the
   scan, the reshard and the einsum. Gate against `compute_kinetic_k`.
2. Add the V_H/V_loc operator (the FFT round trip). Gate against
   `compute_local_V_k` at 1e-12 relative — the sharding reassociates the
   contraction, so bit-identity is not the criterion (numerical-tolerance
   ruling). Delete `compute_local_V_k_2d`.
3. Dipole and V_NL. DONE. Both are apply-to-ket wrappers around kernels
   `psp` already had; the dipole needs a replicated component axis
   (`Operator.ncomp`) so its three directions cost one sweep, not three.
4. The collector + `SlabIO` write of `H[m_X,n_Y,k]`, then the **consumer
   audit** — the direct-field basis rotation
   (`einsum('kpm,kpq,kqn->kmn')`) today wants a full `(nb,nb)` per k, and per
   the owner's ruling that rotation itself must become distributed through the
   linalg FFI in this same layout.

   NOT DONE, and what stands in its place: `blocks_to_host`. The three CLI
   sinks are serial `h5py` writes and `sigma_dispatch`'s gspace route wants a
   replicated global operand, so all four need the whole `(nk, nb, nb)` and
   none can take a sharded block today. The boundary is therefore a NAMED
   gather at the call site, with `owner_only` preserving the rank-0-only
   contract `gather_k_blocks(owner_only=True)` had and leading-axis chunking
   keeping a peer's transient at one chunk. What the sweep removes is
   upstream of it: the per-k full-band FFT box (W1) and the `P ≤ nk` ceiling
   (W3). W2 — the replicated table — survives at the boundary and is exactly
   what this step deletes.
   The one consumer that already stays sharded is
   `sc_iteration.rebuild_hartree_dft_basis`, which calls
   `sweep_matrix_elements` directly and never reaches `blocks_to_host`.
5. Benchmark at b600/P=64 per `rho_vh_2d_design.md` §8. Report per-rank
   `VmHWM` from `/proc/self/status` at exit **on every rank** — a plan that
   merely relocated a peak looks clean on rank 0 alone.

## 7. Gates

* vs the existing k-parallel kernels at 1e-12 relative, **per shard**, every
  rank checking its own block. A global `device_get` would gather the very
  tile this exists to remove, and a rank-0-only check passes a plan that puts
  the right answer on the wrong rank.
* one compile for the whole k sweep (`LORRAX_DEBUG_PRINT=1`).
* no host readback inside the scan.
* the reshard is along `'x'` only, not a global collective.
* per-rank stdout AND stderr to separate files — stderr is lost under
  srun+apptainer at teardown, which is why the 2026-08-02 defect went
  unattributed for a day.
