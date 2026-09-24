# ζ fit by μ-batches (route G): Z(G) first, C⁺ once

Status: the charge channel runs route G on branch
`feat/zeta-mubatch-2026-09-23` (not on main): `isdf.zeta_mubatch`
(`make_route_g_kernel`, `ZStore`, `ZetaG`), `gw.isdf_fitting._fit_mubatch`,
planner `gw.gflat_memory_model.plan_zeta_route_g`, conditioning `isdf.cplus`.
Evidence: `runs/runtime/zeta_mubatch_20260923/` in the sandbox (manifest.yaml).
Owner decision 2026-09-23: route G is the single route; the r-space
machinery (ψ(r) cache and plane regeneration, the Z-row transpose, the
orbit r blocks, the planes/flat switch) is retired.

## The identity

ISDF least squares per selected q:

```text
Z_q(μ, r) = Σ_k Σ_{ab} D^L_{k,ab}(μ, r) · conj(D^R_{k+q,ab}(μ, r))      (k-convolution)
D^X_{k,ab}(μ, r) = Σ_n w^X_n ψ_{nka}(r_μ) ψ*_{nkb}(r)                  (X = L, R windows)
C_q(μ, ν) = Z_q(μ, r_ν)                    ζ_q = C_q⁺ Z_q               (C⁺ acts on μ only)
```

`C_q⁺` is the operator `factor_c_q` builds today (rank-truncated
pseudo-inverse or Cholesky, the same rcond, the same pad-diagonal and
LR+RL completion). Because it acts on μ only, it commutes with the
r→G transform:

```text
ζ_q(μ, G) = C_q⁺ Z_q(μ, G),     Z_q(μ, G) = FFT_r[e^{-iq·r} Z_q(μ, r)] on the ζ sphere
V_q = conj(ζ) diag(v_q) ζᵀ = conj(C_q⁺) M_q conj(C_q⁺),   M_q = conj(Z_q) diag(v_q) Z_qᵀ
```

`C_q⁺` is Hermitian, so `(C⁺)ᵀ = conj(C⁺)`. Both forms of `V_q` are the
same operator. Truncating to the sphere before or after the solve is
identical, so ζ(G) must match the r-chunk fit to rounding. The LR+RL
completion `Z_q ← Z_q + conj(Z_{−q})` is applied in r space, before the
q selection and before the transform, exactly where the r-chunk loop
applied it.

## The loop (route G)

```text
setup:  conj ψ(G) of the parent k (one read, `load_parent_psi_G`), G slots
        over the mesh (ψ(G)/P per rank);  C_q, factor (the CCT path);
        Z store (numpy host blocks or slab_io)
for each μ batch B (b centroids, b a multiple of P, serial):
    X_B = ψ_{nks}(r_μ)                       each rank's partial DFT over its
                                              G slots, one psum (replicated)
    D̃^X_k̄(a, μ, b, G) = Σ_n w^X_n X_{nk̄a}(r_μ) conj c_{nk̄b}(G)
                                              G-space GEMM on the parents k̄,
                                              the rank's slice (isdf.pair_kernels)
    ONE all-to-all: G split → μ owner (rank p owns slots p·c + [0, c), c = b/P,
                    whole centroid orbits per owner)
    on the owner:
        typed unfold to the full zone (the Fourier image of the r-space typed
        transport C_q was built with, `typed_child_G_tables`):
            D̃_k = (U⊗Ū)ᵀ D̃_k̄(perm μ, pslot G) e^{2πiL·k̄} conj(phase)
        cylinder gather + axis DFT onto ALL planes, once per k: the D cylinder
        per group of n_pg planes: 2D FFT, Bloch phase → D(k, μ, r_plane);
            Z_q(μ, r) = Σ_k Σ_ab D^L_k conj D^R_{k+q}
                        (isdf.core.parent_projector_kconv on the identity plan)
            LR+RL completion; e^{-iq·r}; forward 2D FFT;
            one matmul onto the ζ-sphere cylinder (columns × axis values)
        gather the ζ slots → rows Z_q(μ_B, G), μ-owned
    write the rows into the Z store (asynchronous; batch β+1 computes meanwhile)
after all batches:  stream the store by G tiles:  ζ_t = C⁺ Z_t (isdf.cplus),
                    V_q += conj(ζ_t) diag(v_q) ζ_tᵀ,  keep the G≈0 shell;
                    write ζ_t only when a consumer needs the file
```

Two collectives per batch (read from the compiled HLO): the X_B psum and
the pair-projector all-to-all.  There is no distributed FFT and no
accumulator over r: every plane transform, the k-convolution and the
axis transform are local to the owner of whole μ rows.

V_q is formed ζ-first, not as `conj(C⁺) M conj(C⁺)`: the two are the same
operator in exact arithmetic, but the second loses `≈ ε·κ(C)²` (3.3e-14 on
core fixture A, 1.7e-3 on CrI3 8×8 with κ(C) = 1.0e8).

### Z store

`ZStore` is one write-once resource of μ-owned rows, never on the device:
one numpy block `(n_Gt, Q, n_batch, c, G_tile)` per local device (exact
bytes, `Q·n_batch·c·N_G·16` per rank), or a slab_io scratch dataset
`(n_Gt, Q, n_batch·b, G_tile)` when the host share does not hold it.  The
host share is `0.8·MemAvailable` at plan time over the processes on the
node, the minimum over processes; the ψ read's phdf5 staging is released
before the store fills (`WfnLoader.release_read_staging`).  A G-tile read
is one contiguous host block per device and reaches the finalize layout
(q-local for the local solve tier, G-split for the replicated one) with
one all-to-all.  The μ axis is batch-slot order; reads gather the packed
carrier (`OwnerOrbitBatches.slot_of_packed`).

## Per-rank memory model and planner

Unit: the Green's-function tile `G_tile = nk·ns²·μ²·16/P`; the GW run's
bottleneck is about `4·G_tile`.  Feasibility: the smallest configuration
(ψ(G) resident, `b = P`, one plane per group) must fit `4·G_tile`, so
the fit never sets the node count; the plan then uses the device target.

| object | bytes/rank |
|---|---|
| conj ψ(G) slice (parents), Ψ | `n_parent·nb·ns·N_Gψ·16/P` |
| C factor | `Q_loc·μ²·16` (q-local) or `Q·μ²·16` (replicated) |
| X_B (replicated per batch) | `n_parent·nb·ns·b·16` |
| pair projectors, G slice and after the all-to-all | `2·n_parent·ns²·b·N_Gψ·16/P` each |
| D cylinder (all planes, owner) | `nk·n_a·ns²·2c·n_col·16` |
| plane group | `2·nk·n_pg·ns²·2c·(N_r/n_a)·16` |
| k-conv + Z (group) | `(9·nk + 3·Q)·c·n_pg·(N_r/n_a)·16` |
| ζ cylinder accumulator | `Q·c·n_zc·n_za·16` |
| Z rows (+1 lookahead) | `2·Q·c·N_G·16` |

The batch working set is the maximum over its three stages (GEMM and
all-to-all; D cylinder; plane groups), not their sum.  The rule (owner):
ψ(G) resident iff `M_f − Ψ ≥ c_μ·b_min` (`M_f` the target minus the fixed
terms, `c_μ` the per-centroid slope of the working set); then
`b = (M_f − Ψ)/c_μ` in multiples of P.  Candidates over the plane-group
width `n_pg` are costed by the two collectives (`gw.comm_model.comm_time`),
the per-group launches and the owner's per-centroid work; the receipt
prints every object as a `G_tile` ratio, the minimum configuration, the
collectives per batch, the all-to-all floor
`μ·2·nk·ns²·N_Gψ·16/(P·β)`, the X_B bytes, the modelled loop and the
runner-up.  No deck key or environment knob sizes anything.  Owners hold
whole centroid orbits, so the fit packs them at the bin width c ≤ b/P with
the least padded work, n_batch·(c + 1) (`best_owner_orbit_batches`): the
planned width can pack badly (CrI3 8×8 P16: c = 19 packs 8 batches of 18,
c = 12 the same 8 batches of 12).

Padding goes through `runtime.padding`: the stored q rows and the ζ
sphere cut into whole G tiles are `PaddedAxis` records made once in
`_fit_mubatch` and shared by the kernel, the store and the finalize; the
ψ G-slot axis and the plane groups are padded by name; batch slots carry
their pads in `MuOrbitBatches.mu` (−1).

### Simplifications taken

| what | edge case that loses | cost |
|---|---|---|
| route G only (no ψ(r) cache, no flat blocks) | axis-mixing groups with a cheap cache | TaAs 8³ ~1.5× vs flat blocks (auditor) |
| the typed D̃ unfold uses the grid-snapped τ of the r-space transport, not the loader's raw τ | none: C and Z share one transport (owner ruling); the loader's raw-τ unfold differs by 2e-10, 2e-6 in ζ at κ(C) ~ 1e8 (KNOWN_LORRAX_ISSUES) | — |
| the Z store never lives on the device | small decks whose store fits beside the batch | one host round trip |
| host tier is pageable numpy, not pinned (`HostTileStore`) | none measured; VI3 12×12 P16 (33.4 GB/rank of Z) was host-OOM-killed on the pinned tier | pageable D2H/H2D bandwidth; the pinned tier returns when the planner prices its overhead |
| one conditioning procedure (`isdf/cplus.py`, rank truncation) | none measured | per-q eigh |
| the orbit bin width is chosen after the plan, by a scan with a one-centroid fixed-cost guess | decks where the batch fixed cost dominates | the planner does not see orbit sizes; its modelled loop assumes b/P |
| the ζ file, when a consumer wants it, is written by its own pass over the store | runs with `write_restart_tensors` | one extra C⁺ stream (V_q streams again) |
| ψ(G) streaming (two band-chunk buffers) not implemented | decks where Ψ does not fit beside the smallest batch | refuses with `GATE zeta-mubatch-capacity` |

## ζ consumers

The fit returns `ZetaG`, a lazy ζ over the store (`zeta_layout =
'G_flat'`), in place of the file path when no consumer needs the file.

| consumer | needs ζ on disk? | handled by |
|---|---|---|
| scalar V_q (`v_q_g_flat._compute_V_q_g_flat_one_tile`) | no | `ZetaG.contract_v`: streamed ζ-first V_q |
| g0 one-leg unfold (non-IBZ) | the G≈0 shell | `ZetaG.shell` (`|q+G| ≤ max|q_full|`, pads at the FFT-box sentinel) |
| head channel (`compute_head_channel_zeta`) | a few G columns | `ZetaG.head_columns(sel)` from the shell |
| `write_restart_tensors = true`, restart | yes | the same streamed tiles are written (`contract_v(…, zeta_io=…)`) |
| bispinor / transverse V_q, BSE `vq_interp`, downfold, exciton bands | yes | the file is written; consumers open `ZetaG.path` |

## Regime guard

The planner refuses with `GATE zeta-mubatch-capacity` naming the
shortfall when ψ(G) plus the smallest batch does not fit (ψ streaming is
the upgrade path).  The finalize solve tier follows `zeta_auto_tier`:
q-local (each G tile read onto q owners) or replicated.  The distributed
tier refuses (`GATE zeta-mubatch-tier`): route G applies the factor B
with C⁺ = B·Bᴴ, and the distributed 2D factor application for μ ~ 1e5
supercells is future work.  Each finish (V, the G≈0 shell, a ζ tile for
the file) leaves the q-local or G-split accumulator in ONE explicit
collective (`_to_mu_owner`: an all-to-all, or a reduce-scatter of the
partial sums), so SPMD never replicates V.  Nq = Nk = 1
runs (core fixture B).

## Verification

`tests/multi_device/zeta_mubatch_p4.py` (P4: glide ns=2 with an antiunitary
row, A-cubic 48 ops, and a deck where no axis divides P; both store tiers
and read layouts; red twin) against the dense full-BZ sum.  NumPy
prototype and whole-run evidence in the sandbox manifest.

## Scope

Charge channel, ns = 1 and ns = 2.  The current (transverse) channels keep
their r-chunk loop until their vertex tails are ported; the coupled
μ = 1,2,3 coordinator is unchanged.
