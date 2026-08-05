# omega-cube sharded-consumer implementation notes (wk_REL, 2026-07-28)

Owner-approved implementation of DESIGN_MEMO_omega_cube_sharding.md, with the
owner clarification: NO new process grid — consume tiles where the stacked
psum_scatter left them, on the EXISTING 2-D ('x','y') mesh.
Branch fix/zq-band-gather-device-invariance @ b436e47 (working tree, NOT committed).

## Consumer enumeration (from code, this pass)

Full-cube readers of Σ_c(ω,k,m,n) as of b436e47:
1. ppm_pipeline._inject_analytic_head        — elementwise add (head is band-DIAGONAL)
2. ppm_pipeline._eval_sigma_c_at_dft_energies — diag only (extract_sigma_diag_replicated)
3. ppm_pipeline._write_sigma_omega_h5        — sigma_mnk.h5 via SlabIO
4. sigma_dispatch.compute_sigma_xc → qsgw_utils.build_qsgw_sigma_xc — one-shot QSGW
   Σ_xc build (E_DFT); **consumes the cube at its NATIVE P(None,None,'x','y')
   sharding by design** (its docstring says so verbatim)
5. qsgw_utils.solve_qp (fixed_point)         — diag + the same QSGW build
6. gw_jax.main:548                           — diag for the eqp1 Z-factor
7. sc_iteration                              — captures the cube across iterations
   (round-1: REFUSED at config resolve, per memo §4)
8. driver tail jnp.asarray(sigma_kij_host)   — the replicated re-upload (elided)

NOTE — memo §2 enumerates 1,2,3, the diag writers, sc, and the driver tail but
does NOT name the QSGW build (items 4/5 above) as a full-cube consumer.  Not a
design conflict: the QSGW kernel was already written for the sharded layout
(take_along_axis over the replicated ω axis is shard-local; only its (nk,nb,nb)
RESULT is forced replicated), so the tiled path feeds it the tile-sharded cube
unchanged.  Recorded here because the memo said "enumerated from code" and the
enumeration was incomplete.

## Design shape implemented (deviation from memo §3, flagged)

Memo §3 sketches a new small dataclass handle (tiles + index + sharding) with a
.replicated() escape hatch.  Implemented instead: on the flag-gated path the
per-rank host tiles are device_put RANK-LOCALLY (no collective) and assembled
with jax.make_array_from_single_device_arrays into a single jax.Array sharded
P(None,None,'x','y') on the EXISTING mesh — i.e. the handle IS the sharded
jax.Array (which is exactly "tiles + index + sharding"), and consumers 4/5
work unchanged at native sharding.  On XLA:CPU device memory IS host memory,
so this is still "host tiles end-to-end" in the memo's sense.  Escape hatch:
qsgw_utils.gather_sigma_omega_replicated_host().  Rationale: zero new types on
the SigmaResult seam; the QSGW-build consumer (missed by the memo) needs a
jax.Array anyway.

## Config key

sigma_omega_layout = replicated (default) | sharded   [gw.in [cohsex]]
- PPMConfig.omega_layout; refusals at resolve time (doctrine 3 / pattern #6):
  * sharded × qp_solver=self_consistent  → ValueError (round-1)
  * sharded × sigma_omega_accumulation=kij_stream → ValueError
  * sharded × (nb_sigma % p_x != 0 or % p_y != 0) → ValueError (round-1:
    the mesh-pad block cannot ride the sharded QSGW/Hermitize path yet)
  * sharded × slab_io=h5py_allgather at P>1 → ValueError (that writer would
    re-introduce the full-cube gather inside SlabIO)
- announced at Σ-stage start when sharded.

## Gates (to be filled as they run)

- [x] py_compile sweep: login python3.7 parse of the 6 edited files OK;
      definitive in-container (3.12) `compileall src/` runs as job prologue
      of 7878697 (job aborts before any run on failure).
- [x] nb=256 A/B dev job — **ALL GREEN** (job 7878707, OMEGA_ab, isolated
      worktree wt-OMEGACUBE @ b436e47 + this workstream's 6 files):
      * rep rc=0 (wall 280 s), til rc=0 (wall 134 s)
      * BITWISE til-vs-rep: PASS — sigma_diag.dat (4146 lines), eqp0.dat,
        eqp1.dat, eqp_g0w0.dat byte-identical; sigma_mnk.h5 ALL datasets
        byte-identical incl. c128(41,16,256,256) sigma_c/sigma_total
      * continuity rep-vs-run_L1_b256 (standing 1e-12 suite): PASS
      * gather proof: rep dump has the c128[41,16,256,256] all-gather
        (jit__identity_fn class; 27 modules reference the cube shape);
        til dump: ZERO modules reference the full-cube shape AT ALL —
        the cube never exists as a device object on the tiled path
      * til new collectives exactly as designed: all-reduce c128[41,16,256]
        = 2.69 MB (shard_map diag psum; nb²→nb per memo §3) + all-reduce
        c128[16,256,256] = 16.78 MB (QSGW build's replicated result);
        τ-kernel reduce-scatters unchanged (10.22 / 0.52 MB)
      * timing: rep sigma.exec 83.707 / host_gather 0.857; til sigma.exec
        82.785 / tile_finalize 0.004 (Σ-stage neutral, as expected —
        memory/scaling play).  OBSERVATION (unclaimed, single-shot): til
        job wall 134 s vs rep 280 s with recorded rows ~equal (118.7 vs
        122.4) — the delta sits in the UNTIMED sigma_mnk.h5 write
        (replicated write elects dedup writers over full-cube hyperslabs
        vs per-rank 21.5 MB tiles).  Needs its own timed row + repeat
        before any claim.
      (sbatch + strict parity script:
      mos2_4x4_test/omega_ab.sbatch, omega_ab_parity.py — bitwise text
      compare after dropping the provenance timestamp line + raw-byte h5
      dataset compare)

      ⚠ INCIDENT (honesty note): first submission 7878697 was CANCELLED by
      me mid-run.  Its job header revealed a CONCURRENT agent editing the
      SAME main checkout live (src/gw/ppm_tau_kernel.py, common/fft_helpers,
      ffi_loader, new src/ffi/mklfft/ — mtimes 15:29–15:33 while my pass rep
      imported at 15:34): the two A/B passes could have imported DIFFERENT
      τ-kernel states, invalidating the bitwise A/B.  Fix: created isolated
      git worktree /work2/08271/jackmc/frontera/wt-OMEGACUBE @ b436e47
      carrying ONLY this workstream's 6 files (house §11 worktree rule),
      repointed both sbatch SRC there, resubmitted as 7878707.  The main
      checkout still carries my 6 edited files (unavoidable given the task
      pointed there; identical content to the worktree) — orchestrator
      should be aware two workstreams' uncommitted edits now coexist in
      the main tree (no file overlap: theirs = tau-kernel/fft/ffi, mine =
      config/driver/pipeline/qsgw).
- [x] nb=512 confirmation cell — **GREEN** (job 7878722, OMEGA_512, til only,
      restart-gated from run_L3_b512_c5000 tmp isdf_tensors_4951.h5):
      * rc=0, wall 551 s; sigma.exec 366.479 s (L3 replicated ref: 401.225 —
        within/below cross-run band, no speed claim), tile_finalize 0.009 s
      * **2751.46 MB gather GONE**: all-gather ops with c128[41,16,512,512]
        = 0; modules mentioning the full-cube shape AT ALL = 0 (L3 ref: 1,
        module_0962 jit__identity_fn); colltable LARGEST collective now
        199.36 MB (module_0333 jit__res — pre-existing W-stage, not σ);
        "NO collective carries a full (mu,mu) tile (mu=4951)"
      * parity vs run_L3 outputs (informational, cross-tree): PASS
        bit-identical — sigma_diag (8242 lines) / eqp0 / eqp1 / eqp_g0w0 +
        ALL sigma_mnk.h5 datasets incl. the two (41,16,512,512) tensors

## Expected NEW small collectives on the tiled path (movement accounting)

- diag extraction: ONE all-reduce (psum) of c128[41,16,nb] — 2.7 MB@256b,
  5.4 MB@512b (nb² → nb, memo §3) — inside the shard_map diag kernel.
- QSGW build result: all-gather of c128[16,nb,nb] (the with_sharding_constraint
  rep_3d in _qsgw_build_kernel; 16.8 MB@256b, 67 MB@512b) — this object is
  replicated BY DESIGN (it feeds the replicated eigh); on the replicated path
  the constraint was a no-op because the input was already replicated.
- everything else rank-local (head diag add, static fold, device assembly,
  SlabIO per-rank hyperslab writes via PHDF5_FFI).

## Files touched (worktree only, NOT committed)

- src/gw/gw_config.py       — sigma_omega_layout key + PPMConfig.omega_layout
                              + resolve-time refusals (SC, kij_stream)
- src/gw/gw_jax.py          — early geometry/backend gate + announce
- src/gw/head_correction.py — compute_ppm_head_sigma_diag factored out
                              (dense builder now embeds it; single source)
- src/gw/ppm_pipeline.py    — head injection diag path; sharded sigma_mnk
                              write (jit out_shardings total derivation)
- src/gw/ppm_sigma.py       — two-plan finalize: sigma.tile_finalize block,
                              no host cube alloc, sharded SigmaOmegaResult
- src/gw/qsgw_utils.py      — is_band_sharded_sigma_omega, shard_map diag
                              extractor (psum, structural), add_band_diag
                              _sharded, gather_sigma_omega_replicated_host
                              escape hatch; extract_sigma_diag_replicated
                              branches on the array's own sharding

NOTE: manual/05_isdf/5.1_pair_density_factorization.md was ALREADY modified
in the working tree before this workstream started (Gram-build commit's doc
edit, uncommitted) — not part of this change.

Canonical copy for merge: /work2/08271/jackmc/frontera/wt-OMEGACUBE
(b436e47 + exactly these 6 files; byte-identical copies also present in the
main checkout, which additionally carries a CONCURRENT agent's unrelated
mklfft/tau-kernel edits — see the incident note above).

## Residual risks / named-not-done (round 1)

1. Sharded × mesh-padded window REFUSED (nb % p_x / % p_y != 0): the QSGW
   Hermitize needs a square unpadded extent.  Fix = pad-aware QSGW build +
   strip; refusal is at config/driver resolve, loudly.
2. Sharded × self_consistent REFUSED (SC loop's Σ_c capture/rotate seam).
3. fixed_point × sharded is WIRED (diag extractor + native-sharding QSGW
   rebuild) but NOT yet gate-covered — decks gated here are one_shot_dft.
   Configuration-lattice coverage (pattern #2) still owed: P=1, small P,
   rectangular mesh, KIJ_STREAM interplay beyond the announce, fixed_point.
4. Known non-bitwise edge (predicted, not observed): the diag psum adds
   exact zeros — an element exactly equal to -0.0 would flip to +0.0.
   Both gates (nb=256 A/B, nb=512 vs L3) came out byte-identical.
5. The sigma_mnk.h5 write-wall observation (til ~140 s faster job wall at
   nb=256, outside recorded rows) is UNCLAIMED: needs a timing.section row
   around _write_sigma_omega_h5 + a repeat before it becomes a number.
6. GPU backend: not exercised (CPU gates only; design is backend-agnostic
   movement, but pattern #9 says re-verify at the backend jump).
