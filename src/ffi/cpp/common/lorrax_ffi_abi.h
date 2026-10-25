// lorrax_ffi_abi.h — THE handler-signature ABI version of this source tree.
//
// ONE NUMBER, ONE DEFINITION, AND IT IS HERE.  Everything downstream is a
// mirror that a test compares against this file: the two .so files bake it in
// (common/build_config.cc), and the two Python loaders carry a constant they
// refuse to disagree with (src/ffi/common/ffi_loader.py,
// services/distrib_la/src/distrib_la/loader.py).
//
// ---------------------------------------------------------------------------
// WHAT THIS NUMBER IS FOR
// ---------------------------------------------------------------------------
// The Python side of an XLA FFI call and the C++ handler on the other side
// agree on an argument list that appears NOWHERE as a checkable declaration.
// jax.ffi.ffi_call passes Args and Attrs positionally; the handler unpacks
// them positionally; nothing links the two.  Change one side and the other
// still builds, still links, still dlopens, still registers its targets, and
// still passes every probe the loader performs — because every one of those
// asks "is the handler PRESENT", never "does it take what I am about to send".
//
// The failure arrives later, at the first call, as
//
//     INVALID_ARGUMENT: Wrong number of arguments: expected 3 but got 4
//
// which names neither library, neither version, nor the commit that moved.  A
// person then has to know that a .so exists at all, find which of several on
// the machine is pinned, and remember which commit last touched a signature.
//
// TWICE IN TWO DAYS, on this tree:
//
//   ABI 1 -> 2   96a6399d (2026-08-07) "phdf5: three I/O defects, one of which
//                needs a coordinated .so rebuild".  ctx_handle stopped being an
//                Attr on the phdf5 read path; the arity went 3 -> 4.  The commit
//                message had to spell out, in prose, which two files on which
//                filesystem to pin — because there was no mechanism.
//
//   ABI 2 -> 3   5bdc345e, merged to main as a16a241c (2026-08-08).
//                PhdfReadKchunk / PhdfReadKchunkUnion lost their ctx_handle and
//                ds_id Attrs and gained a leading (2,) S64 handle Arg.  The
//                branch's own note says it best: "a mispaired run is green
//                until something calls read_slabs".
//
// THE THIRD BUMP IS ALSO THE MECHANISM'S FIRST REAL TEST, and it happened in
// the same day.  This file was written on a branch based on 0fe41a76, which
// PREDATES the kchunk change, and it RESERVED number 3 for it.  That branch
// merged to main hours later.  Under the old arrangement, the pairing this
// created was tracked by a hand-maintained table in BUILD_NOTES.md whose own
// warning was that everything off the kchunk path stays green; under this one
// the number moved with the merge and every library built before it is refused
// by name.  The table stops being something anyone has to maintain.
//
//   ABI 3 -> 4   (2026-09-24, feat/kconv-mathdx-stage2).  The nvidia-mathdx
//                k-convolution handlers (KConvMathdxPair/Parent) gained a
//                `cubin_dir` string Attr for the disk cubin cache.
//
//   ABI 4 -> 5   (2026-09-24, perf/kconv-fused-load, concurrency audit item 8).
//                No signature moved.  The phdf5 read path changed MEANING in
//                3175fbbb: the >32 MiB synchronous read staging is retired
//                after its H2D instead of held for the process.  An ABI-4
//                provider built before that commit still loads and keeps the
//                largest zeta tile pinned (the VI3 host OOM); the bump is the
//                only way the loaders can refuse it.  The rule's "silent one"
//                clause, applied after the fact.
//
//   ABI 5 -> 6   (2026-09-24, perf/ffi-ctx-operand, phase-2 P2-I).  Every
//                cuSolverMp/cuBLASMp/active-subspace/SLATE/ScaLAPACK handler's
//                `ctx_handle` Attr (a heap address, so a new HLO and a
//                persistent-cache miss per process) became `ctx_key`, a hash
//                of the context configuration resolved through the per-.so
//                registry common/ctx_registry.{h,cc}; new C entry points
//                lrx_ctx_bind / lrx_ctx_unbind (leg-suffixed).
//
//   ABI 6 -> 7   (2026-10-16, slabmr/w8).  PhdfRead, PhdfReadMulti, PhdfWrite
//                and PhdfWriteIndependent lost their S64 control operands
//                (handle / offset / valid_shape, and read_multi's control
//                table).  SlabIO posts those values through
//                lrx_phdf5_post_control, keyed by the address of the call's
//                own buffer (reads: a caller-allocated destination aliased to
//                the result; writes: the operand), and the handler takes them
//                by that address (cpp/phdf5/ctx.h POSTED CONTROL).  An ABI-6
//                library would receive one Arg where it expects three: refused
//                by name at load instead.
//
//   ABI 7 -> 8   (2026-10-16, slabmr/w8).  PhdfReadKchunkUnion lost its three
//                S64 control operands (handle, offset table, count table) and
//                takes one destination aliased to its result; SlabIO posts
//                {ctx, ds_id, n_kchunk, N_file, offsets, this rank's counts}
//                under the destination's address (ctx.h kCtlReadKchunkUnion).
//                An ABI-7 library -- including one built from the first ABI-7
//                trees, whose union handler still took operands -- would get
//                one Arg where it expects three; refused by name at load.
//
//   ABI 8 -> 9   (2026-10-17, dlascale).  lorrax_slate_potrf takes one more
//                attribute, seed_min_bytes (the outside allowance past which a
//                rank's SLATE pool is seeded), and lorrax_slate_eigh one more
//                result, a u8 workspace seeding its device pools (sized by
//                distrib_la._slate.eigh_workspace_bytes; CUDA and ROCM legs).
//                potrf and batched_{solve_lu_factors,getrf,getrs} also bind
//                xla::ffi::ScratchAllocator for their per-rank pool seed
//                (slate/pool_seed.h); their operands and results are ABI 8's.
//                An ABI-8 library would reject the attribute and miss the result;
//                refused by name at load instead.
//
//   ABI 9 -> 10  (2026-10-19, largenmu).  lorrax_cusolvermp_eigh takes one more
//                result, the operand cuSOLVERMp destroys, aliased to its Arg
//                (input_output_aliases={0: 2}); the handler solves in it and its
//                scratch is the vendor workspace alone (no private operand tile),
//                and lrx_eigh_workspace_bytes_block reports that workspace alone.
//                An ABI-9 library would see one Ret more than it binds; refused
//                by name at load instead.
//
//   ABI 10 -> 11 (2026-10-23, SLATE removed).  The ScaLAPACK and ELPA handlers'
//                `ctx_key` now names a "mesh" context (cpp/mesh/; config
//                "lorrax-ctx/v1|mesh|..."), created by lrx_mesh_context_create;
//                the lorrax_slate_* handlers and lrx_slate_* entry points are
//                gone.  An ABI-10 library would refuse every new ctx_key as "not
//                a slate context"; refused by name at load instead.
//
//   ABI 11 -> 12 (2026-10-25, nested collective HDF5 dataset paths).
//                The required ds_name contract now includes parent-group
//                creation inside the same collective transaction (W0_lr/Ft).
//                No C signature moved. An ABI-11 provider could invalidate
//                the artifact and then fail to create that required path;
//                refuse it at load, before any data mutation instead.
//
//   ABI 12 -> 13 (2026-10-25, required cuSOLVERMp release floor).
//                CUDA contexts must reject cuSOLVERMp < 0.9.1 before grid
//                creation or workspace allocation on every grid. Older
//                providers have known large-N eigenvector failures. An
//                ABI-12 library can lack this refusal even with a valid
//                dependency closure; require the rebuilt native contract.
//
// ---------------------------------------------------------------------------
// THE RULE
// ---------------------------------------------------------------------------
// BUMP THIS ON ANY CHANGE TO A REQUIRED HANDLER CONTRACT. This stamp covers
// required behavior as well as binary signatures; it is not a source-version
// stamp. Concretely, in the same commit as any of:
//
//   * adding, removing or reordering an Arg or a Ret of an existing handler;
//   * moving a value between Attr and Arg (both recent bumps were this);
//   * changing the dtype or the rank of an existing Arg/Ret/Attr;
//   * changing the meaning of a positional value while keeping its type
//     (the silent one — same arity, different contract, no runtime error at
//     all, which is worse than the INVALID_ARGUMENT above).
//   * requiring newly supported behavior, even with the same signature
//     (e.g. creating ds_name's missing parent groups in SlabIO).
//
// DO NOT BUMP for: adding a whole new handler (an old library simply does not
// export it, and probe_target already reports that precisely); changing a
// handler's internals while preserving its required behavior; changing which
// vendor library a handler calls; anything that does not alter the required
// contract across the boundary.
//
// The version is deliberately NOT derived from a git hash.  A hash answers
// "were these built from the same source", which is a stricter question than
// the one that matters and would refuse every legitimate pairing of a rebuilt
// Python tree with a still-correct .so.  What has to match is the CONTRACT.
#pragma once

#define LORRAX_FFI_ABI_VERSION 13
