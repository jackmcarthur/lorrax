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
// ---------------------------------------------------------------------------
// THE RULE
// ---------------------------------------------------------------------------
// BUMP THIS ON ANY CHANGE TO A HANDLER SIGNATURE.  Concretely, in the same
// commit as any of:
//
//   * adding, removing or reordering an Arg or a Ret of an existing handler;
//   * moving a value between Attr and Arg (both recent bumps were this);
//   * changing the dtype or the rank of an existing Arg/Ret/Attr;
//   * changing the meaning of a positional value while keeping its type
//     (the silent one — same arity, different contract, no runtime error at
//     all, which is worse than the INVALID_ARGUMENT above).
//
// DO NOT BUMP for: adding a whole new handler (an old library simply does not
// export it, and probe_target already reports that precisely); changing a
// handler's internals; changing which vendor library a handler calls; anything
// that does not alter what crosses the boundary.
//
// The version is deliberately NOT derived from a git hash.  A hash answers
// "were these built from the same source", which is a stricter question than
// the one that matters and would refuse every legitimate pairing of a rebuilt
// Python tree with a still-correct .so.  What has to match is the CONTRACT.
#pragma once

#define LORRAX_FFI_ABI_VERSION 3
