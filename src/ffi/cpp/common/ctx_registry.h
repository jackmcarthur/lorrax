// ctx_registry.h — the per-process map from a CONFIGURATION key to a live
// native context (cuSolverMp/cuBLASMp, SLATE, ScaLAPACK-on-SlateCtx).
//
// WHY.  An FFI attribute is baked into the HLO, so it must be a pure
// function of the configuration.  The vendor handlers used to take the
// context's heap address as the `ctx_handle` attribute: every process
// produced a different module, and the persistent compile cache could never
// hit one (17 of 36 warm-run misses on the MoS2 bispinor chain were the
// cuBLASMp GEMM plans alone).  The attribute is now `ctx_key`, a hash of
// the context's configuration computed identically on every rank and every
// run (distrib_la._ctx_key); the address lives HERE, bound once when
// Python creates the context.  Unlike the phdf5 handle buffer, reading it
// needs no device-to-host copy, so an asynchronous GEMM stays asynchronous.
//
// SAFETY.  Each binding carries the full configuration string beside the
// key: re-binding a key to a different configuration or a different live
// handle refuses by name (a 63-bit collision is loud, never wrong), and
// resolving checks the library family.  Destroying a context forgets every
// key bound to it, so an executable run after teardown refuses instead of
// dereferencing freed memory.
//
// Per library: the table lives in ctx_registry.cc, inside `lorrax_ffi::`,
// which the version scripts keep local to each .so; each platform leg has
// its own registry, and Python binds a context in the leg that created it.
#pragma once

#include <cstdint>
#include <string>

namespace lorrax_ffi::ctx_registry {

// Bind `key` to (`handle`, `config`).  "" on success, including an
// identical re-bind; otherwise the refusal text.
std::string bind(int64_t key, int64_t handle, const std::string& config);

// Drop the binding of `key` (no-op if unbound).
void unbind(int64_t key);

// Drop every key bound to `handle` (called by the context destroy paths).
void forget_handle(int64_t handle);

// The live handle bound to `key`, provided its configuration belongs to the
// library `family` ("cusolvermp" or "slate"); 0 otherwise, with the reason
// written to stderr once per key.  Callers keep their existing null check.
int64_t resolve(int64_t key, const char* family);

}  // namespace lorrax_ffi::ctx_registry
