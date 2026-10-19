// shard_index.h — the sharding-encoding arithmetic every phdf5 handler
// runs before it touches HDF5, plus the two safety predicates that arithmetic
// needs.
//
// write_ffi.cc and read_ffi.cc each carried a private copy of
// ``unravel_rank`` / ``vec_to_string``; four dispatch bodies then indexed
// ``axis_flat`` through a running ``flat_idx`` that nothing bounded.  One
// header, one copy: the two TUs cannot drift, and a fix to the encoding
// checks lands in every handler at once.
//
// ─── Rank-invariance, which is the whole point ────────────────────────────
// Every input validated here is an XLA FFI *Attr* — baked into the compiled
// module, therefore identical on every rank by SPMD construction — plus
// ``ctx->rank`` / ``ctx->world_size``, fixed at open_file from
// jax.process_index() / jax.process_count().  So each predicate below
// reaches the SAME verdict on every rank: it refuses everywhere or nowhere.
// That is the only kind of refusal a collective tolerates (decisions.md
// 2026-08-04; measured cost of the other kind: a 306 s hang at P=4 and a
// silent 420 s read timeout).  Do not add a check here whose inputs are
// rank-local.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <hdf5.h>

#include "xla/ffi/api/ffi.h"

#include "ctx.h"

namespace lorrax_ffi::phdf5 {

namespace ffi = ::xla::ffi;

// INTERNAL LINKAGE, deliberately.  The originals in write_ffi.cc /
// read_ffi.cc were ``static``, so they never reached the .so's dynamic
// symbol table.  Plain header ``inline`` would give them vague linkage and
// export them, and both platform libraries are loaded RTLD_GLOBAL in a
// dual-platform process -- the reason the handlers themselves register under
// DISTINCT names (read_ffi.cc's registration note).  An anonymous namespace
// keeps one copy per TU and the exported-symbol set unchanged, which is the
// control every .so rebuild in this tree is checked against.
namespace {

// Un-ravel a linear rank id through a row-major mesh shape into per-axis
// coordinates.  rank = sum(coord[i] * prod(shape[i+1:])).  PRECONDITION:
// ``validate_shard_encoding`` has accepted ``mesh_shape`` — it is what rules
// out the zero entry that would divide by zero here, and the rank >= prod
// that would silently wrap this rank onto another rank's hyperslab.
[[maybe_unused]] std::vector<int64_t> unravel_rank(
    int64_t rank, ffi::Span<const int64_t> mesh_shape)
{
    std::vector<int64_t> coord(mesh_shape.size(), 0);
    int64_t r = rank;
    for (ssize_t i = (ssize_t)mesh_shape.size() - 1; i >= 0; --i) {
        coord[i] = r % mesh_shape[i];
        r /= mesh_shape[i];
    }
    return coord;
}

template <typename T>
std::string vec_to_string(const std::vector<T>& v)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    os << "]";
    return os.str();
}

[[maybe_unused]] std::string span_to_string(ffi::Span<const int64_t> v)
{
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    os << "]";
    return os.str();
}

// ─── What an impossible descriptor value ACTUALLY IS ─────────────────────
//
// The refusals below print a nonsense ``offset`` or ``valid_shape`` and stop.
// That is honest and useless: the operator sees `offset=-4642951212449158796`
// and has no way to tell a marshalling-width defect from a stale allocation
// from a caller bug.  Every such value measured on this project so far has
// had the SAME answer, and it is recoverable from the bit pattern in one
// line of arithmetic:
//
//   offset_base=4462667732332943029     -> f64  2.225039e-10   (2026-08-15, S3)
//   valid_shape=-9223372036854775808    -> f64 -0.000000e+00   (2026-08-17)
//   offset=-4642951212449158796         -> f64 -1.652707e-02   (2026-08-17)
//   offset_base=[0,0,0,4596944070643295330] -> f64 1.9e-01     (KNOWN_FAILURES L1)
//
// Four sightings, four small-magnitude IEEE-754 doubles.  An int64 index that
// decodes as a plausible float64 is not an arithmetic mistake and not an
// int32 field widened wrongly (a widened int32 leaves the top half zero or
// all-ones, i.e. an ABSURD double: an infinity, a NaN or ~1e-300).  It is
// PHYSICS DATA: the descriptor buffer was read while it still held whatever
// the allocator had left there, which has exactly two known causes on this
// stack, both fixed IN SOURCE and both invisible to a run pinned to an older
// library:
//
//   * the control-operand stream race (SLAB_IO_ROOT_CAUSE_AUDIT §A, fix B1) —
//     ``copy_index_to_host`` read the operand allocation before the XLA
//     stream wrote it;
//   * the cross-.so ``PhdfCtx`` ODR collision (KNOWN_FAILURES L1) — the
//     handler decoded the other build's struct at the wrong field offsets.
//
// So the classifier is worth its ten lines: it turns "impossible descriptor"
// into a named hypothesis with a repair, at the one moment the operator is
// looking.  It states its own uncertainty rather than asserting the cause.
[[maybe_unused]] std::string descriptor_forensics(int64_t v)
{
    double as_f64 = 0.0;
    std::memcpy(&as_f64, &v, sizeof(double));
    const double mag = as_f64 < 0 ? -as_f64 : as_f64;
    const bool plausible_float =
        (as_f64 == 0.0) || (mag > 1e-30 && mag < 1e30);
    std::ostringstream os;
    os << "  [descriptor forensics] raw=0x" << std::hex << (uint64_t)v
       << std::dec << " reinterpreted as f64 = " << as_f64;
    if (plausible_float) {
        os << " -- a SMALL-MAGNITUDE DOUBLE, i.e. this descriptor slot most "
              "likely held PHYSICS DATA and was read before its producer "
              "wrote it.  Both known causes are fixed in source and neither "
              "is in a library built before 2026-08-15: the control-operand "
              "stream race (fix B1) and the cross-.so PhdfCtx ODR collision "
              "(KNOWN_FAILURES L1).  CHECK THE LIBRARY FIRST: "
              "`strings -a \"$LORRAX_FFI_SO\" | grep -q 'stale or foreign "
              "ctx handle'` must succeed, and `nm -D` on the CUDA and host "
              "legs must share no lrx_/lorrax_ffi name.";
    } else {
        os << " -- NOT a plausible double, so the stale-allocation "
              "explanation does not fit this one.  Suspect the marshalling "
              "instead: dump the boundary with LORRAX_DEBUG_PRINT=1"
              "=1 and check that the Python side handed over int64.";
    }
    return os.str();
}

// Say it on stderr, NOW, flushed.
//
// Every error below is returned to XLA as an ffi::Error, and in the ordinary
// case Python prints it.  The case this exists for is the other one: when a
// handler refuses, the rank does not enter the collective its peers are
// inside, and the job then dies minutes later at a barrier deadline with the
// original message still buffered — stderr written just before process death
// is lost under srun+apptainer (audit 2026-08-04; two 32-node legs died with
// no traceback anywhere).  A flushed line is what survives.
//
// This is for ERROR paths only.  Clipping an overhanging slab is NOT an error
// and must stay silent — decisions.md 2026-08-04, "CLIPPING IS SILENT AND
// THAT IS INTENDED".
[[maybe_unused]] void announce_error(const PhdfCtx* ctx, const std::string& msg)
{
    std::fprintf(stderr, "[phdf5 ERROR rank=%d] %s\n",
                 ctx ? ctx->rank : -1, msg.c_str());
    std::fflush(stderr);
}

// The ctx-handle liveness check — SLAB_IO_ROOT_CAUSE_AUDIT.md B2.
//
// Call this on the value that came off seam 2 (``copy_index_to_host``) and
// BEFORE anything dereferences it — including before the ``fail`` lambdas,
// which announce through ``ctx->rank``.  ``ctx_is_live`` consults the
// registry context.cc maintains across open/close; an address that is not in
// it is a stale, doubly-closed, foreign-leg or raced handle, and the only
// safe thing to do with it is refuse by name.
//
// ON RANK-INVARIANCE, since this header's preamble forbids rank-local checks:
// the handle operand IS replicated by construction (the shard_map passes it
// with ``P()``), so in every correct execution this predicate reaches the
// same verdict everywhere, exactly like the ``!ctx`` test it sits beside.
// The case it fires in is by definition the incorrect one — a race, which is
// per-rank — and there a refusal on a proper subset is not the hazard the
// preamble is about.  The alternative on that path is a dereference of a
// garbage pointer: a hang with a printed reason beats a SIGSEGV with none.
[[maybe_unused]] bool check_live_ctx(
    const PhdfCtx* ctx, int64_t raw_handle, const char* who, std::string* err)
{
    size_t n_live = 0;
    if (ctx_is_live(ctx, &n_live)) return true;
    std::ostringstream os;
    os << who << ": stale or foreign ctx handle -- stream race or double "
          "close."
       << "  got=0x" << std::hex << (unsigned long long)raw_handle << std::dec
       << " (not in this library's live-ctx registry; " << n_live
       << " context(s) currently open)."
       << "  want= a handle returned by lrx_phdf5_open on THIS platform leg "
          "and not yet closed."
       << "  fix= see SLAB_IO_ROOT_CAUSE_AUDIT.md B1 (stream-order the "
          "control-operand copies) and B2 (this refusal); if B1 is already "
          "in the loaded .so, look for a close/use ordering bug or a handle "
          "minted by the other leg.";
    *err = os.str();
    announce_error(nullptr, *err);   // NOT through ctx: it may not be memory
    return false;
}

// prod(dims) * elt_bytes, refusing on overflow instead of wrapping.
//
// The wrapped product is not a cosmetic problem: it is what ``ensure_pinned``
// is asked to allocate and what the following ``memset`` uses, while H5Dread
// still transfers the true element count of the dataspace.  A wrap therefore
// buys a heap overflow with no HDF5 error.
[[maybe_unused]] bool checked_buffer_bytes(const std::vector<hsize_t>& dims,
                                 size_t elt_bytes, const char* who,
                                 size_t* out_bytes, std::string* err)
{
    size_t n = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        size_t d = (size_t)dims[i];
        if (d != 0 && n > SIZE_MAX / d) {
            std::ostringstream os;
            os << who << ": element count overflows size_t at dim " << i
               << " (shape " << vec_to_string(dims) << ")";
            *err = os.str();
            return false;
        }
        n *= d;
    }
    if (elt_bytes != 0 && n > SIZE_MAX / elt_bytes) {
        std::ostringstream os;
        os << who << ": buffer size overflows size_t ("
           << n << " elements x " << elt_bytes << " bytes)";
        *err = os.str();
        return false;
    }
    *out_bytes = n * elt_bytes;
    return true;
}

// Validate the (mesh_shape, axis_count_per_dim, axis_flat) encoding BEFORE
// anything indexes through it.
//
// Three distinct hazards, none of which produced a diagnostic before:
//
//  1. ``axis_flat`` is the concatenation of the per-dim axis lists, walked by
//     a running ``flat_idx``; nothing checked that the lengths agree, so a
//     mismatched attr triple read PAST THE END of the span — an out-of-bounds
//     read of XLA-owned memory whose value then became a mesh coordinate and
//     hence a file offset.  Silent wrong-hyperslab, not a crash.
//  2. ``mesh_shape`` entries reach ``%``/``/`` in unravel_rank unchecked: a 0
//     is SIGFPE, a negative is an implementation-defined coordinate.
//  3. ``prod(mesh_shape) != ctx->world_size`` means the un-ravel does not
//     describe this communicator.  With prod < world_size two ranks share a
//     coordinate and write the SAME hyperslab — undefined under collective
//     MPI-IO, and the dedup logic will not catch it because it dedups by
//     coordinate.  ``ffi.io.open_file`` already refuses ``p*q !=
//     jax.process_count()``, so this is that invariant restated where the
//     arithmetic actually happens.
[[maybe_unused]] bool validate_shard_encoding(
    const PhdfCtx* ctx, const char* who,
    ffi::Span<const int64_t> mesh_shape,
    ffi::Span<const int64_t> axis_count_per_dim,
    ffi::Span<const int64_t> axis_flat,
    std::string* err)
{
    std::ostringstream os;
    if (mesh_shape.size() == 0) {
        os << who << ": mesh_shape attr is empty; there is no mesh to "
           << "un-ravel rank " << (ctx ? ctx->rank : -1) << " through";
        *err = os.str();
        return false;
    }
    int64_t prod = 1;
    for (size_t i = 0; i < mesh_shape.size(); ++i) {
        const int64_t m = mesh_shape[i];
        if (m <= 0) {
            os << who << ": mesh_shape" << span_to_string(mesh_shape)
               << " has a non-positive entry at axis " << i
               << "; the rank un-ravel divides by it";
            *err = os.str();
            return false;
        }
        if (prod > INT64_MAX / m) {
            os << who << ": mesh_shape" << span_to_string(mesh_shape)
               << " overflows int64 at axis " << i;
            *err = os.str();
            return false;
        }
        prod *= m;
    }
    if (ctx != nullptr) {
        if (ctx->world_size > 0 && prod != (int64_t)ctx->world_size) {
            os << who << ": mesh_shape" << span_to_string(mesh_shape)
               << " has " << prod << " slots but this context was opened for "
               << ctx->world_size << " processes.  The per-rank hyperslab is "
               << "derived by un-ravelling the rank through that mesh, so a "
               << "mismatch either aliases two ranks onto ONE hyperslab "
               << "(undefined under collective MPI-IO) or sends a rank past "
               << "the mesh.  Refused identically on every rank.";
            *err = os.str();
            return false;
        }
        if (ctx->rank < 0 || (int64_t)ctx->rank >= prod) {
            os << who << ": rank " << ctx->rank << " is outside mesh_shape"
               << span_to_string(mesh_shape) << " (" << prod << " slots); "
               << "the un-ravel would wrap it onto another rank's hyperslab";
            *err = os.str();
            return false;
        }
    }
    int64_t total = 0;
    for (size_t d = 0; d < axis_count_per_dim.size(); ++d) {
        const int64_t na = axis_count_per_dim[d];
        if (na < 0) {
            os << who << ": axis_count_per_dim"
               << span_to_string(axis_count_per_dim)
               << " is negative at dim " << d;
            *err = os.str();
            return false;
        }
        if (na > (int64_t)mesh_shape.size()) {
            os << who << ": axis_count_per_dim[" << d << "]=" << na
               << " exceeds the mesh rank " << mesh_shape.size();
            *err = os.str();
            return false;
        }
        total += na;
    }
    if ((size_t)total != axis_flat.size()) {
        os << who << ": axis_count_per_dim"
           << span_to_string(axis_count_per_dim) << " sums to " << total
           << " but axis_flat" << span_to_string(axis_flat) << " has "
           << axis_flat.size() << " entries.  Walking axis_flat with a "
           << "running index would read past its end.";
        *err = os.str();
        return false;
    }
    return true;
}

// The dataset must have the rank the caller's index vectors have, checked
// BEFORE they are handed to H5S*.  H5Sselect_hyperslab and
// H5Sget_simple_extent_dims both read `file_rank` entries from the pointers
// we pass, so an (N+k)-D dataset reads k entries past the end of an N-entry
// vector.  ``write_ffi.cc`` has always had this guard and ``read_ffi.cc``
// gained it in the 2026-08-04 audit; these are its remaining siblings.
// Rank-invariant: same dataset, same request, on every rank.
[[maybe_unused]] bool check_dataset_rank(hid_t filespace, int want_rank,
                               const char* who, int* file_rank_out,
                               std::string* err)
{
    const int file_rank = H5Sget_simple_extent_ndims(filespace);
    if (file_rank_out) *file_rank_out = file_rank;
    if (file_rank == want_rank) return true;
    std::ostringstream os;
    os << who << ": dataset rank mismatch file_rank=" << file_rank
       << " request_rank=" << want_rank
       << " -- refused identically on every rank";
    *err = os.str();
    return false;
}

}  // namespace

}  // namespace lorrax_ffi::phdf5
