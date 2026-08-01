// scalapack_descriptor.h — pure-math helpers for 2-D block-cyclic layouts.
//
// ScaLAPACK, cuSOLVERMp, and ELPA all share the same block-cyclic
// distribution conventions (column-major tiles, (rsrc, csrc) grid origin,
// numroc-style local-count formula).  These helpers stay library-agnostic
// so both subpackages can use them.

#pragma once

#include <cstdint>

namespace lorrax_ffi::scalapack {

// numroc(n, mb, myrow, rsrc, nprow) — number of matrix rows owned by
// process `myrow` under block-cyclic distribution of `n` global rows
// with block size `mb`, grid origin `rsrc`, and `nprow` process rows.
//
// Direct translation of ScaLAPACK's NUMROC; cusolverMp exports the same
// function as cusolverMpNUMROC.  Provided here so we don't have to
// round-trip through a library call for layout-only math.
inline int64_t numroc(int64_t n, int64_t mb, int myrow, int rsrc, int nprow) {
    int64_t whoseFirstBlock = (myrow - rsrc + nprow) % nprow;
    int64_t nBlocks         = n / mb;
    int64_t myNBlocks       = nBlocks / nprow +
                              (whoseFirstBlock < (nBlocks % nprow) ? 1 : 0);
    int64_t rem             = n - nBlocks * mb;
    int64_t myExtra         = 0;
    if (whoseFirstBlock == (nBlocks % nprow)) myExtra = rem;
    return myNBlocks * mb + myExtra;
}

// Convenience: the exact local tile size given a global (rows, cols)
// distributed on a (p, q) grid with tile (mb, nb), origin (0, 0).
struct LocalTileShape {
    int64_t rows;
    int64_t cols;
};
inline LocalTileShape local_tile_shape(int64_t n_rows, int64_t n_cols,
                                       int64_t mb, int64_t nb,
                                       int myrow, int mycol,
                                       int nprow, int npcol) {
    return {
        numroc(n_rows, mb, myrow, 0, nprow),
        numroc(n_cols, nb, mycol, 0, npcol),
    };
}

// For the special "one tile per rank" case (mb = n_rows/nprow,
// nb = n_cols/npcol, n_rows % nprow == 0, n_cols % npcol == 0) the
// local shape is exactly (n_rows/nprow, n_cols/npcol).  This matches
// JAX's NamedSharding(P('x','y')) block layout.
inline LocalTileShape one_tile_per_rank(int64_t n_rows, int64_t n_cols,
                                        int nprow, int npcol) {
    return {n_rows / nprow, n_cols / npcol};
}

}  // namespace lorrax_ffi::scalapack
