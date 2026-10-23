"""One-pass typed grid pullbacks agree with literal affine inversion."""
import itertools
import weakref

import numpy as np
from ffi import _services
_services.ensure_on_path()
from symmetry_maps import SymMaps
from symmetry_maps.orbit_syms import _fft_grid_pullback_memo


def _typed_rows():
    rotation = np.asarray([[0,-1,0],[1,-1,0],[0,0,1]], dtype=np.int64)
    powers = (np.eye(3, dtype=np.int64), rotation, rotation@rotation)
    matrices = np.asarray([m for _ in range(4) for m in powers])
    translations = np.asarray([[0.,0.,j/4] for j in range(4) for _ in powers])*2*np.pi
    sym = SymMaps.__new__(SymMaps)
    sym.sym_matrices = matrices
    sym.sym_mats_k = np.concatenate((matrices,-matrices))
    sym.translations = translations
    return sym


def test_one_pass_pullback_is_exact_read_only_and_not_retained():
    sym = _typed_rows()
    grid = np.asarray([8,8,12])
    refs = []
    _fft_grid_pullback_memo.cache_clear()
    try:
        for row in range(12):
            table = sym.fft_grid_pullback(np.asarray([row]), grid, cache=False)
            expected = np.empty(int(np.prod(grid)), dtype=np.int32)
            shift = np.rint(sym.translations[row]/(2*np.pi)*grid).astype(np.int64)
            # Independent inverse action at each output grid point:
            # r_old = S*(r_new-tau), on this commensurate integer grid.
            for flat,new in enumerate(itertools.product(*(range(int(n)) for n in grid))):
                old = (sym.sym_matrices[row]@(np.asarray(new)-shift))%grid
                expected[flat] = np.ravel_multi_index(old, tuple(grid))
            np.testing.assert_array_equal(table[0], expected)
            assert not table.flags.writeable
            refs.append(weakref.ref(table))
            del table
        assert _fft_grid_pullback_memo.cache_info().currsize==0
        assert all(ref() is None for ref in refs)
        first = sym.fft_grid_pullback(np.asarray([0]), grid)
        assert first is sym.fft_grid_pullback(np.asarray([0]), grid)
        before = _fft_grid_pullback_memo.cache_info()
        transient = sym.fft_grid_pullback(np.asarray([0]), grid, cache=False)
        assert transient is not first
        np.testing.assert_array_equal(transient, first)
        assert not transient.flags.writeable
        assert _fft_grid_pullback_memo.cache_info()==before
        # Antiunitary rows use the same spatial pullback for a scalar field.
        np.testing.assert_array_equal(
            sym.fft_grid_pullback(np.asarray([12]), grid, cache=False), first)
    finally:
        _fft_grid_pullback_memo.cache_clear()
