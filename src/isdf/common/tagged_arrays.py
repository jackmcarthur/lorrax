import numpy as np
from .gpu_utils import cp, xp, cufft, GPU_AVAILABLE
from numpy import s_
from typing import Union

## LabeledArray module
"""
This module defines a lightweight wrapper around NumPy/CuPy arrays that tracks named axes and supports fast, in-place operations for high-performance electronic structure computations. Axes can be reshaped, transposed, joined, sliced, and Fourier transformed by name instead of index.

In upcoming versions the low-level array operations will be redirected to a
distributed linear algebra backend developed in-house.  Keeping a thin wrapper
around the raw arrays will make that transition mostly transparent to user code.

### Key features:
- Named axes: access and manipulate dimensions without hardcoding index positions
- In-place operations: `.join(...)`, `.unjoin(...)`, `.fft_kgrid()`, `.transpose(...)` mutate the array without returning a copy
- Axis tracking: joins/unjoins maintain internal metadata (`joined_axes`) for reversible reshaping
- CuPy integration: GPU-accelerated FFTs with memory locality via `cupyx.scipy.fftpack.get_fft_plan`
- Initialization from just shape and axis names: `LabeledArray(axes, shape)` creates zero-initialized complex array

Example usage:

from numpy import s_

# Initialize
A = LabeledArray(axes=['nfreq', 'nkx', 'nky', 'nkz', 'nspinor_1', 'nr1', 'nspinor_2', 'nr2'],
                 shape=(8, 4, 4, 4, 2, 5, 2, 5))

# In-place transformations
A.kgrid_to_last()
A.freq_to_last()
A.join('nspinor_1', 'nr1')
A.slice_many({'nkx': s_[1:3], 'nfreq': 0})
A.fft_kgrid()

# Transpose (not in-place): returns a new object with reordered axes and contiguous memory
B = A.transpose('nfreq', 'nspinor_1', 'nr1', 'nspinor_2', 'nr2', 'nkx', 'nky', 'nkz')
print(B.axes)  # ['nfreq', 'nspinor_1', 'nr1', 'nspinor_2', 'nr2', 'nkx', 'nky', 'nkz']
print(A.axes)  # unchanged in A
"""

class LabeledArray:
    __slots__ = ['data', 'axes', 'xp', 'joined_axes', 'original_sizes']

    def __init__(self, data=None, axes=None, shape=None, dtype=np.complex128, joined_axes=None):
        if data is not None:
            assert axes is not None
            self.data = data
            self.xp = cp.get_array_module(data)
            self.axes = list(axes)
        elif axes is not None and shape is not None:
            # interpret None as newaxis (size 1)
            shape_resolved = tuple(1 if dim is None else dim for dim in shape)
            if GPU_AVAILABLE:
                self.xp = cp
            else:
                self.xp = np
            self.data = self.xp.zeros(shape_resolved, dtype=dtype)
            self.axes = list(axes)
        else:
            raise ValueError("Either data+axes or axes+shape must be provided.")
            
        # Initialize original_sizes with current shape
        self.original_sizes = {ax: self.data.shape[i] for i, ax in enumerate(self.axes)}
        self.joined_axes = joined_axes or {}

    def shape(self, axis_name=None):
        if axis_name is None:
            return self.data.shape
        else:
            if axis_name in self.axes:
                return self.data.shape[self.axes.index(axis_name)]
            else:
                raise ValueError(f"Axis '{axis_name}' not found in axes {self.axes}")


    def transpose(self, *new_order):
        if set(new_order) != set(self.axes):
            raise ValueError("new_order must be a permutation of current axes.")
        perm = [self.axes.index(ax) for ax in new_order]
        new_data = self.xp.ascontiguousarray(self.data.transpose(perm))
        # build the new object, carrying over the old joined_axes
        new = LabeledArray(new_data,
                           axes=list(new_order),
                           joined_axes=dict(self.joined_axes))
        # preserve the original sizes so unjoin() still works
        new.original_sizes = dict(self.original_sizes)
        return new

    def slice(self, axis_name: str, slice_val: Union[int, slice], tagged=False):
        idx = self.axes.index(axis_name)
        slicer = [slice(None)] * self.data.ndim
        slicer[idx] = slice_val
        new_data = self.data[tuple(slicer)]
        if not tagged:
            return new_data
        new_axes = self.axes[:idx] + self.axes[idx+1:] if isinstance(slice_val, int) else self.axes
        return LabeledArray(new_data, new_axes, joined_axes=dict(self.joined_axes))

    def slice_many(self, slice_dict: dict, tagged=False):
        slicer = [slice(None)] * self.data.ndim
        new_axes = list(self.axes)
        
        for ax, s_val in slice_dict.items():
            if ax not in self.axes:
                raise ValueError(f"Axis '{ax}' not found.")
            idx = self.axes.index(ax)
            slicer[idx] = s_val
            if isinstance(s_val, int):
                new_axes[idx] = None  # mark for deletion
        
        # Remove any axes marked for deletion
        new_axes = [ax for ax in new_axes if ax is not None]
        new_data = self.data[tuple(slicer)]
        
        if not tagged:
            return new_data
        
        return LabeledArray(new_data, new_axes, joined_axes=dict(self.joined_axes))

    def shape_dict(self):
        return {name: self.data.shape[i] for i, name in enumerate(self.axes)}

    def kgrid_to_last(self):
        grid_axes = ['nkx', 'nky', 'nkz']
        front = [a for a in self.axes if a not in grid_axes]
        return self.transpose(*front, *grid_axes)

    def freq_to_last(self):
        if 'nfreq' not in self.axes:
            raise ValueError("nfreq axis not present.")
        front = [a for a in self.axes if a != 'nfreq']
        return self.transpose(*front, 'nfreq')

    def join(self, *axes_to_join):
        axes_to_join = list(axes_to_join)
        idxs = [self.axes.index(ax) for ax in axes_to_join]
        if sorted(idxs) != list(range(min(idxs), max(idxs)+1)):
            raise ValueError("Axes must be contiguous to join.")

        # Store original sizes using individual axis names
        for ax, idx in zip(axes_to_join, idxs):
            self.original_sizes[ax] = self.data.shape[idx]

        new_dim = int(np.prod([self.data.shape[i] for i in idxs]))
        new_shape = (
            self.data.shape[:idxs[0]] +
            (new_dim,) +
            self.data.shape[idxs[-1]+1:]
        )
        new_axes = (
            self.axes[:idxs[0]] +
            ['*'.join(axes_to_join)] +
            self.axes[idxs[-1]+1:]
        )
        self.data = self.data.reshape(new_shape)
        self.axes = list(new_axes)
        self.joined_axes['*'.join(axes_to_join)] = axes_to_join

    def unjoin(self, *original_axes):
        joined_name = '*'.join(original_axes)
        if joined_name not in self.joined_axes:
            raise ValueError(f"Axes {original_axes} are not currently joined.")

        idx = self.axes.index(joined_name)
        
        # Get original sizes from stored individual axis sizes
        recovered_shapes = [self.original_sizes[ax] for ax in original_axes]
        
        new_shape = (
            self.data.shape[:idx] +
            tuple(recovered_shapes) +
            self.data.shape[idx+1:]
        )
        self.data = self.data.reshape(new_shape)
        self.axes = self.axes[:idx] + list(original_axes) + self.axes[idx+1:]
        del self.joined_axes[joined_name]

    def fft_kgrid(self):
        self._apply_fft_inplace(['nkx', 'nky', 'nkz'], inverse=False)

    def ifft_kgrid(self):
        self._apply_fft_inplace(['nkx', 'nky', 'nkz'], inverse=True)

    def _apply_fft_inplace(self, k_axes, inverse=False):
        k_idxs = [self.axes.index(ax) for ax in k_axes]

        if GPU_AVAILABLE:
            plan = cufft.get_fft_plan(self.data, axes=tuple(k_idxs))
            func = cufft.ifftn if inverse else cufft.fftn
            self.data[...] = func(
                self.data,
                axes=k_idxs,
                norm='ortho',
                plan=plan,
                overwrite_x=True,
            )
        else:
            func = np.fft.ifftn if inverse else np.fft.fftn
            self.data[...] = func(self.data, axes=k_idxs, norm='ortho')

    def __repr__(self):
        return f"LabeledArray(shape={self.data.shape}, axes={self.axes}, joined_axes={self.joined_axes})"


class WfnArray:
    """
    Class to hold both wavefunction coefficients and energies together.
    Uses slots for memory efficiency and to prevent accidental attribute creation.
    """
    __slots__ = ('psi', 'enk')
    
    def __init__(self, psi: LabeledArray, enk: LabeledArray):
        """
        Initialize WfnArray from existing LabeledArrays.
        
        Args:
            psi: LabeledArray containing wavefunction coefficients
                Expected axes: ['nk', 'nb', 'nspinor', 'nrmu']
            enk: LabeledArray containing energies
                Expected axes: ['nb', 'nk']
        """
        # Verify input types
        if not isinstance(psi, LabeledArray) or not isinstance(enk, LabeledArray):
            raise TypeError("Both psi and enk must be LabeledArray instances")
            
        # Verify matching dimensions
        if psi.shape('nk') != enk.shape('nk') or psi.shape('nb') != enk.shape('nb'):
            raise ValueError("Incompatible shapes between psi and enk arrays")
            
        self.psi = psi
        self.enk = enk