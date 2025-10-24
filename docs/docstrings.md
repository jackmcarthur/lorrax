## Docstring style (NumPy)

Use NumPy-style docstrings so tools like pdoc can generate readable Markdown.

```python
def read_Gvecs_to_devices(wfn, sym, bandrange, meta, bispinor, mesh_xy):
    """
    Load plane-wave coefficients cnk(G) and assemble a globally sharded FFT box.

    Parameters
    ----------
    wfn : WFNReader
        Wavefunction reader (plane-wave DFT I/O).
    sym : SymMaps
        Symmetry maps providing k-point mappings and G-vector utilities.
    bandrange : tuple[int, int]
        Inclusive-exclusive band window (start, end) to load.
    meta : Meta
        System metadata and FFT grid sizes.
    bispinor : bool
        If True, constructs 4-component spinors by adding small components.
    mesh_xy : Mesh
        JAX mesh used to shard across the band axis (x,y).

    Returns
    -------
    global_psi_Gtot : jax.Array
        Sharded array (nk, nb_pad, nspinor, nx, ny, nz) in G-space.
    nb_actual : int
        Number of requested (un-padded) bands.
    """
    ...
```

Tips:
- Start with a one-line summary, then a short paragraph if needed.
- Document shapes and units for arrays.
- Prefer explicit types for parameters and returns.
- Add small Examples blocks for nontrivial functions.

