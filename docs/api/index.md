# Table of Contents

* [isdf.gw\_isdf.get\_windows](#isdf.gw_isdf.get_windows)
  * [compute\_dos](#isdf.gw_isdf.get_windows.compute_dos)
  * [find\_optimal\_partitions](#isdf.gw_isdf.get_windows.find_optimal_partitions)
  * [minimize\_cost\_fn](#isdf.gw_isdf.get_windows.minimize_cost_fn)
  * [get\_window\_info](#isdf.gw_isdf.get_windows.get_window_info)
* [isdf.gw\_isdf.cohsex\_jax](#isdf.gw_isdf.cohsex_jax)
  * [read\_cohsex\_input](#isdf.gw_isdf.cohsex_jax.read_cohsex_input)
  * [get\_bandranges](#isdf.gw_isdf.cohsex_jax.get_bandranges)
  * [get\_zeta\_q\_and\_v\_q\_mu\_nu](#isdf.gw_isdf.cohsex_jax.get_zeta_q_and_v_q_mu_nu)
  * [get\_G\_mu\_nu\_jax](#isdf.gw_isdf.cohsex_jax.get_G_mu_nu_jax)
  * [get\_G\_R\_jax](#isdf.gw_isdf.cohsex_jax.get_G_R_jax)
  * [get\_sigma\_x\_mu\_nu\_jax](#isdf.gw_isdf.cohsex_jax.get_sigma_x_mu_nu_jax)
  * [get\_sigma\_x\_kij\_jax](#isdf.gw_isdf.cohsex_jax.get_sigma_x_kij_jax)
  * [compute\_sigma\_pipeline\_jax](#isdf.gw_isdf.cohsex_jax.compute_sigma_pipeline_jax)
  * [preprocess\_q\_loops](#isdf.gw_isdf.cohsex_jax.preprocess_q_loops)
* [isdf.gw\_isdf.w\_isdf](#isdf.gw_isdf.w_isdf)
  * [get\_chi\_lm\_Yt\_jax](#isdf.gw_isdf.w_isdf.get_chi_lm_Yt_jax)
  * [get\_chi0\_jax](#isdf.gw_isdf.w_isdf.get_chi0_jax)
  * [get\_static\_w\_q\_jax](#isdf.gw_isdf.w_isdf.get_static_w_q_jax)
* [isdf.gw\_isdf.vcoul](#isdf.gw_isdf.vcoul)
  * [wrap\_points\_to\_voronoi](#isdf.gw_isdf.vcoul.wrap_points_to_voronoi)
* [isdf.gw\_isdf.cohsex\_jax\_deprecated](#isdf.gw_isdf.cohsex_jax_deprecated)
  * [read\_cohsex\_input](#isdf.gw_isdf.cohsex_jax_deprecated.read_cohsex_input)
  * [get\_bandranges](#isdf.gw_isdf.cohsex_jax_deprecated.get_bandranges)
  * [wrap\_points\_to\_voronoi](#isdf.gw_isdf.cohsex_jax_deprecated.wrap_points_to_voronoi)
  * [fft\_bandrange](#isdf.gw_isdf.cohsex_jax_deprecated.fft_bandrange)
  * [get\_zeta\_q\_and\_v\_q\_mu\_nu](#isdf.gw_isdf.cohsex_jax_deprecated.get_zeta_q_and_v_q_mu_nu)
  * [get\_sigma\_x\_kij](#isdf.gw_isdf.cohsex_jax_deprecated.get_sigma_x_kij)
  * [find\_qpoint\_index](#isdf.gw_isdf.cohsex_jax_deprecated.find_qpoint_index)
  * [write\_labeled\_arrays\_to\_h5](#isdf.gw_isdf.cohsex_jax_deprecated.write_labeled_arrays_to_h5)
  * [read\_labeled\_arrays\_from\_h5](#isdf.gw_isdf.cohsex_jax_deprecated.read_labeled_arrays_from_h5)
* [isdf.gw\_isdf.cohsex\_isdf](#isdf.gw_isdf.cohsex_isdf)
  * [read\_cohsex\_input](#isdf.gw_isdf.cohsex_isdf.read_cohsex_input)
  * [get\_bandranges](#isdf.gw_isdf.cohsex_isdf.get_bandranges)
  * [wrap\_points\_to\_voronoi](#isdf.gw_isdf.cohsex_isdf.wrap_points_to_voronoi)
  * [fft\_bandrange](#isdf.gw_isdf.cohsex_isdf.fft_bandrange)
  * [get\_zeta\_q\_and\_v\_q\_mu\_nu](#isdf.gw_isdf.cohsex_isdf.get_zeta_q_and_v_q_mu_nu)
  * [get\_sigma\_x\_kij](#isdf.gw_isdf.cohsex_isdf.get_sigma_x_kij)
  * [find\_qpoint\_index](#isdf.gw_isdf.cohsex_isdf.find_qpoint_index)
  * [write\_labeled\_arrays\_to\_h5](#isdf.gw_isdf.cohsex_isdf.write_labeled_arrays_to_h5)
  * [read\_labeled\_arrays\_from\_h5](#isdf.gw_isdf.cohsex_isdf.read_labeled_arrays_from_h5)
* [isdf.gw\_isdf.gw\_file\_io](#isdf.gw_isdf.gw_file_io)
  * [write\_labeled\_arrays\_to\_h5](#isdf.gw_isdf.gw_file_io.write_labeled_arrays_to_h5)
  * [read\_labeled\_arrays\_from\_h5](#isdf.gw_isdf.gw_file_io.read_labeled_arrays_from_h5)
  * [load\_labeled\_arrays\_from\_h5](#isdf.gw_isdf.gw_file_io.load_labeled_arrays_from_h5)
  * [save\_restart\_per\_proc](#isdf.gw_isdf.gw_file_io.save_restart_per_proc)
* [isdf.common.wfnreader](#isdf.common.wfnreader)
  * [WFNReader](#isdf.common.wfnreader.WFNReader)
    * [\_\_init\_\_](#isdf.common.wfnreader.WFNReader.__init__)
    * [get\_cnk](#isdf.common.wfnreader.WFNReader.get_cnk)
    * [get\_gvec\_nk](#isdf.common.wfnreader.WFNReader.get_gvec_nk)
* [isdf.common.epsreader](#isdf.common.epsreader)
  * [EPSReader](#isdf.common.epsreader.EPSReader)
    * [\_\_init\_\_](#isdf.common.epsreader.EPSReader.__init__)
    * [\_\_del\_\_](#isdf.common.epsreader.EPSReader.__del__)
    * [get\_eps\_matrix](#isdf.common.epsreader.EPSReader.get_eps_matrix)
    * [get\_eps\_minus\_delta\_matrix](#isdf.common.epsreader.EPSReader.get_eps_minus_delta_matrix)
    * [get\_eps\_diagonal](#isdf.common.epsreader.EPSReader.get_eps_diagonal)
* [isdf.common.load\_wfns](#isdf.common.load_wfns)
  * [get\_enk\_bandrange](#isdf.common.load_wfns.get_enk_bandrange)
  * [read\_Gvecs\_to\_devices](#isdf.common.load_wfns.read_Gvecs_to_devices)
  * [get\_sharded\_wfns](#isdf.common.load_wfns.get_sharded_wfns)
* [isdf.common.tagged\_arrays](#isdf.common.tagged_arrays)
  * [WfnArray](#isdf.common.tagged_arrays.WfnArray)
    * [\_\_init\_\_](#isdf.common.tagged_arrays.WfnArray.__init__)
* [isdf.common.symmetry\_maps](#isdf.common.symmetry_maps)
  * [SymMaps](#isdf.common.symmetry_maps.SymMaps)
    * [\_\_init\_\_](#isdf.common.symmetry_maps.SymMaps.__init__)
    * [create\_kpoint\_symmetry\_map](#isdf.common.symmetry_maps.SymMaps.create_kpoint_symmetry_map)
    * [syms\_crystal\_to\_cartesian](#isdf.common.symmetry_maps.SymMaps.syms_crystal_to_cartesian)
    * [get\_spinor\_rotations](#isdf.common.symmetry_maps.SymMaps.get_spinor_rotations)
    * [get\_kminusq\_map](#isdf.common.symmetry_maps.SymMaps.get_kminusq_map)
    * [find\_qpoint\_index](#isdf.common.symmetry_maps.SymMaps.find_qpoint_index)
* [isdf.isdf\_init.kmeans\_isdf](#isdf.isdf_init.kmeans_isdf)
  * [interpolate\_density](#isdf.isdf_init.kmeans_isdf.interpolate_density)
  * [plot\_density\_and\_centroids](#isdf.isdf_init.kmeans_isdf.plot_density_and_centroids)
  * [weighted\_kmeans\_jax](#isdf.isdf_init.kmeans_isdf.weighted_kmeans_jax)
* [isdf.isdf\_init.get\_charge\_density](#isdf.isdf_init.get_charge_density)
  * [perform\_fft\_3d](#isdf.isdf_init.get_charge_density.perform_fft_3d)
  * [calculate\_charge\_density](#isdf.isdf_init.get_charge_density.calculate_charge_density)
  * [save\_charge\_density](#isdf.isdf_init.get_charge_density.save_charge_density)
  * [analyze\_gvectors](#isdf.isdf_init.get_charge_density.analyze_gvectors)

<a id="isdf.gw_isdf.get_windows"></a>

# isdf.gw\_isdf.get\_windows

get_windows.py

Compute optimal energy windows for the conduction and valence bands (minimizing total quadrature points)
for the O(N^3) polarizability and self energy calculations given in Kim, Martyna, and Ismail-Beigi, PRB 101, 035139 (2020).

These energy windows also define the discrete imaginary-time grids used by the
CTSP method.  Once we move beyond static COHSEX these same windows will control
the frequency resolution of the full GW calculations.

<a id="isdf.gw_isdf.get_windows.compute_dos"></a>

#### compute\_dos

```python
def compute_dos(wfn_file, n_points=2000)
```

Load all band energies from WFN file, flatten them,
and compute a Gaussian‑broadened DOS on a linear grid.

Args:
    wfn_file (str): path to WFN .h5 file
    n_points (int): number of energy grid points

Returns:
    energies (np.ndarray): 1D array of length n_points
    dos      (np.ndarray): 1D DOS array of same length

<a id="isdf.gw_isdf.get_windows.find_optimal_partitions"></a>

#### find\_optimal\_partitions

```python
def find_optimal_partitions(energies, n_windows)
```

Find the optimal partition points for a given number of windows.

Args:
    energies (np.ndarray): Sorted array of energies.
    n_windows (int): Number of windows to partition into.

Returns:
    partitions (list): List of partition indices.

<a id="isdf.gw_isdf.get_windows.minimize_cost_fn"></a>

#### minimize\_cost\_fn

```python
def minimize_cost_fn(wfn,
                     epsq,
                     max_val_windows=8,
                     max_cond_windows=8,
                     cond_bounds=None,
                     nband_max=None)
```

Minimize the cost function by finding optimal partition points for valence and conduction windows.

Args:
    wfn (WFNReader): Wavefunction reader object.
    epsq (float): Epsilon squared value.
    max_val_windows (int): Maximum number of valence windows.
    max_cond_windows (int): Maximum number of conduction windows.
    cond_bounds (tuple[float,float]|None): Optional (emin, emax) to restrict conduction energies.
    nband_max (int|None): If provided, restrict energies to the first nband_max bands.

Returns:
    cost_matrix (np.ndarray): Cost matrix for each combination of valence and conduction windows.
    window_bounds (dict): Dictionary of window boundaries for each combination.

<a id="isdf.gw_isdf.get_windows.get_window_info"></a>

#### get\_window\_info

```python
def get_window_info(epsq, wfn, cond_bounds=None, nband_max=None)
```

Calculate window information for valence and conduction bands.

Args:
    epsq (float): Epsilon squared value.
    wfn (WFNReader): Wavefunction reader object.
    cond_bounds (tuple[float,float]|None): Optional (emin, emax) for conduction energies to restrict to active band range.
    nband_max (int|None): If provided, restrict energies to the first nband_max bands.

Returns:
    list: A list of WindowPair objects containing window information.

<a id="isdf.gw_isdf.cohsex_jax"></a>

# isdf.gw\_isdf.cohsex\_jax

<a id="isdf.gw_isdf.cohsex_jax.read_cohsex_input"></a>

#### read\_cohsex\_input

```python
def read_cohsex_input(filename: str) -> dict
```

Parse input file for the COHSEX driver, allowing a QE K_POINTS block.

We extract the [cohsex] section using a substring to avoid configparser
errors from non-INI blocks like K_POINTS. The K_POINTS {crystal_b} block
is parsed manually and returned under 'kpoints_crystal_b'.

<a id="isdf.gw_isdf.cohsex_jax.get_bandranges"></a>

#### get\_bandranges

```python
def get_bandranges(nv, nc, nband, nelec)
```

Return ranges of bands necessary for \sigma_{X,SX,COH}

<a id="isdf.gw_isdf.cohsex_jax.get_zeta_q_and_v_q_mu_nu"></a>

#### get\_zeta\_q\_and\_v\_q\_mu\_nu

```python
def get_zeta_q_and_v_q_mu_nu(wfn,
                             sym,
                             bandrange_l,
                             bandrange_r,
                             V_qG,
                             meta: Meta,
                             psi_l_rtot,
                             psi_r_rtot,
                             psi_l_rmu,
                             psi_r_rmu,
                             psi_l_rmuT,
                             psi_r_rmuT,
                             *,
                             preprocessed_q_data=None,
                             bispinor=False,
                             mesh_xy=None)
```

Find the interpolative separable density fitting representation.

<a id="isdf.gw_isdf.cohsex_jax.get_G_mu_nu_jax"></a>

#### get\_G\_mu\_nu\_jax

```python
def get_G_mu_nu_jax(psi_vTX, psi_vY)
```

Pure: psi_* (nk, nb, nspinor, n_rmu) -> G_k (nk, nspinor, n_rmu, nspinor, n_rmu).
Zero-comm contraction when left is X-sharded on rmu and right is Y-sharded on rmu.
Einsum order: kxmb,kbyn->kxmyn (spin indices x,y kept separate from rmu m,n).

<a id="isdf.gw_isdf.cohsex_jax.get_G_R_jax"></a>

#### get\_G\_R\_jax

```python
def get_G_R_jax(G_k, nkx, nky, nkz)
```

Pure: (nk, s1,rmu1,s2,rmu2) -> (s1,rmu1,s2,rmu2,nkx,nky,nkz).

<a id="isdf.gw_isdf.cohsex_jax.get_sigma_x_mu_nu_jax"></a>

#### get\_sigma\_x\_mu\_nu\_jax

```python
def get_sigma_x_mu_nu_jax(G_R, V_mu_nu, nk_tot)
```

Pure: G_R (s1,rmu1,s2,rmu2,nkx,nky,nkz), V_mu_nu (rmu1,rmu2,nkx,nky,nkz) -> sigma_k same shape as G_R.

<a id="isdf.gw_isdf.cohsex_jax.get_sigma_x_kij_jax"></a>

#### get\_sigma\_x\_kij\_jax

```python
def get_sigma_x_kij_jax(psi_sigX, psi_sigTY, sigma_k_munu)
```

Pure: psi_* (nk, nb, ns, rmu), sigma_k_munu (s1,rmu1,s2,rmu2,nkx,nky,nkz) -> sigma_kij (nk, nb, nb).

<a id="isdf.gw_isdf.cohsex_jax.compute_sigma_pipeline_jax"></a>

#### compute\_sigma\_pipeline\_jax

```python
def compute_sigma_pipeline_jax(psi_l_rmuT_X, psi_l_rmu_Y, psi_r_rmu_X,
                               psi_r_rmuT_Y, V_mu_nu, nkx: int, nky: int,
                               nkz: int, nk_tot: int, nspinor: int)
```

Pure JAX pipeline: returns sigma_kij (nk, nb, nb).
Uses psi_l for building G (valence-only) and psi_r for projection to bands.

<a id="isdf.gw_isdf.cohsex_jax.preprocess_q_loops"></a>

#### preprocess\_q\_loops

```python
def preprocess_q_loops(wfn, sym, meta, V_qG, mesh_xy=None)
```

Precompute all q-point and k-point mappings for the zeta/v_q function.

This function processes all q-vectors in the Brillouin zone and precomputes:
- k-point index mappings for each q-vector  
- Coulomb potential data V_qG for each q-point
- G-vector components for interpolation

Parameters
----------
wfn : WFNReader
	Wavefunction data
sym : symmetry_maps object  
	Symmetry information
meta : Meta
	System metadata
V_qG : array
	Coulomb potential in G-space

Returns
-------
tuple
	(all_k_l_indices, all_k_r_indices, all_qvecs_wrapped, 
	 all_V_qfullG, all_vcoul_comps, n_G_per_q, all_qvecs_nonneg, all_iq_indices)

<a id="isdf.gw_isdf.w_isdf"></a>

# isdf.gw\_isdf.w\_isdf

<a id="isdf.gw_isdf.w_isdf.get_chi_lm_Yt_jax"></a>

#### get\_chi\_lm\_Yt\_jax

```python
def get_chi_lm_Yt_jax(psi_vTX: jax.Array,
                      psi_vY: jax.Array,
                      psi_cX: jax.Array,
                      psi_cTY: jax.Array,
                      enk_v: jax.Array,
                      enk_c: jax.Array,
                      win,
                      meta: Meta,
                      mesh_xy: Mesh | None = None)
```

Compute chi_lm integrated over tau using only JAX arrays.

Args:
	psi_vTX: (nk, ns, rmu, nb)  left valence, rmu sharded on X, band is fastest
	psi_vY:  (nk, nb, ns, rmu)  right valence, rmu sharded on Y, rmu is fastest
	psi_cX:  (nk, ns, rmu, nb)  left conduction, rmu sharded on X, band is fastest
	psi_cTY: (nk, nb, ns, rmu)  right conduction, rmu sharded on Y, rmu is fastest
	enk_v: (nk, nb_v)
	enk_c: (nk, nb_c)
	win: window object with attributes (tau_i, z_lm, w_i, val_window, cond_window)
	meta: Meta with nkx,nky,nkz
	mesh_xy: optional 2D device mesh for sharding (mu on x, nu on y)

Returns:
	chi_q: (nkx, nky, nkz, npol1=1, nrmu1, npol2=1, nrmu2) complex128

<a id="isdf.gw_isdf.w_isdf.get_chi0_jax"></a>

#### get\_chi0\_jax

```python
def get_chi0_jax(psi_vTX: jax.Array,
                 psi_vY: jax.Array,
                 psi_cX: jax.Array,
                 psi_cTY: jax.Array,
                 enk_v: jax.Array,
                 enk_c: jax.Array,
                 windows,
                 meta: Meta,
                 mesh_xy: Mesh | None = None)
```

Sum chi_lm over windows using JAX arrays.

Args:
	psi_vTX: (nk, ns, rmu, nb)  left valence, rmu sharded on X, band is fastest
	psi_vY:  (nk, nb, ns, rmu)  right valence, rmu sharded on Y, rmu is fastest
	psi_cX:  (nk, ns, rmu, nb)  left conduction, rmu sharded on X, band is fastest
	psi_cTY: (nk, nb, ns, rmu)  right conduction, rmu sharded on Y, rmu is fastest
	enk_v: (nk, nb_v)
	enk_c: (nk, nb_c)
	windows: iterable of window objects
	meta: Meta
	mesh_xy: optional mesh for sharding

Returns:
	chi_q: (nkx,nky,nkz,npol1=1,nrmu1,npol2=1,nrmu2) complex128

<a id="isdf.gw_isdf.w_isdf.get_static_w_q_jax"></a>

#### get\_static\_w\_q\_jax

```python
def get_static_w_q_jax(V_qmunu: jax.Array,
                       chi_q: jax.Array,
                       S_qmunu: jax.Array | None,
                       meta: Meta,
                       mesh_xy: Mesh | None = None)
```

Compute static W_q using JAX under k_XY sharding inside a single jit.

Inputs:
- V_qmunu: (1, npol1=1, npol2=1, nkx, nky, nkz, nrmu, nrmu)
- chi_q:   (nkx, nky, nkz, 1, nrmu, 1, nrmu)
- S_qmunu: (nkx, nky, nkz, nrmu, nrmu) or None (whitening; required for overlap)

Returns:
- W_q: (nkx, nky, nkz, 1, nrmu, 1, nrmu) with mu_X,nu_Y sharding

<a id="isdf.gw_isdf.vcoul"></a>

# isdf.gw\_isdf.vcoul

<a id="isdf.gw_isdf.vcoul.wrap_points_to_voronoi"></a>

#### wrap\_points\_to\_voronoi

```python
def wrap_points_to_voronoi(randcart, bvec, nmax=1)
```

Helper function to get test q-points for mini-BZ average with correct voronoi cell.
Rewritten to use JAX arrays.

<a id="isdf.gw_isdf.cohsex_jax_deprecated"></a>

# isdf.gw\_isdf.cohsex\_jax\_deprecated

<a id="isdf.gw_isdf.cohsex_jax_deprecated.read_cohsex_input"></a>

#### read\_cohsex\_input

```python
def read_cohsex_input(filename: str) -> dict
```

Parse a simple INI-style input file for the COHSEX driver.

<a id="isdf.gw_isdf.cohsex_jax_deprecated.get_bandranges"></a>

#### get\_bandranges

```python
def get_bandranges(nv, nc, nband, nelec)
```

Return ranges of bands necessary for \sigma_{X,SX,COH}

<a id="isdf.gw_isdf.cohsex_jax_deprecated.wrap_points_to_voronoi"></a>

#### wrap\_points\_to\_voronoi

```python
def wrap_points_to_voronoi(randcart, bvec, xp, nmax=1)
```

Helper function to get test q-points for mini-BZ average with correct voronoi cell.

<a id="isdf.gw_isdf.cohsex_jax_deprecated.fft_bandrange"></a>

#### fft\_bandrange

```python
def fft_bandrange(wfn, sym, bandrange, is_left, meta: Meta, bispinor=False)
```

Get psi_nk(r) for all k-points in the full Brillouin zone.
(not u_nk(r)! returns psi_nk(r) = e^{ikr} u_nk(r))
Args:
    wfn/sym: WFNReader/SymMaps objects
    bandrange: Tuple (start, end) for band range
    is_left: Bool indicating if psi = psi_l (gets conjugated)
Returns:
    psi_rtot_out: Array of real-space wavefunctions for all k-points

<a id="isdf.gw_isdf.cohsex_jax_deprecated.get_zeta_q_and_v_q_mu_nu"></a>

#### get\_zeta\_q\_and\_v\_q\_mu\_nu

```python
def get_zeta_q_and_v_q_mu_nu(wfn,
                             sym,
                             centroid_indices,
                             bandrange_l,
                             bandrange_r,
                             V_qG,
                             meta: Meta,
                             xp,
                             bispinor=False)
```

Find the interpolative separable density fitting representation.

<a id="isdf.gw_isdf.cohsex_jax_deprecated.get_sigma_x_kij"></a>

#### get\_sigma\_x\_kij

```python
def get_sigma_x_kij(psi_l, psi_r, sigma_kbar, meta: Meta, xp)
```

Calculate the sigma_x_kij matrix elements.
sigma_mnkbar = \sum_rmu,rnu,s,s' exp(ik(r_nu-r_mu)) u_mk^*(r_mu,s) sigma_kbar,ss'(r_mu,r_nu) u_nk(r_nu,s')

<a id="isdf.gw_isdf.cohsex_jax_deprecated.find_qpoint_index"></a>

#### find\_qpoint\_index

```python
def find_qpoint_index(q_ext, sym, tol=1e-6)
```

Find index of q-point in unfolded k-points list.

Args:
    q_ext: Vector of length 3 (crystal coordinates)
    sym: SymMaps object containing unfolded_kpts
    tol: Tolerance for floating point comparison

Returns:
    Index of matching q-point, or raises ValueError if not found

<a id="isdf.gw_isdf.cohsex_jax_deprecated.write_labeled_arrays_to_h5"></a>

#### write\_labeled\_arrays\_to\_h5

```python
def write_labeled_arrays_to_h5(filename, V_qmunu, psi_l, psi_r)
```

Write the data of LabeledArray and WfnArray objects to an HDF5 file.

Args:
    filename: Name of the HDF5 file
    V_qmunu: LabeledArray for V_qmunu
    psi_l: WfnArray for left states
    psi_r: WfnArray for right states

<a id="isdf.gw_isdf.cohsex_jax_deprecated.read_labeled_arrays_from_h5"></a>

#### read\_labeled\_arrays\_from\_h5

```python
def read_labeled_arrays_from_h5(filename)
```

Read the data arrays from an HDF5 file and reconstruct LabeledArrays and WfnArrays.

Args:
    filename (str): The name of the HDF5 file to read from.

Returns:
    tuple: A tuple containing (V_qmunu, psi_l, psi_r) where V_qmunu is a LabeledArray
          and psi_l/psi_r are WfnArrays.

<a id="isdf.gw_isdf.cohsex_isdf"></a>

# isdf.gw\_isdf.cohsex\_isdf

<a id="isdf.gw_isdf.cohsex_isdf.read_cohsex_input"></a>

#### read\_cohsex\_input

```python
def read_cohsex_input(filename: str) -> dict
```

Parse a simple INI-style input file for the COHSEX driver.

<a id="isdf.gw_isdf.cohsex_isdf.get_bandranges"></a>

#### get\_bandranges

```python
def get_bandranges(nv, nc, nband, nelec)
```

Return ranges of bands necessary for \sigma_{X,SX,COH}

<a id="isdf.gw_isdf.cohsex_isdf.wrap_points_to_voronoi"></a>

#### wrap\_points\_to\_voronoi

```python
def wrap_points_to_voronoi(randcart, bvec, xp, nmax=1)
```

Helper function to get test q-points for mini-BZ average with correct voronoi cell.

<a id="isdf.gw_isdf.cohsex_isdf.fft_bandrange"></a>

#### fft\_bandrange

```python
def fft_bandrange(wfn,
                  sym,
                  bandrange,
                  is_left,
                  psi_rtot_out,
                  xp=cp,
                  bispinor=False)
```

Get psi_nk(r) for all k-points in the full Brillouin zone.
(not u_nk(r)! returns psi_nk(r) = e^{ikr} u_nk(r))
Args:
    wfn/sym: WFNReader/SymMaps objects
    bandrange: Tuple (start, end) for band range
    is_left: Bool indicating if psi = psi_l (gets conjugated)
Returns:
    psi_rtot_out: Array of real-space wavefunctions for all k-points

<a id="isdf.gw_isdf.cohsex_isdf.get_zeta_q_and_v_q_mu_nu"></a>

#### get\_zeta\_q\_and\_v\_q\_mu\_nu

```python
def get_zeta_q_and_v_q_mu_nu(wfn,
                             sym,
                             centroid_indices,
                             bandrange_l,
                             bandrange_r,
                             V_qG,
                             xp,
                             bispinor=False)
```

Find the interpolative separable density fitting representation.

<a id="isdf.gw_isdf.cohsex_isdf.get_sigma_x_kij"></a>

#### get\_sigma\_x\_kij

```python
def get_sigma_x_kij(psi_l, psi_r, sigma_kbar, xp)
```

Calculate the sigma_x_kij matrix elements.
sigma_mnkbar = \sum_rmu,rnu,s,s' exp(ik(r_nu-r_mu)) u_mk^*(r_mu,s) sigma_kbar,ss'(r_mu,r_nu) u_nk(r_nu,s')

<a id="isdf.gw_isdf.cohsex_isdf.find_qpoint_index"></a>

#### find\_qpoint\_index

```python
def find_qpoint_index(q_ext, sym, tol=1e-6)
```

Find index of q-point in unfolded k-points list.

Args:
    q_ext: Vector of length 3 (crystal coordinates)
    sym: SymMaps object containing unfolded_kpts
    tol: Tolerance for floating point comparison

Returns:
    Index of matching q-point, or raises ValueError if not found

<a id="isdf.gw_isdf.cohsex_isdf.write_labeled_arrays_to_h5"></a>

#### write\_labeled\_arrays\_to\_h5

```python
def write_labeled_arrays_to_h5(filename, V_qmunu, psi_l, psi_r)
```

Write the data of LabeledArray and WfnArray objects to an HDF5 file.

Args:
    filename: Name of the HDF5 file
    V_qmunu: LabeledArray for V_qmunu
    psi_l: WfnArray for left states
    psi_r: WfnArray for right states

<a id="isdf.gw_isdf.cohsex_isdf.read_labeled_arrays_from_h5"></a>

#### read\_labeled\_arrays\_from\_h5

```python
def read_labeled_arrays_from_h5(filename)
```

Read the data arrays from an HDF5 file and reconstruct LabeledArrays and WfnArrays.

Args:
    filename (str): The name of the HDF5 file to read from.

Returns:
    tuple: A tuple containing (V_qmunu, psi_l, psi_r) where V_qmunu is a LabeledArray
          and psi_l/psi_r are WfnArrays.

<a id="isdf.gw_isdf.gw_file_io"></a>

# isdf.gw\_isdf.gw\_file\_io

<a id="isdf.gw_isdf.gw_file_io.write_labeled_arrays_to_h5"></a>

#### write\_labeled\_arrays\_to\_h5

```python
def write_labeled_arrays_to_h5(filename,
                               V_qmunu,
                               psi_l,
                               psi_r,
                               enk_l=None,
                               enk_r=None,
                               S_qmunu=None)
```

Write raw JAX/Numpy arrays to an HDF5 file for restart.
Only rank 0 performs the write; arrays are gathered to host first.

<a id="isdf.gw_isdf.gw_file_io.read_labeled_arrays_from_h5"></a>

#### read\_labeled\_arrays\_from\_h5

```python
def read_labeled_arrays_from_h5(filename)
```

Read raw arrays from an HDF5 restart file and return JAX arrays:
(V_qmunu, S_qmunu, psi_l, psi_r, enk_l, enk_r)

<a id="isdf.gw_isdf.gw_file_io.load_labeled_arrays_from_h5"></a>

#### load\_labeled\_arrays\_from\_h5

```python
def load_labeled_arrays_from_h5(filename, mesh_xy)
```

Load restart arrays and apply intended sharding, returning the same tuple
shape as get_zeta_q_and_v_q_mu_nu:
(V_qmunu, S_qmunu, psi_lT, psi_l, psi_r, psi_rT, enk_l, enk_r)

<a id="isdf.gw_isdf.gw_file_io.save_restart_per_proc"></a>

#### save\_restart\_per\_proc

```python
def save_restart_per_proc(prefix: str, V_qmunu, S_qmunu, psi_l, psi_r, enk_l,
                          enk_r, meta: Meta, mesh_xy)
```

Save per-process local shards to HDF5 files named by (x,y) mesh coords.

<a id="isdf.common.wfnreader"></a>

# isdf.common.wfnreader

<a id="isdf.common.wfnreader.WFNReader"></a>

## WFNReader Objects

```python
class WFNReader()
```

<a id="isdf.common.wfnreader.WFNReader.__init__"></a>

#### \_\_init\_\_

```python
def __init__(filename)
```

Initialize WFNReader with WFN file.

<a id="isdf.common.wfnreader.WFNReader.get_cnk"></a>

#### get\_cnk

```python
def get_cnk(ik, ib)
```

Get complex coefficients for both spinor components of a wavefunction.

Args:
    ik (int): k-point index
    ib (int): band index

Returns:
    np.ndarray: Complex coefficients array of shape (ngk[ik], 2) for both spinor components,
               in Fortran order

<a id="isdf.common.wfnreader.WFNReader.get_gvec_nk"></a>

#### get\_gvec\_nk

```python
def get_gvec_nk(ik)
```

Get G-vectors for a specific k-point.

Args:
    ik (int): k-point index

Returns:
    np.ndarray: G-vectors array of shape (ngk[ik], 3) in Fortran order,
               where each row is a G-vector [Gx, Gy, Gz]

<a id="isdf.common.epsreader"></a>

# isdf.common.epsreader

<a id="isdf.common.epsreader.EPSReader"></a>

## EPSReader Objects

```python
class EPSReader()
```

<a id="isdf.common.epsreader.EPSReader.__init__"></a>

#### \_\_init\_\_

```python
def __init__(filename)
```

Initialize EPSMATReader with epsmat.h5 file.

<a id="isdf.common.epsreader.EPSReader.__del__"></a>

#### \_\_del\_\_

```python
def __del__()
```

Clean up by closing the file when the object is destroyed.

<a id="isdf.common.epsreader.EPSReader.get_eps_matrix"></a>

#### get\_eps\_matrix

```python
def get_eps_matrix(iq, ifreq=0, imatrix=0)
```

Get the epsilon matrix for a specific q-point and frequency.

Args:
    iq (int): Q-point index
    ifreq (int): Frequency index (default=0 for static)
    imatrix (int): Matrix index (default=0)

Returns:
    np.ndarray: Complex epsilon matrix of shape (nmtx[iq], nmtx[iq])

<a id="isdf.common.epsreader.EPSReader.get_eps_minus_delta_matrix"></a>

#### get\_eps\_minus\_delta\_matrix

```python
def get_eps_minus_delta_matrix(iq, ifreq=0, imatrix=0)
```

Get the epsilon matrix for a specific q-point and frequency.

Args:
    iq (int): Q-point index
    ifreq (int): Frequency index (default=0 for static)
    imatrix (int): Matrix index (default=0)

Returns:
    np.ndarray: Complex epsilon matrix of shape (nmtx[iq], nmtx[iq])

<a id="isdf.common.epsreader.EPSReader.get_eps_diagonal"></a>

#### get\_eps\_diagonal

```python
def get_eps_diagonal(iq)
```

Get the static diagonal elements for a specific q-point.

Args:
    iq (int): Q-point index

Returns:
    np.ndarray: Complex diagonal elements

<a id="isdf.common.load_wfns"></a>

# isdf.common.load\_wfns

<a id="isdf.common.load_wfns.get_enk_bandrange"></a>

#### get\_enk\_bandrange

```python
def get_enk_bandrange(wfn, sym, bandrange, sigma_bandrange)
```

Return band energies and per-band weights for a given band window.

Args:
	wfn: WFNReader providing energies and Fermi level
	sym: SymMaps with mappings between irreducible and full k sets
	bandrange: tuple[int,int] inclusive-exclusive (start, end) bands to extract
	sigma_bandrange: tuple[int,int] band window used to compute weighting

Returns:
	enk: jax.Array of shape (nk_full, nb)
	weights: jax.Array of shape (nk_full, nb * 2) with simple val/cond weights

<a id="isdf.common.load_wfns.read_Gvecs_to_devices"></a>

#### read\_Gvecs\_to\_devices

```python
def read_Gvecs_to_devices(wfn, sym, bandrange, meta: Meta, bispinor: bool,
                          mesh_xy: Mesh)
```

Non-jitted: load cnk(G) for all k-points and (padded) band shards into a global
sharded G-space FFT box over a 2D mesh ['x','y'] along the band axis.
Returns the global sharded array global_psi_Gtot and nb_actual.

<a id="isdf.common.load_wfns.get_sharded_wfns"></a>

#### get\_sharded\_wfns

```python
def get_sharded_wfns(global_psi_Gtot: jax.Array, sym, meta: Meta,
                     centroid_indices, nb_actual: int, is_left: bool,
                     mesh_xy: Mesh)
```

Jitted: FFT -> apply phase -> normalize/trim -> flatten r -> reshard (Y-only) -> centroid gather ->
build psi_rmu^T with X-only sharding. Returns (psi_rtot_Y, psi_rmu_Y, psi_rmuT_X).

<a id="isdf.common.tagged_arrays"></a>

# isdf.common.tagged\_arrays

<a id="isdf.common.tagged_arrays.WfnArray"></a>

## WfnArray Objects

```python
class WfnArray()
```

Class to hold both wavefunction coefficients and energies together.
Uses slots for memory efficiency and to prevent accidental attribute creation.

<a id="isdf.common.tagged_arrays.WfnArray.__init__"></a>

#### \_\_init\_\_

```python
def __init__(psi: LabeledArray, enk: LabeledArray)
```

Initialize WfnArray from existing LabeledArrays.

Args:
    psi: LabeledArray containing wavefunction coefficients
        Expected axes: ['nk', 'nb', 'nspinor', 'nrmu']
    enk: LabeledArray containing energies
        Expected axes: ['nb', 'nk']

<a id="isdf.common.symmetry_maps"></a>

# isdf.common.symmetry\_maps

<a id="isdf.common.symmetry_maps.SymMaps"></a>

## SymMaps Objects

```python
class SymMaps()
```

<a id="isdf.common.symmetry_maps.SymMaps.__init__"></a>

#### \_\_init\_\_

```python
def __init__(wfn)
```

Initialize symmetry mappings for a given WFN file.
class variables are:
dict: irk_to_k_map[irk] = [k1, k2, k3, ...], kpt id's that map to irk
dict: irk_sym_map[irk] = [sym1, sym2, sym3, ...], sym op sym_matrices[sym1] maps irk to ik
U_spinor[sym_idx] is the spinor rotation matrix for the sym_idx-th symmetry operation.
The matrices are currently 2x2 Pauli-spinor rotations; upcoming work
will expand this to the 4-component formalism used in relativistic
treatments.
R_grid[sym_idx] is the corresponding list of symmetry operations in the WFN file
u_{n,Rk,a}(G) = U_spinor_{a,b} u_{n,k,b}(Rinv G)

Args:
    wfn: WFNReader instance

<a id="isdf.common.symmetry_maps.SymMaps.create_kpoint_symmetry_map"></a>

#### create\_kpoint\_symmetry\_map

```python
def create_kpoint_symmetry_map(wfn)
```

Read k-point mapping from kgrid.log file.
Converts from 1-based to 0-based indexing for kpts.

Args:
    wfn (WfnReader): WFN reader object

Returns:
    tuple: (kpoint_map, full_kpoints)
        - kpoint_map: Array mapping each k-point to its irreducible k-point (full zone)
        - full_kpoints: Array of all k-points in the full grid

<a id="isdf.common.symmetry_maps.SymMaps.syms_crystal_to_cartesian"></a>

#### syms\_crystal\_to\_cartesian

```python
def syms_crystal_to_cartesian(wfn)
```

Convert symmetry matrices from crystal to cartesian coordinates.

Args:
    sym_matrices_crys (numpy.ndarray): Symmetry matrices in crystal coords (nsym, 3, 3)

Returns:
    numpy.ndarray: Symmetry matrices in cartesian coordinates (nsym, 3, 3)

<a id="isdf.common.symmetry_maps.SymMaps.get_spinor_rotations"></a>

#### get\_spinor\_rotations

```python
def get_spinor_rotations(wfn, sym_matrices_cart)
```

Converts a list of rotation matrices to their spinor representations using Markley's modification
of Shepperd's algorithm (aka quaternion representation, see Brad Barker's dissertation).

When the wavefunction files store four-component states these routines will
compute the corresponding 4x4 spinor rotation matrices.

Parameters:
sym_matrices (numpy.ndarray): Array of 3x3 rotation matrices with shape (nsym, 3, 3)

Returns:
numpy.ndarray: Array of spinor matrices with shape (nsym, 2, 2) of complex type

<a id="isdf.common.symmetry_maps.SymMaps.get_kminusq_map"></a>

#### get\_kminusq\_map

```python
def get_kminusq_map(wfn, full_kpts)
```

Create mapping between k and k-q points in the full k-point grid.

Args:
    wfn: WFNReader instance
    full_kpts: Array of all k-points in the full grid

Returns:
    numpy.ndarray: kq_map[ik,iq] = index of k-q in full k-point grid,
                  where ik is index in full grid, iq is index in reduced grid

<a id="isdf.common.symmetry_maps.SymMaps.find_qpoint_index"></a>

#### find\_qpoint\_index

```python
def find_qpoint_index(q_ext, tol=1e-6)
```

Find index of q-point in unfolded k-points list.

Args:
    q_ext: Vector of length 3 (crystal coordinates)
    tol: Tolerance for floating point comparison

Returns:
    Index of matching q-point, or raises ValueError if not found

<a id="isdf.isdf_init.kmeans_isdf"></a>

# isdf.isdf\_init.kmeans\_isdf

<a id="isdf.isdf_init.kmeans_isdf.interpolate_density"></a>

#### interpolate\_density

```python
def interpolate_density(rho_np, zoom_factors=(1, 1, 1))
```

Return a zoomed copy of ``rho_np`` using ``scipy.ndimage.zoom``.

<a id="isdf.isdf_init.kmeans_isdf.plot_density_and_centroids"></a>

#### plot\_density\_and\_centroids

```python
def plot_density_and_centroids(wfn, rho_np, centroids, labels=None)
```

Plot charge density and centroids in 3D.

<a id="isdf.isdf_init.kmeans_isdf.weighted_kmeans_jax"></a>

#### weighted\_kmeans\_jax

```python
def weighted_kmeans_jax(avec,
                        rho_jax,
                        N_k=10,
                        max_steps=200,
                        tolerance=5e-3,
                        seed=0)
```

Weighted k-means using JAX on multiple CPU devices.

<a id="isdf.isdf_init.get_charge_density"></a>

# isdf.isdf\_init.get\_charge\_density

<a id="isdf.isdf_init.get_charge_density.perform_fft_3d"></a>

#### perform\_fft\_3d

```python
def perform_fft_3d(data_1d, gvecs, fft_grid)
```

Transform 1D complex array to real space using an FFT.

Args:
    data_1d: 1D complex array of coefficients (length ngk[ik])
    gvecs: G-vector components for this k-point (ranging from ~ -10 to 10)
    fft_grid: 3D FFT grid dimensions for zero-padding

<a id="isdf.isdf_init.get_charge_density.calculate_charge_density"></a>

#### calculate\_charge\_density

```python
def calculate_charge_density(wfn, sym, nval=None, ncond=None)
```

Calculate charge density in real space from wavefunctions using WFNReader: goes over all occupied states c_nk(G),
FFTs them to c_nk(R) (using GPU acceleration when available), squares and sums to get rho(R).
k-point symmetries are used. The loop order is (nband, nk_irr, n_sym).
n_sym is done on the GPU since symmetry operations over Gvecs can be parallelized.

<a id="isdf.isdf_init.get_charge_density.save_charge_density"></a>

#### save\_charge\_density

```python
def save_charge_density(charge_density)
```

Save the charge density to an HDF5 file.

<a id="isdf.isdf_init.get_charge_density.analyze_gvectors"></a>

#### analyze\_gvectors

```python
def analyze_gvectors(gvecs)
```

Analyze the range and distribution of G-vectors.

Args:
    gvecs: Array of G-vectors, shape (ngvecs, 3)

