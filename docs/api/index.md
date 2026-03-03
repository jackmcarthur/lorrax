# Table of Contents

* [gw\_isdf.gw\_init](#gw_isdf.gw_init)
  * [compute\_optimal\_chunks](#gw_isdf.gw_init.compute_optimal_chunks)
  * [print\_memory\_breakdown](#gw_isdf.gw_init.print_memory_breakdown)
  * [get\_effective\_chunk\_size](#gw_isdf.gw_init.get_effective_chunk_size)
  * [read\_cohsex\_input](#gw_isdf.gw_init.read_cohsex_input)
  * [get\_bandranges](#gw_isdf.gw_init.get_bandranges)
* [gw\_isdf.get\_windows](#gw_isdf.get_windows)
  * [WindowPair](#gw_isdf.get_windows.WindowPair)
    * [init\_hgl\_quadrature](#gw_isdf.get_windows.WindowPair.init_hgl_quadrature)
    * [classify\_frequencies](#gw_isdf.get_windows.WindowPair.classify_frequencies)
  * [classify\_frequencies](#gw_isdf.get_windows.classify_frequencies)
  * [compute\_dos](#gw_isdf.get_windows.compute_dos)
  * [find\_optimal\_partitions](#gw_isdf.get_windows.find_optimal_partitions)
  * [minimize\_cost\_fn](#gw_isdf.get_windows.minimize_cost_fn)
  * [get\_window\_info](#gw_isdf.get_windows.get_window_info)
* [gw\_isdf.compute\_vcoul](#gw_isdf.compute_vcoul)
  * [exp\_ikr\_fftbox](#gw_isdf.compute_vcoul.exp_ikr_fftbox)
  * [fft\_integer\_axes](#gw_isdf.compute_vcoul.fft_integer_axes)
  * [compute\_sqrt\_vcoul\_2d](#gw_isdf.compute_vcoul.compute_sqrt_vcoul_2d)
  * [compute\_phase\_q](#gw_isdf.compute_vcoul.compute_phase_q)
  * [make\_v\_munu\_chunked\_kernel](#gw_isdf.compute_vcoul.make_v_munu_chunked_kernel)
  * [compute\_V\_q\_from\_zeta\_h5](#gw_isdf.compute_vcoul.compute_V_q_from_zeta_h5)
  * [compute\_V\_q\_from\_zeta\_array](#gw_isdf.compute_vcoul.compute_V_q_from_zeta_array)
  * [read\_zeta\_q\_sharded](#gw_isdf.compute_vcoul.read_zeta_q_sharded)
  * [compute\_all\_V\_q\_from\_zeta\_h5](#gw_isdf.compute_vcoul.compute_all_V_q_from_zeta_h5)
  * [make\_v\_munu\_kernel\_chunked](#gw_isdf.compute_vcoul.make_v_munu_kernel_chunked)
* [gw\_isdf.archive.cohsex\_jax\_deprecated](#gw_isdf.archive.cohsex_jax_deprecated)
  * [read\_cohsex\_input](#gw_isdf.archive.cohsex_jax_deprecated.read_cohsex_input)
  * [get\_bandranges](#gw_isdf.archive.cohsex_jax_deprecated.get_bandranges)
  * [wrap\_points\_to\_voronoi](#gw_isdf.archive.cohsex_jax_deprecated.wrap_points_to_voronoi)
  * [fft\_bandrange](#gw_isdf.archive.cohsex_jax_deprecated.fft_bandrange)
  * [get\_zeta\_q\_and\_v\_q\_mu\_nu](#gw_isdf.archive.cohsex_jax_deprecated.get_zeta_q_and_v_q_mu_nu)
  * [get\_sigma\_x\_kij](#gw_isdf.archive.cohsex_jax_deprecated.get_sigma_x_kij)
  * [find\_qpoint\_index](#gw_isdf.archive.cohsex_jax_deprecated.find_qpoint_index)
  * [write\_labeled\_arrays\_to\_h5](#gw_isdf.archive.cohsex_jax_deprecated.write_labeled_arrays_to_h5)
  * [read\_labeled\_arrays\_from\_h5](#gw_isdf.archive.cohsex_jax_deprecated.read_labeled_arrays_from_h5)
* [gw\_isdf.archive.test\_eps\_v\_products](#gw_isdf.archive.test_eps_v_products)
  * [compute\_v\_trunc\_2d\_for\_eps](#gw_isdf.archive.test_eps_v_products.compute_v_trunc_2d_for_eps)
* [gw\_isdf.archive.plot\_whead\_vs\_model](#gw_isdf.archive.plot_whead_vs_model)
  * [v2d\_trunc\_head\_from\_q](#gw_isdf.archive.plot_whead_vs_model.v2d_trunc_head_from_q)
* [gw\_isdf.archive.jax\_fixed\_point\_demo](#gw_isdf.archive.jax_fixed_point_demo)
  * [f\_of\_x](#gw_isdf.archive.jax_fixed_point_demo.f_of_x)
  * [qr\_least\_squares](#gw_isdf.archive.jax_fixed_point_demo.qr_least_squares)
  * [anderson\_fixed\_history](#gw_isdf.archive.jax_fixed_point_demo.anderson_fixed_history)
  * [crop\_family\_fixed\_history](#gw_isdf.archive.jax_fixed_point_demo.crop_family_fixed_history)
  * [crop\_family\_fixed\_history\_map](#gw_isdf.archive.jax_fixed_point_demo.crop_family_fixed_history_map)
* [gw\_isdf.archive.cohsex\_isdf](#gw_isdf.archive.cohsex_isdf)
  * [read\_cohsex\_input](#gw_isdf.archive.cohsex_isdf.read_cohsex_input)
  * [get\_bandranges](#gw_isdf.archive.cohsex_isdf.get_bandranges)
  * [wrap\_points\_to\_voronoi](#gw_isdf.archive.cohsex_isdf.wrap_points_to_voronoi)
  * [fft\_bandrange](#gw_isdf.archive.cohsex_isdf.fft_bandrange)
  * [get\_zeta\_q\_and\_v\_q\_mu\_nu](#gw_isdf.archive.cohsex_isdf.get_zeta_q_and_v_q_mu_nu)
  * [get\_sigma\_x\_kij](#gw_isdf.archive.cohsex_isdf.get_sigma_x_kij)
  * [find\_qpoint\_index](#gw_isdf.archive.cohsex_isdf.find_qpoint_index)
  * [write\_labeled\_arrays\_to\_h5](#gw_isdf.archive.cohsex_isdf.write_labeled_arrays_to_h5)
  * [read\_labeled\_arrays\_from\_h5](#gw_isdf.archive.cohsex_isdf.read_labeled_arrays_from_h5)
* [gw\_isdf.gw\_jax](#gw_isdf.gw_jax)
  * [compute\_CCT\_ZCT\_for\_q](#gw_isdf.gw_jax.compute_CCT_ZCT_for_q)
  * [solve\_zeta\_cholesky](#gw_isdf.gw_jax.solve_zeta_cholesky)
  * [compute\_Sq\_from\_zeta](#gw_isdf.gw_jax.compute_Sq_from_zeta)
  * [exp\_ikr\_fftbox](#gw_isdf.gw_jax.exp_ikr_fftbox)
  * [fft\_integer\_axes](#gw_isdf.gw_jax.fft_integer_axes)
  * [as\_index\_tuple](#gw_isdf.gw_jax.as_index_tuple)
  * [make\_v\_munu\_kernel](#gw_isdf.gw_jax.make_v_munu_kernel)
  * [compute\_v\_munu\_from\_zeta](#gw_isdf.gw_jax.compute_v_munu_from_zeta)
  * [make\_shardings](#gw_isdf.gw_jax.make_shardings)
  * [build\_q\_coulomb\_cache](#gw_isdf.gw_jax.build_q_coulomb_cache)
  * [determine\_wcoul0](#gw_isdf.gw_jax.determine_wcoul0)
  * [get\_zeta\_q\_and\_v\_q\_mu\_nu](#gw_isdf.gw_jax.get_zeta_q_and_v_q_mu_nu)
  * [fit\_zeta\_and\_compute\_V\_q\_chunked](#gw_isdf.gw_jax.fit_zeta_and_compute_V_q_chunked)
  * [get\_G\_mu\_nu\_jax](#gw_isdf.gw_jax.get_G_mu_nu_jax)
  * [get\_G\_mu\_nu\_RI](#gw_isdf.gw_jax.get_G_mu_nu_RI)
  * [get\_G\_R\_jax](#gw_isdf.gw_jax.get_G_R_jax)
  * [get\_sigma\_static\_mu\_nu\_jax](#gw_isdf.gw_jax.get_sigma_static_mu_nu_jax)
  * [get\_sigma\_static\_kij\_jax](#gw_isdf.gw_jax.get_sigma_static_kij_jax)
  * [build\_density\_from\_Gij](#gw_isdf.gw_jax.build_density_from_Gij)
  * [build\_hartree\_potential](#gw_isdf.gw_jax.build_hartree_potential)
  * [project\_potential\_to\_bands](#gw_isdf.gw_jax.project_potential_to_bands)
  * [compute\_sigma\_pipeline\_jax](#gw_isdf.gw_jax.compute_sigma_pipeline_jax)
  * [summarize\_hermitian\_matrix](#gw_isdf.gw_jax.summarize_hermitian_matrix)
  * [preprocess\_q\_loops](#gw_isdf.gw_jax.preprocess_q_loops)
* [gw\_isdf.hgl\_quadrature](#gw_isdf.hgl_quadrature)
  * [hgl\_nodes\_weights](#gw_isdf.hgl_quadrature.hgl_nodes_weights)
  * [n\_tau\_hgl](#gw_isdf.hgl_quadrature.n_tau_hgl)
* [gw\_isdf.vcoul](#gw_isdf.vcoul)
  * [wrap\_points\_to\_voronoi](#gw_isdf.vcoul.wrap_points_to_voronoi)
  * [compute\_vcoul\_comps\_for\_q](#gw_isdf.vcoul.compute_vcoul_comps_for_q)
  * [compute\_V\_qfullG\_for\_q](#gw_isdf.vcoul.compute_V_qfullG_for_q)
  * [compute\_q0\_averages](#gw_isdf.vcoul.compute_q0_averages)
  * [compute\_wcoul0\_with\_S](#gw_isdf.vcoul.compute_wcoul0_with_S)
* [gw\_isdf.w\_isdf\_dynamic](#gw_isdf.w_isdf_dynamic)
  * [get\_chi\_lm\_Yt\_jax\_windowed](#gw_isdf.w_isdf_dynamic.get_chi_lm_Yt_jax_windowed)
  * [get\_chi0\_jax\_windowed](#gw_isdf.w_isdf_dynamic.get_chi0_jax_windowed)
  * [get\_chi\_omega\_jax](#gw_isdf.w_isdf_dynamic.get_chi_omega_jax)
  * [get\_w\_omega\_jax](#gw_isdf.w_isdf_dynamic.get_w_omega_jax)
* [gw\_isdf.kin\_ion\_io\_chunked](#gw_isdf.kin_ion_io_chunked)
  * [get\_kin\_ion\_k](#gw_isdf.kin_ion_io_chunked.get_kin_ion_k)
* [gw\_isdf.w\_from\_eps0](#gw_isdf.w_from_eps0)
  * [compute\_Wmunu\_from\_eps0\_body](#gw_isdf.w_from_eps0.compute_Wmunu_from_eps0_body)
* [gw\_isdf.w\_isdf](#gw_isdf.w_isdf)
  * [compute\_chi0](#gw_isdf.w_isdf.compute_chi0)
  * [get\_static\_w\_q\_jax](#gw_isdf.w_isdf.get_static_w_q_jax)
  * [compute\_chi0\_and\_w](#gw_isdf.w_isdf.compute_chi0_and_w)
  * [get\_chi0\_jax](#gw_isdf.w_isdf.get_chi0_jax)
  * [get\_w\_omega\_jax](#gw_isdf.w_isdf.get_w_omega_jax)
  * [get\_chi\_omega\_jax](#gw_isdf.w_isdf.get_chi_omega_jax)
* [isdf.common.epsreader](#isdf.common.epsreader)
  * [EPSReader](#isdf.common.epsreader.EPSReader)
    * [\_\_init\_\_](#isdf.common.epsreader.EPSReader.__init__)
    * [\_\_del\_\_](#isdf.common.epsreader.EPSReader.__del__)
    * [get\_eps\_matrix](#isdf.common.epsreader.EPSReader.get_eps_matrix)
    * [get\_eps\_minus\_delta\_matrix](#isdf.common.epsreader.EPSReader.get_eps_minus_delta_matrix)
    * [get\_eps\_diagonal](#isdf.common.epsreader.EPSReader.get_eps_diagonal)
* [isdf.common.cholesky\_2d](#isdf.common.cholesky_2d)
  * [dense\_to\_tiles](#isdf.common.cholesky_2d.dense_to_tiles)
  * [tiles\_to\_dense](#isdf.common.cholesky_2d.tiles_to_dense)
  * [cholesky\_2d\_single](#isdf.common.cholesky_2d.cholesky_2d_single)
  * [cholesky\_2d\_batched](#isdf.common.cholesky_2d.cholesky_2d_batched)
  * [solve\_triangular\_2d](#isdf.common.cholesky_2d.solve_triangular_2d)
  * [cholesky\_solve\_2d](#isdf.common.cholesky_2d.cholesky_solve_2d)
* [isdf.common.gpu\_utils](#isdf.common.gpu_utils)
  * [get\_gpu\_memory\_nvidia\_smi](#isdf.common.gpu_utils.get_gpu_memory_nvidia_smi)
  * [get\_cpu\_memory\_total](#isdf.common.gpu_utils.get_cpu_memory_total)
  * [get\_device\_memory\_gb](#isdf.common.gpu_utils.get_device_memory_gb)
  * [get\_device\_memory\_info](#isdf.common.gpu_utils.get_device_memory_info)
* [isdf.common.symmetry\_maps](#isdf.common.symmetry_maps)
  * [SymMaps](#isdf.common.symmetry_maps.SymMaps)
    * [\_\_init\_\_](#isdf.common.symmetry_maps.SymMaps.__init__)
    * [create\_kpoint\_symmetry\_map](#isdf.common.symmetry_maps.SymMaps.create_kpoint_symmetry_map)
    * [syms\_crystal\_to\_cartesian](#isdf.common.symmetry_maps.SymMaps.syms_crystal_to_cartesian)
    * [get\_spinor\_rotations](#isdf.common.symmetry_maps.SymMaps.get_spinor_rotations)
    * [get\_kminusq\_map](#isdf.common.symmetry_maps.SymMaps.get_kminusq_map)
    * [get\_cnk\_fullzone\_batch](#isdf.common.symmetry_maps.SymMaps.get_cnk_fullzone_batch)
    * [find\_qpoint\_index](#isdf.common.symmetry_maps.SymMaps.find_qpoint_index)
* [isdf.common.jax\_profile](#isdf.common.jax_profile)
  * [trace\_section](#isdf.common.jax_profile.trace_section)
  * [step\_annotation](#isdf.common.jax_profile.step_annotation)
  * [annotation](#isdf.common.jax_profile.annotation)
* [isdf.common.load\_wfns](#isdf.common.load_wfns)
  * [compute\_block\_size\_for\_2d\_cholesky](#isdf.common.load_wfns.compute_block_size_for_2d_cholesky)
  * [make\_sharded\_ifftn\_3d](#isdf.common.load_wfns.make_sharded_ifftn_3d)
  * [make\_sharded\_fftn\_3d](#isdf.common.load_wfns.make_sharded_fftn_3d)
  * [load\_kpoint\_fftbox](#isdf.common.load_wfns.load_kpoint_fftbox)
  * [get\_enk\_bandrange](#isdf.common.load_wfns.get_enk_bandrange)
  * [read\_Gvecs\_to\_devices](#isdf.common.load_wfns.read_Gvecs_to_devices)
  * [get\_sharded\_wfns](#isdf.common.load_wfns.get_sharded_wfns)
  * [compute\_pair\_density\_spin\_traced](#isdf.common.load_wfns.compute_pair_density_spin_traced)
  * [compute\_CCT\_from\_left\_right](#isdf.common.load_wfns.compute_CCT_from_left_right)
  * [compute\_ZCT\_from\_left\_right\_zchunk](#isdf.common.load_wfns.compute_ZCT_from_left_right_zchunk)
  * [compute\_L\_q\_from\_CCT](#isdf.common.load_wfns.compute_L_q_from_CCT)
  * [solve\_zeta\_from\_L\_q](#isdf.common.load_wfns.solve_zeta_from_L_q)
  * [fit\_zeta\_chunked\_to\_h5](#isdf.common.load_wfns.fit_zeta_chunked_to_h5)
  * [load\_gspace\_for\_bands](#isdf.common.load_wfns.load_gspace_for_bands)
  * [get\_sharded\_wfns\_rchunk\_slice](#isdf.common.load_wfns.get_sharded_wfns_rchunk_slice)
  * [get\_psi\_rchunk\_from\_cached](#isdf.common.load_wfns.get_psi_rchunk_from_cached)
  * [get\_psi\_rchunk](#isdf.common.load_wfns.get_psi_rchunk)
  * [get\_sharded\_wfns\_centroids](#isdf.common.load_wfns.get_sharded_wfns_centroids)
  * [load\_centroids\_band\_chunked](#isdf.common.load_wfns.load_centroids_band_chunked)
* [isdf.common.wfnreader](#isdf.common.wfnreader)
  * [WFNReader](#isdf.common.wfnreader.WFNReader)
    * [\_\_init\_\_](#isdf.common.wfnreader.WFNReader.__init__)
    * [get\_cnk](#isdf.common.wfnreader.WFNReader.get_cnk)
    * [get\_cnk\_batch](#isdf.common.wfnreader.WFNReader.get_cnk_batch)
    * [get\_gvec\_nk](#isdf.common.wfnreader.WFNReader.get_gvec_nk)
* [isdf.isdf\_init.kmeans\_isdf](#isdf.isdf_init.kmeans_isdf)
  * [precompute\_metric\_tensor](#isdf.isdf_init.kmeans_isdf.precompute_metric_tensor)
  * [pbc\_distance\_sq\_batch](#isdf.isdf_init.kmeans_isdf.pbc_distance_sq_batch)
  * [pbc\_distance\_sq\_single](#isdf.isdf_init.kmeans_isdf.pbc_distance_sq_single)
  * [kmeans\_update\_step](#isdf.isdf_init.kmeans_isdf.kmeans_update_step)
  * [interpolate\_density](#isdf.isdf_init.kmeans_isdf.interpolate_density)
  * [plot\_density\_and\_centroids](#isdf.isdf_init.kmeans_isdf.plot_density_and_centroids)
  * [weighted\_kmeans\_jax](#isdf.isdf_init.kmeans_isdf.weighted_kmeans_jax)
  * [snap\_centroids\_to\_grid](#isdf.isdf_init.kmeans_isdf.snap_centroids_to_grid)
  * [ensure\_unique\_centroids](#isdf.isdf_init.kmeans_isdf.ensure_unique_centroids)
* [isdf.isdf\_init.get\_charge\_density](#isdf.isdf_init.get_charge_density)
  * [perform\_fft\_3d](#isdf.isdf_init.get_charge_density.perform_fft_3d)
  * [calculate\_charge\_density](#isdf.isdf_init.get_charge_density.calculate_charge_density)
  * [save\_charge\_density](#isdf.isdf_init.get_charge_density.save_charge_density)
  * [analyze\_gvectors](#isdf.isdf_init.get_charge_density.analyze_gvectors)

<a id="gw_isdf.gw_init"></a>

# gw\_isdf.gw\_init

Input file parsing and preprocessing for COHSEX calculations.

This module contains functions for:
- Reading and parsing the cohsex input file
- Converting input parameters to effective values
- Computing band ranges from input parameters
- Memory-aware chunk size optimization with full communication buffer accounting

<a id="gw_isdf.gw_init.compute_optimal_chunks"></a>

#### compute\_optimal\_chunks

```python
def compute_optimal_chunks(n_k: int,
                           n_b: int,
                           n_s: int,
                           n_rmu: int,
                           n_r: int,
                           n_q: int,
                           fft_grid: tuple[int, int, int],
                           n_devices: int,
                           memory_budget_gb: float,
                           target_utilization: float = 0.85,
                           p_x: int | None = None,
                           p_y: int | None = None,
                           verbose: bool = True,
                           n_b_left: int | None = None,
                           n_b_right: int | None = None,
                           r_chunk_override: int | None = None) -> dict
```

Derive chunk sizes that saturate (but do not exceed) the memory budget.

<a id="gw_isdf.gw_init.print_memory_breakdown"></a>

#### print\_memory\_breakdown

```python
def print_memory_breakdown(chunks: dict,
                           n_b: int,
                           n_r: int,
                           n_q: int,
                           fft_grid: tuple[int, int, int],
                           memory_source: str = 'auto') -> None
```

Print memory breakdown based on two bottleneck stages.

<a id="gw_isdf.gw_init.get_effective_chunk_size"></a>

#### get\_effective\_chunk\_size

```python
def get_effective_chunk_size(chunk_size: int) -> int | None
```

Convert chunk_size input flag to actual chunk size.

Args:
    chunk_size: Input flag value:
        -1 = no chunking (return None, all bands at once)
         0 = auto (TODO: compute from available RAM; currently 64)
        1-2048 = explicit chunk size

Returns:
    Effective chunk size as int, or None for no chunking.

<a id="gw_isdf.gw_init.read_cohsex_input"></a>

#### read\_cohsex\_input

```python
def read_cohsex_input(filename: str) -> dict
```

Parse input file for the COHSEX driver, allowing a QE K_POINTS block.

We extract the [cohsex] section using a substring to avoid configparser
errors from non-INI blocks like K_POINTS. The K_POINTS {crystal_b} block
is parsed manually and returned under 'kpoints_crystal_b'.

<a id="gw_isdf.gw_init.get_bandranges"></a>

#### get\_bandranges

```python
def get_bandranges(nv, nc, nband, nelec)
```

Return ranges of bands necessary for \sigma_{X,SX,COH}

<a id="gw_isdf.get_windows"></a>

# gw\_isdf.get\_windows

get_windows.py

Compute optimal energy windows for the conduction and valence bands (minimizing total quadrature points)
for the O(N^3) polarizability and self energy calculations given in Kim, Martyna, and Ismail-Beigi, PRB 101, 035139 (2020).

These energy windows also define the discrete imaginary-time grids used by the
CTSP method.  Once we move beyond static COHSEX these same windows will control
the frequency resolution of the full GW calculations.

<a id="gw_isdf.get_windows.WindowPair"></a>

## WindowPair Objects

```python
class WindowPair()
```

<a id="gw_isdf.get_windows.WindowPair.init_hgl_quadrature"></a>

#### init\_hgl\_quadrature

```python
def init_hgl_quadrature(epsq: float | None = None)
```

Initialize HGL nodes/weights. Call once per window that needs HGL.

HGL quadrature is used when a frequency ω causes an energy crossing,
i.e., when E_gap < ω < E_bw for this window pair. The caller is
responsible for determining which windows need HGL and calling this
exactly once for each.

Args:
    epsq: Target fractional error (defaults to self.epsq from __init__)

<a id="gw_isdf.get_windows.WindowPair.classify_frequencies"></a>

#### classify\_frequencies

```python
def classify_frequencies(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]
```

Classify which ω values use GL vs HGL quadrature for this window.

GL (Gauss-Laguerre): |ω| < E_gap or |ω| > E_bw (denominator doesn't change sign)
HGL (Hermite-Gauss-Laguerre): E_gap < |ω| < E_bw (energy crossing region)

Args:
    omega: Array of frequencies to classify (can be complex for contour integration)

Returns:
    gl_indices: Indices into omega array for GL treatment (non-crossing)
    hgl_indices: Indices into omega array for HGL treatment (crossing)

<a id="gw_isdf.get_windows.classify_frequencies"></a>

#### classify\_frequencies

```python
def classify_frequencies(omega: np.ndarray,
                         win: 'WindowPair') -> tuple[np.ndarray, np.ndarray]
```

Classify which ω values use GL vs HGL quadrature for a window pair.

This is a standalone function that can be used without a WindowPair object
(useful for testing or when window bounds are known but not wrapped in a class).

Args:
    omega: Array of frequencies to classify (can be complex)
    win: WindowPair object with val_window and cond_window

Returns:
    gl_indices: Indices into omega array for GL treatment (non-crossing)
    hgl_indices: Indices into omega array for HGL treatment (crossing)

<a id="gw_isdf.get_windows.compute_dos"></a>

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

<a id="gw_isdf.get_windows.find_optimal_partitions"></a>

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

<a id="gw_isdf.get_windows.minimize_cost_fn"></a>

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

<a id="gw_isdf.get_windows.get_window_info"></a>

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

<a id="gw_isdf.compute_vcoul"></a>

# gw\_isdf.compute\_vcoul

Chunked computation of V_q(μ, ν) = Σ_G ζ̃*_μ(G) ζ̃_ν(G) from zeta stored in HDF5.

This module provides memory-efficient routines for computing the ISDF Coulomb
matrix elements when the full zeta_q(μ, r) doesn't fit in GPU memory.

Key features:
- μ-chunked FFT: Process B_μ centroids at a time
- ν-chunked contraction: Compute V blocks without caching FFT outputs
- Hermitian symmetry: Only compute upper triangle, fill lower by conjugation
- 2D sharding: Output V_q sharded P('x', 'y') for downstream use

Memory model:
- FFT workspace: O(B_μ × n_G) per chunk
- V_q output: O(n_μ²) - typically small (e.g., 2304² × 16B = 85 MB)
- Redundant FFT work: O((n_μ/B_μ)²) vs O(n_μ/B_μ) with caching

Note: For future optimization, if a single zeta_q(μ, r) fits on sqrt(P) processors,
      we could batch multiple q-points to amortize FFT setup costs.

<a id="gw_isdf.compute_vcoul.exp_ikr_fftbox"></a>

#### exp\_ikr\_fftbox

```python
def exp_ikr_fftbox(fft_nx: int, fft_ny: int,
                   fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]
```

Return fractional coordinate grids for constructing exp(ik·r) on the FFT box.

<a id="gw_isdf.compute_vcoul.fft_integer_axes"></a>

#### fft\_integer\_axes

```python
def fft_integer_axes(fft_nx: int, fft_ny: int,
                     fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]
```

Return integer FFT frequency grids in numpy.fft.fftfreq order.

<a id="gw_isdf.compute_vcoul.compute_sqrt_vcoul_2d"></a>

#### compute\_sqrt\_vcoul\_2d

```python
def compute_sqrt_vcoul_2d(qvec_wrapped: jax.Array, fft_nx: int, fft_ny: int,
                          fft_nz: int, nkx: int, nky: int, nkz: int,
                          bvec: np.ndarray, cell_volume: float) -> jax.Array
```

Compute √v(q+G) for 2D truncated Coulomb on the FFT grid.

Returns:
    sqrt_v: (fft_nx, fft_ny, fft_nz) array of √v(q+G) values

<a id="gw_isdf.compute_vcoul.compute_phase_q"></a>

#### compute\_phase\_q

```python
def compute_phase_q(qvec_wrapped: jax.Array, fft_nx: int, fft_ny: int,
                    fft_nz: int, nkx: int, nky: int, nkz: int) -> jax.Array
```

Compute exp(-2πi q·r) phase factor for FFT.

Returns:
    phase: (1, fft_nx, fft_ny, fft_nz) array for broadcasting with zeta

<a id="gw_isdf.compute_vcoul.make_v_munu_chunked_kernel"></a>

#### make\_v\_munu\_chunked\_kernel

```python
def make_v_munu_chunked_kernel(fft_nx: int,
                               fft_ny: int,
                               fft_nz: int,
                               nkx: int,
                               nky: int,
                               nkz: int,
                               bvec: np.ndarray,
                               cell_volume: float,
                               sys_dim: int = 2)
```

Factory for jitted kernels that compute V_q blocks from zeta chunks.

This creates two kernels:
1. fft_and_weight: zeta_r(B_μ, n_rtot) → zeta_weighted(B_μ, n_G)
2. contract_block: (zeta_μ, zeta_ν) → V_block(B_μ, B_ν)

Args:
    fft_nx, fft_ny, fft_nz: FFT grid dimensions
    nkx, nky, nkz: k-grid dimensions
    bvec: Reciprocal lattice vectors (3×3)
    cell_volume: Unit cell volume
    sys_dim: System dimensionality (only 2 supported currently)

Returns:
    Namespace with fft_and_weight, contract_block, get_sqrt_v, get_phase kernels

<a id="gw_isdf.compute_vcoul.compute_V_q_from_zeta_h5"></a>

#### compute\_V\_q\_from\_zeta\_h5

```python
def compute_V_q_from_zeta_h5(zeta_h5,
                             q_idx: int,
                             qvec_wrapped: jax.Array,
                             fft_nx: int,
                             fft_ny: int,
                             fft_nz: int,
                             nkx: int,
                             nky: int,
                             nkz: int,
                             bvec: np.ndarray,
                             cell_volume: float,
                             mu_chunk_size: int = 128,
                             mesh_xy: Mesh = None,
                             sys_dim: int = 2) -> tuple[jax.Array, jax.Array]
```

Compute V_q(μ, ν) from zeta stored in HDF5 using μ/ν chunking.

V_q(μ, ν) = Σ_G ζ̃*_μ(G) ζ̃_ν(G)

where ζ̃_μ(G) = √v(q+G) × FFT[phase_q(r) × ζ_μ(r)]

Uses Hermitian symmetry: only computes upper triangle, fills lower by conjugation.
FFTs are recomputed per (μ,ν) block pair (no caching) to minimize memory.

Args:
    zeta_h5: Open HDF5 file or group containing 'zeta_q' dataset
             with shape (nqx, nqy, nqz, n_rmu, n_rtot)
    q_idx: Flat q-point index, or (qx, qy, qz) tuple
    qvec_wrapped: q-vector in wrapped crystal coordinates
    fft_nx, fft_ny, fft_nz: FFT grid dimensions
    nkx, nky, nkz: k-grid dimensions
    bvec: Reciprocal lattice vectors (3×3)
    cell_volume: Unit cell volume
    mu_chunk_size: Number of μ indices to process at once
    mesh_xy: Optional device mesh for 2D sharding of output
    sys_dim: System dimensionality (only 2 supported)

Returns:
    V_q: (n_rmu, n_rmu) Coulomb matrix, optionally sharded P('x', 'y')
    g0_mu: (n_rmu,) ζ_μ(G=0) for head corrections

<a id="gw_isdf.compute_vcoul.compute_V_q_from_zeta_array"></a>

#### compute\_V\_q\_from\_zeta\_array

```python
def compute_V_q_from_zeta_array(
        zeta_q: jax.Array,
        qvec_wrapped: jax.Array,
        fft_nx: int,
        fft_ny: int,
        fft_nz: int,
        nkx: int,
        nky: int,
        nkz: int,
        bvec: np.ndarray,
        cell_volume: float,
        mu_chunk_size: int = 128,
        mesh_xy: Mesh = None,
        sys_dim: int = 2) -> tuple[jax.Array, jax.Array]
```

Compute V_q(μ, ν) from zeta array in memory using μ/ν chunking.

Same as compute_V_q_from_zeta_h5 but takes zeta as a JAX array instead of HDF5.
Useful for testing or when zeta is already in memory.

Args:
    zeta_q: (n_rmu, n_rtot) zeta array for this q-point
    qvec_wrapped: q-vector in wrapped crystal coordinates
    ... (same as compute_V_q_from_zeta_h5)

Returns:
    V_q: (n_rmu, n_rmu) Coulomb matrix
    g0_mu: (n_rmu,) ζ_μ(G=0) for head corrections

<a id="gw_isdf.compute_vcoul.read_zeta_q_sharded"></a>

#### read\_zeta\_q\_sharded

```python
def read_zeta_q_sharded(zeta_h5, qx: int, qy: int, qz: int, n_rmu: int,
                        n_rtot: int, mesh_xy: Mesh) -> jax.Array
```

Read zeta_q from HDF5 with μ-sharding across processes.

Each process reads only its portion of μ indices, then combines
into a globally sharded array. This distributes I/O across nodes.

Args:
    zeta_h5: Open HDF5 file with 'zeta_q' dataset
    qx, qy, qz: q-point indices
    n_rmu: Total number of μ points
    n_rtot: Total number of r points
    mesh_xy: Device mesh for sharding

Returns:
    zeta_q: (n_rmu, n_rtot) array sharded along μ axis

<a id="gw_isdf.compute_vcoul.compute_all_V_q_from_zeta_h5"></a>

#### compute\_all\_V\_q\_from\_zeta\_h5

```python
def compute_all_V_q_from_zeta_h5(zeta_h5,
                                 kgrid: tuple[int, int, int],
                                 fft_grid: tuple[int, int, int],
                                 bvec: np.ndarray,
                                 cell_volume: float,
                                 mu_chunk_size: int = 128,
                                 mesh_xy: Mesh = None,
                                 sys_dim: int = 2,
                                 q_batch_size: int | None = None,
                                 verbose: bool = True) -> jax.Array
```

Compute V_q for all q-points from zeta stored in HDF5.

Loops over all q-points, computing V_q using μ-chunking for each. When the
μ chunks already cover the full set (single chunk), q-points can be batched
to reuse the FFT and contraction kernels.

Args:
    zeta_h5: Open HDF5 file containing 'zeta_q' with shape (nqx, nqy, nqz, n_rmu, n_rtot)
    kgrid: (nkx, nky, nkz) k-point grid dimensions
    fft_grid: (fft_nx, fft_ny, fft_nz) FFT grid dimensions
    bvec: Reciprocal lattice vectors (3×3)
    cell_volume: Unit cell volume
    mu_chunk_size: Number of μ indices per chunk
    mesh_xy: Optional device mesh for 2D sharding
    sys_dim: System dimensionality
    q_batch_size: Number of q-points to process simultaneously when
        mu_chunk_size ≥ n_rmu (default: no batching)
    verbose: Print timing breakdown

Returns:
    V_qmunu: (nqx, nqy, nqz, n_rmu, n_rmu) array of Coulomb matrices
    g0_mu_all: (nqx, nqy, nqz, n_rmu) array of G=0 components

<a id="gw_isdf.compute_vcoul.make_v_munu_kernel_chunked"></a>

#### make\_v\_munu\_kernel\_chunked

```python
def make_v_munu_kernel_chunked(fft_nx: int,
                               fft_ny: int,
                               fft_nz: int,
                               nkx: int,
                               nky: int,
                               nkz: int,
                               bvec: np.ndarray,
                               cell_volume: float,
                               sys_dim: int,
                               mu_chunk_size: int = 128)
```

Factory for chunked V_q kernel with same signature as gw_jax.make_v_munu_kernel.

Returns a kernel function that takes (zeta_q, qvec_wrapped) and returns (v_munu, g0_mu),
but uses μ-chunking internally for memory efficiency.

Drop-in replacement for make_v_munu_kernel when memory is constrained.

<a id="gw_isdf.archive.cohsex_jax_deprecated"></a>

# gw\_isdf.archive.cohsex\_jax\_deprecated

<a id="gw_isdf.archive.cohsex_jax_deprecated.read_cohsex_input"></a>

#### read\_cohsex\_input

```python
def read_cohsex_input(filename: str) -> dict
```

Parse a simple INI-style input file for the COHSEX driver.

<a id="gw_isdf.archive.cohsex_jax_deprecated.get_bandranges"></a>

#### get\_bandranges

```python
def get_bandranges(nv, nc, nband, nelec)
```

Return ranges of bands necessary for \sigma_{X,SX,COH}

<a id="gw_isdf.archive.cohsex_jax_deprecated.wrap_points_to_voronoi"></a>

#### wrap\_points\_to\_voronoi

```python
def wrap_points_to_voronoi(randcart, bvec, xp, nmax=1)
```

Helper function to get test q-points for mini-BZ average with correct voronoi cell.

<a id="gw_isdf.archive.cohsex_jax_deprecated.fft_bandrange"></a>

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

<a id="gw_isdf.archive.cohsex_jax_deprecated.get_zeta_q_and_v_q_mu_nu"></a>

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

<a id="gw_isdf.archive.cohsex_jax_deprecated.get_sigma_x_kij"></a>

#### get\_sigma\_x\_kij

```python
def get_sigma_x_kij(psi_l, psi_r, sigma_kbar, meta: Meta, xp)
```

Calculate the sigma_x_kij matrix elements.
sigma_mnkbar = \sum_rmu,rnu,s,s' exp(ik(r_nu-r_mu)) u_mk^*(r_mu,s) sigma_kbar,ss'(r_mu,r_nu) u_nk(r_nu,s')

<a id="gw_isdf.archive.cohsex_jax_deprecated.find_qpoint_index"></a>

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

<a id="gw_isdf.archive.cohsex_jax_deprecated.write_labeled_arrays_to_h5"></a>

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

<a id="gw_isdf.archive.cohsex_jax_deprecated.read_labeled_arrays_from_h5"></a>

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

<a id="gw_isdf.archive.test_eps_v_products"></a>

# gw\_isdf.archive.test\_eps\_v\_products

<a id="gw_isdf.archive.test_eps_v_products.compute_v_trunc_2d_for_eps"></a>

#### compute\_v\_trunc\_2d\_for\_eps

```python
def compute_v_trunc_2d_for_eps(G_comps_eps: np.ndarray,
                               bvec: np.ndarray) -> np.ndarray
```

Return 2D truncated Coulomb (8π/|G|^2)*f2d in Cartesian, no 1/Ω, for eps order.
Head (G=0) is left as 0 to avoid infs; caller can compare excluding it.

<a id="gw_isdf.archive.plot_whead_vs_model"></a>

# gw\_isdf.archive.plot\_whead\_vs\_model

<a id="gw_isdf.archive.plot_whead_vs_model.v2d_trunc_head_from_q"></a>

#### v2d\_trunc\_head\_from\_q

```python
def v2d_trunc_head_from_q(q_crys: np.ndarray, bvec: np.ndarray) -> float
```

Return 2D truncated head v(q) (8π/|q|^2)*f2d using q in crystal coords.
(Used only for gamma calibration if eps0 is absent.)

<a id="gw_isdf.archive.jax_fixed_point_demo"></a>

# gw\_isdf.archive.jax\_fixed\_point\_demo

JAX fixed-point demo (complex128, CPU) for Anderson, CROP, rCROP — *flattened over k*.

Synthetic linear test from the paper:
  A in R^{100x100}: tridiagonal with (1, -4, 1)  [or seven-diagonal]
  b in R^{100}: b = e_1
  f(x) = b - A x,  g(x) = x + f(x)
  x0 = 0, maxit = 100, tol = 1e-10

This version FLATTENS the k-axis (if any) into x, so the mixer treats a single
vector x ∈ C^N. That removes all per-k logic; early stopping uses a *scalar*
criterion ‖f(x)‖₂ ≤ tol.

Mapping to your SCF (HF/GW in ψ^(0) basis):
- x  ↔ vec(Σ^{in}_{mnk})  (flatten over m,n,k)
- f(x) ↔ Σ^{out}[x] − Σ^{in}[x]
- g(x) = x + f(x)

JIT-friendly choices:
- CPU + 64-bit enabled via env flags before importing JAX.
- complex128 everywhere.
- Fixed-size circular histories of depth m; no dynamic concatenation.
- Early-stop via jax.lax.while_loop with scalar `done`.

<a id="gw_isdf.archive.jax_fixed_point_demo.f_of_x"></a>

#### f\_of\_x

```python
@jax.jit
def f_of_x(A: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray
```

Residual map f(x) = b - A x (complex128). In SCF, this would be Σ_out - Σ_in.

<a id="gw_isdf.archive.jax_fixed_point_demo.qr_least_squares"></a>

#### qr\_least\_squares

```python
def qr_least_squares(F: jnp.ndarray,
                     r: jnp.ndarray,
                     ridge: float = 0.0) -> jnp.ndarray
```

Solve min_γ || r - F γ ||_2 with F ∈ C^{N×m}, r ∈ C^N → γ ∈ C^m.

<a id="gw_isdf.archive.jax_fixed_point_demo.anderson_fixed_history"></a>

#### anderson\_fixed\_history

```python
@partial(jax.jit, static_argnums=(3, 4))
def anderson_fixed_history(A,
                           b,
                           x0,
                           m: int = 3,
                           maxit: int = 100,
                           tol: float = 1e-10)
```

Anderson acceleration (Pulay) with fixed-size circular buffers (flattened x).
Early exit when ‖f‖₂ ≤ tol. Returns (res_buf, iters).

<a id="gw_isdf.archive.jax_fixed_point_demo.crop_family_fixed_history"></a>

#### crop\_family\_fixed\_history

```python
def crop_family_fixed_history(A,
                              b,
                              x0,
                              m: int = 3,
                              maxit: int = 100,
                              tol: float = 1e-10,
                              real_residual: bool = False)
```

CROP / rCROP with fixed-size histories (flattened x). Returns (x_final, residuals, iters).

<a id="gw_isdf.archive.jax_fixed_point_demo.crop_family_fixed_history_map"></a>

#### crop\_family\_fixed\_history\_map

```python
def crop_family_fixed_history_map(residual_fn,
                                  x0,
                                  m: int = 3,
                                  maxit: int = 100,
                                  tol: float = 1e-10,
                                  real_residual: bool = False)
```

CROP / rCROP driven by an arbitrary residual function f(x).

<a id="gw_isdf.archive.cohsex_isdf"></a>

# gw\_isdf.archive.cohsex\_isdf

<a id="gw_isdf.archive.cohsex_isdf.read_cohsex_input"></a>

#### read\_cohsex\_input

```python
def read_cohsex_input(filename: str) -> dict
```

Parse a simple INI-style input file for the COHSEX driver.

<a id="gw_isdf.archive.cohsex_isdf.get_bandranges"></a>

#### get\_bandranges

```python
def get_bandranges(nv, nc, nband, nelec)
```

Return ranges of bands necessary for \sigma_{X,SX,COH}

<a id="gw_isdf.archive.cohsex_isdf.wrap_points_to_voronoi"></a>

#### wrap\_points\_to\_voronoi

```python
def wrap_points_to_voronoi(randcart, bvec, xp, nmax=1)
```

Helper function to get test q-points for mini-BZ average with correct voronoi cell.

<a id="gw_isdf.archive.cohsex_isdf.fft_bandrange"></a>

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

<a id="gw_isdf.archive.cohsex_isdf.get_zeta_q_and_v_q_mu_nu"></a>

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

<a id="gw_isdf.archive.cohsex_isdf.get_sigma_x_kij"></a>

#### get\_sigma\_x\_kij

```python
def get_sigma_x_kij(psi_l, psi_r, sigma_kbar, xp)
```

Calculate the sigma_x_kij matrix elements.
sigma_mnkbar = \sum_rmu,rnu,s,s' exp(ik(r_nu-r_mu)) u_mk^*(r_mu,s) sigma_kbar,ss'(r_mu,r_nu) u_nk(r_nu,s')

<a id="gw_isdf.archive.cohsex_isdf.find_qpoint_index"></a>

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

<a id="gw_isdf.archive.cohsex_isdf.write_labeled_arrays_to_h5"></a>

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

<a id="gw_isdf.archive.cohsex_isdf.read_labeled_arrays_from_h5"></a>

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

<a id="gw_isdf.gw_jax"></a>

# gw\_isdf.gw\_jax

<a id="gw_isdf.gw_jax.compute_CCT_ZCT_for_q"></a>

#### compute\_CCT\_ZCT\_for\_q

```python
@partial(jax.jit, donate_argnums=(0, 1))
def compute_CCT_ZCT_for_q(CCT_buf: jax.Array, ZCT_buf: jax.Array,
                          psi_l_rmu: jax.Array, psi_r_rmu: jax.Array,
                          psi_l_rtot: jax.Array, psi_r_rtot: jax.Array,
                          psi_l_rmuT: jax.Array, psi_r_rmuT: jax.Array,
                          k_l_indices: jax.Array, k_r_indices: jax.Array)
```

Compute CCT and ZCT accumulators for a single q-point.

Args are sharded arrays with shapes:
- psi_*_rmu: (nk, nb, ns, n_rmu)
- psi_*_rtot: (nk, nb, ns, n_rtot)
- psi_*_rmuT: (nk, n_rmu, nb, ns)
- k_*_indices: (n_pairs,)

<a id="gw_isdf.gw_jax.solve_zeta_cholesky"></a>

#### solve\_zeta\_cholesky

```python
def solve_zeta_cholesky(CCT: jax.Array, ZCT: jax.Array) -> jax.Array
```

Regularize CCT, chol factor, and solve for zeta_q (n_rmu, n_rtot).

<a id="gw_isdf.gw_jax.compute_Sq_from_zeta"></a>

#### compute\_Sq\_from\_zeta

```python
@partial(jax.jit)
def compute_Sq_from_zeta(zeta_q: jax.Array) -> jax.Array
```

S_q = zeta^H zeta over rtot: (n_rmu, n_rtot) -> (n_rmu, n_rmu).

<a id="gw_isdf.gw_jax.exp_ikr_fftbox"></a>

#### exp\_ikr\_fftbox

```python
def exp_ikr_fftbox(fft_nx: int, fft_ny: int,
                   fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]
```

Return fractional coordinate grids for constructing exp(ik·r) on the FFT box.

<a id="gw_isdf.gw_jax.fft_integer_axes"></a>

#### fft\_integer\_axes

```python
def fft_integer_axes(fft_nx: int, fft_ny: int,
                     fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]
```

Return integer FFT frequency grids in numpy.fft.fftfreq order.

<a id="gw_isdf.gw_jax.as_index_tuple"></a>

#### as\_index\_tuple

```python
def as_index_tuple(vec) -> tuple[int, ...]
```

Convert an integer vector into a Python index tuple.

<a id="gw_isdf.gw_jax.make_v_munu_kernel"></a>

#### make\_v\_munu\_kernel

```python
def make_v_munu_kernel(fft_nx: int, fft_ny: int, fft_nz: int, nkx: int,
                       nky: int, nkz: int, bvec: np.ndarray,
                       cell_volume: float, sys_dim: int)
```

Factory for a jitted kernel that computes v_{μν} for one q on the dense FFT grid.

<a id="gw_isdf.gw_jax.compute_v_munu_from_zeta"></a>

#### compute\_v\_munu\_from\_zeta

```python
def compute_v_munu_from_zeta(zeta_q: jax.Array, qvec_wrapped: jax.Array,
                             fft_nx: int, fft_ny: int, fft_nz: int, nkx: int,
                             nky: int, nkz: int, bvec: np.ndarray,
                             cell_volume: float, sys_dim: int) -> jax.Array
```

Reference helper that reuses the dense-grid kernel to obtain v_{μν}(q).

<a id="gw_isdf.gw_jax.make_shardings"></a>

#### make\_shardings

```python
def make_shardings(mesh_xy: Mesh) -> SimpleNamespace
```

Centralize all NamedSharding declarations used in this file.

<a id="gw_isdf.gw_jax.build_q_coulomb_cache"></a>

#### build\_q\_coulomb\_cache

```python
def build_q_coulomb_cache(wfn,
                          sym,
                          meta: Meta,
                          do_Dmunu: bool,
                          sys_dim: int,
                          mesh_xy: Mesh | None = None) -> SimpleNamespace
```

Precompute q-grid Coulomb metadata reused inside the q-loop.

Returns batched JAX arrays for the regular shapes so the q-loop can
run with minimal host-device transfers.

<a id="gw_isdf.gw_jax.determine_wcoul0"></a>

#### determine\_wcoul0

```python
def determine_wcoul0(params, input_dir, wfn, sym, meta, print_fn)
```

Resolve (v_c0, w_c0) head averages using user preference fallback order.

<a id="gw_isdf.gw_jax.get_zeta_q_and_v_q_mu_nu"></a>

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

<a id="gw_isdf.gw_jax.fit_zeta_and_compute_V_q_chunked"></a>

#### fit\_zeta\_and\_compute\_V\_q\_chunked

```python
def fit_zeta_and_compute_V_q_chunked(wfn,
                                     sym,
                                     meta: Meta,
                                     centroid_indices: jax.Array,
                                     mesh_xy: Mesh,
                                     output_dir: str,
                                     bispinor: bool = False,
                                     memory_budget_gb: float = 6.0,
                                     sys_dim: int = 2,
                                     r_chunk_override: int = 0)
```

Chunked zeta fitting and V_q computation pipeline.

This replaces the per-q-point zeta fitting in the main loop with a memory-efficient
chunked approach that:
1. Loads wavefunctions for full band range (b0 to b4)
2. Slices for left (b0→b3) and right (b0→b4) with spin-traced pair density
3. Fits zeta via z-chunked algorithm and writes to HDF5
4. Reads zeta back and computes V_qmunu

Physics note:
	Uses spin-traced pair density P_k(μ,ν) = Σ_{n,s} ψ*_{n,k,s}(μ) ψ_{n,k,s}(ν)
	matching gw_jax convention for ISDF fitting. Different from keeping all
	spin combinations which would increase lstsq error.

Args:
	wfn: WFNReader object
	sym: SymMaps object
	meta: Meta object with system info
	centroid_indices: ISDF centroid indices
	mesh_xy: 2D device mesh
	output_dir: Directory for zeta HDF5 file
	bispinor: Whether to use bispinor wavefunctions
	memory_budget_gb: Memory budget per device in GB
	sys_dim: System dimensionality (2 or 3)
r_chunk_override: If > 0, use explicit r-chunk size (flattened xyz index).

Returns:
	Dictionary with:
	- V_qmunu: (1, npol, npol, nkx, nky, nkz, n_rmu, n_rmu) Coulomb matrix
	- v_q0_noG0_munu: (n_rmu, n_rmu) V_q at q=0 with G=0 excluded
	- G0_mu_nu: (n_rmu,) ζ_μ(G=0) for head corrections
	- psi_l_rmu_Y: Left centroid wfns, Y-sharded
	- psi_l_rmuT_X: Left conjugated wfns, X-sharded  
	- psi_r_rmu_Y: Right centroid wfns, Y-sharded
	- psi_r_rmuT_X: Right conjugated wfns, X-sharded
	- zeta_h5_path: Path to zeta HDF5 file

<a id="gw_isdf.gw_jax.get_G_mu_nu_jax"></a>

#### get\_G\_mu\_nu\_jax

```python
def get_G_mu_nu_jax(psi_vTX, psi_vY, Gij_static)
```

Pure: psi_* (nk, nb, nspinor, n_rmu), Gij_static (nk,nb,nb) -> G_k (nk, nspinor, n_rmu, nspinor, n_rmu).

Computes G_μν(k) = Σ_ij ψ*_ik(r_μ) G_ijk ψ_jk(r_ν)

Zero-comm contraction when left is X-sharded on rmu and right is Y-sharded on rmu.
Gij_static should be initialized as zeros with diagonal 0:nelec set to 1.0+0.j
for the static COHSEX Green's function (identity on occupied states).

<a id="gw_isdf.gw_jax.get_G_mu_nu_RI"></a>

#### get\_G\_mu\_nu\_RI

```python
def get_G_mu_nu_RI(psi_vTX, psi_vY)
```

Pure: psi_* (nk, nb, nspinor, n_rmu) -> G_k (nk, nspinor, n_rmu, nspinor, n_rmu).

Computes G_μν(k) = Σ_n ψ*_nk(r_μ) ψ_nk(r_ν) for ALL bands (no occupation weighting).

This is the "resolution of identity" style sum used for the Coulomb hole term.
Zero-comm contraction when left is X-sharded on rmu and right is Y-sharded on rmu.

<a id="gw_isdf.gw_jax.get_G_R_jax"></a>

#### get\_G\_R\_jax

```python
def get_G_R_jax(G_k, nkx, nky, nkz)
```

Pure: (nk, s1,rmu1,s2,rmu2) -> (s1,rmu1,s2,rmu2,nkx,nky,nkz).

<a id="gw_isdf.gw_jax.get_sigma_static_mu_nu_jax"></a>

#### get\_sigma\_static\_mu\_nu\_jax

```python
def get_sigma_static_mu_nu_jax(G_R, V_mu_nu, nk_tot, bispinor=False)
```

Compute sigma in (s1,rmu1,s2,rmu2,nkx,nky,nkz) basis via convolution in real space.

For nspinor=2: Σ_ab(μ,ν,R) = G_ab(μ,ν,R) * V(μ,ν,R)

For nspinor=4 (bispinor): Uses γ⁰ Coulomb vertex:
	Σ_ab = γ⁰_aa γ⁰_bb G_ab V
where γ⁰ = diag(1,1,-1,-1) in the Dirac representation.
This gives sign_a × sign_b × G_ab × V where sign=[1,1,-1,-1].
Large-large and small-small blocks get +1, cross terms get -1.

Args:
	G_R: (nspinor, rmu1, nspinor, rmu2, nkx, nky, nkz) Green's function in real space
	V_mu_nu: (rmu1, rmu2, nkx, nky, nkz) Coulomb interaction
	nk_tot: Total number of k-points for normalization
	bispinor: If True, apply γ⁰ vertex factors for 4-component spinors

Returns:
	sigma_k: Same shape as G_R, self-energy in k-space

<a id="gw_isdf.gw_jax.get_sigma_static_kij_jax"></a>

#### get\_sigma\_static\_kij\_jax

```python
def get_sigma_static_kij_jax(psi_sigX, psi_sigTY, sigma_k_munu)
```

Project self-energy from (spinor,rmu) basis to band basis.

Computes: Σ_mn(k) = Σ_{s,t,μ,ν} ψ*_ms(k,μ) Σ_st(k,μ,ν) ψ_nt(k,ν)

Works for both 2-component (Pauli) and 4-component (bispinor) wavefunctions.
The spinor contraction sums over all spinor components (s,t indices).

Args:
	psi_sigX: (nk, nb, nspinor, rmu) wavefunctions
	psi_sigTY: (nk, nspinor, rmu, nb) transposed wavefunctions
	sigma_k_munu: (nspinor, rmu1, nspinor, rmu2, nkx, nky, nkz) self-energy

Returns:
	sigma_kij: (nk, nb, nb) band-space self-energy matrix

<a id="gw_isdf.gw_jax.build_density_from_Gij"></a>

#### build\_density\_from\_Gij

```python
def build_density_from_Gij(psi_rmu, Gij, nk_tot)
```

Build charge density at centroids from wavefunctions and Green's function.

ρ_μ = (1/Nk) Σ_k Σ_{ij} G_ij(k) ψ*_ik(r_μ) ψ_jk(r_μ)

For diagonal Gij (initial), this reduces to Σ_n f_n |ψ_n|².
For Gij = U @ diag(f) @ U† (self-consistent), this correctly computes
the density from the QP Green's function in the DFT basis.

Args:
	psi_rmu: (nk, nb, nspinor, n_rmu) wavefunctions at centroids
	Gij: (nk, nb, nb) Green's function matrix (FULL matrix, not just diagonal)
	nk_tot: Total number of k-points for BZ averaging

Returns:
	rho_mu: (n_rmu,) density at centroids

<a id="gw_isdf.gw_jax.build_hartree_potential"></a>

#### build\_hartree\_potential

```python
def build_hartree_potential(rho_mu, V0_munu)
```

Build Hartree potential at centroids from density.

[Vρ]_μ = Σ_ν V0_μν ρ_ν

Args:
	rho_mu: (n_rmu,) density at centroids
	V0_munu: (n_rmu, n_rmu) bare Coulomb at q=0 (G=0 excluded)

Returns:
	Vrho_mu: (n_rmu,) Hartree potential at centroids

<a id="gw_isdf.gw_jax.project_potential_to_bands"></a>

#### project\_potential\_to\_bands

```python
def project_potential_to_bands(psi_rmu, Vrho_mu)
```

Project local potential to band matrix elements.

V_mn(k) = Σ_μ,s ψ*_mk(r_μ) V_μ ψ_nk(r_μ)

Args:
	psi_rmu: (nk, nb, nspinor, n_rmu) wavefunctions at centroids
	Vrho_mu: (n_rmu,) potential at centroids

Returns:
	V_kmn: (nk, nb, nb) potential matrix elements

<a id="gw_isdf.gw_jax.compute_sigma_pipeline_jax"></a>

#### compute\_sigma\_pipeline\_jax

```python
def compute_sigma_pipeline_jax(psi_l_rmuT_X,
                               psi_l_rmu_Y,
                               psi_coh_rmuT_X,
                               psi_coh_rmu_Y,
                               psi_proj_rmu_X,
                               psi_proj_rmuT_Y,
                               W_mu_nu,
                               V_mu_nu,
                               V0_munu,
                               Gij_static,
                               nkx: int,
                               nky: int,
                               nkz: int,
                               nk_tot: int,
                               nspinor: int,
                               fft_vol_au: float,
                               bispinor: bool = False)
```

Pure JAX pipeline: compute static COHSEX self-energy components and Hartree.

Returns:
	sigma_sx_kij: (nk, nb_sigma, nb_sigma) complex - screened exchange self-energy
	sigma_coh_kij: (nk, nb_sigma, nb_sigma) complex - Coulomb hole self-energy
	hartree_kmn: (nk, nb_sigma, nb_sigma) complex - Hartree matrix elements

Wavefunctions:
	psi_l: sigma window bands (b0, b3) for SX Green's function + Hartree density
	       shape (nk, nb_sigma, nspinor, n_rmu)
	psi_coh: ALL bands (b0, b4) for COH resolution of identity
	         shape (nk, nband_full, nspinor, n_rmu)
	psi_proj: sigma window bands (b0, b3) for final projection <m|Σ|n>
	          shape (nk, nb_sigma, nspinor, n_rmu)

Gij_static:
	Static Green's function matrix in band space, shape (nk, nb_sigma, nb_sigma).
	For COHSEX: zeros with diagonal 0:nelec set to 1.0+0.j (projector onto occupied).
	Must match psi_l band range.

W_mu_nu:
	Screened Coulomb interaction, shape (nrmu1, nrmu2, nkx, nky, nkz).
	Same shardings as V_mu_nu.

SCREENED EXCHANGE (Σ_sx):
	G_μν(k) = Σ_ij ψ*_ik(r_μ) G_ijk ψ_jk(r_ν)  [Green's function from Gij_static]
	G_μν(R) = FFT[ G_μν(k) ]                    [to real-space lattice]
	Σ_sx_μν(k) = (1/N_k) Σ_R G_μν(R) W_μν(R)   [screened exchange in ISDF basis]
	Σ_sx_ij(k) = Σ_μν ψ*_i(r_μ) Σ_μν ψ_j(r_ν)  [project to sigma bands]

COULOMB HOLE (Σ_coh):
	G_RI_μν(k) = Σ_n ψ*_nk(r_μ) ψ_nk(r_ν)      [RI sum over ALL nband bands]
	G_RI_μν(R) = FFT[ G_RI_μν(k) ]              [to real-space lattice]
	Σ_coh_μν(k) = (1/N_k) Σ_R G_RI_μν(R) [V_μν(R) - W_μν(R)]
	Σ_coh_ij(k) = Σ_μν ψ*_i(r_μ) Σ_μν ψ_j(r_ν) [project to sigma bands]

HARTREE (V_H):
	ρ_μ = (1/N_k) Σ_k,n,s f_n |ψ_nk(r_μ)|²   [density, weighted by Gij diagonal]
	[Vρ]_μ = Σ_ν V0_μν ρ_ν                    [Hartree potential at centroids]
	<m|V_H|n>_k = Σ_μ,s ψ*_mk(r_μ) [Vρ]_μ ψ_nk(r_μ)  [project to sigma bands]

Key: V0_munu is V(q=0) with G=0 component EXCLUDED (to avoid divergence).
     The G=0 piece is added back via the head correction in the main pipeline.

<a id="gw_isdf.gw_jax.summarize_hermitian_matrix"></a>

#### summarize\_hermitian\_matrix

```python
def summarize_hermitian_matrix(name: str,
                               mats: np.ndarray,
                               print_fn=print,
                               warn_threshold: float = 1e-6)
```

Emit diagnostics for a batch of Hermitian matrices shaped (nk, nb, nb).

<a id="gw_isdf.gw_jax.preprocess_q_loops"></a>

#### preprocess\_q\_loops

```python
def preprocess_q_loops(wfn, sym, meta, mesh_xy=None)
```

Compatibility helper that materializes the legacy q-point cache.

Prefer :func:`iter_qpoint_data` plus on-demand evaluation. This function
exists so downstream code that still expects the old tuple-of-arrays
representation keeps working.

<a id="gw_isdf.hgl_quadrature"></a>

# gw\_isdf.hgl\_quadrature

hgl_quadrature.py

Hermite-Gauss-Laguerre (HGL) quadrature for weight function exp(-τ - τ²/2).

Implements the Golub-Welsch algorithm to compute nodes and weights for the
HGL quadrature, as described in Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020),
Appendix H.

The key identity is:
    1/x = γ ∫₀^∞ sin(γxτ) exp(-τ - τ²/2) dτ    [for x > 0, any γ > 0]

For the Complex-Time Shredded Propagator (CTSP) method, HGL is used when
the frequency ω falls within the transition energy range [E_gap, E_bw],
causing energy crossings (denominator changes sign).

<a id="gw_isdf.hgl_quadrature.hgl_nodes_weights"></a>

#### hgl\_nodes\_weights

```python
def hgl_nodes_weights(n: int,
                      n_gl_base: int = 200) -> tuple[np.ndarray, np.ndarray]
```

Compute n-point HGL quadrature nodes and weights using Golub-Welsch algorithm.

For the weight function h(τ) = exp(-τ - τ²/2), computes nodes {τ_u} and weights {w_u}
such that:

    ∫₀^∞ f(τ) exp(-τ - τ²/2) dτ ≈ Σ_u w_u f(τ_u)

This is a direct translation of the MATLAB code from Kim et al. Appendix H:
the key insight is to use GL quadrature to compute inner products stably.

Args:
    n: Number of quadrature points
    n_gl_base: Base number of GL points for inner products (max ~300 before scipy overflows)

Returns:
    nodes: Array of n nodes (sorted ascending)
    weights: Array of n corresponding weights

Example:
    >>> tau, w = hgl_nodes_weights(10)
    >>> # Test: ∫₀^∞ sin(x*τ) exp(-τ - τ²/2) dτ ≈ Σ w_u sin(x*τ_u)
    >>> x = 1.0
    >>> np.sum(w * np.sin(x * tau))  # Should approximate F(x)

<a id="gw_isdf.hgl_quadrature.n_tau_hgl"></a>

#### n\_tau\_hgl

```python
def n_tau_hgl(gamma: float, E_bw: float, epsilon: float = 0.01) -> int
```

Estimate the number of HGL quadrature points needed for fractional error ε.

Uses the empirical fit from Appendix D of Kim et al.:
    N^(τ,HGL) = c_2(ε) x² + c_1(ε) x + c_0(ε)

where x = γ × E_bw (bandwidth in scaled units).

Args:
    gamma: Scaling parameter (= z_lm = 1/√(E_gap × E_bw))
    E_bw: Energy bandwidth of the window
    epsilon: Target fractional error (default 0.01 = 1%)

Returns:
    Recommended number of quadrature points (at least 3)

<a id="gw_isdf.vcoul"></a>

# gw\_isdf.vcoul

Coulomb utilities: Voronoi-cell sampling and per-q V(q,G).

This module now supports Sobol QMC sampling for the q=0 averages by default
and keeps the V(q,G) head zero so head averages are injected explicitly later.

<a id="gw_isdf.vcoul.wrap_points_to_voronoi"></a>

#### wrap\_points\_to\_voronoi

```python
def wrap_points_to_voronoi(randcart, bvec, nmax=1)
```

Helper function to get test q-points for mini-BZ average with correct Voronoi cell.
Rewritten to use JAX arrays.

<a id="gw_isdf.vcoul.compute_vcoul_comps_for_q"></a>

#### compute\_vcoul\_comps\_for\_q

```python
def compute_vcoul_comps_for_q(wfn, sym, meta: Meta, qvec_nonneg)
```

Return vcoul_psiG_comps for a single q (array of shape (nG,3), int32).

<a id="gw_isdf.vcoul.compute_V_qfullG_for_q"></a>

#### compute\_V\_qfullG\_for\_q

```python
def compute_V_qfullG_for_q(wfn, qvec_wrapped, comps_qG, vc0_mean, do_Dmunu,
                           sys_dim)
```

Compute V_q(G) vector (length nG for this q) using 2D truncation if sys_dim==2.

The head (q+G=0) is set to zero; head averages are injected later.

<a id="gw_isdf.vcoul.compute_q0_averages"></a>

#### compute\_q0\_averages

```python
def compute_q0_averages(wfn,
                        epshead,
                        meta: Meta,
                        S_cart: jnp.ndarray | None = None,
                        nsamples: int = 2**18,
                        method: str = "sobol",
                        qmc_reps: int = 1)
```

Compute q=0 averages (vc0_mean, wcoul0) on the same Monte Carlo points.

If ``S_cart`` is provided, compute
	wcoul0 = < v(q) / (1 - v(q) q^T S q) >
using the same mini-BZ Voronoi samples used to compute ``vc0_mean``.

Otherwise, fall back to the historical Ismail–Beigi gamma model using
``epshead`` (for continuity with older runs).

<a id="gw_isdf.vcoul.compute_wcoul0_with_S"></a>

#### compute\_wcoul0\_with\_S

```python
def compute_wcoul0_with_S(bvec: jnp.ndarray,
                          nkx: int,
                          nky: int,
                          nkz: int,
                          S_cart: jnp.ndarray,
                          nsamples: int = 2_500_000) -> jnp.complex128
```

Average wcoul0 over Voronoi-cell q using small-q tensor S(ω).

Computes wcoul0 = ⟨ v(q) · (1 - v(q) · q^T S q)^{-1} ⟩ over the Voronoi cell,
with 2D truncation consistent with vcoul setup.

<a id="gw_isdf.w_isdf_dynamic"></a>

# gw\_isdf.w\_isdf\_dynamic

Dynamic W(ω) routines with per-window JIT compilation.

These routines use dynamic window slicing and compile separate kernels
per (max_val_len, max_cond_len) pair.

For the streamlined static χ₀/W pipeline, see w_isdf.py.

<a id="gw_isdf.w_isdf_dynamic.get_chi_lm_Yt_jax_windowed"></a>

#### get\_chi\_lm\_Yt\_jax\_windowed

```python
def get_chi_lm_Yt_jax_windowed(psi_vTX: jax.Array,
                               psi_vY: jax.Array,
                               psi_cX: jax.Array,
                               psi_cTY: jax.Array,
                               enk_v: jax.Array,
                               enk_c: jax.Array,
                               win,
                               meta: Meta,
                               mesh_xy: Mesh | None = None)
```

Compute chi_lm for a specific window (causes JIT recompilation per window shape).

<a id="gw_isdf.w_isdf_dynamic.get_chi0_jax_windowed"></a>

#### get\_chi0\_jax\_windowed

```python
def get_chi0_jax_windowed(psi_vTX: jax.Array,
                          psi_vY: jax.Array,
                          psi_cX: jax.Array,
                          psi_cTY: jax.Array,
                          enk_v: jax.Array,
                          enk_c: jax.Array,
                          windows,
                          meta: Meta,
                          mesh_xy: Mesh | None = None)
```

Sum chi over windows using window-specific kernels (legacy interface).

<a id="gw_isdf.w_isdf_dynamic.get_chi_omega_jax"></a>

#### get\_chi\_omega\_jax

```python
def get_chi_omega_jax(psi_vTX: jax.Array, psi_vY: jax.Array, psi_cX: jax.Array,
                      psi_cTY: jax.Array, enk_v: jax.Array, enk_c: jax.Array,
                      windows, omega: float, meta: Meta, mesh_xy: Mesh)
```

Compute χ(ω) using GL/HGL per window based on crossing detection.

<a id="gw_isdf.w_isdf_dynamic.get_w_omega_jax"></a>

#### get\_w\_omega\_jax

```python
def get_w_omega_jax(V_qmunu: jax.Array,
                    psi_vTX: jax.Array,
                    psi_vY: jax.Array,
                    psi_cX: jax.Array,
                    psi_cTY: jax.Array,
                    enk_v: jax.Array,
                    enk_c: jax.Array,
                    windows,
                    omega: float,
                    meta: Meta,
                    mesh_xy: Mesh,
                    S_qmunu: jax.Array | None = None)
```

Compute W(ω) = V / (1 - V χ(ω)) for a single frequency.

<a id="gw_isdf.kin_ion_io_chunked"></a>

# gw\_isdf.kin\_ion\_io\_chunked

Chunked version of kin_ion_io that processes k-points in batches to avoid OOM.

This version:
- Processes k-points in configurable chunks (default: 16 at a time)
- Writes results incrementally to HDF5
- Uses much less GPU memory for large k-grids

Usage:
  python -m gw_isdf.kin_ion_io_chunked -i gw.inp -o kin_ion.h5 --kchunk 16

<a id="gw_isdf.kin_ion_io_chunked.get_kin_ion_k"></a>

#### get\_kin\_ion\_k

```python
def get_kin_ion_k(wfn_k, Gk_crys, kvec, plan, pseudos, wfn, meta,
                  species_payload)
```

Compute kin+ion for a single k-point.

<a id="gw_isdf.w_from_eps0"></a>

# gw\_isdf.w\_from\_eps0

<a id="gw_isdf.w_from_eps0.compute_Wmunu_from_eps0_body"></a>

#### compute\_Wmunu\_from\_eps0\_body

```python
def compute_Wmunu_from_eps0_body(wfn, sym, meta, zeta_q_r: np.ndarray,
                                 qvec_wrapped: np.ndarray,
                                 eps0_path: str) -> np.ndarray
```

Compute body-only W_{mu,nu}(q=0) from eps^{-1}_q=0 and dense FFT data.

Args:
    zeta_q_r: (n_mu, n_rtot) real-space zeta for this q (q=0)
    qvec_wrapped: fractional q-vector (length-3) (used for phase; here should be 0)
    eps0_path: path to eps0mat.h5 (assumed eps^{-1} at q=0)
Returns:
    W_{mu,nu} as a complex128 NumPy array (n_mu, n_mu)

<a id="gw_isdf.w_isdf"></a>

# gw\_isdf.w\_isdf

Static χ₀ and W computation with JAX.

Streamlined pipeline for COHSEX:
- Single universal chi kernel with energy masking (no per-window JIT recompilation)
- Two-stage resharding for W solve following load_wfns pattern
- χ computed with P(..., μ_X, ..., ν_Y) sharding
- V, χ resharded to P(q_XY, μ, ν) for Dyson solve

For dynamic W(ω) with window-specific kernels, see w_isdf_dynamic.py.

<a id="gw_isdf.w_isdf.compute_chi0"></a>

#### compute\_chi0

```python
def compute_chi0(psi_vTX: jax.Array, psi_vY: jax.Array, psi_cX: jax.Array,
                 psi_cTY: jax.Array, enk_v: jax.Array, enk_c: jax.Array,
                 windows, meta: Meta, mesh_xy: Mesh) -> jax.Array
```

Compute static χ₀(q) by summing over all window pairs.

Uses a single universal kernel with energy masking - no per-window JIT.

Args:
    psi_vTX: (nk, ns, μ, nb_v) valence, μ sharded on X axis
    psi_vY:  (nk, nb_v, ns, μ) valence, μ sharded on Y axis
    psi_cX:  (nk, nb_c, ns, μ) conduction, μ sharded on X axis
    psi_cTY: (nk, ns, μ, nb_c) conduction, μ sharded on Y axis
    enk_v: (nk, nb_v) valence energies
    enk_c: (nk, nb_c) conduction energies
    windows: list of WindowPair objects
    meta: Meta with nkx, nky, nkz
    mesh_xy: 2D device mesh

Returns:
    chi_q: (nkx, nky, nkz, 1, n_rmu, 1, n_rmu) with μ_X, ν_Y sharding

<a id="gw_isdf.w_isdf.get_static_w_q_jax"></a>

#### get\_static\_w\_q\_jax

```python
def get_static_w_q_jax(V_qmunu: jax.Array, chi_q: jax.Array,
                       S_qmunu: jax.Array | None, meta: Meta,
                       mesh_xy: Mesh) -> jax.Array
```

Compute static W_q = (I - V χ)^{-1} V.

Uses two-stage resharding following load_wfns pattern:
- χ input: P(None, None, None, None, 'x', None, 'y')
- Reshard to q-parallel for solve
- W output: same as χ

Args:
    V_qmunu: (1, 1, 1, nkx, nky, nkz, n_rmu, n_rmu)
    chi_q:   (nkx, nky, nkz, 1, n_rmu, 1, n_rmu)
    S_qmunu: overlap matrix (not yet implemented) or None
    meta: Meta with k-grid info
    mesh_xy: 2D device mesh

Returns:
    W_q: (nkx, nky, nkz, 1, n_rmu, 1, n_rmu) with μ_X, ν_Y sharding

<a id="gw_isdf.w_isdf.compute_chi0_and_w"></a>

#### compute\_chi0\_and\_w

```python
def compute_chi0_and_w(V_qmunu: jax.Array, psi_vTX: jax.Array,
                       psi_vY: jax.Array, psi_cX: jax.Array,
                       psi_cTY: jax.Array, enk_v: jax.Array, enk_c: jax.Array,
                       windows, meta: Meta,
                       mesh_xy: Mesh) -> tuple[jax.Array, jax.Array]
```

Compute static χ₀ and screened interaction W.

Streamlined pipeline:
1. χ₀(q) via universal kernel with energy masking
2. W(q) via two-stage resharding + Dyson solve

Returns:
    chi_q: (nkx, nky, nkz, 1, n_rmu, 1, n_rmu)
    W_q:   (nkx, nky, nkz, 1, n_rmu, 1, n_rmu)

<a id="gw_isdf.w_isdf.get_chi0_jax"></a>

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
                 mesh_xy: Mesh | None = None) -> jax.Array
```

Compute static χ₀ (alias for compute_chi0).

<a id="gw_isdf.w_isdf.get_w_omega_jax"></a>

#### get\_w\_omega\_jax

```python
def get_w_omega_jax(*args, **kwargs)
```

Dynamic W(ω) - delegates to the dynamic implementation.

<a id="gw_isdf.w_isdf.get_chi_omega_jax"></a>

#### get\_chi\_omega\_jax

```python
def get_chi_omega_jax(*args, **kwargs)
```

Dynamic χ(ω) - delegates to the dynamic implementation.

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

<a id="isdf.common.cholesky_2d"></a>

# isdf.common.cholesky\_2d

2D Block-Distributed Cholesky Decomposition for JAX.

This module provides memory-efficient Cholesky decomposition for matrices
sharded across a 2D processor mesh. It avoids the "involuntary full 
rematerialization" issue that occurs when resharding from P(None, 'x', 'y') 
to P(('x','y'), None, None).

Key features:
- Works directly on C_q(μ_X, ν_Y) without resharding
- O(J × log P) communication rounds via psum
- √P factor less bandwidth than 1D distribution
- Uses lax.map for efficient batched processing over q-points

Usage:
    from isdf.common.cholesky_2d import cholesky_2d_batched, tiles_to_dense

    # Build the batched Cholesky function for your mesh and tile sizes
    chol_fn = cholesky_2d_batched(mesh_2d, J=n//b, b=block_size)

    # Apply to all q-points at once (single XLA dispatch!)
    L_tiles = chol_fn(C_q_tiles)  # (nq, J, J, b, b) -> (nq, J, J, b, b)

    # Optionally convert back to dense
    L_dense = tiles_to_dense(L_tiles, b)  # (nq, n, n)

Memory comparison for n=10k, P=128:
    - Reshard strategy: 1.6 GB/device (may OOM)
    - 2D blocked:       5 MB/device

<a id="isdf.common.cholesky_2d.dense_to_tiles"></a>

#### dense\_to\_tiles

```python
def dense_to_tiles(A: jax.Array, b: int) -> jax.Array
```

Convert dense lower-triangular matrix to blocked tile format.

Args:
    A: Dense matrix of shape (..., n, n)
    b: Block size (n must be divisible by b)

Returns:
    Tiles of shape (..., J, J, b, b) where J = n // b
    Upper tiles (i < j) are zeros.

<a id="isdf.common.cholesky_2d.tiles_to_dense"></a>

#### tiles\_to\_dense

```python
def tiles_to_dense(tiles: jax.Array, b: int) -> jax.Array
```

Convert blocked tile format back to dense matrix.

Args:
    tiles: Tiles of shape (..., J, J, b, b)
    b: Block size

Returns:
    Dense matrix of shape (..., n, n) where n = J * b

<a id="isdf.common.cholesky_2d.cholesky_2d_single"></a>

#### cholesky\_2d\_single

```python
def cholesky_2d_single(mesh: Mesh, J: int, b: int)
```

Build a shard_map function for 2D blocked Cholesky of a single matrix.

This is the core algorithm. For batched processing over q-points,
use cholesky_2d_batched() which wraps this with lax.map.

Args:
    mesh: 2D JAX mesh with axes ('x', 'y')
    J: Number of tile blocks per dimension
    b: Block size (each tile is b×b)

Returns:
    Function that takes (J, J, b, b) tiles sharded as P('x', 'y', None, None)
    and returns Cholesky factor in same format/sharding.

Algorithm (right-looking blocked Cholesky):
    for k = 0 to J-1:
        1. POTRF: diagonal owner factors L[k,k] = chol(A[k,k])
        2. Broadcast L[k,k] via psum (O(log P) rounds)
        3. TRSM: column k owners compute L[i>k, k] = A[i,k] @ L[k,k]^{-H}
        4. Broadcast panel L[:,k] via psum
        5. SYRK: all procs update A[i,j] -= L[i,k] @ L[j,k]^H for i,j>k

<a id="isdf.common.cholesky_2d.cholesky_2d_batched"></a>

#### cholesky\_2d\_batched

```python
def cholesky_2d_batched(mesh: Mesh, J: int, b: int)
```

Build a batched 2D Cholesky function that processes all q-points efficiently.

Uses lax.map internally for ~18x speedup over Python loop by using
a single XLA dispatch for all q-points.

Args:
    mesh: 2D JAX mesh with axes ('x', 'y')
    J: Number of tile blocks per dimension  
    b: Block size

Returns:
    JIT-compiled function that takes (nq, J, J, b, b) tiles 
    sharded as P(None, 'x', 'y', None, None) and returns 
    Cholesky factors in same format.

Example:
    chol_fn = cholesky_2d_batched(mesh, J=16, b=64)
    L_all = chol_fn(C_q_tiles)  # Single dispatch for all q!

<a id="isdf.common.cholesky_2d.solve_triangular_2d"></a>

#### solve\_triangular\_2d

```python
def solve_triangular_2d(L_tiles: jax.Array,
                        B_tiles: jax.Array,
                        mesh: Mesh,
                        lower: bool = True,
                        trans: str = 'N') -> jax.Array
```

Solve L @ X = B or L^H @ X = B with 2D sharded triangular matrix.

For Cholesky solve C @ x = b, use:
    L = cholesky_2d(C)
    y = solve_triangular_2d(L, b, lower=True, trans='N')   # L @ y = b
    x = solve_triangular_2d(L, y, lower=True, trans='C')   # L^H @ x = y

Note: Currently this replicates L for simplicity. For very large L,
a distributed triangular solve would be needed.

Args:
    L_tiles: Lower triangular factor (nq, J, J, b, b) or (J, J, b, b)
    B_tiles: Right-hand side, same tile format
    mesh: 2D mesh
    lower: Whether L is lower triangular (always True for Cholesky)
    trans: 'N' for L, 'C' for L^H, 'T' for L^T

Returns:
    X_tiles in same format as B_tiles

<a id="isdf.common.cholesky_2d.cholesky_solve_2d"></a>

#### cholesky\_solve\_2d

```python
def cholesky_solve_2d(C_q_tiles: jax.Array, Z_q_tiles: jax.Array, mesh: Mesh,
                      J: int, b: int) -> jax.Array
```

Solve C_q @ zeta_q = Z_q using 2D blocked Cholesky.

This is the full pipeline for ISDF zeta fitting:
1. L = chol(C_q)
2. y = L^{-1} @ Z_q  (forward solve)
3. zeta = L^{-H} @ y (backward solve)

Args:
    C_q_tiles: CCT matrix (nq, J, J, b, b) sharded P(None, 'x', 'y', None, None)
    Z_q_tiles: ZCT matrix, same shape and sharding
    mesh: 2D mesh with axes ('x', 'y')
    J: Number of blocks
    b: Block size

Returns:
    zeta_q_tiles: Solution, same shape and sharding

<a id="isdf.common.gpu_utils"></a>

# isdf.common.gpu\_utils

<a id="isdf.common.gpu_utils.get_gpu_memory_nvidia_smi"></a>

#### get\_gpu\_memory\_nvidia\_smi

```python
def get_gpu_memory_nvidia_smi() -> float | None
```

Query GPU memory via nvidia-smi.

Returns:
    Total GPU memory in GB, or None if nvidia-smi unavailable.

<a id="isdf.common.gpu_utils.get_cpu_memory_total"></a>

#### get\_cpu\_memory\_total

```python
def get_cpu_memory_total() -> float | None
```

Query total system memory.

Returns:
    Total system memory in GB, or None if unavailable.

<a id="isdf.common.gpu_utils.get_device_memory_gb"></a>

#### get\_device\_memory\_gb

```python
def get_device_memory_gb(n_devices: int | None = None) -> float
```

Get available memory per device for JAX computations.

Detection strategy:
1. If JAX backend is 'gpu' or 'cuda': use nvidia-smi with 50% factor
   (CUDA driver, XLA runtime, and fragmentation consume ~50%)
2. If JAX backend is 'cpu': use system RAM / n_devices with 80% factor
3. Fallback: return conservative 4 GB default

Args:
    n_devices: Number of devices to divide memory among (for CPU).
               If None, auto-detects via jax.device_count().

Returns:
    Memory per device in GB (usable for computation)

<a id="isdf.common.gpu_utils.get_device_memory_info"></a>

#### get\_device\_memory\_info

```python
def get_device_memory_info() -> dict
```

Get detailed memory information for current JAX backend.

Returns:
    Dictionary with:
    - backend: 'gpu' or 'cpu'
    - total_gb: Total memory per device in GB
    - source: How memory was detected ('nvidia-smi', 'psutil', '/proc/meminfo', 'default')
    - n_devices: Number of JAX devices

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

<a id="isdf.common.symmetry_maps.SymMaps.get_cnk_fullzone_batch"></a>

#### get\_cnk\_fullzone\_batch

```python
def get_cnk_fullzone_batch(wfn, band_indices, nk)
```

Apply symmetry operations to multiple bands at once (vectorized).

Args:
    wfn: WFNReader instance
    band_indices: array-like of band indices
    nk: index of k-point in unfolded grid

Returns:
    np.ndarray: Rotated coefficients of shape (nb, 2, ngk)

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

<a id="isdf.common.jax_profile"></a>

# isdf.common.jax\_profile

<a id="isdf.common.jax_profile.trace_section"></a>

#### trace\_section

```python
@contextmanager
def trace_section(section: str) -> Iterator[None]
```

Start a JAX profiler trace when ISDF_JAX_PROFILE_DIR is set.

<a id="isdf.common.jax_profile.step_annotation"></a>

#### step\_annotation

```python
@contextmanager
def step_annotation(name: str,
                    *,
                    step_num: int | None = None,
                    detail: str | None = None) -> Iterator[None]
```

Annotate host-side regions so they show up inside a profiler trace.

<a id="isdf.common.jax_profile.annotation"></a>

#### annotation

```python
@contextmanager
def annotation(name: str) -> Iterator[None]
```

Light-weight annotation that does not bump the step counter.

<a id="isdf.common.load_wfns"></a>

# isdf.common.load\_wfns

<a id="isdf.common.load_wfns.compute_block_size_for_2d_cholesky"></a>

#### compute\_block\_size\_for\_2d\_cholesky

```python
def compute_block_size_for_2d_cholesky(n_rmu: int, Pr: int,
                                       Pc: int) -> tuple[int, int]
```

Compute block size for 2D blocked Cholesky that satisfies distribution constraints.

Constraints (fundamental to 2D blocked algorithms):
    - n_rmu % block_size == 0  (matrix divides into whole tiles)
    - J % Pr == 0              (tile rows distribute evenly on X-axis)
    - J % Pc == 0              (tile cols distribute evenly on Y-axis)

Where J = n_rmu / block_size is the number of tiles per dimension.

The simplest solution: J = lcm(Pr, Pc), giving block_size = n_rmu / J.
If n_rmu doesn't divide evenly, we try multiples of lcm(Pr, Pc).

Args:
    n_rmu: Matrix dimension (number of ISDF centroids)
    Pr: Number of devices on X-axis
    Pc: Number of devices on Y-axis

Returns:
    (block_size, J) tuple

Raises:
    ValueError: If no valid block size exists (n_rmu incompatible with mesh)

<a id="isdf.common.load_wfns.make_sharded_ifftn_3d"></a>

#### make\_sharded\_ifftn\_3d

```python
def make_sharded_ifftn_3d(mesh: Mesh,
                          in_spec: P,
                          out_spec: P,
                          *,
                          norm: str | None = None)
```

Uses shard_map to run FFT independently on each device's local data.
The FFT axes (last 3) must NOT be sharded - only batch dims can be sharded.
Args:
    mesh: The device mesh
    in_spec: PartitionSpec for input (e.g., P(None, ('x','y'), None, None, None, None))
    out_spec: PartitionSpec for output (same as in_spec for FFT)

Returns:
    A function that performs 3D IFFT on sharded data

<a id="isdf.common.load_wfns.make_sharded_fftn_3d"></a>

#### make\_sharded\_fftn\_3d

```python
def make_sharded_fftn_3d(mesh: Mesh,
                         in_spec: P,
                         out_spec: P,
                         *,
                         norm: str | None = None)
```

shard_map local FFT (forward).

This is the forward-FFT counterpart to make_sharded_ifftn_3d.

<a id="isdf.common.load_wfns.load_kpoint_fftbox"></a>

#### load\_kpoint\_fftbox

```python
def load_kpoint_fftbox(wfn, sym, meta, k_idx, nb)
```

Load a single k-point's wavefunction into the FFT box on GPU.

Returns jax array of shape (nb, nspinor, nx, ny, nz), ~0.55 GiB for 12x12.

<a id="isdf.common.load_wfns.get_enk_bandrange"></a>

#### get\_enk\_bandrange

```python
def get_enk_bandrange(wfn, sym, bandrange, sigma_bandrange, nspinor=2)
```

Return band energies and per-band weights for a given band window.

Args:
	wfn: WFNReader providing energies and Fermi level
	sym: SymMaps with mappings between irreducible and full k sets
	bandrange: tuple[int,int] inclusive-exclusive (start, end) bands to extract
	sigma_bandrange: tuple[int,int] band window used to compute weighting
	nspinor: Number of spinor components (2 for Pauli, 4 for bispinor)

Returns:
	enk: jax.Array of shape (nk_full, nb)
	weights: jax.Array of shape (nk_full, nb * nspinor) with simple val/cond weights

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
Uses function caching to avoid JIT recompilation on repeated calls.

<a id="isdf.common.load_wfns.compute_pair_density_spin_traced"></a>

#### compute\_pair\_density\_spin\_traced

```python
def compute_pair_density_spin_traced(psi_rmuT_X: jax.Array,
                                     psi_rmu_Y: jax.Array,
                                     mesh_xy: Mesh) -> jax.Array
```

Compute spin-traced pair density P_k(μ,ν) = Σ_{n,s} ψ*_{n,k,s}(μ) ψ_{n,k,s}(ν).

This matches gw_jax spin treatment for ISDF fitting.

Input shapes and shardings:
	psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
		- conj(psi_nk,s(r_mu)) with mu sharded on X
	psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
		- psi_nk,s(r_nu) with nu sharded on Y

Output:
	P_k: (nk, n_rmu, n_rmu) with P(None, 'x', 'y')
		- P[k, mu, nu] = Σ_{n,s} psi*_nk,s(r_mu) * psi_nk,s(r_nu)

<a id="isdf.common.load_wfns.compute_CCT_from_left_right"></a>

#### compute\_CCT\_from\_left\_right

```python
def compute_CCT_from_left_right(P_l_k: jax.Array, P_r_k: jax.Array,
                                kgrid: tuple[int, int,
                                             int], mesh_xy: Mesh) -> jax.Array
```

Compute CCT from separate left and right spin-traced pair densities.

C_q(μ,ν) = FFT[ conj(IFFT(P_l)) ⊙ IFFT(P_r) ]

This matches gw_jax physics where left and right have different band ranges.

Args:
	P_l_k: (nk, n_rmu, n_rmu) left pair density, P(None, 'x', 'y')
	P_r_k: (nk, n_rmu, n_rmu) right pair density, P(None, 'x', 'y')
	kgrid: (nkx, nky, nkz)
	mesh_xy: Device mesh

Returns:
	C_q: (nqx, nqy, nqz, n_rmu, n_rmu) with P(None, None, None, 'x', 'y')

<a id="isdf.common.load_wfns.compute_ZCT_from_left_right_zchunk"></a>

#### compute\_ZCT\_from\_left\_right\_zchunk

```python
def compute_ZCT_from_left_right_zchunk(P_l_k_muz: jax.Array,
                                       P_r_k_muz: jax.Array,
                                       kgrid: tuple[int, int, int],
                                       mesh_xy: Mesh) -> jax.Array
```

Compute ZCT from left and right pair densities, both at (μ, z-chunk).

Z_q(μ,r) = FFT[ conj(IFFT(P_l(μ,r))) ⊙ IFFT(P_r(μ,r)) ]

Args:
	P_l_k_muz: (nk, n_rmu, n_zchunk) left pair density at z-chunk, P(None, 'x', 'y')
	P_r_k_muz: (nk, n_rmu, n_zchunk) right pair density at z-chunk, P(None, 'x', 'y')
	kgrid: (nkx, nky, nkz)
	mesh_xy: Device mesh

Returns:
	Z_q: (nqx, nqy, nqz, n_rmu, n_zchunk) with P(None, None, None, 'x', 'y')

<a id="isdf.common.load_wfns.compute_L_q_from_CCT"></a>

#### compute\_L\_q\_from\_CCT

```python
def compute_L_q_from_CCT(C_q: jax.Array,
                         mesh_xy: Mesh,
                         block_size: int = None) -> jax.Array
```

Compute Cholesky factor L_q from CCT matrix using 2D blocked algorithm.

Args:
    C_q: (nq, n_rmu, n_rmu) CCT matrix, sharded P(None, 'x', 'y')
    mesh_xy: 2D device mesh
    block_size: Tile block size (auto if None)

Returns:
    L_q: (nq, n_rmu, n_rmu) Cholesky factor, sharded P(None, 'x', 'y')

<a id="isdf.common.load_wfns.solve_zeta_from_L_q"></a>

#### solve\_zeta\_from\_L\_q

```python
def solve_zeta_from_L_q(L_q: jax.Array,
                        Z_q: jax.Array,
                        mesh_xy: Mesh,
                        q_chunk_size: int = 1) -> jax.Array
```

Solve for zeta_q given pre-computed Cholesky factor L_q.

Uses q-chunked all-gather strategy: gather B_q L matrices at a time,
then solve all B_q systems in parallel using vmap.

Memory trade-off:
- q_chunk_size=1: Minimum memory (one L replicated at a time)
- q_chunk_size=nq: Maximum parallelism (all L replicated)

Args:
    L_q: (nq, n_rmu, n_rmu) Cholesky factor, sharded P(None, 'x', 'y')
    Z_q: (nq, n_rmu, n_zchunk) ZCT matrix, sharded P(None, 'x', 'y')
    mesh_xy: 2D device mesh
    q_chunk_size: Number of q-points to solve simultaneously (default 1)

Returns:
    zeta_q: (nq, n_rmu, n_zchunk) solution, sharded P(None, None, ('x','y'))

<a id="isdf.common.load_wfns.fit_zeta_chunked_to_h5"></a>

#### fit\_zeta\_chunked\_to\_h5

```python
def fit_zeta_chunked_to_h5(wfn,
                           sym,
                           meta: Meta,
                           centroid_indices: jax.Array,
                           mesh_xy: Mesh,
                           chunk_r: int,
                           output_file: str,
                           band_chunk_size: int = 16,
                           q_chunk_size: int = 1,
                           bispinor: bool = True,
                           use_gspace_cache: bool = True,
                           band_range_left: tuple[int, int] | None = None,
                           band_range_right: tuple[int, int] | None = None)
```

Full zeta fitting pipeline with r-chunk loop and HDF5 output.

Workflow:
1. Load wavefunctions (band-chunked FFT) for max range
2. Slice to get left (0:b3) and right (0:b4) views
3. Compute C_q from spin-traced P_l × P_r via ortho FFT
4. Compute L_q = chol(C_q) using 2D blocked algorithm
5. For each r-chunk:
   a. Compute psi_nk,a(r_chunk) via FFT
   b. Compute spin-traced P_l and P_r at r-chunk
   c. Compute Z_q via ortho FFT with left/right cross-product
   d. Solve zeta_q = L^{-H}(L^{-1} Z_q) (q-chunked)
   e. Write zeta_q chunk to HDF5

Args:
    wfn: WFNReader object
    sym: SymMaps object
    meta: Meta object with system info
    centroid_indices: ISDF centroid indices
    mesh_xy: 2D device mesh
    chunk_r: Number of flattened r-points per chunk
    output_file: Path to output HDF5 file
    band_chunk_size: Bands to process at once when FFTing wavefunctions (with global r)
    q_chunk_size: Q-points to solve C_q @ zeta_q = Z_q simultaneously
    bispinor: Whether to use bispinor wavefunctions
    use_gspace_cache: If True, cache G-space across r-chunks
    band_range_left: (start, end) for left wfns. Default: (b0, b3)
    band_range_right: (start, end) for right wfns. Default: (b0, b4)

Returns:
    psi_l_rmu_Y: Left centroid wfns (nk, nb_l, ns, n_rmu), Y-sharded
    psi_l_rmuT_X: Left conjugated wfns (nk, n_rmu, nb_l, ns), X-sharded
    psi_r_rmu_Y: Right centroid wfns (nk, nb_r, ns, n_rmu), Y-sharded
    psi_r_rmuT_X: Right conjugated wfns (nk, n_rmu, nb_r, ns), X-sharded

<a id="isdf.common.load_wfns.load_gspace_for_bands"></a>

#### load\_gspace\_for\_bands

```python
def load_gspace_for_bands(wfn,
                          sym,
                          meta,
                          mesh_xy,
                          band_range,
                          bispinor,
                          band_chunk_size: int = 16
                          ) -> list[tuple[jax.Array, tuple[int, int]]]
```

Load G-space wavefunctions for all band chunks ONCE.

This caches the expensive HDF5 read + scatter operation so it can be
reused across multiple z-chunk iterations. Memory cost is ~0.5-1 GB
for typical systems (nk * nb * ns * fft_grid * 16 bytes).

Args:
    wfn: WFNReader
    sym: SymMaps
    meta: Meta object
    mesh_xy: Device mesh
    band_range: (b_start, b_end) - total bands needed
    bispinor: Whether to use bispinor
    band_chunk_size: Bands to process at once

Returns:
    List of (global_psi_Gtot, bc_range) for each band chunk

<a id="isdf.common.load_wfns.get_sharded_wfns_rchunk_slice"></a>

#### get\_sharded\_wfns\_rchunk\_slice

```python
def get_sharded_wfns_rchunk_slice(global_psi_Gtot: jax.Array, meta: Meta,
                                  r_start: int, r_end: int,
                                  kvecs_frac: np.ndarray, mesh_xy: Mesh,
                                  band_range: tuple[int, int]) -> jax.Array
```

FFT wavefunctions and extract r-chunk via flattened r-index slicing.

R-chunking gives CONTIGUOUS r-indices: slicing r in [r_start, r_end)
produces a contiguous block in the flattened xyz order and can be written
to HDF5 in a single sequential operation.

Args:
    global_psi_Gtot: G-space wfns from read_Gvecs_to_devices
    meta: Meta object
    r_start, r_end: R-index range [r_start, r_end)
    kvecs_frac: (nk, 3) k-vectors in fractional coordinates
    mesh_xy: Device mesh
    band_range: (b_start, b_end)

Returns:
    psi_rchunk_Y: (nk, nb, ns, n_rchunk) with P(None, None, None, 'y')

<a id="isdf.common.load_wfns.get_psi_rchunk_from_cached"></a>

#### get\_psi\_rchunk\_from\_cached

```python
def get_psi_rchunk_from_cached(cached_gspace: list[tuple[jax.Array,
                                                         tuple[int, int]]],
                               meta,
                               mesh_xy,
                               band_range,
                               r_start,
                               r_end,
                               kvecs_frac,
                               band_chunk_size: int = 16) -> jax.Array
```

Extract r-chunk from pre-loaded G-space (FFT only, no HDF5 read).

This is the fast path that reuses cached G-space across r-chunk iterations.
R-chunks are contiguous in r-space, enabling efficient HDF5 writes.

Args:
    cached_gspace: Pre-loaded G-space from load_gspace_for_bands()
    meta: Meta object
    mesh_xy: Device mesh
    band_range: (b_start, b_end) - total bands needed
    r_start, r_end: R-index range
    kvecs_frac: (nk, 3) k-vectors in fractional coordinates
    band_chunk_size: Bands to FFT at once

Returns:
    psi_rchunk_Y: (nk, nb, ns, n_rchunk) with P(None, None, None, 'y')

<a id="isdf.common.load_wfns.get_psi_rchunk"></a>

#### get\_psi\_rchunk

```python
def get_psi_rchunk(wfn,
                   sym,
                   meta,
                   mesh_xy,
                   band_range,
                   r_start,
                   r_end,
                   bispinor,
                   band_chunk_size: int = 16) -> jax.Array
```

Load and FFT wavefunctions for a specific r-chunk.

NOTE: This function reloads G-space from HDF5 each call. For multiple
r-chunks, use load_gspace_for_bands() + get_psi_rchunk_from_cached()
to avoid redundant HDF5 reads.

Uses band chunking to limit memory during FFT step:
- Loop over band chunks
- FFT each chunk to real-space (the memory bottleneck)
- Extract r-slice and accumulate into output array

The final psi_rchunk has all bands but only the r-slice, which is
small enough to hold in memory for downstream pair density computation.

Args:
    wfn: WFNReader
    sym: SymMaps
    meta: Meta object
    mesh_xy: Device mesh
    band_range: (b_start, b_end) - total bands needed
    r_start, r_end: R-index range
    bispinor: Whether to use bispinor
    band_chunk_size: Bands to FFT at once (memory control for FFT step)

Returns:
    psi_rchunk_Y: (nk, nb, ns, n_rchunk) with P(None, None, None, 'y')

<a id="isdf.common.load_wfns.get_sharded_wfns_centroids"></a>

#### get\_sharded\_wfns\_centroids

```python
def get_sharded_wfns_centroids(
        global_psi_Gtot: jax.Array, meta: Meta, centroid_indices: jax.Array,
        kvecs_frac: np.ndarray, mesh_xy: Mesh,
        band_range: tuple[int, int]) -> tuple[jax.Array, jax.Array]
```

FFT wavefunctions and extract centroids for a single band chunk.

This is the centroid-extraction counterpart to get_sharded_wfns_rchunk_slice.
Both use the same caching and staging patterns for memory efficiency.

Args:
    global_psi_Gtot: G-space wfns from read_Gvecs_to_devices
    meta: Meta object
    centroid_indices: (n_rmu, 3) centroid grid coordinates
    kvecs_frac: (nk, 3) k-vectors in fractional coordinates
    mesh_xy: Device mesh
    band_range: (b_start, b_end)

Returns:
    psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
    psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)

<a id="isdf.common.load_wfns.load_centroids_band_chunked"></a>

#### load\_centroids\_band\_chunked

```python
def load_centroids_band_chunked(
        wfn,
        sym,
        meta: Meta,
        centroid_indices: jax.Array,
        bispinor: bool,
        mesh_xy: Mesh,
        band_range: tuple[int, int],
        band_chunk_size: int = 64) -> tuple[jax.Array, jax.Array]
```

Load centroid-sampled wavefunctions using band chunking.

Memory-safe version that loops over band chunks to avoid OOM
when loading all bands at once for FFT.

This is the unified band-chunked backend used by fit_zeta_chunked_to_h5.

Args:
    wfn: WFNReader
    sym: SymMaps
    meta: Meta object
    centroid_indices: (n_rmu, 3) centroid grid coordinates
    bispinor: Whether to use bispinor
    mesh_xy: Device mesh
    band_range: (b_start, b_end)
    band_chunk_size: Bands to FFT at once (memory control)

Returns:
    psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
    psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)

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
    np.ndarray: Complex coefficients array of shape (2, ngk[ik]) for both spinor components

<a id="isdf.common.wfnreader.WFNReader.get_cnk_batch"></a>

#### get\_cnk\_batch

```python
def get_cnk_batch(ik, band_indices)
```

Get complex coefficients for multiple bands at once (vectorized).

Args:
    ik (int): k-point index
    band_indices: array-like of band indices

Returns:
    np.ndarray: Complex coefficients of shape (nb, 2, ngk[ik])

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

<a id="isdf.isdf_init.kmeans_isdf"></a>

# isdf.isdf\_init.kmeans\_isdf

Weighted k-means clustering for ISDF sampling point selection.

This module implements density-weighted k-means with periodic boundary conditions (PBC).
The clustering uses the minimal image convention for distances, which is critical for
crystalline systems.

PBC Distance Calculation (Minimal Image Convention)
====================================================
For two points with fractional coordinates r1 and r2:

1. Compute fractional displacement: df = r1 - r2
2. Apply minimal image: df_wrapped = df - round(df)
   - This wraps each component to [-0.5, 0.5)
   - Equivalent to finding the closest image among all 27 neighboring cells
3. Compute Cartesian distance: d = |df_wrapped @ avec|
   - Or equivalently using metric tensor: d² = df_wrapped @ G @ df_wrapped
   - Where G = avec.T @ avec is the metric tensor

Why round() gives the minimum over 27 cells:
- Each fractional component df_i can be shifted by any integer n_i
- The closest image has n_i = round(df_i), giving df_i - round(df_i) ∈ [-0.5, 0.5)
- This is the unique image in the first Brillouin zone (Wigner-Seitz cell in reciprocal space)

For non-orthogonal cells, this approximation works well when the cell is "not too skewed".
For highly skewed cells, one should check neighboring images explicitly.

<a id="isdf.isdf_init.kmeans_isdf.precompute_metric_tensor"></a>

#### precompute\_metric\_tensor

```python
def precompute_metric_tensor(avec: np.ndarray) -> np.ndarray
```

Precompute the metric tensor G = A^T @ A for PBC distance calculations.

The metric tensor allows computing squared Cartesian distances directly
from fractional displacements without explicit coordinate conversion:

    d² = df @ G @ df^T

where df is the (wrapped) fractional displacement vector.

This is equivalent to d² = |df @ avec|² but avoids materializing the
Cartesian displacement vector, saving memory for large arrays.

Args:
    avec: (3, 3) lattice vectors, rows are a1, a2, a3 in Cartesian coords

Returns:
    G: (3, 3) metric tensor

<a id="isdf.isdf_init.kmeans_isdf.pbc_distance_sq_batch"></a>

#### pbc\_distance\_sq\_batch

```python
@jax.jit
def pbc_distance_sq_batch(positions_frac: jnp.ndarray,
                          centroids_frac: jnp.ndarray,
                          metric_tensor: jnp.ndarray) -> jnp.ndarray
```

Compute squared PBC distances between all positions and all centroids.

Uses the minimal image convention: for each pair, finds the minimum distance
over all periodic images (equivalent to checking 27 neighboring cells).

Implementation:
    1. Compute fractional displacement: df = pos - cent
    2. Wrap to [-0.5, 0.5): df = df - round(df)  [minimal image]
    3. Compute squared distance: d² = df @ G @ df^T

Args:
    positions_frac: (P, 3) fractional coordinates of grid points
    centroids_frac: (K, 3) fractional coordinates of centroids
    metric_tensor: (3, 3) G = avec^T @ avec

Returns:
    distances_sq: (P, K) squared Cartesian distances with PBC

<a id="isdf.isdf_init.kmeans_isdf.pbc_distance_sq_single"></a>

#### pbc\_distance\_sq\_single

```python
@jax.jit
def pbc_distance_sq_single(positions_frac: jnp.ndarray,
                           centroid_frac: jnp.ndarray,
                           metric_tensor: jnp.ndarray) -> jnp.ndarray
```

Compute squared PBC distances from all points to a single centroid.

Optimized for k-means++ initialization where we add one centroid at a time.

Args:
    positions_frac: (P, 3) fractional coordinates of grid points
    centroid_frac: (3,) fractional coordinates of single centroid
    metric_tensor: (3, 3) G = avec^T @ avec

Returns:
    distances_sq: (P,) squared distances to the centroid

<a id="isdf.isdf_init.kmeans_isdf.kmeans_update_step"></a>

#### kmeans\_update\_step

```python
@partial(jax.jit, static_argnames=['n_k'])
def kmeans_update_step(positions_frac: jnp.ndarray,
                       centroids_frac: jnp.ndarray, rho_flat: jnp.ndarray,
                       metric_tensor: jnp.ndarray, n_k: int) -> tuple
```

Single k-means update step: assign labels and compute new centroids.

This is the core k-means iteration, JIT-compiled for speed:
    1. Compute all pairwise PBC distances
    2. Assign each point to nearest centroid
    3. Compute weighted mean of assigned points (in wrapped coordinates)
    4. Update centroid positions

The weighted mean is computed in *displacement space* relative to the current
centroid, using PBC-wrapped displacements. This ensures the centroid moves
towards its assigned points correctly even when points wrap around boundaries.

Args:
    positions_frac: (P, 3) fractional coordinates of grid points
    centroids_frac: (K, 3) fractional coordinates of centroids
    rho_flat: (P,) charge density weights
    metric_tensor: (3, 3) G = avec^T @ avec
    n_k: Number of clusters (must match centroids_frac.shape[0])

Returns:
    new_centroids_frac: (K, 3) updated centroid positions
    movement_sq: (K,) squared movement of each centroid (for convergence check)
    labels: (P,) cluster assignment for each point

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
def weighted_kmeans_jax(avec: jnp.ndarray,
                        rho_jax: jnp.ndarray,
                        N_k: int = 10,
                        max_steps: int = 200,
                        tolerance: float = 5e-3,
                        seed: int = 0) -> tuple
```

Density-weighted k-means clustering with periodic boundary conditions.

Uses k-means++ initialization for better initial centroid placement,
then iterates Lloyd's algorithm until convergence.

Key features:
- Minimal image convention for PBC (considers all 27 neighboring cells)
- Metric tensor for efficient squared distance computation
- JIT-compiled inner loop for speed
- Weighted by charge density (centroids concentrate in high-density regions)

Args:
    avec: (3, 3) lattice vectors (rows are a1, a2, a3 in Cartesian coords)
    rho_jax: (Nx, Ny, Nz) charge density on real-space grid
    N_k: Number of cluster centroids (ISDF sampling points)
    max_steps: Maximum k-means iterations
    tolerance: Convergence tolerance for centroid movement (Angstroms)
    seed: Random seed for reproducibility

Returns:
    labels: (P,) cluster assignment for each grid point
    centroids: (N_k, 3) final centroid positions in fractional coordinates
    history: (N_k, max_steps) z-coordinate history (for debugging)
    steps_taken: Number of iterations until convergence

<a id="isdf.isdf_init.kmeans_isdf.snap_centroids_to_grid"></a>

#### snap\_centroids\_to\_grid

```python
def snap_centroids_to_grid(
        centroids_frac: np.ndarray,
        fft_grid: np.ndarray,
        deduplicate: bool = True) -> tuple[np.ndarray, np.ndarray, int]
```

Snap fractional centroids to the nearest FFT grid points and optionally deduplicate.

When N_k is large relative to the FFT grid, multiple k-means centroids may map 
to the same grid point. This function:
1. Converts fractional coords to integer grid indices
2. Handles periodic wrapping
3. Removes duplicate grid points (if deduplicate=True)

Args:
    centroids_frac: (N_k, 3) fractional coordinates in [0, 1)
    fft_grid: (3,) FFT grid dimensions [Nx, Ny, Nz]
    deduplicate: If True, remove duplicate grid points

Returns:
    centroid_indices: (N_unique, 3) integer grid indices
    centroids_frac_snapped: (N_unique, 3) fractional coords of snapped centroids
    n_duplicates: Number of duplicate centroids that were removed

<a id="isdf.isdf_init.kmeans_isdf.ensure_unique_centroids"></a>

#### ensure\_unique\_centroids

```python
def ensure_unique_centroids(centroids_frac: np.ndarray,
                            fft_grid: np.ndarray,
                            rho: np.ndarray = None,
                            metric_tensor: np.ndarray = None) -> np.ndarray
```

Ensure all centroids map to unique FFT grid points.

If duplicates are found, attempts to redistribute them to nearby 
unoccupied grid points (weighted by density if provided).

Args:
    centroids_frac: (N_k, 3) fractional coordinates
    fft_grid: (3,) FFT grid dimensions
    rho: Optional (Nx, Ny, Nz) charge density for weighted redistribution
    metric_tensor: Optional (3,3) for PBC distance calculation

Returns:
    centroids_frac_unique: (N_k, 3) fractional coordinates with no duplicates

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

