# Dipole utility

`psp.get_dipole_mtxels` The velocity matrix $\langle m\bk|\hat v_a|n\bk\rangle$ and the Berry connection from parallel-transport links.
{: .page-summary }

<div class="workflow-map" data-driver="dipole" data-page="files"></div>

Dipole matrix elements fix the $\bq \to 0$ head of $W$, the one element of the screened interaction that a finite k grid cannot sample directly. This utility computes them from the commutator of the Hamiltonian with position, analytically and with the nonlocal pseudopotential included, so that the full anisotropic head tensor comes out right. BerkeleyGW instead evaluates the head at one small but finite $\bq$, a finite-difference approach that samples a single direction.
{: .lead }

The utility writes the Cartesian velocity matrix elements $\langle m\bk|\hat v_a|n\bk\rangle$ to `dipole.h5` for every full-BZ $\bk$, and by default the link matrices and Berry connection between neighbouring k points to `parallel_transport.h5`. GW reads `dipole.h5` for the head, the BSE for its optical dipoles, and the self-consistent QSGW head reads both.

Optical consumers read their selected conduction–valence window directly onto
the distributed band matrix. A QP-rotated BSE reads the full active square,
since mixing occupied and empty states also needs same-class velocity entries.
The file format and its band-coverage and WFN-basis checks are unchanged.

<div class="glance" markdown>

Invoke
:   `python -m psp.get_dipole_mtxels -i gw.in`

Reads
:   input file (`-i`), `WFN.h5`, `*.upf`

Writes
:   `dipole.h5`, `parallel_transport.h5`, `dipole.out`

Operator
:   $\hat v_a = \partial H_\bk/\partial k_a = \hat p_a - i[\hat r_a, V^{\mathrm{NL}}]$ · `--skip-vnl`: $\hat p_a$ alone

</div>

## Dipoles from the commutator {: #commutator }

Position is not a well-defined operator in a periodic solid, but its interband matrix elements are, and they follow from the velocity. With $\hbar = 1$ the velocity is the commutator of the Hamiltonian with position, $\hat{\mathbf v} = i[H,\hat{\mathbf r}]$. Between Bloch states the position operator acts as a k derivative plus the Berry connection,

$$
\langle\psi_{m\bk}|\hat{\mathbf r}|\psi_{n\bk'}\rangle = \big[i\nabla_\bk\,\delta_{mn} + \mathbf A_{mn}(\bk)\big]\,\delta(\bk-\bk'), \qquad \mathbf A_{mn}(\bk) = i\langle u_{m\bk}|\nabla_\bk u_{n\bk}\rangle,
$$

so the commutator of $\hat{\mathbf r}$ with any operator $O$ that conserves crystal momentum has two terms, the k derivative of its matrix elements and its commutator with the connection,

$$
[\hat{\mathbf r},O]_{mn\bk} = i\,\partial_\bk O_{mn} + [\mathbf A_\bk,O]_{mn}.
$$

For the DFT Hamiltonian, which is diagonal with eigenvalues $E_{n\bk}$, this gives $\langle m\bk|\hat{\mathbf v}|n\bk\rangle = i(E_{m\bk}-E_{n\bk})\,\mathbf A_{mn}$ for $m \neq n$, so the velocity determines the interband position matrix elements $\mathbf r_{mn} = \mathbf A_{mn}$.

The head needs the commutator of the full Hamiltonian, nonlocal parts included. For the DFT part the utility evaluates the velocity analytically as $\partial H_\bk/\partial\bk$, including the commutator of the nonlocal pseudopotential with $\hat{\mathbf r}$. On a self-consistent QSGW run the Hamiltonian also contains the correction $\Delta H$, which holds $\Sigma$ and is known only as a matrix in the band basis on the k grid. LORRAX therefore takes its commutator from the second form above, with the Berry connection that this utility builds from parallel-transport links. The sign convention, the links and their validation are derived on [Velocity operator](../theory/velocity-operator.md).

## What it computes {: #what }

<figure class="flowchart" markdown>
[![One-body Hamiltonian, velocity and Berry connection](../assets/figures/p1.png)](../assets/figures/p1.png)
<figcaption><strong>Fig. P1.</strong> The three one-body operators built from the lattice, the atomic positions and the pseudopotentials. This utility runs stages 2 and 3: it differentiates $T_\bk$ and $V^{\mathrm{NL}}_\bk$ with respect to $\bk$ and contracts them with the wavefunctions into $\langle m\bk|\hat v_a|n\bk\rangle$, then overlaps neighbouring k points, takes the polar factor of each overlap as the link matrix and forms the Berry connection from the links. Stage 1 belongs to the <a href="../kin-ion/">kin-ion utility</a>.</figcaption>
</figure>

The velocity is the $\bk$ derivative of the Bloch Hamiltonian: $2(\bk+\bG)_a$ from the kinetic energy plus the analytic derivative of the nonlocal projectors. $V^{\mathrm{loc}}$ commutes with $\hat{\mathbf r}$ and drops out. One band-sharded sweep covers the three Cartesian components on the wedge of k points stored in `WFN.h5`; the result is unfolded to the full BZ as a polar, time-odd vector. No density enters.

The link stage overlaps the cell-periodic states at each $\bk$ with those at $\bk+\mathbf b$ for every step $\mathbf b$ of a Marzari–Vanderbilt shell, keeps the polar factor of each overlap as the link $L_{\mathbf b}(\bk)$, and differences the links into the Berry connection $A_\bk$. It then checks the links on the identity $v = D_\bk\,\mathrm{diag}(E)$ and stamps the relative error $\epsilon_{\mathrm{link}}$. The operator, its sign convention, the links, the stencil and the validation are derived on [Velocity operator](../theory/velocity-operator.md).

!!! note
    GW, the QSGW head and `bse.exciton_bands` open `dipole.h5` under that name in the input file directory; no input key names it. `--out` is resolved against the working directory, so run the utility from the input file directory or keep the default name there.

## Running {: #run }

=== "Local"
    ```bash
    uv run python -m psp.get_dipole_mtxels -i gw.in
    ```
=== "Perlmutter"
    ```bash
    srun -n 4 --gpus-per-task=1 --gpu-bind=closest python -m psp.get_dipole_mtxels -i gw.in
    ```
=== "Frontera"
    ```bash
    $LORRAX_PY -u -m psp.get_dipole_mtxels -i gw.in
    ```

Use the GW input file itself, so that the band extent and spinor model agree with the run that reads the file. The input file lines it reads:

```ini
wfn_file           = WFN.h5
sys_dim            = 3
nval               = 4
ncond              = 4
number_bands       = 200
```

The flags decide which artifacts a run writes:

| Run | `dipole.h5` | `parallel_transport.h5` |
|---|---|---|
| default | written | written beside `--out`; dropped, with one report line, on a k grid with a two-point axis or when its link stage refuses |
| `--parallel-transport-out FILE` | written | written to `FILE`; refuses instead of being dropped |
| `--no-parallel-transport`, `--skip-vnl` | written | not written |

## Keywords {: #keywords }

Defaults are read from the argument parser (`src/psp/get_dipole_mtxels.py`) and the keyword schema (`src/gw/gw_config.py`). Every physical setting comes from the GW input file. Unlike kin-ion, the file covers every band of the $\chi^0$ sum, because the head and the BSE need dipoles between all of them.

Required
{: .kw-group }

**Command line**

### `-i`, `--input` {: .kw data-type="path" data-default="required" }
The GW input file.

**Input keys**

The input file passes through the GW reader, so an unknown key or a missing `sys_dim` refuses.

### `sys_dim` {: .kw data-type="int" data-default="required" }
`3` bulk, `2` slab, `0` molecule. Only validated here; the velocity carries no Coulomb truncation.

`0` `2` `3`
{: .kw-values }

### `nval` {: .kw data-type="int" data-default="required" }
Valence bands of the quasiparticle window. Required unless `number_bands_protected` is given. Stamped (`prov_nval`).

### `ncond` {: .kw data-type="int" data-default="required" }
Number of protected conduction bands, the conduction bands whose quasiparticle energies GW corrects. Required unless `number_bands_protected` is given. Stamped (`prov_ncond`); the file holds at least $n_\mathrm{occ} + \texttt{ncond}$ bands.

### `number_bands_protected` {: .kw data-type="int" data-default="instead of nval, ncond" }
Alternative request for the same window: every occupied band plus protected conduction bands up to this total.

Resolved exactly as in GW and [kin-ion](kin-ion.md#number_bands_protected), giving `nval` = $n_\mathrm{occ}$ and `ncond` = total − $n_\mathrm{occ}$ protected conduction bands. The stamps record the resolved values, so a `dipole.h5` built from a GW input file authenticates against the run of that input file. Giving it beside `nval` or `ncond` refuses (`GATE band_request_forms`).
{: .kw-details }

### `number_bands` {: .kw data-type="int" data-default="required" }
Top of the $\chi^0$ and $\Sigma$ band sums. Alias `nband`.

The matrix holds $n_b = \min\big(N_{\mathrm{WFN}}, \max(n_\mathrm{occ} + \texttt{ncond}, \texttt{number\_bands})\big)$ bands, so the head and the BSE see every band of the $\chi^0$ sum. The resolved value is the larger of `number_bands_chi` and `number_bands_sigma`.
{: .kw-details }

Optional
{: .kw-group }

**Command line**

### `--out` {: .kw data-type="path" data-default="dipole.h5" }
Output file, relative to the working directory.

### `--report-file` {: .kw data-type="path" data-default="dipole.out" }
Run report; by default `dipole.out` beside `--out`.

### `--pseudo-dir`, `--pseudo_dir` {: .kw data-type="path" data-default="input file directory" }
Directory scanned for `*.upf`.

Without the flag, `../qe/scf` and then `../qe/nscf`, relative to the input file, are tried when the input file directory holds none.
{: .kw-details }

### `--skip-vnl` {: .kw data-type="flag" data-default="off" }
Write $\hat p_a = 2(\bk+\bG)_a$ alone, without the nonlocal commutator (BerkeleyGW's `use_momentum`).

For comparison only: the GW head refuses such a file. Needs no pseudopotentials and writes no `parallel_transport.h5`.
{: .kw-details }

### `--parallel-transport-out` {: .kw data-type="path" data-default="parallel_transport.h5 beside --out" }
Link and connection artifact. Naming it makes every failure of the link stage refuse instead of dropping the file.

The default is written as `parallel_transport.h5.partial` and renamed when the links complete.
{: .kw-details }

### `--no-parallel-transport` {: .kw data-type="flag" data-default="off" }
Write no `parallel_transport.h5`.

### `--parallel-transport-validation-rtol` {: .kw data-type="float" data-default="5e-3" }
Relative tolerance of the link check $v = D_\bk\,\mathrm{diag}(E)$.

Above it the run warns that the k grid is coarse for the links and writes the artifact anyway.
{: .kw-details }

**Input keys**

### `wfn_file` {: .kw data-type="string" data-default="WFN.h5" }
Wavefunction file, relative to the input file directory.

### `bispinor` {: .kw data-type="bool" data-default="false" }
Four-component run: the velocity is built on the kinetic-balance four-spinor of `WFN.h5`. Stamped (`prov_bispinor`).

$2(\bk+\bG)$ acts on all four components, $V^{\mathrm{NL}}$ on the two large ones.
{: .kw-details }

### `bispinor_gw` {: .kw data-type="enum" data-default="bare_transverse" }
Four-current screening model; here only checked: `full_static_cohsex` refuses without `bispinor = true`.

`bare_transverse` `full_tangential` `full_static_cohsex`
{: .kw-values }

### `linalg` {: .kw data-type="enum" data-default="local" }
Expert setting; most runs should leave it at `local`. It selects how the link polar factors are solved, `local` one per rank and `distributed` each on the whole mesh.

`local` `distributed`
{: .kw-values }

**Environment**

### `LORRAX_DEBUG_PRINT` {: .kw data-type="env" data-default="unset" }
Prints the $x$-component $\hat p$ and $\hat p + \partial_k V^{\mathrm{NL}}$ blocks between the top six valence and lowest four conduction bands at one k. See [Environment](../using/environment.md#lorrax_debug_print).

## Output format {: #output }

<div class="file-list" markdown>

`dipole.h5`
:   $\langle m\bk|\hat v_a|n\bk\rangle$ in Ry bohr on the full BZ, the band energies, and the provenance stamps. Written collectively through SlabIO from the band shards.

`parallel_transport.h5`
:   The velocity again on the outer band set, the links, the Berry connection, the stencil and the validation stamps. Read by the QSGW head through `parallel_transport_file`.

`dipole.out`
:   Run report: operator, pseudopotentials, bands, timings and the link validation.

</div>

$n_\bk$ is the full-BZ k count, $n_b$ the band extent of `dipole.h5` and $n_b'$ the outer band set of the links, $\min(N_{\mathrm{WFN}},\lceil1.25\,n_b\rceil)$.

### dipole.h5

Generated from `file_io.dipole.write_dipole` and `psp.get_dipole_mtxels.dipole_provenance`. Every root entry below `band_energies` is an HDF5 attribute.

```tree
dipole.h5
  dipole_cart                    complex128[3, nk, nb, nb]
  band_energies                  float64[nk, nb]          E_nk of the star parent, Ry
  @nbands                        int         bands in WFN.h5
  @nk                            int         full-BZ k count
  @skip_vnl                      bool
  @note                          str         operator
  @basis                         str         "dft" or "qp"
  @prov_wfn_sha256               str
  @prov_wfn_fingerprint_scheme   str
  @prov_wfn_file                 str
  @prov_nval                     int
  @prov_ncond                    int
  @prov_nband                    int
  @prov_nb_written               int         nb
  @prov_bispinor                 bool
  @prov_skip_vnl                 bool
  @prov_nspinor                  int
  @prov_soc                      bool
  @prov_q0_operator_scheme       str
```

`dipole_cart[a, k, m, n]` is $\langle m\bk|\hat v_a|n\bk\rangle$. $E_{m\bk} - E_{n\bk}$ is not stored; readers form it from `band_energies`, and the BSE divides by it to put each dipole in position form ([Velocity operator §2](../theory/velocity-operator.md#forms)). `basis` is `"qp"` only when `WFN.h5` carries the QP-WFN stamp. Consumers check the stamps to different depths ([The q → 0 head](../theory/q0-head.md#sources)):

- **GW full head** (`GATE dft_head_dipole_provenance`): refuses a file built for another WFN, band window, spinor model or velocity operator.
- **GW direct and static heads**: refuse another velocity operator (`GATE static_head_dipole_operator`); another WFN, band window or k count only warns, unless `LORRAX_SANITY=strict`.
- **`bse.bse_jax`, `bse.absorption_haydock`, `bse.exciton_bands`**: compare only `basis` against the WFN (`exciton_bands` also the k count); the WFN fingerprint is not read.

The fix in every case is to rerun this utility with the run's input file.

### parallel_transport.h5

Generated from `file_io.parallel_transport.initialize_parallel_transport_artifact`, `_write_link_stage`, `_write_connection_stage`, and `write_velocity_validation`. Every entry is a dataset; the scalars are zero-dimensional. $n_s$ is the number of stored link sources (the IBZ when the links are symmetry-reduced, else the full BZ) and $n_d$ the number of stencil steps, one per ± pair.

```tree
parallel_transport.h5
  velocity_dft_cart                    complex128[3, nk, nb', nb']   v, Ry bohr
  velocity_kinetic_cart                complex128[3, nk, nb', nb']   p alone
  collapsed_position_reduced           complex128[3, nk, nb', nb']   collapsed axes only
  links_ibz                            complex128[ns, nd, nb', nb']
    @k_storage                         str     "ibz_source_edges" or "full_bz_source_edges"
    @orientation                       str     "L_d(k) X(k+b_d) L_d(k)^H"
  singular_values_ibz                  float64[ns, nd, nb']
  berry_connection_reduced             complex128[3, nk, nb', nb']
  berry_connection_cart                complex128[3, nk, nb', nb']
  dft_energies_ry_full                 float64[nk, nb']
  dft_occupations_full                 float64[nk, nb']
  source_steps                         int[nd, 3]
  source_full_ids                      int32[ns]
  source_neighbor_full_ids             int[ns, nd]
  g_wrap_ibz                           int32[ns, nd, 3]
  full_forward_neighbors               int[nk, nd]
  directed_edge_source_row             int     symmetry-reduced links only
  directed_edge_source_direction       int     symmetry-reduced links only
  directed_edge_sym_idx                int     symmetry-reduced links only
  directed_edge_reverse                int     symmetry-reduced links only
  directed_edge_antiunitary            int     symmetry-reduced links only
  link_stencil_orders                  int32[3]        per axis: 0 position, 2 or 4
  link_stencil_step_orders             int32[nd]
  link_stencil_coefficients            float64
  collapsed_axis_flags                 int32[3]
  collapsed_axis_center_frac           float64[3]      NaN on stencil axes
  irr_idx_k                            int32[nk]
  sym_idx_k                            int32[nk]
  kgrid                                int32[3]
  kgrid_shift                          float64[3]
  reduced_spacing                      float64[3]
  reciprocal_lattice_cart              float64[3, 3]   bohr^-1
  schema_version                       int32           4
  band_start                           int32           0
  band_stop                            int32           nb'
  spin_channel                         int32           0
  effective_nspinor                    int32
  bispinor                             int32
  vnl_included                         int32
  polar_rcond                          float64
  wfn_path_utf8                        uint8[]
  wfn_fingerprint_utf8                 int32[]
  energy_units_ry                      int32           1
  velocity_units_ry_bohr               int32           1
  velocity_frame_cartesian             int32           1
  reciprocal_units_bohr_inverse        int32           1
  links_symmetry_reduced               int32
  connection_complete                  int32           1 once A is written
  velocity_validation_complete         int32
  velocity_validation_passed           int32           1 within the tolerances
  velocity_validation_band_stop        int32           nb
  velocity_validation_<metric>         float64         atol, rtol, max_abs, max_rel, ...
  velocity_validation_head_set_relative_l2   float64   eps_link
```

The links are stored on the IBZ only when the steps are the three reduced axes and the point group permutes them; bcc, fcc and hexagonal cells store them on the full BZ. A consumer that needs the links refuses a file with `connection_complete = 0` by name. `velocity_validation_head_set_relative_l2` is $\epsilon_{\mathrm{link}}$ on the elements the head reads; it falls as the k grid is refined and gates nothing here ([Velocity operator §3.6](../theory/velocity-operator.md#berry-valid)).

## Refusals {: #refusals }

!!! refuses
    - No `-i`. An input file without `nval` and `ncond` (or `number_bands_protected`) or `number_bands`, with one line per key (`number_bands is required`).
    - An input file without `sys_dim`, or with an unknown key.
    - `number_bands_protected` beside `nval` or `ncond`, below the occupied-band count or above `number_bands` (`GATE band_request_forms`). 
    - `bispinor_gw = full_static_cohsex` without `bispinor = true`.
    - No `*.upf` found, pseudopotentials that do not cover every species, or `sys_dim` other than `0`, `2` or `3`, unless `--skip-vnl`. Stage the files beside the input file or pass `--pseudo-dir`.
    - Fully relativistic pseudopotentials on a two-component WFN with no `<spinorbit>` record, or `<spinorbit>` true on a one-component WFN.
    - `--parallel-transport-out` equal to `--out`, or combined with `--skip-vnl`.
    - A negative validation tolerance.
    - Symmetry tables that do not map every full-BZ $\bk$ to a star parent in `WFN.h5` (`GATE dipole_energy_star_map`).
    - A named `parallel_transport.h5` on a k grid with a two-point axis, with an incomplete stencil shell (`GATE pt_link_shell_incomplete`), with occupied charge at the branch cut of a collapsed axis (`GATE pt_collapsed_axis_cut_density`), or whose connection assembles to zero. The default artifact is dropped instead, and `dipole.h5` is still written.

## Other features {: #other-features }

The utility can also add the velocity term of a DFT+U mean field, which this manual does not describe further. A `WFN.h5` whose Quantum ESPRESSO schema declares DFT+U refuses unless both keys below are set.

### `hubbard_input` {: .kw data-type="path" data-default="unset" }
DFT+U mean field: the pw.x input with the `HUBBARD` card, relative to the input file. Adds $\partial_{k_a} V^{\mathrm U}$ to $\hat v$.

Required, with `hubbard_occupations`, when the QE schema that authenticates `WFN.h5` declares DFT+U, and refused otherwise. Noncollinear (`nspinor = 2`), `ortho-atomic` projector and norm-conserving pseudopotentials only.
{: .kw-details }

### `hubbard_occupations` {: .kw data-type="path" data-default="unset" }
That run's `prefix.save/occup.txt`; its SHA-256 enters the DFT+U stamp.

