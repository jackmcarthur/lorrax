# Codebase Structure & Architecture

**For AI agents**: This document describes the code organization, module responsibilities, key classes, data flow, and file formats. Read this when working on implementation tasks. For physics/theory, see [`PHYSICS_COMPREHENSIVE.md`](PHYSICS_COMPREHENSIVE.md).

---

## Table of Contents

1. [Module Structure](#1-module-structure)
2. [Key Classes](#2-key-classes)
3. [Data Flow Pipeline](#3-data-flow-pipeline)
4. [File Formats](#4-file-formats)
5. [Entry Points & CLI Tools](#5-entry-points--cli-tools)
6. [Function Call Hierarchy](#6-function-call-hierarchy)
7. [JAX Sharding Patterns](#7-jax-sharding-patterns)
8. [Code Locations Quick Reference](#8-code-locations-quick-reference)

---

## 1. Module Structure

```
src/
├── gw_isdf/            # GW/COHSEX calculations
│   ├── gw_jax.py       # Main GW driver (entry point)
│   ├── gw_init.py      # Input parsing, chunking strategy
│   ├── w_isdf.py       # Screened interaction W(iω) via CTSP
│   ├── w_isdf_dynamic.py# Dynamic W(ω) path
│   ├── get_windows.py  # Energy window functions
│   ├── vcoul.py        # Coulomb interaction utilities
│   └── compute_vcoul.py# V_q computation from zeta
│
└── isdf/
    ├── isdf_init/          # ISDF initialization & k-means clustering
    │   ├── kmeans_isdf.py  # K-means centroid selection (entry point)
    │   └── get_charge_density.py  # Charge density computation
    │
    ├── bse_isdf/           # Bethe-Salpeter equation (BSE)
    │   ├── bse_isdf.py     # BSE driver (entry point)
    │   └── bse_kpm.py      # BSE kernel computation
    │
    ├── common/             # Shared utilities & core algorithms
    │   ├── meta.py         # Meta dataclass (system parameters)
    │   ├── load_wfns.py    # Wavefunction loading, FFT, CCT/ZCT fitting
    │   ├── symmetry_maps.py  # k-point symmetry operations
    │   ├── chi_from_dipole.py  # Dipole-based susceptibility
    │   ├── gamma_matrices.py  # Pauli matrices for spinors
    │   ├── timing.py       # Performance timing utilities
    │   └── jax_profile.py  # JAX profiling context managers
    │
    ├── io/                 # Input/output operations
    │   ├── wfnreader.py    # WFNReader class (HDF5 wavefunction I/O)
    │   ├── epsreader.py    # EPSReader class (dielectric matrix I/O)
    │   ├── centroids.py    # Centroid file I/O
    │   ├── sigma_output.py # Self-energy output (eqp, sigma files)
    │   ├── qp_wfn.py       # Quasiparticle wavefunction I/O
    │   ├── tagged_arrays.py  # HDF5 array I/O with metadata
    │   ├── kin_ion.py      # Kinetic/ionic Hamiltonian I/O
    │   └── paths.py        # Path resolution utilities
    │
    ├── mixing/             # Self-consistent GW mixing/acceleration
    │   └── acceleration.py # Anderson mixing, fixed-point acceleration
    │
    ├── postprocess/        # Post-processing utilities
    │   └── rotate_wfn_to_qp.py  # Rotate wavefunctions to QP basis
    │
    ├── interpolate/        # Hamiltonian interpolation (experimental)
    │   └── htransform.py   # H-matrix interpolation
    │
    └── psp/                # Pseudopotential parsing (UPF, BerkeleyGW)
        ├── load_psp.py     # BerkeleyGW pseudopotential loader
        ├── load_upf.py     # UPF pseudopotential parser (entry point)
        └── upf_model_2_0_1/  # UPF XML schema models (xsdata)
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| **isdf_init** | Select interpolation points (centroids) via k-means clustering on charge density |
| **gw_isdf** | GW/COHSEX self-energy calculations, main computational pipeline |
| **common** | Core algorithms: FFT/NUFFT transforms, CCT/ZCT fitting, wavefunction manipulation |
| **io** | All file I/O: HDF5 wavefunctions, centroids, self-energy output, tagged arrays |
| **mixing** | Self-consistent GW acceleration (Anderson mixing, DIIS) |
| **bse_isdf** | Bethe-Salpeter equation for optical spectra (experimental) |
| **postprocess** | Post-GW analysis tools |

---

## 2. Key Classes

### 2.1 `Meta` (src/isdf/common/meta.py)

**Purpose**: Immutable dataclass containing all system parameters

**Key attributes**:
```python
@dataclass
class Meta:
    # MPI info
    rank: int                    # MPI rank
    n_proc: int                  # Total MPI processes

    # Band indices (b_id_0 ≤ b_id_1 ≤ b_id_2 ≤ b_id_3 ≤ b_id_4)
    b_id_0: int                  # Lowest band (occupied)
    b_id_1: int                  # Valence band minimum (for Σ)
    b_id_2: int                  # VBM (Fermi level)
    b_id_3: int                  # Conduction band maximum (for Σ)
    b_id_4: int                  # Highest band (all bands)

    # FFT grid
    fft_grid: tuple[int, int, int]  # (nx, ny, nz) real-space FFT grid

    # Interpolation points
    n_rtot: int                  # Total centroid points
    n_rmu: int                   # Centroid points (same as n_rtot)

    # Spins
    npol: int                    # Spin polarization (1=unpolarized, 2=polarized)
    nspin: int                   # Number of spin channels
    nspinor: int                 # Spinor components (1=scalar, 2=spinor)

    # k-grid (wavefunctions)
    nkx, nky, nkz: int           # k-grid dimensions
    nk_tot: int                  # Total k-points

    # q-grid (ISDF fitting) [NUFFT BACKEND]
    nqx, nqy, nqz: int           # q-grid dimensions (defaults to k-grid)
    nq_tot: int                  # Total q-points
    use_nufft: bool              # True if q-grid ≠ k-grid

    # Frequency grid
    nfreq: int                   # Frequency points for χ⁰(iω)

    # Computed properties
    band_edges: tuple            # (b0, b1, b2, b3, b4)
    band_ranges: SimpleNamespace # .valence, .conduction, .sigma, etc.
    kgrid: tuple                 # (nkx, nky, nkz)
    qgrid: tuple                 # (nqx, nqy, nqz)
```

**Band range names**:
- `valence`: (b1, b2) — Valence bands for self-energy
- `conduction`: (b2, b3) — Conduction bands for self-energy
- `sigma`: (b1, b3) — All bands for self-energy
- `occupied`: (b0, b2) — Occupied bands (for G^occ)
- `full`: (b0, b4) — All bands (for G^all)

**Usage**:
```python
meta = Meta(rank=0, n_proc=1, b_id_0=0, b_id_1=8, b_id_2=12, ...)
vb_min, vb_max = meta.band_range("valence")  # (8, 12)
```

---

### 2.2 `WFNReader` (src/isdf/io/wfnreader.py)

**Purpose**: Read BerkeleyGW HDF5 wavefunction files (`WFN.h5`, `WFNq.h5`)

**Key methods**:
```python
class WFNReader:
    def __init__(self, filepath: str):
        """Open WFN.h5 file."""

    def read_gvecs(self) -> np.ndarray:
        """Read G-vectors (n_g, 3)."""

    def read_coeffs(self, ik: int, bands: slice) -> np.ndarray:
        """Read wavefunction coefficients for k-point ik.
        Returns: (n_bands, n_spinor, n_g) complex array."""

    def read_evals(self) -> np.ndarray:
        """Read eigenvalues (n_k, n_bands)."""

    @property
    def nkpts(self) -> int:
        """Number of k-points."""

    @property
    def nbands(self) -> int:
        """Number of bands."""

    @property
    def nspinor(self) -> int:
        """Number of spinor components (1 or 2)."""
```

**HDF5 structure** (BerkeleyGW format):
```
WFN.h5
├── /wfns/gvecs           # G-vectors: (n_g, 3) int32
├── /wfns/coeffs          # Coefficients: (n_k, n_bands, n_spinor, n_g) complex128
├── /wfns/ekn             # Eigenvalues: (n_k, n_bands) float64
├── /kpoints/rk           # k-points in crystal coords: (n_k, 3) float64
└── /mf_header/...        # Crystal structure, FFT grid, etc.
```

---

### 2.3 `EPSReader` (src/isdf/io/epsreader.py)

**Purpose**: Read BerkeleyGW dielectric matrix files (`eps0mat.h5`, `epsmat.h5`)

**Key methods**:
```python
class EPSReader:
    def read_chi_head(self, iq: int) -> complex:
        """Read χ_00(q) for q-point iq."""

    def read_full_matrix(self, iq: int) -> np.ndarray:
        """Read full ε_GG'(q) matrix."""
```

**Usage**: Primarily for dipole-based head correction (§6.5 of PHYSICS_COMPREHENSIVE.md)

---

### 2.4 `SymmetryMaps` (src/isdf/common/symmetry_maps.py)

**Purpose**: Handle k-point symmetry operations, map k→k', unfold symmetry-reduced grids

**Key methods**:
```python
class SymmetryMaps:
    def __init__(self, crys, wfn, wfnq=None):
        """Initialize from crystal structure and wavefunction files."""

    def get_k_mapping(self) -> dict:
        """Map each k-point to symmetry-equivalent k' in irreducible zone."""

    def unfold_kgrid(self, data_irr: np.ndarray) -> np.ndarray:
        """Unfold data from irreducible to full k-grid."""
```

---

## 3. Data Flow Pipeline

### 3.1 Full GW/COHSEX Pipeline

```
INPUT FILES
    │
    ├─→ WFN.h5, WFNq.h5         (wavefunctions)
    ├─→ centroids.h5            (interpolation points from k-means)
    ├─→ cohsex.in               (input parameters)
    └─→ eps0mat.h5 (optional)   (dipole matrix elements)
    │
    ↓
┌───────────────────────────────────────────────────────────┐
│ STAGE 1: ISDF Fitting (load_wfns.py)                      │
│ ────────────────────────────────────────────────────────  │
│ For each q-chunk:                                          │
│   1. Load wavefunctions ψ(k) in band chunks               │
│   2. FFT G-space → real-space: ψ(k,r)                    │
│   3. Extract at centroids: ψ(k,r_μ)                       │
│   4. Compute pair density: P_k,ab(μ,ν)                    │
│   5. FFT k→R: P_R,ab(μ,ν)                                │
│   6. Spin-trace: C_R(μ,ν) = Σ_ab |P_R,ab|²               │
│   7. FFT R→q: C_q(μ,ν)                                   │
│   8. Cholesky: L_q(μ,ν) = chol(C_q)                      │
│   9. Load ψ(k,r) in R-chunks, compute Z_q(μ,r)           │
│  10. Solve: ζ_q(μ,r) = L_q^-H L_q^-1 Z_q(μ,r)            │
│  11. Write to zeta_q.h5                                   │
└─────────────────────┬─────────────────────────────────────┘
                      ↓
               zeta_q.h5 (10-100 GB, disk bottleneck)
                      │
                      ↓
┌───────────────────────────────────────────────────────────┐
│ STAGE 2: Coulomb Interaction (compute_vcoul.py)           │
│ ────────────────────────────────────────────────────────  │
│ For each q:                                                │
│   1. Load ζ_q(μ,r)                                        │
│   2. FFT: ζ_q(μ,G)                                        │
│   3. Compute: V_q(μ,ν) = Σ_G ζ*_q(μ,G) v(q+G) ζ_q(ν,G)   │
│   4. (Optional) Head correction for q=0                   │
└─────────────────────┬─────────────────────────────────────┘
                      ↓
                  V_q(μ,ν) arrays
                      │
                      ↓
┌───────────────────────────────────────────────────────────┐
│ STAGE 3: Green's Function & Susceptibility (gw_jax.py)│
│ ────────────────────────────────────────────────────────  │
│   1. Construct G_R^occ(μ,ν), G_R^all(μ,ν) in R-space     │
│   2. Compute χ⁰_R(μ,ν,iω) via CTSP quadrature (w_isdf.py)│
│   3. FFT R→q: χ⁰_q(μ,ν,iω)                               │
│   4. Dyson solve: W_q(μ,ν,iω) = V_q + V_q χ⁰_q W_q       │
│   5. Extract static: W_R(μ,ν) = W_R(μ,ν,ω=0)             │
└─────────────────────┬─────────────────────────────────────┘
                      ↓
              W_R(μ,ν), V_R(μ,ν)
                      │
                      ↓
┌───────────────────────────────────────────────────────────┐
│ STAGE 4: Self-Energy (gw_jax.py)                      │
│ ────────────────────────────────────────────────────────  │
│   1. Exchange: Σ^X_R(μ,ν) = -G^occ_R(μ,ν) ∘ W_R(μ,ν)     │
│   2. COHSEX:   Σ^C_R(μ,ν) = G^all_R(μ,ν) ∘ [W_R - V_R]   │
│   3. FFT R→k: Σ_k(μ,ν)                                   │
│   4. Project to bands: Σ_k,ij = ⟨ψ_ki|Σ_k|ψ_kj⟩          │
│   5. (Optional) Self-consistency loop (experimental)      │
└─────────────────────┬─────────────────────────────────────┘
                      ↓
OUTPUT FILES
    │
    ├─→ sigma_hp.log            (self-energy matrix elements)
    ├─→ eqp.dat                 (quasiparticle energies)
    ├─→ sigma_k.h5              (full Σ_k arrays)
    └─→ qp_rotations.h5         (QP wavefunction rotations)
```

---

### 3.2 Data Array Sizes

Typical system: 3×3×1 k-grid, 12 valence + 12 conduction bands, 100 centroids, 8 frequencies

| Array | Shape | Size (GB) | Location |
|-------|-------|-----------|----------|
| ψ(k,n,s,G) coefficients | (9, 24, 2, 50k) | 0.17 | WFN.h5 |
| ψ(k,n,s,r_μ) at centroids | (9, 24, 2, 100) | 0.0003 | In-memory |
| C_q(μ,ν) CCT matrix | (9, 100, 100) | 0.0007 | In-memory |
| ζ_q(μ,r) interpolation vectors | (9, 100, 64³) | 1.8 | zeta_q.h5 (disk) |
| V_q(μ,ν) Coulomb | (9, 100, 100) | 0.0007 | In-memory |
| χ⁰_q(μ,ν,ω) susceptibility | (9, 100, 100, 8) | 0.006 | In-memory |
| W_q(μ,ν,ω) screened interaction | (9, 100, 100, 8) | 0.006 | In-memory |
| Σ_k,ab(i,j) self-energy | (9, 2, 2, 24, 24) | 0.0002 | In-memory |

**Key bottleneck**: `zeta_q.h5` is often 10-100 GB for production systems, causing significant disk I/O overhead.

---

## 4. File Formats

### 4.1 Input Files

#### `cohsex.in` (text input file)
```
# Example COHSEX input
wfn_file    = WFN.h5
wfnq_file   = WFNq.h5
centroids   = centroids.h5
output_dir  = ./output

# Band ranges (0-indexed)
band_occupied_min = 0
band_valence_min  = 8
band_fermi        = 12
band_conduction_max = 24
band_max          = 30

# ISDF parameters
n_centroids = 100

# Q-grid for ISDF fitting (optional, defaults to k-grid)
q_grid = 2 2 1

# Frequency grid (CTSP quadrature)
n_freq = 8

# Chunking (optional, auto-detected if omitted)
chunk_bands = 6
chunk_q = 3
```

#### `centroids.h5` (HDF5)
```
centroids.h5
├── /centroids      # (n_mu, 3) float64 — centroid positions in crystal coords
└── /weights        # (n_mu,) float64 — charge density weights (optional)
```

Generated by: `kmeans_isdf` CLI tool

---

### 4.2 Intermediate Files

#### `zeta_q.h5` (HDF5)
```
zeta_q.h5
├── /zeta_q_0       # (n_mu, fft_nz, fft_ny, fft_nx) complex128
├── /zeta_q_1       # ...for q-point 1
├── ...
├── /zeta_q_N       # ...for q-point N-1
└── /metadata
    ├── n_q_tot     # Total q-points
    ├── n_mu        # Number of centroids
    ├── fft_grid    # (nx, ny, nz) FFT grid
    └── qgrid       # (nqx, nqy, nqz) q-grid dimensions
```

**Size**: 10-100 GB (disk bottleneck)
**Purpose**: Interpolation vectors ζ_q(μ,r) for reconstructing pair products

---

### 4.3 Output Files

#### `sigma_hp.log` (text, BerkeleyGW format)
```
# k-point    band_i  band_j   E_KS    Σ^X      Σ^C      E_QP
    1         11      11    -2.345  -8.123   7.234   -2.456
    1         12      12     2.134  -7.456   6.789    2.223
    ...
```

#### `eqp.dat` (text, BerkeleyGW format)
```
# k-point  band   E_KS   E_QP   occupation
    1      11    -2.345  -2.456  1.0
    1      12     2.134   2.223  0.0
    ...
```

#### `sigma_k.h5` (HDF5)
```
sigma_k.h5
├── /sigma_x        # (n_k, n_spin, n_spin, n_band, n_band) complex128
├── /sigma_c        # (n_k, n_spin, n_spin, n_band, n_band) complex128
├── /eqp            # (n_k, n_band) float64
└── /metadata
    ├── k_grid      # (nkx, nky, nkz)
    ├── band_range  # (b1, b3)
    └── n_spinor    # 1 or 2
```

---

## 5. Entry Points & CLI Tools

### 5.1 Console Commands (pyproject.toml)

```toml
[project.scripts]
gw_jax = "gw_isdf.gw_jax:main"
cohsex_isdf = "gw_isdf.gw_jax:main"
kmeans_isdf = "isdf.isdf_init.kmeans_isdf:main"
bse_isdf = "isdf.bse_isdf.bse_isdf:main"
load_psp = "isdf.psp.load_psp:main"
load_upf = "isdf.psp.load_upf:main"
```

### 5.2 Usage Examples

#### K-means Centroid Selection
```bash
kmeans_isdf --wfn WFN.h5 --n-centroids 100 --output centroids.h5
```

#### GW/COHSEX Calculation
```bash
gw_jax --input cohsex.in
# or equivalently:
python -m gw_isdf.gw_jax --input cohsex.in
```

#### BSE Calculation (experimental)
```bash
bse_isdf --input bse.in
```

---

## 6. Function Call Hierarchy

### 6.1 ISDF Fitting Call Stack

```
gw_jax.main()
  └─→ fit_zeta_chunked_to_h5()                    [load_wfns.py:1720]
       ├─→ get_sharded_wfns()                     [load_wfns.py:1520]
       │    └─→ read_Gvecs_to_devices()           [load_wfns.py:450]
       ├─→ compute_CCT_from_left_right()          [load_wfns.py:850]
       │    ├─→ compute_pair_density_spin_traced()[load_wfns.py:670]
       │    └─→ (NUFFT or FFT k→R, R→q)
       ├─→ blocked_cholesky_2d()                  [load_wfns.py:1200]
       └─→ compute_ZCT_from_left_right()          [load_wfns.py:950]
            └─→ solve_zeta_cholesky()             [gw_jax.py:172]
```

### 6.2 Self-Energy Call Stack

```
gw_jax.main()
  └─→ compute_sigma_pipeline_jax()                [gw_jax.py:1064]
       ├─→ get_zeta_q_and_v_q_mu_nu()            [gw_jax.py:524]
       │    └─→ compute_all_V_q_from_zeta_h5()   [compute_vcoul.py]
       ├─→ get_chi0_jax()                         [w_isdf.py:150]
       │    └─→ _get_chi_kernel()                 [w_isdf.py:60]
       ├─→ get_static_w_q_jax()                   [w_isdf.py:240]
       │    └─→ (Dyson solve: W = V + V χ W)
       ├─→ get_sigma_static_mu_nu_jax()           [gw_jax.py:941]
       │    └─→ (Σ^X = -G^occ ∘ W, Σ^C = G^all ∘ (W-V))
       └─→ get_sigma_static_kij_jax()             [gw_jax.py:979]
            └─→ project_potential_to_bands()      [gw_jax.py:1050]
```

---

## 7. JAX Sharding Patterns

### 7.1 Mesh Definition

**Global mesh**: `mesh_bands = Mesh(devices, ("bands",))` (gw_jax.py:72)
**2D mesh**: `mesh_xy = Mesh(devices.reshape(Px, Py), ("x", "y"))`

### 7.2 Sharding Specifications

From `PHYSICS_COMPREHENSIVE.md` §7:

| Array | Shape | Sharding | Notes |
|-------|-------|----------|-------|
| ψ(k,n,s,G) | (n_k, n_bands, n_s, n_g) | `P(None, 'bands', None, None)` | Shard bands across devices |
| ψ(k,n,s,r_μ) | (n_k, n_bands, n_s, n_μ) | `P(None, 'bands', None, 'y')` | 2D: bands + centroids |
| P_k,ab(μ,ν) | (n_k, 2, 2, n_μ, n_ν) | `P(None, None, None, 'x', 'y')` | 2D: centroids only |
| C_q(μ,ν) | (n_q, n_μ, n_ν) | `P(None, 'x', 'y')` | 2D blocked |
| ζ_q(μ,r) | (n_q, n_μ, n_r) | `P(None, 'x', None)` | 1D along centroids |
| Σ_k(i,j) | (n_k, n_bands, n_bands) | `P(None, 'bands', None)` | Shard outer band index |

### 7.3 Key JAX Patterns

**shard_map** (load_wfns.py:74): Per-device FFT without communication
```python
@partial(shard_map, mesh=mesh, in_specs=P(...), out_specs=P(...))
def fft_local(x):
    return jnp.fft.fftn(x)  # FFT on local shard only
```

**pjit** (gw_jax.py): Automatic sharding with named axes
```python
@partial(pjit, in_shardings=NamedSharding(mesh, P('x', 'y')),
               out_shardings=NamedSharding(mesh, P('x', 'y')))
def matmul_2d(A, B):
    return A @ B  # JAX inserts collectives automatically
```

---

## 8. Code Locations Quick Reference

### Common Tasks → File Locations

| Task | Primary File | Function/Class |
|------|-------------|----------------|
| **Parse input file** | `gw_isdf/gw_init.py` | `read_cohsex_input()` |
| **Load wavefunctions** | `common/load_wfns.py` | `get_sharded_wfns()` |
| **FFT G→r** | `common/load_wfns.py:74` | `shard_map` FFT |
| **NUFFT k→R** | `common/load_wfns.py:97` | `nufft_k_to_R_batched()` |
| **Compute pair density** | `common/load_wfns.py:552` | `compute_pair_density_...()` |
| **CCT matrix** | `common/load_wfns.py:850` | `compute_CCT_from_left_right()` |
| **Cholesky factorization** | `common/load_wfns.py:1200` | `blocked_cholesky_2d()` |
| **ZCT matrix** | `common/load_wfns.py:950` | `compute_ZCT_from_left_right()` |
| **Solve for ζ** | `gw_isdf/gw_jax.py:172` | `solve_zeta_cholesky()` |
| **Fit zeta (full pipeline)** | `common/load_wfns.py:1720` | `fit_zeta_chunked_to_h5()` |
| **Compute V_q** | `gw_isdf/compute_vcoul.py` | `compute_all_V_q_from_zeta_h5()` |
| **χ⁰ kernel** | `gw_isdf/w_isdf.py:60` | `_get_chi_kernel()` |
| **Dyson solve for W** | `gw_isdf/w_isdf.py:240` | `get_static_w_q_jax()` |
| **Self-energy Σ** | `gw_isdf/gw_jax.py:941` | `get_sigma_static_mu_nu_jax()` |
| **Project to bands** | `gw_isdf/gw_jax.py:1050` | `project_potential_to_bands()` |
| **Head correction** | `gw_isdf/gw_jax.py:1948` | (inline in main loop) |
| **Dipole S(ω)** | `common/chi_from_dipole.py` | `compute_S_omega()` |
| **Write sigma output** | `io/sigma_output.py` | `write_sigma_to_file()` |
| **Symmetry operations** | `common/symmetry_maps.py` | `SymmetryMaps` class |

---

## Next Steps

**For physics/theory**: See [`PHYSICS_COMPREHENSIVE.md`](PHYSICS_COMPREHENSIVE.md)
**For environment setup**: See [`ENVIRONMENT_COMPREHENSIVE.md`](ENVIRONMENT_COMPREHENSIVE.md)
**For agent improvement suggestions**: See [`AGENT_TODO.md`](AGENT_TODO.md)
