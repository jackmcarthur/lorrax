# HSE06 references with Quantum ESPRESSO

A patch for Quantum ESPRESSO 7.5 that removes three run-time defects from its exact-exchange code, so that an HSE06 NSCF with thousands of empty bands takes minutes on GPUs: what it changes, how to build and run it, what LORRAX does with the result, and its limits.
{: .page-summary }

## The patch {: #patch }

Download [`qe-7.5-exx-fixes.patch`](../assets/qe-patches/qe-7.5-exx-fixes.patch) and apply it at the top of the QE 7.5 source tree, before `configure`:

```bash
patch -p1 < qe-7.5-exx-fixes.patch
```

It changes `PW/src/exx.f90` and `Modules/mp_exx.f90`. None of the three changes alters a computed result ([Tests](#tests)). The patch is not part of Quantum ESPRESSO and has not been reviewed by its developers.

With $N_k$ k-points, $N_q$ exchange q-points, $N_b$ computed bands and $N_v$ occupied bands:

| Change | Stock QE 7.5 | Patched |
|---|---|---|
| Occupied-only exchange (`exxinit`, `init_index_over_band`) | The exchange buffer and the partner-band loop run over all $N_b$ bands. Empty partners are multiplied by an occupation of zero after the FFT: $2N_kN_qN_b^2$ FFTs per exchange build. | Both stop at the highest occupied band: $2N_kN_qN_bN_v$ FFTs. |
| Buffer copy (`exxinit`, noncollinear GPU branch) | The whole exchange buffer is copied from the GPU to the host once per band and $\bk+\bq$ point. | One copy per call. |
| Pair scan (`vexx_k_gpu`) | For every band, q-point and block of seven partner bands, a host loop scans all $N_b^2$ entries of the pair table, a cost $\propto N_qN_vN_b^3$ per k-point. | The first and last partner of each band are tabulated once per call. |

The first change matters in an NSCF, where $N_b \gg N_v$. The second affects noncollinear GPU runs only. The third sits in code that collinear runs share.

## Build {: #build }

GPU build on Perlmutter (A100 nodes), with `configure` and `make`:

```bash
module load gpu PrgEnv-nvidia
module load nvidia/25.9 cudatoolkit/13.0 cray-fftw
./configure MPIF90=ftn F90=ftn CC=cc FC=ftn \
  --with-cuda=$CUDA_HOME --with-cuda-runtime=13.0 --with-cuda-cc=80 \
  --enable-openmp --with-scalapack=no \
  BLAS_LIBS=" " LAPACK_LIBS=" " FFT_LIBS="-L${FFTW_DIR} -lfftw3_omp -lfftw3"
```

`configure` selects no FFT backend under `PrgEnv-nvidia`. Set two lines of `make.inc`, then build:

```make
DFLAGS = -D__PGI -D__CUDA -D__FFTW3 -D__MPI -D__MPI_MODULE
IFLAGS = -I. -I$(TOPDIR)/include -I$(FFTW_INC)
```

```bash
make -j48 pw pp
```

- The source tree needs the `external/devxlib` and `external/mbd` submodules at the commits in `external/submodule_commit_hash_records`. The GitLab archive of the tag does not contain them.
- This build has no HDF5. The NSCF reads the SCF's ACE projector files from `<prefix>.save/`, so the SCF and the NSCF come from builds with the same HDF5 setting. Otherwise the NSCF stops with `aceinit0: file with ACE potential not found or not readable`.
- A CPU build takes the same patch but keeps the pair scan ([Limits](#limits)).

## Run {: #run }

Run the SCF with `input_dft = 'HSE'` on the GW k-grid, then the NSCF with the same k list: the NSCF reads the SCF's ACE projector by k-point index. Both input files carry the exchange keys; the NSCF adds the band count and the diagonaliser (example values):

```fortran
input_dft             = 'HSE'
nqx1 = 2, nqx2 = 2, nqx3 = 2
ecutfock              = 180
exxdiv_treatment      = 'gygi-baldereschi'
x_gamma_extrapolation = .true.

calculation           = 'nscf'
nbnd                  = 3264
diagonalization       = 'david'
diago_full_acc        = .true.
```

Launch one MPI rank per GPU and one pool per rank:

```bash
export OMP_NUM_THREADS=8 OMP_PLACES=threads OMP_PROC_BIND=spread
srun -N1 -n4 -c32 --cpu-bind=cores --gpus-per-task=1 pw.x -nk 4 -in nscf.in
```

- **One rank per GPU.** Two ranks per GPU run three times slower.
- **Davidson, not CG.** With more than one k-point per GPU, QE uploads the ACE projector on every application. Davidson applies it 70 times less often than CG.
- **Symmetry.** HSE06 does not need `nosym`. Keep QE's symmetries as on [Required DFT inputs](inputs-from-dft.md), with the same settings in the SCF and the NSCF. The GPU build has run full-zone k lists only ([Tests](#tests)).
- **Export with a CPU build of `pw2bgw.x`** from the same source, then `wfn2hdf.x` ([Required DFT inputs §3](inputs-from-dft.md#pw2bgw)). The GPU build's `pw2bgw.x` is untested.

The NSCF diagonalises twice. The first pass uses the SCF's ACE projector, which is exact only on the SCF's bands: the bands above them come out too high, by 0.3 to 1.6 eV in the Mg₇NiO₈ test below. QE then rebuilds the projector on all `nbnd` bands, reports `ACE projected onto <nbnd> (nbndproj) and applied to <nbnd> (nbnd) bands`, and diagonalises again. The saved wavefunctions are those of the second pass.

!!! note "Band margin"
    The top few bands of the window are the least accurate, because the rebuilt projector does not contain the states just above the window. On Si, a 48-band NSCF differs from a 104-band one by at most 0.03 meV in bands 1–38, by 0.2–0.3 meV in bands 39–46 and by 200 meV in bands 47–48. Compute a few percent more bands than the GW run reads.

## Tests {: #tests }

- **Occupied-only exchange against stock QE.** The logged FFT counts equal $2N_kN_qN_bN_v$. Si with spin–orbit coupling: identical output. Antiferromagnetic NiO with symmetry: total energy within $5\times10^{-5}$ meV. Mg₇NiO₈ SCF: identical in every printed digit.
- **GPU build against the patched CPU build.** NiO, 4×4×4 SCF: the same total energy in every printed digit and eigenvalues within 0.1 meV over 64 k-points × 64 bands. NiO, 2×2×2, 400-band NSCF: eigenvalues within 0.5 meV.
- **Pair-scan change.** NiO, 4×4×4, 800-band NSCF with and without it: all eigenvalues identical as printed.

- **Symmetry.** With symmetry, on the CPU build: Si (8 irreducible k-points), GaAs (29) and bulk CrCl₃ (14). The GPU build has run only full-zone k lists (`nosym`, `noinv`), on NiO and Mg₇NiO₈. Neither GPU change touches the symmetry operations.

Every test is a noncollinear run with spin–orbit coupling.

Wall times on Perlmutter GPU nodes (four A100 per node) against the patched CPU build on CPU nodes:

| System | Run | Patched GPU build | Patched CPU build |
|---|---|---|---|
| Mg₇NiO₈, 2×2×2 k and q, $N_v = 136$ | SCF, 144 bands | 4 min 34 s, 1 node | 24.0 min, 2 nodes |
| | NSCF, 3264 bands | 7 min 16 s, 1 node | unfinished after 3 h 12 min, 2 nodes |
| NiO, 4×4×4 k and q, $N_v = 48$ | SCF, 64 bands | 31 min, 1 node | 36 min, 4 nodes |
| | NSCF, 800 bands | 22 min, 1 node | not run |

The exchange build costs $N_kN_qN_bN_v$ pair FFTs at 30 to 34 µs each per A100, the same rate from 64 to 3264 bands.

## In LORRAX {: #lorrax }

**In progress.** LORRAX has no code specific to a hybrid starting point. `kin-ion`, `dipole` and `gw` read an HSE06 `WFN.h5` like any other, and the QP Hamiltonian $T + V^{\mathrm{ion}} + V^{\mathrm H} + \Sigma^{xc}$ contains no $V^{xc}$ ([kin-ion](../utilities/kin-ion.md)). Magnetic crystals run: NiO, bulk CrCl₃ and Mg₇NiO₈. What is missing:

- **The head.** The [velocity](../theory/velocity-operator.md) is $\hat{\mathbf p} + i[V^{\mathrm{NL}}, \hat{\mathbf r}]$. It has no $i[V^{x}, \hat{\mathbf r}]$ term for the nonlocal exchange of a hybrid, so the position matrix elements that the [q → 0 head](../theory/q0-head.md) takes from it are too small. LORRAX does not refuse this. On NiO at 4×4×4 the gap with [`head_correction`](../drivers/gw/keywords.md#head_correction) `= full` is 0.7 eV above the gap with `off`. The GW runs on the references of this page set `head_correction = off`, which drops the Γ-cell term.
- **Non-magnetic crystals with spin–orbit coupling refuse.** QE's noncollinear HSE06 states break Kramers pairing at $2\times10^{-6}$, above the $10^{-6}$ of the [time-reversal guard](../theory/symmetry.md#sym-trs-verdict). GaAs refuses at `kin-ion` with `GATE trs_qe_nonmagnetic_wfn_consistent`, with and without symmetry. Tightening the SCF threshold from $10^{-10}$ to $10^{-12}$ Ry leaves the residual unchanged.
- **The `nscf` driver refuses hybrids** (`GATE nscf_xc_extension`). The empty bands come from QE.

One step outside LORRAX touched the exchange in preprocessing. In the first Si and NiO tests a script replaced the cross-gap block of a copy of `dipole.h5` with position matrix elements from a PBE run on the same crystal and k list, $r^{\mathrm{HSE}} = U^\dagger r^{\mathrm{PBE}} U$ with $U = \langle\psi^{\mathrm{PBE}}|\psi^{\mathrm{HSE}}\rangle$, multiplied by the HSE06 energy differences. It is first order in the occupied–empty mixing $\lVert U_{vc}\rVert$, which is 0.05 on Si and 0.53 on NiO, so it is not controlled on NiO. It is not part of LORRAX, and no later run uses it.

## Limits {: #limits }

!!! refuses
    - Smearing or tetrahedra: `HYBRID_START occupied-only EXX patch: insulators only`. The patch supports fixed occupations only, because the exchange buffer is sized once at the highest occupied band.
    - More than one band group: `HYBRID_START occupied-only EXX patch: run with -nb 1`.

- **CPU builds keep the pair scan.** The same loop is in the CPU routine `vexx_k`, and the patch does not change it. It is negligible at 400 bands. At 3264 bands the Mg₇NiO₈ CPU NSCF had not finished its exchange build after 2 h 15 min, where the pair FFTs alone take 35 min. Use the GPU build for an NSCF with thousands of bands.
- **Host memory.** QE allocates the ACE projector for all k-points on every rank. NiO on a 4×4×4 grid takes 17 GB of host memory per rank at 800 bands.
- **CG on GPUs.** With more than one k-point per GPU, a CG NSCF runs 2.7 times slower than on one CPU node.
