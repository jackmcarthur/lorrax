# Preparing inputs from DFT

LORRAX does not do DFT. It starts from a converged plane-wave DFT solution exported in
**BerkeleyGW's `WFN.h5` format**, and every other input it needs is derived from that file
by LORRAX's own preprocessing steps. This page is the step before
[Quickstart](quickstart.md): how a crystal becomes a `WFN.h5`.

!!! note "Provenance of this page"
    The **LORRAX side** (what `WFN.h5` must contain, what reads it, what is available on
    Perlmutter) was read off a real file and a real machine on 2026-08-06 and is marked
    *verified* below. The **Quantum ESPRESSO side** (the `&system` and `&input_pw2bgw`
    namelists) is transcribed from the manual draft `manual/03_tutorial/3.1_silicon_end_to_end.md`
    and was **not executed in this pass** — no QE run was performed. Treat the namelists as
    a starting point to check against your QE version's documentation, not as a certified
    recipe.

## The chain

```text
  QE scf  →  QE nscf (empty states, GW k-grid)  →  pw2bgw.x  →  wfn2hdf.x  →  WFN.h5
                                                       │
                                                       └→ vxc.dat, kih.dat (optional)
```

Then, from `WFN.h5`, LORRAX's own three preprocessing steps produce `centroids_frac_<N>.txt`,
`dipole.h5` and `kin_ion.h5` — see
[Quickstart → Your first real calculation](quickstart.md#your-first-real-calculation),
**including the three defects in that chain you should know about before you run it.**

The centroid step has two selectors, `centroid.kmeans_cli --centroid-selector kmeans | pivoted_full_grid`.
The default is the seeded k-means draw and is unchanged; `pivoted_full_grid` is a deterministic whole-grid
pivoted Cholesky that uses no RNG at all, so the same deck gives byte-identical points every run. It is the
maximum-robustness backup rather than the default because its Gram is `O(N_grid^2)` — about 3 GB at a 24³
grid, about 200 GB at 48³. Both selectors stamp `centroid_source:` into the output file, and the GW deck key
`centroid_selector` asserts that stamp back. Full flag table, the affordability rule and the measured Σ_x
numbers: [drivers → centroids](drivers.md#the-deterministic-selector-when-to-reach-for-it).

## 1. What `WFN.h5` must contain — *verified*

Read directly from the bundled fixture `tests/regression/cohsex_debug/WFNsmall.h5`
(2026-08-06). Two top-level groups:

| path | contents |
|---|---|
| `/mf_header/crystal/` | `avec`, `bvec`, `adot`, `bdot`, `alat`, `blat`, `apos`, `atyp`, `nat`, `celvol`, `recvol` |
| `/mf_header/gspace/` | `components` (the G-vector list), `ng`, `FFTgrid`, `ecutrho` |
| `/mf_header/kpoints/` | `rk`, `w`, `nrk`, `ngk`, `ngkmax`, `el`, `occ`, `ifmin`, `ifmax`, `mnband`, `kgrid`, `shift`, `ecutwfc`, `nspin`, `nspinor` |
| `/mf_header/symmetry/` | `mtrx` (`[ntran,3,3]`), `tnp` (`[ntran,3]`), `ntran`, `cell_symmetry` |
| `/mf_header/flavor` | 2 = complex |
| `/wfns/coeffs` | `[mnband, nspinor, ngkmax, 2]` — the trailing 2 is (re, im) |
| `/wfns/gvecs` | `[ngkmax, 3]` |

The fixture is `nspinor = 2`, `nspin = 1`, `flavor = 2`, `mnband = 150`.

**The `tnp` convention is a real trap and is documented separately.** The stored `mtrx` is
the *inverse* of QE's spatial rotation, and `tnp` carries an implicit factor of $2\pi$ —
both are properties of the `pw2bgw` writer, not of the spec, which describes `tnp` only as
"fractional translations". [Theory → Symmetry](theory/symmetry.md) owns this and gives the
writer and reader line references. If you are producing `WFN.h5` with anything other than
`pw2bgw`, read that page first.

## 2. Quantum ESPRESSO — *not executed in this pass*

SCF on a converged grid, then NSCF on the **unshifted GW k-grid** with the empty states and
a tight `conv_thr` (1e-10). LORRAX's production path is noncollinear with spin-orbit
coupling and fully-relativistic ONCV pseudopotentials:

```fortran
&system                          ! both runs
   ibrav = 2, celldm(1) = 10.26
   nat = 2, ntyp = 1
   ecutwfc = 60.0
   noncolin = .true.
   lspinorb = .true.
   no_t_rev = .true.
   nbnd = 80                     ! NSCF: valence + empties
/
```

Export with `pw2bgw.x`, then convert the binary to HDF5:

```fortran
&input_pw2bgw
   prefix = 'Si'
   real_or_complex = 2
   wfng_flag = .true.,  wfng_file = 'WFN'
   wfng_kgrid = .true., wfng_nk1 = 4, wfng_nk2 = 4, wfng_nk3 = 4
   vxc_flag = .true.,   vxc_file = 'vxc.dat'
   kih_flag = .true.,   kih_file = 'kih.dat'
/
```

```bash
pw2bgw.x -in pw2bgw.in > pw2bgw.out
wfn2hdf.x BIN WFN WFN.h5
```

`nbnd` here is the summation band count and becomes `nband` in the LORRAX deck; they should
match. `wfng_nk*` must be the unshifted grid you intend to run GW on.

!!! warning "Stock `pw2bgw` does not export spinor wavefunctions with magnetization"
    LORRAX's spinor path needs a **patched** `pw2bgw`. The patched source
    `pw2bgw_qe7.2_with_spinor_mag.f90` supports both full-spinor and non-spinor
    calculations, and enables magnetization for full-spinor runs **with no symmetries
    allowed**. Install it by copying it over `QE7.2/PP/src/pw2bgw.f90` and rebuilding QE.
    This is also the writer whose `tnp = (S⁻¹·ft)·2π` convention
    [Theory → Symmetry](theory/symmetry.md) documents.

    The patched source and its README are **not in this repository**. They live with the
    BerkeleyGW MeanField sources; the copy this project has been using is under a
    `docs_bgw/` tree alongside `wfn.h5.spec`. Ask before assuming a stock QE build will do.

## 3. On Perlmutter — *verified*

Neither tool is on `PATH` by default. Measured 2026-08-06:

| tool | where |
|---|---|
| `wfn2hdf.x`, `hdf2wfn.x`, `kgrid.x` | `module load berkeleygw/4.0-gcc-12.3` (or `4.0-nvhpc-23.9`), which prepends `/global/common/software/nersc9/berkeleygw/zen3/gcc-12/mpich/berkeleygw/BerkeleyGW-4.0/bin` |
| `pw2bgw.x` | **not** in the BerkeleyGW module — it is a Quantum ESPRESSO post-processing tool and ships in QE's `bin`. NERSC provides `espresso/7.3.1-libxc-6.2.2-{cpu,gpu}` and `espresso/7.5-libxc-7.0.0-{cpu,gpu}` |

Since the spinor path needs a patched `pw2bgw.f90` (§2), the NERSC `espresso` modules will
not supply it — that route requires your own QE build.

Run all of this on a compute node with [`lx run`](environment/machines/perlmutter.md#1-entry-point-lx),
not on a login node.

## 4. `vxc.dat` and `kih.dat`

`pw2bgw` can also export the exchange-correlation (`vxc_flag`) and kinetic+ionic
(`kih_flag`) matrix elements. LORRAX computes its own kinetic+ionic term with
`gw.kin_ion_io` (step 3 of the preprocessing chain), so `kih.dat` is not required — but it
is the natural independent cross-check when the `H0 = kin_ion + V_H` sanity banner fires,
and the banner itself suggests it:

```text
Cross-check H0 against pw2bgw's kih.dat.
```

The consumption path and default filenames on the LORRAX side for these two files are
**not documented and were not verified here**; treat them as an expert route.

## See also

- [Quickstart](quickstart.md) — the bundled fixture, and the first real calculation
- [Theory → Symmetry](theory/symmetry.md) — the `mtrx`/`tnp` conventions, authoritative
- [Input reference](input_reference.md) — every deck key, generated from the parser
- `manual/03_tutorial/3.1_silicon_end_to_end.md` — the fuller silicon walkthrough this page
  draws on (repo only, a working draft with unfrozen numbers)
