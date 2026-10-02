# Preparing inputs from DFT

LORRAX does not do DFT. It starts from a converged plane-wave DFT solution exported in
**BerkeleyGW's `WFN.h5` format**, and every other input it needs is derived from that file
by LORRAX's own preprocessing steps. This page is the step before
[Quickstart](quickstart.md): how a crystal becomes a `WFN.h5`.

!!! note "Scope"
    The **LORRAX side** (what `WFN.h5` must contain, what reads it, what is available on
    Perlmutter) was read off a real file and a real machine and is marked *verified*
    below. The **Quantum ESPRESSO namelists** of §2 are a starting point to check against
    your QE version's documentation. The inputs of a chain that was executed, for a
    magnetic spinor cell, are `tests/hsuite/fixture/{scf,nscf,pw2bgw}.in`.

## The chain

```text
  QE scf  →  QE nscf (empty states, GW k-grid)  →  pw2bgw.x  →  wfn2hdf.x  →  WFN.h5
                  │                                    │
                  │                                    └→ vxc.dat, kih.dat (BerkeleyGW only, §4)
                  └→ <prefix>.save/data-file-schema.xml (keep it beside WFN.h5)
```

A LORRAX run directory holds three things from the DFT side: `WFN.h5`, the `*.upf`
pseudopotentials, and `data-file-schema.xml`.

Then, from `WFN.h5`, LORRAX's own three preprocessing steps produce `centroids_frac_<N>.txt`,
`dipole.h5` and `kin_ion.h5` — see
[Quickstart → A first GW calculation](quickstart.md#first-calculation).
A WFN with every band of the plane-wave basis comes from the QE `.save` through
[`psp.run_dense_h`](how-to/complete-basis-wfn.md).

## 1. What `WFN.h5` must contain — *verified*

Read directly from a `WFN.h5` fixture
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

## 2. Quantum ESPRESSO

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
   vxc_flag = .false.
   kih_flag = .false.
/
```

```bash
pw2bgw.x -in pw2bgw.in > pw2bgw.out
wfn2hdf.x BIN WFN WFN.h5
cp Si.save/data-file-schema.xml .
```

Run `pw2bgw.x` right after the NSCF: each `pw.x` run overwrites `<prefix>.save/`.

- `nbnd` must exceed the deck's `number_bands` by at least one band, and the band at
  `number_bands` must not split a degenerate multiplet. GW refuses a deck that breaks
  either rule and names the fix.
- `wfng_nk*` must be the unshifted grid you intend to run GW on.
- `data-file-schema.xml` goes beside `WFN.h5` (or leave the `.save` directory there).
  The WFN header does not say which symmetry operations include time reversal; LORRAX
  reads that from the schema. Without it a run prints `SYMMETRY PROVENANCE WARNING` and
  treats every header operation as purely spatial, which is wrong for a magnet whose QE
  symmetries include one combined with time reversal.

### Magnetic spinor wavefunctions {#magnetic}

- **Keep every symmetry QE finds.** Do not set `no_t_rev` or `nosym`. A
  magnet's symmetries include operations composed with time reversal; QE
  reduces the k grid with them (Fe and Ni 20³: 641 stored k instead of 1062,
  Co 484 instead of 748).
- **Keep the NSCF's `data-file-schema.xml` beside `WFN.h5`** (the schema
  bullet above). It must be the NSCF's: LORRAX binds the time-reversal flags
  only when the schema's operations and k rows are those of the WFN
  ([`symmetry_maps`](services/symmetry_maps.md)). gwjax.out then prints
  `Active op rows : 8 unitary; 8 TR-composed` (Fe, Ni).
- **Effect.** A WFN with the time-reversal-composed operations and one
  without them, both from the same SCF density, give eqp within 0.07 meV:
  Fe 4³ scalar and bispinor, one-shot and SC; Ni 4³, Co 4³ and Co 6×6×4
  scalar SC (claim 3097).

| QE version | `pw2bgw.x` |
|---|---|
| 7.3.1–7.5 | The stock tool exports a noncollinear magnetic WFN and keeps its symmetry operations (4 operations, 2 with fractional translations, on the bundled H2⁻ fixture). The fixture was written by the 7.4.1 tool; the 7.5 module writes the same header. |
| 7.2 | BerkeleyGW ships a replacement source, `MeanField/ESPRESSO/version-7.2/pw2bgw_qe7.2_with_spinor_mag.f90` in the BerkeleyGW 4.0 tree. Copy it over `PP/src/pw2bgw.f90` in the QE 7.2 source and rebuild QE. Its README allows magnetization only in a run with no symmetries. LORRAX has not been run on its output. |

No patch is kept in this repository.

## 3. On Perlmutter — *verified*

Neither tool is on `PATH` by default:

| tool | where |
|---|---|
| `wfn2hdf.x`, `hdf2wfn.x`, `kgrid.x` | `module load berkeleygw/4.0-gcc-12.3` (or `4.0-nvhpc-23.9`), which prepends `/global/common/software/nersc9/berkeleygw/zen3/gcc-12/mpich/berkeleygw/BerkeleyGW-4.0/bin` |
| `pw.x`, `pw2bgw.x` | `module load espresso/7.3.1-libxc-6.2.2-cpu` (or `espresso/7.5-libxc-7.0.0-cpu`). `pw2bgw.x` is a Quantum ESPRESSO tool and is not in the BerkeleyGW module. |

The scalar Na fixture was built with the 7.5 module. The magnetic spinor H2⁻ fixture was built
with a QE 7.4.1 `pw.x` carrying a two-line SOC branch-selection patch. Its SCF is LDA, which
has no GGA branch, and the stock 7.5 module rebuilds it with Γ band densities symmetric to
4e-9 (bands 1–7). No private QE build is needed.

Run all of this on a compute node (`srun`, as in
[Installation › Perlmutter](installation/perlmutter.md#suite)), not on a login node.

## 4. `vxc.dat` and `kih.dat` {#vxc-kih}

LORRAX reads neither file, and no deck key names them. It builds the mean-field side
itself: `gw.kin_ion_io` writes the kinetic and ionic matrix ($T + V_\mathrm{ion}$,
`kin_ion.h5`), `gw.gw_jax` adds the Hartree term from the density, and the quasiparticle
energies come from $T + V_\mathrm{ion} + V_H + \Sigma$. $V_{xc}$ never enters. For a
LORRAX-only run set `vxc_flag = .false.` and `kih_flag = .false.`, as
`tests/hsuite/fixture/pw2bgw.in` does.

Write them in two cases:

- **A BerkeleyGW comparison.** `sigma.x` needs `vxc.dat` or `kih.dat` (its
  `sigma.inp` documents the choice). Add to `&input_pw2bgw`:

    ```fortran
       vxc_flag = .true.,  vxc_file = 'vxc.dat'
       kih_flag = .true.,  kih_file = 'kih.dat'
       vxc_diag_nmin = 1,  vxc_diag_nmax = 80     ! 1 … nbnd
    ```

- **A check of the mean-field side.** `kih.dat` is QE's own diagonal of
  $T + V_\mathrm{ion} + V_H$. When `gwjax.out` prints the banner
  `H0 = kin_ion + V_H … is UNPHYSICAL`, compare `kih.dat` by hand with LORRAX's same
  quantity: the diagonal of `kin_ion.h5` plus the `VH` column of `sigma_diag.dat`. No
  LORRAX tool reads `kih.dat`.

## See also

- [Quickstart](quickstart.md) — the bundled fixture, and the first real calculation
- [Theory → Symmetry](theory/symmetry.md) — the `mtrx`/`tnp` conventions, authoritative
- [Input reference](input_reference.md) — every deck key, generated from the parser
