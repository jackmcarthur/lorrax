# Inputs from DFT

`WFN.h5` The Quantum ESPRESSO side of a LORRAX run: the input files, the export to BerkeleyGW's wavefunction format, the pseudopotentials, and what spinors and magnets need.
{: .page-summary }

LORRAX does no self-consistent DFT. Every driver starts from a converged plane-wave Kohn–Sham solution in BerkeleyGW's `WFN.h5` format, and every other input is derived from that file by LORRAX's own [preprocessing steps](quickstart.md#preprocessing).
{: .lead }

## Workflow {: #chain }

```text
QE scf ──▶ QE nscf (empty states, GW k-grid) ──▶ pw2bgw.x ──▶ wfn2hdf.x ──▶ WFN.h5
   │              │
   │              └──▶ <prefix>.save/data-file-schema.xml   (symmetry and spin-orbit record)
   └──▶ <prefix>.save/charge-density.*                      (SCF moment, magnets only)

QE scf ──▶ psp.run_nscf ──▶ WFN.h5                           (LORRAX nscf driver)
```

A run directory holds three things from the DFT side:

| File | Read by | Note |
|---|---|---|
| `WFN.h5` | every driver and utility, through `wfn_file` | [Export](#pw2bgw) |
| `*.upf` | [`kin-ion`](../utilities/kin-ion.md), [`dipole`](../utilities/dipole.md), [`htransform`](../utilities/htransform.md), [`nscf`](../drivers/nscf/index.md) | [Pseudopotentials](#pseudopotentials) |
| `data-file-schema.xml`, or the `<prefix>.save` directory | the WFN loader, at the first use of symmetry | [Symmetry record](#schema) |

## 1. SCF {: #scf }

A standard `pw.x` SCF on a converged k-grid with norm-conserving pseudopotentials. The production route is noncollinear with spin–orbit coupling (`noncolin = .true.`, `lspinorb = .true.`) and fully relativistic ONCV pseudopotentials; scalar runs (`nspinor = 1`) are also read. A collinear spin-polarized run (`nspin = 2`) refuses downstream; run a magnet noncollinear instead ([Magnets](#magnets)).

A slab input file (`sys_dim = 2`) corresponds to an SCF with `assume_isolated = '2D'`. [`psp.run_dense_h`](../drivers/nscf/index.md#dense-h), which writes a complete-basis WFN, refuses a mismatch; the QE route checks nothing.

## 2. NSCF {: #nscf }

The NSCF puts the occupied and empty bands on the GW k-grid. There are two routes.

=== "QE nscf"
    The SCF input file with `calculation = 'nscf'`, `nbnd` set to the band count of [§6](#bands), a tight `conv_thr` (`1.0d-10`), and the unshifted GW grid:

    ```fortran
    &system
       nbnd = 80                ! valence + empty bands
    /
    K_POINTS automatic
     4 4 4 0 0 0
    ```

    QE reduces the grid with its own symmetry operations, which `pw2bgw.x` exports with the wavefunctions. Run `pw2bgw.x` right after it: each `pw.x` run overwrites `<prefix>.save/`.

=== "LORRAX nscf"
    ```bash
    python -u -m psp.run_nscf -i nscf.in
    ```

    The [`nscf`](../drivers/nscf/index.md) driver rebuilds the Kohn–Sham Hamiltonian from the SCF `.save` and writes `WFN.h5` directly, replacing the QE NSCF, `pw2bgw.x` and `wfn2hdf.x`. It reproduces only nonmagnetic, norm-conserving PBE insulators and checks none of these conditions. It writes no `data-file-schema.xml`. The SCF one, copied beside `WFN.h5`, is used only when the SCF ran on the same k-points ([Symmetry record](#schema)); otherwise a spinor `WFN.h5` with fully relativistic pseudopotentials refuses in `kin-ion` and `dipole` ([nscf dependencies](../drivers/nscf/dependencies.md)).

### k-points: QE's automatic grid {: #kgrid }

!!! note
    `kgrid.x` is not needed. Give the QE NSCF `K_POINTS automatic` with the GW grid. QE reduces it with every operation it uses, and LORRAX unfolds the stored list to the full grid with the same ones: the space group with its fractional translations, time reversal where it holds, and on a magnet the operations QE composed with time reversal ([Symmetry §1.1](../theory/symmetry.md#sym-used)).

The LORRAX `nscf` driver reduces its grid the same way. `pw2bgw.x` with `wfng_kgrid = .true.` records the grid dimensions in `WFN.h5`; they must be the NSCF grid. A `K_POINTS crystal` list from BerkeleyGW's `kgrid.x` is still read: it unfolds when every operation that reduced it is in the WFN header, and time reversal was used only where it holds.

- **Stage the NSCF schema** ([Symmetry record](#schema)). On a magnet it is what types the time-reversal-composed operations; without it a magnet's reduced list is unfolded with the wrong operations or refuses.
- **On a magnet, do not set `nosym` or `no_t_rev`** ([Magnets](#magnets)).
- **A full-grid k that no permitted operation reaches refuses** ([Refusals](#refusals)): a list reduced by $\bk\to-\bk$ on a WFN whose time reversal is off, or by an operation outside the header. Nothing is substituted for the missing point.

## 3. Export: `pw2bgw.x` and `wfn2hdf.x` {: #pw2bgw }

```fortran
&input_pw2bgw
   prefix          = 'Si'
   real_or_complex = 2
   wfng_flag       = .true.
   wfng_file       = 'WFN'
   wfng_kgrid      = .true.
   wfng_nk1        = 4
   wfng_nk2        = 4
   wfng_nk3        = 4
   wfng_dk1        = 0.0
   wfng_dk2        = 0.0
   wfng_dk3        = 0.0
   vxc_flag        = .false.
   kih_flag        = .false.
/
```

```bash
pw2bgw.x -in pw2bgw.in > pw2bgw.out
wfn2hdf.x BIN WFN WFN.h5
cp Si.save/data-file-schema.xml .
```

- `real_or_complex = 2`. A real (`flavor = 1`) file refuses.
- `wfng_nk1..3` is the NSCF grid and `wfng_dk1..3` stay zero: the GW grid is the unshifted one. `WFN.h5` records the shift in `mf_header/kpoints/shift`.
- LORRAX reads neither `vxc.dat` nor `kih.dat`, so `vxc_flag` and `kih_flag` stay `.false.` unless the run is compared with BerkeleyGW ([For BerkeleyGW users](berkeleygw-users.md)).

| QE version | `pw2bgw.x` |
|---|---|
| 7.3.1 – 7.5 | The stock tool exports a noncollinear magnetic WFN with its symmetry operations. The bundled test fixture was written by 7.4.1. |
| 7.2 | BerkeleyGW 4.0 ships a replacement, `MeanField/ESPRESSO/version-7.2/pw2bgw_qe7.2_with_spinor_mag.f90`, to copy over `PP/src/pw2bgw.f90` before rebuilding QE. LORRAX has not been run on its output. |

On Perlmutter neither tool is on `PATH` by default:

| Tools | Module |
|---|---|
| `wfn2hdf.x`, `hdf2wfn.x`, `kgrid.x` | `berkeleygw/4.0-gcc-12.3` (or `4.0-nvhpc-23.9`) |
| `pw.x`, `pw2bgw.x` | `espresso/7.3.1-libxc-6.2.2-cpu` (or `espresso/7.5-libxc-7.0.0-cpu`) |

## 4. Pseudopotentials {: #pseudopotentials }

Stage the pseudopotentials of the QE run beside the input file. `WFN.h5` holds the wavefunctions and band energies but not the Hamiltonian that produced them, and LORRAX computes its one-body parts itself: [`kin-ion`](../utilities/kin-ion.md) evaluates the kinetic energy and the ionic pseudopotential between the Kohn–Sham states, $\langle m\bk|T+V^{\mathrm{ion}}|n\bk\rangle$, the fixed part of the quasiparticle Hamiltonian; [`dipole`](../utilities/dipole.md) adds the nonlocal pseudopotential's part of the velocity operator; and the [`nscf`](../drivers/nscf/index.md) driver rebuilds the whole Kohn–Sham Hamiltonian. They must be the files QE used.

`kin-ion` and `dipole` read every `*.upf` in the input file's directory (`*.UPF` only when no `*.upf` is present), then `../qe/scf` and `../qe/nscf`; `--pseudo_dir` names another directory. The `nscf` driver reads them from `save_dir`. Each file is matched to atoms by the element in its header and the atomic numbers `atyp` of `WFN.h5`.

The operators assume norm-conserving pseudopotentials: the overlap is the identity, and the nonlocal potential is $\sum_{ij}|\beta_i\rangle D_{ij}\langle\beta_j|$ from `PP_BETA` and `PP_DIJ`. Core corrections (`PP_NLCC`) are read. UPF version 2 files (root element `<UPF>`) are the format in use.

| Condition | `kin-ion`, `dipole` | `nscf` | `psp.run_dense_h` |
|---|---|---|---|
| no UPF found, or a species without one | refuses | **accepted**, those atoms left out | refuses |
| ultrasoft or PAW UPF | **accepted**, used as norm-conserving | **accepted** | refuses |
| UPF functional differs from the QE run | **accepted** | **accepted** | refuses |
| a file that does not parse | skipped with `Warning: failed to load …` | skipped | skipped |
| two files for one element | the last in sorted order wins | the same | the same |
| fully relativistic UPF, spinor WFN, no QE schema | refuses | — | — |
| QE `<spinorbit>` true on a scalar WFN | refuses | — | — |

The `kin-ion` run report lists each species as fully or scalar relativistic (decided by the j labels of its projectors), with its valence charge and core correction. A `DEGENERACY MISMATCH` block in that report means $T + V^{\mathrm{ion}}$ splits a manifold that is degenerate in the WFN energies: the pseudopotentials are not those of the QE run, or the degeneracy is accidental.

## 5. Spinors {: #spinors }

A spinor `WFN.h5` (`nspinor = 2`, from `noncolin = .true.` with or without `lspinorb`) needs no special treatment: every driver reads `nspinor` from the file and handles it. Spinor bands count singly ([Band counts](#bands)). With fully relativistic pseudopotentials, also stage the QE schema ([Symmetry record](#schema)): `kin-ion` and `dipole` read from it whether spin–orbit coupling was on in QE, which a spinor `WFN.h5` alone does not say, since `noncolin = .true., lspinorb = .false.` writes one too. Without the schema such a run refuses.

## 6. Magnets: broken time reversal {: #magnets }

A magnet is run noncollinear, as a spinor `WFN.h5`; a collinear spin-polarized run (`nspin = 2`) refuses. What sets a magnet apart is broken time reversal. LORRAX measures time reversal on the occupied states before anything uses it ([`LORRAX_TRS_CHECK`](environment.md)), and when it is broken the whole calculation changes:

- **Symmetry.** Time reversal enters only through the operations QE marks as composed with it. $\bq$ and $-\bq$ are no longer related, so both are computed.
- **Frequency dependence.** $W^c(z)$ is no longer even in $z$, and every pole model carries an odd part: a second, odd residue per pole in GN-PPM, and the ordered tangential model, which pairs each $\bq$ with $-\bq$ and needs the high-frequency moments $M_0$ to $M_3$ ([Tangential W model](../theory/tangential-w.md#pen-ordered)). More of $W$ is sampled and fitted.
- **Treatments.** `hl_ppm`, `screening_diagrams = w_bse` and, on a metal, `head_correction = full` on the tangential route refuse. The full list: [Symmetry](../theory/symmetry.md#sym-trs-consumers).

For a magnet:

- **Stage the QE schema** beside `WFN.h5` ([Symmetry record](#schema)). It is the only record of which symmetry operations are composed with time reversal; without it every operation is taken as purely spatial, which is wrong for a magnet.
- **Keep every QE symmetry.** Do not set `nosym` or `no_t_rev`: QE reduces the k-grid with the time-reversal-composed operations, and LORRAX unfolds it with the same ones.
- **Keep the SCF `.save` too.** A spinor run whose QE operations include none composed with time reversal is judged by its SCF moment: time reversal holds only below $10^{-4}\,\mu_B$ per cell. The NSCF schema records zero, so LORRAX reads the moment from the SCF `.save` whose charge-density file matches the NSCF one; without it, time reversal is taken as broken and the `[2c-TRS]` report line says why.

## 7. Band counts {: #bands }

- **`nbnd` exceeds the top band sum by at least one.** The ζ-fit window ends at the larger of the χ⁰ and Σ band sums (`number_bands`, or `number_bands_chi` and `number_bands_sigma`). A window that reaches the last band of `WFN.h5` refuses: the band above it is what proves the window does not cut a multiplet.
- **The top band sum does not split a degenerate multiplet.** A ζ-fit window edge inside a multiplet refuses and names the edge, its smallest gap and the k-point. A band-sum edge inside a multiplet only warns when it comes from `number_bands`, and refuses when it is set by name with `number_bands_chi` or `number_bands_sigma`.
- **Spinor bands count singly.** A spinor WFN has one band per electron: Si with spin–orbit coupling has 8 occupied bands, the scalar run 4.
- **Band extrapolation** needs the Σ band sum to reach twice the occupied count ([Band extrapolation](../theory/band-extrapolation.md)).

Keys and defaults: [gw keywords](../drivers/gw/keywords.md).

## 8. Symmetry record {: #schema }

`WFN.h5` records the symmetry operations but not which of them are composed with time reversal, nor whether spin–orbit coupling was on in QE. LORRAX reads both from a QE `data-file-schema.xml` whose operations and k-points match the WFN. A nonmagnetic crystal without spin–orbit coupling runs correctly without one; a magnet, or a spinor run with fully relativistic pseudopotentials, needs it. Copy the NSCF `<prefix>.save/data-file-schema.xml` beside `WFN.h5`. LORRAX also searches the two directories above it, each directly and under `scf/`, `nscf/`, `qe/scf/` and `qe/nscf/`, for `data-file-schema.xml` and `*.save/data-file-schema.xml`; no input key names the path. Without a matching schema the run prints `SYMMETRY PROVENANCE WARNING`.

## Refusals {: #refusals }

!!! refuses
    - `WfnLoader: real-flavor WFN unsupported`: `flavor = 1`. Rerun `pw2bgw.x` with `real_or_complex = 2`.
    - `WfnLoader: spin-polarized WFN unsupported`: `nspin = 2`. Run QE noncollinear.
    - A full-grid k-point that no permitted operation reaches from the stored ones. For a nonmagnetic WFN, stage the NSCF and SCF `.save` beside `WFN.h5`; for a magnet, stage the NSCF schema. With a schema bound, the list was reduced by an operation QE did not record: regenerate it with `K_POINTS automatic`.
    - No pseudopotentials, or none for one of the atomic numbers in `atyp`. Stage the UPF files beside the input file or pass `--pseudo_dir`.
    - `sys_dim` other than 0, 2 or 3.
    - Fully relativistic pseudopotentials on a spinor WFN with no QE `<spinorbit>` record. Stage `data-file-schema.xml` beside `WFN.h5`.
    - QE `<spinorbit>` true on a scalar WFN. Use a spinor WFN or scalar-relativistic pseudopotentials.
    - A ζ-fit window that reaches the last band of `WFN.h5` or cuts a degenerate multiplet. Raise `nbnd`, or move the band count to a legal edge.

