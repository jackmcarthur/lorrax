---
driver: nscf
page: files
---

# Files

`psp.run_nscf` Formats of the files run_nscf reads and writes.
{: .page-summary }

<div class="workflow-map" data-driver="nscf" data-page="files"></div>

## Inputs {: #inputs }

| File | Input key | Produced by | Note |
|---|---|---|---|
| NSCF input file | `-i` | user | INI, one `[nscf]` section; paths relative to the input file. |
| `<prefix>.save/data-file-schema.xml` | `save_dir` | QE `pw.x` SCF | Cell, atoms, symmetry, `ecutwfc`, `ecutrho`, FFT grid, `nelec`, `noncolin`, `lspinorb`, `assume_isolated`. |
| `<prefix>.save/charge-density.hdf5` | `save_dir` | QE `pw.x` SCF, built with HDF5 | $\rho_{\mathrm{val}}(\bG)$: `rhotot_g` on `MillerIndices`. |
| `*.upf`, else `*.UPF` | `--pseudo_dir` | pseudopotential library | The files the QE run used; QE copies them into the `.save`. Keyed by element. |
| reference `WFN.h5` | `--ref_wfn` | any BerkeleyGW-format writer | Optional. Only `mf_header/kpoints/rk` and `w` are read. |
| deterministic `WFN.h5` | `--pb-wfn`, `wfn_file` | an earlier `psp.run_nscf`, or QE + `pw2bgw.x` | Pseudobands-only mode. Same spinor count and G spheres as this run. |

## Outputs {: #outputs }

<div class="file-list" markdown>

`WFN.h5`
:   BerkeleyGW-format Kohn–Sham bands: $N_b$ bands per IBZ k-point, energies in Ry, insulator occupations. Name set by `output`.

`<output stem>_pseudobands.h5`
:   With `pseudobands = true`: the protected Davidson bands followed by the pseudobands, in the same layout. No driver reads it.

stdout
:   Run report: mode, k-grid, FFT grid, spinor count, 2D truncation, per-k timings and the lowest eigenvalue in Ry.

</div>

`run_nscf` writes no `data-file-schema.xml`. Downstream readers look for one beside `WFN.h5` for the spin–orbit flag and the time-reversal type of each symmetry operation; see [Dependencies](dependencies.md#chain).

## WFN.h5 {: #wfn }

Written by `file_io.wfn_writer.WFNWriter`, header first and then one k-point at a time on rank 0. The tree below is read from `wfn_header_tables` and `WFNWriter.__init__`. Shapes are as h5py stores them; `nk` is the number of IBZ k-points, `ngktot` $= \sum_\bk$ `ngk`.

```tree
WFN.h5
  mf_header/
    versionnumber              int        1
    flavor                     int        2 (complex)
    kpoints/
      nspin                    int
      nspinor                  int
      nrk                      int        nk
      mnband                   int        nbnd
      ngkmax                   int
      ecutwfc                  float      Ry
      kgrid                    int[3]
      shift                    float[3]   0 0 0
      ngk                      int[nk]
      w                        float[nk]  sums to 1
      rk                       float[nk,3]  crystal
      el                       float[nspin,nk,nbnd]  Ry
      occ                      float[nspin,nk,nbnd]
      ifmin                    int[nspin,nk]  1
      ifmax                    int[nspin,nk]  N_occ
    gspace/
      ng                       int
      ecutrho                  float      Ry
      FFTgrid                  int[3]
      components               int[ng,3]  QE ggen order
    symmetry/
      ntran                    int        1 with nosym
      cell_symmetry            int        0
      mtrx                     int[48,3,3]
      tnp                      float[48,3]
    crystal/
      celvol                   float      bohr^3
      recvol                   float
      alat                     float      bohr
      blat                     float
      nat                      int
      avec                     float[3,3]
      bvec                     float[3,3]
      adot                     float[3,3]
      bdot                     float[3,3]
      atyp                     int[nat]
      apos                     float[nat,3]
  wfns/
    gvecs                      int[ngktot,3]
    coeffs                     float[nbnd,nspinor,ngktot,2]  re, im
```

The G axis of `wfns/gvecs` and `wfns/coeffs` is concatenated over k at offsets $\sum_{\bk'<\bk}$ `ngk`, in the QE order of each k-point's sphere. `el` and `occ` are filled for the first spin channel only; see the [Overview](index.md#refusals) for collinear `nspin = 2`.

