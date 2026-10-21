---
driver: nscf
page: deps
---

# Dependencies

`psp.run_nscf` What run_nscf needs and what reads its output.
{: .page-summary }

<div class="chain-map" data-driver="nscf" data-page="deps"></div>

## Position in the chain {: #chain }

<div class="updown" markdown>

<div markdown>

#### Reads, produced by

`<prefix>.save`
:   QE `pw.x` SCF, built with HDF5

`*.upf`
:   pseudopotential library, via the QE run

reference `WFN.h5`
:   any BerkeleyGW-format writer (optional, `--ref_wfn`)

</div>

<div markdown>

#### Read by

`centroids`
:   `WFN.h5`

`dipole`
:   `WFN.h5`, with the UPF files

`kin-ion`
:   `WFN.h5`, with the UPF files

`gw`
:   `WFN.h5`

</div>

</div>

The downstream readers look for a QE `data-file-schema.xml` in the directory of `WFN.h5` or the two above it, and use it only if its k rows and operations match the file. `run_nscf` writes none, and the SCF `.save` matches only when the SCF ran on the same k set. Without a match:

- the time-reversal type of each operation is unknown, and the run report says so (`SYMMETRY PROVENANCE WARNING`);
- with fully relativistic UPF files and `nspinor = 2`, building $V^{\mathrm{NL}}$ in dipole and kin-ion refuses, because the spin–orbit mode is not recorded in `WFN.h5`.

## Caches {: #caches }

!!! cache "$SCRATCH/.cache/lorrax/jax_compile/&lt;namespace&gt;/np&lt;P&gt;"
    **Reused when** the JAX persistent compile cache is enabled (the default)<br>
    **and matching** the lowered program, compile options, jaxlib and backend; the Davidson solver is compiled once per run for the padded G sphere.<br>
    <span class="refuse">Recompiles on a miss. `ISDF_JAX_CACHE_DIR` moves the cache; an empty value turns it off.</span>

## Runtime {: #runtime }

- `runtime.bootstrap()` runs at import, before `jax`: environment defaults, `jax.distributed` initialization, and a CPU fallback when no GPU backend is usable.
- k-points are dealt round-robin over processes; each rank sets up every k-point of a round in the same order, so the compiled programs agree across ranks.
- Results are gathered with an all-gather whose memory grows linearly with the process count; rank 0 alone writes the HDF5 files.
- `psp.xc` needs the patched `jax_xc` of `config/xc/README.md` on the import path; the native FFI pair is required as for every driver.
- Launch geometry is owned by the machine pages under [Installation](../../using/installation.md).

