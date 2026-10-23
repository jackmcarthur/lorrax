# WFN axis and I/O layout stress

`build_axis_stress_fixture.py` writes a separate WFN with 521 stored k rows,
20,003 scalar bands, ragged 3/5/7 plane-wave rows, and an 8³ FFT box. Every
coefficient byte is physically written. Dyadic coefficients encode band,
stored-k and G-slot identities; the occupied band is the constant state.
The file is about 1 GB, including energies and occupations.

The bands are intentionally nonorthogonal, and the G sphere is deliberately
small. This is an axis/padding/I/O-layout test. It does not measure physical
large-grid FFT throughput, basis quality, or a real metal's band structure.

After loading the normal LORRAX runtime environment, run on compute nodes:

```bash
python services/wfn_loader/bench/build_axis_stress_fixture.py --out /scratch/WFN_axes.h5
srun -n 64 python services/wfn_loader/bench/centroid_axis_stress.py --wfn /scratch/WFN_axes.h5 --mode raw
srun -n 64 python services/wfn_loader/bench/centroid_axis_stress.py --wfn /scratch/WFN_axes.h5 --mode kaxis --k-chunk 17
srun -n 64 python services/wfn_loader/bench/centroid_axis_stress.py --wfn /scratch/WFN_axes.h5 --mode staged --stages 31 --band-chunk 4097 --k-chunk 3 --band-start 0 --band-stop 20003
```

Use the machine's normal rank binding and CPU/GPU launcher. The builder is
a serial CPU task. `--source` optionally selects a different bundled-compatible
scalar header; the default is the bundled Na fixture. It never modifies that
source and refuses to overwrite its output.

The raw probe permutes all stored k requests and checks every local coefficient
bitwise against the declared formula, including zero band/G padding and
nonaliasing FFT sentinels. Its default band window 3:20002 ends before a real
following file band, so a padded read into that band is detectable. Use
`--band-start 0 --band-stop 20003` to check the complete band axis.

The k-axis probe samples every k row with 103 bands. The staged probe samples
all 20,003 bands in bounded 17-k requests; 31 stages cover all 521 k rows.
The final request has 11 rows. Both probes compare X/Y faces against literal
sparse Fourier sums from the encoded coefficients and typed symmetry metadata,
with no FFT or coefficient-loader reference. Expected arrays are local blocks;
Fourier phases are retained for only the current request. Mu padding is checked
as exact zero. Request-cache keys, identities and retained bytes must be
unchanged after each sampler call. Ordinary full-grid symmetry metadata is
outside that request-cache assertion.

The returned X/Y faces are distributed along different centroid mesh axes.
They replicate the other mesh axis. With only 19 physical centroids, world-size
centroid padding still matters; the staged probe intentionally preserves it.
This avoids conflating raw coefficient sharding with the memory needed to retain
all k/band face outputs simultaneously.
