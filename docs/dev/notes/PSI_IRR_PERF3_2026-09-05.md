# PSIIRR-PERF3: rejected parent-tail conv_kpair experiments

Both raw-parent C/Z tails were routed through the existing fused pair-convolution kernel, preserving typed unfold and conjugation. Actual four-GPU scalar/spinor parity passed below2.89e-16. Neither assembly variant is selected for production; runtime source remains ab2eee4f.

| Si P4 rank0 measurement | decomposed | map assembly | vectorized assembly |
|---|---:|---:|---:|
| Complete Z module median, Nsight ms |87.620692|92.627295|100.520370|
| Tiled HLO peak bytes/rank |6037367604|8120330012|9607329284|
| Full zeta stage seconds |18.24|18.06|17.98|

The native convolution itself takes10.73ms, but open-spin operand assembly and ABI transposes offset that saving. The complete tile slows and peak memory grows34.5–59.1%. Collective counts and compile counts are unchanged. Fitted-zeta normalized difference1.2018e-7 fails the requested1e-10 despite passing kernel parity; retained CCT condition numbers approach1e8. Matched eqp maxima0.497/1.128micro-eV do not waive that gate. Compatible quadrature schedules differ380/382nodes, so total driver wall and eqp increments do not isolate the kernel. Na P16 was skipped for insufficient remaining pool wall.

Evidence in sandbox `runs/Si/99_psi_irr_zeta_2026-09-05/perf3/`: decomposed06 step `lx-Xg4-162527-615890-2881`, map04 step `lx-Xg4-162520-544774-1525`, vectorized09 step `lx-Xg4-163416-668430-2894`, strict comparison10 step `lx-Xg0-163818-696347-3710`; all pool57941637. Full report: `reports/psi_irr_perf3_2026-09-05/report.md`; claims844/845/846/849. Durable evidence directory: `/global/cfs/cdirs/m4598/jackm/lorrax_evidence/psi_irr_perf3_2026-09-05`.

The adjacent `PSI_IRR_PERF3_map_2026-09-05.patch.gz` and `PSI_IRR_PERF3_vmap_2026-09-05.patch.gz` preserve exact tested source and GPU fixture changes. Each passed `git apply --check` against ab2eee4f before compression. To reproduce, use a separate checkout at ab2eee4f and pipe **one** decompressed patch into `git apply`; do not stack them. They are rejected prototypes, not accepted optimizations.

Published only on branch `perf/psi-irr-parent-tail-conv-kpair-2026-09-05`, unmerged. No native FFI implementation or comparison worktree was modified.
