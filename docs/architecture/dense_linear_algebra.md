# Dense solves and GEMMs on P devices

Every GW stage ends in dense linear algebra on matrices whose side is the
centroid count: the W Dyson solve, the ζ factors, the eigensolves of
htransform, QSGW and the shared-pole constructor, and the GEMMs between them.
This page derives where each of those operations runs on a mesh of $P$
devices: the sizes that decide it, the layouts and the exchanges between them,
the distributed product, the two eigh routes, the inverse roots, the memory
rule they are priced against, what changes numerically between routes, which
engine serves each platform, and which stage takes which plan. It ends with
CrI3 24×24 at P64. The caller contract of every function named here is
[`distrib_la`'s API](../services/distrib_la/api.md); which library serves which
request is [its backends page](../services/distrib_la/backends.md). Code is cited
as `file:line` at `5ca246d5b`; read the file rather than the number.

## Symbols

| symbol | meaning |
|---|---|
| $P = p \times p$ | devices on the square mesh, axes `x` and `y` |
| $n$ | side of one matrix (a padded centroid count, a pencil side, a band count) |
| $B$ | matrices in one stack: q parents, k points, or a constructor round |
| $B_p = \lceil B/P\rceil P$ | the stack padded to the device count |
| $s = 16$ B | one complex128 element |
| $\mathcal B$ | the planner budget per device ([memory model](memory-model.md#budget)) |

## 1 Sizes and the square mesh

One $n \times n$ complex matrix holds $s n^2$ bytes. Spread over the whole mesh
it costs $s n^2/P$ per device; held whole it costs $s n^2$ on the device that
holds it. A stack of $B$ matrices costs $s B n^2/P$ per device either way, if
the stack is split evenly. Every placement decision on this page is the
comparison of a few such numbers with $\mathcal B$.

Work scales differently. An eigensolve or LU of one matrix is $O(n^3)$; a
distributed library splits that over $P$ devices but pays a fixed
per-call charge for its communication, so a stack of $B$ matrices on the mesh
costs $B$ sequential solves. Whole matrices per device run $\lceil B/P\rceil$
solves concurrently with no communication inside a solve. When a whole matrix
fits one device, the second is faster for every stack with $B \gtrsim 1$
([capacity route](../services/distrib_la/backends.md#capacity-route)).

LORRAX builds only square meshes ($P_x = P_y = p$; a non-square $P$ refuses at
start-up, [decisions](decisions.md)). On a square mesh the transpose of a face
tile is the tile at the mirrored grid position, so a transposed operand moves
by one `ppermute` and a local transpose, never a distributed transpose; and
the block-cyclic eigensolver's square-grid requirement always holds.

## 2 Layouts and the moves between them

A matrix stack lives in one of four layouts
([API: the mesh and the layouts](../services/distrib_la/api.md#the-mesh-and-the-layouts)):

| layout | `PartitionSpec` | per device | used for |
|---|---|---|---|
| face | `P(None,'x','y')` | $(B, n/p, n/p)$ | every large object at rest; distributed GEMM and eigh |
| slab | `P(None,('x','y'),None)` | $(B, n/P, n)$: whole rows | column selections and joins, between two exchanges |
| batch | `P(('x','y'),None,None)` | $B_p/P$ whole matrices | rank-local solves (route (c)); resident factors |
| local | replicated, or whole per-q tiles on their q owner | whole matrices | the `linalg = local` plan of a small solve |

The face, slab and batch layouts hold the same $s\,B n^2/P$ bytes per device.
The moves between them are explicit `shard_map` exchanges, so every byte
crosses the network once and GSPMD never sees a move it could lower as
replicate-then-partition.

**Face to batch.** Route (c)'s move is one `all_to_all` over both axes
`('x','y')`. It splits the stack and joins the $(n/p)^2$ tiles of every rank
into whole matrices. The way back is its literal inverse
(`_face_to_batch`, `_batch_to_face`,
`services/distrib_la/src/distrib_la/_batch_reshard.py:122,138`; the step table is
[API § batched routes](../services/distrib_la/api.md#batched-routes)). Each device
sends and receives about $s\,B_p n^2/P$ bytes per direction. The move is not a
tile permutation, which is why it is written by hand. Ragged stacks are
zero-padded to $B_p$; a zero pad matrix returns zeros.

**Face to slab.** One `all_to_all` over `y` splits a face tile's $n/p$ rows
into $p$ blocks and joins the $p$ column blocks of its mesh row in global
order. Rank $(x, y)$ then holds rows $[x\,n/p + y\,n/P,\ x\,n/p + (y+1)\,n/P)$
with every column. Rows are zero-padded to a multiple of $p$ inside the tile
first. The inverse exchange returns the face.

**Selections and concatenations.** Taking rows or columns of a face operand
(selecting directions, sorting poles, assembling unequal blocks) goes through
the slab: exchange, select or join locally, exchange back
(`common.staged_reshard`). A sharding constraint on a global `take` can let
GSPMD gather the operand onto one mesh axis.

**Placing panels.** The q-local shared-pole round joins $S$ state panels
$[B, n, r_s]$ and takes each slot's pencil columns from them
([bispinor sectors §5.2](bispinor_shared_pole_w.md#5-construction), equation S 4a).
`pack_panels` (`src/gw/shared_pole_local.py:286`) does this one panel at a time
on the batch layout, before the round program: each rank scatters its own
slots' panel columns as contiguous rows of a row-major accumulator and turns
them back into columns once. No byte moves between ranks. The programs are fixed
by the panel width and $F$, never by $S$, so a later SC map with more line
panels makes more calls and no new program. The face programs join and take
the panels themselves through the slab exchange.

### What GSPMD emits {#what-gspmd-emits}

A face program written in global view (slices, concatenations, `a + a^\dagger`,
masks on face operands) is partitioned by GSPMD. Its collectives can be read off
the optimized HLO and split by origin. Those whose `op_name` ends in a JAX
collective primitive come from `shard_map` regions: `panel_matmul`'s SUMMA, the
slab exchanges and route (c). The rest GSPMD inserted. The census below covers
the staged sector programs at CrI3 24×24 P64 shapes: TT side 25856, $n_{TT}$ 4992,
stage width 4; CT joint side 17408, stage width 4. They were compiled on CPU
host meshes. Bytes are received per device per execution, with in-loop
collectives weighted by their trip counts (claim 4104).

| program | optimized ops (2×2 → 8×8) | GSPMD collectives at 8×8 | GB/device at 8×8, GSPMD vs explicit |
|---|---|---|---|
| TT stage 1 (members) | 9328 → 31394 | 299 all-to-all, 154 collective-permute (45 at 2×2), 8 other | 19.0 vs 6.7 |
| TT stages 2–4 | 4017 → 5034, 2208 → 2964, 3973 → 5855 | 43, 17, 56 | 3.5, 1.7, 2.3 vs 34.2, 49.0, 13.8 |
| CT pencil | 15495 → 37743 | 3 | 0.15 vs 37.3 |

What the attribution of stage 1 shows:
- Every state panel joined inside the program costs its own small all-to-all
  (219 of them, 0.7 GB), and the program depends on the panel count: twelve
  more line panels grow stage 1 to 33662 operations and the CT pencil to 41775
  at 8×8.
- Most of the bytes GSPMD moves in stage 1 are the pencil's block joins
  (`hermitian_block`, `join_columns` at the $F$ and $2 i_S$ offsets: 76
  all-to-alls, 7.2 GB) and the paired-basis slices at the half-extent
  offsets, which are not tile-aligned (138 collective-permutes, 8.1 GB).
- The CT pencil's collectives are all explicit. Its growth with the mesh is its
  per-panel joins (607 all-to-alls) and `panel_matmul`'s own panel structure:
  the panel count and the narrower tail panel follow $n/p$.

## 3 The distributed product: SUMMA and `batch_gram`

A face product forms $C_q = A_q B_q$ for every $q$ of a stack of $B$ matrices,
with $A_q$ of shape $(m,k)$, $B_q$ of shape $(k,n)$, and all three on the face.
On the $p \times p$ mesh write $k_\ell = k/p$ for the contraction columns of
one owner block, and $M_x$, $K_y$, $N_y$ for the $x$-th or $y$-th block of the
row, contraction and column ranges ($m/p$, $k_\ell$ and $n/p$ indices). Rank
$(x,y)$ holds $A_q[M_x,K_y]$ and $B_q[K_x,N_y]$, and its output tile
$C_q[M_x,N_y]$ needs the $A$ blocks of its mesh row and the $B$ blocks of its
mesh column. `distrib_la.panel_matmul`
(`services/distrib_la/src/distrib_la/_panel_matmul.py:29`) delivers them as a
batched 2-D SUMMA inside one `shard_map`.

**Interleaved panels.** Panel $j$ takes the same $w$ local columns
$[jw,(j+1)w)$ of every owner block, the global set
$K^{(j)} = \{\,i k_\ell + jw + t : 0 \le i < p,\ 0 \le t < w\,\}$. One
`all_gather` of $A$'s slice over `y` and one of $B$'s over `x` give rank
$(x,y)$ both $A_q[M_x,K^{(j)}]$ and $B_q[K^{(j)},N_y]$ in the same column
order, so the tile is a sum of local GEMMs with no reduction after it:

$$
C_q[M_x,N_y] = \sum_{j=0}^{n_p-1} A_q[M_x,K^{(j)}]\; B_q[K^{(j)},N_y] .
\tag{D 1}
$$

Every $q$ of the stack rides in each exchange and in each local GEMM: one
collective per panel, not per $q$. The next panel is gathered while the
current one is multiplied, so two panels are live:

$$
\text{panel bytes per device} = s\,B\,2pw\,\Big(\frac{m}{p} + \frac{n}{p}\Big)
\le \texttt{panel\_bytes} .
\tag{D 2}
$$

The width also obeys $pw \le k_\ell$, so a gathered panel never exceeds one
owner block and no device holds a contraction-complete row or column. With
$w_{\max}$ the largest width that meets both bounds, the panel count is
$n_p = \lceil k_\ell / w_{\max} \rceil$ and the width is the even split
$w = \lceil k_\ell / n_p \rceil$ (`_interleaved_width`, `_panel_matmul.py:121`).
Since $pw \le k_\ell$, $n_p \ge p$. Flops per device are $8Bmnk/P$; the
exchanges are $O(sB(m+n)k/p)$ per device.

### The panel loop {#the-panel-loop}

**Zero-padded K.** The split leaves $d = n_p w - k_\ell$ columns,
$0 \le d < \min(w, n_p)$, by which the last panel would run past its block.
Its window is held at $[k_\ell - w, k_\ell)$ instead. The first $d$ columns of
that window were panel $n_p - 2$'s, and they are zeroed in $A$'s slice before
the gather (`gather`, `_panel_matmul.py:255`). Each block is therefore read as
$n_p w$ columns, $d$ of them inert zeros: $k$ is padded to a whole number of
panels without a padded copy of any tile, and the padding is less than one
panel. A zero column adds an exact zero, so (D 1) changes only in summation
order. On CPU host meshes (2×2 and 4×4, 1 to 9 panels, every operand option),
105 of 144 products equal main's bit for bit, and the rest agree to
$2.7\cdot10^{-16}$ relative (claim 4110).

**One scan.** A prologue gathers panels 0 and 1 and multiplies panel 0 into a
fresh output tile, so no zero fill runs. One `lax.scan` then gathers panel
$j+1$ and multiplies panel $j$ at each step. An epilogue multiplies the last
panel, which has nothing left to prefetch (`_panel_matmul.py:300`). Every step
has the same shapes, so the compiled per-device program is the same for every
$n_p \ge 4$, and so for every mesh from 4×4 up. The mesh enters only through
the trip count, the tile shapes and the replica groups. A loop whose last panel
had another width would carry a fourth product site, with its own slices and
GEMM, whenever $w \nmid k_\ell$. That depends on the mesh, since
$k_\ell = k/p$. Optimized HLO instructions, CPU host meshes, compile only,
CrI3 24×24 P64 sector shapes (claim 4110):

| program | 2×2 / 4×4 / 8×8, last panel narrower | 2×2 / 4×4 / 8×8, one scan |
|---|---|---|
| one product, the CT pencil's $(W_{TC}Q_C)^\dagger Q_T$: $q = 4$, $k = 4992$, $m = 18432$, $n = 24576$ | 151 / 151 / 131 | 143 / 143 / 143 |
| one product, $(4, 17408, 17408) \cdot (4, 17408, 4992)$ | 113 / 113 / 128 | 125 / 125 / 125 |
| CT keep stage (face GEMMs and their glue) | 2014 / 2014 / 2308 | 2134 / 2134 / 2134 |
| CT output stage | 488 / 488 / 560 | 524 / 524 / 524 |

The scan's fixed cost is about 12 instructions per product: the zero mask and
the held offset. A face program that still grows with the mesh grows outside
its products ([what GSPMD emits](#what-gspmd-emits)). On the XLA route (CPU),
a narrower last panel also left one accumulation unfused beside the running
sum. The uniform loop removes it: the CT keep stage's compiled temporaries at
8×8 go from 8.37 to 6.23 GB, and the CT pencil's at 2×2 from 88.5 to 67.8 GB
(claim 4110).
On CUDA the in-place GEMM already absorbed that accumulation. On the CrI3 6×6
bispinor SC deck at P4 (forced face route, maps 0–2), the uniform loop moves
$E_{\rm QP}$ by at most 0.001 µeV against main. The warm map-2 sector walls
and the run peak (25.21 GB) are unchanged (claim 4111).

**Accumulation.** With `bounds`, or with three or more panels, every panel
after the first adds into the output tile in place through the local
`beta = 1` GEMM (on CUDA `lorrax_cublas_local_active_range_gemm`, or its
prepared target over every column when there are no bounds;
`_panel_contraction`, `_panel_matmul.py:133`). A two-panel product without
bounds stays on XLA. XLA folds one straight-line `c + a @ b` into its GEMM, but
of two adjacent ones it leaves one as a separate add that holds two more output
tiles. The 3-panel scan has one trip and is inlined, which makes two adjacent
ones; this is why three panels already take the in-place GEMM.

**Options.** With `bounds`, each panel's local GEMM runs only over the row's
live contraction interval, so dead bands cost no flops. `weights` scale each
$A$ slice on its way into the gather. `partner=True` returns
$\bar A\,\mathrm{diag}(w)\,\bar B$ from the same exchange. A transposed operand
crosses the grid diagonal in one `ppermute` and is transposed locally
([API](../services/distrib_la/api.md#bounded-face-products)). Face programs
compile with XLA's latency-hiding scheduler (`FACE_COMPILER_OPTIONS`,
`src/gw/shared_pole_execution.py:264`), which runs the prefetched gather beside
the local GEMM.

`panel_matmul` is the only distributed GEMM: every `matmul` request runs it on
every platform. On the CrI3 24×24 sector shapes at P64 it runs 2.3–4.2× faster
than the cuBLASMp face product it replaced, at 9.2–12.4 TF/s per A100 on
$O^\dagger Q$ for TT (claim 3425). Every face Green build is the same SUMMA, at
equal peak memory to the band gather it replaced (claim 2949).

**Whole rows per device.** When the factors of $W_q = b_q\,\mathrm{diag}(w_q)\,c_q^\dagger$
already sit in the batch layout, `batch_gram`
(`services/distrib_la/src/distrib_la/_panel_matmul.py:167`) contracts each
device's own rows with the same local active-range GEMM (`_panel_contraction`),
and only $W$ moves, batch to face. It has no panel loop: the contraction axis is
whole on each device. The shared-pole Σ uses it when the replicated pole
columns do not fit and whole parents do
([shared-pole model §8](shared_pole_model.md)).

## 4 The eigh routes {#eigh-routes}

A Hermitian stack $(B, n, n)$ on the face has two routes.

**Route (c), whole matrices per device.** Move the stack to the batch layout
(§2), run the native JAX eigh on each device's $\lceil B/P\rceil$ whole
matrices, move the eigenvectors back to the face
(`ROUTE_BATCH_RESHARD`, `services/distrib_la/src/distrib_la/plan.py:121`). No
distributed library runs. Its price beside the operand is

$$
\texttt{eigh\_stack\_bytes} = \Big\lceil \frac{B}{P}\Big\rceil\; 8\,n^2\, s ,
\tag{D 3}
$$

eight $n^2$ tiles per whole matrix a device holds: the input copy, the
exchanges, the local solve with its workspace and the vectors
(`BATCH_EIGH_TILES = 8`, `plan.py:395`, measured at 7.9–8.0 on bundle B11 at
P4 and P64; `eigh_stack_bytes`,
`services/distrib_la/src/distrib_la/workspace.py:306`).

**The capacity route, the whole mesh.** Each matrix is solved by the
platform's distributed eigensolver over all $P$ devices (cuSOLVERMp `syevd` on
CUDA, ScaLAPACK `pzheevd` on CPU), one matrix after another. Per device it
holds the vectors stack $sBn^2/P$, the values and one solve's workspace. It is
the only route for a matrix that does not fit one device.

**The decision.** A caller prices route (c) from the shapes and takes it when

$$
\text{boundary} + \texttt{eigh\_stack\_bytes} + \text{live stages} \le \mathcal B ,
\tag{D 4}
$$

the boundary being the arrays live beside that eigh; otherwise the whole mesh,
with one warning. The shared-pole sectors decide each stack so
(`staged_eigh`, `src/gw/shared_pole_execution.py:678`); a caller that passes a
room instead lets `distrib_la` decide each stack by the same price
([API § eigh-stack](../services/distrib_la/api.md#eigh-stack)). Nothing is
compiled to be measured and no size is exchanged, so every rank decides alike
(claim 3984).

**Checks.** No distributed eigh result is returned unchecked: eight
fixed-seed probe columns test $\|(AZ - Z\Lambda)X\|$ and $\|(Z^\dagger Z - I)X\|$
against $64\,n\,\varepsilon$, and a failure is solved again, shifted, then
gathered when it fits. Exact-zero rows are deflated to sentinels first, because
cuSOLVERMp returned wrong vectors with status 0 on such matrices
([API § dense factorizations](../services/distrib_la/api.md#dense-factorizations)).

## 5 Inverse roots: Newton–Schulz

The ordered shared-pole reduction needs $Z = A^{-1/2}$ of a Hermitian metric
$A$ that is the identity up to a small error. The coupled Newton–Schulz
iteration

$$
T = \tfrac12(3I - ZY), \qquad Y \leftarrow YT, \qquad Z \leftarrow TZ ,
\qquad Y_0 = A,\ Z_0 = I ,
\tag{D 5}
$$

converges to $Y \to A^{1/2}$, $Z \to A^{-1/2}$ while $d = \|I - A\|_\infty < 1$,
with $d_{\rm next} \le d^2(3+d)/4$. Each step is three batched GEMMs, so it runs on
`panel_matmul` in the face layout with no eigensolve. The iteration count is
fixed from the initial bound, $\lceil \log_2(\ln \varepsilon_t / \ln d)\rceil$,
never from an on-device residual, so every rank runs the same program
(`_metric_inverse_root`, `src/gw/shared_pole_reduction.py:18`). For the paired
sector metric, which is the identity to about $10^{-8}$ by construction, that is
one iteration; an eigh-based root of the same stack costs more (claim 3425).
The receipt reports $\max\|ZAZ - I\|_F/\sqrt R$. The checked eigh chain
re-orthonormalizes the vectors of a shifted retry by the related Newton–Schulz
polar iteration, also GEMMs only.

## 6 Memory

Every placement above compares shape prices with one number, the planner
budget $\mathcal B$. The [memory rule](memory-model.md#budget) owns it: how it
follows from the card total and the bytes outside the XLA pool, and how a deck's
`memory_per_device_gb` overrides it. Streamed loops take the fixed tile
`runtime.tiles.TILE_BYTES` instead, so their results do not depend on
$\mathcal B$. A stage priced over $\mathcal B$ warns and runs; no planner refuses
on a price.

## 7 Determinism and round-off

**Every rank decides alike.** Each route decision on this page reads only
shapes, the deck and $\mathcal B$, which is the minimum over processes. No
decision reads free memory or a compiled size, so no two ranks can enter
different collectives (INVARIANTS 21).

**Across routes, results move at round-off.** Changing the route changes the
summation order:

- route (c) against the whole mesh, and the sector q-local rounds against the
  staged face rounds: CrI3 6×6 P4 max $|\Delta E_{\rm QP}|$ 0.001 µeV, Fe 4³
  0.060 µeV (claim 3997);
- `linalg = local` against `linalg = distributed`: the block-cyclic solve sums
  in a grid-dependent order, so the two agree to $\kappa\varepsilon$, not
  bitwise ([the deck dial](../services/distrib_la/backends.md#the-deck-dial));
- a GEMM stage width changes how many parents share one exchange, and moves no
  number.

The charge ζ factor is the one solve kept off the distributed route: its
rank-truncating eigh would make the retained rank depend on the grid, which
GN-PPM amplifies, so it is the replicated whole-tile eigh on every layout.

## 8 Portability {#portability}

The platform is the device vendor, read from the JAX client by
`lxkit.device_vendor` (`services/lxkit/src/lxkit/gate.py:143`): `cpu`, `cuda`,
`rocm`, or the device's own string. `device.platform` alone is `gpu` for every
GPU vendor and is never used. XLA is the reference path on every platform; a
vendor route stays only where it is at least 2× faster on a production stage or
decisive on memory, and it is gated against XLA on the same device
([decisions § XLA reference](decisions.md#xla-reference)).

| operation | CUDA | CPU | ROCm and others |
|---|---|---|---|
| eigh, whole matrices (route (c), `linalg = local`) | `jnp.linalg.eigh` (cuSolverDn) | `jnp.linalg.eigh` (LAPACK) | `jnp.linalg.eigh` |
| eigh, capacity route | cuSOLVERMp `syevd` | ScaLAPACK `pzheevd` | none: no ROCm library is built; refuses |
| LU solve, `linalg = distributed` | cuSOLVERMp batched `getrf`/`getrs` | ScaLAPACK `pXgetrf`/`pXgetrs` | refuses |
| LU solve, `linalg = local` | `jax.scipy.linalg` LU per q | same | same |
| distributed GEMM | `panel_matmul` (XLA; local GEMMs on cuBLAS) | `panel_matmul` (XLA) | `panel_matmul` (XLA) |
| inverse root | Newton–Schulz on `panel_matmul` | same | same |
| k-convolution | nvidia-mathdx kernels; XLA where mathdx cannot hold the grid | FFTW3-ABI host plans | XLA (`jnp.fft`) |

The k-convolution router (`ffi.fft.kconv_backend`, `src/ffi/fft.py:270`) picks
its row from the vendor and the k grid alone
([k-convolution § router](kconv.md#router)). Only `linalg = local` runs on ROCm,
and no ROCm run has been made.

## 9 Which stage takes which plan {#stage-plans}

`linalg = local | distributed` is the one deck key for the layout of every
dense solve (`resolve_linalg`, `src/gw/gw_config.py:851`). `local` keeps whole
per-q matrices, scheduled q-parallel over the devices; it is mesh-invariant and
is the numerical control. `distributed` factors one matrix over the whole mesh;
it is the only plan whose factorization memory divides by $P$. The resolved
fields are tabulated in [the deck dial](../services/distrib_la/backends.md#the-deck-dial).
With $\mu$ the padded centroid count and $Q$ the stage's q extent:

| stage | `linalg = local` | `linalg = distributed` |
|---|---|---|
| ζ `C_q` build | `P(None,'x','y')`, $sQ\mu^2/P$ | same |
| ζ charge factor (`rank_truncate`) | replicated whole-tile eigh pseudo-inverse, one q batch at a time under `LORRAX_ZETA_REPLICATE_CAP_GIB`; q-parallel above $Q\mu^3 \ge 5\cdot10^9$, $\lceil Q/P\rceil\mu^3$ work per device | same |
| ζ back-solve, per G tile | whole-tile factor on its q owner, $s\lceil Q/P\rceil\mu^2$ per device; only the right-hand side moves | same |
| ζ transverse factor (ridge LU) | whole-tile local LU once per q and channel, applied per G tile ([the solve seam](zeta_fit_mubatch.md#the-solve-seam)) | same |
| W Dyson solve (`gw.w_isdf.solve_w`) | per-q dense LU, $\lceil Q/P\rceil$ whole $(\mu,\mu)$ tiles per device | `plan('solve_lu', backend='distributed').batched`, $sQ\mu^2/P$; the μ axes never leave the face |
| W ladder resolvent (`bse.w_ladder`, `screening_diagrams = w_bse`) | one code path for both layouts; no whole-$\mu^2$ object per device | same |
| eigensolves (htransform `fH_q`, `vq_interp` `C_q`, QSGW $H_k$) | route (c): one whole matrix per device | the capacity route: one matrix over the mesh |
| shared-pole constructor | q-local rounds, else face rounds with each eigh stack on route (c) or the mesh by (D 4) | same ([bispinor sectors §4](bispinor_shared_pole_w.md#sector-route)) |
| BSE restart load, matvec, W densify | face tiles through SlabIO and `shard_map`, $s\,n_k\mu^2/P$ | same |

**Where the ζ factor saturates.** The charge factor's only parallel axis is q,
so every rank past $Q$ idles for the whole factor stage, and the run says so
when $P > Q$.

**What does not divide by $P$.**
- `bse/vq_interp` (the `exciton_bands` and `bse_k_grid` paths) keeps `Fch`, the
  $(Q, \mu, n_G)$ long-range form factors, as a host array on every process; with
  `run_diagnostics`, `S` and `V_SRc` add two $(Q,\mu,\mu)$ host mirrors per
  process.
- The BSE ψ and `M` stacks are sharded on one mesh axis, so they divide by $p$
  while `W_q` divides by $P$.

**Thresholds.** `LORRAX_ZETA_REPLICATE_CAP_GIB` (default 4) bounds one
replicated charge-factor q batch; `LORRAX_ZETA_QPARALLEL` overrides the
$5\cdot10^9$ q-parallel threshold; `LORRAX_COLLECTIVE_CHUNK_MB` (default 128)
bounds one emitted collective of the distributed W Dyson build. Spellings and
grammar are in [the registry](../reference/env_vars.md). The launcher owns the
CPU transport ([collective transports](../environment/transports.md)).

## 10 Worked example: CrI3 24×24 at P64

The bispinor QSGW deck of claim 4083: P = 64 on 8×8 (16 nodes of A100-80GB),
`memory_per_device_gb = 72` so $\mathcal B = 72.0$ GB, `linalg = local`,
$n_q = 61$ parents, charge basis $n_C = 3328$, current basis $n_T = 1728$.

**A small solve stays whole.** One charge-basis matrix is
$s\,3328^2 = 177$ MB. With $\lceil 61/64\rceil = 1$ parent per device, every
per-q solve of that size runs whole on its device, and 3 of the 64 devices
idle.

**A large pencil cannot.** The TT sector's reduction pencil at its conservative
side $R = 32000$ is $sR^2 = 16.4$ GB per matrix and is held several times over,
so one whole TT parent per device would need 155.8 GB against 72.0 (CT 212.7, CC
71.1). The sector constructor therefore runs on the face, one round of all 61
parents ([bispinor sectors §8](bispinor_shared_pole_w.md#sector-byte-budget)).

**Its eighs still run whole.** The largest stack is the TT reduced eigh,
$m = 20480$. On the mesh it would hold $s\,m^2/P = 105$ MB per matrix per
device and run 61 sequential distributed solves. On route (c) each device holds
one whole matrix, $8 m^2 s = 53.7$ GB by (D 3), beside a 16.4 GB boundary:
70.1 GB before the held panels, inside 72.0. All eight stacks of the round take
route (c) (claim 4083). The gain is the concurrency: on the same deck, CT with
route-(c) eighs took 155.7 s at map 0 against 1251.5 s on the whole-mesh scan
(claim 3706, main 56a24169c).

**The GEMMs stay distributed.** The reduction's products are face GEMMs on
`panel_matmul`, in stages of 4 (TT and CT) and 16 (CC) parents, the widest
halving of 61 whose program fits beside the stacks. At these shapes they run at
9.2–12.4 TF/s per A100 (claim 3425).

**The inverse roots are one step.** The paired metric's Newton–Schulz
correction leaves $\max\|ZAZ - I\|_F/\sqrt R$ at $8.2$–$8.9\cdot10^{-17}$ (claim
4083).
