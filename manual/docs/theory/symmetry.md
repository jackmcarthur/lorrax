# Symmetry

`symmetry_maps` Space-group operations and time reversal: the irreducible k and q wedges, and the transport of ψ, G, ζ, V, W and Σ back to the full grid.
{: .page-summary }

<div class="section-cards"></div>

## <span class="n">1</span> Operations and wedges {: #sym-ops }

LORRAX keeps wavefunctions only at the k rows stored in `WFN.h5`, the parents. Every full-grid object is obtained from a parent by one space-group operation, possibly composed with time reversal. No deck key turns symmetry on or off: a run uses the operations its `WFN.h5` stores.
{: .intro }

!!! note
    The NSCF k list may be QE's own reduced `K_POINTS automatic` grid, reduced by every operation QE used: the space group with its fractional translations, time reversal on a nonmagnetic crystal, and the time-reversal-composed operations of a magnet. LORRAX unfolds it to the full grid itself (§1.4), so BerkeleyGW's `kgrid.x` is not needed ([Inputs from DFT](../using/inputs-from-dft.md#kgrid)). A full-grid k that no permitted operation reaches refuses (§6.4).

### <span class="n">1.1</span> Operations used {: #sym-used }

`symmetry_maps.SymMaps.active_symmetry_rows`
{: .in-code }

Every operation of the WFN header is used, symmorphic or not, and with it time reversal as far as the mean field has it:

| Operation | Example | Used for | Condition |
|---|---|---|---|
| identity and proper rotations, $\det R_s = +1$ | $C_{4z}$, $C_3$ about [111] | k unfolding, the q wedge, transport of ψ, $G$, ζ, $V$, $W$ and Σ; centroid orbits | stored in `WFN.h5` |
| inversion and other improper rotations, $\det R_s = -1$ | $I$, a mirror, $S_6$ | the same; the spinor factor is that of the proper part $\det(R_s)R_s$ (§3.3) | stored in `WFN.h5` |
| non-symmorphic, $\{R_s\vert\boldsymbol\tau_s\}$ with $\boldsymbol\tau_s$ off the lattice | screw axes, glide planes; Si (Fd$\bar3$m) | the same; $\boldsymbol\tau_s$ enters as the phase of Eq. (3.1) and through $\alpha_s$, $\mathbf L_{s\mu}$ (Eq. 3.2) | stored in `WFN.h5`; a $\boldsymbol\tau_s$ not commensurate with the FFT grid runs unreduced (§5.1) |
| time reversal $\Theta$, alone and composed with every stored operation | $\bk\to-\bk$ in a crystal without inversion | k unfolding; $\bq$ and $-\bq$ from one parent (§4.3); antiunitary transport (§4.2) | time reversal holds (§2): always for a scalar WFN, measured for a spinor WFN |
| magnetic, $\Theta g_s$ for the operations QE marks `t_rev = 1` | bcc Fe with spin–orbit coupling: 8 unitary, 8 composed with $\Theta$ | k unfolding, the q wedge, antiunitary transport, on those rows only | time reversal broken and a bound QE schema (§2.2) |

The table of §1.3 always holds both rows of each operation; `active_symmetry_rows` selects which may be used. With time reversal broken, $\Theta$ alone is never used, even when QE's flags would have allowed $\bk\leftrightarrow-\bk$ folding, and without a bound schema every stored operation is taken as unitary. An operation absent from the header is never added. Centroid orbits are closed under the spatial operations only, since $\Theta$ leaves positions fixed (§5). Magnetism enters only as a noncollinear spinor WFN; a collinear spin-polarized one refuses at load (§3.3).

### <span class="n">1.2</span> Seitz operations {: #sym-seitz }

`symmetry_maps.SymMaps`
{: .in-code }

For each spatial operation $s = 0,\dots,n_t-1$, `WFN.h5` stores an integer matrix $M_s$ (`mtrx`) and a translation $\mathbf t_s$ (`tnp`), both in crystal coordinates. With $\boldsymbol\tau_s = \mathbf t_s/2\pi$, the operation is $g_s = \{R_s|\boldsymbol\tau_s\}$ with $R_s = M_s^{-1}$, and it acts as:

$$
\br \mapsto R_s\br + \boldsymbol\tau_s, \qquad
g_s^{-1}\br = M_s(\br - \boldsymbol\tau_s), \qquad
\bk \mapsto S_s\bk, \quad S_s = M_s^{\mathsf T} \tag{1.1}
$$

Real-space maps use $\boldsymbol\tau_s$; reciprocal-space phases use the raw $\mathbf t_s$. Products compose as $\{R_a|\boldsymbol\tau_a\}\{R_b|\boldsymbol\tau_b\} = \{R_aR_b|R_a\boldsymbol\tau_b+\boldsymbol\tau_a\}$ modulo a lattice vector.

### <span class="n">1.3</span> Antiunitary rows {: #sym-antiunitary }

`WFN.h5` does not record which operations QE composed with time reversal $\Theta$. Each operation therefore has two candidate rows: $s$, the unitary $g_s$, and $n_t+s$, the antiunitary $\Theta g_s$:

$$
S_{n_t+s} = -S_s, \qquad
\Theta = i\sigma_y K \;\;(n_s = 2), \qquad
\Theta = K \;\;(n_s = 1) \tag{1.2}
$$

$K$ is complex conjugation and $n_s$ the number of spinor components. $\Theta$ leaves positions fixed, so both rows share the real-space action of $g_s$. The rows a run may select are all $2n_t$ when time reversal holds; otherwise the QE-typed row of each operation, or only the unitary half when no QE schema is bound ([§2](#sym-trs)). The run report prints the count as `Active op rows`.

### <span class="n">1.4</span> Parent k {: #sym-parent-k }

`symmetry_maps.map_full_kpoints_to_irreducible`
{: .in-code }

The full grid is generated from the WFN's `kgrid` and `shift`. Each full-grid point gets a parent WFN row $p$ and an operation row $s$, matched in fractional coordinates to $10^{-6}$:

$$
\bk = S_s\bar\bk_p + \mathbf g_0, \qquad \mathbf g_0 \in \mathbb Z^3 \tag{1.3}
$$

Among matches the highest stored row wins, then the lowest operation row. The rule is frozen, because changing it moves results. Parents are the WFN rows as written, with no rotation and no gauge fixing. A WFN that stores two k of one orbit leaves the lower row with no children, and that row is never read.

!!! note
    An energy or an occupation is constant on a star and may be read from any member. A wavefunction may not: members differ by a gauge, so ψ at a full-grid k is always the transported image of its own parent row.

### <span class="n">1.5</span> Irreducible q and the two k wedges {: #sym-wedges }

`symmetry_maps.find_irreducible_bz_points`
{: .in-code }

The q parents are one member per orbit of the unshifted mesh: the lexicographically smallest, reached by the lowest operation row, in integer arithmetic. Band-index quantities ($E_{n\bk}$, $\Sigma_{mn}$, $H^{\mathrm{QP}}$, $U$) use one of two reduced k sets:

| Wedge | Rows | Used by |
|---|---|---|
| file wedge | the stored WFN rows, `nk_red` of them | `.dat` outputs, `sigma_mnk.h5`, the Σ projection |
| star wedge | the first full-grid row of each orbit | `kin_ion.h5`, the QSGW loop under [`sc_on_ibz`](../drivers/gw/keywords.md#sc_on_ibz) |

The two coincide when every stored k lies in its own orbit (Si 4×4×4 with 48 operations: 64 → 8 and 8). They differ when the WFN stores two k of one orbit: a test WFN with 9 stored k in 5 orbits. Reducing to the file wedge and unfolding back is then not the identity.

## <span class="n">2</span> Time reversal {: #sym-trs }

Whether $\Theta$ is a symmetry of the mean field is measured once per WFN and read by every consumer. No key or environment variable can assert it.
{: .intro }

### <span class="n">2.1</span> The verdict {: #sym-trs-verdict }

`symmetry_maps.check_spinor_reference_trs`
{: .in-code }

A scalar WFN ($n_s = 1$) has a real Kohn–Sham Hamiltonian, and time reversal holds without a measurement. A two-component WFN has time reversal only if the bound QE schema types no operation with `t_rev = 1` and the SCF absolute magnetization is below $10^{-4}\,\mu_B$ per cell. When both hold, a guard compares occupied one-particle density operators $D_\bk = \sum_n f_{n\bk}\,|\psi_{n\bk}\rangle\langle\psi_{n\bk}|$ at Kramers partners, in G space:

$$
r = \frac{\big\| D_{\bk'} - \Theta D_{\bk}\Theta^{-1} \big\|_F}{\big(\|D_{\bk'}\|_F^2 + \|D_{\bk}\|_F^2\big)^{1/2}}, \qquad r \le 10^{-6} \tag{2.1}
$$

The partner pairs are stored $\bk/{-\bk}$ rows, spatial images of a stored partner, and time-reversal-invariant points, at most 12 k. $r$ is invariant to band phases and to rotations inside degenerate blocks. With QE's moment exactly 0 a failed pair refuses, because $\Theta$ is then exact and the WFN is inconsistent; at $0 < |m| < 10^{-4}\,\mu_B$, or when only a spatial image fails, time reversal is switched off with a warning.

!!! note
    A completed verdict is stamped in the user cache (`wfn_trs`), keyed on the file, a digest of its header and the tolerances, so a rerun on the same WFN skips the measurement. `LORRAX_TRS_CHECK=strict` makes a broken or inconclusive verdict refuse; `LORRAX_TRS_TOL` and `LORRAX_TRS_MAX_K` set the tolerance and the k cap ([`LORRAX_TRS_CHECK`](../using/environment.md#lorrax_trs_check)).

### <span class="n">2.2</span> QE schema provenance {: #sym-schema }

`symmetry_maps.resolve_qe_symmetry_binding`
{: .in-code }

The loader searches near `WFN.h5` for a QE `data-file-schema.xml`; where to stage it is on [Inputs from DFT](../using/inputs-from-dft.md#schema). A `WFN_qp.h5` also searches beside its stamped source WFN. A candidate binds only if it matches the WFN in:

| Checked | Against |
|---|---|
| `kgrid` | the WFN header |
| spinor count | `nspinor` |
| stored k coordinates | every WFN row, modulo a reciprocal lattice vector |
| rotation matrices | `mtrx`, row by row (a transposed match is named as an axis bug) |
| translations | `tnp`, modulo a lattice vector |

Without a binding every stored operation is taken as unitary, time reversal is off on a two-component WFN, and `SymMaps` warns `SYMMETRY PROVENANCE WARNING`. Two binding schemas that disagree on operation typing or spin–orbit refuse. The run report prints `QE schema`, `Stored QE type`, `Active op rows` and `Global TRS`.

### <span class="n">2.3</span> What the verdict decides {: #sym-trs-consumers }

`symmetry_maps.SymMaps.trs_allowed`
{: .in-code }

Every consumer reads the one boolean `SymMaps.trs_allowed`; none accepts an override.

| Consumer | Time reversal holds | Time reversal broken |
|---|---|---|
| k map | all $2n_t$ rows | QE-typed rows; a WFN reduced by $\bk \leftrightarrow -\bk$ refuses |
| q axis | $\bq$ and $-\bq$ share one spatial row (§4.3) | $\bq$ and $-\bq$ are independent parents |
| χ⁰ response store | unordered: $\chi^{\mathsf T}$ and $\chi$ coincide | ordered: the physical orientation |
| GN-PPM fit | one residue per pole | a second, odd residue ([PPM](frequency-treatments.md#ppm-odd)) |
| `hl_ppm` | runs | refuses ([Frequency treatments](frequency-treatments.md#ppm-odd)) |
| `screening_diagrams = w_bse` | runs | refuses (`GATE w_bse_requires_measured_trs`) |
| `bispinor_gw = full_tangential` | refuses (`GATE full_tangential_trs`) | runs |
| `head_correction = full`, tangential | runs | runs with the signed Γ body; refuses on a metal ([q → 0 head](q0-head.md)) |

## <span class="n">3</span> Wavefunctions {: #sym-psi }

Children are transported from parents, never stored. The loader transports plane-wave coefficients; the GW kernels transport values at the centroids.
{: .intro }

### <span class="n">3.1</span> Plane-wave coefficients {: #sym-psi-pw }

`symmetry_maps.unfold_psi`
{: .in-code }

On the unitary row $s$, with $\bk = S_s\bar\bk + \mathbf g_0$, and on the antiunitary row $n_t+s$, with $\bk = -S_s\bar\bk + \mathbf g_0$, the child coefficients are:

$$
c_{\bk}(S_s\bG - \mathbf g_0) = \mathcal U_s\, e^{-i(S_s\bG)\cdot\mathbf t_s}\, c_{\bar\bk}(\bG), \qquad
c_{\bk}(-S_s\bG - \mathbf g_0) = \mathcal U_{n_t+s}\, e^{+i(S_s\bG)\cdot\mathbf t_s}\, \overline{c_{\bar\bk}(\bG)} \tag{3.1}
$$

$\mathcal U$ is the spinor factor of §3.3. On the antiunitary row $\Theta$ has two halves, the spinor factor and the negation of the G list. Applying one without the other replaces $\psi(\br)$ by $\overline{\psi(-\br)}$, which preserves norms, overlaps and the kinetic energy and shows only in position-dependent terms such as $V^{\mathrm{ion}}$ and $V^{\mathrm{NL}}_\bk$. The coefficients are conjugated before the phase is applied. The loader omits the band-independent factor $e^{-2\pi i(S_s\bar\bk)\cdot\boldsymbol\tau_s}$; it is a gauge of $\psi_\bk$ and cancels in every $G_\bk$ and every same-k band matrix.

### <span class="n">3.2</span> At the centroids {: #sym-psi-centroids }

`symmetry_maps.centroid_source_map_and_wrap`
{: .in-code }

Each operation pulls a centroid back onto another centroid plus a lattice vector. This defines the source map $\alpha_s$ and the wrap $\mathbf L_{s\mu}$, and with them the child at the centroids:

$$
M_s(\br_\mu - \boldsymbol\tau_s) = \br_{\alpha_s(\mu)} + \mathbf L_{s\mu}, \qquad
\psi_{n\bk}(\br_\mu) = \mathcal U_s\,\mathcal T_s\Big[e^{2\pi i\bar\bk\cdot\mathbf L_{s\mu}}\,\psi_{n\bar\bk}(\br_{\alpha_s(\mu)})\Big] \tag{3.2}
$$

$\mathcal T_s$ is complex conjugation on an antiunitary row and the identity otherwise. Centroids are FFT-grid points and every operation is commensurate with the grid, so $\alpha_s$ is an exact permutation of a closed set. The image is rounded to grid integers before the wrap is split off, so rounding noise never turns a zero wrap into $-1$. Kernels gather with $\alpha_s$ as tabulated; $\alpha_s^{-1}$ is the table of $g_s^{-1}$, and the two differ for an order-3 operation.

### <span class="n">3.3</span> Spinor factors {: #sym-spinor }

`symmetry_maps.spinor_rotation_for_sym_row`
{: .in-code }

| Representation | Unitary row $\mathcal U_s$ | Antiunitary row $\mathcal U_{n_t+s}$ |
|---|---|---|
| scalar, $n_s = 1$ | $1$ | $1$ |
| Pauli, $n_s = 2$ | $U_s$ | $i\sigma_y\overline{U_s}$ |
| four-component, $n_s = 4$ | $\mathrm{diag}(U_s,\ \det R_s\,U_s)$ | $\mathrm{diag}(i\sigma_y\overline{U_s},\ \det R_s\,i\sigma_y\overline{U_s})$ |

$U_s \in SU(2)$ is the spinor image of the proper rotation $\det(R_s)R_s$ in Cartesian coordinates. $n_s = 2$ means noncollinear, with or without spin–orbit coupling: the code branches on the spinor axis, never on SOC. The four-component case is the [`bispinor`](../drivers/gw/keywords.md#bispinor) route. A collinear spin-polarized WFN (`nspin = 2`) refuses at load.

## <span class="n">4</span> Two-point operators {: #sym-two-point }

$G$, $\chi^0$, $V$, $W$ and Σ are transported in the centroid basis by a double gather with wrap phases. On antiunitary rows a static Hermitian operator is conjugated and an operator at a fixed complex time or frequency is transposed.
{: .intro }

### <span class="n">4.1</span> Unitary rows {: #sym-op-unitary }

`symmetry_maps.unfold_isdf_operator`
{: .in-code }

Let $X_{\mathbf p}$ be a two-point operator with Bloch vector $\mathbf p$: $\mathbf p = \bk$ for $G_\bk$, $\mathbf p = \bq$ for $\chi^0_\bq$, $V_\bq$ and $W_\bq$. Write $X_{\mathbf p}(\mu,\nu)$ for its centroid matrix, the point pair $(\br_\mu,\br_\nu)$ for $G$ and $\chi^0$ and the ζ pair for $V$ and $W$. On a unitary row:

$$
X_{S_s\bar{\mathbf p}}(\mu,\nu) = e^{2\pi i\bar{\mathbf p}\cdot(\mathbf L_{s\mu}-\mathbf L_{s\nu})}\;\mathcal U_s\,X_{\bar{\mathbf p}}\big(\alpha_s(\mu),\alpha_s(\nu)\big)\,\mathcal U_s^\dagger \tag{4.1}
$$

$\mathcal U_s$ acts on the spin index of each endpoint. The translation enters only through $\alpha_s$ and $\mathbf L_s$; no separate phase appears in the centroid basis.

### <span class="n">4.2</span> Antiunitary rows {: #sym-op-antiunitary }

A static Hermitian operator ($V_\bq$, static $W_\bq$) is conjugated (`trs_rule = conj`). An operator at a fixed complex time or frequency ($G_\bk(t)$, the residues of $W^c$) is transposed (`pair_transpose`), because its weight $e^{-iEt}$ or $1/(z-E)$ multiplies $\psi\psi^\dagger$ and is not conjugated:

$$
X_{-S_s\bar{\mathbf p}}(\mu,\nu) = e^{-2\pi i\bar{\mathbf p}\cdot(\mathbf L_{s\mu}-\mathbf L_{s\nu})}\,\overline{X_{\bar{\mathbf p}}\big(\alpha_s(\mu),\alpha_s(\nu)\big)}, \qquad
G_{\Theta\bk}(\br,\br';t) = \Theta\,G_{\bk}(\br',\br;t)^{\mathsf T}\,\Theta^{-1} \tag{4.2}
$$

The transpose rule reads a partner $G^{\mathsf T}_{\bar\bk} = \sum_n w_n\,\overline{\psi_n}\,\psi_n^{\mathsf T}$, built from conjugated faces and unconjugated weights; it equals $\overline{G_{\bar\bk}}$ only for real weights. Conjugating instead would send $G(t)$ to $G(-t)$ and flip the sign of $\mathrm{Im}\,\Sigma$, the lifetime. Tangential residues use the transpose rule; pole positions carry no wrap phase. A rectangular operator between two centroid families (charge × current) uses each family's own $(\alpha,\mathbf L)$ on its endpoint.

### <span class="n">4.3</span> ζ, V and the q axis {: #sym-op-q }

`gw.qgrid_symmetry.qgrid_trs_policy_for`
{: .in-code }

$\zeta$ is fitted and stored in G space on the parent q wedge ([ISDF §2.5](isdf.md#isdf-qirr)). Under row $s$ the child at $\bq+\bG' = S_s(\bar\bq+\bG)$ is:

$$
\zeta_{\bq\mu}(\bG') = e^{-iS_s(\bar\bq+\bG)\cdot\mathbf t_s}\,e^{-2\pi i\bar\bq\cdot\mathbf L_{s\mu}}\,\zeta_{\bar\bq\,\alpha_s(\mu)}(\bG) \tag{4.3}
$$

The child is conjugated as a whole on an antiunitary row. In $V_\bq(\mu,\nu)$ ([ISDF Eq. 3.1](isdf.md#isdf-vq)) the translation phase cancels and Eq. (4.1) follows. The child's $\bG'=0$ column in general comes from a nonzero parent $\bG$, so the head columns of the [q → 0 head](q0-head.md) are transported from the parent sphere.

$V_{-\bq} = \overline{V_\bq}$ holds for any mean field and checks the fit, not time reversal ([ISDF §3.3](isdf.md#isdf-head)). With time reversal, the $\bq$ and $-\bq$ children of one parent use the same spatial row, one of them composed with $\Theta$. At a self-negative $\bq$ (every time-reversal-invariant point of an even mesh, Γ alone on an odd one) $\Theta$ acts as the projector $A \mapsto (A+\overline A)/2$, and the removed part is printed. Without time reversal $\bq$ and $-\bq$ are independent parents and neither step runs. The run report states which, as `q-grid TRS policy`.

!!! note
    The transport assumes each parent tile is invariant under its own little group. That is exact for the continuum operator and approximate for a finite ISDF fit. The residual is measured at every parent with the transport's own formula and reported, so a fit error shows as a fit error.

### <span class="n">4.4</span> Band matrices and Σ {: #sym-op-band }

`symmetry_maps.star_broadcast`
{: .in-code }

A band matrix computed on a parent reaches the other members of its star by row gather. On antiunitary rows an operator that transforms like $G$ is transposed and an observable is conjugated:

$$
\Sigma_{\Theta\bk,mn}(z) = \Sigma_{\bk,nm}(z), \qquad O_{\Theta\bk,mn} = \overline{O_{\bk,mn}} \tag{4.4}
$$

On the diagonal the two differ by the sign of $\mathrm{Im}\,\Sigma$, so the dynamical $\Sigma^c$ written to the file wedge takes the transpose rule. A polar vector index such as the velocity transforms by the forward Cartesian rotation, with an extra $-1$ on antiunitary rows; an axial index by $\det(R)R$. Four-current indices transform by $\Lambda_s = \mathrm{diag}(1,R_s)$ on a unitary row and $\mathrm{diag}(1,-R_s)$ on an antiunitary one, after the scalar transport. A wrong conjugation leaves every Hermitian diagonal unchanged and shows only off the diagonal.

!!! note
    A band window whose edge splits a degenerate multiplet is not a representation of the little group, and Σ then differs across a star: about 2 meV on a Si COHSEX test deck, exactly 0 with a clean edge. The ζ-fit window ends at the [`number_bands`](../drivers/gw/keywords.md#number_bands) edge and refuses a split multiplet there; the band-sum check on the same edge only warns unless the edge is named by `number_bands_chi` or `number_bands_sigma` ([Band counts](../using/inputs-from-dft.md#bands)).

## <span class="n">5</span> Orbit-closed centroid sets {: #sym-closure }

Transport in the centroid basis needs every operation to permute the centroid set. The [centroids](../utilities/centroids.md) utility selects closed sets; the GW run measures closure and runs unreduced when it fails.
{: .intro }

### <span class="n">5.1</span> The closure test {: #sym-closure-test }

`symmetry_maps.verify_centroid_orbit_closure`
{: .in-code }

For every spatial operation of the WFN, the run measures how far the pulled-back centroids land from the set, in fractional coordinates with the minimum image:

$$
d_s = \max_\mu\,\min_\nu\,\big\| M_s(\br_\mu-\boldsymbol\tau_s) - \br_\nu \big\|_{\mathbb T^3}, \qquad
\text{closed} \iff d_s \le 10^{-5}\ \text{for every } s \tag{5.1}
$$

The [selector](centroid-selection.md#cs-symmetry) closes its points under the atomic space group of the crystal. A magnetic WFN, whose group is a subgroup of it, sees a closed set. A closed set can still fail to give tables when $N\boldsymbol\tau$ is not integral on the FFT grid or two centroids collide under rounding; the run reports that as its own reason.

### <span class="n">5.2</span> The unreduced run {: #sym-trivial }

`symmetry_maps.SymMaps.trivial_view`
{: .in-code }

When the charge set, or on a bispinor run the current set, is not closed, the run warns with the file name and restricts its computational group to the identity. Every full-grid k is its own parent, realized from the WFN by Eq. (3.1), and every q is computed. The restricted group uses no antiunitary row but keeps the measured time-reversal verdict, because time reversal is a property of the Hamiltonian. File outputs still reduce through the WFN's own map.

The unreduced run costs a larger restart bundle and a Σ that spreads across a k star ([Centroid selection §5.2](centroid-selection.md#cs-closure-check)). Regenerate the set with the default, symmetric selection of the [centroids utility](../utilities/centroids.md#-no-symmetries) to restore the reduction.

!!! refuses
    - `screening_diagrams = w_bse` or `w_rpa_resolvent` with a set that is not closed (`GATE resolvent_ladder_trivial_view`). The resolvent ladder solves on the reduced q wedge. Regenerate the set, or use `w_rpa`.

### <span class="n">5.3</span> Orbit-packed layout {: #sym-packed }

`common.centroid_basis.PackedCentroidBasis`
{: .in-code }

On the square $s\times s$ process mesh every centroid axis is held in one orbit-packed order. Whole orbits are placed on the $s$ shards of the axis, largest first onto the least-loaded shard, and each shard is padded at its tail to a common size. Every $\alpha_s$ then maps each shard into itself, so Eq. (4.1) on an operator sharded over $(\mu,\nu)$ is a gather inside each rank's tile, with no collective.

Pad slots are fixed points of every $\alpha_s$, with zero wrap, and hold exact zeros. Files keep the canonical centroid order at the logical extent; the conversion happens only where bytes cross a file. A non-square mesh, or a map that leaves its shard, refuses.

## <span class="n">6</span> Where symmetry is used {: #sym-use }

Symmetry replaces per-k and per-q linear algebra and storage, never a lattice convolution. Which operations a run uses is fixed by `WFN.h5`; the deck keys only choose storage.
{: .intro }

### <span class="n">6.1</span> Work on the wedge {: #sym-work }

| Stage | Domain | Role of symmetry |
|---|---|---|
| ψ load | parent k | children transported, never stored |
| Green's functions for χ⁰ and Σ | parent k | built on parents; the k-convolution unfolds them on its load |
| ζ fit, $V_\bq$, Dyson solve, pole fits | parent q | solved on parents, transported to all q |
| k and q convolutions (χ⁰, Σ) | full grid | none: inputs are transported first |
| Σ band projection | parent k | band matrix transported by row gather (§4.4) |
| density, $V^{\mathrm H}$ | real-space grid | symmetrized by the FFT-grid pull-back |

The savings are $N_k/n_{\mathrm{parent}}$ and $N_q/N_q^{\mathrm{irr}}$ (Si 4×4×4 with 48 operations: 64 → 8 for both). A lattice convolution needs every point of a uniform grid, so its cost does not change.

### <span class="n">6.2</span> What each file stores {: #sym-files }

| File | k or q set | Declared by |
|---|---|---|
| `eqp0.dat`, `eqp1.dat`, `sigma_diag.dat` | file wedge | the k coordinates of each block |
| `sigma_mnk.h5` | file wedge | the star spread measured on the full grid before reduction |
| `kin_ion.h5` | star wedge | `k_storage = ibz`, `irr_idx_k`, `sym_idx_k`; no attribute means full grid |
| `qp_wfn_rotations.h5` | file wedge when the round trip reproduces the arrays, otherwise full | `k_storage` |
| `isdf_tensors_<N_mu>.h5` (`V_qmunu`, `W0_qmunu`) | q parents when the set is closed and q reduced | `q_storage` and the unfold tables |
| `zeta_q.h5` | q parents or full | the q extent |
| `dipole.h5` | full grid | none |
| `WFN.h5`, `WFN_qp.h5` | file wedge | the BerkeleyGW header |

A q-parent tensor carries its own tables (`irr_idx_q`, `sym_idx_q`, `sym_perm`, `L_table`) in canonical centroid order. Its reader refuses a digest, version or table mismatch and never re-derives tables from the current run.

### <span class="n">6.3</span> Keys {: #sym-keys }

No key switches symmetry off. To run without it, write the WFN without it: [`nosym = true`](../drivers/nscf/keywords.md#nosym) in `nscf`, or QE `nosym` with `noinv`.

| Key | Effect on symmetry |
|---|---|
| [`centroids_file`](../drivers/gw/keywords.md#centroids_file) | Its closure decides between the reduced and the unreduced run (§5). |
| [`centroids_file_current`](../drivers/gw/keywords.md#centroids_file_current) | The bispinor current set; its closure enters the same decision. |
| [`--no-symmetries`](../utilities/centroids.md#-no-symmetries) | Centroid selection without the crystal's symmetries; symmetric selection is the default. |
| [`restart_q_storage`](../drivers/gw/keywords.md#restart_q_storage) | q set of the restart bundle. |
| [`qp_rotations_k_storage`](../drivers/gw/keywords.md#qp_rotations_k_storage) | k set of `qp_wfn_rotations.h5`. |
| [`sc_on_ibz`](../drivers/gw/keywords.md#sc_on_ibz) | QSGW tables on the star wedge. |
| [`bgw_vcoul_sym_wfn`](../drivers/gw/keywords.md#bgw_vcoul_sym_wfn) | Supplies the group that folds q onto BerkeleyGW's list for a WFN without symmetry; comparison runs only. |

### <span class="n">6.4</span> Refusals {: #sym-refusals }

`symmetry_maps.SymMaps`
{: .in-code }

The action is refused, never approximated: a missing k is never replaced by Γ or by the identity.

!!! refuses
    - `kgrid` without three positive extents, or more stored k than grid points. Fix the WFN header.
    - A stored k off the grid of its own `kgrid` and `shift` by more than $10^{-6}$. Fix the k list or `kgrid`/`shift`.
    - `ntran = 1` with every grid point stored but a non-identity operation, or with stored k that do not cover the mesh.
    - A full-grid k that the allowed rows cannot reach. With a bound schema its operations are incomplete. With time reversal broken and no bound schema: for a nonmagnetic reference co-stage the NSCF and SCF `.save` beside `WFN.h5`; for a magnet stage the NSCF schema, which types the operations QE reduced with (§2.2).
    - Two QE schemas that both match the WFN but disagree on operation typing or spin–orbit. Keep only the WFN-generating NSCF `.save`.
    - A two-component WFN whose schema reports zero moment and no `t_rev` row, but whose Kramers partners fail Eq. (2.1) (`GATE trs_qe_nonmagnetic_wfn_consistent`).
    - `LORRAX_TRS_CHECK = 0` or `off`; with `strict`, a broken or inconclusive verdict.
    - `restart_q_storage = ibz` with a set that is not closed, or with a q axis that did not reduce. Use `auto`.
    - A q-parent tensor whose stored tables, digest or version do not match. Regenerate the file.

## Notation

| Symbol | Meaning |
|---|---|
| $n_t$, $s$ | number of spatial operations in `WFN.h5`; an operation row ($s \ge n_t$: antiunitary) |
| $M_s$, $\mathbf t_s$ | `mtrx` and `tnp` of operation $s$, crystal coordinates |
| $\{R_s\vert\boldsymbol\tau_s\}$ | Seitz operation, $R_s = M_s^{-1}$, $\boldsymbol\tau_s = \mathbf t_s/2\pi$ |
| $S_s$ | reciprocal action $M_s^{\mathsf T}$; $S_{n_t+s} = -S_s$ |
| $\Theta$, $K$ | time reversal; complex conjugation |
| $\bar\bk_p$, $\mathbf g_0$ | parent k of WFN row $p$; umklapp vector |
| $N_k$, $n_{\mathrm{parent}}$, $N_q^{\mathrm{irr}}$ | full-grid k; parent k; parent q |
| $n_s$, $U_s$, $\mathcal U_s$ | spinor components; SU(2) image of $\det(R_s)R_s$; spinor factor of row $s$ |
| $\alpha_s(\mu)$, $\mathbf L_{s\mu}$ | centroid source map; lattice wrap |
| $\mathcal T_s$ | conjugation on an antiunitary row, identity otherwise |
| $X_{\mathbf p}(\mu,\nu)$ | two-point operator with Bloch vector $\mathbf p$ in the centroid basis |
| $D_\bk$, $f_{n\bk}$ | occupied one-particle density operator; occupation |
| $d_s$ | closure residual of operation $s$ |
| $\Lambda_s$ | four-current action $\mathrm{diag}(1,\pm R_s)$ |

## References

1. C. J. Bradley and A. P. Cracknell, *The Mathematical Theory of Symmetry in Solids* (Clarendon Press, Oxford, 1972).
2. J. Deslippe, G. Samsonidze, D. A. Strubbe, M. Jain, M. L. Cohen, and S. G. Louie, Comput. Phys. Commun. 183, 1269 (2012).
3. P. Giannozzi, O. Andreussi, T. Brumme, et al., J. Phys.: Condens. Matter 29, 465901 (2017).

