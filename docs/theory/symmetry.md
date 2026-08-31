# Symmetry conventions and unfolding

Symmetry reduces non-FFT work and irreducible-zone storage. It does not alter
the physical operator: every reduced object has an explicit inverse map to the
full zone, and every map follows BerkeleyGW's stored convention.

## 1. BerkeleyGW convention

For one spatial operation, let

$$
S=\mathrm{mtrx},
\qquad
\boldsymbol\tau=\mathrm{tnp}/(2\pi).
$$

In column notation the four useful actions are

| object | action |
|---|---|
| direct-space forward | \(\mathbf r'=S^{-1}\mathbf r+\boldsymbol\tau\) |
| direct-space inverse | \(\mathbf r'=S(\mathbf r-\boldsymbol\tau)\) |
| reciprocal forward | \(\mathbf k'=S\mathbf k\) |
| reciprocal inverse | \(\mathbf k'=S^{-1}\mathbf k\) |

The raw `translations` array retains BerkeleyGW units \(2\pi\tau\).
Real-space geometry divides by \(2\pi\); wavefunction phases consume the raw
array directly.

LORRAX stores integer k-grid tuples as rows. `SymMaps.sym_mats_k` is therefore
the row-form table

$$
\mathrm{sym\_mats\_k}=[S^\mathsf T,-S^\mathsf T],
$$

where the second half appends time reversal. Integer modular arithmetic
constructs the complete k/q star maps, so star membership does not depend on a
floating tolerance.

At the QE boundary, XML `rotation` text is reshaped without transposition and
must equal the raw WFN `mtrx` row. Reciprocal actions transpose it once later.
QE's affine translation is converted to WFN units as
(2\pi S^{-1}\tau_{\rm QE}), modulo a lattice vector. The schema/WFN binding
checks both arrays, the stored k rows, grid, and spinor count; a transposed
matrix stack is rejected with a major/minor-axis diagnostic.

The corresponding direct-space composition law is

$$
S_c=S_aS_b,
\qquad
\boldsymbol\tau_c
=S_a^{-1}\boldsymbol\tau_b+\boldsymbol\tau_a
\pmod{\mathbb Z^3}.
$$

This convention is load bearing for nonsymmorphic translations. Symmorphic
systems can conceal a forward/inverse mistake because inverse operations
generate the same orbit.

## 2. Unitary and antiunitary rows

`WFN.h5` stores Seitz matrices and translations but not QE's per-operation
`time_reversal` bit. LORRAX therefore treats
\(s\) and \(n_{\rm tran}+s\) as a candidate pair: the first is the unitary
action \(\{S|\tau\}\), the second the antiunitary action
\(T\{S|\tau\}\), where

$$
T=i\sigma_yK,
\qquad
i\sigma_y=
\begin{pmatrix}
0&1\\-1&0
\end{pmatrix}.
$$

Time reversal changes \(\mathbf k\to-\mathbf k\) and complex-conjugates the
wavefunction; it does not move real-space centroids. The augmented centroid
permutation and lattice-wrap tables therefore duplicate their spatial half,
while wavefunction and tensor consumers apply conjugation explicitly.

`WfnLoader.symmetry()` binds a nearby QE `data-file-schema.xml` only when it
authenticates the same WFN. If global TR is broken, each raw WFN row selects
exactly the unitary or antiunitary member named by that receipt. If global TR
holds, both members are valid. Without a matching schema, initialization
prints a loud warning: operation typing is unknown and a TR-broken result is
unsafe if QE used a magnetic antiunitary operation.

The public `symmetry_maps` service owns these tables and their refusal
contracts. Callers use the package door, not private submodules.

## 3. Wavefunction unfolding

Let \(\bar{\mathbf k}\) be an irreducible point and
\(\mathbf k=S\bar{\mathbf k}+\mathbf k_{g0}\). For a spatial row,

$$
\psi_{\mathbf k}(\mathbf G_{\mathrm{rot}})
=U_s\,
e^{-i(S\mathbf G)\cdot\mathrm{tnp}_s}
\psi_{\bar{\mathbf k}}(\mathbf G),
$$

where \(U_s\) is the spinor rotation and
\(\mathbf G_{\mathrm{rot}}=S\mathbf G+\mathbf k_{g0}\). The wavefunction
loader owns the rotated G-list and umklapp; `unfold_psi` owns only the phase,
spin rotation, and time reversal.

For a time-reversal row,

$$
\psi_{TS\bar{\mathbf k}}(\mathbf G_{\mathrm{rot}})
=
\left(i\sigma_y\overline{U_s}\right)
e^{+i(S\mathbf G)\cdot\mathrm{tnp}_s}
\overline{\psi_{\bar{\mathbf k}}(\mathbf G)}.
$$

Conjugation is applied before the stored phase. Reversing that order reverses
the phase a second time. The spinor table contains spatial rotations only;
the \(i\sigma_y\overline U_s\) row is constructed at the unfold site.

## 4. Centroid permutation and lattice wrap

For centroid \(\mathbf r_\mu\), the inverse direct-space action defines

$$
S_s(\mathbf r_\mu-\boldsymbol\tau_s)
=\mathbf r_{\alpha_s(\mu)}+\mathbf L_{s\mu},
\qquad
\mathbf L_{s\mu}\in\mathbb Z^3.
$$

The source permutation \(\alpha_s\), not its inverse, is used by the
\(V_q\) unfold. It matters for operations of order greater than two.

Centroids lie on the FFT grid. The transformed coordinate is therefore
snapped to an FFT-grid integer before integer floor division determines
\(\mathbf L\). Applying floating `floor` directly near a cell boundary can
invent a wrap and hence an order-one Bloch phase.

Validation requires every transformed point to find exactly one centroid and
every row of \(\alpha_s\) to be a permutation. Failure means that the centroid
set is not orbit closed or is incompatible with the FFT grid. The q-reduced
Coulomb path then falls back to a full-zone calculation; it never uses an
incomplete symmetry table.

Centroid closure uses the decorated atoms' spatial Seitz group, independent of
electronic, magnetic, and time-reversal authorization. Wavefunction unfolding
and \(V_q\) continue to use the authenticated electronic group.

## 5. Matrix unfolding

Let full-zone \(q\) have irreducible parent \(\bar q\) and symmetry row \(s\).
For a centroid-basis matrix,

$$
V_q[\mu,\nu]
=
e^{2\pi i\bar{\mathbf q}\cdot
(\mathbf L_{s\mu}-\mathbf L_{s\nu})}
V_{\bar q}[\alpha_s(\mu),\alpha_s(\nu)].
$$

A time-reversal row complex-conjugates the spatial result. The same
double-permutation and phase structure applies to other bilinear
centroid-basis tensors when their physics declares the same covariance.

The implementation preserves `P(None,'x','y')`: each centroid gather acts
along one already-sharded matrix axis, and the output retains both matrix
axes distributed. Padded centroid slots use identity permutations and zero
wraps.

Bispinor \(V_q^{ij}\) first unfolds each tile with the scalar rule. Its two
Pauli-vector indices then use the same row's axial, time-odd action. The two
time-reversal signs cancel; the scalar complex conjugation remains.

## 6. Where symmetry may reduce work

Symmetry is valid before operations that are covariant under the maps above:

- interpolation-vector fitting and G-flat storage;
- Coulomb matrix construction;
- complex-frequency chi/W slabs and fitted MPA pole slabs;
- other q-local matrix algebra.

The lattice k/q FFT convolution requires complete uniform grids. Its inputs
are unfolded before the FFT, and no time-reversal shortcut is applied inside
the convolution. This is why q-wedge storage can reduce I/O without changing
the shape of the expensive spatial kernel.

## 7. Required regression structure

No single material exercises the complete convention:

- a nonsymmorphic case is required to test translations and lattice wraps;
- an order-three operation is required to distinguish a permutation from its
  inverse;
- an SOC case with time-reversal-folded points is required to test
  \(i\sigma_yK\).

Unit tests pin group closure, exact integer star maps, centroid permutations,
wavefunction phases, and matrix unfolding. Observable sym-versus-nosym gates
then test the composed driver. Passing only a symmorphic material is not
evidence that the convention is correct.

The service contract and symbol index are in
[the symmetry_maps service page](../services/symmetry_maps.md). This chapter
owns the equations; the service page owns APIs and refusals.
