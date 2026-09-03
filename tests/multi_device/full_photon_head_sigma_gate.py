"""P4 gate for post-Dyson CC / CT+TC / TT photon-head Sigma attribution.

The mini-BZ geometry is read from a caller-supplied real WFN.  A small
charge-only response with a nonzero Hall coefficient is built by the
production producer (``gw.static_gauge_response.
build_static_photon_head_response``, its scalar direct response replaced by
a synthetic one) and completed by the canonical 4x4 Dyson/cubature path;
its retained factor carrier is then contracted through the production
static-photon Sigma loop.  V/W contain only that completed head in this
discriminator, so the aggregate Sigma diagonal is an independent oracle for
the sum of the three reported sectors.  The ``sigma_H = 0`` twin exercises
the absent-Hall-artifact default of the one packed mode.
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_TESTS))
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

if __name__ == "__main__":
    from runtime import initialize_communicator_stack
    _RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402


def _put(value, mesh, spec):
    return jax.device_put(
        np.asarray(value), NamedSharding(mesh, P(*spec)))


def _gather(value):
    if jax.process_count() == 1:
        return np.asarray(value)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(value, tiled=True))


def _levi_civita(b, a, i):
    if len({b, a, i}) != 3:
        return 0
    return 1 if (b, a, i) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1


def _hall_numpy(sigma_h):
    out = np.zeros((2, 4, 4), dtype=np.complex128)
    for a in range(2):
        for i in range(3):
            # -i: the persisted sigma_H is the occupied-bra Berry sum and
            # the live response is energy-ordered (head_correction owner).
            value = -1j * sum(
                _levi_civita(b, a, i) * sigma_h[b] for b in range(3))
            out[a, 0, i + 1] = value
            out[a, i + 1, 0] = np.conj(value)
    return out


def _direct_moment_oracle(receipt, sigma_h):
    """Independent NumPy spelling of the canonical 4x4 Dyson cubature."""
    hall = _hall_numpy(np.asarray(sigma_h, dtype=np.float64))
    per_order = []
    for chunk in receipt.chunks:
        n = int(chunk.physical_count)
        q = np.asarray(chunk.q_cart[:n], dtype=np.float64)
        D = np.asarray(chunk.D_raw[:n], dtype=np.complex128)
        weight = np.asarray(chunk.sample_weight[:n], dtype=np.float64)
        response = np.einsum("sa,aij->sij", q[:, :2], hall)
        lhs = np.eye(4, dtype=np.complex128)[None] - D @ response
        screened = np.linalg.solve(lhs, D)
        basis = np.column_stack((np.ones(n), q[:, :2]))
        per_order.append(
            np.einsum(
                "s,su,sij,sv->uvij", weight, basis, screened, basis,
                optimize=True)
            / np.sum(weight))
    return per_order[-1]


def _packed_index(layout, mesh, channel, logical):
    """Global packed index of one logical centroid, read off the packer.

    Padding is to a multiple of the device count, not the mesh side, so
    the index is not a closed-form offset; ask the canonical packer with a
    one-hot channel vector instead of hand-addressing the interleaving.
    """
    from gw.photon_layout import pack_photon_channel_vectors

    vectors = []
    for row in range(4):
        vector = np.zeros((1, layout.padded_extent(row)), dtype=np.complex128)
        if row == channel:
            vector[0, logical] = 1.0
        vectors.append(_put(vector, mesh, (None, "x")))
    packed = _gather(pack_photon_channel_vectors(
        tuple(vectors), layout, mesh, axis_name="x")[0][channel])
    hits = np.flatnonzero(np.abs(packed) > 0)
    if hits.size != 1:
        raise AssertionError(
            f"one-hot channel {channel} logical {logical} packed to {hits}")
    return int(hits[0])


def _response(mesh, layout, sigma_h, *, wfn, wfn_binding, meta):
    """Build the bounded response through the production producer.

    The scalar direct response is synthetic: zero ``S`` and nonzero charge
    wings on charge centroid 0 only, with ``Z = conj(Y)`` as the static
    Adler--Wiser weight requires.  The analytic body below couples that
    centroid only to charge centroid 1, so ``Y W Z`` is exactly zero and the
    folded ``S`` stays the Ward-closed zero tensor while ``YW``/``WZ`` are
    nonzero and the mixed ``(1,qx,qy)`` moments reach the packed body.
    """
    import gw.qsgw_head as qsgw_head
    from common.parallel_transport import fingerprint_from_binding
    from gw.static_gauge_response import build_static_photon_head_response

    charge_extent = layout.padded_extent(0)
    Y = np.zeros((1, 3, charge_extent), np.complex128)
    for a in range(2):
        Y[0, a, 0] = (0.11 + 0.03j) * (a + 1)
    Z = np.conj(np.transpose(Y, (0, 2, 1)))
    direct = SimpleNamespace(
        S_direct=_put(np.zeros((1, 3, 3), np.complex128), mesh, ()),
        Y_x=_put(Y, mesh, (None, None, "x")),
        Z_y=_put(Z, mesh, (None, "y", None)),
    )
    saved = qsgw_head.build_dft_head_response
    qsgw_head.build_dft_head_response = lambda *args, **kwargs: direct
    try:
        sigma = np.asarray(sigma_h, dtype=np.float64)
        if np.any(sigma != 0.0):
            hall = qsgw_head._static_gauge_hall_transaction_from_artifact(
                sigma_H=sigma,
                hamiltonian_config_operator_fingerprint="sha256:" + "7" * 64,
                wfn_fingerprint=fingerprint_from_binding(wfn_binding, wfn),
                band_start=int(meta.b_id_0),
                band_stop=int(meta.b_id_4_chi_user),
                nk_tot=int(meta.nk_tot),
                mesh=mesh)
        else:
            hall = None
        return build_static_photon_head_response(
            None, input_dir="unused", mesh=mesh, wfn=wfn, meta=meta,
            config=SimpleNamespace(), layout=layout,
            hall_transaction=hall, wfn_fingerprint_binding=wfn_binding)
    finally:
        qsgw_head.build_dft_head_response = saved


def _read_debug_table(path):
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    header = next(
        line[1:].split() for line in lines
        if line.startswith("# ") and line[2:].split()[:2] == ["k", "n"])
    rows = []
    for line in lines:
        fields = line.split()
        if len(fields) == len(header) and fields[0].isdigit():
            rows.append([float(value) for value in fields])
    return header, np.asarray(rows, dtype=np.float64)


def _write_and_check_receipts(plus, nk, nb, output_dir):
    """Exercise the incumbent GWResults -> canonical table writer seam."""
    from common.units import RYD_TO_EV
    from gw.gw_output import GWResults, write_freq_debug

    zeros = np.zeros((nk, nb, nb), dtype=np.complex128)
    identity = np.broadcast_to(np.eye(nb), (nk, nb, nb)).copy()
    sym = SimpleNamespace(
        kirr_fullids=np.arange(nk, dtype=np.int32),
        unfolded_kpts=np.column_stack((
            np.arange(nk, dtype=np.float64) / nk,
            np.zeros(nk), np.zeros(nk))),
    )
    expected_names = []
    for stem in ("head", "x_head", "sex_head", "coh_head"):
        expected_names.extend((
            f"{stem}_CC.Re", f"{stem}_CC.Im",
            f"{stem}_CTTC.Re", f"{stem}_CTTC.Im",
            f"{stem}_TT.Re", f"{stem}_TT.Im",
            f"{stem}_total.Re", f"{stem}_total.Im",
        ))

    def emit(name, diag, present, explicit):
        path = os.path.join(output_dir, name)
        results = GWResults(
            sig_sx=plus["aggregate"][1],
            sig_coh=plus["aggregate"][2],
            sig_h=zeros,
            sig_x=plus["aggregate"][0],
            E_qp_ry=np.zeros((nk, nb)),
            U_qp=identity,
            E_dft_ry=np.zeros((nk, nb)),
            kin_ion_ry=zeros,
            band_start=0,
            band_stop=nb,
            photon_head_sigma_diag_tskn_ry=diag,
            photon_head_sigma_basis=("dft" if present else None),
        )
        config = SimpleNamespace(
            debug=SimpleNamespace(
                sigma_freq_debug_output=explicit,
                sigma_freq_debug_file=path),
            do_screened=True,
        )
        write_freq_debug(
            results,
            config=config,
            static_head_terms=None,
            omega_dft_rel_ev=None,
            head_sigma_diag_w_kn_ry=None,
            omega_grid_ry=None,
            sym=sym,
            print_fn=lambda *_args, **_kwargs: None,
        )
        if not os.path.isfile(path):
            raise AssertionError(f"canonical Sigma receipt was not emitted: {path}")
        header, rows = _read_debug_table(path)
        missing = [column for column in expected_names if column not in header]
        if missing:
            raise AssertionError(f"Sigma receipt lacks columns {missing}")
        return path, header, rows

    full_path, header, rows = emit(
        "sigma_freq_debug.dat", plus["components"], True, False)
    with open(full_path, "r", encoding="utf-8") as stream:
        full_receipt_text = stream.read()
    required_metadata = (
        "# metadata photon_head_basis=dft\n",
        ("# metadata photon_head_sector_convention="
         "final_post_dyson_lorentz_blocks\n"),
    )
    missing_metadata = [
        line.rstrip() for line in required_metadata
        if line not in full_receipt_text]
    if missing_metadata:
        raise AssertionError(
            f"authenticated Sigma receipt lacks metadata {missing_metadata}")
    expected_total = np.sum(
        plus["components"][1] + plus["components"][2], axis=0) * RYD_TO_EV
    got_re = rows[:, header.index("head_total.Re")].reshape(nk, nb)
    got_im = rows[:, header.index("head_total.Im")].reshape(nk, nb)
    writer_error = float(np.max(np.abs(
        got_re + 1j * got_im - expected_total)))
    if writer_error > 8.0e-7:
        raise AssertionError(
            f"canonical head_total receipt differs by {writer_error:.3e} eV")

    zero_path, zero_header, zero_rows = emit(
        "sigma_freq_debug.zero.dat",
        np.zeros_like(plus["components"]), False, True)
    head_indices = [
        zero_header.index(column) for column in expected_names
        if column.startswith("head_")]
    zero_receipt_leak = float(np.max(np.abs(zero_rows[:, head_indices])))
    if zero_receipt_leak != 0.0:
        raise AssertionError(
            f"ordinary-mode zero receipt leaks {zero_receipt_leak:.3e} eV")
    return {
        "full_receipt": full_path,
        "zero_receipt": zero_path,
        "writer_max_abs_ev": writer_error,
        "zero_receipt_leak_ev": zero_receipt_leak,
    }


def _check_rotated_diagonal(mesh):
    """NumPy oracle for the shared batched diag(U A U^dagger) owner."""
    from gw.qsgw_density import diagonal_rotated_band_matrix

    rng = np.random.default_rng(2026082611)
    nk, nt, nb = 2, 3, 4
    raw = (rng.standard_normal((nk, nb, nb))
           + 1j * rng.standard_normal((nk, nb, nb)))
    unitary = np.stack([np.linalg.qr(raw[k])[0] for k in range(nk)])
    matrices = (rng.standard_normal((nk, nt, nb, nb))
                + 1j * rng.standard_normal((nk, nt, nb, nb)))
    u_device = _put(unitary, mesh, (None, "x", "y"))
    a_device = _put(matrices, mesh, (None, None, "x", "y"))

    @jax.jit
    def qp_to_dft(a, u):
        return diagonal_rotated_band_matrix(
            a, u, mesh=mesh, to_qp=False)

    @jax.jit
    def dft_to_qp(a, u):
        return diagonal_rotated_band_matrix(
            a, u, mesh=mesh, to_qp=True)

    got_dft = _gather(qp_to_dft(a_device, u_device))
    want_dft = np.einsum(
        "kmn,ktnp,kmp->ktm", unitary, matrices,
        np.conj(unitary), optimize=True)
    got_qp = _gather(dft_to_qp(a_device, u_device))
    want_qp = np.einsum(
        "kmn,ktml,kln->ktn", np.conj(unitary), matrices,
        unitary, optimize=True)
    return max(
        float(np.max(np.abs(got_dft - want_dft))),
        float(np.max(np.abs(got_qp - want_qp))),
    )


def _bundle(mesh, psi, enk, occ, slices):
    from gw.wavefunction_bundle import (
        PSI_MUN_SPEC, PSI_NMU_SPEC, Wavefunctions)
    return Wavefunctions(
        psi_mun=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_MUN_SPEC),
        psi_nmu=_put(psi, mesh, PSI_NMU_SPEC),
        enk=_put(enk, mesh, (None, None)),
        occ=_put(occ, mesh, (None, None)),
        slices=slices,
        layout="face",
    )


def run_gate(mesh, wfn_path, output_dir):
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader

    # The producer authenticates the Hall term against the loaded WFN's
    # identity, so the loader stays open for the whole gate.
    with WfnLoader(wfn_path, backend="eager") as wfn:
        return _run_gate(mesh, wfn, output_dir)


def _run_gate(mesh, wfn, output_dir):
    import vcoul
    from common.parallel_transport import bind_wfn_fingerprint

    from gw.head_correction import (
        STATIC_PHOTON_CHARGE_BLOCK_PRESENT,
        complete_static_slab_photon_q0,
    )
    from gw.photon_layout import (
        PhotonBasisLayout, pack_photon_channel_vectors, photon_block_view,
        photon_q0_low_rank_block)
    from gw.photon_sigma import compute_static_photon_sigma
    from gw.wavefunction_bundle import BandSlices

    rotated_diagonal_error = _check_rotated_diagonal(mesh)
    if rotated_diagonal_error > 2.0e-10:
        raise AssertionError(
            "batched rotated-diagonal oracle error "
            f"{rotated_diagonal_error:.3e}")

    geometry = vcoul.CoulombGeometry.from_wfn(wfn)
    material_kgrid = tuple(int(v) for v in wfn.kgrid)
    wfn_binding = bind_wfn_fingerprint(wfn)
    receipt = vcoul.slab_minibz_photon_cubature(
        vcoul.get_kernel(2), geometry, material_kgrid)

    # Two logical charge centroids (wing support on 0, analytic body coupling
    # 0 <-> 1) and two transverse; equal padded extents keep the q0 block
    # graph count at exactly two (bare / screened factor counts).
    layout = PhotonBasisLayout.from_centroid_extents(2, 2, mesh)
    n = layout.packed_extent
    nq = 2
    packed_sharding = NamedSharding(mesh, P(None, "x", "y"))
    x_sharding = NamedSharding(mesh, P(None, "x"))
    y_sharding = NamedSharding(mesh, P(None, "y"))
    def packed_vertex(axis_name):
        sharding = x_sharding if axis_name == "x" else y_sharding
        vectors = []
        for row in range(4):
            value = 1.0 + 0.07j * (row + 1)
            vector = np.zeros(
                (1, layout.padded_extent(row)), dtype=np.complex128)
            vector[0, 0] = value
            vectors.append(jax.device_put(vector, sharding))
        return pack_photon_channel_vectors(
            tuple(vectors), layout, mesh, axis_name=axis_name)[0]

    # Use the canonical mesh-interleaved packer; hand-addressing padded
    # channels here would create a second layout convention in the gate.
    g0_x = packed_vertex("x")
    g0_y = packed_vertex("y")

    rng = np.random.default_rng(2026082607)
    nk, nb, ns, mu = nq, 4, 4, layout.padded_extent(0)
    psi_c = (rng.standard_normal((nk, nb, ns, mu))
             + 1j * rng.standard_normal((nk, nb, ns, mu)))
    psi_t = (rng.standard_normal((nk, nb, ns, mu))
             + 1j * rng.standard_normal((nk, nb, ns, mu)))
    enk = np.tile(np.asarray((-0.7, -0.2, 0.3, 0.8)), (nk, 1))
    occ = np.tile(np.asarray((1.0, 1.0, 0.0, 0.0)), (nk, 1))
    slices = BandSlices.from_band_edges(0, 0, 2, 4, 4)
    wfns_c = _bundle(mesh, psi_c, enk, occ, slices)
    wfns_t = _bundle(mesh, psi_t, enk, occ, slices)
    Gij = np.zeros((nk, nb, nb), dtype=np.complex128)
    idx = np.arange(nb)
    Gij[:, idx, idx] = occ
    Gij = _put(Gij, mesh, (None, None, None))
    meta = SimpleNamespace(
        kgrid=(nq, 1, 1), nk_tot=nq, b_id_0=0, b_id_4_chi_user=nb)

    V_host = np.zeros((nq, n, n), np.complex128)
    W_host = np.zeros((nq, n, n), np.complex128)
    W_host[0] = 0.1 * np.eye(n, dtype=np.complex128)
    # Charge centroid 0 (the wing support) couples only to charge centroid
    # 1, so Y W Z = 0 while W Z and Y W are nonzero (see _response).
    charge_0 = _packed_index(layout, mesh, 0, 0)
    charge_1 = _packed_index(layout, mesh, 0, 1)
    W_host[0, charge_0, charge_0] = 0.0
    W_host[0, charge_0, charge_1] = 0.1
    W_host[0, charge_1, charge_0] = 0.1

    def base_operators():
        return (
            jax.device_put(V_host, packed_sharding),
            jax.device_put(W_host, packed_sharding),
        )

    V_reference, W_reference = base_operators()
    (baseline_x, baseline_sx, baseline_coh, baseline_diagnostics,
     baseline_sigma_diagnostics) = (
        compute_static_photon_sigma(
            wfns_charge=wfns_c,
            wfns_transverse=wfns_t,
            Gij=Gij,
            V_packed=V_reference,
            W_packed=W_reference,
            photon_layout=layout,
            meta=meta,
            mesh_xy=mesh,
            verbose=False,
        ))
    if baseline_diagnostics is not None:
        raise AssertionError("headless photon Sigma returned diagnostics")
    if baseline_sigma_diagnostics is None:
        raise AssertionError("headless photon Sigma omitted its block split")
    baseline = tuple(
        _gather(value) for value in (baseline_x, baseline_sx, baseline_coh))

    def one_case(sigma_h):
        V, W = base_operators()
        response = _response(
            mesh, layout, sigma_h, wfn=wfn, wfn_binding=wfn_binding,
            meta=meta)
        V, W, completion = complete_static_slab_photon_q0(
            V, W, response, g0_x, g0_y, receipt, mesh_xy=mesh,
            charge_block_state=STATIC_PHOTON_CHARGE_BLOCK_PRESENT)
        if not np.array_equal(
                np.asarray(completion.sigma_H),
                np.asarray(sigma_h, dtype=np.float64)):
            raise AssertionError(
                "completion did not record the sigma_H it consumed")
        bare_v_q0 = _gather(V[0])
        bare_v_hermiticity_residual = float(np.max(np.abs(
            bare_v_q0 - np.conj(bare_v_q0.T))))
        bare_v_scale = float(np.max(np.abs(bare_v_q0)))
        (sig_x, sig_sx, sig_coh, diagnostics,
         sigma_diagnostics) = compute_static_photon_sigma(
            wfns_charge=wfns_c,
            wfns_transverse=wfns_t,
            Gij=Gij,
            V_packed=V,
            W_packed=W,
            photon_layout=layout,
            meta=meta,
            mesh_xy=mesh,
            head_completion=completion,
            diagnostic_input_basis="dft",
            verbose=False,
        )
        aggregate = tuple(_gather(x) for x in (sig_x, sig_sx, sig_coh))
        components = _gather(diagnostics.components_tskn_ry)
        sigma_components = _gather(sigma_diagnostics.components_skij_ry)
        np.testing.assert_allclose(
            np.sum(sigma_components, axis=0), aggregate[1] + aggregate[2],
            rtol=1.0e-11, atol=1.0e-13)
        block_errors = []
        for A, B in ((0, 0), (0, 2), (2, 0), (2, 3)):
            for packed, pairs in (
                (V, (completion.q0_factors.bare_pair,)),
                (W, completion.q0_factors.screened_pairs),
            ):
                got = _gather(photon_q0_low_rank_block(
                    pairs, layout, A, B, mesh))
                want = _gather(photon_block_view(
                    packed, layout, A, B, mesh))
                base_packed = V_reference if packed is V else W_reference
                base = _gather(photon_block_view(
                    base_packed, layout, A, B, mesh))
                update = want - base
                block_errors.append(float(np.max(np.abs(
                    got[0] - update[0]))))
                block_errors.append(float(np.max(np.abs(update[1:]))))
        closure_errors = []
        for term, (matrix, base_matrix) in enumerate(zip(aggregate, baseline)):
            direct = np.diagonal(
                matrix - base_matrix, axis1=1, axis2=2)
            sector_sum = np.sum(components[term], axis=0)
            closure_errors.append(float(np.max(np.abs(
                direct - sector_sum))))
        return {
            "aggregate": aggregate,
            "components": components,
            "completion": completion,
            "block_error": max(block_errors),
            "closure_error": max(closure_errors),
            "internal_closure": diagnostics.max_closure_residual_ry,
            "bare_v_hermiticity_residual": bare_v_hermiticity_residual,
            "bare_v_scale": bare_v_scale,
        }

    # Scale the Hall coefficient so max ||D Pi_H|| is 0.05: visibly nonzero
    # while remaining far from a Dyson pole on the actual CrI3 mini-BZ.
    hall_unit = _hall_numpy(np.asarray((0.0, 0.0, 1.0)))
    max_coupling = 0.0
    for chunk in receipt.chunks:
        count = int(chunk.physical_count)
        qxy = np.asarray(chunk.q_cart[:count, :2])
        D = np.asarray(chunk.D_raw[:count], dtype=np.complex128)
        Pi = np.einsum("sa,aij->sij", qxy, hall_unit)
        max_coupling = max(
            max_coupling,
            float(np.linalg.norm(D @ Pi, axis=(1, 2)).max()))
    sigma_amp = 0.05 / max_coupling
    sigma_plus = np.asarray((0.0, 0.0, sigma_amp), dtype=np.float64)
    plus = one_case(sigma_plus)
    zero = one_case(np.zeros(3, dtype=np.float64))
    minus = one_case(-sigma_plus)

    oracle_errors = []
    for case, sigma_h in ((plus, sigma_plus),
                          (zero, np.zeros(3)),
                          (minus, -sigma_plus)):
        oracle = _direct_moment_oracle(receipt, sigma_h)
        oracle_errors.append(float(np.max(np.abs(
            np.asarray(case["completion"].screened_moments) - oracle))))
    oracle_error = max(oracle_errors)
    block_error = max(case["block_error"] for case in (plus, zero, minus))
    closure_error = max(case["closure_error"] for case in (plus, zero, minus))
    bare_v_hermiticity_residual = max(
        case["bare_v_hermiticity_residual"]
        for case in (plus, zero, minus))
    bare_v_scale = max(
        case["bare_v_scale"] for case in (plus, zero, minus))
    scale = max(float(np.max(np.abs(x)))
                for case in (plus, zero, minus)
                for x in case["aggregate"])
    cttc_plus = plus["components"][1, 1] + plus["components"][2, 1]
    cttc_zero = zero["components"][1, 1] + zero["components"][2, 1]
    cttc_minus = minus["components"][1, 1] + minus["components"][2, 1]
    hall_plus = cttc_plus - cttc_zero
    hall_minus = cttc_minus - cttc_zero
    hall_signal = float(np.max(np.abs(0.5 * (hall_plus - hall_minus))))
    cttc_sx_signal = float(np.max(np.abs(plus["components"][1, 1])))
    cttc_coh_signal = float(np.max(np.abs(plus["components"][2, 1])))
    zero_background = float(np.max(np.abs(cttc_zero)))
    sign_error = float(np.max(np.abs(hall_plus + hall_minus)))
    if oracle_error > 2.0e-10:
        raise AssertionError(f"direct NumPy 4x4 oracle error {oracle_error:.3e}")
    if block_error > 2.0e-10:
        raise AssertionError(f"factor/block oracle error {block_error:.3e}")
    if closure_error > 2.0e-10 * max(1.0, scale):
        raise AssertionError(
            f"head sector sum does not close to aggregate Sigma: "
            f"{closure_error:.3e}")
    if bare_v_hermiticity_residual > 2.0e-10 * max(1.0, bare_v_scale):
        raise AssertionError(
            "completed bare q=0 photon operator is not Hermitian: "
            f"residual={bare_v_hermiticity_residual:.3e}, "
            f"scale={bare_v_scale:.3e}")
    if hall_signal <= 1.0e-12:
        raise AssertionError(
            f"nonzero Hall response produced no CT/TC band Sigma: "
            f"total={hall_signal:.3e}, SX={cttc_sx_signal:.3e}, "
            f"COH={cttc_coh_signal:.3e}")
    # Non-Hall wings can carry a sigma_H-independent CT/TC background.  The
    # Hall discriminator is therefore the change from the exact sigma_H=0
    # twin.  At max ||D Pi_H||=0.05 the nonlinear Dyson even part is allowed
    # at O(0.05) of the odd response, but a failed sign reversal is not.
    if sign_error > 0.1 * hall_signal:
        raise AssertionError(
            f"sigma_H sign twin failed in CT/TC band Sigma: "
            f"even={sign_error:.3e}, odd={hall_signal:.3e}")
    receipt_checks = None
    if jax.process_index() == 0:
        receipt_checks = _write_and_check_receipts(
            plus, nk, nb, output_dir)
    from gw import photon_layout as photon_layout_module
    q0_block_executables = len(photon_layout_module._q0_block_cache)
    if q0_block_executables != 2:
        raise AssertionError(
            "equal C/T padded extents with bare/screened factor counts must "
            f"compile exactly two q0 block graphs; got {q0_block_executables}")
    return {
        "material_kgrid": material_kgrid,
        "cell_volume": float(geometry.cell_volume),
        "sigma_Hz": sigma_amp,
        "numpy_4x4_oracle_max_abs": oracle_error,
        "block_oracle_max_abs": block_error,
        "sigma_closure_max_abs": closure_error,
        "bare_v_hermiticity_max_abs": bare_v_hermiticity_residual,
        "internal_closure_max_abs": max(
            case["internal_closure"] for case in (plus, zero, minus)),
        "cttc_sigma_signal": hall_signal,
        "cttc_sx_signal": cttc_sx_signal,
        "cttc_coh_signal": cttc_coh_signal,
        "cttc_sigma_zero_background": zero_background,
        "cttc_sigma_sign_error": sign_error,
        "q0_block_executables": q0_block_executables,
        "rotated_diagonal_oracle_max_abs": rotated_diagonal_error,
        "receipt_checks": receipt_checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--wfn", required=True)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    px, py = (int(v) for v in args.mesh.split("x"))
    if jax.process_count() != px * py:
        raise ValueError(
            f"mesh {args.mesh} needs {px * py} ranks; got {jax.process_count()}")
    from jax.sharding import Mesh
    mesh = _RUNTIME.mesh
    if tuple(mesh.devices.shape) != (px, py):
        mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    result = run_gate(
        mesh, os.path.abspath(args.wfn), os.path.abspath(args.output_dir))
    if jax.process_index() == 0:
        print(f"FULL_PHOTON_HEAD_SIGMA_P4_PASS {result}", flush=True)
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
