"""Focused algebra/source gates for the immutable centroid-WFN receipt."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from file_io.wfn_basis import (
    CENTROID_TABLE_FINGERPRINT_SCHEME,
    WavefunctionBasisReceipt,
    centroid_table_md5,
)


CENTROIDS = np.asarray([
    [0, 0, 0],
    [1, 2, 3],
    [3, 1, 0],
], dtype=np.int32)


def _wfn(*, shift: float = 0.0):
    return SimpleNamespace(
        nbands=6,
        nelec=2,
        nspinor=2,
        energies=np.asarray([[[0.1 + shift, 0.2, 0.3, 0.4, 0.5, 0.6]]]),
        kpoints=np.asarray([[0.0, 0.0, 0.0]]),
    )


def _receipt(*, role="transverse", pad=4, centroids=CENTROIDS,
             grid=(4, 4, 4), band_interval=(1, 5), wfn=None,
             bispinor=False):
    table = np.asarray(centroids, dtype=np.int32)
    return WavefunctionBasisReceipt.from_source(
        wfn=_wfn() if wfn is None else wfn,
        role=role,
        bispinor=bispinor,
        band_interval=band_interval,
        fft_grid=grid,
        centroid_fft_idx=table,
        n_rmu_logical=table.shape[0],
        n_rmu_padded=pad,
    )


def test_receipt_is_immutable_and_reuses_the_restart_centroid_digest():
    receipt = _receipt()
    assert receipt.centroid_fingerprint_scheme == \
        CENTROID_TABLE_FINGERPRINT_SCHEME
    assert receipt.centroid_table_md5 == centroid_table_md5(CENTROIDS)
    with pytest.raises(FrozenInstanceError):
        receipt.role = "charge"


def test_bound_canonical_wfn_fingerprint_is_reused_without_rescan(
        monkeypatch):
    import common.parallel_transport as transport

    wfn = _wfn()
    binding = transport.bind_wfn_fingerprint(wfn)
    canonical = transport.fingerprint_from_binding(binding, wfn)
    with pytest.raises(TypeError, match="host-only and transient"):
        pickle.dumps(binding)

    def unexpected_rescan(_wfn_value):
        raise AssertionError("precomputed canonical WFN fingerprint rescanned")

    monkeypatch.setattr(transport, "wfn_fingerprint", unexpected_rescan)
    charge = WavefunctionBasisReceipt.from_bound_source(
        wfn=wfn, wfn_fingerprint_binding=binding,
        role="charge", bispinor=True, band_interval=(1, 5),
        fft_grid=(4, 4, 4), centroid_fft_idx=CENTROIDS,
        n_rmu_logical=3, n_rmu_padded=4)
    transverse = WavefunctionBasisReceipt.from_bound_source(
        wfn=wfn, wfn_fingerprint_binding=binding,
        role="transverse", bispinor=True, band_interval=(1, 5),
        fft_grid=(4, 4, 4), centroid_fft_idx=CENTROIDS,
        n_rmu_logical=3, n_rmu_padded=4)
    assert charge.wfn_fingerprint == transverse.wfn_fingerprint == canonical
    with pytest.raises(ValueError, match="different loaded WFN object"):
        WavefunctionBasisReceipt.from_bound_source(
            wfn=_wfn(), wfn_fingerprint_binding=binding,
            role="charge", bispinor=True, band_interval=(1, 5),
            fft_grid=(4, 4, 4), centroid_fft_idx=CENTROIDS,
            n_rmu_logical=3, n_rmu_padded=4)
    with pytest.raises(TypeError, match="bind_wfn_fingerprint"):
        WavefunctionBasisReceipt.from_bound_source(
            wfn=wfn, wfn_fingerprint_binding=canonical,
            role="charge", bispinor=True, band_interval=(1, 5),
            fft_grid=(4, 4, 4), centroid_fft_idx=CENTROIDS,
            n_rmu_logical=3, n_rmu_padded=4)


def test_prepare_constructs_receipts_on_host_from_one_canonical_scan():
    source = Path(__file__).parents[1] / "src" / "gw" / "gw_init.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    prepare = next(
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "prepare_isdf_and_wavefunctions")
    assert prepare.decorator_list == []

    stages = [node for node in module.body if isinstance(node, ast.FunctionDef)
              and node.name in ("_prepare_fresh_isdf", "_prepare_fitted_zeta",
                                "_restart_charge_basis", "_restart_current_carrier")]
    assert len(stages) == 4
    assert all(node.decorator_list == [] for node in stages)
    receipt_calls = [
        node for stage in stages for node in ast.walk(stage)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "WavefunctionBasisReceipt"
        and node.func.attr == "from_bound_source"
    ]
    assert len(receipt_calls) == 4
    assert all(
        any(keyword.arg == "wfn_fingerprint_binding"
            for keyword in call.keywords)
        for call in receipt_calls)

    canonical_scans = [
        node for stage in stages for node in ast.walk(stage)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bind_wfn_fingerprint"
    ]
    assert len(canonical_scans) == 1
    assert (len(canonical_scans[0].args) == 1
            and isinstance(canonical_scans[0].args[0], ast.Name)
            and canonical_scans[0].args[0].id == "wfn")
    assert "wfn_fingerprint_value" not in source.read_text(encoding="utf-8")


def test_fresh_and_restart_face_builders_validate_host_receipt():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from gw.wavefunction_bundle import (
        BandSlices,
        build_wavefunctions_face,
        wavefunctions_face_from_restart,
    )

    receipt = _receipt()
    slices = BandSlices.from_band_edges(1, 1, 2, 3, 5)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    psi_y = jnp.zeros((1, 4, 2, 4), dtype=jnp.complex128)
    psi_t_x = jnp.zeros((1, 4, 4, 2), dtype=jnp.complex128)
    enk = jnp.zeros((1, 4), dtype=jnp.float64)

    fresh = build_wavefunctions_face(
        psi_y, psi_t_x, enk_full=enk, slices=slices, mesh_xy=mesh,
        basis_receipt=receipt)
    restart = wavefunctions_face_from_restart(
        fresh.psi_nmu, fresh.psi_mun, enk_full=enk, slices=slices,
        mesh_xy=mesh, basis_receipt=receipt)
    assert not hasattr(fresh, "basis_receipt")
    assert not hasattr(restart, "basis_receipt")
    receipt.assert_matches_carrier(fresh, where="fresh orchestration")
    receipt.assert_matches_carrier(restart, where="restart orchestration")


def test_physical_source_identity_is_layout_and_device_count_independent():
    p4 = _receipt(pad=4)
    p16 = _receipt(pad=16)
    assert p4.source_identity == p16.source_identity
    p4.assert_same_source(p16, where="cross-device receipt")
    assert p4 != p16
    with pytest.raises(ValueError, match="padded centroid extents"):
        p4.assert_same_carrier(p16, where="same-runtime carrier")


def test_charge_and_transverse_are_distinct_even_on_the_same_points():
    charge = _receipt(role="charge")
    transverse = _receipt(role="transverse")
    with pytest.raises(ValueError, match="role"):
        charge.assert_same_source(transverse, where="Lorentz channel")


def test_scalar_spinor_and_kinetic_balance_lift_are_distinct_sources():
    scalar = _receipt(role="charge", bispinor=False)
    kinetic_balance = _receipt(role="charge", bispinor=True)
    assert scalar.nspinor_sampled == 2
    assert kinetic_balance.nspinor_sampled == 4
    assert scalar.bispinor_lift_provenance is None
    with pytest.raises(ValueError, match="nspinor_sampled"):
        scalar.assert_same_source(
            kinetic_balance, where="sampled representation")


@pytest.mark.parametrize(
    "changed,field",
    [
        (_receipt(wfn=_wfn(shift=1.0e-3)), "wfn_fingerprint"),
        (_receipt(band_interval=(0, 4)), "band_interval"),
        (_receipt(grid=(5, 4, 4)), "fft_grid"),
        (_receipt(centroids=CENTROIDS[::-1]), "centroid_table_md5"),
        (_receipt(centroids=CENTROIDS[:2], pad=4), "n_rmu_logical"),
    ],
)
def test_every_physical_source_field_has_a_red_twin(changed, field):
    with pytest.raises(ValueError, match=field):
        _receipt().assert_same_source(changed, where="stale source")


def test_wavefunctions_authenticates_receipt_against_live_mu_carrier():
    import jax.numpy as jnp
    from gw.wavefunction_bundle import BandSlices, Wavefunctions

    slices = BandSlices.from_band_edges(1, 1, 2, 3, 5)
    psi_nmu = jnp.zeros((1, 4, 2, 4), dtype=jnp.complex128)
    psi_mun = jnp.zeros((1, 2, 4, 4), dtype=jnp.complex128)
    wfns = Wavefunctions(
        enk=jnp.zeros((1, 4)), occ=jnp.zeros((1, 4)), slices=slices,
        psi_nmu=psi_nmu, psi_mun=psi_mun, layout="face",
    )
    _receipt(role="charge").assert_matches_carrier(
        wfns, where="host orchestration")

    with pytest.raises(ValueError, match="centroid extent"):
        mismatched_mu = Wavefunctions(
            enk=jnp.zeros((1, 4)), occ=jnp.zeros((1, 4)), slices=slices,
            psi_nmu=psi_nmu[..., :3], psi_mun=psi_mun[:, :, :3],
            layout="face")
        _receipt(role="charge").assert_matches_carrier(
            mismatched_mu, where="host orchestration")

    with pytest.raises(ValueError, match="spinor extent"):
        _receipt(role="charge", bispinor=True).assert_matches_carrier(
            Wavefunctions(
            enk=jnp.zeros((1, 4)), occ=jnp.zeros((1, 4)), slices=slices,
            psi_nmu=psi_nmu, psi_mun=psi_mun, layout="face",
            ), where="host orchestration")


def test_receipt_is_host_only_and_does_not_split_jit_cache_family():
    import jax
    import jax.numpy as jnp
    from gw.wavefunction_bundle import (
        AuthenticatedWavefunctions, BandSlices, Wavefunctions)

    slices = BandSlices.from_band_edges(1, 1, 2, 3, 5)
    carrier = dict(
        enk=jnp.zeros((1, 4)), occ=jnp.zeros((1, 4)), slices=slices,
        psi_nmu=jnp.zeros((1, 4, 2, 4), dtype=jnp.complex128),
        psi_mun=jnp.zeros((1, 2, 4, 4), dtype=jnp.complex128),
        layout="face",
    )
    wfns = Wavefunctions(**carrier)
    first_receipt = _receipt(role="charge")
    second_receipt = _receipt(role="charge", wfn=_wfn(shift=1.0e-3))
    assert first_receipt != second_receipt
    first_receipt.assert_matches_carrier(wfns, where="first source")
    second_receipt.assert_matches_carrier(wfns, where="second source")

    python_traces = []

    @jax.jit
    def numerical_kernel(wfns):
        python_traces.append(wfns.layout)
        return wfns.enk + 1.0

    first = AuthenticatedWavefunctions(wfns, first_receipt)
    second = AuthenticatedWavefunctions(wfns, second_receipt)

    def orchestrate(binding):
        return numerical_kernel(binding.wavefunctions)

    np.testing.assert_array_equal(orchestrate(first), np.ones((1, 4)))
    np.testing.assert_array_equal(orchestrate(second), np.ones((1, 4)))
    assert python_traces == ["face"]
    assert numerical_kernel._cache_size() == 1
    hlo_first = numerical_kernel.lower(first.wavefunctions).as_text()
    hlo_second = numerical_kernel.lower(second.wavefunctions).as_text()
    assert hlo_first == hlo_second
    assert first_receipt.wfn_fingerprint not in hlo_first
    assert second_receipt.wfn_fingerprint not in hlo_second

    # A whole-carrier JIT round-trip cannot silently erase provenance because
    # provenance was never embedded in the numerical pytree.  Its explicit
    # orchestration owner survives independently.
    roundtripped = jax.jit(lambda value: value)(wfns)
    assert not hasattr(roundtripped, "basis_receipt")
    assert first.receipt.wfn_fingerprint != second.receipt.wfn_fingerprint
    with pytest.raises(TypeError, match="Error interpreting argument"):
        jax.jit(lambda value: value)(first)


def test_finite_transfer_endpoint_preserves_pre_receipt_positional_abi():
    import jax
    import jax.numpy as jnp
    from common.mtxel_sweep import FiniteTransferCurrentEndpoint

    assert FiniteTransferCurrentEndpoint._fields[-1] == "basis_receipt"
    assert FiniteTransferCurrentEndpoint.__new__.__defaults__ == (None,)
    old_arity_values = [None] * (len(FiniteTransferCurrentEndpoint._fields) - 1)
    assert FiniteTransferCurrentEndpoint(*old_arity_values).basis_receipt is None

    values = dict.fromkeys(FiniteTransferCurrentEndpoint._fields, None)
    values.update(
        current_nmu=jnp.zeros((1, 1, 3, 4, 1)),
        current_mun=jnp.zeros((1, 3, 4, 1, 1)),
        n_rmu_logical=1,
        iq_irr=0,
        q_irr_kgrid_int=np.zeros(3, dtype=np.int32),
        q_crys=np.zeros(3),
        kminq_idx=np.zeros(1, dtype=np.int32),
        g_wrap=np.zeros((1, 3), dtype=np.int32),
        hamiltonian_config_operator_fingerprint="sha256:" + "0" * 64,
        vnl_path_operator_fingerprint="sha256:" + "1" * 64,
    )
    endpoint = FiniteTransferCurrentEndpoint(**values)
    with pytest.raises(TypeError):
        jax.jit(lambda row: row.current_nmu)(endpoint)


def test_isdf_header_keeps_candidate_import_compatibility_without_ownership():
    from file_io import isdf_header
    from file_io import wfn_basis

    assert (isdf_header.WavefunctionBasisReceipt
            is wfn_basis.WavefunctionBasisReceipt)
    assert isdf_header.centroid_table_md5 is wfn_basis.centroid_table_md5
