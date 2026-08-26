"""P4 receipt gate for the bispinor V/restart artifact join.

This cell has no numerical photon body.  It proves on every rank that one
matching small host record authenticates, while a same-shape V artifact whose
charge-zeta identity differs and a legacy unstamped V artifact both refuse.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np


_TESTS = Path(__file__).resolve().parent
_REPO = _TESTS.parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


if __name__ == "__main__":
    from runtime import initialize_communicator_stack
    _RUNTIME = initialize_communicator_stack(platform="gpu")


def _basis(role, centroid_md5):
    from common.bispinor_init import KINETIC_BALANCE_LIFT_PROVENANCE
    from common.parallel_transport import WFN_FINGERPRINT_SCHEME
    from common.wfn_transforms import FULL_BLOCH_TRANSFORM_SCHEME
    from file_io.wfn_basis import (
        CENTROID_TABLE_FINGERPRINT_SCHEME,
        WavefunctionBasisReceipt,
    )

    return WavefunctionBasisReceipt(
        role=role,
        wfn_fingerprint_scheme=WFN_FINGERPRINT_SCHEME,
        wfn_fingerprint="a" * 64,
        band_interval=(2, 12),
        fft_grid=(8, 8, 4),
        centroid_fingerprint_scheme=CENTROID_TABLE_FINGERPRINT_SCHEME,
        centroid_table_md5=centroid_md5,
        n_rmu_logical=7,
        n_rmu_padded=8,
        source_identity=FULL_BLOCH_TRANSFORM_SCHEME,
        nspinor_sampled=4,
        bispinor_lift_provenance=KINETIC_BALANCE_LIFT_PROVENANCE,
    )


def _binding():
    from file_io.bispinor_vq_restart import BispinorVqRestartBinding
    from file_io.tagged_arrays import (
        coulomb_policy_from_config, format_coulomb_policy)

    head = SimpleNamespace(
        mc_average_vcoul_body=True,
        mc_average_placement="off",
        mc_average_placement_vcoul=None,
        head_minibz_average=False,
        bare_coulomb_cutoff=None,
        use_bgw_vcoul=False,
        bgw_vcoul_file=None,
        bispinor_tt_head_correction=False,
    )
    policy = format_coulomb_policy(coulomb_policy_from_config(
        SimpleNamespace(head=head), SimpleNamespace(sys_dim=2)))
    return BispinorVqRestartBinding.from_sources(
        v_qmunu_format="bispinor_lorentz_v2",
        zeta_fit_provenance=(
            '{"channel":"C","fit":"same"}',
            '{"channel":"T1","fit":"same"}',
            '{"channel":"T2","fit":"same"}',
            '{"channel":"T3","fit":"same"}',
        ),
        charge_basis_receipt=_basis("charge", "1" * 32),
        transverse_basis_receipt=_basis("transverse", "2" * 32),
        coulomb_policy=policy,
    )


def _write(path, binding=None):
    from file_io.bispinor_vq_restart import (
        BISPINOR_VQ_RESTART_BINDING_DATASET)

    with h5py.File(path, "w") as h5:
        # Identical payload shape in every discriminator: only receipt state
        # changes, so a shape-only preflight would accept all three.
        h5.create_dataset("payload", data=np.zeros((2, 3, 3)))
        if binding is not None:
            h5.create_dataset(
                BISPINOR_VQ_RESTART_BINDING_DATASET,
                data=binding.encode())


def run_gate(artifact_root: Path):
    import jax
    from jax.experimental import multihost_utils

    import file_io.bispinor_vq_restart as binding_module
    import file_io.wfn_basis as basis_module
    import gw.gw_init as gw_init_module
    from file_io.bispinor_vq_restart import (
        assert_bispinor_vq_restart_binding)

    if jax.process_count() != 4 or len(jax.devices()) != 4:
        raise AssertionError(
            f"gate requires P4; got process_count={jax.process_count()}, "
            f"devices={len(jax.devices())}")
    rank = jax.process_index()
    rank_dir = artifact_root / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=False)

    binding = _binding()
    restart = rank_dir / "restart.h5"
    matching = rank_dir / "vq_matching.h5"
    stale = rank_dir / "vq_stale_zeta.h5"
    legacy = rank_dir / "vq_legacy.h5"
    _write(restart, binding)
    _write(matching, binding)
    assert assert_bispinor_vq_restart_binding(
        restart_path=restart, v_q_path=matching,
        where=f"P4 rank {rank} matching") == binding

    stale_binding = replace(
        binding,
        zeta_fit_provenance=(
            '{"channel":"C","fit":"different"}',
            *binding.zeta_fit_provenance[1:],
        ))
    _write(stale, stale_binding)
    try:
        assert_bispinor_vq_restart_binding(
            restart_path=restart, v_q_path=stale,
            where=f"P4 rank {rank} stale zeta")
    except ValueError as exc:
        if "zeta_fit_provenance" not in str(exc):
            raise
    else:
        raise AssertionError("same-shape stale zeta receipt was accepted")

    _write(legacy)
    try:
        assert_bispinor_vq_restart_binding(
            restart_path=restart, v_q_path=legacy,
            where=f"P4 rank {rank} legacy")
    except ValueError as exc:
        if "Legacy artifacts cannot authenticate" not in str(exc):
            raise
    else:
        raise AssertionError("legacy unstamped V artifact was accepted")

    all_passed = np.asarray(multihost_utils.process_allgather(
        np.asarray([1, 1, 1], dtype=np.int32)))
    if all_passed.shape != (4, 3) or not np.all(all_passed == 1):
        raise AssertionError(f"incomplete all-rank receipt verdict {all_passed}")
    if rank == 0:
        print(f"ORIGIN_BINDING={binding_module.__file__}", flush=True)
        print(f"ORIGIN_WFN_BASIS={basis_module.__file__}", flush=True)
        print(f"ORIGIN_GW_INIT={gw_init_module.__file__}", flush=True)
        print(
            "P4_BINDING_GATE process_count=4 devices=4 mesh=2x2 "
            "matching=PASS stale_zeta=REFUSED legacy=REFUSED "
            "distributed_payload_opened=false",
            flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run_gate(args.artifact_root)


if __name__ == "__main__":
    main()
