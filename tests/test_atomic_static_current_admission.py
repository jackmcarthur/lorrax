"""Public reconstructed static-current admission and early source refusals."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gw.gw_config import (
    BispinorGWMode, HeadCorrection, LorraxConfig, refuse_unsupported_bispinor_gw,
)


def config(tmp_path, *, restart=False):
    path = tmp_path / 'x.in'
    path.write_text('[cohsex]\nnval=1\nncond=1\nnumber_bands=4\n'
                    'qp_solver=one_shot_dft\nlinalg=local\ncompute_mode=x_only\n'
                    'bispinor=true\nbispinor_gw=bare_transverse\nsys_dim=3\n'
                    'head_correction=off\natomic_reconstruction_dir=atoms\n'
                    f'restart={str(restart).lower()}\n')
    return LorraxConfig.from_input_file(
        str(path), resolve_hardware=False, print_fn=lambda *_: None)


@pytest.mark.parametrize('restart', [False, True])
def test_headless_static_current_parser_and_programmatic_preflight(tmp_path, restart):
    cfg = config(tmp_path, restart=restart)
    refuse_unsupported_bispinor_gw(cfg)
    assert cfg.bispinor_gw is BispinorGWMode.BARE_TRANSVERSE
    assert cfg.paths.atomic_reconstruction_dir == str(tmp_path / 'atoms')


@pytest.mark.parametrize('legacy_screened', [False, True])
@pytest.mark.parametrize('restart', [False, True])
def test_explicit_x_only_preserves_legacy_screening_flag_compatibility(tmp_path, legacy_screened, restart):
    cfg = replace(config(tmp_path, restart=restart), do_screened=legacy_screened)
    refuse_unsupported_bispinor_gw(cfg)
    assert not cfg.compute_mode.needs_screening


@pytest.mark.parametrize('change', [
    'cohsex', 'mpa', 'full_shared_pole', 'full_static_cohsex', 'scalar', 'slab',
    'head_full', 'head_no_local_fields', 'head_overlay', 'self_consistent',
    'smearing', 'broadening', 'density_flag_only',
])
@pytest.mark.parametrize('restart', [False, True])
def test_reconstruction_never_leaks_into_unvalidated_photon_models(tmp_path, change, restart):
    cfg = config(tmp_path, restart=restart)
    if change in ('cohsex', 'mpa'):
        cfg = replace(cfg, compute_mode_raw=change, do_screened=True)
    elif change in ('full_shared_pole', 'full_static_cohsex'):
        cfg = replace(cfg, bispinor_gw=BispinorGWMode(change))
    elif change == 'scalar':
        cfg = replace(cfg, bispinor=False)
    elif change == 'slab':
        cfg = replace(cfg, sys_dim=2)
    elif change.startswith('head_'):
        head = (replace(cfg.head, bispinor_tt_head_correction=True)
                if change == 'head_overlay' else
                replace(cfg.head, correction=HeadCorrection(change.removeprefix('head_'))))
        cfg = replace(cfg, head=head)
    elif change == 'self_consistent':
        cfg = replace(cfg, qp_solver_raw='self_consistent', density_self_consistent=True)
    elif change == 'smearing':
        cfg = replace(cfg, occ_smearing_width_ry=.001, occ_smearing_family='fd')
    elif change == 'broadening':
        # The existing constructor owns this refusal before any atomic
        # admission or data read; do not bypass it to reach a later guard.
        with pytest.raises(ValueError, match='occ_broadening > 0'):
            replace(cfg, screening=replace(cfg.screening, occ_broadening_ev=.01))
        return
    elif change == 'density_flag_only':
        cfg = replace(cfg, density_self_consistent=True)
    with pytest.raises(ValueError, match='GATE atomic_augmentation_domain'):
        refuse_unsupported_bispinor_gw(cfg)


@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.parametrize('metal', [False, True])
def test_public_fresh_and_restart_check_actual_occupations_before_io(tmp_path, monkeypatch, restart, metal):
    from gw import gw_init
    from gw.wavefunction_bundle import BandSlices

    cfg = config(tmp_path, restart=restart)
    slices = BandSlices.from_band_edges(0, 0, 1, 2, 4, b4_logical=4)
    wfn = SimpleNamespace(occs=np.array([[[.75 if metal else 1., 0., 0., 0.]]]))
    meta = SimpleNamespace(b_id_4_user=4, mu_basis=None)
    reached = []

    class OwnerReached(Exception):
        pass

    def stop_before_io(*args, **kwargs):
        reached.append('restart' if restart else 'fresh')
        raise OwnerReached

    monkeypatch.setattr(gw_init, '_prepare_fresh_isdf', stop_before_io)
    monkeypatch.setattr(gw_init, '_prepare_restart_isdf', stop_before_io)
    expected = ValueError if metal else OwnerReached
    with pytest.raises(expected, match='atomic_transverse_fixed_occupations' if metal else None):
        gw_init.prepare_isdf_and_wavefunctions(
            cfg=cfg, wfn=wfn, sym=None, meta=meta, centroid_indices=None,
            band_slices=slices, mesh_xy=None, tmp_dir=str(tmp_path),
            tensors_filename=str(tmp_path / 'forbidden.h5'), print0=lambda *_: None)
    assert reached == ([] if metal else ['restart' if restart else 'fresh'])
    assert not (tmp_path / 'forbidden.h5').exists()
