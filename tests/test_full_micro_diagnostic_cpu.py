"""Partial q0 frequency artifacts precede refusal; no Sigma is admitted."""
import json
import numpy as np
import pytest


@pytest.mark.parametrize('growth', [False, True])
def test_native_q0_partial_capture_and_direct_dyson_before_refusal(tmp_path,monkeypatch,growth):
    import gw.shared_pole_screening as screening
    from tests.test_tdagw_full_microscopic_cpu import test_full_native_screening_builder_matches_literal_full_port_dyson
    original=screening.screen_general_shared_poles
    real_reduce=screening.__dict__.get('collective_general_pencil_batch')
    if growth:
        import gw.shared_pole_constructor as constructor
        real_reduce=constructor.collective_general_pencil_batch
        def poison(*a,**k):
            models=real_reduce(*a,**k)
            models[0]['poles_ry'][0]=complex(models[0]['poles_ry'][0].real,.3)
            return models
        monkeypatch.setattr(constructor,'collective_general_pencil_batch',poison)
    def diagnostic(*a,**kw):
        kw['snapshot']=dict(kw['snapshot'],stop_after_model_q=0)
        return original(*a,**kw)
    monkeypatch.setattr(screening,'screen_general_shared_poles',diagnostic)
    gate='shared_general_laplace' if growth else 'shared_general_diagnostic_stop'
    with pytest.raises((ValueError,RuntimeError),match=gate):
        test_full_native_screening_builder_matches_literal_full_port_dyson(tmp_path,monkeypatch)
    path=tmp_path/'builder_general_shared_pole/q_000_diagnostic.npz'
    record=json.loads(path.with_suffix('.json').read_text())
    assert not record['admitted'] and not record['sigma_computed']
    assert record['status']==('refused_laplace' if growth else 'unadmitted_partial')
    assert record['q']==0 and record['n_q']==3
    with np.load(path) as arrays:
        for key in ('E','A','poles_ry','output_map','input_map','space_right','space_left'):
            assert key in arrays
        np.testing.assert_allclose(arrays['residue_norm_bound'],
                                   arrays['left_column_norm']*arrays['right_column_norm'])
    if growth:
        probes=json.loads((path.parent/record['direct_dyson_probes_file']).read_text())['probes']
        assert len(probes)==2
        assert all('relative_w_defect' in p for p in probes)
        assert record['eta_margin_ry']<0
    else:
        assert max(record['sample_errors'])<.02
        assert record['eta_margin_ry']>0
