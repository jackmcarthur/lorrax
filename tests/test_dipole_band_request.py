"""Producer configuration gate: execute main to Meta with mocked WFN/I/O only.

Run as a P4 script. No wavefunctions, operators or scientific matrices are
constructed; actual input parsing, band resolution and provenance are used.
"""


def main():
    from contextlib import nullcontext
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    import runtime
    import psp.get_dipole_mtxels as d

    class ReachedMeta(Exception):
        pass

    header = SimpleNamespace(nelec=18, nbands=304, kgrid=np.array([4,4,4]),
                             symmetry=lambda: None)
    report = SimpleNamespace(begin=lambda **kw: None, devices=lambda **kw: None)
    captured = []

    def meta(wfn, sym, nval, ncond, nband, *rest):
        captured.append(d.dipole_provenance(wfn=wfn, wfn_path='test-WFN',
            nval=nval, ncond=ncond, nband=nband, nb_written=d._resolve_dipole_nb_written(
                wfn,ncond=ncond,nband=nband),bispinor=False,skip_vnl=False,vnl_mode='analytic'))
        raise ReachedMeta

    with TemporaryDirectory() as tmp, \
            patch.object(d,'WfnLoader',lambda *a,**k: header), \
            patch.object(d,'RunSession',lambda *a,**k: nullcontext(SimpleNamespace(report=report,debug=None))), \
            patch.object(d,'StageSequence',lambda *a,**k: nullcontext(None)), \
            patch.object(d,'Meta',SimpleNamespace(from_system=meta)), \
            patch.object(d,'wfn_fingerprint',lambda wfn: 'fixture-header-identity'):
        deck=Path(tmp)/'dipole.in'
        for request,expected in [('number_bands_protected = 40\n',(18,22)),
                                 ('nval = 5\nncond = 7\n',(5,7))]:
            deck.write_text('[cohsex]\nsys_dim = 3\nnumber_bands = 304\n'+request)
            try:
                d.main(['-i',str(deck),'--no-parallel-transport'])
            except ReachedMeta:
                pass
            else:
                raise AssertionError('did not reach producer Meta')
            attrs=captured[-1]
            assert (attrs['prov_nval'],attrs['prov_ncond'])==expected,attrs
            assert attrs['prov_nband']==304 and attrs['prov_nb_written']==304
        for request in ('number_bands_protected = 17\n',
                        'number_bands_protected = 305\n',
                        'number_bands_protected = 40\nnval = 5\n'):
            deck.write_text('[cohsex]\nsys_dim = 3\nnumber_bands = 304\n'+request)
            before=len(captured)
            try:
                d.main(['-i',str(deck),'--no-parallel-transport'])
            except ValueError as exc:
                assert 'GATE band_request_forms' in str(exc),str(exc)
            else:
                raise AssertionError('invalid request reached producer Meta')
            assert len(captured)==before
    if d.jax.process_index()==0:
        print('DIPOLE_BAND_REQUEST PASS protected40->18/22 explicit5/7 validity/provenance',flush=True)
    return 0


if __name__=='__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
