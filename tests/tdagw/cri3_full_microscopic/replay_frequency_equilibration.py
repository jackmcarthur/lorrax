"""ONE conditioning-only captured-pencil replay, not producer replay/admission."""
import argparse,hashlib,importlib.util,json,os,resource,time
from pathlib import Path
import numpy as np
import file_io
from gw.shared_pole_constructor import reduce_general_pencil

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('capture',type=Path)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
capture=args.capture
out=args.out
if out.exists():raise FileExistsError(out)
spec=importlib.util.spec_from_file_location('pencil_analysis',Path.cwd()/'tests/tdagw/cri3_full_microscopic/analyze_general_pencil.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
start=time.perf_counter()
report=dict(scope='CONDITIONING-ONLY lower-level pencil diagnostic; NOT exact new-producer finite/infinity-unit replay; NOT admission',
    coordinate_units='ALL ONE: old capture lacks authenticated infinity mask; no qi-width assumption',
    capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    reducer_file=reduce_general_pencil.__code__.co_filename,
    reducer_sha256=hashlib.sha256(Path(reduce_general_pencil.__code__.co_filename).read_bytes()).hexdigest(),
    rank_tolerance=1e-8,max_order=3522,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
    status='running')
def save():
    report.update(seconds=time.perf_counter()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(helper.finite(report),indent=2)+'\n')
with np.load(capture) as f:
    e,a=f['E'],f['A'];energy=f['energy_ry'];eta=float(f['eta_ry'])
    omega=max(float(np.ptp(energy)),eta)
    report.update(shape=e.shape,frequency_scale_ry=omega,eta_ry=eta);save()
    t=time.perf_counter()
    try:
        model=reduce_general_pencil(e,a,max_order=3522,rank_tolerance=1e-8,
            frequency_scale_ry=omega,coordinate_units=None,max_workspace_bytes=4*1024**3)
        report.update(reduce_seconds=time.perf_counter()-t,receipt=model['receipt'],
            maximum_growth_ry=float(model['poles_ry'].imag.max()),
            laplace_margin_ry=eta-float(model['poles_ry'].imag.max()),
            pole_count=len(model['poles_ry']),growing_count=int(np.count_nonzero(model['poles_ry'].imag>0)))
        save();print('REDUCTION',report['pole_count'],report['maximum_growth_ry'],report['reduce_seconds'],flush=True)
        sites=f['sites_ry']
        # Raw frequency-only physical Grams, NOT the self-q crossing mixture.
        try:
            report['raw_gram_transfer']=helper.gram_transfer_summary(model,f['GC'],f['GB'],sites,
                probe_sites=f['probe_sites_ry'],mixed=f['probe_mixed_J'],exact_sq=f['probe_exact_norm_sq'])
        except Exception as error:
            report['raw_gram_transfer_unresolved']=str(error)
        save()
        # Small descriptor-coordinate witness remains available even if
        # physical FP64 Grams cannot resolve cancellation.
        b=np.eye(e.shape[0],8,dtype=complex);c=np.eye(8,e.shape[1],dtype=complex)
        report['descriptor_coordinate_transfer'],_=helper.transfer_summary(model,b,c,sites[:3])
        report['status']='complete_unadmitted'
    except Exception as error:
        report.update(status='refused_unadmitted',error=str(error))
save();print(json.dumps(helper.finite({k:report.get(k) for k in ('status','pole_count','maximum_growth_ry','reduce_seconds','seconds','raw_gram_transfer_unresolved')}),indent=2),flush=True)
