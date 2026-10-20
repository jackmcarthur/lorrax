"""One bounded reduced-pencil benchmark; no production edits or admission."""
import argparse,contextlib,hashlib,io,json,os,resource,time
from pathlib import Path
import numpy as np
import scipy,scipy.linalg as la
config=io.StringIO()
with contextlib.redirect_stdout(config):np.show_config()

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('capture',type=Path)
parser.add_argument('--qz-report',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
path=args.capture
if args.out.exists():raise FileExistsError(args.out)
reference=json.loads(args.qz_report.read_text())
report=dict(scope='CPU numerical experiment only; raw descriptor-coordinate transfer, no microscopic W/Sigma admission',
    input=str(path.resolve()),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qz_report_sha256=hashlib.sha256(args.qz_report.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    tolerance=1e-8,status='running',versions=dict(numpy=np.__version__,scipy=scipy.__version__),
    threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
    numpy_build_config=config.getvalue(),timings={})
start=time.perf_counter()
def save():
    report.update(total_seconds=time.perf_counter()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
def stamp(name,before):
    report['timings'][name]=time.perf_counter()-before;save();print(name,report['timings'][name],flush=True)
with np.load(path) as f:e,a=f['E'],f['A']
t=time.perf_counter();u,s,vh=la.svd(e,full_matrices=False,check_finite=False)
n=int(np.count_nonzero(s>1e-8*s[0]));u=u[:,:n];v=vh[:n].conj().T;s=s[:n]
report['order']=n;report['E_condition']=float(s[0]/s[-1]);stamp('descriptor_svd',t)
t=time.perf_counter();er=u.conj().T@e@v;ar=u.conj().T@a@v
d=1/np.sqrt(s);h=d[:,None]*ar*d[None,:]
report['balanced_E_identity_defect']=float(la.norm(d[:,None]*er*d[None,:]-np.eye(n))/np.sqrt(n))
stamp('balance_and_project',t)
t=time.perf_counter();p,wl,vr=la.eig(h,left=True,right=True,check_finite=False);stamp('ordinary_geev',t)
t=time.perf_counter();metric=wl.conj().T@vr
inverse=la.solve(metric,np.eye(n,dtype=complex),check_finite=False)
out=(v*d[None,:])@vr;inp=(u*d[None,:])@wl@inverse.conj().T
report['balanced_eigen_residual']=float(la.norm(h@vr-vr*p[None,:])/la.norm(h@vr))
report['original_descriptor_backward_residual']=float(la.norm(ar@(d[:,None]*vr)-(er@(d[:,None]*vr))*p[None,:])/la.norm(ar@(d[:,None]*vr)))
report['max_growth_ry']=float(p.imag.max());report['growing_count']=int(np.count_nonzero(p.imag>0))
ref=reference['studies'][0]
report['QZ_reference']=dict(receipt=ref['receipt'],growing_count=ref['transfer']['growing_count'],rerun=False)
report['largest_growth_comparison']=[]
for row in ref['transfer']['largest_growth']:
    z=complex(row['pole_ry']['real'],row['pole_ry']['imag']);nearest=p[np.argmin(abs(p-z))]
    report['largest_growth_comparison'].append(dict(QZ=[z.real,z.imag],ordinary_nearest=[nearest.real,nearest.imag],absolute_difference=float(abs(z-nearest))))
stamp('modal_maps_and_backward_checks',t)
t=time.perf_counter();report['descriptor_coordinate_probes']=[]
for row in ref['transfer']['probes'][:3]:
    z=complex(row['site_ry']['real'],row['site_ry']['imag'])
    got=(out[:8]/(z-p)[None,:])@inp[:8].conj().T
    br=u.conj().T[:,:8];cr=v[:8]
    direct=cr@la.solve(z*er-ar,br,check_finite=False)
    balanced=cr@(d[:,None]*la.solve(z*np.eye(n)-h,d[:,None]*br,check_finite=False))
    report['descriptor_coordinate_probes'].append(dict(site_ry=[z.real,z.imag],ordinary_norm=float(la.norm(got)),
        QZ_report_norm=row['transfer_frobenius'],relative_norm_difference_from_QZ=float(abs(la.norm(got)-row['transfer_frobenius'])/row['transfer_frobenius']),
        modal_vs_balanced_resolvent=float(la.norm(got-balanced)/la.norm(balanced)),
        modal_vs_original_descriptor_resolvent=float(la.norm(got-direct)/la.norm(direct))))
stamp('three_transfer_probes',t)
report['status']='complete';save()
print(json.dumps({k:report[k] for k in ('order','max_growth_ry','timings','total_seconds','descriptor_coordinate_probes')},indent=2),flush=True)
