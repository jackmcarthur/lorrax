import numpy as np, sys
def load(p):
    rows=[l.split() for l in open(p) if l.strip() and not l.startswith('#')]
    rows=[r for r in rows if r[5]=='interp']
    E=np.array([[float(x) for x in r[6:]] for r in rows])  # (nQ, n_eig) eV
    Q=np.array([[float(r[2]),float(r[3]),float(r[4])] for r in rows])
    return Q,E
Qn,En=load(sys.argv[1]); Qc,Ec=load(sys.argv[2])
assert En.shape==Ec.shape, (En.shape,Ec.shape)
d=(Ec-En)*1000.0  # meV
print(f"native file: {sys.argv[1]}")
print(f"coarse file: {sys.argv[2]}")
print(f"shape (nQ,n_eig)={En.shape}")
print(f"native E1 at Gamma = {En[0,0]:.4f} eV;  coarse E1 at Gamma = {Ec[0,0]:.4f} eV")
print(f"lowest-exciton (col0) native range [{En[:,0].min():.4f},{En[:,0].max():.4f}] eV")
print(f"Delta (coarse-native) meV: max|Δ|={np.abs(d).max():.3f}  mean|Δ|={np.abs(d).mean():.3f}  rms={np.sqrt((d**2).mean()):.3f}")
print(f"per-eig max|Δ| meV: {np.round(np.abs(d).max(0),3)}")
print(f"lowest-exciton (col0) max|Δ| = {np.abs(d[:,0]).max():.3f} meV")
np.savez(sys.argv[3], Q=Qn, E_native=En, E_coarse=Ec, dE_meV=d)
print("saved", sys.argv[3])
