import h5py, numpy as np, sys
f=h5py.File(sys.argv[1],'r')
print("kgrid=", np.asarray(f['kgrid'][:]).ravel())
print("W0_qmunu attrs:", dict(f['W0_qmunu'].attrs))
print("W0_qmunu shape:", f['W0_qmunu'].shape)
print("V_qmunu shape:", f['V_qmunu'].shape)
print("enk_full shape:", f['enk_full'].shape)
