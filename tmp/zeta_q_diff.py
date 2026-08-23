"""Bounded diff of two zeta_q.h5 files: dataset list/shapes/dtypes match,
plus a sampled numeric comparison (a handful of q-blocks, not the whole
~9 GB file) between the incumbent single-axis fit and the new face-CCT
fit on the SAME k6_c600 deck.
"""
import sys

import h5py
import numpy as np

a_path, b_path = sys.argv[1], sys.argv[2]

with h5py.File(a_path, "r") as fa, h5py.File(b_path, "r") as fb:
    keys_a = set(fa.keys())
    keys_b = set(fb.keys())
    print(f"datasets only in A: {sorted(keys_a - keys_b)}")
    print(f"datasets only in B: {sorted(keys_b - keys_a)}")
    common = sorted(keys_a & keys_b)
    print(f"{len(common)} common top-level datasets/groups")
    for k in common:
        da, db = fa[k], fb[k]
        if isinstance(da, h5py.Dataset) and isinstance(db, h5py.Dataset):
            same_shape = da.shape == db.shape
            same_dtype = da.dtype == db.dtype
            print(f"  {k}: shape A={da.shape} B={db.shape} "
                  f"({'OK' if same_shape else 'MISMATCH'}) "
                  f"dtype A={da.dtype} B={db.dtype} "
                  f"({'OK' if same_dtype else 'MISMATCH'})")

    zq_name = None
    for cand in ("zeta_q_G", "zeta_q", "zeta", "zeta_mu_q", "zeta_q_ibz"):
        if cand in fa:
            zq_name = cand
            break
    if zq_name is None:
        print("no zeta_q-like dataset found by common name; listed keys above")
        sys.exit(0)

    if "g0_mu" in fa:
        ga, gb = np.asarray(fa["g0_mu"]), np.asarray(fb["g0_mu"])
        d = np.abs(ga - gb)
        s = np.abs(ga)
        print(f"\ng0_mu (full, {ga.shape}): max|diff|={float(d.max()):.3e} "
              f"(ref scale {float(s.max()):.3e}) "
              f"max|rel|={float((d / np.maximum(s, 1e-300)).max()):.3e}")

    dset_a = fa[zq_name]
    dset_b = fb[zq_name]
    print(f"\nsampling '{zq_name}' shape {dset_a.shape} dtype {dset_a.dtype}")
    nq = dset_a.shape[0]
    sample_q = sorted(set([0, nq // 3, 2 * nq // 3, nq - 1]))
    max_abs = 0.0
    max_rel = 0.0
    ref_scale = 0.0
    for q in sample_q:
        va = np.asarray(dset_a[q])
        vb = np.asarray(dset_b[q])
        d = np.abs(va - vb)
        s = np.abs(va)
        max_abs = max(max_abs, float(d.max()))
        ref_scale = max(ref_scale, float(s.max()))
        rel = float((d / np.maximum(s, 1e-300)).max())
        max_rel = max(max_rel, rel)
        print(f"  q={q}: max|diff|={float(d.max()):.3e} "
              f"(ref scale {float(s.max()):.3e}) max|rel|={rel:.3e}")
    print(f"\noverall (sampled q={sample_q}): "
          f"max|diff|={max_abs:.3e} ref_scale={ref_scale:.3e} "
          f"max|rel diff|={max_rel:.3e}")
