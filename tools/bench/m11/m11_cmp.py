import sys, numpy as np, json
a = np.load(sys.argv[1])["acc"]; b = np.load(sys.argv[2])["acc"]
same = a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))
d = float(np.abs(a - b).max()) if a.shape == b.shape else -1
print("CMP", json.dumps(dict(a=sys.argv[1].split("/")[-1], b=sys.argv[2].split("/")[-1], bitwise=bool(same), maxdiff=d, absmax=float(np.abs(a).max()))))
