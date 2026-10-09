"""Build the NVRTC k-convolution images of the release grid list into the per-user cache.

    lx run -N 1 -G 4 -n 4 -- env SCRATCH=<fresh dir> python scripts/build_cubin_store.py [GRID ...]

Each process takes every ``nproc``-th (grid, image) task on its own GPU (a 1 x 1 mesh, no
collectives), calls the router factory on tiny operands, and so leaves the image in
``ffi.fft.cubin_cache_dir()`` (``$SCRATCH/.cache/lorrax/kconv_mathdx``, seeded first from
the source tree's store, so only missing images compile).  The image key holds the mode,
k-grid, ``n_s``, right width, precision and variant (``kconv_mathdx_cuda_ffi.cc`` ``build``),
never ``N_mu``, bands or P, so the operands are as small as the factories accept.  The
regular files the cache then holds are the new images; the release copies them into its
``cubin_store/`` (``docs/architecture/kconv.md#build-and-cache``).

Images keyed by the system rather than the k-grid are not built here: mode 10 and the
Fourier pair (FFT plane and G-sphere rows) and the BSE outer kernels (band rank K).  They
come from the release smoke decks run against the same cache.

Per grid and spin width ``n_s`` the tasks are the ones the drivers request (router call
sites): modes 2-5 (Sigma/COHSEX/BSE k transforms and convolutions; modes 4-5 also c64, the
BSE fp32 runner), mode 9 at ``n_s = 1`` (the scalar W wedge), and per ``n_s`` modes 1 and 6
(ISDF pair), 7 (Sigma unfold) and 11 (chi0, complete and not); at ``n_s = 4`` mode 8 at
vertex widths (na, nb) in {1, 3}^2 and the two-spinor chi vertex.  Modes 7, 8, 9 and 11 are
built plain and with the ``live`` operand of a windowed pass.  A grid the router sends to
XLA, or a residency refusal, prints SKIP and builds nothing.
"""
from __future__ import annotations

import os
import sys
import time

#: (k-grid, spin widths): the hsuite fixtures, the release smoke decks and the production grids.
GRIDS = {
    "probe": ((2, 1, 1), ()),             # the router's startup probe (mode 3 only)
    "na_fixture": ((3, 3, 3), (1,)),      # hsuite bcc Na
    "h2_fixture": ((5, 5, 1), (1, 2, 4)),  # hsuite H2- spinor, bispinor stages
    "fe_4": ((4, 4, 4), (1, 2, 4)),       # Fe/Co/Ni 4^3
    "agi_6": ((6, 6, 6), (1, 2, 4)),      # AgI k6
    "cubic_8": ((8, 8, 8), (1, 2, 4)),    # Si, Na, Fe/Co/Ni, AgI k8
    "cubic_20": ((20, 20, 20), (1, 2, 4)),  # Fe/Co/Ni 20^3, Gd hcp 20^3
    "cri3_6": ((6, 6, 1), (1, 2, 4)),     # CrI3 6x6
    "cri3_24": ((24, 24, 1), (1, 2, 4)),  # CrI3 24x24
    "crsbr": ((20, 15, 1), (1, 2, 4)),    # CrSBr 20x15
    "nips3": ((12, 7, 1), (1, 2, 4)),     # NiPS3 12x7
}


def tasks(kg, spins):
    """``(name, thunk)`` for every image of one grid; a thunk builds and runs one call."""
    if not spins:
        yield "m3", lambda c: c.kfft(kg)
        return
    yield "m3", lambda c: c.kfft(kg)
    yield "m2", lambda c: c.klead(kg)
    yield "m5", lambda c: c.kfft_kminor(kg, c.c128)
    yield "m5 c64", lambda c: c.kfft_kminor(kg, c.c64)
    yield "m4", lambda c: c.kminor(kg, c.c128)
    yield "m4 c64", lambda c: c.kminor(kg, c.c64)
    for live in (False, True):
        yield f"m9 live={live}", lambda c, lv=live: c.wedge(kg, lv)
    for ns in spins:
        yield f"m1 ns{ns}", lambda c, s=ns: c.parent(kg, s)
        yield f"m6 ns{ns}", lambda c, s=ns: c.plane(kg, s)
        for live in (False, True):
            yield f"m7 ns{ns} live={live}", lambda c, s=ns, lv=live: c.unfold(kg, s, lv)
            for complete in (False, True):
                yield (f"m11 ns{ns} complete={complete} live={live}",
                       lambda c, s=ns, cp=complete, lv=live: c.chi(kg, s, cp, lv))
    if 4 in spins:
        for live in (False, True):
            for na in (1, 3):
                for nb in (1, 3):
                    yield (f"m8 ns4 w{na}x{nb} live={live}",
                           lambda c, a=na, b=nb, lv=live: c.lorentz(kg, 4, a, b, lv))
            yield f"m11 vertex ns2 live={live}", lambda c, lv=live: c.vertex(kg, lv)


class Calls:
    """Tiny-operand calls of each router factory on a 1 x 1 mesh."""

    def __init__(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from ffi import fft as F
        self.jax, self.jnp, self.np, self.P, self.F = jax, jnp, np, P, F
        self.mesh = Mesh(np.asarray(jax.local_devices()[:1]).reshape(1, 1), ("x", "y"))
        self.NS = lambda spec: NamedSharding(self.mesh, spec)
        self.c128, self.c64 = np.complex128, np.complex64
        self.m = 2                                              # centroids per shard
        self.rng = np.random.default_rng(0)

    def put(self, a, spec):
        return self.jax.device_put(self.np.asarray(a), self.NS(spec))

    def rnd(self, *shape, dtype=None):
        a = self.rng.standard_normal(shape) + 1j * self.rng.standard_normal(shape)
        return a.astype(dtype or self.c128)

    def run(self, f, *args, **kw):
        self.jax.block_until_ready(self.jax.jit(lambda *a: f(*a, **kw))(*args))

    def tables(self, nk, ns, n_l, n_r, n_parent=2):
        np = self.np
        from symmetry_maps import UnfoldLoadTables
        m = self.m
        lsrc = np.tile(np.arange(m * n_l, dtype=np.int32), (nk, 1))
        rsrc = np.tile(np.arange(m * n_r, dtype=np.int32), (nk, 1))
        return UnfoldLoadTables(
            row=(np.arange(nk) % n_parent).astype(np.int32), trs=np.zeros(nk, np.int32),
            lsrc=lsrc, rsrc=rsrc, mph=np.ones(lsrc.shape, np.complex128),
            nph=np.ones(rsrc.shape, np.complex128),
            spin=np.tile(np.eye(ns, dtype=np.complex128), (nk, 1, 1)),
            n_parent=n_parent, mesh_shape=(1, 1), conj_trs=0)

    def live(self, lv):
        return dict(live=self.put(self.np.asarray([0, self.m], self.np.int32), self.P())) if lv else {}

    def kfft(self, kg):
        nk, m, P = int(self.np.prod(kg)), self.m, self.P
        X = self.put(self.rnd(nk, m, m), P(None, "x", "y"))
        for kind in ("ifftn", "fftn"):
            self.run(self.F.make_kfft_klead(self.mesh, kg, P(None, None, None, "x", "y"), kind=kind,
                                            norm="ortho"), X)

    def _klead(self, kg):
        P = self.P
        return self.F.make_kconv_klead(self.mesh, kg, P(None, None, None, None, "x", None, "y"),
                                       P(None, None, None, "x", "y"), norm="ortho")

    def klead(self, kg):
        nk, m, P = int(self.np.prod(kg)), self.m, self.P
        c = self._klead(kg)
        T = self.put(self.rnd(nk, 1, m, 1, m), P(None, None, "x", None, "y"))
        W = self.put(self.rnd(nk, m, m), P(None, "x", "y"))
        self.run(lambda t, w: c.apply(t, c.prep(w)), T, W)

    def kfft_kminor(self, kg, dt):
        m, P = self.m, self.P
        X = self.put(self.rnd(m, m, *kg, dtype=dt), P("x", "y", None, None, None))
        self.run(self.F.make_kfft_kminor(self.mesh, kg, P("x", "y", None, None, None), kind="ifftn",
                                         norm="backward"), X)

    def kminor(self, kg, dt):
        nk, m, P = int(self.np.prod(kg)), self.m, self.P
        X = self.put(self.rnd(1, m, m, 1, 1, nk, dtype=dt), P(None, "x", "y", None, None, None))
        K = self.put(self.rnd(m, m, nk, dtype=dt), P("x", "y", None))
        self.run(self.F.make_kconv_kminor(self.mesh, kg, P(None, "x", "y", None, None, None),
                                          P("x", "y", None), norm="ortho"), X, K)

    def wedge(self, kg, lv):
        from symmetry_maps import device_load_tables
        nk, m, P = int(self.np.prod(kg)), self.m, self.P
        t = self.tables(nk, 1, 1, 1)
        W = self.put(self.rnd(t.n_parent, m, m), P(None, "x", "y"))
        f = self.F.make_kfft_klead_unfold(self.mesh, kg, t, norm="ortho")
        kw = dict(load=device_load_tables(t, self.mesh), **self.live(lv)) if lv else {}
        self.run(lambda w, **k: f(w, w, **k), W, **kw)

    def parent(self, kg, ns):
        np, jnp, m = self.np, self.jnp, self.m
        nk = int(np.prod(kg))
        perm, phase = list(range(ns)), [1] * ns
        f = self.F.make_fused_conv_kparent(self.mesh, kg, ns, (m, m), perm_l=perm, phase_l=phase,
                                           perm_r=perm, phase_r=phase)
        D = jnp.asarray(self.rnd(2, ns, m, ns, m))
        coef = np.tile(np.eye(ns * ns, dtype=np.complex128), (nk, 1, 1))
        z = lambda *s: jnp.zeros(s, jnp.int32)
        tabs = (z(nk), z(nk), z(1, m), z(1, m), jnp.zeros((1, m, 3)), jnp.zeros((1, m, 3)),
                jnp.zeros((2, 3)), z(nk), jnp.asarray(coef), jnp.asarray(coef))
        self.run(lambda d, t: f(d, d, t), D, tabs)

    def plane(self, kg, ns):
        nk, m = int(self.np.prod(kg)), self.m
        perm, phase = list(range(ns)), [1] * ns
        f = self.F.make_fused_conv_kplane(self.mesh, kg, ns, perm_l=perm, phase_l=phase,
                                          perm_r=perm, phase_r=phase)
        self.run(f, self.jnp.asarray(self.rnd(nk, 1, ns, 2 * m, ns, 4)),
                 self.jnp.asarray(self.rnd(nk, 1, 4)))

    def _greens(self, ns, n_parent=2):
        m, P = self.m, self.P
        return self.put(self.rnd(n_parent, m, ns, m, ns), P(None, "x", None, "y", None))

    def unfold(self, kg, ns, lv):
        from symmetry_maps import device_load_tables
        nk, m, P = int(self.np.prod(kg)), self.m, self.P
        t = self.tables(nk, ns, ns, ns)
        c = self._klead(kg)
        f = self.F.make_kconv_klead_unfold(self.mesh, kg, t, store_rows=[0], norm="ortho")
        G = self._greens(ns)
        W = self.put(self.rnd(nk, m, m), P(None, "x", "y"))
        kw = dict(load=device_load_tables(t, self.mesh), **self.live(lv)) if lv else {}
        self.run(lambda g, w, **k: f(g, g, c.prep(w), **k), G, W, **kw)

    def chi(self, kg, ns, complete, lv):
        from symmetry_maps import device_load_tables
        nk, m, P = int(self.np.prod(kg)), self.m, self.P
        t = self.tables(nk, ns, ns, ns)
        f = self.F.make_kconv_chi_unfold(self.mesh, kg, t, n_out=1, complete=complete, norm="ortho")
        G = self._greens(ns)
        acc = self.put(self.rnd(1, nk, m, m), P(None, None, "x", "y"))
        alpha = self.put(self.rnd(1), P())
        kw = dict(load=device_load_tables(t, self.mesh), **self.live(lv)) if lv else {}
        self.run(lambda a, g, al, **k: f(a, g, g, al, **k), acc, G, alpha, **kw)

    def vertex(self, kg, lv):
        from symmetry_maps import device_load_tables
        nk, m, P = int(self.np.prod(kg)), self.m, self.P
        t = self.tables(nk, 2, 2, 2)
        v = (((0, 1), (1, 1)),)
        f = self.F.make_kconv_chi_vertex(self.mesh, kg, t, left_vertices=v, right_vertices=v,
                                         sign_c=self.np.ones(nk), norm="ortho")
        G = self._greens(2)
        acc = self.put(self.rnd(1, nk, m, m), P(None, None, "x", "y"))
        kw = dict(load=device_load_tables(t, self.mesh), **self.live(lv)) if lv else {}
        self.run(lambda a, g, **k: f(a, g, g, **k), acc, G, **kw)

    def lorentz(self, kg, ns, na, nb, lv):
        from symmetry_maps import device_load_tables
        nk, m, P = int(self.np.prod(kg)), self.m, self.P
        t = self.tables(nk, ns, ns, ns)
        tw = self.tables(nk, na, na, nb)._replace(
            spin=self.np.tile(self.np.eye(na, dtype=self.np.complex128), (nk, 1, 1)),
            spin_r=self.np.tile(self.np.eye(nb, dtype=self.np.complex128), (nk, 1, 1)))
        vl = tuple((tuple(range(ns)), (1,) * ns) for _ in range(na))
        vr = tuple((tuple(range(ns)), (1,) * ns) for _ in range(nb))
        f = self.F.make_kconv_lorentz_unfold(self.mesh, kg, t, w_tables=tw, left_vertices=vl,
                                             right_vertices=vr, store_rows=[0], norm="ortho")
        G = self._greens(ns)
        W = self.put(self.rnd(2, m, na, m, nb), P(None, "x", None, "y", None))
        kw = (dict(load=device_load_tables(t, self.mesh), w_load=device_load_tables(tw, self.mesh),
                   **self.live(lv)) if lv else {})
        self.run(lambda g, w, **k: f(g, g, w, w, **k), G, W, **kw)


def main(names) -> int:
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    nproc = int(os.environ.get("SLURM_NTASKS", "1"))
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure(print_fn=lambda *a, **k: None)     # the service packages (lxkit, ...)
    from ffi import fft as F
    c = Calls()
    if F.require_kconv(c.mesh, announce=False) != "mathdx":
        print(f"[cubin-store] rank {rank}: the router is not on nvidia-mathdx here; nothing to build",
              flush=True)
        return 1
    work = [(g, n, t) for g in names for n, t in tasks(*GRIDS[g])]
    cache = F.cubin_cache_dir()
    t0, done, skipped = time.time(), 0, 0
    for i, (g, n, thunk) in enumerate(work):
        if i % nproc != rank:
            continue
        t1 = time.time()
        try:
            thunk(c)
            done += 1
            print(f"[cubin-store] rank {rank} {g} {GRIDS[g][0]} {n}: {time.time() - t1:.1f} s", flush=True)
        except Exception as e:                                        # noqa: BLE001
            skipped += 1
            why = str(e).strip().splitlines()[0][:240] if str(e).strip() else type(e).__name__
            print(f"[cubin-store] rank {rank} SKIP {g} {GRIDS[g][0]} {n}: {why}", flush=True)
    print(f"[cubin-store] rank {rank}: {done} calls, {skipped} skipped, {time.time() - t0:.0f} s; "
          f"cache {cache}; store {F.CUBIN_STORE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(GRIDS)))
