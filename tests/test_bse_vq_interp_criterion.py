"""The long-range channel-fitting criterion (``vq_interp.lr_fit_degrees``).

FIXTURE-FREE, and deliberately so.  ``tests/test_bse_vq_interp.py`` is gated
behind an out-of-repo Perlmutter fixture and skips everywhere else, which is
precisely how a criterion defect survived: the guard that looked like it was
watching the long-range channel set was watching a different object, and no
cell that could see the gap ever ran.  Everything here is a statement about
CELL GEOMETRY, so it runs wherever pytest runs, on numpy alone.

Owner ruling of 2026-08-10 (docs/architecture/decisions.md): the criterion is
an ENERGY CUTOFF with a HARD TWO-SHELL FLOOR.  Each cell below gates one half
of that, and the two red twins are the arms that must FAIL.
"""
import numpy as np
import pytest

from bse import vq_interp as V

# Cell geometries, in bohr.  Si is the bulk control (FCC primitive, a = 10.262
# bohr); MoS2 is the reference slab (hexagonal, a = 5.9715, c = 22.677) and the
# vacuum sweep scales c off it.  These reproduce the recorded set sizes exactly
# -- Si nG = 123 / |G_z| <= 2, MoS2 nG = 337 / |G_z| <= 9 / 23 in-plane
# (FIX_vq_interp.md B.1) -- which is what makes them usable as a gate.
A_SI, A_MOS2, C_MOS2 = 10.262, 5.9715, 22.677


def _fcc(a):
    A = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    return 2 * np.pi * np.linalg.inv(A).T, abs(np.linalg.det(A))


def _hex_slab(a, c):
    A = np.array([[a, 0., 0.], [-0.5 * a, np.sqrt(3) / 2 * a, 0.], [0., 0., c]])
    return 2 * np.pi * np.linalg.inv(A).T, abs(np.linalg.det(A))


def _zx(bvec, celvol):
    return {"bvec": bvec, "celvol": celvol}


def _lost_lr_weight(zx, gz_keep, nq_side=7):
    """Fraction of ``sum_G v_LR(q+G)`` sitting in UNFITTED |G_z| channels,
    over an nq_side x nq_side in-plane q sample.

    This is the quantity the production path actually loses: stage 1 subtracts
    the FULL-sphere V_LR and stage 3 adds back only the fitted channels, so an
    unfitted channel's weight is subtracted and never returned.  Metric and
    sampling are FIX_vq_interp.md B.3's, so the numbers here are comparable to
    the 17.93% that opened the decision.  Uses vq_interp's own kernel.
    """
    GS = V.lr_gset(zx)                       # trim off by default -> full set
    fitted = np.isin(np.abs(GS[2]), np.asarray(sorted(gz_keep)))
    tot = lost = 0.0
    for tx in np.linspace(-0.5, 0.5, nq_side, endpoint=False):
        for ty in np.linspace(-0.5, 0.5, nq_side, endpoint=False):
            v = V.v_slab_on_set(zx, [tx, ty, 0.0], GS, kind="slab_lr",
                                alpha=V.ALPHA)
            tot += float(np.sum(v))
            lost += float(np.sum(v[~fitted]))
    return lost / tot


def _head_channel(zx, Qfrac):
    """|G_z| of the mini-BZ head slot G* = argmin_G |Q+G| -- the channel the
    head magnitude lands on (``minibz_head_vlr``)."""
    GS = V.lr_gset(zx)
    K = zx["bvec"].T @ (np.asarray(Qfrac, dtype=float)[:, None] + GS)
    return int(abs(GS[2, int(np.argmin(np.sum(K * K, axis=0)))]))


# --------------------------------------------------------------------------
# the geometries themselves, so a drift in lr_gset cannot silently rebase
# every number below
# --------------------------------------------------------------------------
def test_reference_cell_geometries_reproduce_the_recorded_supersets():
    si = _zx(*_fcc(A_SI))
    GS = V.lr_gset(si)
    assert GS.shape[1] == 123 and int(np.abs(GS[2]).max()) == 2

    mos2 = _zx(*_hex_slab(A_MOS2, C_MOS2))
    GS = V.lr_gset(mos2)
    assert GS.shape[1] == 337 and int(np.abs(GS[2]).max()) == 9
    assert len({(int(a), int(b)) for a, b in zip(GS[0], GS[1])}) == 23


# --------------------------------------------------------------------------
# GATE (c) -- the small-cell control, and backward compatibility
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,geom", [("si", _fcc(A_SI)),
                                       ("mos2_ref", _hex_slab(A_MOS2, C_MOS2))])
def test_reference_decks_fit_exactly_the_channels_they_fitted_before(name, geom):
    """BIT-IDENTITY ON THE REFERENCE DECKS.  The fit consumes channels via
    ``prep['gz_cols']``, i.e. only the |G_z| values the SUPERSET actually
    carries, so what has to be unchanged is the criterion's set INTERSECTED
    with the superset.  On Si the criterion drops the key |G_z|=3, and that is
    a no-op precisely because Si's superset stops at |G_z|=2 -- there are no
    such columns to visit.  Equal intersections => identical design blocks,
    identical normal equations, identical coefficients."""
    zx = _zx(*geom)
    present = {int(g) for g in np.abs(V.lr_gset(zx)[2])}
    was = {abs(int(g)) for g in V.DEG_B26P} & present
    now = {abs(int(g)) for g in V.lr_fit_degrees(zx)} & present
    assert now == was, f"{name}: fitted channel set moved {was} -> {now}"
    # and the degree ladder each surviving channel is fitted at is unchanged
    d = V.lr_fit_degrees(zx)
    assert all(d[g] == V.DEG_B26P[g] for g in sorted(now))


def test_si_bulk_loses_no_long_range_weight_either_way():
    """The bulk control, as a number rather than a set identity: Si's cutoff
    stops inside the fitted channels, so nothing is lost before or after."""
    si = _zx(*_fcc(A_SI))
    before = _lost_lr_weight(si, [abs(int(g)) for g in V.DEG_B26P])
    after = _lost_lr_weight(si, list(V.lr_fit_degrees(si)))
    assert abs(before) < 1e-12 and abs(after) < 1e-12
    assert after <= before + 1e-15          # never degraded


# --------------------------------------------------------------------------
# GATE (a) -- the thick-slab vacuum recovery
# --------------------------------------------------------------------------
@pytest.mark.parametrize("vac,was_lost", [(1.5, 0.0261), (2.0, 0.0717),
                                          (3.0, 0.1793)])
def test_vacuum_padding_no_longer_strands_long_range_weight(vac, was_lost):
    """THE DEFECT, AND ITS REPAIR.  A fixed |G_z| <= 3 spans a shrinking energy
    window as vacuum grows; an energy cutoff does not.  The FALSE arm here is
    the old rule, and it must still show the loss that opened the decision."""
    zx = _zx(*_hex_slab(A_MOS2, C_MOS2 * vac))
    old = _lost_lr_weight(zx, [abs(int(g)) for g in V.DEG_B26P])
    new = _lost_lr_weight(zx, list(V.lr_fit_degrees(zx)))
    assert old == pytest.approx(was_lost, abs=2e-4), "red twin stopped failing"
    assert new <= 0.01, f"x{vac} vacuum still loses {100 * new:.3f}%"
    assert new < old / 5.0


# --------------------------------------------------------------------------
# GATE (b) -- the umklapp roll of the head channel
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,geom", [("si", _fcc(A_SI)),
                                       ("mos2_ref", _hex_slab(A_MOS2, C_MOS2)),
                                       ("mos2_x3", _hex_slab(A_MOS2, 3 * C_MOS2))])
def test_head_channel_stays_fitted_when_g0_rolls_at_a_zone_boundary(name, geom):
    """G=0 IS NOT THE HEAD SLOT AT A ZONE BOUNDARY.  Past the boundary along z
    the argmin rolls onto G = [0,0,-1], so the head magnitude lands on the
    |G_z| = 1 channel.  If the criterion does not fit that channel its form
    factor is identically zero and the head is multiplied away in silence
    (``minibz_head_vlr`` raises rather than let that happen).  This is the
    owner's stated reason for the floor, gated."""
    zx = _zx(*geom)
    assert _head_channel(zx, [0.1, 0.1, 0.0]) == 0        # interior: G=0
    rolled = _head_channel(zx, [0.1, 0.1, 0.6])           # past the boundary
    assert rolled == 1, f"{name}: expected an umklapp roll, got |G_z|={rolled}"
    assert rolled in V.lr_fit_degrees(zx), (
        f"{name}: the rolled head channel |G_z|={rolled} is not fitted")


@pytest.mark.parametrize("name,geom", [("si", _fcc(A_SI)),
                                       ("mos2_ref", _hex_slab(A_MOS2, C_MOS2)),
                                       ("mos2_x3", _hex_slab(A_MOS2, 3 * C_MOS2))])
def test_red_twin_a_literal_first_shell_criterion_drops_the_rolled_head(name, geom):
    """THE FALSE ARM.  Same criterion, floor lowered to the literal first shell
    (|G_z| = 0 only).  It must fail the cell above on every geometry --
    otherwise the floor is not what is carrying the umklapp guarantee and the
    gate above proves nothing."""
    zx = _zx(*geom)
    only_first = V.lr_fit_degrees(zx, e_cut=1e-12, shell_floor=0)
    assert set(only_first) == {0}
    assert _head_channel(zx, [0.1, 0.1, 0.6]) not in only_first


def test_two_shells_is_the_smallest_floor_that_keeps_a_bulk_cell_whole():
    """WHY 'AT LEAST 2', numerically.  On Si the cutoff alone would keep only
    |G_z| = 0 (|b3| = 1.06, so one shell already costs 1.12 Ry), and the floor
    is the whole of what protects the bulk cell.  A third shell buys nothing
    because Si's superset stops at |G_z| = 2, so two is exactly the minimum --
    which is what the owner's 'at least 2' resolves to on real geometry."""
    si = _zx(*_fcc(A_SI))
    lost = [_lost_lr_weight(si, list(V.lr_fit_degrees(si, shell_floor=f)))
            for f in (0, 1, 2, 3)]
    assert lost[0] > 0.40                    # 48.4% -- catastrophic
    assert 0.01 < lost[1] < 0.02             # 1.37% -- still over the bar
    assert abs(lost[2]) < 1e-12              # 0.000%
    assert lost[3] == pytest.approx(lost[2], abs=1e-15)
    assert V.FIT_SHELL_FLOOR == 2


# --------------------------------------------------------------------------
# the default, and the shape of the criterion
# --------------------------------------------------------------------------
def test_the_default_cutoff_sits_inside_the_backward_compatible_window():
    """The default has to reproduce today's channel set on the MoS2 reference
    deck, which pins it to [(3|b3|)^2, (4|b3|)^2) = [0.691, 1.229) Ry there.
    The 0.5 Ry sketch is BELOW that window -- it would drop |G_z| = 3 and make
    the reference deck WORSE (0.24% -> 1.90% lost) -- which is why the default
    is 1.0 Ry.  This cell is what fails if someone retunes it carelessly."""
    b3 = np.linalg.norm(_hex_slab(A_MOS2, C_MOS2)[0][2])
    assert (3 * b3) ** 2 <= V.E_CUT_FIT < (4 * b3) ** 2
    zx = _zx(*_hex_slab(A_MOS2, C_MOS2))
    assert set(V.lr_fit_degrees(zx, e_cut=0.5)) == {0, 1, 2}      # the sketch
    assert set(V.lr_fit_degrees(zx)) == {0, 1, 2, 3}              # the default


def test_the_criterion_is_an_energy_cutoff_not_a_shell_count():
    """The property the whole ruling turns on: at fixed cutoff the number of
    fitted shells must GROW as |b3| shrinks.  A fixed count would be flat."""
    n = [len(V.lr_fit_degrees(_zx(*_hex_slab(A_MOS2, C_MOS2 * f))))
         for f in (1.0, 1.5, 2.0, 3.0)]
    assert n == sorted(n) and n[-1] > n[0]
    # and each is the largest n with (n |b3|)^2 <= E_cut, floor aside
    for f in (1.5, 2.0, 3.0):
        zx = _zx(*_hex_slab(A_MOS2, C_MOS2 * f))
        b3 = np.linalg.norm(zx["bvec"][2])
        nmax = max(V.lr_fit_degrees(zx))
        assert (nmax * b3) ** 2 <= V.E_CUT_FIT < ((nmax + 1) * b3) ** 2


def test_a_nonpositive_cutoff_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        V.lr_fit_degrees(_zx(*_fcc(A_SI)), e_cut=0.0)
