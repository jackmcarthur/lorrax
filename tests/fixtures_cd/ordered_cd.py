"""Contour-deformation Sigma_c without time-reversal assumptions (lane TRREF).

Same split, anchors, imaginary rule and Hermite real-axis selectors as the CD harness of record
(313_si_control_20260908/decisive_cd.py:14-50 with Run255 selector_jets.py). Its two time-reversal-even steps are
replaced by their exact forms:

  imaginary axis  W_q(-iu) = W_q(iu)^dagger.  The harness folds both half-lines into d/(d^2+u^2) on W_q(iu)
                  (selector_jets.py:48), which assumes W_q(-iu) = W_q(iu).  Here the -iu half-line reads
                  M_ab(W^dagger) = conj(M_ba(W)) with the kernel 1/(d+iu).
  real axis       the occupied residue (x < 0) needs W_q(x - i eta) = W_{-q}(|x| + i eta)^T.  The harness reads
                  W_q(|x| + i eta) (selector_jets.py:65,70).

Per half-line anchors: a+ matches M(W(i eta)) and a- matches M(W(i eta)^dagger). Each cancels the kernel pole
at u = eta that the x = 0 (own-energy) column puts on its own half-line.  The analytic anchor integral is
(1/2pi) int [s_-(u) a+ + s_+(u) a-] = anchor_selector((a+ + a-)/2) + (i/2pi) K (a+ - a-).

Streaming form (used by the CrI3 callback, and by integrate_ordered, so the planted gate covers it): the
imaginary node sum is accumulated WITHOUT anchors, sum_i [s_-(u_i) M(W_i) + s_+(u_i) M(W_i^dagger)].  The anchor
subtraction is linear in (a+, a-) and uses only the selector sums S_-(beta) = sum_i s_-(u_i)/(u_i^2+beta^2) and
S_+(beta).

Layouts (Ry, s = z^2): contractions of -Wc and d/ds are [..., k, bra, ket, l]; x and occ are [k, bra, E, l].
Every function takes ``xp`` (numpy or jax.numpy); the selector jets are numpy and are moved to ``xp`` once.
"""
import numpy as np

EV = 13.605693122994
BETA = np.array([.5, 4.]) / EV


def _apply(a, w, xp=np):
    return xp.einsum('kabl,kael->kabe', a, w)


def _dagger(a, xp=np):
    """[..., bra, ket, l] contraction of X -> contraction of X^dagger."""
    return xp.conj(xp.swapaxes(a, -3, -2))


def _anchor_fit(value, derivative_s, eta):
    """decisive_cd.py:22-26: two-pole anchor matching value and d/ds at i*eta."""
    derivative = -derivative_s
    d1, d2 = eta**2 + BETA**2
    y1 = (derivative + value/d2) / (1/d2 - 1/d1)
    return y1*d1, (value - y1)*d2


def _odd_anchor(d, x, beta, zero_angle, xp=np):
    """K = int_0^inf u du / ((u^2+d^2)(u^2+beta^2)) = ln(beta^2/d^2)/(2(beta^2-d^2)); x -> 0+ branch at x = 0."""
    d2 = d*d
    angle = xp.where(x == 0, zero_angle, xp.angle(d2))
    return (np.log(beta*beta) - (xp.log(xp.abs(d2)) + 1j*angle)) / (2*(beta*beta - d2))


def kernels(x, occ, eta, xp=np):
    f = xp.broadcast_to(xp.asarray(occ), x.shape)
    return f, x - 1j*eta, x + 1j*eta


def anchor_part(value0, deriv0, x, occ, jets, *, eta, xp=np):
    """Analytic anchor integral and the anchor coefficients (cp, cm)."""
    f, dv, dc = kernels(x, occ, eta, xp)
    cp = _anchor_fit(value0, deriv0, eta)
    cm = _anchor_fit(_dagger(value0, xp), _dagger(deriv0, xp), eta)
    fh = np.broadcast_to(np.asarray(occ), np.shape(x))
    xh = np.asarray(x)
    total = 0
    for p, m, b in zip(cp, cm, BETA):
        odd = (f*_odd_anchor(dv, x, b, -np.pi, xp) + (1 - f)*_odd_anchor(dc, x, b, np.pi, xp)) * (1j/(2*np.pi))
        total = total + _apply(.5*(p + m), xp.asarray(jets.anchor_selector(xh, fh, eta, b)), xp)
        total = total + _apply(p - m, odd, xp)
    return total, cp, cm


def imag_node_kernels(u, w, x, occ, eta, xp=np):
    f, dv, dc = kernels(x, occ, eta, xp)
    minus = (f/(dv - 1j*u) + (1 - f)/(dc - 1j*u)) * (w/(2*np.pi))
    plus = (f/(dv + 1j*u) + (1 - f)/(dc + 1j*u)) * (w/(2*np.pi))
    return minus, plus


def imag_raw_node(value_i, u, w, x, occ, *, eta, xp=np):
    """Anchor-free contribution of one imaginary node, as the pair (minus-branch, plus-branch)."""
    minus, plus = imag_node_kernels(u, w, x, occ, eta, xp)
    return _apply(value_i, minus, xp), _apply(_dagger(value_i, xp), plus, xp)


def imag_anchor_sums(u_list, w_list, x, occ, *, eta, xp=np):
    """S_-(beta), S_+(beta) for both betas: selector sums the anchor subtraction needs."""
    sums = [[0, 0], [0, 0]]
    for u, w in zip(u_list, w_list):
        minus, plus = imag_node_kernels(u, w, x, occ, eta, xp)
        for j, b in enumerate(BETA):
            sums[j][0] = sums[j][0] + minus / (u*u + b*b)
            sums[j][1] = sums[j][1] + plus / (u*u + b*b)
    return sums


def imag_anchor_subtraction(cp, cm, sums, xp=np):
    out = 0
    for j in range(len(BETA)):
        out = out + _apply(cp[j], sums[j][0], xp) + _apply(cm[j], sums[j][1], xp)
    return out


def real_part(values, derivatives, values_t, derivatives_t, x, occ, real_ev, jets, *, eta, n_active, h,
              base_ev, xp=np):
    """Hermite residues: x >= 0 reads W_q (values), x < 0 reads W_{-q}^T (values_t); node i is real_ev[i]."""
    xh = np.asarray(x)
    fh = np.broadcast_to(np.asarray(occ), xh.shape)
    active_h = (np.arange(xh.shape[-1]) < n_active)[None, None, None, :]
    weight = np.where(xh < 0, fh, 1 - fh)
    cut = real_ev[-1] / EV
    if np.any(active_h & (weight != 0) & (np.abs(xh) > cut)):
        raise RuntimeError('GATE trref_residue_coverage: an active residue lies beyond the last real node; '
                           'no clipping, no omitted tail')
    rule = jets.hermite_selectors(xh, fh, h/EV)
    index = xp.asarray(rule['index'])
    coef = {k: xp.asarray(rule[k]) for k in ('value_left', 'value_right', 'derivative_left', 'derivative_right')}
    active = xp.asarray(active_h)
    neg = xp.asarray(xh < 0)
    part = 0
    for inode in range(0, len(real_ev), round(h/base_ev)):
        node = round(inode*base_ev/h)
        z = real_ev[inode]/EV + 1j*eta
        for deriv, a, at in ((False, values[inode], values_t[inode]),
                             (True, derivatives[inode]*2*z, derivatives_t[inode]*2*z)):
            lk = 'derivative_left' if deriv else 'value_left'
            rk = 'derivative_right' if deriv else 'value_right'
            sel = xp.where(active, xp.where(index == node, coef[lk], 0j) + xp.where(index + 1 == node, coef[rk], 0j), 0j)
            part = part + _apply(a, xp.where(neg, 0j, sel), xp)
            part = part + _apply(at, xp.where(neg, sel, 0j), xp)
    return part


def integrate_ordered(values, derivatives, values_t, derivatives_t, x, occ, schedule, jets, *, eta, n_active,
                      spacings_ev=(0.0625, 0.125), base_ev=0.0625, xp=np):
    """Returns {f"n{n}_h{h}": [k, bra, ket, E]}; the composition of the streaming pieces above."""
    anchor, cp, cm = anchor_part(values[0], derivatives[0], x, occ, jets, eta=eta, xp=xp)
    imag = {}
    for n in sorted(map(int, schedule['imaginary'])):
        rule = schedule['imaginary'][str(n)]
        raw = 0
        for i, u, w in zip(rule['indices'], rule['u_ry'], rule['weights_ry']):
            a, b = imag_raw_node(values[i], u, w, x, occ, eta=eta, xp=xp)
            raw = raw + a
            raw = raw + b
        sums = imag_anchor_sums(rule['u_ry'], rule['weights_ry'], x, occ, eta=eta, xp=xp)
        imag[n] = raw - imag_anchor_subtraction(cp, cm, sums, xp)
    real = {h: real_part(values, derivatives, values_t, derivatives_t, x, occ, schedule['real_ev'], jets,
                         eta=eta, n_active=n_active, h=h, base_ev=base_ev, xp=xp) for h in spacings_ev}
    return {f'n{n}_h{h}': anchor + imag[n] + real[h] for n in imag for h in real}


def even_inputs(values, derivatives, values_t, derivatives_t, n_real, xp=np):
    """Integrator inputs for W^even_q = (W_q + W_{-q}^T)/2, from the same contractions (coordinator ruling 2026-09-15).

    Real nodes (index < n_real): M(W^even_q; P_q) = (M(W_q; P_q) + M(W_{-q}^T; P_q))/2. Occupied residues read
    W^even_{-q}^T = W^even_q, so the even values serve both branches. Imaginary nodes: W_{-q}(iu)^T = W_q(iu)^dagger,
    so M(W^even) = (M(W) + M(W^dagger))/2 and needs no extra contraction. Only real-node derivatives are read.
    Sigma^odd = integrate_ordered(values, ...) - integrate_ordered(*even_inputs(...)).
    """
    ve = xp.concatenate([0.5*(values[:n_real] + values_t[:n_real]),
                         0.5*(values[n_real:] + _dagger(values[n_real:], xp))])
    de = 0.5*(derivatives[:n_real] + derivatives_t[:n_real])
    return ve, de, ve, de
