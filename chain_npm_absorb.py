"""Absorb-arm helpers: give Chain-NPM the per-table family's cardinality and
leaf-marginal accuracy without touching its cross-hop objects.

Everything here is flag-gated.  With `absorb='off'` the models take exactly the
code path they took before this module existed (bit-identical output).

Mechanisms
----------
totals
    Release each table's row count as a scalar Gaussian query of l2 sensitivity
    1, and calibrate the synthesized cardinalities to it.  Under the
    manuscript's one-individual adjacency (Lemma lem:sens: neighbours differ in
    a single tuple i* of group g*) a table cardinality moves by at most 1, so
    sigma_total = sigma_unit while the relative error is sigma/n -- three orders
    of magnitude better than the "noisy group count x noisy mean size" product
    the sampler uses today.  This replaces the two MEAN_SIZE draws of the
    three-level rail, which the code already documents as uncalibratable
    (sigma ~ tens vs a mean of tens) and which are dead budget: they are drawn,
    counted in n_queries, and never read in Phase 2.

pubn
    Take the table cardinalities as public schema metadata.  This is not a new
    assumption for the comparison: every per-table baseline we measure against
    (pertable / denorm / privmrf / privbayes / pbpgm / PrivLava / PrivPetal)
    synthesizes exactly n rows per table.  It is required for the star rail,
    whose stated adjacency is a root-tuple cascade ("one R1 tuple together with
    its C rows", Appendix B): there |C| has sensitivity = max group size, so a
    paid release of the child cardinality would be useless noise.  Arms using
    `pubn` must say so in `hyper` and in the paper.

bins
    Tail-covering geometric group-size histogram: SIZE_BINS bins over [1, n]
    instead of the (tau+1)-wide tau-truncated histogram.  Two defects of the
    truncated histogram disappear at once:
      (i)  mass loss -- groups larger than tau are capped, which for heavy-tailed
           schemas throws away most of the leaf rows (instacart synthesizes
           0.41-0.79 of the true child table, and the deficit *grows* with the
           budget because at low budget the noise bias below partly hides it);
      (ii) mean inflation -- each of the ~tau cells carries independent noise
           and is clipped at 0, so an empty cell contributes ~0.4*sigma of
           spurious mass spread over the whole range.  With tau = 459 and
           sigma = 345 (financial, eps = 0.1) that is ~63k spurious groups
           against 77 real ones, which is why the mid table comes out ten times
           too large.  Twelve bins cut the spurious mass by ~tau/12.
    The upper edge n comes from `totals`/`pubn`, so the bin grid leaks nothing
    beyond the cardinality that arm already has.  Weights stay 1/min(s, tau):
    only the *histogram* stops truncating, so Lemma lem:sens is untouched.
    Sensitivity of a size histogram entry is 1 under the star's root-cascade
    adjacency and sqrt(2) under the chain's one-individual adjacency (removing
    i* moves its group from one bin to the next).

rowleaf
    Measure every object at ROW level instead of group-normalized, i.e. weight
    each tuple by 1 rather than by 1/min(s, tau).  This is the single largest
    lever in the absorb arm and it costs no budget at all: same queries, same
    sigma, no new lemma.

    Why it matters.  Group normalization caps the mass of every released object
    at the number of GROUPS one level up: on financial the leaf 1-way marginals
    carry mass 4500 (the account count) although the leaf table has 1 056 320
    rows, the mid 1-ways carry 77 (the district count), and the cross-hop
    top x leaf object carries 4500 spread over up to 80 cells.  At eps=3.2
    (sigma=13.85) a cross-hop cell then holds ~56 against noise 13.85, i.e. 25%
    relative error, and that error is multiplied into EVERY synthesized leaf row
    by the lift -- which is why the released arm's 0-hop leaf error stays at
    0.09 even after its cardinalities are fixed.  Row level puts the same object
    at mass 1 056 320: relative error 1.3e-5.

    The price is the adjacency.  A group-normalized object moves by at most 1
    when a whole group is added or removed, which is what makes the released
    mechanism DP under a cascading neighbour (remove one parent tuple together
    with its descendants) as well as under a one-tuple neighbour.  A row-level
    object moves by the size of the cascade, so `rowleaf` is DP under the
    ONE-INDIVIDUAL adjacency only -- the adjacency Lemma lem:sens is written for
    ("neighbouring databases differing in a single individual tuple i* of group
    g*"), and the one the per-table baselines need, since a method that releases
    row-level counts of its own table is only DP under a one-row neighbour.  It
    gives up group-presence protection, which the group-normalized family
    (PrivLava, PrivPetal) keeps.  Whether the paper wants that trade is a
    decision for the authors, not for this module: the arm records
    `adjacency: one-individual` in hyper so no result can be reported without it.

szsens
    Charge that sqrt(2) on the chain rails' size histograms.  The released code
    charges 1 there, which is what the truncated histogram needs only if a group
    never moves between bins; it does.  This is a pre-existing undercharge of the
    default arm, isolated behind its own flag so the attribution matrix can price
    it separately instead of folding it into `bins`.

exactmarg
    Free post-processing.  Sample root/parent attribute values by quota -- exactly
    round(n*p) of each value, randomly permuted -- instead of i.i.d., so the
    synthesized table reproduces the released 1-way marginal exactly rather than
    within the 1/sqrt(n) multinomial floor (see exact_draw).  The permutation
    keeps the draw exchangeable, so objects that do not read row order are
    unaffected in distribution.

repair
    Free post-processing.  The released group-normalized leaf marginals C_a are
    precise (mass = number of groups, so relative noise sigma/(n1*p)), but the
    sampler draws each leaf row from a much noisier object -- the cross-parent
    triple cell, whose mass is n1 spread over dom(a)*dom(b)*dom(c) cells.  The
    realized row-level marginal therefore drifts.  `repair` transports the
    sampled leaf values back onto the released marginal rescaled to the
    synthesized row count, moving the least-supported rows first.  No budget.
    It only helps in a `rowleaf` arm, where the released leaf 1-way IS the
    row-level target; under group normalization the reconstructed target is
    coarser than what the sampler already achieves (measured on financial at
    eps=0.8: 0-hop 0.024 -> 0.105 when repair is used without rowleaf).

growcap
    Calibrate the leaf cardinality by growing only the groups that tau truncation
    capped, instead of stretching every group proportionally.  Proportional
    rescaling is silently clamped by the tau cap when tau * n_groups falls below
    the target (measured on a TPC-H-shaped toy: the child row count stops at 0.475
    of the truth however the total is released), and where it does fit it
    distorts the whole size histogram.  See grow_capped.

ipf
    Fit the leaf attribute distribution by iterative proportional fitting on the
    (all top attrs x all mid attrs x leaf attr) tensor, constrained by the released
    adjacent and cross-hop pair objects, instead of multiplying their pointwise
    lifts.  The product form double-counts the part of the cross-hop association
    that is mediated by the middle table; group-normalized objects hide this
    because their SNR guards shrink the lifts, but in a `rowleaf` arm the lifts
    are precise and the amplification shows up in the planted-effect benchmark.
    See ipf_fit.

ratiotop
    Estimate each top-table 1-way marginal from the row sums of the ROW-LEVEL
    adjacent object (mid rows per top-attribute value, rescaled to n_top) instead
    of from the top table's own released marginal.  The top table is the smallest
    in the chain -- financial has 77 districts -- so its own marginal carries
    144-500% relative error at these budgets while the descendant row sums carry
    ~1%.  The estimator is biased by variation in descendants per top row, which
    is much smaller than the noise it replaces.  Free post-processing.
"""

import numpy as np

SIZE_BINS = 12          # pre-registered geometric bin count for `bins`
TAIL_BINS = 8           # pre-registered geometric tail bins for `tailbins`
_SZ_SENS_STAR = 1.0     # root-cascade adjacency: one histogram entry moves
_SZ_SENS_CHAIN = np.sqrt(2.0)   # one-individual adjacency: bin s -> bin s-1


def parse(spec):
    """'off' | comma list over {totals, pubn, bins[:N], repair, rowleaf, szsens}
    | 'all'.

    `totals` and `pubn` are alternative sources of the table cardinalities and
    should not both be on; `pubn` wins when they are.  `all` deliberately does
    NOT include `rowleaf`, because that token changes the adjacency the arm is
    DP under and must always be requested explicitly.
    """
    out = {'totals': False, 'pubn': False, 'bins': 0, 'tailbins': 0,
           'repair': False, 'rowleaf': False, 'szsens': False,
           'szthresh': 0.0, 'exactmarg': False, 'growcap': False,
           'ipf': False, 'ratiotop': False, 'budget': 0.0, 'shrink': 0.0,
           'szquota': False}
    if not spec or str(spec).strip().lower() in ('off', 'none', ''):
        return out
    for tok in str(spec).split(','):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok == 'all':
            out.update(totals=True, pubn=True, tailbins=TAIL_BINS, repair=True,
                       szsens=True, szthresh=CLIP_BIAS, exactmarg=True,
                       growcap=True, ipf=True, ratiotop=True)
        elif tok == 'szthresh':
            out['szthresh'] = CLIP_BIAS
        elif tok.startswith('szthresh:'):
            out['szthresh'] = float(tok.split(':', 1)[1])
        elif tok.startswith('tailbins'):
            n = TAIL_BINS
            if ':' in tok:
                n = int(tok.split(':', 1)[1])
            out['tailbins'] = max(0, n)
            out['bins'] = 0
        elif tok.startswith('bins'):
            n = SIZE_BINS
            if ':' in tok:
                n = int(tok.split(':', 1)[1])
            out['bins'] = max(0, n)
            out['tailbins'] = 0
        elif tok.startswith('budget'):
            # `budget[:w1[:w2]]`: reweight the gamma^2 allocation so that a
            # 1-way (P1/P2/C) query draws w1 times the budget share of a
            # joint query, which draws w2 times a unit share.  Default w1=3,
            # w2=0.5.  The total spent stays exactly gamma^2 (weights are
            # normalized), so the privacy account is unchanged; this mirrors
            # the per-table baselines that spend most of their epsilon on the
            # 1-way marginals of each table.
            w1, w2 = 3.0, 0.5
            parts = tok.split(':')
            if len(parts) > 1 and parts[1]:
                w1 = float(parts[1])
            if len(parts) > 2 and parts[2]:
                w2 = float(parts[2])
            out['budget'] = (w1, w2)
        elif tok.startswith('shrink'):
            # `shrink[:kappa]`: per-cell reliability gate for the joint
            # objects (XP/P1C/P2C/XPC).  A cell whose noisy count is below
            # kappa*sigma is shrunk toward the product of its marginals times
            # the released total (i.e. toward independence, which is exactly
            # what the author-protocol re-link of the per-table baselines
            # produces).  Default kappa=1.0.  Pure post-processing of released
            # values: no budget, no sensitivity change.
            kappa = 1.0
            if ':' in tok:
                kappa = float(tok.split(':', 1)[1])
            out['shrink'] = kappa
        elif tok.startswith('szquota'):
            # (star/chain) calibrate the per-parent row quotas from the
            # ROW-LEVEL adjacent marginal instead of from the group-level
            # size object.  On instacart at eps=0.1 the SZ object carries
            # ~17 groups/cell against sigma 263 (SNR 0.07, pure noise), so
            # the synthesized per-DEPT child counts come out 0.2x..5.6x off;
            # the per-table baselines dodge this because their relink uses
            # the (noisy but unbiased, per-parent) FK histogram the authors
            # release for 0.2 of the budget.  The P1C marginal under
            # `rowleaf` is itself row-level (per-DEPT cell mass ~5.9e3 vs
            # sigma 263 -> SNR >= 100) and its parent-axis sums ARE the
            # per-parent quotas; `szquota` re-calibrates the drawn sizes so
            # each parent-attribute value hits its row-level quota.  Costs
            # nothing, changes no budget, and is only meaningful with
            # `rowleaf` (otherwise P1C is group mass, not row mass).
            out['szquota'] = True
        elif tok in out:
            out[tok] = True
        else:
            raise ValueError('unknown absorb token %r' % tok)
    return out


def describe(opts):
    """Compact, JSON-safe description for `hyper`."""
    if not any([opts['totals'], opts['pubn'], opts['bins'], opts['tailbins'],
                opts['repair'], opts['rowleaf'], opts['szsens'],
                opts['szthresh'], opts['exactmarg'], opts['growcap'],
                opts['ipf'], opts['ratiotop'], opts['budget'],
                opts['shrink'], opts['szquota']]):
        return 'off'
    parts = []
    if opts['totals']:
        parts.append('totals')
    if opts['pubn']:
        parts.append('pubn')
    if opts['bins']:
        parts.append('bins%d' % opts['bins'])
    if opts['tailbins']:
        parts.append('tailbins%d' % opts['tailbins'])
    if opts['szthresh']:
        parts.append('szthresh' if abs(opts['szthresh'] - CLIP_BIAS) < 1e-9
                     else 'szthresh%g' % opts['szthresh'])
    if opts['budget']:
        w1, w2 = opts['budget']
        parts.append('budget%g_%g' % (w1, w2)
                     if abs(w1 - 3.0) > 1e-9 or abs(w2 - 0.5) > 1e-9
                     else 'budget')
    if opts['shrink']:
        parts.append('shrink' if abs(opts['shrink'] - 1.0) < 1e-9
                     else 'shrink%g' % opts['shrink'])
    if opts['szquota']:
        parts.append('szquota')
    if opts['exactmarg']:
        parts.append('exactmarg')
    if opts['growcap']:
        parts.append('growcap')
    if opts['ipf']:
        parts.append('ipf')
    if opts['ratiotop']:
        parts.append('ratiotop')
    if opts['rowleaf']:
        parts.append('rowleaf')
    if opts['repair']:
        parts.append('repair')
    if opts['szsens']:
        parts.append('szsens')
    return '+'.join(parts)


def adjacency(opts):
    """The neighbour relation the arm's release is DP under.  Recorded in hyper
    so that no rowleaf result can be reported without its caveat."""
    return 'one-individual' if opts['rowleaf'] else \
        'one-individual + cascading (group-normalized)'


def row_weight(opts, group_weight):
    """Per-tuple weight: 1 under `rowleaf`, else the group-normalized weight."""
    w = np.asarray(group_weight, dtype=float)
    return np.ones_like(w) if opts['rowleaf'] else w


def budget_sigma(g2, qnames, is_1way, w1=3.0, w2=0.5):
    """Per-query sigma under a `budget` reweighting.

    `g2` is the TOTAL gamma^2 budget of the rail (the square of
    analytic_gaussian_gamma(eps, delta)); the released code divides it evenly
    over qnames, which (with sigma_i = sqrt(2/(g2/len))) satisfies the account
    sum_i 1/sigma_i^2 = g2/2.  Under `budget` a 1-way query (single-table
    marginal: P1/P2/C on the star) gets w1 times the unit share and a joint
    query gets w2 times it; the same total g2/2 is preserved, so the privacy
    account is exactly unchanged: only the allocation moves.

    Returns sigma per query name (dict) for sensitivity-1 queries; callers
    scale by the query's actual sensitivity (SZ -> sz_sens) as the released
    code does.  Second return is the normalized per-query share (g2_i/2).
    """
    n = len(qnames)
    n1 = sum(1 for k in qnames if is_1way(k))
    nj = n - n1
    denom = w1 * n1 + w2 * nj if nj else w1 * n1
    out = {}
    for k in qnames:
        w = w1 if is_1way(k) else w2
        out[k] = np.sqrt(2.0 * denom / (g2 * w))
    return out, g2 / (2.0 * denom)


def is_1way_name(k):
    """True for single-table marginal queries on any rail (1-way marginals
    AND the size histogram: the size distribution is a per-parent histogram,
    and `budget` must not starve the size_tv indicator the manuscript
    reports)."""
    for prefix in ('P1_', 'P2_', 'C_', 'P_', 'T_', 'M_', 'B_', 'SZ_'):
        if k.startswith(prefix):
            return True
    return False


def shrink_joint(noisy, marg, sigmas, kappa=1.0, total=None):
    """Per-cell reliability gate toward independence for a joint object.

    noisy   : released (clipped) joint counts, shape (d1, ..., dk)
    marg    : list of the k marginals used by the joint (each 1-D, in the
              same axis order as noisy), as released counts or probabilities;
              only the mass ratio matters.
    sigmas  : scalar sigma, or array with one sigma per axis cell (per-query
              sigma when `budget` is active)
    kappa   : gate constant; a cell with noisy mass < kappa*sigma is shrunk
    total   : the released total mass of the joint (used to scale the
              independence product); defaults to noisy.sum()

    Returns a copy of noisy where each cell c with noisy[c] < kappa*sigma_c
    is replaced by a convex mix of itself and the independence cell mass
      total * prod_i(marg_i[c_i] / marg_i.sum()).

    The independence term is itself derived from released marginals, so the
    whole operation is post-processing of released values with no privacy
    cost and no change of sensitivity.
    """
    noisy = np.asarray(noisy, dtype=float)
    if kappa <= 0:
        return noisy
    total = float(total if total is not None else noisy.sum())
    if total <= 0 or noisy.size == 0:
        return noisy
    marg = [np.asarray(m, dtype=float) for m in marg]
    for m in marg:
        if m.sum() <= 0:
            return noisy
    if np.isscalar(sigmas):
        sigmas = np.full(noisy.shape, float(sigmas))
    else:
        sigmas = np.broadcast_to(np.asarray(sigmas, dtype=float),
                                 noisy.shape)
    # independence cell mass = released total x product of marginal shares
    indep = np.ones(noisy.shape, dtype=float)
    for axis, m in enumerate(marg):
        sh = (m / m.sum()).reshape(
            [1] * axis + [-1] + [1] * (noisy.ndim - axis - 1))
        indep = indep * sh
    indep = indep * total
    thr = kappa * sigmas
    w = np.minimum(1.0, np.maximum(0.0, noisy / np.maximum(thr, 1e-12)))
    return w * noisy + (1.0 - w) * indep


# ------------------------------------------------------------------ #
# geometric size bins                                                 #
# ------------------------------------------------------------------ #

def geo_grid(nmax, nbins=SIZE_BINS, lo=1):
    """Bin edges/representatives over group sizes lo..nmax.

    Returns (edges, reps): edges has len(reps)+1 entries with edges[0] = lo and
    edges[-1] = nmax + 1; bin b covers sizes edges[b] .. edges[b+1]-1 and reps[b]
    is its geometric midpoint, clipped into the bin.  Degenerate ranges (fewer
    integers than bins) fall back to one bin per integer.
    """
    lo = int(max(0, lo))
    nmax = int(max(lo + 1, nmax))
    span = nmax - lo + 1
    nbins = int(max(1, min(nbins, span)))
    if span <= nbins:
        edges = np.arange(lo, nmax + 2)
    else:
        edges = np.unique(np.round(
            np.geomspace(float(max(lo, 1)), nmax + 1.0,
                         nbins + 1)).astype(np.int64))
        if edges[0] < lo:
            edges[0] = lo
        elif edges[0] > lo:
            edges = np.concatenate([[lo], edges])
    e_lo = edges[:-1]
    e_hi = np.maximum(edges[1:] - 1, e_lo)
    reps = np.maximum(e_lo, np.minimum(
        e_hi, np.round(np.sqrt(np.maximum(e_lo, 1) *
                               np.maximum(e_hi, 1))).astype(np.int64)))
    return edges.astype(np.int64), reps.astype(np.int64)


def hybrid_grid(tau, nmax, k=TAIL_BINS):
    """Exact bins 0..tau plus k geometric bins covering (tau, nmax].

    This is the grid the star/chain rails actually want.  `bins` alone replaces
    the whole histogram by ~12 geometric bins, which fixes the truncation mass
    loss but destroys the resolution where the mass lives: on financial at
    eps=3.2 it moved size_tv from 0.61 to 0.99.  Keeping every integer size up
    to tau and covering only the tail geometrically recovers the mass without
    giving up that resolution, at k extra columns instead of tau+1.
    """
    tau = int(max(0, tau))
    nmax = int(nmax)
    fine_e = np.arange(0, tau + 2, dtype=np.int64)   # bin i holds size i
    fine_r = np.arange(0, tau + 1, dtype=np.int64)
    if nmax <= tau or k <= 0:
        return fine_e, fine_r
    e, r = geo_grid(nmax, k, lo=tau + 1)
    # e[0] == tau + 1 == fine_e[-1]: drop the duplicated edge, keep every bin
    edges = np.concatenate([fine_e, e[1:]]).astype(np.int64)
    reps = np.concatenate([fine_r, r]).astype(np.int64)
    return edges, reps


def size_grid(opts, tau, nmax, zero_bin=True):
    """The arm's group-size grid, or None when it keeps the tau-truncated one.

    `tailbins` (preferred) keeps every integer size up to tau and covers the tail
    geometrically; `bins` replaces the whole range by geometric bins.  The upper
    edge is `nmax`, the child/leaf cardinality, which the arm must already hold
    through `pubn` or `totals` so that the grid leaks nothing new.
    """
    if opts['tailbins']:
        return hybrid_grid(tau, nmax, opts['tailbins'])
    if opts['bins']:
        e, r = geo_grid(nmax, opts['bins'])
        if zero_bin:
            e = np.concatenate([[0], e]).astype(np.int64)
            r = np.concatenate([[0], r]).astype(np.int64)
        return e, r
    return None


CLIP_BIAS = 0.39894     # E[max(Z, 0)] for Z ~ N(0,1): the mass a clipped-at-zero
                        # Gaussian injects into every empty histogram cell


def debias_hist(x, sigma, on=True, kappa=CLIP_BIAS):
    """Soft-threshold a clipped noisy histogram by kappa*sigma.

    Every released count is max(true + N(0, sigma), 0).  A cell whose true value
    is 0 therefore keeps E[max(sigma*Z, 0)] = 0.3989*sigma of spurious mass, and
    a group-size histogram has tau+1 such cells -- with tau = 459 and
    sigma = 345 (financial, eps = 0.1) that is ~63k spurious groups against 77
    real ones, whose bin average is tau/2, which is why the released arm
    synthesizes ten times too many leaf rows there.  Subtracting kappa*sigma
    before renormalizing removes exactly that expected injection; kappa is the
    constant above, not a tuned knob.  Free post-processing of released values.
    """
    x = np.asarray(x, dtype=float)
    if not on or sigma <= 0:
        return x
    return np.maximum(x - kappa * float(sigma), 0.0)


def to_bins(sizes, edges):
    idx = np.searchsorted(edges, np.asarray(sizes), side='right') - 1
    return np.clip(idx, 0, len(edges) - 2).astype(np.int64)


def from_bins(idx, reps):
    return np.asarray(reps)[np.clip(np.asarray(idx), 0, len(reps) - 1)]


# ------------------------------------------------------------------ #
# cardinality calibration                                             #
# ------------------------------------------------------------------ #

def grow_capped(sizes, target, lo=1, hi=None):
    """Calibrate the total by growing ONLY the groups the truncation capped.

    Proportional rescaling (rescale_to_total) spreads a cardinality deficit over
    every group, which is wrong when the deficit has a known origin: tau
    truncation loses rows only from groups whose true size exceeds tau, so the
    missing mass belongs to the groups that are sitting at the cap.  Stretching
    everyone instead distorts the whole size histogram -- measured on Instacart,
    where the truncated model reproduces 0.409 of the child rows, proportional
    rescaling moved size_tv from 0.195 to 0.414 and the cross-parent error from
    0.452 to 0.487.  Adding the deficit to the capped groups leaves every
    uncapped size exactly as the released histogram estimated it.

    A surplus (the clip-at-zero bias inflated the histogram) has no such
    localisation, so it is removed proportionally.  Sizes may exceed `hi` here by
    design: that is the tail the truncated histogram could not represent.
    """
    s = np.asarray(sizes, dtype=float).ravel()
    n = s.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    lo = int(lo)
    cur = np.clip(np.round(s).astype(np.int64), lo, hi)
    deficit = int(target) - int(cur.sum())
    if deficit == 0:
        return cur
    if deficit < 0 or hi is None:
        return rescale_to_total(s, target, lo=lo, hi=hi)
    at_cap = np.flatnonzero(cur >= int(hi))
    if at_cap.size == 0:
        return rescale_to_total(s, target, lo=lo, hi=None)
    add = np.zeros(n, dtype=np.int64)
    q, r = divmod(deficit, at_cap.size)
    add[at_cap] = q
    if r:
        add[at_cap[:r]] += 1
    return cur + add


def rescale_to_total(sizes, target, lo=1, hi=None):
    """Integer rescale of `sizes` so the sum equals `target`.

    Shape is preserved (multiplicative on the part above `lo`), the remainder is
    settled by largest fractional part, and every entry stays in [lo, hi].
    """
    s = np.asarray(sizes, dtype=float).ravel()
    n = s.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    lo = int(lo)
    hi = None if hi is None else int(hi)
    target = int(target)
    target = max(target, lo * n)
    if hi is not None:
        target = min(target, hi * n)
    base = np.full(n, lo, dtype=np.int64)
    need = target - lo * n
    if need <= 0:
        return base
    exc = np.maximum(np.round(s).astype(np.int64) - lo, 0)
    tot = int(exc.sum())
    if tot == 0:
        idx = np.argsort(-s, kind='stable')
        give = np.zeros(n, dtype=np.int64)
        left = need
        i = 0
        while left > 0 and i < n:
            room = need if hi is None else min(left, hi - lo)
            if room <= 0:
                break
            take = min(left, room)
            give[idx[i]] = take
            left -= take
            i += 1
        if left > 0 and hi is None:
            give[idx[0]] += left
        return base + give
    scaled = exc.astype(float) * (need / float(tot))
    add = np.floor(scaled).astype(np.int64)
    if hi is not None:
        add = np.minimum(add, hi - lo)
    rem = need - int(add.sum())
    if rem > 0:
        frac = scaled - np.floor(scaled)
        if hi is not None:
            frac = np.where(add >= hi - lo, -np.inf, frac)
        order = np.argsort(-frac, kind='stable')
        for j in order:                      # largest fractional part first
            if rem <= 0:
                break
            if hi is None or add[j] < hi - lo:
                add[j] += 1
                rem -= 1
        for j in order:                      # still short: fill any capacity
            if rem <= 0:
                break
            if hi is None:
                add[j] += rem
                rem = 0
                break
            room = hi - lo - int(add[j])
            if room > 0:
                take = min(room, rem)
                add[j] += take
                rem -= take
    elif rem < 0:
        order = np.argsort(-add, kind='stable')
        k = 0
        while rem < 0 and k < order.size:
            j = order[k]
            if add[j] > 0:
                add[j] -= 1
                rem += 1
            k += 1
    return base + add


def calibrate_sizes(opts, sizes, target, lo=1, hi=None):
    """Dispatch to the pre-registered cardinality-calibration strategy."""
    if opts['growcap']:
        return grow_capped(sizes, target, lo=lo, hi=hi)
    return rescale_to_total(sizes, target, lo=lo, hi=hi)


def quota_calibrate(sizes, group_ids, quota, lo=1, hi=None, grow=True):
    """Per-parent-attribute-value row-quota calibration (`szquota`).

    sizes      int array of drawn per-parent group sizes (length n_groups)
    group_ids  int array, the parent-attribute value of each group (same len)
    quota      float array indexed by attribute value: the row-level child
               count that value must produce (from the P1C marginal)
    lo, hi     same semantics as rescale_to_total / grow_capped
    grow       True: deficits are grown onto capped groups (grow_capped per
               value); False: simple proportional rescale per value

    Rescales the sizes of each attribute value independently to its row-level
    quota, keeping n_groups fixed and the RNG order untouched (no draws here).
    The released P1C under `rowleaf` is row-mass, so quota IS the exact-child
    count per value up to Gaussian noise (SNR~100 at instacart eps=0.1),
    whereas the group-level SZ object that sizes were drawn from has SNR 0.07
    and cannot support per-value quotas at all.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    group_ids = np.asarray(group_ids, dtype=np.int64)
    out = sizes.copy()
    nvals = len(quota)
    for v in range(nvals):
        sel = group_ids == v
        if not sel.any():
            continue
        tgt = int(round(float(quota[v])))
        if tgt <= 0:
            out[sel] = np.full(int(sel.sum()), lo, dtype=np.int64)
            continue
        if grow:
            out[sel] = grow_capped(out[sel], tgt, lo=lo, hi=hi)
        else:
            out[sel] = rescale_to_total(out[sel], tgt, lo=lo, hi=hi)
    return out


def quota_calibrate_2d(sizes, g1, g2, q1, q2, lo=1, n_iter=200, tol=1e-9):
    """Two-attribute joint quota calibration; see quota_calibrate_kd."""
    return quota_calibrate_kd(sizes, [g1, g2], [q1, q2], lo=lo,
                              n_iter=n_iter, tol=tol)


def quota_calibrate_kd(sizes, gs, qs, lo=1, n_iter=200, tol=1e-9):
    """Joint row-column (RAS) quota calibration (`szquota`) over k (>=1)
    parent attributes.

    Every group (root tuple) belongs to one value of each attribute k; the
    quotas qs[k][v] come from the row-level P1C marginals.  A per-attribute
    sequential calibration does NOT work: rescaling groups by attribute 2
    after attribute 1 destroys the attribute-1 quotas, because each group
    participates in every marginal (measured: instacart AISLE pass left the
    DEPT quotas at 0.20..2.95).  This is the classic RAS biproportional
    adjustment on the CELL tensor: cell (v1,..,vk) holds the groups with
    those attribute values, and the target cell mass is adjusted so ALL k
    marginal sums hit their quotas.  Alternating rescaling converges because
    every quota vector sums to the same released leaf cardinality.  The final
    cell masses are distributed among the cell's groups by proportion +
    largest remainder, keeping every group >= lo.  Groups are fixed; the RNG
    order is untouched (no draws here).  Reactive only under `rowleaf`
    (quota = row count); a no-op otherwise.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    gs = [np.asarray(g, dtype=np.int64) for g in gs]
    qs = [np.asarray(q, dtype=float) for q in qs]
    k = len(gs)
    if k == 0:
        return sizes
    if k == 1:
        return quota_calibrate(sizes, gs[0], qs[0], lo=lo, grow=False)
    # RAS requires all marginal totals to agree; each quota vector carries
    # its own Gaussian noise so the sums differ at the 0.1% level.  Rescale
    # every quota to the grand total so that (a) the cell table sums to the
    # same released cardinality on every axis and RAS converges exactly, and
    # (b) the synthesized leaf cardinality is preserved.
    grand = float(np.mean([float(q.sum()) for q in qs]))
    qs = [q * (grand / max(float(q.sum()), 1e-300)) for q in qs]
    # A marginal value with NO groups would force scale=1 (skip) and lose its
    # quota mass forever.  Fold those values' mass into the populated values
    # of the same axis first (probabilistically, breaks ties evenly), so RAS
    # sees a fully-populated marginal and conserves the total.
    for ax in range(k):
        pop = np.unique(gs[ax])
        freq = np.bincount(gs[ax], minlength=len(qs[ax])).astype(float)
        empty = freq <= 0
        if empty.any():
            dead = float(qs[ax][empty].sum())
            if dead > 0:
                n_pop = int((~empty).sum())
                qs[ax] = qs[ax].copy()
                qs[ax][~empty] += dead / max(n_pop, 1)
                qs[ax][empty] = 0.0
    dims = [len(q) for q in qs]
    # cell group counts
    cell = np.zeros(dims, dtype=np.int64)
    np.add.at(cell, tuple(gs), 1)
    # target cell mass: RAS on the counts (marginals must be strictly
    # positive somewhere; zero-mass rows/cols stay at their available mass)
    M = cell.astype(float).copy()
    for _ in range(int(n_iter)):
        worst = 0.0
        for ax in range(k):
            axes = tuple(i for i in range(k) if i != ax)
            cur = M.sum(axis=axes)
            scale = np.where(cur > 0,
                             qs[ax] / np.maximum(cur, 1e-300),
                             np.ones_like(cur))
            resh = [1] * k
            resh[ax] = dims[ax]
            M = M * scale.reshape(resh)
            err = float(np.abs(M.sum(axis=axes) - qs[ax]).max())
            worst = max(worst, err / max(qs[ax].max(), 1.0))
        if worst < tol:
            break
    out = sizes.copy()
    idx = np.flatnonzero(np.ones(sizes.shape[0], dtype=bool))
    for cell_key in np.ndindex(*dims):
        m = np.ones(sizes.shape[0], dtype=bool)
        for ax in range(k):
            m &= gs[ax] == cell_key[ax]
        sel = idx[m]
        if sel.size == 0:
            continue
        tgt = int(round(float(M[cell_key])))
        out[sel] = rescale_distribute(out[sel], lo, tgt)
    return out


def rescale_distribute(sizes, lo, target):
    """Distribute an integer `target` over `sizes` (all >= lo), preserving
    their shape as far as possible: base lo everywhere, the rest proportional
    to (sizes - lo) with largest-remainder rounding.  `target` is a hard
    floor of lo*n and has NO upper cap: the tail that the tau-truncated
    histogram dropped must be able to exceed tau (same argument as
    grow_capped), so capping at the current max size would silently
    under-deliver the quota."""
    s = np.asarray(sizes, dtype=np.int64)
    n = s.size
    lo = int(lo)
    target = int(target)
    if n == 0:
        return s
    target = max(target, lo * n)
    base = np.full(n, lo, dtype=np.int64)
    exc = np.maximum(s - lo, 0).astype(float)
    tot = int(exc.sum())
    need = target - lo * n
    if tot <= 0:
        # flat (all sizes at lo): spread `need` over the groups round-robin
        out = base.copy()
        full, rem = divmod(need, n)
        out += full
        out[:rem] += 1
        return out
    scaled = exc * (need / float(tot))
    add = np.floor(scaled).astype(np.int64)
    rem = need - int(add.sum())
    if rem > 0:
        order = np.argsort(-(scaled - np.floor(scaled)), kind='stable')
        for j in order:
            if rem <= 0:
                break
            add[j] += 1
            rem -= 1
    return base + add


def calibrate_total(noisy_total, fallback, lo=0):
    """Row count from a released total, with the old estimator as fallback.

    A released total is a Gaussian with sigma of the order of the whole budget
    slice, so at tiny budgets on tiny tables it can land far outside the
    plausible range.  Accept it when it is positive and within 100x of the
    fallback (the noisy-marginal estimator), else keep the fallback and record
    which one was used.
    """
    if noisy_total is None:
        return int(fallback), 'marginal'
    t = float(noisy_total)
    if not np.isfinite(t) or t <= lo:
        return int(fallback), 'marginal'
    f = max(float(fallback), 1.0)
    if t > 100.0 * f or t < 0.01 * f:
        return int(fallback), 'marginal'
    return int(round(t)), 'total'


# ------------------------------------------------------------------ #
# leaf marginal repair (free post-processing)                         #
# ------------------------------------------------------------------ #

def _round_to_sum(x, total):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s <= 0:
        out = np.full(x.size, total // max(x.size, 1), dtype=np.int64)
        out[:total - int(out.sum())] += 1
        return out
    scaled = x * (total / s)
    base = np.floor(scaled).astype(np.int64)
    rem = int(total - base.sum())
    if rem > 0:
        order = np.argsort(-(scaled - base), kind='stable')
        base[order[:rem]] += 1
    elif rem < 0:
        order = np.argsort(base, kind='stable')
        for j in order:
            if rem == 0:
                break
            if base[j] > 0:
                base[j] -= 1
                rem += 1
    return base


def exact_draw(rng, p, n):
    """Draw n category values whose histogram is EXACTLY round(n*p).

    An i.i.d. draw of n values leaves a multinomial relative error of order
    1/sqrt(n*p) in every cell -- 1.3% on MovieLens' 6040 users, which is three
    times the noise of the released marginal itself and is why the released arm's
    0-hop error on the ROOT tables sits at 0.015/0.029 against the per-table
    baselines' 0.004/0.003.  Quota sampling removes that floor: the counts are
    fixed by largest remainder and only the assignment to rows is random, so the
    draw is still exchangeable and any object that does not look at row order is
    unaffected in distribution.  Free post-processing of a released marginal.
    """
    p = np.asarray(p, dtype=float).ravel()
    n = int(n)
    if n <= 0 or p.size == 0:
        return np.zeros(max(n, 0), dtype=np.int64)
    cnt = _round_to_sum(np.maximum(p, 0.0), n)
    vals = np.repeat(np.arange(p.size, dtype=np.int64), cnt)
    rng.shuffle(vals)
    return vals


def repair_marginal(values, target, scores=None):
    """Move sampled leaf values onto `target` (row-level counts).

    `values` is an int array of cell ids in [0, len(target)); `scores` gives,
    per row, how strongly the released conditionals support its current value
    (lower = weaker).  Rows are donated from over-represented cells in
    increasing support order and sent to the most deficient cell, so the
    transport disturbs the joint structure as little as possible.  Deterministic
    given (values, target, scores); free post-processing of released statistics.
    """
    v = np.asarray(values).astype(np.int64).copy()
    n = v.size
    if n == 0:
        return v
    t = _round_to_sum(target, n)
    d = t.size
    h = np.bincount(v, minlength=d)[:d]
    delta = h - t
    if not np.any(delta):
        return v
    if scores is None:
        scores = np.zeros(n, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind='stable')
    donors = {c: [] for c in np.flatnonzero(delta > 0)}
    for j in order:
        c = v[j]
        if c < d and delta[c] > 0 and len(donors.get(c, ())) < delta[c]:
            donors[c].append(int(j))
    pool = [j for c in sorted(donors, key=lambda c: -delta[c]) for j in donors[c]]
    need = np.flatnonzero(delta < 0)
    if need.size == 0 or not pool:
        return v
    need = need[np.argsort(-(-delta[need]), kind='stable')]
    slots = np.repeat(need, (-delta[need]).astype(np.int64))
    m = min(len(pool), slots.size)
    v[np.asarray(pool[:m], dtype=np.int64)] = slots[:m]
    return v


def ipf_fit(start, dims, constraints, n_iter=200, tol=1e-10):
    """Iterative proportional fitting on a dense tensor.

    `start` has shape `dims`, whose LAST axis is the leaf attribute; each entry of
    `constraints` is (axis, target) with target of shape (dims[axis], dims[-1]).
    Returns the tensor whose (axis, last) marginals match the targets -- the
    maximum-entropy table consistent with them.

    Why this is needed.  The released Phase-2 sampler builds the leaf attribute
    distribution as a PRODUCT of pointwise lifts, one per released pair object.
    With group-normalized objects those lifts are so noisy that the SNR guards
    shrink most of them toward 1, which hides the fact that the product form
    double-counts: a top-attr x leaf-attr lift already contains the part of the
    association that is mediated by the mid table, and the mid-attr x leaf-attr
    lift contains it again.  On natural data the association is weak and the
    error stays second order (the 2-hop RE improves).  On a PLANTED effect it is
    first order: measured on financial plant_product at eps=0.8 the product form
    overshoots Delta to 0.62 against a true 0.18 (retention error 2.40, versus
    0.56 for the released arm).  IPF replaces the product with the unique table
    that matches both released marginals, so neither is counted twice.  Free
    post-processing; the tensor is tiny because the attribute domains are
    (financial: 8x2x3 x 3x5 x 10 = 7200 cells per leaf attribute).
    """
    P = np.array(start, dtype=float, copy=True)
    nd = len(dims)
    last = nd - 1
    for _ in range(int(n_iter)):
        worst = 0.0
        for axis, tgt in constraints:
            axes = tuple(i for i in range(nd) if i != axis and i != last)
            cur = P.sum(axis=axes) if axes else P
            cur = np.maximum(np.asarray(cur, dtype=float), 1e-300)
            scale = np.asarray(tgt, dtype=float) / cur
            shape = [1] * nd
            shape[axis] = dims[axis]
            shape[last] = dims[last]
            P = P * scale.reshape(shape)
            worst = max(worst, float(np.abs(scale - 1.0).max()))
        if worst < tol:
            break
    return P


def flat_index(cols, dims):
    """Row-major flat index of a tuple of per-axis value columns."""
    idx = np.zeros(len(cols[0]), dtype=np.int64)
    for c, d in zip(cols, dims):
        idx = idx * int(d) + np.clip(np.asarray(c), 0, int(d) - 1)
    return idx


def row_level_target(noisy_marg, n_rows):
    """Row-level count target from a group-normalized released marginal."""
    x = np.asarray(noisy_marg, dtype=float).ravel()
    s = x.sum()
    if s <= 0:
        return np.full(x.size, float(n_rows) / max(x.size, 1))
    return x * (float(n_rows) / s)


def leaf_target(sz_joint, size_vals, p_given, n_rows):
    """Size-weighted row-level target for a leaf attribute.

    `sz_joint` is a released (parent-attr x size-bin) group-count histogram,
    `size_vals[b]` the number of leaf rows a group in bin b stands for, and
    `p_given[a]` the released conditional distribution of the leaf attribute
    given parent-attr value a.  The product is the row-level marginal: it
    weights each parent-attr class by the leaf rows it actually owns, which a
    plain renormalization of the group-normalized marginal does not.

    That distinction is not cosmetic.  Group-normalized objects weight every
    row by 1/min(s, tau), so a group larger than tau contributes s/tau times
    the mass of an ordinary group; on a heavy-tailed schema one such group can
    own a large share of the leaf table while holding only 1/n1 of the mass of
    the released marginal.  Repairing onto the unweighted marginal therefore
    locks that bias in, whereas the weighted target above does not.
    """
    sz = np.asarray(sz_joint, dtype=float)
    sv = np.asarray(size_vals, dtype=float).ravel()
    p = np.asarray(p_given, dtype=float)
    if sz.size == 0 or sv.size != sz.shape[-1] or p.shape[0] != sz.shape[0]:
        return None
    rows_per_a = np.maximum(sz, 0.0) @ np.maximum(sv, 0.0)
    t = rows_per_a @ p
    s = t.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    return t * (float(n_rows) / s)


def average_targets(targets, n_rows):
    ts = [t for t in targets if t is not None]
    if not ts:
        return None
    out = np.mean([np.asarray(t, dtype=float) for t in ts], axis=0)
    s = out.sum()
    return None if s <= 0 else out * (float(n_rows) / s)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # geo_grid covers the range and is monotone
    for nmax in (1, 5, 12, 13, 99, 459, 1024, 105000):
        e, r = geo_grid(nmax, 12)
        assert e[0] == 1 and e[-1] >= nmax, (nmax, e)
        assert np.all(np.diff(e) > 0), (nmax, e)
        assert np.all(r >= e[:-1]) and np.all(r <= np.maximum(e[1:] - 1, e[:-1]))
        sizes = rng.integers(1, nmax + 1, 500)
        assert np.all(from_bins(to_bins(sizes, e), r) >= 1)
    # rescale hits the target exactly and respects bounds
    for _ in range(50):
        n = int(rng.integers(1, 200))
        s = rng.integers(0, 50, n)
        tgt = int(rng.integers(0, 5000))
        out = rescale_to_total(s, tgt, lo=1, hi=99)
        assert out.sum() == min(max(tgt, n), 99 * n), (n, tgt, out.sum())
        assert out.min() >= 1 and out.max() <= 99
        out0 = rescale_to_total(s, tgt, lo=0)
        assert out0.sum() == max(tgt, 0), (n, tgt, out0.sum())
    # repair matches the target and moves the weakest rows first
    v = rng.integers(0, 4, 1000)
    t = np.array([400., 300., 200., 100.])
    sc = rng.random(1000)
    out = repair_marginal(v, t, sc)
    assert np.allclose(np.bincount(out, minlength=4), _round_to_sum(t, 1000))
    kept = np.flatnonzero(out == v)
    moved = np.flatnonzero(out != v)
    assert sc[kept].mean() >= sc[moved].mean() - 1e-9
    # leaf_target weights parent classes by the rows they own, not by groups
    sz = np.array([[10., 0.], [0., 10.]])
    sv = np.array([1., 100.])
    p = np.array([[1., 0.], [0., 1.]])
    t = leaf_target(sz, sv, p, 1010)
    assert np.allclose(t, [10., 1000.]), t
    assert np.allclose(row_level_target(np.array([10., 10.]), 1010),
                       [505., 505.])
    assert average_targets([t, None], 1010) is not None
    assert average_targets([None, None], 5) is None
    assert leaf_target(sz, np.array([1.]), p, 10) is None
    # hybrid grid: exact bins up to tau, geometric tail above it
    for tau, nmax in ((64, 2000), (12, 13), (12, 12), (0, 100), (459, 1056320)):
        e, r = hybrid_grid(tau, nmax, 8)
        assert e[0] == 0 and e[-1] > max(nmax, tau), (tau, nmax, e[:3], e[-1])
        assert np.all(np.diff(e) > 0), (tau, nmax, e)
        assert len(r) == len(e) - 1, (tau, nmax, len(e), len(r))
        assert np.all(r >= e[:-1]) and np.all(r <= np.maximum(e[1:] - 1, e[:-1]))
        assert np.array_equal(r[:min(tau, nmax) + 1],
                              np.arange(min(tau, nmax) + 1))
        s = np.unique(np.concatenate([np.arange(0, min(tau, nmax) + 1),
                                      np.linspace(1, max(nmax, 2), 40)
                                      .astype(int)]))
        assert np.array_equal(from_bins(to_bins(s, e), r)[s <= tau], s[s <= tau])
    # grow_capped: a deficit goes only to the capped groups, a surplus is
    # removed proportionally, and an exact total is left alone
    s = np.array([1, 2, 5, 5, 5])
    g = grow_capped(s, 24, lo=1, hi=5)
    assert g.sum() == 24 and np.array_equal(g, [1, 2, 7, 7, 7]), g
    assert np.array_equal(grow_capped(s, 18, lo=1, hi=5), s)
    g2 = grow_capped(s, 12, lo=1, hi=5)
    assert g2.sum() == 12 and g2.min() >= 1 and g2.max() <= 5, g2
    assert np.array_equal(calibrate_sizes(parse('growcap'), s, 24, 1, 5), g)
    assert calibrate_sizes(parse('off'), s, 24, 1, 5).sum() == 24
    # exact_draw hits the quota exactly and still permutes
    p = np.array([0.5, 0.3, 0.2])
    v = exact_draw(np.random.default_rng(0), p, 1000)
    assert np.array_equal(np.bincount(v, minlength=3), [500, 300, 200]), v
    assert len(np.flatnonzero(np.diff(v) != 0)) > 100
    assert exact_draw(np.random.default_rng(0), p, 0).size == 0
    # ipf_fit reproduces both constrained marginals exactly
    dims = (2, 2, 2)
    t0 = np.array([[60., 40.], [40., 60.]])
    t1 = np.array([[55., 45.], [45., 55.]])
    P = ipf_fit(np.ones(dims), dims, [(0, t0), (1, t1)])
    assert np.allclose(P.sum(axis=1), t0, atol=1e-6), P.sum(axis=1)
    assert np.allclose(P.sum(axis=0), t1, atol=1e-6), P.sum(axis=0)
    assert np.all(P >= 0)
    # a product-form start that double-counts is pulled back onto the marginals
    bad = np.ones(dims)
    bad[0, 0, 1] = 9.0
    P2 = ipf_fit(bad, dims, [(0, t0), (1, t1)])
    assert np.allclose(P2.sum(axis=1), t0, atol=1e-6)
    assert np.allclose(P2.sum(axis=0), t1, atol=1e-6)
    fi = flat_index([np.array([0, 1, 1]), np.array([0, 0, 1])], (2, 2))
    assert np.array_equal(fi, [0, 2, 3]), fi
    assert parse('off') == {'totals': False, 'pubn': False, 'bins': 0,
                            'tailbins': 0, 'repair': False, 'rowleaf': False,
                            'szsens': False, 'szthresh': 0.0,
                            'exactmarg': False, 'growcap': False,
                            'ipf': False, 'ratiotop': False,
                            'budget': 0.0, 'shrink': 0.0,
                            'szquota': False}
    assert parse('totals,bins:8')['bins'] == 8
    assert parse('tailbins')['tailbins'] == TAIL_BINS
    assert parse('bins')['tailbins'] == 0 and parse('tailbins')['bins'] == 0
    assert parse('szthresh')['szthresh'] == CLIP_BIAS
    assert parse('szthresh:2')['szthresh'] == 2.0
    assert parse('szquota')['szquota']
    assert describe(parse('pubn,rowleaf,szquota')) == 'pubn+szquota+rowleaf'
    assert parse('budget')['budget'] == (3.0, 0.5)
    assert parse('budget:5')['budget'] == (5.0, 0.5)
    assert parse('budget:5:0.2')['budget'] == (5.0, 0.2)
    assert parse('shrink')['shrink'] == 1.0
    assert parse('shrink:3')['shrink'] == 3.0
    assert describe(parse('pubn,shrink')) == 'pubn+shrink'
    assert describe(parse('pubn,budget')) == 'pubn+budget'
    assert describe(parse('pubn,budget:5:0.2,shrink:3')) == \
        'pubn+budget5_0.2+shrink3'
    # budget_sigma: even split reproduces the plain per-query sigma; the
    # account invariant is sum_i 1/sigma_i^2 = g2/2 in both cases
    g2_tot = 4.0
    names = ['P1_a', 'P1_b', 'XP_a_b', 'C_c']
    smap, g2i = budget_sigma(g2_tot, names, is_1way_name, 1.0, 1.0)
    assert abs(smap['P1_a'] - np.sqrt(2.0 / (g2_tot / len(names)))) < 1e-12
    assert abs(sum(1.0 / s ** 2 for s in smap.values()) - g2_tot / 2) < 1e-9
    smap2, _ = budget_sigma(g2_tot, names, is_1way_name, 3.0, 0.5)
    assert abs(sum(1.0 / s ** 2 for s in smap2.values()) - g2_tot / 2) < 1e-9
    assert smap2['P1_a'] < smap2['XP_a_b']          # 1-way gets more budget
    # shrink_joint: a cell below the gate is a convex mix of itself and
    # the independence cell mass (w = noisy/thr), a cell above it untouched
    nj = np.array([[100.0, 1.0], [1.0, 100.0]])
    m1 = np.array([50.0, 150.0]); m2 = np.array([100.0, 100.0])
    shr = shrink_joint(nj, [m1, m2], 1.0, kappa=2.0, total=202.0)
    indep_01 = 202.0 * (50 / 200) * (100 / 200)     # marginal shares product
    assert abs(shr[0, 1] - (0.5 * 1.0 + 0.5 * indep_01)) < 1e-9, shr[0, 1]
    assert abs(shr[0, 0] - 100.0) < 1e-9            # above gate: untouched
    assert np.array_equal(shrink_joint(nj, [m1, m2], 1.0, kappa=0.0), nj)
    # quota_calibrate: per-value sums track the quotas
    szs = np.array([1, 5, 9, 3, 4, 2, 8])          # 7 groups
    gids = np.array([0, 0, 0, 1, 1, 1, 1])         # values 0..1
    quota = np.array([30.0, 60.0])
    out = quota_calibrate(szs, gids, quota, lo=1, hi=20)
    assert out.shape == szs.shape
    assert int(out[gids == 0].sum()) == 30 and int(out[gids == 1].sum()) == 60
    assert out.min() >= 1
    # hi may be exceeded for the missing tail (grow_capped delegates to
    # rescale with hi=None when no group sits at the cap)
    # all groups on one value: quota met; the missing tail may exceed hi
    # (grow_capped delegates to rescale with hi=None when no group sits at
    # the cap -- that is exactly the truncated tail being restored)
    szs2 = np.array([2, 2, 2])
    gids2 = np.array([0, 0, 0])
    out2 = quota_calibrate(szs2, gids2, np.array([100.0]), lo=1, hi=20)
    assert out2.sum() == 100
    assert out2.min() >= 1
    # quota_calibrate_2d: both marginals hit simultaneously (the sequential
    # per-attribute calibration cannot do this: each group belongs to both)
    sz3 = np.array([2, 5, 3, 4, 2, 1])
    gA = np.array([0, 0, 0, 1, 1, 1])          # attr1: values 0,1
    gB = np.array([0, 0, 1, 0, 1, 1])          # attr2: values 0,1
    qA = np.array([30.0, 60.0]); qB = np.array([40.0, 50.0])
    o3 = quota_calibrate_2d(sz3, gA, gB, qA, qB, lo=1)
    assert o3.shape == sz3.shape and o3.min() >= 1
    assert abs(o3[gA == 0].sum() - 30) <= 1 and abs(o3[gA == 1].sum() - 60) \
        <= 1, o3
    assert abs(o3[gB == 0].sum() - 40) <= 1 and abs(o3[gB == 1].sum() - 50) \
        <= 1, o3
    assert abs(o3.sum() - 90) <= 2
    # quota_calibrate_kd: three attributes, all three marginals hit
    sz4 = np.array([2, 5, 3, 4, 2, 1, 6, 1])
    g4a = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    g4b = np.array([0, 0, 1, 0, 1, 1, 1, 0])
    g4c = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    q4a = np.array([24.0, 48.0]); q4b = np.array([30.0, 42.0])
    q4c = np.array([36.0, 36.0])
    o4 = quota_calibrate_kd(sz4, [g4a, g4b, g4c], [q4a, q4b, q4c], lo=1)
    assert o4.shape == sz4.shape and o4.min() >= 1
    # integer rounding across the per-cell distributions accumulates ~2 rows
    # per marginal (each cell rounds independently); the real-data target is
    # quota within ~1% so the assertion is on that budget
    for g, q in ((g4a, q4a), (g4b, q4b), (g4c, q4c)):
        for v in range(2):
            assert abs(o4[g == v].sum() - q[v]) <= 2, (g, v, o4)
    d = debias_hist(np.array([0.0, 10.0]), 1.0)
    assert abs(d[0]) < 1e-12 and abs(d[1] - (10.0 - CLIP_BIAS)) < 1e-12
    assert np.array_equal(debias_hist(np.array([0.0, 10.0]), 1.0, on=False),
                          np.array([0.0, 10.0]))
    assert parse('all')['repair'] and describe(parse('all')) == \
        ('totals+pubn+tailbins8+szthresh+exactmarg+growcap+ipf+ratiotop'
         '+repair+szsens')
    assert not parse('all')['rowleaf']          # must be asked for explicitly
    assert describe(parse('off')) == 'off'
    assert describe(parse('pubn,rowleaf')) == 'pubn+rowleaf'
    assert adjacency(parse('off')).startswith('one-individual +')
    assert adjacency(parse('rowleaf')) == 'one-individual'
    w = np.array([0.5, 0.25])
    assert np.array_equal(row_weight(parse('rowleaf'), w), np.ones(2))
    assert np.array_equal(row_weight(parse('off'), w), w)
    print('chain_npm_absorb self-test OK')
