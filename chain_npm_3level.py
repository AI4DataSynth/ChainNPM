"""
ChainNPM3: three-level FK chain synthesizer (top <- mid <- bot).

Deep-chain generalization of chain_npm_generic.py (ChainNPM2), e.g.
TPC-H customer <- orders <- lineitem with ALL THREE tables private.

Mechanism (same B+NPM design):
  Phase 1: ONE budget pass — noisy low-order joint marginals:
    - top 1-way + selected top x top pairs (intra-top correlation)
    - mid 1-way + top x mid pairs (group-size-normalized by children/top)
    - bot 1-way + mid x bot pairs (group-size-normalized by children/mid)
    - top x bot CROSS-HOP pairs (optional, see below), measured through the
      top -> mid -> bot path so that cross-grandparent/leaf dependence enters
      the measured set explicitly (the chain analogue of ChainNPMV's XP pair)
    - group size (mid per top) x selected top attrs
    - group size (bot per mid) x selected mid attrs
    - mean group sizes at both levels
  Phase 2: free post-processing — normalize, build lift tables, sample
    top -> (size1, mid attrs) -> (size2, bot attrs).

Multi-hop correlations across all three private tables are preserved
because the mid rows are sampled ONCE consistently and condition both
the top->mid and mid->bot lifts (composition survives by construction).
The optional cross-hop marginals add the *direct* top x bot dependence,
which the composed adjacent-hop lifts alone recover only weakly.

Normalization rule for the cross-hop marginal (``tb_weight``):
  'inner': w = 1 / min(size2, tau2) -- each MID group (the traversed FK
           group, i.e. bot's direct parent) contributes <= 1.  This is
           exactly the weight already used for the B / MB queries at this
           level, so the cross-hop marginal is on the same scale as the
           marginals it is composed with, and the lift it yields is a
           conditional that can be multiplied into them.
  'both' : w = 1 / (min(size1, tau1) * min(size2, tau2)) -- additionally
           normalizes by the TOP group (the ChainNPMV XP weight carried
           over literally), so each TOP group contributes ~1.  Measured to
           be worse on deep chains (over-normalization starves the
           conditional); kept for ablation.

Budget: Gaussian mechanism, equal gamma^2-split across all queries.  Adding
Q_tb cross-hop queries (together with the constant 2 mean-size queries, which
become 3 TOTAL queries when the `totals` token of the absorb arm is on)
therefore raises the per-query noise by a factor sqrt(Q' / Q) for *every*
query, since sigma scales as sqrt(n_queries) under an equal gamma^2 split.
"""

import numpy as np
import logging
import argparse

import chain_npm_absorb as ab

logger = logging.getLogger('ChainNPM3')

# ---------------------------------------------------------------- #
# Cross-hop (top x bot) option surface, shared by every ChainNPM3    #
# driver (p4_common.run_chain_pipeline and chain_npm_tpch_deep.py)   #
# so the flag names and the default cannot drift apart.              #
# ---------------------------------------------------------------- #

CROSSHOP_MODES = ('off', 'inner', 'both')
CROSSHOP_DEFAULT = 'inner'          # blessed default (see the docstring)
SHRINK_ALIAS = {'': (False, False), 'count': (True, False),
                'denom': (False, True), 'both': (True, True)}


def add_crosshop_args(ap):
    """Register the cross-hop CLI on ``ap`` and return it.

    ``--crosshop-marginal`` takes an optional value so that the historical
    store_true spelling (``--crosshop-marginal --crosshop-weight inner``,
    used by the validated prototype grid runner) still parses: a bare flag
    means the blessed ``inner``.
    """
    ap.add_argument('--crosshop-marginal', nargs='?', const=CROSSHOP_DEFAULT,
                    default=CROSSHOP_DEFAULT, choices=list(CROSSHOP_MODES),
                    help='add top x bot cross-hop marginals measured through '
                         'the top -> mid -> bot path: off = pre-cross-hop '
                         'behaviour; inner = w = 1/min(size2, tau2), the '
                         'weight already used for B/MB (DEFAULT); both = '
                         'additionally normalized by the top group '
                         '(ablation arm)')
    ap.add_argument('--crosshop-denom-guard', type=str, default='on',
                    choices=['on', 'off'],
                    help='SNR-shrink the cross-hop lift by the noised top '
                         '1-way marginal it divides by (on by default: on a '
                         'small top table that denominator is pure noise at '
                         'low epsilon and manufactures a spurious cross-hop '
                         'contrast)')
    ap.add_argument('--crosshop-count-guard', type=str, default='off',
                    choices=['on', 'off'],
                    help='SNR-shrink the cross-hop lift by the noised TB '
                         'counts (numerator); off by default, ablation only')
    # legacy aliases of the validated prototype launcher; they override the
    # switches above when given (hidden from --help on purpose)
    ap.add_argument('--crosshop-weight', type=str, default=None,
                    choices=['inner', 'both'],
                    help=argparse.SUPPRESS)
    ap.add_argument('--crosshop-shrink', type=str, default=None,
                    choices=['', 'count', 'denom', 'both'],
                    help=argparse.SUPPRESS)
    return ap


def crosshop_kwargs(args, top_attrs, bot_attrs):
    """Resolve the parsed CLI into ``ChainNPM3`` constructor kwargs.

    Returns ``(tb_pairs, tb_weight, tb_shrink, tb_shrink_denom)``.  With
    ``mode == 'off'`` the returned ``tb_pairs`` is empty, so the executed
    code path is exactly the pre-cross-hop one (the extension is inert).
    """
    mode = getattr(args, 'crosshop_marginal', CROSSHOP_DEFAULT) \
        or CROSSHOP_DEFAULT
    weight_alias = getattr(args, 'crosshop_weight', None)
    shrink_alias = getattr(args, 'crosshop_shrink', None)
    if weight_alias:
        mode = weight_alias
    if shrink_alias is not None:
        tb_shrink, tb_shrink_denom = SHRINK_ALIAS[shrink_alias]
    else:
        tb_shrink = getattr(args, 'crosshop_count_guard', 'off') == 'on'
        tb_shrink_denom = getattr(args, 'crosshop_denom_guard', 'on') == 'on'
    if mode == 'off':
        return (), None, False, False
    tb_pairs = [(a, b) for a in top_attrs for b in bot_attrs]
    return tb_pairs, mode, bool(tb_shrink), bool(tb_shrink_denom)


def analytic_gaussian_gamma(epsilon, delta):
    from scipy.stats import norm
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if mid <= 0:
            lo = mid
            continue
        val = (norm.cdf(mid / 2 - epsilon / mid)
               - np.exp(epsilon) * norm.cdf(-mid / 2 - epsilon / mid))
        if val <= delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return lo


class ChainNPM3:
    """Three-level chain synthesizer (top <- mid <- bot).

    Cross-hop extension (`tb_pairs`, `tb_weight`, `tb_shrink`,
    `tb_shrink_denom`): `tb_pairs` adds noisy top-attribute x bot-attribute
    marginals measured through the top -> mid -> bot path, so that
    cross-grandparent/leaf dependence enters the measured set (the chain
    analogue of ChainNPMV's XP cross-parent pair).  Weights are documented
    in the module docstring.  The extension is inert when `tb_pairs` is
    empty, and is enabled by default by the pipeline driver
    (p4_common.run_chain_pipeline) with `tb_weight='inner'` plus the
    denominator SNR guard; pass `--crosshop-marginal off` there to recover
    the pre-cross-hop behaviour.
    """

    def __init__(self, top_attrs, top_domains, mid_attrs, mid_domains,
                 bot_attrs, bot_domains, tt_pairs=(), tm_pairs=(),
                 mb_pairs=(), sz1_attrs=None, sz2_attrs=None,
                 sz_snr=5.0, tau1=40, tau2=8, seed=0,
                 tb_pairs=(), tb_weight='inner', tb_shrink=False,
                 tb_shrink_denom=False, absorb=None):
        self.top_attrs = top_attrs
        self.tdom = top_domains
        self.mid_attrs = mid_attrs
        self.mdom = mid_domains
        self.bot_attrs = bot_attrs
        self.bdom = bot_domains
        self.tt_pairs = tuple(tt_pairs)
        self.tm_pairs = tuple(tm_pairs)
        self.mb_pairs = tuple(mb_pairs)
        self.sz1_attrs = list(sz1_attrs) if sz1_attrs else list(top_attrs)
        self.sz2_attrs = list(sz2_attrs) if sz2_attrs else list(mid_attrs)
        self.sz_snr = sz_snr
        self.tau1 = tau1
        self.tau2 = tau2
        self.seed = seed
        # Absorb arm (chain_npm_absorb).  absorb=None leaves Phase 1/Phase 2 on
        # exactly the released code path.  Unlike the star rail, this chain's
        # adjacency is one individual tuple of the leaf relation, so a table
        # cardinality has l2 sensitivity 1 and `totals` can buy it: the arm
        # spends the two slots the released code already pays for MEAN_SIZE1/2
        # (drawn, budgeted, never read in Phase 2) on TOTAL_M/TOTAL_B instead.
        self.ab = ab.parse('off') if absorb is None else ab.parse(absorb)
        self.absorb = ab.describe(self.ab)
        self.adjacency = ab.adjacency(self.ab)
        self.pub = None
        self.tot = {}
        self.total_source = {}
        self.sz_sens = 1.0
        self.sz1_grid = None
        self.sz2_grid = None
        self.anchor_source = 'pair' if self.tt_pairs else 'none'
        self.tb_pairs = tuple(tb_pairs)          # cross-hop top x bot pairs
        self.tb_weight = tb_weight if self.tb_pairs else None
        # optional SNR shrinkage of the cross-hop lift (same rule already
        # used for the group-size lifts: clip the deviation from 1 by
        # count/(sz_snr * sigma)).  tb_shrink guards the NUMERATOR (TB
        # counts), tb_shrink_denom guards the DENOMINATOR (the top 1-way
        # marginal the lift divides by, which on a small top table is the
        # ill-conditioned part at low epsilon: the top table of the deep
        # financial chain has 77 rows, ~26 rows per value, while sigma at
        # eps=0.1 is ~346, so the denominator is pure noise and a spurious
        # cross-hop signal appears unless it is guarded).
        self.tb_shrink = bool(tb_shrink) and bool(self.tb_pairs)
        self.tb_shrink_denom = bool(tb_shrink_denom) and bool(self.tb_pairs)
        if self.tb_weight not in (None, 'inner', 'both'):
            raise ValueError('tb_weight must be inner|both, got %r'
                             % (tb_weight,))

    # ---------------- Phase 1 ----------------

    def compute_marginals(self, top, mid, bot, epsilon, delta):
        """top: (n_t, 1+len(top_attrs)) with PK col 0 (dense 0..n_t-1).
        mid:  (n_m, 2+len(mid_attrs)) with PK col 0, FK->top col 1.
        bot:  (n_b, 2+len(bot_attrs)) with PK col 0, FK->mid col 1.
        """
        rng = np.random.default_rng(self.seed)
        n_t, n_m, n_b = top.shape[0], mid.shape[0], bot.shape[0]

        t_idx = {a: i + 1 for i, a in enumerate(self.top_attrs)}
        m_idx = {a: i + 2 for i, a in enumerate(self.mid_attrs)}
        b_idx = {a: i + 2 for i, a in enumerate(self.bot_attrs)}

        queries = {}
        for a in self.top_attrs:
            queries[f'T_{a}'] = ('t1', a)
        for (a, b) in self.tt_pairs:
            queries[f'TT_{a}_{b}'] = ('t2', (a, b))
        for a in self.mid_attrs:
            queries[f'M_{a}'] = ('m1', a)
        for (a, b) in self.tm_pairs:
            queries[f'TM_{a}_{b}'] = ('tm', (a, b))
        for a in self.bot_attrs:
            queries[f'B_{a}'] = ('b1', a)
        for (a, b) in self.mb_pairs:
            queries[f'MB_{a}_{b}'] = ('mb', (a, b))
        # cross-hop top x bot marginals (optional)
        for (a, c) in self.tb_pairs:
            queries[f'TB_{a}_{c}'] = ('tb', (a, c))
        for a in self.sz1_attrs:
            queries[f'SZ1_{a}'] = ('sz1', a)
        for a in self.sz2_attrs:
            queries[f'SZ2_{a}'] = ('sz2', a)
        qnames = list(queries)
        # Absorb arm: the released code pays for two MEAN_SIZE queries that
        # Phase 2 never reads (see the NOTE there).  `totals` spends the same
        # slots -- plus one -- on TOTAL_T/M/B instead.  All are l2 sensitivity 1
        # under this chain's one-individual adjacency, but a total's relative
        # error is sigma/n rather than sigma/mean.
        n_queries = len(qnames) + (3 if self.ab['totals'] else 2)
        gamma_max = analytic_gaussian_gamma(epsilon, delta)
        g2 = gamma_max ** 2 / n_queries

        sigma = np.sqrt(2.0 / g2)  # sensitivity 1 for every query
        # `szsens` charges sqrt(2) on the size histograms: removing one leaf row
        # moves its group from bin s to bin s-1, which is an l2 change of sqrt(2)
        # and not 1.  Independent noise stream, so the marginals below stay on
        # the same realisation as the default arm.
        self.sz_sens = ab._SZ_SENS_CHAIN if (self.ab['szsens']
                                             or self.ab['rowleaf']) else 1.0
        self.tot = {}
        if self.ab['totals']:
            rng_t = np.random.default_rng(self.seed + 777001)
            self.tot = {'TOTAL_T': float(n_t) + rng_t.normal(0, sigma),
                        'TOTAL_M': float(n_m) + rng_t.normal(0, sigma),
                        'TOTAL_B': float(n_b) + rng_t.normal(0, sigma)}
        if self.ab['pubn']:
            self.pub = {'n_t': int(n_t), 'n_m': int(n_m), 'n_b': int(n_b)}
        # size-bin grid: bounded by the cardinality this arm already holds, so
        # the grid leaks nothing beyond it
        self.sz1_grid = self.sz2_grid = None
        if self.ab['bins'] or self.ab['tailbins']:
            if not self.pub and not self.tot:
                raise ValueError('absorb=bins/tailbins needs totals or pubn on '
                                 'the chain rail (the grid bound must come from '
                                 'released or public cardinalities)')
            ub1 = self.pub['n_m'] if self.pub else \
                max(1, int(round(self.tot.get('TOTAL_M', 0))))
            ub2 = self.pub['n_b'] if self.pub else \
                max(1, int(round(self.tot.get('TOTAL_B', 0))))
            # the grids keep the size-0 bin: top rows without children live there
            self.sz1_grid = ab.size_grid(self.ab, self.tau1, ub1)
            self.sz2_grid = ab.size_grid(self.ab, self.tau2, ub2)

        acc = {}
        for name in qnames:
            kind, spec = queries[name]
            if kind == 't1':
                shape = (self.tdom[spec],)
            elif kind == 't2':
                shape = (self.tdom[spec[0]], self.tdom[spec[1]])
            elif kind == 'm1':
                shape = (self.mdom[spec],)
            elif kind == 'tm':
                shape = (self.tdom[spec[0]], self.mdom[spec[1]])
            elif kind == 'b1':
                shape = (self.bdom[spec],)
            elif kind == 'mb':
                shape = (self.mdom[spec[0]], self.bdom[spec[1]])
            elif kind == 'tb':
                shape = (self.tdom[spec[0]], self.bdom[spec[1]])
            elif kind == 'sz1':
                shape = (self.tdom[spec], self.tau1 + 1 if self.sz1_grid is
                         None else len(self.sz1_grid[1]))
            elif kind == 'sz2':
                shape = (self.mdom[spec], self.tau2 + 1 if self.sz2_grid is
                         None else len(self.sz2_grid[1]))
            acc[name] = np.zeros(shape, dtype=np.float64)

        # ---- top-level marginals
        ta = {a: np.clip(top[:, t_idx[a]].astype(int), 0, self.tdom[a] - 1)
              for a in self.top_attrs}
        for a in self.top_attrs:
            np.add.at(acc[f'T_{a}'], ta[a], 1.0)
        for (a, b) in self.tt_pairs:
            np.add.at(acc[f'TT_{a}_{b}'], (ta[a], ta[b]), 1.0)

        # ---- group mid rows by FK -> top
        fk1 = mid[:, 1].astype(int)
        order1 = np.argsort(fk1, kind='stable')
        fk1_s = fk1[order1]
        bnd1 = np.flatnonzero(np.diff(fk1_s)) + 1
        starts1 = np.concatenate([[0], bnd1])
        ends1 = np.concatenate([bnd1, [len(fk1_s)]])
        sizes1_full = ends1 - starts1
        sizes1 = np.minimum(sizes1_full, self.tau1)
        top_pk1 = fk1_s[starts1]
        valid1 = top_pk1 < n_t
        top_pk1 = np.where(valid1, top_pk1, 0)

        # per-mid-row weights 1/min(group_size, tau1), ORIGINAL mid order
        sizes1_by_row = np.repeat(sizes1, sizes1_full)  # sorted order
        inv_order1 = np.empty_like(order1)
        inv_order1[order1] = np.arange(len(order1))
        # `rowleaf` replaces the 1/min(s,tau) group weight by 1, which raises the
        # mass of every mid/leaf object from the group count to the row count at
        # the same l2 sensitivity under the one-individual adjacency
        w1 = ab.row_weight(self.ab,
                           (1.0 / np.maximum(sizes1_by_row, 1))[inv_order1])

        ma = {a: np.clip(mid[:, m_idx[a]].astype(int), 0, self.mdom[a] - 1)
              for a in self.mid_attrs}
        for a in self.mid_attrs:
            np.add.at(acc[f'M_{a}'], ma[a], w1)
        for (a, b) in self.tm_pairs:
            tv = np.clip(ta[a][np.clip(fk1, 0, n_t - 1)],
                         0, self.tdom[a] - 1)
            np.add.at(acc[f'TM_{a}_{b}'], (tv, ma[b]), w1)
        # `bins` histograms the UNtruncated group size: the tau cap is what
        # discards the leaf rows of heavy-tailed schemas.  Weights (w1/w2) keep
        # 1/min(s, tau), so Lemma lem:sens is untouched.
        sz1_idx = sizes1 if self.sz1_grid is None else \
            ab.to_bins(sizes1_full, self.sz1_grid[0])
        for a in self.sz1_attrs:
            np.add.at(acc[f'SZ1_{a}'], (ta[a][top_pk1], sz1_idx),
                      valid1.astype(float))
        # top rows WITHOUT children go into the size-0 bin (so synthesized
        # top counts match the true table, e.g. TPC-H customers w/o orders)
        has_group = np.zeros(n_t, dtype=bool)
        has_group[top_pk1[valid1]] = True
        zero_idx = np.flatnonzero(~has_group)
        if len(zero_idx):
            for a in self.sz1_attrs:
                np.add.at(acc[f'SZ1_{a}'],
                          (ta[a][zero_idx],
                           np.zeros(len(zero_idx), dtype=int)), 1.0)

        # ---- group bot rows by FK -> mid
        fk2 = bot[:, 1].astype(int)
        order2 = np.argsort(fk2, kind='stable')
        fk2_s = fk2[order2]
        bnd2 = np.flatnonzero(np.diff(fk2_s)) + 1
        starts2 = np.concatenate([[0], bnd2])
        ends2 = np.concatenate([bnd2, [len(fk2_s)]])
        sizes2_full = ends2 - starts2
        sizes2 = np.minimum(sizes2_full, self.tau2)
        mid_pk2 = fk2_s[starts2]
        valid2 = mid_pk2 < n_m
        mid_pk2 = np.where(valid2, mid_pk2, 0)

        sizes2_by_row = np.repeat(sizes2, sizes2_full)
        inv_order2 = np.empty_like(order2)
        inv_order2[order2] = np.arange(len(order2))
        w2 = ab.row_weight(self.ab,
                           (1.0 / np.maximum(sizes2_by_row, 1))[inv_order2])

        ba = {a: np.clip(bot[:, b_idx[a]].astype(int), 0, self.bdom[a] - 1)
              for a in self.bot_attrs}
        for a in self.bot_attrs:
            np.add.at(acc[f'B_{a}'], ba[a], w2)
        for (a, b) in self.mb_pairs:
            mv = np.clip(ma[a][np.clip(fk2, 0, n_m - 1)],
                         0, self.mdom[a] - 1)
            np.add.at(acc[f'MB_{a}_{b}'], (mv, ba[b]), w2)
        sz2_idx = sizes2 if self.sz2_grid is None else \
            ab.to_bins(sizes2_full, self.sz2_grid[0])
        for a in self.sz2_attrs:
            np.add.at(acc[f'SZ2_{a}'], (ma[a][mid_pk2], sz2_idx),
                      valid2.astype(float))

        # ---- cross-hop top x bot marginals (optional), measured through the
        # top -> mid -> bot path: each bot row carries the attribute of its
        # ancestor top row, with the same per-FK-group weights used above.
        if self.tb_pairs:
            mid_of_bot = np.clip(fk2, 0, n_m - 1)          # bot -> mid
            top_of_bot = np.clip(mid[:, 1].astype(int)[mid_of_bot], 0,
                                 n_t - 1)                  # bot -> top
            if self.tb_weight == 'both':
                # + top-group normalization (ChainNPMV XP weight carried over)
                w_tb = w2 * w1[mid_of_bot]
            else:                                          # 'inner'
                w_tb = w2
            for (a, c) in self.tb_pairs:
                tv = np.clip(ta[a][top_of_bot], 0, self.tdom[a] - 1)
                np.add.at(acc[f'TB_{a}_{c}'], (tv, ba[c]), w_tb)

        mean_size1 = float(sizes1_full.mean()) if len(sizes1_full) else 1.0
        mean_size2 = float(sizes2_full.mean()) if len(sizes2_full) else 1.0

        noisy = {}
        for name in qnames:
            sens = self.sz_sens if queries[name][0] in ('sz1', 'sz2') else 1.0
            noisy[name] = np.maximum(0.0, acc[name]
                                     + rng.normal(0, sigma * sens,
                                                  acc[name].shape))
        if self.ab['totals']:
            # released cardinalities take the slots MEAN_SIZE1/2 occupied; they
            # are NOT clipped at 0 here -- calibrate_total() decides in Phase 2
            # whether a draw is plausible enough to use.
            noisy.update(self.tot)
        else:
            noisy['MEAN_SIZE1'] = max(0.5, mean_size1 + rng.normal(0, sigma))
            noisy['MEAN_SIZE2'] = max(0.5, mean_size2 + rng.normal(0, sigma))
        self.noisy = noisy
        self.sigma = float(sigma)
        self.n_top = n_t
        self.Q = len(qnames)          # noisy marginals; + 2 mean-size queries
        #                               # (-> 3 TOTAL queries under `totals`)
        self.n_queries = n_queries
        # effective mass of the cross-hop marginals (pre-noise, diagnostic):
        # tells how much per-group weight actually accumulates.
        self.tb_mass = {name: float(acc[name].sum())
                        for (name, (kind, _)) in queries.items()
                        if kind == 'tb'}
        return noisy

    # ---------------- Phase 2 (post-processing) ----------------
    # NOTE: calibrating group-size totals to the released MEAN_SIZE*
    # statistics was tried and rejected — their noise (sigma ~ tens at
    # these budgets) dwarfs the means and rescaling amplifies it.
    # NOTE (absorb arm): the objection is about the statistic, not about
    # calibration.  A row count has the SAME l2 sensitivity 1 as a mean group
    # size, but its relative error is sigma/n instead of sigma/mean, i.e. better
    # by a factor of n.  On financial at eps=0.1 (sigma ~ 345) the mean of the
    # mid-per-top size is ~58, so MEAN_SIZE carries ~600% relative error, while
    # TOTAL_M on 4500 rows carries ~7.7% and TOTAL_B on 1.06M rows ~0.03%.
    # `totals`/`pubn` therefore calibrate to a cardinality and leave MEAN_SIZE
    # alone; `calibrate_total` falls back to the old estimator when the released
    # draw is implausible (tiny table at tiny budget).

    def _bot_vals_ipf(self, noisy, dist, sample_rows, n_b, bot_fk, mid_fk,
                      top_vals, mid_vals, chunk):
        """Leaf attribute values from an IPF fit instead of a lift product.

        The tensor axes are (every top attribute, every mid attribute, the leaf
        attribute); the constraints are the released TB and MB pair objects, both
        renormalized to mass n_b so their totals agree.  The start point is the
        synthesized (top tuple, mid tuple) row count times the leaf 1-way, i.e.
        conditional independence, and IPF moves it to the maximum-entropy table
        consistent with both released marginals.  See chain_npm_absorb.ipf_fit
        for the measured overshoot this removes.  Returns only the attributes it
        could fit; the caller falls back to the lift product for the rest.
        """
        base_dims = tuple(int(self.tdom[a]) for a in self.top_attrs) + \
            tuple(int(self.mdom[b]) for b in self.mid_attrs)
        anc_top = mid_fk[bot_fk]
        cols = ([np.asarray(top_vals[a])[anc_top] for a in self.top_attrs]
                + [np.asarray(mid_vals[b])[bot_fk] for b in self.mid_attrs])
        qidx = ab.flat_index(cols, base_dims)
        Q = np.bincount(qidx, minlength=int(np.prod(base_dims))
                        ).astype(float).reshape(base_dims)
        out = {}
        for c in self.bot_attrs:
            dom_c = int(self.bdom[c])
            pc = dist(noisy[f'B_{c}'])
            cons = []
            for (a, cc) in self.tb_pairs:
                if cc != c:
                    continue
                t = np.maximum(np.asarray(noisy[f'TB_{a}_{c}'], float), 0.0)
                if t.sum() > 0:
                    cons.append((self.top_attrs.index(a), t * (n_b / t.sum())))
            for (b, cc) in self.mb_pairs:
                if cc != c:
                    continue
                t = np.maximum(np.asarray(noisy[f'MB_{b}_{c}'], float), 0.0)
                if t.sum() > 0:
                    cons.append((len(self.top_attrs) + self.mid_attrs.index(b),
                                 t * (n_b / t.sum())))
            if not cons:
                continue
            P = ab.ipf_fit(Q[..., None] * pc, base_dims + (dom_c,), cons)
            Pc = P.reshape(-1, dom_c)
            vals = np.empty(n_b, dtype=int)
            sup = (np.empty(n_b, dtype=float)
                   if (self.ab['repair'] and self.ab['rowleaf']) else None)
            for s0 in range(0, n_b, chunk):
                e0 = min(s0 + chunk, n_b)
                probs = Pc[qidx[s0:e0]]
                probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-300)
                vals[s0:e0] = sample_rows(probs, e0 - s0)
                if sup is not None:
                    sup[s0:e0] = probs[np.arange(e0 - s0), vals[s0:e0]]
            if sup is not None:
                vals = ab.repair_marginal(
                    vals, ab.row_level_target(noisy[f'B_{c}'], n_b), sup)
            out[c] = vals
        return out

    def synthesize(self, n_top=None, chunk=200000):
        rng = np.random.default_rng(self.seed + 12345)
        noisy = self.noisy

        def dist(arr):
            arr = np.asarray(arr, dtype=float)
            s = arr.sum()
            if s <= 0:
                return np.full(arr.shape, 1.0 / arr.size)
            return arr / s

        def sample_from(probs):
            flat = probs.flatten()
            return int(rng.choice(len(flat), p=flat / flat.sum()))

        def lift_table(joint, marg_x, marg_y, smooth=1e-6):
            denom = np.outer(marg_x, marg_y) + smooth
            return (joint + smooth / joint.size) / denom

        def sample_rows(probs, n):
            """probs (n, k) normalized; return draws via Gumbel-free CDF."""
            u = rng.random(n)
            cum = np.cumsum(probs, axis=1)
            return (cum[:, :-1] < u[:, None]).sum(axis=1)

        # ---------- top sampling ----------
        n_t = n_top if n_top is not None else int(round(
            noisy['T_' + self.top_attrs[0]].sum()))
        # where each cardinality came from, recorded in hyper by the driver:
        # 'marginal' = round(sum of a released 1-way), 'total' = released
        # TOTAL_*, 'public' = public schema cardinality.
        self.total_source = {'top': 'marginal', 'mid': 'sum(sizes)',
                             'bot': 'sum(sizes)'}
        if n_top is None:
            if self.pub:
                n_t = self.pub['n_t']
                self.total_source['top'] = 'public'
            elif self.tot:
                n_t, src = ab.calibrate_total(noisy.get('TOTAL_T'), n_t)
                self.total_source['top'] = src
        # ---- ratiotop: recover the top 1-way from its row-level descendants ---
        # The top table is tiny (financial: 77 districts) so its own 1-way
        # marginal is hopeless at these budgets -- 8 REGION cells holding ~9.6
        # rows each against sigma=13.85 at eps=3.2 and 49.4 at eps=0.8, i.e.
        # 144-500% relative error, which is why the released arm synthesizes a
        # top-attribute distribution whose anchor share is 0.102 against a true
        # 0.470.  But the ROW-LEVEL adjacent object TM_{a,b} counts mid rows per
        # top-attribute value (mass n_m, so ~1% error), and rescaling it to n_t
        # is an estimate of the same 1-way.  It is biased by variation in the
        # number of descendants per top row, which is far smaller than the noise
        # it replaces.  Free post-processing of released values.  Used ONLY for
        # sampling the top rows; the released T_a still feeds the lift guards.
        self.top_marg = {a: np.asarray(noisy[f'T_{a}'], float)
                         for a in self.top_attrs}
        self.ratio_source = {a: 'direct' for a in self.top_attrs}
        if self.ab['ratiotop'] and n_t:
            for a in self.top_attrs:
                est = None
                for (aa, bb) in self.tm_pairs:
                    if aa != a:
                        continue
                    rs = np.asarray(noisy[f'TM_{a}_{bb}'], float).sum(axis=1)
                    if rs.sum() > 0:
                        est, self.ratio_source[a] = rs * (n_t / rs.sum()), \
                            'TM:%s' % bb
                    break
                if est is None:
                    for (aa, cc) in self.tb_pairs:
                        if aa != a:
                            continue
                        rs = np.asarray(noisy[f'TB_{a}_{cc}'], float).sum(axis=1)
                        if rs.sum() > 0:
                            est, self.ratio_source[a] = rs * (n_t / rs.sum()), \
                                'TB:%s' % cc
                        break
                if est is not None:
                    self.top_marg[a] = est

        sampled = {}
        # `exactmarg` quota rule.  A quota draw on the anchor JOINT makes the pair
        # and both of its marginals exact at once, which is what a large parent
        # table wants.  On a small one the joint is noise-dominated -- financial's
        # district table has 77 rows over 8x2 cells, i.e. 4.8 per cell against
        # sigma=13.85 -- and forcing an exact match to it makes the 1-ways worse
        # than drawing them independently (measured: 0hop_top 0.019 -> 0.115).
        # Dropping the anchor instead (quota on each 1-way) does buy an exact
        # 1-way marginal, but it makes the parent attributes mutually
        # independent, and the child sampler multiplies one lift per parent
        # attribute: an implausible attribute combination gets a product of lifts
        # that no real parent ever had.  Measured on financial at eps=3.2, that
        # trade is net negative (0hop_top 0.115 -> 0.039 but 1hop_topmid
        # 0.307 -> 0.593 and 2hop_topbot 0.283 -> 0.640).  So when the joint is
        # not estimable the anchor is kept and NO quota is applied to that table;
        # the quota still applies to attributes drawn from their own 1-way.  The
        # threshold reuses the code's pre-registered SNR constant (sz_snr).
        anchor = bool(self.tt_pairs)
        quota_anchor = bool(self.ab['exactmarg'])
        self.anchor_source = 'pair' if anchor else 'none'
        if self.ab['ratiotop'] and anchor:
            # the pair joint cannot be rescaled from descendants the way a 1-way
            # can, and it is noise-dominated on exactly the small tables where
            # ratiotop fires, so the top attributes are drawn from their ratio
            # estimates instead
            anchor = False
            self.anchor_source = 'ratio-oneway'
        if anchor and quota_anchor:
            a0, b0 = self.tt_pairs[0]
            cells = self.tdom[a0] * self.tdom[b0]
            if n_t < self.sz_snr * getattr(self, 'sigma', 0.0) * cells:
                quota_anchor = False
                self.anchor_source = 'pair-iid'
        if anchor:
            a, b = self.tt_pairs[0]
            p_ab = dist(noisy[f'TT_{a}_{b}'])
            idx = ab.exact_draw(rng, p_ab.ravel(), n_t) if quota_anchor \
                else np.array([sample_from(p_ab) for _ in range(n_t)])
            sampled[a] = idx // p_ab.shape[1]
            sampled[b] = idx % p_ab.shape[1]
        for attr in self.top_attrs:
            if attr in sampled:
                continue
            p = dist(self.top_marg[attr])
            # TT_{x}_{y} has axes (x, y): attr==x -> column joint[:, pv];
            # attr==y -> row joint[pv, :].
            partner = None
            for (x, y) in self.tt_pairs:
                if x == attr and y in sampled:
                    partner = (y, f'TT_{x}_{y}', 1)
                    break
                elif y == attr and x in sampled:
                    partner = (x, f'TT_{x}_{y}', 0)
                    break
            if partner is None:
                sampled[attr] = ab.exact_draw(rng, p, n_t) \
                    if self.ab['exactmarg'] else \
                    np.array([sample_from(p) for _ in range(n_t)])
            else:
                pname, key, axis = partner
                joint = dist(noisy[key])
                vals = np.empty(n_t, dtype=int)
                for i in range(n_t):
                    pv = sampled[pname][i]
                    cond = joint[pv, :] if axis == 0 else joint[:, pv]
                    vals[i] = sample_from(dist(cond))
                sampled[attr] = vals
        top_vals = {a: sampled[a].astype(int) for a in self.top_attrs}

        # ---------- level-1 sizes ----------
        t_margs = {a: dist(noisy[f'T_{a}']) for a in self.top_attrs}

        # ---- rowleaf: lift denominators that match the numerator's weighting --
        # A lift is joint / (marg_x (x) marg_y), so both marginals must carry the
        # same weighting as the joint.  Under `rowleaf` the pair objects count
        # ROWS of the child table while T_a counts top rows (77 on financial), so
        # dividing by dist(T_a) injects the top table's own sampling noise (144%
        # per REGION cell at eps=3.2, where sigma=13.85) into a conditional whose
        # numerator is measured to ~1e-5.  The object's own row sum is the
        # matching denominator: child rows per parent-attribute value, mass
        # n_child/dom.  Group-normalized arms keep the released denominators, so
        # this is a no-op for them.
        _mass_cache = {}

        def row_mass(name):
            if name not in _mass_cache:
                _mass_cache[name] = np.asarray(noisy[name],
                                               dtype=float).sum(axis=1)
            return _mass_cache[name]

        def lift_denom(name, fallback):
            if not self.ab['rowleaf']:
                return fallback
            return dist(row_mass(name))

        nb1 = int(noisy[f'SZ1_{self.sz1_attrs[0]}'].shape[1])
        sz1_acc = np.zeros(nb1)
        for a in self.sz1_attrs:
            sz1_acc += ab.debias_hist(
                noisy[f'SZ1_{a}'].sum(axis=0), getattr(self, 'sigma', 0.0),
                on=bool(self.ab['szthresh']),
                kappa=self.ab['szthresh'] or ab.CLIP_BIAS)
        sz1_marg = dist(sz1_acc)
        thr = self.sz_snr * getattr(self, "sigma", 0.0)
        sz1_probs = np.tile(sz1_marg, (n_t, 1))
        for a in self.sz1_attrs:
            lift = lift_table(noisy[f'SZ1_{a}'], t_margs[a], sz1_marg)
            if thr > 0:
                w = np.clip(noisy[f'SZ1_{a}'] / thr, 0.0, 1.0)
                lift = 1.0 + (lift - 1.0) * w
            sz1_probs *= lift[top_vals[a]]
        sz1_probs /= sz1_probs.sum(axis=1, keepdims=True)
        # size 0 allowed (top rows without children)
        idx1 = np.clip(sample_rows(sz1_probs, n_t), 0, nb1 - 1)
        sizes1_out = idx1 if self.sz1_grid is None else self.sz1_grid[1][idx1]
        sizes1_out = np.clip(sizes1_out, 0,
                             None if self.sz1_grid else self.tau1)
        n_m = int(sizes1_out.sum())
        if self.pub or self.tot:
            # cardinality calibration: keep the sampled SHAPE, fix the MASS
            if self.pub:
                tgt, src = self.pub['n_m'], 'public'
            else:
                tgt, src = ab.calibrate_total(noisy.get('TOTAL_M'), n_m)
            self.total_source['mid'] = src
            sizes1_out = ab.calibrate_sizes(
                self.ab, sizes1_out, tgt, lo=0,
                hi=None if self.sz1_grid else self.tau1)
            n_m = int(sizes1_out.sum())
        if self.ab['szquota']:
            # per-top-attribute-value row quotas from the ROW-LEVEL TM
            # marginals (under `rowleaf`), averaged over mid attrs -- the
            # level-1 analogue of the vshape P1C calibration.  The group-level
            # SZ1 object cannot support per-value quotas (SNR ~0.07) but TM is
            # row-mass (SNR >= 100) and its top-axis sums are exactly the
            # per-value mid counts, so a quota draw here is free post-processing.
            gs, qs = [], []
            for a in self.sz1_attrs:
                qas = [np.asarray(noisy[f'TM_{a}_{b}'], float).sum(axis=1)
                       for (aa, b) in self.tm_pairs if aa == a]
                if not qas:
                    continue
                gs.append(top_vals[a])
                qs.append(np.mean(qas, axis=0))
            if len(gs) > 1:
                sizes1_out = ab.quota_calibrate_kd(sizes1_out, gs, qs, lo=0)
            elif len(gs) == 1:
                sizes1_out = ab.quota_calibrate(
                    sizes1_out, gs[0], qs[0], lo=0,
                    hi=None if self.sz1_grid else self.tau1,
                    grow=bool(self.ab['growcap']))
            n_m = int(sizes1_out.sum())
        mid_fk = np.repeat(np.arange(n_t), sizes1_out)

        # ---------- mid attrs (vectorized over all mid rows) ----------
        mid_vals = {}
        for b in self.mid_attrs:
            m_marg = dist(noisy[f'M_{b}'])
            probs = np.tile(m_marg, (n_m, 1))
            for (a, bb) in self.tm_pairs:
                if bb != b:
                    continue
                lift = lift_table(noisy[f'TM_{a}_{b}'],
                                  lift_denom(f'TM_{a}_{b}', t_margs[a]),
                                  m_marg)
                probs *= lift[top_vals[a][mid_fk]]
            probs /= probs.sum(axis=1, keepdims=True)
            mid_vals[b] = sample_rows(probs, n_m).astype(int)

        # ---------- level-2 sizes (chunked over mid rows) ----------
        m_margs = {a: dist(noisy[f'M_{a}']) for a in self.mid_attrs}
        nb2 = int(noisy[f'SZ2_{self.sz2_attrs[0]}'].shape[1])
        sz2_acc = np.zeros(nb2)
        for a in self.sz2_attrs:
            sz2_acc += ab.debias_hist(
                noisy[f'SZ2_{a}'].sum(axis=0), getattr(self, 'sigma', 0.0),
                on=bool(self.ab['szthresh']),
                kappa=self.ab['szthresh'] or ab.CLIP_BIAS)
        sz2_marg = dist(sz2_acc)
        sz2_lifts = {}
        for a in self.sz2_attrs:
            lift = lift_table(noisy[f'SZ2_{a}'], m_margs[a], sz2_marg)
            if thr > 0:
                w = np.clip(noisy[f'SZ2_{a}'] / thr, 0.0, 1.0)
                lift = 1.0 + (lift - 1.0) * w
            sz2_lifts[a] = lift
        sizes2_out = np.empty(n_m, dtype=int)
        for s in range(0, n_m, chunk):
            e = min(s + chunk, n_m)
            probs = np.tile(sz2_marg, (e - s, 1))
            for a in self.sz2_attrs:
                probs *= sz2_lifts[a][mid_vals[a][s:e]]
            probs /= probs.sum(axis=1, keepdims=True)
            sizes2_out[s:e] = sample_rows(probs, e - s)
        if self.sz2_grid is not None:
            sizes2_out = self.sz2_grid[1][np.clip(sizes2_out, 0, nb2 - 1)]
        sizes2_out = np.clip(sizes2_out, 1,
                             None if self.sz2_grid else self.tau2)
        n_b = int(sizes2_out.sum())
        if self.pub or self.tot:
            if self.pub:
                tgt, src = self.pub['n_b'], 'public'
            else:
                tgt, src = ab.calibrate_total(noisy.get('TOTAL_B'), n_b, lo=1)
            self.total_source['bot'] = src
            sizes2_out = ab.calibrate_sizes(
                self.ab, sizes2_out, tgt, lo=1,
                hi=None if self.sz2_grid else self.tau2)
            n_b = int(sizes2_out.sum())
        if self.ab['szquota']:
            # level-2 analogue: per-mid-attribute-value row quotas from the
            # ROW-LEVEL MB marginals (under `rowleaf`), averaged over bot attrs.
            gs, qs = [], []
            for a in self.sz2_attrs:
                qas = [np.asarray(noisy[f'MB_{a}_{b}'], float).sum(axis=1)
                       for (aa, b) in self.mb_pairs if aa == a]
                if not qas:
                    continue
                gs.append(mid_vals[a])
                qs.append(np.mean(qas, axis=0))
            if len(gs) > 1:
                sizes2_out = ab.quota_calibrate_kd(sizes2_out, gs, qs, lo=1)
            elif len(gs) == 1:
                sizes2_out = ab.quota_calibrate(
                    sizes2_out, gs[0], qs[0], lo=1,
                    hi=None if self.sz2_grid else self.tau2,
                    grow=bool(self.ab['growcap']))
            n_b = int(sizes2_out.sum())
        bot_fk = np.repeat(np.arange(n_m), sizes2_out)

        # ---------- bot attrs (chunked) ----------
        # cross-hop lifts: bot attr conditioned on the attribute of its
        # ancestor TOP row (assimilated through the top -> mid -> bot path,
        # which is preserved by construction via mid_fk / bot_fk).
        tb_lifts = {}
        for (a, c) in self.tb_pairs:
            L = lift_table(noisy[f'TB_{a}_{c}'],
                           lift_denom(f'TB_{a}_{c}', t_margs[a]),
                           dist(noisy[f'B_{c}']))
            if (self.tb_shrink or self.tb_shrink_denom) and thr > 0:
                wgt = np.ones_like(L)
                if self.tb_shrink:
                    wgt = wgt * np.clip(noisy[f'TB_{a}_{c}'] / thr, 0.0, 1.0)
                if self.tb_shrink_denom:
                    # The guard measures something different from the lift's
                    # denominator, and the two must not be conflated.  The
                    # denominator normalizes the object and therefore has to carry
                    # the object's own weighting (row-level under `rowleaf`, see
                    # lift_denom).  The guard asks whether the ANCESTOR ATTRIBUTE
                    # ASSIGNMENT is reliable enough to support a conditional at
                    # all, and that is a property of the top table's own marginal
                    # (77 districts, ~9.6 rows per REGION cell against sigma=13.9
                    # at eps=3.2), not of the leaf-level object -- which is
                    # precise under `rowleaf` no matter how unreliable the
                    # ancestors are.  Keeping the parent mass here is what stops
                    # a precise cross-hop lift from being applied at full
                    # strength to a scrambled ancestor partition: measured on
                    # financial plant_product at eps=0.8, full strength overshoots
                    # the planted Delta to 0.62-0.68 against a true 0.18 while the
                    # synthesized anchor share is 0.11-0.24 against 0.47.
                    wgt = wgt * np.clip(noisy[f'T_{a}'] / thr, 0.0, 1.0)[:, None]
                L = 1.0 + (L - 1.0) * wgt
            tb_lifts[(a, c)] = L
        top_of_bot = mid_fk[bot_fk] if self.tb_pairs and n_m else None
        bot_vals = {}
        if self.ab['ipf'] and n_b and n_m:
            bot_vals = self._bot_vals_ipf(noisy, dist, sample_rows, n_b,
                                          bot_fk, mid_fk, top_vals, mid_vals,
                                          chunk)
        for c in self.bot_attrs:
            if c in bot_vals:
                continue          # fitted by IPF above
            b_marg = dist(noisy[f'B_{c}'])
            lifts = [(a, lift_table(noisy[f'MB_{a}_{c}'],
                                    lift_denom(f'MB_{a}_{c}', m_margs[a]),
                                    b_marg))
                     for (a, cc) in self.mb_pairs if cc == c]
            tb_l = [(a, tb_lifts[(a, c)]) for (a, cc) in self.tb_pairs
                    if cc == c]
            vals = np.empty(n_b, dtype=int)
            sup = np.empty(n_b, dtype=float) if self.ab['repair'] else None
            for s in range(0, n_m, chunk):
                e = min(s + chunk, n_m)
                rows = slice(int(sizes2_out[:s].sum()),
                             int(sizes2_out[:e].sum()))
                fk_slice = bot_fk[rows]
                probs = np.tile(b_marg, (fk_slice.size, 1))
                for (a, lift) in lifts:
                    probs *= lift[mid_vals[a][fk_slice]]
                if tb_l:
                    tk = top_of_bot[rows]
                    for (a, lift) in tb_l:
                        probs *= lift[top_vals[a][tk]]
                probs /= probs.sum(axis=1, keepdims=True)
                vals[rows] = sample_rows(probs, fk_slice.size)
                if sup is not None:
                    sup[rows] = probs[np.arange(fk_slice.size), vals[rows]]
            if sup is not None and n_b:
                # free post-processing: transport the sampled leaf values onto
                # the size-weighted row-level marginal estimated from SZ2 x MB
                # (see chain_npm_absorb.leaf_target for why the weighting
                # matters under 1/min(s,tau) group normalization)
                size_vals = np.arange(self.tau2 + 1, dtype=float) \
                    if self.sz2_grid is None \
                    else self.sz2_grid[1].astype(float)
                tgts = []
                if self.ab['rowleaf']:
                    # B_c is a row-level count in this arm (mass n_b), so
                    # rescaling it to n_b IS the row-level target; the
                    # size-weighted reconstruction below is only needed when the
                    # release is group-normalized.
                    tgts.append(ab.row_level_target(noisy[f'B_{c}'], n_b))
                for x in (self.sz2_attrs if not self.ab['rowleaf'] else ()):
                    key = f'MB_{x}_{c}'
                    if key not in noisy or f'SZ2_{x}' not in noisy:
                        continue
                    pj = noisy[key]
                    rs = pj.sum(axis=1, keepdims=True)
                    p_given = np.where(rs > 0, pj / np.maximum(rs, 1e-12),
                                       b_marg[None, :])
                    tgts.append(ab.leaf_target(noisy[f'SZ2_{x}'], size_vals,
                                               p_given, n_b))
                tgt = ab.average_targets(tgts, n_b)
                if tgt is None:
                    tgt = ab.row_level_target(noisy[f'B_{c}'], n_b)
                vals = ab.repair_marginal(vals, tgt, sup)
            bot_vals[c] = vals

        top_rows = np.column_stack([np.arange(n_t)]
                                   + [top_vals[a] for a in self.top_attrs])
        mid_rows = np.column_stack([np.arange(n_m), mid_fk]
                                   + [mid_vals[a] for a in self.mid_attrs]) \
            if n_m else np.zeros((0, 2 + len(self.mid_attrs)), dtype=int)
        bot_rows = np.column_stack([np.arange(n_b), bot_fk]
                                   + [bot_vals[a] for a in self.bot_attrs]) \
            if n_b else np.zeros((0, 2 + len(self.bot_attrs)), dtype=int)
        return top_rows, mid_rows, bot_rows
