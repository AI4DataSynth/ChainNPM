"""
ChainNPM3: three-level FK chain synthesizer (top <- mid <- bot).

Deep-chain generalization of chain_npm_generic.py (ChainNPM2), e.g.
TPC-H customer <- orders <- lineitem with ALL THREE tables private.

Mechanism (same B+NPM design):
  Phase 1: ONE budget pass — noisy low-order joint marginals:
    - top 1-way + selected top x top pairs (intra-top correlation)
    - mid 1-way + top x mid pairs (group-size-normalized by children/top)
    - bot 1-way + mid x bot pairs (group-size-normalized by children/mid)
    - group size (mid per top) x selected top attrs
    - group size (bot per mid) x selected mid attrs
    - mean group sizes at both levels
  Phase 2: free post-processing — normalize, build lift tables, sample
    top -> (size1, mid attrs) -> (size2, bot attrs).

Multi-hop correlations across all three private tables are preserved
because the mid rows are sampled ONCE consistently and condition both
the top->mid and mid->bot lifts (composition survives by construction).

Budget: Gaussian mechanism, equal gamma^2-split across all queries.
"""

import numpy as np
import logging

logger = logging.getLogger('ChainNPM3')


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
    """Three-level chain synthesizer (top <- mid <- bot)."""

    def __init__(self, top_attrs, top_domains, mid_attrs, mid_domains,
                 bot_attrs, bot_domains, tt_pairs=(), tm_pairs=(),
                 mb_pairs=(), sz1_attrs=None, sz2_attrs=None,
                 sz_snr=5.0, tau1=40, tau2=8, seed=0):
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
        for a in self.sz1_attrs:
            queries[f'SZ1_{a}'] = ('sz1', a)
        for a in self.sz2_attrs:
            queries[f'SZ2_{a}'] = ('sz2', a)
        qnames = list(queries)
        n_queries = len(qnames) + 2  # + 2 mean-size queries
        gamma_max = analytic_gaussian_gamma(epsilon, delta)
        g2 = gamma_max ** 2 / n_queries

        sigma = np.sqrt(2.0 / g2)  # sensitivity 1 for every query

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
            elif kind == 'sz1':
                shape = (self.tdom[spec], self.tau1 + 1)
            elif kind == 'sz2':
                shape = (self.mdom[spec], self.tau2 + 1)
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
        w1 = (1.0 / np.maximum(sizes1_by_row, 1))[inv_order1]

        ma = {a: np.clip(mid[:, m_idx[a]].astype(int), 0, self.mdom[a] - 1)
              for a in self.mid_attrs}
        for a in self.mid_attrs:
            np.add.at(acc[f'M_{a}'], ma[a], w1)
        for (a, b) in self.tm_pairs:
            tv = np.clip(ta[a][np.clip(fk1, 0, n_t - 1)],
                         0, self.tdom[a] - 1)
            np.add.at(acc[f'TM_{a}_{b}'], (tv, ma[b]), w1)
        for a in self.sz1_attrs:
            np.add.at(acc[f'SZ1_{a}'], (ta[a][top_pk1], sizes1),
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
        w2 = (1.0 / np.maximum(sizes2_by_row, 1))[inv_order2]

        ba = {a: np.clip(bot[:, b_idx[a]].astype(int), 0, self.bdom[a] - 1)
              for a in self.bot_attrs}
        for a in self.bot_attrs:
            np.add.at(acc[f'B_{a}'], ba[a], w2)
        for (a, b) in self.mb_pairs:
            mv = np.clip(ma[a][np.clip(fk2, 0, n_m - 1)],
                         0, self.mdom[a] - 1)
            np.add.at(acc[f'MB_{a}_{b}'], (mv, ba[b]), w2)
        for a in self.sz2_attrs:
            np.add.at(acc[f'SZ2_{a}'], (ma[a][mid_pk2], sizes2),
                      valid2.astype(float))

        mean_size1 = float(sizes1_full.mean()) if len(sizes1_full) else 1.0
        mean_size2 = float(sizes2_full.mean()) if len(sizes2_full) else 1.0

        noisy = {}
        for name in qnames:
            noisy[name] = np.maximum(0.0, acc[name]
                                     + rng.normal(0, sigma, acc[name].shape))
        noisy['MEAN_SIZE1'] = max(0.5, mean_size1 + rng.normal(0, sigma))
        noisy['MEAN_SIZE2'] = max(0.5, mean_size2 + rng.normal(0, sigma))
        self.noisy = noisy
        self.sigma = float(sigma)
        self.n_top = n_t
        return noisy

    # ---------------- Phase 2 (post-processing) ----------------
    # NOTE: calibrating group-size totals to the released MEAN_SIZE*
    # statistics was tried and rejected — their noise (sigma ~ tens at
    # these budgets) dwarfs the means and rescaling amplifies it.

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
        sampled = {}
        if self.tt_pairs:
            a, b = self.tt_pairs[0]
            p_ab = dist(noisy[f'TT_{a}_{b}'])
            idx = np.array([sample_from(p_ab) for _ in range(n_t)])
            sampled[a] = idx // p_ab.shape[1]
            sampled[b] = idx % p_ab.shape[1]
        for attr in self.top_attrs:
            if attr in sampled:
                continue
            p = dist(noisy[f'T_{attr}'])
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
                sampled[attr] = np.array([sample_from(p)
                                          for _ in range(n_t)])
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
        sz1_acc = np.zeros(self.tau1 + 1)
        for a in self.sz1_attrs:
            sz1_acc += noisy[f'SZ1_{a}'].sum(axis=0)
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
        sizes1_out = np.clip(sample_rows(sz1_probs, n_t), 0, self.tau1)
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
                lift = lift_table(noisy[f'TM_{a}_{b}'], t_margs[a], m_marg)
                probs *= lift[top_vals[a][mid_fk]]
            probs /= probs.sum(axis=1, keepdims=True)
            mid_vals[b] = sample_rows(probs, n_m).astype(int)

        # ---------- level-2 sizes (chunked over mid rows) ----------
        m_margs = {a: dist(noisy[f'M_{a}']) for a in self.mid_attrs}
        sz2_acc = np.zeros(self.tau2 + 1)
        for a in self.sz2_attrs:
            sz2_acc += noisy[f'SZ2_{a}'].sum(axis=0)
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
        sizes2_out = np.clip(sizes2_out, 1, self.tau2)
        n_b = int(sizes2_out.sum())
        bot_fk = np.repeat(np.arange(n_m), sizes2_out)

        # ---------- bot attrs (chunked) ----------
        bot_vals = {}
        for c in self.bot_attrs:
            b_marg = dist(noisy[f'B_{c}'])
            lifts = [(a, lift_table(noisy[f'MB_{a}_{c}'], m_margs[a], b_marg))
                     for (a, cc) in self.mb_pairs if cc == c]
            vals = np.empty(n_b, dtype=int)
            for s in range(0, n_m, chunk):
                e = min(s + chunk, n_m)
                rows = slice(int(sizes2_out[:s].sum()),
                             int(sizes2_out[:e].sum()))
                fk_slice = bot_fk[rows]
                probs = np.tile(b_marg, (fk_slice.size, 1))
                for (a, lift) in lifts:
                    probs *= lift[mid_vals[a][fk_slice]]
                probs /= probs.sum(axis=1, keepdims=True)
                vals[rows] = sample_rows(probs, fk_slice.size)
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
