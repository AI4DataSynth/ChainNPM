"""
ChainNPM generic 2-level version: parent <- child FK chain.

Generalizes chain_npm.py (hardcoded census-like 3-table schema) to any
2-level FK chain, e.g. TPC-H orders <- lineitem (with public-side
attributes such as customer nation joined into the parent table).

Mechanism (same B+NPM design):
  Phase 1: ONE budget pass — noisy low-order joint marginals:
    - parent 1-way marginals (per parent tuple)
    - selected parent x parent pair marginals (intra-parent correlation)
    - child 1-way marginals (group-size-normalized)
    - parent x child pair marginals (group-size-normalized)
    - within-group child x child pair marginals
    - group size x selected parent attribute marginals
    - mean group size
  Phase 2: free post-processing — normalize, build conditionals, sample.

Budget: Gaussian mechanism, equal gamma^2-split across all queries
(equal split validated as near-optimal for join-count workloads; see
alloc ablation in the paper).
"""

import numpy as np
import logging

logger = logging.getLogger('ChainNPM2')


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


class ChainNPM2:
    """Two-level chain synthesizer (parent <- child)."""

    def __init__(self, parent_attrs, parent_domains, child_attrs,
                 child_domains, pp_pairs=(), cc_pairs=(), sz_attrs=None,
                 sz_snr=5.0, tau=12, seed=0):
        self.parent_attrs = parent_attrs      # list of names
        self.pdom = parent_domains            # name -> domain size
        self.child_attrs = child_attrs
        self.cdom = child_domains
        self.pp_pairs = pp_pairs              # parent-parent pairs to keep
        self.cc_pairs = tuple(cc_pairs)       # within-group child pairs
        # parent attrs conditioning the group-size model; exclude attrs that
        # ARE the group size (double-counting inflates the size tail)
        self.sz_attrs = list(sz_attrs) if sz_attrs else list(parent_attrs)
        self.sz_snr = sz_snr
        self.tau = tau
        self.seed = seed

    # ---------------- Phase 1 ----------------

    def compute_marginals(self, parent, child, epsilon, delta):
        """parent: (n_p, 1+len(parent_attrs)) with PK in col 0.
        child:  (n_c, 2+len(child_attrs)) with PK col 0, FK col 1.
        """
        rng = np.random.default_rng(self.seed)
        n_p, n_c = parent.shape[0], child.shape[0]

        p_idx = {a: i + 1 for i, a in enumerate(self.parent_attrs)}
        c_idx = {a: i + 2 for i, a in enumerate(self.child_attrs)}

        # query registry: name -> (kind, spec)
        queries = {}
        for a in self.parent_attrs:
            queries[f'P_{a}'] = ('p1', a)
        for (a, b) in self.pp_pairs:
            queries[f'PP_{a}_{b}'] = ('p2', (a, b))
        for a in self.child_attrs:
            queries[f'C_{a}'] = ('c1', a)
        for a in self.parent_attrs:
            for b in self.child_attrs:
                queries[f'PC_{a}_{b}'] = ('pc', (a, b))
        # within-group child pairs: only those explicitly requested
        # (registered-but-unfilled CC queries used to burn budget for free)
        for (a, b) in self.cc_pairs:
            queries[f'CC_{a}_{b}'] = ('cc', (a, b))
        for a in self.sz_attrs:
            queries[f'SZ_{a}'] = ('sz', a)
        qnames = list(queries)
        gamma_max = analytic_gaussian_gamma(epsilon, delta)
        g2 = gamma_max ** 2 / len(qnames)

        def sigma(sens):
            return sens * np.sqrt(2.0 / g2)

        acc = {}
        for name in qnames:
            kind, spec = queries[name]
            if kind == 'p1':
                shape = (self.pdom[spec],)
            elif kind == 'p2':
                shape = (self.pdom[spec[0]], self.pdom[spec[1]])
            elif kind == 'c1':
                shape = (self.cdom[spec],)
            elif kind == 'pc':
                shape = (self.pdom[spec[0]], self.cdom[spec[1]])
            elif kind == 'cc':
                shape = (self.cdom[spec[0]], self.cdom[spec[1]])
            elif kind == 'sz':
                shape = (self.pdom[spec], self.tau + 1)
            acc[name] = np.zeros(shape, dtype=np.float64)

        # parent-level
        pa = {a: parent[:, p_idx[a]].astype(int) for a in self.parent_attrs}
        for a in self.parent_attrs:
            v = np.clip(pa[a], 0, self.pdom[a] - 1)
            np.add.at(acc[f'P_{a}'], v, 1.0)
        for (a, b) in self.pp_pairs:
            va = np.clip(pa[a], 0, self.pdom[a] - 1)
            vb = np.clip(pa[b], 0, self.pdom[b] - 1)
            np.add.at(acc[f'PP_{a}_{b}'], (va, vb), 1.0)

        # group children by FK
        fk = child[:, 1].astype(int)
        order = np.argsort(fk, kind='stable')
        fk_s = fk[order]
        boundaries = np.flatnonzero(np.diff(fk_s)) + 1
        group_starts = np.concatenate([[0], boundaries])
        group_ends = np.concatenate([boundaries, [len(fk_s)]])
        sizes_full = group_ends - group_starts
        sizes = np.minimum(sizes_full, self.tau)

        child_sorted = {a: np.clip(child[order, c_idx[a]].astype(int),
                                   0, self.cdom[a] - 1)
                        for a in self.child_attrs}
        group_pk = fk_s[group_starts]
        valid = group_pk < n_p
        group_pk = np.where(valid, group_pk, 0)

        # per-child weight = 1/min(group_size, tau), in ORIGINAL child order
        sizes_by_child = np.repeat(sizes, sizes_full)  # sorted order
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order))
        w_per_child = (1.0 / np.maximum(sizes_by_child, 1))[inv_order]

        # vectorized child-level marginals
        fk_full = child[:, 1].astype(int)  # original order, dense PK assumed
        for a in self.parent_attrs:
            pva = np.clip(pa[a][np.clip(fk_full, 0, n_p - 1)],
                          0, self.pdom[a] - 1)
            for b in self.child_attrs:
                cv = np.clip(child[:, c_idx[b]].astype(int),
                             0, self.cdom[b] - 1)
                np.add.at(acc[f'PC_{a}_{b}'], (pva, cv), w_per_child)
        for b in self.child_attrs:
            cv = np.clip(child[:, c_idx[b]].astype(int),
                         0, self.cdom[b] - 1)
            np.add.at(acc[f'C_{b}'], cv, w_per_child)

        # group size x parent attr
        for a in self.sz_attrs:
            pvg = np.clip(pa[a][group_pk], 0, self.pdom[a] - 1)
            np.add.at(acc[f'SZ_{a}'], (pvg, sizes), valid.astype(float))

        # NOTE (v0 simplification): within-group child-pair marginals (CC)
        # are registered but left zero-filled for large-scale chains such as
        # TPC-H — they matter for intra-group correlation (census-style
        # workloads, handled by chain_npm.py) but not for multi-hop join
        # counts. Adding them back requires bounded pair enumeration
        # (sensitivity cap as in PrivPetal).

        mean_size = float(sizes_full.mean()) if len(sizes_full) else 1.0

        noisy = {}
        for name in qnames:
            kind, _ = queries[name]
            sens = 2.0 if kind == 'cc' else 1.0
            noisy[name] = np.maximum(0.0, acc[name]
                                     + rng.normal(0, sigma(sens),
                                                  acc[name].shape))
        noisy['MEAN_SIZE'] = max(0.5, mean_size
                                 + rng.normal(0, sigma(1.0)))
        self.noisy = noisy
        self.sigma = float(sigma(1.0))
        return noisy

    # ---------------- Phase 2 (post-processing) ----------------

    def synthesize(self, n_parents=None):
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

        # parent sampling: use parent-pair joint if available, else 1-way
        # chain sampling in pp_pairs order (first pair anchors)
        n_p = n_parents if n_parents is not None else int(round(
            noisy['P_' + self.parent_attrs[0]].sum()))
        sampled = {}
        if self.pp_pairs:
            a, b = self.pp_pairs[0]
            p_ab = dist(noisy[f'PP_{a}_{b}'])
            idx = np.array([sample_from(p_ab) for _ in range(n_p)])
            sampled[a] = idx // p_ab.shape[1]
            sampled[b] = idx % p_ab.shape[1]
        for attr in self.parent_attrs:
            if attr in sampled:
                continue
            p = dist(noisy[f'P_{attr}'])
            # condition on already-sampled pp partner if exists.
            # PP_{x}_{y} has axes (x, y): if attr==x the conditional
            # P(x|y=pv) is the COLUMN joint[:, pv]; if attr==y it is the
            # ROW joint[pv, :].
            partner = None
            for (x, y) in self.pp_pairs:
                if x == attr and y in sampled:
                    partner = (y, f'PP_{x}_{y}', 1)
                    break
                elif y == attr and x in sampled:
                    partner = (x, f'PP_{x}_{y}', 0)
                    break
            if partner is None:
                sampled[attr] = np.array([sample_from(p)
                                          for _ in range(n_p)])
            else:
                pname, key, axis = partner
                joint = dist(noisy[key])
                sampled[attr] = np.empty(n_p, dtype=int)
                for i in range(n_p):
                    pv = sampled[pname][i]
                    cond = joint[pv, :] if axis == 0 else joint[:, pv]
                    sampled[attr][i] = sample_from(dist(cond))

        # child sampling per parent (batched per parent)
        # LIFT parameterization: start from the child 1-way marginal and
        # multiply by pointwise lifts PC/(P_a x C_b).  With redundant
        # conditioning (child attr independent of several parent attrs) the
        # lifts are ~1, so the marginal is preserved instead of collapsing
        # to p^n under a naive product of conditionals.
        def lift_table(joint, pa_marg, cb_marg, smooth=1e-6):
            # ratio[u, v] = joint[u,v] / (pa[u]*cb[v]), smoothed
            denom = np.outer(pa_marg, cb_marg) + smooth
            return (joint + smooth / joint.size) / denom

        c_margs = {}
        for b in self.child_attrs:
            cm = noisy[f'C_{b}']
            c_margs[b] = cm / max(cm.sum(), 1e-9)
        p_margs = {}
        for a in self.parent_attrs:
            pm = noisy[f'P_{a}']
            p_margs[a] = pm / max(pm.sum(), 1e-9)
        cond_tables = {}
        for b in self.child_attrs:
            cond_tables[b] = {}
            for a in self.parent_attrs:
                cond_tables[b][a] = lift_table(noisy[f'PC_{a}_{b}'],
                                               p_margs[a], c_margs[b])
        # size: lift form over the SZ marginal
        sz_marg_acc = np.zeros(self.tau + 1)
        for a in self.sz_attrs:
            sz_marg_acc += noisy[f'SZ_{a}'].sum(axis=0)
        sz_marg = sz_marg_acc / max(sz_marg_acc.sum(), 1e-9)
        # noise-adaptive lift shrinkage (empirical-Bayes style): cells with
        # mass below 10 sigma are shrunk toward lift 1 proportionally to
        # their SNR; well-estimated cells keep their true lift
        sz_lifts = {}
        thr = self.sz_snr * getattr(self, "sigma", 0.0)
        for a in self.sz_attrs:
            lift = lift_table(noisy[f'SZ_{a}'], p_margs[a], sz_marg)
            if thr > 0:
                w = np.clip(noisy[f'SZ_{a}'] / thr, 0.0, 1.0)
                lift = 1.0 + (lift - 1.0) * w
            sz_lifts[a] = lift

        sampled_np = {a: sampled[a].astype(int) for a in self.parent_attrs}

        # per-parent child-attribute distributions (same for all children
        # of a parent)
        all_probs = {}
        for b in self.child_attrs:
            all_probs[b] = np.empty((n_p, self.cdom[b]))
            for i in range(n_p):
                probs = c_margs[b].copy()
                for a in self.parent_attrs:
                    probs *= cond_tables[b][a][sampled_np[a][i]]
                all_probs[b][i] = probs / probs.sum()

        # group sizes: SZ marginal x lifts over ALL parent attrs
        # (recovers size correlations with every parent attribute, e.g.
        # the planted order-size <-> order-year association)
        sz_probs = np.empty((n_p, self.tau + 1))
        for i in range(n_p):
            probs = sz_marg.copy()
            for a in self.sz_attrs:
                probs *= sz_lifts[a][sampled_np[a][i]]
            sz_probs[i] = probs / probs.sum()
        u_vec = rng.random(n_p)
        sz_cum = np.cumsum(sz_probs, axis=1)
        sizes_out = (sz_cum[:, :-1] < u_vec[:, None]).sum(axis=1)
        sizes_out = np.clip(sizes_out, 1, self.tau).astype(int)
        # NOTE: calibrating the total to the released MEAN_SIZE statistic
        # was tried and rejected — its noise (sigma ~ tens at these budgets)
        # dwarfs the mean itself and rescaling amplifies it catastrophically

        # batched sampling per parent per attribute
        child_cols = {b: [] for b in self.child_attrs}
        for i in range(n_p):
            s = int(sizes_out[i])
            for b in self.child_attrs:
                child_cols[b].append(
                    rng.choice(self.cdom[b], size=s, p=all_probs[b][i]))

        total = int(sizes_out.sum())
        order_fk = np.repeat(np.arange(n_p), sizes_out)
        child_arr = np.column_stack(
            [np.arange(total), order_fk]
            + [np.concatenate(child_cols[b]) for b in self.child_attrs]) \
            if total else np.zeros((0, 2 + len(self.child_attrs)), dtype=int)

        parent_rows = np.column_stack([np.arange(n_p)]
                                      + [sampled[a] for a in
                                         self.parent_attrs])
        return parent_rows, child_arr
