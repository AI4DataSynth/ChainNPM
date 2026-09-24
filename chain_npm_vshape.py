"""ChainNPM V-shape (star) extension: root1 <- child -> root2.

The child table references TWO parents (e.g. ratings -> users, movies).
Per-FK methods measure each parent-child pair separately and merge on the
child, so the cross-parent object (root1 attr x root2 attr) enters no
queried marginal.  ChainNPMV adds it to the measured set directly:

  Phase 1 (one budget pass):
    - root1 / root2 1-way marginals
    - XP: root1 x root2 cross-parent marginals, weight 1/size1 per child row
    - child 1-way and root1 x child / root2 x child, weight 1/size1
    - SZ: root1 group-size marginals
  Phase 2 (free post-processing): sample root1; per root1 tuple sample its
    root2 partners from the XP conditional; sample group size; sample child
    attributes via lifts conditioned on both sampled parents.

Sensitivity: with 1/size1 weights and tau-truncation of root1 groups, each
query's l2 sensitivity is <= 1 for a root1-cascading neighbor (same per-group
argument as Lemma 5.1); the root2-side bound follows by the same argument
with roles swapped (Appendix B of the paper).
"""

import numpy as np
from chain_npm_generic import analytic_gaussian_gamma
import chain_npm_absorb as ab


class ChainNPMV:
    def __init__(self, r1_attrs, r1_dom, r2_attrs, r2_dom,
                 child_attrs, child_dom, xp_pairs=(), sz_attrs=None,
                 tau=12, seed=0, absorb=None):
        self.r1_attrs, self.r1_dom = list(r1_attrs), dict(r1_dom)
        self.r2_attrs, self.r2_dom = list(r2_attrs), dict(r2_dom)
        self.child_attrs, self.cdom = list(child_attrs), dict(child_dom)
        self.xp_pairs = list(xp_pairs) or [(a, b) for a in self.r1_attrs
                                           for b in self.r2_attrs]
        self.sz_attrs = list(sz_attrs) if sz_attrs else list(r1_attrs)
        self.tau = tau
        self.seed = seed
        # Absorb arm (chain_npm_absorb): absorb=None keeps the path below
        # bit-identical to the released code.  `pubn` is the cardinality arm --
        # the star adjacency of Appendix B is a root-tuple cascade, under which
        # |C| has sensitivity = max group size, so the child cardinality cannot
        # be bought at sensitivity 1 here; the arm instead takes table sizes as
        # public schema metadata, which is what every per-table baseline in the
        # comparison already does.
        self.ab = ab.parse('off') if absorb is None else ab.parse(absorb)
        self.absorb = ab.describe(self.ab)
        self.adjacency = ('one-child-row' if self.ab['rowleaf']
                          else ab.adjacency(self.ab) + '; star: root-cascade')
        self.pub = None
        self.sz_grid = None
        self.sz_sens = 1.0

    def compute_marginals(self, root1, root2, child, epsilon, delta):
        """root1: (n1, 1+len(r1_attrs)) PK col0.
        root2: (n2, 1+len(r2_attrs)) PK col0.
        child: (nc, 3+len(child_attrs)) PK col0, fk1 col1, fk2 col2.
        """
        rng = np.random.default_rng(self.seed)
        n1, n2 = root1.shape[0], root2.shape[0]
        i1 = {a: i + 1 for i, a in enumerate(self.r1_attrs)}
        i2 = {a: i + 1 for i, a in enumerate(self.r2_attrs)}
        ic = {a: i + 3 for i, a in enumerate(self.child_attrs)}

        queries = {}
        for a in self.r1_attrs:
            queries[f'P1_{a}'] = ('p1', a)
        for b in self.r2_attrs:
            queries[f'P2_{b}'] = ('p2', b)
        for (a, b) in self.xp_pairs:
            queries[f'XP_{a}_{b}'] = ('xp', (a, b))
        for c in self.child_attrs:
            queries[f'C_{c}'] = ('c1', c)
            for a in self.r1_attrs:
                queries[f'P1C_{a}_{c}'] = ('p1c', (a, c))
            for b in self.r2_attrs:
                queries[f'P2C_{b}_{c}'] = ('p2c', (b, c))
        # cross-parent x child triples: the star analogue of cross-hop
        # triples; needed for XOR-type cross-parent interactions
        for (a, b) in self.xp_pairs:
            for c in self.child_attrs:
                queries[f'XPC_{a}_{b}_{c}'] = ('xpc', (a, b, c))
        for a in self.sz_attrs:
            queries[f'SZ_{a}'] = ('sz', a)
        qnames = list(queries)
        g2_total = analytic_gaussian_gamma(epsilon, delta) ** 2
        # even split (released behaviour): sigma=lambda s: s*sqrt(2/g2) with
        # g2 = g2_total/len(qnames).  `budget` re-allocates the same g2_total
        # across queries (1-way marginals richer, joints poorer) so that the
        # single-table marginals get the same accuracy advantage the per-table
        # baselines buy by spending epsilon per table; the sum of 1/sigma^2 is
        # unchanged, i.e. the privacy account is identical.
        g2 = g2_total / len(qnames)
        sigma = lambda s: s * np.sqrt(2.0 / g2)
        self.sigma_map = None
        self.sigma_sz = None
        if self.ab['budget']:
            w1, w2 = self.ab['budget']
            smap, _ = ab.budget_sigma(g2_total, qnames, ab.is_1way_name, w1, w2)
            self.sigma_map = smap
            szq = 'SZ_' + self.sz_attrs[0]
            self.sigma_sz = self.sz_sens * smap[szq]

        # ---- absorb arm setup ------------------------------------------------
        # `bins`: geometric size bins over [1, nc] instead of 0..tau.  The upper
        # edge is the child cardinality, which this arm already holds (pubn) or
        # buys (totals), so the grid itself leaks nothing new.  `bins` on its own
        # does not touch the 1/min(s, tau) weights, hence Lemma lem:sens is
        # untouched and under the star's root-cascade adjacency a histogram entry
        # still moves by at most 1 (see `rowleaf` for the arm that changes that).
        nc = int(child.shape[0])
        self.sz_grid = None
        # Under the root-cascade neighbour a size histogram entry moves by 1.
        # Under `rowleaf` (one-child-row neighbour) removing a child row moves
        # its group from bin s to bin s-1, i.e. sqrt(2); `szsens` charges the
        # same on demand.
        self.sz_sens = ab._SZ_SENS_CHAIN if (self.ab['rowleaf']
                                             or self.ab['szsens']) else 1.0
        self.sz_grid = ab.size_grid(self.ab, self.tau, nc, zero_bin=False)
        nb_sz = self.tau + 1 if self.sz_grid is None \
            else int(len(self.sz_grid[1]))
        if self.ab['pubn']:
            self.pub = {'n1': int(n1), 'n2': int(n2), 'nc': nc}

        shapes = {
            'p1': lambda a: (self.r1_dom[a],),
            'p2': lambda b: (self.r2_dom[b],),
            'xp': lambda ab: (self.r1_dom[ab[0]], self.r2_dom[ab[1]]),
            'c1': lambda c: (self.cdom[c],),
            'p1c': lambda ac: (self.r1_dom[ac[0]], self.cdom[ac[1]]),
            'p2c': lambda bc: (self.r2_dom[bc[0]], self.cdom[bc[1]]),
            'xpc': lambda abc: (self.r1_dom[abc[0]], self.r2_dom[abc[1]],
                                self.cdom[abc[2]]),
            'sz': lambda a: (self.r1_dom[a], nb_sz),
        }
        acc = {n: np.zeros(shapes[k](s), float)
               for n, (k, s) in queries.items()}

        fk1 = np.clip(child[:, 1].astype(int), 0, n1 - 1)
        fk2 = np.clip(child[:, 2].astype(int), 0, n2 - 1)

        # root1 group sizes (tau-truncated) with per-child weight 1/size
        order = np.argsort(fk1, kind='stable')
        bnd = np.flatnonzero(np.diff(fk1[order])) + 1
        gs = np.concatenate([[0], bnd])
        ge = np.concatenate([bnd, [len(fk1)]])
        sizes_full = ge - gs
        sizes = np.minimum(sizes_full, self.tau)
        gid = np.zeros(len(fk1), int)
        gid[order] = np.repeat(np.arange(len(sizes)), sizes_full)
        # `rowleaf` weights every child row by 1 instead of 1/min(s,tau), which
        # lifts the mass of XP/C/P1C/P2C/XPC from the root1 group count to the
        # child row count (a factor of ~166 on MovieLens).  NOTE: Appendix B
        # states this rail's neighbour as "one R1 tuple together with its C
        # rows", under which a row-level object has sensitivity = max group size.
        # A rowleaf arm on the star is therefore DP only under the ONE-CHILD-ROW
        # neighbour, i.e. the same relation the chain rails and the per-table
        # baselines use; hyper records that and the arm must not be reported
        # without it.
        w = ab.row_weight(self.ab, 1.0 / np.maximum(sizes[gid], 1))

        r1v = {a: root1[:, i1[a]].astype(int) for a in self.r1_attrs}
        r2v = {b: root2[:, i2[b]].astype(int) for b in self.r2_attrs}
        cv = {c: np.clip(child[:, ic[c]].astype(int), 0, self.cdom[c] - 1)
              for c in self.child_attrs}
        a1 = {a: np.clip(r1v[a][fk1], 0, self.r1_dom[a] - 1)
              for a in self.r1_attrs}
        a2 = {b: np.clip(r2v[b][fk2], 0, self.r2_dom[b] - 1)
              for b in self.r2_attrs}

        for a in self.r1_attrs:
            np.add.at(acc[f'P1_{a}'], r1v[a], 1.0)
        for b in self.r2_attrs:
            np.add.at(acc[f'P2_{b}'], r2v[b], 1.0)
        for (a, b) in self.xp_pairs:
            np.add.at(acc[f'XP_{a}_{b}'], (a1[a], a2[b]), w)
        for c in self.child_attrs:
            np.add.at(acc[f'C_{c}'], cv[c], w)
            for a in self.r1_attrs:
                np.add.at(acc[f'P1C_{a}_{c}'], (a1[a], cv[c]), w)
            for b in self.r2_attrs:
                np.add.at(acc[f'P2C_{b}_{c}'], (a2[b], cv[c]), w)
        for (a, b) in self.xp_pairs:
            for c in self.child_attrs:
                np.add.at(acc[f'XPC_{a}_{b}_{c}'], (a1[a], a2[b], cv[c]), w)
        gpk = fk1[order][gs]
        # `bins` histograms the UNtruncated group size (the tau cap is what
        # throws the heavy-tailed leaf mass away); weights above keep 1/min(s,tau)
        sz_idx = sizes if self.sz_grid is None else \
            ab.to_bins(sizes_full, self.sz_grid[0])
        for a in self.sz_attrs:
            np.add.at(acc[f'SZ_{a}'], (np.clip(r1v[a][gpk], 0,
                      self.r1_dom[a] - 1), sz_idx), 1.0)

        noisy = {}
        sigma_base = float(sigma(1.0))
        for n in qnames:
            sens = self.sz_sens if queries[n][0] == 'sz' else 1.0
            sig = sens * (self.sigma_map[n] if self.sigma_map is not None
                          else sigma_base)
            noisy[n] = np.maximum(0.0, acc[n]
                                  + rng.normal(0, sig, acc[n].shape))
        # `shrink`: per-cell reliability gate toward independence for the
        # joint objects.  Replaces the noisy count of a low-mass cell by a
        # convex mix of itself and total x (marginal shares product), which is
        # exactly the independence estimate the per-table baselines produce
        # through their FK re-link.  At low epsilon the released joints are
        # noise-dominated and the shrunk version (much closer to the baselines)
        # lifts 1hop_r1c / cross_r1r2; at high epsilon every cell clears the
        # gate (noisy ~ truth >> sigma) and the arm is unchanged.  Marginal
        # inputs are the released 1-ways stripped of the clipping bias if
        # `szthresh`'s debias constant is available; mass scale is irrelevant
        # (shares only).
        if self.ab['shrink']:
            kappa = self.ab['shrink']
            m1d = {a: ab.debias_hist(noisy[f'P1_{a}'], sigma_base, on=True)
                   for a in self.r1_attrs}
            m2d = {b: ab.debias_hist(noisy[f'P2_{b}'], sigma_base, on=True)
                   for b in self.r2_attrs}
            mcd = {c: ab.debias_hist(noisy[f'C_{c}'], sigma_base, on=True)
                   for c in self.child_attrs}
            for (a, b) in self.xp_pairs:
                k = f'XP_{a}_{b}'
                sig = self.sigma_map[k] if self.sigma_map is not None \
                    else sigma_base
                noisy[k] = ab.shrink_joint(noisy[k], [m1d[a], m2d[b]], sig,
                                           kappa)
            for a in self.r1_attrs:
                for c in self.child_attrs:
                    k = f'P1C_{a}_{c}'
                    sig = self.sigma_map[k] if self.sigma_map is not None \
                        else sigma_base
                    noisy[k] = ab.shrink_joint(noisy[k], [m1d[a], mcd[c]], sig,
                                               kappa)
            for b in self.r2_attrs:
                for c in self.child_attrs:
                    k = f'P2C_{b}_{c}'
                    sig = self.sigma_map[k] if self.sigma_map is not None \
                        else sigma_base
                    noisy[k] = ab.shrink_joint(noisy[k], [m2d[b], mcd[c]], sig,
                                               kappa)
            for (a, b) in self.xp_pairs:
                for c in self.child_attrs:
                    k = f'XPC_{a}_{b}_{c}'
                    sig = self.sigma_map[k] if self.sigma_map is not None \
                        else sigma_base
                    noisy[k] = ab.shrink_joint(noisy[k],
                                               [m1d[a], m2d[b], mcd[c]], sig,
                                               kappa)
        # The MEAN_SIZE draw is gone: Phase 2 never read it (see the calibration
        # NOTE there) and it was not in the gamma^2 account, so it was a dead,
        # unaccounted noise draw.  It is drawn after every released marginal, so
        # removing it leaves the released values bit-identical.
        self.noisy, self.sigma = noisy, sigma_base
        self.Q = len(qnames)
        self.n_queries = len(qnames)
        return noisy

    def synthesize(self, n_root1=None):
        rng = np.random.default_rng(self.seed + 12345)
        nz = self.noisy

        def dist(x):
            x = np.asarray(x, float)
            s = x.sum()
            return x / s if s > 0 else np.full(x.shape, 1.0 / x.size)

        n1 = n_root1 or int(round(nz['P1_' + self.r1_attrs[0]].sum()))
        if n_root1 is None and self.ab['pubn'] and self.pub:
            n1 = self.pub['n1']
        s1 = {}
        for a in self.r1_attrs:
            p1 = dist(nz[f'P1_{a}'])
            # `exactmarg`: quota draw, so the synthesized root table reproduces
            # the released 1-way marginal exactly instead of within 1/sqrt(n1)
            s1[a] = ab.exact_draw(rng, p1, n1) if self.ab['exactmarg'] \
                else rng.choice(self.r1_dom[a], size=n1, p=p1)

        def lift(joint, ma, mb, smooth=1e-6):
            return (joint + smooth / joint.size) / (np.outer(ma, mb) + smooth)

        m1 = {a: dist(nz[f'P1_{a}']) for a in self.r1_attrs}
        m2 = {b: dist(nz[f'P2_{b}']) for b in self.r2_attrs}
        mc = {c: dist(nz[f'C_{c}']) for c in self.child_attrs}
        xp_lift = {(a, b): lift(nz[f'XP_{a}_{b}'], m1[a], m2[b])
                   for (a, b) in self.xp_pairs}
        p1c_lift = {(a, c): lift(nz[f'P1C_{a}_{c}'], m1[a], mc[c])
                    for a in self.r1_attrs for c in self.child_attrs}
        p2c_lift = {(b, c): lift(nz[f'P2C_{b}_{c}'], m2[b], mc[c])
                    for b in self.r2_attrs for c in self.child_attrs}

        # `szthresh` removes the mass that clipping at zero injects into every
        # empty size bin (see chain_npm_absorb.debias_hist); a no-op otherwise.
        # Under `budget` the SZ query carries a joint weight, so its own sigma
        # (possibly > the 1-way sigma) must drive the debias threshold.
        sz_sigma = self.sigma_sz if self.sigma_sz is not None \
            else getattr(self, 'sigma', 0.0)
        sz_clean = nz
        if bool(self.ab['szthresh']):
            # Per-CELL debias of the size object, not just its marginal.
            # The SZ object is GROUP-level mass: instacart 38.5k groups over
            # 16 x ~140 cells = ~17 groups/cell against sigma 263 (SNR 0.07),
            # so almost every cell is clipping-biased noise.  Debiasing only
            # the marginal (the old code) left the raw noisy cells inside
            # sz_lift, which then scrambled every parent's size draw by a
            # noise lift -> per-parent child counts off by 0.2x..5.6x at
            # eps=0.1.  Debiasing each cell first makes noise-only cells
            # (clean <= 0) contribute lift = 1, i.e. fall back to the global
            # size distribution, exactly what the per-table baselines achieve
            # (they never touch this object).  Strong cells keep their lift.
            sz_clean = {a: ab.debias_hist(nz[f'SZ_{a}'], sz_sigma,
                                          on=True, kappa=self.ab['szthresh']
                                          or ab.CLIP_BIAS)
                        for a in self.sz_attrs}
        sz_acc = None
        for a in self.sz_attrs:
            x = sz_clean[a] if self.ab['szthresh'] \
                else nz[f'SZ_{a}'].sum(0)
            x = x.sum(0) if self.ab['szthresh'] else x
            sz_acc = x if sz_acc is None else sz_acc + x
        sz_marg = dist(sz_acc)
        thr = 5.0 * sz_sigma
        sz_lift = {}
        for a in self.sz_attrs:
            x = sz_clean[a] if self.ab['szthresh'] \
                else nz[f'SZ_{a}']
            L = lift(x, m1[a], sz_marg)
            wgt = np.clip(x / thr, 0, 1) if thr > 0 else 1.0
            sz_lift[a] = 1.0 + (L - 1.0) * wgt

        # ---- absorb arm: pre-draw every group size, then calibrate ----------
        # Two reasons to hoist the size draws out of the generation loop: with
        # `bins` the draws come from geometric bin representatives rather than
        # 0..tau, and with a known cardinality the whole vector is rescaled so
        # that sum(sizes) equals it.  sizes_pre is None in the default arm, which
        # keeps drawing the size inside the loop (bit-identical rng order).
        sizes_pre = None
        if self.ab['bins'] or (self.ab['pubn'] and self.pub) \
                or self.ab['szquota']:
            reps = None if self.sz_grid is None else self.sz_grid[1]
            sizes_pre = np.empty(n1, dtype=np.int64)
            chunk = 200000
            for lo_i in range(0, n1, chunk):
                hi_i = min(lo_i + chunk, n1)
                sp = np.tile(sz_marg, (hi_i - lo_i, 1))
                for x in self.sz_attrs:
                    sp *= sz_lift[x][s1[x][lo_i:hi_i]]
                sp /= sp.sum(axis=1, keepdims=True)
                u = rng.random(hi_i - lo_i)
                cum = np.cumsum(sp, axis=1)
                idx = (cum[:, :-1] < u[:, None]).sum(axis=1)
                if reps is None:
                    sizes_pre[lo_i:hi_i] = np.clip(idx, 1, self.tau)
                else:
                    sizes_pre[lo_i:hi_i] = np.maximum(
                        1, reps[np.clip(idx, 0, len(reps) - 1)])
            if self.ab['pubn'] and self.pub:
                sizes_pre = ab.calibrate_sizes(
                    self.ab, sizes_pre, self.pub['nc'], lo=1,
                    hi=None if self.sz_grid else self.tau)
            if self.ab['szquota']:
                # per-parent-attribute-value row quotas from the ROW-LEVEL
                # P1C marginal (under `rowleaf`), averaged over child attrs.
                # At eps=0.1 the group-level SZ object cannot support them
                # (SNR 0.07) but P1C is row-mass (SNR >= 100) and its
                # parent-axis sums are exactly the per-value child counts.
                # With MORE THAN ONE parent attribute a per-attribute
                # sequential calibration destroys the earlier marginals
                # (each group sits in one value of every attribute), so a
                # joint RAS adjustment across ALL r1 attributes is used
                # (quota_calibrate_kd).
                gs, qs = [], []
                for a in self.r1_attrs:
                    qas = [nz[f'P1C_{a}_{c}'].sum(axis=1)
                           for c in self.child_attrs
                           if f'P1C_{a}_{c}' in nz]
                    if not qas:
                        continue
                    gs.append(s1[a])
                    qs.append(np.mean(qas, axis=0))
                if len(gs) > 1:
                    sizes_pre = ab.quota_calibrate_kd(sizes_pre, gs, qs,
                                                      lo=1)
                elif len(gs) == 1:
                    sizes_pre = ab.quota_calibrate(
                        sizes_pre, gs[0], qs[0], lo=1,
                        hi=None if self.sz_grid else self.tau,
                        grow=bool(self.ab['growcap']))

        a, b = self.xp_pairs[0]
        cols = {c: [] for c in self.child_attrs}
        sco = {c: [] for c in self.child_attrs} if self.ab['repair'] else None
        r2_out = {b2: [] for b2 in self.r2_attrs}
        fks = []
        for i in range(n1):
            p = m2[b].copy() * xp_lift[(a, b)][s1[a][i]]
            for (x, y) in self.xp_pairs[1:]:
                if x == a:
                    p *= xp_lift[(x, y)][s1[x][i]]
            p = p / p.sum()
            if sizes_pre is None:
                sp = sz_marg.copy()
                for x in self.sz_attrs:
                    sp *= sz_lift[x][s1[x][i]]
                sp = sp / sp.sum()
                size = int(rng.choice(self.tau + 1, p=sp))
                size = max(1, min(size, self.tau))
            else:
                size = int(sizes_pre[i])
            for _ in range(size):
                rv2 = int(rng.choice(self.r2_dom[b], p=p))
                r2vals = {b: rv2}
                for b2 in self.r2_attrs:
                    if b2 == b:
                        continue
                    q = m2[b2].copy()
                    for (x, y) in self.xp_pairs:
                        if y == b2 and x == a:
                            q *= xp_lift[(x, y)][s1[x][i], :]
                    r2vals[b2] = int(rng.choice(self.r2_dom[b2],
                                                p=q / q.sum()))
                for c in self.child_attrs:
                    q = nz[f'XPC_{a}_{b}_{c}'][s1[a][i], r2vals[b]].copy()
                    if q.sum() <= 0:
                        q = mc[c].copy()
                        for x in self.r1_attrs:
                            q *= p1c_lift[(x, c)][s1[x][i]]
                        for y in self.r2_attrs:
                            q *= p2c_lift[(y, c)][r2vals[y]]
                    qq = q / q.sum()
                    v = int(rng.choice(self.cdom[c], p=qq))
                    cols[c].append(v)
                    if sco is not None:
                        # support of the drawn value under the conditional that
                        # produced it: the repair pass moves weak rows first
                        sco[c].append(float(qq[v]))
                for b2 in self.r2_attrs:
                    r2_out[b2].append(r2vals[b2])
                fks.append(i)
        total = len(fks)
        if sco is not None and total:
            # Free post-processing.  The released leaf marginal C_c carries the
            # mass of the group count (relative noise ~sigma/(n1*p)) while each
            # row above was drawn from one cross-parent triple cell, whose mass
            # n1 is spread over dom(a)*dom(b)*dom(c) cells and is therefore far
            # noisier per cell.  Transport the sampled values onto the released
            # marginal rescaled to `total`, moving the least-supported rows
            # first.  Costs no budget and does not touch the FK structure.
            size_vals = np.arange(self.tau + 1, dtype=float) \
                if self.sz_grid is None else self.sz_grid[1].astype(float)
            for c in self.child_attrs:
                if self.ab['rowleaf']:
                    # C_c is itself a row-level count in this arm (mass n_c), so
                    # rescaling it to `total` IS the target: no size weighting
                    # needed, and it beats the i.i.d. sampling floor of ~1/sqrt(n)
                    # that leaves the released arm's 0-hop error an order of
                    # magnitude above the per-table baselines.
                    cols[c] = ab.repair_marginal(
                        np.asarray(cols[c], dtype=np.int64),
                        ab.row_level_target(nz[f'C_{c}'], total),
                        np.asarray(sco[c], dtype=float))
                    continue
                tgts = []
                for x in self.sz_attrs:
                    key = f'P1C_{x}_{c}'
                    if key not in nz or f'SZ_{x}' not in nz:
                        continue
                    pj = nz[key]
                    rs = pj.sum(axis=1, keepdims=True)
                    p_given = np.where(rs > 0, pj / np.maximum(rs, 1e-12),
                                       dist(nz[f'C_{c}'])[None, :])
                    tgts.append(ab.leaf_target(nz[f'SZ_{x}'], size_vals,
                                               p_given, total))
                tgt = ab.average_targets(tgts, total)
                if tgt is None:
                    tgt = ab.row_level_target(nz[f'C_{c}'], total)
                cols[c] = ab.repair_marginal(
                    np.asarray(cols[c], dtype=np.int64), tgt,
                    np.asarray(sco[c], dtype=float))
        child_arr = np.column_stack([np.arange(total), fks, fks]
                                    + [np.array(cols[c]) for c in
                                       self.child_attrs]) if total else \
            np.zeros((0, 3 + len(self.child_attrs)), int)
        r1_arr = np.column_stack([np.arange(n1)] + [s1[a2] for a2 in
                                                    self.r1_attrs])
        return r1_arr, child_arr, r2_out, total


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n1, n2 = 800, 600
    root1 = np.column_stack([np.arange(n1), rng.binomial(1, .5, n1),
                             rng.integers(0, 3, n1)])
    root2 = np.column_stack([np.arange(n2), rng.binomial(1, .5, n2)])
    nc = 6000
    fk1 = rng.integers(0, n1, nc)
    fk2 = rng.integers(0, n2, nc)
    g = root1[fk1, 1]
    mv = root2[fk2, 1]
    # planted cross-parent interaction: rating depends on g XOR mv
    c = (rng.random(nc) < np.where(g ^ mv == 1, 0.8, 0.3)).astype(int)
    child = np.column_stack([np.arange(nc), fk1, fk2, c])

    for eps in (3.2, 0.8):
        m = ChainNPMV(['gen', 'age'], {'gen': 2, 'age': 3},
                      ['cat'], {'cat': 2}, ['rate'], {'rate': 2},
                      xp_pairs=[('gen', 'cat')], tau=12, seed=0)
        m.compute_marginals(root1, root2, child, eps, 1e-4)
        r1s, ch, r2o, tot = m.synthesize()
        # cross-parent signal: Delta = Pr[rate=1 | g xor mv =1] - Pr[. | =0]
        def delta(gg, mm, rr):
            z = (gg ^ mm) == 1
            return np.mean(rr[z] == 1) - np.mean(rr[~z] == 1)
        real_d = delta(root1[fk1, 1], root2[fk2, 1], c)
        syn_d = delta(r1s[ch[:, 1], 1], np.array(r2o['cat']), ch[:, 3])
        print(f"eps={eps}: Q={m.Q} real_delta={real_d:.3f} "
              f"syn_delta={syn_d:.3f}")
        assert syn_d > 0.5 * real_d - 0.05, (eps, syn_d, real_d)

    print("vshape self-test OK")
