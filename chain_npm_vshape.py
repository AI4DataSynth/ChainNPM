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


class ChainNPMV:
    def __init__(self, r1_attrs, r1_dom, r2_attrs, r2_dom,
                 child_attrs, child_dom, xp_pairs=(), sz_attrs=None,
                 tau=12, seed=0):
        self.r1_attrs, self.r1_dom = list(r1_attrs), dict(r1_dom)
        self.r2_attrs, self.r2_dom = list(r2_attrs), dict(r2_dom)
        self.child_attrs, self.cdom = list(child_attrs), dict(child_dom)
        self.xp_pairs = list(xp_pairs) or [(a, b) for a in self.r1_attrs
                                           for b in self.r2_attrs]
        self.sz_attrs = list(sz_attrs) if sz_attrs else list(r1_attrs)
        self.tau = tau
        self.seed = seed

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
        g2 = analytic_gaussian_gamma(epsilon, delta) ** 2 / len(qnames)
        sigma = lambda s: s * np.sqrt(2.0 / g2)

        shapes = {
            'p1': lambda a: (self.r1_dom[a],),
            'p2': lambda b: (self.r2_dom[b],),
            'xp': lambda ab: (self.r1_dom[ab[0]], self.r2_dom[ab[1]]),
            'c1': lambda c: (self.cdom[c],),
            'p1c': lambda ac: (self.r1_dom[ac[0]], self.cdom[ac[1]]),
            'p2c': lambda bc: (self.r2_dom[bc[0]], self.cdom[bc[1]]),
            'xpc': lambda abc: (self.r1_dom[abc[0]], self.r2_dom[abc[1]],
                                self.cdom[abc[2]]),
            'sz': lambda a: (self.r1_dom[a], self.tau + 1),
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
        w = 1.0 / np.maximum(sizes[gid], 1)

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
        for a in self.sz_attrs:
            np.add.at(acc[f'SZ_{a}'], (np.clip(r1v[a][gpk], 0,
                      self.r1_dom[a] - 1), sizes), 1.0)

        noisy = {n: np.maximum(0.0, acc[n] + rng.normal(0, sigma(1.0),
                               acc[n].shape)) for n in qnames}
        noisy['MEAN_SIZE'] = max(0.5, float(sizes_full.mean())
                                 + rng.normal(0, sigma(1.0)))
        self.noisy, self.sigma = noisy, float(sigma(1.0))
        self.Q = len(qnames)
        return noisy

    def synthesize(self, n_root1=None):
        rng = np.random.default_rng(self.seed + 12345)
        nz = self.noisy

        def dist(x):
            x = np.asarray(x, float)
            s = x.sum()
            return x / s if s > 0 else np.full(x.shape, 1.0 / x.size)

        n1 = n_root1 or int(round(nz['P1_' + self.r1_attrs[0]].sum()))
        s1 = {}
        for a in self.r1_attrs:
            s1[a] = rng.choice(self.r1_dom[a], size=n1,
                               p=dist(nz[f'P1_{a}']))

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

        sz_marg = dist(sum(nz[f'SZ_{a}'].sum(0) for a in self.sz_attrs))
        thr = 5.0 * getattr(self, 'sigma', 0.0)
        sz_lift = {}
        for a in self.sz_attrs:
            L = lift(nz[f'SZ_{a}'], m1[a], sz_marg)
            wgt = np.clip(nz[f'SZ_{a}'] / thr, 0, 1) if thr > 0 else 1.0
            sz_lift[a] = 1.0 + (L - 1.0) * wgt

        a, b = self.xp_pairs[0]
        cols = {c: [] for c in self.child_attrs}
        r2_out = {b2: [] for b2 in self.r2_attrs}
        fks = []
        for i in range(n1):
            p = m2[b].copy() * xp_lift[(a, b)][s1[a][i]]
            for (x, y) in self.xp_pairs[1:]:
                if x == a:
                    p *= xp_lift[(x, y)][s1[x][i]]
            p = p / p.sum()
            n_part = 1
            sp = sz_marg.copy()
            for x in self.sz_attrs:
                sp *= sz_lift[x][s1[x][i]]
            sp = sp / sp.sum()
            size = int(rng.choice(self.tau + 1, p=sp))
            size = max(1, min(size, self.tau))
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
                    cols[c].append(int(rng.choice(self.cdom[c],
                                                  p=q / q.sum())))
                for b2 in self.r2_attrs:
                    r2_out[b2].append(r2vals[b2])
                fks.append(i)
        total = len(fks)
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
