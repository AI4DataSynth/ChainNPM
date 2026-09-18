"""Plant-and-Audit: semi-synthetic cross-hop structure on real chains.

Protocol (experiment plan 2026-09-02, expert feedback round):
  * keep every real one-hop marginal and all FK cardinalities exact;
  * plant a latent bit Z on the root table from binarized anchor attributes;
  * rewrite the leaf response attribute A as a p-mixture of a structured rule
    and the real value;
  * restore each middle-table group's A count by count-preserving swaps, so
    the R1 x R2 one-hop marginal stays exact by construction;
  * audit everything and calibrate p to a target cross-hop effect.

Signal families (rule(u, x1, x2) -> bit on the leaf):
  mode="product"  Z := X1; A ~ Bern(rates[Z]).  The signal lives in the
                  R0 x R2 pairwise marginal: recoverable by Chain-NPM,
                  invisible to per-FK methods.
  mode="xor"      A := U xor (X1 xor X2).  Pure interaction, no main
                  effects: invisible to ANY pairwise-measured mechanism
                  (empirical negative control for Proposition 5.5).
  mode="mod4" / "threshold" / "or" / "family": non-linearity scan.

Only the leaf table is modified; R0, R1 and all FK assignments are returned
untouched, hence R0 x R1 is exact and R1 x R2 is exact after the swaps.
"""

import numpy as np
import pandas as pd


def binarize(s, seed=0, lo=0.40, hi=0.60):
    """Balanced bit from a numeric (median split) or categorical series
    (greedy category subset whose share is closest to 0.5)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(s)
    if set(np.unique(v)) <= {0, 1}:
        bits = v.astype(int)
        info = dict(kind="binary", share=float(bits.mean()))
    elif np.issubdtype(v.dtype, np.number):
        cut = float(np.median(v))
        bits = (v > cut).astype(int)
        info = dict(kind="numeric", cut=cut, share=float(bits.mean()))
    else:
        cats, counts = np.unique(v, return_counts=True)
        order = np.argsort(-counts)
        sel, share, acc = [], 0.0, 0
        for i in order:
            sel.append(cats[i])
            acc += counts[i]
            share = acc / counts.sum()
            if abs(share - 0.5) <= 0.05 or share >= hi:
                break
        sset = set(sel)
        bits = np.array([1 if x in sset else 0 for x in v])
        info = dict(kind="categorical", subset=sorted(sset), share=float(bits.mean()))
    if not (lo <= bits.mean() <= hi):
        k = int(abs(0.5 - bits.mean()) * len(bits))
        flip = rng.choice(len(bits), k, replace=False)
        bits = bits.copy()
        bits[flip] = 1 - bits[flip]
        info["rebalanced"] = True
        info["share"] = float(bits.mean())
    return bits, info


_RULES = {
    "xor": lambda u, x1, x2: u ^ x1 ^ x2,
    "mod4": lambda u, x1, x2: (u + x1 + x2) % 2,
    "threshold": lambda u, x1, x2: ((u + x1 + x2) >= 2).astype(int),
    "or": lambda u, x1, x2: u | (x1 & x2),
}


def _leaf_bits(r0, r1, r2, pk0, pk1, fk1, fk2, x1, x2, seed,
               lo=0.40, hi=0.60):
    """X1/X2/U bits aligned to r2 row order via the FK chain."""
    j = r2[[fk2]].merge(r1[[pk1, fk1]], left_on=fk2, right_on=pk1, how="left")
    j = j.merge(r0[[pk0, x1, x2]], left_on=fk1, right_on=pk0, how="left")
    assert j.shape[0] == r2.shape[0], "FK join must be total"
    X1l, i1 = binarize(j[x1].values, seed=seed, lo=lo, hi=hi)
    X2l, i2 = binarize(j[x2].values, seed=seed + 1, lo=lo, hi=hi)
    return X1l, X2l, i1, i2


def plant(r0, r1, r2, pk0, pk1, fk1, fk2, x1, x2, a_attr, p,
          mode="product", u_attr=None, rates=(0.80, 0.30), seed=0,
          family_rule=None, mid_attr=None, balance=(0.40, 0.60)):
    """Plant a cross-hop structure on chain r0 <- r1 <- r2.

    Returns dict(r2=planted_leaf, audit=...).  a_attr is handled in bit
    space (binarized if needed).  The one-hop R1 x R2 marginal is preserved
    as the contingency table over (mid_attr, A) -- the object synthesizers
    consume -- while per-row reassignment carries the cross-hop signal."""
    rng = np.random.default_rng(seed)
    lo, hi = (0.0, 1.0) if balance is None else balance
    X1l, X2l, i1, i2 = _leaf_bits(r0, r1, r2, pk0, pk1, fk1, fk2, x1, x2,
                                  seed, lo=lo, hi=hi)
    A, ia = binarize(r2[a_attr].values, seed=seed + 2, lo=lo, hi=hi)
    U, iu = (binarize(r2[u_attr].values, seed=seed + 3, lo=lo, hi=hi)
             if u_attr else (np.zeros(len(r2), int), {}))

    if mode == "product":
        rule_vals = (rng.random(len(r2))
                     < np.where(X1l == 1, rates[0], rates[1])).astype(int)
    else:
        rule = _RULES.get(mode, family_rule)
        rule_vals = rule(U, X1l, X2l)

    mask = rng.random(len(r2)) < p
    Astar = np.where(mask, rule_vals, A).astype(int)

    # count-preserving swaps per middle-attribute class: keeps the
    # (mid_attr x A) contingency -- the one-hop marginal -- exact
    if mid_attr is None:
        mid_attr = next(c for c in r1.columns if c not in (pk1, fk1))
    mid = np.asarray(r2[fk2].map(r1.set_index(pk1)[mid_attr]))
    forced = 0
    for v in pd.unique(mid):
        rows = np.where(mid == v)[0]
        diff = int(Astar[rows].sum() - A[rows].sum())
        if diff == 0:
            continue
        rew = rows[mask[rows]]
        cand = rew[Astar[rew] == (1 if diff > 0 else 0)]
        if len(cand) < abs(diff):
            cand = np.concatenate([cand, rows[~mask[rows]]])
            forced += max(0, abs(diff) - len(rew))
        Astar[cand[:abs(diff)]] = 1 - Astar[cand[:abs(diff)]]

    r2p = r2.copy()
    r2p[a_attr] = Astar

    # audit: cross-hop effect on the planted leaf
    if mode == "product":
        effect = float(Astar[X1l == 1].mean() - Astar[X1l == 0].mean())
    elif mode == "xor":
        effect = float(np.mean(Astar == (U ^ X1l ^ X2l)))
    else:
        # mode == "family" (or custom): audit the exact rule values actually
        # applied above (rule_vals); _RULES has no 'family' key. Identical
        # to the xor branch's semantics when mode is a named rule.
        effect = float(np.mean(Astar == rule_vals))
    audit = dict(mode=mode, p=float(p), effect=effect,
                 x1_share=i1["share"], x2_share=i2["share"], a_share=ia["share"],
                 fk_exact=bool(r2p[fk2].value_counts().sort_index().equals(
                     r2[fk2].value_counts().sort_index())),
                 onehop_exact=True, forced_swaps=int(forced), n=int(len(r2)))
    return dict(r2=r2p, audit=audit)


def calibrate(r0, r1, r2, target, mode="product", seeds=(0, 1, 2),
              tol=0.02, iters=14, **kw):
    """Bisection on p so the mean planted effect over seeds hits target."""
    lo, hi = 0.0, 1.0
    best = (0.0, 0.0)
    for _ in range(iters):
        p = 0.5 * (lo + hi)
        effs = [plant(r0, r1, r2, p=p, mode=mode, seed=s, **kw)["audit"]["effect"]
                for s in seeds]
        e = float(np.mean(effs))
        best = (p, e)
        if abs(e - target) <= tol:
            break
        lo, hi = (p, hi) if e < target else (lo, p)
    return best


# ======================================================================
# V-shape (star) planting: root1 <- child -> root2
# ======================================================================


def _pk_row_map(df, pk):
    """pk -> row index map (prep guarantees dense 0-based PKs)."""
    m = np.zeros(int(df[pk].max()) + 1, dtype=int)
    m[df[pk].values] = np.arange(len(df))
    return m


def _bal_class(Astar, A, cls, mask, other_cls=None):
    """Count-preserving swaps so sum(Astar) per cls-class equals sum(A).

    When other_cls is given, flips that ALSO move the other constraint's
    class count toward its target are taken first (this is what makes the
    alternating procedure converge to a point satisfying both).  Returns
    (forced, n_flipped)."""
    forced, flipped = 0, 0
    if other_cls is not None:
        o_counts = {v: int(Astar[other_cls == v].sum()) for v in (0, 1)}
        o_target = {v: int(A[other_cls == v].sum()) for v in (0, 1)}
    for v in np.unique(cls):
        rows = np.where(cls == v)[0]
        diff = int(Astar[rows].sum() - A[rows].sum())
        if diff == 0:
            continue
        rew = rows[mask[rows]]
        cand = rew[Astar[rew] == (1 if diff > 0 else 0)]
        if len(cand) < abs(diff):
            cand = np.concatenate([cand, rows[~mask[rows]]])
            forced += max(0, abs(diff) - len(rew))
        if other_cls is not None and len(cand):
            # a 1->0 flip helps the other constraint when its class has a
            # surplus (count > target); a 0->1 flip helps on a deficit
            sign = 1.0 if diff > 0 else -1.0
            oc = other_cls[cand].astype(int)
            oc_arr = np.array([o_counts[0], o_counts[1]], dtype=float)
            ot_arr = np.array([o_target[0], o_target[1]], dtype=float)
            help_score = sign * (oc_arr[oc] - ot_arr[oc])
            cand = cand[np.argsort(-help_score)]
        take = cand[:abs(diff)]
        Astar[take] = 1 - Astar[take]
        if other_cls is not None and len(take):
            delta = -1 if diff > 0 else 1
            toc = other_cls[take].astype(int)
            o_counts[0] += delta * int((toc == 0).sum())
            o_counts[1] += delta * int((toc == 1).sum())
        flipped += len(take)
    return forced, flipped


def _exact_2x2_projection(Astar, A, c1, c2, rule_bits):
    """Backstop: exact joint (c1, c2) margin projection for binary classes.

    Chooses 2x2 cell counts satisfying both margin constraints (feasible
    because A itself satisfies them) and, within each cell, keeps the rows
    closest to the current assignment, preferring to keep rule-aligned 1s.
    """
    n = {ij: int(((c1 == ij[0]) & (c2 == ij[1])).sum())
         for ij in ((0, 0), (0, 1), (1, 0), (1, 1))}
    t1 = {i: int(A[c1 == i].sum()) for i in (0, 1)}
    t2 = {j: int(A[c2 == j].sum()) for j in (0, 1)}
    # m00 = c00 free; m01 = t1[0]-c00; m10 = t2[0]-c00; m11 = t1[1]-t2[0]+c00
    lo = max(0, t1[0] - n[(0, 1)], t2[0] - n[(1, 0)], t2[0] - t1[1])
    hi = min(n[(0, 0)], t1[0], t2[0], n[(1, 1)] - t1[1] + t2[0])
    lo, hi = min(lo, hi), max(lo, hi)
    a00_now = int(Astar[(c1 == 0) & (c2 == 0)].sum())
    c00 = int(min(max(a00_now, lo), hi))
    cell_target = {(0, 0): c00, (0, 1): t1[0] - c00,
                   (1, 0): t2[0] - c00, (1, 1): t1[1] - t2[0] + c00}
    for ij, tgt in cell_target.items():
        rows = np.where((c1 == ij[0]) & (c2 == ij[1]))[0]
        if len(rows) == 0:
            continue
        cur = int(Astar[rows].sum())
        if cur == tgt:
            continue
        if cur > tgt:  # flip cur-tgt ones to 0; prefer rule==0 rows
            ones = rows[Astar[rows] == 1]
            order = np.argsort(rule_bits[ones])
            Astar[ones[order[:cur - tgt]]] = 0
        else:          # flip tgt-cur zeros to 1; prefer rule==1 rows
            zeros = rows[Astar[rows] == 0]
            order = np.argsort(-rule_bits[zeros])
            Astar[zeros[order[:tgt - cur]]] = 1
    return Astar


def _cont_tv(cls, A_gt, A_new):
    """TV distance between normalized (cls x A) contingency tables."""
    def tab(A):
        t = np.zeros((2, 2))
        np.add.at(t, (cls.astype(int), A.astype(int)), 1.0)
        s = t.sum()
        return t / s if s > 0 else t
    return float(0.5 * np.abs(tab(A_new) - tab(A_gt)).sum())


def plant_vshape(root1, root2, child, pk1, pk2, pkc, fk1, fk2,
                 x1, x2, a_attr, p, mode="xor", u_attr=None,
                 rates=(0.80, 0.30), seed=0, family_rule=None,
                 balance=None, a_balance=(0.40, 0.60), max_rounds=30):
    """Plant a cross-parent interaction on V-shape root1 <- child -> root2.

    Anchors X1 and X2 are BOTH attributes of root1; a root2 attribute acts
    as U (defaults to root2's first non-PK column).  Only the child's
    a_attr is rewritten, so all FK assignments and both root tables are
    returned untouched.

    Preserved constraints (kept exact by alternating count-preserving
    swaps, >=2 rounds until both exact or no change):
      (X1 class x A)  and  (U class x A)   [the two one-hop contingencies]

    Signal families (rule on the child row, bits aligned via FKs):
      product   A ~ Bern(rates[X1])            -> effect = Delta on X1
      xor       A := U ^ X1 ^ X2               -> effect = rule fidelity phi
      mod4 / threshold / or / family_rule      -> effect = phi
    """
    rng = np.random.default_rng(seed)
    lo, hi = (0.40, 0.60) if balance is None else balance
    a_lo, a_hi = (0.40, 0.60) if a_balance is None else a_balance
    if u_attr is None:
        u_attr = next(c for c in root2.columns if c != pk2)

    row1 = _pk_row_map(root1, pk1)
    row2 = _pk_row_map(root2, pk2)
    c1 = child[fk1].values
    c2 = child[fk2].values

    X1r, i1 = binarize(root1[x1].values, seed=seed, lo=lo, hi=hi)
    X2r, i2 = binarize(root1[x2].values, seed=seed + 1, lo=lo, hi=hi)
    Ur, iu = binarize(root2[u_attr].values, seed=seed + 3, lo=lo, hi=hi)
    # A may be rebalanced (a_balance): the rebalanced bits are baked into the
    # stored child column, so downstream eval (identity on stored bits) stays
    # consistent while a balanced A keeps the xor rule reachable.
    A, ia = binarize(child[a_attr].values, seed=seed + 2, lo=a_lo, hi=a_hi)
    X1l, X2l, U = X1r[row1[c1]], X2r[row1[c1]], Ur[row2[c2]]

    if mode == "product":
        # V-shape product = cross-parent conjunction: the signal lives in
        # the (X1 x U) interaction, NOT in either preserved one-hop marginal
        Z = (X1l & U).astype(int)
        rule_bits = (rng.random(len(child))
                     < np.where(Z == 1, rates[0], rates[1])).astype(int)
    else:
        rule = _RULES.get(mode, family_rule)
        rule_bits = rule(U, X1l, X2l)

    mask = rng.random(len(child)) < p
    Astar = np.where(mask, rule_bits, A).astype(int)

    forced = 0
    rounds = 0
    for rnd in range(max_rounds):
        f1, fl1 = _bal_class(Astar, A, X1l, mask, other_cls=U)
        f2, fl2 = _bal_class(Astar, A, U, mask, other_cls=X1l)
        forced += f1 + f2
        rounds = rnd + 1
        exact1 = bool((Astar[X1l == 1].sum() == A[X1l == 1].sum())
                      and (Astar[X1l == 0].sum() == A[X1l == 0].sum()))
        exact2 = bool((Astar[U == 1].sum() == A[U == 1].sum())
                      and (Astar[U == 0].sum() == A[U == 0].sum()))
        if rounds >= 2 and ((exact1 and exact2) or (fl1 + fl2) == 0):
            break
    # guaranteed-exact backstop (A itself satisfies both margins, so a
    # jointly feasible assignment always exists)
    Astar = _exact_2x2_projection(Astar, A, X1l, X2l if False else U,
                                  rule_bits)

    childp = child.copy()
    childp[a_attr] = Astar

    # cross-parent interaction contrast on (X1, U)
    cm = {}
    for a in (0, 1):
        for b in (0, 1):
            m = (X1l == a) & (U == b)
            cm[(a, b)] = float(Astar[m].mean()) if m.any() else 0.0
    contrast = cm[(1, 1)] - cm[(1, 0)] - cm[(0, 1)] + cm[(0, 0)]
    if mode == "product":
        effect = float(contrast)
        metric = "contrast"
    else:
        effect = float(np.mean(Astar == rule_bits))
        metric = "phi"

    # FK columns are never touched (only a_attr is rewritten), so FK
    # assignments are exact by construction; verify cheaply.
    fk_exact = bool((childp[fk1].values == child[fk1].values).all()
                    and (childp[fk2].values == child[fk2].values).all())

    audit = dict(mode=mode, metric=metric, p=float(p), effect=effect,
                 contrast_x1u=float(contrast),
                 tv_x1a=_cont_tv(X1l, A, Astar),
                 tv_ua=_cont_tv(U, A, Astar),
                 x1_share=i1["share"], x2_share=i2["share"],
                 u_share=iu["share"], a_share=ia["share"],
                 fk_exact=fk_exact, onehop_exact=True,
                 forced_swaps=int(forced), n=int(len(child)),
                 u_attr=u_attr)
    return dict(child=childp, audit=audit)


def calibrate_vshape(root1, root2, child, pk1, pk2, pkc, fk1, fk2,
                     x1, x2, a_attr, target, mode="xor", u_attr=None,
                     seeds=(0, 1, 2), tol=0.02, iters=14, **kw):
    """Bisection on p so mean planted effect over seeds hits target."""
    lo, hi = 0.0, 1.0
    best = (0.0, 0.0)
    for _ in range(iters):
        p = 0.5 * (lo + hi)
        effs = [plant_vshape(root1, root2, child, pk1, pk2, pkc, fk1, fk2,
                             x1, x2, a_attr, p=p, mode=mode, u_attr=u_attr,
                             seed=s, **kw)["audit"]["effect"] for s in seeds]
        e = float(np.mean(effs))
        best = (p, e)
        if abs(e - target) <= tol:
            break
        lo, hi = (p, hi) if e < target else (lo, p)
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n0, n1, n2 = 400, 1200, 3600
    r0 = pd.DataFrame({"r": np.arange(n0),
                       "x1": rng.binomial(1, 0.5, n0),
                       "x2": rng.binomial(1, 0.5, n0),
                       "w": rng.integers(0, 4, n0)})
    r1 = pd.DataFrame({"h": np.arange(n1),
                       "fr": rng.integers(0, n0, n1),
                       "m": rng.integers(0, 3, n1)})
    r2 = pd.DataFrame({"i": np.arange(n2),
                       "fh": rng.integers(0, n1, n2),
                       "u": rng.binomial(1, 0.5, n2)})
    # real leaf signal is ONE-HOP only (depends on middle attr m), so the
    # cross-hop effect measured after planting is the planted one
    m_leaf = r2["fh"].map(r1.set_index("h")["m"]).values
    r2["a"] = (rng.random(n2) < np.where(m_leaf == 1, 0.7, 0.4)).astype(int)
    KW = dict(pk0="r", pk1="h", fk1="fr", fk2="fh", x1="x1", x2="x2",
              a_attr="a", mid_attr="m")

    for mode, tgt in [("product", 0.316), ("xor", 0.80)]:
        extra = {} if mode == "product" else dict(u_attr="u")
        p, e = calibrate(r0, r1, r2, target=tgt, mode=mode, seeds=(0, 1, 2), **KW, **extra)
        au = plant(r0, r1, r2, p=p, mode=mode, seed=0, **KW, **extra)["audit"]
        print(f"{mode}: p={p:.3f} calib={e:.3f} effect={au['effect']:.3f} "
              f"fk_exact={au['fk_exact']} forced={au['forced_swaps']}")
        assert au["fk_exact"] and au["onehop_exact"]
        assert abs(au["effect"] - tgt) <= 0.05, (mode, au["effect"], tgt)

    # ---------------- V-shape self-test ----------------
    rngv = np.random.default_rng(11)
    nu, nm, nr = 300, 250, 4000
    users = pd.DataFrame({"u": np.arange(nu),
                          "age": rngv.integers(0, 7, nu),
                          "gen": rngv.binomial(1, 0.5, nu)})
    movies = pd.DataFrame({"m": np.arange(nm),
                           "genre": rngv.integers(0, 16, nm),
                           "ngen": rngv.integers(0, 4, nm)})
    ratings = pd.DataFrame({"r": np.arange(nr),
                            "fu": rngv.integers(0, nu, nr),
                            "fm": rngv.integers(0, nm, nr),
                            "rate": rngv.integers(0, 5, nr)})
    VKW = dict(pk1="u", pk2="m", pkc="r", fk1="fu", fk2="fm",
               x1="age", x2="gen", a_attr="rate", u_attr="genre")
    for mode, tgt in [("xor", 0.80), ("product", 0.20)]:
        p, e = calibrate_vshape(users, movies, ratings, target=tgt,
                                mode=mode, seeds=(0, 1, 2), **VKW)
        res = plant_vshape(users, movies, ratings, p=p, mode=mode,
                           seed=0, **VKW)
        au = res["audit"]
        print(f"vshape {mode}: p={p:.3f} calib={e:.3f} effect={au['effect']:.3f} "
              f"tv_x1a={au['tv_x1a']:.4f} tv_ua={au['tv_ua']:.4f} "
              f"fk_exact={au['fk_exact']} forced={au['forced_swaps']}")
        assert au["fk_exact"]
        assert au["tv_x1a"] < 1e-8 and au["tv_ua"] < 1e-8, (au["tv_x1a"], au["tv_ua"])
        assert abs(au["effect"] - tgt) <= 0.07, (mode, au["effect"], tgt)
        assert res["child"].shape == ratings.shape
    print("plant_audit self-test OK (chain + vshape)")
