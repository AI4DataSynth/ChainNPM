"""Unified evaluator for P4 new datasets (chain + V-shape schemas).

Called by the per-dataset drivers.  Responsibilities:

  1. Random conjunctive count queries, generated ONCE per schema shape from
     prereg.QUERY_SEED (150 per hop class, pre-registered), shared by all
     methods so comparisons are paired.
  2. Median relative error per hop class, all queries and large-count-only
     (real answer >= prereg.LARGE_COUNT).
  3. Cross-hop (chain: Delta = Pr[A|X1=1] - Pr[A|X1=0]) and cross-parent
     (V: interaction contrast E[A|1,1]-E[A|1,0]-E[A|0,1]+E[A|0,0]) effect
     metrics, reported for real and synthetic data.
  4. Group-size distribution TV.

Canonical table layouts expected by this module
------------------------------------------------
chain  : top = [PK, attrs...]
         mid = [PK, FK->top, attrs...]
         bot = [PK, FK->mid, attrs...]
vshape : root1 = [PK, attrs...]
         root2 = [PK, attrs...]
         child = [PK, fk1->root1, fk2->root2, attrs...]
         (plus optional 'child_r2_attrs' (nc, n_r2_attrs): per-child-row
          root2 attribute values, used when the method does not emit root2
          entity identity -- ChainNPMV samples root2 attributes per child
          row and sets child.fk2 = child.fk1.)

Hop classes (pre-registered query families)
-------------------------------------------
chain  : 0hop_top, 0hop_bot            (single table)
         1hop_topmid, 1hop_midbot      (parent-child conjunction)
         2hop_topbot                    (cross grandparent-leaf)
vshape : 0hop_r1, 0hop_r2, 0hop_c      (single table)
         1hop_r1c, 1hop_r2c             (parent-child conjunction)
         cross_r1r2                      (cross-parent conjunction)
"""

import json
import os

import numpy as np

import prereg

# ======================================================================
# binarization (fit on GT, apply to syn) -- mirrors plant_audit.binarize
# but with a fit/apply split so synthetic data is transformed identically
# ======================================================================


def fit_binarize(values, seed=0):
    """Fit a deterministic binarization; returns params dict.

    Mirrors plant_audit.binarize WITHOUT the rebalance flips (anchors must
    remain pure functions of attribute values so they can be re-applied to
    synthetic data)."""
    v = np.asarray(values)
    uniq = np.unique(v)
    if set(uniq.tolist()) <= {0, 1}:
        return dict(kind='binary')
    if np.issubdtype(v.dtype, np.number):
        return dict(kind='numeric', cut=float(np.median(v)))
    # NB: mirrors plant_audit.binarize with lo=0, hi=1 (no rebalance flips)
    # so anchors stay pure functions of attribute values and can be re-applied
    # to synthetic data; drivers must plant with balance=(0.0, 1.0).
    cats, counts = np.unique(v, return_counts=True)
    order = np.argsort(-counts)
    sel, acc = [], 0
    for i in order:
        sel.append(cats[i])
        acc += counts[i]
        share = acc / counts.sum()
        if abs(share - 0.5) <= 0.05 or share >= 1.0:
            break
    return dict(kind='categorical', subset=sorted(set(sel)))


def apply_binarize(values, params):
    v = np.asarray(values)
    if params['kind'] == 'binary':
        return (v == 1).astype(int)
    if params['kind'] == 'numeric':
        return (v > params['cut']).astype(int)
    sset = set(params['subset'])
    return np.array([1 if x in sset else 0 for x in v], dtype=int)


# ======================================================================
# query generation
# ======================================================================


def _subset(rng, dom):
    k = int(rng.integers(1, min(4, dom)))
    return sorted(rng.choice(dom, size=k, replace=False).astype(int).tolist())


def _pred(rng, role, attr, domains):
    return (role, str(attr), _subset(rng, domains[(role, str(attr))]))


def _multi(rng, role, attrlist, domains, k):
    picks = rng.choice(attrlist, size=int(rng.integers(1, min(k + 1,
                     len(attrlist) + 1))), replace=False)
    return [_pred(rng, role, a, domains) for a in picks]


def _single(rng, role, attrlist, domains):
    return [_pred(rng, role, rng.choice(attrlist), domains)]


def make_queries_chain(rng, attrs, domains, n_per_class):
    """attrs: {'top','mid','bot'} attr lists; domains {(role,attr): size}.

    One query drawn per iteration for each of the five pre-registered hop
    classes (150 draws each by default)."""
    qs = []
    top, mid, bot = attrs['top'], attrs['mid'], attrs['bot']
    for _ in range(n_per_class):
        qs.append({'cls': '0hop_top', 'count_role': 'top',
                   'preds': _single(rng, 'top', top, domains)})
        qs.append({'cls': '0hop_bot', 'count_role': 'bot',
                   'preds': _single(rng, 'bot', bot, domains)})
        qs.append({'cls': '1hop_topmid', 'count_role': 'mid',
                   'preds': [_pred(rng, 'top', rng.choice(top), domains),
                             _pred(rng, 'mid', rng.choice(mid), domains)]})
        qs.append({'cls': '1hop_midbot', 'count_role': 'bot',
                   'preds': [_pred(rng, 'mid', rng.choice(mid), domains),
                             _pred(rng, 'bot', rng.choice(bot), domains)]})
        qs.append({'cls': '2hop_topbot', 'count_role': 'bot',
                   'preds': [_pred(rng, 'top', rng.choice(top), domains),
                             _pred(rng, 'bot', rng.choice(bot), domains)]})
    return qs


def make_queries_vshape(rng, attrs, domains, n_per_class):
    """attrs: {'r1','r2','c'} attr lists; domains {(role,attr): size}."""
    qs = []
    r1, r2, c = attrs['r1'], attrs['r2'], attrs['c']
    for _ in range(n_per_class):
        qs.append({'cls': '0hop_r1', 'count_role': 'r1',
                   'preds': _single(rng, 'r1', r1, domains)})
        qs.append({'cls': '0hop_r2', 'count_role': 'r2',
                   'preds': _single(rng, 'r2', r2, domains)})
        qs.append({'cls': '0hop_c', 'count_role': 'c',
                   'preds': _single(rng, 'c', c, domains)})
        qs.append({'cls': '1hop_r1c', 'count_role': 'c',
                   'preds': [_pred(rng, 'r1', rng.choice(r1), domains),
                             _pred(rng, 'c', rng.choice(c), domains)]})
        qs.append({'cls': '1hop_r2c', 'count_role': 'c',
                   'preds': [_pred(rng, 'r2', rng.choice(r2), domains),
                             _pred(rng, 'c', rng.choice(c), domains)]})
        qs.append({'cls': 'cross_r1r2', 'count_role': 'c',
                   'preds': [_pred(rng, 'r1', rng.choice(r1), domains),
                             _pred(rng, 'r2', rng.choice(r2), domains)]})
    return qs


# ======================================================================
# views
# ======================================================================


def _join(parent_attrs, fk, n_parent):
    """parent_attrs (np, n_parent x k); fk (n,). Returns (joined, keep)."""
    fk = np.asarray(fk, dtype=int)
    keep = (fk >= 0) & (fk < n_parent)
    idx = np.where(keep, fk, 0)
    return parent_attrs[idx], keep


def build_chain_views(tables, attrs):
    """tables: {'top','mid','bot'} canonical arrays; attrs per role."""
    top, mid, bot = tables['top'], tables['mid'], tables['bot']
    ta = top[:, 1:1 + len(attrs['top'])].astype(int)
    ma = mid[:, 2:2 + len(attrs['mid'])].astype(int)
    ba = bot[:, 2:2 + len(attrs['bot'])].astype(int)
    mid_fk = mid[:, 1].astype(int)
    bot_fk = bot[:, 1].astype(int)

    views = {}
    views['top'] = {'data': ta, 'orphans': 0,
                    'col': {('top', a): i for i, a in enumerate(attrs['top'])}}
    top_at_mid, keep_m = _join(ta, mid_fk, top.shape[0])
    views['mid'] = {'data': np.hstack([top_at_mid, ma]),
                    'orphans': int((~keep_m).sum())}
    # two-step join bot -> mid -> top (guard dangling mid FKs)
    keep_midfk = (bot_fk >= 0) & (bot_fk < mid.shape[0])
    mid_of_bot = np.where(keep_midfk, bot_fk, 0)
    topfk_of_bot = mid[:, 1].astype(int)[mid_of_bot]
    top_at_bot, keep_b2 = _join(ta, topfk_of_bot, top.shape[0])
    mid_at_bot, _ = _join(ma, mid_of_bot, mid.shape[0])
    views['bot'] = {'data': np.hstack([top_at_bot, mid_at_bot, ba]),
                    'orphans': int((~(keep_b2 & keep_midfk)).sum())}
    # column maps
    nt, nm = len(attrs['top']), len(attrs['mid'])
    mid_view_cols = {}
    for i, a in enumerate(attrs['top']):
        mid_view_cols[('top', a)] = i
    for i, a in enumerate(attrs['mid']):
        mid_view_cols[('mid', a)] = nt + i
    bot_view_cols = {}
    for i, a in enumerate(attrs['top']):
        bot_view_cols[('top', a)] = i
    for i, a in enumerate(attrs['mid']):
        bot_view_cols[('mid', a)] = nt + i
    for i, a in enumerate(attrs['bot']):
        bot_view_cols[('bot', a)] = nt + nm + i
    views['mid']['col'] = mid_view_cols
    views['bot']['col'] = bot_view_cols
    return views


def build_vshape_views(tables, attrs, child_r2_attrs=None):
    root1, root2, child = tables['r1'], tables['r2'], tables['c']
    r1a = root1[:, 1:1 + len(attrs['r1'])].astype(int)
    r2a = root2[:, 1:1 + len(attrs['r2'])].astype(int)
    ca = child[:, 3:3 + len(attrs['c'])].astype(int)
    fk1 = child[:, 1].astype(int)
    fk2 = child[:, 2].astype(int)

    views = {}
    views['r1'] = {'data': r1a, 'orphans': 0,
                   'col': {('r1', a): i for i, a in enumerate(attrs['r1'])}}
    views['r2'] = {'data': r2a, 'orphans': 0,
                   'col': {('r2', a): i for i, a in enumerate(attrs['r2'])}}
    r1_at_c, keep1 = _join(r1a, fk1, root1.shape[0])
    if child_r2_attrs is not None:
        r2_at_c = np.asarray(child_r2_attrs, dtype=int)
        keep2 = np.ones(child.shape[0], dtype=bool)
    else:
        r2_at_c, keep2 = _join(r2a, fk2, root2.shape[0])
    ccol = {}
    off = 0
    for i, a in enumerate(attrs['r1']):
        ccol[('r1', a)] = off + i
    off += len(attrs['r1'])
    for i, a in enumerate(attrs['r2']):
        ccol[('r2', a)] = off + i
    off += len(attrs['r2'])
    for i, a in enumerate(attrs['c']):
        ccol[('c', a)] = off + i
    views['c'] = {'data': np.hstack([r1_at_c, r2_at_c, ca]),
                  'orphans': int((~(keep1 & keep2)).sum()), 'col': ccol}
    return views


def answer_query(q, views):
    v = views[q['count_role']]
    data, col = v['data'], v['col']
    m = np.ones(data.shape[0], dtype=bool)
    for role, attr, vals in q['preds']:
        key = (role, attr)
        if key not in col:
            return None
        m &= np.isin(data[:, col[key]], vals)
    return int(m.sum())


def query_re(gt_views, syn_views, queries, large_count):
    re_all, re_large = {}, {}
    for q in queries:
        a_real = answer_query(q, gt_views)
        a_syn = answer_query(q, syn_views)
        if a_real is None or a_syn is None:
            continue
        re = abs(a_syn - a_real) / max(a_real, 1.0)
        re_all.setdefault(q['cls'], []).append(re)
        if a_real >= large_count:
            re_large.setdefault(q['cls'], []).append(re)
    out_all = {c: float(np.median(v)) for c, v in re_all.items() if v}
    out_large = {c: float(np.median(v)) for c, v in re_large.items() if v}
    n_all = {c: len(v) for c, v in re_all.items()}
    n_large = {c: len(v) for c, v in re_large.items()}
    return {'re_median': out_all, 're_median_large': out_large,
            'n_queries': n_all, 'n_queries_large': n_large}


# ======================================================================
# effect metrics
# ======================================================================


def chain_effect(view, col, x1, a, x1_params, a_params):
    """Delta = Pr[A=1 | X1=1] - Pr[A=1 | X1=0] on the bot view."""
    data = view['data']
    xb = apply_binarize(data[:, col[x1]], x1_params)
    ab = apply_binarize(data[:, col[a]], a_params)
    m1 = xb == 1
    m0 = xb == 0
    p1 = ab[m1].mean() if m1.any() else 0.0
    p0 = ab[m0].mean() if m0.any() else 0.0
    return {'delta': float(p1 - p0), 'pr_a_x1_1': float(p1),
            'pr_a_x1_0': float(p0), 'share_x1': float(xb.mean())}


def vshape_effect(view, col, x1, x2, a, p1_, p2_, pa, u=None, pu=None):
    """Interaction contrast over (x1, x2) on the child view plus, when u is
    given, xor rule fidelity phi = mean(a == u ^ x1 ^ x2)."""
    data = view['data']
    b1 = apply_binarize(data[:, col[x1]], p1_)
    b2 = apply_binarize(data[:, col[x2]], p2_)
    ba = apply_binarize(data[:, col[a]], pa)
    means = {}
    for i in (0, 1):
        for j in (0, 1):
            m = (b1 == i) & (b2 == j)
            means[(i, j)] = float(ba[m].mean()) if m.any() else 0.0
    contrast = (means[(1, 1)] - means[(1, 0)] - means[(0, 1)] + means[(0, 0)])
    out = {'contrast': float(contrast),
           'cell_means': {'%d%d' % k: v for k, v in means.items()},
           'share_x1': float(b1.mean()), 'share_x2': float(b2.mean())}
    if u is not None:
        bu = apply_binarize(data[:, col[u]], pu)
        out['phi_xor'] = float((ba == (bu ^ b1 ^ b2)).mean())
        out['share_u'] = float(bu.mean())
    return out


# ======================================================================
# size TV
# ======================================================================


def size_tv(sizes_real, sizes_syn, cap):
    """TV between group-size distributions, bins 1..cap plus overflow."""
    nb = cap + 1  # bins: 1..cap, cap+1 = ">cap"
    r = np.bincount(np.clip(sizes_real, 1, cap + 1), minlength=nb + 2)[1:nb + 1]
    s = np.bincount(np.clip(sizes_syn, 1, cap + 1), minlength=nb + 2)[1:nb + 1]
    r = r / max(r.sum(), 1)
    s = s / max(s.sum(), 1)
    return float(0.5 * np.abs(r - s).sum())


def group_sizes(fk, n_parent=None):
    fk = np.asarray(fk, dtype=int)
    u, c = np.unique(fk, return_counts=True)
    if n_parent is not None:
        full = np.zeros(n_parent, dtype=int)
        full[u] = c
        return full
    return c


# ======================================================================
# one-shot convenience for drivers (not strictly required)
# ======================================================================


def load_int_csv(path):
    import pandas as pd
    return pd.read_csv(path).values.astype(int)


if __name__ == '__main__':
    # tiny self-test: chain + vshape views and queries on toy data
    rng = np.random.default_rng(0)
    nt, nm, nb = 20, 100, 600
    top = np.column_stack([np.arange(nt), rng.integers(0, 3, nt)])
    mid = np.column_stack([np.arange(nm), rng.integers(0, nt, nm),
                           rng.integers(0, 2, nm)])
    bot = np.column_stack([np.arange(nb), rng.integers(0, nm, nb),
                           rng.integers(0, 4, nb)])
    attrs = {'top': ['T'], 'mid': ['M'], 'bot': ['B']}
    dom = {('top', 'T'): 3, ('mid', 'M'): 2, ('bot', 'B'): 4}
    views = build_chain_views({'top': top, 'mid': mid, 'bot': bot}, attrs)
    q = {'cls': '2hop_topbot', 'count_role': 'bot',
         'preds': [('top', 'T', [0, 1]), ('bot', 'B', [2])]}
    a = answer_query(q, views)
    # brute force
    bf = 0
    for r in bot:
        mrow = mid[r[1]]
        trow = top[mrow[1]]
        if trow[1] in (0, 1) and r[2] == 2:
            bf += 1
    assert a == bf, (a, bf)
    print('chain view answer OK', a)

    n1, n2, nc = 30, 25, 500
    r1 = np.column_stack([np.arange(n1), rng.integers(0, 2, n1)])
    r2 = np.column_stack([np.arange(n2), rng.integers(0, 3, n2)])
    ch = np.column_stack([np.arange(nc), rng.integers(0, n1, nc),
                          rng.integers(0, n2, nc), rng.integers(0, 2, nc)])
    va = {'r1': ['X'], 'r2': ['Y'], 'c': ['A']}
    vv = build_vshape_views({'r1': r1, 'r2': r2, 'c': ch}, va)
    q2 = {'cls': 'cross_r1r2', 'count_role': 'c',
          'preds': [('r1', 'X', [1]), ('r2', 'Y', [0, 2])]}
    a2 = answer_query(q2, vv)
    bf2 = sum(1 for r in ch if r1[r[1]][1] == 1 and r2[r[2]][1] in (0, 2))
    assert a2 == bf2, (a2, bf2)
    print('vshape view answer OK', a2)
    tv = size_tv(rng.integers(1, 10, 500), rng.integers(1, 10, 480), 8)
    print('size_tv', round(tv, 4))
    print('eval_newdata self-test OK')
