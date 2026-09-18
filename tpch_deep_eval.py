"""
Deep-chain multi-hop evaluator for TPC-H (discretized space).

GT:  a TPC-H-deep dir (make_deep_gt.py output) with
     customer.csv / orders.csv / lineitem.csv.
Syn: <prefix>_syn_c.csv / _syn_o.csv / _syn_l.csv in the same deep
     schema (extra columns such as O_ORDERDATE3 / L_PARTKEY are ignored).
     Works for both Chain-NPM output and PrivPetal discretized output.

Query classes (random conjunctive counts over the 3-table join):
  3hop       nation x segment x year x qbin      (3 private tables!)
  3hop_disc  nation x segment x year x dbin
  2hop_cn    nation x year x qbin
  2hop_seg   segment x year x qbin
  1hop       year x qbin
  1hop_seg   segment x dbin
  0hop_year  orders year marginal
  0hop_seg   customer segment marginal

Planted-association preservation (syn/real, 1 = perfect):
  assoc_seg_nation  (within customer, planted)
  assoc_nat_year    (customer->orders, planted)
  assoc_seg_year    (3-table composition signal)
  assoc_size_year   (orders->lineitem, planted)
  mean_qty          (quantity scaled by part ratio)

Run:
  python tpch_deep_eval.py --gt_dir .../TPC-H-deep \
      --syn_prefix .../chainnpm3_eps3.2 --out results.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_side(prefix):
    c = pd.read_csv(prefix + '_syn_c.csv')
    o = pd.read_csv(prefix + '_syn_o.csv')
    l = pd.read_csv(prefix + '_syn_l.csv')
    return c, o, l


def keymap(keys):
    k = np.asarray(keys, dtype=np.int64)
    m = np.full(int(k.max()) + 1 if len(k) else 1, -1, dtype=np.int64)
    m[k] = np.arange(len(k))
    return m


def build_views(c, o, l):
    """Returns dict of numpy arrays aligned to lineitem rows, plus
    order-level arrays."""
    cmap = keymap(c['C_CUSTKEY'].values)
    omap = keymap(o['O_ORDERKEY'].values)

    o_cust = o['O_CUSTKEY'].astype(np.int64).values
    o_cidx = cmap[o_cust] if len(cmap) else np.full(len(o), -1)
    keep_o = o_cidx >= 0
    o_year = o['O_ORDERDATE1'].astype(int).values
    o_pri = o['O_ORDERPRIORITY'].astype(int).values

    c_nat = c['C_NATIONKEY'].astype(int).values
    c_seg = c['C_MKTSEGMENT'].astype(int).values

    l_fk = l['L_ORDERKEY'].astype(np.int64).values
    l_oidx = omap[l_fk] if len(omap) else np.full(len(l), -1)
    keep_l = l_oidx >= 0
    l_oidx = np.where(keep_l, l_oidx, 0)
    o_idx_of_l = l_oidx

    year = o_year[o_idx_of_l]
    pri = o_pri[o_idx_of_l]
    cust_of_order = o_cidx[o_idx_of_l]
    cust_of_order = np.where(cust_of_order >= 0, cust_of_order, 0)
    nation = c_nat[cust_of_order]
    segment = c_seg[cust_of_order]
    qbin = np.clip(l['L_QUANTITY1'].astype(int).values, 0, 4)
    dbin = np.clip(l['L_DISCOUNT'].astype(int).values // 3, 0, 3)
    qty = (l['L_QUANTITY1'].astype(int).values * 10
           + l['L_QUANTITY2'].astype(int).values)

    # order-level sizes
    sizes = np.bincount(l_oidx[keep_l], minlength=len(o))

    return {
        'l_keep': keep_l,
        'nation': nation[keep_l], 'segment': segment[keep_l],
        'year': year[keep_l], 'pri': pri[keep_l],
        'qbin': qbin[keep_l], 'dbin': dbin[keep_l], 'qty': qty[keep_l],
        'o_keep': keep_o, 'o_year': o_year[keep_o],
        'o_nat': np.where(o_cidx[keep_o] >= 0,
                          c_nat[np.clip(o_cidx[keep_o], 0, None)], 0),
        'o_seg': np.where(o_cidx[keep_o] >= 0,
                          c_seg[np.clip(o_cidx[keep_o], 0, None)], 0),
        'o_size': sizes[keep_o],
        'c_nat': c_nat, 'c_seg': c_seg,
        'drop_orders': int((~keep_o).sum()),
        'drop_lineitem': int((~keep_l).sum()),
    }


def make_queries(rng, n):
    queries = []
    for _ in range(n):
        yset = sorted(rng.choice(7, size=int(rng.integers(1, 4)),
                                 replace=False).tolist())
        nset = sorted(rng.choice(25, size=int(rng.integers(2, 8)),
                                 replace=False).tolist())
        sset = sorted(rng.choice(5, size=int(rng.integers(1, 4)),
                                 replace=False).tolist())
        qset = sorted(rng.choice(5, size=int(rng.integers(1, 3)),
                                 replace=False).tolist())
        dset = sorted(rng.choice(4, size=int(rng.integers(1, 3)),
                                 replace=False).tolist())
        pset = sorted(rng.choice(5, size=int(rng.integers(1, 3)),
                                 replace=False).tolist())
        queries.append({'cls': '3hop', 'nset': nset, 'sset': sset,
                        'yset': yset, 'qset': qset})
        queries.append({'cls': '3hop_disc', 'nset': nset, 'sset': sset,
                        'yset': yset, 'dset': dset})
        queries.append({'cls': '2hop_cn', 'nset': nset, 'yset': yset,
                        'qset': qset})
        queries.append({'cls': '2hop_seg', 'sset': sset, 'yset': yset,
                        'qset': qset})
        queries.append({'cls': '1hop', 'yset': yset, 'qset': qset})
        queries.append({'cls': '1hop_seg', 'sset': sset, 'dset': dset})
        queries.append({'cls': '0hop_year', 'yset': yset})
        queries.append({'cls': '0hop_seg', 'sset': sset})
    return queries


def answer(q, v):
    cls = q['cls']
    if cls == '0hop_year':
        return int(np.isin(v['o_year'], q['yset']).sum())
    if cls == '0hop_seg':
        return int(np.isin(v['c_seg'], q['sset']).sum())
    m = np.ones(len(v['year']), dtype=bool)
    if 'yset' in q:
        m &= np.isin(v['year'], q['yset'])
    if 'nset' in q:
        m &= np.isin(v['nation'], q['nset'])
    if 'sset' in q:
        m &= np.isin(v['segment'], q['sset'])
    if 'qset' in q:
        m &= np.isin(v['qbin'], q['qset'])
    if 'dset' in q:
        m &= np.isin(v['dbin'], q['dset'])
    return int(m.sum())


def tv_cond(x, y, nx, ny):
    """TV distance between P(y|x) rows and the marginal P(y)."""
    ct = np.zeros((nx, ny), dtype=float)
    np.add.at(ct, (x, y), 1.0)
    p = ct / np.maximum(ct.sum(axis=1, keepdims=True), 1.0)
    marg = ct.sum(axis=0) / max(ct.sum(), 1.0)
    return float(0.5 * np.abs(p - marg).sum())


def planted_stats(v):
    out = {}
    out['assoc_seg_nation'] = tv_cond(v['c_nat'], v['c_seg'], 25, 5)
    out['assoc_nat_year'] = tv_cond(v['o_nat'], v['o_year'], 25, 7)
    out['assoc_seg_year'] = tv_cond(v['o_seg'], v['o_year'], 5, 7)
    big = (v['o_size'] >= 4).astype(int)
    early = v['o_year'] <= 2
    late = v['o_year'] >= 4
    out['assoc_size_year'] = float(big[late].mean() - big[early].mean()
                                   if early.sum() and late.sum() else 0.0)
    out['mean_qty'] = float(v['qty'].mean()) if len(v['qty']) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--syn_prefix', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_queries', type=int, default=200)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    gt_c = pd.read_csv(os.path.join(args.gt_dir, 'customer.csv'))
    gt_o = pd.read_csv(os.path.join(args.gt_dir, 'orders.csv'))
    gt_l = pd.read_csv(os.path.join(args.gt_dir, 'lineitem.csv'))
    v_gt = build_views(gt_c, gt_o, gt_l)

    syn_c, syn_o, syn_l = load_side(args.syn_prefix)
    v_syn = build_views(syn_c, syn_o, syn_l)

    rng = np.random.default_rng(args.seed)
    queries = make_queries(rng, args.n_queries)

    cls_names = ['3hop', '3hop_disc', '2hop_cn', '2hop_seg', '1hop',
                 '1hop_seg', '0hop_year', '0hop_seg']
    res = {c: [] for c in cls_names}
    res_large = {c: [] for c in cls_names}
    for q in queries:
        a_real = answer(q, v_gt)
        a_syn = answer(q, v_syn)
        re = abs(a_syn - a_real) / max(a_real, 1.0)
        res[q['cls']].append(re)
        if a_real >= 50:
            res_large[q['cls']].append(re)

    p_gt = planted_stats(v_gt)
    p_syn = planted_stats(v_syn)

    out = {
        'config': vars(args),
        'counts': {
            'customers_gt': len(gt_c), 'customers_syn': len(syn_c),
            'orders_gt': len(gt_o), 'orders_syn': len(syn_o),
            'lineitem_gt': len(gt_l), 'lineitem_syn': len(syn_l),
            'syn_drop_orders': v_syn['drop_orders'],
            'syn_drop_lineitem': v_syn['drop_lineitem'],
        },
        're_median': {c: float(np.median(v)) if v else None
                      for c, v in res.items()},
        're_median_large': {c: float(np.median(v)) if v else None
                            for c, v in res_large.items()},
        'planted_real': p_gt,
        'planted_syn': p_syn,
        'planted_ratio': {k: (p_syn[k] / p_gt[k]
                              if abs(p_gt.get(k, 0)) > 1e-9 else None)
                          for k in p_gt},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ['counts', 're_median',
                                          're_median_large',
                                          'planted_ratio']}, indent=2))


if __name__ == '__main__':
    main()
