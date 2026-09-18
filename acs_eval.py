"""
Evaluator for the ACS PUMS household<-individual DP synthesis (PO-2).

Compares synthetic _syn_h/_syn_i against GT household.csv/individual.csv:
  - random conjunctive counts: 2-hop (hh attr x person attr over the join),
    1-hop (person-only), 0-hop (household-only)
  - household size distribution TV + mean size
  - pairwise association TV on the join for key (hh, person) pairs
    (lower = better; reported as absolute TV difference syn vs GT)

Run:
  python acs_eval.py --data_dir .../acs/processed \
      --syn_prefix .../chainnpm2_acs_eps3.2 --out result.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

H_ATTRS = ['ACR', 'BLD', 'HHT', 'TEN', 'VEH', 'NPbin', 'HINCPbin']
I_ATTRS = ['AGEbin', 'SEX', 'SCHLbin', 'ESR', 'MAR', 'RELPbin', 'WKHPbin']
KEY_PAIRS = [('HHT', 'AGEbin'), ('HINCPbin', 'ESR'), ('NPbin', 'RELPbin'),
             ('TEN', 'MAR'), ('HHT', 'SCHLbin')]


def build_view(h, i):
    """Return per-individual attribute dict + per-household arrays."""
    hmap = np.full(int(h['HHKEY'].max()) + 1, -1, dtype=int)
    hmap[h['HHKEY'].values] = np.arange(len(h))
    fk = i['HHKEY'].astype(int).values
    idx = hmap[fk]
    keep = idx >= 0
    idx = np.where(keep, idx, 0)
    view = {}
    for a in H_ATTRS:
        view[a] = h[a].astype(int).values[idx][keep]
    for a in I_ATTRS:
        view[a] = i[a].astype(int).values[keep]
    sizes = np.bincount(i['HHKEY'].astype(int).values[keep],
                        minlength=len(h))
    return view, keep, sizes


def make_queries(rng, n):
    queries = []
    for _ in range(n):
        ha = sorted(rng.choice(H_ATTRS,
                               size=int(rng.integers(1, 3)),
                               replace=False).tolist())
        ia = sorted(rng.choice(I_ATTRS,
                               size=int(rng.integers(1, 3)),
                               replace=False).tolist())
        hv = {a: sorted(rng.choice(12, size=int(rng.integers(1, 4)),
                                   replace=False).tolist()) for a in ha}
        iv = {a: sorted(rng.choice(10, size=int(rng.integers(1, 4)),
                                   replace=False).tolist()) for a in ia}
        queries.append({'cls': '2hop', 'h': {a: [v for v in hv[a]]
                                             for a in ha},
                        'i': {a: iv[a] for a in ia}})
        queries.append({'cls': '1hop', 'i': iv})
        queries.append({'cls': '0hop', 'h': hv})
    return queries


def answer(q, view, h_view):
    cls = q['cls']
    if cls == '0hop':
        m = np.ones(len(h_view[H_ATTRS[0]]), dtype=bool)
        for a, vals in q['h'].items():
            m &= np.isin(h_view[a], vals)
        return int(m.sum())
    m = np.ones(len(view[I_ATTRS[0]]), dtype=bool)
    for a, vals in q.get('i', {}).items():
        m &= np.isin(view[a], vals)
    if cls == '2hop':
        for a, vals in q['h'].items():
            m &= np.isin(view[a], vals)
    return int(m.sum())


def pair_tv(view, ha, ia, dh, di):
    """TV between joint P(ha,ia) and product of marginals? No: we report
    TV distance of the syn joint vs gt joint (both normalized)."""
    ct = np.zeros((dh, di))
    np.add.at(ct, (view[ha], view[ia]), 1.0)
    return ct / max(ct.sum(), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--syn_prefix', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_queries', type=int, default=200)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    gt_h = pd.read_csv(os.path.join(args.data_dir, 'household.csv'))
    gt_i = pd.read_csv(os.path.join(args.data_dir, 'individual.csv'))
    syn_h = pd.read_csv(args.syn_prefix + '_syn_h.csv')
    syn_i = pd.read_csv(args.syn_prefix + '_syn_i.csv')

    v_gt, keep_gt, sz_gt = build_view(gt_h, gt_i)
    v_syn, keep_syn, sz_syn = build_view(syn_h, syn_i)

    h_dom = {a: int(gt_h[a].max()) + 1 for a in H_ATTRS}
    i_dom = {a: int(gt_i[a].max()) + 1 for a in I_ATTRS}

    # household-only view for 0hop
    hv_gt = {a: gt_h[a].astype(int).values for a in H_ATTRS}
    hv_syn = {a: syn_h[a].astype(int).values for a in H_ATTRS}

    rng = np.random.default_rng(args.seed)
    queries = make_queries(rng, args.n_queries)

    res = {c: [] for c in ['2hop', '1hop', '0hop']}
    res_large = {c: [] for c in res}
    for q in queries:
        a_real = answer(q, v_gt, hv_gt)
        a_syn = answer(q, v_syn, hv_syn)
        re = abs(a_syn - a_real) / max(a_real, 1.0)
        res[q['cls']].append(re)
        if a_real >= 50:
            res_large[q['cls']].append(re)

    # size distribution TV (bins 1..20)
    tau = 20
    gt_hist = np.bincount(np.clip(sz_gt, 1, tau), minlength=tau + 1)[1:]
    syn_hist = np.bincount(np.clip(sz_syn, 1, tau), minlength=tau + 1)[1:]
    size_tv = float(0.5 * np.abs(gt_hist / gt_hist.sum()
                                 - syn_hist / max(syn_hist.sum(), 1)).sum())

    pair_tvs = {}
    for (ha, ia) in KEY_PAIRS:
        p_gt = pair_tv(v_gt, ha, ia, h_dom[ha], i_dom[ia])
        p_syn = pair_tv(v_syn, ha, ia, h_dom[ha], i_dom[ia])
        pair_tvs[f'{ha}x{ia}'] = float(0.5 * np.abs(p_gt - p_syn).sum())

    out = {
        'config': vars(args),
        'counts': {'households_gt': len(gt_h), 'households_syn': len(syn_h),
                   'individuals_gt': len(gt_i),
                   'individuals_syn': len(syn_i),
                   'syn_drop_individuals': int((~keep_syn).sum())},
        're_median': {c: float(np.median(v)) if v else None
                      for c, v in res.items()},
        're_median_large': {c: float(np.median(v)) if v else None
                            for c, v in res_large.items()},
        'size_tv': size_tv,
        'mean_size_gt': float(sz_gt.mean()),
        'mean_size_syn': float(sz_syn.mean()),
        'pair_tv_syn_vs_gt': pair_tvs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ['counts', 're_median',
                                          're_median_large', 'size_tv',
                                          'mean_size_gt', 'mean_size_syn',
                                          'pair_tv_syn_vs_gt']}, indent=2))


if __name__ == '__main__':
    main()
