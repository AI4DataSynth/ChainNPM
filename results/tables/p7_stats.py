#!/usr/bin/env python3
"""P7-2 配对显著检验：每 (dataset, eps) 下 Chain-NPM vs 各基线，
natural 模式 2-hop RE，按 seed 配对 Wilcoxon signed-rank（scipy.stats.wilcoxon）。
零差对剔除后 n<6 → NaN 标注；p 按数据集内全系对比族做 Holm 校正（α=0.05）。
产物 tmp/p7_stats.json。"""
import json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p7_common import load_rows, hop2_re, META, HOP2_KEY, DATASETS, EPS_ORDER

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p7_stats.json')
ALPHA = 0.05
MIN_N = 6  # 零差剔除后最小配对样本量

PREF = ['pertable', 'privmrf', 'chainnpm', 'privpetal', 'lavaprop',
        'privbayes', 'pbpgm']
BASELINES = ['privpetal', 'lavaprop', 'privmrf', 'privbayes', 'pbpgm']


def main():
    rows = load_rows(pref_order=PREF)
    # natural 模式 per (ds, series, eps) → {seed: re}
    per = {}
    for d in rows:
        if d.get('mode') != 'natural':
            continue
        v = hop2_re(d)
        if v is None:
            continue
        key = (d.get('dataset') or d['_dataset'], d['method'], d.get('eps'))
        per.setdefault(key, {}).setdefault(d.get('seed'), []).append(float(v))
    # 同 (ds, series, eps, seed) 理论唯一；若重复取均值（防御）
    for k in per:
        for s in list(per[k]):
            v = per[k][s]
            per[k][s] = float(np.mean(v))

    comparisons = []
    families = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        hopkey = HOP2_KEY[ds]
        for eps in EPS_ORDER:
            base = f'chainnpm_{ds}_{eps}'
            chain = per.get((ds, 'chainnpm', eps))
            for b in BASELINES:
                if b == 'lavaprop' and ds != 'financial':
                    comparisons.append({
                        'dataset': ds, 'eps': eps, 'baseline': b,
                        'chain_series': 'chainnpm', 'metric': hopkey,
                        'n_pairs': 0, 'n_nonzero': 0, 'w_stat': None,
                        'p_raw': None, 'p_holm': None, 'median_diff': None,
                        'chain_downside_idx': None, 'significant': False,
                        'note': 'no data — PrivLava 仅 financial 有 re（vshape 预注册降级）',
                    })
                    continue
                bvals = per.get((ds, b, eps))
                if chain is None and bvals is None:
                    comparisons.append({
                        'dataset': ds, 'eps': eps, 'baseline': b,
                        'chain_series': 'chainnpm', 'metric': hopkey,
                        'n_pairs': 0, 'n_nonzero': 0, 'w_stat': None,
                        'p_raw': None, 'p_holm': None, 'median_diff': None,
                        'chain_downside_idx': None, 'significant': False,
                        'note': 'no data for both series',
                    })
                    continue
                if chain is None or bvals is None:
                    comparisons.append({
                        'dataset': ds, 'eps': eps, 'baseline': b,
                        'chain_series': 'chainnpm', 'metric': hopkey,
                        'n_pairs': 0, 'n_nonzero': 0, 'w_stat': None,
                        'p_raw': None, 'p_holm': None, 'median_diff': None,
                        'chain_downside_idx': None, 'significant': False,
                        'note': 'no data for one series (chain=%s base=%s)'
                                % (chain is not None, bvals is not None),
                    })
                    continue
                seeds = sorted(set(chain) & set(bvals))
                n_pairs = len(seeds)
                if n_pairs == 0:
                    comparisons.append({
                        'dataset': ds, 'eps': eps, 'baseline': b,
                        'chain_series': 'chainnpm', 'metric': hopkey,
                        'n_pairs': 0, 'n_nonzero': 0, 'w_stat': None,
                        'p_raw': None, 'p_holm': None, 'median_diff': None,
                        'chain_downside_idx': None, 'significant': False,
                        'note': 'no shared seeds',
                    })
                    continue
                diff = np.array([chain[s] - bvals[s] for s in seeds])
                nz = diff[np.abs(diff) > 1e-12]
                med_diff = float(np.median(diff))
                rec = {
                    'dataset': ds, 'eps': eps, 'baseline': b,
                    'chain_series': 'chainnpm', 'metric': hopkey,
                    'n_pairs': n_pairs, 'n_nonzero': int(len(nz)),
                    'w_stat': None, 'p_raw': None, 'p_holm': None,
                    'median_diff': med_diff,
                    'chain_lower_median': bool(med_diff < 0),  # 更低 RE = 更好
                    'significant': False,
                    'note': None,
                }
                if len(nz) < MIN_N:
                    rec['p_raw'] = None
                    rec['note'] = ('insufficient nonzero pairs n=%d (<%d)'
                                   % (len(nz), MIN_N))
                    if b == 'privpetal':
                        rec['note'] += '; PrivPetal n=3 证据不足'
                    comparisons.append(rec)
                    continue
                res = stats.wilcoxon(nz)
                rec['w_stat'] = float(res.statistic)
                rec['p_raw'] = float(res.pvalue)
                comparisons.append(rec)
                families[ds].append(rec)

    # Holm 校正（每数据集一个对比族）
    for ds in DATASETS:
        fam = [r for r in families[ds] if r['p_raw'] is not None]
        fam.sort(key=lambda r: r['p_raw'])
        m = len(fam)
        for i, r in enumerate(fam):
            r['p_holm'] = min(1.0, r['p_raw'] * (m - i))
            r['significant'] = bool(r['p_holm'] < ALPHA)

    # 有 n<6 或 no data 的记录 p_holm/significant 保持 None/False
    out = {
        'title': 'P7-2 paired significance: Chain-NPM vs baselines',
        'metric': 'natural-mode 2-hop RE = re_median_large[2hop_topbot | cross_r1r2]',
        'test': 'two-sided Wilcoxon signed-rank on per-seed paired differences',
        'alpha': ALPHA,
        'min_n_nonzero': MIN_N,
        'holm_family': 'per dataset (all present eps x baseline cells)',
        'comparisons': comparisons,
        'notes': {
            'privpetal': '仅 seeds 42/43/44（n=3），任何 p 值都不足以下显著结论',
            'privmrf': '系列 = privmrf+pertable 合并，pertable 优先；privmrf 单文件已并入',
            'lavaprop': '仅 financial 有 re；vshape 三数据集为预注册降级（无 re）',
            'direction': 'median_diff = median(chain - baseline)（2-hop RE，越低越好）',
        },
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print('written:', OUT)
    print('total comparison rows:', len(comparisons))
    n_sig = sum(1 for r in comparisons if r['significant'])
    print('significant after Holm:', n_sig)
    for r in comparisons:
        if r['significant']:
            direction = 'chainbetter' if r['chain_lower_median'] else 'baselineBetter'
            print('  SIG %s eps%.1f vs %s  p_raw=%.4g p_holm=%.4g med_diff=%+.4f (%s)'
                  % (r['dataset'], r['eps'], r['baseline'], r['p_raw'],
                     r['p_holm'], r['median_diff'], direction))


if __name__ == '__main__':
    main()