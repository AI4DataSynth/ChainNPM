#!/usr/bin/env python3
"""P7-1 汇总表：natural 模式 2-hop RE（re_median_large，大查询）。
(dataset, series, eps) × {n, mean, median, std, bootstrap95%_lo/hi}
bootstrap: 1000 次 seeds 重抽样均值，RNG 固定 42。产物 tmp/p7_summary.tsv。"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p7_common import (load_rows, hop2_re, META, HOP2_KEY, DATASETS, EPS_ORDER)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p7_summary.tsv')
NBOOT = 1000

PREF_ORDER = ['pertable', 'privmrf', 'chainnpm', 'privpetal', 'lavaprov',
              'privbayes', 'pbpgm']


def bootstrap95(vals, rng):
    arr = np.asarray(vals, dtype=float)
    means = np.empty(NBOOT)
    n = len(arr)
    for b in range(NBOOT):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    # 修正 pref 表（前面写错了 lavaprop 标签）
    pref = ['pertable', 'privmrf', 'chainnpm', 'privpetal', 'lavaprop',
            'privbayes', 'pbpgm']
    rows = load_rows(pref_order=pref)
    rng = np.random.default_rng(42)

    # 按 (dataset, series, eps) 聚合 natural 模式
    agg = {}
    dropped_invalid = 0
    for d in rows:
        if d.get('mode') != 'natural':
            continue
        v = hop2_re(d)
        if v is None:
            dropped_invalid += 1
            continue
        key = (d.get('dataset') or d['_dataset'], d['method'], d.get('eps'))
        agg.setdefault(key, []).append(float(v))

    lines = ['# P7-1 summary: natural mode, 2-hop RE = '
             're_median_large[%s] (large queries, prereg query_seed=123, '
             '150 queries/class, large_count=50)' % ' | '.join(
                 '%s=%s' % (ds, HOP2_KEY[ds]) for ds in DATASETS),
             '# bootstrap: %d resamples of per-seed means, RNG seed 42; '
             'series labels follow final figures (denorm excluded; '
             'PrivMRF = privmrf+pertable, pertable preferred)' % NBOOT,
             'dataset\tseries\teps\tn\tmean\tmedian\tstd\tboot_lo\tboot_hi']
    for ds in DATASETS:
        for series in META:
            for eps in EPS_ORDER:
                vals = agg.get((ds, series, eps))
                if not vals:
                    continue
                v = np.array(vals)
                lo, hi = bootstrap95(v, rng)
                lines.append('%s\t%s\t%.1f\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f'
                             % (ds, series, eps, len(v), float(v.mean()),
                                float(np.median(v)), float(v.std(ddof=1)),
                                lo, hi))
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # 与 p5_preview 数值表对齐（median 必须一致）——6 档 ε 全覆盖
    ref = {  # (dataset, series): (ε0.1, ε0.2, ε0.4, ε0.8, ε1.6, ε3.2) median
        ('financial', 'chainnpm'):  (3.881, 2.438, 1.683, 0.905, 0.690, 0.435),
        ('financial', 'privpetal'): (0.292, 0.508, 0.732, 0.842, 0.917, 0.943),
        ('financial', 'lavaprop'):  (0.340, 0.380, 0.343, 0.369, 0.330, 0.343),
        ('financial', 'privmrf'):   (0.288, 0.218, 0.170, 0.097, 0.136, 0.156),
        ('financial', 'privbayes'): (0.518, 0.496, 0.402, 0.138, 0.173, 0.119),
        ('financial', 'pbpgm'):     (1.000, 0.851, 0.691, 0.280, 0.267, 0.137),
        ('imdb', 'chainnpm'):       (0.230, 0.197, 0.178, 0.161, 0.151, 0.149),
        ('imdb', 'privpetal'):      (0.641, 0.306, 0.313, 0.293, 0.332, 0.364),
        ('imdb', 'lavaprop'):       (0.999, 0.999, 0.999, 0.999, 0.999, 0.999),
        ('imdb', 'privmrf'):        (0.207, 0.203, 0.201, 0.200, 0.196, 0.199),
        ('imdb', 'privbayes'):      (0.224, 0.216, 0.210, 0.204, 0.204, 0.204),
        ('imdb', 'pbpgm'):          (0.224, 0.210, 0.213, 0.206, 0.202, 0.205),
        ('instacart', 'chainnpm'):  (0.567, 0.558, 0.419, 0.348, 0.383, 0.452),
        ('instacart', 'privpetal'): (1.207, 1.032, 0.454, 0.444, 0.443, 0.438),
        ('instacart', 'lavaprop'):  (0.904, 0.901, 0.908, 0.920, 0.934, 0.953),
        ('instacart', 'privmrf'):   (0.506, 0.497, 0.471, 0.502, 0.491, 0.481),
        ('instacart', 'privbayes'): (0.666, 0.548, 0.501, 0.488, 0.479, 0.481),
        ('instacart', 'pbpgm'):     (0.501, 0.477, 0.469, 0.479, 0.479, 0.476),
        ('movielens', 'chainnpm'):  (0.976, 0.777, 0.727, 0.665, 0.616, 0.516),
        ('movielens', 'privpetal'): (0.667, 0.774, 0.818, 0.801, 0.811, 0.817),
        ('movielens', 'lavaprop'):  (0.981, 0.983, 0.982, 0.969, 0.974, 0.963),
        ('movielens', 'privmrf'):   (0.389, 0.354, 0.347, 0.341, 0.338, 0.331),
        ('movielens', 'privbayes'): (0.411, 0.381, 0.361, 0.344, 0.346, 0.327),
        ('movielens', 'pbpgm'):     (0.453, 0.393, 0.379, 0.342, 0.351, 0.326),
    }
    bad = []
    for key, refs in ref.items():
        for i, eps in enumerate(EPS_ORDER):
            r = refs[i]
            if r is None:
                continue
            vals = agg.get((key[0], key[1], eps))
            if vals is None:
                bad.append((key, eps, 'eps missing'))
                continue
            med = float(np.median(vals))
            if abs(med - r) > 1e-3:
                bad.append((key, eps, med, r))
    print('summary written:', OUT)
    print('cells:', len(agg))
    print('drops_no_2hop_re (vshape lavaprop 降级等):', dropped_invalid)
    print('p5-alignment mismatches:', bad if bad else 'NONE — median 与 p5 期望基准全对齐')


if __name__ == '__main__':
    main()