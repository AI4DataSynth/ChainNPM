#!/usr/bin/env python3
"""P5.5 六基线预览图：每数据集一 panel，x=ε(ln2 轴)，y=跨模式的2跳 RE
（chain financial 用 2hop_topbot，vshape 用 cross_r1r2；re_median_large）。
按 seed 取中位数、min-max 为误差带。natural 模式。

六条线：Chain-NPM / PrivPetal / PrivLava / PrivMRF / PrivBayes / PB-PGM。
- denorm：用户已裁定不进手稿，不绘。
- PrivMRF == 手稿 PrivMRF（逐表 PrivMRF + 随机 FK 重连）；线吸收
  'privmrf' 与 'pertable' 两个 method 标签（P5 pertable ε 盲修复重跑后回填，
  去重时优先 pertable）。
- 单点散点标注：只有 1 个 ε 点的方法（P5 服务器续跑未回填）。
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
D = os.path.join(ROOT, 'tmp', 'results')
OUT = os.path.join(ROOT, 'tmp', 'fig_p5_preview_6lines_natural_crosshop.png')

DATASETS = ['financial', 'imdb', 'instacart', 'movielens']
EPS_ORDER = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]

# chain（financial）与 vshape（其余）的 2 跳查询类键名不同
HOP2_KEY = {ds: ('2hop_topbot' if ds == 'financial' else 'cross_r1r2')
            for ds in DATASETS}

# 方法显示名 / 颜色 / 标记（series → (label, color, marker)）
META = {
    'chainnpm':  ('Chain-NPM',  '#1d4ed8', 'o'),
    'privpetal': ('PrivPetal',  '#b45309', 's'),
    'lavaprop':  ('PrivLava',   '#9333ea', 'D'),
    'privmrf':   ('PrivMRF',    '#be123c', 'v'),
    'privbayes': ('PrivBayes',  '#059669', '^'),
    'pbpgm':     ('PB-PGM',     '#0e7490', 'P'),
}
# 吸收的 method 标签：系列 → JSON method 字段取值
SOURCE_METHODS = {
    'privmrf': ['privmrf', 'pertable'],
}


def load_rows():
    seen = set()
    rows = []
    for ds in DATASETS:
        for fp in sorted(glob.glob(os.path.join(D, ds, '*.json'))):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            if not (isinstance(d, dict) and d.get('mode') == 'natural'):
                continue
            if '_sub' in os.path.basename(fp):
                continue
            m = d.get('method')
            for series, meths in SOURCE_METHODS.items():
                if m in meths:
                    m = series
                    break
            if m not in META:
                continue
            # 去重：(ds, series, eps, seed)，同格只留一个（优先 pertable 源）；
            # 并回写归一化后的 method，使 aggregate() 按系列名键控。
            # 降级记录（无 re 指标）不占用去重名额：同格的有效记录
            # （如 lavaprop vshape _r1 版）必须能覆盖先读到的降级文件。
            if not (isinstance(d.get('re'), dict)
                    and d['re'].get('re_median_large')):
                d = dict(d)
                d['method'] = m
                rows.append(d)
                continue
            key = (ds, m, d.get('eps'), d.get('seed'))
            if key in seen:
                continue
            seen.add(key)
            d = dict(d)
            d['method'] = m
            rows.append(d)
    return rows


def aggregate(rows, metric='auto'):
    """(ds, series) -> {eps: list[val]}，取 re.re_median_large[metric]；
    metric='auto' 按数据集选 2 跳键"""
    agg = {}
    for d in rows:
        k = (d['dataset'], d['method'])
        reobj = d.get('re')
        m = reobj.get('re_median_large') if isinstance(reobj, dict) else None
        if not isinstance(m, dict) or not m:
            continue  # 降级记录（如 movielens lavaprop）无指标，跳过
        key = HOP2_KEY[d['dataset']] if metric == 'auto' else metric
        if key not in m:
            continue
        agg.setdefault(k, {})
        agg[k].setdefault(d['eps'], []).append(m[key])
    return agg


def main():
    rows = load_rows()
    agg = aggregate(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    fig.suptitle('Six-baseline preview — natural mode, 2-hop cross-schema RE\n'
                 'median over seeds of re_median_large (large queries only)',
                 fontsize=11)
    for ax, ds in zip(axes.ravel(), DATASETS):
        for series, (label, color, marker) in META.items():
            pts = agg.get((ds, series))
            if not pts:
                continue
            eps = sorted(pts, key=lambda e: e if e is not None else 1e9)
            med = [float(np.median(pts[e])) for e in eps]
            lo = [float(np.min(pts[e])) for e in eps]
            hi = [float(np.max(pts[e])) for e in eps]
            x = eps
            if len(eps) >= 2:
                ax.plot(x, med, marker=marker, color=color, linewidth=1.8,
                        markersize=5.5, label=label, zorder=3)
                ax.fill_between(x, lo, hi, color=color, alpha=0.13,
                                linewidth=0)
                ax.annotate(label, (eps[-1], med[-1]),
                            textcoords='offset points', xytext=(7, 0),
                            fontsize=8.5, color=color, va='center')
            else:  # 单点：散点 + 标注
                ax.scatter(x, med, marker=marker, color=color, s=46, zorder=4,
                           edgecolor='white', linewidth=0.6)
                ax.annotate(('* ' + label), (x[0], med[0]),
                            textcoords='offset points', xytext=(7, -2),
                            fontsize=8, color=color, va='center',
                            style='italic')
        ax.set_xscale('log', base=2)
        ax.set_xticks(EPS_ORDER)
        ax.set_xticklabels([str(e) for e in EPS_ORDER])
        ax.set_xlabel('privacy budget $\\varepsilon$ (log₂)')
        ax.set_ylabel('median RE (2-hop, large queries)')
        ax.set_title(ds, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlim(0.075, 4.9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=200)
    print('saved', OUT)
    # 打印数值表，供汇报
    print('\n== median 2-hop RE by (dataset, method, eps) ==')
    for ds in DATASETS:
        for series in META:
            pts = agg.get((ds, series))
            if pts:
                s = '  '.join('ε%.1f: %.3f(n=%d)' % (e, np.median(pts[e]),
                                                     len(pts[e]))
                              for e in sorted(pts))
                print('%s %-10s %s' % (ds, META[series][0], s))


if __name__ == '__main__':
    main()