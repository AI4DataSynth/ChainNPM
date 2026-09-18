#!/usr/bin/env python3
"""P7 统计汇总共享加载器（只读，不改任何数据）。

数据源解析：优先随仓归档 ``results/by_dataset/``，否则用现场战役树
``tmp/results/``。项目根按"是否存在这两个数据目录"向上探测——同一份脚本曾同时
存在于 ``<root>/tmp/`` 与 ``<root>/results/tables/``，后者用
``dirname(dirname(__file__))`` 会落在 ``results/``，导致 D 指向不存在的
``results/tmp/results``，所有汇总静默退化成 0 行。
"""
import json, os, glob


def _has_data(d):
    return (os.path.isdir(os.path.join(d, 'results', 'by_dataset'))
            or os.path.isdir(os.path.join(d, 'tmp', 'results')))


def _find_root(start):
    d = os.path.abspath(start)
    for _ in range(6):
        if _has_data(d):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # 回退到旧口径（<root>/tmp/ 下的拷贝）
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
# Prefer the live campaign tree when it exists: it holds both the natural cells
# and the planted-mode records, while results/by_dataset ships only the natural
# grid.  In the published repository tmp/results is absent and the archive wins.
_ALL = [os.path.join(ROOT, 'tmp', 'results'),
        os.path.join(ROOT, 'results', 'by_dataset')]
D = next((c for c in _ALL if os.path.isdir(c)), _ALL[-1])

DATASETS = ['financial', 'imdb', 'instacart', 'movielens']
EPS_ORDER = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
PLANT_PRODUCT_T = [0.1, 0.2, 0.316, 0.4]

# chain（financial）与 vshape（其余）的 2 跳查询类键名不同
HOP2_KEY = {ds: ('2hop_topbot' if ds == 'financial' else 'cross_r1r2')
            for ds in DATASETS}

# 终图方法系列 → (显示名, JSON method 标签集合)；denorm 已裁定剔除不载入
META = {
    'chainnpm':  ('Chain-NPM',  ['chainnpm']),
    'privpetal': ('PrivPetal',  ['privpetal']),
    'lavaprop':  ('PrivLava',   ['lavaprop']),
    'privmrf':   ('PrivMRF',    ['privmrf', 'pertable']),  # 吸收 pertable（ε 修复后值）
    'privbayes': ('PrivBayes',  ['privbayes']),
    'pbpgm':     ('PB-PGM',     ['pbpgm']),
}
# pertable 优先于 privmrf 的 1 个冗余文件
SOURCE_PREFERENCE = {'privmrf': ['pertable', 'privmrf']}


def iter_files():
    for ds in DATASETS:
        for fp in glob.glob(os.path.join(D, ds, '*.json')):
            if os.path.basename(fp).startswith('._') or '_sub' in os.path.basename(fp):
                continue
            try:
                with open(fp) as f:
                    d = json.load(f)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            d['_dataset'] = ds
            d['_file'] = fp
            yield d


def canonical_series(method_label):
    """JSON method 标签 → 终图系列 key（不在终图系列则返回 None）。"""
    for series, (_, tags) in META.items():
        if method_label in tags:
            return series
    return None


def plant_t_from_filename(fp_or_name):
    """从文件名提取 plant t（如 t0.1 / t0.316）。"""
    import re
    m = re.search(r'_t([0-9.]+)_', os.path.basename(fp_or_name))
    return float(m.group(1)) if m else None


def load_rows(pref_order=None):
    """同 (ds, series, eps, seed, mode, t) 去重（plant 模式 t 取自卫文件名；
    natural 模式 t=None），同格保留 pref 横靠前的源码文件。

    降级记录（无 re.re_median_large，例如 movielens 上被历史 bug 写坏的
    lavaprop_natural_eps0.1/3.2_seed42.json）永不顶掉同格的可用记录；可用
    记录则一定顶掉降级记录，避免去重结果依赖 glob 的返回顺序。
    """
    best = {}
    for d in iter_files():
        series = canonical_series(d.get('method'))
        if series is None:
            continue
        t = plant_t_from_filename(d.get('_file'))
        key = (d['_dataset'], series, d.get('eps'), d.get('seed'),
               d.get('mode'), t)
        if key in best:
            old = best[key]
            new_ok = hop2_re(d) is not None
            old_ok = hop2_re(old) is not None
            if new_ok and not old_ok:
                best[key] = d
            elif new_ok == old_ok and pref_order is not None and \
                    pref_order.index(d.get('method')) < \
                    pref_order.index(old.get('method')):
                best[key] = d
        else:
            best[key] = d
    rows = []
    for d in best.values():
        d = dict(d)
        d['method'] = canonical_series(d.get('method'))
        rows.append(d)
    return rows


def hop2_re(d):
    """extract 2-hop RE from re.re_median_large；缺失返回 None。"""
    reobj = d.get('re')
    m = reobj.get('re_median_large') if isinstance(reobj, dict) else None
    if not isinstance(m, dict) or not m:
        return None
    key = HOP2_KEY.get(d.get('dataset') or d.get('_dataset'))
    if key not in m:
        return None
    return m[key]


def effect_scalars(d):
    """返回 {target: (real, syn)}：
    financial(chain) → delta；vshape → contrast（plant_xor 另有 phi 时附 phi）。"""
    ef = d.get('effect') or {}
    ds = d.get('dataset') or d.get('_dataset')
    out = {}
    if ds == 'financial':
        if 'delta_real' in ef and 'delta_syn' in ef:
            r = ef['delta_real'].get('delta')
            s = ef['delta_syn'].get('delta')
            if r is not None and s is not None:
                out['delta'] = (r, s)
    else:
        if 'contrast_real' in ef and 'contrast_syn' in ef:
            r = ef['contrast_real'].get('contrast')
            s = ef['contrast_syn'].get('contrast')
            if r is not None and s is not None:
                out['contrast'] = (r, s)
        if 'phi_real' in ef and 'phi_syn' in ef and 'phi_xor' in ef['phi_real']:
            r = ef['phi_real'].get('phi_xor')
            s = ef['phi_syn'].get('phi_xor')
            if r is not None and s is not None:
                out['phi'] = (r, s)
    return out


if __name__ == '__main__':
    # 自检：打印各 (dataset, series, mode, eps) 的 seed 数
    from collections import defaultdict
    cc = defaultdict(list)
    pref = ['pertable', 'privmrf', 'chainnpm', 'privpetal', 'lavaprop',
            'privbayes', 'pbpgm', 'denorm']
    for d in iter_files():
        s = canonical_series(d.get('method'))
        if s is None:
            continue
        cc[(d['_dataset'], s, d.get('mode'), d.get('eps'))].append(d.get('seed'))
    for k in sorted(cc):
        print(k, 'n=%d' % len(cc[k]))