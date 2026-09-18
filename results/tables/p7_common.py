#!/usr/bin/env python3
"""P7 统计汇总共享加载器（只读 tmp/results，不改任何数据）。"""
import json, os, glob

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Read the shipped archive when present (results/by_dataset/), otherwise a live
# campaign tree (tmp/results/).
_cands = [os.path.join(ROOT, 'results', 'by_dataset'), os.path.join(ROOT, 'tmp', 'results')]
D = next((c for c in _cands if os.path.isdir(c)), _cands[-1])

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
    natural 模式 t=None），同格保留 pref 横靠前的源码文件。"""
    best = {}
    for d in iter_files():
        series = canonical_series(d.get('method'))
        if series is None:
            continue
        t = plant_t_from_filename(d.get('_file'))
        key = (d['_dataset'], series, d.get('eps'), d.get('seed'),
               d.get('mode'), t)
        if key in best:
            # 保留 pref 更靠前者；并列保留先见
            if pref_order is not None and \
               pref_order.index(d.get('method')) < pref_order.index(best[key].get('method')):
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