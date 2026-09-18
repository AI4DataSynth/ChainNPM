#!/usr/bin/env python3
"""P7-3 effect/contrast 二次分析（跨跳相关保留）：
planted 模式 plant_xor(eps 0.8/3.2, t=0.8) 与 plant_product(全 t x eps) 下，每
(dataset, series, eps, t, target) 输出 real（植入真实效应）与 syn（合成保留效应），
及保留误差 |syn - real| / |real|（per-seed 均值/中位数）。
- financial(chain) → target=delta；vshape → target=contrast（plant_xor 另有 phi）。
- 系列：chainnpm / privmrf(=pertable) / privpetal；denorm 已剔除不入档。
- privpetal 仅 eps3.2、n=3（标注 n 受限）；privbayes/pbpgm/lavaprop 无 planted 数据。
产物 tmp/p7_effect.tsv。"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p7_common import (load_rows, effect_scalars, plant_t_from_filename,
                       META, DATASETS, EPS_ORDER, PLANT_PRODUCT_T)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p7_effect.tsv')
PREF = ['pertable', 'privmrf', 'chainnpm', 'privpetal', 'lavaprop',
        'privbayes', 'pbpgm']
MODES = ['plant_product', 'plant_xor']
SERIES_OF_INTEREST = ['chainnpm', 'privmrf', 'privpetal']


def ret_err(real, syn):
    if real == 0:
        return None
    return abs(syn - real) / abs(real)


def main():
    rows = load_rows(pref_order=PREF)
    # (dataset, mode, eps, t, series, target) -> list[(real, syn, err, seed)]
    cells = {}
    for d in rows:
        if d.get('mode') not in MODES:
            continue
        series = d['method']
        if series not in SERIES_OF_INTEREST:
            continue
        t = plant_t_from_filename(d.get('_file', ''))
        if t is None:
            # privpetal 植入文件文件名无 t；plant.target 携带 t（product=0.316, xor=0.8）
            t = (d.get('plant') or {}).get('target')
        if t is None:
            t = 0.8  # plant_xor 的 t0.8 由文件名给出
        sc = effect_scalars(d)
        for target, (real, syn) in sc.items():
            key = (d.get('dataset') or d['_dataset'], d['mode'], d.get('eps'),
                   t, series, target)
            cells.setdefault(key, []).append(
                (real, syn, ret_err(real, syn), d.get('seed')))

    lines = ['# P7-3 effect/contrast retention (planted mode): '
             'retention error = |syn - real| / |real| per seed, '
             'aggregated by mean/median over seeds',
             '# financial -> target=delta; vshape -> target=contrast '
             '(plant_xor also phi=xor interaction when present)',
             '# series: chainnpm / privmrf(=privmrf+pertable, pertable '
             'preferred) / privpetal; denorm excluded; '
             'privpetal n=3 eps3.2 only; privbayes/pbpgm/lavaprop no planted data',
             'dataset\tmode\teps\tt\tseries\ttarget\tn\t'
             'real_mean\treal_med\tsyn_mean\tsyn_med\tret_err_mean\tret_err_med']
    n_privpetal = 0
    for ds in DATASETS:
        for mode in MODES:
            for eps in EPS_ORDER:
                tlist = PLANT_PRODUCT_T if mode == 'plant_product' else [0.8]
                for t in tlist:
                    for series in SERIES_OF_INTEREST:
                        for target in (['delta'] if ds == 'financial'
                                       else ['contrast', 'phi']):
                            key = (ds, mode, eps, t, series, target)
                            recs = cells.get(key)
                            if not recs:
                                continue
                            reals = np.array([r[0] for r in recs])
                            syns = np.array([r[1] for r in recs])
                            errs = np.array([r[2] for r in recs if r[2] is not None])
                            if series == 'privpetal':
                                n_privpetal += 1
                            err_mean = float(np.mean(errs)) if len(errs) else np.nan
                            err_med = float(np.median(errs)) if len(errs) else np.nan
                            lines.append(
                                '%s\t%s\t%.1f\t%.3f\t%s\t%s\t%d\t%.6f\t%.6f\t'
                                '%.6f\t%.6f\t%.6f\t%.6f' % (
                                    ds, mode, eps, t, series, target, len(recs),
                                    float(reals.mean()), float(np.median(reals)),
                                    float(syns.mean()), float(np.median(syns)),
                                    err_mean, err_med))
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('written:', OUT)
    print('privpetal effect cells (n=3, eps3.2):', n_privpetal)

    # ---- Chain-NPM 相对基线的保留优势 ----
    print('\n== Chain-NPM retention-advantage vs baseline ==')
    for base in ['privmrf', 'privpetal']:
        wins, losses, ties, total = 0, 0, 0, 0
        print('vs %s:' % META[base][0])
        for ds in DATASETS:
            for mode in MODES:
                for eps in EPS_ORDER:
                    tlist = PLANT_PRODUCT_T if mode == 'plant_product' else [0.8]
                    for t in tlist:
                        for target in (['delta'] if ds == 'financial'
                                       else ['contrast', 'phi']):
                            kc = (ds, mode, eps, t, 'chainnpm', target)
                            kb = (ds, mode, eps, t, base, target)
                            if kc not in cells or kb not in cells:
                                continue
                            ec = cells[kc]; eb = cells[kb]
                            erc = np.mean([r[2] for r in ec if r[2] is not None])
                            erb = np.mean([r[2] for r in eb if r[2] is not None])
                            if np.isnan(erc) or np.isnan(erb):
                                continue
                            total += 1
                            if erc < erb - 1e-6:
                                wins += 1
                            elif erb < erc - 1e-6:
                                losses += 1
                            else:
                                ties += 1
        print('  wins(lower err)=%d losses=%d ties=%d total=%d'
              % (wins, losses, ties, total))
    # 逐格明细（保留误差最低者高亮）
    print('\n== per-cell retention error (mean over seeds); * = best among '
          'chainnpm/privmrf/privpetal ==')
    best_count = {'chainnpm': 0, 'privmrf': 0, 'privpetal': 0}
    for ds in DATASETS:
        for mode in MODES:
            for eps in EPS_ORDER:
                tlist = PLANT_PRODUCT_T if mode == 'plant_product' else [0.8]
                for t in tlist:
                    for target in (['delta'] if ds == 'financial'
                                   else ['contrast', 'phi']):
                        row = {}
                        for series in SERIES_OF_INTEREST:
                            recs = cells.get((ds, mode, eps, t, series, target))
                            if recs:
                                errs = [r[2] for r in recs if r[2] is not None]
                                row[series] = np.mean(errs) if errs else np.nan
                        if not row:
                            continue
                        best = min(row, key=row.get)
                        best_count[best] += 1
                        ps = '  '.join('%s=%.3f%s' % (s, row[s],
                                                      ('*' if s == best and not (
                                                          s == 'privpetal' and False)
                                                       else '')) for s in row)
                        print('%s %s eps%.1f t%.3f %s | %s | best=%s'
                              % (ds, mode, eps, t, target, ps, best))
    print('\nbest-cell tally:', best_count)


if __name__ == '__main__':
    main()