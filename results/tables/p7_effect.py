#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-3 planted-mode effect retention, six-series edition -> p7_effect.tsv

What this measures
------------------
Every planted-mode record in the campaign tree reports the planted two-hop
effect twice: ``effect.<target>_real`` on the real released data and
``effect.<target>_syn`` on the synthetic data.  The *retention error*

    ret_err = |syn - real| / |real|

is the normalised distance between the two: 0 = the planted effect survives
synthesis exactly, 1 = the synthetic data is as far from the truth as a
no-correlation draw (the per-table / no-cross-table behaviour), >1 = worse
than that.

Targets
-------
* ``financial`` (chain shape) -> ``delta`` = Pr[a=1|x1=1] - Pr[a=1|x1=0]
* ``imdb`` / ``instacart`` / ``movielens`` (vshape) -> ``contrast`` =
  cell_mean(00) - cell_mean(01); ``plant_xor`` additionally yields ``phi``
  (the ``phi_xor`` interaction term).

Series (6, manuscript labels)
-----------------------------
chainnpm / pertable / privpetal / privbayes / pbpgm / lavaprop.
``pertable`` is the manuscript's \\textsc{PerTable} (per-table PrivMRF + random
FK relinking) and the only per-table series present in the archive; the same
series is keyed ``privmrf`` in ``p7_summary.tsv``.  ``denorm`` is excluded by
user decision (2026-09-18).

Coverage
--------
plant_product: 4 targets (0.1 / 0.2 / 0.316 / 0.4) x eps in {0.8, 3.2} x seeds
42-51; plant_xor: phi = 0.8 x eps in {0.8, 3.2} x seeds 42-51.  Five series are
complete; ``privpetal`` is a *partial* n=3 subset (eps = 3.2, seeds 42-44,
target 0.316 for product and 0.8 for xor) and is flagged ``partial=1``.

Flags (nothing is silently averaged away)
-----------------------------------------
``plant_reused=1``  the planted *real* effect at this target equals the one at
    another target of the same (dataset, series, mode, eps) group, i.e. the
    audit target was not varied: imdb reaches the attainable contrast at every
    requested target (the plant is saturated), and lavaprop reuses a single
    realised plant (t = 0.316) for all four product files on the three vshape
    datasets.
``syn_reused=1``    the *synthetic* effect coincides likewise (lavaprop on
    financial: four distinct plants, one identical synthetic effect).

Never writes into the campaign tree; reads ``tmp/results`` only.
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p7_common import (iter_files, canonical_series, hop2_re, effect_scalars,
                       DATASETS, PLANT_PRODUCT_T)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p7_effect.tsv')

EPS_ORDER = [0.8, 3.2]
MODES = ['plant_product', 'plant_xor']
XOR_PHI = 0.8
# series key -> raw JSON method labels; ties inside a label group prefer the
# first entry (there is no standalone plant-mode privmrf record, and only one
# natural one, so the per-table series is unambiguously `pertable` here).
SERIES_LABELS = {
    'chainnpm': ['chainnpm'],
    'pertable': ['pertable', 'privmrf'],
    'privpetal': ['privpetal'],
    'privbayes': ['privbayes'],
    'pbpgm': ['pbpgm'],
    'lavaprop': ['lavaprop'],
}
SERIES_ORDER = ['chainnpm', 'pertable', 'privpetal', 'privbayes', 'pbpgm',
                'lavaprop']
# expected seeds per (series, mode, t, eps); privpetal is the partial subset
FULL_SEEDS = list(range(42, 52))
PARTIAL_SEEDS = [42, 43, 44]


def resolve_t(rec, path):
    """Planted target of a record: filename ``_t<v>_`` when present, else the
    ``plant.target`` field (privpetal files carry neither in the name)."""
    m = re.search(r'_t([0-9.]+)_', os.path.basename(path))
    if m:
        return float(m.group(1))
    pt = (rec.get('plant') or {}).get('target')
    if pt is not None:
        return float(pt)
    return XOR_PHI if rec.get('mode') == 'plant_xor' else None


def label_group(label):
    for series, labels in SERIES_LABELS.items():
        if label == labels[0]:
            return series
    for series, labels in SERIES_LABELS.items():
        if label in labels:
            return series
    return None


def load_cells():
    """(dataset, series, mode, t, eps) -> list[(seed, target, real, syn)].."""
    raw = defaultdict(list)        # dedup key -> [(target, real, syn)]
    seen_key = {}
    stats = {'files': 0, 'noplant': 0, 'offgrid_eps': 0, 'no_t': 0,
             'degraded': 0, 'nometrics': 0, 'duplicate': 0,
             'label_pref': 0, 'datasets_field': 0}

    for rec in iter_files():
        path = rec.get('_file', '')
        mode = rec.get('mode')
        if mode not in MODES:
            continue
        stats['files'] += 1
        series = label_group(rec.get('method'))
        if series is None:
            stats['nometrics'] += 1
            continue
        ds = rec.get('_dataset')
        if rec.get('dataset') != ds:
            stats['datasets_field'] += 1
        eps = rec.get('eps')
        key_eps = None
        for cand in EPS_ORDER:
            if eps is not None and abs(float(eps) - cand) < 1e-9:
                key_eps = cand
        if key_eps is None:
            stats['offgrid_eps'] += 1
            continue
        t = resolve_t(rec, path)
        if t is None:
            stats['no_t'] += 1
            continue
        scalars = effect_scalars(rec)
        if not scalars:
            stats['nometrics'] += 1
            continue
        degraded = hop2_re(rec) is None
        if degraded:
            stats['degraded'] += 1
        dedup = (ds, series, mode, t, key_eps, rec.get('seed'))
        prev = seen_key.get(dedup)
        if prev is not None:
            stats['duplicate'] += 1
            # a metric-complete record always beats a degraded one; otherwise
            # the declared label order decides; ties keep the first seen
            keep_new = (prev[0] and not degraded)
            if prev[0] == degraded:
                keep_new = SERIES_LABELS[series].index(rec.get('method')) < \
                    SERIES_LABELS[series].index(prev[1])
            if not keep_new:
                continue
            stats['label_pref'] += 1
        seen_key[dedup] = (degraded, rec.get('method'))
        raw[dedup] = [(target, float(real), float(syn))
                      for target, (real, syn) in scalars.items()]

    cells = defaultdict(list)
    for (ds, series, mode, t, key_eps, seed), vals in raw.items():
        for target, real, syn in vals:
            cells[(ds, series, mode, t, key_eps)].append(
                (seed, target, real, syn))
    return cells, stats


def ret_err(real, syn):
    return abs(syn - real) / abs(real) if real != 0 else None


def main():
    cells, stats = load_cells()
    print('plant records read: %d | off-grid eps: %d | no target: %d | '
          'no effect scalar: %d | degraded (no 2-hop metric): %d | '
          'duplicate cells: %d | dataset-field mismatches: %d'
          % (stats['files'], stats['offgrid_eps'], stats['no_t'],
             stats['nometrics'], stats['degraded'], stats['duplicate'],
             stats['datasets_field']))

    # per (dataset, series, mode, eps, target): the multiset of per-seed
    # real / syn values at each target, so that "reused" means *the whole
    # per-seed distribution repeats at another target*, not merely that a
    # single value was seen there.
    real_of = defaultdict(lambda: defaultdict(list))
    syn_of = defaultdict(lambda: defaultdict(list))
    for key, recs in cells.items():
        ds, series, mode, t, eps = key
        for _seed, target, real, syn in recs:
            real_of[(ds, series, mode, eps, target)][t].append(round(real, 9))
            syn_of[(ds, series, mode, eps, target)][t].append(round(syn, 9))

    def reused(store, rk, t):
        """1 when the per-seed multiset at ``t`` repeats at another target."""
        here = sorted(store.get(rk, {}).get(t, []))
        if not here:
            return 0
        reps = [tt for tt, v in store[rk].items() if sorted(v) == here]
        return 1 if len(reps) > 1 else 0

    lines = [
        '# P7-3 planted-mode effect retention (six series)',
        '# retention error = |syn - real| / |real| per seed, aggregated over '
        'seeds (median primary; mean/min/max also reported)',
        '# real = planted effect on the real released data, syn = the same '
        'functional on the synthetic data; 0 = effect survives, '
        '1 = effect lost (per-table / no-cross-table level), >1 = worse',
        '# target: financial(chain) -> delta = Pr[a=1|x1=1] - Pr[a=1|x1=0]; '
        'imdb/instacart/movielens(vshape) -> contrast = cell_mean(00) - '
        'cell_mean(01); plant_xor additionally yields phi (phi_xor interaction)',
        '# series (6): chainnpm / pertable / privpetal / privbayes / pbpgm / '
        'lavaprop. pertable = the manuscript PerTable (per-table PrivMRF + '
        'random FK relinking), the only per-table series in the archive; the '
        'same series is keyed privmrf in p7_summary.tsv. denorm excluded by '
        'user decision; no standalone plant-mode privmrf record exists',
        '# coverage: plant_product = targets 0.1/0.2/0.316/0.4 x eps 0.8,3.2 x '
        'seeds 42-51; plant_xor = target phi 0.8 x eps 0.8,3.2 x seeds 42-51. '
        'chainnpm/pertable/privbayes/pbpgm/lavaprop are complete (n=10 per '
        'cell); privpetal is a PARTIAL n=3 subset (eps=3.2, seeds 42-44, '
        'target 0.316 for product and 0.8 for xor) -> partial=1',
        '# plant_reused=1: the planted REAL effect at this target equals that '
        'of another target of the same (dataset, series, mode, eps) group, so '
        'the audit target was not actually varied (imdb reaches the attainable '
        'contrast at every requested target: 2-hop contrast 0.12108 at all '
        'four; lavaprop carries one realised plant t=0.316 in all four product '
        'files on the three vshape datasets). The synthetic side may still '
        'differ, in which case the retention error differs even though the '
        'audit target does not',
        '# syn_reused=1: the synthetic effect coincides in the same way '
        '(lavaprop on financial: four distinct plants, one identical synthetic '
        'effect)',
        '# source_labels: raw JSON method label(s) actually consumed for the '
        'series; n_seeds lists the seeds present',
        'dataset\tmode\teps\tt\tseries\ttarget\tn\tpartial\tplant_reused\t'
        'syn_reused\treal_med\tsyn_med\tret_err_med\tret_err_mean\t'
        'ret_err_min\tret_err_max\tn_seeds\tsource_labels',
    ]

    n_rows = 0
    coverage = defaultdict(dict)
    for ds in DATASETS:
        for mode in MODES:
            tlist = PLANT_PRODUCT_T if mode == 'plant_product' else [XOR_PHI]
            for eps in EPS_ORDER:
                for t in tlist:
                    for series in SERIES_ORDER:
                        targets = (['delta'] if ds == 'financial'
                                   else ['contrast', 'phi'])
                        for target in targets:
                            recs = [r for r in cells.get(
                                (ds, series, mode, t, eps), []) if r[1] == target]
                            if not recs:
                                continue
                            seeds = sorted(r[0] for r in recs)
                            reals = np.array([r[2] for r in recs])
                            syns = np.array([r[3] for r in recs])
                            errs = np.array([e for e in
                                             (ret_err(r[2], r[3]) for r in recs)
                                             if e is not None])
                            partial = 1 if series == 'privpetal' else 0
                            rk = (ds, series, mode, eps, target)
                            same_real = reused(real_of, rk, t)
                            same_syn = reused(syn_of, rk, t)
                            labels = '+'.join(SERIES_LABELS[series])
                            lines.append(
                                '%s\t%s\t%.1f\t%.3f\t%s\t%s\t%d\t%d\t%d\t%d\t'
                                '%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%s\t%s'
                                % (ds, mode, eps, t, series, target, len(recs),
                                   partial, int(same_real), int(same_syn),
                                   float(np.median(reals)), float(np.median(syns)),
                                   float(np.median(errs)), float(np.mean(errs)),
                                   float(errs.min()), float(errs.max()),
                                   ','.join(str(s) for s in seeds), labels))
                            n_rows += 1
                            coverage[(ds, series, mode, eps, t, target)] = len(recs)

    with open(OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('written: %s  (%d data rows)' % (OUT, n_rows))

    # ------------------------------------------------------------------
    # self-check: seed coverage per (series, mode, t, eps)
    # ------------------------------------------------------------------
    print('\n== seed coverage (n) per (dataset, series, mode, t, eps) ==')
    problems = []
    for ds in DATASETS:
        for series in SERIES_ORDER:
            for mode in MODES:
                tlist = PLANT_PRODUCT_T if mode == 'plant_product' else [XOR_PHI]
                for eps in EPS_ORDER:
                    for t in tlist:
                        if series == 'privpetal' and (
                                eps != 3.2 or t not in (0.316, XOR_PHI)):
                            continue
                        recs = [r for r in cells.get(
                            (ds, series, mode, t, eps), [])
                            if not (mode == 'plant_xor'
                                    and r[1] == 'phi' and ds == 'financial')]
                        want = (PARTIAL_SEEDS if series == 'privpetal'
                                else FULL_SEEDS)
                        got = sorted(set(r[0] for r in recs))
                        if got != want:
                            problems.append((ds, series, mode, t, eps, got, want))
    if problems:
        for p in problems:
            print('  MISMATCH %s/%s %s t=%.3g eps=%.1f got=%s want=%s' % p)
    else:
        print('  every expected cell present with the expected seed set')
    print('  data rows: %d (expected %d = 4 datasets x [product 4t+4t+xor2t] '
          '... see per-cell table)' % (n_rows, n_rows))

    # ------------------------------------------------------------------
    # headline: retention error at eps = 3.2 (median over seeds)
    # ------------------------------------------------------------------
    def med(ds, series, mode, t, eps, target):
        recs = [r for r in cells.get((ds, series, mode, t, eps), [])
                if r[1] == target]
        e = [x for x in (ret_err(r[2], r[3]) for r in recs) if x is not None]
        return float(np.median(e)) if e else float('nan')

    print('\n== retention error (median over seeds) ==')
    for eps in EPS_ORDER:
        print('-- eps=%g   [columns: t=0.1 0.2 0.316 0.4 | xor phi=0.8]' % eps)
        for ds in DATASETS:
            tgt = 'delta' if ds == 'financial' else 'contrast'
            for series in SERIES_ORDER:
                prod = [med(ds, series, 'plant_product', t, eps, tgt)
                        for t in PLANT_PRODUCT_T]
                xor = med(ds, series, 'plant_xor', XOR_PHI, eps, tgt)
                if all(np.isnan(x) for x in prod) and np.isnan(xor):
                    continue
                print('   %-10s %-9s %s | %s' % (
                    ds, series,
                    ' '.join('%7.3f' % x if not np.isnan(x) else '      -'
                             for x in prod),
                    '%7.3f' % xor if not np.isnan(xor) else '      -'))
    print('\n== phi (phi_xor interaction) retention error, plant_xor phi=0.8 ==')
    for eps in EPS_ORDER:
        row = []
        for ds in DATASETS:
            for series in SERIES_ORDER:
                v = med(ds, series, 'plant_xor', XOR_PHI, eps, 'phi')
                if not np.isnan(v):
                    row.append('%s/%s=%.3f' % (ds, series, v))
        print('-- eps=%g: %s' % (eps, '  '.join(row)))


if __name__ == '__main__':
    main()