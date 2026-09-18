#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Appendix RE table (paper-ready wide layout) -> tab_re_appendix.tsv

Caliber
-------
Natural (non-planted) mode, the *large-query* median relative error of the
2-hop query class, i.e. ``re.re_median_large[2hop_topbot]`` on the chain shape
(financial) and ``re_median_large[cross_r1r2]`` on the vshape datasets
(imdb / instacart / movielens).  One value per (dataset, method, eps), median
over the archived seeds 42-51 (n = 10).

Provenance
----------
Every number is transcribed 1:1 from the ``median`` column of
``results/tables/p7_summary.tsv``, which ``results/tables/p7_summary.py``
recomputes from the campaign tree ``tmp/results`` (read only) together with
its own bootstrap CI.  This script never reads the campaign tree itself, so the
appendix table and the summary cannot drift apart: it *fails* if any of the
4 x 6 x 6 cells is absent, instead of silently emitting a hole.

Series labels
-------------
``pertable`` is the manuscript's \\textsc{PerTable} (per-table PrivMRF + random
FK relinking); it is the only per-table series in the archive and is keyed
``privmrf`` inside ``p7_summary.tsv``.  ``denorm`` is excluded by user decision.

2026-09-18 data repair
----------------------
``movielens / chainnpm / eps = 3.2`` was a degenerate cell (``hyper.tau = 12``
instead of 556, ``c_syn`` = 72K against 1.0M true rows, ``size_tv`` = 1.0,
2.0 s runtime).  It was re-run and the archive updated; the row below carries
the repaired value (see the ``note`` column).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, 'p7_summary.tsv')
OUT = os.path.join(HERE, 'tab_re_appendix.tsv')

DATASETS = ['financial', 'imdb', 'instacart', 'movielens']
EPS_ORDER = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
# (tsv series key, manuscript label, order)
SERIES = [('chainnpm', 'Chain-NPM'), ('pertable', 'PerTable'),
          ('privpetal', 'PrivPetal'), ('privbayes', 'PrivBayes'),
          ('pbpgm', 'PB-PGM'), ('lavaprop', 'PrivLava')]
SUMMARY_KEY = {'chainnpm': 'chainnpm', 'pertable': 'privmrf',
               'privpetal': 'privpetal', 'privbayes': 'privbayes',
               'pbpgm': 'pbpgm', 'lavaprop': 'lavaprop'}
HOP2 = {ds: ('2hop_topbot' if ds == 'financial' else 'cross_r1r2')
        for ds in DATASETS}
NOTE_FIX = '2026-09-18 repaired (was tau=12/c_syn=72K); median unchanged'


def read_summary():
    vals = {}
    for line in open(SUMMARY):
        if line.startswith('#'):
            continue
        f = line.rstrip('\n').split('\t')
        if f[0] == 'dataset':
            continue
        ds, series, eps = f[0], f[1], float(f[2])
        vals[(ds, series, eps)] = {'n': int(f[3]), 'mean': float(f[4]),
                                   'median': float(f[5]), 'std': float(f[6]),
                                   'lo': float(f[7]), 'hi': float(f[8])}
    return vals


def main():
    vals = read_summary()
    missing = []
    for ds in DATASETS:
        for series, _ in SERIES:
            for eps in EPS_ORDER:
                if (ds, SUMMARY_KEY[series], eps) not in vals:
                    missing.append((ds, series, eps))
    if missing:
        sys.exit('refusing to write a table with holes: %d missing cells, '
                 'first %s' % (len(missing), missing[0]))

    lines = [
        '# Appendix table data: natural mode, 2-hop LARGE-query median relative '
        'error (RE)',
        '# query class: %s' % ' | '.join(
            '%s=%s' % (ds, HOP2[ds]) for ds in DATASETS),
        '# median over seeds 42-51 (n=10) of re.re_median_large (prereg '
        'query_seed=123, 150 queries per class, large_count=50)',
        '# provenance: transcribed 1:1 from the median column of '
        'results/tables/p7_summary.tsv, which results/tables/p7_summary.py '
        'recomputes from tmp/results (read only). This script reads only '
        'p7_summary.tsv, so the two cannot drift',
        '# series: pertable = the manuscript PerTable (per-table PrivMRF + '
        'random FK relinking); it appears as privmrf in p7_summary.tsv. '
        'denorm excluded by user decision',
        '# n: seeds behind each cell (must be 10); values are medians, 4 dp',
        '# note on 2026-09-18: the movielens/chainnpm/eps=3.2 cell was a '
        'degenerate archive record (tau=12 vs 556, c_syn=72,408 vs 1,000,209, '
        'size_tv=1.0). It was re-run; the value below is the repaired one. Its '
        'median is unchanged (0.5158) while the mean moves 0.5569 -> 0.5164 '
        'and the std 0.1419 -> 0.0579',
        'dataset\tmethod\tmethod_label\t' +
        '\t'.join('eps_%g' % e for e in EPS_ORDER) +
        '\t' + '\t'.join('n_%g' % e for e in EPS_ORDER) + '\tnote',
    ]
    for ds in DATASETS:
        for series, label in SERIES:
            key = SUMMARY_KEY[series]
            med, ns = [], []
            for eps in EPS_ORDER:
                c = vals[(ds, key, eps)]
                med.append('%.4f' % c['median'])
                ns.append(str(c['n']))
            note = (NOTE_FIX if (ds, series) == ('movielens', 'chainnpm')
                    else '')
            lines.append('%s\t%s\t%s\t%s\t%s\t%s' % (
                ds, series, label, '\t'.join(med), '\t'.join(ns), note))
    with open(OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('written: %s (%d data rows)' % (OUT, len(DATASETS) * len(SERIES)))

    # ---- self-check: echo the key cell and the envelope ----
    allv = [vals[(ds, SUMMARY_KEY[s], e)]['median']
            for ds in DATASETS for s, _ in SERIES for e in EPS_ORDER]
    print('cells: %d, n(present)=%d, all n==10: %s'
          % (len(allv), len(vals),
             all(vals[(ds, SUMMARY_KEY[s], e)]['n'] == 10
                 for ds in DATASETS for s, _ in SERIES for e in EPS_ORDER)))
    c = vals[('movielens', 'chainnpm', 3.2)]
    print('movielens/chainnpm/eps3.2 -> median=%.4f mean=%.4f std=%.4f '
          'boot=[%.4f, %.4f] n=%d'
          % (c['median'], c['mean'], c['std'], c['lo'], c['hi'], c['n']))
    print('median envelope: %.4f .. %.4f' % (min(allv), max(allv)))


if __name__ == '__main__':
    main()