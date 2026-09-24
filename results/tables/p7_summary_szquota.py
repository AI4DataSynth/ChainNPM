#!/usr/bin/env python3
"""P7-1 szquota-arm summary: natural-mode 2-hop RE with Chain-NPM swapped to
the reported arm.

Same output schema as ``p7_summary.tsv`` and the same aggregation code path
(bootstrap CI over the per-seed means, RNG seed 42):

    dataset  series  eps  n  mean  median  std  boot_lo  boot_hi

Every series except ``chainnpm`` is read from the released archive
(``results/by_dataset``) through ``p7_common.load_rows``; the ``chainnpm``
rows are the reported arm, read from the szquota grids through
``szquota_common`` (``results/arms/absorb`` in the release repository,
``tmp/absorb_proto/campaign_*`` in the internal analysis tree -- whichever
exists).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from p7_common import (hop2_re, DATASETS, META, EPS_ORDER,   # noqa: E402
                       HOP2_KEY)
import szquota_common                                        # noqa: E402

OUT = os.path.join(HERE, 'p7_summary_szquota.tsv')
NBOOT = 1000


def bootstrap95(vals, rng):
    arr = np.asarray(vals, dtype=float)
    means = np.empty(NBOOT)
    n = len(arr)
    for b in range(NBOOT):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rng = np.random.default_rng(42)
    agg = {}
    dropped = 0
    for d in szquota_common.szquota_records():
        if d.get('mode') != 'natural':
            continue
        v = hop2_re(d)
        if v is None:
            dropped += 1
            continue
        key = (d.get('dataset') or d['_dataset'], d.get('method'),
               d.get('eps'))
        agg.setdefault(key, []).append(float(v))

    descr = ' | '.join('%s=%s' % (ds, HOP2_KEY[ds]) for ds in DATASETS)
    lines = ['# P7-1 szquota-arm summary: natural mode, 2-hop RE = '
             're_median_large[%s] (large queries, prereg query_seed=123)'
             % descr,
             '# chainnpm = reported arm (row-level quota calibration) read from '
             'the szquota grids (results/arms/absorb or tmp/absorb_proto/'
             'campaign_*); other series = released archive results/by_dataset',
             '# bootstrap: %d resamples of per-seed means, RNG seed 42; '
             'series labels follow final figures (PrivMRF = privmrf+pertable, '
             'pertable preferred)' % NBOOT,
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
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    print('written: %s' % OUT)
    print('cells: %d | dropped (no 2-hop re): %d' % (len(agg), dropped))


if __name__ == '__main__':
    main()