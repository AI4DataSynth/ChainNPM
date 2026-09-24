"""szquota-arm mirror of p7_common.load_rows for the manuscript pipeline.

The released archive (results/by_dataset) carries the released Chain-NPM
grid; the szquota campaign grids under tmp/absorb_proto/campaign_szquota{2}
(+ campaign_fin_szquota once it lands) carry the same cells with the
``szquota`` absorb arm.  ``szquota_records()`` returns the load_rows() record
list with the chainnpm natural rows of the four datasets replaced by the
szquota-arm grid values; a dataset whose grid directory is absent (financial
until the campaign lands) falls back to the released chainnpm rows, so the
pipeline keeps working with a placeholder.

Record structure is identical to p7_common.load_rows() output (dict with
``_dataset``, ``method`` (canonical series key), ``eps``, ``seed``, ``mode``,
``shape``, ``re``, ``size_tv``), so downstream plot/table scripts consume it
without change; only the loader call differs.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # results/tables
RESULTS_DIR = os.path.dirname(HERE)                        # results
ROOT = os.path.dirname(RESULTS_DIR)                        # project root
sys.path.insert(0, HERE)
from p7_common import load_rows                             # noqa: E402

_LIVE_NAT = {'instacart': 'campaign_szquota/instacart',
             'movielens': 'campaign_szquota2/movielens',
             'imdb': 'campaign_szquota2/imdb',
             'financial': 'campaign_fin_szquota/financial'}


def _camp_dir(ds, which):
    """Grid directory for a dataset, preferring the shipped archive layout.

    In the release repository the reported-arm cells live under
    ``results/arms/absorb/{natural,plant}/<ds>``; in the internal analysis
    tree they live under ``tmp/absorb_proto/campaign_*``.  Whichever exists is
    used, so the same scripts run in both trees.
    """
    repo = os.path.join(ROOT, 'results', 'arms', 'absorb', which, ds)
    if os.path.isdir(repo):
        return repo
    live = (_LIVE_NAT[ds] if which == 'natural'
            else 'campaign_plant_szquota/%s' % ds)
    return os.path.join(ROOT, 'tmp', 'absorb_proto', live)


def _load_grid():
    """(dataset, eps, seed) -> record, from the szquota campaign grids."""
    repl = {}
    for ds in _LIVE_NAT:
        cdir = _camp_dir(ds, 'natural')
        if not os.path.isdir(cdir):
            continue
        for fp in sorted(glob.glob(os.path.join(cdir, '*.json'))):
            name = os.path.basename(fp)
            if 'budget' in name:          # budget_szquota arm is NOT the pick
                continue
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            if d.get('mode') != 'natural':
                continue
            d['_dataset'] = ds
            d['_file'] = fp
            repl[(ds, d.get('eps'), d.get('seed'))] = d
    return repl


def szquota_records(pref_order=None):
    """load_rows() with chainnpm natural cells swapped to the szquota arm."""
    repl = _load_grid()
    out = []
    for d in load_rows(pref_order=pref_order):
        if d.get('method') == 'chainnpm' and d.get('mode') == 'natural' \
                and (d['_dataset'], d.get('eps'), d.get('seed')) in repl:
            continue
        out.append(d)
    for (ds, eps, seed), d in repl.items():
        dd = dict(d)
        dd['method'] = 'chainnpm'
        if dd.get('dataset') != ds:
            dd['dataset'] = ds
        out.append(dd)
    return out


def szquota_plant_records():
    """Yield campaign_plant_szquota records (chainnpm plant mode, szquota arm).

    Record structure matches p7_common.iter_files() output (plain JSON with
    ``_dataset``/``_file`` attached), so p7_effect.py can consume it as the
    planted-mode source for the szquota arm.
    """
    for ds in ('imdb', 'instacart', 'movielens'):
        cdir = _camp_dir(ds, 'plant')
        if not os.path.isdir(cdir):
            continue
        for fp in sorted(glob.glob(os.path.join(cdir, '*.json'))):
            name = os.path.basename(fp)
            if 'family' in name:
                continue
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            d['_dataset'] = ds
            d['_file'] = fp
            yield d