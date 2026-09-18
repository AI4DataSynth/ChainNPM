"""Generate planted-variant CSVs for the PrivPetal baseline (P4 campaign).

Replicates EXACTLY the planting protocol of p4_common.run_vshape_pipeline /
run_chain_pipeline (calibrate seeds=(0,1), iters=12, tol=0.02,
balance=(0.0,1.0); plant seed fixed to 42) so the PrivPetal runs consume the
same planted data as the other methods.  Only the child/leaf table changes
(a_attr binarized); parents and FK assignments are untouched.

Output: tmp/data_planted/<name>/<mode>/ with the SAME csv layout and domain
jsons as tmp/data/<name>/processed/ (child keeps its original column order;
a_attr domain size becomes 2) plus plant_audit.json.

Usage:
  python make_planted_csvs.py --name movielens --mode plant_xor --target 0.8
  python make_planted_csvs.py --all     # 4 datasets x {product 0.316, xor 0.8}
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

import plant_audit as pa
import chain_npm_financial as d_fin
import chain_npm_movielens as d_mov
import chain_npm_imdb as d_imdb
import chain_npm_instacart as d_inst

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DRIVERS = {'financial': d_fin.CFG, 'movielens': d_mov.CFG,
           'imdb': d_imdb.CFG, 'instacart': d_inst.CFG}


def _proc_dir(name):
    return os.path.join(PROJECT_ROOT, 'tmp', 'data', name, 'processed')


def _out_dir(name, mode):
    return os.path.join(PROJECT_ROOT, 'tmp', 'data_planted', name, mode)


def make_vshape(name, cfg, mode, target, seed=42):
    d = _proc_dir(name)
    t1, t2, tc = cfg['tables']['r1'], cfg['tables']['r2'], cfg['tables']['c']
    r1 = pd.read_csv(os.path.join(d, t1['file']))
    r2 = pd.read_csv(os.path.join(d, t2['file']))
    c = pd.read_csv(os.path.join(d, tc['file']))
    plant = cfg['plant']
    plant_mode = mode[len('plant_'):]
    kw = dict(pk1=t1['pk'], pk2=t2['pk'], pkc=tc['pk'],
              fk1=tc['fk1'], fk2=tc['fk2'],
              x1=plant['x1'], x2=plant['x2'], a_attr=plant['a_attr'],
              u_attr=plant.get('u_attr'), balance=(0.0, 1.0))
    t0 = time.time()
    p, calib = pa.calibrate_vshape(r1, r2, c, target=target, mode=plant_mode,
                                   seeds=(0, 1), iters=12, tol=0.02, **kw)
    res = pa.plant_vshape(r1, r2, c, p=p, mode=plant_mode, seed=seed, **kw)
    audit = res['audit']
    audit.update(calib_p=float(p), calib_effect=float(calib),
                 target=float(target), plant_seed=int(seed),
                 calibrate_sec=round(time.time() - t0, 1))

    out = _out_dir(name, mode)
    os.makedirs(out, exist_ok=True)
    # parents untouched: copy csv + domain
    for t in (t1, t2):
        shutil.copy(os.path.join(d, t['file']), os.path.join(out, t['file']))
        shutil.copy(os.path.join(d, t['domain_file']),
                    os.path.join(out, t['domain_file']))
    # child: original processed column order, a_attr now binary
    childp = res['child'].reindex(columns=list(c.columns))
    childp.to_csv(os.path.join(out, tc['file']), index=False)
    dom = json.load(open(os.path.join(d, tc['domain_file'])))
    dom[plant['a_attr']] = {'size': 2}
    json.dump(dom, open(os.path.join(out, tc['domain_file']), 'w'))
    json.dump(audit, open(os.path.join(out, 'plant_audit.json'), 'w'),
              indent=2)
    print('[%s %s] p=%.4f calib=%.4f effect=%.4f forced=%d %.1fs -> %s'
          % (name, mode, p, calib, audit['effect'], audit['forced_swaps'],
             audit['calibrate_sec'], out))


def make_chain(name, cfg, mode, target, seed=42):
    d = _proc_dir(name)
    tt, tm, tb = (cfg['tables']['top'], cfg['tables']['mid'],
                  cfg['tables']['bot'])
    top = pd.read_csv(os.path.join(d, tt['file']))
    mid = pd.read_csv(os.path.join(d, tm['file']))
    bot = pd.read_csv(os.path.join(d, tb['file']))
    plant = cfg['plant']
    plant_mode = mode[len('plant_'):]
    kw = dict(pk0=tt['pk'], pk1=tm['pk'], fk1=tm['fk'], fk2=tb['fk'],
              x1=plant['x1'], x2=plant['x2'], a_attr=plant['a_attr'],
              u_attr=plant.get('u_attr'), mid_attr=plant.get('mid_attr'),
              balance=(0.0, 1.0))
    t0 = time.time()
    p, calib = pa.calibrate(top, mid, bot, target=target, mode=plant_mode,
                            seeds=(0, 1), iters=12, tol=0.02, **kw)
    res = pa.plant(top, mid, bot, p=p, mode=plant_mode, seed=seed, **kw)
    audit = res['audit']
    audit.update(calib_p=float(p), calib_effect=float(calib),
                 target=float(target), plant_seed=int(seed),
                 calibrate_sec=round(time.time() - t0, 1))

    out = _out_dir(name, mode)
    os.makedirs(out, exist_ok=True)
    for t in (tt, tm):
        shutil.copy(os.path.join(d, t['file']), os.path.join(out, t['file']))
        shutil.copy(os.path.join(d, t['domain_file']),
                    os.path.join(out, t['domain_file']))
    botp = res['r2'].reindex(columns=list(bot.columns))
    botp.to_csv(os.path.join(out, tb['file']), index=False)
    dom = json.load(open(os.path.join(d, tb['domain_file'])))
    dom[plant['a_attr']] = {'size': 2}
    json.dump(dom, open(os.path.join(out, tb['domain_file']), 'w'))
    json.dump(audit, open(os.path.join(out, 'plant_audit.json'), 'w'),
              indent=2)
    print('[%s %s] p=%.4f calib=%.4f effect=%.4f forced=%d %.1fs -> %s'
          % (name, mode, p, calib, audit['effect'], audit['forced_swaps'],
             audit['calibrate_sec'], out))


def make_one(name, mode, target, seed=42):
    cfg = DRIVERS[name]
    if name == 'financial':
        make_chain(name, cfg, mode, target, seed)
    else:
        make_vshape(name, cfg, mode, target, seed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', type=str)
    ap.add_argument('--mode', type=str)
    ap.add_argument('--target', type=float)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if args.all:
        for name in ('financial', 'movielens', 'imdb', 'instacart'):
            make_one(name, 'plant_product', 0.316, args.seed)
            make_one(name, 'plant_xor', 0.8, args.seed)
    else:
        make_one(args.name, args.mode, args.target, args.seed)
