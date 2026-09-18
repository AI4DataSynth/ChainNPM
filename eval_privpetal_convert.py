"""Convert PrivPetal (official, P4 adapter) synthetic outputs into the
campaign result-JSON schema and re-evaluate with the SAME query workload
(prereg.QUERY_SEED=123) as all other methods.

Reads
  cluster sync dir : tmp/results/<name>/privpetal_runs/<exp>/
                     {syn_*.csv, meta.json}   (scp'd back from the cluster)
  GT               : tmp/data/<name>/processed/              (natural)
                     tmp/data_planted/<name>/<mode>/         (plant_*)
Writes
  tmp/results/<name>/privpetal_<mode>_eps<e>_seed<s>.json
with fields compatible with p4_common outputs (re / effect / size_tv /
counts / runtime_sec ...).

Usage:
  python eval_privpetal_convert.py --name movielens --mode natural \
      --eps 3.2 --seed 42
  python eval_privpetal_convert.py --all
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import prereg
import eval_newdata as ev
import p4_common as p4
import chain_npm_financial as d_fin
import chain_npm_movielens as d_mov
import chain_npm_imdb as d_imdb
import chain_npm_instacart as d_inst

PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DRIVERS = {'financial': d_fin.CFG, 'movielens': d_mov.CFG,
           'imdb': d_imdb.CFG, 'instacart': d_inst.CFG}
SHAPES = {'financial': 'chain', 'movielens': 'vshape', 'imdb': 'vshape',
          'instacart': 'vshape'}


def run_dir(name, mode, eps, seed):
    exp = 'P4_%s_%s_eps%.1f_seed%d' % (name, mode, eps, seed)
    return os.path.join(PROJECT_ROOT, 'tmp', 'results', name,
                        'privpetal_runs', exp)


def gt_dir(name, mode):
    if mode == 'natural':
        return os.path.join(PROJECT_ROOT, 'tmp', 'data', name, 'processed')
    return os.path.join(PROJECT_ROOT, 'tmp', 'data_planted', name, mode)


def load_tables_vshape(cfg, rdir, ddir, syn=True):
    t1, t2, tc = cfg['tables']['r1'], cfg['tables']['r2'], cfg['tables']['c']
    base = rdir if syn else ddir
    pf = 'syn_r1.csv' if syn else t1['file']
    r1 = pd.read_csv(os.path.join(base, pf))
    r2 = pd.read_csv(os.path.join(base, ('syn_r2.csv' if syn
                                         else t2['file'])))
    c = pd.read_csv(os.path.join(base, ('syn_c.csv' if syn else tc['file'])))
    if syn:
        # adapter emits child in canonical [pk, fk1, fk2, attrs]
        arr_c = c.values.astype(int)
    else:
        # processed layout [pk, attrs..., fk1, fk2] -> canonical
        cols = [tc['pk'], tc['fk1'], tc['fk2']] + tc['attrs']
        arr_c = c.reindex(columns=cols).values.astype(int)
    return {'r1': r1.values.astype(int), 'r2': r2.values.astype(int),
            'c': arr_c}


def load_tables_chain(cfg, rdir, ddir, syn=True):
    tt, tm, tb = (cfg['tables']['top'], cfg['tables']['mid'],
                  cfg['tables']['bot'])
    base = rdir if syn else ddir
    top = pd.read_csv(os.path.join(base, 'syn_top.csv' if syn
                                   else tt['file']))
    mid = pd.read_csv(os.path.join(base, 'syn_mid.csv' if syn
                                   else tm['file']))
    bot = pd.read_csv(os.path.join(base, 'syn_bot.csv' if syn
                                   else tb['file']))
    if not syn:  # processed layout: mid/bot = [pk, attrs..., fk]
        mid = mid.reindex(columns=[tm['pk'], tm['fk']] + tm['attrs'])
        bot = bot.reindex(columns=[tb['pk'], tb['fk']] + tb['attrs'])
    return {'top': top.values.astype(int), 'mid': mid.values.astype(int),
            'bot': bot.values.astype(int)}


def load_domains_vshape(cfg, ddir):
    return p4.load_domains(ddir, cfg['tables'])


def convert(name, mode, eps, seed):
    cfg = DRIVERS[name]
    shape = SHAPES[name]
    rdir = run_dir(name, mode, eps, seed)
    ddir = gt_dir(name, mode)
    if not os.path.isdir(rdir):
        print('MISSING run dir:', rdir)
        return None
    meta_p = os.path.join(rdir, 'meta.json')
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    if meta.get('status') != 'ok':
        print('INCOMPLETE run (no meta/status ok):', rdir)
        return None

    if shape == 'vshape':
        attrs = {r: cfg['tables'][r]['attrs'] for r in ('r1', 'r2', 'c')}
        gt = load_tables_vshape(cfg, rdir, ddir, syn=False)
        syn = load_tables_vshape(cfg, rdir, ddir, syn=True)
        domains = load_domains_vshape(cfg, ddir)
        qdom = {(role, a): domains[role][a]
                for role in attrs for a in attrs[role]}
        qrng = np.random.default_rng(prereg.QUERY_SEED)
        queries = ev.make_queries_vshape(qrng, attrs, qdom, prereg.N_QUERIES)
        gt_views = ev.build_vshape_views(gt, attrs)
        syn_views = ev.build_vshape_views(syn, attrs)
        sizes_gt = ev.group_sizes(gt['c'][:, 1])
        sizes_syn = ev.group_sizes(syn['c'][:, 1])
        tau = prereg.pick_tau(sizes_gt)
        result = {
            'dataset': name, 'shape': 'vshape', 'method': 'privpetal',
            'mode': mode, 'eps': eps, 'seed': seed,
            'runtime_sec': meta.get('total_wall_sec'),
            'hyper': {'sample_sizes': {
                'r1': None, 'r2': None}, 'theta': None, 'tau': tau,
                'delta': prereg.delta_for(int(gt['c'].shape[0])),
                'privpetal_budget': meta.get('budget_total')},
            'prereg': {'query_seed': prereg.QUERY_SEED,
                       'n_queries': prereg.N_QUERIES,
                       'large_count': prereg.LARGE_COUNT,
                       'tau_choices': list(prereg.TAU_CHOICES)},
            'counts': {'r1_gt': int(gt['r1'].shape[0]),
                       'r2_gt': int(gt['r2'].shape[0]),
                       'c_gt': int(gt['c'].shape[0]),
                       'r1_syn': int(syn['r1'].shape[0]),
                       'r2_syn': int(syn['r2'].shape[0]),
                       'c_syn': int(syn['c'].shape[0]),
                       'syn_orphans_c': int(syn_views['c']['orphans'])},
            're': ev.query_re(gt_views, syn_views, queries,
                              prereg.LARGE_COUNT),
            'size_tv': ev.size_tv(sizes_gt, sizes_syn, tau),
            'mean_size': {'gt': float(sizes_gt.mean()),
                          'syn': float(sizes_syn.mean())
                          if len(sizes_syn) else 0.0},
            'privpetal_meta': {
                'pass_timings_sec': meta.get('pass_timings_sec'),
                'counts': meta.get('counts'),
                'host': meta.get('host'), 'gpu': meta.get('gpu'),
                'tuple_num': meta.get('tuple_num'),
                'process_num': meta.get('process_num')},
        }
        # sample sizes used (from per-pass cfg if present)
        for tag in ('r1', 'r2'):
            c = meta.get('cfg_' + tag)
            if c:
                result['hyper']['sample_sizes'][tag] = \
                    max(c.get('size_bins', [0]))
                result['hyper']['theta'] = c.get('theta')
        mode_kind = mode[len('plant_'):] if mode.startswith('plant_') \
            else 'natural'
        result['effect'] = p4.vshape_effects(gt_views, syn_views, cfg,
                                             mode_kind)
    else:  # chain
        attrs = {r: cfg['tables'][r]['attrs']
                 for r in ('top', 'mid', 'bot')}
        gt = load_tables_chain(cfg, rdir, ddir, syn=False)
        syn = load_tables_chain(cfg, rdir, ddir, syn=True)
        domains = p4.load_domains(ddir, cfg['tables'])
        qdom = {(role, a): domains[role][a]
                for role in attrs for a in attrs[role]}
        qrng = np.random.default_rng(prereg.QUERY_SEED)
        queries = ev.make_queries_chain(qrng, attrs, qdom, prereg.N_QUERIES)
        gt_views = ev.build_chain_views(gt, attrs)
        syn_views = ev.build_chain_views(syn, attrs)
        sizes2_gt = ev.group_sizes(gt['bot'][:, 1])
        sizes1_gt = ev.group_sizes(gt['mid'][:, 1])
        tau2 = prereg.pick_tau(sizes2_gt)
        tau1 = prereg.pick_tau(sizes1_gt)
        result = {
            'dataset': name, 'shape': 'chain', 'method': 'privpetal',
            'mode': mode, 'eps': eps, 'seed': seed,
            'runtime_sec': meta.get('total_wall_sec'),
            'hyper': {'tau1': tau1, 'tau2': tau2, 'theta': None,
                      'delta': prereg.delta_for(int(gt['bot'].shape[0])),
                      'privpetal_budget': meta.get('budget_total')},
            'prereg': {'query_seed': prereg.QUERY_SEED,
                       'n_queries': prereg.N_QUERIES,
                       'large_count': prereg.LARGE_COUNT,
                       'tau_choices': list(prereg.TAU_CHOICES)},
            'counts': {'top_gt': int(gt['top'].shape[0]),
                       'mid_gt': int(gt['mid'].shape[0]),
                       'bot_gt': int(gt['bot'].shape[0]),
                       'top_syn': int(syn['top'].shape[0]),
                       'mid_syn': int(syn['mid'].shape[0]),
                       'bot_syn': int(syn['bot'].shape[0]),
                       'syn_orphans_bot': int(syn_views['bot']['orphans'])},
            're': ev.query_re(gt_views, syn_views, queries,
                              prereg.LARGE_COUNT),
            'size_tv': {
                'bot_per_mid': ev.size_tv(
                    sizes2_gt, ev.group_sizes(syn['bot'][:, 1]), tau2),
                'mid_per_top': ev.size_tv(
                    sizes1_gt, ev.group_sizes(syn['mid'][:, 1]), tau1)},
            'mean_size': {'bot_per_mid_gt': float(sizes2_gt.mean()),
                          'mid_per_top_gt': float(sizes1_gt.mean())},
            'privpetal_meta': {
                'pass_timings_sec': meta.get('pass_timings_sec'),
                'counts': meta.get('counts'),
                'host': meta.get('host'), 'gpu': meta.get('gpu'),
                'tuple_num': meta.get('tuple_num'),
                'process_num': meta.get('process_num')},
        }
        for tag in ('stage1', 'stage2'):
            c = meta.get('cfg_' + tag)
            if c:
                result['hyper']['theta'] = c.get('theta')
        result['effect'] = p4.chain_effects(gt_views, syn_views, cfg)

    # plant audit passthrough (achieved effect is the reference)
    audit_p = os.path.join(ddir, 'plant_audit.json')
    if mode != 'natural' and os.path.exists(audit_p):
        result['plant'] = json.load(open(audit_p))

    out = os.path.join(PROJECT_ROOT, 'tmp', 'results', name,
                       'privpetal_%s_eps%.1f_seed%d.json'
                       % (mode, eps, seed))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print('wrote', out)
    if result.get('re'):
        print('  re_median_large:', json.dumps(
            result['re']['re_median_large'], default=float))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', type=str)
    ap.add_argument('--mode', type=str, default='natural')
    ap.add_argument('--eps', type=float, default=3.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force', action='store_true',
                    help='re-convert even if output json exists')
    ap.add_argument('--all', action='store_true',
                    help='convert every run dir present under '
                         'tmp/results/*/privpetal_runs/')
    args = ap.parse_args()
    if args.all:
        pattern = os.path.join(PROJECT_ROOT, 'tmp', 'results', '*',
                               'privpetal_runs', 'P4_*')
        for d in sorted(glob.glob(pattern)):
            exp = os.path.basename(d)
            try:
                _, name, mode, eps_s, seed_s = exp.split('_')
                if mode.startswith('plant'):
                    name = '%s_%s' % (name, mode.split(':')[0])
                    # re-split robustly
                eps = float(eps_s[3:])
                seed = int(seed_s[4:])
            except Exception:
                # exp format: P4_<name>_<mode>_eps<e>_seed<s>; mode may
                # contain '_' (plant_product) -> parse from the right
                parts = exp.split('_')
                seed = int(parts[-1][4:])
                eps = float(parts[-2][3:])
                name = parts[1]
                mode = '_'.join(parts[2:-2])
            outp = os.path.join(PROJECT_ROOT, 'tmp', 'results', name,
                                'privpetal_%s_eps%.1f_seed%d.json'
                                % (mode, eps, seed))
            if os.path.exists(outp) and not args.force:
                continue
            convert(name, mode, eps, seed)
    else:
        convert(args.name, args.mode, args.eps, args.seed)
