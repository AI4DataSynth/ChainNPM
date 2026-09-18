"""Shared P4 pipeline: load -> (plant) -> synthesize -> evaluate -> dump.

The four dataset drivers (chain_npm_{financial,movielens,imdb,instacart}.py)
each declare a schema config and call run_chain_pipeline / run_vshape_pipeline
here, so the protocol (query workload, effect metrics, recording) is identical
across datasets and methods.
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import prereg
import eval_newdata as ev
import baselines as bl
import plant_audit as pa
from chain_npm_vshape import ChainNPMV
from chain_npm_3level import ChainNPM3

# Root of the data/result tree (expects <root>/tmp/data/<dataset>/processed and
# writes <root>/tmp/results/...).  Override with CHAINNPM_ROOT, or per call with
# --data_dir / --out.  The default keeps our campaign layout working.
PROJECT_ROOT = os.environ.get('CHAINNPM_ROOT') or os.path.dirname(os.path.dirname(_HERE))


# ------------------------------------------------------------------ #
# CLI                                                                 #
# ------------------------------------------------------------------ #

def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=3.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mode', type=str, default='natural',
                    help='natural | plant_product | plant_xor | '
                         'plant_family:<rule>')
    ap.add_argument('--target', type=float, default=None,
                    help='target effect (Delta or phi/contrast) for planting')
    ap.add_argument('--method', type=str, default='chainnpm',
                    choices=['chainnpm', 'pertable', 'denorm', 'lavaprop',
                             'privmrf', 'privbayes', 'pbpgm'])
    ap.add_argument('--subset_frac', type=float, default=1.0)
    ap.add_argument('--subset_seed', type=int, default=None)
    ap.add_argument('--data_dir', type=str, default=None)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--tau', type=int, default=None,
                    help='override pre-registered tau')
    return ap


def parse_mode(mode):
    """Return (is_plant, plant_mode, family_rule)."""
    if mode == 'natural':
        return False, None, None
    if not mode.startswith('plant_'):
        raise ValueError('unknown mode %s' % mode)
    rest = mode[len('plant_'):]
    if rest.startswith('family:'):
        rule = rest.split(':', 1)[1]
        return True, 'family', _make_family_rule(rule)
    return True, rest, None


def _make_family_rule(spec):
    import plant_audit as _pa
    if spec in _pa._RULES:
        return _pa._RULES[spec]
    # allow a python expression in (u, x1, x2)
    fn = eval('lambda u, x1, x2: (%s)' % spec)
    return lambda u, x1, x2: np.asarray(fn(u, x1, x2), dtype=int)


# ------------------------------------------------------------------ #
# loading                                                             #
# ------------------------------------------------------------------ #

def _canonical_vshape(df, spec, role):
    pk = spec['pk']; attrs = spec['attrs']
    if role in ('r1', 'r2'):
        cols = [pk] + attrs
    else:
        cols = [pk, spec['fk1'], spec['fk2']] + attrs
    return df[cols].values.astype(int)


def _canonical_chain(df, spec, role):
    pk = spec['pk']; attrs = spec['attrs']
    if role == 'top':
        cols = [pk] + attrs
    else:
        cols = [pk, spec['fk']] + attrs
    return df[cols].values.astype(int)


def load_domains(data_dir, tables):
    domains = {}
    for role, spec in tables.items():
        dj = json.load(open(os.path.join(data_dir, spec['domain_file'])))
        domains[role] = {a: int(dj[a]['size']) for a in spec['attrs']}
    return domains


def load_vshape(data_dir, tables):
    out = {}
    for role, spec in tables.items():
        df = pd.read_csv(os.path.join(data_dir, spec['file']))
        out[role] = _canonical_vshape(df, spec, role)
    return out


def load_chain(data_dir, tables):
    out = {}
    for role, spec in tables.items():
        df = pd.read_csv(os.path.join(data_dir, spec['file']))
        out[role] = _canonical_chain(df, spec, role)
    return out


def subset_vshape(tables, frac, seed):
    """Deterministic root1-stratified subset with FK closure."""
    if frac >= 1.0:
        return tables, {'subset_frac': 1.0, 'subset_seed': None}
    r1, r2, c = tables['r1'], tables['r2'], tables['c']
    rng = np.random.default_rng(seed)
    n1, n2 = r1.shape[0], r2.shape[0]
    k1 = max(1, int(round(n1 * frac)))
    keep1 = np.sort(rng.choice(n1, size=k1, replace=False))
    map1 = -np.ones(n1, dtype=int); map1[keep1] = np.arange(k1)
    ckeep = map1[c[:, 1].astype(int)] >= 0
    c_sub = c[ckeep].copy()
    c_sub[:, 1] = map1[c_sub[:, 1].astype(int)]
    used2 = np.unique(c_sub[:, 2].astype(int))
    map2 = -np.ones(n2, dtype=int); map2[used2] = np.arange(len(used2))
    c_sub[:, 2] = map2[c_sub[:, 2].astype(int)]
    r1_sub = r1[keep1].copy(); r1_sub[:, 0] = np.arange(k1)
    r2_sub = r2[used2].copy(); r2_sub[:, 0] = np.arange(len(used2))
    c_sub[:, 0] = np.arange(c_sub.shape[0])
    return ({'r1': r1_sub, 'r2': r2_sub, 'c': c_sub},
            {'subset_frac': float(frac), 'subset_seed': int(seed)})


def _htilde_summary(htilde):
    """Mean group-size TVD over the FKs (H~ protocol diagnostic; None if the
    backend returned no H~ record, e.g. old baselines.py on other mirrors)."""
    if not htilde:
        return None
    tvds = [v['tvd'] for v in htilde.values()
            if isinstance(v, dict) and 'tvd' in v]
    return float(np.mean(tvds)) if tvds else None


def _baseline_result(out):
    """Unpack baselines returns: aligned backends give (syn_tables, htilde),
    legacy ones a bare dict.  Normalizes to (syn_tables, htilde-or-None)."""
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, None


# ------------------------------------------------------------------ #
# effect helpers                                                      #
# ------------------------------------------------------------------ #

def _fit_on_views(views, keys):
    """Fit each anchor's binarization on its OWN entity-level view so the
    cut/subset is not popularity-weighted (must match planting, which
    binarizes anchors on the root tables and A on the child rows)."""
    params = {}
    for k in keys:
        if k in params:
            continue
        role = k[0]
        view = views[role]
        params[k] = ev.fit_binarize(view['data'][:, view['col'][k]])
    return params


def vshape_effects(gt_views, syn_views, cfg, mode_kind):
    """Return dict of real/syn effect metrics for the V-shape."""
    eff = cfg['effect']
    x1 = tuple(eff['x1']); x2 = tuple(eff['x2']); a = tuple(eff['a'])
    gv, sv = gt_views['c'], syn_views['c']
    keys = [x1, x2, a]
    plant = cfg.get('plant')
    phi_keys = []
    if plant:
        u = ('r2', plant['u_attr'])
        px1 = ('r1', plant['x1']); px2 = ('r1', plant['x2']); aa = ('c', plant['a_attr'])
        phi_keys = [px1, px2, aa, u]
    params = _fit_on_views(gt_views, keys + phi_keys)
    out = {}
    # main cross-parent contrast
    for tag, view in (('real', gv), ('syn', sv)):
        r = ev.vshape_effect(view, view['col'], x1, x2, a,
                             params[x1], params[x2], params[a])
        out['contrast_' + tag] = r
    # phi (xor) on plant anchors
    if plant and mode_kind in ('xor', 'family', 'mod4', 'threshold', 'or'):
        aa = ('c', plant['a_attr'])
        for tag, view in (('real', gv), ('syn', sv)):
            r = ev.vshape_effect(view, view['col'], phi_keys[0], phi_keys[1],
                                 aa, params[phi_keys[0]], params[phi_keys[1]],
                                 params[aa], u=phi_keys[3], pu=params[phi_keys[3]])
            out['phi_' + tag] = r
    return out


def chain_effects(gt_views, syn_views, cfg):
    eff = cfg['effect']
    x1 = tuple(eff['x1']); a = tuple(eff['a'])
    gv, sv = gt_views['bot'], syn_views['bot']
    params = _fit_on_views(gt_views, [x1, a])
    out = {}
    for tag, view in (('real', gv), ('syn', sv)):
        out['delta_' + tag] = ev.chain_effect(view, view['col'], x1, a,
                                              params[x1], params[a])
    return out


# ------------------------------------------------------------------ #
# V-shape pipeline                                                    #
# ------------------------------------------------------------------ #

def _materialize_chainnpmv(r1_arr, child_arr, r2_out, cfg, domains, n2_gt,
                           noisy, seed):
    r2_attrs = cfg['tables']['r2']['attrs']
    child_r2 = np.column_stack([np.asarray(r2_out[a], dtype=int)
                                for a in r2_attrs]) if r2_out else \
        np.zeros((child_arr.shape[0], len(r2_attrs)), dtype=int)
    rng = np.random.default_rng(seed + 777)
    cols = [np.arange(n2_gt)]
    for a in r2_attrs:
        p = noisy.get('P2_' + a)
        if p is not None and p.sum() > 0:
            prob = np.maximum(p, 0); prob = prob / prob.sum()
            cols.append(rng.choice(len(prob), size=n2_gt, p=prob))
        else:
            cols.append(np.zeros(n2_gt, dtype=int))
    r2_tab = np.column_stack(cols)
    return r1_arr, r2_tab, child_arr, child_r2


def run_vshape_pipeline(cfg, args):
    t_start = time.time()
    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, 'tmp', 'data',
                                             cfg['name'], 'processed')
    tables = load_vshape(data_dir, cfg['tables'])
    domains = load_domains(data_dir, cfg['tables'])
    attrs = {role: cfg['tables'][role]['attrs'] for role in ('r1', 'r2', 'c')}

    # optional subset
    sub_seed = args.subset_seed if args.subset_seed is not None else args.seed
    tables, sub_info = subset_vshape(tables, args.subset_frac, sub_seed)
    r1, r2, c = tables['r1'], tables['r2'], tables['c']

    is_plant, plant_mode, family_rule = parse_mode(args.mode)
    plant_audit = None
    if is_plant:
        plant = cfg['plant']
        kw = dict(pk1=cfg['tables']['r1']['pk'], pk2=cfg['tables']['r2']['pk'],
                  pkc=cfg['tables']['c']['pk'],
                  fk1=cfg['tables']['c']['fk1'], fk2=cfg['tables']['c']['fk2'],
                  x1=plant['x1'], x2=plant['x2'], a_attr=plant['a_attr'],
                  u_attr=plant.get('u_attr'), balance=(0.0, 1.0))
        r1_df = pd.DataFrame(r1, columns=[cfg['tables']['r1']['pk']] + attrs['r1'])
        r2_df = pd.DataFrame(r2, columns=[cfg['tables']['r2']['pk']] + attrs['r2'])
        c_df = pd.DataFrame(c, columns=[cfg['tables']['c']['pk'],
                                        cfg['tables']['c']['fk1'],
                                        cfg['tables']['c']['fk2']] + attrs['c'])
        tgt = args.target if args.target is not None else 0.8
        p, calib = pa.calibrate_vshape(r1_df, r2_df, c_df, target=tgt,
                                       mode=plant_mode, family_rule=family_rule,
                                       seeds=(0, 1), iters=12, tol=0.02, **kw)
        res = pa.plant_vshape(r1_df, r2_df, c_df, p=p, mode=plant_mode,
                              family_rule=family_rule, seed=args.seed, **kw)
        plant_audit = res['audit']
        plant_audit['calib_p'] = float(p); plant_audit['calib_effect'] = float(calib)
        # swap in planted child (a_attr now binary), canonical
        # [pk, fk1, fk2, attrs] order
        cols = [cfg['tables']['c']['pk'], cfg['tables']['c']['fk1'],
                cfg['tables']['c']['fk2']] + attrs['c']
        tables['c'] = res['child'].reindex(columns=cols).values.astype(int)
        domains['c'][plant['a_attr']] = 2  # a_attr binarized

    n_leaf = tables['c'].shape[0]
    # tau from the pre-registered p95 rule (children per root1)
    sizes = ev.group_sizes(tables['c'][:, 1])
    tau = args.tau or prereg.pick_tau(sizes)

    # queries (built once, shared across methods)
    qdom = {}
    for role in ('r1', 'r2', 'c'):
        for a in attrs[role]:
            qdom[(role, a)] = domains[role][a]
    qrng = np.random.default_rng(prereg.QUERY_SEED)
    queries = ev.make_queries_vshape(qrng, attrs, qdom, prereg.N_QUERIES)

    gt_views = ev.build_vshape_views(tables, attrs)

    # ---- synthesize ----
    t0 = time.time()
    child_r2 = None
    if args.method == 'chainnpm':
        xp = list(cfg.get('xp_pairs') or [])
        model = ChainNPMV(attrs['r1'], domains['r1'], attrs['r2'], domains['r2'],
                          attrs['c'], domains['c'], xp_pairs=xp,
                          tau=tau, seed=args.seed)
        delta = prereg.delta_for(n_leaf)
        model.compute_marginals(tables['r1'], tables['r2'], tables['c'],
                                args.eps, delta)
        r1s, ch, r2o, tot = model.synthesize()
        syn_r1, syn_r2, syn_c, child_r2 = _materialize_chainnpmv(
            r1s, ch, r2o, cfg, domains, tables['r2'].shape[0],
            model.noisy, args.seed)
        syn_tables = {'r1': syn_r1, 'r2': syn_r2, 'c': syn_c}
        hyper = {'tau': tau, 'Q': model.Q, 'sigma': model.sigma,
                 'delta': delta, 'k': None}
    elif args.method == 'pertable':
        syn_tables, htilde = _baseline_result(
            bl.pertable_vshape(tables, attrs, domains, args.eps, args.seed))
        hyper = {'tau': tau, 'k': None}
        if htilde:
            hyper['htilde'] = htilde
            hyper['htilde_score'] = _htilde_summary(htilde)
    elif args.method == 'denorm':
        syn_tables = bl.denorm_vshape(tables, attrs, domains, args.eps,
                                      args.seed)
        hyper = {'tau': tau, 'k': None}
    elif args.method == 'privmrf':
        syn_tables, htilde = _baseline_result(
            bl.privmrf_vshape(tables, attrs, domains, args.eps, args.seed))
        hyper = {'tau': tau, 'k': None}
        if htilde:
            hyper['htilde'] = htilde
            hyper['htilde_score'] = _htilde_summary(htilde)
    elif args.method == 'privbayes':
        syn_tables, htilde = _baseline_result(
            bl.privbayes_vshape(tables, attrs, domains, args.eps, args.seed))
        hyper = {'tau': tau, 'k': None, 'pb': bl.pb_config(args.seed)}
        if htilde:
            hyper['htilde'] = htilde
            hyper['htilde_score'] = _htilde_summary(htilde)
    elif args.method == 'pbpgm':
        syn_tables, htilde = _baseline_result(
            bl.pbpgm_vshape(tables, attrs, domains, args.eps, args.seed))
        hyper = {'tau': tau, 'k': None, 'pb': bl.pb_config(args.seed)}
        if htilde:
            hyper['htilde'] = htilde
            hyper['htilde_score'] = _htilde_summary(htilde)
    elif args.method == 'lavaprop':
        try:
            syn_tables = bl.lavaprop_vshape(tables, attrs, domains, args.eps,
                                            args.seed, tau=tau,
                                            k=prereg.K_DEFAULT)
            hyper = {'tau': tau, 'k': prereg.K_DEFAULT}
        except Exception as e:
            syn_tables = None
            hyper = {'tau': tau, 'k': prereg.K_DEFAULT,
                     'degraded': str(e)}
    runtime = time.time() - t0

    result = {
        'dataset': cfg['name'], 'shape': 'vshape',
        'method': args.method, 'mode': args.mode, 'eps': args.eps,
        'seed': args.seed, 'runtime_sec': runtime,
        'hyper': hyper, 'prereg': {
            'query_seed': prereg.QUERY_SEED, 'n_queries': prereg.N_QUERIES,
            'large_count': prereg.LARGE_COUNT, 'tau_choices': list(prereg.TAU_CHOICES)},
        'counts': {'r1_gt': int(tables['r1'].shape[0]),
                   'r2_gt': int(tables['r2'].shape[0]),
                   'c_gt': int(tables['c'].shape[0])},
        'subset': sub_info,
    }
    if plant_audit is not None:
        result['plant'] = plant_audit

    if syn_tables is None:
        result['degraded'] = hyper.get('degraded', 'method returned None')
        result['re'] = None
    else:
        syn_views = ev.build_vshape_views(syn_tables, attrs,
                                          child_r2_attrs=child_r2)
        result['counts'].update(
            {'r1_syn': int(syn_tables['r1'].shape[0]),
             'r2_syn': int(syn_tables['r2'].shape[0]),
             'c_syn': int(syn_tables['c'].shape[0]),
             'syn_orphans_c': int(syn_views['c']['orphans'])})
        result['re'] = ev.query_re(gt_views, syn_views, queries,
                                   prereg.LARGE_COUNT)
        mode_kind = plant_mode if is_plant else 'natural'
        result['effect'] = vshape_effects(gt_views, syn_views, cfg, mode_kind)
        gt_sizes = ev.group_sizes(tables['c'][:, 1])
        syn_sizes = ev.group_sizes(syn_tables['c'][:, 1])
        result['size_tv'] = ev.size_tv(gt_sizes, syn_sizes, tau)
        result['mean_size'] = {'gt': float(gt_sizes.mean()),
                               'syn': float(syn_sizes.mean()) if len(syn_sizes) else 0.0}

    result['total_runtime_sec'] = time.time() - t_start
    return result


# ------------------------------------------------------------------ #
# chain pipeline                                                      #
# ------------------------------------------------------------------ #

def run_chain_pipeline(cfg, args):
    t_start = time.time()
    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, 'tmp', 'data',
                                             cfg['name'], 'processed')
    tables = load_chain(data_dir, cfg['tables'])
    domains = load_domains(data_dir, cfg['tables'])
    attrs = {role: cfg['tables'][role]['attrs'] for role in ('top', 'mid', 'bot')}
    top, mid, bot = tables['top'], tables['mid'], tables['bot']

    is_plant, plant_mode, family_rule = parse_mode(args.mode)
    plant_audit = None
    if is_plant:
        plant = cfg['plant']
        top_df = pd.DataFrame(top, columns=[cfg['tables']['top']['pk']] + attrs['top'])
        mid_df = pd.DataFrame(mid, columns=[cfg['tables']['mid']['pk'],
                                            cfg['tables']['mid']['fk']] + attrs['mid'])
        bot_df = pd.DataFrame(bot, columns=[cfg['tables']['bot']['pk'],
                                            cfg['tables']['bot']['fk']] + attrs['bot'])
        kw = dict(pk0=cfg['tables']['top']['pk'], pk1=cfg['tables']['mid']['pk'],
                  fk1=cfg['tables']['mid']['fk'], fk2=cfg['tables']['bot']['fk'],
                  x1=plant['x1'], x2=plant['x2'], a_attr=plant['a_attr'],
                  u_attr=plant.get('u_attr'), mid_attr=plant.get('mid_attr'),
                  balance=(0.0, 1.0))
        tgt = args.target if args.target is not None else 0.3
        p, calib = pa.calibrate(top_df, mid_df, bot_df, target=tgt,
                                mode=plant_mode, family_rule=family_rule,
                                seeds=(0, 1), iters=12, tol=0.02, **kw)
        res = pa.plant(top_df, mid_df, bot_df, p=p, mode=plant_mode,
                       family_rule=family_rule, seed=args.seed, **kw)
        plant_audit = res['audit']
        plant_audit['calib_p'] = float(p); plant_audit['calib_effect'] = float(calib)
        cols = [cfg['tables']['bot']['pk'], cfg['tables']['bot']['fk']] + attrs['bot']
        bot = res['r2'].reindex(columns=cols).values.astype(int)
        tables['bot'] = bot
        domains['bot'][plant['a_attr']] = 2

    n_leaf = tables['bot'].shape[0]
    sizes2 = ev.group_sizes(tables['bot'][:, 1])
    sizes1 = ev.group_sizes(tables['mid'][:, 1])
    tau2 = args.tau or prereg.pick_tau(sizes2)
    tau1 = prereg.pick_tau(sizes1)

    qdom = {}
    for role in ('top', 'mid', 'bot'):
        for a in attrs[role]:
            qdom[(role, a)] = domains[role][a]
    qrng = np.random.default_rng(prereg.QUERY_SEED)
    queries = ev.make_queries_chain(qrng, attrs, qdom, prereg.N_QUERIES)
    gt_views = ev.build_chain_views(tables, attrs)

    t0 = time.time()
    if args.method == 'chainnpm':
        tt = [(attrs['top'][i], attrs['top'][j])
              for i in range(len(attrs['top'])) for j in range(i + 1, len(attrs['top']))]
        tm = [(a, b) for a in attrs['top'] for b in attrs['mid']]
        mb = [(a, b) for a in attrs['mid'] for b in attrs['bot']]
        model = ChainNPM3(attrs['top'], domains['top'], attrs['mid'],
                          domains['mid'], attrs['bot'], domains['bot'],
                          tt_pairs=tt, tm_pairs=tm, mb_pairs=mb,
                          sz1_attrs=attrs['top'], sz2_attrs=attrs['mid'],
                          tau1=tau1, tau2=tau2, seed=args.seed)
        delta = prereg.delta_for(n_leaf)
        model.compute_marginals(tables['top'], tables['mid'], tables['bot'],
                                args.eps, delta)
        ts, ms, bs = model.synthesize()
        syn_tables = {'top': ts, 'mid': ms, 'bot': bs}
        hyper = {'tau1': tau1, 'tau2': tau2, 'sigma': model.sigma,
                 'delta': delta, 'k': None}
    elif args.method == 'pertable':
        syn_tables, htilde = _baseline_result(
            bl.pertable_chain(tables, attrs, domains, args.eps, args.seed))
        hyper = {'tau1': tau1, 'tau2': tau2, 'k': None}
        if htilde:
            hyper['htilde'] = htilde
            hyper['htilde_score'] = _htilde_summary(htilde)
    elif args.method == 'denorm':
        syn_tables = bl.denorm_chain(tables, attrs, domains, args.eps, args.seed)
        hyper = {'tau1': tau1, 'tau2': tau2, 'k': None}
    elif args.method == 'privmrf':
        syn_tables, htilde = _baseline_result(
            bl.privmrf_chain(tables, attrs, domains, args.eps, args.seed))
        hyper = {'tau1': tau1, 'tau2': tau2, 'k': None}
        if htilde:
            hyper['htilde'] = htilde
            hyper['htilde_score'] = _htilde_summary(htilde)
    elif args.method == 'privbayes':
        syn_tables, htilde = _baseline_result(
            bl.privbayes_chain(tables, attrs, domains, args.eps, args.seed))
        hyper = {'tau1': tau1, 'tau2': tau2, 'k': None,
                 'pb': bl.pb_config(args.seed)}
        if htilde:
            hyper['htilde'] = htilde
            hyper['htilde_score'] = _htilde_summary(htilde)
    elif args.method == 'pbpgm':
        syn_tables, htilde = _baseline_result(
            bl.pbpgm_chain(tables, attrs, domains, args.eps, args.seed))
        hyper = {'tau1': tau1, 'tau2': tau2, 'k': None,
                 'pb': bl.pb_config(args.seed)}
        if htilde:
            hyper['htilde'] = htilde
            hyper['htilde_score'] = _htilde_summary(htilde)
    elif args.method == 'lavaprop':
        try:
            syn_tables = bl.lavaprop_chain(tables, attrs, domains, args.eps,
                                           args.seed, tau=tau2,
                                           k=prereg.K_DEFAULT)
            hyper = {'tau1': tau1, 'tau2': tau2, 'k': prereg.K_DEFAULT}
        except Exception as e:
            syn_tables = None
            hyper = {'tau1': tau1, 'tau2': tau2, 'k': prereg.K_DEFAULT,
                     'degraded': str(e)}
    runtime = time.time() - t0

    result = {
        'dataset': cfg['name'], 'shape': 'chain',
        'method': args.method, 'mode': args.mode, 'eps': args.eps,
        'seed': args.seed, 'runtime_sec': runtime, 'hyper': hyper,
        'prereg': {'query_seed': prereg.QUERY_SEED, 'n_queries': prereg.N_QUERIES,
                   'large_count': prereg.LARGE_COUNT,
                   'tau_choices': list(prereg.TAU_CHOICES)},
        'counts': {'top_gt': int(top.shape[0]), 'mid_gt': int(mid.shape[0]),
                   'bot_gt': int(bot.shape[0])},
    }
    if plant_audit is not None:
        result['plant'] = plant_audit

    if syn_tables is None:
        result['degraded'] = hyper.get('degraded', 'method returned None')
        result['re'] = None
    else:
        syn_views = ev.build_chain_views(syn_tables, attrs)
        result['counts'].update(
            {'top_syn': int(syn_tables['top'].shape[0]),
             'mid_syn': int(syn_tables['mid'].shape[0]),
             'bot_syn': int(syn_tables['bot'].shape[0]),
             'syn_orphans_bot': int(syn_views['bot']['orphans'])})
        result['re'] = ev.query_re(gt_views, syn_views, queries,
                                   prereg.LARGE_COUNT)
        result['effect'] = chain_effects(gt_views, syn_views, cfg)
        result['size_tv'] = {
            'bot_per_mid': ev.size_tv(sizes2, ev.group_sizes(syn_tables['bot'][:, 1]), tau2),
            'mid_per_top': ev.size_tv(sizes1, ev.group_sizes(syn_tables['mid'][:, 1]), tau1)}
        result['mean_size'] = {'bot_per_mid_gt': float(sizes2.mean()),
                               'mid_per_top_gt': float(sizes1.mean())}

    result['total_runtime_sec'] = time.time() - t_start
    return result


def dump(result, cfg, args, default_name):
    out = args.out or os.path.join(PROJECT_ROOT, 'tmp', 'results',
                                   cfg['name'], default_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print('wrote %s' % out)
    return out
