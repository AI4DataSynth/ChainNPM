"""Baseline methods for campaign P4 (per-table, denorm, lava-prop).

These are the comparison points for Chain-NPM on the four new datasets.
PrivMRF / PrivLava calls REUSE the existing implementations under
baseline-repos (unified_comparison.run_privmrf_single_table and the
PrivLava orchestrator); if a backend fails or is too costly the driver
records the degradation instead of blocking.

Canonical layouts are the same as eval_newdata's:
  chain : top=[PK,attrs], mid=[PK,FK,attrs], bot=[PK,FK,attrs]
  vshape: r1=[PK,attrs], r2=[PK,attrs], c=[PK,fk1,fk2,attrs]
"""

import os
import sys
import contextlib
import io
import logging

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# The PrivMRF / PrivLava / PB-PGM backends are third-party implementations and
# are NOT vendored here.  Point BASELINE_REPOS at the directory that holds them
# (see README); the default assumes this file sits in a "<baseline-repos>/ChainNPM"
# checkout, as in our own campaign tree.
_BASE = os.environ.get('BASELINE_REPOS') or os.path.dirname(_HERE)
for p in (_BASE, os.path.join(_BASE, 'PrivLava'),
          os.path.join(_BASE, 'pbpgm')):
    if p not in sys.path:
        sys.path.insert(0, p)


def _quiet():
    """Silence the very chatty PrivMRF/PrivLava stdout+logging."""
    return contextlib.redirect_stdout(io.StringIO())


def _single_attr_table(data, attr_names, domain_dict, epsilon, seed):
    """PrivMRF degenerates on <=1 attr (empty junction tree); sample the
    attribute from a noisy 1-way marginal instead (Gaussian mechanism)."""
    from chain_npm_generic import analytic_gaussian_gamma
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    if len(attr_names) == 0:
        return np.zeros((n, 0), dtype=int)
    d = int(domain_dict[attr_names[0]])
    hist = np.bincount(data[:, 0], minlength=d).astype(float)
    delta = 1.0 / max(n, 1)
    gamma = analytic_gaussian_gamma(epsilon, delta)
    sigma = 1.0 / gamma if gamma > 0 else 1.0
    noisy = np.maximum(0.0, hist + rng.normal(0, sigma, d))
    p = noisy / max(noisy.sum(), 1e-9)
    return rng.choice(d, size=n, p=p).reshape(-1, 1).astype(int)


def privmrf_table(data, attr_names, domain_dict, epsilon, name, seed=0):
    """Single-table PrivMRF. data (n, d) int; returns (n, d) int."""
    from unified_comparison import run_privmrf_single_table
    data = np.asarray(data, dtype=int)
    if len(attr_names) <= 1:
        return _single_attr_table(data, attr_names, domain_dict, epsilon, seed)
    from chain_npm_generic import analytic_gaussian_gamma
    n = data.shape[0]
    delta = 1.0 / max(n, 1)
    gamma = analytic_gaussian_gamma(epsilon, delta)
    sigma = 1.0 / gamma if gamma > 0 else 1.0
    np.random.seed(seed)
    logging.disable(logging.CRITICAL)
    with _quiet():
        syn = run_privmrf_single_table(
            data, list(attr_names), dict(domain_dict), epsilon, name,
            noise_sigma=sigma)
    logging.disable(logging.NOTSET)
    return np.asarray(syn, dtype=int)


# ------------------------------------------------------------------ #
# per-table PrivMRF + author-protocol FK re-linking (80/20 split)     #
# ------------------------------------------------------------------ #
# Protocol (PrivLava Sec.7.1, quoted verbatim by PrivPetal Sec.6.1):
#   "First, we use 80% of the privacy cost to create synthetic versions for
#    private relations. Then, we generate a noisy histogram H~ recording the
#    group size distribution for each private foreign key FK(R, R'), with the
#    remaining 20%. Finally, we randomly link the synthetic tuples in R with
#    those in R', ensuring that the group size distribution in H~ is
#    maintained."
# Per-FK H~ mechanics here (see relink_with_group_sizes): the per-parent
# group sizes are true counts with sensitivity 1 (Gaussian mechanism, sigma
# from analytic_gaussian_gamma(0.2*eps, delta=1/n_child)); negative noisy
# counts are clamped to zero, the vector is normalized to a distribution and
# every synthetic child row draws its parent from it (rng.choice), so each
# parent's expected synthetic degree tracks its true noisy degree.
# DP accounting (same split as the authors): a child row enters its own table
# model (0.8*eps) and exactly one FK histogram (0.2*eps) -> <= 1 * eps by
# basic composition; a top/parent row enters only its table model (0.8*eps).
# Advanced composition would only lower the total, so the 80/20 split is a
# valid per-cell epsilon upper bound identical to the author text.

def _true_group_counts(child_fk, parent_pk, n_parent):
    """True per-parent-row child counts of an FK.

    Maps the child's FK values (parent-PK space) to parent ROW indices via a
    stable sort + searchsorted, then bincounts into row space.  Robust to
    non-contiguous / 1-based parent PKs (financial DKEY = {0,1,2} over 77
    district rows, IMDb ids, etc.).
    """
    fk = np.asarray(child_fk, dtype=int)
    pk = np.asarray(parent_pk, dtype=int)
    order = np.argsort(pk, kind='stable')
    idx = np.searchsorted(pk[order], fk)
    idx = np.clip(idx, 0, len(pk) - 1)
    return np.bincount(order[idx], minlength=int(n_parent)).astype(float)


def relink_with_group_sizes(parent_groups, child_per_group_gt, epsilon_hist,
                            delta, seed, rng=None):
    """Author-protocol FK re-linking through a noisy group-size histogram H~.

    parent_groups       : int array, the true parent-group id of every child
                          row (used for n_child and the diagnostic TVD).
    child_per_group_gt  : float array of length n_parent, the true child
                          count per parent group (row space).
    epsilon_hist        : privacy budget of this FK's histogram (the 20%
                          share of the cell's total epsilon).
    delta               : delta of the Gaussian mechanism (1/n_child of the
                          FK's child table, per campaign rule).
    seed, rng           : reproducibility; an explicit rng wins over seed.

    Returns (fk, info): fk is the synthetic parent id of every child row,
    drawn via rng.choice from the noise-normalized H~ (negatives clamped to
    zero); info carries the mechanism parameters and the TVD between the
    resulting synthetic and the true group-size distributions.
    """
    from chain_npm_generic import analytic_gaussian_gamma
    from eval_newdata import size_tv as _size_tv
    counts = np.asarray(child_per_group_gt, dtype=float)
    rng = rng if rng is not None else np.random.default_rng(seed)
    n_child = int(len(parent_groups))
    n_parent = int(counts.shape[0])
    gamma = analytic_gaussian_gamma(float(epsilon_hist), float(delta))
    sigma = 1.0 / gamma if gamma > 0 else 1.0
    noisy = np.maximum(0.0, counts + rng.normal(0.0, sigma, size=n_parent))
    p = noisy / max(noisy.sum(), 1e-9)
    fk = rng.choice(n_parent, size=n_child, p=p)
    # diagnostic: TVD (drop-zero rows) between true and synthetic group sizes
    gt = counts[counts > 0].astype(np.int64)
    sy = np.bincount(fk, minlength=n_parent).astype(np.int64)
    sy = sy[sy > 0]
    cap = max(int(gt.max()) if gt.size else 1, int(sy.max()) if sy.size else 1)
    tvd = _size_tv(gt, sy, cap)
    info = dict(mechanism='gaussian', eps_hist=float(epsilon_hist),
                sigma=float(sigma), delta=float(delta), tvd=float(tvd))
    return fk, info


def pertable_chain(tables, attrs, domains, epsilon, seed=0):
    """Per-table PrivMRF (0.8 eps) + author-protocol FK re-link (0.2 eps).

    Returns (syn_tables, htilde); htilde records the per-FK H~ parameters and
    the synthetic-vs-true group-size TVD (hyper.htilde_score in the JSON).
    """
    ta = _table_cols(tables, attrs, 'top')
    ma = _table_cols(tables, attrs, 'mid')
    ba = _table_cols(tables, attrs, 'bot')
    s_top = privmrf_table(ta, attrs['top'], domains['top'], 0.8 * epsilon,
                          'pertable_top', seed)
    s_mid = privmrf_table(ma, attrs['mid'], domains['mid'], 0.8 * epsilon,
                          'pertable_mid', seed + 1)
    s_bot = privmrf_table(ba, attrs['bot'], domains['bot'], 0.8 * epsilon,
                          'pertable_bot', seed + 2)
    return _relink_chain(tables, s_top, s_mid, s_bot, epsilon, seed)


def pertable_vshape(tables, attrs, domains, epsilon, seed=0):
    """Per-table PrivMRF (0.8 eps) + author-protocol FK re-link (0.2 eps)."""
    s1 = privmrf_table(_table_cols(tables, attrs, 'r1'), attrs['r1'],
                       domains['r1'], 0.8 * epsilon, 'pertable_r1', seed)
    s2 = privmrf_table(_table_cols(tables, attrs, 'r2'), attrs['r2'],
                       domains['r2'], 0.8 * epsilon, 'pertable_r2', seed + 1)
    sc = privmrf_table(_table_cols(tables, attrs, 'c'), attrs['c'],
                       domains['c'], 0.8 * epsilon, 'pertable_c', seed + 2)
    return _relink_vshape(tables, s1, s2, sc, epsilon, seed)


# ------------------------------------------------------------------ #
# denorm: flatten full join -> single PrivMRF -> decompose            #
# ------------------------------------------------------------------ #

def _unique_with_first(sigs):
    """np.unique on 2-D rows + index of first occurrence of each."""
    uniq, inv = np.unique(sigs, axis=0, return_inverse=True)
    first = np.empty(len(uniq), dtype=int)
    seen = np.zeros(len(uniq), dtype=bool)
    for i, e in enumerate(inv):
        if not seen[e]:
            first[e] = i
            seen[e] = True
    return uniq, inv, first


def denorm_chain(tables, attrs, domains, epsilon, seed=0):
    """Flatten every bot row with its mid+top attrs, single PrivMRF, then
    re-derive top/mid entities from the synthetic attribute signatures."""
    from eval_newdata import build_chain_views
    views = build_chain_views(tables, attrs)
    flat = views['bot']['data']                      # [top|mid|bot] per bot row
    nt, nm = len(attrs['top']), len(attrs['mid'])
    flat_attrs = attrs['top'] + attrs['mid'] + attrs['bot']
    flat_dom = {}
    for role in ('top', 'mid', 'bot'):
        for a in attrs[role]:
            flat_dom[a] = domains[role][a]
    syn = privmrf_table(flat, flat_attrs, flat_dom, epsilon, 'denorm_chain',
                        seed)
    n_b = syn.shape[0]

    top_sigs, top_inv = np.unique(syn[:, :nt], axis=0, return_inverse=True)
    mid_sigs, mid_inv, mid_first = _unique_with_first(syn[:, :nt + nm])
    mid_top_fk = top_inv[mid_first]
    return {
        'top': np.column_stack([np.arange(len(top_sigs)), top_sigs]),
        'mid': np.column_stack([np.arange(len(mid_sigs)), mid_top_fk,
                                mid_sigs[:, nt:]]),
        'bot': np.column_stack([np.arange(n_b), mid_inv, syn[:, nt + nm:]]),
    }


def denorm_vshape(tables, attrs, domains, epsilon, seed=0):
    """Join both parents' attrs into each child row, single PrivMRF, then
    re-derive root entities from the synthetic attribute signatures."""
    r1, r2, c = tables['r1'], tables['r2'], tables['c']
    r1a = r1[:, 1:1 + len(attrs['r1'])].astype(int)
    r2a = r2[:, 1:1 + len(attrs['r2'])].astype(int)
    ca = c[:, 3:3 + len(attrs['c'])].astype(int)
    flat = np.hstack([r1a[c[:, 1]], r2a[c[:, 2]], ca])
    n1, n2 = len(attrs['r1']), len(attrs['r2'])
    flat_attrs = attrs['r1'] + attrs['r2'] + attrs['c']
    flat_dom = {}
    for role in ('r1', 'r2', 'c'):
        for a in attrs[role]:
            flat_dom[a] = domains[role][a]
    syn = privmrf_table(flat, flat_attrs, flat_dom, epsilon, 'denorm_vshape',
                        seed)
    nc = syn.shape[0]
    r1_sigs, r1_inv = np.unique(syn[:, :n1], axis=0, return_inverse=True)
    r2_sigs, r2_inv = np.unique(syn[:, n1:n1 + n2], axis=0,
                                return_inverse=True)
    return {
        'r1': np.column_stack([np.arange(len(r1_sigs)), r1_sigs]),
        'r2': np.column_stack([np.arange(len(r2_sigs)), r2_sigs]),
        'c': np.column_stack([np.arange(nc), r1_inv, r2_inv,
                              syn[:, n1 + n2:]]),
    }


# ------------------------------------------------------------------ #
# lava-prop: PrivLava latent propagation (best effort)                 #
# ------------------------------------------------------------------ #

def _privlava_config(epsilon, n_leaf, tau, k):
    from privacy_accountant import PrivLavaConfig
    # Paper defaults (PrivLava Sec. 6 + Sec. 4.7): T=6 EM iterations,
    # T_C=2 marginal-selection rounds, n_C=400 candidate latent marginals per
    # round; delta = 1 / leaf-table rows (the campaign's delta_for rule).
    # (Previous P4 grid used T=4/T_C=1/n_C=200 — superseded by this align.)
    return PrivLavaConfig(epsilon=epsilon, delta=1.0 / max(int(n_leaf), 1),
                          T=6, T_C=2, n_C=400, n_latent=2,
                          tau_quantile=0.99)


def lavaprop_chain(tables, attrs, domains, epsilon, seed=0, tau=None, k=4):
    """PrivLava pairwise latent-propagation on the chain. Best effort."""
    from multi_table_data import MultiTableDatabase, ForeignKey, Relation
    from multi_fk_orchestrator import MultiFKOrchestrator
    top, mid, bot = tables['top'], tables['mid'], tables['bot']

    # seed the pipeline's PRNG so each grid cell is reproducible (all other
    # methods in this module already control the seed; lavaprop consumed
    # unseeded np.random draws and a fixed default_rng(42) sampler before)
    np.random.seed(seed)
    # PrivLava graph search writes ./temp/graph_*.png relative to the cwd;
    # make sure the directory exists so the CFS marginal path is not silently
    # replaced by 2-way marginals
    os.makedirs('temp', exist_ok=True)

    t_rel = Relation('Top', top, ['PK'] + attrs['top'],
                     {**{a: int(top[:, i + 1].max()) + 1
                         for i, a in enumerate(attrs['top'])},
                      'PK': int(top[:, 0].max()) + 2},
                     pk_attr='PK', is_private=True, is_primary_private=True)
    m_rel = Relation('Mid', mid, ['PK', 'FK'] + attrs['mid'],
                     {**{a: int(mid[:, i + 2].max()) + 1
                         for i, a in enumerate(attrs['mid'])},
                      'PK': int(mid[:, 0].max()) + 2,
                      'FK': int(top[:, 0].max()) + 2},
                     pk_attr='PK', is_private=True)
    b_rel = Relation('Bot', bot, ['PK', 'FK'] + attrs['bot'],
                     {**{a: int(bot[:, i + 2].max()) + 1
                         for i, a in enumerate(attrs['bot'])},
                      'PK': int(bot[:, 0].max()) + 2,
                      'FK': int(mid[:, 0].max()) + 2},
                     pk_attr='PK', is_private=True)
    fk_tm = ForeignKey('Mid', 'FK', 'Top', 'PK',
                       multiplicity=tau or 40, is_private=True)
    fk_bm = ForeignKey('Bot', 'FK', 'Mid', 'PK',
                       multiplicity=tau or 12, is_private=True)
    db = MultiTableDatabase({'Top': t_rel, 'Mid': m_rel, 'Bot': b_rel},
                            [fk_tm, fk_bm], 'Top')
    cfg = _privlava_config(epsilon, bot.shape[0], tau or 12, k)
    orch = MultiFKOrchestrator(config=cfg)
    with _quiet():
        syn = orch.model_and_synthesize(db)
    # syn['Top']=[PK,attrs...]; syn['Mid']=[PK,FK,attrs]; syn['Bot']=[PK,FK,attrs]
    return {'top': syn['Top'], 'mid': syn['Mid'], 'bot': syn['Bot']}


def lavaprop_vshape(tables, attrs, domains, epsilon, seed=0, tau=None, k=4):
    """PrivLava per-FK latent propagation on the star (V-shape), degraded.

    Intended merge strategy (recorded per the protocol): model FK(child->r1)
    and FK(child->r2) INDEPENDENTLY, each with its own latent variable
    (PrivLava Algorithm 2), synthesize the two child-attribute draws
    separately, then merge them on the shared child row (one child row gets
    its r1-side and r2-side attributes from the two per-FK models).  This is
    exactly the per-FK composition the cross-parent experiment is meant to
    separate from Chain-NPM's joint modeling.

    Degraded here because the bundled PrivLava orchestrator is
    chain-oriented (a single primary table with a linear FK topological
    order) and does not expose a star merge; wiring a faithful two-FK merge
    is left as follow-up work (does not block the P4 deliverables)."""
    raise NotImplementedError(
        "lavaprop_vshape: PrivLava orchestrator is chain-oriented; the "
        "per-FK star merge (see docstring) is not implemented. Degraded.")


# ------------------------------------------------------------------ #
# PrivBayes / PB-PGM + privmrf alias (added P5.5 baseline campaign)   #
# ------------------------------------------------------------------ #
# Method mapping (per manuscript alignment, see campaign report):
#   - 'privmrf':  per-table PrivMRF + author-protocol FK re-linking == the
#                 manuscript's \textsc{PerTable}.  Uses the eps-aware
#                 privmrf_table above (Gaussian mechanism, sigma from
#                 analytic_gaussian_gamma(eps, delta=1/n)); the P5.5 aligned
#                 grid reruns this recipe under its 'pertable' name with the
#                 same fixed backend.
#   - 'privbayes': per-table PrivBayes (Zhang et al.) + author-protocol FK
#                  re-linking; canonical Laplace accounting (eps/2 structure,
#                  eps/2 measurements), pure eps-DP per table.
#   - 'pbpgm':     per-table PB-PGM (PrivBayes structure + PGM
#                  post-processing, McKenna et al. '19) + author-protocol FK
#                  re-linking.  Same network and noisy measurements as
#                  PrivBayes for the same seed (CRN), differing only in the
#                  estimator: max-entropy PGM fit (MD engine) + i.i.d.
#                  sampling vs BN forward conditional sampling.
# Table synthesis always spends the 80% share (0.8*eps), each FK's noisy
# group-size histogram H~ the 20% share (0.2*eps) via relink_with_group_sizes
# — the same protocol and budget split as PrivLava Sec.7.1 / PrivPetal
# Sec.6.1 — so the three methods differ only in the single-table synthesizer.


def privbayes_table(data, attr_names, domain_dict, epsilon, seed):
    """Single-table PrivBayes (pbpgm.privbayes_table).  Returns (syn, meta)."""
    from privbayes_baseline import privbayes_table as _pb
    data = np.asarray(data, dtype=int)
    syn = _pb(data, list(attr_names), dict(domain_dict), epsilon, seed=int(seed))
    return np.asarray(syn, dtype=int)


def pbpgm_table(data, attr_names, domain_dict, epsilon, seed):
    """Single-table PB-PGM (pbpgm.pbpgm_table).  Returns (syn, meta)."""
    from privbayes_baseline import pbpgm_table as _pgm
    data = np.asarray(data, dtype=int)
    syn = _pgm(data, list(attr_names), dict(domain_dict), epsilon, seed=int(seed))
    return np.asarray(syn, dtype=int)


def _relink_chain(tables, s_top, s_mid, s_bot, epsilon, seed):
    """Author-protocol FK re-linking for chain (top<-mid<-bot).

    Each FK spends its 20% share of the budget on a Gaussian noisy per-parent
    group-size histogram H~ (delta = 1/n_child of the FK's child table) and
    every synthetic child row independently draws its parent from the
    noise-normalized H~ (relink_with_group_sizes).  Returns
    (syn_tables, htilde): htilde = per-FK H~ parameters + TVD of the
    synthetic-vs-true group-size distributions, plus the 80/20 split record.
    """
    n_m, n_b = s_mid.shape[0], s_bot.shape[0]
    n_t = s_top.shape[0]
    eps_h = 0.2 * epsilon
    true_mid = _true_group_counts(tables['mid'][:, 1], tables['top'][:, 0], n_t)
    mid_fk, h1 = relink_with_group_sizes(tables['mid'][:, 1], true_mid,
                                         eps_h, 1.0 / max(n_m, 1), seed + 100)
    true_bot = _true_group_counts(tables['bot'][:, 1], tables['mid'][:, 0], n_m)
    bot_fk, h2 = relink_with_group_sizes(tables['bot'][:, 1], true_bot,
                                         eps_h, 1.0 / max(n_b, 1), seed + 200)
    syn = {
        'top': np.column_stack([np.arange(n_t), s_top]),
        'mid': np.column_stack([np.arange(n_m), mid_fk, s_mid]),
        'bot': np.column_stack([np.arange(n_b), bot_fk, s_bot]),
    }
    htilde = {'mid_per_top': h1, 'bot_per_mid': h2,
              'table_frac': 0.8, 'fk_hist_frac': 0.2}
    return syn, htilde


def _relink_vshape(tables, s1, s2, sc, epsilon, seed):
    """Author-protocol FK re-linking for the star (r1 <- c -> r2).

    Same H~ protocol as _relink_chain; both FKs share the child-table delta
    1/n_c.  Returns (syn_tables, htilde)."""
    n1, n2, nc = s1.shape[0], s2.shape[0], sc.shape[0]
    eps_h = 0.2 * epsilon
    delta_c = 1.0 / max(nc, 1)
    true_c1 = _true_group_counts(tables['c'][:, 1], tables['r1'][:, 0], n1)
    fk1, h1 = relink_with_group_sizes(tables['c'][:, 1], true_c1, eps_h,
                                      delta_c, seed + 100)
    true_c2 = _true_group_counts(tables['c'][:, 2], tables['r2'][:, 0], n2)
    fk2, h2 = relink_with_group_sizes(tables['c'][:, 2], true_c2, eps_h,
                                      delta_c, seed + 200)
    syn = {
        'r1': np.column_stack([np.arange(n1), s1]),
        'r2': np.column_stack([np.arange(n2), s2]),
        'c': np.column_stack([np.arange(nc), fk1, fk2, sc]),
    }
    htilde = {'c_per_r1': h1, 'c_per_r2': h2,
              'table_frac': 0.8, 'fk_hist_frac': 0.2}
    return syn, htilde


def _table_cols(tables, attrs, role):
    """Attribute block (without PK/FK columns) for one role."""
    if role in ('top', 'r1', 'r2'):
        return tables[role][:, 1:1 + len(attrs[role])]
    if role == 'mid':
        return tables[role][:, 2:2 + len(attrs['mid'])]
    return tables['bot'][:, 2:2 + len(attrs['bot'])] \
        if role == 'bot' else tables['c'][:, 3:3 + len(attrs['c'])]


def privmrf_chain(tables, attrs, domains, epsilon, seed=0):
    """PrivMRF == per-table PrivMRF + author-protocol re-link (PerTable).

    Identical recipe to pertable_chain; registered under its own name so the
    figure's 'PrivMRF' series has a dedicated method tag.  Uses the
    eps-aware privmrf_table (Gaussian sigma from the analytic gamma); table
    synthesis spends 0.8*eps, each FK's H~ relink 0.2*eps."""
    ta = _table_cols(tables, attrs, 'top')
    ma = _table_cols(tables, attrs, 'mid')
    ba = _table_cols(tables, attrs, 'bot')
    s_top = privmrf_table(ta, attrs['top'], domains['top'], 0.8 * epsilon,
                          'privmrf_top', seed)
    s_mid = privmrf_table(ma, attrs['mid'], domains['mid'], 0.8 * epsilon,
                          'privmrf_mid', seed + 1)
    s_bot = privmrf_table(ba, attrs['bot'], domains['bot'], 0.8 * epsilon,
                          'privmrf_bot', seed + 2)
    return _relink_chain(tables, s_top, s_mid, s_bot, epsilon, seed)


def privmrf_vshape(tables, attrs, domains, epsilon, seed=0):
    """PrivMRF per-table + author-protocol re-link on the star."""
    s1 = privmrf_table(_table_cols(tables, attrs, 'r1'), attrs['r1'],
                       domains['r1'], 0.8 * epsilon, 'privmrf_r1', seed)
    s2 = privmrf_table(_table_cols(tables, attrs, 'r2'), attrs['r2'],
                       domains['r2'], 0.8 * epsilon, 'privmrf_r2', seed + 1)
    sc = privmrf_table(_table_cols(tables, attrs, 'c'), attrs['c'],
                       domains['c'], 0.8 * epsilon, 'privmrf_c', seed + 2)
    return _relink_vshape(tables, s1, s2, sc, epsilon, seed)


def privbayes_chain(tables, attrs, domains, epsilon, seed=0):
    """Per-table PrivBayes (0.8 eps) + author-protocol re-link (0.2 eps)."""
    ta = _table_cols(tables, attrs, 'top')
    ma = _table_cols(tables, attrs, 'mid')
    ba = _table_cols(tables, attrs, 'bot')
    s_top = privbayes_table(ta, attrs['top'], domains['top'], 0.8 * epsilon, seed)
    s_mid = privbayes_table(ma, attrs['mid'], domains['mid'], 0.8 * epsilon, seed + 1)
    s_bot = privbayes_table(ba, attrs['bot'], domains['bot'], 0.8 * epsilon, seed + 2)
    return _relink_chain(tables, s_top, s_mid, s_bot, epsilon, seed)


def privbayes_vshape(tables, attrs, domains, epsilon, seed=0):
    """Per-table PrivBayes (0.8 eps) + author-protocol re-link (0.2 eps)."""
    s1 = privbayes_table(_table_cols(tables, attrs, 'r1'), attrs['r1'],
                         domains['r1'], 0.8 * epsilon, seed)
    s2 = privbayes_table(_table_cols(tables, attrs, 'r2'), attrs['r2'],
                         domains['r2'], 0.8 * epsilon, seed + 1)
    sc = privbayes_table(_table_cols(tables, attrs, 'c'), attrs['c'],
                         domains['c'], 0.8 * epsilon, seed + 2)
    return _relink_vshape(tables, s1, s2, sc, epsilon, seed)


def pbpgm_chain(tables, attrs, domains, epsilon, seed=0):
    """Per-table PB-PGM (0.8 eps) + author-protocol re-link (0.2 eps)."""
    ta = _table_cols(tables, attrs, 'top')
    ma = _table_cols(tables, attrs, 'mid')
    ba = _table_cols(tables, attrs, 'bot')
    s_top = pbpgm_table(ta, attrs['top'], domains['top'], 0.8 * epsilon, seed)
    s_mid = pbpgm_table(ma, attrs['mid'], domains['mid'], 0.8 * epsilon, seed + 1)
    s_bot = pbpgm_table(ba, attrs['bot'], domains['bot'], 0.8 * epsilon, seed + 2)
    return _relink_chain(tables, s_top, s_mid, s_bot, epsilon, seed)


def pbpgm_vshape(tables, attrs, domains, epsilon, seed=0):
    """Per-table PB-PGM (0.8 eps) + author-protocol re-link (0.2 eps)."""
    s1 = pbpgm_table(_table_cols(tables, attrs, 'r1'), attrs['r1'],
                     domains['r1'], 0.8 * epsilon, seed)
    s2 = pbpgm_table(_table_cols(tables, attrs, 'r2'), attrs['r2'],
                     domains['r2'], 0.8 * epsilon, seed + 1)
    sc = pbpgm_table(_table_cols(tables, attrs, 'c'), attrs['c'],
                     domains['c'], 0.8 * epsilon, seed + 2)
    return _relink_vshape(tables, s1, s2, sc, epsilon, seed)


def pb_config(seed=0):
    """Hyper config record for the PrivBayes-family methods."""
    from privbayes_baseline import THETA, PGM_ITERS
    return {'theta': THETA, 'pgm_iters': PGM_ITERS, 'mechanism': 'laplace',
            'budget_split': {'structure': 0.5, 'measurements': 0.5}}
