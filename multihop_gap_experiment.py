"""
Multi-hop correlation gap experiment for DP relational data synthesis.

Research question
-----------------
All existing DP relational synthesis methods (PrivLava SIGMOD'23, PrivPetal
SIGMOD'25) model each foreign key pairwise and compose the per-FK models.
Do they preserve correlations that span more than one FK hop?

Schema: REGION <- HOUSEHOLD <- INDIVIDUAL   (chain, 2 FK hops)

Planted ground-truth structure:
  - REGION:  URBAN (0/1), BAND (0-2, filler).
  - Household latent type Z_H:  P(Z=1|URBAN=1)=0.85, P(Z=1|URBAN=0)=0.20.
  - Household observables OWN/SIZE depend on URBAN (1-hop signal) but NOT
    on Z_H  ->  the REGION<->INDIVIDUAL correlation is mediated purely by
    the latent household type, not by intermediate observed attributes.
  - Individual EMP/AGE depend strongly on Z_H:
        P(EMP=1|Z=1)=0.80, P(EMP=1|Z=0)=0.30
    => planted 2-hop correlation:
        P(EMP=1|URBAN=1) ~= 0.725 vs P(EMP=1|URBAN=0) ~= 0.40

Methods
-------
  1. PerTable-MRF   : PrivMRF per table + uniform-random FK re-linking
                      (no cross-table modeling at all; lower bound).
  2. Lava-AsIs      : PrivLava reproduction pairwise FK models + the
                      reproduction's multi-FK synthesis (budget-fair: each
                      FK gets half of the gamma^2 budget).
  3. Lava-Propagate : faithful PrivLava-style latent propagation (paper
                      Algorithm 3): HOUSEHOLD is augmented with the
                      EM-estimated Z of FK(Individual->Household) before
                      modeling FK(Household->Region); synthesis conditions
                      each hop on its own latent variable.
  4. Denorm-Full    : flatten the full 3-table join, run one PrivMRF on it,
                      decompose back (joint-modeling reference point).

Metrics
-------
  - 1-hop RE : conjunctive count queries over region><household and
               household><individual joins.
  - 2-hop RE : conjunctive count queries over region><household><individual.
  - Correlation preservation: Delta = P(EMP=1|URBAN=1) - P(EMP=1|URBAN=0);
    report Delta_syn / Delta_real.
"""

import sys
import os
import json
import time
import logging
import argparse

import numpy as np

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger('multihop_gap')
logger.setLevel(logging.INFO)

# Path bootstrap.  The release modules (chain_npm.py, baselines.py) live next
# to this driver and must win every import; the third-party backends used by
# the comparison methods (a PrivLava checkout plus unified_comparison.py) are
# NOT vendored, so they are located through CHAINNPM_BASELINES (see README).
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get('CHAINNPM_BASELINES',
                      os.path.join(os.path.dirname(_HERE), 'baseline-repos'))
for _p in (os.path.join(BASE, 'pbpgm'), os.path.join(BASE, 'PrivLava'), BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)   # the release modules shadow anything in BASE

# `run_privmrf_single_table` (needed by the `denorm` control only) is imported
# lazily inside run_denorm_full, so the chain_* and pertable methods run in a
# checkout that has no baseline tree.

# ============================================================
# 1. Data generation: planted 2-hop latent-mediated correlation
# ============================================================

P_Z_GIVEN_URBAN1 = 0.85
P_Z_GIVEN_URBAN0 = 0.20
P_EMP_GIVEN_Z1 = 0.80
P_EMP_GIVEN_Z0 = 0.30
P_OWN_GIVEN_URBAN1 = 0.60
P_OWN_GIVEN_URBAN0 = 0.35


def generate_multihop_data(n_regions=80, seed=42):
    rng = np.random.default_rng(seed)

    region_rows, hh_rows, ind_rows = [], [], []
    hh_id, ind_id = 1, 1
    latent_log = []  # (region_id, urban, hh_id, z)

    for r in range(1, n_regions + 1):
        urban = int(rng.random() < 0.5)
        band = int(rng.integers(0, 3))
        region_rows.append([r, urban, band])

        n_hh = int(rng.integers(2, 7))  # 2..6 households per region
        for _ in range(n_hh):
            p_z = P_Z_GIVEN_URBAN1 if urban else P_Z_GIVEN_URBAN0
            z = int(rng.random() < p_z)
            own = int(rng.random() < (P_OWN_GIVEN_URBAN1 if urban else P_OWN_GIVEN_URBAN0))
            size = int(np.clip(rng.integers(0, 4) + urban, 0, 4))  # encoded 0..4
            hh_rows.append([hh_id, r, own, size])
            latent_log.append((r, urban, hh_id, z))

            for _ in range(size + 1):  # household size = encoded value + 1
                age = int(rng.integers(2, 8)) if z else int(rng.integers(0, 6))
                emp = int(rng.random() < (P_EMP_GIVEN_Z1 if z else P_EMP_GIVEN_Z0))
                edu = int(np.clip(age - int(rng.integers(0, 4)), 0, 3))
                ind_rows.append([ind_id, age, emp, edu, hh_id])
                ind_id += 1
            hh_id += 1

    region = np.array(region_rows, dtype=int)
    household = np.array(hh_rows, dtype=int)
    individual = np.array(ind_rows, dtype=int)

    r_domain = {'R_ID': n_regions + 1, 'URBAN': 2, 'BAND': 3}
    h_domain = {'H_ID': hh_id, 'REGION_ID': n_regions + 1, 'OWN': 2, 'SIZE': 5}
    i_domain = {'I_ID': ind_id, 'AGE': 8, 'EMP': 2, 'EDU': 4, 'H_ID': hh_id}

    return {
        'region': region, 'household': household, 'individual': individual,
        'r_domain': r_domain, 'h_domain': h_domain, 'i_domain': i_domain,
        'n_regions': n_regions,
    }


# ============================================================
# 2. Joins + query evaluation
# ============================================================

def join_all(region, household, individual):
    """region >< household >< individual, returns wide rows."""
    hh_by_r = {int(t[0]): t for t in region}
    r_by_h = {int(t[0]): hh_by_r.get(int(t[1])) for t in household}
    rows = []
    for t in individual:
        h = household[int(t[4]) - 1] if int(t[4]) - 1 < len(household) else None
        if h is None or int(h[0]) != int(t[4]):
            h = household[household[:, 0] == int(t[4])]
            h = h[0] if len(h) else None
        if h is None:
            continue
        r = r_by_h.get(int(h[0]))
        if r is None:
            continue
        rows.append([int(r[1]), int(h[2]), int(t[2])])  # URBAN, OWN, EMP
    return np.array(rows, dtype=int).reshape(-1, 3) if rows else np.zeros((0, 3), dtype=int)


def make_queries(rng, n_per_class=150):
    """Random conjunctive count queries by hop class.

    Predicates are random non-empty subsets of each attribute domain:
      URBAN:2, OWN:2, EMP:2, BAND:3, SIZE:5, AGE:8, EDU:4
    Returns list of dicts: {cls, tables, preds} where preds maps
    table -> (attr, subset).
    """
    queries = []

    def subset(dom):
        k = int(rng.integers(1, dom))  # 1..dom-1
        return sorted(rng.choice(dom, size=min(k, dom), replace=False).tolist())

    for _ in range(n_per_class):
        # 1-hop: region x household
        queries.append({'cls': '1hop_rh',
                        'r_pred': ('URBAN', subset(2)), 'h_pred': ('OWN', subset(2))})
        # 1-hop: household x individual
        queries.append({'cls': '1hop_hi',
                        'h_pred': ('OWN', subset(2)), 'i_pred': ('EMP', subset(2))})
        # 2-hop: region x household x individual
        queries.append({'cls': '2hop',
                        'r_pred': ('URBAN', subset(2)), 'i_pred': ('EMP', subset(2))})
        # 2-hop with 2 predicates per side
        queries.append({'cls': '2hop_wide',
                        'r_pred': ('BAND', subset(3)), 'i_pred': ('AGE', subset(8))})
    return queries


def run_query(q, region, household, individual):
    """Answer one conjunctive count query on the given data."""
    r_idx = {int(t[0]): i for i, t in enumerate(region)}
    h_idx = {int(t[0]): i for i, t in enumerate(household)}

    r_attr, r_set = q['r_pred'] if 'r_pred' in q else (None, None)
    h_attr, h_set = q['h_pred'] if 'h_pred' in q else (None, None)
    i_attr, i_set = q['i_pred'] if 'i_pred' in q else (None, None)

    r_col = {'URBAN': 1, 'BAND': 2}
    h_col = {'REGION_ID': 1, 'OWN': 2, 'SIZE': 3}
    i_col = {'AGE': 1, 'EMP': 2, 'EDU': 3, 'H_ID': 4}

    # qualifying households
    qual_h = set()
    for t in household:
        hid = int(t[0])
        ok = True
        if r_attr is not None:
            r = region[int(t[1]) - 1] if 1 <= int(t[1]) <= len(region) else None
            if r is None or int(r[0]) != int(t[1]):
                matches = region[region[:, 0] == int(t[1])]
                r = matches[0] if len(matches) else None
            if r is None or int(r[r_col[r_attr]]) not in r_set:
                ok = False
        if ok and h_attr is not None and int(t[h_col[h_attr]]) not in h_set:
            ok = False
        if ok:
            qual_h.add(hid)

    cnt = 0
    if i_attr is None:
        cnt = len(qual_h)
    else:
        for t in individual:
            if int(t[4]) in qual_h and int(t[i_col[i_attr]]) in i_set:
                cnt += 1
    return cnt


def relative_error(ans_syn, ans_real):
    return abs(ans_syn - ans_real) / max(ans_real, 1.0)


def evaluate(real, syn, queries):
    """real/syn: dict with region/household/individual arrays."""
    res = {c: [] for c in ['1hop_rh', '1hop_hi', '2hop', '2hop_wide']}
    res_large = {c: [] for c in res}  # only queries with real answer >= 20
    for q in queries:
        a_real = run_query(q, real['region'], real['household'], real['individual'])
        a_syn = run_query(q, syn['region'], syn['household'], syn['individual'])
        res[q['cls']].append(relative_error(a_syn, a_real))
        if a_real >= 20:
            res_large[q['cls']].append(relative_error(a_syn, a_real))
    out = {c: float(np.median(v)) if v else None for c, v in res.items()}
    for c, v in res_large.items():
        out[c + '_large'] = float(np.median(v)) if v else None

    # correlation preservation: Delta = P(EMP=1|URBAN=1) - P(EMP=1|URBAN=0)
    def delta(d):
        wide = join_all(d['region'], d['household'], d['individual'])
        if len(wide) == 0:
            return 0.0
        p1 = wide[wide[:, 0] == 1][:, 2].mean() if (wide[:, 0] == 1).any() else 0.0
        p0 = wide[wide[:, 0] == 0][:, 2].mean() if (wide[:, 0] == 0).any() else 0.0
        return float(p1 - p0)

    d_real, d_syn = delta(real), delta(syn)
    out['delta_real'] = d_real
    out['delta_syn'] = d_syn
    out['delta_ratio'] = float(d_syn / d_real) if abs(d_real) > 1e-9 else None
    return out


# ============================================================
# 3. Methods
# ============================================================

def build_db(data, aug_hh=None, z2_size=None):
    from multi_table_data import MultiTableDatabase, ForeignKey, Relation
    h_data = aug_hh if aug_hh is not None else data['household']
    h_attrs = ['H_ID', 'REGION_ID', 'OWN', 'SIZE']
    h_domain = dict(data['h_domain'])
    if z2_size is not None:
        h_attrs = h_attrs + ['Z2']
        h_domain['Z2'] = z2_size

    region = Relation('Region', data['region'], ['R_ID', 'URBAN', 'BAND'],
                      data['r_domain'], pk_attr='R_ID', is_private=True,
                      is_primary_private=True)
    household = Relation('Household', h_data, h_attrs, h_domain,
                         pk_attr='H_ID', is_private=True, is_primary_private=False)
    individual = Relation('Individual', data['individual'],
                          ['I_ID', 'AGE', 'EMP', 'EDU', 'H_ID'],
                          data['i_domain'], pk_attr='I_ID', is_private=True,
                          is_primary_private=False)
    fk_hr = ForeignKey('Household', 'REGION_ID', 'Region', 'R_ID',
                       multiplicity=8, is_private=True)
    fk_ih = ForeignKey('Individual', 'H_ID', 'Household', 'H_ID',
                       multiplicity=6, is_private=True)
    db = MultiTableDatabase({'Region': region, 'Household': household,
                             'Individual': individual},
                            [fk_hr, fk_ih], 'Region')
    return db, fk_hr, fk_ih


def run_lava_pairwise(data, epsilon, propagate=False, seed=42):
    """PrivLava-reproduction pairwise FK modeling, budget-fair (gamma^2/2 per FK).

    propagate=False : the reproduction's synthesis logic (as-is).
    propagate=True  : faithful paper Algorithm-3 latent propagation.
    """
    from multi_table_data import MultiTableDatabase, ForeignKey, Relation
    from fk_modeler import ForeignKeyModeler
    from synthesizer import Synthesizer
    from privacy_accountant import PrivLavaConfig, analytic_gaussian_gamma

    config = PrivLavaConfig(epsilon=epsilon, delta=1.0 / max(
        data['individual'].shape[0], 1), T=4, T_C=1, n_C=200)
    modeler = ForeignKeyModeler(config=config)
    gamma_half = analytic_gaussian_gamma(config.epsilon, config.delta) ** 2 / 2.0

    db, fk_hr, fk_ih = build_db(data)
    db.truncate_multiplicity(fk_ih)
    db.truncate_multiplicity(fk_hr)

    # Model FK(Individual -> Household) on original data
    model_ih = modeler.model(db, fk_ih)

    rng = np.random.default_rng(seed)

    if not propagate:
        # ---- As-is reproduction pipeline (manual budget-fair version) ----
        model_hr = modeler.model(db, fk_hr)
        synth = Synthesizer()
        models = {
            f"{fk_hr.ref_table}->{fk_hr.target_table}": model_hr,
            f"{fk_ih.ref_table}->{fk_ih.target_table}": model_ih,
        }
        synthetic = synth.synthesize_multi_fk(
            models=models, relations=db.relations,
            foreign_keys=db.foreign_keys, fk_topo_order=db.fk_topo_order,
            primary_table='Region')
        return {'region': synthetic['Region'], 'household': synthetic['Household'],
                'individual': synthetic['Individual']}

    # ---- Faithful latent propagation (paper Algorithm 3) ----
    em_ih = model_ih.em_estimator
    z2_size = em_ih._get_z_size()

    # augment HOUSEHOLD with estimated Z2 (hard assignment from EM)
    z2_by_hh = {}
    for g in model_ih.groups:
        z2_by_hh[int(g.target_pk_value)] = int(g.z)
    hh = data['household']
    z2_col = np.array([z2_by_hh.get(int(h[0]), 0) for h in hh], dtype=int)
    z2_col = np.clip(z2_col, 0, z2_size - 1)
    hh_aug = np.column_stack([hh, z2_col])

    db_aug, fk_hr_aug, fk_ih_aug = build_db(data, aug_hh=hh_aug, z2_size=z2_size)
    db_aug.truncate_multiplicity(fk_hr_aug)
    model_hr = modeler.model(db_aug, fk_hr_aug)

    # ---- Synthesis: Region -> Household(+Z2) -> Individual ----
    synth = Synthesizer()
    n_r0 = data['n_regions'] + int(rng.integers(-3, 4))
    n_r0 = max(1, n_r0)
    syn_region, syn_z1 = synth._synthesize_r0(
        db_aug.relations['Region'], model_hr, n_r0)

    em_hr = model_hr.em_estimator
    hh_rows, z2_stars = [], []
    h_id = 1
    hh_obs_cols = None  # obs order of Household_aug minus PK/FK: [OWN, SIZE, Z2]
    for i in range(syn_region.shape[0]):
        region_pk = int(syn_region[i, 0])
        z1 = int(syn_z1[i])
        s = int(em_hr.sample_group_size(z1, rng))
        s = max(0, min(s, 8))
        for _ in range(s):
            t = em_hr.sample_tuple(z1, rng)  # [OWN, SIZE, Z2]
            own = int(np.clip(t[0], 0, 1))
            size = int(np.clip(t[1], 0, 4))
            z2 = int(np.clip(t[2], 0, z2_size - 1))
            hh_rows.append([h_id, region_pk, own, size])
            z2_stars.append(z2)
            h_id += 1
    syn_hh = np.array(hh_rows, dtype=int) if hh_rows else np.zeros((0, 4), dtype=int)

    ind_rows = []
    i_id = 1
    for j in range(syn_hh.shape[0]):
        z2 = z2_stars[j]
        s2 = int(em_ih.sample_group_size(z2, rng))
        s2 = max(0, min(s2, 6))
        for _ in range(s2):
            t2 = em_ih.sample_tuple(z2, rng)  # [AGE, EMP, EDU]
            ind_rows.append([i_id, int(np.clip(t2[0], 0, 7)),
                             int(np.clip(t2[1], 0, 1)),
                             int(np.clip(t2[2], 0, 3)), int(syn_hh[j, 0])])
            i_id += 1
    syn_ind = np.array(ind_rows, dtype=int) if ind_rows else np.zeros((0, 5), dtype=int)

    return {'region': syn_region, 'household': syn_hh, 'individual': syn_ind}


def run_pertable_mrf(data, epsilon, seed=42):
    """Per-table PrivMRF + author-protocol H~ re-linking (no cross-table model).

    This is the manuscript's per-table baseline and now runs the SAME code path
    as the four-database archive (baselines.privmrf_table / baselines._relink_chain),
    so the figure's PrivMRF series is one object across the whole paper:

      * each table model gets 0.8*eps with the eps-aware Gaussian sigma
        (sigma = 1 / analytic_gaussian_gamma(0.8*eps, delta = 1/n_table));
      * each FK's group-size histogram H~ gets 0.2*eps and the synthetic child
        rows draw their parent from the noise-normalized H~ -> 1.0*eps total;
      * every draw is seeded from the run seed, so the three runs of a point
        differ in the re-linking as well as in the synthesis.

    Replaces the eps-blind variant (fixed sigma = 1.0, uniform-random re-link
    from a constant rng seed) whose cells are archived under
    results/arms/controlled_epsblind/.
    """
    from baselines import privmrf_table, _relink_chain
    tables = {'top': data['region'],
              'mid': data['household'],
              'bot': data['individual'][:, [0, 4, 1, 2, 3]]}
    s_top = privmrf_table(tables['top'][:, 1:], ['URBAN', 'BAND'],
                          {'URBAN': 2, 'BAND': 3}, 0.8 * epsilon, 'region', seed)
    s_mid = privmrf_table(tables['mid'][:, [2, 3]], ['OWN', 'SIZE'],
                          {'OWN': 2, 'SIZE': 5}, 0.8 * epsilon, 'household',
                          seed + 1)
    s_bot = privmrf_table(tables['bot'][:, [2, 3, 4]], ['AGE', 'EMP', 'EDU'],
                          {'AGE': 8, 'EMP': 2, 'EDU': 4}, 0.8 * epsilon,
                          'individual', seed + 2)
    syn, _ = _relink_chain(tables, s_top, s_mid, s_bot, epsilon, seed)
    bot = syn['bot'][:, [0, 2, 3, 4, 1]]  # back to [I_ID, AGE, EMP, EDU, H_ID]
    return {'region': syn['top'].astype(int),
            'household': syn['mid'].astype(int),
            'individual': bot.astype(int)}


def run_denorm_full(data, epsilon, max_gs=5, seed=42):
    """Flatten full 3-table join -> single PrivMRF -> decompose. Joint-modeling ref."""
    region = {int(t[0]): t for t in data['region']}
    household = {int(t[0]): t for t in data['household']}

    flat_rows = []
    for h in data['household']:
        hid, rid, own, size = int(h[0]), int(h[1]), int(h[2]), int(h[3])
        r = region.get(rid)
        if r is None:
            continue
        urban, band = int(r[1]), int(r[2])
        members = data['individual'][data['individual'][:, 4] == hid]
        for j in range(max_gs):
            if j < len(members):
                m = members[j]
                flat_rows.append([urban, band, own, size,
                                  int(m[1]), int(m[2]), int(m[3]), 1])
            else:
                flat_rows.append([urban, band, own, size, 0, 0, 0, 0])
    flat_data = np.array(flat_rows, dtype=int)
    attrs = ['URBAN', 'BAND', 'OWN', 'SIZE', 'AGE', 'EMP', 'EDU', 'PRES']
    dom = {'URBAN': 2, 'BAND': 3, 'OWN': 2, 'SIZE': 5, 'AGE': 8, 'EMP': 2,
           'EDU': 4, 'PRES': 2}

    # eps-aware noise: this reference spends its whole budget on the ONE
    # flattened model (there is no FK structure of its own), so sigma follows
    # the declared budget instead of the fixed sigma = 1.0 it used before.
    from chain_npm_generic import analytic_gaussian_gamma
    try:
        from unified_comparison import run_privmrf_single_table
    except ImportError as exc:      # baseline tree not present in this checkout
        raise RuntimeError(
            'the `denorm` control needs the PrivMRF backend: set '
            'CHAINNPM_BASELINES to the directory holding unified_comparison.py '
            f'(currently {BASE})') from exc
    gamma = analytic_gaussian_gamma(float(epsilon),
                                    1.0 / max(flat_data.shape[0], 1))
    sigma = 1.0 / gamma if gamma > 0 else 1.0
    syn_flat = run_privmrf_single_table(flat_data, attrs, dom, epsilon, 'denorm3',
                                       noise_sigma=sigma)

    # decompose: each household = max_gs consecutive rows
    n_h = data['household'].shape[0]
    n_r = data['n_regions']
    rng = np.random.default_rng(seed)
    hh_rows, ind_rows = [], []
    i_id = 1
    hh_region_attr = []  # (urban, band) per synthetic household, for region rebuild
    for h in range(n_h):
        start = h * max_gs
        chunk = syn_flat[start:start + max_gs]
        if len(chunk) == 0:
            continue
        urban = int(np.clip(chunk[0, 0], 0, 1))
        band = int(np.clip(chunk[0, 1], 0, 2))
        own = int(np.clip(chunk[0, 2], 0, 1))
        size = int(np.clip(chunk[0, 3], 0, 4))  # encoded 0..4
        hh_region_attr.append((urban, band))
        hh_rows.append([h + 1, 0, own, size])  # REGION_ID patched below
        members = [j for j in range(len(chunk)) if int(chunk[j, 7]) == 1]
        for j in members[:size + 1]:
            ind_rows.append([i_id, int(np.clip(chunk[j, 4], 0, 7)),
                             int(np.clip(chunk[j, 5], 0, 1)),
                             int(np.clip(chunk[j, 6], 0, 3)), h + 1])
            i_id += 1

    # region table: group households by their sampled (urban, band) so that
    # each synthetic region is attribute-pure; split large buckets across
    # multiple regions. Region attrs then exactly match member households.
    from collections import defaultdict
    buckets = defaultdict(list)
    for h in range(len(hh_rows)):
        buckets[hh_region_attr[h]].append(h)
    cap = max(1, int(np.ceil(len(hh_rows) / max(n_r, 1))))
    syn_region_rows = []
    r_id = 1
    for (urban, band), mem in sorted(buckets.items()):
        for k in range(0, len(mem), cap):
            sub = mem[k:k + cap]
            for h in sub:
                hh_rows[h][1] = r_id
            syn_region_rows.append([r_id, urban, band])
            r_id += 1

    return {'region': np.array(syn_region_rows, dtype=int),
            'household': np.array(hh_rows, dtype=int) if hh_rows else np.zeros((0, 4), dtype=int),
            'individual': np.array(ind_rows, dtype=int) if ind_rows else np.zeros((0, 5), dtype=int)}


# ============================================================
# 4. Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_regions', type=int, default=80)
    ap.add_argument('--epsilons', type=str, default='0.8,3.2,8.0')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--n_queries', type=int, default=150)
    ap.add_argument('--methods', type=str,
                    default='pertable,lava_asis,lava_prop,denorm')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    epsilons = [float(x) for x in args.epsilons.split(',')]
    methods = args.methods.split(',')

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'multihop_gap_results.json')

    all_results = {'config': vars(args), 'results': {}}
    rng_q = np.random.default_rng(123)
    queries = make_queries(rng_q, args.n_queries)

    for eps in epsilons:
        eps_key = str(eps)
        all_results['results'][eps_key] = {}
        for m in methods:
            run_metrics = []
            timings = []
            for run in range(args.runs):
                seed = 42 + run
                data = generate_multihop_data(args.n_regions, seed=seed)
                real = {'region': data['region'], 'household': data['household'],
                        'individual': data['individual']}
                t0 = time.time()
                try:
                    if m == 'pertable':
                        syn = run_pertable_mrf(data, eps, seed=seed)
                    elif m == 'lava_asis':
                        syn = run_lava_pairwise(data, eps, propagate=False, seed=seed)
                    elif m == 'lava_prop':
                        syn = run_lava_pairwise(data, eps, propagate=True, seed=seed)
                    elif m == 'denorm':
                        syn = run_denorm_full(data, eps, seed=seed)
                    elif m in ('chain_direct', 'chain_mixture'):
                        from chain_npm import run_chain_npm
                        syn = run_chain_npm(
                            data, eps,
                            mode='direct' if m == 'chain_direct' else 'mixture')
                    elif m.startswith('chain_') and (
                            '_cross' in m):
                        from chain_npm import run_chain_npm
                        base, alloc = m.rsplit('_', 1)  # ..._cross / ..._cross8
                        alloc = 'cross' if alloc == 'cross' else alloc
                        syn = run_chain_npm(
                            data, eps,
                            mode='direct' if base == 'chain_direct' else 'mixture',
                            alloc=alloc)
                    else:
                        raise ValueError(f'unknown method {m}')
                    metrics = evaluate(real, syn, queries)
                except Exception as e:
                    logger.error(f'{m} eps={eps} run={run} FAILED: {e}')
                    import traceback; traceback.print_exc()
                    metrics = {'error': str(e)}
                dt = time.time() - t0
                timings.append(dt)
                run_metrics.append(metrics)
                logger.info(f'[{m} eps={eps} run={run}] {dt:.1f}s '
                            + (f"2hop={metrics.get('2hop'):.3f} "
                               f"delta_ratio={metrics.get('delta_ratio')}"
                               if 'error' not in metrics else metrics['error']))

            # aggregate medians across runs
            agg = {}
            keys = ['1hop_rh', '1hop_hi', '2hop', '2hop_wide',
                    '1hop_rh_large', '1hop_hi_large', '2hop_large',
                    '2hop_wide_large', 'delta_ratio', 'delta_syn', 'delta_real']
            for k in keys:
                vals = [r[k] for r in run_metrics if r.get(k) is not None]
                agg[k] = float(np.median(vals)) if vals else None
            agg['time_s'] = float(np.median(timings))
            agg['runs'] = run_metrics
            all_results['results'][eps_key][m] = agg

        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # summary table
    print('\n' + '=' * 100)
    print(f"{'method':<14} {'eps':<6} {'1hop_rh':<9} {'1hop_hi':<9} "
          f"{'2hop':<9} {'2hop_w':<9} {'2hop_L':<9} {'delta_ratio':<11}")
    print('=' * 100)
    for eps_key, ms in all_results['results'].items():
        for m, agg in ms.items():
            fmt = lambda x: f'{x:.3f}' if x is not None else 'NA'
            print(f"{m:<14} {eps_key:<6} {fmt(agg.get('1hop_rh_large')):<9} "
                  f"{fmt(agg.get('1hop_hi_large')):<9} {fmt(agg.get('2hop_large')):<9} "
                  f"{fmt(agg.get('2hop_wide_large')):<9} {fmt(agg.get('2hop')):<9} "
                  f"{fmt(agg.get('delta_ratio')):<11}")
    print('=' * 100)
    print(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
