"""
Chain-NPM: joint chain marginals + EM-as-post-processing
============================================================

Core mechanism (B+NPM direction):
  Phase 1 (ONLY budget-consuming step): ONE pass over the FK chain computing
  low-order JOINT marginals that span multiple FK hops (incl. the 2-hop
  region x individual marginals that per-FK methods never compute).
  Gaussian mechanism with gamma^2-split across queries.

  Phase 2 (FREE post-processing): normalization, latent-mixture EM, and
  synthesis all operate on the noisy statistics only.  EM is demoted from
  "budget consumer" (PrivLava: fresh noisy marginals every iteration) to
  "free post-processing on noisy NPMs".

DP accounting (prototype; exact sensitivity analysis belongs in the paper):
  - Every marginal is a count query computed once on the raw data.
  - Group contributions are truncated at tau and normalized per-group so a
    single individual's worst-case L2 influence is <= 1 for the 1/size-
    normalized marginals (m_HI, m_RI, m_RHI), <= 1 for household-level
    marginals (m_R, m_H, m_RH), and <= 2 for within-group pair marginals.
  - gamma_i^2 = gamma_max^2 / Q per query; sigma_i = sens_i * sqrt(2 / gamma_i^2).
  - Total (eps, delta)-DP by standard Gaussian composition over gamma^2.

Schema (matches tmp/multihop_gap_experiment.py):
  REGION[R_ID, URBAN, BAND] <- HOUSEHOLD[H_ID, REGION_ID, OWN, SIZE]
    <- INDIVIDUAL[I_ID, AGE, EMP, EDU, H_ID]
"""

import numpy as np
import logging

logger = logging.getLogger('ChainNPM')


# ----------------------------------------------------------------------
# Budget
# ----------------------------------------------------------------------

def analytic_gaussian_gamma(epsilon, delta):
    """Bisection for gamma where Gaussian mechanism with sensitivity gamma
    achieves (epsilon, delta)-DP (same convention as PrivHybrid/PrivLava)."""
    from scipy.stats import norm
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if mid <= 0:
            lo = mid
            continue
        val = (norm.cdf(mid / 2 - epsilon / mid)
               - np.exp(epsilon) * norm.cdf(-mid / 2 - epsilon / mid))
        if val <= delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return lo


def make_sigmas(epsilon, delta, n_queries, tau):
    """Equal gamma^2-split across n_queries marginal queries."""
    gamma_max = analytic_gaussian_gamma(epsilon, delta)
    gamma_i_sq = gamma_max ** 2 / n_queries
    sigma_unit = np.sqrt(2.0 / gamma_i_sq)  # sensitivity-1 sigma
    return {'sigma1': sigma_unit,            # sens <= 1 queries
            'sigma2': 2.0 * sigma_unit,      # within-group pair queries
            'gamma_max': gamma_max,
            'n_queries': n_queries}


def make_sigmas_weighted(epsilon, delta, weights):
    """Importance-weighted gamma^2-split (adaptive allocation).

    gamma_i^2 = gamma_max^2 * (w_i * sens_i^2) / sum_j(w_j * sens_j^2)
    => noise sigma_i = sens_i * sqrt(2 / gamma_i^2), so high-importance
    marginals (cross-hop) receive more budget / less noise.

    weights: dict marginal-name -> importance weight (>0).
    """
    gamma_max = analytic_gaussian_gamma(epsilon, delta)
    sens = {name: s for name, (_, _), s in MARGINALS}
    sens['HH_PER_REGION'] = 1  # extra statistic outside MARGINALS
    denom = sum(weights[name] * sens[name] ** 2 for name in weights)
    sigmas = {}
    for name in weights:
        g2 = gamma_max ** 2 * weights[name] * sens[name] ** 2 / denom
        sigmas[name] = sens[name] * np.sqrt(2.0 / g2)
    return {'per_query': sigmas, 'gamma_max': gamma_max}


def alloc_weights(scheme):
    """Preset importance-weight schemes for ablation."""
    w = {name: 1.0 for name, _, _ in MARGINALS}
    if scheme in ('cross', 'cross4'):
        mult = 4.0
    elif scheme == 'cross8':
        mult = 8.0
    elif scheme == 'cross2':
        mult = 2.0
    else:
        return w
    for name, (tbl, _), _ in MARGINALS:
        if tbl in ('ri', 'rhi'):
            w[name] = mult
    return w


# ----------------------------------------------------------------------
# Phase 1: one-pass noisy joint chain marginals
# ----------------------------------------------------------------------

# Marginal registry: (name, tables, attrs, sensitivity class)
# tables: r=region, h=household, i=individual
MARGINALS = [
    # region level (per-region counts)
    ('R_URBAN',    ('r', ('URBAN',)),          1),
    ('R_BAND',     ('r', ('BAND',)),           1),
    ('RB',         ('r', ('URBAN', 'BAND')),   1),
    # household level (per-household counts)
    ('H_OWN',      ('h', ('OWN',)),            1),
    ('H_SIZE',     ('h', ('SIZE',)),           1),
    ('HS',         ('h', ('OWN', 'SIZE')),     1),
    ('RH_OWN',     ('rh', ('URBAN', 'OWN')),   1),
    ('RH_SIZE',    ('rh', ('URBAN', 'SIZE')),  1),
    ('RH_BOWN',    ('rh', ('BAND', 'OWN')),    1),
    # 1-hop household x individual (per-individual / group_size)
    ('HI_OWN_EMP', ('hi', ('OWN', 'EMP')),     1),
    ('HI_OWN_AGE', ('hi', ('OWN', 'AGE')),     1),
    ('HI_OWN_EDU', ('hi', ('OWN', 'EDU')),     1),
    ('HI_SIZE_EMP', ('hi', ('SIZE', 'EMP')),   1),
    ('HI_SIZE_AGE', ('hi', ('SIZE', 'AGE')),   1),
    # 2-HOP region x individual  <-- the object per-FK methods never compute
    ('RI_URB_EMP', ('ri', ('URBAN', 'EMP')),   1),
    ('RI_URB_AGE', ('ri', ('URBAN', 'AGE')),   1),
    ('RI_URB_EDU', ('ri', ('URBAN', 'EDU')),   1),
    ('RI_BAND_EMP', ('ri', ('BAND', 'EMP')),   1),
    # 2-hop triple (small cells): the core cross-hop conditional
    ('RHI_U_O_E',  ('rhi', ('URBAN', 'OWN', 'EMP')), 1),
    ('RHI_U_O_A',  ('rhi', ('URBAN', 'OWN', 'AGE')), 1),
    ('RHI_U_S_E',  ('rhi', ('URBAN', 'SIZE', 'EMP')), 1),
    # within-group individual pairs
    ('II_AGE_EMP', ('ii', ('AGE', 'EMP')),     2),
    ('II_AGE_EDU', ('ii', ('AGE', 'EDU')),     2),
    ('II_EMP_EDU', ('ii', ('EMP', 'EDU')),     2),
]

R_COL = {'URBAN': 1, 'BAND': 2}
H_COL = {'REGION_ID': 1, 'OWN': 2, 'SIZE': 3}
I_COL = {'AGE': 1, 'EMP': 2, 'EDU': 3, 'H_ID': 4}

DOMAINS = {'URBAN': 2, 'BAND': 3, 'OWN': 2, 'SIZE': 5,
           'AGE': 8, 'EMP': 2, 'EDU': 4}


def compute_chain_marginals(region, household, individual, sigmas, tau=6):
    """Single raw-data pass; returns dict name -> noisy count ndarray.

    sigmas: either the legacy uniform dict (sigma1/sigma2) or a weighted
    dict {'per_query': {name: sigma}} from make_sigmas_weighted().
    """
    rng = np.random.default_rng(0)
    per_query = sigmas.get('per_query')

    def noise_sigma(name, sens):
        if per_query is not None:
            return per_query[name]
        return sigmas['sigma1'] if sens == 1 else sigmas['sigma2']

    r_by_id = {int(t[0]): t for t in region}
    h_by_id = {int(t[0]): t for t in household}
    members = {}
    for t in individual:
        members.setdefault(int(t[4]), []).append(t)

    hhs_per_region = {}
    for h in household:
        hhs_per_region.setdefault(int(h[1]), []).append(int(h[0]))

    acc = {name: np.zeros([DOMAINS[a] for a in attrs], dtype=np.float64)
           for name, (tbl, attrs), sens in MARGINALS}

    def idx(vals):
        return tuple(vals)

    # region-level
    for r in region:
        u, b = int(r[1]), int(r[2])
        acc['R_URBAN'][u] += 1
        acc['R_BAND'][b] += 1
        acc['RB'][u, b] += 1

    # household-level + region x household
    for h in household:
        r = r_by_id.get(int(h[1]))
        own, size = int(h[2]), int(h[3])
        acc['H_OWN'][own] += 1
        acc['H_SIZE'][size] += 1
        acc['HS'][own, size] += 1
        if r is not None:
            u, b = int(r[1]), int(r[2])
            acc['RH_OWN'][u, own] += 1
            acc['RH_SIZE'][u, size] += 1
            acc['RH_BOWN'][b, own] += 1

    # individual-level contributions, 1/size-normalized per group, tau-truncated
    for hid, mem in members.items():
        h = h_by_id.get(hid)
        if h is None:
            continue
        r = r_by_id.get(int(h[1]))
        own, size = int(h[2]), int(h[3])
        u = int(r[1]) if r is not None else 0
        b = int(r[2]) if r is not None else 0
        mem_t = mem[:tau]
        w = 1.0 / max(len(mem_t), 1)
        for t in mem_t:
            age, emp, edu = int(t[1]), int(t[2]), int(t[3])
            acc['HI_OWN_EMP'][own, emp] += w
            acc['HI_OWN_AGE'][own, age] += w
            acc['HI_OWN_EDU'][own, edu] += w
            acc['HI_SIZE_EMP'][size, emp] += w
            acc['HI_SIZE_AGE'][size, age] += w
            acc['RI_URB_EMP'][u, emp] += w
            acc['RI_URB_AGE'][u, age] += w
            acc['RI_URB_EDU'][u, edu] += w
            acc['RI_BAND_EMP'][b, emp] += w
            acc['RHI_U_O_E'][u, own, emp] += w
            acc['RHI_U_O_A'][u, own, age] += w
            acc['RHI_U_S_E'][u, size, emp] += w
        # within-group pairs (bounded count like PrivHybrid prototype)
        n_pairs = min(len(mem_t) * (len(mem_t) - 1), 20)
        if n_pairs > 0:
            pw = 1.0 / n_pairs
            cnt = 0
            for a_i in range(len(mem_t)):
                for b_i in range(a_i + 1, len(mem_t)):
                    if cnt >= n_pairs:
                        break
                    t1, t2 = mem_t[a_i], mem_t[b_i]
                    acc['II_AGE_EMP'][int(t1[1]), int(t2[2])] += pw
                    acc['II_AGE_EDU'][int(t1[1]), int(t2[3])] += pw
                    acc['II_EMP_EDU'][int(t1[2]), int(t2[3])] += pw
                    cnt += 1

    noisy = {}
    for name, (tbl, attrs), sens in MARGINALS:
        sigma = noise_sigma(name, sens)
        noisy[name] = np.maximum(0.0, acc[name]
                                 + rng.normal(0, sigma, acc[name].shape))
    # household-count stat (sens 1): mean households per region
    cnts = np.array([len(v) for v in hhs_per_region.values()], dtype=float)
    cnt_sigma = (per_query['HH_PER_REGION'] if per_query is not None
                 and 'HH_PER_REGION' in per_query
                 else sigmas.get('sigma1', 1.0))
    noisy['HH_PER_REGION'] = max(0.0, cnts.mean() + rng.normal(0, cnt_sigma))
    return noisy


def _cond(joint, axis):
    """p(rest | axis values) from a noisy joint count array.

    axis: int or tuple of ints — the axes to condition ON.
    """
    if isinstance(axis, int):
        axis = (axis,)
    rest = tuple(i for i in range(joint.ndim) if i not in axis)
    total = joint.sum(axis=rest, keepdims=True) if rest else joint
    total = np.where(total <= 0, 1.0, total)
    return joint / total


# ----------------------------------------------------------------------
# Phase 2a: direct synthesis from noisy marginals (free post-processing)
# ----------------------------------------------------------------------

class ChainDirectSynthesizer:
    def __init__(self, noisy, seed=42):
        self.m = noisy
        self.rng = np.random.default_rng(seed)

    def synthesize(self, n_regions):
        m = self.m
        rng = self.rng

        # regions
        p_rb = m['RB'].copy()
        p_rb /= max(p_rb.sum(), 1e-9)
        syn_region = []
        for r in range(1, n_regions + 1):
            u, b = self._sample2(p_rb)
            syn_region.append([r, u, b])
        syn_region = np.array(syn_region, dtype=int)

        # households per region
        hh_per_r = max(1, int(round(m['HH_PER_REGION'])))

        # conditionals
        p_own_given_u = _cond(m['RH_OWN'], 0)          # p(OWN|URBAN)
        p_size_given_u = _cond(m['RH_SIZE'], 0)        # p(SIZE|URBAN)
        p_emp_given_u_o = _cond(m['RHI_U_O_E'], (0, 1))  # p(EMP|URBAN,OWN)
        p_age_given_u_o = _cond(m['RHI_U_O_A'], (0, 1))  # p(AGE|URBAN,OWN)
        p_edu_given_age = _cond(m['II_AGE_EDU'], 0)      # p(EDU|AGE)

        syn_hh, syn_ind = [], []
        h_id, i_id = 1, 1
        for r in range(1, n_regions + 1):
            u = int(syn_region[r - 1, 1])
            n_h = max(1, hh_per_r + int(rng.integers(-1, 2)))
            for _ in range(n_h):
                own = self._sample1(p_own_given_u[u])
                size = self._sample1(p_size_given_u[u])
                syn_hh.append([h_id, r, own, size])
                for _ in range(size + 1):
                    emp = self._sample1(p_emp_given_u_o[u, own])
                    age = self._sample1(p_age_given_u_o[u, own])
                    edu = self._sample1(p_edu_given_age[age])
                    syn_ind.append([i_id, age, emp, edu, h_id])
                    i_id += 1
                h_id += 1

        return {'region': syn_region,
                'household': np.array(syn_hh, dtype=int),
                'individual': np.array(syn_ind, dtype=int)}

    def _sample1(self, p):
        p = np.asarray(p, dtype=float)
        s = p.sum()
        if s <= 0:
            return int(self.rng.integers(len(p)))
        return int(self.rng.choice(len(p), p=p / s))

    def _sample2(self, p):
        p = np.asarray(p, dtype=float)
        s = p.sum()
        if s <= 0:
            return (0, 0)
        flat = p.flatten() / s
        ij = int(self.rng.choice(len(flat), p=flat))
        return (ij // p.shape[1], ij % p.shape[1])


# ----------------------------------------------------------------------
# Phase 2b: latent-mixture EM on noisy marginals (free post-processing)
# ----------------------------------------------------------------------

class ChainMixtureSynthesizer:
    """EM learns k latent household types Z2 from noisy 2-hop marginals.

    Generative sketch per household:
        URBAN ~ p_R ;  Z2 ~ p(Z2|URBAN) ;
        (OWN, SIZE) ~ p(.|Z2, URBAN)  [v0: p(OWN|URBAN), p(SIZE|URBAN)]
        individuals: EMP ~ p(EMP|Z2, URBAN), AGE ~ p(AGE|Z2), EDU ~ p(EDU|AGE)
    All parameters estimated by soft cell attribution on noisy counts.
    """

    def __init__(self, noisy, k=4, n_iter=30, seed=42):
        self.m = noisy
        self.k = k
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def _fit(self):
        m = self.m
        k = self.k
        rng = self.rng

        p_z_u = np.full((2, k), 1.0 / k)            # p(Z2|URBAN)
        p_emp_z_u = np.full((k, 2, 2), 0.5)         # p(EMP|Z2,URBAN)
        p_age_z = np.full((k, 8), 1.0 / 8)          # p(AGE|Z2)
        p_own_u = _cond(m['RH_OWN'], 0)
        p_size_u = _cond(m['RH_SIZE'], 0)

        # observed noisy joints used for attribution
        riu = m['RHI_U_O_E']     # URBAN x OWN x EMP  (OWN not in model -> marginalize)
        riu_age = m['RHI_U_O_A']  # URBAN x OWN x AGE
        ri_emp = m['RI_URB_EMP']  # URBAN x EMP

        for _ in range(self.n_iter):
            # ---- E-step: attribute EMP-cell mass over (URBAN, OWN, EMP)
            num_z_u = np.zeros((2, k))
            num_emp = np.zeros((k, 2, 2))
            num_age = np.zeros((k, 8))

            for u in range(2):
                for o in range(2):
                    for e in range(2):
                        mass = riu[u, o, e]
                        if mass <= 0:
                            continue
                        w = p_z_u[u] * p_emp_z_u[:, u, e]
                        if w.sum() <= 0:
                            continue
                        w = w / w.sum()
                        num_z_u[u] += mass * w
                        for z in range(k):
                            num_emp[z, u, e] += mass * w[z]
            # AGE attribution via URBAN x OWN x AGE
            for u in range(2):
                for o in range(2):
                    for a in range(8):
                        mass = riu_age[u, o, a]
                        if mass <= 0:
                            continue
                        w = p_z_u[u].copy()
                        if w.sum() <= 0:
                            continue
                        w = w / w.sum()
                        for z in range(k):
                            num_age[z, a] += mass * w[z]

            # ---- M-step
            for u in range(2):
                if num_z_u[u].sum() > 0:
                    p_z_u[u] = num_z_u[u] / num_z_u[u].sum()
            for z in range(k):
                for u in range(2):
                    tot = num_emp[z, u].sum()
                    if tot > 0:
                        p_emp_z_u[z, u] = num_emp[z, u] / tot
                tot = num_age[z].sum()
                if tot > 0:
                    p_age_z[z] = num_age[z] / tot

        self.p_z_u, self.p_emp_z_u, self.p_age_z = p_z_u, p_emp_z_u, p_age_z
        self.p_own_u, self.p_size_u = p_own_u, p_size_u
        self.p_edu_age = _cond(m['II_AGE_EDU'], 0)

    def synthesize(self, n_regions):
        if not hasattr(self, 'p_z_u'):
            self._fit()
        m, rng = self.m, self.rng

        p_rb = m['RB'].copy()
        p_rb /= max(p_rb.sum(), 1e-9)
        syn_region = []
        for r in range(1, n_regions + 1):
            flat = p_rb.flatten()
            ij = int(rng.choice(len(flat), p=flat / flat.sum()))
            syn_region.append([r, ij // p_rb.shape[1], ij % p_rb.shape[1]])
        syn_region = np.array(syn_region, dtype=int)

        hh_per_r = max(1, int(round(m['HH_PER_REGION'])))
        syn_hh, syn_ind = [], []
        h_id, i_id = 1, 1
        for r in range(1, n_regions + 1):
            u = int(syn_region[r - 1, 1])
            n_h = max(1, hh_per_r + int(rng.integers(-1, 2)))
            for _ in range(n_h):
                own = self._sample(self.p_own_u[u])
                size = self._sample(self.p_size_u[u])
                z = self._sample(self.p_z_u[u])
                syn_hh.append([h_id, r, own, size])
                for _ in range(size + 1):
                    emp = self._sample(self.p_emp_z_u[z, u])
                    age = self._sample(self.p_age_z[z])
                    edu = self._sample(self.p_edu_age[age])
                    syn_ind.append([i_id, age, emp, edu, h_id])
                    i_id += 1
                h_id += 1
        return {'region': syn_region,
                'household': np.array(syn_hh, dtype=int),
                'individual': np.array(syn_ind, dtype=int)}

    def _sample(self, p):
        p = np.asarray(p, dtype=float)
        s = p.sum()
        if s <= 0:
            return int(self.rng.integers(len(p)))
        return int(self.rng.choice(len(p), p=p / s))


# ----------------------------------------------------------------------
# One-call entry used by the gap experiment
# ----------------------------------------------------------------------

def run_chain_npm(data, epsilon, mode='direct', k=4, alloc='uniform'):
    """data: dict from multihop_gap_experiment.generate_multihop_data().

    alloc: 'uniform' (equal gamma^2 split) or 'cross'/'cross8'
    (importance-weighted split favoring cross-hop marginals).
    """
    n_ind = data['individual'].shape[0]
    delta = 1.0 / max(n_ind, 1)
    if alloc == 'uniform':
        sigmas = make_sigmas(epsilon, delta, n_queries=len(MARGINALS), tau=6)
    else:
        weights = alloc_weights(alloc)
        weights['HH_PER_REGION'] = 1.0
        sigmas = make_sigmas_weighted(epsilon, delta, weights)
    noisy = compute_chain_marginals(data['region'], data['household'],
                                    data['individual'], sigmas, tau=6)
    n_regions = data['region'].shape[0] + int(
        np.random.default_rng(0).integers(-3, 4))
    n_regions = max(1, n_regions)
    if mode == 'mixture':
        synth = ChainMixtureSynthesizer(noisy, k=k, seed=0)
    else:
        synth = ChainDirectSynthesizer(noisy, seed=0)
    return synth.synthesize(n_regions)
