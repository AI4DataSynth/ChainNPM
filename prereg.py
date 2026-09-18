"""Pre-registered constants for experiment campaign P4 (2026-09-02).

Everything here is FIXED before any result is produced so the campaign is
reproducible and free of post-hoc tuning.  Drivers and the evaluator must
import from here rather than re-declare their own defaults.

Conventions
-----------
- 10 DP seeds per configuration (SEEDS); query workload is seeded once by
  QUERY_SEED so all methods answer the exact same query set.
- delta = 1 / n where n is the number of rows of the LEAF (child/bot)
  table, matching chain_npm_acs.py / multihop_gap_experiment.py.
- A count query is "large" (trusted denominator) when its real answer is
  >= LARGE_COUNT.
- tau is the Chain-NPM group-size truncation, pre-registered by the rule
  tau = clip(p95(group sizes), 12, 1024) (see pick_tau); the p95 keeps
  truncation bias small on heavy-tailed group-size schemas while bounding
  the SZ table width.  The picked value is recorded in every result JSON.
  The smaller grid {6, 8, 12} remains the tau-ablation grid of the
  controlled benchmark only.  k is the latent domain size handed to the
  PrivLava latent-propagation baseline.
"""

import numpy as np

# --- DP sweep -------------------------------------------------------------
SEEDS = tuple(range(42, 52))          # 10 seeds
EPS = [0.2, 0.8, 3.2]                 # privacy budgets

# --- query workload -------------------------------------------------------
QUERY_SEED = 123                      # one shared workload seed
N_QUERIES = 150                       # random conjunctive queries per class
LARGE_COUNT = 50                      # "large count" RE threshold

# --- defaults -------------------------------------------------------------
K_DEFAULT = 4                         # latent domain size for lavaprop
TAU_CHOICES = (6, 8, 12)              # tau-ablation grid (controlled only)
TAU_LO, TAU_HI = 12, 1024             # clip range of the p95 rule


def delta_for(n_leaf):
    """delta = 1 / n for the leaf table row count n."""
    return 1.0 / max(int(n_leaf), 1)


def pick_tau(group_sizes, lo=TAU_LO, hi=TAU_HI):
    """Pre-registered truncation rule: tau = clip(p95(group sizes), lo, hi).

    Deterministic given the data; recorded in every result JSON.
    """
    return int(np.clip(np.percentile(np.asarray(group_sizes), 95), lo, hi))


if __name__ == "__main__":
    print("SEEDS", SEEDS)
    print("EPS", EPS)
    print("QUERY_SEED", QUERY_SEED, "N_QUERIES", N_QUERIES,
          "LARGE_COUNT", LARGE_COUNT)
    print("K_DEFAULT", K_DEFAULT, "TAU_CHOICES", TAU_CHOICES)
    print("delta_for(1000000)", delta_for(1000000))
    for sizes in [np.full(50, 5), np.arange(1, 100), np.full(50, 500)]:
        print("pick_tau(p95 of %s)=%d" % (sizes[[0, -1]], pick_tau(sizes)))
