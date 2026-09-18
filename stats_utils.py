"""Statistical rigor utilities (experiment plan 2026-09-02).

Conventions (pre-registered): 10 seeds per configuration; common random
numbers shared across methods (same data seeds, query-workload seed,
planting seeds); 95% CIs by bootstrap; paired comparisons per seed with
Wilcoxon signed-rank; family-wise control by Holm on the declared primary
family (Chain-NPM vs each baseline per dataset/epsilon/metric).
"""

import numpy as np
from scipy import stats


def ci95(xs, n_boot=2000, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) < 3:
        return float(xs.mean()), float(xs.min()), float(xs.max())
    rng = np.random.default_rng(seed)
    med = float(np.median(xs))
    boots = np.array([np.median(rng.choice(xs, len(xs), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, float(lo), float(hi)


def paired_test(a, b):
    """Per-seed paired comparison; Wilcoxon signed-rank with a paired-t
    fallback when the rank test is degenerate (many ties)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    try:
        p = stats.wilcoxon(a, b, zero_method='wilcox').pvalue
    except ValueError:
        p = stats.ttest_rel(a, b).pvalue
    return float(p)


def holm(pvals):
    """Holm-Bonferroni; returns (reject_bool_array, adjusted_p)."""
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj <= 0.05, adj


def summarize(per_seed):
    """per_seed: dict seed -> value. Returns the pre-registered summary."""
    xs = np.array([per_seed[s] for s in sorted(per_seed)])
    med, lo, hi = ci95(xs)
    return dict(n=len(xs), median=med, ci_lo=lo, ci_hi=hi,
                runs=xs.tolist())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 10)
    b = rng.normal(0.2, 0.08, 10)
    print("ci95(a):", ["%.3f" % x for x in ci95(a)])
    print("paired p:", "%.2e" % paired_test(a, b))
    rej, adj = holm([1e-4, 0.02, 0.4])
    print("holm:", rej, ["%.3f" % x for x in adj])
    print("summarize:", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in summarize({i: a[i] for i in range(10)}).items()
                         if k != 'runs'})
