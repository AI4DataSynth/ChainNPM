# Reproducing the Chain-NPM experiments

All experiments are driven by the per-dataset drivers, which share the pipeline in
`p4_common.py`; the query workload and the reported statistics are fixed in
`prereg.py` (pre-registered before any result was produced).

## 1. Environment

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt           # numpy, pandas, scipy
export CHAINNPM_ROOT=$PWD/work            # where inputs and outputs live
```

Everything is CPU-only: no GPU is required for Chain-NPM itself (the comparison
methods PrivPetal / LavaProp require a GPU and their own environments).

## 2. Input data

Each dataset is stored as discretized CSVs plus one domain JSON per table, under
`$CHAINNPM_ROOT/tmp/data/<dataset>/processed/` (the planted variants of
`plant_product` / `plant_xor` are generated below). File names, primary keys,
attributes and FK columns per dataset are declared in the driver's `CFG`.

| Dataset | Shape | Tables (file).`[pk, attrs..., fk]` | Raw source |
|---|---|---|---|
| `financial` | chain `district <- account <- trans` | `district.csv`, `account.csv`, `trans.csv` | PKDD'99 CTU financial dataset |
| `imdb` | V-shape `names <- principals -> titles` | `names.csv`, `titles.csv`, `principals.csv` | IMDb official snapshot subset |
| `instacart` | V-shape `products <- order_products -> orders` | `products.csv`, `orders.csv`, `order_products.csv` | Instacart online-grocery dataset |
| `movielens` | V-shape `users <- ratings -> movies` | `users.csv`, `movies.csv`, `ratings.csv` | MovieLens |
| TPC-H | chain `customer <- orders <- lineitem` | `tpch_2tab`, `TPC-H-deep` (see below) | TPC-H `dbgen` (SF=1) |
| ACS PUMS CA'22 | chain `household <- individual` | `household.csv`, `individual.csv` | ACS PUMS 2022, California |

Preprocessing discretizes every attribute and writes the matching
`*_domain.json` with `{attribute: {"size": domain_size}}`. Continuous attributes
are binned; identifiers stay as their original integer keys. `delta = 1 / n` with
`n` the leaf-table row count, and `tau` (the group-size truncation) is picked by the
pre-registered rule `clip(p95(group sizes), 12, 1024)`; both are recorded in every
result JSON.

TPC-H variants:

```bash
# 2-table private chain (customer/public-side attrs joined into orders)
python chain_npm_tpch_v2.py --data_dir <tpch_data_dir> --epsilon 3.2 \
    --out_prefix $CHAINNPM_ROOT/tmp/tpch/chainnpm_eps3.2

# deep 3-table chain: build the ground truth first, then synthesize
python make_deep_gt.py --data_root <tpch_data_root> --out_dir <tpch_data_root>/TPC-H-deep
python chain_npm_tpch_deep.py --data_dir <tpch_data_root>/TPC-H-deep --epsilon 3.2 \
    --out_prefix $CHAINNPM_ROOT/tmp/deep/chainnpm_eps3.2

# ACS PUMS (household <- individual)
python chain_npm_acs.py --data_dir <acs_data_dir> --epsilon 3.2 --seed 42 \
    --out_prefix $CHAINNPM_ROOT/tmp/acs/chainnpm_eps3.2
```

## 3. The four-dataset natural grid (paper Figure 1, Table "per-hop RE")

ε ∈ {0.1, 0.2, 0.4, 0.8, 1.6, 3.2}, seeds 42–51, methods
`chainnpm`, `pertable`, `privmrf`, `privbayes`, `pbpgm`, `lavaprop`
(`denorm` is an internal control that the paper does not report):

```bash
for ds in financial imdb instacart movielens; do
  for eps in 0.1 0.2 0.4 0.8 1.6 3.2; do
    for seed in $(seq 42 51); do
      for m in chainnpm pertable privmrf privbayes pbpgm lavaprop; do
        python chain_npm_${ds}.py --eps $eps --seed $seed --mode natural --method $m
      done
    done
  done
done
```

Each run evaluates 150 pre-registered random conjunctive count queries per hop
class (`prereg.QUERY_SEED`), and reports the median relative error over all queries
and over large-count queries (`answer >= prereg.LARGE_COUNT`). Recorded results
reproduce the files shipped in `results/by_dataset/`.

## 4. Plant-and-Audit (cross-hop retention)

`plant_audit.py` keeps every real one-hop marginal and all FK cardinalities exact
and plants a latent bit plus a structured leaf response at a controlled strength,
so that the true cross-hop effect is known. The grid used in the paper:

* `plant_product`, targets Δ ∈ {0.10, 0.20, 0.316, 0.40}, ε ∈ {0.8, 3.2}, seeds 42–51
* `plant_xor`, φ = 0.8, ε ∈ {0.8, 3.2}, seeds 42–51
* `plant_family:{mod4,threshold,or}` (rule-family variants), ε = 3.2, seeds 42–44

```bash
python chain_npm_financial.py --eps 0.8 --seed 42 --mode plant_product \
    --target 0.10 --method chainnpm
python chain_npm_financial.py --eps 0.8 --seed 42 --mode plant_xor --method chainnpm
python chain_npm_financial.py --eps 3.2 --seed 42 --mode plant_family:mod4 --method chainnpm
```

Planted inputs are materialized once and shared by every method (this is what makes
the comparison paired):

```bash
python make_planted_csvs.py            # writes $CHAINNPM_ROOT/tmp/data_planted/<ds>/<mode>/
```

For third-party baselines that run as external processes, feed them the planted
CSVs and convert their outputs with
`python eval_privpetal_convert.py --name <ds> --mode <mode> --eps <e> --seed <s>`.

## 5. Summary tables

`results/tables/` contains the aggregation scripts used for the paper's summary
tables; they read the shipped archive (`results/by_dataset/`) when present, or a
live `tmp/results/` tree:

```bash
python results/tables/p7_summary.py     # median per (dataset, series, eps) + bootstrap CI
python results/tables/p7_effect.py      # plant-and-audit retention per seed -> mean/median
python results/tables/p7_stats.py       # Wilcoxon signed-rank across margins + Holm correction
```

Outputs: `p7_summary.tsv`, `p7_effect.tsv`, `p7_stats.json`.

## 6. Runtime

Chain-NPM is CPU-only (pure numpy/scipy; the method code spawns no workers).
Measured per-cell wall-clock times for
the four-dataset natural grid (one core, leaf tables of 1.0–1.5M rows):

| Dataset | min | median | max |
|---|---|---|---|
| `financial` | 0.9 s | 3.8 s | 39 s |
| `imdb` | 60 s | 100 s | 164 s |
| `instacart` | 29 s | 66 s | 159 s |
| `movielens` | 2 s | 57 s | 110 s |

The full 1440-cell natural grid therefore costs a few CPU-hours and parallelizes
trivially across cores (one process per cell). The baselines dominate the total
cost: PrivPetal and LavaProp need GPU hours per cell, and the TPC-H deep-chain
baseline runs take hours.

## 7. Notes and caveats

* The processed inputs (discretized CSVs + domain JSONs) are not stored in this
  repository; they are derived from the raw sources listed in §2 by the
  preprocessing described there. Bin edges are fixed and recorded in the domain
  JSONs, and `prereg.py` fixes every other campaign constant.
* Comments and docstrings inside `results/tables/*.py` are partly in Chinese, as
  these scripts were extracted from the internal analysis tree.
* `results/by_dataset/` reports exactly the cells that entered the paper; a few
  cells have fewer than 10 seeds where a run was still in flight, and each record
  carries its own `seed`, so the aggregation is transparent.
* **Reproducibility of a single cell.** Chain-NPM's own cells are deterministic
  given `--seed` (no multiprocessing, no GPU, numpy legacy RNG only), so re-running
  one reproduces it exactly. The PrivPetal baseline cells are **not** bit-reproducible:
  its implementation dispatches per-attribute jobs through `multiprocessing.Pool`
  with `apply_async`, so which job a forked worker picks up — and therefore the
  random stream it consumes — varies from run to run even on one machine with the
  same seed. Two same-seed runs on the same GPU differ on the synthetic foreign-key
  assignment by as much as a cross-platform pair does (measured: same-machine TVD
  35% vs cross-machine 33% on the FK column), while the synthetic *parent* content is
  identical as a multiset and per-table marginal TVDs stay below 1%. Seeds are
  therefore a Monte-Carlo handle, not a bit-level reproducibility handle, for that
  baseline; the aggregate statistics (medians over 10 seeds) are unaffected.
* **PerTable naming.** The per-table baseline appears as `pertable` in
  `results/by_dataset/`, `results/tables/p7_effect.tsv` and
  `results/tables/tab_re_appendix.tsv`, and as `privmrf` in
  `results/tables/p7_summary.tsv`; both are the manuscript's *PerTable*.
* **Planted-mode cells.** The planted grid is shipped as aggregated tables and
  figures only; the per-cell planted records live in the live campaign tree
  (`tmp/results/<dataset>/`), which is not part of this repository.