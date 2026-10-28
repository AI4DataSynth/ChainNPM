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

**The reported arm.**  The paper's Chain-NPM numbers enable one extra
post-processing step:

```bash
--absorb 'pubn,rowleaf,szthresh,exactmarg,repair,growcap,szquota'
```

`szquota` recalibrates the drawn child group sizes against the per-parent-value
row counts read off the row-level adjacent marginals the mechanism already
releases; `rowleaf` makes those marginals row-level, which is the
**one-individual** adjacency the paper uses throughout (`all` deliberately
excludes `rowleaf` because it changes the adjacency, so it must be requested
explicitly).  The step spends no budget and changes no released marginal or
query; with the extra flag the loop above becomes

```bash
python chain_npm_${ds}.py --eps $eps --seed $seed --mode natural \
    --method chainnpm \
    --absorb 'pubn,rowleaf,szthresh,exactmarg,repair,growcap,szquota'
```

and the per-cell archives land in `results/arms/absorb/natural/` (these feed
Table `tab:re-appendix` and Figures `fig:dataset_sweep` / `fig:query_class_sweep`).
Without the flag the default run is bit-identical to the plain release
(verified byte-for-byte; `results/by_dataset/` is that default arm).

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
* **PrivMRF naming.** The per-table baseline appears as `pertable` in
  `results/by_dataset/`, `results/tables/p7_effect.tsv` and
  `results/tables/tab_re_appendix.tsv`, and as `privmrf` in
  `results/tables/p7_summary.tsv`; both are the manuscript's *PrivMRF*.
* **Planted-mode cells.** The per-cell planted records of the reported arm are
  shipped under `results/arms/absorb/plant/` (the released-arm planted cells live
  in `results/by_dataset/`, where the planted grid also enters as aggregated
  tables and figures).
* **The reported arm and the adjacency convention.** Every Chain-NPM number that
  enters the paper comes from the `--absorb 'pubn,rowleaf,szthresh,exactmarg,
  repair,growcap,szquota'` run archived under `results/arms/absorb/`, and every
  number — arm or default — is under the **one-individual** adjacency (a single
  base-record insertion or deletion). `rowleaf` is what makes the row-level
  objects the arm reads exact under that convention; the cascade convention is
  not used anywhere in the paper.

## 8. The controlled multi-hop benchmark

`multihop_gap_experiment.py` runs the self-contained controlled benchmark
(`REGION <- HOUSEHOLD <- INDIVIDUAL`, 200 regions, planted latent-mediated
two-hop structure, 150 pre-registered random conjunctive queries per hop class).
It needs no input data — the planted database is generated from the seeds — and
it feeds the controlled-benchmark results of the paper: Figure
`fig:controlled` (both panels: the Δ-preservation ratio and the 2-hop query error)
and, in the appendix, the allocation study `fig:alloc` and the ancestor-size sweep
`tab:scale`.

The paper's grid is ten runs (seeds 42–51) at ε ∈ {0.8, 3.2, 8.0}. One JSON per
budget, as shipped in `results/controlled/`:

```bash
for eps in 0.8 3.2 8.0; do
  # the two Chain-NPM variants (the reported arm: sampled-object budget split,
  # household count derived from RH_SIZE, EMP calibration after drawing)
  python multihop_gap_experiment.py --n_regions 200 --epsilons $eps --runs 10 \
      --n_queries 150 --methods chain_direct,chain_mixture \
      --out results/controlled/chain_results_eps${eps}_10run.json
done
```

The four comparison methods (`pertable`, `lava_asis`, `lava_prop`, `denorm`) run
through the same driver and write `base_results_eps<ε>_10run.json`; they need the
third-party backends (a PrivLava checkout plus `unified_comparison.py`), which are
not vendored — point `CHAINNPM_BASELINES` at the directory holding them:

```bash
export CHAINNPM_BASELINES=/path/to/baseline-repos   # contains unified_comparison.py
python multihop_gap_experiment.py --n_regions 200 --epsilons 0.8 --runs 10 \
    --n_queries 150 --methods pertable,lava_asis,lava_prop,denorm \
    --out results/controlled/base_results_eps0.8_10run.json
```

Runtime: about **2 s per cell** on one core (measured 1.1–1.7 s for the Chain-NPM
methods at this scale); the ten-run, two-method, three-budget grid is therefore
under a minute, and the comparison methods are slower but still CPU-only except
where their own backends require a GPU. Every run is deterministic given its seed:
re-running the grid reproduces both the medians and the per-seed entries
(`runs[]`) of the shipped files exactly, so a partial grid can be checked against
the archive without re-running all ten seeds — `--runs 2` reproduces the first two
`runs[]` entries of every cell bit-for-bit.

**The ARM the benchmark measures.** `chain_npm.py` here is the reported arm: the
γ²-split covers only the eight objects the samplers actually consume
(`RB`, `RH_OWN`, `RH_SIZE`, `RI_URB_EMP`, `RHI_U_O_E`, `RHI_U_O_A`, `RHI_U_S_E`,
`II_AGE_EDU`) plus the households-per-region count; the mean household count is
derived from the released `RH_SIZE` table rather than noised separately; and after
drawing, EMP is re-weighted within each `(URBAN, OWN)` group by a bisection
multiplier so the synthetic URBAN–EMP contrast matches the noisy `RI_URB_EMP`
marginal. Both steps are pure post-processing of the released statistics.