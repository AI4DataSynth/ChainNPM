# Chain-NPM

Reference implementation and experiment code for

> **Chain-NPM: Preserving Multi-Hop Correlations in Differentially Private Relational Data Synthesis**

Chain-NPM is a differentially private (DP) synthesizer for relational databases.
State-of-the-art multi-table methods model each foreign key (FK) in isolation and
compose the per-FK models afterwards; a correlation spanning more than one FK hop
then has to travel through lossy intermediate representations, each consuming
privacy budget. Chain-NPM instead

1. spends the entire budget on a **single pass of noisy low-order joint marginals**
   that explicitly include the cross-hop statistics, and
2. demotes latent-type inference (EM) to **budget-free post-processing** on the
   noisy statistics.

## Repository layout

| Path | Contents |
|---|---|
| `chain_npm.py` | the original three-table census-style synthesizer (one-hop + NPM design) |
| `chain_npm_generic.py` | ChainNPM2: generic two-level FK chain (`parent <- child`) |
| `chain_npm_3level.py` | ChainNPM3: three-level FK chain (`top <- mid <- bot`), all tables private |
| `chain_npm_vshape.py` | ChainNPMV: V-shape / star extension (`root1 <- child -> root2`) |
| `p4_common.py` | shared pipeline (load → plant → synthesize → evaluate → dump), campaign protocol (query workload, effect metrics, recording) and per-method dispatch |
| `chain_npm_{financial,imdb,instacart,movielens}.py` | dataset drivers with the schema configs used in the paper |
| `chain_npm_tpch.py`, `chain_npm_tpch_v2.py`, `chain_npm_tpch_deep.py` | TPC-H drivers (two-table and deep three-table private chains) |
| `chain_npm_acs.py` | ACS PUMS driver (household ← individual) |
| `eval_newdata.py` | evaluator: pre-registered random conjunctive count queries, per-hop median relative error, cross-hop effect |
| `acs_eval.py`, `tpch_deep_eval.py`, `eval_privpetal_convert.py` | ACS, deep-TPC-H and baseline-directory evaluators |
| `plant_audit.py` | Plant-and-Audit protocol: semi-synthetic cross-hop ground truth on real chains |
| `make_planted_csvs.py`, `make_deep_gt.py` | planted-input generation (for third-party baselines) and deep-TPC-H ground truth |
| `baselines.py` | baseline adapters: per-table PrivMRF + FK re-linking, full denormalization, LavaProp |
| `prereg.py`, `stats_utils.py` | pre-registered query seeds/thresholds; bootstrap and Wilcoxon helpers |
| `results/by_dataset/` | per-cell metrics for every (dataset, method, mode, ε, seed) reported in the paper |
| `results/tables/` | summary tables (median + bootstrap CI, significance, effect retention) |
| `docs/REPRODUCE.md` | data preparation and exact reproduction commands |
| `preprocess/` | rebuilds the discretized inputs of every dataset from the public raw sources, with a per-source licence review; `preprocess/MANIFEST.md` lists the reference build's file hashes |

## Install

Python ≥ 3.9 (we use 3.9):

```bash
pip install -r requirements.txt
```

## Quickstart

The four relational datasets used in the paper are `financial` (PKDD'99 CTU chain)
and the V-shapes `imdb`, `instacart`, `movielens`. Every driver shares the same CLI:

```bash
# natural mode: Chain-NPM on the financial chain (district <- account <- trans)
python chain_npm_financial.py --eps 3.2 --seed 42 --mode natural --method chainnpm

# planted cross-hop audit: product planting at a given target strength
python chain_npm_imdb.py --eps 0.8 --seed 42 --mode plant_product --target 0.20

# the same driver produces every baseline comparison point
python chain_npm_movielens.py --eps 3.2 --seed 42 --mode natural --method pertable
```

| Option | Values |
|---|---|
| `--eps` | privacy budget ε (paper grid: 0.1, 0.2, 0.4, 0.8, 1.6, 3.2) |
| `--seed` | RNG seed (paper grid: 42–51) |
| `--mode` | `natural` \| `plant_product` \| `plant_xor` \| `plant_family:<rule>` |
| `--target` | target effect for planting (`Δ` for product, `φ` for xor); defaults reproduce the paper grid |
| `--method` | `chainnpm` \| `pertable` \| `denorm` \| `privmrf` \| `privbayes` \| `pbpgm` \| `lavaprop` |
| `--data_dir`, `--out` | override the input directory and the output JSON path |
| `--subset_frac`, `--subset_seed` | run on a random subset of the leaf table (smoke tests) |
| `--tau` | override the pre-registered truncation threshold |

Each run writes the synthetic tables and `meta.json` under `result/p4out/` and a
result JSON with the per-hop errors and the cross-hop effect.

## Input data layout

Inputs are discretized CSVs plus a domain JSON per table. By default they are read
from

```
$CHAINNPM_ROOT/tmp/data/<dataset>/processed/          # natural mode
$CHAINNPM_ROOT/tmp/data_planted/<dataset>/<mode>/     # planted modes
```

with one `<file>.csv` and one `<file>_domain.json` per table; the exact file names,
primary keys, attributes and FK columns are declared in each driver's `CFG`
(e.g. `chain_npm_financial.py` uses `district.csv`, `account.csv`, `trans.csv` with
`district_domain.json` and so on). `CHAINNPM_ROOT` defaults to the parent of the
checkout; `--data_dir` overrides it per call.

Table conventions:

* parent table: `[pk, attrs...]`
* child table: `[pk, attrs..., fk]`; V-shape child: `[pk, attrs..., fk1, fk2]`
* domain JSON: `{attr: {"size": d}}`

Result JSONs are written to `$CHAINNPM_ROOT/tmp/results/<dataset>/`.

`make_planted_csvs.py` materializes the planted variants (product / xor) that the
third-party baselines consume, using exactly the same planting protocol as
`p4_common`, so that all methods see identical planted inputs.

## Data availability

The discretized inputs that produced `results/by_dataset/` are **not
redistributed here**: the IMDb, MovieLens, Instacart and PKDD'99 terms do not
permit redistribution of raw or derived data (the per-source review, with
quotations and links, is in `preprocess/README.md`). Instead,

* `preprocess/` rebuilds every dataset's inputs from its public source with a
  scripted, verified pipeline, and
* `preprocess/MANIFEST.md` lists the reference build's file sizes, row counts and
  sha256 hashes, so a rebuild can be checked byte-for-byte against the inputs
  behind the archived numbers.

All other artifacts — the method code, the experiment drivers, the evaluators and
the per-cell result archive — are included and runnable as-is.

## Baselines

`baselines.py` adapts the comparison methods reported in the paper. The
implementations themselves are **not vendored here**; place them in one directory
and point `BASELINE_REPOS` at it:

```bash
export BASELINE_REPOS=/path/to/baseline-repos   # must contain unified_comparison.py,
                                               # PrivLava/, pbpgm/
```

Without that directory only `--method chainnpm` (and the evaluators) run; the
baseline methods record a degradation instead of failing the pipeline.

## Results

`results/by_dataset/<dataset>/<method>_<mode>_eps<ε>_seed<seed>.json` holds one
record per experiment cell (metadata, hyper-parameters, per-hop median relative
errors, cross-hop effect, size statistics, runtime). `results/tables/` aggregates
them into the summary tables of the paper (median with bootstrap CI, Wilcoxon/Holm
significance, plant-and-audit retention).

## License

Apache-2.0 (see `LICENSE`).