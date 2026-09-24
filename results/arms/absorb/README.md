# `results/arms/absorb/` — the reported Chain-NPM arm (row-level quota calibration)

Every Chain-NPM number in the paper is produced by the released code with one
optional post-processing step enabled:

```
--absorb 'pubn,rowleaf,szthresh,exactmarg,repair,growcap,szquota'
```

The step spends no budget and changes no released marginal or query: it reads
the per-parent-value row counts off the row-level adjacent marginals that the
mechanism already releases at full strength and recalibrates the pre-drawn
child group sizes against them (`szquota`; see `chain_npm_absorb.py` for the
token reference).  The default flags (`--absorb off`) leave every released
code path bit-identical — the archives under `results/by_dataset/` are that
default arm, so this directory and that one differ only by the extra flag.

All numbers use the **one-individual** adjacency (a single base-record
insertion or deletion); `rowleaf` is what makes the row-level objects exact
under it, and it is never included in `all`.

## Layout

```
natural/<dataset>/   60 cells each (6 budgets x seeds 42-51), 4 databases
                     chainnpm_natural_eps<e>_seed<s>_<tokens>.json
plant/<dataset>/    100 cells each (plant_product t in {0.1,0.2,0.316,0.4}
                     + plant_xor t=0.8, budgets 0.8/3.2, seeds 42-51),
                     3 V-shaped databases
```

The `budget`-arm cells (`*_budget_szquota.json`) are not part of the reported
arm and are not archived here.

## What consumes these files

* `results/tables/p7_summary_szquota.tsv` — natural-mode 2-hop medians
  (Table `tab:re-appendix` and Figures `fig:dataset_sweep` /
  `fig:query_class_sweep`).
* `results/tables/p7_effect.tsv` — planted-effect retention
  (Table `tab:four-ds` and Figure `fig:four-ds-effect`), with financial rows
  kept at the released arm.

## Reproducing a cell

```
python chain_npm_financial.py --eps 0.8 --seed 42 --mode natural \
    --method chainnpm \
    --absorb 'pubn,rowleaf,szthresh,exactmarg,repair,growcap,szquota'
```

(or `chain_npm_instacart.py` / `chain_npm_movielens.py` / `chain_npm_imdb.py`
for the V-shaped schemas; `--mode plant_product --target 0.2` for the planted
cells).  Verified byte-for-byte against this archive at merge time.