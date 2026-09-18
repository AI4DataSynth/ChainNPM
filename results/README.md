# Results archive

`by_dataset/<dataset>/<method>_<mode>_eps<ε>_seed<seed>.json` — one record per
experiment cell, exactly the cells that entered the paper:

| Field | Meaning |
|---|---|
| `dataset`, `shape`, `method`, `mode`, `eps`, `seed` | cell identity (`shape` is `chain` or `vshape`) |
| `runtime_sec`, `total_runtime_sec` | wall-clock cost of the cell |
| `hyper` | δ, σ, truncation thresholds (τ), and other cell-level hyper-parameters |
| `prereg` | pre-registered constants in force (`prereg.py`) |
| `counts` | row counts of the ground-truth and synthetic tables |
| `re` | relative-error statistics per hop class (median over all queries and over large-count queries) |
| `effect` | cross-hop effect measured on the real and synthetic data, and the `syn`/`real` pair used for retention |
| `size_tv`, `mean_size` | group-size distribution statistics |

`method` values: `chainnpm`, `pertable` (per-table PrivMRF + author-protocol FK
re-linking, the paper's PerTable), `privmrf` (single-table PrivMRF), `privpetal`,
`lavaprop` (PrivLava), `privbayes`, `pbpgm`. `denorm` records (an internal control)
are not shipped. `mode` values: `natural`, `plant_product`, `plant_xor`,
`plant_family:<rule>`.

Two notes on provenance:

* records for `privpetal` carry a `privpetal_meta` block copied from the baseline's
  own `meta.json`; the host field has been redacted to `cluster` here;
* a few cells have fewer than 10 seeds because a run was still in flight when the
  archive was cut; each record carries its own `seed`, so aggregations stay exact.

`tables/` holds the aggregation outputs used by the paper together with the
scripts that produced them: `p7_summary.tsv` (natural-mode per-hop RE, median
with bootstrap CI), `p7_effect.tsv` (planted-mode effect retention, six series),
`p7_stats.json` (Wilcoxon/Holm across margins) and `tab_re_appendix.tsv` (the
appendix RE table). `figures/` holds the two main figures (`fig_main_effect_plant.*`
for effect retention and `fig_main_crosshop_natural.*` for the natural-grid RE)
plus the scripts and per-cell coverage reports that generate them.

Two naming notes:

* the per-table baseline series appears as **`pertable`** in `by_dataset/`,
  `p7_effect.tsv` and `tab_re_appendix.tsv`, but as **`privmrf`** in
  `p7_summary.tsv`; both denote the manuscript's *PerTable* (per-table PrivMRF
  with author-protocol FK re-linking), and the alias is also recorded in the
  header of `tab_re_appendix.tsv`;
* the planted-mode grid is only shipped as the aggregated tables and figures —
  the per-cell planted records live in the live campaign tree, not in
  `by_dataset/`.