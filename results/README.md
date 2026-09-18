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

`tables/` holds the aggregation outputs used by the paper (`p7_summary.tsv`,
`p7_effect.tsv`, `p7_stats.json`) together with the scripts that produced them.