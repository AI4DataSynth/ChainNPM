# Processed-input manifest (reference build)

This file records the **reference inputs** that produced the results archived in
`results/by_dataset/`: the discretized CSVs and domain JSONs of the four
datasets used in the natural-mode grid (`financial`, `imdb`, `instacart`,
`movielens`), exactly as the drivers read them from
`$CHAINNPM_ROOT/tmp/data/<dataset>/processed/`.

The derived files themselves are **deliberately not redistributed**: the IMDb,
MovieLens, Instacart and PKDD'99 terms do not permit redistribution of raw or
derived data (see [README.md](README.md), section *Data licensing*). Rebuild
them from the public sources with the scripts in this directory and then verify
your build against the sha256 values below.

| Property | Value |
|---|---|
| datasets | `financial` (PKDD'99 CTU chain), `imdb`, `instacart`, `movielens` (V-shapes) |
| files | 31 (one `<table>.csv` plus one `<table>_domain.json` per table, plus per-dataset reports) |
| total CSV data rows | 5,741,978 |
| largest single file | `instacart/order_products.csv`, 35,287,048 B (33.7 MiB) |
| total size on disk | 114,638,297 B (109.3 MiB) |

## Layout produced by the prep scripts

```
<dataset>/            <- the archive is flat per dataset
  <table>.csv
  <table>_domain.json
  <dataset>_report.json
```

which is copied into the driver layout as
`$CHAINNPM_ROOT/tmp/data/<dataset>/processed/`.

## Rebuild

Every script takes `--dir` (default: the script's own directory), reads its raw
inputs from `<dir>/raw/` and writes `<dir>/processed/`:

```bash
python preprocess/financial/prep_financial.py --dir preprocess/financial   # + export_*.py for the trans dump
python preprocess/imdb/prep_imdb.py         --dir preprocess/imdb
python preprocess/instacart/prep_instacart.py --dir preprocess/instacart
python preprocess/movielens/prep_movielens.py --dir preprocess/movielens
python preprocess/acs/prep_acs.py           --dir preprocess/acs
```

`README.md` lists, per dataset, the raw files and URLs the script expects.
Four of the five pipelines were verified byte-identical on a re-run from the
same raw inputs; the IMDb pipeline differs only in its docstring (it re-reads a
6.6 GB stream).

## Verify

```bash
# for each file listed below
shasum -a 256 <dataset>/<file>
```

If you want to package your own build the way we did:

```bash
tar -czf chainnpm-processed-inputs.tar.gz -C <build-root> chainnpm-processed-inputs
# our reference bundle: 28,780,432 B,
# sha256 4e3bc6325c2e4c3113008c4b7c2280da5721ff167c5709afa125ff42a5da2343
```

---

## `financial` — PKDD'99 financial chain (`district <- account <- trans`)

| File | Rows | Bytes | sha256 |
|---|---|---|---|
| `financial/account.csv` | 4,500 | 51,910 | `e385125d425b72b5f3de66461ef7588258bee2eb570b886f8f5cdeba10f5ab1a` |
| `financial/account_domain.json` | — | 43 | `9fe92a15485304a1cf16e0e3666694dd161c29bff3a49fa980614dcdbb8a5003` |
| `financial/district.csv` | 77 | 710 | `c56c6036e7ea1761a3fe7c9e9ff97176d60c00024a2540edf9f03ab66cc32562` |
| `financial/district_domain.json` | — | 70 | `0351a579e587747b517c40be2b0fd8701c9825a25b08e20d061c6d2e67399d17` |
| `financial/financial_report.json` | — | 1,190 | `377e74eab9f73771d437c721f7ca7abfd78b3e113cba6c5dd6ceb0fa56e1d27b` |
| `financial/trans.csv` | 1,056,320 | 20,814,684 | `585921224ed1273b705ead417f49821804a1edd32f0083d566de3652a4c628dd` |
| `financial/trans_domain.json` | — | 91 | `ed4c2a9bb9cd945484a48e6a5f95ab75341fdd99b8bd414d4a4d8a8e204dae98` |

Subtotal: **20,868,698 bytes** (19.9 MiB), CSV rows **1,060,897**.

## `imdb` — IMDb V-shape (`names <- principals -> titles`)

| File | Rows | Bytes | sha256 |
|---|---|---|---|
| `imdb/imdb_report.json` | — | 464 | `837dacb28a8a7e624da1bc3bdcdfd5563702951c096e8b21af3554a2007c073f` |
| `imdb/names.csv` | 639,533 | 8,880,394 | `75b8f2d7cacd7a627778268cb5b5a18ddb4e62bc2090eeb47269a412d1f37b55` |
| `imdb/names_domain.json` | — | 71 | `8248245ed56a4fdaef125127be7b7d6f9a54f3759b0b8f796bf3367195378077` |
| `imdb/principals.csv` | 1,216,178 | 26,717,365 | `f76f9a97aa2a8744366877b2b9369814af9507fd3fedb8a03f7ffdf8517fb5c2` |
| `imdb/principals_domain.json` | — | 26 | `17930518f313e06a428476b2a67a1a865f6972af1671886fafe4d10f5b062ca7` |
| `imdb/titles.csv` | 110,468 | 1,786,769 | `1aae73501105ea3d14fe66926fd52c315185345386d265771609b4ecd2dc78e6` |
| `imdb/titles_domain.json` | — | 120 | `33ebb6ddb052ded2ceefa324d03437955f66b9446fed4d9a85d2a757612218d5` |
| `imdb/vshape_spec.json` | — | 1,143 | `a8ab94eb0104d5023e3e4a97ad3ec263956aa8a44f7f06d4dd129f4d7faaf43c` |

Subtotal: **37,386,352 bytes** (35.7 MiB), CSV rows **1,966,179**.

## `instacart` — Instacart V-shape (`products <- order_products -> orders`)

| File | Rows | Bytes | sha256 |
|---|---|---|---|
| `instacart/instacart_report.json` | — | 445 | `6d9aa0b222642f16e9f20d61d3392c4a028e32832f89deccbd80a04e2608c996` |
| `instacart/order_products.csv` | 1,516,279 | 35,287,048 | `68759384d6b8dbcbd950b2be15894611b428fb696104459ac912fc43f5f45b62` |
| `instacart/order_products_domain.json` | — | 49 | `2655ef8deb8c51f59c3b96dea2c18dbcd3db0d1594af915d906bd2bfa2d308cf` |
| `instacart/orders.csv` | 150,000 | 2,138,922 | `2c1a426f3518e75ffd9a3a1490a9f2d958ed8b1b3c1d7725125d7fcaa820d6ee` |
| `instacart/orders_domain.json` | — | 91 | `e5107c9edf354e2def9fc061d69e00dc9fe0382a32b2433b46dd1611b1e69de4` |
| `instacart/products.csv` | 38,491 | 412,722 | `5d634d9599258dd2fe138ca7dd88d7a57821bee9a91685e22e334171f7a987d8` |
| `instacart/products_domain.json` | — | 45 | `96f04a83a9dfc13731bfc026722565cc18b3b4d61aaaadb450dbd5d25c59b436` |
| `instacart/vshape_spec.json` | — | 1,188 | `8f2b3f253c3e1a3fb62be1eec43b27ebb289a1571e2cf10c81d1dc564dd43cb3` |

Subtotal: **37,840,510 bytes** (36.1 MiB), CSV rows **1,704,770**.

## `movielens` — MovieLens-1M V-shape (`users <- ratings -> movies`)

| File | Rows | Bytes | sha256 |
|---|---|---|---|
| `movielens/movielens_report.json` | — | 462 | `4691a94adda105c10ad012d705abc6edb9e75c4bee0902bb779dfcf79c586b22` |
| `movielens/movies.csv` | 3,883 | 34,083 | `d2bf0d3b2891adddde35d2b88b7b36b4d87295d754a182a2e87f3036371333da` |
| `movielens/movies_domain.json` | — | 45 | `106f9573aa75f5329283259b2dba0506fde655c419c53be1b0141126e8368fa9` |
| `movielens/ratings.csv` | 1,000,209 | 18,440,442 | `55ec9bab6675ca8d53e19bfeb311686995948a284c87b19633c95fb80f0527ef` |
| `movielens/ratings_domain.json` | — | 23 | `87000e3b4716eb977947ccd8d9b58abb750ace711b5ccf25b529ef04d34245e2` |
| `movielens/users.csv` | 6,040 | 66,732 | `7720655a556a3d0a2fe4e9e472f060ed55c23a7d17aff3742316d6fba017820b` |
| `movielens/users_domain.json` | — | 64 | `1f7871706ab9f51b5f6fe7ccffd874c372348c5333f2e02181771f01e47ec835` |
| `movielens/vshape_spec.json` | — | 886 | `546dae389524b60ea344e561a4494a2f75a574ca0bb300b4d01f61e16f9fa219` |

Subtotal: **18,542,737 bytes** (17.7 MiB), CSV rows **1,010,132**.

## All four datasets

| Dataset | Files | CSV rows | Bytes |
|---|---|---|---|
| `financial` | 7 | 1,060,897 | 20,868,698 |
| `imdb` | 8 | 1,966,179 | 37,386,352 |
| `instacart` | 8 | 1,704,770 | 37,840,510 |
| `movielens` | 8 | 1,010,132 | 18,542,737 |
| **total** | **31** | **5,741,978** | **114,638,297** |

Row counts are data rows (CSV lines minus the header). Each dataset also carries
its `<dataset>_report.json` (group-size and cross-hop statistics) and, for the
three V-shapes, a `vshape_spec.json` that declares the root/child roles, primary
keys, attributes, FK columns and the sampling parameters.

## What is *not* in this archive

* raw source data (`raw/`) for any dataset — downloaded from the providers in
  `preprocess/README.md` §1–§5;
* the planted variants (`plant_product` / `plant_xor` / `plant_family:*`), which
  `make_planted_csvs.py` regenerates deterministically from these inputs;
* the TPC-H variants (regenerable with `dbgen`, see `preprocess/README.md` §6);
* the ACS PUMS inputs (auxiliary; `preprocess/acs/prep_acs.py` rebuilds them);
* any synthetic output tables or result JSONs (those live under `results/` in
  the code repository).

## Provenance and reproducibility

The four per-dataset reports inside the archive were produced by the scripts in
`preprocess/<dataset>/`. During the preparation of this archive the scripts were
re-run against the same raw files and compared byte-wise with the archived
copies:

| Dataset | Re-run | Result |
|---|---|---|
| `financial` | yes (`prep_financial.py --dir …`) | all 3 CSVs and all 3 domain JSONs sha256-identical |
| `instacart` | yes (`prep_instacart.py --dir …`) | all 3 CSVs sha256-identical |
| `movielens` | yes (`prep_movielens.py --dir …`) | all 3 CSVs and all 3 domain JSONs sha256-identical |
| `imdb` | no | not re-run: it needs three streaming passes over 6.6 GB of TSV. The shipped script differs from the one that produced the archive **only in a docstring** (verified by `diff`), so the code path is unchanged. |

Two documentation-only divergences between the shipped scripts and the
originals that produced the archive: the licensing/docstring additions, and the
`source` string that `prep_instacart.py` writes into a regenerated
`vshape_spec.json` (the archived copy names the mirror the data was fetched
from). Neither affects any CSV or domain JSON.

Row counts, group-size statistics and cross-hop figures quoted in
`preprocess/README.md` were recomputed from the files in this archive.

## Licensing verdict — do not publish this archive as-is

**This archive should not be uploaded to Zenodo/Figshare (or any other public
host) in its current form.** It is a redistribution of *derived* versions of
four third-party datasets, and the providers' terms restrict that (verbatim
clauses are collected in `preprocess/README.md` §7):

| Dataset | Redistribution status |
|---|---|
| `imdb` | **Prohibited.** IMDb's non-commercial terms: the data "must not be altered/republished/resold/repurposed to create any kind of online/offline database of movie information". This covers derived files, so the IMDb V-shape is the clearest blocker. |
| `movielens` | **Not permitted without permission.** GroupLens usage license: "The user may not redistribute the data without separate permission." |
| `instacart` | **No permission granted.** The provider page states the dataset is "provided as-is for non-commercial use"; the mirror the paper used declares no license at all and the official host is currently unavailable. |
| `financial` | **Unclear.** Neither the CTU repository nor the archived PKDD'99 call for contributions states a license. |

Recommended paths, in order of preference:

1. **Publish code only** (`preprocess/` scripts + this manifest). Reviewers
   reconstruct the inputs from the providers; the sha256 and row-count tables
   above let them confirm their reconstruction matches the paper's inputs.
2. If a data artifact is wanted, restrict it to the **TPC-H** variant, which is
   regenerable from the public `dbgen` fixture and carries no third-party terms.
3. Publish the other four datasets only after obtaining written permission from
   GroupLens, Instacart and the CTU repository, and treat IMDb as excluded
   (its terms do not offer a permission route for republication).
4. A hybrid that is usually acceptable: publish the manifest plus a
   deterministic **derivation recipe** (bin edges, sampling seeds, filters —
   all already recorded in `preprocess/` and in the domain files) instead of the
   data bytes.