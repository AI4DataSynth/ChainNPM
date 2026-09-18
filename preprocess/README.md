# Preprocessing: raw data → Chain-NPM input files

This directory holds the scripts that turn the raw public data sources into the
discretized CSVs and domain JSONs that every Chain-NPM driver consumes.

Each driver reads its inputs from

```
$CHAINNPM_ROOT/tmp/data/<dataset>/processed/
```

one `<table>.csv` plus one `<table>_domain.json` per table (see the top-level
`README.md` and `docs/REPRODUCE.md` for the per-dataset `CFG` that names the
files, primary keys, attributes and FK columns).

Two conventions are shared by every script in this directory:

* `--dir <dataset dir>` selects the dataset directory. The default is the
  directory the script itself lives in, so
  `python3 preprocess/movielens/prep_movielens.py` reads
  `preprocess/movielens/raw/` and writes `preprocess/movielens/processed/`.
  Passing `--dir $CHAINNPM_ROOT/tmp/data/<dataset>` writes the files directly
  into the tree the drivers read, which is what the commands below do.
* Domain JSON format: `{"<ATTR>": {"size": <domain size>}}`, where every
  attribute value is a non-negative integer in `[0, size)`.
* Table layout: parent `[pk, attrs...]`, child `[pk, attrs..., fk]`, and for the
  V-shape datasets child `[pk, attrs..., fk1, fk2]` — the FK columns are always
  the **last** columns.

All scripts are deterministic (seed 42) and CPU-only; they depend on
`numpy`, `pandas` and (IMDb only) `pyarrow`, plus `pymysql` for the PKDD'99
export.

## Summary

| Dataset | Shape | Tables | Processed rows | Raw source |
|---|---|---|---|---|
| `financial` | chain `district <- account <- trans` | 3 | 77 + 4,500 + 1,056,320 = **1,060,897** | PKDD'99 / Berka 2000, CTU Relational Repository |
| `imdb` | V-shape `names <- principals -> titles` | 3 | 110,468 + 639,533 + 1,216,178 = **1,966,179** | IMDb non-commercial snapshot 2026-08-19 |
| `instacart` | V-shape `products <- order_products -> orders` | 3 | 38,491 + 150,000 + 1,516,279 = **1,704,770** | Instacart Market Basket Analysis (2017), `prior` split |
| `movielens` | V-shape `users <- ratings -> movies` | 3 | 6,040 + 3,883 + 1,000,209 = **1,010,132** | MovieLens-1M (GroupLens) |
| `acs` (auxiliary) | chain `household <- individual` | 2 | 151,045 + 382,957 = **534,002** | ACS PUMS 2022 1-Year, **California** |
| TPC-H (auxiliary) | chain `customer <- orders <- lineitem` | 3 | see below | TPC-H `dbgen` SF=1 |

The row counts above were re-measured from the delivered `processed/*.csv`
files (`wc -l` minus the header) and agree exactly with the per-dataset schema
reports recorded when the paper's inputs were produced.

---

## 1. `financial` — PKDD'99 (Berka) bank chain

### 1.1 Source and download

* Repository: <https://relational.fel.cvut.cz/dataset/Financial>
  ("PKDD'99 Financial dataset", alternative name "loan application").
* Original data: the PKDD'99 Discovery Challenge bank data set
  (Berka 2000; 8 tables describing clients, accounts, transactions,
  permanent orders, loans and credit cards).
* Access used by the paper: the repository's **public MariaDB**, queried
  directly — host `relational.fel.cvut.cz`, port `3306`, user `guest`,
  password `ctu-relational`, database `financial`. No file download is
  involved.

### 1.2 Raw files

`raw/` holds three CSVs exported from that database (dumped as
`SELECT *`, empty string for SQL NULL):

| File | Rows | Bytes |
|---|---|---|
| `raw/district.csv` | 77 | 6,197 |
| `raw/account.csv` | 4,500 | 164,348 |
| `raw/trans.csv` | 1,056,320 | 58,718,776 |

Only these three tables are part of the chain. The database also contains
`card`, `client`, `disp`, `loan` and `order`, which were not exported.

`trans` needs care: a full-table `SELECT *` exceeds the server-side 300 s
statement limit, so it is fetched with `export_trans_resume.py` in `trans_id`
range chunks (200,000 ids per query, 2 parallel connections — 4 connections
were dropped by the server). Every chunk is written to its own part file, so
the export is resumable, and the parts are concatenated into `raw/trans.csv` at
the end. The `trans_id` values are sparse in `[1, 3,682,987]`, which is why the
19 chunks yield 1,056,320 rows; that number matches `COUNT(*)` on the table
exactly.

### 1.3 Preprocessing steps (`prep_financial.py`)

Column-numbering note: in this schema `A2` is the district *name* and `A3` the
region; `A11` is the average salary and `A12`/`A13` the 1995/1996 unemployment
rates. The paper discretizes `A13`, `A11` and `A3`:

* **district** (root, 77 rows): sorted by `district_id`, dense PK `DKEY` =
  row index.
  * `REGION` = `A3`, 8 region names sorted lexicographically → `0..7`.
  * `UNEMP96` = `A13 > median(A13)` with median **3.6** → binary.
  * `SALARY` = `A11` in tertiles, edges **8572** and **9055** → `0,1,2`.
* **account** (middle, 4,500 rows): sorted by `account_id`, dense PK `AKEY`.
  * `FREQ` = the 3 original frequency values (monthly / weekly / per
    transaction), sorted → `0..2`.
  * `AYEAR` = `year(date) - min(year)` → 5 values (accounts opened 1993–1997).
  * `DKEY` = foreign key into `district` (last column).
* **trans** (leaf, 1,056,320 rows): sorted by `trans_id`, dense PK `TKEY`.
  * `TYEAR` = `year(date) - 1993` → `0..5` (transactions 1993–1998).
  * `TTYPE` = the 3 original type values (`PRIJEM`/`VYBER`/`VYDAJ`), sorted.
  * `TOPER` = the 5 non-empty operation codes sorted, plus a **separate 6th
    bin** for the 183,114 rows with no operation code → size 6.
  * `TAMOUNT` = deciles of `amount`, edges
    15 / 92 / 201 / 870 / 2100 / 3400 / 5439 / 9120 / 18400 → `0..9`.
  * `AKEY` = foreign key into `account` (last column).

No subsetting: 1.06 M rows is already comparable to the other datasets.
`balance`, `k_symbol`, `bank` and the counterparty account were dropped as
high-cardinality or redundant.

### 1.4 Outputs (measured)

| File | Rows | Bytes |
|---|---|---|
| `processed/district.csv` | 77 | 710 |
| `processed/account.csv` | 4,500 | 51,910 |
| `processed/trans.csv` | 1,056,320 | 20,814,684 |
| `processed/district_domain.json` | — | 70 |
| `processed/account_domain.json` | — | 43 |
| `processed/trans_domain.json` | — | 91 |
| `processed/financial_report.json` | — | 1,190 |

Domain sizes: `REGION` 8, `UNEMP96` 2, `SALARY` 3; `FREQ` 3, `AYEAR` 5;
`TYEAR` 6, `TTYPE` 3, `TOPER` 6, `TAMOUNT` 10.

FK integrity: `account.DKEY → district` 4,500/4,500 and
`trans.AKEY → account` 1,056,320/1,056,320 — **zero orphans** on both hops.

Group sizes: `trans` per `account` median **208**, max **675**, Gini 0.305;
`account` per `district` median **48**, max **554**, Gini 0.243.

Cross-hop signal (recomputed from the delivered `processed/*.csv`): the two-hop
district ↔ transaction coupling is weak — the share of top-decile transactions
(amount above 18400) is 0.1032 for accounts in the lowest salary tertile and
0.0999 for accounts in the highest, and the credit (`PRIJEM`) share is
0.3831 / 0.3794 / 0.386 across the three salary tertiles (≈0.38 overall).
No income-related structure survives at this hop. This dataset therefore serves
as the **weak-cross-hop** control; the strong signal of the Berka data sits in
the `account`/`loan` layer, which is not part of this chain.

### 1.5 Commands

```bash
export CHAINNPM_ROOT=/path/to/work
export DATA_ROOT=$CHAINNPM_ROOT/tmp/data

mkdir -p $DATA_ROOT/financial/raw
# small tables: a plain full-table SELECT is enough
python3 preprocess/financial/export_financial.py --dir $DATA_ROOT/financial district account
# trans: resumable trans_id-range export (200k ids/query, 2 connections)
python3 preprocess/financial/export_trans_resume.py --dir $DATA_ROOT/financial

python3 preprocess/financial/prep_financial.py --dir $DATA_ROOT/financial
```

`export_financial.py` also accepts `trans` in its table list, but a full-table
`SELECT` on `trans` is cut off by the 300 s server limit — use
`export_trans_resume.py` for it. `export_trans_resume.py` additionally accepts
`--lo/--hi/--chunk/--workers`.

---

## 2. `imdb` — IMDb V-shape

### 2.1 Source and download

* Snapshot used: **2026-08-19**, from the official non-commercial dataset host
  <https://datasets.imdbws.com/>:
  * <https://datasets.imdbws.com/title.basics.tsv.gz>
  * <https://datasets.imdbws.com/title.principals.tsv.gz>
  * <https://datasets.imdbws.com/name.basics.tsv.gz>
* The gzip files were checked against the server `Content-Length` and with
  `gzip -t`. The 2026-08-19 snapshot is a *dated copy of a daily-refreshed
  source*: the counts below are the ones the paper used, and a fresh download
  will differ slightly.

### 2.2 Raw files

| File | Rows | Bytes |
|---|---|---|
| `raw/title.basics.tsv.gz` / `.tsv` | 12,727,068 | 225,672,858 / 1,104,507,522 |
| `raw/title.principals.tsv.gz` / `.tsv` | 101,266,607 | 779,098,548 / 4,525,898,670 |
| `raw/name.basics.tsv.gz` / `.tsv` | 15,585,043 | 308,374,937 / 963,085,635 |

That is ≈1.31 GB gzipped, ≈6.6 GB uncompressed. `prep_imdb.py` reads the
**uncompressed** `.tsv` files and streams them with `pyarrow.csv`
(`title.principals` alone is 101 M rows and does not fit in memory as a pandas
frame), so both the `.tsv.gz` and the `.tsv` must be present.

### 2.3 Subset and preprocessing steps (`prep_imdb.py`)

Structure is the V-shape `names <- principals -> titles` (`principals` is the
child carrying both FKs). The official snapshot has **no gender column** (that
column exists only in the CTU mirror of IMDb), so `ALIVE` + `PROF1` stand in
for the demographic attributes without reducing the categorical content.

Deterministic subset (seed 42), built to stay under a 2 M total-row cap:

1. Candidate titles = `titleType ∈ {movie, tvSeries}` **and**
   `startYear ∈ [2000, 2019]` → **400,465** candidates.
2. Uniform sample of candidates; the first pass with `T_INITIAL = 140,000`
   exceeded the 2 M row cap, so the target shrank to **117,059** titles and one
   retry produced the delivered set.
3. Cascade `title.principals` for those titles and `name.basics` for the
   referenced `nconst`s; FK closure prunes the frontier.
4. Drop snapshot drift: **33** `nconst` present in `principals` but absent from
   `name.basics`, and the **4** titles that became orphaned as a result →
   **110,468** final titles.
5. Encode attributes:
   * `names`: `ALIVE` = `deathYear` missing (2); `BYEARBIN` = `birthYear`
     floored to decades 1870s–2010s, clipped, with a separate NA bin (14);
     `PROF1` = first entry of `primaryProfession`, 16 values (rarest merged to
     `OTHER`).
   * `titles`: `TITLETYPE` = movie / tvSeries (2); `GENRE1` = first listed
     genre, 16 values (3 rarest of the 18 merged to `OTHER`); `YEARBIN` =
     5-year bins over 2000–2019 (4); `ISADULT` (2); `RTBIN` =
     `runtimeMinutes` into NA / <60 / 60–89 / 90–119 / ≥120 (5).
   * `principals`: `CATEGORY` = original `category` (13 values, no merging).

`primaryTitle`, `job` and `characters` (free text) are not used.

### 2.4 Outputs (measured)

| File | Rows | Bytes |
|---|---|---|
| `processed/names.csv` | 639,533 | 8,880,394 |
| `processed/titles.csv` | 110,468 | 1,786,769 |
| `processed/principals.csv` | 1,216,178 | 26,717,365 |
| `processed/names_domain.json` | — | 71 |
| `processed/titles_domain.json` | — | 120 |
| `processed/principals_domain.json` | — | 26 |
| `processed/vshape_spec.json` | — | 1,143 |
| `processed/imdb_report.json` | — | 464 |

Total **1,966,179** rows ≤ 2 M. Domain sizes: `ALIVE` 2, `BYEARBIN` 14,
`PROF1` 16; `TITLETYPE` 2, `GENRE1` 16, `YEARBIN` 4, `ISADULT` 2, `RTBIN` 5;
`CATEGORY` 13.

Column layout: `names.csv [NKEY, ALIVE, BYEARBIN, PROF1]`,
`titles.csv [TKEY, TITLETYPE, GENRE1, YEARBIN, ISADULT, RTBIN]`,
`principals.csv [PKEY, CATEGORY, NKEY, TKEY]`.

FK integrity: after the drift pruning, **zero** orphans on both FKs by
construction (`PKEY`/`NKEY`/`TKEY` are dense row indices).

Group sizes: `principals` per title median **10**, max **56**, Gini 0.342;
`principals` per name median **1**, max **480**, Gini 0.389.

Cross-hop signal (recomputed from the delivered `processed/*.csv`): titles
reached through persons born in the 1890s–1920s are 15.98 % tvSeries, versus
24.51 % for persons born from the 1970s on — a 1.5× two-hop contrast that
per-FK marginalization cannot keep.

### 2.5 Commands

```bash
export DATA_ROOT=$CHAINNPM_ROOT/tmp/data
mkdir -p $DATA_ROOT/imdb/raw && cd $DATA_ROOT/imdb/raw

for f in title.basics.tsv.gz title.principals.tsv.gz name.basics.tsv.gz; do
  curl -sL -C - --retry 3 -o "$f" "https://datasets.imdbws.com/$f"
  gzip -t "$f" && gzip -dk "$f"        # prep_imdb.py reads the .tsv files
done

python3 /path/to/ChainNPM/preprocess/imdb/prep_imdb.py --dir $DATA_ROOT/imdb
```

Preprocessing this dataset is the expensive step of the four (three streaming
passes over 6.6 GB of TSV); the `title.principals` pass is the dominant cost.

---

## 3. `instacart` — Instacart V-shape

### 3.1 Source and download

* Dataset: "Instacart Market Basket Analysis" (Bo Yang, Instacart, 2017), the
  data accompanying the Kaggle competition of the same name.
* Provider page: <https://www.instacart.com/datasets/grocery-shopping-2017>
  (that page states the dataset is *"provided as-is for non-commercial use"*;
  at the time of writing it also says "Dataset is temporarily unavailable").
* Because the provider page was unavailable, the paper's copy came from a
  **public Hugging Face mirror** (`attik/Instacart-Market-Basket-Analysis`,
  reached through the `hf-mirror.com` mirror). All six files were byte-compared
  against the source listing. That mirror declares **no license** — see §7.
* Alternatively, with a Kaggle account:
  `kaggle competitions download -c instacart-market-basket-analysis`.

### 3.2 Raw files

| File | Rows | Bytes | Used |
|---|---|---|---|
| `raw/orders.csv` | 3,421,083 | 108,968,645 | yes (`eval_set == prior`) |
| `raw/order_products__prior.csv` | 32,434,489 | 577,550,706 | yes |
| `raw/products.csv` | 49,688 | 2,166,953 | yes |
| `raw/order_products__train.csv` | 1,384,617 | 24,680,147 | no |
| `raw/aisles.csv` | 134 | 2,603 | no (aisle ids only) |
| `raw/departments.csv` | 21 | 270 | no (department ids only) |

The `prior` split has 3,214,874 orders (of 3,421,083 total; the remainder are
`train` and `test`). The `train` and `test` splits are deliberately unused,
matching the PrivPetal setup the paper compares against.

### 3.3 Structure note: chain notation vs. FK-faithful V-shape

`order_products` references **both** `products` and `orders`, so the three
tables cannot be written as a directed three-table chain. They are delivered in
the pipeline's V-shape convention — identical in form to
`users <- ratings -> movies`:

```
products <- order_products -> orders
```

### 3.4 Subset and preprocessing steps (`prep_instacart.py`)

Deterministic subset (seed 42), under a 2 M total-row cap:

1. Keep `orders` with `eval_set == prior`.
2. Uniform sample of `S_INITIAL = 150,000` prior orders (the row cap was met at
   the first attempt, so no shrink-retry was needed).
3. Stream `order_products__prior.csv` in 1 M-row chunks and keep the rows whose
   `order_id` is in the sample.
4. Keep only products that were actually purchased (`products` is then FK-closed
   by construction: 38,491 of 49,688 products).
5. Encode attributes:
   * `products`: `DEPT` = `department_id`, 21 departments → 16 (smallest 6
     merged into `OTHER`); `AISLE` = `aisle_id`, 134 aisles → 16 (top 15 +
     `OTHER`).
   * `orders`: `DOW` = `order_dow` (7); `HOURBIN` =
     `order_hour_of_day // 4` (6 four-hour blocks); `DPOBIN` =
     `days_since_prior_order` into 0 / 1–3 / 4–7 / 8–14 / 15–21 / 22–30 / NA
     (7); `ONUMBIN` = `order_number` clipped to 1..6 plus a "7+" bin (7).
   * `order_products`: `REORDERED` = reorder flag (2); `ATCBIN` =
     `add_to_cart_order` clipped to 1..6 plus a "7+" bin (7); plus `PKEY` and
     `OKEY` FKs. Rows are sorted by `(OKEY, PKEY)`.

`product_name` (free text) is not used.

### 3.5 Outputs (measured)

| File | Rows | Bytes |
|---|---|---|
| `processed/products.csv` | 38,491 | 412,722 |
| `processed/orders.csv` | 150,000 | 2,138,922 |
| `processed/order_products.csv` | 1,516,279 | 35,287,048 |
| `processed/products_domain.json` | — | 45 |
| `processed/orders_domain.json` | — | 91 |
| `processed/order_products_domain.json` | — | 49 |
| `processed/vshape_spec.json` | — | 1,188 |
| `processed/instacart_report.json` | — | 445 |

Total **1,704,770** rows ≤ 2 M. Domain sizes: `DEPT` 16, `AISLE` 16; `DOW` 7,
`HOURBIN` 6, `DPOBIN` 7, `ONUMBIN` 7; `REORDERED` 2, `ATCBIN` 7.

Column layout: `products.csv [PKEY, DEPT, AISLE]`,
`orders.csv [OKEY, DOW, HOURBIN, DPOBIN, ONUMBIN]`,
`order_products.csv [OPKEY, REORDERED, ATCBIN, PKEY, OKEY]`.

FK integrity: both FKs are complete by construction — `PKEY` is a dense index
over purchased products and `OKEY` a dense index over sampled orders.

Group sizes: items per order median **8**, max **104**, Gini 0.392; purchases
per product median **5**, max **22,085**, Gini 0.842 (head items are extremely
concentrated).

Cross-hop signal (recomputed from the delivered `processed/*.csv`): weak. The
reorder rate is 0.5925 overall, 0.5654 for the 00:00–04:00 block and 0.5862 for
the 12:00–16:00 block. As with `financial`, this dataset is used as a
**weak-cross-hop** control; its strong signal is child-internal
(product popularity, add-to-cart position).

### 3.6 Commands

```bash
export DATA_ROOT=$CHAINNPM_ROOT/tmp/data
mkdir -p $DATA_ROOT/instacart/raw     # put orders.csv, products.csv,
                                      # order_products__prior.csv there
python3 /path/to/ChainNPM/preprocess/instacart/prep_instacart.py --dir $DATA_ROOT/instacart
```

`prep_instacart.py` reads exactly three raw files (`orders.csv`,
`products.csv`, `order_products__prior.csv`); `aisles.csv`,
`departments.csv` and `order_products__train.csv` are not read. The
`S_INITIAL`/`ROW_CAP` constants at the top of the script control the subset
size.

---

## 4. `movielens` — MovieLens-1M V-shape

### 4.1 Source and download

* Source: <https://grouplens.org/datasets/movielens/1m/>
  (GroupLens Research, University of Minnesota).
* File: <https://files.grouplens.org/datasets/movielens/ml-1m.zip>,
  **5,917,549 bytes**, matching the official size. When the paper's copy was
  taken the site's HTTPS certificate had expired, so the download was made with
  certificate verification disabled; that is a property of the mirror at that
  moment, not a recommendation.

### 4.2 Raw files

`raw/ml-1m/` (24594131 B ratings.dat, 134368 B users.dat, 171308 B movies.dat,
README):

| File | Rows |
|---|---|
| `raw/ml-1m/ratings.dat` | 1,000,209 |
| `raw/ml-1m/users.dat` | 6,040 |
| `raw/ml-1m/movies.dat` | 3,883 |

These are the original `::`-separated, no-header files (read with
`sep='::'`, `engine='python'`, `encoding='latin-1'`).

### 4.3 Preprocessing steps (`prep_movielens.py`)

Full data, no subsetting — 1 M ratings is within the 2 M row budget. Structure
is the V-shape `users <- ratings -> movies`.

* **users** (6,040): sorted by `UserID`, dense PK `UKEY`.
  * `GENDER`: M → 1, F → 0 (2).
  * `AGE`: the 7 official brackets (1/18/25/35/45/50/56) sorted → `0..6`.
  * `OCC`: `Occupation`, 21 values → 16 (6 rarest merged into `OTHER`).
* **movies** (3,883): sorted by `MovieID`, dense PK `MKEY`.
  * `GENRE1`: primary genre = first entry of the pipe-separated `Genres`
    list, 18 values → 16 (3 rarest merged into `OTHER`).
  * `NGEN`: number of genres, clipped to 1..4 and shifted to `0..3`, so the
    multi-label information survives as a separate attribute.
* **ratings** (1,000,209): dense PK `RKEY` = row index after sorting by
  `(UKEY, MKEY)`; `RATING` = 1–5 stars shifted to `0..4`; the last two columns
  are the FKs `UKEY` (→ users) and `MKEY` (→ movies).

`Zip` is excluded (high cardinality).

### 4.4 Outputs (measured)

| File | Rows | Bytes |
|---|---|---|
| `processed/users.csv` | 6,040 | 66,732 |
| `processed/movies.csv` | 3,883 | 34,083 |
| `processed/ratings.csv` | 1,000,209 | 18,440,442 |
| `processed/users_domain.json` | — | 64 |
| `processed/movies_domain.json` | — | 45 |
| `processed/ratings_domain.json` | — | 23 |
| `processed/vshape_spec.json` | — | 886 |
| `processed/movielens_report.json` | — | 462 |

Total **1,010,132** rows. Domain sizes: `GENDER` 2, `AGE` 7, `OCC` 16;
`GENRE1` 16, `NGEN` 4; `RATING` 5.

Column layout: `users.csv [UKEY, GENDER, AGE, OCC]`,
`movies.csv [MKEY, GENRE1, NGEN]`,
`ratings.csv [RKEY, RATING, UKEY, MKEY]`.

FK integrity: the original data has no missing keys, so both FKs are complete —
`ratings.UKEY → users` 6,040/6,040 and `ratings.MKEY → movies` 3,883/3,883 hit,
**zero orphans**.

Group sizes: ratings per user median **96**, max **2,314**, Gini 0.529; ratings
per movie median **123.5**, max **3,428**, Gini 0.634.

Cross-hop signal (recomputed from the delivered `processed/*.csv`):
Animation/Children's titles are 12.98 % of the ratings of users under 18 versus
3.86 % for users aged 56+ — a 3.4× two-hop contrast that per-FK
marginalization flattens.

### 4.5 Commands

```bash
export DATA_ROOT=$CHAINNPM_ROOT/tmp/data
mkdir -p $DATA_ROOT/movielens/raw && cd $DATA_ROOT/movielens/raw
curl -L -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -q ml-1m.zip            # produces raw/ml-1m/{ratings,users,movies}.dat

python3 /path/to/ChainNPM/preprocess/movielens/prep_movielens.py --dir $DATA_ROOT/movielens
```

---

## 5. `acs` — ACS PUMS 2022 California (auxiliary real-schema validation)

This dataset is **not** part of the four-dataset natural grid; it backs the
paper's real-schema validation. It is included here because it is produced by
the same preprocessing convention.

### 5.1 Source and download

* Source: US Census Bureau, ACS PUMS **2022 1-Year**,
  <https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/>
* Files: `csv_hca.zip` (25,702,107 B) and `csv_pca.zip` (69,416,422 B), which
  unzip to `psam_h06.csv` (100,116,798 B) and `psam_p06.csv` (276,125,309 B).
  The `06` suffix is the **state FIPS code for California**, and the files
  carry `ST = 6` on every row. The official `ACS2022_PUMS_README.pdf` documents
  the columns.

> **State discrepancy, resolved.** An earlier acquisition pass had downloaded
> the **Wyoming** 2022 1-Year files (`csv_hwy.zip`/`csv_pwy.zip` →
> `psam_h56.csv` / `psam_p56.csv`, state FIPS 56; 2,965 households / 5,962
> persons, person-per-household mean 2.23, max 11, `SERIALNO` ⊆ household
> 100 %). Those files were **never used by the paper** and are not processed
> here. The paper, the driver `chain_npm_acs.py`, and the script in this
> directory all use the **California 2022** files (`psam_h06` / `psam_p06`).
> Same URL pattern, same schema; only the state code differs.

### 5.2 Preprocessing steps (`prep_acs.py`)

1. Drop group quarters (`TYPEHUGQ == 2`) and vacant units (`NP == 0`, no
   person-level information, and they break the group-size model).
2. Re-index the surviving households densely (`HHKEY`) and keep only the person
   rows whose `SERIALNO` is in that set → **151,045** households and
   **382,957** persons (persons per household mean 2.535, max 20).
3. Encode attributes (NA always gets its own bin):
   * household: `ACR` 4, `BLD` 11, `HHT` 9, `TEN` 4, `VEH` 8 (0..7, 8+ and NA
     folded into 7), `NPbin` = `NP` clipped to 1..10 → 10 (9 = "10+"),
     `HINCPbin` = deciles of `HINCP` plus an NA bin → 11.
   * individual: `AGEbin` = `AGEP // 10` clipped to 10 (0..9); `SEX` 2;
     `SCHLbin` = education recoded into <high-school / HS / some college /
     associate / bachelor / master / professional / doctorate / NA → 9;
     `ESR` 7 (6 = NA); `MAR` 5 (4 = NA); `RELPbin` = `RELSHIPP` collapsed into
     8 relationship classes; `WKHPbin` = weekly hours into
     `[0, 1-19, 20-34, 35-39, 40-49, 50+]` → 6.
4. Write `household.csv [HHKEY, attrs...]`,
   `individual.csv [PKEY, attrs..., HHKEY]` (FK last, persons sorted by the
   synthetic household key), the two domain JSONs and `acs_report.json`.

### 5.3 Outputs

| File | Rows | Bytes |
|---|---|---|
| `processed/household.csv` | 151,045 | 3,086,422 |
| `processed/individual.csv` | 382,957 | 10,354,455 |
| `processed/household_domain.json` | — | 150 |
| `processed/individual_domain.json` | — | 156 |
| `processed/acs_report.json` | — | 711 |

FK integrity: `individual.HHKEY → household` is complete by construction.
Group size: persons per household mean 2.535, max 20.

(These five figures come from the delivered `processed/` files and
`acs_report.json`; unlike the four main datasets there is no separate
`schema_report.md` for ACS in the archive.)

### 5.4 Commands

```bash
export DATA_ROOT=$CHAINNPM_ROOT/tmp/data
mkdir -p $DATA_ROOT/acs && cd $DATA_ROOT/acs
curl -LO https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/csv_hca.zip
curl -LO https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/csv_pca.zip
unzip -o csv_hca.zip && unzip -o csv_pca.zip     # psam_h06.csv, psam_p06.csv

python3 /path/to/ChainNPM/preprocess/acs/prep_acs.py --dir $DATA_ROOT/acs
```

---

## 6. TPC-H (auxiliary; not in this directory)

TPC-H raw data is not distributed with this repository and is not one of the
four datasets preprocessed here — it can be regenerated from the official
generator, which is why it is treated separately.

### 6.1 Obtaining the raw data

Build the TPC-H tools and generate scale factor 1 with `dbgen`:

```bash
# TPC-H tools (tpc.org), e.g.:
make -f makefile.suite CC=gcc DATABASE=ORACLE MACHINE=LINUX WORKLOAD=TPCH
./dbgen -s 1 -f
```

This produces `customer.tbl`, `orders.tbl`, `lineitem.tbl`, `part.tbl`,
`partsupp.tbl`, `supplier.tbl`, `nation.tbl`, `region.tbl` — pipe-separated
with a trailing `|` and no header, which is exactly the format
`chain_npm_tpch.py` reads from `<DATA_ROOT>/input_data/*.tbl`. `dbgen`
output is a public benchmark fixture and is freely regenerable, so no part of
TPC-H raises the licensing questions discussed in §7.

### 6.2 Discretized / planted TPC-H tables

The paper's TPC-H cells use a discretized, planted variant
(`orders.csv`, `lineitem.csv`, `part.csv`, `partsupp.csv`, `supplier.csv` plus
domain JSONs) under `<DATA_ROOT>/TPC-H/`. Those files were prepared by the
PrivLava/PrivPetal preprocessing of the same SF=1 tables — they are **not**
shipped in this repository and the script that produced them is not part of it.
The planted structure they carry is documented in `make_deep_gt.py`: the
`nation ↔ order year` association is planted in `orders.csv` via
`CUSTKEY` reassignment, and the `order year ↔ order size` association, with
quantity scaled by the part brand/type ratio, is planted in `lineitem.csv`.

### 6.3 `make_deep_gt.py`

`make_deep_gt.py` builds the *deep-chain* variant, in which `customer` is
private as well, giving the three-private-table chain
`customer -> orders -> lineitem` across which planted correlations compose:
`C_MKTSEGMENT ↔ C_NATIONKEY` (planted here) ∘ `C_NATIONKEY ↔ O_YEAR` ∘
`O_YEAR ↔ order size`. It

1. reads `<DATA_ROOT>/input_data/customer.tbl`;
2. bins `C_ACCTBAL` into 22 bins (`C_ACCTBALBIN`) and overrides
   `C_MKTSEGMENT` with `C_NATIONKEY % 5` on a random `--plant_prob` fraction of
   rows (default 0.6, `--seed` default 42);
3. writes `customer.csv` + `customer_domain.json` to `--out_dir` and copies the
   already-planted `orders/lineitem/part/partsupp/supplier` CSVs and domain
   JSONs from `<DATA_ROOT>/TPC-H/`;
4. writes `deep_gt_report.json` with the planted associations (total-variation
   distances for segment↔nation, nation↔year, segment↔year) and the group-size
   statistics, so the ground truth of the deep chain is auditable.

```bash
python make_deep_gt.py --data_root <DATA_ROOT> --out_dir <DATA_ROOT>/TPC-H-deep
python chain_npm_tpch_deep.py --data_dir <DATA_ROOT>/TPC-H-deep \
    --epsilon 3.2 --out_prefix <OUT_DIR>/chainnpm3_eps3.2
```

For the standard two-private-table TPC-H setup (private = `orders` + `lineitem`,
everything else public) the driver reads the raw `.tbl` files directly:

```bash
python chain_npm_tpch.py --gt_dir <DATA_ROOT>/input_data --epsilon 3.2 \
    --out_prefix <OUT_DIR>/chainnpm_tpch_eps3.2
```

---

## 7. Data licensing / redistribution

Chain-NPM's inputs are **derived from third-party data**, and several of the
sources are published under research/non-commercial terms that restrict
re-publication. Summary of what we checked (quoted text is verbatim from the
sources; see the URLs):

| Source | What the provider says | Re-distributable? |
|---|---|---|
| **PKDD'99 / Berka financial** (CTU Relational Repository) | Neither the repository pages (<https://relational.fel.cvut.cz/dataset/Financial>, `/about`) nor the archived PKDD'99 call for contributions state any license. The archived call only says the data sets are available to prospective conference participants. | **No explicit license — unclear; treat as not permitted.** |
| **IMDb datasets** | IMDb non-commercial licensing: the data "can only be used for **personal and non-commercial** use and must not be **altered/republished/resold/repurposed to create any kind of online/offline database of movie information** (except for individual personal use)". | **No — expressly prohibited** (raw *and* altered/derived). |
| **Instacart Market Basket Analysis** | Provider page: the dataset "is provided as-is for **non-commercial use** and is subject to our Terms of Service". The provider page currently reports the dataset as temporarily unavailable, and the public mirror the paper used declares no license. | **No explicit permission; non-commercial only.** |
| **MovieLens-1M** (GroupLens) | Usage license in `ml-1m-README.txt`: "The user may not **redistribute the data** without separate permission." | **No — requires separate permission.** |
| **ACS PUMS 2022** (US Census Bureau) | US federal government statistical data; PUMS microdata are published for public use. We found **no explicit redistribution clause** on the Census pages consulted: the Census Bureau Data API terms of service (which we could retrieve) govern *API* use, require the notice "This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau", and say nothing about redistributing data files. | Widely treated as US government works not subject to copyright, but **confirm with the Census Bureau** before publishing. |
| **TPC-H** | Regenerable public benchmark fixture via `dbgen`. | Yes (regenerate with `dbgen`; do not redistribute our planted variant without review). |

Sources consulted (checked when this archive was prepared):

* IMDb — "Can I use IMDb data in my software?" (the non-commercial licensing
  article linked from <https://data.imdb.com/non-commercial-datasets/>):
  <https://help.imdb.com/article/imdb/general-information/non-commercial-licensing/G5JTRESSHJBBHTGX>.
* MovieLens — usage license in `ml-1m-README.txt`:
  <https://files.grouplens.org/datasets/movielens/ml-1m-README.txt>.
* Instacart — provider page <https://www.instacart.com/datasets/grocery-shopping-2017>
  (archived copy: <http://web.archive.org/web/20230319033637/https://www.instacart.com/datasets/grocery-shopping-2017>)
  and the platform terms <https://www.instacart.com/terms>. The Kaggle
  competition page and its rules page are no longer reachable (HTTP 404), so no
  competition-specific data clause could be consulted.
* PKDD'99 / CTU — <https://relational.fel.cvut.cz/dataset/Financial>,
  <https://relational.fel.cvut.cz/about>, and the archived PKDD'99 call for
  contributions
  <http://web.archive.org/web/20180506061559/http://lisp.vse.cz/pkdd99/Challenge/chall.htm>.
* ACS PUMS — <https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/>
  and the Census Bureau Data API terms of service
  <https://www.census.gov/data/developers/about/terms-of-service.html>.

Nothing here is legal advice; the intent is only to record what each provider's
published terms say so that the release decision is made with open eyes.

Consequences for this repository:

* This repository ships **code and aggregated result files only**. The
  processed inputs are *not* committed, and no raw data is included.
* Anyone reproducing the experiments should **download the raw sources
  themselves** (URLs in §1–§5) and run the scripts in this directory. That is
  the only route that is unambiguously consistent with all four licenses.
* A redistribution of the discretized CSVs would be a redistribution of
  *derived* versions of IMDb, MovieLens and Instacart data; the IMDb terms
  cover "altered" data explicitly, and the GroupLens terms require separate
  permission for "the data". **The IMDb-derived V-shape in particular should
  not be published as a data artifact**, and the MovieLens and Instacart
  derivatives should only be published after obtaining permission from
  GroupLens and Instacart respectively.
* For a public artifact we recommend publishing only the TPC-H variant
  (regenerable at will) and, at most, an artifact that ships *scripts plus
  checksums and row counts* for the other datasets rather than the data itself.
* Readers of the paper who want to work with the exact inputs should be pointed
  at the sources, the reconstruction commands in this file, and the row
  counts/`sha256` in the archive manifest, which let them verify that their
  reconstruction matches the one used in the paper without shipping the data.