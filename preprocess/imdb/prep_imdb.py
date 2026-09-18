"""
IMDb (official 2026-08-19 snapshot, datasets.imdbws.com) -> Chain-NPM
V-shape deliverables, subset <= 2M total rows.

Structure (V-shape):  names <- principals -> titles
  names.csv      : NKEY, ALIVE, BYEARBIN, PROF1
  titles.csv     : TKEY, TITLETYPE, GENRE1, YEARBIN, ISADULT, RTBIN
  principals.csv : PKEY, CATEGORY, NKEY, TKEY  (last two cols = FKs)
  + *_domain.json + vshape_spec.json + imdb_report.json

Sampling (deterministic, SEED=42): candidate titles =
titleType in {movie, tvSeries} & startYear in [2000,2019]; sample
T_TARGET of them, cascade principals and names, drop FK-orphaned
titles; if total rows > ROW_CAP, shrink T_TARGET and retry (<=2 iters).

Official snapshot has NO gender column; ALIVE (deathYear missing) +
PROF1 substitute. GENRE1 = first listed genre (rare merged to OTHER).

Licensing: the IMDb non-commercial terms forbid republishing or repurposing
the data ("must not be altered/republished/resold/repurposed"), so neither the
raw snapshot nor these processed CSVs may be redistributed. See
../README.md ("Data licensing / redistribution").

Run: python3 prep_imdb.py [--dir <this dir>]
"""
import argparse, json, os
import numpy as np
import pandas as pd
import pyarrow.csv as pcsv

SEED = 42
MAXDOM = 16
ROW_CAP = 2_000_000
T_INITIAL = 140_000
RAW = None

pa_read = dict(block_size=1 << 22)


def read_opts():
    import pyarrow.csv as pc
    return pc.ReadOptions(block_size=1 << 22), \
        pc.ParseOptions(delimiter='\t', quote_char=False), \
        pc.ConvertOptions(null_values=['\\N', '', 'NA'],
                          strings_can_be_null=True)


def merge_small(counts: pd.Series, maxdom: int):
    if len(counts) <= maxdom:
        return {v: i for i, v in enumerate(sorted(counts.index.astype(str)))}, False
    order = counts.sort_values(ascending=False).index.astype(str).tolist()
    m = {v: i for i, v in enumerate(order[:maxdom - 1])}
    for v in order[maxdom - 1:]:
        m[v] = maxdom - 1
    return m, True


def gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() - (n + 1) * v.sum()) / (n * v.sum()))


def stream_table(path, columns):
    ro, po, co = read_opts()
    co = pcsv.ConvertOptions(null_values=['\\N', '', 'NA'],
                             strings_can_be_null=True,
                             include_columns=columns)
    reader = pcsv.open_csv(path, read_options=ro, parse_options=po,
                           convert_options=co)
    for batch in reader:
        yield batch.to_pandas()


def main():
    global RAW
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    RAW = os.path.join(args.dir, 'raw')
    out = os.path.join(args.dir, 'processed')
    os.makedirs(out, exist_ok=True)
    tb = os.path.join(RAW, 'title.basics.tsv')
    tp = os.path.join(RAW, 'title.principals.tsv')
    nb = os.path.join(RAW, 'name.basics.tsv')

    # ---------- pass 1: candidate titles ----------
    print('pass 1: candidate titles', flush=True)
    cand_t = []
    cand_rows = []
    n_scanned = 0
    for df in stream_table(tb, ['tconst', 'titleType', 'isAdult',
                                'startYear', 'runtimeMinutes', 'genres']):
        n_scanned += len(df)
        m = (df['titleType'].isin(['movie', 'tvSeries'])) & \
            (pd.to_numeric(df['startYear'], errors='coerce') >= 2000) & \
            (pd.to_numeric(df['startYear'], errors='coerce') <= 2019)
        sub = df[m]
        cand_rows.append(sub)
    cand = pd.concat(cand_rows, ignore_index=True)
    print(f'  scanned {n_scanned}, candidates {len(cand)}', flush=True)

    rng = np.random.default_rng(SEED)
    t_target = T_INITIAL
    for attempt in range(3):
        idx = rng.choice(len(cand), size=min(t_target, len(cand)),
                         replace=False)
        idx.sort()
        tsel = cand.iloc[idx].reset_index(drop=True)
        tset = set(tsel['tconst'])

        # ---------- pass 2: principals cascade ----------
        print(f'pass 2 (attempt {attempt}): principals for {len(tset)} titles',
              flush=True)
        keep = []
        for df in stream_table(tp, ['tconst', 'ordering', 'nconst',
                                    'category']):
            sub = df[df['tconst'].isin(tset)]
            if len(sub):
                keep.append(sub)
        prin = pd.concat(keep, ignore_index=True)
        prin = prin.dropna(subset=['nconst'])
        nset = set(prin['nconst'])
        titles_ok = set(prin['tconst'])
        tsel = tsel[tsel['tconst'].isin(titles_ok)].reset_index(drop=True)
        total = len(tsel) + len(prin) + len(nset)
        print(f'  titles {len(tsel)} principals {len(prin)} '
              f'names~{len(nset)} total~{total}', flush=True)
        if total <= ROW_CAP:
            break
        t_target = int(t_target * ROW_CAP / total * 0.97)
        rng = np.random.default_rng(SEED)  # deterministic retry path
        print(f'  over cap; resample with T_TARGET={t_target}', flush=True)

    # ---------- pass 3: names ----------
    print('pass 3: names', flush=True)
    nkeep = []
    for df in stream_table(nb, ['nconst', 'birthYear', 'deathYear',
                                'primaryProfession']):
        sub = df[df['nconst'].isin(nset)]
        if len(sub):
            nkeep.append(sub)
    names = pd.concat(nkeep, ignore_index=True)
    # snapshot drift: a few principals.nconst absent from name.basics
    n_missing = len(nset) - names['nconst'].nunique()
    if n_missing:
        print(f'  dropping {n_missing} principal-names absent from '
              f'name.basics (snapshot drift)', flush=True)
        keep_n = set(names['nconst'])
        prin = prin[prin['nconst'].isin(keep_n)].reset_index(drop=True)
        titles_ok = set(prin['tconst'])
        n_dropped_titles = (~tsel['tconst'].isin(titles_ok)).sum()
        tsel = tsel[tsel['tconst'].isin(titles_ok)].reset_index(drop=True)
        if n_dropped_titles:
            print(f'  dropped {n_dropped_titles} orphaned titles',
                  flush=True)

    # ---------- attribute encoding ----------
    # names
    names = names.sort_values('nconst').reset_index(drop=True)
    nkey = {c: i for i, c in enumerate(names['nconst'])}
    alive = names['deathYear'].isna().astype(int).values
    by = pd.to_numeric(names['birthYear'], errors='coerce')
    byear = np.full(len(names), 13, dtype=int)          # 13 = NA bin
    valid = by.notna()
    dec = ((by[valid].values // 10) * 10).clip(1870, 2010)
    byear[valid.values] = ((dec - 1870) // 10).astype(int)  # 0..14->clip
    byear[valid.values] = np.clip(byear[valid.values], 0, 12)
    prof1 = names['primaryProfession'].fillna('').str.split(',').str[0].fillna('')
    prof_counts = prof1[prof1 != ''].value_counts()
    prof_map, prof_merged = merge_small(prof_counts, MAXDOM)
    prof1_code = prof1.map(lambda v: prof_map.get(v, MAXDOM - 1)).astype(int).values
    prof1_code[prof1.values == ''] = MAXDOM - 1
    names_df = pd.DataFrame({'NKEY': np.arange(len(names)), 'ALIVE': alive,
                             'BYEARBIN': byear, 'PROF1': prof1_code})

    # titles
    tsel = tsel.sort_values('tconst').reset_index(drop=True)
    tkey = {c: i for i, c in enumerate(tsel['tconst'])}
    ttype = (tsel['titleType'] == 'tvSeries').astype(int).values
    genre1 = tsel['genres'].fillna('').str.split(',').str[0].fillna('')
    g_counts = genre1[genre1 != ''].value_counts()
    g_map, g_merged = merge_small(g_counts, MAXDOM)
    g1 = genre1.map(lambda v: g_map.get(v, MAXDOM - 1)).astype(int).values
    g1[genre1.values == ''] = MAXDOM - 1
    yr = pd.to_numeric(tsel['startYear'], errors='coerce').astype(float)
    ybin = ((yr - 2000) // 5).astype(int).clip(0, 3).values
    adult = tsel['isAdult'].fillna('0').astype(str).str.strip()
    adult = (adult == '1').astype(int).values
    rt = pd.to_numeric(tsel['runtimeMinutes'], errors='coerce')
    rtbin = np.zeros(len(tsel), dtype=int)               # 0=NA
    rtbin[(rt >= 1) & (rt < 60)] = 1
    rtbin[(rt >= 60) & (rt < 90)] = 2
    rtbin[(rt >= 90) & (rt < 120)] = 3
    rtbin[rt >= 120] = 4
    titles_df = pd.DataFrame({'TKEY': np.arange(len(tsel)),
                              'TITLETYPE': ttype, 'GENRE1': g1,
                              'YEARBIN': ybin, 'ISADULT': adult,
                              'RTBIN': rtbin})

    # principals
    prin['TKEY'] = prin['tconst'].map(tkey)
    prin['NKEY'] = prin['nconst'].map(nkey)
    assert prin['TKEY'].notna().all() and prin['NKEY'].notna().all()
    cat_counts = prin['category'].fillna('OTHER').value_counts()
    c_map, c_merged = merge_small(cat_counts, MAXDOM)
    cat = prin['category'].fillna('OTHER').map(c_map).astype(int).values
    prin = prin.assign(CATEGORY=cat)
    prin = prin.sort_values(['NKEY', 'TKEY']).reset_index(drop=True)
    prin_df = pd.DataFrame({'PKEY': np.arange(len(prin)),
                            'CATEGORY': prin['CATEGORY'].values,
                            'NKEY': prin['NKEY'].astype(int).values,
                            'TKEY': prin['TKEY'].astype(int).values})

    # ---------- write ----------
    names_df.to_csv(os.path.join(out, 'names.csv'), index=False)
    titles_df.to_csv(os.path.join(out, 'titles.csv'), index=False)
    prin_df.to_csv(os.path.join(out, 'principals.csv'), index=False)

    n_dom = {'ALIVE': {'size': 2},
             'BYEARBIN': {'size': int(names_df['BYEARBIN'].max()) + 1},
             'PROF1': {'size': int(names_df['PROF1'].max()) + 1}}
    t_dom = {'TITLETYPE': {'size': 2},
             'GENRE1': {'size': int(titles_df['GENRE1'].max()) + 1},
             'YEARBIN': {'size': int(titles_df['YEARBIN'].max()) + 1},
             'ISADULT': {'size': 2},
             'RTBIN': {'size': int(titles_df['RTBIN'].max()) + 1}}
    p_dom = {'CATEGORY': {'size': int(prin_df['CATEGORY'].max()) + 1}}
    json.dump(n_dom, open(os.path.join(out, 'names_domain.json'), 'w'))
    json.dump(t_dom, open(os.path.join(out, 'titles_domain.json'), 'w'))
    json.dump(p_dom, open(os.path.join(out, 'principals_domain.json'), 'w'))

    spec = {
        'structure': 'vshape',
        'root1': {'file': 'names.csv', 'pk': 'NKEY',
                  'attrs': ['ALIVE', 'BYEARBIN', 'PROF1'],
                  'domain_json': 'names_domain.json'},
        'root2': {'file': 'titles.csv', 'pk': 'TKEY',
                  'attrs': ['TITLETYPE', 'GENRE1', 'YEARBIN', 'ISADULT',
                            'RTBIN'],
                  'domain_json': 'titles_domain.json'},
        'child': {'file': 'principals.csv', 'pk': 'PKEY',
                  'attrs': ['CATEGORY'], 'fk1': 'NKEY', 'fk2': 'TKEY',
                  'fk_order': 'last two columns are NKEY then TKEY',
                  'domain_json': 'principals_domain.json'},
        'seed': SEED,
        'subset': {'candidate_filter': "titleType in {movie,tvSeries} and "
                   "startYear in [2000,2019]",
                   'titles_sampled': int(len(tsel)),
                   'row_cap': ROW_CAP,
                   'note': 'uniform sample over candidate titles, cascade '
                           'closure, orphan titles dropped'},
        'source': 'IMDb official non-commercial snapshot '
                  '(datasets.imdbws.com, downloaded 2026-08-19)',
        'notes': 'no gender column in official snapshot; ALIVE/PROF1 used; '
                 'GENRE1=first listed genre, rare -> OTHER; CATEGORY rare '
                 '-> OTHER',
    }
    json.dump(spec, open(os.path.join(out, 'vshape_spec.json'), 'w'), indent=2)

    # ---------- stats ----------
    per_t = prin_df.groupby('TKEY').size()
    per_n = prin_df.groupby('NKEY').size()
    # two-hop cue: GENRE1 == Animation share of titles reached via
    # persons born <=1949 (bin<=7) vs >=1980 (bin>=11)
    anim = titles_df['GENRE1'].eq(
        g_map.get('Animation', -1)).values if 'Animation' in g_map else None
    cue = {}
    if anim is not None:
        t_anim = anim[prin_df['TKEY'].values]
        nb_arr = names_df['BYEARBIN'].values[prin_df['NKEY'].values]
        cue = {'anim_share_via_person_born_<=1949':
                   round(float(t_anim[nb_arr <= 6].mean()), 4),
               'anim_share_via_person_born_>=1980':
                   round(float(t_anim[nb_arr >= 11].mean()), 4)}
    stats = {
        'titles': int(len(titles_df)), 'names': int(len(names_df)),
        'principals': int(len(prin_df)),
        'total_rows': int(len(titles_df) + len(names_df) + len(prin_df)),
        'principals_per_title': {'median': float(per_t.median()),
                                 'max': int(per_t.max()),
                                 'gini': round(gini(per_t.values), 4)},
        'principals_per_name': {'median': float(per_n.median()),
                                'max': int(per_n.max()),
                                'gini': round(gini(per_n.values), 4)},
        'category_merged': c_merged, 'genre1_merged': g_merged,
        'prof1_merged': prof_merged,
        'cross_hop_cue': cue,
    }
    json.dump(stats, open(os.path.join(out, 'imdb_report.json'), 'w'),
              indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
