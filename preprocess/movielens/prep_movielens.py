"""
MovieLens-1M -> Chain-NPM V-shape deliverables.

Structure (V-shape):  users <- ratings -> movies
  users.csv  : col0 UKEY (dense PK), attrs GENDER, AGE, OCC
  movies.csv : col0 MKEY (dense PK), attrs GENRE1 (primary genre,
               rare genres merged to OTHER so domain<=16), NGEN (num genres)
  ratings.csv: col0 RKEY (dense PK), attr RATING, last two cols UKEY, MKEY
               (FKs into users / movies)
  + users_domain.json / movies_domain.json / ratings_domain.json
  + vshape_spec.json

Source: https://grouplens.org/datasets/movielens/1m/ (GroupLens,
non-commercial research license; Harper & Konstan 2015).
Licensing: the GroupLens usage license states "The user may not redistribute
the data without separate permission", so neither the raw archive nor these
processed CSVs may be redistributed. See ../README.md ("Data licensing /
redistribution").
Deterministic; SEED fixed to 42.

Run: python3 prep_movielens.py [--dir <this dir>]
"""
import argparse, json, os
import numpy as np
import pandas as pd

SEED = 42
MAXDOM = 16


def merge_small(counts: pd.Series, maxdom: int, other_name='OTHER'):
    """Keep the maxdom-1 most frequent categories, merge rest into OTHER.
    Deterministic (sort by freq desc, then name)."""
    if len(counts) <= maxdom:
        return {v: i for i, v in enumerate(counts.index)}, False
    order = counts.sort_values(ascending=False).index.tolist()
    keep = order[:maxdom - 1]
    mapping = {v: i for i, v in enumerate(keep)}
    for v in order[maxdom - 1:]:
        mapping[v] = maxdom - 1
    return mapping, True


def gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() - (n + 1) * v.sum()) / (n * v.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)  # registered for reproducibility
    raw = os.path.join(args.dir, 'raw', 'ml-1m')
    out = os.path.join(args.dir, 'processed')
    os.makedirs(out, exist_ok=True)

    users = pd.read_csv(os.path.join(raw, 'users.dat'), sep='::',
                        engine='python', header=None,
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip'],
                        encoding='latin-1')
    ratings = pd.read_csv(os.path.join(raw, 'ratings.dat'), sep='::',
                          engine='python', header=None,
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                          encoding='latin-1')
    movies = pd.read_csv(os.path.join(raw, 'movies.dat'), sep='::',
                         engine='python', header=None,
                         names=['MovieID', 'Title', 'Genres'],
                         encoding='latin-1')
    print('raw:', users.shape, ratings.shape, movies.shape, flush=True)

    # ---------------- users ----------------
    users = users.sort_values('UserID').reset_index(drop=True)
    ukey = {int(u): i for i, u in enumerate(users['UserID'])}
    gender = (users['Gender'] == 'M').astype(int).values
    age_vals = [1, 18, 25, 35, 45, 50, 56]
    age = users['Age'].map({v: i for i, v in enumerate(age_vals)}).astype(int).values
    occ_counts = users['Occupation'].value_counts()
    occ_map, occ_merged = merge_small(occ_counts, MAXDOM)
    occ = users['Occupation'].map(occ_map).astype(int).values
    users_df = pd.DataFrame({'UKEY': np.arange(len(users)),
                             'GENDER': gender, 'AGE': age, 'OCC': occ})

    # ---------------- movies ----------------
    movies = movies.sort_values('MovieID').reset_index(drop=True)
    mkey = {int(m): i for i, m in enumerate(movies['MovieID'])}
    genres_split = movies['Genres'].str.split('|')
    ngen = genres_split.str.len()
    primary = genres_split.str[0]
    p_counts = primary.value_counts()
    g_map, g_merged = merge_small(p_counts, MAXDOM)
    genre1 = primary.map(g_map).astype(int).values
    ngen_bin = ngen.clip(1, 4).values - 1          # 1,2,3,4+ -> 0..3
    movies_df = pd.DataFrame({'MKEY': np.arange(len(movies)),
                              'GENRE1': genre1, 'NGEN': ngen_bin})

    # ---------------- ratings (child of V) ----------------
    ratings = ratings.rename(columns={})
    ratings['UKEY'] = ratings['UserID'].map(ukey)
    ratings['MKEY'] = ratings['MovieID'].map(mkey)
    assert ratings['UKEY'].notna().all() and ratings['MKEY'].notna().all()
    ratings = ratings.sort_values(['UKEY', 'MKEY']).reset_index(drop=True)
    ratings_df = pd.DataFrame({
        'RKEY': np.arange(len(ratings)),
        'RATING': (ratings['Rating'].astype(int) - 1).values,
        'UKEY': ratings['UKEY'].astype(int).values,
        'MKEY': ratings['MKEY'].astype(int).values})

    # ---------------- write ----------------
    users_df.to_csv(os.path.join(out, 'users.csv'), index=False)
    movies_df.to_csv(os.path.join(out, 'movies.csv'), index=False)
    ratings_df.to_csv(os.path.join(out, 'ratings.csv'), index=False)

    u_dom = {'GENDER': {'size': 2}, 'AGE': {'size': len(age_vals)},
             'OCC': {'size': int(users_df['OCC'].max()) + 1}}
    m_dom = {'GENRE1': {'size': int(movies_df['GENRE1'].max()) + 1},
             'NGEN': {'size': int(movies_df['NGEN'].max()) + 1}}
    r_dom = {'RATING': {'size': 5}}
    json.dump(u_dom, open(os.path.join(out, 'users_domain.json'), 'w'))
    json.dump(m_dom, open(os.path.join(out, 'movies_domain.json'), 'w'))
    json.dump(r_dom, open(os.path.join(out, 'ratings_domain.json'), 'w'))

    spec = {
        'structure': 'vshape',
        'child_on_top': False,
        'root1': {'file': 'users.csv', 'pk': 'UKEY',
                  'attrs': ['GENDER', 'AGE', 'OCC'],
                  'domain_json': 'users_domain.json'},
        'root2': {'file': 'movies.csv', 'pk': 'MKEY',
                  'attrs': ['GENRE1', 'NGEN'],
                  'domain_json': 'movies_domain.json'},
        'child': {'file': 'ratings.csv', 'pk': 'RKEY',
                  'attrs': ['RATING'],
                  'fk1': 'UKEY', 'fk2': 'MKEY',
                  'fk_order': 'last two columns are UKEY then MKEY',
                  'domain_json': 'ratings_domain.json'},
        'seed': SEED,
        'source': 'MovieLens-1M (Harper & Konstan, ACM TiiS 2015), '
                  'https://grouplens.org/datasets/movielens/1m/',
        'notes': 'OCC rare occupations merged to OTHER (domain<=16); '
                 'GENRE1 rare primary genres merged to OTHER; '
                 'NGEN=1,2,3,4+ genre count bins (multi-label record).',
    }
    json.dump(spec, open(os.path.join(out, 'vshape_spec.json'), 'w'), indent=2)

    # ---------------- stats ----------------
    per_u = ratings_df.groupby('UKEY').size()
    per_m = ratings_df.groupby('MKEY').size()
    # cross-hop cue: Children's/Animation share of ratings per age group
    childlike = set(g for g in p_counts.index if g in ("Animation", "Children's"))
    childlike_idx = [g_map[g] for g in childlike if g in g_map]
    is_child = movies_df['GENRE1'].isin(childlike_idx).values
    r_genre_child = is_child[ratings_df['MKEY'].values]
    young = ratings_df['UKEY'].map(
        users_df.set_index('UKEY')['AGE'].to_dict()).values == 0
    old = ratings_df['UKEY'].map(
        users_df.set_index('UKEY')['AGE'].to_dict()).values == 6
    stats = {
        'users': int(len(users_df)), 'movies': int(len(movies_df)),
        'ratings': int(len(ratings_df)),
        'ratings_per_user': {'median': float(per_u.median()),
                             'max': int(per_u.max()),
                             'gini': round(gini(per_u.values), 4)},
        'ratings_per_movie': {'median': float(per_m.median()),
                              'max': int(per_m.max()),
                              'gini': round(gini(per_m.values), 4)},
        'occ_merged_to_other': occ_merged,
        'genre1_merged_to_other': g_merged,
        'genre1_map_size': int(movies_df['GENRE1'].max()) + 1,
        'cross_hop_cue': {
            "share_Animation/Children's_among_age_Under18":
                round(float(r_genre_child[young].mean()), 4),
            "share_Animation/Children's_among_age_56+":
                round(float(r_genre_child[old].mean()), 4)},
    }
    json.dump(stats, open(os.path.join(out, 'movielens_report.json'), 'w'),
              indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
