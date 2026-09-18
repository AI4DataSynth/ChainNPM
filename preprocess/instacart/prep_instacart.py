"""
Instacart Market Basket Analysis -> Chain-NPM V-shape deliverables,
subset of the "prior" orders (same split PrivPetal uses).

FK-faithful structure (order_products carries BOTH fks):
    products <- order_products -> orders
delivered as V-shape (same convention as users <- ratings -> movies):
  products.csv      : PKEY, DEPT, AISLE
  orders.csv        : OKEY, DOW, HOURBIN, DPOBIN, ONUMBIN
  order_products.csv: OPKEY, REORDERED, ATCBIN, PKEY, OKEY
                      (last two cols = FKs to products / orders)
  + *_domain.json + vshape_spec.json + instacart_report.json

Note: the task sheet's "products<-order_products<-orders chain" is not
FK-realizable (order_products references BOTH products and orders); the
V-shape above is the faithful representation and matches the pipeline's
vshape_spec convention.

Sampling (deterministic, SEED=42): uniform sample of S_TARGET prior
orders, cascade order_products, keep only products that appear (FK
closure); if total rows > ROW_CAP, shrink S_TARGET and retry.

Domain control: departments 21 -> merge smallest to OTHER (16);
aisles 134 -> top-15 + OTHER (16); hours -> 4h blocks (6); days_since
prior order -> 7 bins incl NA; order_number -> 7 bins; add_to_cart ->
7 bins.

Source: Kaggle/Instacart "Instacart Market Basket Analysis" (Bo Yang,
Instacart, 2017). The provider page states the dataset is "provided as-is
for non-commercial use"; see ../README.md ("Data licensing / redistribution")
before redistributing anything derived from it. The copy used for the paper
was mirrored from a public Hugging Face mirror, because the original
Instacart hosting page was marked "temporarily unavailable".

Run: python3 prep_instacart.py [--dir <this dir>]
"""
import argparse, json, os
import numpy as np
import pandas as pd

SEED = 42
MAXDOM = 16
ROW_CAP = 2_000_000
S_INITIAL = 150_000


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
    raw = os.path.join(args.dir, 'raw')
    out = os.path.join(args.dir, 'processed')
    os.makedirs(out, exist_ok=True)

    orders = pd.read_csv(os.path.join(raw, 'orders.csv'))
    products = pd.read_csv(os.path.join(raw, 'products.csv'))
    print('raw orders', orders.shape, 'products', products.shape, flush=True)

    prior = orders[orders['eval_set'] == 'prior'].reset_index(drop=True)
    print('prior orders:', len(prior), flush=True)

    op_path = os.path.join(raw, 'order_products__prior.csv')
    rng = np.random.default_rng(SEED)
    s_target = S_INITIAL
    for attempt in range(3):
        sel = prior.sample(n=min(s_target, len(prior)),
                           random_state=SEED).sort_values('order_id')
        oset = set(sel['order_id'])
        print(f'attempt {attempt}: streaming order_products for '
              f'{len(oset)} orders', flush=True)
        keep = []
        for chunk in pd.read_csv(op_path, chunksize=1_000_000):
            sub = chunk[chunk['order_id'].isin(oset)]
            if len(sub):
                keep.append(sub)
        op = pd.concat(keep, ignore_index=True)
        total = len(sel) + len(op) + op['product_id'].nunique()
        print(f'  order_products {len(op)}, products~'
              f"{op['product_id'].nunique()}, total~{total}", flush=True)
        if total <= ROW_CAP:
            break
        s_target = int(s_target * ROW_CAP / total * 0.97)
        print(f'  over cap; resample with S_TARGET={s_target}', flush=True)

    # ---------------- products ----------------
    prod = products[products['product_id'].isin(op['product_id'])]
    prod = prod.sort_values('product_id').reset_index(drop=True)
    pkey = {int(v): i for i, v in enumerate(prod['product_id'])}
    dept_counts = prod['department_id'].value_counts()
    if len(dept_counts) > MAXDOM:
        order_ = dept_counts.sort_values(ascending=False).index.tolist()
        dept_map = {v: i for i, v in enumerate(order_[:MAXDOM - 1])}
        for v in order_[MAXDOM - 1:]:
            dept_map[v] = MAXDOM - 1
        dept_merged = True
    else:
        dept_map = {v: i for i, v in enumerate(
            sorted(dept_counts.index))}
        dept_merged = False
    aisle_counts = prod['aisle_id'].value_counts()
    aisle_order = aisle_counts.sort_values(ascending=False).index.tolist()
    aisle_map = {v: i for i, v in enumerate(aisle_order[:MAXDOM - 1])}
    for v in aisle_order[MAXDOM - 1:]:
        aisle_map[v] = MAXDOM - 1
    products_df = pd.DataFrame({
        'PKEY': np.arange(len(prod)),
        'DEPT': prod['department_id'].map(dept_map).astype(int).values,
        'AISLE': prod['aisle_id'].map(aisle_map).astype(int).values})

    # ---------------- orders ----------------
    okey = {int(v): i for i, v in enumerate(sel['order_id'])}
    dow = sel['order_dow'].astype(int).values
    hourbin = (sel['order_hour_of_day'].astype(int) // 4).values     # 0..5
    dpo = sel['days_since_prior_order']
    dpobin = np.full(len(sel), 6, dtype=int)                        # 6=NA
    dpobin[(dpo == 0).values] = 0
    dpobin[(dpo >= 1) & (dpo <= 3)] = 1
    dpobin[(dpo >= 4) & (dpo <= 7)] = 2
    dpobin[(dpo >= 8) & (dpo <= 14)] = 3
    dpobin[(dpo >= 15) & (dpo <= 21)] = 4
    dpobin[(dpo >= 22) & (dpo <= 30)] = 5
    onum = sel['order_number'].astype(int).clip(1, 7).values - 1    # 0..6
    orders_df = pd.DataFrame({'OKEY': np.arange(len(sel)), 'DOW': dow,
                              'HOURBIN': hourbin, 'DPOBIN': dpobin,
                              'ONUMBIN': onum})

    # ---------------- order_products (child) ----------------
    op['PKEY'] = op['product_id'].map(pkey)
    op['OKEY'] = op['order_id'].map(okey)
    assert op['PKEY'].notna().all() and op['OKEY'].notna().all()
    atc = op['add_to_cart_order'].astype(int).clip(1, 7).values - 1  # 0..6
    op = op.sort_values(['OKEY', 'PKEY']).reset_index(drop=True)
    op_df = pd.DataFrame({
        'OPKEY': np.arange(len(op)),
        'REORDERED': op['reordered'].astype(int).values,
        'ATCBIN': atc,
        'PKEY': op['PKEY'].astype(int).values,
        'OKEY': op['OKEY'].astype(int).values})

    # ---------------- write ----------------
    products_df.to_csv(os.path.join(out, 'products.csv'), index=False)
    orders_df.to_csv(os.path.join(out, 'orders.csv'), index=False)
    op_df.to_csv(os.path.join(out, 'order_products.csv'), index=False)
    p_dom = {'DEPT': {'size': int(products_df['DEPT'].max()) + 1},
             'AISLE': {'size': int(products_df['AISLE'].max()) + 1}}
    o_dom = {'DOW': {'size': 7},
             'HOURBIN': {'size': int(orders_df['HOURBIN'].max()) + 1},
             'DPOBIN': {'size': int(orders_df['DPOBIN'].max()) + 1},
             'ONUMBIN': {'size': int(orders_df['ONUMBIN'].max()) + 1}}
    op_dom = {'REORDERED': {'size': 2},
              'ATCBIN': {'size': int(op_df['ATCBIN'].max()) + 1}}
    json.dump(p_dom, open(os.path.join(out, 'products_domain.json'), 'w'))
    json.dump(o_dom, open(os.path.join(out, 'orders_domain.json'), 'w'))
    json.dump(op_dom, open(os.path.join(out, 'order_products_domain.json'),
                           'w'))

    spec = {
        'structure': 'vshape',
        'root1': {'file': 'products.csv', 'pk': 'PKEY',
                  'attrs': ['DEPT', 'AISLE'],
                  'domain_json': 'products_domain.json'},
        'root2': {'file': 'orders.csv', 'pk': 'OKEY',
                  'attrs': ['DOW', 'HOURBIN', 'DPOBIN', 'ONUMBIN'],
                  'domain_json': 'orders_domain.json'},
        'child': {'file': 'order_products.csv', 'pk': 'OPKEY',
                  'attrs': ['REORDERED', 'ATCBIN'],
                  'fk1': 'PKEY', 'fk2': 'OKEY',
                  'fk_order': 'last two columns are PKEY then OKEY',
                  'domain_json': 'order_products_domain.json'},
        'seed': SEED,
        'subset': {'split': 'eval_set == prior',
                   'orders_sampled': int(len(sel)),
                   'row_cap': ROW_CAP,
                   'note': 'uniform sample over prior orders, cascade '
                           'closure over order_products, products kept '
                           'only if purchased'},
        'source': 'Instacart Market Basket Analysis (Instacart, 2017); '
                  'official page https://www.instacart.com/datasets/'
                  'grocery-shopping-2017 (dataset provided as-is for '
                  'non-commercial use), copy taken from a public Hugging '
                  'Face mirror',
        'notes': 'DEPT 21->16 (smallest merged to OTHER); AISLE 134->16 '
                 '(top-15 + OTHER); hour->4h blocks; FK-faithful V-shape '
                 'instead of the 3-table chain notation in the task sheet',
    }
    json.dump(spec, open(os.path.join(out, 'vshape_spec.json'), 'w'),
              indent=2)

    # ---------------- stats ----------------
    per_o = op_df.groupby('OKEY').size()
    per_p = op_df.groupby('PKEY').size()
    # two-hop cue: reorder rate of breakfast-dept products in early-hour
    # vs late-hour orders (dept <- op -> hour crosses both roots)
    dept_of_p = products_df['DEPT'].values[op_df['PKEY'].values]
    hour_of_o = orders_df['HOURBIN'].values[op_df['OKEY'].values]
    r = op_df['REORDERED'].values.astype(bool)
    cue = {
        'reorder_rate_overall': round(float(r.mean()), 4),
        'reorder_rate_hourbin0_morning': round(
            float(r[hour_of_o == 0].mean()), 4) if (hour_of_o == 0).any()
            else None,
        'reorder_rate_hourbin3_afternoon': round(
            float(r[hour_of_o == 3].mean()), 4) if (hour_of_o == 3).any()
            else None,
    }
    stats = {
        'products': int(len(products_df)),
        'orders': int(len(orders_df)),
        'order_products': int(len(op_df)),
        'total_rows': int(len(products_df) + len(orders_df)
                          + len(op_df)),
        'items_per_order': {'median': float(per_o.median()),
                            'max': int(per_o.max()),
                            'gini': round(gini(per_o.values), 4)},
        'purchases_per_product': {'median': float(per_p.median()),
                                  'max': int(per_p.max()),
                                  'gini': round(gini(per_p.values), 4)},
        'dept_merged': dept_merged,
        'cross_hop_cue': cue,
    }
    json.dump(stats, open(os.path.join(out, 'instacart_report.json'), 'w'),
              indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
