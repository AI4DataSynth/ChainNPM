"""
Build the DEEP-CHAIN TPC-H benchmark variant.

Standard (PrivLava/PrivPetal) TPC-H setup: 2 private tables
(orders, lineitem); customer/nation/part are public, so the planted
nation<->year correlation collapses into a single FK neighborhood and
per-FK methods do not lose it.

Deep variant: customer is PRIVATE too, giving a 3-private-table chain
    customer -> orders -> lineitem
with planted correlations composing across all three private tables:
  (A) within customer :  C_MKTSEGMENT <-> C_NATIONKEY   (planted here)
  (B) customer->orders:  C_NATIONKEY  <-> O_YEAR        (already planted in
                          data/TPC-H/orders.csv via custkey reassignment)
  (C) orders->lineitem:  O_YEAR <-> order size; quantity scaled by part
                          brand/type ratio              (already planted in
                          data/TPC-H/lineitem.csv)
Composition A o B o C is a genuine multi-hop signal (segment <-> year <->
quantity) that no single per-FK neighborhood sees in its joint form.

Outputs to --out_dir (e.g. <DATA_ROOT>/TPC-H-deep):
  customer.csv  C_CUSTKEY,C_NATIONKEY,C_MKTSEGMENT,C_ACCTBALBIN (+domain)
  orders.csv/lineitem.csv/part/partsupp/supplier (+domains) copied from
  the existing planted data/TPC-H/.

Run (any numpy/pandas env, e.g. conda mhgap):
  python make_deep_gt.py --data_root <DATA_ROOT> \
      --out_dir <DATA_ROOT>/TPC-H-deep \
      [--plant_prob 0.6] [--seed 42]
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)   # .../privpetal_official/data
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--plant_prob', type=float, default=0.6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    inp = os.path.join(args.data_root, 'input_data')
    planted = os.path.join(args.data_root, 'TPC-H')
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- customer from raw .tbl
    c = pd.read_csv(os.path.join(inp, 'customer.tbl'), sep='|',
                    header=None, usecols=range(8))
    c.columns = ['C_CUSTKEY', 'C_NAME', 'C_ADDRESS', 'C_NATIONKEY',
                 'C_PHONE', 'C_ACCTBAL', 'C_MKTSEGMENT', 'C_COMMENT']
    c = c.sort_values('C_CUSTKEY').reset_index(drop=True)

    nation = c['C_NATIONKEY'].astype(int).values
    seg_vals = sorted(c['C_MKTSEGMENT'].unique())
    seg_map = {v: i for i, v in enumerate(seg_vals)}
    segment = c['C_MKTSEGMENT'].map(seg_map).astype(int).values
    bal = c['C_ACCTBAL'].astype(float).clip(-1000.0, 9999.99).values
    acctbin = ((bal + 1000.0) / 500.0).astype(int)   # 0..21, 22 bins

    # plant (A): segment <-> nation
    planted_seg = nation % 5
    mask = rng.random(len(c)) < args.plant_prob
    segment = np.where(mask, planted_seg, segment)

    cust = pd.DataFrame({
        'C_CUSTKEY': c['C_CUSTKEY'].astype(int).values,
        'C_NATIONKEY': nation,
        'C_MKTSEGMENT': segment,
        'C_ACCTBALBIN': acctbin,
    })
    cust.to_csv(os.path.join(args.out_dir, 'customer.csv'), index=False)
    json.dump({'C_NATIONKEY': {'size': 25},
               'C_MKTSEGMENT': {'size': 5},
               'C_ACCTBALBIN': {'size': int(acctbin.max()) + 1}},
              open(os.path.join(args.out_dir, 'customer_domain.json'), 'w'))

    # ---- copy already-planted private tables + public side tables
    for name in ['orders.csv', 'orders_domain.json',
                 'lineitem.csv', 'lineitem_domain.json',
                 'part.csv', 'part_domain.json',
                 'partsupp.csv', 'partsupp_domain.json',
                 'supplier.csv', 'supplier_domain.json']:
        shutil.copy(os.path.join(planted, name),
                    os.path.join(args.out_dir, name))

    # ---- planted-association audit
    o = pd.read_csv(os.path.join(args.out_dir, 'orders.csv'))
    j = o.merge(cust, left_on='O_CUSTKEY', right_on='C_CUSTKEY')
    ct = pd.crosstab(j['C_NATIONKEY'], j['O_ORDERDATE1'], normalize='index')
    assoc_nat_year = float(0.5 * np.abs(ct - ct.mean(axis=0)).values.sum())
    ct2 = pd.crosstab(j['C_MKTSEGMENT'], j['O_ORDERDATE1'], normalize='index')
    assoc_seg_year = float(0.5 * np.abs(ct2 - ct2.mean(axis=0)).values.sum())
    ct3 = pd.crosstab(cust['C_NATIONKEY'], cust['C_MKTSEGMENT'],
                      normalize='index')
    assoc_seg_nation = float(0.5 * np.abs(ct3 - ct3.mean(axis=0)).values.sum())
    sizes = o.groupby('O_CUSTKEY').size()
    report = {
        'n_customers': int(len(cust)),
        'n_customers_with_orders': int(len(sizes)),
        'n_orders': int(len(o)),
        'orders_per_customer_mean': float(sizes.mean()),
        'orders_per_customer_max': int(sizes.max()),
        'plant_prob': args.plant_prob,
        'assoc_seg_nation_TV': assoc_seg_nation,
        'assoc_nat_year_TV': assoc_nat_year,
        'assoc_seg_year_TV': assoc_seg_year,
        'segment_domain': seg_map,
    }
    json.dump(report, open(os.path.join(args.out_dir, 'deep_gt_report.json'),
                           'w'), indent=2)
    print(json.dumps(report, indent=2))
    print('DEEP GT SAVED:', args.out_dir, flush=True)


if __name__ == '__main__':
    main()
