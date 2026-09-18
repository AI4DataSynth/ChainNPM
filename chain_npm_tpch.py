"""
Chain-NPM on TPC-H (standard PrivLava/PrivPetal setup):
private = Orders + Lineitem; Customer/Nation/Part public.
Nation is joined into Orders as a parent attribute (public join = free).

Runs on the GPU cluster where the data lives:
  python chain_npm_tpch.py --gt_dir <DATA_ROOT>/input_data \
      --epsilon 3.2 --out_prefix <OUT_DIR>/chainnpm_tpch_eps3.2

Outputs (original TPC-H schema, ready for tpch_multihop_eval.py):
  <prefix>_syn_o.csv, <prefix>_syn_l.csv, <prefix>_meta.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain_npm_generic import ChainNPM2  # noqa: E402

PRIORITIES = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW']
SHIPMODES = ['TRUCK', 'MAIL', 'REG AIR', 'AIR', 'FOB', 'RAIL', 'SHIP']
SHIPINSTRUCTS = ['DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN']
STATUSES = ['O', 'F', 'P']
DBIN_REPR = [0.01, 0.03, 0.06, 0.09]


def load(gt_dir):
    def tbl(name, cols):
        df = pd.read_csv(os.path.join(gt_dir, name + '.tbl'), sep='|',
                         header=None, usecols=range(len(cols)))
        df.columns = cols
        return df

    orders = tbl('orders', ['O_ORDERKEY', 'O_CUSTKEY', 'O_ORDERSTATUS',
                            'O_TOTALPRICE', 'O_ORDERDATE', 'O_ORDERPRIORITY',
                            'O_CLERK', 'O_SHIPPRIORITY', 'O_COMMENT'])
    lineitem = tbl('lineitem', ['L_ORDERKEY', 'L_PARTKEY', 'L_SUPPKEY',
                                'L_LINENUMBER', 'L_QUANTITY',
                                'L_EXTENDEDPRICE', 'L_DISCOUNT', 'L_TAX',
                                'L_RETURNFLAG', 'L_LINESTATUS', 'L_SHIPDATE',
                                'L_COMMITDATE', 'L_RECEIPTDATE',
                                'L_SHIPINSTRUCT', 'L_SHIPMODE', 'L_COMMENT'])
    customer = tbl('customer', ['C_CUSTKEY', 'C_NAME', 'C_ADDRESS',
                                'C_NATIONKEY', 'C_PHONE', 'C_ACCTBAL',
                                'C_MKTSEGMENT', 'C_COMMENT'])
    return orders, lineitem, customer


def discretize(orders, lineitem, customer):
    """Build dense-PK parent/child arrays for ChainNPM2."""
    o = orders.sort_values('O_ORDERKEY').reset_index(drop=True)
    okey_to_idx = {k: i for i, k in enumerate(o['O_ORDERKEY'])}

    cust_nation = customer.set_index('C_CUSTKEY')['C_NATIONKEY']
    nation = o['O_CUSTKEY'].map(cust_nation).fillna(0).astype(int).values
    year = o['O_ORDERDATE'].astype(str).str[:4].astype(int).values - 1992
    month = o['O_ORDERDATE'].astype(str).str[5:7].astype(int).values - 1
    status = o['O_ORDERSTATUS'].map(
        {s: i for i, s in enumerate(STATUSES)}).fillna(0).astype(int).values
    prio = o['O_ORDERPRIORITY'].map(
        {s: i for i, s in enumerate(PRIORITIES)}).fillna(0).astype(int).values

    parent = np.column_stack([np.arange(len(o)), status, year, month,
                              prio, nation])

    l = lineitem.copy()
    l['oidx'] = l['L_ORDERKEY'].map(okey_to_idx)
    l = l.dropna(subset=['oidx']).reset_index(drop=True)
    l['oidx'] = l['oidx'].astype(int)

    qbin = (l['L_QUANTITY'].astype(int) // 10).clip(0, 5).values
    disc = l['L_DISCOUNT'].astype(float).values
    dbin = np.digitize(disc, [0.02, 0.05, 0.08]).clip(0, 3)
    sm = l['L_SHIPMODE'].map(
        {s: i for i, s in enumerate(SHIPMODES)}).fillna(0).astype(int).values
    si = l['L_SHIPINSTRUCT'].map(
        {s: i for i, s in enumerate(SHIPINSTRUCTS)}).fillna(0).astype(int).values

    child = np.column_stack([np.arange(len(l)), l['oidx'].values,
                             qbin, dbin, sm, si])
    return parent, child, o, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_dir', required=True)
    ap.add_argument('--epsilon', type=float, required=True)
    ap.add_argument('--out_prefix', required=True)
    args = ap.parse_args()

    t0 = time.time()
    orders, lineitem, customer = load(args.gt_dir)
    parent, child, o_sorted, l_df = discretize(orders, lineitem, customer)
    print(f'loaded: orders={parent.shape[0]} lineitem={child.shape[0]} '
          f'({time.time()-t0:.0f}s)', flush=True)

    parent_attrs = ['STATUS', 'YEAR', 'MONTH', 'PRIORITY', 'NATION']
    parent_domains = {'STATUS': 3, 'YEAR': 7, 'MONTH': 12, 'PRIORITY': 5,
                      'NATION': 25}
    child_attrs = ['QBIN', 'DBIN', 'SHIPMODE', 'SHIPINSTRUCT']
    child_domains = {'QBIN': 6, 'DBIN': 4, 'SHIPMODE': 7, 'SHIPINSTRUCT': 4}

    model = ChainNPM2(parent_attrs, parent_domains, child_attrs,
                      child_domains, pp_pairs=[('NATION', 'YEAR'),
                                               ('YEAR', 'MONTH')],
                      tau=8, seed=0)
    n_ind = child.shape[0]
    delta = 1.0 / n_ind
    t1 = time.time()
    model.compute_marginals(parent, child, args.epsilon, delta)
    print(f'marginals done ({time.time()-t1:.0f}s)', flush=True)

    t2 = time.time()
    syn_parent, syn_child = model.synthesize(n_parents=parent.shape[0])
    print(f'synthesis done: parents={syn_parent.shape[0]} '
          f'children={syn_child.shape[0]} ({time.time()-t2:.0f}s)',
          flush=True)

    # ---- convert to original TPC-H schema ----
    rng = np.random.default_rng(1)
    cust_by_nation = customer.groupby('C_NATIONKEY')['C_CUSTKEY'].apply(
        lambda s: s.values)

    n_orders = syn_parent.shape[0]
    cols = {a: syn_parent[:, i + 1] for i, a in enumerate(parent_attrs)}
    okeys = np.arange(1, n_orders + 1)
    custkeys = np.empty(n_orders, dtype=int)
    for i in range(n_orders):
        nat = int(cols['NATION'][i])
        cands = cust_by_nation.get(nat, customer['C_CUSTKEY'].values)
        custkeys[i] = int(cands[rng.integers(len(cands))])
    syn_o = pd.DataFrame({
        'O_ORDERKEY': okeys,
        'O_CUSTKEY': custkeys,
        'O_ORDERSTATUS': [STATUSES[int(v)] for v in cols['STATUS']],
        'O_TOTALPRICE': rng.uniform(1000, 500000, n_orders).round(2),
        'O_ORDERDATE': [f'{1992+int(y)}-{int(m)+1:02d}-'
                        f'{int(rng.integers(1,29)):02d}'
                        for y, m in zip(cols['YEAR'], cols['MONTH'])],
        'O_ORDERPRIORITY': [PRIORITIES[int(v)] for v in cols['PRIORITY']],
        'O_CLERK': 'Clerk#000000001',
        'O_SHIPPRIORITY': 0,
        'O_COMMENT': 'synthetic',
    })

    # child: FK is dense parent index -> map to O_ORDERKEY
    n_l = syn_child.shape[0]
    ccols = {a: syn_child[:, i + 2] for i, a in enumerate(child_attrs)}
    okeys_l = okeys[syn_child[:, 1]]
    qty = np.clip(ccols['QBIN'] * 10
                  + rng.integers(0, 10, n_l), 1, 50)
    disc = np.array([DBIN_REPR[int(v)] for v in ccols['DBIN']])
    syn_l = pd.DataFrame({
        'L_ORDERKEY': okeys_l,
        'L_PARTKEY': rng.integers(1, 200001, n_l),
        'L_SUPPKEY': rng.integers(1, 10001, n_l),
        'L_LINENUMBER': 1,
        'L_QUANTITY': qty,
        'L_EXTENDEDPRICE': (qty * rng.uniform(900, 2000, n_l)).round(2),
        'L_DISCOUNT': disc,
        'L_TAX': rng.choice([0.0, 0.02, 0.04, 0.06, 0.08], n_l),
        'L_RETURNFLAG': rng.choice(['A', 'N', 'R'], n_l),
        'L_LINESTATUS': rng.choice(['O', 'F'], n_l),
        'L_SHIPDATE': '1994-01-01',
        'L_COMMITDATE': '1994-02-01',
        'L_RECEIPTDATE': '1994-03-01',
        'L_SHIPINSTRUCT': [SHIPINSTRUCTS[int(v)] for v in ccols['SHIPINSTRUCT']],
        'L_SHIPMODE': [SHIPMODES[int(v)] for v in ccols['SHIPMODE']],
        'L_COMMENT': 'synthetic',
    })

    syn_o.to_csv(args.out_prefix + '_syn_o.csv', index=False)
    syn_l.to_csv(args.out_prefix + '_syn_l.csv', index=False)
    meta = {'epsilon': args.epsilon, 'delta': delta,
            'n_orders_syn': int(n_orders), 'n_lineitem_syn': int(n_l),
            'n_queries_in_phase1': 42, 'runtime_s': time.time() - t0}
    json.dump(meta, open(args.out_prefix + '_meta.json', 'w'), indent=2)
    print('SAVED', args.out_prefix, meta, flush=True)


if __name__ == '__main__':
    main()
