"""
Chain-NPM driver for the DEEP-CHAIN TPC-H benchmark (3 private tables:
customer -> orders -> lineitem). Data dir is produced by make_deep_gt.py
(customer.csv with C_NATIONKEY/C_MKTSEGMENT/C_ACCTBALBIN + the existing
planted orders.csv/lineitem.csv).

Outputs synthetic tables in the SAME discretized deep schema:
  <prefix>_syn_c.csv  C_CUSTKEY,C_NATIONKEY,C_MKTSEGMENT,C_ACCTBALBIN
  <prefix>_syn_o.csv  O_ORDERKEY,O_ORDERSTATUS,O_ORDERDATE1,O_ORDERDATE2,
                      O_ORDERDATE3,O_ORDERPRIORITY,O_CUSTKEY
  <prefix>_syn_l.csv  LINEITEMKEY,L_QUANTITY1,L_QUANTITY2,L_DISCOUNT,
                      L_SHIPMODE,L_SHIPINSTRUCT,L_ORDERKEY
(evaluator tpch_deep_eval.py consumes this schema directly)

Run (conda mhgap):
  python chain_npm_tpch_deep.py \
      --data_dir <DATA_ROOT>/TPC-H-deep \
      --epsilon 3.2 --out_prefix <OUT_DIR>/chainnpm3_eps3.2
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain_npm_3level import ChainNPM3  # noqa: E402


def load_deep(data_dir):
    c = pd.read_csv(os.path.join(data_dir, 'customer.csv'))
    o = pd.read_csv(os.path.join(data_dir, 'orders.csv'))
    l = pd.read_csv(os.path.join(data_dir, 'lineitem.csv'))

    c = c.sort_values('C_CUSTKEY').reset_index(drop=True)
    # dense customer PK -> index
    ckey_to_idx = np.full(int(c['C_CUSTKEY'].max()) + 1, -1, dtype=int)
    ckey_to_idx[c['C_CUSTKEY'].values] = np.arange(len(c))

    o = o.sort_values('O_ORDERKEY').reset_index(drop=True)
    okey_to_idx = np.full(int(o['O_ORDERKEY'].max()) + 1, -1, dtype=int)
    okey_to_idx[o['O_ORDERKEY'].values] = np.arange(len(o))

    top = np.column_stack([
        np.arange(len(c)),
        c['C_NATIONKEY'].astype(int).values,
        c['C_MKTSEGMENT'].astype(int).values,
        c['C_ACCTBALBIN'].astype(int).values,
    ])
    mid = np.column_stack([
        np.arange(len(o)),
        ckey_to_idx[o['O_CUSTKEY'].astype(int).values],
        o['O_ORDERSTATUS'].astype(int).values,
        o['O_ORDERDATE1'].astype(int).values,   # year 0..6
        o['O_ORDERDATE2'].astype(int).values,   # month 0..11
        o['O_ORDERPRIORITY'].astype(int).values,
    ])
    # drop orders whose customer key was not found (shouldn't happen)
    mid = mid[mid[:, 1] >= 0]
    dbin = np.clip(l['L_DISCOUNT'].astype(int).values // 3, 0, 3)
    bot = np.column_stack([
        np.arange(len(l)),
        okey_to_idx[l['L_ORDERKEY'].astype(int).values],
        l['L_QUANTITY1'].astype(int).values,
        l['L_QUANTITY2'].astype(int).values,
        dbin,
        l['L_SHIPMODE'].astype(int).values,
        l['L_SHIPINSTRUCT'].astype(int).values,
    ])
    bot = bot[bot[:, 1] >= 0]
    return top, mid, bot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--epsilon', type=float, required=True)
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--tau1', type=int, default=40)
    ap.add_argument('--tau2', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    top, mid, bot = load_deep(args.data_dir)
    print('top', top.shape, 'mid', mid.shape, 'bot', bot.shape, flush=True)

    top_attrs = ['NATION', 'SEGMENT', 'ACCTBIN']
    top_domains = {'NATION': 25, 'SEGMENT': 5,
                   'ACCTBIN': int(top[:, 3].max()) + 1}
    mid_attrs = ['STATUS', 'YEAR', 'MONTH', 'PRIORITY']
    mid_domains = {'STATUS': 3, 'YEAR': 7, 'MONTH': 12, 'PRIORITY': 5}
    bot_attrs = ['Q1', 'Q2', 'DBIN', 'SHIPMODE', 'SHIPINSTRUCT']
    bot_domains = {'Q1': int(bot[:, 2].max()) + 1, 'Q2': 10, 'DBIN': 4,
                   'SHIPMODE': 7, 'SHIPINSTRUCT': 4}

    model = ChainNPM3(
        top_attrs, top_domains, mid_attrs, mid_domains,
        bot_attrs, bot_domains,
        tt_pairs=[('NATION', 'SEGMENT'), ('NATION', 'ACCTBIN'),
                  ('SEGMENT', 'ACCTBIN')],
        tm_pairs=[('NATION', 'YEAR'), ('NATION', 'STATUS'),
                  ('SEGMENT', 'YEAR'), ('SEGMENT', 'PRIORITY'),
                  ('ACCTBIN', 'YEAR')],
        mb_pairs=[('YEAR', 'Q1'), ('YEAR', 'Q2'), ('YEAR', 'DBIN'),
                  ('PRIORITY', 'SHIPMODE'), ('STATUS', 'DBIN')],
        sz1_attrs=['NATION', 'SEGMENT', 'ACCTBIN'],
        sz2_attrs=['YEAR', 'STATUS', 'PRIORITY'],
        tau1=args.tau1, tau2=args.tau2, seed=args.seed)

    delta = 1.0 / bot.shape[0]
    model.compute_marginals(top, mid, bot, args.epsilon, delta)
    print('marginals done: {:.1f}s'.format(time.time() - t0), flush=True)

    top_out, mid_out, bot_out = model.synthesize(n_top=top.shape[0])
    print('synthesize done: {:.1f}s'.format(time.time() - t0), flush=True)
    print('syn sizes: top', top_out.shape, 'mid', mid_out.shape,
          'bot', bot_out.shape, flush=True)

    syn_c = pd.DataFrame(top_out, columns=[
        'C_CUSTKEY', 'C_NATIONKEY', 'C_MKTSEGMENT', 'C_ACCTBALBIN'])
    syn_o = pd.DataFrame(mid_out, columns=[
        'O_ORDERKEY', 'O_CUSTKEY', 'O_ORDERSTATUS', 'O_ORDERDATE1',
        'O_ORDERDATE2', 'O_ORDERPRIORITY'])
    syn_o = syn_o[['O_ORDERKEY', 'O_ORDERSTATUS', 'O_ORDERDATE1',
                   'O_ORDERDATE2', 'O_ORDERPRIORITY', 'O_CUSTKEY']]
    syn_l = pd.DataFrame(bot_out, columns=[
        'LINEITEMKEY', 'L_ORDERKEY', 'L_QUANTITY1', 'L_QUANTITY2',
        'L_DISCOUNT', 'L_SHIPMODE', 'L_SHIPINSTRUCT'])
    # emit discount on the GT scale (0..10) so the evaluator's //3 binning
    # recovers the modeled DBIN instead of double-binning
    syn_l['L_DISCOUNT'] = syn_l['L_DISCOUNT'] * 3
    syn_l = syn_l[['LINEITEMKEY', 'L_QUANTITY1', 'L_QUANTITY2',
                   'L_DISCOUNT', 'L_SHIPMODE', 'L_SHIPINSTRUCT',
                   'L_ORDERKEY']]

    outdir = os.path.dirname(os.path.abspath(args.out_prefix))
    os.makedirs(outdir, exist_ok=True)
    syn_c.to_csv(args.out_prefix + '_syn_c.csv', index=False)
    syn_o.to_csv(args.out_prefix + '_syn_o.csv', index=False)
    syn_l.to_csv(args.out_prefix + '_syn_l.csv', index=False)

    stats = {'epsilon': args.epsilon, 'seed': args.seed,
             'runtime_sec': time.time() - t0,
             'n_top': int(top_out.shape[0]), 'n_mid': int(mid_out.shape[0]),
             'n_bot': int(bot_out.shape[0])}
    json.dump(stats, open(args.out_prefix + '_stats.json', 'w'), indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
