"""
Chain-NPM2 driver for ACS PUMS California 2022 (household <- individual),
the non-artificial real-schema validation (PO-2).

Input:  <data_dir>/household.csv  HHKEY,ACR,BLD,HHT,TEN,VEH,NPbin,HINCPbin
        <data_dir>/individual.csv PKEY,<attrs>,HHKEY(last col, FK)
Output: <out_prefix>_syn_h.csv / _syn_i.csv in the same schema
        (individual FK references synthetic HHKEYs)

Run:
  python chain_npm_acs.py --data_dir .../acs/processed \
      --epsilon 3.2 --out_prefix .../chainnpm2_acs_eps3.2
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--epsilon', type=float, required=True)
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--tau', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    h = pd.read_csv(os.path.join(args.data_dir, 'household.csv'))
    i = pd.read_csv(os.path.join(args.data_dir, 'individual.csv'))
    h_dom = json.load(open(os.path.join(args.data_dir,
                                        'household_domain.json')))
    i_dom = json.load(open(os.path.join(args.data_dir,
                                        'individual_domain.json')))

    parent_attrs = ['ACR', 'BLD', 'HHT', 'TEN', 'VEH', 'NPbin', 'HINCPbin']
    child_attrs = ['AGEbin', 'SEX', 'SCHLbin', 'ESR', 'MAR', 'RELPbin',
                   'WKHPbin']
    parent_domains = {a: h_dom[a]['size'] for a in parent_attrs}
    child_domains = {a: i_dom[a]['size'] for a in child_attrs}

    parent = np.column_stack([h['HHKEY'].values]
                             + [h[a].astype(int).values for a in parent_attrs])
    child = np.column_stack([i['PKEY'].values, i['HHKEY'].astype(int).values]
                            + [i[a].astype(int).values for a in child_attrs])
    print('parent', parent.shape, 'child', child.shape, flush=True)

    model = ChainNPM2(
        parent_attrs, parent_domains, child_attrs, child_domains,
        pp_pairs=[('HHT', 'NPbin'), ('TEN', 'HINCPbin'),
                  ('HHT', 'HINCPbin'), ('BLD', 'VEH'), ('ACR', 'BLD')],
        cc_pairs=[],
        # NPbin IS the group size; conditioning SZ on it double-counts
        sz_attrs=['ACR', 'BLD', 'HHT', 'TEN', 'VEH', 'HINCPbin'],
        tau=args.tau, seed=args.seed)

    delta = 1.0 / child.shape[0]
    model.compute_marginals(parent, child, args.epsilon, delta)
    print('marginals done: {:.1f}s'.format(time.time() - t0), flush=True)

    parent_out, child_out = model.synthesize(n_parents=parent.shape[0])
    print('synthesize done: {:.1f}s'.format(time.time() - t0), flush=True)
    print('syn sizes:', parent_out.shape, child_out.shape, flush=True)

    syn_h = pd.DataFrame(parent_out, columns=['HHKEY'] + parent_attrs)
    syn_i = pd.DataFrame(child_out,
                         columns=['PKEY', 'HHKEY'] + child_attrs)
    outdir = os.path.dirname(os.path.abspath(args.out_prefix))
    os.makedirs(outdir, exist_ok=True)
    syn_h.to_csv(args.out_prefix + '_syn_h.csv', index=False)
    syn_i.to_csv(args.out_prefix + '_syn_i.csv', index=False)

    stats = {'epsilon': args.epsilon, 'seed': args.seed,
             'runtime_sec': time.time() - t0,
             'n_households': int(parent_out.shape[0]),
             'n_individuals': int(child_out.shape[0])}
    json.dump(stats, open(args.out_prefix + '_stats.json', 'w'), indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
