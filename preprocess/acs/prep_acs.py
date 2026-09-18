"""
Discretize ACS PUMS 2022 California (psam_h06/psam_p06) into the PrivPetal
Census format (household.csv / individual.csv / *_domain.json) for the
PO-2 real-schema validation.

Schema (all domains modest for PrivMRF clique budgets):
  household attrs: ACR, BLD, HHT, TEN, VEH, NPbin, HINCPbin
  individual attrs: AGEbin, SEX, SCHLbin, ESR, MAR, RELPbin, WKHPbin

individual.csv layout (PrivPetal Data.load_data convention):
  col 0 = individual PK, middle cols = attrs, LAST col = household FK (dense)
household.csv: col 0 = dense household PK, then attrs.

NOTE on the state: the paper uses CALIFORNIA 2022 (psam_h06 / psam_p06,
state FIPS 6), not Wyoming. An earlier acquisition pass also downloaded the
Wyoming files (psam_h56 / psam_p56, state FIPS 56); those were never used by
the paper. This script reads the California files as given by --dir.

Run: python prep_acs.py [--dir tmp/data/acs]
Writes: <dir>/processed/{household.csv,individual.csv,
        household_domain.json,individual_domain.json,acs_report.json}
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

NA = 'NA'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    h = pd.read_csv(os.path.join(args.dir, 'psam_h06.csv'),
                    low_memory=False)
    p = pd.read_csv(os.path.join(args.dir, 'psam_p06.csv'),
                    low_memory=False)
    print('housing', h.shape, 'person', p.shape, flush=True)

    # housing file is all housing units; TYPEHUGQ==2 marks group quarters
    if 'TYPEHUGQ' in h.columns:
        h = h[h['TYPEHUGQ'] != 2].copy()
    # vacant housing units (NP==0) carry no person-level information and
    # break the group-size model; drop them before computing attributes
    n_vacant = int((h['NP'].astype(int) == 0).sum())
    h = h[h['NP'].astype(int) > 0].copy()
    print('dropped vacant units:', n_vacant, flush=True)

    # ---------- household attributes ----------
    hattrs = {}

    def cat(col, vals):
        """map column to 0..len(vals)-1 by value list; anything else -> NA bin"""
        s = h[col].fillna(NA)
        m = {v: i for i, v in enumerate(vals)}
        out = s.map(m).fillna(len(vals)).astype(int)
        return out.values, len(vals) + 1

    hattrs['ACR'], d = cat('ACR', [1, 2, 3]);             # lot size, NA bin
    hattrs['BLD'], d = cat('BLD', list(range(1, 11)));     # building size
    hattrs['HHT'], d = cat('HHT', list(range(1, 9)));      # household type
    hattrs['TEN'], d = cat('TEN', [1, 2, 3]);              # tenure
    veh = h['VEH'].fillna(7).astype(int).clip(0, 7)
    hattrs['VEH'] = veh.values                              # 0..7 (7=NA/8+)
    np_ = h['NP'].astype(int).clip(1, 10)
    hattrs['NPbin'] = np_.values - 1                        # 0..9 (9=10+)
    hincp = h['HINCP']                                       # NaN = missing
    valid = hincp.notna()
    qs = np.quantile(hincp[valid].astype(float), np.linspace(0, 1, 11))
    hincp_bin = np.full(len(h), 10, dtype=int)              # NA bin
    hincp_bin[valid.values] = np.searchsorted(
        qs[1:-1], hincp[valid].astype(float).values)         # 0..9
    hattrs['HINCPbin'] = hincp_bin

    # drop persons in dropped households (e.g. group quarters) BEFORE
    # computing attribute arrays
    h = h.reset_index(drop=True)
    serial_to_idx = {s: i for i, s in enumerate(h['SERIALNO'])}
    p = p[p['SERIALNO'].isin(serial_to_idx)].reset_index(drop=True)
    print('after join: households', len(h), 'persons', len(p), flush=True)

    # ---------- individual attributes ----------
    iattrs = {}
    age = p['AGEP'].astype(int)
    iattrs['AGEbin'] = (age // 10).clip(0, 9).values        # 0..9
    iattrs['SEX'] = (p['SEX'].astype(int) - 1).values       # 0/1
    schl = p['SCHL'].fillna(0).astype(int)
    schl_bin = np.zeros(len(p), dtype=int)
    schl_bin[(schl >= 1) & (schl <= 15)] = 0                # < HS
    schl_bin[schl == 16] = 1                                # HS
    schl_bin[schl == 17] = 2                                # some college
    schl_bin[schl == 18] = 3                                # associate
    schl_bin[schl == 19] = 4                                # bachelor
    schl_bin[schl == 20] = 5                                # master
    schl_bin[schl == 21] = 6                                # professional
    schl_bin[schl >= 22] = 7                                # doctorate
    schl_bin[schl == 0] = 8                                 # NA (age < 3)
    iattrs['SCHLbin'] = schl_bin
    esr = p['ESR'].fillna(7).astype(int).clip(1, 7)
    iattrs['ESR'] = esr.values - 1                          # 0..6 (6=NA)
    mar = p['MAR'].fillna(6).astype(int).clip(1, 6)
    iattrs['MAR'] = mar.values - 1                          # 0..5 (5=NA)
    # RELSHIPP (2022 vintage): 0 ref person, 1 spouse, 2-4 children,
    # 5 sibling, 6 parent, 7-10 other relatives, 11-15 nonrelatives,
    # 16-17 group quarters
    relp = p['RELSHIPP'].fillna(18).astype(int)
    relp_map = np.array([0, 1, 2, 2, 2, 3, 4, 5, 5, 5, 5, 6, 6, 1, 2, 6,
                         7, 7, 7])
    iattrs['RELPbin'] = relp_map[np.clip(relp.values, 0, 18)]
    wkhp = p['WKHP'].fillna(0).astype(int)
    wk_bin = np.digitize(wkhp.values, [1, 20, 35, 40, 49])  # 0..5
    iattrs['WKHPbin'] = wk_bin

    # ---------- assemble ----------
    os.makedirs(os.path.join(args.dir, 'processed'), exist_ok=True)

    h_names = list(hattrs)
    household = pd.DataFrame({'HHKEY': np.arange(len(h))})
    for a in h_names:
        household[a] = hattrs[a]
    household.to_csv(os.path.join(args.dir, 'processed', 'household.csv'),
                     index=False)

    i_names = list(iattrs)
    individual = pd.DataFrame({'PKEY': np.arange(len(p))})
    for a in i_names:
        individual[a] = iattrs[a]
    individual['HHKEY'] = p['SERIALNO'].map(serial_to_idx).astype(int)
    individual = individual.sort_values('HHKEY').reset_index(drop=True)
    individual['PKEY'] = np.arange(len(individual))
    individual.to_csv(os.path.join(args.dir, 'processed', 'individual.csv'),
                      index=False)

    h_domain = {a: {'size': int(household[a].max()) + 1} for a in h_names}
    i_domain = {a: {'size': int(individual[a].max()) + 1} for a in i_names}
    json.dump(h_domain, open(os.path.join(args.dir, 'processed',
                                          'household_domain.json'), 'w'))
    json.dump(i_domain, open(os.path.join(args.dir, 'processed',
                                          'individual_domain.json'), 'w'))

    sizes = individual.groupby('HHKEY').size()
    report = {
        'households': int(len(household)),
        'individuals': int(len(individual)),
        'persons_per_hh_mean': float(sizes.mean()),
        'persons_per_hh_max': int(sizes.max()),
        'household_domain': h_domain,
        'individual_domain': i_domain,
    }
    json.dump(report, open(os.path.join(args.dir, 'processed',
                                        'acs_report.json'), 'w'), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
