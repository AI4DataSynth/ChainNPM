"""
Financial PKDD'99 (CTU relational repo, Berka schema) -> Chain-NPM
3-table chain deliverables.

Chain: district <- account <- trans
  district.csv : DKEY (dense PK), REGION, UNEMP96, SALARY
  account.csv  : AKEY (dense PK), FREQ, AYEAR, last col DKEY (FK)
  trans.csv    : TKEY (dense PK), TYEAR, TTYPE, TOPER, TAMOUNT,
                 last col AKEY (FK)
  + district_domain.json / account_domain.json / trans_domain.json
  + financial_report.json

Column note: in the CTU `financial` schema A2 is the district NAME and
A3 the region; unemployment rate is A12/A13 and average salary is A11
(the task sheet's "A2/A13" labels follow a different numbering). We
discretize A13 (unemployment rate 1996, binary at median), A11 (average
salary, tertiles) and A3 (region, 8 values).

Source: https://relational.fel.cvut.cz/dataset/Financial (guest account
on public MariaDB; Berka 2000, PKDD'99 Discovery Challenge).
Licensing: neither the CTU repository pages nor the archived PKDD'99 call for
contributions state a license for this dataset, so its redistribution status
is unclear -- do not redistribute the raw or processed files without checking
first. See ../README.md ("Data licensing / redistribution").
Deterministic; SEED=42 (used for NA-bin tie-handling only).

Run: python3 prep_financial.py [--dir <this dir>]
"""
import argparse, json, os
import numpy as np
import pandas as pd

SEED = 42


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
    rng = np.random.default_rng(SEED)
    raw = os.path.join(args.dir, 'raw')
    out = os.path.join(args.dir, 'processed')
    os.makedirs(out, exist_ok=True)

    d = pd.read_csv(os.path.join(raw, 'district.csv'), dtype={'A2': str},
                    keep_default_na=False)
    a = pd.read_csv(os.path.join(raw, 'account.csv'),
                    keep_default_na=False)
    t = pd.read_csv(os.path.join(raw, 'trans.csv'),
                    keep_default_na=False)
    print('raw:', d.shape, a.shape, t.shape, flush=True)

    # ---------------- district ----------------
    d = d.sort_values('district_id').reset_index(drop=True)
    dkey = {int(v): i for i, v in enumerate(d['district_id'])}
    regions = sorted(d['A3'].unique())
    region = d['A3'].map({v: i for i, v in enumerate(regions)}).astype(int).values
    a13 = d['A13'].astype(float)
    med = float(a13.median())
    unemp = (a13 > med).astype(int).values          # binary at median
    a11 = d['A11'].astype(float)
    q1, q2 = np.quantile(a11, [1/3, 2/3])
    salary = np.digitize(a11.values, [q1, q2]).astype(int)   # 0/1/2 tertiles
    district_df = pd.DataFrame({'DKEY': np.arange(len(d)),
                                'REGION': region, 'UNEMP96': unemp,
                                'SALARY': salary})

    # ---------------- account ----------------
    a = a.sort_values('account_id').reset_index(drop=True)
    akey = {int(v): i for i, v in enumerate(a['account_id'])}
    freqs = sorted(a['frequency'].unique())
    freq = a['frequency'].map({v: i for i, v in enumerate(freqs)}).astype(int).values
    ayear = a['date'].str[:4].astype(int)
    ayear_bin = (ayear - int(ayear.min())).values
    a_dk = a['district_id'].map(dkey)
    assert a_dk.notna().all()
    a_dk = a_dk.astype(int).values
    account_df = pd.DataFrame({'AKEY': np.arange(len(a)), 'FREQ': freq,
                               'AYEAR': ayear_bin, 'DKEY': a_dk})

    # ---------------- trans ----------------
    t = t.sort_values('trans_id').reset_index(drop=True)
    tyear = t['date'].str[:4].astype(int)
    tyear_bin = (tyear - 1993).values                        # 0..5
    types = sorted(t['type'].unique())
    ttype = t['type'].map({v: i for i, v in enumerate(types)}).astype(int).values
    ops = sorted(t.loc[t['operation'] != '', 'operation'].unique())
    op_map = {v: i for i, v in enumerate(ops)}
    toper = t['operation'].map(op_map).fillna(len(ops)).astype(int).values  # NA bin
    amount = t['amount'].astype(float).values
    qs = np.quantile(amount, np.linspace(0, 1, 11)[1:-1])
    tamount = np.searchsorted(qs, amount, side='right').astype(int)   # 0..9
    t_ak = t['account_id'].map(akey)
    assert t_ak.notna().all()
    t_ak = t_ak.astype(int).values
    trans_df = pd.DataFrame({'TKEY': np.arange(len(t)), 'TYEAR': tyear_bin,
                             'TTYPE': ttype, 'TOPER': toper,
                             'TAMOUNT': tamount, 'AKEY': t_ak})

    # ---------------- write ----------------
    district_df.to_csv(os.path.join(out, 'district.csv'), index=False)
    account_df.to_csv(os.path.join(out, 'account.csv'), index=False)
    trans_df.to_csv(os.path.join(out, 'trans.csv'), index=False)
    d_dom = {'REGION': {'size': len(regions)}, 'UNEMP96': {'size': 2},
             'SALARY': {'size': 3}}
    a_dom = {'FREQ': {'size': len(freqs)},
             'AYEAR': {'size': int(account_df['AYEAR'].max()) + 1}}
    t_dom = {'TYEAR': {'size': 6}, 'TTYPE': {'size': len(types)},
             'TOPER': {'size': len(ops) + 1},
             'TAMOUNT': {'size': int(trans_df['TAMOUNT'].max()) + 1}}
    json.dump(d_dom, open(os.path.join(out, 'district_domain.json'), 'w'))
    json.dump(a_dom, open(os.path.join(out, 'account_domain.json'), 'w'))
    json.dump(t_dom, open(os.path.join(out, 'trans_domain.json'), 'w'))

    # ---------------- stats ----------------
    per_a = trans_df.groupby('AKEY').size()
    per_d = account_df.groupby('DKEY').size()
    # two-hop cue: top-decile transaction share by salary tertile
    sal_of_acct = district_df['SALARY'].values[account_df['DKEY'].values]
    sal_of_trans = sal_of_acct[trans_df['AKEY'].values]
    top = (trans_df['TAMOUNT'].values == 9)
    cue = {'top_amount_share_salary_tertile0':
               round(float(top[sal_of_trans == 0].mean()), 4),
           'top_amount_share_salary_tertile2':
               round(float(top[sal_of_trans == 2].mean()), 4)}
    stats = {
        'districts': int(len(district_df)),
        'accounts': int(len(account_df)),
        'transactions': int(len(trans_df)),
        'trans_per_account': {'median': float(per_a.median()),
                              'max': int(per_a.max()),
                              'gini': round(gini(per_a.values), 4)},
        'accounts_per_district': {'median': float(per_d.median()),
                                  'max': int(per_d.max()),
                                  'gini': round(gini(per_d.values), 4)},
        'freq_values': freqs, 'type_values': types,
        'operation_values': ops, 'regions': regions,
        'unemp96_median': med,
        'salary_tertile_edges': [float(q1), float(q2)],
        'amount_decile_edges': [float(x) for x in qs],
        'cross_hop_cue': cue,
    }
    json.dump(stats, open(os.path.join(out, 'financial_report.json'), 'w'),
              indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
