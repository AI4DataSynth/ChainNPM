"""Export small financial PKDD'99 tables (district, account) from the CTU
relational MariaDB (relational.fel.cvut.cz:3306, guest/ctu-relational, db
`financial`) into ``<dir>/raw/<table>.csv``. Acquisition script; idempotent
(each table is fully rewritten).

Note: the default table list also names ``trans``, but a full-table
``SELECT * FROM trans`` exceeds the server-side 300 s statement limit and will
be cut off. Export ``trans`` with ``export_trans_resume.py`` instead, which
chunks the scan by ``trans_id``:

  python3 export_financial.py --dir <DATA_ROOT>/financial district account
  python3 export_trans_resume.py --dir <DATA_ROOT>/financial

Run:
  python3 export_financial.py [--dir <this dir>] [table ...]
"""
import argparse
import csv
import os

import pymysql

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('tables', nargs='*', default=None,
                help='tables to export (default: district account trans)')
ap.add_argument('--dir', default=HERE,
                help='dataset directory holding raw/ (default: this '
                     'script\'s directory)')
args = ap.parse_args()

RAW = os.path.join(args.dir, 'raw')
os.makedirs(RAW, exist_ok=True)

conn = pymysql.connect(host='relational.fel.cvut.cz', port=3306,
                       user='guest', password='ctu-relational',
                       database='financial', connect_timeout=30,
                       read_timeout=600,
                       cursorclass=pymysql.cursors.SSCursor)

for table in (args.tables or ["district", "account", "trans"]):
    cur = conn.cursor()
    cur.execute(f"SHOW COLUMNS FROM `{table}`")
    cols = [r[0] for r in cur.fetchall()]
    cur.execute(f"SELECT * FROM `{table}`")
    path = os.path.join(RAW, f'{table}.csv')
    n = 0
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        while True:
            rows = cur.fetchmany(5000)
            if not rows:
                break
            for r in rows:
                w.writerow(['' if v is None else v for v in r])
                n += 1
    cur.close()
    print(f'{table}: {n} rows -> {path}', flush=True)

conn.close()
