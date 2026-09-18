"""Resumable export of the ``trans`` table from the CTU relational MariaDB
(relational.fel.cvut.cz:3306, guest/ctu-relational, db `financial`) into
``<dir>/raw/trans.csv``.

Why this script exists: a full-table ``SELECT * FROM trans`` exceeds the
server-side 300 s statement limit, so `trans` is fetched in trans_id range
chunks. Each range is written to its own part file, which makes the export
resumable -- re-running skips every range whose part file already exists and
re-fetches only the missing ones. Two parallel connections are used; four
connections were dropped by the server mid-query ("Lost connection to MySQL
server during query"). Part files are concatenated into ``raw/trans.csv`` and
removed at the end.

This is the exporter that produced the paper's ``raw/trans.csv``
(1,056,320 rows, 19 ranges, 2 workers, 199.8 s); see ``schema_report`` in the
upstream preprocessing notes for the row-count cross-check against
``COUNT(*)`` in the database.

Run:
  python3 export_trans_resume.py [--dir <this dir>] [--lo 1] [--hi 3682987] \
      [--chunk 200000] [--workers 2]

  # write into the pipeline input tree instead of next to this script:
  python3 export_trans_resume.py --dir $CHAINNPM_ROOT/tmp/data/financial
"""
import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pymysql

HERE = os.path.dirname(os.path.abspath(__file__))

DB = dict(host='relational.fel.cvut.cz', port=3306, user='guest',
          password='ctu-relational', database='financial',
          connect_timeout=30, read_timeout=900)
COLS = ['trans_id', 'account_id', 'date', 'type', 'operation', 'amount',
        'balance', 'k_symbol', 'bank', 'account']
SEL = ("SELECT trans_id, account_id, date, type, operation, amount, "
       "balance, k_symbol, bank, account FROM trans "
       "WHERE trans_id BETWEEN {a} AND {b} ORDER BY trans_id")

# trans_id bounds of the CTU `financial` snapshot used in the paper.
LO, HI, CHUNK = 1, 3_682_987, 200_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=HERE,
                    help='dataset directory holding raw/ (default: this '
                         'script\'s directory)')
    ap.add_argument('--lo', type=int, default=LO,
                    help='lowest trans_id to fetch')
    ap.add_argument('--hi', type=int, default=HI,
                    help='highest trans_id to fetch')
    ap.add_argument('--chunk', type=int, default=CHUNK,
                    help='trans_id range width per query')
    ap.add_argument('--workers', type=int, default=2,
                    help='parallel connections (4 were dropped by the server)')
    args = ap.parse_args()

    raw = os.path.join(args.dir, 'raw')
    os.makedirs(raw, exist_ok=True)

    def part_path(a):
        return os.path.join(raw, 'trans_part_%08d.csv' % a)

    def fetch_range(a, b):
        p = part_path(a)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return a, 'skip'
        for attempt in range(3):
            try:
                conn = pymysql.connect(**DB)
                cur = conn.cursor()
                cur.execute(SEL.format(a=a, b=b))
                tmp = p + '.tmp'
                n = 0
                with open(tmp, 'w', newline='') as f:
                    w = csv.writer(f)
                    for r in cur.fetchall():
                        w.writerow(['' if v is None else v for v in r])
                        n += 1
                cur.close()
                conn.close()
                os.rename(tmp, p)
                return a, f'{n} rows'
            except Exception as e:
                print(f'  range {a}-{b} attempt {attempt+1} failed: {e}',
                      flush=True)
                time.sleep(5 * (attempt + 1))
        return a, 'FAILED'

    ranges = [(a, min(a + args.chunk - 1, args.hi))
              for a in range(args.lo, args.hi + 1, args.chunk)]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for a, status in ex.map(lambda r: fetch_range(*r), ranges):
            done += 1
            print(f'[{done}/{len(ranges)}] range {a}: {status}', flush=True)

    def missing_ranges():
        return [a for a, _ in ranges
                if not (os.path.exists(part_path(a))
                        and os.path.getsize(part_path(a)) > 0)]

    missing = missing_ranges()
    if missing:
        print('missing ranges, sequential retry:', missing, flush=True)
        for a in missing:
            _, status = fetch_range(a, min(a + args.chunk - 1, args.hi))
            print(f'  retry range {a}: {status}', flush=True)

    missing = missing_ranges()
    if missing:
        raise SystemExit(f'still missing: {missing}')

    out = os.path.join(raw, 'trans.csv')
    with open(out, 'w', newline='') as f:
        f.write(','.join(COLS) + '\n')
        total = 0
        for a, _ in ranges:
            with open(part_path(a)) as g:
                for line in g:
                    f.write(line)
                    total += 1
            os.remove(part_path(a))
    print('done', total, 'rows ->', out, 'in', round(time.time() - t0, 1),
          's')


if __name__ == '__main__':
    main()
