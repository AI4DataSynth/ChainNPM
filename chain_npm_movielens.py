"""Driver: MovieLens-1M.  V-shape  users <- ratings -> movies.

Chain-NPMV with cross-parent pair (AGE, GENRE1).  See p4_common.

Example:
  python chain_npm_movielens.py --eps 3.2 --seed 42 --mode natural --method chainnpm
  python chain_npm_movielens.py --eps 3.2 --seed 42 --mode plant_xor --target 0.8
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4_common as p4

CFG = dict(
    name='movielens',
    tables={
        'r1': dict(file='users.csv', pk='UKEY',
                   attrs=['GENDER', 'AGE', 'OCC'],
                   domain_file='users_domain.json'),
        'r2': dict(file='movies.csv', pk='MKEY',
                   attrs=['GENRE1', 'NGEN'],
                   domain_file='movies_domain.json'),
        'c': dict(file='ratings.csv', pk='RKEY', attrs=['RATING'],
                  fk1='UKEY', fk2='MKEY',
                  domain_file='ratings_domain.json'),
    },
    xp_pairs=[('AGE', 'GENRE1')],
    effect=dict(x1=('r1', 'AGE'), x2=('r2', 'GENRE1'), a=('c', 'RATING')),
    plant=dict(x1='AGE', x2='GENDER', u_attr='GENRE1', a_attr='RATING'),
)


def main():
    args = p4.build_arg_parser().parse_args()
    result = p4.run_vshape_pipeline(CFG, args)
    name = '%s_%s_eps%.2f_seed%d.json' % (args.method, args.mode,
                                          args.eps, args.seed)
    p4.dump(result, CFG, args, name)
    if result.get('re'):
        eff = result['effect']
        print(json.dumps({'re_median_large': result['re']['re_median_large'],
                          'contrast': {'real': eff['contrast_real']['contrast'],
                                       'syn': eff['contrast_syn']['contrast']},
                          'phi': ({'real': eff['phi_real']['phi_xor'],
                                   'syn': eff['phi_syn']['phi_xor']}
                                  if 'phi_real' in eff else None),
                          'size_tv': result.get('size_tv'),
                          'runtime_sec': round(result['runtime_sec'], 1)},
                         indent=2))


if __name__ == '__main__':
    main()
