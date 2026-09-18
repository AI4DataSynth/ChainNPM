"""Driver: IMDb (official snapshot subset).  V-shape names <- principals -> titles.

Chain-NPMV with cross-parent pair (BYEARBIN, TITLETYPE).  See p4_common.
Smoke runs use a 20% subset (--subset_frac 0.2) -- record the subset seed.

Example:
  python chain_npm_imdb.py --eps 3.2 --seed 42 --mode natural --method chainnpm \\
      --subset_frac 0.2 --subset_seed 42
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4_common as p4

CFG = dict(
    name='imdb',
    tables={
        'r1': dict(file='names.csv', pk='NKEY',
                   attrs=['ALIVE', 'BYEARBIN', 'PROF1'],
                   domain_file='names_domain.json'),
        'r2': dict(file='titles.csv', pk='TKEY',
                   attrs=['TITLETYPE', 'GENRE1', 'YEARBIN', 'ISADULT', 'RTBIN'],
                   domain_file='titles_domain.json'),
        'c': dict(file='principals.csv', pk='PKEY', attrs=['CATEGORY'],
                  fk1='NKEY', fk2='TKEY',
                  domain_file='principals_domain.json'),
    },
    xp_pairs=[('BYEARBIN', 'TITLETYPE')],
    effect=dict(x1=('r1', 'BYEARBIN'), x2=('r2', 'TITLETYPE'),
                a=('c', 'CATEGORY')),
    plant=dict(x1='BYEARBIN', x2='ALIVE', u_attr='TITLETYPE',
               a_attr='CATEGORY'),
)


def main():
    args = p4.build_arg_parser().parse_args()
    result = p4.run_vshape_pipeline(CFG, args)
    name = '%s_%s_eps%.2f_seed%d%s.json' % (
        args.method, args.mode, args.eps, args.seed,
        ('_sub%.2f' % args.subset_frac) if args.subset_frac < 1.0 else '')
    p4.dump(result, CFG, args, name)
    if result.get('re'):
        eff = result['effect']
        print(json.dumps({'re_median_large': result['re']['re_median_large'],
                          'contrast': {'real': eff['contrast_real']['contrast'],
                                       'syn': eff['contrast_syn']['contrast']},
                          'size_tv': result.get('size_tv'),
                          'runtime_sec': round(result['runtime_sec'], 1)},
                         indent=2))


if __name__ == '__main__':
    main()
