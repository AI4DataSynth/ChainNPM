"""Driver: Instacart Market Basket.  V-shape products <- order_products -> orders.

Chain-NPMV with cross-parent pair (DEPT, HOURBIN).  See p4_common.
Smoke runs use a 20% subset (--subset_frac 0.2) -- record the subset seed.

Example:
  python chain_npm_instacart.py --eps 3.2 --seed 42 --mode natural --method chainnpm \\
      --subset_frac 0.2 --subset_seed 42
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4_common as p4

CFG = dict(
    name='instacart',
    tables={
        'r1': dict(file='products.csv', pk='PKEY',
                   attrs=['DEPT', 'AISLE'],
                   domain_file='products_domain.json'),
        'r2': dict(file='orders.csv', pk='OKEY',
                   attrs=['DOW', 'HOURBIN', 'DPOBIN', 'ONUMBIN'],
                   domain_file='orders_domain.json'),
        'c': dict(file='order_products.csv', pk='OPKEY',
                  attrs=['REORDERED', 'ATCBIN'],
                  fk1='PKEY', fk2='OKEY',
                  domain_file='order_products_domain.json'),
    },
    xp_pairs=[('DEPT', 'HOURBIN')],
    effect=dict(x1=('r1', 'DEPT'), x2=('r2', 'HOURBIN'),
                a=('c', 'REORDERED')),
    plant=dict(x1='DEPT', x2='AISLE', u_attr='HOURBIN', a_attr='REORDERED'),
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
