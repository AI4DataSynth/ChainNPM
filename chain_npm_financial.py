"""Driver: financial (PKDD'99 CTU).  Chain  district <- account <- trans.

Chain-NPM3 (top=district, mid=account, bot=trans).  See p4_common for the
shared protocol (query workload, effect metrics, recording).

Example:
  python chain_npm_financial.py --eps 3.2 --seed 42 --mode natural --method chainnpm
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4_common as p4

CFG = dict(
    name='financial',
    tables={
        'top': dict(file='district.csv', pk='DKEY',
                    attrs=['REGION', 'UNEMP96', 'SALARY'],
                    domain_file='district_domain.json'),
        'mid': dict(file='account.csv', pk='AKEY',
                    attrs=['FREQ', 'AYEAR'], fk='DKEY',
                    domain_file='account_domain.json'),
        'bot': dict(file='trans.csv', pk='TKEY',
                    attrs=['TYEAR', 'TTYPE', 'TOPER', 'TAMOUNT'], fk='AKEY',
                    domain_file='trans_domain.json'),
    },
    effect=dict(x1=('top', 'SALARY'), a=('bot', 'TAMOUNT')),
    plant=dict(x1='SALARY', x2='UNEMP96', a_attr='TAMOUNT',
               u_attr='TTYPE', mid_attr='FREQ'),
)


def main():
    args = p4.build_arg_parser().parse_args()
    result = p4.run_chain_pipeline(CFG, args)
    name = '%s_%s_eps%.2f_seed%d.json' % (args.method, args.mode,
                                          args.eps, args.seed)
    p4.dump(result, CFG, args, name)
    if result.get('re'):
        print(json.dumps({'re_median_large': result['re']['re_median_large'],
                          'effect': {k: v for k, v in result['effect'].items()},
                          'size_tv': result.get('size_tv'),
                          'runtime_sec': round(result['runtime_sec'], 1)},
                         indent=2))


if __name__ == '__main__':
    main()
