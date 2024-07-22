"""Example of training a policy with EPO_PMN algorithm."""

import argparse

import epo_pmn
from epo_pmn.utils.tools import custom_cfgs_to_dict, update_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        metavar='ALGO',
        default='EPO_PMN',
        help='algorithm to train',
        choices=epo_pmn.ALGORITHMS['all'],
    )
    parser.add_argument(
        '--env-id',
        type=str,
        metavar='ENV',
        default='SafetyPointGoal1-v0',
        help='the name of test environment',
    )
    parser.add_argument(
        '--parallel',
        default=1,
        type=int,
        metavar='N',
        help='number of paralleled progress for calculations.',
    )
    parser.add_argument(
        '--total-steps',
        type=int,
        default=10240000,
        metavar='STEPS',
        help='total number of steps to train for algorithm',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        metavar='DEVICES',
        help='device to use for training',
    )
    parser.add_argument(
        '--vector-env-nums',
        type=int,
        default=32,
        metavar='VECTOR-ENV',
        help='number of vector envs to use for training',
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=16,
        metavar='THREADS',
        help='number of threads to use for torch',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='SEED',
        help='seed for random number generator',
    )
    parser.add_argument(
        '--cost-limit',
        type=int,
        default=25,
        metavar='COST-LIMIT',
        help='Tolerance of constraint violation',
    )
    parser.add_argument(
        '--penalty-factor',
        type=float,
        default=0.1,
        metavar='PENALTY-FACTOR',
        help='penalty factor',
    )
    parser.add_argument(
        '--type-penalty',
        type=str,
        default='mix',  # quadratic，linear
        metavar='TYPE-PENALTY',
        help='the type of penalty function',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default='0.5',
        metavar='ALPHA',
        help='the weight of linear and quadratic penalty term',
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./debug_results',
        metavar='LOG-DIR',
        help='save logger path',
    )
    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    var_args = vars(args)
    if var_args['algo'] in {'EPO_PMN'}:
        custom_cfgs = {
            'seed': var_args['seed'],
            'algo_cfgs': {
                'update_cycle': 32000,
                'update_iters': 40,
                'cost_limit': var_args['cost_limit'],
                'penalty_factor': var_args['penalty_factor'],
                'type_penalty': var_args['type_penalty'],
                'alpha': var_args['alpha'],
            },
            'logger_cfgs': {
                'log_dir': var_args['log_dir'],
            },
        }

    else:
        custom_cfgs = {
            'seed': var_args['seed'],
            'algo_cfgs': {
                'update_cycle': 32000,
                'update_iters': 40,
                'cost_limit': var_args['cost_limit'],
            },
            'logger_cfgs': {
                'log_dir': var_args['log_dir'],
            },
        }
    train_cfgs = {
        'device': var_args['device'],
        'torch_threads': var_args['torch_threads'],
        'vector_env_nums': var_args['vector_env_nums'],
        'parallel': var_args['parallel'],
        'total_steps': var_args['total_steps'],
    }
    for k, v in unparsed_args.items():
        update_dic(custom_cfgs, custom_cfgs_to_dict(k, v))

    agent = epo_pmn.Agent(
        args.algo,
        args.env_id,
        train_terminal_cfgs=train_cfgs,
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
