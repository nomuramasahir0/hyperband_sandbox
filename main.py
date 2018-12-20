import argparse
from utils.distributions import RandInt, Uniform
from functions.mnist import MLPWithMNIST
import numpy as np
import os
import datetime
from hyperband import Hyperband


def get_path_with_time(alg_name):
    time_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    path = 'log/' + alg_name + '/' + time_name
    return path


def get_param_with_bench(bench):
    params = {}

    if bench == 'MLPWithMNIST':
        # maximum iterations/epochs per configuration
        params['max_iter'] = 27
        # downsampling rate
        params['eta'] = 3
        # hyperparameters
        params['hparams'] = {
            'lr': Uniform(0.001, 0.20),
            'momentum': Uniform(0.80, 0.999),
            'fc1_unit': RandInt(50, 500)
        }
        params['obj_func'] = MLPWithMNIST

    params['seed'] = np.random.randint(0, 2 ** 32 - 1)
    params['path'] = get_path_with_time('random_search')
    if not os.path.isdir(params['path']):
        os.makedirs(params['path'])
    print('create directory which is ' + params['path'])
    return params


def main():
    parser = argparse.ArgumentParser(description='Hyperband main script')
    parser.add_argument('bench',
                        action='store',
                        nargs=None,
                        const=None,
                        default=None,
                        type=str,
                        choices=['MLPWithMNIST'],
                        help='specify the benchmark function you want to run',
                        metavar=None)
    parser.add_argument('--gcp',
                        action='store_true')

    args = parser.parse_args()
    params = get_param_with_bench(args.bench)

    # run optimization
    hb = Hyperband(**params)
    hb.run()


if __name__ == '__main__':
    main()
