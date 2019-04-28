import argparse
from utils.distributions import RandInt, Uniform
from functions.mnist import MLPWithMNIST
import numpy as np
import os
import datetime
from hyperband import Hyperband
from utils import plot_util
import time
import pandas as pd


def get_path_with_time(alg_name):
    time_name = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    path = 'log/' + alg_name + '/' + time_name
    return path


def get_param_with_bench(bench):
    params = {}

    if bench == 'MLPWithMNIST':
        # hyperparameters
        params['hparams'] = {
            'lr': Uniform(0.001, 0.30),
            'momentum': Uniform(0.50, 0.999),
            'fc1_unit': RandInt(30, 1000),
            'fc2_unit': RandInt(30, 1000)
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
                        help='the benchmark function you want to run',
                        metavar=None)
    parser.add_argument('--max_iter',
                        type=int,
                        default=27,
                        help='maximum amount of resource that can be allocated to a single configuration')
    parser.add_argument('--eta',
                        type=int,
                        default=3,
                        help='proportion of the configurations discarded in each round of SH')
    parser.add_argument('--patience',
                        type=int,
                        default=5,
                        help='threshold for original early-stopping')
    parser.add_argument('--gcp',
                        action='store_true')

    args = parser.parse_args()
    params = get_param_with_bench(args.bench)
    params['max_iter'] = args.max_iter
    params['eta'] = args.eta
    params['patience'] = args.patience
    params['homedir'] = '/hyperband_sandbox/' if args.gcp else './'

    # run optimization
    hb = Hyperband(**params)
    best = hb.run()
    print("best:{}".format(best))

    separate_history = hb.separate_history
    print("separate_history:{}".format(separate_history))
    i = 0
    for k, v in separate_history.items():
        df = pd.DataFrame(v)
        df.to_csv("./log_{}.csv".format(i))
        i += 1

    plot_util.plot_separately(separate_history, homedir=params['homedir'])


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("elapsed_time:{}".format(elapsed_time) + "[sec]")
