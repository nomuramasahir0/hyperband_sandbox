import math
import numpy as np
import uuid
import gc


class Hyperband:
    def __init__(self, **params):
        # maximum iterations/epochs per configuration
        self.max_iter = params['max_iter']
        # downsampling rate
        self.eta = params['eta']
        self.logeta = lambda x: math.log(x) / math.log(self.eta)
        # number of unique executions of Successive Halving (minus one)
        self.s_max = int(self.logeta(self.max_iter))
        # total number of iterations (without reuse) per execution of Successive Halving (n, r)
        self.B = (self.s_max + 1) * self.max_iter
        # hyperparameters
        self.hparams = params['hparams']
        # objective function
        self.obj_func = params['obj_func']
        # history (separate)
        self.separate_history = {}

    def random_sampling(self):
        ps = {}
        for k, v in self.hparams.items():
            ps[k] = v.sample()
        return ps

    def run(self):
        # Begin Finite Horizon Hyperband outerloop. Repeat indefinitely.
        # hedge strategy
        print("outer loop start:")
        for s in reversed(range(self.s_max + 1)):
            print("s:{}".format(s))
            # initial number of configuration
            n = int(math.ceil(int(self.B / self.max_iter / (s + 1)) * self.eta ** s))
            print("n:{}".format(n))
            # initial numnber of iterations to run configuration
            r = self.max_iter * self.eta ** (-s)
            print("r:{}".format(r))

            # Begin Finite Horizon Successive Halving with (n, r)
            # early-stopping procedure
            hparams = [self.random_sampling() for _ in range(n)]
            print("hparams:{}".format(hparams))
            # obj_funcには，initする前のclassだけ渡しておく
            obj_names = [str(uuid.uuid4().hex) for _ in range(n)]
            # history logging
            for obj_name in obj_names:
                self.separate_history[obj_name] = []

            print("inner loop start")
            for i in range(s + 1):
                print("i:{}".format(i))
                # Run each of the n_i configs for r_i iterations and keep best n_i / eta
                n_i = n * self.eta ** (-i)
                print("n_i:{}".format(n_i))
                r_i = r * self.eta ** i
                print("r_i:{}".format(r_i))

                # model
                val_losses = []
                for (hparam, obj_name) in zip(hparams, obj_names):
                    obj = self.obj_func(hparam, obj_name, self.separate_history)
                    val_loss = obj.evaluate(num_iter=int(r_i))
                    # val_loss = np.random.rand()
                    val_losses.append(val_loss)
                    del obj
                    gc.collect()

                arg_val_losses = np.argsort(val_losses)
                hparams = [hparams[j] for j in arg_val_losses[0:int(n_i / self.eta)]]
                obj_names = [obj_names[j] for j in arg_val_losses[0:int(n_i / self.eta)]]
                print('replace finished.')
                print("hparams:{}".format(hparams))
        # End Finite Horizon Successive Halving
