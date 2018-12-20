import numpy as np


class Uniform(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.uniform(self.low, self.high)


class RandInt(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.randint(self.low, self.high)


class Choice(object):
    def __init__(self, candidates):
        self.candidates = candidates

    def sample(self):
        return self.candidates[np.random.randint(len(self.candidates))]


class PUniform(object):
    def __init__(self, low, high, base=10):
        self.low = low
        self.high = high
        self.base = base

    def sample(self):
        return np.power(self.base, np.random.uniform(self.low, self.high))
