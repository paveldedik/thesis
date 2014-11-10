# -*- coding: utf-8 -*-


import sys

import numpy as np
import pandas as pd
from scipy import optimize
from IPython.display import clear_output
from sklearn.metrics import mean_squared_error


def echo(msg, clear=True):
    """Prints given message."""
    if clear:
        clear_output()
    print msg
    sys.stdout.flush()


def load(path):
    """Loads data into :class:`pandas.DataFrame`."""
    return pd.read_csv(path)


def sigmoid(x):
    """Sigmoid function implementation."""
    return 1 / (1 + np.exp(-x))


def shuffle(data):
    """Shuffles given set of data."""
    items = np.random.permutation(data.index)
    return data.reindex(items)


def take_random(data, n=1000):
    """Randomly selects rows from given data set."""
    items = np.random.permutation(data.index)[:n]
    return data.ix[items]


def add_ones(data):
    """Adds column with ones as the first column of the given data set."""
    columns = list(data.columns)
    data['ones'] = np.ones(len(data))
    data = data[['ones'] + columns]
    return data


def rmse(y_true, y_pred):
    """Calculates Root Mean Squered Error for vectors
    *y_pred* and *y_true*."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def cost(theta, data, y):
    """Cost function implementation."""
    hyph = sigmoid(np.dot(data, theta))
    calc = (-1 * y) * np.log(hyph) - (1 - y) * np.log(1 - hyph)
    return sum(calc) / len(y)


def grad(theta, data, y):
    """Returns gradient for given data set."""
    hyph = sigmoid(np.dot(data, theta))
    return np.dot(hyph - y, data) / y.size


def find_theta(theta, data, y, maxiter=1000):
    """Runs BFGS algoritm for minimizing cost function on given data set."""
    return optimize.fmin_bfgs(
        cost, theta, fprime=grad, args=(data, y), maxiter=maxiter
    )


def decay_rate(strength, c=0.2, a=0.18):
    """Calculates decay rate for given memory strength."""
    return c * np.exp(strength) + a


def memory_strength(practices, spacing_rate=0.2, decay_rate=0.18):
    """Calculates memory strength based on given vector of practices.
    Each element *t_i* in the vector indicates how long ago the *i*-th
    practice occured.
    """
    decays = []
    strengths = [-np.inf]

    get_decay = lambda x: decay_rate(x, c=spacing_rate, a=decay_rate)

    for i in range(len(practices)):
        decays.append(get_decay(strengths[i]))
        rates = practices[:i+1] ** -np.array(decays)
        strengths.append(np.log(sum(rates)))

    return strengths[-1]


def retrieval_prob(strength, tau=-0.704, s=0.255):
    """Returns the probability of item retrieval based on the *strength*
    of the memory.
    """
    val = (tau - strength) / s
    return 1 / (1 + np.exp(val))


def timing(x):
    """Calculates probability of correct answer based on response time."""
    result = (4.07446031e-03 * x -
              1.18475468e+00 * x ** (1 / 2) -
              1.01545130e-05 * x ** (3 / 2) +
              4.68002306e+00 * x ** (1 / 3))
    return max(result - 12.28, 0)


def get_prior(data):
    """Returns only prior answers for given data set."""
    data = data.sort(
        ['user', 'place_asked', 'inserted']
    ).groupby(
        ['user', 'place_asked']
    ).first()
    return data.reset_index()


def split_data(data, ratio=0.7):
    """Splits data into cross validation set and training set."""
    data_shuffeled = shuffle(data)
    threshold = int(len(data_shuffeled) * ratio)
    train_set = data_shuffeled[:threshold]
    valid_set = data_shuffeled[threshold:]
    return train_set, valid_set


def prepare_data(data):
    """Prepares data for models (e.g. addes *correct* column."""
    data['correct'] = (data['place_asked'] ==
                       data['place_answered']).astype(int)
    data['prediction'] = np.nan
    return data


class cached_property(object):
    """A decorator that converts a method into a property. The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::

        class Person(object):

            @cached_property
            def stats(self):
                # calculate something important here
                return Stats(self.domain)

    The class has to have a `__dict__` in order for this property to
    work.
    """

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__)
        if value is None:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value
