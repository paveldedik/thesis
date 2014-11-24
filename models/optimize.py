# -*- coding: utf-8 -*-

"""
Optimization Methods
====================

"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize

from . import tools
from .tests import PFATest, EloTest


class GridResult(object):
    """Represents grid search result."""

    def __init__(self, grid, extent, minimum, xlabel=None, ylabel=None):
        self.grid = grid
        self.extent = extent
        self.minimum = minimum
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self):
        """Plots the result of grid search.
        Uses :func:`~matplotlib.pyplot.imshow` to plot the data.
        """
        plot = plt.imshow(self.grid,
                          cmap=cm.Greys_r,
                          interpolation='nearest',
                          extent=self.extent,
                          aspect='auto')
        plt.colorbar(plot)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()
        return plot

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        grid = self.grid.round(3)
        minimum = self.minimum.round(3)
        return ('Minimum:\n {}\n'
                'Grid:\n {}').format(minimum, grid)


class GridSearch(object):
    """Encapsulates grid searches for various models.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

        self.elo_result = None
        self.pfa_result = None

    def search_elo(self, alphas, betas):
        """Performes grid search on ELO model using given parameters.

        :param alphas: Alpha parameters (see :class:`EloModel`).
        :param betas: Beta paramters (see :class:`EloModel`).
        """
        minimum = (0, 0)
        test = EloTest(self.data, train=False)

        m, n = len(alphas), len(betas)
        grid = np.matrix([[0] * m] * n, dtype=float)

        for x, y in itertools.product(range(m), range(n)):
            kwargs = {'alpha': alphas[x], 'beta': betas[y]}
            test.retrain_model(model_kwargs=kwargs)
            grid[y, x] = test.rmse()

            minimum = (y, x) if grid[minimum] > grid[y, x] else minimum
            tools.echo('ELO: {}/{} {}/{}'.format(x+1, n, y+1, n))

        self.elo_result = GridResult(
            grid=grid,
            extent=np.array([
                min(alphas), max(alphas),
                min(betas), max(betas),
            ]),
            minimum=np.array([alphas[minimum[0]], betas[minimum[1]]]),
            xlabel='alpha',
            ylabel='beta',
        )

        return self.elo_result

    def search_pfa(self, gammas, deltas):
        """Performes grid search on ELO model using given parameters.

        :param gammas: Gamma parameters (see :class:`PFAModel`).
        :param deltas: Delta paramters (see :class:`PFAModel`).
        """
        minimum = (0, 0)
        test = PFATest(self.data, train=False)

        m, n = len(gammas), len(deltas)
        grid = np.matrix([[0] * m] * n, dtype=float)

        for x, y in itertools.product(range(m), range(n)):
            kwargs = {'gamma': gammas[x], 'delta': deltas[y]}
            test.retrain_model(model_kwargs=kwargs)
            grid[y, x] = test.rmse()

            minimum = (y, x) if grid[minimum] > grid[y, x] else minimum
            tools.echo('PFA: {}/{} {}/{}'.format(x+1, m, y+1, n))

        self.pfa_result = GridResult(
            grid=grid,
            extent=np.array([
                min(gammas), max(gammas),
                max(deltas), min(deltas),
            ]),
            minimum=np.array([gammas[minimum[0]], deltas[minimum[1]]]),
            xlabel='gamma',
            ylabel='delta',
        )

        return self.pfa_result

    def plot_elo(self):
        """Plots the result of ELO model grid search."""
        if self.elo_result is None:
            raise RuntimeError('Run GridSearch.elo_search first.')
        return self.elo_result.plot()

    def plot_pfa(self):
        """Plots the result of PFA model grid search."""
        if self.pfa_result is None:
            raise RuntimeError('Run GridSearch.pfa_search first.')
        return self.pfa_result.plot()


class RandomSearch(object):
    """Encapsulates grid searches for various models.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

        self.elo_result = None
        self.pfa_result = None

    def search_elo(self, alpha, beta):
        """Performes random search on ELO model using given initial
        parameters.

        :param alpha: Initial alpha value (see :class:`EloModel`).
        :param beta: Initial beta value (see :class:`EloModel`).
        """
        test = EloTest(self.data, train=False)

        def fun(x):
            kwargs = {'alpha': x[0], 'beta': x[1]}
            test.retrain_model(model_kwargs=kwargs)

            tools.echo('alpha={x[0]} beta={x[1]}'.format(x=x))
            return test.rmse()

        self.elo_result = optimize.minimize(fun, [alpha, beta])
        return self.elo_result

    def search_pfa(self, gamma, delta):
        """Performes random search on ELO model using given initial
        parameters.

        :param gamma: Initial gamma value (see :class:`PFAModel`).
        :param delta: Initial delta value (see :class:`PFAModel`).
        """
        test = PFATest(self.data, train=False)

        def fun(x):
            kwargs = {'gamma': x[0], 'delta': x[1]}
            test.retrain_model(model_kwargs=kwargs)

            tools.echo('gamma={x[0]} delta={x[1]}'.format(x=x))
            return test.rmse()

        self.pfa_result = optimize.minimize(fun, [gamma, delta])
        return self.pfa_result
