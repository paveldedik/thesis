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
from .tests import PerformanceTest
from .models import EloModel, PFAModel, PFAWithSpacing


class GridResult(object):
    """Represents a GRID search result.

    :param grid: A matrix representing the results of the search.
    :type grid: :class:`numpy.matrix`
    :param xlabel: Name of the x-axis.
    :type xlabel: str
    :param ylavel: Name of the y-axis.
    :type ylabel: str
    :param xvalues: Values on the x-axis.
    :type xvalues: list
    :param yvalues: Values on the y-axis.
    :type yvalues: list
    """

    def __init__(self, grid,
                 xlabel=None, ylabel=None,
                 xvalues=None, yvalues=None):
        self.grid = grid
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xvalues = xvalues
        self.yvalues = yvalues

        self.extent = np.array([
            min(self.xvalues), max(self.xvalues),
            max(self.yvalues), min(self.yvalues),
        ]),

    @tools.cached_property
    def rmse(self):
        """Grid Search errors estimations using RMSE."""
        return np.array([
            [result.rmse().value for result in row]
            for row in self.grid
        ])

    @tools.cached_property
    def auc(self):
        """Grid Search errors estimations using AUC."""
        return np.array([
            [result.auc().value for result in row]
            for row in self.grid
        ])

    @tools.cached_property
    def rmse_min(self):
        """Values of `xvalues` and `yvalues` with best RMSE."""
        minimum = np.unravel_index(self.rmse.argmax(), self.rmse.shape)
        return np.array([self.xvalues[minimum[0]], self.yvalues[minimum[1]]])

    @tools.cached_property
    def auc_min(self):
        """Values of `xvalues` and `yvalues` with best AUC."""
        minimum = np.unravel_index(self.auc.argmax(), self.auc.shape)
        return np.array([self.xvalues[minimum[0]], self.yvalues[minimum[1]]])

    def plot(self, metric='RMSE'):
        """Plots the result of the GRID search.
        Uses :func:`~matplotlib.pyplot.imshow` to plot the data.
        """
        grids = {
            'RMSE': self.rmse,
            'AUC': self.auc,
        }
        plot = plt.imshow(grids[metric],
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
        return (
            'RMSE:\n min: {0}\n {1}'
            '\n\n'
            'AUC:\n min: {2}\n {3}'
        ).format(
            self.rmse_min.round(3),
            self.rmse.round(3),
            self.auc_min.round(3),
            self.auc.round(3),
        )


class GridSearch(object):
    """Encapsulates GRID searches for various models.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

    def search_elo(self, alphas, betas):
        """Performes grid search on ELO model using given parameters.

        :param alphas: Alpha parameters (see :class:`EloModel`).
        :type alphas: list or :class:`numpy.array`
        :param betas: Beta paramters (see :class:`EloModel`).
        :type betas: list or :class:`numpy.array`
        """
        m, n = len(alphas), len(betas)
        grid = [[None] * m] * n

        for x, y in itertools.product(range(m), range(n)):
            elo = EloModel(alpha=alphas[x], beta=betas[y])
            test = PerformanceTest(elo, self.data)
            test.run()

            grid[y, x] = test
            tools.echo('ELO: {}/{} {}/{}'.format(x+1, n, y+1, n))

        return GridResult(
            grid=grid,
            xlabel='alpha',
            ylabel='beta',
            xvalues=alphas,
            yvalues=betas,
        )

    def search_pfa(self, gammas, deltas):
        """Performes grid search on PFA extended model using given parameters.

        :param gammas: Gamma parameters (see :class:`PFAModel`).
        :type gammas: list or :class:`numpy.array`
        :param deltas: Delta paramters (see :class:`PFAModel`).
        :type deltas: list or :class:`numpy.array`
        """
        m, n = len(gammas), len(deltas)

        elo = EloModel()
        grid = [[None] * m] * n

        for x, y in itertools.product(range(m), range(n)):
            pfa = PFAModel(elo, gamma=gammas[x], delta=deltas[y])
            test = PerformanceTest(pfa, self.data)
            test.run()

            grid[y, x] = test
            tools.echo('PFA: {}/{} {}/{}'.format(x+1, m, y+1, n))

        return GridResult(
            grid=grid,
            xlabel='gamma',
            ylabel='delta',
            xvalues=gammas,
            yvalues=deltas,
        )

    def search_pfas(self, decays, spacings):
        """Performes grid search on PFA extended with spacing and forgetting
        using given parameters.

        :param decays: Decay rates (see :class:`PFAWithSpacing`).
        :type decays: list or :class:`numpy.array`
        :param spacings: Spacing rates (see :class:`PFAWithSpacing`).
        :type spacings: list or :class:`numpy.array`
        """
        m, n = len(decays), len(spacings)

        elo = EloModel()
        grid = [[None] * m] * n

        for x, y in itertools.product(range(m), range(n)):
            pfas = PFAWithSpacing(elo,
                                  decay_rate=decays[x],
                                  spacing_rate=spacings[y])
            test = PerformanceTest(pfas, self.data)
            test.run()

            grid[y, x] = test
            tools.echo('PFA: {}/{} {}/{}'.format(x+1, m, y+1, n))

        return GridResult(
            grid=grid,
            xlabel='decay rates',
            ylabel='spacing rates',
            xvalues=decays,
            yvalues=spacings,
        )


class RandomSearch(object):
    """Encapsulates random searches for various models.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

    def search_elo(self, alpha, beta):
        """Performes random search on ELO model using given initial
        parameters.

        :param alpha: Initial alpha value (see :class:`EloModel`).
        :type alpha: float
        :param beta: Initial beta value (see :class:`EloModel`).
        :type beta: float
        """
        def fun(x):
            elo = EloModel(alpha=x[0], beta=x[1])
            test = PerformanceTest(elo)

            test.run()

            tools.echo('alpha={x[0]} beta={x[1]}'.format(x=x))
            return test.rmse().value

        return optimize.minimize(fun, [alpha, beta])

    def search_pfa(self, gamma, delta):
        """Performes random search on ELO model using given initial
        parameters.

        :param gamma: Initial gamma value (see :class:`PFAModel`).
        :type gamma: float
        :param delta: Initial delta value (see :class:`PFAModel`).
        :type delta: float
        """
        elo = EloModel()

        def fun(x):
            pfa = PFAModel(elo, gamma=x[0], delta=x[1])
            test = PerformanceTest(pfa)

            test.run()

            tools.echo('gamma={x[0]} delta={x[1]}'.format(x=x))
            return test.rmse().value

        return optimize.minimize(fun, [gamma, delta])
